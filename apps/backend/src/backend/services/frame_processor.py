from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from activity_classifier.inference import ClassificationResult
from backend.core.telemetry import RuntimeMetrics
from backend.db.repository import SessionRepository
from backend.inference.container import InferenceContainer
from backend.services.analytics import StudyAnalyticsTracker
from shared.schemas import (
    ClientFrameMetadata,
    EducationLabel,
    FrameAnalysis,
    OcrBlock,
    StudyAnalytics,
    TutorSuggestion,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameWorkItem:
    user_id: str
    session_id: str
    metadata: ClientFrameMetadata
    payload: bytes
    received_at_monotonic: float = field(default_factory=time.perf_counter)
    queue_depth: int = 0


class FrameProcessor:
    def __init__(
        self,
        container: InferenceContainer,
        repository: SessionRepository,
        analytics: StudyAnalyticsTracker,
        metrics: RuntimeMetrics | None = None,
        *,
        cpu_optimized: bool = False,
        lightweight_context: bool = False,
        lightweight_tutor: bool = False,
        memory_write_interval_seconds: float = 10.0,
    ) -> None:
        self.container = container
        self.repository = repository
        self.analytics = analytics
        self.metrics = metrics
        self.cpu_optimized = cpu_optimized
        self.lightweight_context = lightweight_context
        self.lightweight_tutor = lightweight_tutor
        self.memory_write_interval_seconds = memory_write_interval_seconds
        self._last_memory_write_monotonic = 0.0
        self._memory_task: asyncio.Task[None] | None = None

    async def process(self, item: FrameWorkItem) -> FrameAnalysis:
        started = time.perf_counter()
        queue_wait_ms = max(0.0, (started - item.received_at_monotonic) * 1000.0)
        capture_lag_ms = max(
            0.0,
            (datetime.now(UTC) - item.metadata.captured_at).total_seconds() * 1000.0,
        )
        try:
            analysis = await self._process(
                item,
                queue_wait_ms=queue_wait_ms,
                capture_lag_ms=capture_lag_ms,
            )
        except Exception:
            if self.metrics is not None:
                self.metrics.frame_failed()
            raise
        if self.metrics is not None:
            self.metrics.frame_processed(
                (time.perf_counter() - started) * 1000.0,
                queue_wait_ms=queue_wait_ms,
                capture_lag_ms=capture_lag_ms,
                timings_ms=analysis.timings_ms,
                ocr_status=analysis.ocr_status,
                ocr_latency_ms=analysis.ocr_latency_ms,
                ocr_age_ms=analysis.ocr_age_ms,
            )
        return analysis

    async def _process(
        self,
        item: FrameWorkItem,
        *,
        queue_wait_ms: float,
        capture_lag_ms: float,
    ) -> FrameAnalysis:
        vision_result = await self.container.vision.process(item.payload, frame_id=item.metadata.frame_id)
        timings_ms = dict(vision_result.timings_ms)

        classifier_started = time.perf_counter()
        classification = await asyncio.to_thread(
            self.container.classifier.predict,
            vision_result.image_bgr,
        )
        timings_ms["classifier_ms"] = (time.perf_counter() - classifier_started) * 1000.0

        context_started = time.perf_counter()
        if self.cpu_optimized and self.lightweight_context:
            education_context, education_confidence = _context_from_prior(
                classification.education_probabilities
            )
        else:
            context = await self.container.context.classify(
                vision_result.ocr_text,
                neural_prior=classification.education_probabilities,
            )
            education_context = context.label
            education_confidence = context.confidence
        timings_ms["context_ms"] = (time.perf_counter() - context_started) * 1000.0

        distraction_started = time.perf_counter()
        distraction_score = self.container.distraction.score(
            classification.activity_probabilities,
            education_confidence,
            vision_result.ocr_text,
        )
        timings_ms["distraction_ms"] = (time.perf_counter() - distraction_started) * 1000.0

        tutor_started = time.perf_counter()
        try:
            if self.cpu_optimized and self.lightweight_tutor:
                suggestions = [_fallback_suggestion(distraction_score)]
            else:
                suggestions = await self.container.tutor.suggest(
                    user_id=item.user_id,
                    ocr_text=vision_result.ocr_text,
                    activity=classification.activity,
                    education_context=education_context,
                    distraction_score=distraction_score,
                )
        except Exception:
            logger.exception("Tutor suggestion generation failed", extra={"frame_id": item.metadata.frame_id})
            suggestions = [_fallback_suggestion(distraction_score)]
        timings_ms["tutor_ms"] = (time.perf_counter() - tutor_started) * 1000.0

        analytics_started = time.perf_counter()
        analytics = self.analytics.update(
            session_id=item.session_id,
            activity=classification.activity,
            education_context=education_context,
            distraction_score=distraction_score,
        )
        timings_ms["analytics_ms"] = (time.perf_counter() - analytics_started) * 1000.0

        analysis = self._build_analysis(
            item=item,
            classification=classification,
            ocr_text=vision_result.ocr_text,
            ocr_blocks=vision_result.ocr_blocks,
            ocr_status=vision_result.ocr_status,
            ocr_refreshed_at=vision_result.ocr_refreshed_at,
            ocr_latency_ms=vision_result.ocr_latency_ms,
            ocr_age_ms=vision_result.ocr_age_ms,
            education_context=education_context,
            education_confidence=education_confidence,
            distraction_score=distraction_score,
            suggestions=suggestions,
            analytics=analytics,
            timings_ms=timings_ms,
            queue_wait_ms=queue_wait_ms,
            capture_lag_ms=capture_lag_ms,
            cpu_optimized=self.cpu_optimized,
        )

        save_started = time.perf_counter()
        await self.repository.save_analysis(analysis)
        analysis.timings_ms["repository_ms"] = (time.perf_counter() - save_started) * 1000.0

        memory_started = time.perf_counter()
        try:
            if self.cpu_optimized:
                if self._should_write_memory(analysis):
                    self._schedule_memory_write(item.user_id, item.session_id, analysis)
                    self._last_memory_write_monotonic = time.perf_counter()
                    analysis.timings_ms["memory_deferred"] = 1.0
            elif self._should_write_memory(analysis):
                await self.container.memory.remember(
                    user_id=item.user_id,
                    session_id=item.session_id,
                    text=_memory_text(analysis),
                    metadata={
                        "frame_id": analysis.frame_id,
                        "activity": analysis.activity.value,
                        "education_context": analysis.education_context.value,
                        "distraction_score": analysis.distraction_score,
                    },
                )
                self._last_memory_write_monotonic = time.perf_counter()
        except Exception:
            logger.exception("Memory write failed", extra={"frame_id": analysis.frame_id})
        analysis.timings_ms["memory_ms"] = (time.perf_counter() - memory_started) * 1000.0
        return analysis

    def _schedule_memory_write(
        self,
        user_id: str,
        session_id: str,
        analysis: FrameAnalysis,
    ) -> None:
        if self._memory_task is not None and not self._memory_task.done():
            return
        text = _memory_text(analysis)
        metadata = {
            "frame_id": analysis.frame_id,
            "activity": analysis.activity.value,
            "education_context": analysis.education_context.value,
            "distraction_score": analysis.distraction_score,
        }
        self._memory_task = asyncio.create_task(
            self._remember_background(
                user_id=user_id,
                session_id=session_id,
                text=text,
                metadata=metadata,
                frame_id=analysis.frame_id,
            )
        )

    async def _remember_background(
        self,
        *,
        user_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, str | float],
        frame_id: str,
    ) -> None:
        try:
            await self.container.memory.remember(
                user_id=user_id,
                session_id=session_id,
                text=text,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Background memory write failed", extra={"frame_id": frame_id})

    def _should_write_memory(self, analysis: FrameAnalysis) -> bool:
        if not self.cpu_optimized:
            return True
        if not analysis.ocr_text.strip():
            return False
        if analysis.ocr_status not in {"fresh", "scheduled", "cached"}:
            return False
        now = time.perf_counter()
        return now - self._last_memory_write_monotonic >= self.memory_write_interval_seconds

    @staticmethod
    def _build_analysis(
        item: FrameWorkItem,
        classification: ClassificationResult,
        ocr_text: str,
        ocr_blocks: list[OcrBlock],
        ocr_status: str,
        ocr_refreshed_at: datetime | None,
        ocr_latency_ms: float | None,
        ocr_age_ms: float | None,
        education_context: EducationLabel,
        education_confidence: float,
        distraction_score: float,
        suggestions: list[TutorSuggestion],
        analytics: StudyAnalytics,
        timings_ms: dict[str, float],
        queue_wait_ms: float,
        capture_lag_ms: float,
        cpu_optimized: bool,
    ) -> FrameAnalysis:
        return FrameAnalysis(
            session_id=item.session_id,
            frame_id=item.metadata.frame_id,
            processed_at=datetime.now(UTC),
            ocr_text=ocr_text,
            ocr_blocks=ocr_blocks,
            ocr_status=ocr_status,
            ocr_refreshed_at=ocr_refreshed_at,
            ocr_latency_ms=ocr_latency_ms,
            ocr_age_ms=ocr_age_ms,
            activity=classification.activity,
            activity_confidence=classification.activity_confidence,
            activity_probabilities=classification.activity_probabilities,
            education_context=education_context,
            education_confidence=education_confidence,
            distraction_score=distraction_score,
            suggestions=suggestions,
            analytics=analytics,
            timings_ms=timings_ms,
            queue_wait_ms=queue_wait_ms,
            capture_lag_ms=capture_lag_ms,
            cpu_optimized=cpu_optimized,
        )


def _memory_text(analysis: FrameAnalysis) -> str:
    return (
        f"Activity: {analysis.activity.value}\n"
        f"Context: {analysis.education_context.value}\n"
        f"Distraction: {analysis.distraction_score:.2f}\n"
        f"Visible text:\n{analysis.ocr_text[:3000]}"
    )


def _fallback_suggestion(distraction_score: float) -> TutorSuggestion:
    if distraction_score >= 0.6:
        return TutorSuggestion(
            title="Refocus on the active task",
            body="The current screen looks distracting. Return to the study material or pause capture while switching tasks.",
            confidence=0.62,
            actions=["Return to study", "Pause capture"],
        )
    return TutorSuggestion(
        title="Continue the study thread",
        body="Stay with the current material and capture the next useful question or concept before moving on.",
        confidence=0.58,
        actions=["Write a note", "Review next concept"],
    )


def _context_from_prior(probabilities: dict[str, float]) -> tuple[EducationLabel, float]:
    if not probabilities:
        return EducationLabel.GENERAL, 0.5
    label, confidence = max(probabilities.items(), key=lambda item: item[1])
    try:
        return EducationLabel(label), float(confidence)
    except ValueError:
        return EducationLabel.GENERAL, float(confidence)
