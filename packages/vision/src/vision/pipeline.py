from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray
from shared.schemas import OcrBlock

from vision.ocr import OcrEngine
from vision.preprocess import compute_frame_diff, decode_image_bytes, resize_for_inference

ImageArray = NDArray[np.uint8]
OcrObserver = Callable[[str, float], None]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionResult:
    image_bgr: ImageArray
    ocr_blocks: list[OcrBlock]
    ocr_text: str
    changed_ratio: float
    mean_delta: float
    ocr_status: str
    ocr_refreshed_at: datetime | None
    ocr_latency_ms: float | None
    ocr_age_ms: float | None
    ocr_interval_seconds: float
    ocr_pending: bool
    timings_ms: dict[str, float] = field(default_factory=dict)


class VisionPipeline:
    def __init__(
        self,
        ocr_engine: OcrEngine,
        *,
        inference_max_side: int = 960,
        ocr_interval_seconds: float = 4.0,
        ocr_start_delay_seconds: float = 0.2,
        ocr_every_n_frames: int = 10,
        min_ocr_changed_ratio: float = 0.025,
        ocr_max_stale_seconds: float = 30.0,
        enable_ocr: bool = True,
        ocr_observer: OcrObserver | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine
        self.inference_max_side = inference_max_side
        self.ocr_interval_seconds = ocr_interval_seconds
        self.ocr_start_delay_seconds = ocr_start_delay_seconds
        self.ocr_every_n_frames = max(1, ocr_every_n_frames)
        self.min_ocr_changed_ratio = min_ocr_changed_ratio
        self.ocr_max_stale_seconds = ocr_max_stale_seconds
        self.enable_ocr = enable_ocr
        self.ocr_observer = ocr_observer
        self._previous_frame: ImageArray | None = None
        self._previous_ocr_blocks: list[OcrBlock] = []
        self._ocr_refreshed_at: datetime | None = None
        self._last_ocr_latency_ms: float | None = None
        self._last_ocr_started_monotonic = 0.0
        self._last_ocr_completed_monotonic = 0.0
        self._last_ocr_frame_index = 0
        self._frame_index = 0
        self._last_ocr_error: str | None = None
        self._ocr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    async def process(self, payload: bytes, frame_id: str | None = None) -> VisionResult:
        started = time.perf_counter()
        decoded = decode_image_bytes(payload)
        decode_ms = (time.perf_counter() - started) * 1000.0

        resize_started = time.perf_counter()
        image = resize_for_inference(decoded, max_side=self.inference_max_side)
        resize_ms = (time.perf_counter() - resize_started) * 1000.0

        state_started = time.perf_counter()
        async with self._lock:
            self._frame_index += 1
            diff = compute_frame_diff(self._previous_frame, image)
            self._previous_frame = image.copy()
            now = time.monotonic()
            should_ocr = self._should_schedule_ocr(diff.changed_ratio, now)
            ocr_status = self._cached_status(now)
            if should_ocr:
                if self._schedule_ocr_locked(image.copy(), frame_id):
                    ocr_status = "scheduled"
                else:
                    ocr_status = "running"
            ocr_blocks = list(self._previous_ocr_blocks)
            ocr_refreshed_at = self._ocr_refreshed_at
            ocr_latency_ms = self._last_ocr_latency_ms
            ocr_age_ms = self._ocr_age_ms(now)
            ocr_pending = self._ocr_task is not None and not self._ocr_task.done()
            if self._last_ocr_error and not ocr_blocks:
                ocr_status = "failed"
        state_ms = (time.perf_counter() - state_started) * 1000.0

        ocr_text = "\n".join(block.text for block in ocr_blocks)
        total_ms = (time.perf_counter() - started) * 1000.0
        return VisionResult(
            image_bgr=image,
            ocr_blocks=ocr_blocks,
            ocr_text=ocr_text,
            changed_ratio=diff.changed_ratio,
            mean_delta=diff.mean_delta,
            ocr_status=ocr_status,
            ocr_refreshed_at=ocr_refreshed_at,
            ocr_latency_ms=ocr_latency_ms,
            ocr_age_ms=ocr_age_ms,
            ocr_interval_seconds=self.ocr_interval_seconds,
            ocr_pending=ocr_pending,
            timings_ms={
                "vision_decode_ms": decode_ms,
                "vision_resize_ms": resize_ms,
                "vision_state_ms": state_ms,
                "vision_total_ms": total_ms,
                "screen_changed_ratio": diff.changed_ratio,
                "screen_mean_delta": diff.mean_delta,
            },
        )

    async def close(self) -> None:
        task = self._ocr_task
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    async def wait_for_ocr_idle(self) -> None:
        task = self._ocr_task
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    def set_ocr_observer(self, observer: OcrObserver | None) -> None:
        self.ocr_observer = observer

    def _should_schedule_ocr(self, changed_ratio: float, now: float) -> bool:
        if not self.enable_ocr:
            return False
        if self._ocr_task is not None and not self._ocr_task.done():
            return False
        if not self._previous_ocr_blocks and self._last_ocr_completed_monotonic == 0.0:
            return True

        seconds_since_start = now - self._last_ocr_started_monotonic
        frames_since_ocr = self._frame_index - self._last_ocr_frame_index
        if seconds_since_start < self.ocr_interval_seconds:
            return False
        if frames_since_ocr < self.ocr_every_n_frames:
            return False

        stale = self._last_ocr_completed_monotonic > 0.0 and (
            now - self._last_ocr_completed_monotonic >= self.ocr_max_stale_seconds
        )
        changed = changed_ratio >= self.min_ocr_changed_ratio
        return stale or changed

    def _schedule_ocr_locked(self, image: ImageArray, frame_id: str | None) -> bool:
        if self._ocr_task is not None and not self._ocr_task.done():
            return False
        self._last_ocr_started_monotonic = time.monotonic()
        self._last_ocr_frame_index = self._frame_index
        self._last_ocr_error = None
        self._ocr_task = asyncio.create_task(self._run_ocr(image, frame_id))
        return True

    async def _run_ocr(self, image: ImageArray, frame_id: str | None) -> None:
        if self.ocr_start_delay_seconds > 0:
            await asyncio.sleep(self.ocr_start_delay_seconds)
        started = time.perf_counter()
        try:
            blocks = await asyncio.to_thread(self.ocr_engine.read, image)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.exception("OCR failed", extra={"frame_id": frame_id})
            async with self._lock:
                self._last_ocr_latency_ms = elapsed_ms
                self._last_ocr_completed_monotonic = time.monotonic()
                self._last_ocr_error = str(exc)
                self._ocr_task = None
            self._notify_ocr_observer("failed", elapsed_ms)
            return

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        async with self._lock:
            self._previous_ocr_blocks = blocks
            self._ocr_refreshed_at = datetime.now(UTC)
            self._last_ocr_latency_ms = elapsed_ms
            self._last_ocr_completed_monotonic = time.monotonic()
            self._last_ocr_error = None
            self._ocr_task = None
        self._notify_ocr_observer("fresh", elapsed_ms)

    def _cached_status(self, now: float) -> str:
        if not self.enable_ocr:
            return "disabled"
        if self._ocr_task is not None and not self._ocr_task.done():
            return "running"
        if self._last_ocr_error:
            return "failed"
        if self._ocr_refreshed_at is None:
            return "pending"
        ocr_age_ms = self._ocr_age_ms(now)
        if ocr_age_ms is not None and ocr_age_ms < 500.0:
            return "fresh"
        return "cached"

    def _ocr_age_ms(self, now: float) -> float | None:
        if self._last_ocr_completed_monotonic == 0.0:
            return None
        return max(0.0, (now - self._last_ocr_completed_monotonic) * 1000.0)

    def _notify_ocr_observer(self, status: str, elapsed_ms: float) -> None:
        if self.ocr_observer is None:
            return
        try:
            self.ocr_observer(status, elapsed_ms)
        except Exception:
            logger.exception("OCR observer failed")
