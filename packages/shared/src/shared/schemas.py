from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

WebSocketEventType = Literal[
    "session.created",
    "frame.metadata",
    "frame.analysis",
    "queue.backpressure",
    "error",
    "heartbeat",
]


class ActivityLabel(StrEnum):
    CODING = "coding"
    READING = "reading"
    WRITING = "writing"
    VIDEO = "video"
    BROWSER = "browser"
    MESSAGING = "messaging"
    GAMING = "gaming"
    IDLE = "idle"
    UNKNOWN = "unknown"


class EducationLabel(StrEnum):
    PROGRAMMING = "programming"
    MATH = "math"
    SCIENCE = "science"
    LANGUAGE = "language"
    RESEARCH = "research"
    EXAM_PREP = "exam_prep"
    NOTE_TAKING = "note_taking"
    GENERAL = "general"


class OcrBlock(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[list[float]] = Field(default_factory=list)


class CaptureMetadata(BaseModel):
    width: int
    height: int
    monitor_id: str | None = None
    app_name: str | None = None
    window_title: str | None = None
    captured_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ClientFrameMetadata(CaptureMetadata):
    frame_id: str = Field(default_factory=lambda: str(uuid4()))
    mime_type: Literal["image/jpeg", "image/png"] = "image/jpeg"


class TutorSuggestion(BaseModel):
    title: str
    body: str
    confidence: float = Field(ge=0.0, le=1.0)
    actions: list[str] = Field(default_factory=list)


class StudyAnalytics(BaseModel):
    focus_score: float = Field(ge=0.0, le=1.0)
    distraction_score: float = Field(ge=0.0, le=1.0)
    study_seconds: int = 0
    distraction_seconds: int = 0
    active_context: EducationLabel = EducationLabel.GENERAL
    activity: ActivityLabel = ActivityLabel.UNKNOWN


class FrameAnalysis(BaseModel):
    session_id: str
    frame_id: str
    processed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ocr_text: str
    ocr_blocks: list[OcrBlock] = Field(default_factory=list)
    ocr_status: str = "unknown"
    ocr_refreshed_at: datetime | None = None
    ocr_latency_ms: float | None = Field(default=None, ge=0.0)
    ocr_age_ms: float | None = Field(default=None, ge=0.0)
    activity: ActivityLabel
    activity_confidence: float = Field(ge=0.0, le=1.0)
    activity_probabilities: dict[str, float] = Field(default_factory=dict)
    education_context: EducationLabel
    education_confidence: float = Field(ge=0.0, le=1.0)
    distraction_score: float = Field(ge=0.0, le=1.0)
    suggestions: list[TutorSuggestion] = Field(default_factory=list)
    analytics: StudyAnalytics
    timings_ms: dict[str, float] = Field(default_factory=dict)
    queue_wait_ms: float = Field(default=0.0, ge=0.0)
    capture_lag_ms: float = Field(default=0.0, ge=0.0)
    cpu_optimized: bool = False


class SessionCreated(BaseModel):
    session_id: str
    user_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WebSocketEnvelope(BaseModel):
    type: WebSocketEventType
    payload: dict[str, Any]
