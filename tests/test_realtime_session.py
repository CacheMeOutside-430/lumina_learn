from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from backend.core.telemetry import RuntimeMetrics
from backend.db.repository import SessionRepository
from backend.services.frame_processor import FrameProcessor, FrameWorkItem
from backend.services.realtime import RealtimeSession
from shared.schemas import (
    ActivityLabel,
    ClientFrameMetadata,
    EducationLabel,
    FrameAnalysis,
    StudyAnalytics,
)


@dataclass(frozen=True)
class _CreatedSession:
    id: str


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.sent: list[dict[str, Any]] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict[str, Any]:
        if self.messages:
            return self.messages.pop(0)
        await asyncio.sleep(0.05)
        return {"type": "websocket.disconnect"}

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)


class _FakeRepository:
    def __init__(self) -> None:
        self.closed_session_id: str | None = None

    async def create_session(self, user_id: str) -> _CreatedSession:
        assert user_id == "user-1"
        return _CreatedSession(id="session-1")

    async def close_session(self, session_id: str) -> None:
        self.closed_session_id = session_id


class _FakeProcessor:
    async def process(self, item: FrameWorkItem) -> FrameAnalysis:
        return FrameAnalysis(
            session_id=item.session_id,
            frame_id=item.metadata.frame_id,
            processed_at=datetime(2026, 5, 15, tzinfo=UTC),
            ocr_text="lesson",
            activity=ActivityLabel.CODING,
            activity_confidence=0.9,
            activity_probabilities={ActivityLabel.CODING.value: 0.9},
            education_context=EducationLabel.PROGRAMMING,
            education_confidence=0.8,
            distraction_score=0.1,
            analytics=StudyAnalytics(
                focus_score=0.9,
                distraction_score=0.1,
                active_context=EducationLabel.PROGRAMMING,
                activity=ActivityLabel.CODING,
            ),
        )


@pytest.mark.asyncio
async def test_realtime_session_processes_metadata_and_binary_frame() -> None:
    websocket = _FakeWebSocket(
        [
            {
                "text": (
                    '{"type":"frame.metadata","payload":{"frame_id":"frame-1",'
                    '"width":1280,"height":720,"mime_type":"image/jpeg"}}'
                )
            },
            {"bytes": b"jpeg-bytes"},
        ]
    )
    repository = _FakeRepository()
    session = RealtimeSession(
        websocket=cast(Any, websocket),
        user_id="user-1",
        repository=cast(SessionRepository, repository),
        processor=cast(FrameProcessor, _FakeProcessor()),
        max_frame_queue=4,
        max_frame_bytes=1024,
    )

    await session.run()

    assert websocket.accepted
    assert repository.closed_session_id == "session-1"
    assert [message["type"] for message in websocket.sent] == ["session.created", "frame.analysis"]
    assert websocket.sent[1]["payload"]["frame_id"] == "frame-1"


@pytest.mark.asyncio
async def test_realtime_session_rejects_invalid_json_without_crashing() -> None:
    websocket = _FakeWebSocket([{"text": "not-json"}])
    repository = _FakeRepository()
    session = RealtimeSession(
        websocket=cast(Any, websocket),
        user_id="user-1",
        repository=cast(SessionRepository, repository),
        processor=cast(FrameProcessor, _FakeProcessor()),
        max_frame_queue=4,
        max_frame_bytes=1024,
    )

    await session.run()

    assert repository.closed_session_id == "session-1"
    assert [message["type"] for message in websocket.sent] == ["session.created", "error"]
    assert websocket.sent[1]["payload"]["message"] == "Invalid JSON message"


@pytest.mark.asyncio
async def test_realtime_session_keeps_only_latest_pending_frame() -> None:
    metrics = RuntimeMetrics()
    session = RealtimeSession(
        websocket=cast(Any, _FakeWebSocket([])),
        user_id="user-1",
        repository=cast(SessionRepository, _FakeRepository()),
        processor=cast(FrameProcessor, _FakeProcessor()),
        max_frame_queue=2,
        max_frame_bytes=1024,
        metrics=metrics,
        target_processing_fps=0,
        latest_frame_only=True,
    )
    session._pending_item = FrameWorkItem(  # noqa: SLF001
        user_id="user-1",
        session_id="session-1",
        metadata=ClientFrameMetadata(frame_id="old", width=10, height=10),
        payload=b"old",
    )

    await session._enqueue_latest(  # noqa: SLF001
        FrameWorkItem(
            user_id="user-1",
            session_id="session-1",
            metadata=ClientFrameMetadata(frame_id="new", width=10, height=10),
            payload=b"new",
        )
    )

    assert session._pending_item is not None  # noqa: SLF001
    assert session._pending_item.metadata.frame_id == "new"  # noqa: SLF001
    assert metrics.snapshot()["frames_dropped"] == 1
