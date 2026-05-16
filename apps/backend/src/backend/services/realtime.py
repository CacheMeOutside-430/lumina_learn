from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from contextlib import suppress
from json import JSONDecodeError
from typing import Any

from backend.core.events import envelope
from backend.core.telemetry import RuntimeMetrics
from backend.db.repository import SessionRepository
from backend.services.frame_processor import FrameProcessor, FrameWorkItem
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from shared.schemas import ClientFrameMetadata, SessionCreated

logger = logging.getLogger(__name__)


class RealtimeSession:
    def __init__(
        self,
        websocket: WebSocket,
        user_id: str,
        repository: SessionRepository,
        processor: FrameProcessor,
        max_frame_queue: int,
        max_frame_bytes: int,
        metrics: RuntimeMetrics | None = None,
        *,
        target_processing_fps: float = 2.0,
        latest_frame_only: bool = True,
    ) -> None:
        self.websocket = websocket
        self.user_id = user_id
        self.repository = repository
        self.processor = processor
        self.max_frame_queue = max_frame_queue
        self.max_frame_bytes = max_frame_bytes
        self.target_processing_fps = target_processing_fps
        self.latest_frame_only = latest_frame_only
        self._send_lock = asyncio.Lock()
        self._pending_metadata: ClientFrameMetadata | None = None
        self._pending_item: FrameWorkItem | None = None
        self._pending_event = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._closing = False
        self._processing = False
        self._last_accepted_monotonic = 0.0
        self.metrics = metrics

    async def run(self) -> None:
        await self.websocket.accept()
        session = await self.repository.create_session(self.user_id)
        self._worker_task = asyncio.create_task(self._frame_worker())
        if self.metrics is not None:
            self.metrics.session_opened()
        await self._send(
            envelope(
                "session.created",
                SessionCreated(session_id=session.id, user_id=self.user_id).model_dump(mode="json"),
            )
        )
        try:
            while True:
                message = await self.websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                await self._handle_message(session.id, message)
        except WebSocketDisconnect:
            pass
        finally:
            self._closing = True
            self._pending_event.set()
            if self._worker_task is not None:
                self._worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._worker_task
            await self.repository.close_session(session.id)
            if self.metrics is not None:
                self.metrics.session_closed()

    async def _handle_message(self, session_id: str, message: Mapping[str, Any]) -> None:
        text = message.get("text")
        if isinstance(text, str):
            await self._handle_text(text)
            return
        payload = message.get("bytes")
        if payload is None:
            return
        if not isinstance(payload, bytes):
            await self._send(envelope("error", {"message": "Binary frame payload is invalid"}))
            return
        metadata = self._pending_metadata or ClientFrameMetadata(width=0, height=0)
        self._pending_metadata = None
        if len(payload) > self.max_frame_bytes:
            await self._send(
                envelope(
                    "error",
                    {
                        "message": "Frame exceeds configured maximum size",
                        "frame_id": metadata.frame_id,
                    },
                )
            )
            return

        queue_depth = self._queue_depth()
        if self.metrics is not None:
            self.metrics.frame_received(queue_depth)

        if self._should_drop_for_fps():
            await self._drop_frame(queue_depth)
            return

        item = FrameWorkItem(
            user_id=self.user_id,
            session_id=session_id,
            metadata=metadata,
            payload=payload,
            queue_depth=queue_depth,
        )
        await self._enqueue_latest(item)

    async def _handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
        except JSONDecodeError:
            await self._send(envelope("error", {"message": "Invalid JSON message"}))
            return
        if not isinstance(payload, dict):
            await self._send(envelope("error", {"message": "WebSocket message must be an object"}))
            return
        message_type = payload.get("type")
        if message_type == "frame.metadata":
            try:
                self._pending_metadata = ClientFrameMetadata.model_validate(payload.get("payload"))
            except ValidationError as exc:
                await self._send(
                    envelope(
                        "error",
                        {
                            "message": "Invalid frame metadata",
                            "details": exc.errors(include_url=False, include_input=False),
                        },
                    )
                )
        elif message_type == "heartbeat":
            await self._send(envelope("heartbeat", {"ok": True}))
        else:
            await self._send(envelope("error", {"message": f"Unsupported message type: {message_type}"}))

    async def _enqueue_latest(self, item: FrameWorkItem) -> None:
        queue_depth = self._queue_depth()
        if self.latest_frame_only:
            if self._pending_item is not None and self.metrics is not None:
                self.metrics.frame_dropped(queue_depth)
            self._pending_item = item
            self._last_accepted_monotonic = time.monotonic()
            self._pending_event.set()
            if self.metrics is not None:
                self.metrics.observe_queue_depth(self._queue_depth())
            return

        if queue_depth >= self.max_frame_queue:
            await self._drop_frame(queue_depth)
            return
        self._pending_item = item
        self._last_accepted_monotonic = time.monotonic()
        self._pending_event.set()
        if self.metrics is not None:
            self.metrics.observe_queue_depth(self._queue_depth())

    async def _frame_worker(self) -> None:
        while not self._closing:
            await self._pending_event.wait()
            self._pending_event.clear()
            item = self._pending_item
            self._pending_item = None
            if item is None:
                continue
            self._processing = True
            if self.metrics is not None:
                self.metrics.observe_queue_depth(self._queue_depth())
            try:
                await self._process_and_send(item)
            finally:
                self._processing = False
                if self._pending_item is not None:
                    self._pending_event.set()

    async def _process_and_send(self, item: FrameWorkItem) -> None:
        try:
            analysis = await self.processor.process(item)
            await self._send(envelope("frame.analysis", analysis.model_dump(mode="json")))
        except Exception:
            logger.exception("Frame processing failed", extra={"frame_id": item.metadata.frame_id})
            await self._send(
                envelope(
                    "error",
                    {
                        "message": "Frame processing failed",
                        "frame_id": item.metadata.frame_id,
                    },
                )
            )

    def _should_drop_for_fps(self) -> bool:
        if self.target_processing_fps <= 0:
            return False
        min_interval = 1.0 / self.target_processing_fps
        return time.monotonic() - self._last_accepted_monotonic < min_interval and self._queue_depth() > 0

    async def _drop_frame(self, queue_depth: int) -> None:
        if self.metrics is not None:
            self.metrics.frame_dropped(queue_depth)
            self.metrics.backpressure(queue_depth)
        await self._send(
            envelope(
                "queue.backpressure",
                {
                    "queued": queue_depth,
                    "dropped": True,
                },
            )
        )

    def _queue_depth(self) -> int:
        return int(self._processing) + int(self._pending_item is not None)

    async def _send(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(payload)
