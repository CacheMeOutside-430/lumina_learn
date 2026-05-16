from __future__ import annotations

import secrets

from backend.core.config import Settings
from backend.services.realtime import RealtimeSession
from fastapi import APIRouter, Query, WebSocket, status

router = APIRouter()


@router.websocket("/ws/realtime")
async def realtime_ws(
    websocket: WebSocket,
    user_id: str = Query(default="local-user", min_length=1, max_length=128),
    api_key: str | None = Query(default=None, max_length=4096),
) -> None:
    state = websocket.scope["app"].state
    origin = websocket.headers.get("origin")
    if origin and origin not in state.settings.cors_origin_list:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    if not _websocket_authorized(state.settings, websocket, api_key):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    session = RealtimeSession(
        websocket=websocket,
        user_id=user_id,
        repository=state.repository,
        processor=state.frame_processor,
        max_frame_queue=state.settings.effective_frame_queue(
            state.settings.cpu_optimized_enabled(state.inference.classifier.device.type)
        ),
        max_frame_bytes=state.settings.max_frame_bytes,
        metrics=state.metrics,
        target_processing_fps=state.settings.frame_processing_fps,
        latest_frame_only=state.settings.latest_frame_only,
    )
    await session.run()


def _websocket_authorized(settings: Settings, websocket: WebSocket, api_key: str | None) -> bool:
    if not settings.api_key:
        return True
    authorization = websocket.headers.get("authorization", "")
    bearer = authorization.removeprefix("Bearer ").strip()
    header_key = websocket.headers.get("x-api-key", "")
    candidates = (api_key or "", bearer, header_key)
    return any(secrets.compare_digest(candidate, settings.api_key) for candidate in candidates)
