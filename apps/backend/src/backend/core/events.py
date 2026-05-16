from __future__ import annotations

from typing import Any

from shared.schemas import WebSocketEnvelope, WebSocketEventType


def envelope(event_type: WebSocketEventType, payload: dict[str, Any]) -> dict[str, Any]:
    return WebSocketEnvelope(type=event_type, payload=payload).model_dump(mode="json")
