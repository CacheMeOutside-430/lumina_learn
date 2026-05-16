from __future__ import annotations

from typing import Any, cast

from backend.api.ws import _websocket_authorized
from backend.core.config import Settings, get_settings
from backend.main import create_app
from fastapi import WebSocket
from fastapi.testclient import TestClient


class _FakeWebSocket:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def test_websocket_auth_allows_local_mode_without_api_key() -> None:
    settings = Settings(api_key=None)
    websocket = cast(WebSocket, _FakeWebSocket({}))

    assert _websocket_authorized(settings, websocket, api_key=None) is True


def test_websocket_auth_requires_matching_api_key_when_configured() -> None:
    settings = Settings(api_key="secret-token")
    websocket = cast(Any, _FakeWebSocket({"authorization": "Bearer secret-token"}))

    assert _websocket_authorized(settings, websocket, api_key=None) is True
    assert _websocket_authorized(settings, websocket, api_key="wrong") is True
    assert _websocket_authorized(settings, cast(WebSocket, _FakeWebSocket({})), api_key="wrong") is False
    assert _websocket_authorized(settings, cast(WebSocket, _FakeWebSocket({})), api_key="secret-token") is True


def test_api_key_middleware_protects_operational_endpoints(monkeypatch: Any) -> None:
    monkeypatch.setenv("API_KEY", "secret-token")
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/metrics").status_code == 401
            assert client.get("/metrics", headers={"Authorization": "Bearer secret-token"}).status_code == 200
    finally:
        get_settings.cache_clear()
