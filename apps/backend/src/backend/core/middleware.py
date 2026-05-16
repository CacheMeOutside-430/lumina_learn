from __future__ import annotations

import logging
import secrets
import time
from collections import defaultdict, deque
from uuid import uuid4

from backend.core.config import Settings
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


def install_middlewares(app: FastAPI, settings: Settings) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        limit=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
    if settings.api_key:
        app.add_middleware(ApiKeyAuthMiddleware, api_key=settings.api_key)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid4())
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, limit: int, window_seconds: int) -> None:
        super().__init__(app)
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: defaultdict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS" or request.url.path.startswith("/health"):
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        timestamps = self._requests[client]
        while timestamps and now - timestamps[0] > self.window_seconds:
            timestamps.popleft()
        if len(timestamps) >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - timestamps[0])))
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        timestamps.append(now)
        return await call_next(request)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, api_key: str) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS" or request.url.path.startswith("/health"):
            return await call_next(request)
        authorization = request.headers.get("authorization", "")
        bearer = authorization.removeprefix("Bearer ").strip()
        header_key = request.headers.get("x-api-key", "")
        if secrets.compare_digest(bearer, self.api_key) or secrets.compare_digest(header_key, self.api_key):
            return await call_next(request)
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
