from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.feedback import router as feedback_router
from backend.api.http_routes import router as http_router
from backend.api.ws import router as ws_router
from backend.core.config import get_settings
from backend.core.lifespan import lifespan
from backend.core.middleware import install_middlewares


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Lumina Learn Backend", version="0.1.0", lifespan=lifespan)
    install_middlewares(app, settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["authorization", "content-type", "x-api-key", "x-request-id"],
    )
    app.include_router(http_router)
    app.include_router(ws_router)
    app.include_router(feedback_router)
    return app


app = create_app()
