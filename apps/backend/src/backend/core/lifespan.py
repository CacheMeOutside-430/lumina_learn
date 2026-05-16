from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from backend.core.config import get_settings
from backend.core.telemetry import RuntimeMetrics
from backend.db.base import create_engine, create_sessionmaker, init_db
from backend.db.repository import SessionRepository
from backend.inference.container import build_inference_container, close_inference_container
from backend.services.analytics import StudyAnalyticsTracker
from backend.services.frame_processor import FrameProcessor
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())
    engine = create_engine(settings.database_url)
    sessionmaker = create_sessionmaker(engine)
    await init_db(engine)
    repository = SessionRepository(sessionmaker)
    container = await build_inference_container(settings)
    analytics = StudyAnalyticsTracker()
    metrics = RuntimeMetrics()
    container.vision.set_ocr_observer(metrics.ocr_completed)
    cpu_optimized = settings.cpu_optimized_enabled(container.classifier.device.type)
    processor = FrameProcessor(
        container=container,
        repository=repository,
        analytics=analytics,
        metrics=metrics,
        cpu_optimized=cpu_optimized,
        lightweight_context=settings.cpu_lightweight_context,
        lightweight_tutor=settings.cpu_lightweight_tutor,
        memory_write_interval_seconds=settings.memory_write_interval_seconds,
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.sessionmaker = sessionmaker
    app.state.repository = repository
    app.state.inference = container
    app.state.metrics = metrics
    app.state.frame_processor = processor
    try:
        yield
    finally:
        await close_inference_container(container)
        await engine.dispose()
