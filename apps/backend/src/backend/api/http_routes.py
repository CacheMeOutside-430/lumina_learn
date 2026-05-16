from __future__ import annotations

from typing import Any

from ai_core.device import choose_device
from backend.core.telemetry import RuntimeMetrics
from backend.db.repository import SessionRepository
from backend.services.model_registry import (
    CheckpointSummary,
    TrainingMetricRow,
    read_training_metrics,
)
from backend.services.model_registry import list_checkpoints as list_model_checkpoints
from fastapi import APIRouter, Path, Query, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "env": settings.app_env,
        "production": settings.is_production,
    }


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, str | bool]:
    engine: AsyncEngine = request.app.state.engine
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))
    classifier = request.app.state.inference.classifier
    settings = request.app.state.settings
    checkpoint_ready = (
        classifier.checkpoint_path is not None and classifier.checkpoint_path.exists()
        if settings.require_activity_checkpoint
        else True
    )
    return {
        "status": "ready" if checkpoint_ready else "degraded",
        "checkpoint_ready": checkpoint_ready,
    }


@router.get("/metrics")
async def metrics(request: Request) -> dict[str, Any]:
    runtime_metrics: RuntimeMetrics = request.app.state.metrics
    return runtime_metrics.snapshot()


@router.get("/system/status")
async def system_status(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    classifier = request.app.state.inference.classifier
    device = choose_device(settings.enable_gpu)
    cpu_optimized = settings.cpu_optimized_enabled(device.type)
    checkpoint_path = classifier.checkpoint_path or settings.activity_checkpoint
    return {
        "env": settings.app_env,
        "production": settings.is_production,
        "device": str(device),
        "gpu_enabled": settings.enable_gpu and device.type == "cuda",
        "gpu_requested": settings.enable_gpu,
        "cpu_optimized_active": cpu_optimized,
        "cpu_optimized_mode": settings.cpu_optimized_mode,
        "database": _database_kind(settings.database_url),
        "vector_backend": settings.vector_backend,
        "ocr_engine": settings.ocr_engine,
        "ocr_engine_resolved": type(request.app.state.inference.vision.ocr_engine).__name__,
        "ocr_interval_seconds": settings.ocr_interval_seconds,
        "ocr_start_delay_seconds": settings.ocr_start_delay_seconds,
        "ocr_every_n_frames": settings.ocr_every_n_frames,
        "ocr_max_side": settings.ocr_max_side,
        "ocr_roi_enabled": settings.ocr_roi_enabled,
        "ocr_threads": settings.ocr_threads,
        "ocr_min_changed_ratio": settings.ocr_min_changed_ratio,
        "inference_max_side": settings.inference_max_side,
        "latest_frame_only": settings.latest_frame_only,
        "cpu_lightweight_context": settings.cpu_lightweight_context,
        "cpu_lightweight_tutor": settings.cpu_lightweight_tutor,
        "local_inference": settings.local_inference,
        "reasoning_model": settings.reasoning_model,
        "embedding_model": settings.embedding_model,
        "activity_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "activity_checkpoint_loaded": classifier.model is not None,
        "heuristic_classifier_allowed": settings.allow_heuristic_classifier,
        "require_activity_checkpoint": settings.require_activity_checkpoint,
        "max_frame_queue": settings.effective_frame_queue(cpu_optimized),
        "max_frame_bytes": settings.max_frame_bytes,
        "frame_processing_fps": settings.frame_processing_fps,
    }


@router.get("/models/checkpoints")
async def model_checkpoints(request: Request) -> list[CheckpointSummary]:
    settings = request.app.state.settings
    return list_model_checkpoints(settings.model_dir, active_checkpoint=settings.activity_checkpoint)


@router.get("/training/metrics")
async def training_metrics(
    request: Request,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[TrainingMetricRow]:
    settings = request.app.state.settings
    return read_training_metrics(settings.resolved_training_metrics_path, limit=limit)


@router.get("/users/{user_id}/events")
async def recent_events(
    request: Request,
    user_id: str = Path(min_length=1, max_length=128),
    limit: int = Query(default=50, ge=1, le=200),
) -> list[dict[str, Any]]:
    repository: SessionRepository = request.app.state.repository
    events = await repository.recent_events(user_id=user_id, limit=limit)
    return [event.payload for event in events]


def _database_kind(database_url: str) -> str:
    return database_url.split(":", 1)[0]
