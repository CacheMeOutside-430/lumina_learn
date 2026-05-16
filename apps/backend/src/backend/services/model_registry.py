from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointSummary:
    name: str
    path: str
    bytes: int
    modified_at: datetime
    is_active: bool
    epoch: int | None = None
    image_size: int | None = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    model_config: dict[str, str | int | float | bool] = field(default_factory=dict)
    readable: bool = True
    error: str | None = None


@dataclass(frozen=True)
class TrainingMetricRow:
    epoch: int
    train: dict[str, float]
    val: dict[str, float]


def list_checkpoints(model_dir: Path, active_checkpoint: Path | None = None) -> list[CheckpointSummary]:
    active_resolved = active_checkpoint.resolve() if active_checkpoint else None
    candidates = sorted(model_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [_summarize_checkpoint(path, active_resolved) for path in candidates]


def read_training_metrics(metrics_path: Path, limit: int = 100) -> list[TrainingMetricRow]:
    if not metrics_path.exists():
        return []
    rows: list[TrainingMetricRow] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                TrainingMetricRow(
                    epoch=int(payload["epoch"]),
                    train=_float_metrics(payload.get("train", {}), line_number),
                    val=_float_metrics(payload.get("val", {}), line_number),
                )
            )
    return rows[-limit:]


def _summarize_checkpoint(path: Path, active_checkpoint: Path | None) -> CheckpointSummary:
    stat = path.stat()
    name = path.name
    checkpoint_path = str(path)
    size_bytes = stat.st_size
    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    is_active = active_checkpoint == path.resolve()
    try:
        import torch

        payload: Any = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload if isinstance(payload, dict) else {}
        return CheckpointSummary(
            name=name,
            path=checkpoint_path,
            bytes=size_bytes,
            modified_at=modified_at,
            is_active=is_active,
            epoch=_optional_int(metadata.get("epoch")),
            image_size=_optional_int(metadata.get("image_size")),
            train_metrics=_float_metrics(metadata.get("train_metrics", {}), 0),
            val_metrics=_float_metrics(metadata.get("val_metrics", {}), 0),
            model_config=_primitive_config(metadata.get("model_config", {})),
        )
    except Exception as exc:
        return CheckpointSummary(
            name=name,
            path=checkpoint_path,
            bytes=size_bytes,
            modified_at=modified_at,
            is_active=is_active,
            readable=False,
            error=str(exc)[:240],
        )


def _float_metrics(value: object, line_number: int) -> dict[str, float]:
    if not isinstance(value, dict):
        if line_number:
            raise ValueError(f"Metric row {line_number} must contain object train/val metrics")
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(key, str) and isinstance(raw, int | float):
            result[key] = float(raw)
    return result


def _primitive_config(value: object) -> dict[str, str | int | float | bool]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str | int | float | bool] = {}
    for key, raw in value.items():
        if isinstance(key, str) and isinstance(raw, str | int | float | bool):
            result[key] = raw
    return result


def _optional_int(value: object) -> int | None:
    return int(value) if isinstance(value, int | float) else None
