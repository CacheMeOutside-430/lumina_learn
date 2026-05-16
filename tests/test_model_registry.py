from __future__ import annotations

import json
from pathlib import Path

import torch
from backend.services.model_registry import list_checkpoints, read_training_metrics


def test_model_registry_summarizes_checkpoint_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "activity_classifier_best.pt"
    torch.save(
        {
            "epoch": 3,
            "image_size": 224,
            "model_state_dict": {},
            "train_metrics": {"loss": 0.8, "activity_accuracy": 0.7},
            "val_metrics": {"loss": 0.9, "activity_accuracy": 0.6},
            "model_config": {"architecture": "ScreenActivityNet", "version": 1},
        },
        checkpoint,
    )

    summaries = list_checkpoints(tmp_path, active_checkpoint=checkpoint)

    assert len(summaries) == 1
    assert summaries[0].is_active is True
    assert summaries[0].epoch == 3
    assert summaries[0].val_metrics["loss"] == 0.9
    assert summaries[0].model_config["architecture"] == "ScreenActivityNet"


def test_model_registry_reads_training_metrics_tail(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    rows = [
        {"epoch": 1, "train": {"loss": 2.0}, "val": {"loss": 2.2}},
        {"epoch": 2, "train": {"loss": 1.5}, "val": {"loss": 1.7}},
    ]
    metrics_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    metrics = read_training_metrics(metrics_path, limit=1)

    assert len(metrics) == 1
    assert metrics[0].epoch == 2
    assert metrics[0].val["loss"] == 1.7
