from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from activity_classifier.dataset import (
    ScreenshotManifestDataset,
    default_eval_transform,
    default_train_transform,
)
from activity_classifier.model import ScreenActivityNet


@dataclass(frozen=True)
class TrainConfig:
    manifest: Path
    val_manifest: Path
    output: Path
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    image_size: int = 224
    num_workers: int = 4
    amp: bool = True
    seed: int = 20260515
    grad_clip_norm: float = 1.0
    resume_checkpoint: Path | None = None


def build_loader(
    manifest: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    transform = default_train_transform(image_size) if train else default_eval_transform(image_size)
    dataset = ScreenshotManifestDataset(manifest, transform=transform)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(dataset) > batch_size,
        generator=generator if train else None,
    )


def train_one_epoch(
    model: ScreenActivityNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any,
    use_amp: bool,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct_activity = 0
    correct_education = 0

    for images, activity_targets, education_targets in loader:
        images = images.to(device, non_blocking=True)
        activity_targets = activity_targets.to(device, non_blocking=True)
        education_targets = education_targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[attr-defined]
            activity_logits, education_logits = model(images)
            activity_loss = criterion(activity_logits, activity_targets)
            education_loss = criterion(education_logits, education_targets)
            loss = activity_loss + 0.7 * education_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        correct_activity += int((activity_logits.argmax(dim=1) == activity_targets).sum().item())
        correct_education += int((education_logits.argmax(dim=1) == education_targets).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "activity_accuracy": correct_activity / max(total, 1),
        "education_accuracy": correct_education / max(total, 1),
    }


@torch.inference_mode()
def evaluate(
    model: ScreenActivityNet,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    correct_activity = 0
    correct_education = 0

    for images, activity_targets, education_targets in loader:
        images = images.to(device, non_blocking=True)
        activity_targets = activity_targets.to(device, non_blocking=True)
        education_targets = education_targets.to(device, non_blocking=True)
        activity_logits, education_logits = model(images)
        loss = criterion(activity_logits, activity_targets) + 0.7 * criterion(
            education_logits, education_targets
        )
        batch_size = images.size(0)
        total += batch_size
        total_loss += float(loss.item()) * batch_size
        correct_activity += int((activity_logits.argmax(dim=1) == activity_targets).sum().item())
        correct_education += int((education_logits.argmax(dim=1) == education_targets).sum().item())

    return {
        "loss": total_loss / max(total, 1),
        "activity_accuracy": correct_activity / max(total, 1),
        "education_accuracy": correct_education / max(total, 1),
    }


def train(config: TrainConfig) -> Path:
    if config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if config.grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive")
    config.output.mkdir(parents=True, exist_ok=True)
    _set_reproducibility(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_loader(
        config.manifest,
        config.image_size,
        config.batch_size,
        config.num_workers,
        train=True,
        seed=config.seed,
    )
    val_loader = build_loader(
        config.val_manifest,
        config.image_size,
        config.batch_size,
        config.num_workers,
        train=False,
        seed=config.seed,
    )

    model = ScreenActivityNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")  # type: ignore[attr-defined]
    best_loss = float("inf")
    best_path = config.output / "activity_classifier_best.pt"
    start_epoch = 1

    if config.resume_checkpoint is not None:
        payload = _load_checkpoint(config.resume_checkpoint, device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        if "scheduler_state_dict" in payload:
            scheduler.load_state_dict(payload["scheduler_state_dict"])
        if "scaler_state_dict" in payload and config.amp and device.type == "cuda":
            scaler.load_state_dict(payload["scaler_state_dict"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_loss = float(payload.get("best_loss", best_loss))

    metrics_path = config.output / "metrics.jsonl"
    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp=config.amp and device.type == "cuda",
            grad_clip_norm=config.grad_clip_norm,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_loss": min(best_loss, val_metrics["loss"]),
            "image_size": config.image_size,
            "seed": config.seed,
            "model_config": {"architecture": "ScreenActivityNet", "version": 1},
        }
        epoch_path = config.output / f"activity_classifier_epoch_{epoch:03d}.pt"
        _save_checkpoint(checkpoint, epoch_path)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            checkpoint["best_loss"] = best_loss
            _save_checkpoint(checkpoint, best_path)
        metric_row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric_row, sort_keys=True) + "\n")
        print(metric_row, flush=True)

    return best_path


def _set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Unsupported training checkpoint format: {path}")
    return payload


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the Lumina screen activity classifier.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    return TrainConfig(
        manifest=args.manifest,
        val_manifest=args.val_manifest,
        output=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        seed=args.seed,
        grad_clip_norm=args.grad_clip_norm,
        resume_checkpoint=args.resume_checkpoint,
    )


if __name__ == "__main__":
    train(parse_args())
