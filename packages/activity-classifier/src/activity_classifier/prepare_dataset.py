from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from activity_classifier.labels import ACTIVITY_TO_INDEX, EDUCATION_TO_INDEX

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_dataset(source: Path, output: Path, val_ratio: float, seed: int) -> tuple[Path, Path]:
    rows = []
    for image_path in source.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            activity_label = image_path.relative_to(source).parts[0]
            education_label = image_path.relative_to(source).parts[1]
        except IndexError as exc:
            raise ValueError(
                "Expected dataset layout: source/activity_label/education_label/image.jpg"
            ) from exc
        if activity_label not in ACTIVITY_TO_INDEX:
            raise ValueError(f"Unknown activity label in path: {activity_label}")
        if education_label not in EDUCATION_TO_INDEX:
            raise ValueError(f"Unknown education label in path: {education_label}")
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "activity_label": activity_label,
                "education_label": education_label,
            }
        )
    if not rows:
        raise ValueError(f"No images found under {source}")
    random.Random(seed).shuffle(rows)
    split = max(1, int(len(rows) * (1.0 - val_ratio)))
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[-1:]
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    val_path = output / "val.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)
    return train_path, val_path


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation manifests from screenshots.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path, val_path = prepare_dataset(args.source, args.output, args.val_ratio, args.seed)
    print({"train_manifest": str(train_path), "val_manifest": str(val_path)}, flush=True)


if __name__ == "__main__":
    main()
