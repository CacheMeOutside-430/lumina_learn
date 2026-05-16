#!/usr/bin/env python3
r"""
Lightweight benchmark for activity classifier inference on CPU.

Usage: run this script from the project root in the project's virtualenv.
Example:
  .venv\Scripts\python.exe scripts/benchmark_inference.py

It will look for a model at models/activity_classifier_best.pt and sample images
under data/screenshots/. It reports simple latency statistics.
"""
import statistics
import sys
import time
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
for package_src in (
    ROOT / "packages" / "shared" / "src",
    ROOT / "packages" / "activity-classifier" / "src",
):
    sys.path.insert(0, str(package_src))

MODEL = ROOT / "models" / "activity_classifier_best.pt"
SAMPLES_DIR = ROOT / "data" / "screenshots"
ImageArray = NDArray[np.uint8]


def find_images(directory: Path) -> list[Path]:
    return (
        list(directory.rglob("*.png"))
        + list(directory.rglob("*.jpg"))
        + list(directory.rglob("*.jpeg"))
    )


def main() -> None:
    import cv2
    from activity_classifier.inference import ActivityClassifier

    images = find_images(SAMPLES_DIR)
    if not images:
        print("No sample images found in data/screenshots. Place a few sample PNG/JPGs there.")
        return

    classifier = ActivityClassifier(checkpoint_path=MODEL if MODEL.exists() else None, device="cpu")
    if MODEL.exists():
        print("Loading checkpoint:", MODEL)
        classifier.load_checkpoint(MODEL)
    else:
        print("No model checkpoint found; using heuristic fallback (faster but lower fidelity)")

    # warmup
    raw_frame = cv2.imread(str(images[0]))
    if raw_frame is None:
        raise RuntimeError(f"Unable to read sample image: {images[0]}")
    frame = cast(ImageArray, raw_frame)
    classifier.predict(frame)

    runs = 20
    times: list[float] = []
    for i in range(runs):
        raw_frame = cv2.imread(str(images[i % len(images)]))
        if raw_frame is None:
            continue
        frame = cast(ImageArray, raw_frame)
        t0 = time.time()
        _ = classifier.predict(frame)
        t1 = time.time()
        times.append(t1 - t0)
    if not times:
        raise RuntimeError("No readable sample images were available for benchmarking")

    times_sorted = sorted(times)
    print(f"runs: {runs}")
    p95_index = max(0, int(len(times_sorted) * 0.95) - 1)
    print(
        f"min: {min(times_sorted):.4f}s  "
        f"median: {statistics.median(times_sorted):.4f}s  "
        f"mean: {statistics.mean(times_sorted):.4f}s  "
        f"p95: {times_sorted[p95_index]:.4f}s"
    )


if __name__ == "__main__":
    main()
