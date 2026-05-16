#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
for package_src in (
    ROOT / "packages" / "shared" / "src",
    ROOT / "packages" / "vision" / "src",
):
    sys.path.insert(0, str(package_src))

ImageArray = NDArray[np.uint8]

def find_images(directory: Path) -> list[Path]:
    return (
        list(directory.rglob("*.png"))
        + list(directory.rglob("*.jpg"))
        + list(directory.rglob("*.jpeg"))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OCR engines on local screenshots.")
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "screenshots")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    from vision.ocr import EasyOcrEngine, OcrEngine, RapidOcrEngine, TesseractOcrEngine

    images = find_images(args.samples)[: args.limit]
    if not images:
        print(f"No images found under {args.samples}")
        return

    engines: list[tuple[str, OcrEngine]] = []
    if RapidOcrEngine.available():
        engines.append(("rapidocr-onnx", RapidOcrEngine(max_side=args.max_side, threads=args.threads)))
    engines.append(("easyocr-cpu", EasyOcrEngine(["en"], gpu=False, max_side=args.max_side)))
    if TesseractOcrEngine.available():
        engines.append(("tesseract", TesseractOcrEngine(max_side=args.max_side)))
    else:
        print("tesseract: unavailable on PATH")

    for name, engine in engines:
        timings: list[float] = []
        block_counts: list[int] = []
        for path in images:
            image = cv2.imread(str(path))
            if image is None:
                continue
            image_array = cast(ImageArray, image.astype(np.uint8, copy=False))
            started = time.perf_counter()
            blocks = engine.read(image_array)
            timings.append((time.perf_counter() - started) * 1000.0)
            block_counts.append(len(blocks))
        if not timings:
            print(f"{name}: no readable samples")
            continue
        print(
            f"{name}: runs={len(timings)} "
            f"median={statistics.median(timings):.1f}ms "
            f"mean={statistics.mean(timings):.1f}ms "
            f"max={max(timings):.1f}ms "
            f"blocks_avg={statistics.mean(block_counts):.1f}"
        )


if __name__ == "__main__":
    main()
