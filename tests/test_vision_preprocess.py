from __future__ import annotations

import cv2
import numpy as np
import pytest
from vision.preprocess import (
    compute_frame_diff,
    decode_image_bytes,
    prepare_ocr_frame,
    resize_for_inference,
)


def test_decode_image_bytes_round_trips_jpeg() -> None:
    image = np.full((32, 48, 3), 180, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)

    assert ok
    decoded = decode_image_bytes(encoded.tobytes())

    assert decoded.shape == image.shape
    assert decoded.dtype == np.uint8


def test_decode_image_bytes_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="Unable to decode"):
        decode_image_bytes(b"not-an-image")


def test_resize_for_inference_preserves_aspect_ratio() -> None:
    image = np.zeros((1000, 2000, 3), dtype=np.uint8)

    resized = resize_for_inference(image, max_side=500)

    assert resized.shape[:2] == (250, 500)


def test_compute_frame_diff_detects_static_and_changed_frames() -> None:
    previous = np.zeros((64, 64, 3), dtype=np.uint8)
    changed = previous.copy()
    changed[:, 32:] = 255

    static_diff = compute_frame_diff(previous, previous)
    changed_diff = compute_frame_diff(previous, changed)

    assert static_diff.changed_ratio == 0.0
    assert changed_diff.changed_ratio > 0.45


def test_prepare_ocr_frame_crops_and_downscales_large_desktop() -> None:
    image = np.full((1080, 1920, 3), 240, dtype=np.uint8)
    cv2.putText(image, "Visible lesson text", (520, 360), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4)

    prepared = prepare_ocr_frame(image, max_side=640, roi_enabled=True)

    assert max(prepared.image_rgb.shape[:2]) <= 640
    assert prepared.roi_bbox[3] <= 1080
    assert prepared.scale <= 1.0
