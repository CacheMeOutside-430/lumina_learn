from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vision.frame import FrameDiff

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class OcrPreprocessResult:
    image_rgb: ImageArray
    roi_bbox: tuple[int, int, int, int]
    scale: float


def decode_image_bytes(payload: bytes) -> ImageArray:
    buffer = np.frombuffer(payload, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode frame bytes as an image")
    return cast(ImageArray, image)


def resize_for_inference(image_bgr: ImageArray, max_side: int = 1280) -> ImageArray:
    resized, _ = resize_with_scale(image_bgr, max_side=max_side)
    return resized


def resize_with_scale(image_bgr: ImageArray, max_side: int = 1280) -> tuple[ImageArray, float]:
    height, width = image_bgr.shape[:2]
    side = max(height, width)
    if side <= max_side:
        return image_bgr, 1.0
    scale = max_side / float(side)
    target = (int(width * scale), int(height * scale))
    resized = cast(ImageArray, cv2.resize(image_bgr, target, interpolation=cv2.INTER_AREA))
    return resized, scale


def compute_frame_diff(previous_bgr: ImageArray | None, current_bgr: ImageArray) -> FrameDiff:
    if previous_bgr is None:
        return FrameDiff(changed_ratio=1.0, mean_delta=255.0)
    previous = cv2.resize(previous_bgr, (128, 72), interpolation=cv2.INTER_AREA)
    current = cv2.resize(current_bgr, (128, 72), interpolation=cv2.INTER_AREA)
    delta = cv2.absdiff(previous, current)
    gray = cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY)
    changed = float((gray > 18).mean())
    return FrameDiff(changed_ratio=changed, mean_delta=float(gray.mean()))


def detect_ocr_roi(
    image_bgr: ImageArray,
    *,
    exclude_bottom_ratio: float = 0.08,
    padding_ratio: float = 0.025,
) -> tuple[int, int, int, int]:
    height, width = image_bgr.shape[:2]
    if height < 32 or width < 32:
        return (0, 0, width, height)

    usable_bottom = max(1, int(height * (1.0 - exclude_bottom_ratio)))
    search = image_bgr[:usable_bottom, :]
    search_small, scale = resize_with_scale(search, max_side=640)
    gray = cv2.cvtColor(search_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gradient = cv2.morphologyEx(
        gray,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3)),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    small_h, small_w = search_small.shape[:2]
    min_area = max(12.0, small_w * small_h * 0.00002)
    max_area = small_w * small_h * 0.35
    for contour in contours:
        x, y, box_w, box_h = cv2.boundingRect(contour)
        area = box_w * box_h
        if area < min_area or area > max_area:
            continue
        if box_w < 6 or box_h < 4:
            continue
        boxes.append((x, y, x + box_w, y + box_h))

    if not boxes:
        return (0, 0, width, usable_bottom)

    inv_scale = 1.0 / max(scale, 1e-6)
    left = int(min(box[0] for box in boxes) * inv_scale)
    top = int(min(box[1] for box in boxes) * inv_scale)
    right = int(max(box[2] for box in boxes) * inv_scale)
    bottom = int(max(box[3] for box in boxes) * inv_scale)
    padding = int(max(width, height) * padding_ratio)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(usable_bottom, bottom + padding)

    if right - left < 32 or bottom - top < 32:
        return (0, 0, width, usable_bottom)
    return (left, top, right, bottom)


def prepare_ocr_image(
    image_bgr: ImageArray,
    *,
    max_side: int = 960,
    roi_enabled: bool = True,
) -> ImageArray:
    return prepare_ocr_frame(
        image_bgr,
        max_side=max_side,
        roi_enabled=roi_enabled,
    ).image_rgb


def prepare_ocr_frame(
    image_bgr: ImageArray,
    *,
    max_side: int = 960,
    roi_enabled: bool = True,
) -> OcrPreprocessResult:
    height, width = image_bgr.shape[:2]
    roi = detect_ocr_roi(image_bgr) if roi_enabled else (0, 0, width, height)
    left, top, right, bottom = roi
    cropped = image_bgr[top:bottom, left:right]
    resized, scale = resize_with_scale(cropped, max_side=max_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.6)
    sharpened = cv2.addWeighted(enhanced, 1.45, blurred, -0.45, 0)
    image_rgb = cast(ImageArray, cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
    return OcrPreprocessResult(image_rgb=image_rgb, roi_bbox=roi, scale=scale)
