from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from shared.schemas import OcrBlock
from vision.pipeline import VisionPipeline


class _FakeOcr:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, image_bgr: np.ndarray) -> list[OcrBlock]:
        self.calls += 1
        time.sleep(0.05)
        return [OcrBlock(text="hello", confidence=0.9, bbox=[])]


@pytest.mark.asyncio
async def test_vision_pipeline_schedules_ocr_without_blocking_frame_result() -> None:
    ocr = _FakeOcr()
    pipeline = VisionPipeline(
        ocr,
        inference_max_side=320,
        ocr_interval_seconds=60.0,
        ocr_every_n_frames=100,
    )
    payload = _jpeg_payload()

    result = await pipeline.process(payload, frame_id="frame-1")

    assert result.ocr_status == "scheduled"
    assert result.ocr_text == ""

    await pipeline.wait_for_ocr_idle()
    cached = await pipeline.process(payload, frame_id="frame-2")

    assert ocr.calls == 1
    assert cached.ocr_text == "hello"
    assert cached.ocr_status in {"cached", "fresh"}
    await pipeline.close()


def _jpeg_payload() -> bytes:
    image = np.full((120, 220, 3), 245, dtype=np.uint8)
    cv2.putText(image, "hello", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
