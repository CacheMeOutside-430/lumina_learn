from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import cv2
import mss
import numpy as np
from shared.schemas import ClientFrameMetadata


@dataclass(frozen=True)
class CapturedFrame:
    metadata: ClientFrameMetadata
    encoded: bytes


class MssScreenCapture:
    def __init__(self, monitor_index: int = 1, jpeg_quality: int = 72) -> None:
        self.monitor_index = monitor_index
        self.jpeg_quality = jpeg_quality

    async def frames(self, fps: float) -> AsyncIterator[CapturedFrame]:
        interval = 1.0 / max(fps, 0.1)
        with mss.mss() as screen:
            monitor = screen.monitors[self.monitor_index]
            while True:
                raw = await asyncio.to_thread(screen.grab, monitor)
                image = np.asarray(raw)
                image_bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                ok, buffer = cv2.imencode(
                    ".jpg",
                    image_bgr,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("Failed to JPEG-encode captured frame")
                metadata = ClientFrameMetadata(
                    width=int(monitor["width"]),
                    height=int(monitor["height"]),
                    monitor_id=str(self.monitor_index),
                    captured_at=datetime.now(UTC),
                )
                yield CapturedFrame(metadata=metadata, encoded=buffer.tobytes())
                await asyncio.sleep(interval)
