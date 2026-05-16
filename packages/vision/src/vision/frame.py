from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from shared.schemas import ClientFrameMetadata

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ScreenFrame:
    image_bgr: ImageArray
    metadata: ClientFrameMetadata
    received_at: datetime


@dataclass(frozen=True)
class FrameDiff:
    changed_ratio: float
    mean_delta: float
