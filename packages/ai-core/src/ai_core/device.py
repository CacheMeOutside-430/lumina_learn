from __future__ import annotations

import torch


def choose_device(enable_gpu: bool = True) -> torch.device:
    if enable_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if enable_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
