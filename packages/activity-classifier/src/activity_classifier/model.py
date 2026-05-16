from __future__ import annotations

from typing import cast

import torch
from torch import nn

from activity_classifier.labels import ACTIVITY_LABELS, EDUCATION_LABELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.activation(x + self.net(x)))


class ScreenActivityNet(nn.Module):
    """Compact multi-head CNN for realtime screenshot classification."""

    def __init__(
        self,
        activity_classes: int = len(ACTIVITY_LABELS),
        education_classes: int = len(EDUCATION_LABELS),
        width: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResidualBlock(width),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.SiLU(inplace=True),
            ResidualBlock(width * 2),
            nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 4),
            nn.SiLU(inplace=True),
            ResidualBlock(width * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        hidden = width * 4
        self.embedding = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.SiLU())
        self.activity_head = nn.Linear(hidden, activity_classes)
        self.education_head = nn.Linear(hidden, education_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.embedding(self.features(x))
        return cast(torch.Tensor, self.activity_head(features)), cast(torch.Tensor, self.education_head(features))
