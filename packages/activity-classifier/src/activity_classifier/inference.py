from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from shared.schemas import ActivityLabel, EducationLabel
from torchvision import transforms

from activity_classifier.labels import (
    ACTIVITY_LABELS,
    EDUCATION_LABELS,
    INDEX_TO_ACTIVITY,
    INDEX_TO_EDUCATION,
)
from activity_classifier.model import ScreenActivityNet

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True)
class ClassificationResult:
    activity: ActivityLabel
    activity_confidence: float
    education_context: EducationLabel
    education_confidence: float
    activity_probabilities: dict[str, float]
    education_probabilities: dict[str, float]


class ActivityClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: str | torch.device,
        image_size: int = 224,
        allow_heuristic_fallback: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.model: ScreenActivityNet | None = None
        self._model_lock = Lock()
        if self.checkpoint_path is not None and not self.checkpoint_path.exists():
            if not self.allow_heuristic_fallback:
                raise FileNotFoundError(f"Activity checkpoint does not exist: {self.checkpoint_path}")
            self.checkpoint_path = None
        if self.checkpoint_path is None and not self.allow_heuristic_fallback:
            raise RuntimeError("A trained activity checkpoint is required when heuristic fallback is disabled")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        with self._model_lock:
            model = ScreenActivityNet().to(self.device)
            checkpoint: Any = torch.load(path, map_location=self.device, weights_only=True)
            state_dict = (
                checkpoint.get("model_state_dict", checkpoint)
                if isinstance(checkpoint, dict)
                else checkpoint
            )
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.checkpoint_path = path

    @torch.inference_mode()
    def predict_batch(self, frames: Sequence[ImageArray]) -> list[ClassificationResult]:
        if not frames:
            return []
        model = self._ensure_model()
        if model is None:
            return [self._heuristic_result(frame) for frame in frames]
        tensors = torch.stack([self._prepare_frame(frame) for frame in frames]).to(self.device)
        activity_logits, education_logits = model(tensors)
        activity_probs = torch.softmax(activity_logits, dim=-1).detach().cpu()
        education_probs = torch.softmax(education_logits, dim=-1).detach().cpu()
        return [
            self._to_result(activity_probs[index], education_probs[index])
            for index in range(activity_probs.shape[0])
        ]

    def predict(self, frame: ImageArray) -> ClassificationResult:
        return self.predict_batch([frame])[0]

    def _prepare_frame(self, frame: ImageArray) -> torch.Tensor:
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[-1] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.ndim == 3 and frame.shape[-1] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported frame shape for classification: {frame.shape}")
        image = Image.fromarray(rgb)
        return cast(torch.Tensor, self.transform(image))

    def _ensure_model(self) -> ScreenActivityNet | None:
        if self.model is None and self.checkpoint_path is not None:
            with self._model_lock:
                if self.model is None and self.checkpoint_path is not None:
                    path = self.checkpoint_path
                    model = ScreenActivityNet().to(self.device)
                    checkpoint: Any = torch.load(path, map_location=self.device, weights_only=True)
                    state_dict = (
                        checkpoint.get("model_state_dict", checkpoint)
                        if isinstance(checkpoint, dict)
                        else checkpoint
                    )
                    model.load_state_dict(state_dict)
                    model.eval()
                    self.model = model
        return self.model

    @staticmethod
    def _to_result(
        activity_probs: torch.Tensor,
        education_probs: torch.Tensor,
    ) -> ClassificationResult:
        activity_index = int(torch.argmax(activity_probs).item())
        education_index = int(torch.argmax(education_probs).item())
        return ClassificationResult(
            activity=INDEX_TO_ACTIVITY[activity_index],
            activity_confidence=float(activity_probs[activity_index].item()),
            education_context=INDEX_TO_EDUCATION[education_index],
            education_confidence=float(education_probs[education_index].item()),
            activity_probabilities={
                INDEX_TO_ACTIVITY[i].value: float(activity_probs[i].item())
                for i in range(activity_probs.shape[0])
            },
            education_probabilities={
                INDEX_TO_EDUCATION[i].value: float(education_probs[i].item())
                for i in range(education_probs.shape[0])
            },
        )

    @staticmethod
    def _heuristic_result(frame: ImageArray) -> ClassificationResult:
        is_color = frame.ndim == 3 and frame.shape[-1] == 3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if is_color else frame
        gray = gray.astype(np.uint8, copy=False)
        edges = cv2.Canny(gray, 60, 160)
        edge_density = float((edges > 0).mean())
        brightness = float(gray.mean() / 255.0)
        contrast = float(gray.std() / 128.0)
        bright_ratio = float((gray > 215).mean())

        if is_color:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = float(hsv[..., 1].mean() / 255.0)
            hue = hsv[..., 0]
            green_ratio = float(((hue > 38) & (hue < 88) & (hsv[..., 1] > 70)).mean())
            blue_ratio = float(((hue > 90) & (hue < 132) & (hsv[..., 1] > 60)).mean())
        else:
            saturation = 0.0
            green_ratio = 0.0
            blue_ratio = 0.0

        activity = ActivityLabel.UNKNOWN
        activity_confidence = 0.36
        education_context = EducationLabel.GENERAL
        education_confidence = 0.44

        if edge_density < 0.018 and contrast < 0.25:
            activity = ActivityLabel.IDLE
            activity_confidence = 0.62
        elif green_ratio > 0.18 and saturation > 0.34:
            activity = ActivityLabel.GAMING
            activity_confidence = 0.6
        elif edge_density > 0.11 and brightness < 0.58:
            activity = ActivityLabel.CODING
            activity_confidence = 0.58
            education_context = EducationLabel.PROGRAMMING
            education_confidence = 0.52
        elif bright_ratio > 0.48 and edge_density > 0.05 and saturation < 0.34:
            activity = ActivityLabel.READING
            activity_confidence = 0.54
            education_context = EducationLabel.RESEARCH
            education_confidence = 0.5
        elif blue_ratio > 0.08 and edge_density > 0.04:
            activity = ActivityLabel.BROWSER
            activity_confidence = 0.5
        elif edge_density > 0.075 and bright_ratio > 0.24:
            activity = ActivityLabel.WRITING
            activity_confidence = 0.48
            education_context = EducationLabel.NOTE_TAKING
            education_confidence = 0.48

        return ClassificationResult(
            activity=activity,
            activity_confidence=activity_confidence,
            education_context=education_context,
            education_confidence=education_confidence,
            activity_probabilities=_label_probabilities(ACTIVITY_LABELS, activity, activity_confidence),
            education_probabilities=_label_probabilities(
                EDUCATION_LABELS,
                education_context,
                education_confidence,
            ),
        )


def _label_probabilities(
    labels: tuple[ActivityLabel, ...] | tuple[EducationLabel, ...],
    selected: ActivityLabel | EducationLabel,
    confidence: float,
) -> dict[str, float]:
    remainder = max(1.0 - confidence, 0.0)
    baseline = remainder / max(len(labels) - 1, 1)
    return {label.value: confidence if label == selected else baseline for label in labels}
