from __future__ import annotations

from typing import Any

__all__ = ["ActivityClassifier", "ClassificationResult", "ScreenActivityNet"]


def __getattr__(name: str) -> Any:
    if name in {"ActivityClassifier", "ClassificationResult"}:
        from activity_classifier.inference import ActivityClassifier, ClassificationResult

        return {"ActivityClassifier": ActivityClassifier, "ClassificationResult": ClassificationResult}[name]
    if name == "ScreenActivityNet":
        from activity_classifier.model import ScreenActivityNet

        return ScreenActivityNet
    raise AttributeError(name)
