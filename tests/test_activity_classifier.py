from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from activity_classifier.inference import ActivityClassifier
from activity_classifier.labels import ACTIVITY_LABELS, EDUCATION_LABELS
from shared.schemas import ActivityLabel


def test_activity_classifier_uses_deterministic_fallback_without_checkpoint() -> None:
    classifier = ActivityClassifier(checkpoint_path=None, device="cpu")
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    result = classifier.predict(frame)

    assert result.activity in {ActivityLabel.IDLE, ActivityLabel.UNKNOWN}
    assert set(result.activity_probabilities) == {label.value for label in ACTIVITY_LABELS}
    assert set(result.education_probabilities) == {label.value for label in EDUCATION_LABELS}
    assert sum(result.activity_probabilities.values()) == pytest.approx(1.0)
    assert sum(result.education_probabilities.values()) == pytest.approx(1.0)


def test_activity_classifier_requires_checkpoint_when_fallback_is_disabled(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Activity checkpoint does not exist"):
        ActivityClassifier(
            checkpoint_path=tmp_path / "missing.pt",
            device="cpu",
            allow_heuristic_fallback=False,
        )


def test_activity_classifier_accepts_grayscale_and_bgra_frames() -> None:
    classifier = ActivityClassifier(checkpoint_path=None, device="cpu")

    gray = np.zeros((64, 64), dtype=np.uint8)
    bgra = np.zeros((64, 64, 4), dtype=np.uint8)

    assert classifier.predict(gray).activity in set(ACTIVITY_LABELS)
    assert classifier.predict(bgra).activity in set(ACTIVITY_LABELS)
