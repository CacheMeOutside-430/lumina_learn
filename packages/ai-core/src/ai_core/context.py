from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray
from shared.schemas import ActivityLabel, EducationLabel

from ai_core.embeddings import EmbeddingService

FloatArray = NDArray[np.float32]

EDUCATION_PROTOTYPES: dict[EducationLabel, str] = {
    EducationLabel.PROGRAMMING: "software engineering code debugging programming IDE compiler terminal",
    EducationLabel.MATH: "mathematics algebra calculus equation geometry proof problem solving",
    EducationLabel.SCIENCE: "science physics biology chemistry research paper experiment data",
    EducationLabel.LANGUAGE: "language learning grammar vocabulary translation reading writing",
    EducationLabel.RESEARCH: "academic research article citation literature review PDF notes",
    EducationLabel.EXAM_PREP: "exam practice flashcards quiz assessment revision study plan",
    EducationLabel.NOTE_TAKING: "notes outline lecture summary concept map writing",
    EducationLabel.GENERAL: "general learning study education browser reading",
}


@dataclass(frozen=True)
class ContextClassification:
    label: EducationLabel
    confidence: float
    probabilities: dict[str, float]


class ContextReasoner:
    def __init__(self, embeddings: EmbeddingService) -> None:
        self.embeddings = embeddings
        self._prototype_matrix: FloatArray | None = None
        self._labels = list(EDUCATION_PROTOTYPES.keys())

    async def classify(self, text: str, neural_prior: dict[str, float] | None = None) -> ContextClassification:
        if not text.strip() and neural_prior:
            label, confidence = max(neural_prior.items(), key=lambda item: item[1])
            return ContextClassification(
                label=EducationLabel(label),
                confidence=float(confidence),
                probabilities=neural_prior,
            )
        if self._prototype_matrix is None:
            self._prototype_matrix = await self.embeddings.embed_texts(
                [EDUCATION_PROTOTYPES[label] for label in self._labels]
            )
        text_vector = await self.embeddings.embed_text(text or "empty desktop")
        similarities = self.embeddings.cosine_similarity(text_vector, self._prototype_matrix)
        similarity_probs = _softmax(similarities)
        probabilities = {
            label.value: float(similarity_probs[index]) for index, label in enumerate(self._labels)
        }
        if neural_prior:
            for label, value in neural_prior.items():
                probabilities[label] = 0.55 * probabilities.get(label, 0.0) + 0.45 * float(value)
            total = sum(probabilities.values()) or 1.0
            probabilities = {key: value / total for key, value in probabilities.items()}
        best_label, confidence = max(probabilities.items(), key=lambda item: item[1])
        return ContextClassification(
            label=EducationLabel(best_label),
            confidence=float(confidence),
            probabilities=probabilities,
        )


class DistractionScorer:
    """Model-output scorer that uses classifier probabilities as neural evidence."""

    DISTRACTION_WEIGHTS = {
        ActivityLabel.MESSAGING.value: 0.95,
        ActivityLabel.GAMING.value: 1.0,
        ActivityLabel.IDLE.value: 0.65,
        ActivityLabel.VIDEO.value: 0.35,
        ActivityLabel.BROWSER.value: 0.25,
        ActivityLabel.CODING.value: -0.35,
        ActivityLabel.READING.value: -0.25,
        ActivityLabel.WRITING.value: -0.2,
        ActivityLabel.UNKNOWN.value: 0.1,
    }

    def score(
        self,
        activity_probabilities: dict[str, float],
        education_confidence: float,
        ocr_text: str,
    ) -> float:
        weighted = sum(
            self.DISTRACTION_WEIGHTS.get(label, 0.0) * probability
            for label, probability in activity_probabilities.items()
        )
        sparse_screen_penalty = 0.12 if len(ocr_text.strip()) < 12 else 0.0
        study_signal = 0.28 * education_confidence
        return float(np.clip(0.5 + weighted - study_signal + sparse_screen_penalty, 0.0, 1.0))


def _softmax(values: FloatArray) -> FloatArray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return cast(FloatArray, (exp / max(float(exp.sum()), 1e-8)).astype(np.float32, copy=False))
