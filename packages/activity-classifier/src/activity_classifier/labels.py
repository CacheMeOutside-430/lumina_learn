from __future__ import annotations

from shared.schemas import ActivityLabel, EducationLabel

ACTIVITY_LABELS: tuple[ActivityLabel, ...] = (
    ActivityLabel.CODING,
    ActivityLabel.READING,
    ActivityLabel.WRITING,
    ActivityLabel.VIDEO,
    ActivityLabel.BROWSER,
    ActivityLabel.MESSAGING,
    ActivityLabel.GAMING,
    ActivityLabel.IDLE,
    ActivityLabel.UNKNOWN,
)

EDUCATION_LABELS: tuple[EducationLabel, ...] = (
    EducationLabel.PROGRAMMING,
    EducationLabel.MATH,
    EducationLabel.SCIENCE,
    EducationLabel.LANGUAGE,
    EducationLabel.RESEARCH,
    EducationLabel.EXAM_PREP,
    EducationLabel.NOTE_TAKING,
    EducationLabel.GENERAL,
)

ACTIVITY_TO_INDEX = {label.value: index for index, label in enumerate(ACTIVITY_LABELS)}
INDEX_TO_ACTIVITY = dict(enumerate(ACTIVITY_LABELS))
EDUCATION_TO_INDEX = {label.value: index for index, label in enumerate(EDUCATION_LABELS)}
INDEX_TO_EDUCATION = dict(enumerate(EDUCATION_LABELS))
