from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.services.analytics import StudyAnalyticsTracker
from shared.schemas import ActivityLabel, EducationLabel


def test_study_analytics_tracks_focus_and_distraction_time() -> None:
    tracker = StudyAnalyticsTracker()
    start = datetime(2026, 5, 15, tzinfo=UTC)

    first = tracker.update(
        session_id="session-1",
        activity=ActivityLabel.CODING,
        education_context=EducationLabel.PROGRAMMING,
        distraction_score=0.2,
        now=start,
    )
    second = tracker.update(
        session_id="session-1",
        activity=ActivityLabel.MESSAGING,
        education_context=EducationLabel.GENERAL,
        distraction_score=0.8,
        now=start + timedelta(seconds=6),
    )

    assert first.study_seconds == 0
    assert second.distraction_seconds == 6
    assert second.study_seconds == 0
    assert second.focus_score == 0.0
