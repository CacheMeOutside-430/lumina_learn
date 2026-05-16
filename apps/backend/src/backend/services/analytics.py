from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from shared.schemas import ActivityLabel, EducationLabel, StudyAnalytics


@dataclass
class _SessionCounters:
    last_seen: datetime
    study_seconds: float = 0.0
    distraction_seconds: float = 0.0


class StudyAnalyticsTracker:
    def __init__(self) -> None:
        self._sessions: dict[str, _SessionCounters] = {}

    def update(
        self,
        session_id: str,
        activity: ActivityLabel,
        education_context: EducationLabel,
        distraction_score: float,
        now: datetime | None = None,
    ) -> StudyAnalytics:
        timestamp = now or datetime.now(UTC)
        counters = self._sessions.setdefault(session_id, _SessionCounters(last_seen=timestamp))
        delta = min(max((timestamp - counters.last_seen).total_seconds(), 0.0), 10.0)
        counters.last_seen = timestamp
        if distraction_score >= 0.58:
            counters.distraction_seconds += delta
        else:
            counters.study_seconds += delta
        total = max(counters.study_seconds + counters.distraction_seconds, 1.0)
        focus_score = counters.study_seconds / total
        return StudyAnalytics(
            focus_score=float(focus_score),
            distraction_score=distraction_score,
            study_seconds=int(counters.study_seconds),
            distraction_seconds=int(counters.distraction_seconds),
            active_context=education_context,
            activity=activity,
        )
