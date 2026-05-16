from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from backend.db.base import create_engine, create_sessionmaker, init_db
from backend.db.repository import SessionRepository
from shared.schemas import ActivityLabel, EducationLabel, FrameAnalysis, StudyAnalytics


@pytest.mark.asyncio
async def test_session_repository_persists_and_reads_recent_events(tmp_path: Path) -> None:
    db_path = tmp_path / "lumina-test.sqlite3"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path.as_posix()}")
    sessionmaker = create_sessionmaker(engine)
    await init_db(engine)
    repository = SessionRepository(sessionmaker)

    session = await repository.create_session("user-1")
    analysis = FrameAnalysis(
        session_id=session.id,
        frame_id="frame-1",
        processed_at=datetime(2026, 5, 15, tzinfo=UTC),
        ocr_text="Visible lesson text",
        activity=ActivityLabel.CODING,
        activity_confidence=0.9,
        activity_probabilities={ActivityLabel.CODING.value: 0.9},
        education_context=EducationLabel.PROGRAMMING,
        education_confidence=0.8,
        distraction_score=0.1,
        analytics=StudyAnalytics(
            focus_score=0.9,
            distraction_score=0.1,
            active_context=EducationLabel.PROGRAMMING,
            activity=ActivityLabel.CODING,
        ),
    )

    await repository.save_analysis(analysis)
    events = await repository.recent_events("user-1", limit=10)
    await repository.close_session(session.id)
    await engine.dispose()

    assert len(events) == 1
    assert events[0].payload["frame_id"] == "frame-1"
    assert events[0].ocr_text == "Visible lesson text"
