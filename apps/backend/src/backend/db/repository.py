from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from backend.db.models import FrameEvent, FrameFeedback, UserSession
from shared.schemas import FrameAnalysis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class SessionRepository:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    async def create_session(self, user_id: str) -> UserSession:
        session = UserSession(id=str(uuid4()), user_id=user_id)
        async with self.sessionmaker() as db:
            db.add(session)
            await db.commit()
            await db.refresh(session)
        return session

    async def close_session(self, session_id: str) -> None:
        async with self.sessionmaker() as db:
            session = await db.get(UserSession, session_id)
            if session:
                session.ended_at = datetime.now(UTC)
                await db.commit()

    async def save_analysis(self, analysis: FrameAnalysis) -> None:
        async with self.sessionmaker() as db:
            statement = select(FrameEvent).where(FrameEvent.frame_id == analysis.frame_id)
            result = await db.execute(statement)
            event = result.scalars().first()
            if event is None:
                event = FrameEvent(session_id=analysis.session_id, frame_id=analysis.frame_id)
                db.add(event)
            event.created_at = analysis.processed_at
            event.ocr_text = analysis.ocr_text[:8000]
            event.activity = analysis.activity.value
            event.activity_confidence = analysis.activity_confidence
            event.education_context = analysis.education_context.value
            event.education_confidence = analysis.education_confidence
            event.distraction_score = analysis.distraction_score
            event.payload = analysis.model_dump(mode="json")
            await db.commit()

    async def recent_events(self, user_id: str, limit: int = 100) -> list[FrameEvent]:
        async with self.sessionmaker() as db:
            statement = (
                select(FrameEvent)
                .join(UserSession)
                .where(UserSession.user_id == user_id)
                .order_by(FrameEvent.created_at.desc())
                .limit(limit)
            )
            result = await db.execute(statement)
            return list(result.scalars().all())

    async def save_feedback(
        self,
        frame_id: str,
        corrected_activity: str | None,
        reason: str | None,
    ) -> bool:
        async with self.sessionmaker() as db:
            statement = select(FrameEvent).where(FrameEvent.frame_id == frame_id)
            result = await db.execute(statement)
            event = result.scalars().first()
            if not event:
                return False
            feedback = FrameFeedback(frame_event_id=event.id, corrected_activity=corrected_activity, reason=reason)
            db.add(feedback)
            await db.commit()
            return True
