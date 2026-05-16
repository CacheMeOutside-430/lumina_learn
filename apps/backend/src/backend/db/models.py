from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class UserSession(Base):
    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    events: Mapped[list[FrameEvent]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
    )


class FrameEvent(Base):
    __tablename__ = "frame_events"
    __table_args__ = (UniqueConstraint("frame_id", name="uq_frame_events_frame_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("user_sessions.id"), index=True)
    frame_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    activity: Mapped[str] = mapped_column(String(64), index=True)
    activity_confidence: Mapped[float] = mapped_column(Float)
    education_context: Mapped[str] = mapped_column(String(64), index=True)
    education_confidence: Mapped[float] = mapped_column(Float)
    distraction_score: Mapped[float] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)

    session: Mapped[UserSession] = relationship(back_populates="events")
    feedbacks: Mapped[list[FrameFeedback]] = relationship(back_populates="event", cascade="all, delete-orphan")


class FrameFeedback(Base):
    __tablename__ = "frame_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    frame_event_id: Mapped[int] = mapped_column(ForeignKey("frame_events.id"), index=True)
    corrected_activity: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    event: Mapped[FrameEvent] = relationship(back_populates="feedbacks")
