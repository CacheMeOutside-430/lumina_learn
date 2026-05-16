from __future__ import annotations

from backend.db.repository import SessionRepository
from fastapi import APIRouter, HTTPException, Path, Request
from pydantic import BaseModel, Field
from shared.schemas import ActivityLabel

router = APIRouter()


class FeedbackPayload(BaseModel):
    corrected_activity: ActivityLabel | None = None
    reason: str | None = Field(default=None, max_length=1000)


@router.post("/frames/{frame_id}/feedback")
async def submit_feedback(
    request: Request,
    payload: FeedbackPayload,
    frame_id: str = Path(min_length=1, max_length=128),
) -> dict[str, str]:
    repository: SessionRepository = request.app.state.repository
    saved = await repository.save_feedback(
        frame_id=frame_id,
        corrected_activity=payload.corrected_activity.value if payload.corrected_activity else None,
        reason=payload.reason,
    )
    if not saved:
        raise HTTPException(status_code=404, detail="Frame analysis was not found")
    return {"status": "ok"}
