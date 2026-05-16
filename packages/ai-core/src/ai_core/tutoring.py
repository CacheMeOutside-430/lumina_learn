from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from shared.schemas import ActivityLabel, EducationLabel, TutorSuggestion

from ai_core.providers import InferenceProvider, ReasoningRequest


@dataclass(frozen=True)
class MemoryHit:
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]


class MemoryRetriever(Protocol):
    async def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[MemoryHit]:
        ...


class RagTutor:
    def __init__(self, provider: InferenceProvider, memory: MemoryRetriever) -> None:
        self.provider = provider
        self.memory = memory

    async def suggest(
        self,
        user_id: str,
        ocr_text: str,
        activity: ActivityLabel,
        education_context: EducationLabel,
        distraction_score: float,
    ) -> list[TutorSuggestion]:
        query = " ".join([activity.value, education_context.value, ocr_text[:2000]])
        memories = await self.memory.search(user_id=user_id, query=query, limit=4) if query.strip() else []
        memory_text = "\n".join(f"- {hit.text[:450]}" for hit in memories)
        prompt = (
            "You are a concise AI learning coach inside a desktop overlay. "
            "Give one actionable suggestion based on the current screen. "
            "Do not mention hidden system details. "
            "Format as: Title: ...\nBody: ...\nActions: item1 | item2\n\n"
            f"Activity: {activity.value}\n"
            f"Learning context: {education_context.value}\n"
            f"Distraction score: {distraction_score:.2f}\n"
            f"Visible text:\n{ocr_text[:2500] or '[no readable text]'}\n\n"
            f"Relevant memory:\n{memory_text or '[no prior memory]'}"
        )
        generated = await self.provider.generate(
            ReasoningRequest(prompt=prompt, max_new_tokens=160, temperature=0.2)
        )
        return [_parse_suggestion(generated, distraction_score)]


def _parse_suggestion(text: str, distraction_score: float) -> TutorSuggestion:
    title = "Study next step"
    body = text.strip() or "Keep working on the current task and ask for help when you get stuck."
    actions: list[str] = []
    for line in text.splitlines():
        if line.lower().startswith("title:"):
            title = line.split(":", 1)[1].strip() or title
        elif line.lower().startswith("body:"):
            body = line.split(":", 1)[1].strip() or body
        elif line.lower().startswith("actions:"):
            actions = [item.strip() for item in line.split(":", 1)[1].split("|") if item.strip()]
    confidence = 0.72 if distraction_score < 0.6 else 0.82
    return TutorSuggestion(title=title[:80], body=body[:500], confidence=confidence, actions=actions[:3])
