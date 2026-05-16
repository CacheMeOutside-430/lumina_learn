from __future__ import annotations

from ai_core.providers import InferenceProvider, ReasoningRequest


class ContextSummarizer:
    def __init__(self, provider: InferenceProvider) -> None:
        self.provider = provider

    async def summarize(self, text: str) -> str:
        if not text.strip():
            return ""
        prompt = (
            "Summarize this learning context in one concise sentence. "
            "Keep technical details that would help a tutor respond later.\n\n"
            f"{text[:4000]}"
        )
        return await self.provider.generate(ReasoningRequest(prompt=prompt, max_new_tokens=90))
