from __future__ import annotations

from ai_core.embeddings import EmbeddingService
from ai_core.tutoring import MemoryHit

from memory_store.security import redact_sensitive_text
from memory_store.vector_store import MemoryRecord, VectorStore


class MemoryRagRetriever:
    def __init__(self, embeddings: EmbeddingService, store: VectorStore) -> None:
        self.embeddings = embeddings
        self.store = store

    async def remember(
        self,
        user_id: str,
        session_id: str,
        text: str,
        metadata: dict[str, str | int | float | bool],
    ) -> None:
        redacted_text = redact_sensitive_text(text)
        if not redacted_text.strip():
            return
        memory_text = redacted_text[:4000]
        embedding = await self.embeddings.embed_text(memory_text)
        await self.store.add(
            MemoryRecord(
                user_id=user_id,
                session_id=session_id,
                text=memory_text,
                embedding=embedding,
                metadata=metadata,
            )
        )

    async def search(self, user_id: str, query: str, limit: int = 5) -> list[MemoryHit]:
        if not query.strip():
            return []
        embedding = await self.embeddings.embed_text(query[:4000])
        results = await self.store.search(user_id=user_id, embedding=embedding, limit=limit)
        return [
            MemoryHit(text=result.text, score=result.score, metadata=result.metadata)
            for result in results
        ]
