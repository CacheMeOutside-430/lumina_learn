from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

EmbeddingVector = NDArray[np.float32]


@dataclass(frozen=True)
class MemoryRecord:
    user_id: str
    session_id: str
    text: str
    embedding: EmbeddingVector
    metadata: dict[str, str | int | float | bool] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class MemorySearchResult:
    id: str
    text: str
    score: float
    metadata: dict[str, str | int | float | bool]


class VectorStore(Protocol):
    async def add(self, record: MemoryRecord) -> None:
        ...

    async def search(
        self,
        user_id: str,
        embedding: EmbeddingVector,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        ...


class ChromaVectorStore:
    def __init__(self, path: str | Path, collection_name: str = "learning_memory") -> None:
        import chromadb
        from chromadb.config import Settings

        self._lock = asyncio.Lock()
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(
                anonymized_telemetry=False,
                chroma_product_telemetry_impl="memory_store.chroma_telemetry.NoopTelemetry",
                chroma_telemetry_impl="memory_store.chroma_telemetry.NoopTelemetry",
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, record: MemoryRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._add_sync, record)

    def _add_sync(self, record: MemoryRecord) -> None:
        metadata = {"user_id": record.user_id, "session_id": record.session_id, **record.metadata}
        collection = cast(Any, self.collection)
        collection.add(
            ids=[record.id],
            embeddings=[cast(list[float], record.embedding.astype(float).tolist())],
            documents=[record.text],
            metadatas=[metadata],
        )

    async def search(
        self,
        user_id: str,
        embedding: EmbeddingVector,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, user_id, embedding, limit)

    def _search_sync(
        self,
        user_id: str,
        embedding: EmbeddingVector,
        limit: int,
    ) -> list[MemorySearchResult]:
        collection = cast(Any, self.collection)
        result = cast(
            dict[str, Any],
            collection.query(
                query_embeddings=[cast(list[float], embedding.astype(float).tolist())],
                n_results=limit,
                where={"user_id": user_id},
                include=["documents", "distances", "metadatas"],
            ),
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return [
            MemorySearchResult(
                id=str(ids[index]),
                text=str(documents[index]),
                score=1.0 - float(distances[index]),
                metadata=dict(metadatas[index] or {}),
            )
            for index in range(len(ids))
        ]


class FaissVectorStore:
    def __init__(self, index_path: str | Path, dimension: int) -> None:
        import faiss

        self.faiss = faiss
        self._lock = asyncio.Lock()
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(".jsonl")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index = (
            faiss.read_index(str(self.index_path))
            if self.index_path.exists()
            else faiss.IndexFlatIP(dimension)
        )
        self.records: list[dict[str, Any]] = self._load_metadata()

    async def add(self, record: MemoryRecord) -> None:
        async with self._lock:
            await asyncio.to_thread(self._add_sync, record)

    def _add_sync(self, record: MemoryRecord) -> None:
        vector = record.embedding.astype("float32")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
        self.index.add(vector.reshape(1, -1))
        payload = {
            **asdict(record),
            "embedding": None,
            "metadata": {"user_id": record.user_id, "session_id": record.session_id, **record.metadata},
        }
        self.records.append(payload)
        self._persist()

    async def search(
        self,
        user_id: str,
        embedding: EmbeddingVector,
        limit: int = 5,
    ) -> list[MemorySearchResult]:
        async with self._lock:
            return await asyncio.to_thread(self._search_sync, user_id, embedding, limit)

    def _search_sync(
        self,
        user_id: str,
        embedding: EmbeddingVector,
        limit: int,
    ) -> list[MemorySearchResult]:
        if self.index.ntotal == 0:
            return []
        vector = embedding.astype("float32")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
        scores, indices = self.index.search(vector.reshape(1, -1), min(limit * 4, self.index.ntotal))
        results: list[MemorySearchResult] = []
        for score, index in zip(scores[0], indices[0], strict=False):
            if index < 0:
                continue
            record = self.records[int(index)]
            metadata = dict(record.get("metadata") or {})
            if metadata.get("user_id") != user_id:
                continue
            results.append(
                MemorySearchResult(
                    id=str(record["id"]),
                    text=str(record["text"]),
                    score=float(score),
                    metadata=metadata,
                )
            )
            if len(results) >= limit:
                break
        return results

    def _load_metadata(self) -> list[dict[str, Any]]:
        if not self.meta_path.exists():
            return []
        with self.meta_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def _persist(self) -> None:
        self.faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
