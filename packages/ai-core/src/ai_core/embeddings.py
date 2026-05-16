from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ai_core.device import choose_device

FloatArray = NDArray[np.float32]


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        enable_gpu: bool = True,
        normalize: bool = True,
        dimension: int = 384,
    ) -> None:
        self.device = choose_device(enable_gpu)
        self.model_name = model_name
        self._model: Any | None = None
        self.normalize = normalize
        self._dimension = dimension
        self._lock = asyncio.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> FloatArray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        async with self._lock:
            model = await asyncio.to_thread(self._load_model)
            embeddings = await asyncio.to_thread(
                model.encode,
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return cast(FloatArray, embeddings.astype(np.float32, copy=False))

    async def embed_text(self, text: str) -> FloatArray:
        return cast(FloatArray, (await self.embed_texts([text]))[0])

    @staticmethod
    def cosine_similarity(query: FloatArray, candidates: FloatArray) -> FloatArray:
        if candidates.size == 0:
            return np.empty((0,), dtype=np.float32)
        query_norm = query / max(float(np.linalg.norm(query)), 1e-8)
        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True).clip(min=1e-8)
        return cast(FloatArray, candidates / candidate_norms @ query_norm)

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=str(self.device))
            dimension = self._model.get_sentence_embedding_dimension()
            if dimension is None:
                raise RuntimeError(f"Embedding model did not report a dimension: {self.model_name}")
            self._dimension = int(dimension)
        return self._model
