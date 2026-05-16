from __future__ import annotations

import asyncio
from dataclasses import dataclass

import cv2
import numpy as np
from activity_classifier.inference import ActivityClassifier
from ai_core.context import ContextReasoner, DistractionScorer
from ai_core.device import choose_device
from ai_core.embeddings import EmbeddingService
from ai_core.providers import CloudInferenceProvider, InferenceProvider, LocalTransformerProvider
from ai_core.tutoring import RagTutor
from backend.core.config import Settings
from memory_store.rag import MemoryRagRetriever
from memory_store.vector_store import ChromaVectorStore, FaissVectorStore, VectorStore
from vision.ocr import EasyOcrEngine, OcrEngine, RapidOcrEngine, TesseractOcrEngine
from vision.pipeline import VisionPipeline


@dataclass
class InferenceContainer:
    vision: VisionPipeline
    classifier: ActivityClassifier
    embeddings: EmbeddingService
    context: ContextReasoner
    distraction: DistractionScorer
    memory: MemoryRagRetriever
    tutor: RagTutor
    provider: InferenceProvider


async def build_inference_container(settings: Settings) -> InferenceContainer:
    device = choose_device(settings.enable_gpu)
    cpu_optimized = settings.cpu_optimized_enabled(device.type)
    if settings.require_activity_checkpoint:
        if not settings.activity_checkpoint:
            raise RuntimeError("ACTIVITY_CHECKPOINT is required in this environment")
        if not settings.activity_checkpoint.exists():
            raise RuntimeError(f"Activity checkpoint does not exist: {settings.activity_checkpoint}")
    ocr = _build_ocr(settings, gpu_enabled=device.type == "cuda", cpu_optimized=cpu_optimized)
    if cpu_optimized and isinstance(ocr, RapidOcrEngine):
        warmup_ocr = np.full((240, 640, 3), 255, dtype=np.uint8)
        cv2.putText(warmup_ocr, "warmup", (32, 140), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        await asyncio.to_thread(ocr.read, warmup_ocr)
    vision = VisionPipeline(
        ocr,
        inference_max_side=settings.inference_max_side,
        ocr_interval_seconds=settings.ocr_interval_seconds,
        ocr_start_delay_seconds=settings.ocr_start_delay_seconds,
        ocr_every_n_frames=settings.ocr_every_n_frames,
        min_ocr_changed_ratio=settings.ocr_min_changed_ratio,
        ocr_max_stale_seconds=settings.ocr_max_stale_seconds,
    )
    classifier = ActivityClassifier(
        settings.activity_checkpoint,
        device=device,
        allow_heuristic_fallback=settings.allow_heuristic_classifier
        and not settings.require_activity_checkpoint,
    )
    if classifier.checkpoint_path is not None and classifier.checkpoint_path.exists():
        await asyncio.to_thread(classifier.load_checkpoint, classifier.checkpoint_path)
        warmup_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        await asyncio.to_thread(classifier.predict, warmup_frame)
    embeddings = EmbeddingService(
        settings.embedding_model,
        enable_gpu=settings.enable_gpu,
        dimension=settings.embedding_dimension,
    )
    vector_store = _build_vector_store(settings, embeddings.dimension)
    memory = MemoryRagRetriever(embeddings, vector_store)
    provider = _build_provider(settings)
    return InferenceContainer(
        vision=vision,
        classifier=classifier,
        embeddings=embeddings,
        context=ContextReasoner(embeddings),
        distraction=DistractionScorer(),
        memory=memory,
        tutor=RagTutor(provider=provider, memory=memory),
        provider=provider,
    )


def _build_ocr(settings: Settings, *, gpu_enabled: bool, cpu_optimized: bool) -> OcrEngine:
    languages = settings.ocr_language_list
    engine = settings.ocr_engine.lower()
    if engine == "auto":
        if cpu_optimized and RapidOcrEngine.available():
            engine = "rapidocr"
        elif cpu_optimized and TesseractOcrEngine.available():
            engine = "tesseract"
        else:
            engine = "easyocr"
    if engine == "rapidocr":
        return RapidOcrEngine(
            max_side=settings.ocr_max_side,
            roi_enabled=settings.ocr_roi_enabled,
            threads=settings.ocr_threads,
        )
    if engine == "tesseract":
        language = "eng" if languages[0] == "en" else languages[0]
        return TesseractOcrEngine(
            language=language,
            max_side=settings.ocr_max_side,
            roi_enabled=settings.ocr_roi_enabled,
        )
    return EasyOcrEngine(
        languages,
        gpu=gpu_enabled,
        max_side=settings.ocr_max_side,
        roi_enabled=settings.ocr_roi_enabled,
    )


def _build_vector_store(settings: Settings, embedding_dimension: int) -> VectorStore:
    if settings.vector_backend.lower() == "faiss":
        return FaissVectorStore(settings.faiss_index_path, dimension=embedding_dimension)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(settings.chroma_path)


def _build_provider(settings: Settings) -> InferenceProvider:
    if not settings.local_inference:
        if not settings.cloud_inference_endpoint or not settings.cloud_inference_api_key:
            raise RuntimeError("Cloud inference requires CLOUD_INFERENCE_ENDPOINT and API key")
        return CloudInferenceProvider(
            endpoint=settings.cloud_inference_endpoint,
            api_key=settings.cloud_inference_api_key,
        )
    return LocalTransformerProvider(settings.reasoning_model, enable_gpu=settings.enable_gpu)


async def close_inference_container(container: InferenceContainer) -> None:
    await container.vision.close()
    close = getattr(container.provider, "aclose", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result
