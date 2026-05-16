# Implementation Plan

## Stage 1: Foundation

- Establish a monorepo with isolated Python packages and a Tauri desktop app.
- Use typed shared contracts for WebSocket and persistence payloads.
- Keep frame capture, OCR, ML inference, memory, and tutoring as separate modules.

## Stage 2: Realtime Runtime

- Desktop captures frames locally and streams JPEG bytes over WebSockets.
- Backend applies backpressure with bounded async queues.
- Each frame becomes a `FrameAnalysis` event containing OCR text, classifier probabilities, educational context, distraction score, and tutoring suggestions.

## Stage 3: ML Pipelines

- CNN classifier trains on labeled screenshots with activity and educational context labels.
- OCR uses EasyOCR by default with Tesseract fallback support.
- Embeddings use sentence-transformer models and persist into Chroma or FAISS-backed vector memory.
- RAG retrieves recent and semantically relevant context before generating tutoring suggestions with a local transformer model.

## Stage 4: Product UX

- React/Tailwind overlay shows current state, suggestion cards, OCR snippets, and session analytics.
- Tauri handles native capture and starts from a minimal shell with clear extension points.

## Stage 5: Production Hardening

- Add GPU workers and model serving once usage outgrows in-process inference.
- Move SQLite to PostgreSQL while preserving repository contracts.
- Add privacy controls, telemetry opt-in, encrypted local storage, and per-user model cache management.
