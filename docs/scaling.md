# Scaling Recommendations

## Millions Of Users

- Keep the desktop runtime local-first; send only summaries unless a user opts into cloud reasoning.
- Move API traffic behind a global edge gateway with regional inference clusters.
- Split realtime session orchestration from heavy inference workers.
- Batch embeddings and CNN inference on GPU workers.
- Use PostgreSQL for session/event data and Chroma server, Milvus, Weaviate, or pgvector for hosted memory.
- Add model registry/versioning so classifier outputs remain auditable.

## Performance Targets

- Capture: 1-4 FPS adaptive sampling.
- OCR: only every N frames or when visual diff exceeds a threshold.
- CNN: batch adjacent frames when queues grow.
- WebSocket payloads: send JPEG bytes upstream and compact JSON events downstream.
- Memory: store summaries and embeddings, not raw screenshots.
