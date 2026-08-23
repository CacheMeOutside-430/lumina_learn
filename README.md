# Lumina Learn

A local-first AI desktop learning assistant: watches your screen in real time, understands what you're studying, and tutors you through it — with capture, inference, and memory staying on your machine by default.

![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN%20classifier-EE4C2C?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

<!-- TODO(you): add a screenshot of the Tauri overlay console here (run backend + pnpm --filter @lumina/desktop dev), e.g. docs/screenshot.png -->

## Features

- Real-time screen capture streamed to a FastAPI backend over WebSockets
- OCR context extraction with a tunable pipeline (ROI, fps, interval, staleness)
- PyTorch CNN activity classifier — deterministic heuristic fallback when no checkpoint is trained
- Vector memory (Chroma or FAISS) + RAG tutoring grounded in your own activity
- React/Tailwind overlay: live suggestions, OCR context, activity confidence, analytics
- Pluggable inference providers — local transformers by default, cloud optional
- SQLite persistence through an async repository layer that can move to PostgreSQL
- Secret-pattern redaction before anything reaches vector memory

## Architecture

```
apps/
  desktop/    Tauri + React + TypeScript shell (capture client + console)
  backend/    FastAPI: WebSockets, orchestration, persistence, ops endpoints
packages/
  vision/               screen capture, preprocessing, OCR
  activity-classifier/  CNN model, training, inference, datasets
  memory/               vector memory and RAG retrieval
  ai-core/              embeddings, reasoning providers, tutoring pipeline
  shared/               cross-service Pydantic schemas and event contracts
infrastructure/
  docker/, k8s/         deployment artifacts
```

Frame flow: desktop captures → compresses frames over `ws://127.0.0.1:8765/ws/realtime` → backend preprocesses (OpenCV) → OCR → activity classification → memory write → tutoring response back to the overlay.

## Setup

Prerequisites: Python 3.12, Node.js 20+ with Corepack, optional Rust/Cargo for native Tauri packaging, optional Tesseract binary.

Backend:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH="apps/backend/src;packages/shared/src;packages/ai-core/src;packages/vision/src;packages/memory/src;packages/activity-classifier/src"
uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload
```

Desktop:

```powershell
corepack enable
pnpm install
pnpm --filter @lumina/desktop dev
```

Development defaults run without a trained classifier by using the heuristic classifier. For production, set `REQUIRE_ACTIVITY_CHECKPOINT=true` and provide `ACTIVITY_CHECKPOINT`. All configuration lives in `.env` (see [.env.example](.env.example) — no API key needed for local mode).

## Training

```powershell
.\scripts\generate_seed_screenshots.ps1        # starter dataset
.\scripts\capture_training_screenshot.ps1 -Activity coding -Education programming
.\scripts\train_activity_model.ps1 -Epochs 12 -BatchSize 16
```

## Security & privacy notes

- Local-first: raw screens stay on-machine unless cloud inference is explicitly enabled
- Capture starts paused; screen sharing is explicit
- Set `API_KEY` to require bearer auth on operational APIs and WebSocket sessions
- CORS/WebSocket origins restricted via `CORS_ORIGINS`; uploads bounded by `MAX_FRAME_BYTES`
- Ops endpoints: `/health/live`, `/health/ready`, `/metrics`, `/system/status`

## Status

Functional solo monorepo: the capture → OCR → classify → tutor pipeline is implemented and tested locally; k8s manifests are provided but not load-tested. Developed in bursts between other projects.

## License

[MIT](LICENSE)
