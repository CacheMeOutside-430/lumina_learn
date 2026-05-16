# Lumina Learn

Production-oriented AI desktop learning assistant with a local-first realtime vision pipeline, OCR, PyTorch activity classification, vector memory, RAG tutoring, and a Tauri overlay UI.

## Implementation Plan

1. Bootstrap a modular monorepo with Python packages, a FastAPI backend, a Tauri desktop app, shared schemas, deployment assets, and architecture docs.
2. Run desktop frame capture locally and stream compressed frames to the backend over WebSockets.
3. Process frames asynchronously with OpenCV preprocessing, OCR, CNN activity classification, educational context classification, distraction scoring, embedding memory, and RAG tutoring.
4. Persist sessions, frame events, study analytics, and memory metadata in SQLite through an async repository layer that can move to PostgreSQL.
5. Keep inference providers pluggable for local transformer models and future cloud backends.
6. Ship a React/Tailwind overlay for suggestions, live OCR/context, activity confidence, and analytics.
7. Provide model training, dataset preparation, Docker, and Kubernetes-ready deployment artifacts.

## Repository Layout

```text
apps/
  backend/              FastAPI, WebSockets, persistence, orchestration
  desktop/              Tauri + React + TypeScript desktop shell
packages/
  shared/               Cross-service Pydantic schemas and event contracts
  ai-core/              Embeddings, reasoning providers, tutoring pipeline
  vision/               Screen capture, preprocessing, OCR
  memory/               Vector memory and RAG retrieval
  activity-classifier/  CNN model, training, inference, datasets
  overlay-ui/           Reusable React overlay components
infrastructure/
  docker/               Backend image
  k8s/                  Kubernetes manifests
docs/                   Architecture, ML, privacy, scaling notes
```

## Quick Start

Prerequisites:

- Python 3.12
- Node.js 20+ with Corepack
- Rust/Cargo for native Tauri packaging (`pnpm build:desktop`)
- Optional: Tesseract binary if `OCR_ENGINE=tesseract`

### Backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH="apps/backend/src;packages/shared/src;packages/ai-core/src;packages/vision/src;packages/memory/src;packages/activity-classifier/src"
uvicorn backend.main:app --host 127.0.0.1 --port 8765 --reload
```

Development defaults run without a trained classifier checkpoint by using a deterministic heuristic classifier. Production should set `REQUIRE_ACTIVITY_CHECKPOINT=true`, `ALLOW_HEURISTIC_CLASSIFIER=false`, and provide `ACTIVITY_CHECKPOINT`.

### Desktop

```powershell
corepack enable
pnpm install
pnpm --filter @lumina/desktop dev
```

The desktop app expects the backend at `ws://127.0.0.1:8765/ws/realtime` by default. The React shell opens as an operational console with live guidance, runtime metrics, checkpoint inspection, training metrics, and recent event history. Capture starts paused so screen sharing is explicit.

For a web-only production build of the React shell:

```powershell
pnpm build:web
```

For native Tauri packaging after Rust/Cargo is installed:

```powershell
pnpm build:desktop
```

## Training

This repo includes a seed screenshot generator and a one-command training script.

Generate the included screenshot-style starter dataset:

```powershell
.\scripts\generate_seed_screenshots.ps1
```

Add real screenshots over time with:

```powershell
.\scripts\capture_training_screenshot.ps1 -Activity coding -Education programming
.\scripts\capture_training_screenshot.ps1 -Activity reading -Education research
```

Then train:

```powershell
.\scripts\train_activity_model.ps1 -Epochs 12 -BatchSize 16
```

The backend `.env` is already configured to load:

```env
ACTIVITY_CHECKPOINT=./models/activity_classifier_best.pt
REQUIRE_ACTIVITY_CHECKPOINT=true
```

You can also prepare a JSONL manifest manually:

```json
{"image_path":"data/screenshots/001.jpg","activity_label":"coding","education_label":"programming"}
```

Run:

```powershell
$env:PYTHONPATH="packages/activity-classifier/src"
python -m activity_classifier.trainer --manifest data/train.jsonl --val-manifest data/val.jsonl --output runs/activity
```

## Production Notes

- Local-first capture and OCR keep raw screens on the user's machine unless cloud inference is explicitly enabled.
- WebSocket origins and CORS origins are restricted through `CORS_ORIGINS`.
- Set `API_KEY` to require bearer auth for operational APIs and WebSocket sessions. Browser WebSocket clients may pass the same value as the `api_key` query parameter.
- Frame uploads are bounded by `MAX_FRAME_BYTES`.
- OCR text is redacted for common secret patterns before vector-memory insertion.
- Chroma telemetry is disabled by the local vector-store adapter.
- Operational endpoints include `/health/live`, `/health/ready`, `/metrics`, `/system/status`, `/models/checkpoints`, and `/training/metrics`.
- Inference modules are isolated from transport and storage code to support GPU workers, batching, and model serving later.
- SQLite is the default development store; repository interfaces avoid SQLite-specific assumptions where practical.
- Chroma is the preferred vector store. FAISS support is included for embedded local indexes.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pip_audit -r requirements.txt
pnpm lint
pnpm typecheck
pnpm build:web
pnpm audit --audit-level moderate
```

## Deployment

Docker backend:

```powershell
docker compose up --build backend
```

Kubernetes manifests are in `infrastructure/k8s`. Before production rollout, place a trained classifier checkpoint at `/models/activity_classifier_best.pt` in the `lumina-models` PVC or update `ACTIVITY_CHECKPOINT` to the mounted model path.
