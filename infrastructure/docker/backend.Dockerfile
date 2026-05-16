FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/apps/backend/src:/app/packages/shared/src:/app/packages/ai-core/src:/app/packages/vision/src:/app/packages/memory/src:/app/packages/activity-classifier/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY apps ./apps
COPY packages ./packages
COPY models ./models

RUN addgroup --system lumina \
    && adduser --system --ingroup lumina --home /app lumina \
    && mkdir -p /app/data \
    && chown -R lumina:lumina /app/data /app/models

EXPOSE 8765
USER lumina

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health/live', timeout=3).read()"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8765"]
