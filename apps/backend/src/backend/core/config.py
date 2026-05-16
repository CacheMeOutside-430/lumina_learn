from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    database_url: str = "sqlite+aiosqlite:///./lumina.sqlite3"
    cors_origins: str = (
        "http://localhost:1420,"
        "http://127.0.0.1:1420,"
        "http://tauri.localhost,"
        "tauri://localhost"
    )
    vector_backend: str = "chroma"
    chroma_path: Path = Path("./data/chroma")
    faiss_index_path: Path = Path("./data/faiss/activity.index")
    local_inference: bool = True
    cloud_inference_provider: str = "disabled"
    cloud_inference_endpoint: str | None = None
    cloud_inference_api_key: str | None = None
    ocr_engine: str = "easyocr"
    ocr_languages: str = "en"
    ocr_interval_seconds: float = Field(default=4.0, ge=0.5, le=120.0)
    ocr_start_delay_seconds: float = Field(default=0.2, ge=0.0, le=5.0)
    ocr_every_n_frames: int = Field(default=10, ge=1, le=10_000)
    ocr_max_side: int = Field(default=640, ge=320, le=2400)
    ocr_roi_enabled: bool = True
    ocr_threads: int = Field(default=4, ge=1, le=32)
    ocr_min_changed_ratio: float = Field(default=0.025, ge=0.0, le=1.0)
    ocr_max_stale_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    model_dir: Path = Path("./models")
    activity_checkpoint: Path | None = None
    require_activity_checkpoint: bool = False
    allow_heuristic_classifier: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, ge=1)
    reasoning_model: str = "google/flan-t5-base"
    frame_processing_fps: float = Field(default=2.0, gt=0.0, le=30.0)
    max_frame_queue: int = Field(default=8, ge=1, le=128)
    max_frame_bytes: int = Field(default=5_000_000, ge=128_000, le=25_000_000)
    inference_max_side: int = Field(default=960, ge=320, le=2400)
    latest_frame_only: bool = True
    cpu_optimized_mode: str = "auto"
    cpu_lightweight_context: bool = True
    cpu_lightweight_tutor: bool = True
    memory_write_interval_seconds: float = Field(default=10.0, ge=0.0, le=600.0)
    enable_gpu: bool = True
    log_level: str = "INFO"
    api_key: str | None = None
    rate_limit_requests: int = Field(default=120, ge=1, le=10_000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)
    training_metrics_path: Path | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def ocr_language_list(self) -> list[str]:
        languages = [item.strip() for item in self.ocr_languages.split(",") if item.strip()]
        return languages or ["en"]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}

    @property
    def resolved_training_metrics_path(self) -> Path:
        return self.training_metrics_path or (self.model_dir / "metrics.jsonl")

    def cpu_optimized_enabled(self, device_type: str) -> bool:
        mode = self.cpu_optimized_mode.strip().lower()
        if mode in {"1", "true", "yes", "on", "enabled"}:
            return True
        if mode in {"0", "false", "no", "off", "disabled"}:
            return False
        return device_type == "cpu"

    def effective_frame_queue(self, cpu_optimized: bool) -> int:
        if cpu_optimized and self.latest_frame_only:
            return min(self.max_frame_queue, 2)
        return self.max_frame_queue


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
