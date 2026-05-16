from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class RuntimeMetrics:
    started_at_monotonic: float = field(default_factory=time.monotonic)
    active_sessions: int = 0
    sessions_opened: int = 0
    sessions_closed: int = 0
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    frame_errors: int = 0
    backpressure_events: int = 0
    total_frame_processing_ms: float = 0.0
    total_queue_wait_ms: float = 0.0
    total_capture_lag_ms: float = 0.0
    max_queue_depth: int = 0
    latest_queue_depth: int = 0
    total_stage_ms: dict[str, float] = field(default_factory=dict)
    last_stage_ms: dict[str, float] = field(default_factory=dict)
    ocr_runs_observed: int = 0
    ocr_failures_observed: int = 0
    total_ocr_latency_ms: float = 0.0
    last_ocr_latency_ms: float | None = None
    last_ocr_age_ms: float | None = None
    last_ocr_status: str = "unknown"
    last_frame_processing_ms: float = 0.0
    _last_snapshot_wall: float = field(default_factory=time.monotonic)
    _last_snapshot_process_cpu: float = field(default_factory=time.process_time)
    _last_process_cpu_percent: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def session_opened(self) -> None:
        with self._lock:
            self.active_sessions += 1
            self.sessions_opened += 1

    def session_closed(self) -> None:
        with self._lock:
            self.active_sessions = max(0, self.active_sessions - 1)
            self.sessions_closed += 1

    def frame_received(self, queue_depth: int = 0) -> None:
        with self._lock:
            self.frames_received += 1
            self.latest_queue_depth = queue_depth
            self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def frame_processed(
        self,
        elapsed_ms: float,
        *,
        queue_wait_ms: float = 0.0,
        capture_lag_ms: float = 0.0,
        timings_ms: dict[str, float] | None = None,
        ocr_status: str | None = None,
        ocr_latency_ms: float | None = None,
        ocr_age_ms: float | None = None,
    ) -> None:
        with self._lock:
            self.frames_processed += 1
            self.total_frame_processing_ms += elapsed_ms
            self.last_frame_processing_ms = elapsed_ms
            self.total_queue_wait_ms += queue_wait_ms
            self.total_capture_lag_ms += capture_lag_ms
            if timings_ms:
                self.last_stage_ms = dict(timings_ms)
                for key, value in timings_ms.items():
                    self.total_stage_ms[key] = self.total_stage_ms.get(key, 0.0) + float(value)
            if ocr_status is not None:
                self.last_ocr_status = ocr_status
                if ocr_status == "failed":
                    self.ocr_failures_observed += 1
            if ocr_latency_ms is not None:
                if self.last_ocr_latency_ms != ocr_latency_ms:
                    self.ocr_runs_observed += 1
                    self.total_ocr_latency_ms += ocr_latency_ms
                self.last_ocr_latency_ms = ocr_latency_ms
            self.last_ocr_age_ms = ocr_age_ms

    def frame_failed(self) -> None:
        with self._lock:
            self.frame_errors += 1

    def ocr_completed(self, status: str, elapsed_ms: float) -> None:
        with self._lock:
            self.last_ocr_status = status
            self.last_ocr_latency_ms = elapsed_ms
            self.last_ocr_age_ms = 0.0
            if status == "failed":
                self.ocr_failures_observed += 1
            else:
                self.ocr_runs_observed += 1
                self.total_ocr_latency_ms += elapsed_ms

    def frame_dropped(self, queue_depth: int = 0) -> None:
        with self._lock:
            self.frames_dropped += 1
            self.latest_queue_depth = queue_depth
            self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def backpressure(self, queue_depth: int) -> None:
        with self._lock:
            self.backpressure_events += 1
            self.latest_queue_depth = queue_depth
            self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def observe_queue_depth(self, queue_depth: int) -> None:
        with self._lock:
            self.latest_queue_depth = queue_depth
            self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_ms = self.total_frame_processing_ms / max(self.frames_processed, 1)
            average_queue_wait_ms = self.total_queue_wait_ms / max(self.frames_processed, 1)
            average_capture_lag_ms = self.total_capture_lag_ms / max(self.frames_processed, 1)
            average_ocr_ms = self.total_ocr_latency_ms / max(self.ocr_runs_observed, 1)
            averages_by_stage = {
                key: round(value / max(self.frames_processed, 1), 3)
                for key, value in self.total_stage_ms.items()
            }
            now = time.monotonic()
            cpu_now = time.process_time()
            wall_delta = max(now - self._last_snapshot_wall, 1e-6)
            cpu_delta = max(cpu_now - self._last_snapshot_process_cpu, 0.0)
            cpu_count = max(os.cpu_count() or 1, 1)
            self._last_process_cpu_percent = min(
                100.0,
                (cpu_delta / wall_delta) * 100.0 / cpu_count,
            )
            self._last_snapshot_wall = now
            self._last_snapshot_process_cpu = cpu_now
            uptime_seconds = now - self.started_at_monotonic
            processed_fps = self.frames_processed / max(uptime_seconds, 1e-6)
            return {
                "uptime_seconds": round(uptime_seconds, 3),
                "active_sessions": self.active_sessions,
                "sessions_opened": self.sessions_opened,
                "sessions_closed": self.sessions_closed,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "frames_dropped": self.frames_dropped,
                "frame_errors": self.frame_errors,
                "backpressure_events": self.backpressure_events,
                "max_queue_depth": self.max_queue_depth,
                "latest_queue_depth": self.latest_queue_depth,
                "average_frame_processing_ms": round(average_ms, 3),
                "last_frame_processing_ms": round(self.last_frame_processing_ms, 3),
                "average_queue_wait_ms": round(average_queue_wait_ms, 3),
                "average_capture_lag_ms": round(average_capture_lag_ms, 3),
                "processed_fps": round(processed_fps, 3),
                "process_cpu_percent": round(self._last_process_cpu_percent, 2),
                "ocr_runs_observed": self.ocr_runs_observed,
                "ocr_failures_observed": self.ocr_failures_observed,
                "average_ocr_latency_ms": round(average_ocr_ms, 3),
                "last_ocr_latency_ms": (
                    round(self.last_ocr_latency_ms, 3)
                    if self.last_ocr_latency_ms is not None
                    else None
                ),
                "last_ocr_age_ms": (
                    round(self.last_ocr_age_ms, 3) if self.last_ocr_age_ms is not None else None
                ),
                "last_ocr_status": self.last_ocr_status,
                "last_stage_ms": {
                    key: round(value, 3) for key, value in self.last_stage_ms.items()
                },
                "average_stage_ms": averages_by_stage,
            }
