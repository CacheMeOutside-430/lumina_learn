from __future__ import annotations

from backend.core.telemetry import RuntimeMetrics


def test_runtime_metrics_records_session_frame_and_backpressure() -> None:
    metrics = RuntimeMetrics()

    metrics.session_opened()
    metrics.frame_processed(25.0)
    metrics.backpressure(3)
    metrics.session_closed()

    snapshot = metrics.snapshot()

    assert snapshot["active_sessions"] == 0
    assert snapshot["sessions_opened"] == 1
    assert snapshot["sessions_closed"] == 1
    assert snapshot["frames_processed"] == 1
    assert snapshot["backpressure_events"] == 1
    assert snapshot["max_queue_depth"] == 3
    assert snapshot["average_frame_processing_ms"] == 25.0
