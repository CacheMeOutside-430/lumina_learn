export type ActivityLabel =
  | "coding"
  | "reading"
  | "writing"
  | "video"
  | "browser"
  | "messaging"
  | "gaming"
  | "idle"
  | "unknown";

export type EducationLabel =
  | "programming"
  | "math"
  | "science"
  | "language"
  | "research"
  | "exam_prep"
  | "note_taking"
  | "general";

export interface TutorSuggestion {
  title: string;
  body: string;
  confidence: number;
  actions: string[];
}

export interface StudyAnalytics {
  focus_score: number;
  distraction_score: number;
  study_seconds: number;
  distraction_seconds: number;
  active_context: EducationLabel;
  activity: ActivityLabel;
}

export interface FrameAnalysis {
  session_id: string;
  frame_id: string;
  processed_at: string;
  ocr_text: string;
  ocr_status: string;
  ocr_refreshed_at: string | null;
  ocr_latency_ms: number | null;
  ocr_age_ms: number | null;
  activity: ActivityLabel;
  activity_confidence: number;
  activity_probabilities: Record<string, number>;
  education_context: EducationLabel;
  education_confidence: number;
  distraction_score: number;
  suggestions: TutorSuggestion[];
  analytics: StudyAnalytics;
  timings_ms: Record<string, number>;
  queue_wait_ms: number;
  capture_lag_ms: number;
  cpu_optimized: boolean;
}

export interface RuntimeMetrics {
  uptime_seconds: number;
  active_sessions: number;
  sessions_opened: number;
  sessions_closed: number;
  frames_received: number;
  frames_processed: number;
  frames_dropped: number;
  frame_errors: number;
  backpressure_events: number;
  max_queue_depth: number;
  latest_queue_depth: number;
  average_frame_processing_ms: number;
  last_frame_processing_ms: number;
  average_queue_wait_ms: number;
  average_capture_lag_ms: number;
  processed_fps: number;
  process_cpu_percent: number;
  ocr_runs_observed: number;
  ocr_failures_observed: number;
  average_ocr_latency_ms: number;
  last_ocr_latency_ms: number | null;
  last_ocr_age_ms: number | null;
  last_ocr_status: string;
  last_stage_ms: Record<string, number>;
  average_stage_ms: Record<string, number>;
}

export interface SystemStatus {
  env: string;
  production: boolean;
  device: string;
  gpu_enabled: boolean;
  gpu_requested: boolean;
  cpu_optimized_active: boolean;
  cpu_optimized_mode: string;
  database: string;
  vector_backend: string;
  ocr_engine: string;
  ocr_engine_resolved: string;
  ocr_interval_seconds: number;
  ocr_start_delay_seconds: number;
  ocr_every_n_frames: number;
  ocr_max_side: number;
  ocr_roi_enabled: boolean;
  ocr_threads: number;
  ocr_min_changed_ratio: number;
  inference_max_side: number;
  latest_frame_only: boolean;
  cpu_lightweight_context: boolean;
  cpu_lightweight_tutor: boolean;
  local_inference: boolean;
  reasoning_model: string;
  embedding_model: string;
  activity_checkpoint: string | null;
  activity_checkpoint_loaded: boolean;
  heuristic_classifier_allowed: boolean;
  require_activity_checkpoint: boolean;
  max_frame_queue: number;
  max_frame_bytes: number;
  frame_processing_fps: number;
}

export interface ModelCheckpoint {
  name: string;
  path: string;
  bytes: number;
  modified_at: string;
  is_active: boolean;
  epoch: number | null;
  image_size: number | null;
  train_metrics: Record<string, number>;
  val_metrics: Record<string, number>;
  model_config: Record<string, string | number | boolean>;
  readable: boolean;
  error: string | null;
}

export interface TrainingMetric {
  epoch: number;
  train: Record<string, number>;
  val: Record<string, number>;
}

export type ConnectionStatus = "connecting" | "online" | "offline" | "error";
