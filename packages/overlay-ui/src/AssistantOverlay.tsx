import {
  Activity,
  BarChart3,
  Brain,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  Eye,
  Gauge,
  History,
  Pause,
  Play,
  RefreshCw,
  ScanText,
  Server,
  Sparkles,
  Wifi,
  WifiOff,
} from "lucide-react";
import React from "react";
import type {
  ConnectionStatus,
  FrameAnalysis,
  ModelCheckpoint,
  RuntimeMetrics,
  SystemStatus,
  TrainingMetric,
} from "./types";

interface AssistantOverlayProps {
  analysis: FrameAnalysis | null;
  status: ConnectionStatus;
  paused: boolean;
  backpressure: number;
  metrics: RuntimeMetrics | null;
  systemStatus: SystemStatus | null;
  checkpoints: ModelCheckpoint[];
  trainingMetrics: TrainingMetric[];
  recentEvents: FrameAnalysis[];
  onTogglePaused: () => void;
  onRefreshOperationalData: () => void;
  onMarkSuggestionIrrelevant?: (frameId: string) => Promise<void>;
}

type Tab = "live" | "models" | "events";

const percent = (value: number) => `${Math.round(value * 100)}%`;
const seconds = (value: number) => `${Math.round(value)}s`;
const millis = (value: number | null | undefined) => (typeof value === "number" ? `${value.toFixed(0)}ms` : "--");
const bytes = (value: number) => {
  if (value >= 1024 * 1024) return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${value} B`;
};
const timeLabel = (value: string | null | undefined) => (value ? new Date(value).toLocaleTimeString() : "--");

export function AssistantOverlay({
  analysis,
  status,
  paused,
  backpressure,
  metrics,
  systemStatus,
  checkpoints,
  trainingMetrics,
  recentEvents,
  onTogglePaused,
  onRefreshOperationalData,
  onMarkSuggestionIrrelevant,
}: AssistantOverlayProps) {
  const [feedbackSent, setFeedbackSent] = React.useState(false);
  const [tab, setTab] = React.useState<Tab>("live");
  const suggestion = analysis?.suggestions[0];
  const online = status === "online";

  React.useEffect(() => {
    setFeedbackSent(false);
  }, [analysis?.frame_id]);

  async function sendFeedback() {
    if (!analysis || !onMarkSuggestionIrrelevant) return;
    setFeedbackSent(true);
    try {
      await onMarkSuggestionIrrelevant(analysis.frame_id);
    } catch (e) {
      console.error(e);
      setFeedbackSent(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#0a0f14] text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-4 px-4 py-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div className="flex min-w-0 items-center gap-3">
            <div className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-emerald-400 text-slate-950 shadow-sm shadow-emerald-500/20">
              <Brain size={21} aria-hidden="true" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-lg font-semibold tracking-normal">Lumina Learn</h1>
              <p className="truncate text-xs text-slate-400">
                {analysis?.education_context ?? "general"} / {analysis?.activity ?? "waiting"} /{" "}
                {systemStatus?.device ?? "device pending"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusPill status={status} />
            <button
              type="button"
              onClick={onRefreshOperationalData}
              title="Refresh"
              className="grid h-9 w-9 place-items-center rounded-md border border-slate-700 bg-slate-900 text-slate-100 hover:border-cyan-400"
            >
              <RefreshCw size={16} aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={onTogglePaused}
              title={paused ? "Resume capture" : "Pause capture"}
              className="grid h-9 w-9 place-items-center rounded-md border border-slate-700 bg-slate-900 text-slate-100 hover:border-emerald-400"
            >
              {paused ? <Play size={17} aria-hidden="true" /> : <Pause size={17} aria-hidden="true" />}
            </button>
          </div>
        </header>

        <nav className="flex gap-2 overflow-x-auto">
          <TabButton active={tab === "live"} icon={<Activity size={15} />} label="Live" onClick={() => setTab("live")} />
          <TabButton active={tab === "models"} icon={<Server size={15} />} label="Models" onClick={() => setTab("models")} />
          <TabButton active={tab === "events"} icon={<History size={15} />} label="Events" onClick={() => setTab("events")} />
        </nav>

        {tab === "live" && (
          <section className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-[1.25fr_0.75fr]">
            <div className="flex flex-col gap-4">
              <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
                <PanelTitle icon={<Sparkles size={16} />} title="Current suggestion" />
                {suggestion ? (
                  <div className="space-y-3">
                    <h2 className="text-xl font-semibold tracking-normal">{suggestion.title}</h2>
                    <p className="max-w-4xl text-sm leading-6 text-slate-300">{suggestion.body}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      {suggestion.actions.map((action) => (
                        <span key={action} className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-xs text-slate-300">
                          {action}
                        </span>
                      ))}
                      <button
                        type="button"
                        onClick={sendFeedback}
                        disabled={feedbackSent || !onMarkSuggestionIrrelevant}
                        className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1 text-xs text-slate-300 hover:border-emerald-400 disabled:opacity-60"
                      >
                        {feedbackSent ? "Sent" : "Not relevant"}
                      </button>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-slate-400">
                    {online ? "Analyzing the desktop stream." : "Start the backend to receive live guidance."}
                  </p>
                )}
              </section>

              <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
                <PanelTitle icon={<Eye size={16} />} title="OCR context" />
                <div className="mb-3 grid grid-cols-2 gap-x-4 gap-y-2 text-sm md:grid-cols-4">
                  <KeyValue label="Status" value={analysis?.ocr_status ?? metrics?.last_ocr_status ?? "--"} />
                  <KeyValue label="Age" value={millis(analysis?.ocr_age_ms ?? metrics?.last_ocr_age_ms)} />
                  <KeyValue label="OCR latency" value={millis(analysis?.ocr_latency_ms ?? metrics?.last_ocr_latency_ms)} />
                  <KeyValue label="Refreshed" value={timeLabel(analysis?.ocr_refreshed_at)} />
                </div>
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-300">
                  {analysis?.ocr_text?.trim() || "No readable screen text yet."}
                </pre>
              </section>
            </div>

            <aside className="flex flex-col gap-4">
              <div className="grid grid-cols-2 gap-3">
                <MetricCard icon={<Gauge size={17} />} label="Focus" value={analysis ? percent(analysis.analytics.focus_score) : "--"} tone="emerald" />
                <MetricCard icon={<BarChart3 size={17} />} label="Latency" value={metrics ? millis(metrics.last_frame_processing_ms) : "--"} tone="cyan" />
                <MetricCard icon={<Activity size={17} />} label="FPS" value={metrics ? (metrics.processed_fps ?? 0).toFixed(1) : "--"} tone="sky" />
                <MetricCard icon={<Cpu size={17} />} label="CPU" value={metrics ? `${(metrics.process_cpu_percent ?? 0).toFixed(0)}%` : "--"} tone="amber" />
                <MetricCard icon={<ScanText size={17} />} label="OCR" value={metrics ? millis(metrics.last_ocr_latency_ms) : "--"} tone="cyan" />
                <MetricCard icon={<Clock3 size={17} />} label="Drops" value={metrics ? String(metrics.frames_dropped ?? 0) : "--"} tone="amber" />
              </div>
              <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
                <PanelTitle icon={<Activity size={16} />} title="Classifier probabilities" />
                <ProbabilityBars values={analysis?.activity_probabilities ?? {}} highlight={analysis?.activity} />
                <div className="mt-4 border-t border-slate-800 pt-4">
                  <ProbabilityBars values={analysis ? { [analysis.education_context]: analysis.education_confidence } : {}} highlight={analysis?.education_context} />
                </div>
              </section>
              <SystemPanel systemStatus={systemStatus} metrics={metrics} />
            </aside>
          </section>
        )}

        {tab === "models" && (
          <section className="grid flex-1 grid-cols-1 gap-4 xl:grid-cols-[1fr_0.8fr]">
            <CheckpointTable checkpoints={checkpoints} />
            <TrainingPanel trainingMetrics={trainingMetrics} systemStatus={systemStatus} />
          </section>
        )}

        {tab === "events" && <EventTimeline events={recentEvents} />}
      </div>
    </main>
  );
}

function TabButton({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex h-9 items-center gap-2 rounded-md border px-3 text-sm ${
        active ? "border-emerald-400 bg-emerald-400 text-slate-950" : "border-slate-700 bg-slate-900 text-slate-300 hover:border-cyan-400"
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function StatusPill({ status }: { status: ConnectionStatus }) {
  const online = status === "online";
  return (
    <div className="flex h-9 items-center gap-2 rounded-md border border-slate-700 bg-slate-900 px-3 text-xs text-slate-300">
      {online ? <Wifi size={15} aria-hidden="true" /> : <WifiOff size={15} aria-hidden="true" />}
      {status}
    </div>
  );
}

function PanelTitle({ icon, title }: { icon: React.ReactNode; title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-200">
      {icon}
      {title}
    </div>
  );
}

function MetricCard({ icon, label, value, tone }: { icon: React.ReactNode; label: string; value: string; tone: "emerald" | "cyan" | "sky" | "amber" }) {
  const tones = {
    emerald: "text-emerald-300",
    cyan: "text-cyan-300",
    sky: "text-sky-300",
    amber: "text-amber-300",
  };
  return (
    <div className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
      <div className={`mb-2 flex items-center gap-2 text-sm ${tones[tone]}`}>
        {icon}
        {label}
      </div>
      <div className="truncate text-2xl font-semibold tracking-normal">{value}</div>
    </div>
  );
}

function ProbabilityBars({ values, highlight }: { values: Record<string, number>; highlight?: string }) {
  const entries = Object.entries(values)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8);
  if (!entries.length) return <p className="text-sm text-slate-500">No probabilities yet.</p>;
  return (
    <div className="space-y-3">
      {entries.map(([label, value]) => (
        <div key={label}>
          <div className="mb-1 flex justify-between gap-3 text-xs text-slate-400">
            <span className={label === highlight ? "text-emerald-300" : undefined}>{label}</span>
            <span>{percent(value)}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-sm bg-slate-800">
            <div className={label === highlight ? "h-full bg-emerald-400" : "h-full bg-cyan-400"} style={{ width: percent(value) }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function SystemPanel({ systemStatus, metrics }: { systemStatus: SystemStatus | null; metrics: RuntimeMetrics | null }) {
  return (
    <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
      <PanelTitle icon={<Database size={16} />} title="Runtime" />
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
        <KeyValue label="Env" value={systemStatus?.env ?? "--"} />
        <KeyValue label="Device" value={systemStatus?.device ?? "--"} />
        <KeyValue label="Mode" value={systemStatus?.cpu_optimized_active ? "CPU optimized" : "standard"} />
        <KeyValue label="DB" value={systemStatus?.database ?? "--"} />
        <KeyValue label="Vector" value={systemStatus?.vector_backend ?? "--"} />
        <KeyValue label="OCR" value={systemStatus?.ocr_engine_resolved ?? systemStatus?.ocr_engine ?? "--"} />
        <KeyValue label="OCR interval" value={systemStatus?.ocr_interval_seconds ? `${systemStatus.ocr_interval_seconds}s` : "--"} />
        <KeyValue label="Queue" value={String(metrics?.latest_queue_depth ?? systemStatus?.max_frame_queue ?? "--")} />
        <KeyValue label="Uptime" value={metrics ? seconds(metrics.uptime_seconds) : "--"} />
      </div>
    </section>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="truncate text-slate-200">{value}</div>
    </div>
  );
}

function CheckpointTable({ checkpoints }: { checkpoints: ModelCheckpoint[] }) {
  return (
    <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
      <PanelTitle icon={<Server size={16} />} title="Checkpoints" />
      <div className="overflow-auto">
        <table className="w-full min-w-[640px] border-collapse text-left text-sm">
          <thead className="text-xs uppercase text-slate-500">
            <tr className="border-b border-slate-800">
              <th className="py-2 pr-3 font-medium">Model</th>
              <th className="py-2 pr-3 font-medium">Epoch</th>
              <th className="py-2 pr-3 font-medium">Val loss</th>
              <th className="py-2 pr-3 font-medium">Accuracy</th>
              <th className="py-2 pr-3 font-medium">Size</th>
              <th className="py-2 pr-3 font-medium">State</th>
            </tr>
          </thead>
          <tbody>
            {checkpoints.map((checkpoint) => (
              <tr key={checkpoint.path} className="border-b border-slate-800/80">
                <td className="py-3 pr-3">
                  <div className="font-medium text-slate-100">{checkpoint.name}</div>
                  <div className="max-w-sm truncate text-xs text-slate-500">{checkpoint.path}</div>
                </td>
                <td className="py-3 pr-3 text-slate-300">{checkpoint.epoch ?? "--"}</td>
                <td className="py-3 pr-3 text-slate-300">{formatMetric(checkpoint.val_metrics.loss)}</td>
                <td className="py-3 pr-3 text-slate-300">{formatMetric(checkpoint.val_metrics.activity_accuracy)}</td>
                <td className="py-3 pr-3 text-slate-300">{bytes(checkpoint.bytes)}</td>
                <td className="py-3 pr-3">
                  <span className={`inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs ${checkpoint.is_active ? "border-emerald-400 text-emerald-300" : "border-slate-700 text-slate-400"}`}>
                    {checkpoint.is_active && <CheckCircle2 size={13} aria-hidden="true" />}
                    {checkpoint.readable ? (checkpoint.is_active ? "active" : "ready") : "unreadable"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!checkpoints.length && <p className="text-sm text-slate-500">No checkpoints found.</p>}
    </section>
  );
}

function TrainingPanel({ trainingMetrics, systemStatus }: { trainingMetrics: TrainingMetric[]; systemStatus: SystemStatus | null }) {
  const latest = trainingMetrics.at(-1);
  return (
    <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
      <PanelTitle icon={<BarChart3 size={16} />} title="Training" />
      <div className="mb-4 grid grid-cols-2 gap-3">
        <MetricCard icon={<Activity size={17} />} label="Epoch" value={latest ? String(latest.epoch) : "--"} tone="sky" />
        <MetricCard icon={<Gauge size={17} />} label="Val loss" value={latest ? formatMetric(latest.val.loss) : "--"} tone="amber" />
      </div>
      <Trend metrics={trainingMetrics} />
      <div className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <KeyValue label="Reasoning" value={systemStatus?.reasoning_model ?? "--"} />
        <KeyValue label="Embeddings" value={systemStatus?.embedding_model ?? "--"} />
        <KeyValue label="Checkpoint" value={systemStatus?.activity_checkpoint ?? "--"} />
        <KeyValue label="Classifier loaded" value={systemStatus?.activity_checkpoint_loaded ? "yes" : "lazy"} />
      </div>
    </section>
  );
}

function Trend({ metrics }: { metrics: TrainingMetric[] }) {
  if (metrics.length < 2) return <p className="text-sm text-slate-500">No training trend yet.</p>;
  const points = metrics.slice(-24);
  const losses = points.map((row) => row.val.loss ?? row.train.loss ?? 0);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const coords = losses
    .map((loss, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * 100;
      const y = 36 - ((loss - min) / Math.max(max - min, 1e-6)) * 32;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
  return (
    <svg viewBox="0 0 100 40" className="h-32 w-full rounded-md bg-slate-950" role="img" aria-label="Validation loss trend">
      <polyline points={coords} fill="none" stroke="#34d399" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function EventTimeline({ events }: { events: FrameAnalysis[] }) {
  return (
    <section className="rounded-md border border-slate-800 bg-slate-900/70 p-4">
      <PanelTitle icon={<History size={16} />} title="Recent events" />
      <div className="space-y-3">
        {events.map((event) => (
          <div key={event.frame_id} className="rounded-md border border-slate-800 bg-slate-950 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="font-medium text-slate-100">
                {event.activity} / {event.education_context}
              </div>
              <div className="text-xs text-slate-500">{new Date(event.processed_at).toLocaleTimeString()}</div>
            </div>
            <div className="mt-2 line-clamp-2 text-sm text-slate-400">{event.ocr_text || "No OCR text"}</div>
          </div>
        ))}
      </div>
      {!events.length && <p className="text-sm text-slate-500">No recent events yet.</p>}
    </section>
  );
}

function formatMetric(value: number | undefined) {
  return typeof value === "number" ? value.toFixed(3) : "--";
}
