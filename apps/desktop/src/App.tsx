import {
  AssistantOverlay,
  type ConnectionStatus,
  type FrameAnalysis,
  type ModelCheckpoint,
  type RuntimeMetrics,
  type SystemStatus,
  type TrainingMetric
} from "@lumina/overlay-ui";
import { useEffect, useMemo, useRef, useState } from "react";
import { captureFrame, stopCapture } from "./lib/capture";
import { RealtimeClient } from "./lib/realtime";

const rawWsUrl = import.meta.env.VITE_BACKEND_WS ?? "ws://127.0.0.1:8765/ws/realtime?user_id=local-user";
const httpUrl =
  import.meta.env.VITE_BACKEND_HTTP ??
  rawWsUrl.replace(/^ws:/, "http:").replace(/^wss:/, "https:").replace(/\/ws\/realtime.*$/, "");
const apiKey = import.meta.env.VITE_API_KEY as string | undefined;
const wsUrl = withApiKey(rawWsUrl, apiKey);
const userId = userIdFromWsUrl(rawWsUrl);

export function App() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [analysis, setAnalysis] = useState<FrameAnalysis | null>(null);
  const [metrics, setMetrics] = useState<RuntimeMetrics | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [checkpoints, setCheckpoints] = useState<ModelCheckpoint[]>([]);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [recentEvents, setRecentEvents] = useState<FrameAnalysis[]>([]);
  const [paused, setPaused] = useState(true);
  const [backpressure, setBackpressure] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const clientRef = useRef<RealtimeClient | null>(null);

  const client = useMemo(
    () =>
      new RealtimeClient(wsUrl, {
        onOpen: () => {
          setStatus("online");
          setLastError(null);
        },
        onClose: () => setStatus("offline"),
        onError: (message) => {
          setStatus("error");
          setLastError(message);
        },
        onAnalysis: (value) => {
          setAnalysis(value);
          setBackpressure(0);
        },
        onBackpressure: (queued) => setBackpressure(queued),
        onSession: () => undefined
      }),
    []
  );

  useEffect(() => {
    clientRef.current = client;
    client.connect();
    return () => client.close();
  }, [client]);

  useEffect(() => {
    let cancelled = false;
    const targetFps = Math.max(0.2, Math.min(systemStatus?.frame_processing_fps ?? 1, 5));
    const baseDelayMs = Math.max(250, Math.round(1000 / targetFps));
    async function loop() {
      while (!cancelled) {
        if (!paused && clientRef.current?.isOpen()) {
          try {
            const frame = await captureFrame();
            clientRef.current.sendFrame(frame);
          } catch (error) {
            setLastError(error instanceof Error ? error.message : String(error));
          }
        }
        const overloadDelayMs = backpressure > 0 ? 1000 : 0;
        await new Promise((resolve) => window.setTimeout(resolve, baseDelayMs + overloadDelayMs));
      }
    }
    void loop();
    return () => {
      cancelled = true;
    };
  }, [paused, status, systemStatus?.frame_processing_fps, backpressure]);

  useEffect(() => () => stopCapture(), []);

  async function refreshOperationalData(): Promise<void> {
    try {
      const [runtime, statusPayload, checkpointPayload, trainingPayload, eventPayload] = await Promise.all([
        apiGet<RuntimeMetrics>("/metrics"),
        apiGet<SystemStatus>("/system/status"),
        apiGet<ModelCheckpoint[]>("/models/checkpoints"),
        apiGet<TrainingMetric[]>("/training/metrics?limit=100"),
        apiGet<FrameAnalysis[]>(`/users/${encodeURIComponent(userId)}/events?limit=20`)
      ]);
      setMetrics(runtime);
      setSystemStatus(statusPayload);
      setCheckpoints(checkpointPayload);
      setTrainingMetrics(trainingPayload);
      setRecentEvents(eventPayload);
    } catch (error) {
      setLastError(error instanceof Error ? error.message : String(error));
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function poll() {
      while (!cancelled) {
        await refreshOperationalData();
        await new Promise((resolve) => window.setTimeout(resolve, 5000));
      }
    }
    void poll();
    return () => {
      cancelled = true;
    };
  }, []);

  async function markSuggestionIrrelevant(frameId: string): Promise<void> {
    const response = await fetch(`${httpUrl}/frames/${encodeURIComponent(frameId)}/feedback`, {
      method: "POST",
      headers: apiHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ corrected_activity: null, reason: "not relevant" })
    });
    if (!response.ok) {
      throw new Error(`Feedback failed with HTTP ${response.status}`);
    }
  }

  return (
    <>
      <AssistantOverlay
        analysis={analysis}
        status={status}
        paused={paused}
        backpressure={backpressure}
        metrics={metrics}
        systemStatus={systemStatus}
        checkpoints={checkpoints}
        trainingMetrics={trainingMetrics}
        recentEvents={recentEvents}
        onTogglePaused={() => setPaused((value) => !value)}
        onRefreshOperationalData={() => void refreshOperationalData()}
        onMarkSuggestionIrrelevant={markSuggestionIrrelevant}
      />
      {lastError && (
        <div className="fixed bottom-3 left-3 max-w-lg rounded-md border border-red-500/40 bg-red-950 px-3 py-2 text-xs text-red-100">
          {lastError}
        </div>
      )}
    </>
  );
}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${httpUrl}${path}`, { headers: apiHeaders() });
  if (!response.ok) {
    throw new Error(`GET ${path} failed with HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

function apiHeaders(extra: Record<string, string> = {}): Record<string, string> {
  return apiKey ? { ...extra, Authorization: `Bearer ${apiKey}` } : extra;
}

function withApiKey(url: string, key?: string): string {
  if (!key) return url;
  try {
    const parsed = new URL(url);
    parsed.searchParams.set("api_key", key);
    return parsed.toString();
  } catch {
    return url;
  }
}

function userIdFromWsUrl(url: string): string {
  try {
    return new URL(url).searchParams.get("user_id") || "local-user";
  } catch {
    return "local-user";
  }
}
