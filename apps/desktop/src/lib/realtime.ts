import type { FrameAnalysis } from "@lumina/overlay-ui";
import type { CapturedFrame } from "./capture";

export interface RealtimeHandlers {
  onOpen: () => void;
  onClose: () => void;
  onError: (message: string) => void;
  onAnalysis: (analysis: FrameAnalysis) => void;
  onBackpressure: (queued: number) => void;
  onSession: (sessionId: string) => void;
}

export interface RealtimeClientOptions {
  reconnect?: boolean;
  reconnectBaseMs?: number;
  reconnectMaxMs?: number;
  maxBufferedBytes?: number;
}

export class RealtimeClient {
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private reconnectAttempts = 0;
  private shouldReconnect = false;
  private readonly options: Required<RealtimeClientOptions>;

  constructor(
    private readonly url: string,
    private readonly handlers: RealtimeHandlers,
    options: RealtimeClientOptions = {}
  ) {
    this.options = {
      reconnect: options.reconnect ?? true,
      reconnectBaseMs: options.reconnectBaseMs ?? 750,
      reconnectMaxMs: options.reconnectMaxMs ?? 8000,
      maxBufferedBytes: options.maxBufferedBytes ?? 2_000_000
    };
  }

  connect(): void {
    if (this.socket?.readyState === WebSocket.OPEN || this.socket?.readyState === WebSocket.CONNECTING) {
      return;
    }
    this.shouldReconnect = true;
    this.socket = new WebSocket(this.url);
    this.socket.binaryType = "arraybuffer";
    this.socket.onopen = () => {
      this.reconnectAttempts = 0;
      this.handlers.onOpen();
    };
    this.socket.onclose = () => {
      this.socket = null;
      this.handlers.onClose();
      this.scheduleReconnect();
    };
    this.socket.onerror = () => this.handlers.onError("WebSocket connection failed");
    this.socket.onmessage = (event) => this.handleMessage(event);
  }

  close(): void {
    this.shouldReconnect = false;
    if (this.reconnectTimer !== null) {
      window.clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.socket?.close();
    this.socket = null;
  }

  isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  sendFrame(frame: CapturedFrame): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return false;
    }
    if (this.socket.bufferedAmount > this.options.maxBufferedBytes) {
      return false;
    }
    const metadata = {
      type: "frame.metadata",
      payload: {
        frame_id: crypto.randomUUID(),
        width: frame.width,
        height: frame.height,
        monitor_id: frame.monitorId,
        mime_type: "image/jpeg",
        captured_at: new Date().toISOString()
      }
    };
    this.socket.send(JSON.stringify(metadata));
    this.socket.send(frame.bytes);
    return true;
  }

  private handleMessage(event: MessageEvent<string>): void {
    let envelope: { type?: unknown; payload?: unknown };
    try {
      envelope = JSON.parse(event.data) as { type?: unknown; payload?: unknown };
    } catch {
      this.handlers.onError("Backend sent invalid JSON");
      return;
    }
    const payload = asRecord(envelope.payload);
    if (envelope.type === "frame.analysis") {
      if (isFrameAnalysis(envelope.payload)) {
        this.handlers.onAnalysis(envelope.payload);
      } else {
        this.handlers.onError("Backend sent an invalid analysis payload");
      }
    } else if (envelope.type === "queue.backpressure") {
      this.handlers.onBackpressure(Number(payload.queued ?? 0));
    } else if (envelope.type === "session.created") {
      this.handlers.onSession(String(payload.session_id));
    } else if (envelope.type === "error") {
      this.handlers.onError(String(payload.message ?? "Unknown backend error"));
    }
  }

  private scheduleReconnect(): void {
    if (!this.options.reconnect || !this.shouldReconnect || this.reconnectTimer !== null) {
      return;
    }
    const delay = Math.min(
      this.options.reconnectMaxMs,
      this.options.reconnectBaseMs * 2 ** this.reconnectAttempts
    );
    this.reconnectAttempts += 1;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delay);
  }
}

function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function isFrameAnalysis(value: unknown): value is FrameAnalysis {
  const payload = asRecord(value);
  const analytics = asRecord(payload.analytics);
  return (
    typeof payload.session_id === "string" &&
    typeof payload.frame_id === "string" &&
    typeof payload.processed_at === "string" &&
    typeof payload.ocr_text === "string" &&
    typeof payload.activity === "string" &&
    typeof payload.activity_confidence === "number" &&
    typeof payload.activity_probabilities === "object" &&
    typeof payload.education_context === "string" &&
    typeof payload.education_confidence === "number" &&
    typeof payload.distraction_score === "number" &&
    Array.isArray(payload.suggestions) &&
    typeof analytics.focus_score === "number" &&
    typeof analytics.distraction_score === "number"
  );
}
