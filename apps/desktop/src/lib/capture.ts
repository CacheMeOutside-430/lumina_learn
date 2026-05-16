import { invoke } from "@tauri-apps/api/core";

export interface CapturedFrame {
  bytes: Uint8Array;
  width: number;
  height: number;
  monitorId: string;
}

interface NativeCaptureResponse {
  base64: string;
  width: number;
  height: number;
  monitor_id: string;
}

const captureMaxSide = 1280;
const captureJpegQuality = 0.62;

let browserStream: MediaStream | null = null;
let browserVideo: HTMLVideoElement | null = null;

export async function captureFrame(): Promise<CapturedFrame> {
  if (isTauri()) {
    const frame = await invoke<NativeCaptureResponse>("capture_screen_jpeg");
    return {
      bytes: base64ToBytes(frame.base64),
      width: frame.width,
      height: frame.height,
      monitorId: frame.monitor_id
    };
  }
  return captureBrowserFrame();
}

function isTauri(): boolean {
  return "__TAURI_INTERNALS__" in window;
}

async function captureBrowserFrame(): Promise<CapturedFrame> {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("Screen capture is not supported in this browser");
  }
  if (!browserStream) {
    browserStream = await navigator.mediaDevices.getDisplayMedia({
      video: { frameRate: 2 },
      audio: false
    });
    const [track] = browserStream.getVideoTracks();
    track?.addEventListener("ended", stopCapture, { once: true });
    browserVideo = document.createElement("video");
    browserVideo.srcObject = browserStream;
    browserVideo.muted = true;
    browserVideo.playsInline = true;
    await browserVideo.play();
  }
  if (!browserVideo) {
    throw new Error("Browser capture video is not initialized");
  }
  const width = browserVideo.videoWidth || 1280;
  const height = browserVideo.videoHeight || 720;
  const [targetWidth, targetHeight] = scaledDimensions(width, height, captureMaxSide);
  const canvas = document.createElement("canvas");
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Unable to create capture canvas");
  }
  context.drawImage(browserVideo, 0, 0, targetWidth, targetHeight);
  const blob = await new Promise<Blob>((resolve, reject) =>
    canvas.toBlob(
      (value) => (value ? resolve(value) : reject(new Error("JPEG encode failed"))),
      "image/jpeg",
      captureJpegQuality
    )
  );
  return {
    bytes: new Uint8Array(await blob.arrayBuffer()),
    width: targetWidth,
    height: targetHeight,
    monitorId: "browser"
  };
}

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

export function stopCapture(): void {
  browserStream?.getTracks().forEach((track) => track.stop());
  browserStream = null;
  browserVideo?.remove();
  browserVideo = null;
}

function scaledDimensions(width: number, height: number, maxSide: number): [number, number] {
  const side = Math.max(width, height);
  if (side <= maxSide) {
    return [width, height];
  }
  const scale = maxSide / side;
  return [Math.round(width * scale), Math.round(height * scale)];
}
