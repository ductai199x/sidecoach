// The backend is the Python sidecar's FastAPI server. In a browser the page is served by
// that same server, so every path is same-origin; inside the desktop shell the page comes
// from the shell's own origin and the sidecar's port (and bearer token) arrive from the
// shell, so the calls go cross-origin to 127.0.0.1:<port> (CORS is allowed for the
// shell's origins in sidecoach/ui/server.py).

import { isTauri, invoke } from "@tauri-apps/api/core";

let base = "";
let token = "";
let ready: Promise<void> | null = null;

/** Resolve where the API lives. In the shell, wait for the sidecar's port. */
export function connect(): Promise<void> {
  if (ready) return ready;
  ready = (async () => {
    if (!isTauri()) {
      reportErrors();
      return;
    }
    for (;;) {
      const port = await invoke<number | null>("sidecar_port").catch(() => null);
      if (port) {
        base = `http://127.0.0.1:${port}`;
        token = await invoke<string>("sidecar_token").catch(() => "");
        reportErrors();
        return;
      }
      await new Promise((r) => setTimeout(r, 300));
    }
  })();
  return ready;
}

let reporting = false;

/** Uncaught errors and rejections of this page go to the panel's terminal (sys channel). */
export function reportErrors(): void {
  if (reporting || typeof window === "undefined") return;
  reporting = true;
  const send = (text: string) => {
    try {
      void fetch(`${base}/api/client_log`, {
        method: "POST",
        headers: headers(true),
        body: JSON.stringify({ text: text.slice(0, 2000), page: location.pathname }),
        keepalive: true,
      }).catch(() => {});
    } catch { /* nothing to do about it */ }
  };
  window.addEventListener("error", (e) => send(`${e.message} @ ${e.filename || "?"}:${e.lineno || 0}`));
  window.addEventListener("unhandledrejection", (e) => {
    const r = e.reason as { stack?: string; message?: string } | undefined;
    send(`unhandled rejection: ${r?.stack || r?.message || String(e.reason)}`);
  });
}

/** The sidecar's origin ("" in the browser), for links that must reach it directly. */
export function apiBase(): string {
  return base;
}

function headers(json: boolean): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

export async function get<T = unknown>(path: string): Promise<T> {
  const r = await fetch(base + path, { headers: headers(false) });
  return (await r.json()) as T;
}

export async function post<T = unknown>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(base + path, {
    method: "POST",
    headers: headers(true),
    body: JSON.stringify(body ?? {}),
  });
  return (await r.json()) as T;
}

export async function put<T = unknown>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(base + path, {
    method: "PUT",
    headers: headers(true),
    body: JSON.stringify(body ?? {}),
  });
  return (await r.json()) as T;
}

/** A beacon on unload (the puzzle room's close); falls back to a fetch with keepalive. */
export function beacon(path: string): void {
  const blob = new Blob(["{}"], { type: "application/json" });
  if (!token && navigator.sendBeacon && navigator.sendBeacon(base + path, blob)) return;
  fetch(base + path, { method: "POST", headers: headers(true), body: "{}", keepalive: true }).catch(() => {});
}

// ---- the panel's controls -----------------------------------------------------------------

export type Poll = { state?: PanelState; lines?: LogLine[]; cursor?: number };

export const panel = {
  poll: (since: number) => get<Poll>(`/api/poll?since=${since | 0}`),
  health: () => get<{ ok: boolean; version: string; pid: number; running: boolean }>("/api/health"),
  start: () => post("/api/start"),
  stop: () => post("/api/stop"),
  hint: () => post("/api/hint"),
  move: () => post("/api/move"),
  line: () => post("/api/line"),
  flip: () => post("/api/flip"),
  analyze: () => post("/api/analyze"),
  ask: (text: string) => post("/api/ask", { text }),
  setVoice: (voice: string) => post("/api/voice", { voice }),
  setMuted: (muted: boolean) => post("/api/muted", { muted }),
  setListen: (mode: string) => post("/api/listen", { mode }),
  end: (outcome: string) => post("/api/end", { outcome }),
  resetProfile: () => post("/api/reset_profile"),
  clearCaches: () => post("/api/clear_caches"),
  correct: (body: { game_id: string; segment: number; ply: number; cause: string }) => post("/api/correct", body),
  exclude: (rect: { x: number; y: number; w: number; h: number }) => post("/api/exclude", rect),
  ptt: (down: boolean) => post("/api/ptt", { down }),
  dump: () => post<{ ok: boolean; path: string }>("/api/dump"),
  perf: (label: string, sample: Record<string, unknown>) => post("/api/perf", { label, sample }),
};

export const puzzle = (id: string) => {
  const p = `/api/puzzle/${encodeURIComponent(id)}`;
  return {
    open: () => post<{ ok: boolean; error?: string; next?: string[]; state?: PuzzleState }>(`${p}/open`),
    poll: (since: number) =>
      get<{ ok: boolean; closed?: boolean; state?: PuzzleState; lines?: LogLine[]; cursor?: number }>(`${p}/poll?since=${since | 0}`),
    targets: (square: string) => post<{ ok: boolean; targets: string[] }>(`${p}/targets`, { square }),
    move: (uci: string) => post(`${p}/move`, { uci }),
    retract: () => post(`${p}/retract`),
    giveup: () => post(`${p}/giveup`),
    ask: (text: string) => post(`${p}/ask`, { text }),
    voice: (on: boolean) => post(`${p}/voice`, { on }),
    close: () => post(`${p}/close`),
    closePath: `${p}/close`,
  };
};

// ---- what the sidecar sends -------------------------------------------------------------------

export type Mark = { kind: "arrow" | "square" | "ring"; squares: string[]; color: string; opacity?: number | null };

export type LogLine = {
  seq?: number;
  t: number;
  chan: string;
  level: string;
  text: string;
  data?: Record<string, any> | null;
};

export type Health = {
  llm?: { calls: number; prompt: number; cached: number; completion: number; reasoning: number; cost: number | null; estimated?: number } | null;
  latency?: {
    move_to_speech_s: number | null; move_to_speech_last_s: number | null; move_to_speech_n: number;
    first_token_s: number | null; first_token_last_s: number | null; first_token_n: number;
  } | null;
  rss_mb?: number | null;
  tts_device?: string | null;
  stt_device?: string | null;
  mic_dropped?: number;
  fps?: number;
  lock?: string;
  stage?: string;
  pipeline_ms?: [number, number];
  lag_ms?: number | null;
  recording?: string;
  coach_pending?: number;
};

export type PanelState = {
  running?: boolean;
  status?: string;
  session_dir?: string | null;
  user_side?: string | null;
  ply?: number;
  following?: boolean;
  kind?: string;
  bottom?: string;
  view_flip?: boolean;
  muted?: boolean;
  fen?: string | null;
  last_move?: string | null;
  fens?: string[];
  ucis?: string[];
  moves?: string[];
  labels?: Record<string, string>;
  health?: Health;
  listen?: string;
  heard?: string;
  voices?: string[];
  voice?: string;
  clock?: { white_ms: number | null; black_ms: number | null; running: string | null } | null;
  [key: string]: unknown;
};

export type PuzzleState = {
  id: string;
  fen: string;
  bottom: string;
  your_move: boolean;
  last_move?: string | null;
  check?: string | null;
  wrong?: string | null;
  status: string;
  message?: string;
  remaining?: number;
  misses?: number;
  motif?: string | null;
  rating?: number;
  url?: string;
  voice?: boolean;
  closed?: boolean;
  llm?: { calls: number; prompt: number; completion: number; cost: number | null } | null;
};
