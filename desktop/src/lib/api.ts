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

export async function del<T = unknown>(path: string): Promise<T> {
  const r = await fetch(base + path, { method: "DELETE", headers: headers(false) });
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
  setRecord: (on: boolean) => post<{ ok: boolean; on: boolean; running: boolean; folder: string | null }>("/api/record", { on }),
  setListen: (mode: string) => post("/api/listen", { mode }),
  end: (outcome: string) => post("/api/end", { outcome }),
  resetProfile: () => post("/api/reset_profile"),
  clearCaches: () => post("/api/clear_caches"),
  correct: (body: { game_id: string; segment: number; ply: number; cause: string }) => post("/api/correct", body),
  exclude: (rect: { x: number; y: number; w: number; h: number }) => post("/api/exclude", rect),
  ptt: (down: boolean) => post("/api/ptt", { down }),
  cancelDictation: () => post<{ ok: boolean; text: string }>("/api/dictation/cancel"),
  dump: () => post<{ ok: boolean; path: string }>("/api/dump"),
  perf: (label: string, sample: Record<string, unknown>) => post("/api/perf", { label, sample }),
};

export const puzzle = (id: string) => {
  const p = `/api/puzzle/${encodeURIComponent(id)}`;
  return {
    open: () => post<{ ok: boolean; error?: string; next?: string[]; state?: PuzzleState }>(`${p}/open`),
    poll: (since: number) =>
      get<{ ok: boolean; closed?: boolean; state?: PuzzleState; lines?: LogLine[]; cursor?: number }>(`${p}/poll?since=${since | 0}`),
    // Played at once, answered with the state they leave.
    move: (uci: string) => post<{ ok: boolean; state?: PuzzleState }>(`${p}/move`, { uci }),
    retract: () => post<{ ok: boolean; state?: PuzzleState }>(`${p}/retract`),
    giveup: () => post<{ ok: boolean; state?: PuzzleState }>(`${p}/giveup`),
    ask: (text: string) => post(`${p}/ask`, { text }),
    voice: (on: boolean) => post(`${p}/voice`, { on }),
    close: () => post(`${p}/close`),
    closePath: `${p}/close`,
  };
};

// ---- Game Review: the library and a review (profile/library.py, ui/review_room.py) -----------

export type LibraryGame = {
  key: string;
  source: "watched" | "chesscom" | "lichess" | "pgn";
  added_at: number;
  played_at: number | null;
  white: string | null;
  black: string | null;
  white_elo: number | null;
  black_elo: number | null;
  result: string | null;
  time_class: string | null;
  user_side: "white" | "black" | null;
  reviewed: boolean;
  accuracy: { white: number | null; black: number | null } | null;
  performance: { white: number | null; black: number | null } | null;
};

export const games = {
  list: () => get<{ games: LibraryGame[]; users: { chesscom: string; lichess: string } }>("/api/games"),
  bringIn: (source: "pgn" | "chesscom" | "lichess", text = "", months = 3) =>
    post<{ ok: boolean; added?: number; keys?: string[]; error?: string }>("/api/games/import", { source, text, months }),
  remove: (key: string) => del<{ ok: boolean }>(`/api/games/${encodeURIComponent(key)}`),
};

export type Eval = { cp: number | null; mate: number | null; result: string | null };

/** A move of a line the board plays: in SAN and UCI, and the position it leaves. */
export type LineStep = { san: string; uci: string; fen: string };

export type ReviewMove = {
  ply: number;
  san: string;
  side: "white" | "black";
  uci: string;
  from: string;
  to: string;
  fen_before: string;
  fen_after: string;
  label: string | null;
  accuracy: number | null;
  loss: number;
  best_uci: string | null;
  best_san: string | null;
  best_line: LineStep[];
  follow_line: LineStep[];
  eval: Eval;
  white_score: number;
  tags: string[];
  opening: string | null;
  clock_ms: number | null;
  spent_ms: number | null;
};

export type PhaseGrade = { phase: "opening" | "middlegame" | "endgame"; moves: number; accuracy: number; grade: string };

export type SideSummary = {
  side: "white" | "black";
  name: string | null;
  rating: number | null;
  accuracy: number | null;
  counts: Record<string, number>;
  phases: PhaseGrade[];
  performance: { rating: number; low: number; high: number } | null;
};

export type GameSummary = {
  white: SideSummary;
  black: SideSummary;
  opening: string | null;
  last_book_ply: number | null;
  key_moments: { ply: number; side: "white" | "black"; kind: string }[];
  graph: number[];
  shape: string | null;
};

export type ReviewState = {
  key: string;
  status: "starting" | "analysing" | "ready" | "failed";
  progress: { done: number; total: number };
  error: string | null;
  strength: string;
  game: Omit<LibraryGame, "key" | "added_at" | "reviewed" | "accuracy" | "performance">;
  start_fen: string | null;
  summary: GameSummary | null;
  moves: ReviewMove[];
  closed: boolean;
};

export const review = (key: string) => {
  const p = `/api/review/${encodeURIComponent(key)}`;
  type Polled = { ok: boolean; closed?: boolean; error?: string; state?: ReviewState; lines?: LogLine[]; cursor?: number };
  return {
    open: (strength = "standard") => post<Polled>(`${p}/open`, { strength }),
    poll: (since: number) => get<Polled>(`${p}/poll?since=${since | 0}`),
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

export type Captions = {
  on: boolean;
  keep: boolean;
  text: string;
  background: string;
  opacity: number;
  size: number;
  /** Raised by one at each change of their look: a sample line shows. */
  preview: number;
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
  record_wanted?: boolean;
  recording_dir?: string | null;
  fen?: string | null;
  last_move?: string | null;
  fens?: string[];
  ucis?: string[];
  moves?: string[];
  labels?: Record<string, string>;
  health?: Health;
  listen?: string;
  /** Whether speech recognition loaded; false leaves the listening button out. */
  stt_ready?: boolean;
  ptt?: boolean;
  ptt_key?: string;
  heard?: string;
  /** What is being said, as recognised so far; `live` while the microphone still captures
   * it, `sends_at` (ms since the epoch) once it is about to be sent on its own. */
  dictation?: { text: string; live: boolean; sends_at: number | null };
  voices?: string[];
  voice?: string;
  clock?: { white_ms: number | null; black_ms: number | null; running: string | null } | null;
  /** The captions' settings, the words given to the voice (numbered, the last few), whether
   * it is speaking now, and the board the vision follows, on the screen (routes/captions). */
  captions?: Captions;
  caption_lines?: [number, string, boolean?][];
  speaking?: boolean;
  board_screen?: [number, number, number, number] | null;
  [key: string]: unknown;
};

/** A position of a puzzle: its placement, the move that made it (none for the start), the
 * king in check, and whether the solver played it. */
export type PuzzlePosition = { fen: string; uci: string | null; check: string | null; by_user: boolean };

export type PuzzleState = {
  id: string;
  /** Counts every change: an answer with a lower one is older than one shown. */
  version: number;
  /** The start, then one per ply. */
  positions: PuzzlePosition[];
  bottom: string;
  your_move: boolean;
  /** Where each of the solver's pieces may go, while it is their move. */
  legal: Record<string, string[]>;
  wrong?: string | null;
  status: string;
  message?: string;
  remaining?: number;
  misses?: number;
  motif?: string | null;
  rating?: number;
  url?: string;
  voice?: boolean;
  // Whether the room has a voice at all; the voice button shows only when it has.
  voice_ready?: boolean;
  closed?: boolean;
  llm?: { calls: number; prompt: number; completion: number; cost: number | null } | null;
};
