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
      const status = await invoke<{ port: number | null; error: string | null }>("sidecar_status").catch(() => null);
      if (status?.port) {
        base = `http://127.0.0.1:${status.port}`;
        token = await invoke<string>("sidecar_token").catch(() => "");
        reportErrors();
        return;
      }
      // A sidecar that failed to start never serves: say so, and let a later call try again.
      if (status?.error) {
        ready = null;
        throw new Error(status.error);
      }
      await new Promise((r) => setTimeout(r, 300));
    }
  })();
  return ready;
}

/** In the shell: start the sidecar afresh (it died, or never started) and wait for it. */
export async function restartSidecar(): Promise<void> {
  if (!isTauri()) return;
  await invoke("restart_sidecar");
  ready = null;
  await connect();
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

/** In the shell: whether the sidecar now serves on another port than this window's (a
 *  Restart started a new one); this window follows it when it does. */
async function moved(): Promise<boolean> {
  if (!isTauri() || !base) return false;
  const status = await invoke<{ port: number | null }>("sidecar_status").catch(() => null);
  const now = status?.port ? `http://127.0.0.1:${status.port}` : "";
  if (!now || now === base) return false;
  base = now;
  token = await invoke<string>("sidecar_token").catch(() => token);
  return true;
}

/** A request to the sidecar. One that cannot reach it asks whether the sidecar moved: the
 *  panel's Restart starts a new one on another port, and every other window (Settings, Your
 *  games, a review, a puzzle, the captions) kept calling the old one, dead until reopened.
 *  It is sent once more to where the sidecar is now; otherwise the failure stands. */
async function send(path: string, init: () => RequestInit): Promise<Response> {
  try {
    return await fetch(base + path, init());
  } catch (e) {
    if (await moved()) return fetch(base + path, init());
    throw e;
  }
}

/** The sidecar answered, with an error: it is running (a 4xx or 5xx, a body that is not JSON). */
export class ServerError extends Error {
  readonly status: number;
  constructor(status: number, detail: string) {
    super(`Sidecoach answered an error (${status})${detail ? ": " + detail : ""}`);
    this.status = status;
  }
}

/** No answer in the time a request is given: the sidecar may be alive but stuck, or the
 *  connection hangs without closing. Told apart from an error answer and from no connection. */
export class TimedOut extends Error {
  constructor(path: string, ms: number) {
    super(`Sidecoach didn't answer in ${Math.round(ms / 1000)} s (${path.split("?")[0]})`);
  }
}

/** How long a request is given by default: the sidecar answers its routes in milliseconds, so
 *  this is only for one that hangs (the panel's one-at-a-time poll froze on stale state with no
 *  offline banner, the 2026-10-07 audit). */
export const TIMEOUT_MS = 10_000;
/** Routes that do real work before they answer (an engine search, the stats over every game,
 *  a dump, settings that reload a model, a puzzle database opened on the phone). */
export const SLOW_MS = 60_000;

/** `timeout`: ms, or null for none (a route whose time depends on the network or on the
 *  player, such as games fetched from chess.com, or a PGN saved through a dialog). */
export type RequestOpts = { timeout?: number | null; keepalive?: boolean };

/** Every request to the sidecar: sent with a timeout, its answer checked. Resolves with the
 *  JSON body of a 2xx; rejects with a ServerError for an error answer (its `detail` or `error`
 *  said, a body that is not JSON quoted), a TimedOut for none in time, or the fetch's own
 *  failure when nothing listens. */
export async function request<T = unknown>(method: string, path: string, body?: unknown, opts: RequestOpts = {}): Promise<T> {
  const ms = opts.timeout === undefined ? TIMEOUT_MS : opts.timeout;
  const hasBody = method !== "GET" && method !== "DELETE";
  let r: Response;
  try {
    r = await send(path, () => ({
      method,
      headers: headers(hasBody),
      body: hasBody ? JSON.stringify(body ?? {}) : undefined,
      keepalive: !!opts.keepalive,
      // A fresh signal for each send: a request sent again to a moved sidecar has its own time.
      signal: ms ? AbortSignal.timeout(ms) : undefined,
    }));
  } catch (e) {
    if (ms && (e as { name?: string } | null)?.name === "TimeoutError") throw new TimedOut(path, ms);
    throw e;
  }
  let text: string;
  try {
    text = await r.text();
  } catch (e) {
    if (ms && (e as { name?: string } | null)?.name === "TimeoutError") throw new TimedOut(path, ms);
    throw e;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new ServerError(r.status, text.trim().slice(0, 200));
  }
  if (!r.ok) {
    const b = parsed as { detail?: unknown; error?: unknown } | null;
    const d = typeof b?.detail === "string" ? b.detail : typeof b?.error === "string" ? b.error : "";
    throw new ServerError(r.status, d);
  }
  return parsed as T;
}

/** What the panel says when a poll got no good answer, or null for nothing to say.
 *  "stopped" only when the sidecar's process has ended, which only the shell knows (it holds
 *  the child): a live sidecar answering an error shows that error, one that does not answer in
 *  time says so, and one the shell still holds alive with the connection refused is a blip (the
 *  next poll goes on). In a browser there is no shell: a request that reached nobody says so as
 *  it is, and the next answer clears it (Tai, 2026-10-02: no count of failed polls decides that
 *  the sidecar stopped). */
export async function noAnswer(e: unknown): Promise<{ text: string; stopped: boolean } | null> {
  if (e instanceof ServerError) return { text: e.message, stopped: false };
  if (!isTauri()) return { text: e instanceof TimedOut ? "Sidecoach isn't answering" : "Can't reach Sidecoach", stopped: false };
  const status = await invoke<{ exited?: boolean }>("sidecar_status").catch(() => null);
  if (status?.exited) return { text: "Sidecoach stopped", stopped: true };
  return e instanceof TimedOut ? { text: "Sidecoach isn't answering", stopped: false } : null;
}

/** The log cursor to poll from next, given ours and the one the sidecar answered. Polls go one
 *  at a time, so the sidecar's only moves on -- unless it is a new sidecar (stopped and started
 *  again), numbering its lines afresh: then from its start, or its lines, the reply to a question
 *  among them, stayed below our old cursor and never came (the typing dots stayed for good). */
export function followCursor(ours: number, theirs: number | undefined): number {
  if (theirs === undefined) return ours;
  return theirs < ours ? 0 : theirs;
}

/** What to tell the player when a request failed, in plain words, with what to do. */
export function failureText(e: unknown): string {
  if (e instanceof ServerError) return `${e.message}. Try again; if it keeps happening, restart Sidecoach.`;
  if (e instanceof TimedOut) return "Sidecoach isn't answering. Try again in a moment.";
  return "Can't reach Sidecoach. Check it is running, then try again.";
}

export function get<T = unknown>(path: string, opts?: RequestOpts): Promise<T> {
  return request<T>("GET", path, undefined, opts);
}

export function post<T = unknown>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
  return request<T>("POST", path, body, opts);
}

/** `keepalive`: the request is sent even if the page is closing (a save on the way out). */
export function put<T = unknown>(path: string, body?: unknown, opts?: RequestOpts): Promise<T> {
  return request<T>("PUT", path, body, opts);
}

export function del<T = unknown>(path: string, opts?: RequestOpts): Promise<T> {
  return request<T>("DELETE", path, undefined, opts);
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
  /** The state alone, no log lines (the captions' window; the phone ignores `lines` and its
   *  page has no such window). */
  pollState: () => get<Poll>("/api/poll?lines=0"),
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
  resetProfile: () => post("/api/reset_profile", {}, { timeout: SLOW_MS }),
  clearCaches: () => post("/api/clear_caches", {}, { timeout: SLOW_MS }),
  correct: (body: { game_id: string; segment: number; ply: number; cause: string }) => post("/api/correct", body),
  exclude: (rect: { x: number; y: number; w: number; h: number }) => post("/api/exclude", rect),
  ptt: (down: boolean) => post("/api/ptt", { down }),
  cancelDictation: () => post<{ ok: boolean; text: string }>("/api/dictation/cancel"),
  dump: () => post<{ ok: boolean; path: string }>("/api/dump", {}, { timeout: SLOW_MS }),
  perf: (label: string, sample: Record<string, unknown>) => post("/api/perf", { label, sample }),
};

export const puzzle = (id: string) => {
  const p = `/api/puzzle/${encodeURIComponent(id)}`;
  return {
    /** `next`: the session's recommended puzzles but this one (Next while it is played);
     * `after`: where Next goes once it has ended, however it was opened. */
    open: () => post<{ ok: boolean; error?: string; next?: string[]; after?: string[]; state?: PuzzleState }>(`${p}/open`, {}, { timeout: SLOW_MS }),
    /** `hidden`: the page is out of view, so the room does not take its quiet for its end. */
    poll: (since: number, hidden = false) =>
      get<{ ok: boolean; closed?: boolean; state?: PuzzleState; lines?: LogLine[]; cursor?: number }>(
        `${p}/poll?since=${since | 0}${hidden ? "&hidden=1" : ""}`,
      ),
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

/** The background reviewer (ui/backlog.py): the games it has still to review, the one under
 * way, whether it waits for the player's own work; null beside another panel. */
/** The background review of the library (ui/backlog.py): a job the player sees and runs. */
export type Backlog = {
  status: "reviewing" | "paused" | "waiting" | "done" | "empty";
  total: number;
  done: number;
  /** Games not reviewed yet. */
  waiting: number;
  reviewing: string | null;
  current: { key: string; white: string | null; black: string | null; done: number; total: number } | null;
  eta_s: number | null;
  paused_by_you: boolean;
  /** Stepped aside for the player's own game or review. */
  paused: boolean;
} | null;

export const jobs = {
  /** Pause or resume the background review; kept across restarts. */
  pauseReviews: (paused: boolean) => post<{ ok: boolean; reviews?: NonNullable<Backlog> }>("/api/jobs/reviews", { paused }),
};

export const games = {
  list: () => get<{ games: LibraryGame[]; users: { chesscom: string; lichess: string }; backlog: Backlog }>("/api/games"),
  /** Games in: how many are new and how many were there already; on a failure, what to do,
   * and the setting to fix when the username is at fault (`field`). */
  bringIn: (source: "pgn" | "chesscom" | "lichess", text = "", months = 3) =>
    post<{ ok: boolean; added?: number; already?: number; left_out?: number; keys?: string[]; error?: string; field?: string }>("/api/games/import", { source, text, months }, { timeout: null }),
  /** Games out of the player's record, each with the game it is paired with (`removed`). */
  remove: (keys: string[]) => post<{ ok: boolean; removed?: string[] }>("/api/games/remove", { keys }),
  /** Games removed moments ago, put back. */
  restore: (keys: string[]) => post<{ ok: boolean; restored?: string[]; error?: string }>("/api/games/restore", { keys }),
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
  /** The loss's band: none, inaccuracy, mistake or blunder (`score.severity`). */
  severity: string;
  best_uci: string | null;
  best_san: string | null;
  best_line: LineStep[];
  /** How far the next best move stood behind the best in the position the move was played
   *  from, in expected score; null with a single line. */
  gap: number | null;
  follow_line: LineStep[];
  /** What the move threatens (UCI): the mover's best next move were it their turn again. */
  idea: string | null;
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
  /** Whether the opening book is there to name the opening (Setup fetches it). */
  book: boolean;
  last_book_ply: number | null;
  key_moments: { ply: number; side: "white" | "black"; kind: string }[];
  graph: number[];
  shape: string | null;
};

/** The last card's training: a motif the game's mistakes passed up, and puzzles of it. */
export type Practice = { motif: string; missed: number; puzzles: { id: string; rating: number }[] };

/** Retry: the position a move was played from, and a new move from it graded. */
export type RetryPosition = { ply: number; fen: string; side: "white" | "black"; legal: Record<string, string[]> };
export type Tried = { ply: number; uci: string; san: string; fen: string; label: string | null; loss: number; accuracy: number | null; only: boolean };

/** The AI coach's words on a review's key moves (coach/review_words.py), by ply: why the move
 * mattered, a lesson, and for a bad move the better one, shown as written. */
export type ReviewWords = {
  model: string;
  cost: number | null;
  texts: Record<string, { why?: string; better?: string; lesson?: string }>;
  /** Each paragraph as its links to the board, by ply then part (coach/review_links.py). */
  links: Record<string, Record<string, AiPiece[]>>;
};
/** A link in the AI coach's words: a move (its arrow, coloured by whose it is), a piece, square
 * or file (its squares), a term (its meaning, and the move's motif's squares when tagged). */
export type AiLink = {
  text: string;
  kind: "move" | "piece" | "square" | "file" | "term";
  squares?: string[];
  arrow?: [string, string, "best" | "threat" | "idea"];
  /** A file end to end, drawn as a double-ended arrow. */
  span?: [string, string];
  meaning?: string;
};
export type AiPiece = string | AiLink;
/** The words asked for only by the player: none yet, being written, ready, or failed; why they
 * cannot be (no AI coach set up); the key moves they cover and what they should cost (dollars,
 * none when the model's prices are not known). */
export type ReviewAi = {
  status: "none" | "writing" | "ready" | "failed";
  error: string | null;
  words: ReviewWords | null;
  unavailable: string | null;
  moves: number;
  /** The key moves answered already when the words were cut short (the app closed): kept. */
  answered: number;
  estimate: number | null;
};

export type ReviewState = {
  key: string;
  status: "starting" | "analysing" | "ready" | "failed";
  progress: { done: number; total: number };
  error: string | null;
  strength: string;
  game: Omit<LibraryGame, "key" | "added_at" | "reviewed" | "accuracy" | "performance">;
  start_fen: string | null;
  /** While the game is analysed: its positions, the first before any move, each with the move
   *  that led to it -- the board plays the game out as the progress goes. */
  playback?: { fen: string; uci: string | null }[] | null;
  summary: GameSummary | null;
  moves: ReviewMove[];
  practice: Practice[];
  ai: ReviewAi;
  closed: boolean;
};

export const review = (key: string) => {
  const p = `/api/review/${encodeURIComponent(key)}`;
  type Polled = { ok: boolean; closed?: boolean; error?: string; state?: ReviewState; lines?: LogLine[]; cursor?: number };
  return {
    /** The game reviewed at `strength`; none named, at the server's own: Standard on a computer,
     *  Fast on the phone, whose engine is slower (the plan's depth 14). */
    open: (strength?: string) => post<Polled & { voice?: boolean }>(`${p}/open`, strength ? { strength } : {}, { timeout: SLOW_MS }),
    poll: (since: number) => get<Polled>(`${p}/poll?since=${since | 0}`),
    /** The card said in the panel's voice, cutting off the one before; hush stops it. */
    say: (text: string) => post<{ ok: boolean }>(`${p}/say`, { text }),
    hush: () => post<{ ok: boolean }>(`${p}/hush`),
    /** The AI coach set writing on the key moves; the polls bring the words. */
    explain: () => post<{ ok: boolean; error: string | null }>(`${p}/ai`),
    retry: (ply: number) => get<{ ok: boolean; error?: string } & RetryPosition>(`${p}/retry?ply=${ply | 0}`, { timeout: SLOW_MS }),
    attempt: (ply: number, uci: string) => post<{ ok: boolean; error?: string } & Tried>(`${p}/try`, { ply, uci }, { timeout: SLOW_MS }),
    hint: (ply: number, level: number) =>
      post<{ ok: boolean; error?: string; level: number; text: string; best: string | null }>(`${p}/hint`, { ply, level }, { timeout: SLOW_MS }),
    // Analysis (ui/review_analysis.py): any move from any position, the engine's lines for the
    // board, what the player's games went on with, the game as a PGN.
    position: (fen: string) => post<{ ok: boolean; error?: string } & AnalysisPosition>(`${p}/position`, { fen }, { timeout: SLOW_MS }),
    move: (fen: string, uci: string) =>
      post<{ ok: boolean; error?: string } & AnalysisPosition & { uci: string; san: string }>(`${p}/move`, { fen, uci }, { timeout: SLOW_MS }),
    lines: (fen: string) => post<{ ok: boolean; error?: string } & EngineLines>(`${p}/lines`, { fen }, { timeout: SLOW_MS }),
    explore: (fen: string, moves: string[]) => post<{ ok: boolean; error?: string } & Explored>(`${p}/explore`, { fen, moves }, { timeout: SLOW_MS }),
    /** `copy`: the phone's app puts it on the clipboard (its web view's clipboard refuses writes). */
    pgn: (tree: PgnTree[], annotated: boolean, save: boolean, copy = false) =>
      // Saved through the system's dialog, which waits on the player: no timeout then.
      post<{ ok: boolean; error?: string; pgn: string; name: string; saved?: string; copied?: boolean }>(`${p}/pgn`, { tree, annotated, save, copy }, { timeout: save ? null : SLOW_MS }),
    close: () => post(`${p}/close`),
    closePath: `${p}/close`,
  };
};

/** A position of the analysis board: whose move, every legal move (a square -> where its piece
 *  may go), and the result when the game is over there. */
export type AnalysisPosition = { fen: string; side: "white" | "black"; legal: Record<string, string[]>; over: string | null };
/** The engine's best lines for a position, each from White's side and played out. */
export type EngineLines = {
  fen: string;
  depth: number;
  white_score: number | null;
  lines: { eval: { cp: number | null; mate: number | null }; steps: LineStep[] }[];
};
/** What the player's games went on with from a position, and how those games went for them. */
export type Explored = {
  games: number;
  opening: string | null;
  /** Whether the opening book is there to name the opening (Setup fetches it). */
  book: boolean;
  moves: { uci: string; san: string; games: number; won: number; drawn: number; lost: number; last: { key: string; played_at: number } | null }[];
};
/** The analysis board's moves for the PGN: each with what follows it, the game's own first. */
export type PgnTree = { uci: string; children: PgnTree[] };

// ---- Stats: the player's games added up (profile/stats.py) ------------------------------------

/** A mean with how many it is of; a rate with its hits. With the player's level known: the
 * peers' figure there, and a grade for how far ahead of it the player's is (a kind of move's
 * badge; none where the peers' figure hardly follows the rating). */
export type Figure = { value: number | null; n: number; peers?: number; grade?: string | null };
/** A rating on the page's scale, from so many moves; a category's with its likely range. */
export type StatRating = { rating: number; moves: number; low?: number; high?: number } | null;
export type Rate = Figure & { hits: number };

export type StatsFilter = { time_class: string; days: number; opponents: "all" | "people" | "bots" };

export type StatsNumbers = {
  overview: { games: number; wins: number; draws: number; losses: number; accuracy: Figure; moves: number };
  openings: {
    accuracy: Figure;
    mistakes: Figure;
    book_moves: Figure;
    by_opening: { opening: string; side: "white" | "black"; games: number; score: number | null; accuracy: number | null }[];
  };
  tactics: { found: Record<string, Rate>; allowed: Record<string, Rate>; overall: Rate };
  strategy: Record<string, Record<string, Figure>>;
  endgames: { win: Record<string, Figure>; accuracy: Record<string, Figure> };
  time: { by_phase: Record<string, Figure>; pressure: Rate; calm: Rate; lost_on_time: number; lost_on_time_winning: number };
  results: Record<string, { won: number; drawn: number; lost: number }>;
  /** How the games went by their shape (smooth, sudden, giveaway...), the shapes they had. */
  shapes: { shape: string; games: number; won: number; drawn: number; lost: number }[];
  galleries: Record<string, number>;
  /** Game by game, oldest first: the rating as of each game (the last ten games' moves pooled;
   * `game`, the game's own), and each category's figure. */
  trends: Record<string, { t: number | null; key: string; v: number; game?: number }[]>;
  /** The player's rating over every move here, and each category's. */
  ratings: { overall: StatRating; openings: StatRating; tactics: StatRating; strategy: StatRating; endgames: StatRating };
  /** Who the peers are: players of this rating, the player's own in their games ("rating") or,
   * when none is recorded, of their play ("play"). */
  peers: { rating: number; from: "rating" | "play" } | null;
  /** What each area of the player's game cost them: expected score lost, a game. */
  costs: Record<string, Figure | Record<string, Figure>>;
  /** The areas that cost most, dearest first, with puzzles of their motif. */
  work: WorkArea[];
  /** The reviewed games left out as their player's side is not known (no username set). */
  unsided: number;
};

/** An area to work on: `area` its name ("missed.fork", "opening", "endgame.rook"...), what it
 * cost a game over so many games and the peers' cost, the grade, its puzzles. */
export type WorkArea = {
  area: string;
  cost: number;
  games: number;
  /** The moves it counts. */
  moves: number;
  peers?: number | null;
  grade?: string | null;
  motif: string | null;
  puzzles: { id: string; rating: number }[];
};

/** A move of the player's behind a figure: its game, and the position it was played from. */
export type StatPosition = {
  key: string;
  ply: number;
  fen: string;
  uci: string;
  san: string;
  label: string | null;
  side: "white" | "black";
  played_at: number | null;
  opening: string | null;
  /** Whether the opening book is there to name the opening (Setup fetches it). */
  book: boolean;
};

const statsQuery = (f: StatsFilter) => `time_class=${encodeURIComponent(f.time_class)}&days=${f.days | 0}&opponents=${f.opponents}`;

export const stats = {
  get: (f: StatsFilter) => get<{ ok: boolean; stats: StatsNumbers; backlog: Backlog }>(`/api/stats?${statsQuery(f)}`, { timeout: SLOW_MS }),
  positions: (what: string, arg: string, f: StatsFilter) =>
    get<{ ok: boolean; positions: StatPosition[] }>(`/api/stats/positions?what=${encodeURIComponent(what)}&arg=${encodeURIComponent(arg)}&${statsQuery(f)}`, { timeout: SLOW_MS }),
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
  /** Raised by every change the server makes to the settings or a piece set. */
  settings_version?: number;
  /** Game Review's accuracy of each game whose end-of-game card was shown, by its review key,
   * once the game's review is done: the card's live accuracy until then. */
  review_accuracy?: Record<string, number>;
  caption_lines?: [number, string, boolean?][];
  caption_again?: number;
  /** A reply being written, sentence by sentence, until the whole message replaces it. */
  coach_draft?: string;
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
  /** The moves played, in SAN: moves[k] made positions[k + 1]. */
  moves?: string[];
  motif?: string | null;
  rating?: number;
  url?: string;
  voice?: boolean;
  // Whether the room has a voice at all; the voice button shows only when it has.
  voice_ready?: boolean;
  closed?: boolean;
  llm?: { calls: number; prompt: number; completion: number; cost: number | null } | null;
  /** The outcome as the profile took it, once the puzzle has ended and the room recorded it:
   * solved or not, the puzzle rating after it as Lichess shows it, how far it moved, and whether
   * it is still provisional (Lichess writes "1500?"). */
  result?: { solved: boolean; level: number; change: number; provisional: boolean } | null;
};
