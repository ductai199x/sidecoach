// The panel's state, fed by one poll of the sidecar every 250 ms and read by every
// component through runes. What the old app.js kept in module variables lives here;
// what it rendered by hand the components derive.

import { connect, panel, type LogLine, type Mark, type PanelState } from "./api";

export type TermRow = { key: string; count: number; line: LogLine; open: boolean };

export type CoachItem =
  | { kind: "message"; id: number; trigger: string; disclosure: number | string; ply: number | null; fallback: boolean; full: string; marks: Mark[]; fen: string | null }
  | { kind: "review"; id: number; review: Record<string, any>; insights: string[] }
  | { kind: "practice"; id: number; practice: Record<string, any> }
  | { kind: "pending"; id: number };

export type ConvoTurn = {
  id: number;
  who: "you" | "coach" | "game";
  label: string;
  text: string;
  unspoken?: boolean;
  body?: string;
  calls: { title: string; prompt?: string; raw?: string }[];
};

export const CHANNELS = ["sys", "vision", "game", "engine", "coach", "voice", "user", "puzzle"] as const;
export const MODE_NEXT: Record<string, string> = { off: "ptt", ptt: "vad", vad: "off" };
export const MODE_TITLES: Record<string, string> = {
  off: "Listening: off. Click for hold-to-talk",
  ptt: "Listening: hold to talk (hold the round button, or Space). Click for always-on voice",
  vad: "Listening: voice (the coach hears you whenever you speak). Click to turn off",
};

/** The sizes and switches the player set, remembered across sessions. */
export const prefs = {
  get(key: string): number | null {
    try {
      const v = parseFloat(localStorage.getItem("sidecoach." + key) ?? "");
      return Number.isFinite(v) ? v : null;
    } catch {
      return null;
    }
  },
  set(key: string, v: number | null): void {
    try {
      if (v === null) localStorage.removeItem("sidecoach." + key);
      else localStorage.setItem("sidecoach." + key, String(Math.round(v)));
    } catch {
      /* private mode */
    }
  },
  getStr(key: string): string | null {
    try { return localStorage.getItem("sidecoach." + key); } catch { return null; }
  },
  setStr(key: string, v: string | null): void {
    try { if (v === null) localStorage.removeItem("sidecoach." + key); else localStorage.setItem("sidecoach." + key, v); } catch { /* private mode */ }
  },
};

let nextId = 1;

export class PanelStore {
  state = $state<PanelState>({});
  rows = $state<TermRow[]>([]);
  coach = $state<CoachItem[]>([]);
  convo = $state<ConvoTurn[]>([]);
  filters = $state({ chans: Object.fromEntries(CHANNELS.map((c) => [c, true])) as Record<string, boolean>, verbose: false, query: "" });
  autoscroll = $state(true);
  pane = $state<"terminal" | "conversation">("terminal");

  // What the board shows: the live position unless a move is browsed or a message hovered.
  pinnedPly = $state<number | null>(null);
  viewPly = $state<number | null>(null);
  ghostFen = $state<string | null>(null);
  hoverMarks = $state<Mark[] | null>(null);
  hoverSquare = $state<string | null>(null);
  latestMarks = $state<{ ply: number | null; marks: Mark[] } | null>(null);

  listenMode = $state("off");
  pttDown = $state(false);
  connected = $state(false);
  cursor = 0;
  private lastCoachTurn: ConvoTurn | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  polls: number[] = [];

  get running(): boolean { return !!this.state.running; }
  get bottom(): string {
    const base = this.state.bottom || "white";
    return this.state.view_flip ? (base === "white" ? "black" : "white") : base;
  }
  get ply(): number { return this.state.ply || 0; }
  get liveFen(): string | null { return this.state.fen ?? null; }

  /** The position the board shows and the move it highlights. */
  get shown(): { fen: string | null; lastMove: string | null } {
    if (this.ghostFen) return { fen: this.ghostFen, lastMove: null };
    const ply = this.viewPly;
    if (ply != null) {
      const fen = ply > 0 ? this.state.fens?.[ply - 1] ?? null : null;
      return { fen, lastMove: this.state.ucis?.[ply - 1] ?? null };
    }
    return { fen: this.liveFen, lastMove: this.state.last_move ?? null };
  }

  /** The marks on the board: a hovered message's, else the latest while it still applies. */
  get marks(): Mark[] {
    let marks = this.hoverMarks;
    const plyShown = this.viewPly == null ? this.ply : this.viewPly;
    const latest = this.latestMarks;
    if (!marks && latest && (latest.ply == null || (plyShown >= latest.ply && plyShown - latest.ply <= 1))) marks = latest.marks;
    const list = [...(marks || [])];
    if (this.hoverSquare) list.push({ kind: "square", squares: [this.hoverSquare], color: "mention", opacity: 0.9 });
    return list;
  }

  async start(): Promise<void> {
    await connect();
    this.connected = true;
    const first = await panel.poll(0);
    if (first.state) this.state = first.state;
    if (first.lines) for (const line of first.lines) this.addLine(line);
    if (first.cursor !== undefined) this.cursor = first.cursor;
    this.tick();
  }

  stop(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  // One poll at a time: the next is scheduled only once this one has been handled.
  private async tick(): Promise<void> {
    try {
      const t0 = performance.now();
      const res = await panel.poll(this.cursor);
      this.polls.push(performance.now() - t0);
      if (this.polls.length > 400) this.polls = this.polls.slice(-200);
      if (res.cursor !== undefined) this.cursor = Math.max(this.cursor, res.cursor);
      if (res.lines) for (const line of res.lines) this.addLine(line);
      if (res.state) this.apply(res.state);
    } catch {
      /* the window closing, or the sidecar not ready */
    }
    this.timer = setTimeout(() => this.tick(), 250);
  }

  private apply(s: PanelState): void {
    this.state = s;
    if (s.listen !== undefined && s.listen !== this.listenMode) {
      this.listenMode = s.listen;
      if (this.listenMode !== "ptt") this.pttDown = false;
    }
    if (this.pinnedPly !== null && this.pinnedPly > (s.moves?.length ?? 0)) this.showLive();
  }

  addLine(line: LogLine): void {
    const key = line.chan + "|" + line.level + "|" + line.text;
    const last = this.rows[this.rows.length - 1];
    if (last && last.key === key) {
      last.count++;
    } else {
      this.rows.push({ key, count: 1, line, open: false });
    }
    const d = line.data;
    if (line.chan === "coach" && d && d.full) this.addCoach(d);
    if (line.chan === "coach" && d && d.review) this.addReview(d);
    if (line.chan === "coach" && d && d.practice) this.coach.push({ kind: "practice", id: nextId++, practice: d.practice });
    this.addConvo(line);
  }

  clearTerminal(): void {
    this.rows = [];
  }

  private dropPending(): void {
    this.coach = this.coach.filter((c) => c.kind !== "pending");
  }

  showPending(): void {
    if (this.coach.some((c) => c.kind === "pending")) return;
    this.coach.push({ kind: "pending", id: nextId++ });
  }

  private addCoach(d: Record<string, any>): void {
    this.dropPending();
    const marks: Mark[] = d.annotations || [];
    this.coach.push({
      kind: "message", id: nextId++, trigger: String(d.trigger || "").replace("_", " "), disclosure: d.disclosure,
      ply: d.ply ?? null, fallback: !!d.fallback, full: d.full, marks, fen: d.fen ?? null,
    });
    this.latestMarks = { ply: d.ply ?? null, marks };
  }

  private addReview(d: Record<string, any>): void {
    this.dropPending();
    this.coach.push({ kind: "review", id: nextId++, review: d.review, insights: d.insights || [] });
  }

  private addConvo(line: LogLine): void {
    const d = line.data || {};
    if (line.chan === "user" && line.text.startsWith("you: ")) {
      this.convo.push({ id: nextId++, who: "you", label: "you", text: line.text.slice(5), calls: [] });
      this.lastCoachTurn = null;
    } else if (line.chan === "coach" && d.full) {
      const turn: ConvoTurn = {
        id: nextId++, who: "coach",
        label: "coach · " + String(d.trigger || "").replace("_", " ") + (d.fallback ? " · template" : ""),
        text: d.full, unspoken: d.voice === "stale" || d.voice === "skip", calls: [],
      };
      this.convo.push(turn);
      this.lastCoachTurn = this.convo[this.convo.length - 1];
    } else if (line.chan === "coach" && line.text.startsWith("turn (") && d.kind) {
      const first = String(d.body || "").split("\n").find((l: string) => l.startsWith("Move:")) || String(d.text || "").split("\n")[0];
      this.convo.push({ id: nextId++, who: "game", label: d.kind, text: first, body: d.body && d.body.length > first.length ? d.body : undefined, calls: [] });
      this.lastCoachTurn = null;
    } else if (line.chan === "coach" && line.text.startsWith("model reply") && this.lastCoachTurn) {
      this.lastCoachTurn.calls.push({ title: line.text, prompt: d.prompt, raw: d.raw });
    }
  }

  // ---- what the board shows ----------------------------------------------------------------

  showPly(ply: number): void {
    this.ghostFen = null;
    this.viewPly = ply;
  }

  showLive(): void {
    this.pinnedPly = null;
    this.viewPly = null;
    this.ghostFen = null;
  }

  hoverPly(ply: number | null): void {
    if (this.pinnedPly !== null) return;
    if (ply === null) this.viewPly = null; else this.showPly(ply);
  }

  pinPly(ply: number): void {
    if (this.pinnedPly === ply) this.showLive();
    else { this.pinnedPly = ply; this.showPly(ply); }
  }

  /** A message hovered: its marks, and its own position when the board has moved on. */
  hoverMessage(item: Extract<CoachItem, { kind: "message" }> | null): void {
    if (item && item.marks.length) {
      this.hoverMarks = item.marks;
      if (item.fen && item.fen !== this.liveFen) this.ghostFen = item.fen;
    } else {
      this.hoverMarks = null;
      this.ghostFen = null;
      if (this.pinnedPly !== null) this.showPly(this.pinnedPly);
    }
  }

  hoverFen(fen: string | null): void {
    if (fen) this.ghostFen = fen;
    else { this.ghostFen = null; if (this.pinnedPly !== null) this.showPly(this.pinnedPly); }
  }

  // ---- talking ---------------------------------------------------------------------------------

  ask(text: string): void {
    const t = text.trim();
    if (!t) return;
    this.showPending();
    panel.ask(t);
  }

  quick(cmd: "hint" | "move" | "line"): void {
    this.showPending();
    panel[cmd]();
  }

  setPtt(down: boolean): void {
    if (this.listenMode !== "ptt" || this.pttDown === down) return;
    this.pttDown = down;
    panel.ptt(down);
  }

  cycleListen(): void {
    const next = MODE_NEXT[this.listenMode] || "ptt";
    this.listenMode = next; // optimistic; the next poll confirms
    if (next !== "ptt") this.pttDown = false;
    panel.setListen(next);
  }

  setMuted(muted: boolean): void {
    this.state.muted = muted; // optimistic
    panel.setMuted(muted);
  }

  toggleRunning(): void {
    if (this.running) panel.stop(); else panel.start();
  }

  async resetProfile(): Promise<void> {
    await panel.resetProfile();
    this.coach = [];
  }
}

export const store = new PanelStore();
