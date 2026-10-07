// The panel's state, fed by one poll of the sidecar every 250 ms and read by every
// component through runes. What the old app.js kept in module variables lives here;
// what it rendered by hand the components derive.

import { connect, followCursor, noAnswer, panel, put, restartSidecar, type LogLine, type Mark, type PanelState } from "./api";

export type TermRow = { key: string; count: number; line: LogLine; open: boolean };

export type CoachItem =
  | { kind: "message"; id: number; trigger: string; disclosure: number | string; ply: number | null; fallback: boolean; full: string; marks: Mark[]; fen: string | null }
  | { kind: "review"; id: number; review: Record<string, any>; insights: string[]; reviewKey: string | null }
  | { kind: "practice"; id: number; practice: Record<string, any> }
  | { kind: "you"; id: number; text: string; spoken: boolean }
  | { kind: "divider"; id: number; text: string }
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

const MAX_ROWS = 1500;
// The coach pane and the conversation tab are rendered whole, so they are bounded too:
// three games' worth of coach items, and the conversation's turns likewise.
const MAX_COACH = 200;
const MAX_CONVO = 400;
export const CHANNELS = ["sys", "vision", "game", "engine", "coach", "voice", "user", "puzzle"] as const;

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

/** A setting changed from the panel: an open Settings window shows it. */
function told(): void {
  void import("./shell.svelte").then((s) => s.announce("settings")).catch(() => {});
}

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
  // The latest message's marks and the ply they were drawn at (a reply with no ply of its
  // own, at the ply shown when it came), on the line of positions it came in.
  latestMarks = $state<{ ply: number; marks: Mark[] } | null>(null);

  listenMode = $state("off");
  pttDown = $state(false);
  connected = $state(false);
  /** Why the sidecar gives no good answer, when it does not: the page says so. */
  offline = $state<string | null>(null);
  /** The sidecar's process has ended (or could not start): the page offers a Restart. */
  stopped = $state(false);
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
    if (!marks && latest && plyShown >= latest.ply && plyShown - latest.ply <= 1) marks = latest.marks;
    const list = [...(marks || [])];
    if (this.hoverSquare) list.push({ kind: "square", squares: [this.hoverSquare], color: "mention", opacity: 0.9 });
    return list;
  }

  async start(): Promise<void> {
    this.stop();
    try {
      await connect();
    } catch (e) {
      // The shell could not start the sidecar: said plainly, with a Restart.
      this.offline = `Couldn't start: ${e instanceof Error ? e.message : String(e)}`;
      this.stopped = true;
      return;
    }
    this.connected = true;
    this.offline = null;
    this.stopped = false;
    try {
      const first = await panel.poll(0);
      if (first.state) this.state = first.state;
      if (first.lines) for (const line of first.lines) this.addLine(line);
      if (first.cursor !== undefined) this.cursor = first.cursor;
    } catch {
      /* the tick says so if it stays unreachable */
    }
    this.tick();
  }

  /** The sidecar is gone or never started: the shell starts a new one, and the poll resumes. */
  async restart(): Promise<void> {
    this.stop();
    this.offline = "Restarting...";
    this.stopped = false;
    this.cursor = 0;
    try {
      await restartSidecar();
    } catch (e) {
      this.offline = `Couldn't start: ${e instanceof Error ? e.message : String(e)}`;
      this.stopped = true;
      return;
    }
    await this.start();
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
      this.cursor = followCursor(this.cursor, res.cursor);
      if (res.lines) for (const line of res.lines) this.addLine(line);
      if (res.state) this.apply(res.state);
      this.stopped = false;
      if (this.offline) this.offline = null;
    } catch (e) {
      // An error answer, a blip, or the sidecar's process gone: the shell tells them apart.
      const why = await noAnswer(e);
      this.stopped = !!why?.stopped;
      this.offline = why?.text ?? null;
    }
    this.timer = setTimeout(() => this.tick(), 250);
  }

  private apply(s: PanelState): void {
    // The server changed the settings or a piece set, however it came to (a Settings page, a
    // switch here, the captions' box, another device): this window draws them again and tells
    // the others -- no window waits for a peer's word, and none asks on its own (2026-10-01).
    const was = this.state.settings_version;
    if (was !== undefined && s.settings_version !== undefined && s.settings_version !== was) {
      void import("./look").then((l) => l.refreshLook()).catch(() => {});
      told();
    }
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
      // Bounded: the 1h42 soak of 2026-09-21 grew the old page from 480 to 700 MB on an
      // unbounded terminal, and the 40-minute soak with the meter's counters showed the
      // renderer's memory tracking the terminal's DOM -- 4000 rows were 26k nodes and
      // 650 MB steady; 1500 rows read as far back as anyone scrolls (the sidecar keeps
      // 3,000 records, and a dump writes them all).
      if (this.rows.length > MAX_ROWS) this.rows.splice(0, this.rows.length - MAX_ROWS);
    }
    const d = line.data;
    // A new game, a position joined or a resync starts a line of positions numbered from its
    // own start: the last message's marks belonged to the old one (a puzzle's arrows sat on
    // the next puzzles' boards, Tai's Pixel, 2026-09-23).
    if (line.chan === "game" && ((d && d.start_fen) || line.text.startsWith("resync:"))) this.latestMarks = null;
    if (line.chan === "game" && d && d.start_fen && d.kind !== "puzzle") this.addDivider(d.kind === "game" ? "New game" : "New position");
    // An ask with no coach to answer it (panel.py's _drain_commands): no reply is coming.
    if (line.chan === "user" && line.text.startsWith("no coach to ask")) this.dropPending();
    if (line.chan === "coach" && d && d.full) this.addCoach(d);
    if (line.chan === "coach" && d && d.review) this.addReview(d);
    if (line.chan === "coach" && d && d.practice) this.coach.push({ kind: "practice", id: nextId++, practice: d.practice });
    if (line.chan === "user" && line.text.startsWith("you: ")) this.addYou(line.text.slice(5), !!(d && d.spoken));
    this.addConvo(line);
    if (this.coach.length > MAX_COACH) this.coach.splice(0, this.coach.length - MAX_COACH);
    if (this.convo.length > MAX_CONVO) this.convo.splice(0, this.convo.length - MAX_CONVO);
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
    this.latestMarks = { ply: d.ply ?? this.ply, marks };
  }

  /** What you asked, typed or spoken, in the pane with the coach's words (Tai, 2026-09-29: "similar to like a chat interface like one you find on iMessage"). Its
   * reply is already on its way (the typing dots went up when it was sent): the question goes
   * above them. */
  private addYou(text: string, spoken: boolean): void {
    const item: CoachItem = { kind: "you", id: nextId++, text, spoken };
    const pending = this.coach.findIndex((c) => c.kind === "pending");
    if (pending < 0) this.coach.push(item);
    else this.coach.splice(pending, 0, item);
  }

  /** A game's start, in the pane that keeps the last games' words: a line between them, as a
   * chat app's between conversations. None above the first; a game with no words gives its line
   * to the next. */
  private addDivider(text: string): void {
    if (!this.coach.length) return;
    const item: CoachItem = { kind: "divider", id: nextId++, text };
    const last = this.coach.length - 1;
    if (this.coach[last].kind === "divider") this.coach[last] = item;
    else this.coach.push(item);
  }

  private addReview(d: Record<string, any>): void {
    this.dropPending();
    this.coach.push({ kind: "review", id: nextId++, review: d.review, insights: d.insights || [], reviewKey: d.review_key ?? null });
  }

  private addConvo(line: LogLine): void {
    const d = line.data || {};
    if (line.chan === "user" && line.text.startsWith("you: ")) {
      this.convo.push({ id: nextId++, who: "you", label: d.spoken ? "you · spoken" : "you", text: line.text.slice(5), calls: [] });
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

  /** Asked: resolves once the sidecar took the question, rejects when it did not (no answer, an
   *  error, or a refusal) -- the typing dots go then, as no reply is coming; the caller keeps
   *  the words to send again. */
  async ask(text: string): Promise<void> {
    const t = text.trim();
    if (!t) return;
    await this.sent(panel.ask(t));
  }

  quick(cmd: "hint" | "move" | "line"): Promise<void> {
    return this.sent(panel[cmd]());
  }

  private async sent(request: Promise<unknown>): Promise<void> {
    this.showPending();
    try {
      const res = (await request) as { ok?: boolean } | null;
      if (res?.ok === false) throw new Error("Sidecoach did not take the question");
    } catch (e) {
      this.dropPending();
      throw e;
    }
  }

  setPtt(down: boolean): void {
    if (this.listenMode !== "ptt" || this.pttDown === down) return;
    this.pttDown = down;
    panel.ptt(down);
  }

  /** How the coach hears you: off, hold to talk (ptt), or always listening (vad). Kept for
   * the next session, as muted is. */
  setListen(mode: string): void {
    this.listenMode = mode; // optimistic; the next poll confirms
    if (mode !== "ptt") this.pttDown = false;
    void panel.setListen(mode).then(told);
  }

  setMuted(muted: boolean): void {
    this.state.muted = muted; // optimistic
    void panel.setMuted(muted).then(told);
  }

  /** Captions over other windows on or off, and kept on screen or not: the settings saved,
   * as the Settings page would (the box opens with a sample line, or goes). */
  setCaptions(on: boolean, keep?: boolean): void {
    const was = this.state.captions;
    if (was) this.state.captions = { ...was, on, keep: keep ?? was.keep }; // optimistic
    const voice: Record<string, boolean> = { captions: on };
    if (keep !== undefined) voice.captions_keep = keep;
    void put("/api/settings", { settings: { voice }, secrets: {} }).then(told).catch(() => {});
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
