// A line on the board, stepped: the review's Best, Follow-up and a retry's line, and a puzzle's
// solution once given up. Its moves are shown one at a time and only as the player asks: ◀ ▶
// (or the arrow keys) step it, Play runs it a move every LINE_MS (a reading pace), and any step
// stops Play. `i` is the move shown, -1 the position the line starts from (Tai, 2026-09-30: a
// line that ran by itself, fast, on a click could not be read).
//
// The clock is injected (tests/lineplayer.test.mjs runs this under node with a manual one).

export const LINE_MS = 1100;
export const TAKEBACK_MS = 350;

export type LineClock = {
  set: (fn: () => void, ms: number) => unknown;
  clear: (handle: unknown) => void;
};

const realClock: LineClock = {
  set: (fn, ms) => setTimeout(fn, ms),
  clear: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export type LineHooks = {
  /** A step on by one move, heard: `i` is the move now shown. */
  sound?: (i: number) => void;
  /** The move shown changed (any way: a step, a jump, Play). */
  changed?: (i: number) => void;
};

export class LinePlayer {
  /** The move shown, -1 for the position the line starts from. */
  i = $state(-1);
  /** How many moves the line has. */
  length = $state(0);
  /** Whether Play runs it. */
  autoplay = $state(false);
  private timer: unknown = null;
  private readonly hooks: LineHooks;
  private readonly clock: LineClock;

  constructor(hooks: LineHooks = {}, clock: LineClock = realClock) {
    this.hooks = hooks;
    this.clock = clock;
  }

  get atStart(): boolean { return this.i < 0; }
  get atEnd(): boolean { return this.i >= this.length - 1; }
  /** "Move N of M", or the line's start. */
  get where(): string { return this.i < 0 ? "Before the line" : `Move ${this.i + 1} of ${this.length}`; }

  /** A new line of `length` moves, at `i`; nothing runs. */
  reset(length: number, i = -1): void {
    this.pause();
    this.length = length;
    this.i = Math.max(-1, Math.min(length - 1, i));
  }

  /** The line's first move shown a moment after its start (the board takes back what was there,
   *  then plays the move), and nothing more until the player asks. */
  showFirst(): void {
    this.pause();
    this.timer = this.clock.set(() => {
      this.timer = null;
      if (this.i === -1) this.to(0);
    }, TAKEBACK_MS);
  }

  /** To move `i` (clamped); a step on by one is heard unless `heard` is false. */
  to(i: number, heard = true): void {
    const to = Math.max(-1, Math.min(this.length - 1, i));
    if (heard && to === this.i + 1 && to >= 0) this.hooks.sound?.(to);
    const was = this.i;
    this.i = to;
    if (to !== was) this.hooks.changed?.(to);
  }

  /** The player's own step: Play stops. */
  step(delta: number): void {
    this.pause();
    this.to(this.i + delta);
  }

  /** The line's start or end. */
  toEnd(last: boolean): void {
    this.pause();
    this.to(last ? this.length - 1 : -1, false);
  }

  pause(): void {
    if (this.timer !== null) this.clock.clear(this.timer);
    this.timer = null;
    this.autoplay = false;
  }

  /** Play from the move shown (from the start when at the end), a move every LINE_MS, to the
   *  end; pressed again, Pause. */
  playPause(): void {
    if (this.autoplay) {
      this.pause();
      return;
    }
    this.pause();
    if (this.atEnd) this.to(-1, false);
    this.autoplay = true;
    const advance = () => {
      if (this.atEnd) {
        this.timer = null;
        this.autoplay = false;
        return;
      }
      this.to(this.i + 1);
      this.timer = this.clock.set(advance, LINE_MS);
    };
    this.timer = this.clock.set(advance, this.i === -1 ? TAKEBACK_MS : 0);
  }

  /** The keys a stepped line answers: ← → step, Home/End its ends, Space plays. True when the
   *  key was the line's. */
  key(e: KeyboardEvent): boolean {
    if (e.key === "ArrowRight") this.step(1);
    else if (e.key === "ArrowLeft") this.step(-1);
    else if (e.key === "Home" || e.key === "ArrowUp") this.toEnd(false);
    else if (e.key === "End" || e.key === "ArrowDown") this.toEnd(true);
    else if (e.key === " " && !(e.target as HTMLElement | null)?.closest?.("button, a, input")) this.playPause();
    else return false;
    e.preventDefault();
    return true;
  }
}
