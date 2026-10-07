// Retry (docs/game-review.md, D): another move found in a position a move of a reviewed
// game was played from, graded as the review graded the game's (ui/review_retry.py). The review
// page retries its own moves; the Stats window retries the positions behind a figure. A solving
// move stays; any other shows red and goes back (`wrong`, then `redraw`); the hints are the
// coach's ladder, the third drawing the move; giving up plays it. What fails is said (`error`),
// never swallowed: a sidecar that does not answer left Retry doing nothing (Tai, 2026-09-30).
import type { RetryPosition, Tried } from "$lib/api";
import { SOLVED } from "./words";

export type RetryApi = {
  position: (ply: number) => Promise<{ ok: boolean; error?: string } & RetryPosition>;
  attempt: (ply: number, uci: string) => Promise<{ ok: boolean; error?: string } & Tried>;
  hint: (ply: number, level: number) => Promise<{ ok: boolean; error?: string; level: number; text: string; best: string | null }>;
};

export type Retrying = RetryPosition & {
  tried: Tried | null;
  wrong: string | null;
  busy: boolean;
  hint: number;
  hintText: string;
  hintBest: string | null;
  solved: boolean;
  gaveUp: boolean;
};

const WRONG_MS = 900;
const NO_ANSWER = "Sidecoach is not answering. Try again in a moment.";

export class RetryDesk {
  now = $state<Retrying | null>(null);
  redraw = $state(0);
  /** What went wrong the last time the desk asked the sidecar; empty when all is well. */
  error = $state("");
  /** The best try at each position retried, by ply: what "accuracy after retries" counts. */
  best = $state<Record<number, Tried>>({});

  constructor(
    private api: RetryApi,
    private sound: (fen: string, uci: string) => void,
  ) {}

  async open(ply: number): Promise<boolean> {
    const at = await this.api.position(ply).catch(() => null);
    if (!at || !at.ok) {
      this.error = at?.error || NO_ANSWER;
      return false;
    }
    this.error = "";
    this.now = { ply: at.ply, fen: at.fen, side: at.side, legal: at.legal, tried: null, wrong: null, busy: false, hint: 0, hintText: "", hintBest: null, solved: false, gaveUp: false };
    return true;
  }

  close(): void {
    this.now = null;
    this.error = "";
    this.redraw++;
  }

  // A try, on the board already (the hand put it there), graded.
  play = async (uci: string): Promise<void> => {
    const at = this.now;
    if (!at || at.busy) return;
    this.sound(at.fen, uci);
    this.now = { ...at, busy: true };
    const graded = await this.api.attempt(at.ply, uci).catch(() => null);
    if (!this.now || this.now.ply !== at.ply) return;
    if (!graded || !graded.ok) {
      this.error = graded?.error || NO_ANSWER;
      this.now = { ...this.now, busy: false };
      this.redraw++;
      return;
    }
    this.error = "";
    const kept = this.best[graded.ply];
    if (!kept || (graded.accuracy ?? 0) > (kept.accuracy ?? 0)) this.best = { ...this.best, [graded.ply]: graded };
    if (SOLVED.has(graded.label ?? "")) {
      this.now = { ...this.now, tried: graded, busy: false, solved: true, hintBest: null };
      return;
    }
    this.now = { ...this.now, tried: graded, wrong: uci };
    setTimeout(() => {
      if (!this.now || this.now.wrong !== uci) return;
      this.now = { ...this.now, wrong: null, busy: false };
      this.redraw++;
    }, WRONG_MS);
  };

  async hint(): Promise<void> {
    const at = this.now;
    if (!at || at.hint >= 3) return;
    const got = await this.api.hint(at.ply, at.hint + 1).catch(() => null);
    if (!this.now || this.now.ply !== at.ply) return;
    if (!got || !got.ok) {
      this.error = got?.error || NO_ANSWER;
      return;
    }
    this.error = "";
    this.now = { ...this.now, hint: got.level, hintText: got.text, hintBest: got.best };
  }

  // Given up: the move, graded and played; not counted as found.
  async giveUp(): Promise<void> {
    const at = this.now;
    if (!at || at.solved) return;
    const best = at.hintBest ?? (await this.api.hint(at.ply, 3).catch(() => null))?.best;
    if (!best) {
      this.error = NO_ANSWER;
      return;
    }
    const graded = await this.api.attempt(at.ply, best).catch(() => null);
    if (!this.now || this.now.ply !== at.ply) return;
    if (!graded || !graded.ok) {
      this.error = graded?.error || NO_ANSWER;
      return;
    }
    this.error = "";
    this.now = { ...this.now, tried: graded, solved: true, gaveUp: true, hintBest: null };
  }
}
