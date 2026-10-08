// "Why this label" (the audit of 2026-10-07, features #1): one plain line under a move's
// headline in Game Review saying what earned the move its label, from what the review kept --
// the move's loss, how far the next best move stood behind (`gap`), the evals before and after,
// the move before's severity -- by the rule that labels it (docs/move-classification.md).
// Numbers are expected score, as the app shows a loss; nothing is made again.
import type { Eval, ReviewMove } from "$lib/api";

const NAMES: Record<string, string> = {
  brilliant: "Brilliant",
  great: "Great",
  best: "Best",
  excellent: "Excellent",
  good: "Good",
  book: "Book",
  inaccuracy: "Inaccuracy",
  mistake: "Mistake",
  miss: "Miss",
  blunder: "Blunder",
};

/** Expected score to two places, cut rather than rounded: a loss of 0.097 reads 0.09, so a
 *  number never reads as the next label's line (0.10 is a mistake). */
export function points(x: number): string {
  return (Math.floor(Math.round(x * 10000) / 100) / 100).toFixed(2);
}

/** The forced mate an eval (from White's side) gives `side`, in moves; null when none. */
function mateFor(e: Eval | null | undefined, side: "white" | "black"): number | null {
  if (!e || e.mate == null) return null;
  const m = side === "white" ? e.mate : -e.mate;
  return m > 0 ? m : null;
}

/** The line for `m`; `before` is the move played just before it (null for the first), and
 *  `reveal` whether a move the player could retry may name the better one (as the card's own
 *  sentence: off while the best move's arrow is off). Null for a move with no label. The
 *  label is not said again: the headline above says it (Tai, 2026-10-08: "drop the prefix"). */
export function why(m: ReviewMove, before: ReviewMove | null, reveal: boolean): string | null {
  const line = labelled(m, before, reveal);
  if (line === null) return null;
  const reason = line.slice(line.indexOf(": ") + 2);
  return reason.charAt(0).toUpperCase() + reason.slice(1);
}

function labelled(m: ReviewMove, before: ReviewMove | null, reveal: boolean): string | null {
  const label = m.label ?? "";
  const name = NAMES[label];
  if (!name) return null;
  const better = m.best_san && m.best_san !== m.san ? m.best_san : null;
  const kept = better && reveal ? `; ${better} kept it` : "";
  const behind = (loss: number) => (loss < 0.01 ? "less than 0.01" : points(loss));
  const first = m.best_uci === m.uci;
  switch (label) {
    case "book":
      return `${name}: a standard move of this opening.`;
    case "brilliant":
      return `${name}: a sound piece sacrifice, ${first ? "the engine's first choice" : "nearly the engine's first choice"}; no other move was winning.`;
    case "great": {
      const rest = m.gap != null && m.gap >= 0.01 ? `; the next best move gives away ${points(m.gap)}` : "";
      if (before?.severity === "blunder") return `${name}: it punishes the blunder${rest}.`;
      return `${name}: the one move that holds${rest}.`;
    }
    case "best":
      return `${name}: the engine's first choice.`;
    case "excellent":
    case "good":
      return `${name}: ${behind(m.loss)} behind the best move${better ? `, ${better}` : ""}.`;
    case "miss": {
      const had = mateFor(before?.eval, m.side);
      if (had != null) {
        const left = mateFor(m.eval, m.side);
        if (had === 1) return `${name}: there was mate in one${better && reveal ? `, ${better}` : ""}.`;
        if (left == null) return `${name}: there was a forced mate in ${had}, and it is gone${kept}.`;
        return `${name}: the mate in ${had} became a mate in ${left}${kept}.`;
      }
      return `${name}: a chance to press the advantage went by; it gives away ${points(m.loss)}${kept}.`;
    }
    case "blunder":
      if (mateFor(m.eval, m.side === "white" ? "black" : "white") != null) return `${name}: it allows a forced mate${kept}.`;
      return `${name}: it gives away ${points(m.loss)}${kept}.`;
    default:
      return `${name}: it gives away ${points(m.loss)}${kept}.`;
  }
}
