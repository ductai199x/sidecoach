// Run by tests/test_page_api.py (node --test): src/lib/review/why.ts, the "why this label" line
// under a move's headline in Game Review (the audit of 2026-10-07, features #1). Each label's
// line says what earned it, from the numbers the review kept, by the rule that gives it
// (docs/move-classification.md); a move the player could retry names the better move only when
// the card may (`reveal`).
import assert from "node:assert/strict";
import { test } from "node:test";

const { why, points } = await import("../src/lib/review/why.ts");

/** A move as the review page gets it: White's Nf3-less move with Nf3 best, by default. */
const move = (over = {}) => ({
  ply: 11, san: "Nge2", side: "white", uci: "g1e2", label: "best", loss: 0, severity: "none",
  best_uci: "g1f3", best_san: "Nf3", gap: null, eval: { cp: 20, mate: null, result: null },
  ...over,
});

test("a loss reads to two places, cut so it never reads as the next label's line", () => {
  assert.equal(points(0.0997), "0.09");
  assert.equal(points(0.1192), "0.11");
  assert.equal(points(0.29), "0.29");
  assert.equal(points(0.2), "0.20");
});

test("inaccuracy, mistake and blunder say what the move gave away, and the better move when the card may", () => {
  assert.equal(why(move({ label: "mistake", loss: 0.1192 }), null, true), "It gives away 0.11; Nf3 kept it.");
  assert.equal(why(move({ label: "mistake", loss: 0.1192 }), null, false), "It gives away 0.11.");
  assert.equal(why(move({ label: "inaccuracy", loss: 0.0997 }), null, false), "It gives away 0.09.");
  assert.equal(why(move({ label: "blunder", loss: 0.2409 }), null, true), "It gives away 0.24; Nf3 kept it.");
});

test("a blunder into a forced mate says so", () => {
  const into = move({ label: "blunder", loss: 0.12, eval: { cp: null, mate: -3, result: null } });
  assert.equal(why(into, null, false), "It allows a forced mate.");
  const black = move({ side: "black", label: "blunder", loss: 0.3, eval: { cp: null, mate: 2, result: null } });
  assert.equal(why(black, null, true), "It allows a forced mate; Nf3 kept it.");
});

test("a great move says every other move fell behind, or that it answered a blunder", () => {
  const great = move({ san: "Nf3", uci: "g1f3", label: "great", gap: 0.1432 });
  assert.equal(why(great, move({ severity: "mistake" }), true), "The one move that holds; the next best move gives away 0.14.");
  assert.equal(why(great, move({ severity: "blunder" }), true), "It punishes the blunder; the next best move gives away 0.14.");
  assert.equal(why({ ...great, gap: null }, move({ severity: "blunder" }), true), "It punishes the blunder.");
});

test("brilliant, best, excellent, good and book", () => {
  assert.equal(why(move({ san: "Nf3", uci: "g1f3", label: "brilliant" }), null, false), "A sound piece sacrifice, the engine's first choice; no other move was winning.");
  assert.equal(why(move({ label: "brilliant", loss: 0.0025 }), null, false), "A sound piece sacrifice, nearly the engine's first choice; no other move was winning.");
  // Best and book: the card's own sentence already says it, so no line (the walk of 2026-10-09).
  assert.equal(why(move({ san: "Nf3", uci: "g1f3" }), null, false), null);
  assert.equal(why(move({ label: "excellent", loss: 0.0044 }), null, false), "Less than 0.01 behind the best move, Nf3.");
  assert.equal(why(move({ label: "excellent", loss: 0.011 }), null, false), "0.01 behind the best move, Nf3.");
  assert.equal(why(move({ label: "good", loss: 0.0404 }), null, false), "0.04 behind the best move, Nf3.");
  assert.equal(why(move({ label: "book", loss: 0.0175 }), null, false), null);
});

test("a miss names the mate let go, or the chance", () => {
  const had = (mate) => move({ eval: { cp: null, mate, result: null } });
  const miss = (over) => move({ label: "miss", loss: 0.0246, ...over });
  assert.equal(why(miss({ eval: { cp: 900, mate: null, result: null } }), had(3), false), "There was a forced mate in 3, and it is gone.");
  assert.equal(why(miss({ eval: { cp: null, mate: 8, result: null } }), had(3), true), "The mate in 3 became a mate in 8; Nf3 kept it.");
  assert.equal(why(miss({ eval: { cp: null, mate: 2, result: null } }), had(1), true), "There was mate in one, Nf3.");
  assert.equal(why(miss({ eval: { cp: null, mate: 2, result: null } }), had(1), false), "There was mate in one.");
  // Black's mates are negative from White's side.
  const blacks = move({ side: "black", label: "miss", loss: 0.05, eval: { cp: -900, mate: null, result: null } });
  assert.equal(why(blacks, had(-2), false), "There was a forced mate in 2, and it is gone.");
  assert.equal(why(miss({ loss: 0.1048 }), had(null), true), "A chance to press the advantage went by; it gives away 0.10; Nf3 kept it.");
});

test("a move with no label has no line", () => {
  assert.equal(why(move({ label: null }), null, true), null);
});
