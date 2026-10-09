// Run by tests/test_page_api.py (node --test): src/lib/gamefilters.ts, what Your games' filters
// hide. A game brought in while Lost + Rapid are on is hidden and the import line says so, with
// Clear filters; the starting filters hide nothing.
import assert from "node:assert/strict";
import { test } from "node:test";

const { NO_FILTERS, passes, seenOf, hiddenWords, broughtInWords } = await import("../src/lib/gamefilters.ts");

const NOW = 1_800_000_000;
/** A library row: the player's rapid loss as White on chess.com, by default. */
const game = (over = {}) => ({
  key: "chesscom:1", source: "chesscom", added_at: NOW - 100, played_at: NOW - 3600, white: "nosegazm", black: "Ace",
  white_elo: 1000, black_elo: 1100, result: "0-1", time_class: "rapid", user_side: "white", reviewed: false,
  accuracy: null, performance: null, ...over,
});
const LOST_RAPID = { ...NO_FILTERS, outcome: "lost", speed: "rapid" };

test("the starting filters show every game", () => {
  const games = [game(), game({ key: "pgn:1", source: "pgn", user_side: null, result: "1-0", time_class: null, played_at: null, added_at: 5 })];
  assert.deepEqual(games.filter((g) => passes(g, NO_FILTERS, NOW)).length, 2);
});

test("Lost + Rapid hide a pasted game whose side is unknown, and the import line says so", () => {
  const old = game();
  const pasted = game({ key: "pgn:x", source: "pgn", user_side: null, result: "1-0", time_class: "blitz", white: "A", black: "B" });
  const list = [old, pasted];
  const shown = list.filter((g) => passes(g, LOST_RAPID, NOW));
  assert.deepEqual(shown.map((g) => g.key), ["chesscom:1"]);
  const seen = seenOf(["pgn:x"], list, shown);
  assert.deepEqual(seen, { shown: 0, hidden: 1 });
  assert.equal(broughtInWords(1, seen, "", "Press Review on one."), "1 game brought in — hidden by your filters.");
  // Every filter back to the start: it shows, and the line is the import's own.
  const all = list.filter((g) => passes(g, NO_FILTERS, NOW));
  assert.equal(broughtInWords(1, seenOf(["pgn:x"], list, all), "", "Press Review on one."), null);
});

test("some of the new games hidden: how many, the rest of the line kept", () => {
  const seen = { shown: 2, hidden: 1 };
  assert.equal(
    broughtInWords(3, seen, " · 1 already here", "Press Review on one."),
    "3 games brought in — 1 hidden by your filters · 1 already here. Press Review on one.",
  );
  assert.equal(broughtInWords(3, { shown: 0, hidden: 3 }, "", "x"), "3 games brought in — all hidden by your filters.");
});

test("a game removed since is neither shown nor hidden", () => {
  assert.deepEqual(seenOf(["gone"], [game()], [game()]), { shown: 0, hidden: 0 });
});

test("the search hides too, and the list's end says how many", () => {
  const list = [game(), game({ key: "chesscom:2", black: "Zed" })];
  const shown = list.filter((g) => passes(g, { ...NO_FILTERS, query: "  zed " }, NOW));
  assert.deepEqual(shown.map((g) => g.key), ["chesscom:2"]);
  assert.equal(hiddenWords(list.length - shown.length), "1 game hidden by your filters.");
  assert.equal(hiddenWords(127), "127 games hidden by your filters.");
});

test("the date filter hides older games", () => {
  const old = game({ played_at: NOW - 40 * 86400 });
  assert.equal(passes(old, { ...NO_FILTERS, since: 30 }, NOW), false);
  assert.equal(passes(old, { ...NO_FILTERS, since: 90 }, NOW), true);
});
