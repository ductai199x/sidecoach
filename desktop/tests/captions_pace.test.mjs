// Run by tests/test_page_api.py (node --test): the captions' box going through words longer than
// it shows (src/lib/captions.ts `Pace`, voice finding 18; Tai, 2026-10-02: never cut a line off,
// scroll through it at a reading pace, then the next line). The time is given by the test, the
// layout is a fixed number of characters a row: nothing waits, nothing is measured.
import assert from "node:assert/strict";
import { test } from "node:test";

const { KEEP_CHARS, Pace, READ_CPS, appended } = await import("../src/lib/captions.ts");

const PER_ROW = 40;
/** Words laid out `PER_ROW` characters a row. */
const rows = (text) => ({
  count: Math.max(1, Math.ceil(text.length / PER_ROW)),
  rowOf: (offset) => Math.floor(offset / PER_ROW),
});
/** The time the reader takes over `chars` characters, in ms. */
const reading = (chars) => (chars * 1000) / READ_CPS;

test("a line longer than the box is shown from its start, row by row at the reading pace", () => {
  const pace = new Pace();
  const words = "x".repeat(5 * PER_ROW); // five rows in a box of two
  pace.from(0, 0);
  // The start, not the end (the box showed rows 4 and 5 only).
  assert.equal(pace.top(rows(words), 2, words.length), 0);
  pace.advance(reading(PER_ROW) - 1, words.length);
  assert.equal(pace.top(rows(words), 2, words.length), 0, "the first row is still being read");
  pace.advance(reading(PER_ROW) + 1, words.length);
  assert.equal(pace.top(rows(words), 2, words.length), 1, "one row on, not a jump");
  pace.advance(reading(2 * PER_ROW) + 1, words.length);
  assert.equal(pace.top(rows(words), 2, words.length), 2);
  assert.ok(!pace.done(words.length));
  pace.advance(reading(words.length), words.length);
  // The end: the last two rows, as the box always ended.
  assert.equal(pace.top(rows(words), 2, words.length), 3);
  assert.ok(pace.done(words.length));
  // Nothing more to read: the box stays at its foot however long it waits.
  pace.advance(reading(10 * words.length), words.length);
  assert.equal(pace.top(rows(words), 2, words.length), 3);
});

test("the next line comes after the one being read, and from when it came once all is read", () => {
  const pace = new Pace();
  let said = "a".repeat(3 * PER_ROW - 1); // three rows, the third ends with the newline below
  pace.from(0, 0);
  // A line comes while the first is being read: the reading goes on where it was.
  pace.advance(reading(PER_ROW / 2), said.length);
  let next = appended(said, "b".repeat(2 * PER_ROW - 1), true);
  said = next.text;
  pace.shift(next.cut);
  assert.equal(next.cut, 0);
  assert.equal(next.from, 3 * PER_ROW);
  assert.equal(pace.top(rows(said), 2, said.length), 0, "still on the first line");
  pace.advance(reading(3 * PER_ROW) + 1, said.length);
  assert.equal(pace.top(rows(said), 2, said.length), 3, "the new line's first row");
  // All read; a line coming a minute later is read from when it comes, not skipped.
  const caughtUp = reading(said.length) + 1;
  pace.advance(caughtUp, said.length);
  assert.ok(pace.done(said.length));
  const later = caughtUp + 60_000;
  pace.advance(later, said.length);
  next = appended(said, "c".repeat(4 * PER_ROW - 1), true);
  said = next.text;
  pace.shift(next.cut);
  pace.advance(later, said.length);
  assert.equal(pace.top(rows(said), 2, said.length), 4, "the last row read, then the new line");
  pace.advance(later + reading(2), said.length); // the newline, then the first "c"
  assert.equal(pace.top(rows(said), 2, said.length), 5, "the new line's first row, at the reading pace");
});

test("a box coming up starts at its new line; the oldest words cut keep the reader's place", () => {
  const pace = new Pace();
  const old = "o".repeat(6 * PER_ROW - 1);
  const next = appended(old, "n".repeat(3 * PER_ROW - 1), true);
  pace.from(next.from, 0);
  assert.equal(pace.top(rows(next.text), 2, next.text.length), 6, "the new line, not the old words' end");
  // The words past KEEP_CHARS lose their oldest: the reader stays on the same character.
  const long = "w ".repeat(KEEP_CHARS / 2);
  const reader = new Pace();
  reader.from(KEEP_CHARS - 10, 0);
  const cut = appended(long, "z".repeat(50), true);
  reader.shift(cut.cut);
  assert.ok(cut.cut > 0 && cut.text.length <= KEEP_CHARS);
  assert.equal(cut.text[Math.floor(reader.read)], long[KEEP_CHARS - 10]);
  assert.equal(cut.text.slice(cut.from), "z".repeat(50));
});
