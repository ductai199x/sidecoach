// Run by tests/test_page_api.py (node --test): src/lib/review/lineplayer.svelte.ts, the stepped
// line the review's Best / Follow-up and a puzzle's given-up solution share, on a manual clock.
// The 2026-10-07 audit (#8): Give up played the solution itself, a move every 400 ms, a 5-move
// line over in 2 s; Tai's rule is that a line is stepped and never runs by itself. The runes are
// plain values here ($state is what Svelte compiles; node runs the class as it is).
import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.$state = (v) => v;

const { LinePlayer, LINE_MS, TAKEBACK_MS } = await import("../src/lib/review/lineplayer.svelte.ts");

/** Timers fired only when the test moves the clock. */
function manualClock() {
  let now = 0;
  let next = 1;
  const due = new Map();
  return {
    set: (fn, ms) => { const id = next++; due.set(id, { at: now + ms, fn }); return id; },
    clear: (id) => { due.delete(id); },
    pending: () => due.size,
    advance(ms) {
      const end = now + ms;
      for (;;) {
        let first = null;
        for (const [id, t] of due) if (t.at <= end && (!first || t.at < first[1].at)) first = [id, t];
        if (!first) break;
        due.delete(first[0]);
        now = first[1].at;
        first[1].fn();
      }
      now = end;
    },
  };
}

function player() {
  const clock = manualClock();
  const heard = [];
  const shown = [];
  const p = new LinePlayer({ sound: (i) => heard.push(i), changed: (i) => shown.push(i) }, clock);
  return { p, clock, heard, shown };
}

test("a line shows its first move and nothing more until asked", () => {
  const { p, clock, heard, shown } = player();
  p.reset(5);
  p.showFirst();
  assert.equal(p.i, -1);
  clock.advance(TAKEBACK_MS);
  assert.equal(p.i, 0);
  assert.deepEqual(heard, [0]);
  // However long it stays, it does not run on.
  assert.equal(clock.pending(), 0);
  clock.advance(60_000);
  assert.equal(p.i, 0);
  assert.deepEqual(shown, [0]);
  assert.equal(p.where, "Move 1 of 5");
});

test("the player steps it, either way, within its ends; a step on is heard, a step back not", () => {
  const { p, heard } = player();
  p.reset(3);
  p.step(1);
  p.step(1);
  p.step(1);
  p.step(1);
  assert.equal(p.i, 2);
  assert.ok(p.atEnd);
  assert.deepEqual(heard, [0, 1, 2]);
  p.step(-1);
  assert.equal(p.i, 1);
  p.toEnd(false);
  assert.equal(p.i, -1);
  assert.ok(p.atStart);
  assert.equal(p.where, "Before the line");
  assert.deepEqual(heard, [0, 1, 2]);
});

test("Play runs it at a reading pace, and any step stops it", () => {
  const { p, clock } = player();
  p.reset(5, 0);
  p.playPause();
  assert.ok(p.autoplay);
  clock.advance(0);
  assert.equal(p.i, 1);
  clock.advance(LINE_MS - 1);
  assert.equal(p.i, 1, "not before LINE_MS");
  clock.advance(1);
  assert.equal(p.i, 2);
  p.step(-1);
  assert.equal(p.autoplay, false);
  assert.equal(clock.pending(), 0);
  clock.advance(10 * LINE_MS);
  assert.equal(p.i, 1);
});

test("Play at the end plays it again from the start, and stops at the end", () => {
  const { p, clock } = player();
  p.reset(2, 1);
  p.playPause();
  assert.equal(p.i, -1);
  clock.advance(TAKEBACK_MS);
  assert.equal(p.i, 0);
  clock.advance(LINE_MS);
  assert.equal(p.i, 1);
  clock.advance(LINE_MS);
  assert.equal(p.autoplay, false);
  assert.equal(clock.pending(), 0);
});

test("the keys: arrows step, Home and End its ends, Space plays", () => {
  const { p } = player();
  p.reset(4);
  const key = (k) => {
    let prevented = false;
    const used = p.key({ key: k, target: null, preventDefault: () => (prevented = true) });
    return used && prevented;
  };
  assert.ok(key("ArrowRight"));
  assert.equal(p.i, 0);
  assert.ok(key("End"));
  assert.equal(p.i, 3);
  assert.ok(key("ArrowLeft"));
  assert.equal(p.i, 2);
  assert.ok(key("Home"));
  assert.equal(p.i, -1);
  assert.ok(key(" "));
  assert.ok(p.autoplay);
  p.pause();
  assert.equal(key("x"), false);
});
