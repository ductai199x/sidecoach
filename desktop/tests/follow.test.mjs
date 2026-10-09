// Run by tests/test_page_api.py (node --test): src/lib/follow.ts, the move shown kept in view in a
// move strip. A user walk (2026-10-08) saw Game Review's strip on a phone show moves 1-4 with the
// board on move 9 (Bb6??); the strip followed each new move but not a change of its own size: a
// phone turned and back dropped its scroll to 0. A fake strip and a fake ResizeObserver here, the
// test sizing them by hand.
import assert from "node:assert/strict";
import { test } from "node:test";

const { keepInView, followOnResize, sideways } = await import("../src/lib/follow.ts");

/** A box of `n` moves laid out along one line, `w` px each, on screen at x 10..10+width. As a
 *  browser does, scrollLeft is clamped to the room, and a box no wider than its content (the
 *  list) scrolls not at all. */
function strip(n, w = 50, width = 340) {
  const box = {
    left: 10, width, n, w, _x: 0,
    get scrollWidth() { return this.oneLine ? this.n * this.w : this.width; },
    get clientWidth() { return this.width; },
    scrollHeight: 40, clientHeight: 40, scrollTop: 0, oneLine: true,
    get scrollLeft() { return this._x; },
    set scrollLeft(v) { this._x = Math.max(0, Math.min(v, this.scrollWidth - this.clientWidth)); },
    getBoundingClientRect() { return rect(this.left, this.width, 0, 40); },
    item(i) { const self = this; return { getBoundingClientRect: () => rect(self.left + i * self.w - self._x, self.w, 0, 40) }; },
    shown(i) { const r = this.item(i).getBoundingClientRect(); return r.left >= this.left && r.right <= this.left + this.width; },
  };
  return box;
}
const rect = (left, width, top, height) => ({ left, right: left + width, top, bottom: top + height, width, height });

/** A ResizeObserver whose notices the test sends: `resize(w, h)` as the browser would after a layout. */
function observer() {
  const made = [];
  class Fake {
    constructor(cb) { this.cb = cb; this.on = false; made.push(this); }
    observe() { this.on = true; }
    disconnect() { this.on = false; }
    resize(width, height) { if (this.on) this.cb([{ contentRect: { width, height } }]); }
  }
  return { Fake, made };
}

test("a move past the strip's edge comes to the middle; one clear of both edges stays put", () => {
  const box = strip(142);
  keepInView(box, box.item(17)); // Bb6, ply 18
  assert.ok(box.shown(17));
  const r = box.item(17).getBoundingClientRect();
  assert.ok(Math.abs((r.left + r.right) / 2 - (10 + 170)) <= 1, "centred");
  const at = box.scrollLeft;
  keepInView(box, box.item(18)); // the next move, still clear of the edge
  assert.equal(box.scrollLeft, at);
});

test("the strip turned back from the list finds the move shown again", () => {
  const box = strip(142);
  const { Fake, made } = observer();
  const follow = () => keepInView(box, box.item(17));
  followOnResize(box, follow, Fake);
  made[0].resize(340, 40); // the first measure: the follow's own business
  follow();
  assert.ok(box.shown(17));
  // The phone turned: the page lays the moves out as a list; the browser drops the scroll.
  box.oneLine = false; box.width = 501; box.scrollLeft = 0;
  made[0].resize(501, 300);
  assert.equal(box.scrollLeft, 0, "the list is left alone");
  // And back upright: the strip again, at 0 -- move 1 showing, the board on Bb6.
  box.oneLine = true; box.width = 340;
  assert.equal(box.shown(17), false);
  made[0].resize(340, 40);
  assert.ok(box.shown(17), "Bb6 back in view");
});

test("a narrower strip keeps the newest move it was following", () => {
  const box = strip(60, 50, 762); // wide: 60 moves, the newest at the end
  const { Fake, made } = observer();
  const follow = () => { box.scrollLeft = box.scrollWidth; };
  followOnResize(box, follow, Fake);
  made[0].resize(762, 40);
  follow();
  box.width = 322; // narrower: the browser keeps the old offset, the newest moves off the right
  assert.equal(box.shown(59), false);
  made[0].resize(322, 40);
  assert.ok(box.shown(59));
});

test("only a change of size follows: not the first measure, not the same size again, not after the stop", () => {
  const box = strip(142);
  const { Fake, made } = observer();
  let calls = 0;
  const stop = followOnResize(box, () => calls++, Fake);
  made[0].resize(340, 40);
  made[0].resize(340, 40);
  assert.equal(calls, 0);
  made[0].resize(300, 40);
  assert.equal(calls, 1);
  stop();
  made[0].resize(340, 40);
  assert.equal(calls, 1);
  assert.equal(sideways(box), true);
});

test("no ResizeObserver (an old WebView): nothing breaks", () => {
  const stop = followOnResize(strip(10), () => {}, undefined);
  stop();
});
