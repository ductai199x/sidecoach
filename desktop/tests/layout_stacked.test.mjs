// Run by tests/test_page_api.py (node --test): src/lib/layout.ts's mobileBoard, the board's side
// when the panel is stacked (the board over the coach). At 720x900 the board took the whole width
// (640 px) and pushed the ask box and the Hint / Best move / Line chips under the fold, one line
// of the coach on screen (Tai's 720 px window, 2026-10-09). Now the board leaves room for the
// coach at its least, the ask box and the chips. A fake page: each box's height is given, the
// board's side is what --mboard (or the stylesheet's default) makes it. The module's import of
// the panel's store is replaced by nothing (mobileBoard does not use it); TypeScript turns the
// rest into JavaScript.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

const path = process.env.LAYOUT_TS ?? new URL("../src/lib/layout.ts", import.meta.url);
const source = readFileSync(path, "utf8").replace(/import \{ prefs \} from "\.\/panel\.svelte";/, "const prefs = {};");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { mobileBoard } = await import("data:text/javascript;base64," + Buffer.from(js).toString("base64"));

const BAR = 61;
const css = (o = {}) => ({ paddingTop: "0px", paddingBottom: "0px", rowGap: "0px", marginTop: "0px", marginBottom: "0px", borderTopWidth: "0px", borderBottomWidth: "0px", minHeight: "0px", display: "block", ...o });
const box = (h, cs = css(), cls = []) => ({ cs, classList: { contains: (c) => cls.includes(c) }, getBoundingClientRect: () => ({ height: typeof h === "function" ? h() : h }) });

/** The stacked panel in a window `w` x `h`: the board's column (the board, then the handle and the
 * game line, `under(side)` tall: they wrap as the column narrows), and the panel under it (the
 * moves' line, the coach at least 80 px, the ask box and the chips). */
function page(w, h, under = () => 89) {
  const vars = new Map();
  const classes = new Set(["stacked"]);
  globalThis.document = {
    documentElement: {
      classList: { contains: (c) => classes.has(c), toggle: (c, on) => (on ? classes.add(c) : classes.delete(c)) },
      style: { getPropertyValue: (k) => vars.get(k) ?? "", setProperty: (k, v) => vars.set(k, v), removeProperty: (k) => vars.delete(k) },
    },
  };
  globalThis.getComputedStyle = (el) => el.cs;
  const mainW = w;
  const side = () => (vars.has("--mboard") ? parseFloat(vars.get("--mboard")) : Math.min(mainW - 20, 640));
  const wrap = box(side);
  const left = box(() => side() + under(side()));
  left.querySelector = () => wrap;
  const right = box(1, css({ paddingTop: "8px", paddingBottom: "8px", borderTopWidth: "1px", borderBottomWidth: "1px", rowGap: "8px" }));
  right.children = [box(40), box(999, css({ minHeight: "80px" }), ["coach"]), box(88), box(0, css({ display: "none" }))];
  const main = { clientWidth: mainW, clientHeight: h - BAR, cs: css({ paddingTop: "8px", paddingBottom: "10px", rowGap: "8px" }) };
  main.querySelector = (s) => (s === "#left" ? left : s === "#right" ? right : null);
  const needed = () => 8 + side() + under(side()) + 8 + (1 + 8 + 40 + 8 + 80 + 8 + 88 + 8 + 1) + 10;
  return { main, side, vars, classes, fits: () => needed() <= main.clientHeight };
}

test("at 720x900 the board leaves the coach, the ask box and the chips on screen", () => {
  const p = page(720, 900);
  mobileBoard(p.main, null);
  assert.ok(p.fits(), `the board at ${p.side()} px pushes the chips under the fold`);
  // 839 px of main, less its padding (18), the handle and game line (89), the gap (8) and the
  // panel with the coach at its least (242): the board takes the rest, as the real page's 482.
  assert.equal(p.side(), 839 - 18 - 89 - 8 - 242, "and takes the rest of the room");
});

test("every stacked window where the whole width would not fit gets a board that fits", () => {
  for (const [w, h] of [[600, 900], [720, 800], [720, 900], [640, 700], [560, 820]]) {
    const p = page(w, h);
    mobileBoard(p.main, null);
    assert.ok(p.fits(), `${w}x${h}: board ${p.side()}`);
  }
});

test("where the whole width fits (a phone), the stylesheet's default is left as it is", () => {
  for (const [w, h] of [[360, 800], [412, 900], [720, 1080]]) {
    const p = page(w, h);
    mobileBoard(p.main, null);
    assert.equal(p.vars.has("--mboard"), false, `${w}x${h}`);
  }
});

test("the size the player dragged to is kept where it fits, and only shown smaller where it does not", () => {
  const small = page(720, 900);
  mobileBoard(small.main, 300);
  assert.equal(small.side(), 300);
  const roomy = page(720, 1080);
  mobileBoard(roomy.main, 600);
  assert.equal(roomy.side(), 600);
  const big = page(720, 900);
  mobileBoard(big.main, 640);
  assert.ok(big.fits() && big.side() < 640);
});

test("rows under the board that wrap taller as it narrows are measured again", () => {
  const p = page(720, 900, (side) => (side < 560 ? 122 : 89));
  mobileBoard(p.main, null);
  assert.ok(p.fits(), `board ${p.side()}`);
});
