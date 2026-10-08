// Run by tests/test_page_api.py (node --test): src/lib/review/analysis.svelte.ts, Game Review's
// Analysis. The 2026-10-07 audit: an error the analysis said (the engine's lines or the position
// not answered) stayed on screen after the analysis answered again. Now what is said belongs to
// the position on the board: asked again, the old error goes, and a new failure is said. The
// runes are plain values here ($state is what Svelte compiles), and the wait before the engine is
// asked fires at once (logic, not timing). The module has constructor parameter properties, which
// node's type stripping does not take: TypeScript (svelte-check's) turns it into JavaScript.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import ts from "typescript";

globalThis.$state = (v) => v;
globalThis.setTimeout = (fn) => { queueMicrotask(fn); return 0; };

const source = readFileSync(new URL("../src/lib/review/analysis.svelte.ts", import.meta.url), "utf8");
const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { AnalysisDesk } = await import("data:text/javascript;base64," + Buffer.from(js).toString("base64"));

const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1";
const MOVES = [{ uci: "e2e4", san: "e4", fen_after: E4, ply: 1, label: null }];

/** A sidecar that answers each kind of ask as `fail` says at the time of asking. */
function fakeApi(fail) {
  return {
    position: async (fen) => (fail.position ? { ok: false, error: "position failed" } : { ok: true, fen, side: "white", legal: {}, over: null }),
    move: async () => ({ ok: false, error: "no" }),
    lines: async (fen) => (fail.lines ? { ok: false, error: "lines failed" } : { ok: true, fen, depth: 18, lines: [] }),
    explore: async () => ({ ok: true, games: 0, moves: [] }),
  };
}

/** Every ask the board's move started has answered. */
const settled = () => new Promise((r) => setImmediate(r));

test("an old error goes once the analysis answers again", async () => {
  const fail = { lines: true, position: false };
  const desk = new AnalysisDesk(fakeApi(fail), () => {});
  desk.load(START, MOVES);
  desk.go(0);
  await settled();
  assert.equal(desk.error, "lines failed", "the failure is said");
  fail.lines = false;
  desk.go(1);
  await settled();
  assert.ok(desk.lines, "the lines came");
  assert.equal(desk.error, "", "the old error stayed on screen");
});

test("a failure of the position now asked is still said", async () => {
  const fail = { lines: false, position: false };
  const desk = new AnalysisDesk(fakeApi(fail), () => {});
  desk.load(START, MOVES);
  desk.go(0);
  await settled();
  assert.equal(desk.error, "");
  fail.position = true;
  desk.go(1);
  await settled();
  assert.equal(desk.error, "position failed");
});
