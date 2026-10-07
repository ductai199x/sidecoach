// Run by tests/test_page_api.py (node --test): src/lib/api.ts's one request helper, in a browser
// window (no shell), with the network and the timeout signal faked -- the timeouts are fired by
// the test, never waited for. The 2026-10-07 audit (#7): get/post/put/del never checked `r.ok`
// (a 4xx JSON error read as success), a plain-text 500 threw "SyntaxError: Unexpected token",
// and nothing had a timeout, so a sidecar that hung froze the panel's poll on stale state.
import assert from "node:assert/strict";
import { test } from "node:test";

globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.location = { pathname: "/" };

// The timeouts asked for, each a signal the test aborts as the timer would.
const timers = [];
AbortSignal.timeout = (ms) => {
  const c = new AbortController();
  timers.push({ ms, fire: () => c.abort(new DOMException("The operation timed out.", "TimeoutError")) });
  return c.signal;
};

let answer = null; // { status, body } | "hang" | null (nothing listens)
const calls = [];
globalThis.fetch = (url, init = {}) => {
  calls.push({ url, init });
  if (answer === "hang") {
    // The socket stays open: only the request's signal ends it.
    return new Promise((_, reject) => {
      if (!init.signal) return;
      init.signal.addEventListener("abort", () => reject(init.signal.reason));
    });
  }
  if (!answer) return Promise.reject(new TypeError("Failed to fetch"));
  const { status, body } = answer;
  return Promise.resolve({
    status,
    ok: status >= 200 && status < 300,
    text: async () => body,
    json: async () => JSON.parse(body),
  });
};

const api = await import("../src/lib/api.ts");
await api.connect();

test("a 4xx JSON answer is an error, its detail said", async () => {
  answer = { status: 422, body: '{"detail":"the body does not fit the request"}' };
  await assert.rejects(api.post("/api/ask", { text: "why" }), (e) => {
    assert.ok(e instanceof api.ServerError);
    assert.equal(e.status, 422);
    assert.equal(e.message, "Sidecoach answered an error (422): the body does not fit the request");
    return true;
  });
  answer = { status: 401, body: '{"ok":false,"error":"unauthorised"}' };
  await assert.rejects(api.get("/api/settings"), /answered an error \(401\): unauthorised/);
});

test("a plain-text 500 is a readable error, not a SyntaxError", async () => {
  answer = { status: 500, body: "Internal Server Error" };
  for (const call of [() => api.get("/api/poll"), () => api.post("/api/ask", { text: "x" }), () => api.put("/api/settings", {}), () => api.del("/api/pieces/own/x")]) {
    await assert.rejects(call(), (e) => {
      assert.ok(e instanceof api.ServerError, String(e));
      assert.equal(e.message, "Sidecoach answered an error (500): Internal Server Error");
      return true;
    });
  }
});

test("a 2xx answer is its JSON body", async () => {
  answer = { status: 200, body: '{"ok":true,"n":3}' };
  assert.deepEqual(await api.post("/api/ask", { text: "x" }), { ok: true, n: 3 });
  assert.equal(calls.at(-1).init.body, '{"text":"x"}');
  assert.equal(calls.at(-1).init.method, "POST");
});

test("a request that hangs ends at its timeout, as no answer", async () => {
  answer = "hang";
  timers.length = 0;
  const asked = api.panel.poll(5);
  assert.equal(timers.length, 1);
  assert.equal(timers[0].ms, api.TIMEOUT_MS);
  timers[0].fire();
  await assert.rejects(asked, (e) => {
    assert.ok(e instanceof api.TimedOut, String(e));
    return true;
  });
  // The panel says so: in a browser and in the shell alike (here, a browser).
  assert.deepEqual(await api.noAnswer(new api.TimedOut("/api/poll", api.TIMEOUT_MS)), { text: "Sidecoach isn't answering", stopped: false });
});

test("every request has a timeout; the routes that work first a longer one, or none", async () => {
  answer = { status: 200, body: '{"ok":true}' };
  const given = async (call) => {
    timers.length = 0;
    calls.length = 0;
    await call();
    assert.equal(calls.length, 1);
    return calls[0].init.signal ? timers[0].ms : null;
  };
  assert.equal(await given(() => api.panel.ask("why")), api.TIMEOUT_MS);
  assert.equal(await given(() => api.panel.poll(0)), api.TIMEOUT_MS);
  assert.equal(await given(() => api.puzzle("p1").giveup()), api.TIMEOUT_MS);
  assert.equal(await given(() => api.review("k").lines("8/8/8/8/8/8/8/8 w - - 0 1")), api.SLOW_MS);
  assert.equal(await given(() => api.stats.get({ time_class: "all", days: 0, opponents: "all" })), api.SLOW_MS);
  assert.equal(await given(() => api.puzzle("p1").open()), api.SLOW_MS);
  // Games fetched from chess.com, and a PGN saved through a dialog: as long as they take.
  assert.equal(await given(() => api.games.bringIn("chesscom")), null);
  assert.equal(await given(() => api.review("k").pgn([], false, true)), null);
  assert.equal(await given(() => api.review("k").pgn([], false, false)), api.SLOW_MS);
});

test("a sidecar started again is read from its start, not from the old one's cursor", () => {
  // Found running the panel with its sidecar stopped and started: the new one numbers its lines
  // from 1, the page kept asking for lines after the old 500th, and the reply never came.
  assert.equal(api.followCursor(500, 520), 520);
  assert.equal(api.followCursor(500, 500), 500);
  assert.equal(api.followCursor(500, undefined), 500);
  assert.equal(api.followCursor(500, 7), 0);
});

test("what the player is told, by the kind of failure", () => {
  assert.match(api.failureText(new api.ServerError(500, "boom")), /answered an error \(500\): boom\. Try again/);
  assert.match(api.failureText(new api.TimedOut("/api/x", 10_000)), /isn't answering\. Try again/);
  assert.match(api.failureText(new TypeError("Failed to fetch")), /Can't reach Sidecoach\. Check it is running/);
});
