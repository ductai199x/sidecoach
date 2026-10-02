// Run by tests/test_page_api.py (node --test): src/lib/api.ts in a window of the desktop shell,
// with the shell's commands and the network faked. Plain JS, so svelte-check leaves it be;
// node strips the types of the module it tests.
import assert from "node:assert/strict";
import { test } from "node:test";

const shell = { status: { port: 1111, error: null } };
let live = "http://127.0.0.1:1111";
const calls = [];

globalThis.isTauri = true;
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.location = { pathname: "/settings" };
globalThis.__TAURI_INTERNALS__ = {
  invoke: async (cmd) => (cmd === "sidecar_status" ? shell.status : cmd === "sidecar_token" ? "tok" : null),
};
globalThis.fetch = async (url, init = {}) => {
  calls.push({ url, body: init.body, auth: init.headers?.Authorization });
  if (!url.startsWith(live)) throw new TypeError("Failed to fetch");
  return { json: async () => ({ ok: true, from: url }) };
};

const api = await import("../src/lib/api.ts");
await api.connect();

test("a window follows the sidecar a Restart started on another port", async () => {
  calls.length = 0;
  shell.status = { port: 2222, error: null };
  live = "http://127.0.0.1:2222";
  const got = await api.post("/api/settings", { a: 1 });
  assert.equal(got.from, "http://127.0.0.1:2222/api/settings");
  assert.deepEqual(calls.map((c) => c.url), ["http://127.0.0.1:1111/api/settings", "http://127.0.0.1:2222/api/settings"]);
  // The same request again: its body and the bearer.
  assert.deepEqual(calls.map((c) => c.body), ['{"a":1}', '{"a":1}']);
  assert.deepEqual(calls.map((c) => c.auth), ["Bearer tok", "Bearer tok"]);
  // From now on, straight to the new one.
  calls.length = 0;
  await api.get("/api/poll");
  assert.deepEqual(calls.map((c) => c.url), ["http://127.0.0.1:2222/api/poll"]);
});

test("a sidecar that has not moved: the failure stands, nothing is sent twice", async () => {
  calls.length = 0;
  live = "nowhere";
  await assert.rejects(api.get("/api/poll"), TypeError);
  assert.equal(calls.length, 1);
  // Restarting (no port yet): the same.
  calls.length = 0;
  shell.status = { port: null, error: null };
  await assert.rejects(api.get("/api/poll"), TypeError);
  assert.equal(calls.length, 1);
  live = "http://127.0.0.1:2222";
  shell.status = { port: 2222, error: null };
});
