// Run by tests/test_page_api.py (node --test): what the panel says when a poll gets no good
// answer (src/lib/api.ts noAnswer), in a window of the desktop shell with the shell's
// commands and the network faked. "Sidecoach stopped" only when the shell says the
// sidecar's process ended (Tai, 2026-10-02), never from failed polls of a live one.
import assert from "node:assert/strict";
import { test } from "node:test";

const shell = { status: { port: 1111, error: null, exited: false }, asked: 0 };
let answer = null; // null: nothing listens; else { status, body }

globalThis.isTauri = true;
globalThis.window = globalThis;
globalThis.addEventListener = () => {};
globalThis.location = { pathname: "/" };
globalThis.__TAURI_INTERNALS__ = {
  invoke: async (cmd) => {
    if (cmd === "sidecar_status") {
      shell.asked++;
      return shell.status;
    }
    return cmd === "sidecar_token" ? "tok" : null;
  },
};
globalThis.fetch = async () => {
  if (!answer) throw new TypeError("Failed to fetch");
  const { status, body } = answer;
  return { status, ok: status >= 200 && status < 300, text: async () => body, json: async () => JSON.parse(body) };
};

const api = await import("../src/lib/api.ts");
await api.connect();

async function said() {
  try {
    await api.panel.poll(0);
    return "answered";
  } catch (e) {
    return await api.noAnswer(e);
  }
}

test("a live sidecar answering an error shows the error, not stopped", async () => {
  shell.status = { port: 1111, error: null, exited: false };
  answer = { status: 500, body: "Internal Server Error" };
  for (let i = 0; i < 20; i++) {
    assert.deepEqual(await said(), { text: "Sidecoach answered an error (500): Internal Server Error", stopped: false });
  }
  answer = { status: 503, body: '{"detail":"busy"}' };
  assert.deepEqual(await said(), { text: "Sidecoach answered an error (503): busy", stopped: false });
  answer = { status: 200, body: '{"cursor":3}' };
  assert.equal(await said(), "answered");
});

test("no answer while the shell holds the process alive: nothing said, however many", async () => {
  answer = null;
  shell.status = { port: 1111, error: null, exited: false };
  for (let i = 0; i < 20; i++) assert.equal(await said(), null);
});

test("the shell says the process ended: stopped, at the first poll", async () => {
  answer = null;
  shell.status = { port: 1111, error: null, exited: true };
  shell.asked = 0;
  assert.deepEqual(await said(), { text: "Sidecoach stopped", stopped: true });
  assert.ok(shell.asked >= 1);
});
