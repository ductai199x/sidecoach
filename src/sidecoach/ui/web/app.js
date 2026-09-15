"use strict";

// Filled glyphs for both colours; colour comes from a CSS class so white and black match.
const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

const el = (id) => document.getElementById(id);
let api = null;
let cursor = 0;
let running = false;
let lastFen = null;
let lastMovesLen = -1;
let coachCount = 0;

function ready() {
  return new Promise((resolve) => {
    if (window.pywebview && window.pywebview.api) return resolve(window.pywebview.api);
    window.addEventListener("pywebviewready", () => resolve(window.pywebview.api), { once: true });
  });
}

// --- rendering ---------------------------------------------------------------

function renderBoard(fen, bottom) {
  const board = el("board");
  if (!fen) {
    if (lastFen !== null) { board.innerHTML = ""; lastFen = null; }
    return;
  }
  if (fen === lastFen && board.dataset.bottom === bottom) return;
  lastFen = fen;
  board.dataset.bottom = bottom || "white";

  const rows = fen.split(" ")[0].split("/"); // rank 8 .. rank 1
  let grid = [];
  for (const row of rows) {
    const cells = [];
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") { for (let i = 0; i < +ch; i++) cells.push(null); }
      else cells.push(ch);
    }
    grid.push(cells);
  }
  const blackBottom = bottom === "black";
  const order = blackBottom ? [...grid].reverse().map((r) => [...r].reverse()) : grid;

  const frag = document.createDocumentFragment();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const div = document.createElement("div");
      // colour of a1 dark; parity depends on absolute rank/file, not display order.
      const rank = blackBottom ? r : 7 - r;
      const file = blackBottom ? 7 - c : c;
      div.className = "sq " + (((rank + file) % 2 === 0) ? "dark" : "light");
      const piece = order[r][c];
      if (piece) {
        const s = document.createElement("span");
        s.className = piece === piece.toUpperCase() ? "wp" : "bp";
        s.textContent = GLYPH[piece.toLowerCase()];
        div.appendChild(s);
      }
      frag.appendChild(div);
    }
  }
  board.replaceChildren(frag);
}

function renderMoves(sans) {
  if (sans.length === lastMovesLen) return;
  lastMovesLen = sans.length;
  const ol = el("moves");
  const frag = document.createDocumentFragment();
  for (let i = 0; i < sans.length; i += 2) {
    const num = document.createElement("li"); num.className = "num"; num.textContent = (i / 2 + 1) + ".";
    const w = document.createElement("li"); w.className = "mv"; w.textContent = sans[i] || "";
    const b = document.createElement("li"); b.className = "mv"; b.textContent = sans[i + 1] || "";
    if (i + 1 >= sans.length - 1) { (sans[i + 1] ? b : w).classList.add("last"); }
    frag.append(num, w, b);
  }
  ol.replaceChildren(frag);
  const wrap = ol.parentElement; wrap.scrollTop = wrap.scrollHeight;
}

function addLine(line) {
  const term = el("terminal");
  const row = document.createElement("div");
  row.className = "row lvl-" + line.level;
  const m = document.createElement("span"); m.className = "m"; m.textContent = line.text;
  row.appendChild(m);
  term.appendChild(row);
  if (el("autoscroll").checked) term.scrollTop = term.scrollHeight;

  if (line.level === "coach") {
    const c = el("coach");
    const msg = document.createElement("div"); msg.className = "msg"; msg.textContent = line.text;
    c.appendChild(msg);
    c.scrollTop = c.scrollHeight;
    coachCount++;
  }
}

function renderState(s) {
  el("status").textContent = s.status || "";
  el("side").textContent = s.user_side ? ("you play " + s.user_side) : "—";
  el("ply").textContent = "ply " + (s.ply || 0);
  const f = el("following");
  f.textContent = s.following ? "following a game" : "waiting for a start";
  f.classList.toggle("muted", !s.following);
  renderBoard(s.fen, s.bottom);
  renderMoves(s.moves || []);
  setRunning(s.running);
}

function setRunning(on) {
  running = on;
  const b = el("startStop");
  b.textContent = on ? "Stop" : "Start";
  b.classList.toggle("running", on);
}

// --- polling -----------------------------------------------------------------

async function tick() {
  try {
    const res = await api.poll(cursor);
    if (res.cursor !== undefined) cursor = Math.max(cursor, res.cursor);
    if (res.lines) for (const line of res.lines) addLine(line);
    if (res.state) renderState(res.state);
  } catch (e) {
    /* window closing or api not ready; ignore this tick */
  }
}

// --- controls ----------------------------------------------------------------

function fillVoices(voices, current) {
  const sel = el("voice");
  if (sel.dataset.filled) return;
  sel.dataset.filled = "1";
  for (const v of voices) {
    const o = document.createElement("option");
    o.value = v; o.textContent = v;
    if (v === current) o.selected = true;
    sel.appendChild(o);
  }
}

function wire() {
  el("startStop").onclick = () => (running ? api.stop() : api.start());
  el("flip").onclick = () => api.flip();
  el("analyze").onclick = () => api.analyze();
  el("mute").onchange = (e) => api.set_muted(e.target.checked);
  el("voice").onchange = (e) => api.set_voice(e.target.value);
  el("askBtn").onclick = sendAsk;
  el("askInput").addEventListener("keydown", (e) => { if (e.key === "Enter") sendAsk(); });
  for (const b of document.querySelectorAll(".quick button")) {
    b.onclick = () => api[b.dataset.cmd]();
  }
  el("clearTerm").onclick = () => { el("terminal").replaceChildren(); };
}

function sendAsk() {
  const input = el("askInput");
  const text = input.value.trim();
  if (!text) return;
  api.ask(text);
  input.value = "";
}

async function main() {
  api = await ready();
  wire();
  const first = await api.poll(0);
  if (first.state) { fillVoices(first.state.voices || [], first.state.voice); renderState(first.state); }
  setInterval(tick, 250);
}

main();
