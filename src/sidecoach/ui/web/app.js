"use strict";

const FILES = "abcdefgh";
const el = (id) => document.getElementById(id);
let cursor = 0;
let running = false;
let lastBoardKey = null;
let lastMovesLen = -1;
let coachCount = 0;

// The backend is a local FastAPI server: poll for state, POST the controls. Same shape the
// rest of the page expects, so nothing below cares that the transport is HTTP.
function post(url, body) {
  return fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  }).then((r) => r.json());
}
const api = {
  poll: (since) => fetch("/api/poll?since=" + (since | 0)).then((r) => r.json()),
  start: () => post("/api/start"),
  stop: () => post("/api/stop"),
  hint: () => post("/api/hint"),
  move: () => post("/api/move"),
  line: () => post("/api/line"),
  flip: () => post("/api/flip"),
  analyze: () => post("/api/analyze"),
  ask: (text) => post("/api/ask", { text }),
  set_voice: (voice) => post("/api/voice", { voice }),
  set_muted: (muted) => post("/api/muted", { muted }),
};

// --- rendering ---------------------------------------------------------------

// A piece char from the FEN -> sprite class: 'K' -> wK, 'n' -> bN.
const spriteOf = (ch) => (ch === ch.toUpperCase() ? "w" : "b") + ch.toUpperCase();

// "e2e4" -> the two square names it touches, for the last-move highlight.
function moveSquares(uci) {
  if (!uci || uci.length < 4) return new Set();
  return new Set([uci.slice(0, 2), uci.slice(2, 4)]);
}

function renderBoard(fen, bottom, lastMove) {
  const board = el("board");
  if (!fen) {
    if (lastBoardKey !== null) { board.innerHTML = ""; lastBoardKey = null; }
    return;
  }
  const key = fen + "|" + (bottom || "white") + "|" + (lastMove || "");
  if (key === lastBoardKey) return;
  lastBoardKey = key;

  // The FEN's placement, rank 8 down to rank 1, expanded to 8 cells per rank.
  const ranks = fen.split(" ")[0].split("/").map((row) => {
    const cells = [];
    for (const ch of row) {
      if (ch >= "1" && ch <= "8") { for (let i = 0; i < +ch; i++) cells.push(null); }
      else cells.push(ch);
    }
    return cells;
  });
  const blackBottom = bottom === "black";
  const hl = moveSquares(lastMove);

  // Draw display rows top to bottom. Each display cell maps to an absolute (rank, file):
  // White at the bottom shows rank 8 on top and file a at the left; Black at the bottom
  // shows rank 1 on top and file h at the left. The square colour and the coordinate
  // labels come from the absolute square, as on chess.com.
  const frag = document.createDocumentFragment();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const rank = blackBottom ? r : 7 - r;        // 0 = rank 1
      const file = blackBottom ? 7 - c : c;        // 0 = file a
      const name = FILES[file] + (rank + 1);
      const sq = document.createElement("div");
      sq.className = "sq " + (((rank + file) % 2 === 0) ? "dark" : "light");
      if (hl.has(name)) sq.classList.add("hl");
      const piece = ranks[7 - rank][file];
      if (piece) {
        const pc = document.createElement("div");
        pc.className = "pc " + spriteOf(piece);
        sq.appendChild(pc);
      }
      if (c === 0) {   // rank labels down the left edge
        const s = document.createElement("span"); s.className = "coord rank"; s.textContent = rank + 1; sq.appendChild(s);
      }
      if (r === 7) {   // file letters along the bottom edge
        const s = document.createElement("span"); s.className = "coord file"; s.textContent = FILES[file]; sq.appendChild(s);
      }
      frag.appendChild(sq);
    }
  }
  board.replaceChildren(frag);
}

function renderMoves(sans) {
  if (sans.length === lastMovesLen) return;
  lastMovesLen = sans.length;
  const list = el("moves");
  const frag = document.createDocumentFragment();
  for (let i = 0; i < sans.length; i += 2) {
    const row = document.createElement("div"); row.className = "mrow";
    const num = document.createElement("span"); num.className = "num"; num.textContent = (i / 2 + 1) + ".";
    const w = document.createElement("span"); w.className = "mv"; w.textContent = sans[i] || "";
    const b = document.createElement("span"); b.className = "mv"; b.textContent = sans[i + 1] || "";
    if (i + 1 === sans.length - 1) b.classList.add("cur");
    else if (i === sans.length - 1) w.classList.add("cur");
    row.append(num, w, b);
    frag.appendChild(row);
  }
  list.replaceChildren(frag);
  const wrap = list.parentElement; wrap.scrollTop = wrap.scrollHeight;
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
  el("following").textContent = s.following ? "following a game" : "waiting for a start";
  // The pipeline's orientation is what it read from the screen; a view flip only turns
  // the panel's board around.
  const base = s.bottom || "white";
  const shown = s.view_flip ? (base === "white" ? "black" : "white") : base;
  renderBoard(s.fen, shown, s.last_move);
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
  wire();
  const first = await api.poll(0);
  if (first.state) { fillVoices(first.state.voices || [], first.state.voice); renderState(first.state); }
  if (first.lines) for (const line of first.lines) addLine(line);
  if (first.cursor !== undefined) cursor = first.cursor;
  setInterval(tick, 250);
}

main();
