// The puzzle page (/puzzle/<id>): a board with chess.com's manners, the coach beside
// it, the room's log below. The room lives on the server (ui/puzzle_room.py); this page
// polls it and sends what the player does.
//
// The board: click a piece of yours and it lights yellow with its legal squares dotted
// (none if it cannot move, the king in check); click a square to move there, or drag
// the piece and drop it. A right-click marks a square red; a right-drag draws an arrow;
// a left-click anywhere clears them. Moves slide, captures fade, a pawn reaching the
// last rank offers a picker over the square. A wrong move lands, shows red, and comes
// back. The coach says nothing unasked: Hint, Best move, Line, or a typed question.

"use strict";

const FILES = "abcdefgh";
const el = (id) => document.getElementById(id);
const post = (path, body) => fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }).then((r) => r.json());
const PUZZLE_ID = decodeURIComponent(location.pathname.split("/").filter(Boolean).pop() || "");
const api = {
  open: () => post(`/api/puzzle/${PUZZLE_ID}/open`),
  poll: (since) => fetch(`/api/puzzle/${PUZZLE_ID}/poll?since=${since | 0}`).then((r) => r.json()),
  targets: (square) => post(`/api/puzzle/${PUZZLE_ID}/targets`, { square }),
  move: (uci) => post(`/api/puzzle/${PUZZLE_ID}/move`, { uci }),
  retract: () => post(`/api/puzzle/${PUZZLE_ID}/retract`),
  giveup: () => post(`/api/puzzle/${PUZZLE_ID}/giveup`),
  ask: (text) => post(`/api/puzzle/${PUZZLE_ID}/ask`, { text }),
  voice: (on) => post(`/api/puzzle/${PUZZLE_ID}/voice`, { on }),
  close: () => post(`/api/puzzle/${PUZZLE_ID}/close`),
};

// --- the position ---------------------------------------------------------------------------

let state = null;        // the room's state as last polled
let bottom = "white";    // the side at the bottom
let cursor = 0;
let nextIds = [];

function placement(fen) {
  // square -> piece letter, from the FEN's first field
  const out = new Map();
  const rows = fen.split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") { file += +ch; continue; }
      out.set(FILES[file] + (8 - r), ch);
      file++;
    }
  }
  return out;
}
const spriteOf = (letter) => (letter === letter.toUpperCase() ? "w" : "b") + letter.toUpperCase();
const colourOf = (letter) => (letter === letter.toUpperCase() ? "white" : "black");

function cellOf(square) {
  // display column/row (0..7 from the left/top) of a square for the current orientation
  const file = FILES.indexOf(square[0]), rank = +square[1] - 1;
  const col = bottom === "black" ? 7 - file : file;
  const row = bottom === "black" ? rank : 7 - rank;
  return { x: col, y: row };
}
function squareAt(x, y) {
  // display column/row -> square name
  const file = bottom === "black" ? 7 - x : x;
  const rank = bottom === "black" ? y : 7 - y;
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return FILES[file] + (rank + 1);
}
function squareUnder(clientX, clientY) {
  const r = el("board").getBoundingClientRect();
  const x = Math.floor(((clientX - r.left) / r.width) * 8);
  const y = Math.floor(((clientY - r.top) / r.height) * 8);
  return squareAt(x, y);
}

// --- the squares ----------------------------------------------------------------------------

let squares = null; // square name -> element
function buildSquares() {
  const board = el("board");
  const frag = document.createDocumentFragment();
  squares = new Map();
  for (let y = 0; y < 8; y++) {
    for (let x = 0; x < 8; x++) {
      const name = squareAt(x, y);
      const file = FILES.indexOf(name[0]), rank = +name[1] - 1;
      const sq = document.createElement("div");
      sq.className = "sq " + (((rank + file) % 2 === 0) ? "dark" : "light");
      if (y % 2 === 1) sq.classList.add("odd");
      sq.dataset.square = name;
      if (x === 0) { const s = document.createElement("span"); s.className = "coord rank"; s.textContent = rank + 1; sq.appendChild(s); }
      if (y === 7) { const s = document.createElement("span"); s.className = "coord file"; s.textContent = FILES[file]; sq.appendChild(s); }
      frag.appendChild(sq);
      squares.set(name, sq);
    }
  }
  board.replaceChildren(frag);
}

// --- the pieces: a layer of their own, moved by transforms so they slide ------------------

let pieces = new Map(); // square -> element
function place(node, square) {
  const { x, y } = cellOf(square);
  node.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
}
function renderPieces(fen) {
  const want = placement(fen);
  const layer = el("pieces");
  const stay = new Map();
  const gone = [];   // [square, letter, node] no longer where they were
  for (const [sq, node] of pieces) {
    if (want.get(sq) === node.dataset.letter) stay.set(sq, node);
    else gone.push([sq, node.dataset.letter, node]);
  }
  const next = new Map(stay);
  for (const [sq, letter] of want) {
    if (stay.has(sq)) continue;
    // A piece of this kind that left a square slides here (a move, a castling rook);
    // otherwise it appears.
    const i = gone.findIndex(([, l]) => l === letter);
    if (i >= 0) {
      const [[, , node]] = gone.splice(i, 1);
      place(node, sq);
      next.set(sq, node);
    } else {
      const node = document.createElement("div");
      node.className = "piece " + spriteOf(letter);
      node.dataset.letter = letter;
      node.style.transition = "none";
      place(node, sq);
      layer.appendChild(node);
      node.getBoundingClientRect(); // settle before the transition returns
      node.style.transition = "";
      next.set(sq, node);
    }
  }
  for (const [, , node] of gone) {  // taken, or a promoted pawn: fades out
    node.classList.add("fading");
    setTimeout(() => node.remove(), 140);
  }
  pieces = next;
}

// --- marks: the last move, the check, the selection, the targets, the user's ---------------

let selected = null, targets = [];
let marked = new Set();          // right-clicked squares
let arrows = [];                 // [[from, to], ...] the user's arrows
let coachMarks = null;           // the coach's annotations for the current position
let wrongShown = null;           // the wrong move being shown (uci)

function kingInCheck(fen) {
  // The FEN says whose turn; the server tells us the position is check through
  // state.check_square when it is; we keep it simple: state.check (a square or null).
  return state && state.check ? state.check : null;
}

function renderMarks() {
  for (const [name, sq] of squares) {
    sq.classList.remove("hl", "sel", "target", "occupied", "marked", "check", "wrong", "grab");
    if (state && state.your_move && !wrongShown) {
      const p = pieces.get(name);
      if (p && colourOf(p.dataset.letter) === bottom) sq.classList.add("grab");
    }
  }
  if (state && state.last_move && !wrongShown) {
    for (const s of [state.last_move.slice(0, 2), state.last_move.slice(2, 4)]) squares.get(s)?.classList.add("hl");
  }
  if (wrongShown) {
    squares.get(wrongShown.slice(0, 2))?.classList.add("hl");
    squares.get(wrongShown.slice(2, 4))?.classList.add("wrong");
  }
  const check = kingInCheck();
  if (check) squares.get(check)?.classList.add("check");
  if (selected) squares.get(selected)?.classList.add("sel");
  for (const t of targets) {
    const sq = squares.get(t);
    if (!sq) continue;
    sq.classList.add("target");
    if (pieces.has(t)) sq.classList.add("occupied");
  }
  for (const m of marked) squares.get(m)?.classList.add("marked");
  renderOverlay();
}

function svgEl(name, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}
function arrow(frag, from, to, cls, opacity, wide) {
  const a = cellOf(from), b = cellOf(to);
  const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
  const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
  const k = 0.32 / len;
  frag.appendChild(svgEl("line", { x1, y1, x2: x2 - dx * k, y2: y2 - dy * k, class: cls + (wide ? " user" : ""), opacity, "marker-end": `url(#ah-${cls.replace("mk-", "")})` }));
}
function renderOverlay() {
  const svg = el("overlay");
  const frag = document.createDocumentFragment();
  const defs = svgEl("defs", {});
  const colours = new Set(["user", ...(coachMarks || []).filter((m) => m.kind === "arrow").map((m) => m.color)]);
  for (const c of colours) {
    const marker = svgEl("marker", { id: "ah-" + c, class: "mk-" + c, viewBox: "0 0 10 10", refX: "7", refY: "5", markerWidth: "3.2", markerHeight: "3.2", orient: "auto-start-reverse" });
    marker.appendChild(svgEl("path", { d: "M0,0 L10,5 L0,10 z" }));
    defs.appendChild(marker);
  }
  frag.appendChild(defs);
  for (const m of coachMarks || []) {
    if (m.kind === "square") {
      const { x, y } = cellOf(m.squares[0]);
      frag.appendChild(svgEl("rect", { x, y, width: 1, height: 1, class: "mk-" + m.color, opacity: (m.opacity == null ? 1 : m.opacity) * 0.55 }));
    } else if (m.kind === "ring") {
      const { x, y } = cellOf(m.squares[0]);
      frag.appendChild(svgEl("circle", { cx: x + 0.5, cy: y + 0.5, r: 0.42, class: "mk-" + m.color, opacity: m.opacity == null ? 1 : m.opacity }));
    } else if (m.kind === "arrow") {
      arrow(frag, m.squares[0], m.squares[1], "mk-" + m.color, m.opacity == null ? 1 : m.opacity, false);
    }
  }
  for (const [from, to] of arrows) arrow(frag, from, to, "mk-user", 0.85, true);
  svg.replaceChildren(frag);
}

// --- what the player does ---------------------------------------------------------------------

let drag = null;   // {square, node, moved}
let rightDrag = null; // {from}
let selecting = null; // the targets on their way for the piece just picked up

function select(square) {
  const node = pieces.get(square);
  if (!state || !state.your_move || wrongShown || !node || colourOf(node.dataset.letter) !== bottom) {
    selected = null; targets = []; renderMarks(); return Promise.resolve();
  }
  selected = square;
  targets = [];
  renderMarks();
  selecting = api.targets(square).then((res) => {
    if (selected !== square) return;
    targets = (res && res.targets) || [];
    renderMarks();
  }).finally(() => { selecting = null; });
  return selecting;
}

function isPromotion(from, to) {
  const node = pieces.get(from);
  if (!node || node.dataset.letter.toUpperCase() !== "P") return false;
  return to[1] === "8" || to[1] === "1";
}

async function tryMove(from, to) {
  if (!targets.includes(to)) return false;
  selected = null; targets = [];
  if (isPromotion(from, to)) {
    const piece = await promotionPicker(to);
    renderMarks();
    if (!piece) return false;
    await api.move(from + to + piece);
    return true;
  }
  // Slide the piece at once; the poll confirms (or shows the wrong move and takes it back).
  const node = pieces.get(from);
  if (node) place(node, to);
  renderMarks();
  await api.move(from + to);
  return true;
}

function promotionPicker(square) {
  return new Promise((resolve) => {
    const box = el("promo");
    const { x, y } = cellOf(square);
    const down = y > 3;  // the column grows away from the edge the pawn reached
    box.className = "promo" + (down ? " down" : "");
    box.style.left = x * 12.5 + "%";
    box.style.top = (down ? (y - 3) * 12.5 : y * 12.5) + "%";
    box.innerHTML = "";
    const colour = bottom === "white" ? "w" : "b";
    for (const p of ["Q", "R", "B", "N"]) {
      const b = document.createElement("button");
      b.className = colour + p; b.title = p;
      b.onclick = (e) => { e.stopPropagation(); box.hidden = true; resolve(p.toLowerCase()); };
      box.appendChild(b);
    }
    const x_ = document.createElement("button"); x_.className = "x"; x_.textContent = "×";
    x_.onclick = (e) => { e.stopPropagation(); box.hidden = true; resolve(null); };
    box.appendChild(x_);
    box.hidden = false;
  });
}

function onPointerDown(e) {
  const wrap = el("boardwrap");
  if (e.button === 2) {  // a right-click: a mark, or an arrow if dragged
    e.preventDefault();
    rightDrag = { from: squareUnder(e.clientX, e.clientY) };
    wrap.setPointerCapture(e.pointerId);
    return;
  }
  if (e.button !== 0) return;
  // A left-click clears the user's marks, as on chess.com.
  if (marked.size || arrows.length) { marked.clear(); arrows = []; renderMarks(); }
  const square = squareUnder(e.clientX, e.clientY);
  if (!square) return;
  if (selected && targets.includes(square)) { tryMove(selected, square); return; }
  const node = pieces.get(square);
  if (node && state && state.your_move && !wrongShown && colourOf(node.dataset.letter) === bottom) {
    drag = { square, node, moved: false, startX: e.clientX, startY: e.clientY };
    wrap.setPointerCapture(e.pointerId);
    select(square);
  } else if (selected) {
    selected = null; targets = []; renderMarks();
  }
}

function onPointerMove(e) {
  if (drag) {
    const dx = e.clientX - drag.startX, dy = e.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    if (!drag.moved) { drag.moved = true; drag.node.classList.add("dragging"); document.body.classList.add("grabbing"); }
    const r = el("board").getBoundingClientRect();
    const cell = r.width / 8;
    const px = e.clientX - r.left - cell / 2, py = e.clientY - r.top - cell / 2;
    drag.node.style.transform = `translate(${(px / cell) * 100}%, ${(py / cell) * 100}%)`;
  }
}

async function onPointerUp(e) {
  const wrap = el("boardwrap");
  if (rightDrag) {
    const to = squareUnder(e.clientX, e.clientY);
    const from = rightDrag.from;
    rightDrag = null;
    if (from && to && from !== to) {
      const i = arrows.findIndex(([a, b]) => a === from && b === to);
      if (i >= 0) arrows.splice(i, 1); else arrows.push([from, to]);
    } else if (from) {
      if (marked.has(from)) marked.delete(from); else marked.add(from);
    }
    renderMarks();
    return;
  }
  if (!drag) return;
  const d = drag; drag = null;
  document.body.classList.remove("grabbing");
  d.node.classList.remove("dragging");
  if (!d.moved) { place(d.node, d.square); return; }  // a click: stays selected
  if (selecting) await selecting;  // a quick drag: the legal squares may still be on their way
  const to = squareUnder(e.clientX, e.clientY);
  if (to && to !== d.square && targets.includes(to)) {
    await tryMove(d.square, to);
  } else {
    place(d.node, d.square);  // back where it was, still selected
    renderMarks();
  }
}

// --- the coach ---------------------------------------------------------------------------------

let pending = null;
function addMessage(line) {
  const c = el("coach");
  if (pending) { pending.remove(); pending = null; }
  const d = line.data || {};
  const msg = document.createElement("div");
  msg.className = "msg" + (line.chan === "user" ? " you" : "");
  const meta = document.createElement("div"); meta.className = "meta";
  meta.textContent = line.chan === "user" ? "you" : (d.trigger || "coach").replace("_", " ");
  const body = document.createElement("div");
  body.textContent = line.chan === "user" ? line.text.replace(/^you: /, "") : (d.full || line.text);
  msg.append(meta, body);
  if (line.chan === "coach" && d.annotations && d.annotations.length) {
    coachMarks = d.annotations;
    msg.onmouseenter = () => { coachMarks = d.annotations; renderOverlay(); };
    renderOverlay();
  }
  c.appendChild(msg);
  c.scrollTop = c.scrollHeight;
}
function showPending() {
  const c = el("coach");
  if (pending) return;
  pending = document.createElement("div");
  pending.className = "msg pending";
  pending.innerHTML = '<div class="meta">coach</div><div>thinking<span class="dots"></span></div>';
  c.appendChild(pending);
  c.scrollTop = c.scrollHeight;
}
function ask(text) {
  if (!text) return;
  showPending();
  api.ask(text);
}

// --- the log ----------------------------------------------------------------------------------

const stamp = (t) => { const m = Math.floor(t / 60), s = (t % 60).toFixed(1).padStart(4, "0"); return `${String(m).padStart(2, "0")}:${s}`; };
function addLine(line) {
  const term = el("terminal");
  const row = document.createElement("div");
  row.className = `row chan-${line.chan} lvl-${line.level}`;
  const t = document.createElement("span"); t.className = "t"; t.textContent = stamp(line.t || 0);
  const c = document.createElement("span"); c.className = "c"; c.textContent = line.chan;
  const m = document.createElement("span"); m.className = "m"; m.textContent = line.text;
  row.append(t, c, m);
  term.appendChild(row);
  if (!term.hidden) term.scrollTop = term.scrollHeight;
}

// --- the state --------------------------------------------------------------------------------

const motifWords = (m) => (m || "").replace(/([a-z])([A-Z0-9])/g, "$1 $2").toLowerCase();
const kilo = (n) => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n);
let retracting = false;

function render(s) {
  const first = state === null;
  state = s;
  if (first) {
    bottom = s.bottom;
    buildSquares();
    el("pz-title").textContent = `puzzle ${s.id} · ${s.motif ? motifWords(s.motif) + " · " : ""}${s.rating} · you play ${s.bottom}`;
    el("lichess").href = s.url;
    document.title = `sidecoach · puzzle ${s.id}`;
  }
  // A wrong move: shown where it landed, red, then taken back by the server.
  if (s.wrong && !wrongShown) {
    wrongShown = s.wrong;
    const node = pieces.get(s.wrong.slice(0, 2));
    if (node) place(node, s.wrong.slice(2, 4));
    selected = null; targets = [];
    if (!retracting) { retracting = true; setTimeout(() => { retracting = false; api.retract(); }, 800); }
  } else if (!s.wrong && wrongShown) {
    wrongShown = null;
  }
  if (!wrongShown) renderPieces(s.fen);
  const msg = el("pz-msg");
  msg.textContent = s.message || "";
  msg.className = "pz-msg " + (s.status === "solved" ? "good" : s.wrong ? "bad" : "");
  el("side").textContent = `you play ${s.bottom}`;
  el("remaining").textContent = s.status === "playing" ? `${s.remaining} move${s.remaining === 1 ? "" : "s"} to find` : s.status.replace("_", " ");
  el("h-misses").textContent = String(s.misses || 0);
  el("giveup").disabled = s.status !== "playing";
  el("voice").setAttribute("aria-pressed", s.voice ? "true" : "false");
  if (s.llm && s.llm.calls) {
    el("h-tokens").textContent = `${kilo(s.llm.prompt)} in · ${kilo(s.llm.completion)} out · ${s.llm.calls} calls`;
    el("h-cost").textContent = s.llm.cost == null ? "–" : "$" + (s.llm.cost < 0.01 ? s.llm.cost.toFixed(4) : s.llm.cost < 0.1 ? s.llm.cost.toFixed(3) : s.llm.cost.toFixed(2));
  }
  if (s.closed) { msg.textContent = "closed"; msg.className = "pz-msg"; }
  renderMarks();
}

async function tick() {
  try {
    const res = await api.poll(cursor);
    if (res.closed && !res.state) { el("pz-msg").textContent = "closed"; return; }
    if (res.cursor !== undefined) cursor = Math.max(cursor, res.cursor);
    for (const line of res.lines || []) {
      addLine(line);
      if (line.chan === "coach" && line.level === "info") addMessage(line);
      if (line.chan === "user") addMessage(line);
    }
    if (res.state) render(res.state);
    if (res.state && res.state.closed) return;
  } catch (e) { /* a poll missed */ }
  setTimeout(tick, 250);
}

function wire() {
  const wrap = el("boardwrap");
  wrap.addEventListener("pointerdown", onPointerDown);
  wrap.addEventListener("pointermove", onPointerMove);
  wrap.addEventListener("pointerup", onPointerUp);
  wrap.addEventListener("contextmenu", (e) => e.preventDefault());
  wrap.addEventListener("pointercancel", () => { if (drag) { place(drag.node, drag.square); drag.node.classList.remove("dragging"); drag = null; } rightDrag = null; document.body.classList.remove("grabbing"); });
  el("giveup").onclick = () => api.giveup();
  el("close").onclick = async () => { await api.close(); window.close(); el("pz-msg").textContent = "closed — you can close this tab"; };
  el("next").onclick = () => { if (nextIds.length) location.href = `/puzzle/${nextIds[0]}`; };
  el("voice").onclick = () => { const on = el("voice").getAttribute("aria-pressed") !== "true"; el("voice").setAttribute("aria-pressed", on ? "true" : "false"); api.voice(on); };
  for (const b of document.querySelectorAll(".chips-row .qchip")) b.onclick = () => ask(b.dataset.ask);
  const input = el("askInput");
  const sendIt = () => { const t = input.value.trim(); if (t) { ask(t); input.value = ""; } };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendIt(); });
  el("send").onclick = sendIt;
  el("fold").onclick = () => { const t = el("terminal"); t.hidden = !t.hidden; el("fold").textContent = t.hidden ? "show log" : "hide log"; if (!t.hidden) t.scrollTop = t.scrollHeight; };
  window.addEventListener("beforeunload", () => { navigator.sendBeacon && navigator.sendBeacon(`/api/puzzle/${PUZZLE_ID}/close`, new Blob(["{}"], { type: "application/json" })); });
}

function fitBoard() {
  const root = document.documentElement;
  const mainEl = document.querySelector("main");
  const cs = getComputedStyle(mainEl);
  const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
  const byHeight = mainEl.clientHeight - padY - 30;
  const byWidth = mainEl.clientWidth - padX - 14 - 340;
  root.style.setProperty("--board", Math.max(240, Math.floor(Math.min(byHeight, byWidth))) + "px");
}

async function main() {
  wire();
  fitBoard();
  window.addEventListener("resize", fitBoard);
  const opened = await api.open();
  if (!opened.ok) { el("pz-msg").textContent = opened.error || "no such puzzle"; return; }
  nextIds = opened.next || [];
  el("next").hidden = !nextIds.length;
  render(opened.state);
  tick();
}

main();
