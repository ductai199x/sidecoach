"use strict";

const FILES = "abcdefgh";
const el = (id) => document.getElementById(id);
let cursor = 0;
let running = false;
let lastBoardKey = null;
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
  set_listen: (mode) => post("/api/listen", { mode }),
  end: (outcome) => post("/api/end", { outcome }),
  exclude: (rect) => post("/api/exclude", rect),
  ptt: (down) => post("/api/ptt", { down }),
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
      if (r % 2 === 1) sq.classList.add("odd");  // the panel's own-board signature
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

// --- the move list ------------------------------------------------------------
//
// Each move carries the engine's label as a badge (the ones chess.com marks: brilliant,
// great, best, inaccuracy, mistake, miss, blunder; good and book stay plain). Hovering a
// move shows its position on the board; clicking pins it there, for as long as you like,
// and clicking it again (or Escape) comes back to the live board.

const BADGED = new Set(["brilliant", "great", "best", "inaccuracy", "mistake", "miss", "blunder"]);
let moveFens = [];    // fen after each ply
let moveUcis = [];    // uci of each ply
let pinnedPly = null; // a move the user clicked
let lastMovesKey = "";

function showPly(ply) {
  // Show the position after `ply` (1-based) with that move highlighted.
  ghost = true;
  viewPly = ply;
  const fen = ply > 0 ? moveFens[ply - 1] : null;
  if (fen) renderBoard(fen, currentBottom, moveUcis[ply - 1] || null);
  for (const mv of document.querySelectorAll("#moves .mv.view")) mv.classList.remove("view");
  const cell = document.querySelector("#moves .mv[data-ply='" + ply + "']");
  if (cell) cell.classList.add("view");
  renderOverlay();
}

function showLive() {
  ghost = false;
  viewPly = null;
  for (const mv of document.querySelectorAll("#moves .mv.view")) mv.classList.remove("view");
  renderBoard(liveFen, currentBottom, liveLastMove);
  renderOverlay();
}

function renderMoves(sans, labels) {
  const key = sans.length + "|" + JSON.stringify(labels || {});
  if (key === lastMovesKey) return;
  lastMovesKey = key;
  const list = el("moves");
  const frag = document.createDocumentFragment();
  const cell = (san, ply) => {
    const m = document.createElement("span"); m.className = "mv";
    if (!san) return m;
    m.dataset.ply = String(ply);
    m.appendChild(document.createTextNode(san));
    const label = labels && labels[String(ply)];
    if (label && BADGED.has(label)) {
      const b = document.createElement("i"); b.className = "badge " + label; b.title = label; m.appendChild(b);
    }
    if (ply === sans.length) m.classList.add("cur");
    if (ply === pinnedPly) m.classList.add("view");
    m.onmouseenter = () => { if (pinnedPly === null) showPly(ply); };
    m.onmouseleave = () => { if (pinnedPly === null) showLive(); };
    m.onclick = () => {
      if (pinnedPly === ply) { pinnedPly = null; showLive(); }
      else { pinnedPly = ply; showPly(ply); }
    };
    return m;
  };
  for (let i = 0; i < sans.length; i += 2) {
    const row = document.createElement("div"); row.className = "mrow";
    const num = document.createElement("span"); num.className = "num"; num.textContent = (i / 2 + 1) + ".";
    row.append(num, cell(sans[i], i + 1), cell(sans[i + 1], i + 2));
    frag.appendChild(row);
  }
  list.replaceChildren(frag);
  if (pinnedPly !== null && pinnedPly > sans.length) { pinnedPly = null; showLive(); }
  if (pinnedPly === null) { const wrap = list.parentElement; wrap.scrollTop = wrap.scrollHeight; }
}

// --- the debug terminal -------------------------------------------------------
//
// A record is {seq, t, chan, level, text, data}. The line shows `mm:ss.s  CHAN  text`; a
// click expands `data`. Identical consecutive lines collapse into one with a ×N badge.
// Filters (channel chips, verbose, search) only hide rows, so the whole session stays in
// the DOM and can be re-examined without polling again.

const filters = { chans: new Set(["sys", "vision", "game", "engine", "coach", "voice", "user"]), verbose: false, query: "" };
let lastRow = null; // {key, el, count}

function stamp(t) {
  const m = Math.floor(t / 60), s = (t - m * 60).toFixed(1).padStart(4, "0");
  return String(m).padStart(2, "0") + ":" + s;
}

function visible(row) {
  if (!filters.chans.has(row.dataset.chan)) return false;
  if (row.dataset.level === "debug" && !filters.verbose) return false;
  if (filters.query && !row.dataset.text.includes(filters.query)) return false;
  return true;
}

function applyFilters() {
  for (const row of el("terminal").children) row.hidden = !visible(row);
}

function addLine(line) {
  const term = el("terminal");
  const key = line.chan + "|" + line.level + "|" + line.text;
  if (lastRow && lastRow.key === key) {
    lastRow.count++;
    lastRow.badge.textContent = "×" + lastRow.count;
    return;
  }
  const row = document.createElement("div");
  const hasData = line.data && Object.keys(line.data).length > 0;
  row.className = "row chan-" + line.chan + " lvl-" + line.level + (hasData ? " has-data" : "");
  if (line.data && line.data.severity) row.classList.add("sev-" + line.data.severity);
  row.dataset.chan = line.chan; row.dataset.level = line.level;
  row.dataset.text = (line.chan + " " + line.text).toLowerCase();

  const t = document.createElement("span"); t.className = "t"; t.textContent = stamp(line.t || 0);
  const c = document.createElement("span"); c.className = "c"; c.textContent = line.chan;
  const m = document.createElement("span"); m.className = "m";
  if (hasData) { const caret = document.createElement("span"); caret.className = "caret"; caret.textContent = "▸"; m.appendChild(caret); }
  m.appendChild(document.createTextNode(line.text));
  const badge = document.createElement("span"); badge.className = "n"; m.appendChild(badge);
  row.append(t, c, m);
  if (hasData) {
    const d = document.createElement("pre"); d.className = "d"; d.hidden = true;
    d.textContent = JSON.stringify(line.data, null, 1).replace(/\n\s*/g, (x) => x.length > 1 ? "\n " : x);
    row.appendChild(d);
    row.onclick = () => { d.hidden = !d.hidden; m.firstChild.textContent = d.hidden ? "▸" : "▾"; };
  }
  row.hidden = !visible(row);
  term.appendChild(row);
  lastRow = { key, el: row, count: 1, badge };
  if (el("autoscroll").checked) term.scrollTop = term.scrollHeight;

  if (line.chan === "coach" && line.data && line.data.full) addCoach(line);
}

// --- board marks: the arrows and highlights that go with a coach message --------------
//
// Each coach message carries marks built from the same disclosure-filtered facts it was
// written from. The latest message's marks show while the board is still at (or one ply
// past) the position they describe; hovering any message shows its marks regardless, and
// hovering a square or move the coach named lights that square up.

let currentBottom = "white";
let currentPly = 0;
let liveFen = null, liveLastMove = null; // the board on screen, to come back to after a hover
let latestMarks = null; // {ply, marks}
let hoverMarks = null;  // marks while a message is hovered
let hoverSquare = null; // a square while a token is hovered
let ghost = false;      // the board is showing a message's own position, not the live one
let viewPly = null;     // the ply the board shows while browsing the move list
const MARK_PLY_TOLERANCE = 1;
const SQ_TOKEN = /(?<![A-Za-z0-9])((?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?|[a-h][1-8])(?![A-Za-z0-9])/g;

function cellOf(square) {
  const file = FILES.indexOf(square[0]), rank = +square[1] - 1;
  const col = currentBottom === "black" ? 7 - file : file;
  const row = currentBottom === "black" ? rank : 7 - rank;
  return { x: col, y: row };
}

function svgEl(name, attrs) {
  const n = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function renderOverlay() {
  const svg = el("overlay");
  let marks = hoverMarks;
  const plyShown = viewPly == null ? currentPly : viewPly;
  if (!marks && latestMarks && (latestMarks.ply == null || (plyShown >= latestMarks.ply && plyShown - latestMarks.ply <= MARK_PLY_TOLERANCE))) marks = latestMarks.marks;
  const list = [...(marks || [])];
  if (hoverSquare) list.push({ kind: "square", squares: [hoverSquare], color: "mention", opacity: 0.9 });
  const frag = document.createDocumentFragment();
  const defs = svgEl("defs", {});
  const colors = new Set(list.filter((m) => m.kind === "arrow").map((m) => m.color));
  for (const c of colors) {
    const marker = svgEl("marker", { id: "ah-" + c, class: "mk-" + c, viewBox: "0 0 10 10", refX: "7", refY: "5", markerWidth: "3.2", markerHeight: "3.2", orient: "auto-start-reverse" });
    marker.appendChild(svgEl("path", { d: "M0,0 L10,5 L0,10 z" }));
    defs.appendChild(marker);
  }
  frag.appendChild(defs);
  // squares and rings first, arrows on top
  for (const m of list.filter((m) => m.kind !== "arrow")) {
    const { x, y } = cellOf(m.squares[0]);
    const op = m.opacity == null ? 1 : m.opacity;
    if (m.kind === "square") frag.appendChild(svgEl("rect", { x, y, width: 1, height: 1, class: "mk-" + m.color, opacity: op * 0.55 }));
    else if (m.kind === "ring") frag.appendChild(svgEl("circle", { cx: x + 0.5, cy: y + 0.5, r: 0.42, class: "mk-" + m.color, opacity: op }));
  }
  for (const m of list.filter((m) => m.kind === "arrow")) {
    const a = cellOf(m.squares[0]), b = cellOf(m.squares[1]);
    const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const k = 0.32 / len; // stop short of the centre so the head lands on it
    frag.appendChild(svgEl("line", { x1, y1, x2: x2 - dx * k, y2: y2 - dy * k, class: "mk-" + m.color, opacity: m.opacity == null ? 1 : m.opacity, "marker-end": "url(#ah-" + m.color + ")" }));
  }
  svg.replaceChildren(frag);
}

function linkedText(text, container) {
  // Wrap the squares and moves the coach named so hovering one lights its square.
  let last = 0;
  for (const m of text.matchAll(SQ_TOKEN)) {
    container.appendChild(document.createTextNode(text.slice(last, m.index)));
    const tok = document.createElement("span"); tok.className = "tok"; tok.textContent = m[0];
    const sq = m[0].match(/[a-h][1-8]/g); // the destination square of a move, or the square
    if (sq) {
      const dest = sq[sq.length - 1];
      tok.onmouseenter = () => { hoverSquare = dest; renderOverlay(); };
      tok.onmouseleave = () => { hoverSquare = null; renderOverlay(); };
    }
    container.appendChild(tok);
    last = m.index + m[0].length;
  }
  container.appendChild(document.createTextNode(text.slice(last)));
}

let pendingMsg = null; // the "thinking" placeholder after an ask, until the reply lands

function showPending() {
  if (pendingMsg) return;
  const c = el("coach");
  const msg = document.createElement("div"); msg.className = "msg pending";
  const meta = document.createElement("div"); meta.className = "meta"; meta.textContent = "coach";
  const body = document.createElement("div"); body.className = "dots";
  msg.append(meta, body);
  c.appendChild(msg);
  c.scrollTop = c.scrollHeight;
  pendingMsg = msg;
}

function addCoach(line) {
  const c = el("coach");
  const d = line.data;
  if (pendingMsg) { pendingMsg.remove(); pendingMsg = null; }
  const marks = d.annotations || [];
  const msg = document.createElement("div"); msg.className = "msg" + (marks.length ? " linked" : "");
  const meta = document.createElement("div"); meta.className = "meta";
  meta.textContent = d.trigger.replace("_", " ") + " · level " + d.disclosure + (d.ply != null ? " · ply " + d.ply : "") + (d.fallback ? " · template" : "");
  if (marks.length) { const s = document.createElement("span"); s.className = "marks"; s.textContent = " · " + marks.length + " mark" + (marks.length > 1 ? "s" : ""); meta.appendChild(s); }
  const body = document.createElement("div");
  linkedText(d.full, body);
  msg.append(meta, body);
  if (marks.length) {
    // Hovering shows the position the message was about, with its marks, so an arrow
    // from an earlier moment is never drawn over pieces that have since moved.
    msg.onmouseenter = () => {
      hoverMarks = marks;
      if (d.fen && d.fen !== liveFen) { ghost = true; renderBoard(d.fen, currentBottom, null); }
      renderOverlay();
    };
    msg.onmouseleave = () => {
      hoverMarks = null;
      if (pinnedPly !== null) showPly(pinnedPly);
      else if (ghost) { ghost = false; renderBoard(liveFen, currentBottom, liveLastMove); }
      renderOverlay();
    };
  }
  c.appendChild(msg);
  c.scrollTop = c.scrollHeight;
  coachCount++;
  latestMarks = { ply: d.ply, marks };
  renderOverlay();
}

// --- the status strip ----------------------------------------------------------

function clockText(clock) {
  if (!clock) return "–";
  const fmt = (ms) => { if (ms == null) return "–"; const s = Math.round(ms / 1000); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
  const run = clock.running ? " (" + clock.running + " running)" : "";
  return "W " + fmt(clock.white_ms) + "  B " + fmt(clock.black_ms) + run;
}

function renderHealth(h, s) {
  if (!h) return;
  el("h-fps").textContent = h.fps != null ? h.fps.toFixed(1) : "–";
  const lock = el("h-lock"); lock.textContent = h.lock || "–";
  lock.className = "hv" + (h.lock === "no board" ? " bad" : h.lock === "lock in doubt" ? " warn" : "");
  el("h-stage").textContent = (h.stage || "–") + (s.ply ? ", ply " + s.ply : "");
  el("h-pipe").textContent = h.pipeline_ms ? h.pipeline_ms[0] + "/" + h.pipeline_ms[1] + " ms" : "–";
  const lag = el("h-lag"); lag.textContent = h.lag_ms != null ? h.lag_ms + " ms" : "–";
  lag.className = "hv" + (h.lag_ms > 1500 ? " bad" : h.lag_ms > 500 ? " warn" : "");
  el("h-clock").textContent = clockText(s.clock);
  const q = el("h-coach"); q.textContent = h.coach_pending != null ? String(h.coach_pending) : "–";
  q.className = "hv" + (h.coach_pending > 1 ? " warn" : "");
  el("h-rec").textContent = h.recording || "–";
  el("h-voice").textContent = h.tts_device || "–";
  el("h-ear").textContent = h.stt_device || "–";
  el("h-rss").textContent = h.rss_mb != null ? h.rss_mb + " MB" : "–";
}

// --- talking to the coach ----------------------------------------------------------------
//
// Hold the button (or Space, outside a text box) to talk in push-to-talk mode; in voice mode
// the server's VAD decides and the button is just an indicator. What was heard, corrected
// against the board, shows under the buttons and the coach answers it like a typed ask.

let pttDown = false;
let listenMode = "off";
const MODE_TITLES = {
  off: "Listening: off. Click for hold-to-talk",
  ptt: "Listening: hold to talk (hold the round button, or Space). Click for always-on voice",
  vad: "Listening: voice (the coach hears you whenever you speak). Click to turn off",
};
const MODE_NEXT = { off: "ptt", ptt: "vad", vad: "off" };

function setPtt(down) {
  if (listenMode !== "ptt" || pttDown === down) return;
  pttDown = down;
  renderSend();
  api.ptt(down);
}

// The round button is Send while there is text; empty, it is the hold-to-talk button in
// push-to-talk mode, the listening indicator in voice mode, and just a mic otherwise.
function renderSend() {
  const btn = el("send");
  const hasText = el("askInput").value.trim().length > 0;
  const state = hasText ? "send" : listenMode === "ptt" ? "ptt" : listenMode === "vad" ? "vad" : "idle";
  btn.dataset.state = state;
  btn.classList.toggle("live", state === "ptt" && pttDown);
  btn.title = state === "send" ? "Send (Enter)" : state === "ptt" ? "Hold to talk (or hold Space)" : state === "vad" ? "Listening" : "Type a question, or turn listening on";
}

function renderListen(s) {
  if (s.listen !== undefined && s.listen !== listenMode) {
    listenMode = s.listen;
    const mode = el("listenMode");
    mode.dataset.mode = listenMode;
    mode.title = MODE_TITLES[listenMode] || "";
    if (listenMode !== "ptt") pttDown = false;
    renderSend();
  }
  const h = el("heard");
  if (s.heard) { h.hidden = false; h.innerHTML = ""; h.appendChild(document.createTextNode("heard: ")); const b = document.createElement("b"); b.textContent = s.heard; h.appendChild(b); }
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
  liveFen = s.fen; liveLastMove = s.last_move;
  moveFens = s.fens || []; moveUcis = s.ucis || [];
  if (!ghost) renderBoard(s.fen, shown, s.last_move);
  renderMoves(s.moves || [], s.labels || {});
  renderHealth(s.health, s);
  renderListen(s);
  setRunning(s.running);
  const ply = s.ply || 0;
  if (shown !== currentBottom || ply !== currentPly) {
    currentBottom = shown; currentPly = ply;
    renderOverlay();
  }
}

function setRunning(on) {
  running = on;
  const b = el("startStop");
  b.textContent = on ? "Stop" : "Start";
  b.classList.toggle("running", on);
}

// --- keeping the vision off our own board ----------------------------------------------------
//
// If the game shares this monitor, the pipeline could lock onto the panel's board instead
// of the game's. The page knows where its board is on screen, so it tells the engine, which
// masks that rectangle before a frame is read. Screen coordinates are approximate (window
// chrome, DPI); the engine adds a margin.

let lastRect = "";
function reportBoardRect() {
  const r = el("board").getBoundingClientRect();
  if (!r.width) return;
  const dpr = window.devicePixelRatio || 1;
  const chromeH = Math.max(0, window.outerHeight - window.innerHeight);
  const rect = {
    x: Math.round((window.screenX + r.left) * dpr),
    y: Math.round((window.screenY + chromeH + r.top) * dpr),
    w: Math.round(r.width * dpr),
    h: Math.round(r.height * dpr),
  };
  const key = JSON.stringify(rect);
  if (key !== lastRect) { lastRect = key; api.exclude(rect); }
}

// --- polling -----------------------------------------------------------------

// One poll at a time: the next is scheduled only once this one has been handled. On a
// fixed interval, a server that stalled for a moment (the end of the first live game) had
// several polls in flight with the same cursor, and every one of them brought back the
// same new lines: "game over" seven times over.
async function tick() {
  try {
    reportBoardRect();
    const res = await api.poll(cursor);
    if (res.cursor !== undefined) cursor = Math.max(cursor, res.cursor);
    if (res.lines) for (const line of res.lines) addLine(line);
    if (res.state) renderState(res.state);
  } catch (e) {
    /* window closing or api not ready; ignore this tick */
  }
  setTimeout(tick, 250);
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
  const input = el("askInput");
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendAsk(); });
  input.addEventListener("input", renderSend);
  for (const b of document.querySelectorAll(".chips-row .qchip")) {
    b.onclick = () => { showPending(); api[b.dataset.cmd](); };
  }
  el("listenMode").onclick = () => {
    // Optimistic: the next click must see this one, even before the engine confirms it.
    const next = MODE_NEXT[listenMode] || "ptt";
    renderListen({ listen: next });
    api.set_listen(next);
  };
  const send = el("send");
  send.onclick = () => { if (send.dataset.state === "send") sendAsk(); else if (send.dataset.state === "idle") input.focus(); };
  send.onmousedown = () => { if (send.dataset.state === "ptt") setPtt(true); };
  send.onmouseup = send.onmouseleave = () => setPtt(false);
  send.ontouchstart = (e) => { if (send.dataset.state === "ptt") { e.preventDefault(); setPtt(true); } };
  send.ontouchend = () => setPtt(false);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && pinnedPly !== null) { pinnedPly = null; showLive(); } });
  el("clearTerm").onclick = () => { el("terminal").replaceChildren(); lastRow = null; };
  for (const chip of document.querySelectorAll("#chans .chip")) {
    chip.onclick = () => {
      const chan = chip.dataset.chan;
      if (filters.chans.has(chan)) filters.chans.delete(chan); else filters.chans.add(chan);
      chip.classList.toggle("on", filters.chans.has(chan));
      applyFilters();
    };
  }
  el("verbose").onchange = (e) => { filters.verbose = e.target.checked; applyFilters(); };
  el("end").onchange = (e) => { if (e.target.value) { api.end(e.target.value); e.target.value = ""; } };
  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement && document.activeElement.tagName);
  document.addEventListener("keydown", (e) => { if (e.code === "Space" && !e.repeat && !typing()) { e.preventDefault(); setPtt(true); } });
  document.addEventListener("keyup", (e) => { if (e.code === "Space" && !typing()) { e.preventDefault(); setPtt(false); } });
  window.addEventListener("blur", () => setPtt(false));
  el("search").oninput = (e) => { filters.query = e.target.value.trim().toLowerCase(); applyFilters(); };
}

function sendAsk() {
  const input = el("askInput");
  const text = input.value.trim();
  if (!text) return;
  showPending();
  api.ask(text);
  input.value = "";
  renderSend();
}

async function main() {
  wire();
  const first = await api.poll(0);
  if (first.state) { fillVoices(first.state.voices || [], first.state.voice); renderState(first.state); }
  if (first.lines) for (const line of first.lines) addLine(line);
  if (first.cursor !== undefined) cursor = first.cursor;
  renderSend();
  setTimeout(tick, 250);
}

main();
