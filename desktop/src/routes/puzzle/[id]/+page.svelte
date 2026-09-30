<script lang="ts">
  // The puzzle page (/puzzle/<id>): a board with chess.com's manners, the coach beside
  // it, the room's log below. The room lives on the sidecar (ui/puzzle_room.py); this
  // page polls it and sends what the player does.
  //
  // The board: press a piece of yours and it lifts to the pointer, its square yellow and its
  // legal squares dotted (none if it cannot move, the king in check); drop it on one, or
  // let go and click one. A dropped piece lands where it is dropped; a clicked move slides;
  // the line's reply slides in a moment later, as chess.com plays it. A right-click marks a
  // square red; a right-drag draws an arrow; a left-click anywhere clears them. Captures
  // fade, a pawn reaching the last rank offers a picker over the square. A wrong move lands,
  // shows red, and comes back. The coach says nothing unasked: Hint, Best move, Line, or a
  // typed question.
  //
  // The room answers a move with the state it leaves, and every state carries a version:
  // an answer older than the one shown is dropped. (The page used to take every poll as it
  // came, and one that had left before a move put the piece back on its square until the
  // next brought it forward again, Tai, 2026-09-27.)
  import { onMount, tick } from "svelte";
  import { page } from "$app/state";
  import "$lib/styles/puzzle.css";
  import { beacon, connect, puzzle as puzzleApi, type LogLine, type Mark, type PuzzlePosition, type PuzzleState } from "$lib/api";
  import { FILES, cellOf, colourOf, motifWords, placement, squareAt } from "$lib/chess";
  import { PieceLayer } from "$lib/pieces";
  import { layoutMode, mobileBoard } from "$lib/layout";
  import { prefs } from "$lib/panel.svelte";
  import { captureSound, moveSound } from "$lib/sounds";
  import { stickToBottom } from "$lib/stick";

  const id = decodeURIComponent(page.params.id ?? "");
  const api = puzzleApi(id);

  // ---- the room's state ----------------------------------------------------------------------
  let room = $state<PuzzleState | null>(null);
  let bottom = $state("white");
  let nextIds = $state<string[]>([]);
  let message = $state("…");
  let messageClass = $state("");
  let cursor = 0;
  let closed = false;

  // ---- what is on the board ------------------------------------------------------------------
  let selected = $state<string | null>(null);
  let targets = $state<string[]>([]);
  let marked = $state<string[]>([]);
  let arrows = $state<[string, string][]>([]);
  let coachMarks = $state<Mark[] | null>(null);
  let wrongShown = $state<string | null>(null);
  let occupied = $state<Record<string, string>>({});  // square -> letter, for the marks
  let hover = $state<string | null>(null);  // the square under a dragged piece
  let retracting = false;

  // ---- which position is shown -----------------------------------------------------------
  // The positions the room played are shown one by one: the solver's own move at once (it is
  // on the board already), the line's reply REPLY_MS later, and a solution given up a move
  // every REPLY_MS.
  const REPLY_MS = 400;
  let shown = $state(0);
  let optimistic = $state<string | null>(null);  // the solver's move, shown before the answer
  let inFlight: number | null = null;  // the version the move was sent from
  let stepping: ReturnType<typeof setTimeout> | null = null;
  const position = $derived<PuzzlePosition | null>(room?.positions[shown] ?? null);
  const settled = $derived(!!room && shown === room.positions.length - 1);
  const canMove = $derived(!!room && room.your_move && settled && !wrongShown && !optimistic);

  let boardEl = $state<HTMLDivElement | null>(null);
  let piecesEl = $state<HTMLDivElement | null>(null);
  let wrapEl = $state<HTMLDivElement | null>(null);
  let mainEl = $state<HTMLElement | null>(null);

  // The squares in display order for the orientation, with the classes the marks give them.
  const squares = $derived.by(() => {
    const out: { name: string; dark: boolean; odd: boolean; rankLabel: string | null; fileLabel: string | null; cls: string }[] = [];
    const move = wrongShown ? null : (optimistic ?? position?.uci ?? null);
    const last = move ? [move.slice(0, 2), move.slice(2, 4)] : [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const name = squareAt(x, y, bottom)!;
        const file = FILES.indexOf(name[0]), rank = +name[1] - 1;
        const cls: string[] = [];
        if (canMove && occupied[name] && colourOf(occupied[name]) === bottom) cls.push("grab");
        if (last.includes(name)) cls.push("hl");
        if (wrongShown && name === wrongShown.slice(0, 2)) cls.push("hl");
        if (wrongShown && name === wrongShown.slice(2, 4)) cls.push("wrong");
        if (!optimistic && position?.check === name) cls.push("check");
        if (selected === name) cls.push("sel");
        if (targets.includes(name)) { cls.push("target"); if (occupied[name]) cls.push("occupied"); }
        if (marked.includes(name)) cls.push("marked");
        if (hover === name) cls.push("hover");
        out.push({ name, dark: (rank + file) % 2 === 0, odd: y % 2 === 1, rankLabel: x === 0 ? String(rank + 1) : null, fileLabel: y === 7 ? FILES[file] : null, cls: cls.join(" ") });
      }
    }
    return out;
  });

  const arrowColours = $derived([...new Set(["user", ...(coachMarks || []).filter((m) => m.kind === "arrow").map((m) => m.color)])]);
  function arrowLine(from: string, to: string) {
    const a = cellOf(from, bottom), b = cellOf(to, bottom);
    const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const k = 0.32 / len;
    return { x1, y1, x2: x2 - dx * k, y2: y2 - dy * k };
  }

  // ---- the pieces: a layer of their own, moved by transforms so they slide -----------------
  let layer: PieceLayer | null = null;
  const pieceAt = (square: string) => layer?.nodes.get(square);
  const place = (node: HTMLDivElement, square: string) => layer?.place(node, square);
  const placeNow = (node: HTMLDivElement, square: string) => layer?.placeNow(node, square);
  // The pieces as `fen` has them; `uci`, the move that made it, names the piece that moves.
  function renderPieces(fen: string, uci: string | null = null) {
    if (!piecesEl) return;
    layer ??= new PieceLayer(piecesEl, bottom);
    occupied = layer.render(fen, uci);
  }

  // ---- what the player does ------------------------------------------------------------------
  let drag: { square: string; node: HTMLDivElement; moved: boolean; startX: number; startY: number } | null = null;
  let rightDrag: { from: string | null } | null = null;
  let promo = $state<{ square: string; x: number; y: number; down: boolean; resolve: (p: string | null) => void } | null>(null);

  function squareUnder(clientX: number, clientY: number): string | null {
    if (!boardEl) return null;
    const r = boardEl.getBoundingClientRect();
    const x = Math.floor(((clientX - r.left) / r.width) * 8);
    const y = Math.floor(((clientY - r.top) / r.height) * 8);
    return squareAt(x, y, bottom);
  }

  function select(square: string) {
    const node = pieceAt(square);
    if (!room || !canMove || !node || colourOf(node.dataset.letter!) !== bottom) {
      selected = null; targets = []; return;
    }
    selected = square;
    targets = room.legal[square] ?? [];
  }

  function isPromotion(from: string, to: string): boolean {
    const node = pieceAt(from);
    if (!node || node.dataset.letter!.toUpperCase() !== "P") return false;
    return to[1] === "8" || to[1] === "1";
  }

  function promotionPicker(square: string): Promise<string | null> {
    return new Promise((resolve) => {
      const { x, y } = cellOf(square, bottom);
      promo = { square, x, y, down: y > 3, resolve: (p) => { promo = null; resolve(p); } };
    });
  }

  // The sound of a move made from `fen`: a capture when it lands on a piece, or a pawn steps
  // aside onto an empty square (en passant); a move otherwise.
  function sound(fen: string, uci: string) {
    const board = placement(fen);
    const mover = board.get(uci.slice(0, 2));
    const takes = board.has(uci.slice(2, 4)) || (mover?.toUpperCase() === "P" && uci[0] !== uci[2]);
    if (takes) captureSound(); else moveSound();
  }

  // The solver's move: on the board at once -- dropped where it was let go, or slid there
  // from a click -- and sent; the answer brings the reply, or shows it wrong.
  async function tryMove(from: string, to: string, dropped: boolean): Promise<boolean> {
    if (!targets.includes(to)) return false;
    selected = null; targets = [];
    const node = pieceAt(from);
    let uci = from + to;
    if (isPromotion(from, to)) {
      if (node) placeNow(node, to);
      const piece = await promotionPicker(to);
      if (!piece) { if (node) placeNow(node, from); return false; }
      uci += piece;
    } else if (node) {
      if (dropped) placeNow(node, to); else place(node, to);
    }
    if (room) sound(room.positions[shown].fen, uci);
    optimistic = uci;
    inFlight = room?.version ?? null;
    const res = await api.move(uci);
    inFlight = null;
    if (res && res.state) render(res.state);
    else { optimistic = null; if (room) renderPieces(room.positions[shown].fen); }
    return true;
  }

  // Under the pointer: a piece picked up is centred on it, as on chess.com.
  function follow(node: HTMLDivElement, clientX: number, clientY: number) {
    if (!boardEl) return;
    const r = boardEl.getBoundingClientRect();
    const cell = r.width / 8;
    const px = clientX - r.left - cell / 2, py = clientY - r.top - cell / 2;
    node.style.transform = `translate(${(px / cell) * 100}%, ${(py / cell) * 100}%)`;
  }

  function onPointerDown(e: PointerEvent) {
    if (!wrapEl) return;
    if (e.button === 2) { // a right-click: a mark, or an arrow if dragged
      e.preventDefault();
      rightDrag = { from: squareUnder(e.clientX, e.clientY) };
      wrapEl.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    if (marked.length || arrows.length) { marked = []; arrows = []; } // a left-click clears the user's marks
    const square = squareUnder(e.clientX, e.clientY);
    if (!square) return;
    if (selected && targets.includes(square)) { tryMove(selected, square, false); return; }
    const node = pieceAt(square);
    if (node && canMove && colourOf(node.dataset.letter!) === bottom) {
      // Lifted at the press, centred under the pointer.
      drag = { square, node, moved: false, startX: e.clientX, startY: e.clientY };
      wrapEl.setPointerCapture(e.pointerId);
      node.classList.add("dragging");
      follow(node, e.clientX, e.clientY);
      hover = square;
      select(square);
    } else if (selected) {
      selected = null; targets = [];
    }
  }

  function onPointerMove(e: PointerEvent) {
    if (!drag) return;
    if (!drag.moved && Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) >= 4) {
      drag.moved = true;
      document.body.classList.add("grabbing");
    }
    follow(drag.node, e.clientX, e.clientY);
    hover = squareUnder(e.clientX, e.clientY);
  }

  // A piece put down, from a drag: on a legal square it is the move, landed where it was
  // dropped; anywhere else it is back on its square at once, still picked.
  function putDown(d: { square: string; node: HTMLDivElement }, to: string | null) {
    hover = null;
    document.body.classList.remove("grabbing");
    if (to && to !== d.square && targets.includes(to)) {
      tryMove(d.square, to, true);
    } else {
      placeNow(d.node, d.square);
    }
    d.node.classList.remove("dragging");
  }

  function onPointerUp(e: PointerEvent) {
    if (rightDrag) {
      const to = squareUnder(e.clientX, e.clientY);
      const from = rightDrag.from;
      rightDrag = null;
      if (from && to && from !== to) {
        const i = arrows.findIndex(([a, b]) => a === from && b === to);
        if (i >= 0) arrows = arrows.filter((_, j) => j !== i); else arrows = [...arrows, [from, to]];
      } else if (from) {
        marked = marked.includes(from) ? marked.filter((m) => m !== from) : [...marked, from];
      }
      return;
    }
    if (!drag) return;
    const d = drag; drag = null;
    putDown(d, d.moved ? squareUnder(e.clientX, e.clientY) : null); // a click: stays picked
  }

  function onPointerCancel() {
    if (drag) { putDown(drag, null); drag = null; }
    rightDrag = null;
  }

  // ---- the coach and the log ------------------------------------------------------------------
  type Msg = { id: number; you: boolean; meta: string; text: string; marks: Mark[] | null };
  let messages = $state<Msg[]>([]);
  let pending = $state(false);
  let text = $state("");
  let coachEl = $state<HTMLDivElement | null>(null);
  let nextId = 1;

  function addMessage(line: LogLine) {
    pending = false;
    const d = line.data || {};
    const you = line.chan === "user";
    const marks: Mark[] | null = !you && d.annotations && d.annotations.length ? d.annotations : null;
    messages.push({ id: nextId++, you, meta: you ? "you" : String(d.trigger || "coach").replace("_", " "), text: you ? line.text.replace(/^you: /, "") : d.full || line.text, marks });
    if (marks) coachMarks = marks;
    tick().then(() => { if (coachEl) coachEl.scrollTop = coachEl.scrollHeight; });
  }
  function ask(t: string) {
    if (!t) return;
    pending = true;
    tick().then(() => { if (coachEl) coachEl.scrollTop = coachEl.scrollHeight; });
    api.ask(t);
  }
  function send() { const t = text.trim(); if (t) { ask(t); text = ""; } }

  // ---- the room ----------------------------------------------------------------------------------
  function render(s: PuzzleState) {
    if (room && s.version < room.version) return; // an answer older than the one shown
    if (inFlight !== null && s.version <= inFlight) return; // a poll from before the move
    const first = room === null;
    room = s;
    if (first) {
      bottom = s.bottom;
      document.title = `sidecoach · puzzle ${s.id}`;
      shown = s.positions.length - 1;
      renderPieces(s.positions[shown].fen);
    }
    // A wrong move: shown where it landed, red, then taken back by the room.
    if (s.wrong && !wrongShown) {
      wrongShown = s.wrong;
      optimistic = null;
      const node = pieceAt(s.wrong.slice(0, 2));
      if (node) place(node, s.wrong.slice(2, 4));
      selected = null; targets = [];
      if (!retracting) {
        retracting = true;
        setTimeout(async () => {
          const res = await api.retract();
          retracting = false;
          if (res && res.state) render(res.state);
        }, 800);
      }
    } else if (!s.wrong && wrongShown) {
      wrongShown = null;
      renderPieces(s.positions[shown].fen); // the piece goes back
    }
    advance();
    message = s.closed ? "closed" : s.message || "";
    messageClass = s.closed ? "" : s.status === "solved" ? "good" : s.wrong ? "bad" : "";
  }

  // On to the next position the room played: the solver's own move at once, it is there
  // already; the line's moves a moment apart.
  function advance() {
    if (!room || stepping !== null || shown >= room.positions.length - 1) return;
    const before = room.positions[shown].fen;
    const next = room.positions[shown + 1];
    const mine = next.uci !== null && next.uci === optimistic;
    const step = () => {
      stepping = null;
      shown += 1;
      if (mine) optimistic = null;
      else if (next.uci) sound(before, next.uci); // the solver's own was heard as it went down
      renderPieces(next.fen, next.uci);
      advance();
    };
    if (mine) step();
    else stepping = setTimeout(step, REPLY_MS);
  }

  async function giveUp() {
    const res = await api.giveup();
    if (res && res.state) render(res.state);
  }

  async function poll() {
    if (closed) return;
    try {
      const res = await api.poll(cursor);
      if (res.closed && !res.state) { message = "closed"; return; }
      if (res.cursor !== undefined) cursor = Math.max(cursor, res.cursor);
      for (const line of res.lines || []) {
        if (line.chan === "coach" && line.level === "info") addMessage(line);
        if (line.chan === "user") addMessage(line);
      }
      if (res.state) render(res.state);
      if (res.state && res.state.closed) return;
    } catch { /* a poll missed */ }
    setTimeout(poll, 250);
  }

  function fitBoard() {
    if (!mainEl) return;
    // Stacked (a phone): the board across the width, or at the size the player chose on the
    // panel; the column layout needs no --board.
    if (layoutMode()) {
      document.documentElement.style.removeProperty("--board");
      mobileBoard(mainEl, prefs.get("mboard"));
      return;
    }
    const cs = getComputedStyle(mainEl);
    const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
    const byHeight = mainEl.clientHeight - padY - 30;
    const byWidth = mainEl.clientWidth - padX - 14 - 340;
    document.documentElement.style.setProperty("--board", Math.max(240, Math.floor(Math.min(byHeight, byWidth))) + "px");
  }

  const title = $derived(room ? `puzzle ${room.id} · ${room.motif ? motifWords(room.motif) + " · " : ""}${room.rating} · you play ${room.bottom}` : "puzzle");
  const remaining = $derived(room ? (room.status === "playing" ? `${room.remaining} move${room.remaining === 1 ? "" : "s"} to find` : room.status.replace("_", " ")) : "");
  const misses = $derived(room?.misses ? `${room.misses} miss${room.misses === 1 ? "" : "es"}` : "");

  // The next recommended puzzle, loaded in this page's place: a page of its own, so this
  // room closes as the page goes (its beacon) and the next opens fresh, and a tab opened
  // for the first puzzle can still be closed by script.
  function next(e: MouseEvent): void {
    if (!nextIds.length || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault();
    location.replace("/puzzle/" + encodeURIComponent(nextIds[0]));
  }

  // Close the room, then leave: a window or a tab of its own closes; a page with the panel
  // behind it (a phone, where the panel opened it in its own view) goes back to the panel.
  async function leave(): Promise<void> {
    await api.close();
    window.close();
    if (history.length > 1) { history.back(); return; }
    message = "closed — you can close this tab";
    messageClass = "";
  }

  onMount(() => {
    document.body.classList.add("puzzle-page");
    fitBoard();
    window.addEventListener("resize", fitBoard);
    // The bar's height, for the stacked first screen's sum, as on the panel.
    const bar = document.getElementById("bar");
    const barRo = bar && window.ResizeObserver ? new ResizeObserver(() => {
      document.documentElement.style.setProperty("--bar-h", bar.getBoundingClientRect().height + "px");
      fitBoard();
    }) : null;
    if (barRo && bar) barRo.observe(bar);
    const onUnload = () => beacon(api.closePath);
    window.addEventListener("beforeunload", onUnload);
    (async () => {
      await connect();
      const opened = await api.open();
      if (!opened.ok) { message = opened.error || "no such puzzle"; return; }
      nextIds = opened.next || [];
      if (opened.state) render(opened.state);
      poll();
    })();
    return () => {
      closed = true;
      document.body.classList.remove("puzzle-page");
      window.removeEventListener("resize", fitBoard);
      window.removeEventListener("beforeunload", onUnload);
      barRo?.disconnect();
    };
  });
</script>

<svelte:head><title>sidecoach · puzzle</title></svelte:head>

<header id="bar">
  <div class="group">
    <button class="pz-back" aria-label="Back to the panel" title="Back to the panel" onclick={leave}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <img class="pz-mark" src="/logo.svg" alt="" draggable="false" />
    <div class="pz-head">
      <div class="pz-name">Puzzle{room?.motif ? " · " + motifWords(room.motif) : ""}</div>
      <div class="pz-sub">{room ? `rated ${room.rating} · you play ${room.bottom}` : "…"}</div>
    </div>
    <span class="brand">sidecoach</span>
    <span class="strong">{title}</span>
    <span class="pz-msg {messageClass}">{message}</span>
  </div>
  <button class="ghost pz-giveup" title="Play the solution out" disabled={room?.status !== "playing"} onclick={giveUp}>Give up</button>
  <div class="group controls">
    {#if room?.voice_ready !== false}
    <div class="ctlgroup" role="group" aria-label="Voice">
      <span class="ctllabel">voice</span>
      <button id="voice" class="iconbtn small" aria-pressed={!!room?.voice} title="The coach's answers aloud" onclick={() => { if (room) { room.voice = !room.voice; api.voice(room.voice); } }}>
        <svg class="ic-on" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
        <svg class="ic-off" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3L3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4L9.9 6.1 12 8.2V4z"/></svg>
      </button>
    </div>
    {/if}
    <div class="ctlgroup" role="group" aria-label="Puzzle">
      <span class="ctllabel">puzzle</span>
      <button class="ghost" title="Play the solution out" disabled={room?.status !== "playing"} onclick={giveUp}>Give up</button>
      {#if nextIds.length}<a class="ghost linkbtn" href="/puzzle/{encodeURIComponent(nextIds[0])}" title="The next recommended puzzle" onclick={next}>Next</a>{/if}
      {#if room?.url}<a class="ghost linkbtn" href={room.url} target="_blank" rel="noopener" title="This puzzle on Lichess">Lichess ↗</a>{/if}
      <button class="ghost" title="Close this puzzle" onclick={leave}>Close</button>
    </div>
  </div>
</header>

<main bind:this={mainEl}>
  <section id="left">
    <div
      class="board-wrap"
      bind:this={wrapEl}
      role="application"
      aria-label="Chess board"
      onpointerdown={onPointerDown}
      onpointermove={onPointerMove}
      onpointerup={onPointerUp}
      onpointercancel={onPointerCancel}
      oncontextmenu={(e) => e.preventDefault()}
    >
      <div class="board cb" bind:this={boardEl}>
        {#each squares as sq (sq.name)}
          <div class="sq {sq.dark ? 'dark' : 'light'} {sq.cls}" class:odd={sq.odd} data-square={sq.name}>
            {#if sq.rankLabel}<span class="coord rank">{sq.rankLabel}</span>{/if}
            {#if sq.fileLabel}<span class="coord file">{sq.fileLabel}</span>{/if}
          </div>
        {/each}
      </div>
      <div class="pieces" bind:this={piecesEl}></div>
      <svg class="overlay" viewBox="0 0 8 8" aria-hidden="true">
        <defs>
          {#each arrowColours as c (c)}
            <marker id="ah-{c}" class="mk-{c}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" /></marker>
          {/each}
        </defs>
        {#each coachMarks || [] as m}
          {#if m.kind === "square"}
            {@const p = cellOf(m.squares[0], bottom)}
            <rect x={p.x} y={p.y} width="1" height="1" class="mk-{m.color}" opacity={(m.opacity == null ? 1 : m.opacity) * 0.55} />
          {:else if m.kind === "ring"}
            {@const p = cellOf(m.squares[0], bottom)}
            <circle cx={p.x + 0.5} cy={p.y + 0.5} r="0.42" class="mk-{m.color}" opacity={m.opacity == null ? 1 : m.opacity} />
          {:else}
            {@const l = arrowLine(m.squares[0], m.squares[1])}
            <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-{m.color}" opacity={m.opacity == null ? 1 : m.opacity} marker-end="url(#ah-{m.color})" />
          {/if}
        {/each}
        {#each arrows as [from, to] (from + to)}
          {@const l = arrowLine(from, to)}
          <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-user user" opacity="0.85" marker-end="url(#ah-user)" />
        {/each}
      </svg>
      {#if promo}
        <div class="promo" class:down={promo.down} style="left: {promo.x * 12.5}%; top: {(promo.down ? (promo.y - 3) * 12.5 : promo.y * 12.5)}%">
          {#each ["Q", "R", "B", "N"] as p (p)}
            <button class="{bottom === 'white' ? 'w' : 'b'}{p}" title={p} aria-label="Promote to {p}" onclick={(e) => { e.stopPropagation(); promo?.resolve(p.toLowerCase()); }}></button>
          {/each}
          <button class="x" onclick={(e) => { e.stopPropagation(); promo?.resolve(null); }}>×</button>
        </div>
      {/if}
    </div>
    <div class="gameinfo">
      <span class="strong">{room ? "you play " + room.bottom : "—"}</span>
      <span>{remaining}</span>
      <span>{misses}</span>
    </div>
    <!-- Stacked: what the bar says on a wide window, under the board -- the message, what is
         left to find, and where to go next. -->
    <div class="pz-foot">
      <span class="pz-say {messageClass}">{message}</span>
      <span class="pz-left">{[room?.status === "playing" ? remaining : "", misses].filter(Boolean).join(" · ")}</span>
      <span class="pz-links">
        {#if nextIds.length}<a class="pz-link" class:ready={room && room.status !== "playing"} href="/puzzle/{encodeURIComponent(nextIds[0])}" title="The next recommended puzzle" onclick={next}>Next ›</a>{/if}
        {#if room?.url}<a class="pz-link" href={room.url} target="_blank" rel="noopener" title="This puzzle on Lichess">Lichess ↗</a>{/if}
      </span>
    </div>
  </section>

  <section id="right">
    <div class="panel-title">Coach</div>
    <div class="coach" bind:this={coachEl} use:stickToBottom>
      {#each messages as m, i (m.id)}
        <div class="msg" class:you={m.you} class:tail={pending && i === messages.length - 1 ? m.you : messages[i + 1]?.you !== m.you} role="note" title={m.meta} onmouseenter={() => { if (m.marks) coachMarks = m.marks; }}>{m.text}</div>
      {/each}
      {#if pending}<div class="msg typing tail" role="status" aria-label="The coach is writing"><span></span><span></span><span></span></div>{/if}
    </div>
    <div class="composer">
      <div class="box">
        <input type="text" placeholder="Ask the coach…" autocomplete="off" spellcheck="false" bind:value={text} onkeydown={(e) => { if (e.key === "Enter") send(); }} />
        <button class="sendbtn" data-state={text.trim() ? "send" : "quiet"} title="Send (Enter)" aria-label="Send" onclick={send}>
          <svg class="ic ic-send" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 11.5 21 3l-8.5 18-2.5-7.5L3 11.5z"/></svg>
        </button>
      </div>
      <div class="chips-row">
        <button class="qchip" title="A nudge at the current disclosure level" onclick={() => ask("hint")}>Hint</button>
        <button class="qchip" title="The engine's move, then why" onclick={() => ask("what should I play")}>Best move</button>
        <button class="qchip" title="The engine's line" onclick={() => ask("show me the line")}>Line</button>
      </div>
    </div>
  </section>
</main>
