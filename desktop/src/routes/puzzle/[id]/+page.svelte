<script lang="ts">
  // The puzzle page (/puzzle/<id>): a board with chess.com's manners, the coach beside
  // it, the room's log below. The room lives on the sidecar (ui/puzzle_room.py); this
  // page polls it and sends what the player does.
  //
  // The board: click a piece of yours and it lights yellow with its legal squares dotted
  // (none if it cannot move, the king in check); click a square to move there, or drag
  // the piece and drop it. A right-click marks a square red; a right-drag draws an arrow;
  // a left-click anywhere clears them. Moves slide, captures fade, a pawn reaching the
  // last rank offers a picker over the square. A wrong move lands, shows red, and comes
  // back. The coach says nothing unasked: Hint, Best move, Line, or a typed question.
  import { onMount, tick } from "svelte";
  import { page } from "$app/state";
  import "$lib/styles/puzzle.css";
  import { beacon, connect, puzzle as puzzleApi, type LogLine, type Mark, type PuzzleState } from "$lib/api";
  import { FILES, cellOf, colourOf, motifWords, placement, spriteOf, squareAt } from "$lib/chess";
  import { layoutMode, mobileBoard } from "$lib/layout";
  import { prefs } from "$lib/panel.svelte";
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
  let retracting = false;

  let boardEl = $state<HTMLDivElement | null>(null);
  let piecesEl = $state<HTMLDivElement | null>(null);
  let wrapEl = $state<HTMLDivElement | null>(null);
  let mainEl = $state<HTMLElement | null>(null);

  // The squares in display order for the orientation, with the classes the marks give them.
  const squares = $derived.by(() => {
    const out: { name: string; dark: boolean; odd: boolean; rankLabel: string | null; fileLabel: string | null; cls: string }[] = [];
    const last = room?.last_move && !wrongShown ? [room.last_move.slice(0, 2), room.last_move.slice(2, 4)] : [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const name = squareAt(x, y, bottom)!;
        const file = FILES.indexOf(name[0]), rank = +name[1] - 1;
        const cls: string[] = [];
        if (room?.your_move && !wrongShown && occupied[name] && colourOf(occupied[name]) === bottom) cls.push("grab");
        if (last.includes(name)) cls.push("hl");
        if (wrongShown && name === wrongShown.slice(0, 2)) cls.push("hl");
        if (wrongShown && name === wrongShown.slice(2, 4)) cls.push("wrong");
        if (room?.check && name === room.check) cls.push("check");
        if (selected === name) cls.push("sel");
        if (targets.includes(name)) { cls.push("target"); if (occupied[name]) cls.push("occupied"); }
        if (marked.includes(name)) cls.push("marked");
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
  let pieces = new Map<string, HTMLDivElement>();
  function place(node: HTMLDivElement, square: string) {
    const { x, y } = cellOf(square, bottom);
    node.style.transform = `translate(${x * 100}%, ${y * 100}%)`;
  }
  function renderPieces(fen: string) {
    if (!piecesEl) return;
    const want = placement(fen);
    const stay = new Map<string, HTMLDivElement>();
    const gone: [string, string, HTMLDivElement][] = [];
    for (const [sq, node] of pieces) {
      if (want.get(sq) === node.dataset.letter) {
        // Placed again even so: a piece slid ahead of the server (a move sent, or a wrong
        // one shown) stands where its picture is until the position puts it back.
        place(node, sq);
        stay.set(sq, node);
      } else gone.push([sq, node.dataset.letter!, node]);
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
        piecesEl.appendChild(node);
        node.getBoundingClientRect(); // settle before the transition returns
        node.style.transition = "";
        next.set(sq, node);
      }
    }
    for (const [, , node] of gone) { // taken, or a promoted pawn: fades out
      node.classList.add("fading");
      setTimeout(() => node.remove(), 140);
    }
    pieces = next;
    const occ: Record<string, string> = {};
    for (const [sq, node] of pieces) occ[sq] = node.dataset.letter!;
    occupied = occ;
  }

  // ---- what the player does ------------------------------------------------------------------
  let drag: { square: string; node: HTMLDivElement; moved: boolean; startX: number; startY: number } | null = null;
  let rightDrag: { from: string | null } | null = null;
  let selecting: Promise<void> | null = null;
  let promo = $state<{ square: string; x: number; y: number; down: boolean; resolve: (p: string | null) => void } | null>(null);

  function squareUnder(clientX: number, clientY: number): string | null {
    if (!boardEl) return null;
    const r = boardEl.getBoundingClientRect();
    const x = Math.floor(((clientX - r.left) / r.width) * 8);
    const y = Math.floor(((clientY - r.top) / r.height) * 8);
    return squareAt(x, y, bottom);
  }

  function select(square: string): Promise<void> {
    const node = pieces.get(square);
    if (!room || !room.your_move || wrongShown || !node || colourOf(node.dataset.letter!) !== bottom) {
      selected = null; targets = []; return Promise.resolve();
    }
    selected = square;
    targets = [];
    selecting = api.targets(square).then((res) => {
      if (selected !== square) return;
      targets = (res && res.targets) || [];
    }).finally(() => { selecting = null; });
    return selecting;
  }

  function isPromotion(from: string, to: string): boolean {
    const node = pieces.get(from);
    if (!node || node.dataset.letter!.toUpperCase() !== "P") return false;
    return to[1] === "8" || to[1] === "1";
  }

  function promotionPicker(square: string): Promise<string | null> {
    return new Promise((resolve) => {
      const { x, y } = cellOf(square, bottom);
      promo = { square, x, y, down: y > 3, resolve: (p) => { promo = null; resolve(p); } };
    });
  }

  async function tryMove(from: string, to: string): Promise<boolean> {
    if (!targets.includes(to)) return false;
    selected = null; targets = [];
    if (isPromotion(from, to)) {
      const piece = await promotionPicker(to);
      if (!piece) return false;
      await api.move(from + to + piece);
      return true;
    }
    // Slide the piece at once; the poll confirms (or shows the wrong move and takes it back).
    const node = pieces.get(from);
    if (node) place(node, to);
    await api.move(from + to);
    return true;
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
    if (selected && targets.includes(square)) { tryMove(selected, square); return; }
    const node = pieces.get(square);
    if (node && room && room.your_move && !wrongShown && colourOf(node.dataset.letter!) === bottom) {
      drag = { square, node, moved: false, startX: e.clientX, startY: e.clientY };
      wrapEl.setPointerCapture(e.pointerId);
      select(square);
    } else if (selected) {
      selected = null; targets = [];
    }
  }

  function onPointerMove(e: PointerEvent) {
    if (!drag || !boardEl) return;
    const dx = e.clientX - drag.startX, dy = e.clientY - drag.startY;
    if (!drag.moved && Math.hypot(dx, dy) < 4) return;
    if (!drag.moved) { drag.moved = true; drag.node.classList.add("dragging"); document.body.classList.add("grabbing"); }
    const r = boardEl.getBoundingClientRect();
    const cell = r.width / 8;
    const px = e.clientX - r.left - cell / 2, py = e.clientY - r.top - cell / 2;
    drag.node.style.transform = `translate(${(px / cell) * 100}%, ${(py / cell) * 100}%)`;
  }

  async function onPointerUp(e: PointerEvent) {
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
    document.body.classList.remove("grabbing");
    d.node.classList.remove("dragging");
    if (!d.moved) { place(d.node, d.square); return; } // a click: stays selected
    if (selecting) await selecting; // a quick drag: the legal squares may still be on their way
    const to = squareUnder(e.clientX, e.clientY);
    if (to && to !== d.square && targets.includes(to)) {
      await tryMove(d.square, to);
    } else {
      place(d.node, d.square); // back where it was, still selected
    }
  }

  function onPointerCancel() {
    if (drag) { place(drag.node, drag.square); drag.node.classList.remove("dragging"); drag = null; }
    rightDrag = null;
    document.body.classList.remove("grabbing");
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
    const first = room === null;
    room = s;
    if (first) {
      bottom = s.bottom;
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
    message = s.closed ? "closed" : s.message || "";
    messageClass = s.closed ? "" : s.status === "solved" ? "good" : s.wrong ? "bad" : "";
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
  <button class="ghost pz-giveup" title="Play the solution out" disabled={room?.status !== "playing"} onclick={() => api.giveup()}>Give up</button>
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
      <button class="ghost" title="Play the solution out" disabled={room?.status !== "playing"} onclick={() => api.giveup()}>Give up</button>
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
      {#each messages as m (m.id)}
        <div class="msg" class:you={m.you} role="note" onmouseenter={() => { if (m.marks) coachMarks = m.marks; }}>
          <div class="meta">{m.meta}</div>
          <div>{m.text}</div>
        </div>
      {/each}
      {#if pending}<div class="msg pending"><div class="meta">coach</div><div>thinking<span class="dots"></span></div></div>{/if}
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
