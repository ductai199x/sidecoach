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
  import { beacon, connect, failureText, followCursor, puzzle as puzzleApi, type LogLine, type Mark, type PuzzlePosition, type PuzzleState } from "$lib/api";
  import { FILES, cellOf, colourOf, motifWords, placement, squareAt } from "$lib/chess";
  import { MoveInput } from "$lib/moveinput.svelte";
  import { PieceLayer } from "$lib/pieces";
  import { layoutMode, mobileBoard } from "$lib/layout";
  import { prefs } from "$lib/panel.svelte";
  import { captureSound, moveSound } from "$lib/sounds";
  import { stickToBottom } from "$lib/stick";
  import LineControls from "$lib/review/LineControls.svelte";
  import { LinePlayer } from "$lib/review/lineplayer.svelte";

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
  let coachMarks = $state<Mark[] | null>(null);
  let wrongShown = $state<string | null>(null);
  let occupied = $state<Record<string, string>>({});  // square -> letter, for the marks
  let retracting = false;

  // ---- which position is shown -----------------------------------------------------------
  // The positions the room played are shown one by one: the solver's own move at once (it is
  // on the board already), the line's reply REPLY_MS later. A solution given up is not run: its
  // first move is shown, the rest stepped as the solver asks (below).
  const REPLY_MS = 400;
  let shown = $state(0);
  let optimistic = $state<string | null>(null);  // the solver's move, shown before the answer
  let inFlight: number | null = null;  // the version the move was sent from
  let stepping: ReturnType<typeof setTimeout> | null = null;
  const position = $derived<PuzzlePosition | null>(room?.positions[shown] ?? null);
  const settled = $derived(!!room && shown === room.positions.length - 1);
  const canMove = $derived(!!room && room.your_move && settled && !wrongShown && !optimistic);

  // ---- a solution given up: stepped, never run by itself ---------------------------------------
  // Its first move shown, then ◀ ▶ (or the arrow keys) step it and Play runs it at a reading
  // pace, as the review's lines (lib/review/lineplayer; Tai: lines are stepped, never auto-run
  // fast). `from` is the position it starts from; its move i is positions[from + 1 + i]. A page
  // opened on a puzzle given up already steps it from the start, at its end.
  let solution = $state<{ from: number; length: number } | null>(null);
  let solutionShown = -1;
  const solver = new LinePlayer({
    sound: (i) => {
      const p = solution && room ? room.positions[solution.from + 1 + i] : null;
      if (p?.uci && room && solution) sound(room.positions[solution.from + i].fen, p.uci);
    },
    changed: (i) => showSolution(i),
  });
  const solutionSan = $derived(solution && solver.i >= 0 ? (room?.moves?.[solution.from + solver.i] ?? "") : "");
  let givingUp = $state(false);

  function showSolution(i: number): void {
    if (!solution || !room) return;
    const at = solution.from + 1 + i;
    const next = room.positions[at];
    if (!next) return;
    shown = at;
    // A step on slides the piece that moved; a step back or a jump puts the pieces as they are.
    renderPieces(next.fen, i === solutionShown + 1 ? next.uci : null);
    solutionShown = i;
  }

  function startSolution(s: PuzzleState, first: boolean): void {
    if (stepping !== null) { clearTimeout(stepping); stepping = null; }
    optimistic = null;
    const from = first ? 0 : shown;
    const length = s.positions.length - 1 - from;
    if (length <= 0) return;
    solution = { from, length };
    solutionShown = first ? length - 1 : -1;
    solver.reset(length, solutionShown);
    if (!first) solver.showFirst();
    tick().then(fitBoard);
  }

  function onKey(e: KeyboardEvent): void {
    if (!solution || e.ctrlKey || e.metaKey || e.altKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes((e.target as HTMLElement)?.tagName)) return;
    // Esc: the play stops, the board goes to the solution's end, where the puzzle stands.
    if (e.key === "Escape") { e.preventDefault(); solver.toEnd(true); return; }
    solver.key(e);
  }

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
        if (input.selected === name) cls.push("sel");
        if (input.targets.includes(name)) { cls.push("target"); if (occupied[name]) cls.push("occupied"); }
        if (input.marked.includes(name)) cls.push("marked");
        if (input.hover === name) cls.push("hover");
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
  // The pieces as `fen` has them; `uci`, the move that made it, names the piece that moves.
  function renderPieces(fen: string, uci: string | null = null) {
    if (!piecesEl) return;
    layer ??= new PieceLayer(piecesEl, bottom);
    occupied = layer.render(fen, uci);
  }

  // ---- what the player does: the hand (lib/moveinput), and what a move means here -----------
  const input = new MoveInput({
    board: () => boardEl,
    wrap: () => wrapEl,
    layer: () => layer,
    bottom: () => bottom,
    canMove: () => canMove,
    side: () => bottom,
    legal: () => room?.legal ?? {},
    play,
  });

  // The sound of a move made from `fen`: a capture when it lands on a piece, or a pawn steps
  // aside onto an empty square (en passant); a move otherwise.
  function sound(fen: string, uci: string) {
    const board = placement(fen);
    const mover = board.get(uci.slice(0, 2));
    const takes = board.has(uci.slice(2, 4)) || (mover?.toUpperCase() === "P" && uci[0] !== uci[2]);
    if (takes) captureSound(); else moveSound();
  }

  // The solver's move, on the board already: sent; the answer brings the reply, or shows it
  // wrong.
  async function play(uci: string): Promise<void> {
    if (room) sound(room.positions[shown].fen, uci);
    optimistic = uci;
    inFlight = room?.version ?? null;
    // A failed request must not leave the board waiting on it: the polls are heard again.
    const res = await api.move(uci).catch(() => null);
    inFlight = null;
    if (res && res.state) render(res.state);
    else { optimistic = null; if (room) renderPieces(room.positions[shown].fen); }
  }

  // ---- the coach and the log ------------------------------------------------------------------
  type Msg = { id: number; you: boolean; meta: string; text: string; marks: Mark[] | null };
  let messages = $state<Msg[]>([]);
  let pending = $state(false);
  let text = $state("");
  // A question the room did not take (no answer, or an error): kept in the box, said, Retry.
  let notSent = $state<string | null>(null);
  let sending = $state(false);
  let coachEl = $state<HTMLDivElement | null>(null);
  let nextId = 1;

  function addMessage(line: LogLine) {
    const d = line.data || {};
    const you = line.chan === "user";
    // The echo of the player's own line is not the answer: the dots stay until the coach's
    // (or a warning that none is coming).
    if (!you || line.level !== "info") pending = false;
    // Only "you: ..." is the player's own turn; "no coach to ask: ..." ends the dots and is
    // no bubble of theirs (as on the panel).
    if (you && !line.text.startsWith("you: ")) return;
    const marks: Mark[] | null = !you && d.annotations && d.annotations.length ? d.annotations : null;
    messages.push({ id: nextId++, you, meta: you ? "you" : String(d.trigger || "coach").replace("_", " "), text: you ? line.text.replace(/^you: /, "") : d.full || line.text, marks });
    if (marks) coachMarks = marks;
    tick().then(() => { if (coachEl) coachEl.scrollTop = coachEl.scrollHeight; });
  }
  /** Asked; true once the room took it. When it did not (no answer, an error, no such room), the
   *  typing dots go: no reply is coming. */
  async function ask(t: string): Promise<boolean> {
    if (!t) return false;
    pending = true;
    tick().then(() => { if (coachEl) coachEl.scrollTop = coachEl.scrollHeight; });
    try {
      const res = (await api.ask(t)) as { ok?: boolean } | null;
      if (res?.ok === false) throw new Error("not taken");
      return true;
    } catch {
      pending = false;
      return false;
    }
  }
  // The box is cleared only once the question is sent; a failure leaves it there, with Retry.
  async function send() {
    const t = text.trim();
    if (!t || sending) return;
    sending = true;
    notSent = null;
    const ok = await ask(t);
    sending = false;
    if (!ok) { notSent = t; return; }
    if (text.trim() === t) text = "";
  }
  async function quick(t: string) {
    notSent = null;
    if (!(await ask(t))) notSent = t;
  }
  function retrySend() {
    const t = notSent;
    if (!t) return;
    if (!text.trim()) text = t;
    if (text.trim() === t) void send();
    else void quick(t);
  }

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
    // Given up (by this page's button, or a state that says so): the solution is stepped from
    // the position on the board, however the state came (an answer or a poll).
    if (s.status === "given_up" && !solution) startSolution(s, first);
    // A wrong move: shown where it landed, red, then taken back by the room.
    if (s.wrong && !wrongShown) {
      wrongShown = s.wrong;
      optimistic = null;
      const node = layer?.nodes.get(s.wrong.slice(0, 2));
      if (node) layer?.place(node, s.wrong.slice(2, 4));
      input.clear();
      if (!retracting) {
        retracting = true;
        setTimeout(async () => {
          const res = await api.retract().catch(() => null);
          retracting = false;
          if (res && res.state) render(res.state);
          // Not taken back (the request failed): the piece goes back here, and the board
          // takes moves again; the room's next state says the rest.
          else if (wrongShown && room) { wrongShown = null; renderPieces(room.positions[shown].fen); }
        }, 800);
      }
    } else if (!s.wrong && wrongShown) {
      wrongShown = null;
      renderPieces(s.positions[shown].fen); // the piece goes back
    }
    advance();
    message = s.closed ? "closed" : solution ? "The solution. Step through it." : s.message || "";
    messageClass = s.closed ? "" : s.status === "solved" ? "good" : s.wrong ? "bad" : "";
  }

  // On to the next position the room played: the solver's own move at once, it is there
  // already; the line's moves a moment apart.
  function advance() {
    if (!room || solution || stepping !== null || shown >= room.positions.length - 1) return;
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

  // Once at a time, and not while a move is on its way; a failure is said, and the puzzle goes on.
  async function giveUp() {
    if (!room || room.status !== "playing" || givingUp || optimistic) return;
    givingUp = true;
    try {
      const res = await api.giveup();
      if (!res.state) throw new Error("no state");
      render(res.state);
    } catch (e) {
      message = `Couldn't show the solution. ${failureText(e)}`;
      messageClass = "bad";
    } finally {
      givingUp = false;
    }
  }

  async function poll() {
    if (closed) return;
    try {
      const res = await api.poll(cursor, document.hidden);
      if (res.closed && !res.state) { message = "closed"; return; }
      cursor = followCursor(cursor, res.cursor);
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
    // The strip under the board (the message, what is left, Next) keeps its room.
    const foot = (mainEl.querySelector(".pz-foot") as HTMLElement | null)?.getBoundingClientRect().height ?? 56;
    const byHeight = mainEl.clientHeight - padY - foot - 8;
    const byWidth = mainEl.clientWidth - padX - 14 - 340;
    document.documentElement.style.setProperty("--board", Math.max(240, Math.floor(Math.min(byHeight, byWidth))) + "px");
  }

  const capital = (w: string) => w.charAt(0).toUpperCase() + w.slice(1);
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
    await api.close().catch(() => {});
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
    // The room closes when this page goes, and only then (Tai, 2026-10-02): told as the page
    // unloads, and as it is left inside the app (a phone's back), where nothing unloads.
    const onUnload = () => beacon(api.closePath);
    window.addEventListener("beforeunload", onUnload);
    // Out of view, the page's polls slow or stop: the room is told at once, so it never takes
    // that quiet for a page that vanished.
    const onVisibility = () => { if (document.hidden && !closed) void api.poll(cursor, true).catch(() => {}); };
    document.addEventListener("visibilitychange", onVisibility);
    (async () => {
      let opened: Awaited<ReturnType<typeof api.open>>;
      try {
        await connect();
        opened = await api.open();
      } catch (e) {
        message = `Couldn't open the puzzle. ${failureText(e)}`;
        messageClass = "bad";
        return;
      }
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
      document.removeEventListener("visibilitychange", onVisibility);
      onUnload();
      barRo?.disconnect();
    };
  });
</script>

<svelte:head><title>sidecoach · puzzle</title></svelte:head>
<svelte:window onkeydown={onKey} />

<header id="bar">
  <div class="group">
    <button class="iconbtn pz-back" aria-label="Close the puzzle" title="Close the puzzle" onclick={leave}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <img class="pz-mark" src="/logo.svg" alt="" draggable="false" />
    <div class="pz-head">
      <div class="pz-name">Puzzle{room?.motif ? " · " + motifWords(room.motif) : ""}</div>
      <div class="pz-sub">{room ? `Rated ${room.rating} · You play ${capital(room.bottom)}` : "…"}</div>
    </div>
  </div>
  <div class="group controls">
    {#if room?.voice_ready !== false}
      <button id="voice" class="iconbtn" aria-pressed={!!room?.voice} title="The coach's answers aloud" aria-label="The coach's voice" onclick={() => { if (room) { const on = !room.voice; room.voice = on; api.voice(on).catch(() => { if (room) room.voice = !on; }); } }}>
        <svg class="ic-on" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
        <svg class="ic-off" viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3L3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4L9.9 6.1 12 8.2V4z"/></svg>
      </button>
    {/if}
    {#if room?.url}
      <a class="iconbtn" href={room.url} target="_blank" rel="noopener" title="This puzzle on Lichess" aria-label="This puzzle on Lichess">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path fill="currentColor" d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>
      </a>
    {/if}
    <button class="ghost pz-giveup" title="Play the solution out" disabled={room?.status !== "playing" || givingUp || !!optimistic} aria-busy={givingUp} onclick={giveUp}>Give up</button>
  </div>
</header>

<main bind:this={mainEl}>
  <section id="left">
    <div
      class="board-wrap"
      bind:this={wrapEl}
      role="application"
      aria-label="Chess board"
      onpointerdown={input.down}
      onpointermove={input.move}
      onpointerup={input.up}
      onpointercancel={input.cancel}
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
        {#each input.arrows as [from, to] (from + to)}
          {@const l = arrowLine(from, to)}
          <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-user user" opacity="0.85" marker-end="url(#ah-user)" />
        {/each}
      </svg>
      {#if input.promo}
        {@const promo = input.promo}
        <div class="promo" class:down={promo.down} style="left: {promo.x * 12.5}%; top: {(promo.down ? (promo.y - 3) * 12.5 : promo.y * 12.5)}%">
          {#each ["Q", "R", "B", "N"] as p (p)}
            <button class="{bottom === 'white' ? 'w' : 'b'}{p}" title={p} aria-label="Promote to {p}" onclick={(e) => { e.stopPropagation(); promo?.resolve(p.toLowerCase()); }}></button>
          {/each}
          <button class="x" onclick={(e) => { e.stopPropagation(); promo?.resolve(null); }}>×</button>
        </div>
      {/if}
    </div>
    <!-- Under the board, at every width: what the puzzle says (green when solved, red on a miss),
         what is left to find, and Next, lit once it is done. -->
    <div class="pz-foot">
      <span class="pz-say {messageClass}">{message}</span>
      {#if solution}
        <span class="pz-left pz-solution"><LineControls player={solver} label="the solution" extra={solutionSan} /></span>
      {:else}
        <span class="pz-left">{[room?.status === "playing" ? remaining : "", misses].filter(Boolean).join(" · ")}</span>
      {/if}
      <span class="pz-links">
        {#if nextIds.length}<a class="pz-link" class:ready={room && room.status !== "playing"} href="/puzzle/{encodeURIComponent(nextIds[0])}" title="The next recommended puzzle" onclick={next}>Next puzzle ›</a>{/if}
      </span>
    </div>
  </section>

  <section id="right">
    <div class="panel-title">Coach</div>
    <div class="coach" bind:this={coachEl} use:stickToBottom>
      {#if !messages.length && !pending}
        <div class="coach-empty"><b>Find the best move.</b><span>Stuck? Press Hint, or ask the coach anything below.</span></div>
      {/if}
      {#each messages as m, i (m.id)}
        <div class="msg" class:you={m.you} class:tail={pending && i === messages.length - 1 ? m.you : messages[i + 1]?.you !== m.you} role="note" title={m.meta} onmouseenter={() => { if (m.marks) coachMarks = m.marks; }}>{m.text.replaceAll(" -- ", " — ")}</div>
      {/each}
      {#if pending}<div class="msg typing tail" role="status" aria-label="The coach is writing"><span></span><span></span><span></span></div>{/if}
    </div>
    <div class="composer">
      <div class="box">
        <input type="text" placeholder="Ask the coach…" autocomplete="off" spellcheck="false" bind:value={text} oninput={() => (notSent = null)} onkeydown={(e) => { if (e.key === "Enter") void send(); }} />
        <button class="sendbtn" data-state={text.trim() && !sending ? "send" : "quiet"} disabled={sending} title="Send (Enter)" aria-label="Send" onclick={() => void send()}>
          <svg class="ic ic-send" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l7 7-1.4 1.4L13 7.8V20h-2V7.8l-4.6 4.6L5 11z"/></svg>
        </button>
      </div>
      {#if notSent}
        <div class="not-sent" role="alert">Not sent — <button onclick={retrySend}>Retry</button></div>
      {/if}
      <div class="chips-row">
        <button class="qchip" title="A nudge at the current disclosure level" onclick={() => quick("hint")}>Hint</button>
        <button class="qchip" title="The engine's move, then why" onclick={() => quick("what should I play")}>Best move</button>
        <button class="qchip" title="The engine's line" onclick={() => quick("show me the line")}>Line</button>
      </div>
    </div>
  </section>
</main>
