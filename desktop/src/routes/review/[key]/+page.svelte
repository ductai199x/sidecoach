<script lang="ts">
  // The review page (/review/<key>): a game reviewed whole, as chess.com's Game Review goes --
  // the analysis with its progress, then the summary (the coach's line on the game, the graph,
  // each side's accuracy, kinds of move, game rating and phases, the game's shape), then the
  // game move by move: the board with the move's class on it and the best move's arrow, the
  // eval bar, the coach's card on the move, the lines it can play, the move list, the graph.
  // The room lives on the sidecar (ui/review_room.py); this page polls it while it analyses.
  import { onMount } from "svelte";
  import { page } from "$app/state";
  import "$lib/styles/review.css";
  import { beacon, connect, review as reviewApi, type LineStep, type RetryPosition, type ReviewMove, type ReviewState, type Tried } from "$lib/api";
  import { inShell, openWindow } from "$lib/shell.svelte";
  import { placement } from "$lib/chess";
  import { layoutMode, mobileBoard } from "$lib/layout";
  import { prefs } from "$lib/panel.svelte";
  import { captureSound, moveSound } from "$lib/sounds";
  import EvalBar from "$lib/review/EvalBar.svelte";
  import EvalGraph from "$lib/review/EvalGraph.svelte";
  import ReviewBoard from "$lib/review/ReviewBoard.svelte";
  import { BAD, CLASSES, FOLDED, KEY, SOLVED, className, evalSide, evalText, headline, numbered, openingName, sentence, tryWords, verdict } from "$lib/review/words";
  import { motifWords } from "$lib/chess";

  const key = decodeURIComponent(page.params.key ?? "");
  const api = reviewApi(key);

  let room = $state<ReviewState | null>(null);
  let failure = $state("");
  let view = $state<"summary" | "moves">("summary");
  let ply = $state(0);
  let bottom = $state<"white" | "black">("white");
  let unfolded = $state(false);
  // A line the board is playing instead of the game: the best from before the move, or what
  // follows it; `i` is the step shown (-1: the position the line starts from).
  let line = $state<{ kind: "best" | "follow"; steps: LineStep[]; from: string; i: number } | null>(null);
  let playing: ReturnType<typeof setTimeout> | null = null;
  let closed = false;
  // Retry (the plan's D): another move found in the position a move was played from, graded as
  // the review graded the game's (ui/review_retry.py). `wrong`: a try shown red, then put back;
  // `hint`: the coach's hints given (the third draws the move); the best arrow stays hidden.
  type Retry = RetryPosition & {
    tried: Tried | null;
    wrong: string | null;
    busy: boolean;
    hint: number;
    hintText: string;
    hintBest: string | null;
    solved: boolean;
    gaveUp: boolean;
  };
  let retry = $state<Retry | null>(null);
  let redraw = $state(0);
  // The best try at each move retried: what "accuracy after retries" counts.
  let retried = $state<Record<number, Tried>>({});
  const WRONG_MS = 900;

  // ---- the review's settings (the plan's H), kept for the next review ----
  type Look = { arrow: boolean; marks: boolean; keys: "both" | "white" | "black"; spent: boolean };
  const LOOK: Look = { arrow: true, marks: true, keys: "both", spent: false };
  let look = $state<Look>(readLook());
  let settingsOpen = $state(false);
  function readLook(): Look {
    try {
      return { ...LOOK, ...JSON.parse(prefs.getStr("reviewLook") ?? "{}") };
    } catch {
      return { ...LOOK };
    }
  }
  function setLook(change: Partial<Look>): void {
    look = { ...look, ...change };
    prefs.setStr("reviewLook", JSON.stringify(look));
  }

  const moves = $derived(room?.moves ?? []);
  const summary = $derived(room?.summary ?? null);
  const graph = $derived(summary?.graph ?? []);
  const move = $derived<ReviewMove | null>(ply > 0 ? (moves[ply - 1] ?? null) : null);
  const whiteBefore = $derived(graph[ply - 1] ?? 0.5);
  // The moves the review stops at: every side's, or the one side the settings name.
  const keyMoments = $derived(
    (summary?.key_moments ?? []).filter((k) => k.kind !== "book" && (look.keys === "both" || k.side === look.keys)),
  );

  // What the board shows: the game at `ply`, a line's step, or the position being retried.
  const shown = $derived.by(() => {
    if (retry) {
      const kept = retry.solved && retry.tried;
      return { fen: kept ? retry.tried!.fen : retry.fen, uci: kept ? retry.tried!.uci : null, cls: kept && look.marks ? retry.tried!.label : null };
    }
    if (line) {
      const step = line.i >= 0 ? line.steps[line.i] : null;
      return { fen: step ? step.fen : line.from, uci: step ? step.uci : null, cls: null as string | null };
    }
    if (!room) return null;
    if (!move) return { fen: room.start_fen ?? "", uci: null, cls: null };
    return { fen: move.fen_after, uci: move.uci, cls: look.marks ? move.label : null };
  });
  // The best move's arrow, on the position the move left, when the move was not it: shown
  // unless a line is playing or the settings turn it off.
  // Retrying, only the third hint's arrow.
  const arrows = $derived.by(() => {
    if (retry) return retry.hintBest && !retry.solved ? [{ from: retry.hintBest.slice(0, 2), to: retry.hintBest.slice(2, 4), kind: "best" as const }] : [];
    if (!look.arrow || line || !move || !move.best_uci || move.best_uci === move.uci || view !== "moves") return [];
    return [{ from: move.best_uci.slice(0, 2), to: move.best_uci.slice(2, 4), kind: "best" as const }];
  });
  const barScore = $derived(line || retry ? whiteBefore : (graph[ply] ?? 0.5));
  const barEval = $derived(line || retry ? null : (move?.eval ?? null));

  // Each side's accuracy with its retried moves' best tries in their place (chess.com's
  // "accuracy after retries"): the plain mean, as the review's own.
  const afterRetries = $derived.by(() => {
    const out: Partial<Record<"white" | "black", { before: number; after: number }>> = {};
    for (const side of ["white", "black"] as const) {
      const own = moves.filter((m) => m.side === side);
      if (!own.some((m) => retried[m.ply])) continue;
      const mean = (pick: (m: ReviewMove) => number) => own.reduce((s, m) => s + pick(m), 0) / own.length;
      out[side] = {
        before: mean((m) => m.accuracy ?? 0),
        after: mean((m) => Math.max(m.accuracy ?? 0, retried[m.ply]?.accuracy ?? 0)),
      };
    }
    return out;
  });
  // The moves a retry improved on: You played / New move.
  const improvedMoves = $derived(moves.filter((m) => retried[m.ply] && (retried[m.ply].accuracy ?? 0) > (m.accuracy ?? 0)));

  // ---- Retry ----
  async function startRetry(): Promise<void> {
    if (!move) return;
    stopLine();
    const at = await api.retry(move.ply);
    if (!at.ok) return;
    retry = { ply: at.ply, fen: at.fen, side: at.side, legal: at.legal, tried: null, wrong: null, busy: false, hint: 0, hintText: "", hintBest: null, solved: false, gaveUp: false };
  }

  function endRetry(): void {
    retry = null;
    redraw++;
  }

  // A try: on the board already (the hand put it there); graded; a solving move stays, any
  // other shows red and goes back.
  async function playTry(uci: string): Promise<void> {
    if (!retry || retry.busy) return;
    soundOf(retry.fen, uci);
    retry = { ...retry, busy: true };
    const graded = await api.attempt(retry.ply, uci);
    if (!retry) return;
    if (!graded.ok) { retry = { ...retry, busy: false }; redraw++; return; }
    const best = retried[graded.ply];
    if (!best || (graded.accuracy ?? 0) > (best.accuracy ?? 0)) retried = { ...retried, [graded.ply]: graded };
    if (SOLVED.has(graded.label ?? "")) {
      retry = { ...retry, tried: graded, busy: false, solved: true, hintBest: null };
      return;
    }
    retry = { ...retry, tried: graded, wrong: uci };
    setTimeout(() => {
      if (!retry || retry.wrong !== uci) return;
      retry = { ...retry, wrong: null, busy: false };
      redraw++;
    }, WRONG_MS);
  }

  async function nextHint(): Promise<void> {
    if (!retry || retry.hint >= 3) return;
    const got = await api.hint(retry.ply, retry.hint + 1);
    if (!retry || !got.ok) return;
    retry = { ...retry, hint: got.level, hintText: got.text, hintBest: got.best };
  }

  // Given up: the best move, on the board.
  function showMove(): void {
    if (!retry || !move || !move.best_line.length) return;
    const first = move.best_line[0];
    retry = {
      ...retry,
      tried: { ply: retry.ply, uci: first.uci, san: first.san, fen: first.fen, label: "best", loss: 0, accuracy: 100, only: false },
      solved: true,
      gaveUp: true,
      hintBest: null,
    };
  }

  // ---- going through the game ----
  function soundOf(fen: string, uci: string): void {
    const board = placement(fen);
    const mover = board.get(uci.slice(0, 2));
    const takes = board.has(uci.slice(2, 4)) || (mover?.toUpperCase() === "P" && uci[0] !== uci[2]);
    if (takes) captureSound(); else moveSound();
  }

  function stopLine(): void {
    if (playing) clearTimeout(playing);
    playing = null;
    line = null;
  }

  function go(to: number, heard = false): void {
    stopLine();
    retry = null;
    const next = Math.max(0, Math.min(moves.length, to));
    if (heard && next === ply + 1 && moves[next - 1]) soundOf(moves[next - 1].fen_before, moves[next - 1].uci);
    ply = next;
    view = "moves";
  }

  function step(delta: number): void {
    if (line) {
      const i = Math.max(-1, Math.min(line.steps.length - 1, line.i + delta));
      if (i > line.i && line.steps[i]) soundOf(i > 0 ? line.steps[i - 1].fen : line.from, line.steps[i].uci);
      line = { ...line, i };
      return;
    }
    go(ply + delta, delta === 1);
  }

  function nextKey(): void {
    const next = keyMoments.find((k) => k.ply > ply);
    go(next ? next.ply : moves.length);
  }

  // A line played out on the board, a move every LINE_MS; the arrows step through it after.
  const LINE_MS = 750;
  function playLine(kind: "best" | "follow"): void {
    if (!move) return;
    stopLine();
    const steps = kind === "best" ? move.best_line : move.follow_line;
    if (!steps.length) return;
    line = { kind, steps, from: kind === "best" ? move.fen_before : move.fen_after, i: -1 };
    const advance = () => {
      if (!line || line.i >= line.steps.length - 1) { playing = null; return; }
      step(1);
      playing = setTimeout(advance, LINE_MS);
    };
    playing = setTimeout(advance, 350);
  }

  function onKey(e: KeyboardEvent): void {
    if (!room || room.status !== "ready" || (e.target as HTMLElement)?.tagName === "INPUT") return;
    if (e.key === "ArrowRight") { e.preventDefault(); if (view === "summary") go(1, true); else step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "Home" || e.key === "ArrowUp") { e.preventDefault(); go(0); }
    else if (e.key === "End" || e.key === "ArrowDown") { e.preventDefault(); go(moves.length); }
    else if (e.key === "Escape" && settingsOpen) { e.preventDefault(); settingsOpen = false; }
    else if (e.key === "Escape" && retry) { e.preventDefault(); endRetry(); }
    else if (e.key === "Escape" && line) { e.preventDefault(); stopLine(); }
    else if (e.key === "f") bottom = bottom === "white" ? "black" : "white";
  }

  // ---- the coach's voice: the summary's line, then each card, said as it is shown ----
  // A card is said once the player stops on it (SAY_MS), so stepping through quickly says
  // only where they stop; a new one cuts off the last. On unless turned off here.
  const SAY_MS = 450;
  let voiceReady = $state(false);
  let voiceOn = $state(prefs.getStr("reviewVoice") !== "off");
  let saying: ReturnType<typeof setTimeout> | null = null;
  const spoken = $derived.by(() => {
    if (!room || room.status !== "ready" || !summary) return null;
    if (view === "summary") return verdict(summary, room.game.result);
    if (retry) return retry.tried && !retry.gaveUp ? tryWords(retry.tried) : retry.hintText || null;
    if (line || !move) return null;
    return `${headline(move)}. ${sentence(move, whiteBefore)}`;
  });
  $effect(() => {
    const text = spoken, on = voiceOn && voiceReady;
    if (saying) clearTimeout(saying);
    saying = on && text ? setTimeout(() => api.say(text), SAY_MS) : null;
  });
  function toggleVoice(): void {
    voiceOn = !voiceOn;
    prefs.setStr("reviewVoice", voiceOn ? null : "off");
    if (!voiceOn) api.hush();
  }

  // The move shown stays in view in the list, as the arrows step past its edge: the list's
  // own box scrolls under the card (the desktop), never the page (stacked, one long page).
  let listEl = $state<HTMLDivElement | null>(null);
  $effect(() => {
    void ply;
    const box = listEl;
    const el = listEl?.querySelector(".mv.at") as HTMLElement | null;
    if (!box || !el || box.scrollHeight <= box.clientHeight) return;
    const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
    if (r.top < b.top) box.scrollTop -= b.top - r.top + 8;
    else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom + 8;
  });

  // ---- the move list: two columns ----
  const rows = $derived.by(() => {
    const out: { n: number; white: ReviewMove | null; black: ReviewMove | null }[] = [];
    for (const m of moves) {
      const n = Math.ceil(m.ply / 2);
      if (!out.length || out[out.length - 1].n !== n) out.push({ n, white: null, black: null });
      out[out.length - 1][m.side] = m;
    }
    return out;
  });

  // The clock each side had at the position shown: its last move's, or none.
  function clockOf(side: "white" | "black"): string {
    for (let i = ply - 1; i >= 0; i--) {
      const m = moves[i];
      if (m.side === side && m.clock_ms != null) {
        const s = Math.floor(m.clock_ms / 1000);
        return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
      }
    }
    return "";
  }

  const PHASES = ["opening", "middlegame", "endgame"] as const;
  function gradeOf(side: "white" | "black", phase: string) {
    return summary?.[side].phases.find((p) => p.phase === phase) ?? null;
  }

  // ---- the room ----
  async function poll(): Promise<void> {
    if (closed) return;
    try {
      const res = await api.poll(0);
      if (res.state) take(res.state);
      if (res.closed && !res.state) { failure = "This review was closed."; return; }
      if (res.state && (res.state.status === "ready" || res.state.status === "failed" || res.state.closed)) return;
    } catch { /* a poll missed */ }
    setTimeout(poll, 400);
  }

  function take(s: ReviewState): void {
    const first = !room || room.status !== "ready";
    room = s;
    if (s.status === "ready" && first) {
      bottom = s.game.user_side ?? "white";
      document.title = `sidecoach · review · ${s.game.white ?? "White"} v ${s.game.black ?? "Black"}`;
      requestAnimationFrame(fitBoard);
    }
  }

  async function open(): Promise<void> {
    await connect();
    const opened = await api.open();
    if (!opened.ok) { failure = opened.error || "There is no such game."; return; }
    voiceReady = !!opened.voice;
    if (opened.state) take(opened.state);
    poll();
  }

  // Reviewed again at another strength (a strength reviewed before opens at once).
  async function reviewAt(strength: string): Promise<void> {
    if (!room || room.strength === strength) return;
    stopLine();
    const opened = await api.open(strength);
    if (!opened.ok || !opened.state) return;
    room = null;
    view = "summary";
    ply = 0;
    take(opened.state);
    poll();
  }

  /** Time spent on a move as chess.com writes it: 0.4s, 12s, 1m 5s. */
  function spentText(ms: number | null): string {
    if (ms == null) return "";
    const s = ms / 1000;
    if (s < 9.95) return `${s.toFixed(1)}s`;
    const whole = Math.round(s);
    return whole < 60 ? `${whole}s` : `${Math.floor(whole / 60)}m ${whole % 60}s`;
  }

  async function leave(): Promise<void> {
    closed = true;
    await api.close();
    window.close();
    if (history.length > 1) history.back();
  }

  function fitBoard(): void {
    const mainEl = document.querySelector("main") as HTMLElement | null;
    if (!mainEl) return;
    if (layoutMode()) {
      document.documentElement.style.removeProperty("--board");
      mobileBoard(mainEl, prefs.get("mboard"));
      return;
    }
    const cs = getComputedStyle(mainEl);
    const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
    const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
    const byHeight = mainEl.clientHeight - padY - 2 * 30;
    const byWidth = mainEl.clientWidth - padX - 30 - 14 - 360;
    document.documentElement.style.setProperty("--board", Math.max(260, Math.floor(Math.min(byHeight, byWidth))) + "px");
  }

  const game = $derived(room?.game ?? null);
  const title = $derived(game ? `${game.white ?? "White"} v ${game.black ?? "Black"}` : "Game Review");
  const sub = $derived(
    game
      ? [game.result && game.result !== "*" ? game.result.replaceAll("1/2", "½") : null, game.time_class, game.played_at ? new Date(game.played_at * 1000).toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" }) : null].filter(Boolean).join(" · ")
      : "…",
  );
  const top = $derived(bottom === "white" ? "black" : "white");

  onMount(() => {
    document.body.classList.add("review-page");
    fitBoard();
    window.addEventListener("resize", fitBoard);
    const onUnload = () => beacon(api.closePath);
    window.addEventListener("beforeunload", onUnload);
    open();
    return () => {
      closed = true;
      stopLine();
      document.body.classList.remove("review-page");
      window.removeEventListener("resize", fitBoard);
      window.removeEventListener("beforeunload", onUnload);
    };
  });
</script>

<svelte:window onkeydown={onKey} onpointerdown={(e) => { if (settingsOpen && !(e.target as HTMLElement).closest(".rv-set")) settingsOpen = false; }} />
<svelte:head><title>sidecoach · review</title></svelte:head>

<header id="bar">
  <div class="group">
    <button class="iconbtn rv-back" aria-label="Close the review" title="Close the review" onclick={leave}>
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <img class="rv-logo" src="/logo.svg" alt="" draggable="false" />
    <div>
      <div class="rv-name">{title}</div>
      <div class="rv-sub">{sub}</div>
    </div>
  </div>
  <div class="group controls">
    {#if voiceReady}
      <button id="voice" class="iconbtn small" aria-pressed={voiceOn} title="The coach reads each move aloud" aria-label="The coach's voice" onclick={toggleVoice}>
        <svg class="ic-on" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
        <svg class="ic-off" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3L3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4L9.9 6.1 12 8.2V4z"/></svg>
      </button>
    {/if}
    <button class="ghost small" title="Turn the board round (F)" onclick={() => (bottom = top)}>Flip board</button>
    <div class="rv-set">
      <button class="iconbtn small" aria-expanded={settingsOpen} aria-label="Review settings" title="Review settings" onclick={() => (settingsOpen = !settingsOpen)}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M19.14 12.94a7.07 7.07 0 0 0 0-1.88l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.61-.22l-2.39.96a7 7 0 0 0-1.63-.94l-.36-2.54A.5.5 0 0 0 13.9 2h-3.84a.5.5 0 0 0-.49.42l-.36 2.54a7 7 0 0 0-1.63.94l-2.39-.96a.5.5 0 0 0-.61.22L2.66 8.48a.5.5 0 0 0 .12.64l2.03 1.58a7.07 7.07 0 0 0 0 1.88l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.13.22.39.3.61.22l2.39-.96c.5.39 1.05.7 1.63.94l.36 2.54c.04.24.25.42.49.42h3.84c.24 0 .45-.18.49-.42l.36-2.54a7 7 0 0 0 1.63-.94l2.39.96c.22.08.48 0 .61-.22l1.92-3.32a.5.5 0 0 0-.12-.64zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z"/></svg>
      </button>
      {#if settingsOpen}
        <div class="rv-settings" role="dialog" aria-label="Review settings">
          <div class="row">
            <span>How deep the engine looks</span>
            <div class="seg small">
              {#each ["fast", "standard", "deep"] as s (s)}
                <button class="segbtn" aria-pressed={room?.strength === s} onclick={() => reviewAt(s)}>{className(s)}</button>
              {/each}
            </div>
          </div>
          <div class="hint">Deeper finds more and takes longer. A depth reviewed before opens at once.</div>
          <label class="row"><span>The best move's arrow</span><input type="checkbox" checked={look.arrow} onchange={(e) => setLook({ arrow: e.currentTarget.checked })} /></label>
          <label class="row"><span>Each move's kind on the board</span><input type="checkbox" checked={look.marks} onchange={(e) => setLook({ marks: e.currentTarget.checked })} /></label>
          <label class="row"><span>Time spent on each move</span><input type="checkbox" checked={look.spent} onchange={(e) => setLook({ spent: e.currentTarget.checked })} /></label>
          <div class="row">
            <span>Key moves to stop at</span>
            <div class="seg small">
              {#each [["both", "Both"], ["white", game?.white ?? "White"], ["black", game?.black ?? "Black"]] as [value, name] (value)}
                <button class="segbtn" aria-pressed={look.keys === value} onclick={() => setLook({ keys: value as Look["keys"] })}>{name}</button>
              {/each}
            </div>
          </div>
        </div>
      {/if}
    </div>
  </div>
</header>

<main>
  <section class="rv-left">
    <div class="rv-player">
      <i class="sw {top === 'white' ? 'w' : 'b'}"></i>{game?.[top] ?? (top === "white" ? "White" : "Black")}
      {#if game?.[`${top}_elo`]}<small>({game[`${top}_elo`]})</small>{/if}
      {#if clockOf(top)}<span class="clk">{clockOf(top)}</span>{/if}
    </div>
    <div class="rv-boardrow">
      <EvalBar score={barScore} value={barEval} {bottom} />
      {#if shown}
        <ReviewBoard
          fen={shown.fen}
          uci={shown.uci}
          cls={shown.cls}
          {bottom}
          {arrows}
          {redraw}
          legal={retry && !retry.solved && !retry.busy ? retry.legal : null}
          side={retry?.side ?? "white"}
          wrong={retry?.wrong ?? null}
          onplay={playTry}
        />
      {:else}
        <div class="board-wrap rv-board rv-blank"></div>
      {/if}
    </div>
    <div class="rv-player">
      <i class="sw {bottom === 'white' ? 'w' : 'b'}"></i>{game?.[bottom] ?? (bottom === "white" ? "White" : "Black")}
      {#if game?.[`${bottom}_elo`]}<small>({game[`${bottom}_elo`]})</small>{/if}
      {#if clockOf(bottom)}<span class="clk">{clockOf(bottom)}</span>{/if}
    </div>
  </section>

  <section class="rv-pane" aria-label="Game Review">
    <div class="rv-pane-head">
      {#if view === "moves" && room?.status === "ready"}
        <button class="iconbtn small" title="The summary" aria-label="Back to the summary" onclick={() => { stopLine(); view = "summary"; }}>
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>
        </button>
      {/if}
      Game Review
      {#if room}<span class="strength">{room.strength}</span>{/if}
    </div>

    {#if failure || room?.status === "failed"}
      <div class="rv-scroll"><div class="rv-failed">{failure || room?.error || "The review failed."}</div></div>
    {:else if !room || room.status !== "ready" || !summary}
      <div class="rv-scroll">
        <div class="rv-progress" role="status">
          <div>{room?.status === "analysing" ? "Analysing the game…" : "Opening the game…"}</div>
          <div class="bar"><i style="width: {room && room.progress.total ? (100 * room.progress.done) / room.progress.total : 0}%"></i></div>
          {#if room?.progress.total}<div>{room.progress.done} of {room.progress.total} positions</div>{/if}
        </div>
      </div>
    {:else if view === "summary"}
      <div class="rv-scroll">
        <div class="rv-say"><i class="who"></i><div class="bubble">{verdict(summary, game?.result ?? null)}</div></div>
        <EvalGraph {graph} marks={keyMoments} height={84} onpick={(p) => go(p)} />
        <div class="rv-players">
          {#each ["white", "black"] as const as side (side)}
            <div>
              <div class="name">{summary[side].name ?? className(side)}</div>
              {#if summary[side].rating}<div class="rating">{summary[side].rating}</div>{/if}
              <b class="acc {side === 'white' ? 'w' : 'b'}" title="Accuracy">{summary[side].accuracy?.toFixed(1) ?? "–"}</b>
              {#if afterRetries[side]}
                {@const r = afterRetries[side]!}
                <div class="after" title="Accuracy with your retries in place of the moves played">after retries <b>{r.after.toFixed(1)}</b> <span class="up">(+{(r.after - r.before).toFixed(1)})</span></div>
              {/if}
            </div>
          {/each}
        </div>
        <div class="rv-table">
          {#each unfolded ? CLASSES : FOLDED as c (c)}
            {@const w = summary.white.counts[c] ?? 0}
            {@const b = summary.black.counts[c] ?? 0}
            <span class="n cls-text {c}" class:zero={!w}>{w}</span>
            <span class="kind"><i class="badge {c}"></i>{className(c)}</span>
            <span class="n cls-text {c}" class:zero={!b}>{b}</span>
          {/each}
          <button class="ghost small rv-more" style="grid-column: 1 / -1" onclick={() => (unfolded = !unfolded)}>{unfolded ? "Fewer ▴" : "Every kind ▾"}</button>
          {#if summary.white.performance != null || summary.black.performance != null}
            <span class="sep"></span>
            {@const w = summary.white.performance}
            {@const b = summary.black.performance}
            <span class="n" title={w ? `Somewhere from ${w.low} to ${w.high}: one game says only so much` : ""}>{w?.rating ?? "–"}</span>
            <span class="label" title="The rating the game was played at, read from the moves alone">Game rating</span>
            <span class="n" title={b ? `Somewhere from ${b.low} to ${b.high}: one game says only so much` : ""}>{b?.rating ?? "–"}</span>
          {/if}
          <span class="sep"></span>
          {#each PHASES as phase (phase)}
            {@const w = gradeOf("white", phase)}
            {@const b = gradeOf("black", phase)}
            {#if w || b}
              <span class="grade" title={w ? `${w.accuracy.toFixed(1)} over ${w.moves} moves` : ""}>{#if w}<i class="badge {w.grade}"></i>{:else}–{/if}</span>
              <span class="label">{className(phase)}</span>
              <span class="grade" title={b ? `${b.accuracy.toFixed(1)} over ${b.moves} moves` : ""}>{#if b}<i class="badge {b.grade}"></i>{:else}–{/if}</span>
            {/if}
          {/each}
        </div>
        {#if summary.shape}<div class="rv-shape">Game shape: <b>{summary.shape}</b></div>{/if}
      </div>
      <div class="rv-foot">
        <button class="primary" onclick={() => go(1, true)}>Start Review</button>
      </div>
    {:else}
      <div class="rv-scroll rv-step">
        {#if retry}
          <div class="rv-card rv-retry">
            <div class="head">{retry.solved ? (retry.gaveUp ? "The move" : "Found it") : `Find a better move for ${className(retry.side)}`}</div>
            {#if retry.tried}
              <div class="words">
                {#if retry.tried.label}<i class="badge {retry.tried.label}"></i>{/if}
                <span class="cls-text {retry.tried.label ?? ''}">{retry.gaveUp ? `${retry.tried.san} was the move.` : tryWords(retry.tried)}</span>
              </div>
            {/if}
            {#if retry.busy && !retry.wrong}<div class="words muted" role="status">Checking the move…</div>{/if}
            {#if retry.hintText && !retry.solved}<div class="words hint">{retry.hintText}</div>{/if}
            <div class="rv-actions">
              {#if !retry.solved}
                <button onclick={nextHint} disabled={retry.hint >= 3} title="The coach's hints: the idea, where to look, then the move">{retry.hint ? `Hint ${retry.hint + 1 > 3 ? 3 : retry.hint + 1} of 3` : "Hint"}</button>
                <button onclick={showMove}>Show the move</button>
              {/if}
              <button class={retry.solved ? "primary" : "ghost"} onclick={endRetry}>Back to the game</button>
            </div>
          </div>
        {:else if line}
          <div class="rv-card">
            <div class="head">{line.kind === "best" ? "The best line" : "What follows"}</div>
            <div class="words">
              {#each numbered(line.from, line.steps.map((s) => s.san)) as text, i (i)}<b class="mv" class:dim={i > line.i}>{text}</b>{" "}{/each}
            </div>
            <div class="rv-actions"><button class="ghost" onclick={stopLine}>Back to the game</button></div>
          </div>
        {:else if move}
          <div class="rv-card">
            <div class="head">
              {#if move.label}<i class="badge {move.label}"></i>{/if}
              <span class="cls-text {move.label ?? ''}">{headline(move)}</span>
              <span class="pill {evalSide(move.eval)}">{evalText(move.eval)}</span>
            </div>
            <div class="words">{sentence(move, whiteBefore)}</div>
            <div class="rv-actions">
              {#if BAD.has(move.label ?? "")}
                <button class="primary" onclick={startRetry} title="Find a better move yourself">Retry</button>
              {/if}
              {#if move.best_line.length && move.best_uci !== move.uci}
                <button onclick={() => playLine("best")}>Best: {move.best_san}</button>
              {/if}
              {#if move.follow_line.length}<button onclick={() => playLine("follow")}>Show follow-up</button>{/if}
              <button onclick={nextKey} title="The next move worth a word">Next key move ›</button>
            </div>
          </div>
          {#if ply === moves.length}
            <div class="rv-last">
              <div class="rv-say"><i class="who"></i><div class="bubble">{verdict(summary, game?.result ?? null)}</div></div>
              <div class="rv-last-grid">
                {#each ["white", "black"] as const as side (side)}
                  {@const s = summary[side]}
                  {@const r = afterRetries[side]}
                  <div class="rv-last-side">
                    <div class="name">{s.name ?? className(side)}</div>
                    <div><b class="acc {side === 'white' ? 'w' : 'b'}">{s.accuracy?.toFixed(1) ?? "–"}</b> accuracy</div>
                    {#if r}<div class="after">after retries <b>{r.after.toFixed(1)}</b> <span class="up">(+{(r.after - r.before).toFixed(1)})</span></div>{/if}
                    {#if s.performance}<div title="Somewhere from {s.performance.low} to {s.performance.high}">game rating <b>{s.performance.rating}</b></div>{/if}
                  </div>
                {/each}
              </div>
              {#if improvedMoves.length}
                <div class="rv-table rv-tries">
                  <span></span><span class="label">You played</span><span class="label">New move</span>
                  {#each improvedMoves as m (m.ply)}
                    <span class="muted">{Math.ceil(m.ply / 2)}{m.side === 'white' ? '.' : '...'}</span>
                    <span class="cls-text {m.label ?? ''}"><i class="badge {m.label}"></i> {m.san}</span>
                    <span class="cls-text {retried[m.ply].label ?? ''}"><i class="badge {retried[m.ply].label}"></i> {retried[m.ply].san}</span>
                  {/each}
                </div>
              {/if}
              {#if room?.practice.length}
                <div class="rv-practice">
                  <div class="label">Practise what this game missed</div>
                  {#each room.practice as p (p.motif)}
                    <div class="row">
                      <span>{motifWords(p.motif)} <span class="muted">· missed {p.missed}×</span></span>
                      <span class="chips">
                        {#each p.puzzles as z (z.id)}
                          <a class="chip" href="/puzzle/{encodeURIComponent(z.id)}" target="_blank" rel="noopener" onclick={(e) => { if (inShell()) { e.preventDefault(); openWindow('puzzle/' + encodeURIComponent(z.id)); } }}>puzzle · {z.rating}</a>
                        {/each}
                      </span>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        {:else}
          <div class="rv-say"><i class="who"></i><div class="bubble">{summary.opening ? `The ${openingName(summary.opening)}.` : "The start."} Step through with the arrows, or jump to a key move.</div></div>
        {/if}
        <div class="rv-movebox" bind:this={listEl}><div class="rv-moves">
          {#each rows as r, i (r.n)}
            <span class="num" class:row-odd={i % 2 === 1}>{r.n}.</span>
            {#each [r.white, r.black] as m, j (j)}
              {#if m}
                <span class="mv" class:at={m.ply === ply} class:row-odd={i % 2 === 1} role="button" tabindex="-1" onclick={() => go(m.ply)} onkeydown={(e) => { if (e.key === "Enter") go(m.ply); }}>
                  {#if m.label && KEY.has(m.label)}<i class="badge {m.label}"></i>{/if}
                  <span class={m.label && KEY.has(m.label) ? `cls-text ${m.label}` : ""}>{m.san}</span>
                  {#if look.spent && m.spent_ms != null}<small class="spent">{spentText(m.spent_ms)}</small>{/if}
                </span>
              {:else}
                <span class="mv" class:row-odd={i % 2 === 1}></span>
              {/if}
            {/each}
          {/each}
        </div></div>
      </div>
      <div class="rv-foot">
        <EvalGraph {graph} marks={keyMoments} at={ply} height={44} onpick={(p) => go(p)} />
        <div class="rv-nav">
          <button aria-label="The start" title="The start (Home)" onclick={() => go(0)}><svg viewBox="0 0 24 24"><path d="M18.41 16.59 13.82 12l4.59-4.59L17 6l-6 6 6 6zM6 6h2v12H6z"/></svg></button>
          <button aria-label="Back a move" title="Back (←)" onclick={() => step(-1)}><svg viewBox="0 0 24 24"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg></button>
          <button aria-label="On a move" title="On (→)" onclick={() => step(1)}><svg viewBox="0 0 24 24"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg></button>
          <button aria-label="The end" title="The end (End)" onclick={() => go(moves.length)}><svg viewBox="0 0 24 24"><path d="M5.59 7.41 10.18 12l-4.59 4.59L7 18l6-6-6-6zM16 6h2v12h-2z"/></svg></button>
        </div>
      </div>
    {/if}
  </section>
</main>
