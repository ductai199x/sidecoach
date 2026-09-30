<script lang="ts">
  // The review page (/review/<key>): a game reviewed whole, as chess.com's Game Review goes --
  // the analysis with its progress, then the summary (the coach's line on the game, the graph,
  // each side's accuracy, kinds of move, game rating and phases, the game's shape), then the
  // game move by move: the board with every move's badge on it and the arrows the settings turn
  // on, the eval bar, the coach's bubble on the move, the action bar (Retry, Best, Follow-up,
  // Next), the lines it can play, the move list, the graph.
  // The room lives on the sidecar (ui/review_room.py); this page polls it while it analyses.
  import Badge from "$lib/Badge.svelte";
  import { onMount } from "svelte";
  import { page } from "$app/state";
  import "$lib/styles/review.css";
  import { beacon, connect, review as reviewApi, type AiLink, type AiPiece, type LineStep, type ReviewMove, type ReviewState } from "$lib/api";
  import { inShell, openWindow } from "$lib/shell.svelte";
  import { placement } from "$lib/chess";
  import { boardHandle, layoutMode, splitter } from "$lib/layout";
  import { prefs } from "$lib/panel.svelte";
  import { captureSound, moveSound } from "$lib/sounds";
  import AiText from "$lib/review/AiText.svelte";
  import EvalBar from "$lib/review/EvalBar.svelte";
  import { RetryDesk } from "$lib/review/retry.svelte";
  import EvalGraph from "$lib/review/EvalGraph.svelte";
  import ReviewBoard from "$lib/review/ReviewBoard.svelte";
  import { BAD, CLASSES, FOLDED, KEY, bestWords, className, evalSide, evalText, headline, numbered, openingName, sentence, tryWords, verdict } from "$lib/review/words";
  import { motifWords } from "$lib/chess";

  const key = decodeURIComponent(page.params.key ?? "");
  const api = reviewApi(key);

  let room = $state<ReviewState | null>(null);
  let failure = $state("");
  let view = $state<"summary" | "moves">("summary");
  let ply = $state(0);
  let bottom = $state<"white" | "black">("white");
  let unfolded = $state(false);
  // A line the board shows instead of the game: the best from before the move, or what follows
  // it; `i` is the step shown (-1: the position the line starts from); `open` whether its moves
  // are out (Best shows the one move, its line on asking). `autoplay` while Play runs it.
  let line = $state<{ kind: "best" | "follow"; steps: LineStep[]; from: string; i: number; open: boolean } | null>(null);
  let playing: ReturnType<typeof setTimeout> | null = null;
  let autoplay = $state(false);
  let closed = false;
  // Retry (the plan's D; lib/review/retry): the review's own moves found again. `retry` the
  // one under way, `retried` the best try at each; the best arrow stays hidden meanwhile.
  const desk = new RetryDesk({ position: (p) => api.retry(p), attempt: (p, u) => api.attempt(p, u), hint: (p, l) => api.hint(p, l) }, (fen, uci) => soundOf(fen, uci));
  const retry = $derived(desk.now);
  const retried = $derived(desk.best);

  // ---- the review's settings (the plan's H), kept for the next review ----
  // The arrows are off to begin with (Tai, 2026-09-30: the best move's gives away what Retry
  // asks for); each has its checkbox.
  type Look = { arrow: boolean; ideas: boolean; marks: boolean; keys: "both" | "white" | "black"; spent: boolean };
  const LOOK: Look = { arrow: false, ideas: false, marks: true, keys: "both", spent: false };
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
  // The moves the review stops at: every side's, or the one side the settings name; and the
  // last book move for both, where the game leaves known theory (Tai, 2026-09-30).
  const keyMoments = $derived(
    (summary?.key_moments ?? []).filter((k) => k.kind === "book" || look.keys === "both" || k.side === look.keys),
  );

  // What the board shows: the game at `ply`, a line's step, or the position being retried.
  const shown = $derived.by(() => {
    if (retry) {
      const kept = retry.solved && retry.tried;
      return { fen: kept ? retry.tried!.fen : retry.fen, uci: kept ? retry.tried!.uci : null, cls: kept && look.marks ? retry.tried!.label : null };
    }
    if (line) {
      // Each move of a line is the engine's first choice: the best move's badge.
      const step = line.i >= 0 ? line.steps[line.i] : null;
      return { fen: step ? step.fen : line.from, uci: step ? step.uci : null, cls: step && look.marks ? "best" : null };
    }
    if (!room) return null;
    if (!move) return { fen: room.start_fen ?? "", uci: null, cls: null };
    return { fen: move.fen_after, uci: move.uci, cls: look.marks ? move.label : null };
  });
  // The arrows the settings turn on, on the position the move left: the best move in gold when
  // the move was not it; the reply that punishes a bad move in red, or what a sound move threatens
  // in blue. Retrying, only the third hint's (the move, in gold).
  type Drawn = { from: string; to: string; kind: "best" | "threat" | "idea" | "file"; both?: boolean };
  const arrows = $derived.by(() => {
    const out: Drawn[] = [];
    const add = (uci: string | null | undefined, kind: Drawn["kind"], both = false) => {
      if (uci && !out.some((a) => a.from === uci.slice(0, 2) && a.to === uci.slice(2, 4))) out.push({ from: uci.slice(0, 2), to: uci.slice(2, 4), kind, both });
    };
    if (retry && !retry.solved) {
      if (retry.hintBest) add(retry.hintBest, "best");
      return out;
    }
    if (!move || view !== "moves") return out;
    // What the coach's words name, while it is shown: a move's arrow, a file end to end (the
    // better move's words show on its line and on a retry found too).
    if (shownLink?.arrow) add(shownLink.arrow[0] + shownLink.arrow[1], shownLink.arrow[2]);
    if (shownLink?.span) add(shownLink.span[0] + shownLink.span[1], "file", true);
    if (line || retry) return out;
    if (look.arrow && move.best_uci !== move.uci) add(move.best_uci, "best");
    if (look.ideas) {
      if (BAD.has(move.label ?? "")) add(move.follow_line[0]?.uci, "threat");
      else add(move.idea, "idea");
    }
    return out;
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

  // ---- the AI coach's words: asked for on the summary, a game at a time (ui/review_room.py) ----
  const ai = $derived(room?.ai ?? null);
  let asking = $state(false);
  // Whether the better move is out at a ply: the player asked (What was better?), played Best, or
  // found it (or gave up) in Retry, or the gold arrow shows it. "Better" opens then, on every move
  // (the bubble stays short); on the player's own -- every move when their side is not known --
  // a "why" that names it waits too. The opponent's moves are not theirs to find.
  let revealed = $state<Record<number, boolean>>({});
  function reveal(at: number): void {
    if (!revealed[at]) revealed = { ...revealed, [at]: true };
  }
  $effect(() => {
    if (retry?.solved) reveal(retry.ply);
  });
  const betterOut = (m: ReviewMove): boolean => look.arrow || !!revealed[m.ply];
  function answerOut(m: ReviewMove): boolean {
    const own = !room?.game.user_side || m.side === room.game.user_side;
    return !own || betterOut(m);
  }
  const said = $derived(move && ai?.words ? (ai.words.texts[String(move.ply)] ?? null) : null);
  // Each part of it as its links to the board (the text whole when it has none).
  const saidLinks = $derived(move && ai?.words ? (ai.words.links?.[String(move.ply)] ?? {}) : {});
  const linksOf = (part: "why" | "better" | "lesson"): AiPiece[] => saidLinks[part] ?? [said?.[part] ?? ""];
  // A link shown on the board: the one the pointer is on, else the one clicked (or tapped) till
  // another click; none on another move.
  let hovered = $state<AiLink | null>(null);
  let pinned = $state<AiLink | null>(null);
  const shownLink = $derived(hovered ?? pinned);
  function mark(link: AiLink | null, pin: boolean): void {
    if (pin) pinned = pinned === link ? null : link;
    else hovered = link;
  }
  // None carried to another move, or from the move's card to a line's or a retry's.
  const card = $derived(retry ? "retry" : line ? `line-${line.kind}` : "move");
  $effect(() => {
    void ply;
    void card;
    hovered = pinned = null;
  });
  // The AI's "why" once it may be read: one that names the better move waits for the answer, the
  // review's own sentence standing in.
  const aiWhy = $derived(move && said?.why && (!ai?.words?.reveals.includes(move.ply) || answerOut(move)) ? said.why : null);
  const aiCount = $derived(Object.keys(ai?.words?.texts ?? {}).length);

  /** Dollars as the card says them: "about 2¢", "under 1¢"; spent, to a tenth: "1.7¢". */
  function cents(usd: number, spent = false): string {
    const c = usd * 100;
    if (spent) return `${c < 0.1 ? "<0.1" : c.toFixed(1)}¢`;
    return c < 1 ? "under 1¢" : `about ${Math.round(c)}¢`;
  }

  async function explain(): Promise<void> {
    if (asking || !room) return;
    asking = true;
    try {
      let asked = await api.explain().catch(() => ({ ok: false, error: "Sidecoach is not answering. Try again in a moment." }));
      // The room gone with a sidecar that started again: opened again, and asked once more.
      if (!asked.ok && /not open/.test(asked.error ?? "") && (await reopen())) asked = await api.explain();
      if (!asked.ok && room) room = { ...room, ai: { ...room.ai, status: "failed", error: asked.error } };
      else poll();
    } finally {
      asking = false;
    }
  }

  // ---- Retry ----
  async function startRetry(): Promise<void> {
    if (!move) return;
    stopLine();
    if (await desk.open(move.ply)) return;
    // The room went with a sidecar that started again: opened again, and asked once more.
    if (/not open/.test(desk.error)) {
      const at = move.ply;
      desk.error = "";
      reopening = true;
      const ready = await reopen();
      reopening = false;
      if (!ready) desk.error = "The review could not be opened again. Close it and open it from Your games.";
      else if (move?.ply === at) await desk.open(at);
    }
  }
  let reopening = $state(false);

  // The room opened again on the sidecar, and read back (its stored review, a moment): whether
  // it is ready within REOPEN_MS.
  const REOPEN_MS = 20000;
  async function reopen(): Promise<boolean> {
    if (!room) return false;
    const opened = await api.open(room.strength).catch(() => null);
    if (!opened?.ok) return false;
    let state = opened.state;
    const until = Date.now() + REOPEN_MS;
    while (state?.status !== "ready" && Date.now() < until) {
      if (state?.status === "failed") return false;
      await new Promise((r) => setTimeout(r, 250));
      state = (await api.poll(0).catch(() => null))?.state ?? state;
    }
    return state?.status === "ready";
  }

  // Retry done (found, or shown): the best move's line, its moves out; on its first move at
  // once when that is the move on the board, else that move taken back and the line's played.
  function seeLine(): void {
    const onBoard = desk.now?.tried?.uci;
    desk.close();
    showLine("best", true, onBoard === move?.best_line[0]?.uci);
  }

  // ---- going through the game ----
  function soundOf(fen: string, uci: string): void {
    const board = placement(fen);
    const mover = board.get(uci.slice(0, 2));
    const takes = board.has(uci.slice(2, 4)) || (mover?.toUpperCase() === "P" && uci[0] !== uci[2]);
    if (takes) captureSound(); else moveSound();
  }

  function pause(): void {
    if (playing) clearTimeout(playing);
    playing = null;
    autoplay = false;
  }

  function stopLine(): void {
    pause();
    line = null;
  }

  // To a move of the game: the board goes there (a step on is heard); a line or a retry ends.
  function moveTo(to: number, heard = false): void {
    stopLine();
    if (desk.now) desk.close();
    desk.error = "";
    const next = Math.max(0, Math.min(moves.length, to));
    if (heard && next === ply + 1 && moves[next - 1]) soundOf(moves[next - 1].fen_before, moves[next - 1].uci);
    ply = next;
    view = "moves";
  }

  // The player's own going somewhere stops Next's run.
  function go(to: number, heard = false): void {
    stopRun();
    moveTo(to, heard);
  }

  function toSummary(): void {
    stopRun();
    stopLine();
    if (desk.now) desk.close();
    view = "summary";
  }

  // Next: on to the next move worth a word, the moves between played out on the board: a move
  // every RUN_MS, faster over a long stretch so few take much over RUN_TOTAL_MS; never faster
  // than RUN_FASTEST_MS, so each piece lands (a 200 ms slide) and its badge shows. A third
  // slower than at first (Tai, 2026-09-30: "too fast. make it about 30% slower").
  const RUN_MS = 340;
  const RUN_FASTEST_MS = 220;
  const RUN_TOTAL_MS = 3100;
  let running: ReturnType<typeof setTimeout> | null = null;
  function stopRun(): void {
    if (running) clearTimeout(running);
    running = null;
  }

  function step(delta: number): void {
    if (line) {
      lineStep(delta);
      return;
    }
    go(ply + delta, delta === 1);
  }

  // A line's move shown: -1 the position it starts from. A step past its first move brings its
  // moves out: the arrows go through the line once it is on the board.
  function lineTo(i: number, heard = true): void {
    if (!line) return;
    const to = Math.max(-1, Math.min(line.steps.length - 1, i));
    if (heard && to === line.i + 1 && line.steps[to]) soundOf(to > 0 ? line.steps[to - 1].fen : line.from, line.steps[to].uci);
    line = { ...line, i: to, open: line.open || to > 0 };
  }

  // The player's own step stops Play.
  function lineStep(delta: number): void {
    pause();
    if (line) lineTo(line.i + delta);
  }

  function nextKey(): void {
    stopRun();
    const target = keyMoments.find((k) => k.ply > ply)?.ply ?? moves.length;
    const pace = Math.max(RUN_FASTEST_MS, Math.min(RUN_MS, RUN_TOTAL_MS / Math.max(1, target - ply)));
    const advance = () => {
      running = null;
      if (ply >= target) return;
      moveTo(ply + 1, true);
      if (ply < target) running = setTimeout(advance, pace);
    };
    advance();
  }

  // A line on the board: its first move only, played from where the line starts (the move
  // taken back first, for Best), and nothing more till the player asks -- the arrows or ◀ ▶
  // step it, Play runs it a move every LINE_MS. Best shows the move, its line on asking;
  // Follow-up and a retry's line come with their moves out (Tai, 2026-09-30: a line that ran
  // by itself, fast, on a click could not be read).
  const LINE_MS = 1100;
  const TAKEBACK_MS = 350;
  function showLine(kind: "best" | "follow", open = kind === "follow", played = false): void {
    if (!move) return;
    stopRun();
    stopLine();
    const steps = kind === "best" ? move.best_line : move.follow_line;
    if (!steps.length) return;
    if (kind === "best") reveal(move.ply);
    line = { kind, steps, from: kind === "best" ? move.fen_before : move.fen_after, i: played ? 0 : -1, open };
    if (played) return;
    playing = setTimeout(() => {
      playing = null;
      if (line?.i === -1) lineTo(0);
    }, TAKEBACK_MS);
  }

  // Best and Follow-up press in and out: pressed again, back to the game.
  function toggleLine(kind: "best" | "follow"): void {
    if (line?.kind === kind) stopLine();
    else showLine(kind);
  }

  // A line's start or end, or the game's when no line is shown.
  function toEnd(last: boolean): void {
    if (line) {
      pause();
      lineTo(last ? line.steps.length - 1 : -1, false);
    } else go(last ? moves.length : 0);
  }

  // Play from the move shown (from the line's start when it is at its end), a move every
  // LINE_MS, to its end; Pause, or any step, stops it.
  function playPause(): void {
    if (!line) return;
    if (autoplay) {
      pause();
      return;
    }
    pause();
    line = { ...line, open: true, i: line.i >= line.steps.length - 1 ? -1 : line.i };
    autoplay = true;
    const advance = () => {
      if (!line || line.i >= line.steps.length - 1) {
        playing = null;
        autoplay = false;
        return;
      }
      lineTo(line.i + 1);
      playing = setTimeout(advance, LINE_MS);
    };
    playing = setTimeout(advance, line.i === -1 ? TAKEBACK_MS : 0);
  }

  function onKey(e: KeyboardEvent): void {
    if (!room || room.status !== "ready" || (e.target as HTMLElement)?.tagName === "INPUT") return;
    if (e.key === "ArrowRight") { e.preventDefault(); if (view === "summary") go(1, true); else step(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); step(-1); }
    else if (e.key === "Home" || e.key === "ArrowUp") { e.preventDefault(); toEnd(false); }
    else if (e.key === "End" || e.key === "ArrowDown") { e.preventDefault(); toEnd(true); }
    else if (e.key === " " && line && !(e.target as HTMLElement)?.closest("button, a, input")) { e.preventDefault(); playPause(); }
    else if (e.key === "Escape" && settingsOpen) { e.preventDefault(); settingsOpen = false; }
    else if (e.key === "Escape" && retry) { e.preventDefault(); desk.close(); }
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
    if (!move) return null;
    if (line) return line.kind === "best" ? `${move.best_san} was best. ${said?.better ?? bestWords(move, whiteBefore)}` : null;
    return [`${headline(move)}.`, aiWhy ?? sentence(move, whiteBefore, look.arrow, move.ply === summary?.last_book_ply), said?.lesson].filter(Boolean).join(" ");
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

  // A card read from its top: a new move's (the AI coach's words run long) starts there.
  let stepEl = $state<HTMLDivElement | null>(null);
  $effect(() => {
    void ply;
    if (stepEl) stepEl.scrollTop = 0;
  });

  // The move shown stays in view in the list, as the arrows step past its edge: out of view,
  // or at the very edge, it goes to the middle, so the moves around it show too -- down the
  // list side by side, along the one line stacked.
  let listEl = $state<HTMLDivElement | null>(null);
  $effect(() => {
    void ply;
    const box = listEl;
    const el = listEl?.querySelector(".mv.at") as HTMLElement | null;
    if (!box || !el) return;
    const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
    if (box.scrollWidth > box.clientWidth) {
      if (r.left < b.left + r.width || r.right > b.right - r.width) box.scrollLeft += r.left + r.width / 2 - (b.left + b.width / 2);
    } else if (box.scrollHeight > box.clientHeight) {
      if (r.top < b.top + r.height || r.bottom > b.bottom - r.height) box.scrollTop += r.top + r.height / 2 - (b.top + b.height / 2);
    }
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

  const SIDES = ["white", "black"] as const;
  const PHASES = ["opening", "middlegame", "endgame"] as const;
  function gradeOf(side: "white" | "black", phase: string) {
    return summary?.[side].phases.find((p) => p.phase === phase) ?? null;
  }

  // ---- the room ----
  // Polled while it analyses, and while the AI coach writes; one loop at a time.
  let polling = false;
  async function poll(): Promise<void> {
    if (closed || polling) return;
    polling = true;
    let again = true;
    try {
      const res = await api.poll(0);
      if (res.state) take(res.state);
      if (res.closed && !res.state) { failure = "This review was closed."; again = false; }
      const s = res.state;
      if (s && ((s.status === "ready" && s.ai.status !== "writing") || s.status === "failed" || s.closed)) again = false;
    } catch { /* a poll missed */ }
    polling = false;
    if (again) setTimeout(poll, 400);
  }

  function take(s: ReviewState): void {
    const first = !room || room.status !== "ready";
    room = s;
    if (s.status === "ready" && first) {
      bottom = s.game.user_side ?? "white";
      document.title = `sidecoach · review · ${s.game.white ?? "White"} v ${s.game.black ?? "Black"}`;
      requestAnimationFrame(fitBoard);
      // Opened at a move (Stats' positions: `#ply-12`): straight to it.
      const at = /^#ply-(\d+)$/.exec(location.hash);
      if (at) go(+at[1]);
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

  /** How often, in words: once, twice, 3 times. */
  function timesWords(n: number): string {
    return n === 1 ? "once" : n === 2 ? "twice" : `${n} times`;
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

  // The board's side from the room the main area has, as the panel's: its height less the two
  // player lines, its width less the eval bar, the divider and the pane at its least width
  // (review.css); smaller if the player dragged the divider. Stacked when the window is
  // portrait, or a board beside the pane would be smaller than BOARD_MIN: one screen, the board
  // across the width but leaving the pane STACKED_PANE under it (the player lines and the eval
  // bar STACKED_LINES), the handle under the board sizing it within that. Both sizes are this
  // window's own, kept for the next review.
  const BOARD_MIN = 260;
  const STACKED_PANE = 348;
  const STACKED_LINES = 100;
  const STACKED_MIN = 200;
  // Compact, in a short window or stacked: the foot's controls smaller (review.css main.compact),
  // its graph shorter, so the moves keep room.
  const SHORT_WINDOW = 640;
  let windowHeight = $state(window.innerHeight);
  let stackedNow = $state(false);
  const compact = $derived(windowHeight < SHORT_WINDOW || stackedNow);
  let splitBoard = $state<HTMLElement | null>(null);
  let splitMobile = $state<HTMLElement | null>(null);
  function fitBoard(): void {
    const mainEl = document.querySelector("main") as HTMLElement | null;
    if (!mainEl) return;
    const root = document.documentElement;
    stackedNow = false;
    if (!layoutMode()) {
      const cs = getComputedStyle(mainEl);
      const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
      const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
      const paneMin = parseFloat(getComputedStyle(document.body).getPropertyValue("--rv-pane-min")) || 380;
      const divider = splitBoard?.getBoundingClientRect().width || 14;
      const room = Math.floor(Math.min(mainEl.clientHeight - padY - 2 * 28, mainEl.clientWidth - padX - 30 - divider - paneMin));
      if (room >= BOARD_MIN) {
        root.style.setProperty("--board", Math.max(BOARD_MIN, Math.min(room, prefs.get("reviewBoard") ?? room)) + "px");
        return;
      }
      root.classList.add("stacked");
    }
    stackedNow = true;
    root.style.removeProperty("--board");
    const bar = document.getElementById("bar");
    if (bar) root.style.setProperty("--bar-h", bar.getBoundingClientRect().height + "px");
    const room = Math.floor(Math.min(mainEl.clientWidth - 20, 640, mainEl.clientHeight - STACKED_PANE - STACKED_LINES));
    const side = Math.max(STACKED_MIN, Math.min(room, prefs.get("reviewMboard") ?? room));
    root.style.setProperty("--mboard", side + "px");
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
    // Sent to another move of the game while open (Stats' positions, through the shell).
    const onPlace = () => {
      const at = /^#ply-(\d+)$/.exec(location.hash);
      if (at && room?.status === "ready") go(+at[1]);
    };
    window.addEventListener("hashchange", onPlace);
    const root = document.documentElement;
    const cleanups: (() => void)[] = [];
    if (splitBoard) cleanups.push(splitter(splitBoard, "x", () => parseFloat(getComputedStyle(root).getPropertyValue("--board")) || 0,
      (start, d) => { prefs.set("reviewBoard", Math.max(BOARD_MIN, start + d)); fitBoard(); },
      () => { prefs.set("reviewBoard", null); fitBoard(); }));
    if (splitMobile) cleanups.push(boardHandle(splitMobile,
      (w) => { prefs.set("reviewMboard", w); fitBoard(); },
      () => { prefs.set("reviewMboard", null); fitBoard(); }));
    open();
    return () => {
      for (const c of cleanups) c();
      closed = true;
      stopRun();
      stopLine();
      document.body.classList.remove("review-page");
      window.removeEventListener("resize", fitBoard);
      window.removeEventListener("beforeunload", onUnload);
      window.removeEventListener("hashchange", onPlace);
    };
  });
</script>

{#snippet sides()}
  {#if summary}
    <span class="lbl">Players</span>
    {#each SIDES as side (side)}
      {@const s = summary[side]}
      <span class="{side === 'white' ? 'w' : 'b'} name" title={s.name ?? className(side)}><b>{s.name ?? className(side)}</b>{#if s.rating}<small>{s.rating}</small>{/if}</span>
    {/each}
    <span class="lbl">Accuracy</span>
    {#each SIDES as side (side)}
      <b class="{side === 'white' ? 'w' : 'b'} box">{summary[side].accuracy?.toFixed(1) ?? "–"}</b>
    {/each}
    {#if afterRetries.white || afterRetries.black}
      <span class="lbl sub" title="Accuracy with your retries in place of the moves played">After retries</span>
      {#each SIDES as side (side)}
        {@const r = afterRetries[side]}
        <span class="{side === 'white' ? 'w' : 'b'} after">{#if r}{r.after.toFixed(1)}<em>+{(r.after - r.before).toFixed(1)}</em>{:else}<i class="none">–</i>{/if}</span>
      {/each}
    {/if}
  {/if}
{/snippet}

{#snippet spark()}
  <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 2.5 12.1 8.4 18 10.5 12.1 12.6 10 18.5 7.9 12.6 2 10.5 7.9 8.4z" /><path d="M18.5 14 19.5 16.5 22 17.5 19.5 18.5 18.5 21 17.5 18.5 15 17.5 17.5 16.5z" /></svg>
{/snippet}

{#snippet explainCard()}
  {#if ai && ai.moves > 0}
    <div class="rv-ai" class:busy={ai.status === "writing"} role={ai.status === "writing" ? "status" : undefined}>
      <span class="spark">{@render spark()}</span>
      <div class="txt">
        {#if ai.status === "writing"}
          <b>Going over the key moves…</b>
          <small>The AI coach is writing on {ai.moves} key moves.</small>
        {:else if ai.status === "ready" && ai.words}
          <b>The key moves explained</b>
          <small>By the AI coach, on {aiCount} moves{ai.words.cost != null ? ` · ${cents(ai.words.cost, true)}` : ""}. Step through to read them.</small>
        {:else if ai.status === "failed"}
          <b>The AI coach could not finish</b>
          <small class="err">{ai.error ?? "Something went wrong."}</small>
        {:else}
          <b>Explain the key moves</b>
          <small>{ai.unavailable ?? `The AI coach on ${ai.moves} key moves${ai.estimate != null ? `, ${cents(ai.estimate)}` : ""}.`}</small>
        {/if}
      </div>
      {#if ai.status === "none" || ai.status === "failed"}
        <button class="rv-ask" disabled={!!ai.unavailable || asking} title={ai.unavailable ?? "Ask the AI coach why each key move mattered"} onclick={explain}>{ai.status === "failed" ? "Try again" : "Explain"}</button>
      {/if}
      {#if ai.status === "writing"}<i class="bar"></i>{/if}
    </div>
  {/if}
{/snippet}

{#snippet rating()}
  {#if summary && (summary.white.performance || summary.black.performance)}
    <span class="lbl" title="The rating the game was played at, read from the moves alone">Game rating</span>
    {#each SIDES as side (side)}
      {@const p = summary[side].performance}
      <b class="{side === 'white' ? 'w' : 'b'} box" title={p ? `Somewhere from ${p.low} to ${p.high}: one game says only so much` : ""}>{p?.rating ?? "–"}</b>
    {/each}
  {/if}
{/snippet}

<svelte:window bind:innerHeight={windowHeight} onkeydown={onKey} onpointerdown={(e) => { const at = e.target as HTMLElement; if (settingsOpen && !at.closest(".rv-set")) settingsOpen = false; if (pinned && !at.closest(".lk")) pinned = null; }} />
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
    <button class="iconbtn small" aria-label="Flip the board" title="Flip the board (F)" onclick={() => (bottom = top)}>
      <svg viewBox="0 0 24 24" width="19" height="19" aria-hidden="true"><path fill="currentColor" d="M16 17.01V10h-2v7.01h-3L15 21l4-3.99h-3zM9 3 5 6.99h3V14h2V6.99h3L9 3z"/></svg>
    </button>
    <div class="rv-set">
      <button class="iconbtn small" aria-expanded={settingsOpen} aria-label="Review settings" title="Review settings" onclick={() => (settingsOpen = !settingsOpen)}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M19.14 12.94a7.07 7.07 0 0 0 0-1.88l2.03-1.58a.5.5 0 0 0 .12-.64l-1.92-3.32a.5.5 0 0 0-.61-.22l-2.39.96a7 7 0 0 0-1.63-.94l-.36-2.54A.5.5 0 0 0 13.9 2h-3.84a.5.5 0 0 0-.49.42l-.36 2.54a7 7 0 0 0-1.63.94l-2.39-.96a.5.5 0 0 0-.61.22L2.66 8.48a.5.5 0 0 0 .12.64l2.03 1.58a7.07 7.07 0 0 0 0 1.88l-2.03 1.58a.5.5 0 0 0-.12.64l1.92 3.32c.13.22.39.3.61.22l2.39-.96c.5.39 1.05.7 1.63.94l.36 2.54c.04.24.25.42.49.42h3.84c.24 0 .45-.18.49-.42l.36-2.54a7 7 0 0 0 1.63-.94l2.39.96c.22.08.48 0 .61-.22l1.92-3.32a.5.5 0 0 0-.12-.64zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z"/></svg>
      </button>
      {#if settingsOpen}
        <div class="rv-settings" role="dialog" aria-label="Review settings">
          <section>
            <h3>On the board</h3>
            <label class="rv-toggle">
              <svg class="swatch best" viewBox="0 0 28 14" aria-hidden="true"><path d="M2 5.5h16V1.5L27 7l-9 5.5v-4H2z" /></svg>
              <span class="txt"><b>Best move arrow</b><small>The engine's move in gold. It gives away what Retry asks you to find.</small></span>
              <input type="checkbox" role="switch" checked={look.arrow} onchange={(e) => setLook({ arrow: e.currentTarget.checked })} /><i class="track"></i>
            </label>
            <label class="rv-toggle">
              <svg class="swatch ideas" viewBox="0 0 28 14" aria-hidden="true"><path class="threat" d="M2 2.1h16V0l7 3.1-7 3.1V4.1H2z" /><path class="idea" d="M2 9.9h16V7.8l7 3.1-7 3.1v-2.1H2z" /></svg>
              <span class="txt"><b>Threat and idea arrows</b><small>Red: the reply a mistake allows. Blue: what a sound move threatens next.</small></span>
              <input type="checkbox" role="switch" checked={look.ideas} onchange={(e) => setLook({ ideas: e.currentTarget.checked })} /><i class="track"></i>
            </label>
            <label class="rv-toggle">
              <span class="swatch"><Badge kind="best" size={16} /></span>
              <span class="txt"><b>Move badges</b><small>Each move's kind on the square it went to.</small></span>
              <input type="checkbox" role="switch" checked={look.marks} onchange={(e) => setLook({ marks: e.currentTarget.checked })} /><i class="track"></i>
            </label>
          </section>
          <section>
            <h3>The moves</h3>
            <label class="rv-toggle">
              <svg class="swatch clock" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zm0 18a8 8 0 1 1 0-16 8 8 0 0 1 0 16zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z" /></svg>
              <span class="txt"><b>Time spent</b><small>How long each move took, in the move list.</small></span>
              <input type="checkbox" role="switch" checked={look.spent} onchange={(e) => setLook({ spent: e.currentTarget.checked })} /><i class="track"></i>
            </label>
            <div class="rv-pick">
              <span>Next stops at the key moves of</span>
              <div class="seg small">
                {#each [["both", "Both"], ["white", game?.white ?? "White"], ["black", game?.black ?? "Black"]] as [value, name] (value)}
                  <button class="segbtn" aria-pressed={look.keys === value} onclick={() => setLook({ keys: value as Look["keys"] })}>{name}</button>
                {/each}
              </div>
            </div>
          </section>
          <section>
            <h3>The engine</h3>
            <div class="rv-pick">
              <span>How deep it looks</span>
              <div class="seg small">
                {#each ["fast", "standard", "deep"] as s (s)}
                  <button class="segbtn" aria-pressed={room?.strength === s} onclick={() => reviewAt(s)}>{className(s)}</button>
                {/each}
              </div>
            </div>
            <small class="note">Deeper finds more and takes longer. A depth reviewed before opens at once.</small>
          </section>
        </div>
      {/if}
    </div>
  </div>
</header>

<main class:summary={view === "summary"} class:compact>
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
          redraw={desk.redraw}
          spots={!line && !retry ? (shownLink?.squares ?? []) : []}
          trail={!line && !retry && shownLink?.arrow ? [shownLink.arrow[0], shownLink.arrow[1]] : null}
          legal={retry && !retry.solved && !retry.busy ? retry.legal : null}
          side={retry?.side ?? "white"}
          wrong={retry?.wrong ?? null}
          onplay={desk.play}
        />
      {:else}
        <div class="board-wrap rv-board rv-blank"></div>
      {/if}
    </div>
    <div class="msplit" bind:this={splitMobile} role="separator" aria-orientation="horizontal" aria-label="Board size" title="Drag to resize the board; double-tap to reset"></div>
    <div class="rv-player">
      <i class="sw {bottom === 'white' ? 'w' : 'b'}"></i>{game?.[bottom] ?? (bottom === "white" ? "White" : "Black")}
      {#if game?.[`${bottom}_elo`]}<small>({game[`${bottom}_elo`]})</small>{/if}
      {#if clockOf(bottom)}<span class="clk">{clockOf(bottom)}</span>{/if}
    </div>
  </section>

  <div class="vsplit" bind:this={splitBoard} role="separator" aria-orientation="vertical" aria-label="Board size" title="Drag to resize the board; double-click to reset"></div>

  <section class="rv-pane" aria-label="Game Review">
    <div class="rv-pane-head">
      {#if view === "moves" && room?.status === "ready"}
        <button class="iconbtn small" title="The summary" aria-label="Back to the summary" onclick={toSummary}>
          <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true"><path fill="currentColor" d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>
        </button>
      {/if}
      Game Review
      {#if room}<span class="strength" title="How deep the engine looked: the settings change it">{className(room.strength)} depth</span>{/if}
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
        <div class="rv-coach"><i class="who"></i><div class="rv-bubble"><p class="first">{verdict(summary, game?.result ?? null)}</p></div></div>
        {@render explainCard()}
        <EvalGraph {graph} marks={keyMoments} height={80} onpick={(p) => go(p)} />
        <div class="rv-sum">
          {@render sides()}
          <span class="rule"></span>
          {#each unfolded ? CLASSES : FOLDED as c (c)}
            <span class="lbl">{className(c)}</span>
            <b class="w count cls-text {c}">{summary.white.counts[c] ?? 0}</b>
            <span class="mid"><Badge kind={c} size={24} /></span>
            <b class="b count cls-text {c}">{summary.black.counts[c] ?? 0}</b>
          {/each}
          <button class="fold" aria-expanded={unfolded} aria-label={unfolded ? "Fewer kinds of move" : "Every kind of move"} title={unfolded ? "Fewer kinds" : "Every kind"} onclick={() => (unfolded = !unfolded)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z"/></svg></button>
          <span class="rule"></span>
          {@render rating()}
          {#each PHASES as phase (phase)}
            {@const w = gradeOf("white", phase)}
            {@const b = gradeOf("black", phase)}
            {#if w || b}
              <span class="lbl">{className(phase)}</span>
              <span class="w" title={w ? `${w.accuracy.toFixed(1)} over ${w.moves} moves` : ""}>{#if w}<Badge kind={w.grade} size={24} />{:else}<i class="none">–</i>{/if}</span>
              <span class="b" title={b ? `${b.accuracy.toFixed(1)} over ${b.moves} moves` : ""}>{#if b}<Badge kind={b.grade} size={24} />{:else}<i class="none">–</i>{/if}</span>
            {/if}
          {/each}
          {#if summary.shape}
            <span class="lbl" title="How the game went: the swings of the graph">Game shape</span>
            <span class="shape">{summary.shape}</span>
          {/if}
        </div>
      </div>
      <div class="rv-foot">
        <button class="rv-go" onclick={() => go(1, true)}>Start Review</button>
      </div>
    {:else}
      <div class="rv-scroll rv-step" bind:this={stepEl} class:ended={ply === moves.length && !retry && !line} class:long={!!said && !retry && !line}>
        {#if retry}
          <div class="rv-coach">
            <i class="who"></i>
            <div class="rv-bubble">
              <div class="rv-bubble-head">
                {#if retry.tried}<Badge kind={retry.tried.label} size={22} />{/if}
                <span>{retry.tried ? (retry.gaveUp ? `${retry.tried.san} was the move` : tryWords(retry.tried)) : `Find a better move for ${className(retry.side)}`}</span>
              </div>
              {#if !retry.tried && !retry.busy}<p>Play it on the board. Hint when you are stuck; Show gives the move.</p>{/if}
              {#if retry.busy && !retry.wrong}<p class="muted" role="status">Checking the move…</p>{/if}
              {#if retry.hintText && !retry.solved}<p class="hint">{retry.hintText}</p>{/if}
              {#if retry.solved && move}
                <!-- The teaching moment: why the best move works, the AI coach's words when asked
                     for (headed with the move when another was found), else the review's own. -->
                {#if said?.better && retry.tried?.uci !== move.best_uci}
                  <div class="rv-better">
                    <span class="head"><Badge kind="best" size={18} />{move.best_san} was the engine's first choice</span>
                    <p><AiText pieces={linksOf("better")} shown={shownLink} onmark={mark} /></p>
                  </div>
                {:else if said?.better}
                  <p><AiText pieces={linksOf("better")} shown={shownLink} onmark={mark} /></p>
                {:else}
                  <p>{retry.tried?.uci !== move.best_uci ? `${move.best_san} was the engine's first choice. ` : ""}{bestWords(move, whiteBefore)}</p>
                {/if}
                <div class="rv-bubble-foot">
                  {#if move.best_line.length > 1}<button class="rv-reveal" title="The best move's line, to step through or play" onclick={seeLine}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>See the line · {move.best_line.length} moves</button>{/if}
                  {#if said?.better}<span class="rv-ai-mark" title="Written by the AI coach ({ai?.words?.model})">{@render spark()}AI coach</span>{/if}
                </div>
              {/if}
              {#if desk.error}<p class="rv-err" role="alert">{desk.error}</p>{/if}
            </div>
          </div>
        {:else if line && move}
          <div class="rv-coach">
            <i class="who"></i>
            <div class="rv-bubble">
              <div class="rv-bubble-head">
                {#if line.kind === "best"}<Badge kind="best" size={22} />{/if}
                <span>{line.kind === "best" ? `${move.best_san ?? "The best move"} was best` : `What follows ${move.san}`}</span>
                <span class="rv-line-nav">
                  <button class="iconbtn shut" aria-label="Back to the game" title="Back to the game (Esc)" onclick={stopLine}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg></button>
                </span>
              </div>
              {#if line.kind === "best"}
                {#if said?.better}
                  <p><AiText pieces={linksOf("better")} shown={shownLink} onmark={mark} /></p>
                {:else}
                  <p>{bestWords(move, whiteBefore)}</p>
                {/if}
              {:else}
                <p>{BAD.has(move.label ?? "") ? "The reply it allows, and how it goes on." : "How the game could go on from here."}</p>
              {/if}
              {#if line.open}
                <p class="rv-line">
                  {#each numbered(line.from, line.steps.map((s) => s.san)) as t, i (i)}<button class="mv" class:at={i === line.i} class:dim={i > line.i} onclick={() => { pause(); lineTo(i, false); }}>{t}</button>{" "}{/each}
                </p>
                <div class="rv-line-play">
                  <span class="pair">
                    <button class="iconbtn" aria-label="Back a move of the line" title="Back (←)" disabled={line.i < 0} onclick={() => lineStep(-1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg></button>
                    <button class="iconbtn play" aria-label={autoplay ? "Pause the line" : "Play the line"} title={autoplay ? "Pause (Space)" : line.i >= line.steps.length - 1 ? "Play it again (Space)" : "Play (Space)"} onclick={playPause}>
                      {#if autoplay}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
                      {:else if line.i >= line.steps.length - 1}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5V1L7 6l5 5V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8z"/></svg>
                      {:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>{/if}
                    </button>
                    <button class="iconbtn" aria-label="On a move of the line" title="On (→)" disabled={line.i >= line.steps.length - 1} onclick={() => lineStep(1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg></button>
                  </span>
                  <span class="where" aria-live="polite">{line.i < 0 ? "Before the line" : `Move ${line.i + 1} of ${line.steps.length}`}</span>
                </div>
              {/if}
              {#if (!line.open && line.steps.length > 1) || (said?.better && line.kind === "best")}
                <div class="rv-bubble-foot">
                  {#if !line.open && line.steps.length > 1}
                    <button class="rv-reveal" title="The moves that follow it, to step through or play" onclick={() => line && (line = { ...line, open: true })}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>Show the line · {line.steps.length} moves</button>
                  {/if}
                  {#if said?.better && line.kind === "best"}<span class="rv-ai-mark" title="Written by the AI coach ({ai?.words?.model})">{@render spark()}AI coach</span>{/if}
                </div>
              {/if}
            </div>
          </div>
        {:else if move}
          <div class="rv-coach">
            <i class="who"></i>
            <div class="rv-bubble">
              <div class="rv-bubble-head">
                <Badge kind={move.label} size={22} />
                <span>{headline(move)}</span>
                <span class="pill {evalSide(move.eval)}">{evalText(move.eval)}</span>
              </div>
              {#if aiWhy}
                <p><AiText pieces={linksOf("why")} shown={shownLink} onmark={mark} /></p>
              {:else}
                <p>{sentence(move, whiteBefore, look.arrow, move.ply === summary?.last_book_ply)}</p>
              {/if}
              {#if said?.lesson}<p class="rv-lesson"><b>Lesson</b><AiText pieces={linksOf("lesson")} shown={shownLink} onmark={mark} /></p>{/if}
              {#if said?.better && betterOut(move)}
                <div class="rv-better">
                  <span class="head"><Badge kind="best" size={18} />{move.best_san ?? "The better move"} was better</span>
                  <p><AiText pieces={linksOf("better")} shown={shownLink} onmark={mark} /></p>
                </div>
              {/if}
              {#if reopening}<p class="muted" role="status">Opening the review again…</p>{/if}
              {#if desk.error}<p class="rv-err" role="alert">{desk.error}</p>{/if}
              {#if said}
                <div class="rv-bubble-foot">
                  {#if said.better && !betterOut(move)}
                    <button class="rv-reveal" onclick={() => reveal(move!.ply)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"/></svg>What was better?</button>
                  {/if}
                  <span class="rv-ai-mark" title="Written by the AI coach ({ai?.words?.model})">{@render spark()}AI coach</span>
                </div>
              {/if}
            </div>
          </div>
          {#if ply === moves.length}
            <div class="rv-last">
              <div class="rv-end">
                <span class="head">Game over{game?.result && game.result !== "*" ? ` · ${game.result.replaceAll("1/2", "½")}` : ""}</span>
                <p>{verdict(summary, game?.result ?? null)}</p>
              </div>
              <div class="rv-sum">
                {@render sides()}
                {@render rating()}
              </div>
              {#if improvedMoves.length}
                <div class="rv-tries">
                  <span></span><span class="head">You played</span><span class="head">New move</span>
                  {#each improvedMoves as m (m.ply)}
                    <span class="muted">{Math.ceil(m.ply / 2)}{m.side === 'white' ? '.' : '...'}</span>
                    <span class="cls-text {m.label ?? ''}"><Badge kind={m.label} /> {m.san}</span>
                    <span class="cls-text {retried[m.ply].label ?? ''}"><Badge kind={retried[m.ply].label} /> {retried[m.ply].san}</span>
                  {/each}
                </div>
              {/if}
              {#if room?.practice.length}
                <div class="rv-practice">
                  <span class="head">Practise what this game missed</span>
                  {#each room.practice as p (p.motif)}
                    <div class="row">
                      <span><span class="motif">{motifWords(p.motif)}</span> <span class="muted">· missed {timesWords(p.missed)}</span></span>
                      <span class="chips">
                        {#each p.puzzles as z (z.id)}
                          <a class="chip" title="A {motifWords(p.motif)} puzzle, rated {z.rating}" href="/puzzle/{encodeURIComponent(z.id)}" target="_blank" rel="noopener" onclick={(e) => { if (inShell()) { e.preventDefault(); openWindow('puzzle/' + encodeURIComponent(z.id)); } }}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 11H19V7a2 2 0 0 0-2-2h-4V3.5a2.5 2.5 0 0 0-5 0V5H4a2 2 0 0 0-2 2v3.8h1.5a2.7 2.7 0 0 1 0 5.4H2V20a2 2 0 0 0 2 2h3.8v-1.5a2.7 2.7 0 0 1 5.4 0V22H17a2 2 0 0 0 2-2v-4h1.5a2.5 2.5 0 0 0 0-5z"/></svg>{z.rating}</a>
                        {/each}
                      </span>
                    </div>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
        {:else}
          <div class="rv-coach">
            <i class="who"></i>
            <div class="rv-bubble"><p class="first">{summary.opening ? `The ${openingName(summary.opening)}.` : "The start."} Step through with the arrows, or press Next for the next key move.</p></div>
          </div>
        {/if}
        <div class="rv-movebox" bind:this={listEl}>
          {#each rows as r (r.n)}
            <div class="rv-row">
              <span class="num">{r.n}.</span>
              {#each [r.white, r.black] as m, j (j)}
                {#if m}
                  <span class="cell">
                    <span class="mv" class:at={m.ply === ply} role="button" tabindex="-1" onclick={() => go(m.ply)} onkeydown={(e) => { if (e.key === "Enter") go(m.ply); }}>
                      {#if m.label && KEY.has(m.label)}<Badge kind={m.label} size={16} />{/if}
                      <span class={m.label && KEY.has(m.label) ? `cls-text ${m.label}` : ""}>{m.san}</span>
                    </span>
                    {#if look.spent && m.spent_ms != null}<small class="spent">{spentText(m.spent_ms)}</small>{/if}
                  </span>
                {:else}
                  <span class="cell"></span>
                {/if}
              {/each}
            </div>
          {/each}
        </div>
      </div>
      <div class="rv-foot">
        <div class="rv-bar">
          {#if retry}
            <button class="rv-tool" disabled={retry.solved || retry.hint >= 3} title="The coach's hints: the idea, where to look, then the move" onclick={() => desk.hint()}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 21c0 .55.45 1 1 1h4c.55 0 1-.45 1-1v-1H9v1zm3-19C8.14 2 5 5.14 5 9c0 2.38 1.19 4.47 3 5.74V17c0 .55.45 1 1 1h6c.55 0 1-.45 1-1v-2.26c1.81-1.27 3-3.36 3-5.74 0-3.86-3.14-7-7-7z"/></svg><span>{retry.hint && !retry.solved ? `Hint ${Math.min(3, retry.hint + 1)}/3` : "Hint"}</span></button>
            <button class="rv-tool" disabled={retry.solved} title="Show the move" onclick={() => desk.giveUp()}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z"/></svg><span>Show</span></button>
            <button class="rv-tool" title="Back to the game (Esc)" onclick={() => desk.close()}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg><span>Back</span></button>
          {:else}
            <button class="rv-tool" disabled={!move || !BAD.has(move.label ?? "")} title="Find a better move yourself" onclick={startRetry}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4z"/></svg><span>Retry</span></button>
            <button class="rv-tool" aria-pressed={line?.kind === "best"} disabled={!move || !move.best_line.length || move.best_uci === move.uci} title={line?.kind === "best" ? "Back to the game" : "The best move; its line on asking"} onclick={() => toggleLine("best")}><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.2" cy="10.2" r="6.4" fill="none" stroke="currentColor" stroke-width="2.3"/><path d="M15 15l5.2 5.2" stroke="currentColor" stroke-width="2.7" stroke-linecap="round"/><path d="M10.2 6.9l.95 2.3 2.5.2-1.9 1.63.6 2.44-2.15-1.33-2.15 1.33.6-2.44-1.9-1.63 2.5-.2z"/></svg><span>Best</span></button>
            <button class="rv-tool" aria-pressed={line?.kind === "follow"} disabled={!move || !move.follow_line.length} title={line?.kind === "follow" ? "Back to the game" : "What follows the move played"} onclick={() => toggleLine("follow")}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.5 18l8-6-8-6zM13 18l8-6-8-6z"/></svg><span>Follow-up</span></button>
          {/if}
          <button class="rv-go" disabled={ply >= moves.length && !retry} title="On to the next key move, the moves between played out" onclick={nextKey}>Next<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg></button>
        </div>
        <EvalGraph {graph} marks={keyMoments} at={ply} height={compact ? 30 : 44} onpick={(p) => go(p)} />
        <div class="rv-nav">
          <button aria-label={line ? "The line's start" : "The start"} title={line ? "The line's start (Home)" : "The start (Home)"} onclick={() => toEnd(false)}><svg viewBox="0 0 24 24"><path d="M18.41 16.59 13.82 12l4.59-4.59L17 6l-6 6 6 6zM6 6h2v12H6z"/></svg></button>
          <button aria-label="Back a move" title="Back (←)" onclick={() => step(-1)}><svg viewBox="0 0 24 24"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg></button>
          <button aria-label="On a move" title="On (→)" onclick={() => step(1)}><svg viewBox="0 0 24 24"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg></button>
          <button aria-label={line ? "The line's end" : "The end"} title={line ? "The line's end (End)" : "The end (End)"} onclick={() => toEnd(true)}><svg viewBox="0 0 24 24"><path d="M5.59 7.41 10.18 12l-4.59 4.59L7 18l6-6-6-6zM16 6h2v12h-2z"/></svg></button>
        </div>
      </div>
    {/if}
  </section>
</main>
