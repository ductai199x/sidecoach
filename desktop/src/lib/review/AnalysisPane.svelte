<script lang="ts">
  // The review's Analysis (docs/game-review-plan.md, F): the engine's best lines for the board's
  // position, each move of them played on the board at a click; then the moves -- the game's,
  // with the player's own lines off it, any of them gone back to -- what the player's games went
  // on with from the position (the explorer), the game's details and its PGN to copy or keep,
  // and its charts: the score, the time each move took, the themes.
  import Badge from "$lib/Badge.svelte";
  import { motifWords } from "$lib/chess";
  import type { Explored, ReviewMove, ReviewState } from "$lib/api";
  import type { AnalysisDesk, Node } from "./analysis.svelte";
  import EvalGraph from "./EvalGraph.svelte";
  import { KEY, className, evalSide, evalText, numbered, openingName } from "./words";

  let {
    desk,
    room,
    graph,
    marks,
    savePgn,
  }: {
    desk: AnalysisDesk;
    room: ReviewState;
    graph: number[];
    marks: { ply: number; kind: string }[];
    /** The game as a PGN with the board's lines: copied (`save` false) or kept in Downloads. */
    savePgn: (annotated: boolean, save: boolean) => Promise<{ ok: boolean; error?: string; pgn?: string; saved?: string }>;
  } = $props();

  type Tab = "moves" | "explorer" | "details" | "charts";
  const TABS: [Tab, string][] = [["moves", "Moves"], ["explorer", "Explorer"], ["details", "Details"], ["charts", "Charts"]];
  let tab = $state<Tab>("moves");

  const node = $derived(desk.node);
  // The engine's lines when they are this position's; while it thinks, none.
  const lines = $derived(desk.lines && node && desk.lines.fen === node.fen ? desk.lines : null);

  // ---- the moves, as a PGN reads: each move, then the lines played instead of it in brackets,
  // then the line goes on ----
  type Tok = { kind: "move"; id: number; num: string; san: string; depth: number; game: boolean; label: string | null } | { kind: "open"; id: number; depth: number; mine: boolean } | { kind: "close"; depth: number };
  const toks = $derived.by(() => {
    const nodes = desk.nodes;
    const out: Tok[] = [];
    const walk = (start: number, depth: number, number: boolean) => {
      let id = start;
      let force = number;
      for (;;) {
        const n = nodes[id];
        const before = nodes[n.parent ?? 0];
        const [, side, , , , full] = before.fen.split(" ");
        const white = side !== "b";
        const num = white ? `${full}.` : force ? `${full}...` : "";
        out.push({ kind: "move", id, num, san: n.san, depth, game: n.game, label: n.label });
        force = false;
        if (before.children[0] === id) {
          for (const sib of before.children.slice(1)) {
            out.push({ kind: "open", id: sib, depth: depth + 1, mine: !nodes[sib].game });
            walk(sib, depth + 1, true);
            out.push({ kind: "close", depth: depth + 1 });
            force = true;
          }
        }
        if (!n.children.length) break;
        id = n.children[0];
      }
    };
    if (nodes[0]?.children.length) walk(nodes[0].children[0], 0, true);
    return out;
  });

  // The move shown stays in view in the list.
  let listEl = $state<HTMLDivElement | null>(null);
  $effect(() => {
    void desk.at;
    const el = listEl?.querySelector<HTMLElement>(".an-mv.at");
    el?.scrollIntoView({ block: "nearest" });
  });

  // ---- the explorer ----
  const explored = $derived<Explored | null>(desk.explored);
  const pctOf = (n: number, of: number) => (of ? (100 * n) / of : 0);
  const when = (t: number) => new Date(t * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });

  // ---- details: the game, its PGN ----
  const game = $derived(room.game);
  let annotated = $state(true);
  let said = $state("");
  let saidBad = $state(false);
  let busy = $state(false);
  async function copy(): Promise<void> {
    busy = true;
    const got = await savePgn(annotated, false).catch(() => null);
    busy = false;
    if (!got?.ok || !got.pgn) return void ((said = got?.error || "Sidecoach is not answering."), (saidBad = true));
    try {
      await navigator.clipboard.writeText(got.pgn);
      said = "Copied: paste it anywhere a PGN goes.";
      saidBad = false;
    } catch {
      said = "The PGN could not be copied here; Save it instead.";
      saidBad = true;
    }
  }
  async function keep(): Promise<void> {
    busy = true;
    const got = await savePgn(annotated, true).catch(() => null);
    busy = false;
    said = got?.ok && got.saved ? `Saved to ${got.saved}` : got?.error || "Sidecoach is not answering.";
    saidBad = !(got?.ok && got.saved);
  }
  const resultWords = (r: string | null) => (r === "1-0" ? "White won" : r === "0-1" ? "Black won" : r === "1/2-1/2" ? "Drawn" : "—");

  // ---- charts: the time each move took, the themes of each side's moves ----
  const moves = $derived(room.moves);
  const spentMax = $derived(Math.max(1, ...moves.map((m) => m.spent_ms ?? 0)));
  const hasTime = $derived(moves.some((m) => m.spent_ms != null));
  const seconds = (ms: number) => (ms < 10000 ? (ms / 1000).toFixed(1) : Math.round(ms / 1000).toString()) + " s";
  // The themes are what the moves create and what their lines do (motifs.TAG_ORDER's B and C),
  // not what a move merely is (its A: a capture, a check, a quiet move).
  const MOVE_KINDS = new Set(["capture", "enPassant", "promotion", "underPromotion", "castling", "advancedPawn", "check", "doubleCheck", "quietMove", "defensiveMove"]);
  const themes = $derived.by(() => {
    const count = (side: "white" | "black") => {
      const c: Record<string, number> = {};
      for (const m of moves) if (m.side === side) for (const t of m.tags) if (!MOVE_KINDS.has(t)) c[t] = (c[t] ?? 0) + 1;
      return c;
    };
    const w = count("white"), b = count("black");
    return [...new Set([...Object.keys(w), ...Object.keys(b)])]
      .map((t) => ({ t, w: w[t] ?? 0, b: b[t] ?? 0 }))
      .sort((x, y) => y.w + y.b - (x.w + x.b) || x.t.localeCompare(y.t))
      .slice(0, 12);
  });
  const gameAt = $derived(node?.game ? node.ply : null);
  const moveText = (m: ReviewMove) => `${Math.ceil(m.ply / 2)}${m.side === "white" ? "." : "..."} ${m.san}`;
</script>

<div class="an-engine">
  <label class="an-switch" title="The engine's best lines for the position on the board">
    <input type="checkbox" checked={desk.engineOn} onchange={(e) => desk.setEngine((e.currentTarget as HTMLInputElement).checked)} />
    <span>Engine</span>
  </label>
  <span class="an-depth">{#if !desk.engineOn}off{:else if desk.thinking && !lines}thinking…{:else if lines}depth {lines.depth}{/if}</span>
</div>
{#if desk.engineOn}
  <div class="an-lines" aria-live="polite">
    {#if lines && node}
      {#each lines.lines as l, i (i)}
        <div class="an-line">
          <span class="pill {evalSide({ ...l.eval, result: null })}">{evalText({ ...l.eval, result: null })}</span>
          <span class="an-steps">
            {#each numbered(node.fen, l.steps.map((s) => s.san)) as t, j (j)}<button class="an-step" title="Play the line to here" onclick={() => desk.playLine(l.steps, j)}>{t}</button>{" "}{/each}
          </span>
        </div>
      {:else}
        <div class="an-line muted">{desk.here?.over ? `The game is over: ${desk.here.over.replaceAll("1/2", "½")}.` : "No move here."}</div>
      {/each}
    {:else}
      {#each [0, 1, 2] as i (i)}<div class="an-line ghost"></div>{/each}
    {/if}
  </div>
{/if}

<div class="an-tabs" role="tablist">
  {#each TABS as [id, name] (id)}
    <button role="tab" aria-selected={tab === id} class:on={tab === id} onclick={() => (tab = id)}>{name}</button>
  {/each}
</div>

<div class="rv-scroll an-body">
  {#if desk.error}<p class="rv-err" role="alert">{desk.error}</p>{/if}
  {#if tab === "moves"}
    <div class="an-moves" bind:this={listEl}>
      {#each toks as t, i (i)}
        {#if t.kind === "move"}
          <span class="an-mv" class:at={t.id === desk.at} class:side={t.depth > 0} class:mine={!t.game} role="button" tabindex="-1" onclick={() => desk.go(t.id)} onkeydown={(e) => { if (e.key === "Enter") desk.go(t.id); }}>
            {#if t.num}<span class="an-num">{t.num}</span>{/if}{#if t.game && t.label && KEY.has(t.label)}<Badge kind={t.label} size={14} />{/if}{t.san}
          </span>{" "}
        {:else if t.kind === "open"}
          <span class="an-par" class:side={true}>(</span>{#if t.mine}<button class="an-del" title="Take this line out" aria-label="Take this line out" onclick={() => desk.remove(t.id)}>×</button>{/if}
        {:else}
          <span class="an-par side">)</span>{" "}
        {/if}
      {:else}
        <p class="muted">No moves.</p>
      {/each}
    </div>
    <p class="an-tip muted">Play any move on the board: it becomes a line of yours here, kept with the PGN.</p>
  {:else if tab === "explorer"}
    {#if explored}
      <div class="an-opening">{explored.opening ? openingName(explored.opening) : node?.ply === 0 ? "The start position" : "Out of the opening book"}</div>
      {#if explored.moves.length}
        <div class="an-explore">
          <span class="head">Move</span><span class="head">Games</span><span class="head">Your results</span>
          {#each explored.moves as m (m.uci)}
            {@const known = m.won + m.drawn + m.lost}
            <button class="an-xmv" title="Play {m.san}" onclick={() => desk.play(m.uci)}>{m.san}</button>
            <span class="an-n" title={m.last ? `Last played ${when(m.last.played_at)}` : ""}>{m.games}</span>
            <span class="an-wdl" title="{m.won} won, {m.drawn} drawn, {m.lost} lost{known < m.games ? ` (${m.games - known} with your side not known)` : ''}">
              {#if known}
                <i class="w" style:width="{pctOf(m.won, known)}%">{pctOf(m.won, known) >= 18 ? `${Math.round(pctOf(m.won, known))}%` : ""}</i><i class="d" style:width="{pctOf(m.drawn, known)}%"></i><i class="l" style:width="{pctOf(m.lost, known)}%">{pctOf(m.lost, known) >= 18 ? `${Math.round(pctOf(m.lost, known))}%` : ""}</i>
              {:else}<i class="d" style:width="100%"></i>{/if}
            </span>
          {/each}
        </div>
        <p class="an-tip muted">From {explored.games} of your games that reached this position, this one included.</p>
      {:else}
        <p class="muted">None of your games went on from this position.</p>
      {/if}
    {:else}
      <p class="muted">Looking through your games…</p>
    {/if}
  {:else if tab === "details"}
    <div class="an-details">
      <span class="k">White</span><span>{game.white ?? "White"}{game.white_elo ? ` (${game.white_elo})` : ""}</span>
      <span class="k">Black</span><span>{game.black ?? "Black"}{game.black_elo ? ` (${game.black_elo})` : ""}</span>
      <span class="k">Result</span><span>{game.result && game.result !== "*" ? `${game.result.replaceAll("1/2", "½")} · ${resultWords(game.result)}` : "—"}</span>
      <span class="k">Played</span><span>{game.played_at ? when(game.played_at) : "—"}</span>
      <span class="k">Time</span><span>{game.time_class ? className(game.time_class) : "—"}</span>
      <span class="k">Opening</span><span>{room.summary?.opening ? openingName(room.summary.opening) : "—"}</span>
      <span class="k">Source</span><span>{game.source === "chesscom" ? "chess.com" : game.source === "lichess" ? "Lichess" : game.source === "watched" ? "Watched by Sidecoach" : "PGN"}</span>
    </div>
    <label class="an-check"><input type="checkbox" bind:checked={annotated} /> With the review's notes: each move's kind, the evaluation, the AI coach's words</label>
    <div class="an-actions">
      <button class="rv-tool" disabled={busy} onclick={copy}>Copy PGN</button>
      <button class="rv-tool" disabled={busy} onclick={keep}>Save PGN</button>
    </div>
    {#if said}<p class={saidBad ? "rv-err" : "muted"} role="status">{said}</p>{/if}
    <p class="an-tip muted">The PGN has the game, your lines from the board, and the clocks.</p>
  {:else}
    <div class="an-chart">
      <span class="head">Score</span>
      <EvalGraph {graph} {marks} at={gameAt} height={70} onpick={(p) => desk.go(desk.gameNode(p))} />
    </div>
    <div class="an-chart">
      <span class="head">Time per move</span>
      {#if hasTime}
        <svg class="an-time" viewBox="0 0 {Math.max(1, moves.length)} 100" preserveAspectRatio="none" role="img" aria-label="The time each move took, White above, Black below">
          <line x1="0" y1="50" x2={moves.length} y2="50" class="axis" />
          {#each moves as m (m.ply)}
            {@const h = (48 * (m.spent_ms ?? 0)) / spentMax}
            <rect x={m.ply - 1 + 0.12} width="0.76" y={m.side === "white" ? 50 - h : 50} height={h} class={m.side === "white" ? "w" : "b"} class:at={gameAt === m.ply}><title>{moveText(m)}: {seconds(m.spent_ms ?? 0)}</title></rect>
          {/each}
        </svg>
        <div class="an-legend muted"><span>White above, Black below</span><span>longest {seconds(spentMax)}</span></div>
      {:else}
        <p class="muted">The game's PGN has no clocks.</p>
      {/if}
    </div>
    <div class="an-chart">
      <span class="head">Themes</span>
      {#if themes.length}
        <div class="an-themes">
          <span class="head"></span><span class="head">White</span><span class="head">Black</span>
          {#each themes as x (x.t)}<span>{motifWords(x.t)}</span><b>{x.w || "–"}</b><b>{x.b || "–"}</b>{/each}
        </div>
      {:else}
        <p class="muted">No themes in this game's moves.</p>
      {/if}
    </div>
  {/if}
</div>
