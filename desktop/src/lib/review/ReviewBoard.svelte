<script lang="ts">
  // The review's board: the position the review is at, its pieces sliding from move to move;
  // the move that led here tinted in its class's colour on both squares and its badge on the
  // destination's corner, as chess.com marks a reviewed move (a line's move is lit yellow, as a
  // move on any board); the arrows the review draws
  // (the best move, a line being shown); and the player's own marks -- a right-click lights a
  // square, a right-drag draws an arrow, a left-click clears them.
  import { onMount } from "svelte";
  import { FILES, cellOf, squareAt } from "$lib/chess";
  import { PieceLayer } from "$lib/pieces";

  export type Arrow = { from: string; to: string; kind: "best" | "follow" };

  let {
    fen,
    uci = null,
    cls = null,
    bottom = "white",
    arrows = [],
  }: { fen: string; uci?: string | null; cls?: string | null; bottom?: string; arrows?: Arrow[] } = $props();

  let boardEl = $state<HTMLDivElement | null>(null);
  let piecesEl = $state<HTMLDivElement | null>(null);
  let layer: PieceLayer | null = null;
  let marked = $state<string[]>([]);
  let drawn = $state<[string, string][]>([]);

  const from = $derived(uci ? uci.slice(0, 2) : null);
  const to = $derived(uci ? uci.slice(2, 4) : null);
  const squares = $derived.by(() => {
    const out: { name: string; dark: boolean; odd: boolean; rankLabel: string | null; fileLabel: string | null }[] = [];
    for (let y = 0; y < 8; y++) {
      for (let x = 0; x < 8; x++) {
        const name = squareAt(x, y, bottom)!;
        const file = FILES.indexOf(name[0]), rank = +name[1] - 1;
        out.push({ name, dark: (rank + file) % 2 === 0, odd: y % 2 === 1, rankLabel: x === 0 ? String(rank + 1) : null, fileLabel: y === 7 ? FILES[file] : null });
      }
    }
    return out;
  });
  const badgeAt = $derived(to ? cellOf(to, bottom) : null);

  function line(a: string, b: string) {
    const p = cellOf(a, bottom), q = cellOf(b, bottom);
    const x1 = p.x + 0.5, y1 = p.y + 0.5, x2 = q.x + 0.5, y2 = q.y + 0.5;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const k = 0.32 / len; // stop short of the centre so the head lands on it
    return { x1, y1, x2: x2 - dx * k, y2: y2 - dy * k };
  }

  // The pieces follow the position; the move that made it slides.
  $effect(() => {
    const at = fen, move = uci;
    if (!piecesEl) return;
    layer ??= new PieceLayer(piecesEl, bottom);
    if (layer.bottom !== bottom) layer.turn(bottom);
    layer.render(at, move);
    marked = [];
    drawn = [];
  });

  // ---- the player's own marks ----
  let rightFrom: string | null = null;
  function squareUnder(e: PointerEvent): string | null {
    if (!boardEl) return null;
    const r = boardEl.getBoundingClientRect();
    return squareAt(Math.floor(((e.clientX - r.left) / r.width) * 8), Math.floor(((e.clientY - r.top) / r.height) * 8), bottom);
  }
  function down(e: PointerEvent) {
    if (e.button === 2) { rightFrom = squareUnder(e); return; }
    if (e.button === 0) { marked = []; drawn = []; }
  }
  function up(e: PointerEvent) {
    if (e.button !== 2 || !rightFrom) return;
    const a = rightFrom, b = squareUnder(e);
    rightFrom = null;
    if (b && b !== a) {
      const i = drawn.findIndex(([p, q]) => p === a && q === b);
      drawn = i >= 0 ? drawn.filter((_, j) => j !== i) : [...drawn, [a, b]];
    } else marked = marked.includes(a) ? marked.filter((m) => m !== a) : [...marked, a];
  }

  onMount(() => () => { layer = null; });
</script>

<div
  class="board-wrap rv-board"
  role="img"
  aria-label="The position"
  onpointerdown={down}
  onpointerup={up}
  oncontextmenu={(e) => e.preventDefault()}
>
  <div class="board cb" bind:this={boardEl}>
    {#each squares as sq (sq.name)}
      <div
        class="sq {sq.dark ? 'dark' : 'light'}"
        class:odd={sq.odd}
        class:cls={cls && (sq.name === from || sq.name === to)}
        class:hl={!cls && (sq.name === from || sq.name === to)}
        class:from={sq.name === from}
        class:marked={marked.includes(sq.name)}
        style={cls && (sq.name === from || sq.name === to) ? `--tint: color-mix(in srgb, var(--cls-${cls}) ${sq.name === from ? 38 : 52}%, transparent)` : undefined}
      >
        {#if sq.rankLabel}<span class="coord rank">{sq.rankLabel}</span>{/if}
        {#if sq.fileLabel}<span class="coord file">{sq.fileLabel}</span>{/if}
      </div>
    {/each}
  </div>
  <div class="pieces" bind:this={piecesEl}></div>
  <svg class="overlay" viewBox="0 0 8 8" aria-hidden="true">
    <defs>
      {#each ["best", "follow", "user"] as c (c)}
        <marker id="rv-ah-{c}" class="mk-{c}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" /></marker>
      {/each}
    </defs>
    {#each arrows as a (a.kind + a.from + a.to)}
      {@const l = line(a.from, a.to)}
      <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-{a.kind}" marker-end="url(#rv-ah-{a.kind})" />
    {/each}
    {#each drawn as [a, b] (a + b)}
      {@const l = line(a, b)}
      <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-user user" marker-end="url(#rv-ah-user)" />
    {/each}
  </svg>
  {#if cls && badgeAt}
    <div class="rv-mark" style="left: {badgeAt.x * 12.5}%; top: {badgeAt.y * 12.5}%"><i class="badge {cls}"></i></div>
  {/if}
</div>
