<script lang="ts">
  // The review's board: the position the review is at, its pieces sliding from move to move;
  // the move that led here tinted in its class's colour on both squares and its badge on the
  // destination's corner, as chess.com marks a reviewed move (a line's move is lit yellow, as a
  // move on any board); the arrows the review draws (the best move, a line being shown); the
  // player's own marks. Given `legal`, it takes the player's moves (Retry): the hand is the
  // puzzle board's (lib/moveinput), a tried move is `onplay`'s, and a `wrong` one shows red
  // until `redraw` puts the position back.
  import { FILES, cellOf, colourOf, squareAt } from "$lib/chess";
  import { MoveInput } from "$lib/moveinput.svelte";
  import { PieceLayer } from "$lib/pieces";

  type Arrow = { from: string; to: string; kind: "best" | "follow" };

  let {
    fen,
    uci = null,
    cls = null,
    bottom = "white",
    arrows = [],
    legal = null,
    side = "white",
    wrong = null,
    redraw = 0,
    onplay,
  }: {
    fen: string;
    uci?: string | null;
    cls?: string | null;
    bottom?: string;
    arrows?: Arrow[];
    legal?: Record<string, string[]> | null;
    side?: string;
    wrong?: string | null;
    redraw?: number;
    onplay?: (uci: string) => void;
  } = $props();

  let boardEl = $state<HTMLDivElement | null>(null);
  let wrapEl = $state<HTMLDivElement | null>(null);
  let piecesEl = $state<HTMLDivElement | null>(null);
  let layer: PieceLayer | null = null;
  let occupied = $state<Record<string, string>>({}); // square -> letter: a capture's ring

  const input = new MoveInput({
    board: () => boardEl,
    wrap: () => wrapEl,
    layer: () => layer,
    bottom: () => bottom,
    canMove: () => !!legal && !wrong,
    side: () => side,
    legal: () => legal ?? {},
    play: (move) => onplay?.(move),
  });

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

  // The pieces follow the position; the move that made it slides. `redraw` puts a tried move
  // back.
  $effect(() => {
    const at = fen, move = uci;
    void redraw;
    if (!piecesEl) return;
    layer ??= new PieceLayer(piecesEl, bottom);
    if (layer.bottom !== bottom) layer.turn(bottom);
    occupied = layer.render(at, move);
    input.clear();
    input.clearMarks();
  });
</script>

<div
  class="board-wrap rv-board"
  class:takes={!!legal}
  bind:this={wrapEl}
  role="application"
  aria-label="The position"
  onpointerdown={input.down}
  onpointermove={input.move}
  onpointerup={input.up}
  onpointercancel={input.cancel}
  oncontextmenu={(e) => e.preventDefault()}
>
  <div class="board cb" bind:this={boardEl}>
    {#each squares as sq (sq.name)}
      {@const lit = sq.name === from || sq.name === to}
      <div
        class="sq {sq.dark ? 'dark' : 'light'}"
        class:odd={sq.odd}
        class:cls={cls && lit}
        class:hl={!cls && lit && !wrong}
        class:wrong={wrong && sq.name === wrong.slice(2, 4)}
        class:sel={input.selected === sq.name}
        class:target={input.targets.includes(sq.name)}
        class:occupied={input.targets.includes(sq.name) && !!occupied[sq.name]}
        class:grab={!!legal && !wrong && !!occupied[sq.name] && colourOf(occupied[sq.name]) === side}
        class:hover={input.hover === sq.name}
        class:marked={input.marked.includes(sq.name)}
        style={cls && lit ? `--tint: color-mix(in srgb, var(--cls-${cls}) ${sq.name === from ? 38 : 52}%, transparent)` : undefined}
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
    {#each input.arrows as [a, b] (a + b)}
      {@const l = line(a, b)}
      <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-user user" marker-end="url(#rv-ah-user)" />
    {/each}
  </svg>
  {#if cls && badgeAt}
    <div class="rv-mark" style="left: {badgeAt.x * 12.5}%; top: {badgeAt.y * 12.5}%"><i class="badge {cls}"></i></div>
  {/if}
  {#if input.promo}
    {@const promo = input.promo}
    <div class="promo" class:down={promo.down} style="left: {promo.x * 12.5}%; top: {promo.down ? (promo.y - 3) * 12.5 : promo.y * 12.5}%">
      {#each ["Q", "R", "B", "N"] as p (p)}
        <button class="{side === 'white' ? 'w' : 'b'}{p}" title={p} aria-label="Promote to {p}" onclick={(e) => { e.stopPropagation(); promo.resolve(p.toLowerCase()); }}></button>
      {/each}
      <button class="x" onclick={(e) => { e.stopPropagation(); promo.resolve(null); }}>×</button>
    </div>
  {/if}
</div>
