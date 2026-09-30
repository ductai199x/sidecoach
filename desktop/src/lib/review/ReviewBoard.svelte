<script lang="ts">
  // The review's board: the position the review is at, its pieces sliding from move to move;
  // the move that led here tinted in its class's colour on both squares and its badge on the
  // destination's corner, as chess.com marks a reviewed move (a line's move is lit yellow, as a
  // move on any board); the arrows the review draws (the best move, a line being shown); the
  // player's own marks. Given `legal`, it takes the player's moves (Retry): the hand is the
  // puzzle board's (lib/moveinput), a tried move is `onplay`'s, and a `wrong` one shows red
  // until `redraw` puts the position back.
  import Badge from "$lib/Badge.svelte";
  import { FILES, cellOf, colourOf, squareAt } from "$lib/chess";
  import { MoveInput } from "$lib/moveinput.svelte";
  import { PieceLayer } from "$lib/pieces";
  import { arrow } from "./arrows";

  // The review's arrows: the best move (gold), a threat (red), the move's idea (blue).
  type Arrow = { from: string; to: string; kind: "best" | "threat" | "idea" };

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
  <svg class="overlay rv-arrows" viewBox="0 0 8 8" aria-hidden="true">
    {#each [...arrows.map((a) => ({ ...a, id: a.kind + a.from + a.to })), ...input.arrows.map(([from, to]) => ({ from, to, kind: "user", id: "user" + from + to }))] as a (a.id)}
      {@const d = arrow(a.from, a.to, bottom)}
      <g class="rv-arrow {a.kind}">
        <polyline points={d.shaft} stroke-width={d.width} />
        <polygon points={d.head} />
      </g>
    {/each}
  </svg>
  {#if cls && badgeAt}
    {#key fen}
      <div class="rv-mark" style="left: {badgeAt.x * 12.5}%; top: {badgeAt.y * 12.5}%"><Badge kind={cls} /></div>
    {/key}
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
