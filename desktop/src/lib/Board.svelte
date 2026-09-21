<script lang="ts">
  // The mirrored board: 64 squares from a FEN, the last move lit, coordinates on the
  // edges, and an SVG overlay for the marks that go with a coach message. White at the
  // bottom shows rank 8 on top and file a at the left; Black at the bottom the reverse.
  // The square colours and the labels come from the absolute square, as on chess.com.
  import { FILES, cellOf, moveSquares, ranksOf, spriteOf } from "./chess";
  import type { Mark } from "./api";

  let {
    fen = null,
    bottom = "white",
    lastMove = null,
    marks = [],
    board = $bindable<HTMLDivElement | null>(null),
  }: { fen?: string | null; bottom?: string; lastMove?: string | null; marks?: Mark[]; board?: HTMLDivElement | null } = $props();

  const squares = $derived.by(() => {
    if (!fen) return [];
    const ranks = ranksOf(fen);
    const blackBottom = bottom === "black";
    const hl = moveSquares(lastMove);
    const out: { name: string; dark: boolean; odd: boolean; hl: boolean; piece: string | null; rankLabel: string | null; fileLabel: string | null }[] = [];
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const rank = blackBottom ? r : 7 - r;
        const file = blackBottom ? 7 - c : c;
        const name = FILES[file] + (rank + 1);
        out.push({
          name,
          dark: (rank + file) % 2 === 0,
          odd: r % 2 === 1,
          hl: hl.has(name),
          piece: ranks[7 - rank]?.[file] ?? null,
          rankLabel: c === 0 ? String(rank + 1) : null,
          fileLabel: r === 7 ? FILES[file] : null,
        });
      }
    }
    return out;
  });

  const arrowColours = $derived([...new Set(marks.filter((m) => m.kind === "arrow").map((m) => m.color))]);
  const shapes = $derived(marks.filter((m) => m.kind !== "arrow"));
  const arrows = $derived(marks.filter((m) => m.kind === "arrow"));

  function arrowLine(m: Mark) {
    const a = cellOf(m.squares[0], bottom), b = cellOf(m.squares[1], bottom);
    const x1 = a.x + 0.5, y1 = a.y + 0.5, x2 = b.x + 0.5, y2 = b.y + 0.5;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const k = 0.32 / len; // stop short of the centre so the head lands on it
    return { x1, y1, x2: x2 - dx * k, y2: y2 - dy * k };
  }
</script>

<div class="board-wrap">
  <div class="board" bind:this={board}>
    {#each squares as sq (sq.name)}
      <div class="sq {sq.dark ? 'dark' : 'light'}" class:odd={sq.odd} class:hl={sq.hl}>
        {#if sq.piece}<div class="pc {spriteOf(sq.piece)}"></div>{/if}
        {#if sq.rankLabel}<span class="coord rank">{sq.rankLabel}</span>{/if}
        {#if sq.fileLabel}<span class="coord file">{sq.fileLabel}</span>{/if}
      </div>
    {/each}
  </div>
  <svg class="overlay" viewBox="0 0 8 8" aria-hidden="true">
    <defs>
      {#each arrowColours as c (c)}
        <marker id="ah-{c}" class="mk-{c}" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" />
        </marker>
      {/each}
    </defs>
    {#each shapes as m}
      {@const p = cellOf(m.squares[0], bottom)}
      {@const op = m.opacity == null ? 1 : m.opacity}
      {#if m.kind === "square"}
        <rect x={p.x} y={p.y} width="1" height="1" class="mk-{m.color}" opacity={op * 0.55} />
      {:else}
        <circle cx={p.x + 0.5} cy={p.y + 0.5} r="0.42" class="mk-{m.color}" opacity={op} />
      {/if}
    {/each}
    {#each arrows as m}
      {@const l = arrowLine(m)}
      <line x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} class="mk-{m.color}" opacity={m.opacity == null ? 1 : m.opacity} marker-end="url(#ah-{m.color})" />
    {/each}
  </svg>
</div>
