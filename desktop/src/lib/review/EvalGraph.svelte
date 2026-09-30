<script lang="ts">
  // The game's graph, as chess.com draws it: White's expected score after every ply (ply 0
  // the start), White's share filled light from the bottom, the key moments dotted in their
  // class's colour, the move shown marked; a click (or a drag) goes to the move under it.
  let {
    graph,
    marks = [],
    at = null,
    height = 90,
    onpick,
  }: {
    graph: number[];
    marks?: { ply: number; kind: string }[];
    at?: number | null;
    height?: number;
    onpick?: (ply: number) => void;
  } = $props();

  let el = $state<HTMLDivElement | null>(null);
  let hover = $state<number | null>(null);
  const last = $derived(Math.max(1, graph.length - 1));
  // The fill: the line across the plies, closed along the bottom.
  const area = $derived(
    graph.length
      ? `M0,100 ` + graph.map((s, i) => `L${i},${(100 * (1 - s)).toFixed(2)}`).join(" ") + ` L${graph.length - 1},100 Z`
      : "",
  );
  const dots = $derived(
    marks
      .filter((m) => m.kind !== "book" && m.ply < graph.length)
      .map((m) => ({ ...m, x: (100 * m.ply) / last, y: 100 * (1 - graph[m.ply]) })),
  );

  function plyAt(clientX: number): number {
    if (!el) return 0;
    const r = el.getBoundingClientRect();
    return Math.max(0, Math.min(graph.length - 1, Math.round(((clientX - r.left) / r.width) * last)));
  }
  let pressed = false;
</script>

<div
  class="rv-graph"
  style="height: {height}px"
  bind:this={el}
  role="slider"
  tabindex="-1"
  aria-label="The game's evaluation, move by move"
  aria-valuemin={0}
  aria-valuemax={graph.length - 1}
  aria-valuenow={at ?? 0}
  onpointerdown={(e) => { pressed = true; el?.setPointerCapture(e.pointerId); onpick?.(plyAt(e.clientX)); }}
  onpointermove={(e) => { hover = plyAt(e.clientX); if (pressed) onpick?.(hover); }}
  onpointerup={() => (pressed = false)}
  onpointerleave={() => (hover = null)}
>
  <svg viewBox="0 0 {last} 100" preserveAspectRatio="none" aria-hidden="true">
    <path class="white" d={area} />
    <line class="mid" x1="0" y1="50" x2={last} y2="50" />
    {#if hover !== null}<line class="hover" x1={hover} y1="0" x2={hover} y2="100" />{/if}
    {#if at !== null}<line class="at" x1={at} y1="0" x2={at} y2="100" />{/if}
  </svg>
  {#each dots as d (d.ply)}
    <i class="gdot" style="left: {d.x}%; top: {d.y}%; background: var(--cls-{d.kind})" title="{d.kind}, move {Math.ceil(d.ply / 2)}"></i>
  {/each}
</div>

<style>
  .gdot {
    position: absolute; width: 8px; height: 8px; margin: -4px 0 0 -4px; border-radius: 50%;
    box-shadow: 0 0 0 1.5px #262522; pointer-events: none;
  }
</style>
