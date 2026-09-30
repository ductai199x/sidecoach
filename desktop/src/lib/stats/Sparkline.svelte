<script lang="ts">
  // A figure's trend in little, for a card: the running mean of `smooth` games as a line over a
  // soft fill, the last value dotted; no scale (the card says the figure). Drawn at its own size.
  let { points, smooth = 5, height = 44 }: { points: { v: number }[]; smooth?: number; height?: number } = $props();

  let width = $state(200);
  const line = $derived(
    points.map((_, i) => {
      const window = points.slice(Math.max(0, i - smooth + 1), i + 1);
      return window.reduce((s, p) => s + p.v, 0) / window.length;
    }),
  );
  const lo = $derived(line.length ? Math.min(...line) : 0);
  const hi = $derived(line.length ? Math.max(...line) : 1);
  const x = (i: number) => 3 + (line.length > 1 ? i / (line.length - 1) : 1) * (width - 6);
  const y = (v: number) => 4 + (1 - (hi - lo > 0.5 ? (v - lo) / (hi - lo) : 0.5)) * (height - 8);
  const path = $derived(line.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "));
</script>

<div class="st-spark" bind:clientWidth={width} style="height: {height}px">
  {#if line.length > 1}
    <svg {width} {height} viewBox="0 0 {width} {height}" aria-hidden="true">
      <path class="fill" d="{path} L{x(line.length - 1)},{height} L{x(0)},{height} Z" />
      <path class="line" d={path} />
      <circle cx={x(line.length - 1)} cy={y(line[line.length - 1])} r="3" />
    </svg>
  {/if}
</div>
