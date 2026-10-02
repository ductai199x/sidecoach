<script lang="ts">
  // A figure's trend in little, for a card: the running mean of `smooth` games as a line over a
  // soft fill, the last value dotted; no scale (the card says the figure). Drawn at its own size.
  // The height spans `span` of the figure at least (points of a percentage): scaled to its own
  // range, 81.0 to 81.6 filled the height as a big climb (the review of 2026-10-01).
  let { points, smooth = 5, height = 44, span = 10 }: { points: { v: number }[]; smooth?: number; height?: number; span?: number } = $props();

  let width = $state(200);
  const line = $derived(
    points.map((_, i) => {
      const window = points.slice(Math.max(0, i - smooth + 1), i + 1);
      return window.reduce((s, p) => s + p.v, 0) / window.length;
    }),
  );
  const least = $derived(line.length ? Math.min(...line) : 0);
  const most = $derived(line.length ? Math.max(...line) : 1);
  const range = $derived(Math.max(most - least, span));
  const lo = $derived((least + most) / 2 - range / 2);
  const x = (i: number) => 3 + (line.length > 1 ? i / (line.length - 1) : 1) * (width - 6);
  const y = (v: number) => 4 + (1 - (v - lo) / range) * (height - 8);
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
