<script lang="ts">
  // A figure game by game, as chess.com's Stats draw a rating: a filled line under the points,
  // the last value's level, the first and last games' dates below. With `smooth`, a running mean
  // of that many games is the line (one game says little), the games themselves dots.
  let {
    points,
    smooth = 0,
    height = 150,
    unit = "",
  }: { points: { t: number | null; v: number }[]; smooth?: number; height?: number; unit?: string } = $props();

  const W = 600;
  const line = $derived.by(() => {
    if (!smooth) return points.map((p) => p.v);
    return points.map((_, i) => {
      const from = Math.max(0, i - smooth + 1);
      const window = points.slice(from, i + 1);
      return window.reduce((s, p) => s + p.v, 0) / window.length;
    });
  });
  const range = $derived.by(() => {
    const all = [...points.map((p) => p.v), ...line];
    if (!all.length) return { lo: 0, hi: 1 };
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = Math.max(1, (hi - lo) * 0.12);
    return { lo: lo - pad, hi: hi + pad };
  });
  const x = (i: number) => (points.length > 1 ? (i / (points.length - 1)) * W : W / 2);
  const y = (v: number) => height - 18 - ((v - range.lo) / (range.hi - range.lo)) * (height - 30);
  const path = $derived(line.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "));
  const area = $derived(line.length ? `${path} L${x(line.length - 1).toFixed(1)},${height - 18} L${x(0).toFixed(1)},${height - 18} Z` : "");
  const date = (t: number | null) => (t ? new Date(t * 1000).toLocaleDateString([], { month: "short", day: "numeric" }) : "");
  const ticks = $derived([range.hi, (range.hi + range.lo) / 2, range.lo].map((v) => Math.round(v)));
</script>

{#if points.length}
  <svg class="st-chart" viewBox="0 0 {W} {height}" role="img" aria-label="The figure game by game">
    <defs>
      <linearGradient id="st-fill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0" stop-color="var(--st-line)" stop-opacity="0.35" />
        <stop offset="1" stop-color="var(--st-line)" stop-opacity="0" />
      </linearGradient>
    </defs>
    {#each ticks as v, i (i)}
      <line class="tick" x1="0" x2={W} y1={y(v)} y2={y(v)} />
      <text class="lab" x="4" y={y(v) - 3}>{v}{unit}</text>
    {/each}
    <path d={area} fill="url(#st-fill)" />
    <path class="line" d={path} />
    {#if smooth}
      {#each points as p, i (i)}<circle class="dot" cx={x(i)} cy={y(p.v)} r="2.2" />{/each}
    {/if}
    <text class="lab" x="4" y={height - 4}>{date(points[0].t)}</text>
    <text class="lab end" x={W - 4} y={height - 4}>{date(points[points.length - 1].t)}</text>
  </svg>
{:else}
  <div class="st-chart empty" style="height: {height}px">Not enough games yet</div>
{/if}
