<script lang="ts">
  // A figure game by game, as chess.com's Stats draw a rating: a filled line, the scale's round
  // values in a gutter on the left, the dates along the bottom; the pointer on it shows a game's
  // value and date. With `smooth`, the line is a running mean of that many games (one game says
  // little) and the games themselves are dots. Drawn at the size it is shown, so its text is the
  // page's size at any width.
  let {
    points,
    smooth = 0,
    height = 170,
    unit = "",
  }: { points: { t: number | null; v: number }[]; smooth?: number; height?: number; unit?: string } = $props();

  const PAD = { l: 44, r: 14, t: 14, b: 26 };
  let width = $state(600);
  let hover = $state<number | null>(null);

  const line = $derived.by(() => {
    if (!smooth) return points.map((p) => p.v);
    return points.map((_, i) => {
      const window = points.slice(Math.max(0, i - smooth + 1), i + 1);
      return window.reduce((s, p) => s + p.v, 0) / window.length;
    });
  });

  // Round values for the scale: a step of 1, 2, 2.5 or 5 times a power of ten, about four of them.
  const scale = $derived.by(() => {
    const all = [...points.map((p) => p.v), ...line];
    let lo = all.length ? Math.min(...all) : 0, hi = all.length ? Math.max(...all) : 1;
    if (hi - lo < 2) { lo -= 1; hi += 1; }
    const raw = (hi - lo) / 3;
    const mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
    const start = Math.floor(lo / step) * step, end = Math.ceil(hi / step) * step;
    const ticks: number[] = [];
    for (let v = start; v <= end + step / 1000; v += step) ticks.push(+v.toFixed(6));
    return { lo: start, hi: end, ticks, decimals: step < 1 ? 1 : 0 };
  });

  const plotW = $derived(Math.max(10, width - PAD.l - PAD.r));
  const x = (i: number) => PAD.l + (points.length > 1 ? (i / (points.length - 1)) * plotW : plotW / 2);
  const y = (v: number) => PAD.t + (1 - (v - scale.lo) / (scale.hi - scale.lo || 1)) * (height - PAD.t - PAD.b);
  const path = $derived(line.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" "));
  const area = $derived(line.length > 1 ? `${path} L${x(line.length - 1).toFixed(1)},${height - PAD.b} L${x(0).toFixed(1)},${height - PAD.b} Z` : "");
  const date = (t: number | null) => (t ? new Date(t * 1000).toLocaleDateString([], { month: "short", day: "numeric" }) : "");
  // The dates below: the first and the last game's, and the middle one's when there is room.
  const dates = $derived.by(() => {
    if (!points.length) return [];
    const at = points.length > 2 && width > 420 ? [0, Math.floor((points.length - 1) / 2), points.length - 1] : points.length > 1 ? [0, points.length - 1] : [0];
    const out: { x: number; text: string; anchor: string }[] = [];
    for (const i of at) {
      const text = date(points[i].t);
      if (!text || out.some((d) => d.text === text)) continue;
      out.push({ x: x(i), text, anchor: at.length === 1 ? "middle" : i === 0 ? "start" : i === points.length - 1 ? "end" : "middle" });
    }
    return out;
  });
  const fmt = (v: number) => `${v.toFixed(scale.decimals)}${unit}`;

  function pick(e: PointerEvent): void {
    const r = (e.currentTarget as SVGElement).getBoundingClientRect();
    const f = (e.clientX - r.left - PAD.l) / plotW;
    hover = points.length ? Math.max(0, Math.min(points.length - 1, Math.round(f * (points.length - 1)))) : null;
  }
</script>

<div class="st-chart" bind:clientWidth={width} style="height: {height}px">
  {#if points.length}
    <svg {width} {height} viewBox="0 0 {width} {height}" role="img" aria-label="The figure game by game" onpointermove={pick} onpointerleave={() => (hover = null)}>
      <defs>
        <linearGradient id="st-fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stop-color="var(--st-line)" stop-opacity="0.32" />
          <stop offset="1" stop-color="var(--st-line)" stop-opacity="0" />
        </linearGradient>
      </defs>
      {#each scale.ticks as v (v)}
        <line class="tick" x1={PAD.l} x2={width - PAD.r} y1={y(v)} y2={y(v)} />
        <text class="lab num" x={PAD.l - 8} y={y(v) + 4}>{fmt(v)}</text>
      {/each}
      {#each dates as d (d.text)}
        <text class="lab" x={d.x} y={height - 7} text-anchor={d.anchor}>{d.text}</text>
      {/each}
      {#if area}<path d={area} fill="url(#st-fill)" />{/if}
      {#if smooth}
        {#each points as p, i (i)}<circle class="dot" cx={x(i)} cy={y(p.v)} r="2.4" />{/each}
      {/if}
      {#if line.length > 1}<path class="line" d={path} />{/if}
      {#if hover !== null}
        <line class="cross" x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={height - PAD.b} />
        <circle class="at" cx={x(hover)} cy={y(line[hover])} r="4.5" />
      {:else if line.length}
        <circle class="at" cx={x(line.length - 1)} cy={y(line[line.length - 1])} r="4.5" />
      {/if}
    </svg>
    {#if hover !== null}
      {@const p = points[hover]}
      <div class="st-tip" style="left: {x(hover)}px; top: {y(line[hover])}px" class:flip={x(hover) > width - 130}>
        <b>{fmt(line[hover])}</b>
        {#if smooth && points.length > 1}<span>this game {fmt(p.v)}</span>{/if}
        <small>{date(p.t)}</small>
      </div>
    {/if}
    {#if points.length < 3}<div class="st-few">{points.length === 1 ? "One game so far" : "Two games so far"}: the line grows with every game reviewed.</div>{/if}
  {:else}
    <div class="st-few">No games in this range yet.</div>
  {/if}
</div>
