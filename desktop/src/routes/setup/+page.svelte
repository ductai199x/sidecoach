<script lang="ts">
  // First-run setup: what the app needs beyond its code, fetched into the data folder
  // with one progress bar for the lot (docs/app-plan.md, section 6 step 4 and section 7).
  // The required items are ticked and cannot be unticked; the optional ones (the puzzle
  // sample, the CUDA pack when this machine's GPU can use it) start ticked and can be
  // left out. Re-runnable from Help; nothing already present is fetched again.
  import { onMount } from "svelte";
  import { connect, get, post } from "$lib/api";
  import { inShell } from "$lib/shell.svelte";

  type Item = { id: string; title: string; size: number; present: boolean; optional: boolean; detail: string };
  type Status = {
    items: Item[];
    ready: boolean;
    run: { running?: boolean; item?: string | null; done?: number; total?: number; errors?: Record<string, string>; finished?: string[]; queue?: string[] };
    data_dir: string;
    free_bytes?: number | null;
    gpu?: { mode: string; gpu: { found: boolean; eligible: boolean; reason: string; name?: string; driver?: string; vram?: number }; installed: string | null; active: { pack: string | null; cuda: boolean; detail: string } };
  };

  let status = $state<Status | null>(null);
  let picked = $state<Record<string, boolean>>({});
  let timer: ReturnType<typeof setTimeout> | null = null;

  const gb = (n: number) => (n >= 1e9 ? (n / 1e9).toFixed(1) + " GB" : n >= 1e6 ? Math.round(n / 1e6) + " MB" : Math.round(n / 1e3) + " KB");
  const missing = $derived((status?.items ?? []).filter((i) => !i.present));
  const chosen = $derived(missing.filter((i) => !i.optional || picked[i.id]));
  const totalBytes = $derived(chosen.reduce((a, i) => a + i.size, 0));
  const running = $derived(!!status?.run?.running);
  const percent = $derived(status?.run?.total ? Math.min(100, Math.round((100 * (status.run.done ?? 0)) / status.run.total)) : 0);
  const current = $derived(status?.items.find((i) => i.id === status?.run?.item));
  // The download's speed and what is left, from the bytes between polls, smoothed over
  // the last few seconds so the figure does not jitter.
  let samples: { t: number; done: number }[] = [];
  let speed = $state(0); // bytes per second
  const remaining = $derived.by(() => {
    const total = status?.run?.total ?? 0, done = status?.run?.done ?? 0;
    if (!speed || total <= done) return "";
    const s = (total - done) / speed;
    return s < 60 ? `${Math.ceil(s)} s left` : s < 3600 ? `${Math.ceil(s / 60)} min left` : `${(s / 3600).toFixed(1)} h left`;
  });
  const rate = (bps: number) => (bps >= 1e6 ? (bps / 1e6).toFixed(1) + " MB/s" : Math.round(bps / 1e3) + " KB/s");

  async function refresh() {
    status = await get<Status>("/api/setup/status");
    for (const i of status.items) if (!(i.id in picked)) picked[i.id] = true;
    if (status.run?.running) {
      const now = performance.now(), done = status.run.done ?? 0;
      // An item change resets the count (each item's bytes start again).
      if (samples.length && done < samples[samples.length - 1].done) samples = [];
      samples.push({ t: now, done });
      samples = samples.filter((x) => now - x.t <= 5000);
      const first = samples[0];
      speed = samples.length > 1 && now > first.t ? ((done - first.done) * 1000) / (now - first.t) : 0;
    } else {
      samples = [];
      speed = 0;
    }
    timer = setTimeout(refresh, running ? 500 : 3000);
  }
  async function start() {
    await post("/api/setup/start", { ids: chosen.map((i) => i.id) });
    if (timer) clearTimeout(timer);
    await refresh();
  }
  async function cancel() {
    await post("/api/setup/cancel");
  }

  onMount(() => {
    connect().then(refresh);
    return () => { if (timer) clearTimeout(timer); };
  });
</script>

<svelte:head><title>Sidecoach · Setup</title></svelte:head>

<div class="setup">
  <h1>Set up Sidecoach</h1>
  {#if !status}
    <p class="muted">Loading…</p>
  {:else}
    <p class="muted">Into <code>{status.data_dir}</code>{#if status.free_bytes != null} · {gb(status.free_bytes)} free{/if}</p>
    <ul class="items">
      {#each status.items as item (item.id)}
        <li class:present={item.present} class:err={!!status.run?.errors?.[item.id]}>
          <label>
            <input type="checkbox" checked={item.present || !!picked[item.id]} disabled={item.present || !item.optional || running} onchange={(e) => (picked[item.id] = e.currentTarget.checked)} />
            <span class="title">{item.title}</span>
            <span class="size">{item.present ? "ready" : gb(item.size)}</span>
          </label>
          {#if item.detail}<p class="desc">{item.detail}</p>{/if}
          {#if status.run?.errors?.[item.id]}<p class="desc error">{status.run.errors[item.id]}</p>{/if}
        </li>
      {/each}
    </ul>
    {#if status.gpu}
      {@const g = status.gpu.gpu}
      <p class="desc">
        GPU: {g.found ? `${g.name}, driver ${g.driver}, ${gb(g.vram ?? 0)}` : "none found"}{g.found && !g.eligible ? ` · ${g.reason}` : ""}
        {#if status.gpu.installed} · pack installed{/if}{#if status.gpu.active?.detail} · {status.gpu.active.detail}{/if}
      </p>
    {/if}
    {#if running}
      <div class="bar"><div class="fill" style="width: {percent}%"></div></div>
      <p class="muted">{current ? current.title : "…"} · {gb(status.run.done ?? 0)} of {gb(status.run.total ?? 0)} ({percent}%){#if speed > 0} · {rate(speed)} · {remaining}{/if}</p>
      <button class="ghost" onclick={cancel}>Cancel</button>
    {:else if missing.length === 0}
      <p class="ok">Everything is in place.</p>
    {:else}
      <button class="primary" onclick={start} disabled={chosen.length === 0}>Download {gb(totalBytes)}</button>
      {#if status.run?.finished?.length}<p class="muted">Done: {status.run.finished.join(", ")}</p>{/if}
    {/if}
  {/if}
  {#if !inShell()}
    <p class="back"><button type="button" class="link" onclick={() => (window.location.href = "/")}>&larr; Back to the panel</button></p>
  {/if}
</div>

<style>
  .setup { max-width: 720px; margin: 0 auto; padding: 24px 18px; }
  h1 { font-size: 18px; margin: 0 0 8px; }
  .items { list-style: none; padding: 0; margin: 16px 0; }
  .items li { padding: 10px 12px; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 8px; background: var(--panel); }
  .items li.present { opacity: 0.7; }
  .items li.err { border-color: var(--error); }
  .items label { display: flex; align-items: center; gap: 10px; cursor: pointer; }
  .title { flex: 1; font-weight: 600; }
  .size { color: var(--muted); font-size: 12px; }
  .desc { margin: 4px 0 0 26px; color: var(--muted); font-size: 12px; }
  .desc.error { color: var(--error); }
  .muted { color: var(--muted); font-size: 12px; }
  .ok { color: var(--accent); font-weight: 600; }
  .bar { height: 10px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin: 12px 0 6px; }
  .fill { height: 100%; background: var(--accent); transition: width 0.4s; }
  code { font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
  .back { margin-top: 20px; }
  .back .link { background: none; border: 0; padding: 0; font: inherit; font-weight: 600; color: var(--accent); cursor: pointer; }
</style>
