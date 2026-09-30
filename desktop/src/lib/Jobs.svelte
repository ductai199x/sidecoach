<script lang="ts">
  // The panel's bell: the long jobs that run in the background -- the review of the library's
  // games (ui/backlog.py), a fetch of the setup (setup/) -- seen and run from one place, as a
  // browser shows its downloads (Tai, 2026-09-30: "we don't want to kick off a long running
  // intensive cpu job without the user's notice"). There only while a job runs, or one that ran
  // here has not been cleared; its ring the jobs' progress. Hovered, a line on each; opened, the
  // list: in progress with a bar, the game or item under way, the time left, pause or resume,
  // cancel, and the window it belongs to; then what finished, cleared at will.
  import { onMount } from "svelte";
  import { connect, get, post } from "./api";
  import { inShell, openWindow } from "./shell.svelte";

  type Reviews = {
    status: "reviewing" | "paused" | "waiting" | "done" | "empty";
    total: number; done: number; waiting: number;
    current: { key: string; white: string | null; black: string | null; done: number; total: number } | null;
    eta_s: number | null;
  };
  type Setup = { running?: boolean; item?: string | null; done?: number; total?: number; errors?: Record<string, string>; finished?: string[]; titles?: Record<string, string> };
  type Job = {
    id: "reviews" | "setup";
    title: string;
    detail: string;
    meta: string;
    frac: number;
    state: "running" | "paused" | "waiting" | "done" | "failed";
    sign: string; // what a clearing is of: a job cleared stays hidden until it changes
  };

  let reviews = $state<Reviews | null>(null);
  let setup = $state<Setup | null>(null);
  let open = $state(false);
  let hover = $state(false);
  let seen = $state<Record<string, boolean>>({}); // the jobs seen running here
  let cleared = $state<Record<string, string>>({}); // job -> the sign it was cleared at
  let root = $state<HTMLDivElement | null>(null);

  function minutes(s: number): string {
    if (s < 90) return "under 2 min left";
    const m = Math.round(s / 60);
    return m < 60 ? `about ${m} min left` : `about ${Math.floor(m / 60)} h ${m % 60} min left`;
  }
  function size(bytes: number): string {
    return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
  }

  const jobs = $derived.by(() => {
    const out: Job[] = [];
    const r = reviews;
    if (r && r.status !== "empty") {
      const share = r.current && r.current.total ? r.current.done / r.current.total : 0;
      const game = r.current ? `${r.current.white ?? "White"} v ${r.current.black ?? "Black"}` : "";
      out.push({
        id: "reviews",
        title: r.status === "done" ? "Every game reviewed" : r.status === "paused" ? "Game reviews paused" : r.status === "waiting" ? "Game reviews wait for you" : "Reviewing your games",
        detail:
          r.status === "reviewing" && r.current ? `${game} · ${r.current.done} of ${r.current.total} positions`
          : r.status === "waiting" ? "They go on when your game or your review is done."
          : r.status === "paused" ? "Resume when the computer is free: nothing is lost."
          : r.status === "done" ? "Stats counts all of them." : "Getting the games ready…",
        meta: `${r.done} of ${r.total} games${r.eta_s != null ? " · " + minutes(r.eta_s) : ""}`,
        frac: r.total ? Math.min(1, (r.done + share) / r.total) : 0,
        state: r.status === "done" ? "done" : r.status === "paused" ? "paused" : r.status === "waiting" ? "waiting" : "running",
        sign: `${r.status === "done" ? "done" : "on"}:${r.total}`,
      });
    }
    const s = setup;
    if (s && (s.running || (s.total ?? 0) > 0)) {
      const errors = Object.keys(s.errors ?? {});
      const item = s.item ? (s.titles?.[s.item] ?? s.item) : "";
      out.push({
        id: "setup",
        title: s.running ? "Setting up Sidecoach" : errors.length ? "Setup could not finish" : "Setup finished",
        detail: s.running ? item : errors.length ? `${errors.map((e) => s.titles?.[e] ?? e).join(", ")}: see Set up` : `${(s.finished ?? []).map((f) => s.titles?.[f] ?? f).join(", ")}`,
        meta: s.total ? `${size(s.done ?? 0)} of ${size(s.total)}` : "",
        frac: s.total ? Math.min(1, (s.done ?? 0) / s.total) : 0,
        state: s.running ? "running" : errors.length ? "failed" : "done",
        sign: `${s.running ? "on" : "done"}:${s.total}:${(s.finished ?? []).length}`,
      });
    }
    return out;
  });
  const active = $derived(jobs.filter((j) => j.state === "running" || j.state === "paused" || j.state === "waiting"));
  // Finished here, and not cleared: a job the app started with already done is none of these.
  const finished = $derived(jobs.filter((j) => (j.state === "done" || j.state === "failed") && seen[j.id] && cleared[j.id] !== j.sign));
  const shown = $derived(active.length > 0 || finished.length > 0);
  const ring = $derived(active.length ? active.reduce((s, j) => s + j.frac, 0) / active.length : 1);
  const runningNow = $derived(active.some((j) => j.state === "running"));

  $effect(() => {
    for (const j of active) if (!seen[j.id]) seen = { ...seen, [j.id]: true };
  });
  $effect(() => {
    if (!shown) open = false;
  });

  async function read(): Promise<void> {
    try {
      const got = await get<{ reviews: Reviews | null; setup: Setup }>("/api/jobs");
      reviews = got.reviews;
      setup = got.setup;
    } catch {
      /* the sidecar not up yet, or an older one */
    }
  }

  async function pause(paused: boolean): Promise<void> {
    const res = await post<{ ok: boolean; reviews?: Reviews }>("/api/jobs/reviews", { paused }).catch(() => null);
    if (res?.reviews) reviews = res.reviews;
  }
  async function cancelSetup(): Promise<void> {
    await post("/api/setup/cancel").catch(() => null);
    void read();
  }
  function go(route: "games" | "setup"): void {
    open = false;
    if (inShell()) openWindow(route);
    else window.open("/" + route, "_blank");
  }
  function clear(): void {
    const next = { ...cleared };
    for (const j of finished) next[j.id] = j.sign;
    cleared = next;
  }

  onMount(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let gone = false;
    const loop = async () => {
      await read();
      if (!gone) timer = setTimeout(loop, open ? 1000 : 2000);
    };
    void connect().then(loop);
    const away = (e: PointerEvent) => { if (open && root && !root.contains(e.target as Node)) open = false; };
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") open = false; };
    window.addEventListener("pointerdown", away);
    window.addEventListener("keydown", esc);
    return () => {
      gone = true;
      if (timer) clearTimeout(timer);
      window.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", esc);
    };
  });
</script>

{#snippet bar(j: Job)}
  <span class="jb-bar" class:paused={j.state !== "running"} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(j.frac * 100)}><i style:width="{Math.round(j.frac * 100)}%"></i></span>
{/snippet}

{#snippet kindIcon(j: Job)}
  {#if j.id === "reviews"}
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>
  {:else}
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19.35 10.04A7.49 7.49 0 0 0 12 4C9.11 4 6.6 5.64 5.35 8.04A5.994 5.994 0 0 0 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM17 13l-5 5-5-5h3V9h4v4h3z"/></svg>
  {/if}
{/snippet}

{#if shown}
  <div class="jobs" bind:this={root}>
    <button
      class="jb-bell"
      class:on={runningNow}
      aria-label="Background jobs"
      aria-expanded={open}
      onclick={() => { open = !open; hover = false; }}
      onmouseenter={() => (hover = true)}
      onmouseleave={() => (hover = false)}
    >
      <svg class="jb-ring" viewBox="0 0 36 36" aria-hidden="true">
        <circle cx="18" cy="18" r="16" class="track" />
        {#if active.length}<circle cx="18" cy="18" r="16" class="fill" style:stroke-dasharray="{(ring * 100.53).toFixed(1)} 100.53" />{/if}
      </svg>
      <svg class="jb-icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 22a2 2 0 0 0 2-2h-4a2 2 0 0 0 2 2zm6-6v-5c0-3.07-1.64-5.64-4.5-6.32V4a1.5 1.5 0 0 0-3 0v.68C7.63 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2z"/></svg>
      {#if finished.length && !active.length}<span class="jb-dot" aria-hidden="true"></span>{/if}
    </button>

    {#if hover && !open}
      <div class="jb-peek" role="tooltip">
        {#each [...active, ...finished] as j (j.id)}
          <div class="jb-peek-row">
            <b>{j.title}</b>
            {#if j.meta}<small>{j.meta}</small>{/if}
            {#if j.state === "running" || j.state === "paused" || j.state === "waiting"}{@render bar(j)}{/if}
          </div>
        {/each}
      </div>
    {/if}

    {#if open}
      <div class="jb-panel" role="dialog" aria-label="Background jobs">
        <div class="jb-head">
          <span>Background jobs</span>
          <button class="jb-x" aria-label="Close" title="Close (Esc)" onclick={() => (open = false)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg></button>
        </div>
        {#if active.length}
          <div class="jb-section">In progress</div>
          {#each active as j (j.id)}
            <div class="jb-row">
              <span class="jb-kind">{@render kindIcon(j)}</span>
              <div class="jb-main">
                <b>{j.title}</b>
                {#if j.detail}<small class="jb-detail">{j.detail}</small>{/if}
                {@render bar(j)}
                {#if j.meta}<small class="jb-meta">{j.meta}</small>{/if}
              </div>
              <div class="jb-acts">
                {#if j.id === "reviews"}
                  {#if j.state === "paused"}
                    <button class="jb-act" title="Resume the reviews" aria-label="Resume" onclick={() => pause(false)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg></button>
                  {:else}
                    <button class="jb-act" title="Pause the reviews: what is done is kept" aria-label="Pause" onclick={() => pause(true)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg></button>
                  {/if}
                  <button class="jb-act" title="Your games" aria-label="Your games" onclick={() => go("games")}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg></button>
                {:else}
                  <button class="jb-act" title="Stop the setup" aria-label="Cancel" onclick={cancelSetup}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg></button>
                  <button class="jb-act" title="Set up" aria-label="Set up" onclick={() => go("setup")}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg></button>
                {/if}
              </div>
            </div>
          {/each}
        {/if}
        {#if finished.length}
          <div class="jb-section">
            <span>Completed</span>
            <button class="jb-clear" onclick={clear}>Clear</button>
          </div>
          {#each finished as j (j.id)}
            <div class="jb-row done" class:failed={j.state === "failed"}>
              <span class="jb-kind">{@render kindIcon(j)}</span>
              <div class="jb-main">
                <b>{j.title}</b>
                {#if j.detail}<small class="jb-detail">{j.detail}</small>{/if}
                {#if j.meta}<small class="jb-meta">{j.meta}</small>{/if}
              </div>
              <div class="jb-acts">
                <button class="jb-act" title={j.id === "reviews" ? "Your games" : "Set up"} aria-label={j.id === "reviews" ? "Your games" : "Set up"} onclick={() => go(j.id === "reviews" ? "games" : "setup")}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg></button>
              </div>
            </div>
          {/each}
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .jobs { position: relative; flex: 0 0 auto; }
  /* The bell: round as the menu button, the jobs' progress a ring around it. */
  .jb-bell {
    position: relative; display: grid; place-items: center; width: 40px; height: 40px; padding: 0;
    border: 0; border-radius: 50%; background: transparent; color: var(--ink); cursor: pointer;
  }
  .jb-bell:hover, .jb-bell[aria-expanded="true"] { background: var(--panel2); }
  .jb-icon { width: 20px; height: 20px; fill: currentColor; }
  .jb-bell.on .jb-icon { color: var(--accent); }
  .jb-ring { position: absolute; inset: 2px; width: 36px; height: 36px; transform: rotate(-90deg); }
  .jb-ring circle { fill: none; stroke-width: 2.5; }
  .jb-ring .track { stroke: var(--line); }
  .jb-ring .fill { stroke: var(--accent); stroke-linecap: round; transition: stroke-dasharray 300ms ease; }
  .jb-dot { position: absolute; top: 7px; right: 7px; width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 2px var(--panel); }

  /* Hovered: a line on each job. */
  .jb-peek {
    position: absolute; top: calc(100% + 6px); right: 0; z-index: 40; width: 260px;
    display: flex; flex-direction: column; gap: 8px; padding: 10px 12px; border-radius: 8px;
    background: var(--panel2); border: 1px solid var(--line); box-shadow: 0 8px 24px #0008; pointer-events: none;
  }
  .jb-peek-row { display: flex; flex-direction: column; gap: 3px; }
  .jb-peek-row b { font-size: 13px; }
  .jb-peek-row small { font-size: 12px; color: var(--muted); }

  /* Opened: the list, as a browser's downloads panel is laid out. */
  .jb-panel {
    position: absolute; top: calc(100% + 6px); right: 0; z-index: 40; width: min(380px, calc(100vw - 24px));
    max-height: min(70vh, 520px); overflow: auto; border-radius: 12px;
    background: var(--panel2); border: 1px solid var(--line); box-shadow: 0 12px 32px #000a;
  }
  .jb-head { display: flex; align-items: center; justify-content: space-between; padding: 14px 12px 8px 16px; font-size: 16px; font-weight: 700; }
  .jb-x { display: grid; place-items: center; width: 30px; height: 30px; padding: 0; border: 0; border-radius: 50%; background: transparent; color: var(--muted); cursor: pointer; }
  .jb-x:hover { background: var(--panel3); color: var(--ink); }
  .jb-x svg { width: 18px; height: 18px; fill: currentColor; }
  .jb-section {
    display: flex; align-items: center; justify-content: space-between; padding: 8px 16px 6px;
    font-size: 12px; font-weight: 700; color: var(--muted); border-top: 1px solid var(--line);
  }
  .jb-clear { padding: 2px 6px; border: 0; background: transparent; color: var(--accent); font: inherit; font-weight: 600; cursor: pointer; }
  .jb-clear:hover { color: var(--ink); }
  .jb-row { display: flex; align-items: flex-start; gap: 12px; padding: 10px 12px 12px 16px; border-top: 1px solid color-mix(in srgb, var(--line) 50%, transparent); }
  .jb-kind { flex: 0 0 auto; display: grid; place-items: center; width: 28px; height: 28px; margin-top: 2px; border-radius: 6px; background: var(--panel3); color: var(--muted); }
  .jb-kind svg { width: 17px; height: 17px; fill: currentColor; }
  .jb-main { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 4px; }
  .jb-main b { font-size: 13.5px; }
  .jb-detail { font-size: 12px; color: var(--muted); line-height: 1.35; overflow-wrap: anywhere; }
  .jb-meta { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .jb-row.failed b { color: var(--error); }
  .jb-bar { display: block; height: 5px; border-radius: 3px; background: var(--panel3); overflow: hidden; }
  .jb-bar i { display: block; height: 100%; background: var(--accent); border-radius: 3px; transition: width 400ms ease; }
  .jb-bar.paused i { background: var(--muted); }
  .jb-acts { flex: 0 0 auto; display: flex; gap: 2px; margin-top: 1px; }
  .jb-act { display: grid; place-items: center; width: 30px; height: 30px; padding: 0; border: 0; border-radius: 6px; background: transparent; color: var(--muted); cursor: pointer; }
  .jb-act:hover { background: var(--panel3); color: var(--ink); }
  .jb-act svg { width: 18px; height: 18px; fill: currentColor; }
</style>
