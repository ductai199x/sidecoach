<script lang="ts">
  // The header: one row everywhere -- the menu, the knight and the name, the status, then the
  // controls a player needs at any moment: the coach's voice and, on the desktop, keeping the
  // window on top, then Start. The rest is in the drawer the menu button opens (Tai,
  // 2026-09-24: "replace the ugly top bar with the collapsible nav bar on the left too";
  // 2026-10-01: muting must not take the drawer, a scroll, a click and a close).
  import { onMount } from "svelte";
  import { connect, get } from "./api";
  import { store } from "./panel.svelte";
  import { alwaysOnTop, inShell, openWindow, setAlwaysOnTop } from "./shell.svelte";
  import Drawer from "./Drawer.svelte";
  import Jobs from "./Jobs.svelte";

  let drawer = $state<ReturnType<typeof Drawer> | null>(null);
  const shell = inShell();
  const voiced = $derived(!!store.state.voices?.length);
  const muted = $derived(!!store.state.muted);

  // Kept over the game's window (the desktop only): the pin, pressed while it is.
  let onTop = $state(false);
  async function toggleOnTop(): Promise<void> {
    await setAlwaysOnTop(!onTop);
    onTop = await alwaysOnTop();
  }
  onMount(() => {
    if (shell) void alwaysOnTop().then((v) => (onTop = v));
  });

  // What the app still needs (models, Stockfish): a banner until the setup is done.
  let missing = $state<string[]>([]);
  onMount(() => {
    let timer: ReturnType<typeof setTimeout> | null = null;
    const check = async () => {
      let again = 60000;
      try {
        await connect();
        const st = await get<{ items: { title: string; present: boolean; optional: boolean }[]; ready: boolean }>("/api/setup/status");
        missing = st.items.filter((i) => !i.present && !i.optional).map((i) => i.title);
        if (missing.length) again = 5000;
      } catch {
        again = 3000; // the sidecar is not up yet
      }
      timer = setTimeout(check, again);
    };
    check();
    return () => { if (timer) clearTimeout(timer); };
  });
  function openSetup(e: MouseEvent) {
    if (inShell()) { e.preventDefault(); openWindow("setup"); }
  }
</script>

<header id="bar">
  <div class="group">
    <button class="menu-toggle" aria-label="Menu" title="Menu" onclick={() => drawer?.show()}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 6h18v2H3zm0 5h18v2H3zm0 5h18v2H3z"/></svg>
      {#if missing.length}<span class="badge" aria-label="setup unfinished"></span>{/if}
    </button>
    <img class="brand-mark" src="/logo.svg" alt="" draggable="false" />
    <span class="brand">sidecoach</span>
    <span class="status" class:live={store.running}>{store.state.status || ""}</span>
    {#if missing.length}
      <a class="setup-banner" href="/setup" target="_blank" rel="noopener" title={missing.join(", ")} onclick={openSetup}>Set up: {missing.length} item{missing.length === 1 ? "" : "s"} missing →</a>
    {/if}
    <div class="ends">
      <Jobs />
      {#if voiced}
        <button class="iconbtn hbtn" class:off={muted} aria-pressed={muted} aria-label={muted ? "Unmute the coach" : "Mute the coach"} title={muted ? "Unmute the coach (M)" : "Mute the coach (M)"} onclick={() => store.setMuted(!muted)}>
          {#if muted}
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3 3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4 9.9 6.1 12 8.2V4z" /></svg>
          {:else}
            <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z" /></svg>
          {/if}
        </button>
      {/if}
      {#if shell}
        <button class="iconbtn hbtn" class:on={onTop} aria-pressed={onTop} aria-label="Keep Sidecoach on top" title={onTop ? "Kept on top of other windows: press to let it go behind" : "Keep Sidecoach on top of the game's window"} onclick={toggleOnTop}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 9V4h1a1 1 0 0 0 0-2H7a1 1 0 0 0 0 2h1v5a3 3 0 0 1-3 3v2h5.97v7l1 1 1-1v-7H19v-2a3 3 0 0 1-3-3z" /></svg>
        </button>
      {/if}
      <button class="primary" class:running={store.running} onclick={() => store.toggleRunning()}>{store.running ? "Stop" : "Start"}</button>
    </div>
  </div>
</header>

<Drawer bind:this={drawer} {missing} />

<style>
  /* One row: the menu, the knight and the name, the status, Start at the far end. */
  #bar {
    flex-wrap: nowrap; min-height: 56px; box-sizing: border-box;
    padding: 6px 12px 6px 6px;
  }
  #bar > .group { flex: 1 1 auto; flex-wrap: nowrap; gap: 8px; min-width: 0; }
  .menu-toggle {
    position: relative; display: inline-flex; align-items: center; justify-content: center;
    flex: 0 0 auto; width: 44px; height: 44px; padding: 0;
    border: 0; border-radius: 50%; background: transparent; color: var(--ink); cursor: pointer;
    -webkit-tap-highlight-color: transparent;
  }
  .menu-toggle:hover, .menu-toggle:active { background: var(--panel2); }
  .menu-toggle svg { width: 24px; height: 24px; fill: currentColor; }
  .badge {
    position: absolute; top: 9px; right: 9px; width: 9px; height: 9px; border-radius: 50%;
    background: #f2b441; box-shadow: 0 0 0 2px var(--panel);
  }
  .brand-mark {
    display: block; width: 26px; height: 26px; flex: 0 0 26px;
    filter: drop-shadow(0 0 8px rgba(242, 180, 65, 0.3));
  }
  #bar .brand { font-size: 17px; margin-right: 2px; }
  /* A block, not a flex box: an ellipsis is drawn only in a block's own line, so a long
     status ("game over 1-0 · waiting for the next game") ends in "…" rather than mid-word.
     The whole of it is in the drawer. */
  #bar .status {
    position: relative; display: block; min-width: 0; padding-left: 13px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  #bar .status::before {
    content: ""; position: absolute; left: 0; top: 50%; width: 7px; height: 7px;
    margin-top: -3.5px; border-radius: 50%; background: var(--muted); opacity: 0.6;
  }
  #bar .status.live::before {
    background: var(--accent); opacity: 1; box-shadow: 0 0 0 3px rgba(129, 182, 76, 0.2);
  }
  #bar .setup-banner { flex: 0 0 auto; white-space: nowrap; }
  /* A narrow window has no room for the banner: the dot on the menu button says it. */
  :global(html.stacked) #bar .setup-banner { display: none; }
  /* At the far end: the bell (only while a job runs; lib/Jobs.svelte), then Start. */
  #bar .ends { margin-left: auto; flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
  #bar button.primary { flex: 0 0 auto; margin-left: 4px; padding: 8px 18px; }
  /* The voice and the pin: round icons, as the review's header has them; muted is the warning
     colour (a coach that will not speak), kept on top the accent. */
  .hbtn.iconbtn { width: 38px; height: 38px; color: var(--muted); }
  .hbtn svg { width: 22px; height: 22px; fill: currentColor; }
  .hbtn.off { color: var(--warn); }
  .hbtn.on { color: var(--accent); background: #81b64c22; }
  /* A phone's header has room for the knight, the status and the controls, not the name too. */
  @media (max-width: 480px) { #bar .brand { display: none; } }
</style>
