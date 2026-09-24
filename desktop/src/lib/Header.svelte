<script lang="ts">
  // The header: one row everywhere -- the menu, the knight and the name, Start, the status --
  // with every other control in the drawer the menu button opens. The phone had it first;
  // the desktop took it in place of its menu bar and its row of controls (Tai, 2026-09-24:
  // "replace the ugly top bar with the collapsible nav bar on the left too").
  import { onMount } from "svelte";
  import { connect, get } from "./api";
  import { store } from "./panel.svelte";
  import { inShell, openWindow } from "./shell.svelte";
  import Drawer from "./Drawer.svelte";

  let drawer = $state<ReturnType<typeof Drawer> | null>(null);

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
    <button class="primary" class:running={store.running} onclick={() => store.toggleRunning()}>{store.running ? "Stop" : "Start"}</button>
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
  #bar button.primary { margin-left: auto; flex: 0 0 auto; padding: 8px 18px; }
</style>
