<script lang="ts">
  // The header: start/stop and the status; the board's orientation as the coach reads it
  // and a re-read; the game menu (what the board cannot tell, and forgetting); the data
  // menu (the caches, the profile); the voice (mute, which one).
  import { onMount } from "svelte";
  import { connect, get, panel } from "./api";
  import { store } from "./panel.svelte";
  import { inShell, openWindow } from "./shell.svelte";

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

  let openMenu = $state<"game" | "data" | null>(null);
  const voices = $derived(store.state.voices || []);
  const shown = $derived(store.bottom);

  function closeMenus() { openMenu = null; }
  async function resetProfile() {
    closeMenus();
    if (window.confirm("Reset the profile? Every game, verdict, puzzle attempt and insight the player model holds is deleted. This cannot be undone.")) {
      await store.resetProfile();
    }
  }
</script>

<svelte:document onclick={closeMenus} onkeydown={(e) => { if (e.key === "Escape") closeMenus(); }} />

<header id="bar">
  <div class="group">
    <span class="brand">sidecoach</span>
    <button class="primary" class:running={store.running} onclick={() => store.toggleRunning()}>{store.running ? "Stop" : "Start"}</button>
    <span class="status">{store.state.status || ""}</span>
    {#if missing.length}
      <a class="setup-banner" href="/setup" target="_blank" rel="noopener" title={missing.join(", ")} onclick={openSetup}>Set up: {missing.length} item{missing.length === 1 ? "" : "s"} missing →</a>
    {/if}
  </div>
  <div class="group controls">
    <div class="ctlgroup" role="group" aria-label="Board">
      <span class="ctllabel">board</span>
      <div class="seg" title="Which side is at the bottom of the board the coach reads. For a game seen from its start this only turns the panel's board around; for a puzzle or a game joined mid-way it re-reads the screen the other way up.">
        <button class="segbtn" aria-pressed={shown === "white"} onclick={() => { if (shown !== "white") panel.flip(); }}>&#9651; white</button>
        <button class="segbtn" aria-pressed={shown === "black"} onclick={() => { if (shown !== "black") panel.flip(); }}>&#9651; black</button>
      </div>
      <button class="ghost" title="Read the position on the screen now (a puzzle, an analysis board, a game in progress). Done on its own when a board is up and no game is followed." onclick={() => panel.analyze()}>Re-read</button>
    </div>
    <div class="ctlgroup menu">
      <span class="ctllabel">game</span>
      <button class="ghost menubtn" aria-haspopup="menu" aria-expanded={openMenu === "game"} title="End the game when the board cannot tell, or forget it" onclick={(e) => { e.stopPropagation(); openMenu = openMenu === "game" ? null : "game"; }}>End&hellip; &#9662;</button>
      <div class="menulist" role="menu" hidden={openMenu !== "game"}>
        <button role="menuitem" onclick={() => { closeMenus(); panel.end("resigned"); }}>I resigned / lost on time</button>
        <button role="menuitem" onclick={() => { closeMenus(); panel.end("won"); }}>Opponent resigned / flagged</button>
        <button role="menuitem" onclick={() => { closeMenus(); panel.end("draw"); }}>Draw agreed</button>
        <div class="menusep" role="separator"></div>
        <button role="menuitem" onclick={() => { closeMenus(); panel.end("forget"); }}>Forget this game</button>
      </div>
    </div>
    <div class="ctlgroup menu">
      <span class="ctllabel">data</span>
      <button class="ghost menubtn" aria-haspopup="menu" aria-expanded={openMenu === "data"} title="The player model and the caches" onclick={(e) => { e.stopPropagation(); openMenu = openMenu === "data" ? null : "data"; }}>Reset&hellip; &#9662;</button>
      <div class="menulist" role="menu" hidden={openMenu !== "data"}>
        <button role="menuitem" title="The theme cache and the puzzle store; both come back on their own" onclick={() => { closeMenus(); panel.clearCaches(); }}>Clear caches</button>
        <div class="menusep" role="separator"></div>
        <button role="menuitem" data-action="profile" title="Every game, verdict, attempt and insight the profile holds" onclick={resetProfile}>Reset profile&hellip;</button>
      </div>
    </div>
    <div class="ctlgroup" role="group" aria-label="Voice">
      <span class="ctllabel">voice</span>
      <button id="mute" class="iconbtn small" aria-pressed={!!store.state.muted} title={store.state.muted ? "Unmute the coach's voice" : "Mute the coach's voice"} onclick={() => store.setMuted(!store.state.muted)}>
        <svg class="ic-on" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
        <svg class="ic-off" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3L3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4L9.9 6.1 12 8.2V4z"/></svg>
      </button>
      <select title="The coach's voice" value={store.state.voice} onchange={(e) => panel.setVoice((e.currentTarget as HTMLSelectElement).value)}>
        {#each voices as v (v)}<option value={v}>{v}</option>{/each}
      </select>
    </div>
  </div>
</header>
