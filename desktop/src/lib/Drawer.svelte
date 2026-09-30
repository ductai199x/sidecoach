<script lang="ts">
  // Every control but Start, in a drawer from the left: the board's side and a re-read,
  // ending or forgetting the game, the panes, the caches and the profile, the voice when
  // there is one, the settings and the setup -- each a row with room for a thumb. The phone
  // had it first; on the desktop it took the menu bar's place (Tai, 2026-09-24), and there
  // it carries what only the shell has too: the terminal, always on top, the debug tools
  // (the dump that shows a problem first), updates, quitting and the version.
  //
  // It is a history entry (SvelteKit's shallow routing), so the phone's Back closes it,
  // as it closes a drawer in any app; a tap outside it, Esc, or any of its rows too.
  import { onMount } from "svelte";
  import { pushState } from "$app/navigation";
  import { page } from "$app/state";
  import { panel } from "./api";
  import { store } from "./panel.svelte";
  import {
    alwaysOnTop, appVersion, checkForUpdates, inShell, openDataFolder, openInBrowser, openWindow,
    PANES, panes, quitApp, resetLayout, setAlwaysOnTop, terminal, togglePane, toggleTerminal,
  } from "./shell.svelte";
  import { voiceLabel } from "./voices";

  let { missing = [] }: { missing?: string[] } = $props();

  const open = $derived(!!page.state.drawer);
  const shown = $derived(store.bottom);
  const voices = $derived(store.state.voices || []);
  const side = $derived(store.state.user_side as string | null | undefined);
  let closeButton = $state<HTMLButtonElement | null>(null);
  const shell = inShell();
  // The phone's app marks its page (LocalServer): Debug -> Dump is there too, the one debug
  // tool it has -- what the coach saw, saved for Tai to pull off the phone.
  const onPhone = typeof document !== "undefined" && document.documentElement.dataset.app === "phone";
  let onTop = $state(false);
  let version = $state("");
  let dumped = $state("");

  onMount(() => {
    if (shell) void appVersion().then((v) => (version = v));
  });

  async function toggleOnTop(): Promise<void> {
    onTop = !onTop;
    await setAlwaysOnTop(onTop);
    onTop = await alwaysOnTop();
  }

  async function toggleRecord(): Promise<void> {
    const r = await panel.setRecord(!store.state.record_wanted);
    store.state.record_wanted = r.on;
  }

  // The dump stays open with where it went: the one thing to pass on with a problem.
  async function dump(): Promise<void> {
    dumped = "saving…";
    try {
      const r = await panel.dump();
      dumped = r.path ? "saved to " + r.path : "saved";
    } catch (e) {
      dumped = "could not save: " + String(e);
    }
  }

  export function show(): void {
    if (!page.state.drawer) pushState("", { drawer: true });
  }

  function hide(): void {
    if (page.state.drawer) history.back();
  }

  // Close, then act: the drawer's entry leaves the history first.
  function act(fn: () => unknown): void {
    hide();
    void fn();
  }

  // Settings and Setup take the drawer's place in the history, so Back from them comes
  // back to the panel, not to an open drawer. In the shell they are windows of their own.
  function go(route: "settings" | "setup" | "games"): void {
    if (inShell()) {
      hide();
      openWindow(route);
    } else {
      location.replace("/" + route);
    }
  }

  async function resetProfile(): Promise<void> {
    hide();
    if (window.confirm("Reset the profile? Every game, verdict, puzzle attempt and insight the player model holds is deleted. This cannot be undone.")) {
      await store.resetProfile();
    }
  }

  $effect(() => {
    if (open) closeButton?.focus({ preventScroll: true });
    if (open && shell) void alwaysOnTop().then((v) => (onTop = v));
    if (!open) dumped = "";
  });
</script>

<svelte:window onkeydown={(e) => { if (open && e.key === "Escape") hide(); }} />

<div class="scrim" class:open aria-hidden="true" onclick={hide}></div>

<nav class="drawer" class:open aria-label="Menu" aria-hidden={!open} inert={!open}>
  <div class="dhead">
    <img src="/logo.svg" alt="" draggable="false" />
    <div class="dtitle">
      <div class="dname">Sidecoach</div>
      <div class="dsub">
        <span class="dot" class:on={store.running}></span>
        {store.state.status || (store.running ? "watching" : "idle")}{side ? ` · you play ${side}` : ""}
      </div>
    </div>
    <button class="dclose" bind:this={closeButton} aria-label="Close the menu" onclick={hide}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
    </button>
  </div>

  {#if missing.length}
    <div class="dsec">
      <button class="ditem setup" onclick={() => go("setup")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
        <span>Finish setting up<span class="hint">{missing.join(", ")}</span></span>
      </button>
    </div>
  {/if}

  <div class="dsec">
    <div class="dlabel">Board &middot; side at the bottom</div>
    <div class="seg dseg" title="Which side is at the bottom of the board the coach reads. For a game seen from its start this only turns the panel's board around; for a puzzle or a game joined mid-way it re-reads the screen the other way up.">
      <button class="segbtn" aria-pressed={shown === "white"} onclick={() => { if (shown !== "white") panel.flip(); }}>&#9651; White</button>
      <button class="segbtn" aria-pressed={shown === "black"} onclick={() => { if (shown !== "black") panel.flip(); }}>&#9651; Black</button>
    </div>
    <button class="ditem" onclick={() => act(() => panel.analyze())}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/></svg>
      <span>Re-read the screen<span class="hint">A puzzle, an analysis board, a game joined late</span></span>
    </button>
  </div>

  <div class="dsec">
    <div class="dlabel">Games</div>
    <button class="ditem" onclick={() => go("games")}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/></svg>
      <span>Your games<span class="hint">Review any game move by move: watched, chess.com, Lichess, a PGN</span></span>
    </button>
  </div>

  <div class="dsec">
    <div class="dlabel">End the game</div>
    <button class="ditem" onclick={() => act(() => panel.end("resigned"))}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.4 6 14 4H5v17h2v-7h5.6l.4 2h7V6z"/></svg>
      <span>I resigned or lost on time</span>
    </button>
    <button class="ditem" onclick={() => act(() => panel.end("won"))}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5h-2V3H7v2H5c-1.1 0-2 .9-2 2v1c0 2.55 1.92 4.63 4.39 4.94A5.01 5.01 0 0 0 11 15.9V19H7v2h10v-2h-4v-3.1a5.01 5.01 0 0 0 3.61-2.96C19.08 12.63 21 10.55 21 8V7c0-1.1-.9-2-2-2zM5 8V7h2v3.82C5.84 10.4 5 9.3 5 8zm14 0c0 1.3-.84 2.4-2 2.82V7h2v1z"/></svg>
      <span>My opponent resigned or flagged</span>
    </button>
    <button class="ditem" onclick={() => act(() => panel.end("draw"))}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 8.5h14v2.5H5zm0 4.5h14v2.5H5z"/></svg>
      <span>Draw agreed</span>
    </button>
    <button class="ditem warn" onclick={() => act(() => panel.end("forget"))}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM8 9h8v10H8V9zm7.5-5-1-1h-5l-1 1H5v2h14V4z"/></svg>
      <span>Forget this game<span class="hint">Wait for the next start position</span></span>
    </button>
  </div>

  <div class="dsec">
    <div class="dlabel">View</div>
    {#each PANES as pane (pane.id)}
      <button class="ditem" role="menuitemcheckbox" aria-checked={panes[pane.id]} onclick={() => togglePane(pane.id)}>
        {#if panes[pane.id]}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"/></svg>
        {/if}
        <span>{pane.name}<span class="hint">{pane.hint}</span></span>
      </button>
    {/each}
    {#if shell}
      <button class="ditem" role="menuitemcheckbox" aria-checked={terminal.open} onclick={toggleTerminal}>
        {#if terminal.open}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"/></svg>
        {/if}
        <span>Terminal<span class="hint">The pipeline's log, under the status strip</span></span>
      </button>
      <button class="ditem" role="menuitemcheckbox" aria-checked={onTop} onclick={toggleOnTop}>
        {#if onTop}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"/></svg>
        {/if}
        <span>Always on top<span class="hint">Keep this window over the game's</span></span>
      </button>
    {/if}
    <button class="ditem" onclick={() => act(resetLayout)}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5V1L7 6l5 5V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8z"/></svg>
      <span>Reset layout<span class="hint">Every pane shown, every size back to its default</span></span>
    </button>
  </div>

  {#if voices.length}
    <div class="dsec">
      <div class="dlabel">Voice</div>
      <button class="ditem" aria-pressed={!!store.state.muted} onclick={() => store.setMuted(!store.state.muted)}>
        {#if store.state.muted}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16.5 12A4.5 4.5 0 0 0 14 8v2.2l2.5 2.5V12zM19 12a7 7 0 0 1-1.6 4.4l1.5 1.5A9 9 0 0 0 21 12a9 9 0 0 0-7-8.8v2.1A7 7 0 0 1 19 12zM4.3 3 3 4.3 7.7 9H3v6h4l5 5v-6.7l4.3 4.3a7 7 0 0 1-2.3 1.2v2.1a9 9 0 0 0 3.7-1.8l2 2L21 19.7 4.3 3zM12 4 9.9 6.1 12 8.2V4z"/></svg>
          <span>Unmute the coach</span>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
          <span>Mute the coach</span>
        {/if}
      </button>
      <label class="dselect">
        <span>Voice</span>
        <select value={store.state.voice} onchange={(e) => panel.setVoice((e.currentTarget as HTMLSelectElement).value)}>
          {#each voices as v (v)}<option value={v}>{voiceLabel(v)}</option>{/each}
        </select>
      </label>
    </div>
  {/if}

  <div class="dsec">
    <div class="dlabel">Data</div>
    <button class="ditem" onclick={() => act(() => panel.clearCaches())}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m19 8-4 4h3a6 6 0 0 1-8.8 5.3l-1.46 1.46A8 8 0 0 0 20 12h3l-4-4zM6 12a6 6 0 0 1 8.8-5.3l1.46-1.46A8 8 0 0 0 4 12H1l4 4 4-4H6z"/></svg>
      <span>Clear caches<span class="hint">The learned board styles; they come back on their own</span></span>
    </button>
    <button class="ditem danger" onclick={resetProfile}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 8c0-2.21-1.79-4-4-4S6 5.79 6 8s1.79 4 4 4 4-1.79 4-4zm3 2v2h6v-2h-6zM2 18v2h16v-2c0-2.66-5.33-4-8-4s-8 1.34-8 4z"/></svg>
      <span>Reset profile&hellip;</span>
    </button>
  </div>

  {#if onPhone && !shell}
    <div class="dsec">
      <div class="dlabel">Debug</div>
      <button class="ditem" onclick={dump}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6h-8l-2-2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2zm-6 10h-2v-3H9l4-4 4 4h-3v3z"/></svg>
        <span>Dump the last frames and the log<span class="hint">{dumped || "What the coach saw just now, saved to show a problem"}</span></span>
      </button>
    </div>
  {/if}

  {#if shell}
    <div class="dsec">
      <div class="dlabel">Debug</div>
      <button class="ditem" onclick={dump}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6h-8l-2-2H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2zm-6 10h-2v-3H9l4-4 4 4h-3v3z"/></svg>
        <span>Dump the last frames and the log<span class="hint">{dumped || "What the coach saw just now, saved to show a problem"}</span></span>
      </button>
      <button class="ditem" role="menuitemcheckbox" aria-checked={!!store.state.record_wanted} onclick={toggleRecord}>
        {#if store.state.record_wanted}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"/></svg>
        {/if}
        <span>Record sessions<span class="hint">Every session's frames and log to the data folder, to show a bug</span></span>
      </button>
      <button class="ditem" role="menuitemcheckbox" aria-checked={store.filters.verbose} onclick={() => (store.filters.verbose = !store.filters.verbose)}>
        {#if store.filters.verbose}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2zm-9 14-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>
        {:else}
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5v14H5V5h14m0-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z"/></svg>
        {/if}
        <span>Verbose terminal<span class="hint">Every line the pipeline writes, not only the notable ones</span></span>
      </button>
      <button class="ditem" onclick={() => act(openInBrowser)}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z"/></svg>
        <span>Open the panel in a browser</span>
      </button>
      <button class="ditem" onclick={() => act(openDataFolder)}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4H4a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-8l-2-2z"/></svg>
        <span>Open the data folder</span>
      </button>
    </div>
  {/if}

  <div class="dsec dfoot">
    <button class="ditem" onclick={() => go("settings")}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19.4 13a7.6 7.6 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-1.7-1L15 3.2h-4l-.3 2.7a7.6 7.6 0 0 0-1.7 1l-2.4-1-2 3.4L6.6 11a7.6 7.6 0 0 0 0 2l-2 1.6 2 3.4 2.4-1a7.6 7.6 0 0 0 1.7 1l.3 2.7h4l.3-2.7a7.6 7.6 0 0 0 1.7-1l2.4 1 2-3.4-2-1.6zM13 15a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"/></svg>
      <span>Settings</span>
    </button>
    {#if !missing.length}
      <button class="ditem" onclick={() => go("setup")}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/></svg>
        <span>Setup<span class="hint">The models and the engine</span></span>
      </button>
    {/if}
    {#if shell}
      <button class="ditem" onclick={() => act(() => checkForUpdates(true))}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M21 10.12h-6.78l2.74-2.82a7.04 7.04 0 0 0-9.88-.1 6.88 6.88 0 0 0 0 9.8 7.04 7.04 0 0 0 9.88 0A6.5 6.5 0 0 0 19 12.1h2a9.34 9.34 0 0 1-2.64 6.46 9.1 9.1 0 0 1-12.72 0A8.98 8.98 0 0 1 5.62 5.8a9.02 9.02 0 0 1 12.68 0L21 3v7.12z"/></svg>
        <span>Check for updates</span>
      </button>
      <button class="ditem" onclick={quitApp}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M13 3h-2v10h2V3zm4.83 2.17-1.42 1.42A6.92 6.92 0 0 1 19 12a7 7 0 1 1-11.42-5.42L6.17 5.17A8.93 8.93 0 0 0 3 12a9 9 0 0 0 18 0 8.97 8.97 0 0 0-3.17-6.83z"/></svg>
        <span>Quit Sidecoach</span>
      </button>
      {#if version}<div class="dversion">Sidecoach {version}</div>{/if}
    {/if}
  </div>
</nav>

<style>
  .scrim {
    position: fixed; inset: 0; z-index: 50;
    background: rgba(12, 11, 10, 0.55);
    backdrop-filter: blur(2px); -webkit-backdrop-filter: blur(2px);
    opacity: 0; pointer-events: none;
    transition: opacity 220ms ease;
  }
  .scrim.open { opacity: 1; pointer-events: auto; }

  .drawer {
    position: fixed; top: 0; bottom: 0; left: 0; z-index: 51;
    width: min(86vw, 340px);
    display: flex; flex-direction: column;
    overflow-y: auto; overscroll-behavior: contain;
    scrollbar-width: thin; scrollbar-color: var(--panel3) transparent;
    background: var(--panel);
    border-right: 1px solid var(--line);
    box-shadow: 10px 0 36px rgba(0, 0, 0, 0.5);
    transform: translateX(-105%);
    transition: transform 260ms cubic-bezier(0.2, 0.8, 0.2, 1);
    color: var(--ink);
  }
  .drawer.open { transform: none; }
  @media (prefers-reduced-motion: reduce) { .scrim, .drawer { transition: none; } }

  .dhead {
    display: flex; align-items: center; gap: 12px;
    padding: calc(18px + var(--inset-top, 0px)) 10px 16px 18px;
    border-bottom: 1px solid var(--line);
    background: radial-gradient(120% 110% at 0% 0%, rgba(242, 180, 65, 0.12) 0%, rgba(242, 180, 65, 0) 60%);
  }
  .dhead img { width: 40px; height: 40px; flex: 0 0 40px; filter: drop-shadow(0 0 12px rgba(242, 180, 65, 0.35)); }
  .dtitle { flex: 1 1 auto; min-width: 0; }
  .dname { font-size: 19px; font-weight: 750; letter-spacing: 0.2px; }
  /* The whole status, wrapped: the bar above shows it cut short, so here it is in full
     ("game over 1-0 · waiting for the next game · you play white"). The dot sits on the
     first line. */
  .dsub {
    display: flex; align-items: flex-start; gap: 7px; margin-top: 3px;
    font-size: 12.5px; line-height: 17px; color: var(--muted);
  }
  .dot { width: 7px; height: 7px; flex: 0 0 7px; margin-top: 5px; border-radius: 50%; background: var(--muted); opacity: 0.6; }
  .dot.on { background: var(--accent); opacity: 1; box-shadow: 0 0 0 3px rgba(129, 182, 76, 0.2); }
  .dclose {
    flex: 0 0 auto; width: 40px; height: 40px; padding: 0;
    display: inline-flex; align-items: center; justify-content: center;
    border: 0; border-radius: 50%; background: transparent; color: var(--muted); cursor: pointer;
  }
  .dclose:hover, .dclose:active { background: var(--panel2); color: var(--ink); }
  /* Focus moves here when the drawer opens (so a keyboard can close it); a quiet ring,
     not the panel's accent one, and none at all for a tap. */
  .dclose:focus { outline: none; }
  .dclose:focus-visible { box-shadow: 0 0 0 2px var(--line); }
  .dclose svg { width: 22px; height: 22px; fill: currentColor; }

  .dsec { padding: 12px 10px 8px; }
  .dsec + .dsec { border-top: 1px solid var(--line); }
  .dfoot { margin-top: auto; padding-bottom: calc(14px + var(--inset-bottom, 0px)); }
  .dlabel {
    padding: 2px 10px 8px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase;
    color: var(--muted);
  }

  .ditem {
    display: flex; align-items: center; gap: 16px;
    width: 100%; min-height: 48px; padding: 6px 10px;
    border: 0; border-radius: 10px; background: transparent;
    color: var(--ink); font: inherit; font-size: 15px; font-weight: 500; line-height: 1.3;
    text-align: left; cursor: pointer;
    -webkit-tap-highlight-color: transparent;
    transition: background 120ms ease;
  }
  .ditem:hover, .ditem:active, .ditem:focus-visible { background: var(--panel2); }
  .ditem svg { width: 21px; height: 21px; flex: 0 0 21px; fill: var(--muted); }
  .ditem .hint { display: block; margin-top: 2px; font-size: 12px; font-weight: 400; color: var(--muted); }
  .ditem.warn { color: var(--warn); }
  .ditem.warn svg { fill: var(--warn); }
  .ditem.danger { color: var(--error); }
  .ditem.danger svg { fill: var(--error); }
  .ditem.setup { background: rgba(242, 180, 65, 0.1); color: #f2b441; }
  .ditem.setup svg { fill: #f2b441; }

  .dseg { display: flex; width: calc(100% - 20px); margin: 0 10px 8px; border-radius: 10px; }
  .dseg .segbtn { flex: 1 1 0; padding: 11px 6px; font-size: 14px; white-space: nowrap; }

  .dselect {
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 6px 10px 4px 47px; font-size: 14px; color: var(--muted);
  }
  .dselect select { max-width: 60%; }
  .dversion { padding: 10px 10px 2px 47px; font-size: 12px; color: var(--muted); }
</style>
