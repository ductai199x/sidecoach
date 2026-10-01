<script lang="ts">
  // The board's own tools, in the row under it (as chess.com's under its board): flip it,
  // re-read the screen, end the game -- what a player reaches for mid-game, in one click, not
  // behind the drawer (Tai, 2026-10-01: muting took the drawer, a scroll, a click and a close).
  // Ending the game opens a small menu of its four ways.
  import { panel } from "./api";
  import { store } from "./panel.svelte";

  let menuOpen = $state(false);
  let menuEl = $state<HTMLDivElement | null>(null);
  let toggleEl = $state<HTMLButtonElement | null>(null);
  const other = $derived(store.bottom === "white" ? "Black" : "White");

  function end(outcome: "resigned" | "won" | "draw" | "forget"): void {
    menuOpen = false;
    void panel.end(outcome);
  }
  // A press anywhere else closes the menu, as any menu.
  function outside(e: PointerEvent): void {
    if (menuOpen && !menuEl?.contains(e.target as Node) && !toggleEl?.contains(e.target as Node)) menuOpen = false;
  }
  function onKey(e: KeyboardEvent): void {
    if (menuOpen && e.key === "Escape") {
      menuOpen = false;
      toggleEl?.focus();
    }
  }
</script>

<svelte:window onpointerdown={outside} onkeydown={onKey} />

<div class="board-tools" role="toolbar" aria-label="The board">
  <button class="iconbtn bt" title="Flip the board: {other} at the bottom (F)" aria-label="Flip the board" onclick={() => panel.flip()}>
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M16 17.01V10h-2v7.01h-3L15 21l4-3.99h-3zM9 3 5 6.99h3V14h2V6.99h3L9 3z" /></svg><span class="bt-label">Flip</span>
  </button>
  <button
    class="iconbtn bt"
    disabled={!store.running}
    title={store.running ? "Re-read the screen: a puzzle, an analysis board, a game joined late" : "Start first: then Sidecoach can re-read the screen"}
    aria-label="Re-read the screen"
    onclick={() => panel.analyze()}
  >
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M17.65 6.35A7.96 7.96 0 0 0 12 4a8 8 0 1 0 7.73 10h-2.08A6 6 0 1 1 12 6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z" /></svg><span class="bt-label">Re-read</span>
  </button>
  <div class="bt-end">
    <button
      bind:this={toggleEl}
      class="iconbtn bt"
      disabled={!store.running}
      aria-haspopup="menu"
      aria-expanded={menuOpen}
      title={store.running ? "End the game: a resignation, a flag or a draw Sidecoach did not see" : "No game is being followed"}
      aria-label="End the game"
      onclick={() => (menuOpen = !menuOpen)}
    >
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.4 6 14 4H5v17h2v-7h5.6l.4 2h7V6z" /></svg><span class="bt-label">End game</span>
      <svg class="bt-caret" viewBox="0 0 24 24" aria-hidden="true"><path d="M7 10l5 5 5-5z" /></svg>
    </button>
    {#if menuOpen}
      <div class="bt-menu" role="menu" aria-label="End the game" bind:this={menuEl}>
        <div class="bt-head">End the game</div>
        <button role="menuitem" onclick={() => end("resigned")}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14.4 6 14 4H5v17h2v-7h5.6l.4 2h7V6z" /></svg>
          <span>I resigned or lost on time</span>
        </button>
        <button role="menuitem" onclick={() => end("won")}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 5h-2V3H7v2H5c-1.1 0-2 .9-2 2v1c0 2.55 1.92 4.63 4.39 4.94A5.01 5.01 0 0 0 11 15.9V19H7v2h10v-2h-4v-3.1a5.01 5.01 0 0 0 3.61-2.96C19.08 12.63 21 10.55 21 8V7c0-1.1-.9-2-2-2zM5 8V7h2v3.82C5.84 10.4 5 9.3 5 8zm14 0c0 1.3-.84 2.4-2 2.82V7h2v1z" /></svg>
          <span>My opponent resigned or flagged</span>
        </button>
        <button role="menuitem" onclick={() => end("draw")}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 8.5h14v2.5H5zm0 4.5h14v2.5H5z" /></svg>
          <span>Draw agreed</span>
        </button>
        <div class="bt-sep" role="separator"></div>
        <button role="menuitem" class="warn" onclick={() => end("forget")}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM8 9h8v10H8V9zm7.5-5-1-1h-5l-1 1H5v2h14V4z" /></svg>
          <span>Forget this game<small>Wait for the next start position</small></span>
        </button>
      </div>
    {/if}
  </div>
</div>

<style>
  /* Small labelled buttons, quiet until pointed at; a narrow row (a phone, a small board) keeps
     the icons alone, each with its tooltip. */
  .board-tools { margin-left: auto; display: flex; align-items: center; gap: 4px; }
  .bt.iconbtn {
    width: auto; height: 30px; gap: 6px; padding: 0 10px 0 8px; border-radius: 6px;
    font-size: 12.5px; font-weight: 600; color: var(--muted); white-space: nowrap;
  }
  .bt svg { width: 18px; height: 18px; flex: 0 0 18px; fill: currentColor; }
  .bt .bt-caret { width: 16px; height: 16px; flex-basis: 16px; margin: 0 -4px 0 -4px; }
  .bt:hover:not(:disabled) { background: var(--panel2); color: var(--ink); }
  .bt:disabled { opacity: 0.4; cursor: default; }
  .bt:disabled:hover { background: transparent; }
  .bt[aria-expanded="true"] { background: var(--panel2); color: var(--ink); }
  @container gameinfo (max-width: 470px) {
    .bt-label, .bt .bt-caret { display: none; }
    .bt.iconbtn { width: 34px; padding: 0; justify-content: center; }
  }
  .bt-end { position: relative; }
  /* Opens upward, over the board, its right edge on the button's. */
  .bt-menu {
    position: absolute; right: 0; bottom: calc(100% + 6px); z-index: 30; width: max-content; max-width: calc(100vw - 24px);
    display: flex; flex-direction: column; padding: 6px;
    background: var(--panel2); border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 10px 28px #000b;
  }
  .bt-head { padding: 6px 10px 6px; font-size: 11px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase; color: var(--muted); }
  .bt-menu button {
    display: flex; align-items: center; gap: 12px; padding: 9px 10px; border: 0; border-radius: 6px;
    background: transparent; color: var(--ink); font: inherit; font-size: 14px; font-weight: 500; text-align: left; cursor: pointer;
    white-space: nowrap;
  }
  .bt-menu button:hover, .bt-menu button:focus-visible { background: var(--panel3); outline: none; }
  .bt-menu svg { width: 20px; height: 20px; flex: 0 0 20px; fill: var(--muted); }
  .bt-menu .warn, .bt-menu .warn svg { color: var(--warn); fill: var(--warn); }
  .bt-menu small { display: block; margin-top: 1px; font-size: 12px; font-weight: 400; color: var(--muted); }
  .bt-sep { height: 1px; margin: 4px 6px; background: var(--line); }
</style>
