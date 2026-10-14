<script lang="ts">
  // The panel: the board as the pipeline sees it beside the move list and the coach, the
  // status strip and the debug terminal below. One store, polled every 250 ms, and the
  // layout logic of the old page: the board sized to the room it has, three dividers
  // to drag, the terminal that folds, the exclusion rectangle that keeps the vision off
  // our own board when the game shares this monitor.
  import { onMount } from "svelte";
  import { invoke } from "@tauri-apps/api/core";
  import Board from "$lib/Board.svelte";
  import BoardTools from "$lib/BoardTools.svelte";
  import Coach from "$lib/Coach.svelte";
  import Composer from "$lib/Composer.svelte";
  import Header from "$lib/Header.svelte";
  import Moves from "$lib/Moves.svelte";
  import Strip from "$lib/Strip.svelte";
  import Terminal from "$lib/Terminal.svelte";
  import { panel } from "$lib/api";
  import { boardHandle, fitBoard, layoutMode, mobileBoard as sizeMobileBoard, splitter } from "$lib/layout";
  import { prefs, store } from "$lib/panel.svelte";
  import { perf } from "$lib/perf.svelte";
  import { buildTray, checkForUpdates, inShell, onPhone, pageOrigin, panes, terminal, toggleTerminal } from "$lib/shell.svelte";

  let mainEl = $state<HTMLElement | null>(null);
  let footer = $state<HTMLElement | null>(null);
  let boardEl = $state<HTMLDivElement | null>(null);
  let infoEl = $state<HTMLElement | null>(null);
  let right = $state<HTMLElement | null>(null);
  let splitBoard = $state<HTMLElement | null>(null);
  let splitMoves = $state<HTMLElement | null>(null);
  let splitTerm = $state<HTMLElement | null>(null);
  let splitMobile = $state<HTMLElement | null>(null);

  // Stacked, the board's side is the player's: the handle under it drags it, and the
  // coach takes the room it gives up. Never so big the coach, the ask box and the chips
  // leave the first screen, never tiny.
  const mobileBoard = (wanted: number | null) => sizeMobileBoard(mainEl, wanted);

  // The debug area folds to its status strip (the drawer's Terminal, the strip's button).
  const folded = $derived(!terminal.open);

  const shown = $derived(store.shown);

  // Side by side, the moves go to one line (as stacked) when the panel is too short for a list
  // that leaves the coach room, or when the list was dragged under four rows; dragged down past
  // them it is a list again. Never a list too short to read (Tai, 2026-10-01). As stacked, the
  // line and the coach need no titles.
  const LIST_MIN = 112;
  let rightH = $state(0);
  let movesPref = $state<number | null>(prefs.get("moves"));
  const shortPanel = $derived(rightH > 0 && rightH < 540);
  const oneline = $derived(shortPanel || (movesPref !== null && movesPref < LIST_MIN));

  function refit() {
    fitBoard(mainEl, splitBoard, infoEl, prefs.get("board"));
  }

  // If the game shares this monitor, the pipeline could lock onto the panel's board
  // instead of the game's; the page tells the engine where its board is on screen.
  let lastRect = "";
  function reportBoardRect() {
    const r = boardEl?.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    // In the shell, where its page begins on the screen (`pageOrigin`); in a browser tab, the
    // window's place and its chrome's height.
    const origin = pageOrigin();
    const chromeH = Math.max(0, window.outerHeight - window.innerHeight);
    const left = origin ? origin.x : window.screenX * dpr;
    const top = origin ? origin.y : (window.screenY + chromeH) * dpr;
    // No board on screen (its pane hidden, the window hidden or minimised), or the shell has
    // not said where the window is yet: nothing to mask.
    const rect = !r || !r.width || !panes.board || document.hidden || origin === null
      ? { x: 0, y: 0, w: 0, h: 0 }
      : {
        x: Math.round(left + r.left * dpr),
        y: Math.round(top + r.top * dpr),
        w: Math.round(r.width * dpr),
        h: Math.round(r.height * dpr),
      };
    const key = JSON.stringify(rect);
    if (key !== lastRect) { lastRect = key; panel.exclude(rect).catch(() => { lastRect = ""; }); }
  }

  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName ?? "");

  onMount(() => {
    layoutMode();
    document.documentElement.classList.toggle("noboard", !panes.board);
    refit();
    const onResize = () => { refit(); mobileBoard(prefs.get("mboard")); };
    window.addEventListener("resize", onResize);
    const ro = footer && window.ResizeObserver ? new ResizeObserver(refit) : null;
    if (ro && footer) ro.observe(footer);

    const cleanups: (() => void)[] = [];
    const root = document.documentElement;
    if (splitBoard) cleanups.push(splitter(splitBoard, "x", () => parseFloat(getComputedStyle(root).getPropertyValue("--board")) || 0,
      (start, d) => { prefs.set("board", Math.max(200, start + d)); refit(); },
      () => { prefs.set("board", null); refit(); }));
    if (splitMoves && right) cleanups.push(splitter(splitMoves, "y", () => (right?.querySelector(".moves-wrap") as HTMLElement | null)?.getBoundingClientRect().height ?? 0,
      (start, d) => {
        const room = Math.max(LIST_MIN, (right?.clientHeight ?? 0) - 260);
        const h = Math.max(40, Math.min(room, start + d));
        right?.style.setProperty("--moves", h + "px"); prefs.set("moves", h); movesPref = h;
      },
      () => { right?.style.removeProperty("--moves"); prefs.set("moves", null); movesPref = null; }));
    if (prefs.get("moves") !== null && right) right.style.setProperty("--moves", prefs.get("moves") + "px");
    const rightRo = right && window.ResizeObserver ? new ResizeObserver(() => { rightH = right?.clientHeight ?? 0; }) : null;
    if (rightRo && right) rightRo.observe(right);
    cleanups.push(() => rightRo?.disconnect());
    if (splitTerm) cleanups.push(splitter(splitTerm, "y", () => (footer?.querySelector(store.pane === "terminal" ? ".terminal" : ".conversation") as HTMLElement | null)?.getBoundingClientRect().height ?? 0,
      (start, d) => {
        const h = Math.max(60, Math.min(window.innerHeight * 0.6, start - d));
        root.style.setProperty("--term", h + "px"); prefs.set("term", h);
      },
      () => { root.style.removeProperty("--term"); prefs.set("term", null); }));
    if (prefs.get("term") !== null) root.style.setProperty("--term", prefs.get("term") + "px");

    // The phone's handle: the desktop's dividers are off when stacked, this one is only on.
    if (splitMobile) cleanups.push(boardHandle(splitMobile,
      (w) => { mobileBoard(w); prefs.set("mboard", w); },
      () => { prefs.set("mboard", null); mobileBoard(null); }));
    mobileBoard(prefs.get("mboard"));

    // The header's height, for the first screen's sum (it is one row stacked, but a row
    // whose height the font decides).
    const bar = document.getElementById("bar");
    const barRo = bar && window.ResizeObserver ? new ResizeObserver(() => {
      root.style.setProperty("--bar-h", bar.getBoundingClientRect().height + "px");
      mobileBoard(prefs.get("mboard"));
    }) : null;
    if (barRo && bar) barRo.observe(bar);
    cleanups.push(() => barRo?.disconnect());
    // And the rows that share the first screen with the board (the game line, the ask box and
    // the chips): when they grow or shrink mid-game, the board is fitted to them again.
    const sharedRo = window.ResizeObserver ? new ResizeObserver(() => mobileBoard(prefs.get("mboard"))) : null;
    for (const el of [infoEl, right?.querySelector(".composer")]) if (sharedRo && el) sharedRo.observe(el);
    cleanups.push(() => sharedRo?.disconnect());

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && store.pinnedPly !== null) store.showLive();
      // Space is hold-to-talk only while that mode is on; else it keeps its own job.
      if (e.code === "Space" && !e.repeat && !typing() && store.listenMode === "ptt") { e.preventDefault(); store.setPtt(true); }
      // The header's and the board's one-key controls: M mutes the coach, C turns the captions
      // on or off where there are any, F flips the board.
      if (typing() || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
      if (e.key.toLowerCase() === "m" && store.state.voices?.length) store.setMuted(!store.state.muted);
      if (e.key.toLowerCase() === "c" && store.state.captions && (inShell() || onPhone())) {
        store.setCaptions(!store.state.captions.on);
      }
      if (e.key.toLowerCase() === "f") void panel.flip();
    };
    const onKeyUp = (e: KeyboardEvent) => { if (e.code === "Space" && !typing() && store.listenMode === "ptt") { e.preventDefault(); store.setPtt(false); } };
    const onBlur = () => store.setPtt(false);
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);

    const rectTimer = setInterval(reportBoardRect, 250);
    store.start();
    const label = new URLSearchParams(location.search).get("perf");
    if (label) perf.start(label);
    if (inShell()) {
      buildTray();
      // Once a day, quietly; the drawer's Check for updates asks out loud.
      const last = Number(prefs.getStr("updateCheck") || 0);
      if (Date.now() - last > 24 * 3600 * 1000) {
        prefs.setStr("updateCheck", String(Date.now()));
        setTimeout(() => checkForUpdates(false), 20000);
      }
    }
    return () => {
      window.removeEventListener("resize", onResize);
      ro?.disconnect();
      for (const c of cleanups) c();
      document.removeEventListener("keydown", onKeyDown);
      document.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      clearInterval(rectTimer);
      store.stop();
    };
  });

  // The captions' windows once captions are on (Settings > Captions): built hidden, they show
  // and place themselves (routes/captions). Asked when they turn on, not at every poll.
  const captionsOn = $derived(store.state.captions?.on ?? false);
  $effect(() => {
    if (inShell() && captionsOn) invoke("open_captions").catch((e) => console.warn("captions:", e));
  });

  $effect(() => {
    document.documentElement.classList.toggle("noboard", !panes.board);
    void folded;
    refit();
  });
</script>

<svelte:head><title>sidecoach</title></svelte:head>

<Header />

{#if store.offline}
  <div class="offline" role="alert">
    <span>{store.offline}</span>
    {#if inShell() && store.stopped}<button onclick={() => store.restart()}>Restart</button>{/if}
  </div>
{/if}

<main bind:this={mainEl}>
  <section id="left" hidden={!panes.board}>
    <Board fen={shown.fen} bottom={store.bottom} lastMove={shown.lastMove} marks={store.marks} bind:board={boardEl} />
    <div class="msplit" bind:this={splitMobile} role="separator" aria-orientation="horizontal" aria-label="Board size" title="Drag to resize the board; double-tap to reset"></div>
    <div class="gameinfo" bind:this={infoEl}>
      {#if store.state.user_side}<span class="strong">You play {store.state.user_side === "white" ? "White" : "Black"}</span>{/if}
      {#if store.ply}<span>Move {Math.ceil(store.ply / 2)}</span>{/if}
      {#if store.state.kind === "position"}<span>A position: a puzzle or an analysis board</span>{/if}
      <BoardTools />
    </div>
  </section>

  <div id="splitBoard" class="vsplit" bind:this={splitBoard} role="separator" aria-orientation="vertical" aria-label="Board size" title="Drag to resize the board; double-click to reset" hidden={!panes.board}></div>

  <section id="right" bind:this={right}>
    {#if panes.moves}
      <div class="panel-title" hidden={oneline}>Moves</div>
      <Moves {oneline} />
      <div
        class="hsplit"
        bind:this={splitMoves}
        role="separator"
        aria-orientation="horizontal"
        aria-label="Move list size"
        hidden={shortPanel}
        title={oneline ? "Drag down to show the moves as a list; double-click to reset" : "Drag to resize the move list; double-click to reset"}
      ></div>
    {/if}
    <div class="panel-title" hidden={oneline}>Coach</div>
    <Coach />
    <Composer />
  </section>
</main>

<footer bind:this={footer} class:folded class:bare={folded && !panes.strip}>
  <div class="tsplit" bind:this={splitTerm} role="separator" aria-orientation="horizontal" aria-label="Terminal size" title="Drag to resize the terminal; double-click to reset"></div>
  {#if panes.strip}<Strip {folded} onfold={toggleTerminal} />{/if}
  <Terminal />
</footer>
