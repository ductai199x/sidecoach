<script lang="ts">
  // The panel: the board as the pipeline sees it beside the move list and the coach, the
  // status strip and the debug terminal below. One store, polled every 250 ms, and the
  // layout logic of the old page: the board sized to the room it has, three dividers
  // to drag, the terminal that folds, the exclusion rectangle that keeps the vision off
  // our own board when the game shares this monitor.
  import { onMount } from "svelte";
  import Board from "$lib/Board.svelte";
  import Coach from "$lib/Coach.svelte";
  import Composer from "$lib/Composer.svelte";
  import Header from "$lib/Header.svelte";
  import Moves from "$lib/Moves.svelte";
  import Strip from "$lib/Strip.svelte";
  import Terminal from "$lib/Terminal.svelte";
  import { panel } from "$lib/api";
  import { fitBoard, layoutMode, splitter, stacked } from "$lib/layout";
  import { prefs, store } from "$lib/panel.svelte";
  import { perf } from "$lib/perf.svelte";
  import { buildShell, buildTray, checkForUpdates, inShell, panes } from "$lib/shell.svelte";

  let mainEl = $state<HTMLElement | null>(null);
  let footer = $state<HTMLElement | null>(null);
  let boardEl = $state<HTMLDivElement | null>(null);
  let infoEl = $state<HTMLElement | null>(null);
  let right = $state<HTMLElement | null>(null);
  let splitBoard = $state<HTMLElement | null>(null);
  let splitMoves = $state<HTMLElement | null>(null);
  let splitTerm = $state<HTMLElement | null>(null);

  // The debug area folds to its status strip. It starts folded (Tai, 2026-09-21: the
  // terminal is off by default; View > Terminal opens it) and the choice is remembered.
  let folded = $state(true);
  function toggleFold() {
    folded = !folded;
    prefs.setStr("fold", folded ? "1" : "0");
  }

  const shown = $derived(store.shown);
  const following = $derived(!store.state.following ? "waiting for a start" : store.state.kind === "position" ? "a position (puzzle or analysis)" : "a game");

  function refit() {
    fitBoard(mainEl, splitBoard, infoEl, prefs.get("board"));
  }

  // If the game shares this monitor, the pipeline could lock onto the panel's board
  // instead of the game's; the page tells the engine where its board is on screen.
  let lastRect = "";
  function reportBoardRect() {
    if (!boardEl || !panes.board) return;
    const r = boardEl.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    const chromeH = Math.max(0, window.outerHeight - window.innerHeight);
    const rect = {
      x: Math.round((window.screenX + r.left) * dpr),
      y: Math.round((window.screenY + chromeH + r.top) * dpr),
      w: Math.round(r.width * dpr),
      h: Math.round(r.height * dpr),
    };
    const key = JSON.stringify(rect);
    if (key !== lastRect) { lastRect = key; panel.exclude(rect); }
  }

  const typing = () => ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName ?? "");

  onMount(() => {
    layoutMode();
    const saved = prefs.getStr("fold");
    folded = saved === null ? true : saved === "1";
    document.documentElement.classList.toggle("noboard", !panes.board);
    refit();
    const onResize = () => refit();
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
        const room = Math.max(80, (right?.clientHeight ?? 0) - 260);
        const h = Math.max(80, Math.min(room, start + d));
        right?.style.setProperty("--moves", h + "px"); prefs.set("moves", h);
      },
      () => { right?.style.removeProperty("--moves"); prefs.set("moves", null); }));
    if (prefs.get("moves") !== null && right) right.style.setProperty("--moves", prefs.get("moves") + "px");
    if (splitTerm) cleanups.push(splitter(splitTerm, "y", () => (footer?.querySelector(store.pane === "terminal" ? ".terminal" : ".conversation") as HTMLElement | null)?.getBoundingClientRect().height ?? 0,
      (start, d) => {
        const h = Math.max(60, Math.min(window.innerHeight * 0.6, start - d));
        root.style.setProperty("--term", h + "px"); prefs.set("term", h);
      },
      () => { root.style.removeProperty("--term"); prefs.set("term", null); }));
    if (prefs.get("term") !== null) root.style.setProperty("--term", prefs.get("term") + "px");

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && store.pinnedPly !== null) store.showLive();
      if (e.code === "Space" && !e.repeat && !typing()) { e.preventDefault(); store.setPtt(true); }
    };
    const onKeyUp = (e: KeyboardEvent) => { if (e.code === "Space" && !typing()) { e.preventDefault(); store.setPtt(false); } };
    const onBlur = () => store.setPtt(false);
    document.addEventListener("keydown", onKeyDown);
    document.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", onBlur);

    const rectTimer = setInterval(reportBoardRect, 250);
    store.start();
    const label = new URLSearchParams(location.search).get("perf");
    if (label) perf.start(label);
    if (inShell()) {
      buildShell({ shown: () => !folded, toggle: toggleFold });
      buildTray();
      // Once a day, quietly; Help > Check for updates asks out loud.
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

  $effect(() => {
    document.documentElement.classList.toggle("noboard", !panes.board);
    void folded;
    refit();
  });
</script>

<svelte:head><title>sidecoach</title></svelte:head>

<Header />

<main bind:this={mainEl}>
  <section id="left" hidden={!panes.board}>
    <Board fen={shown.fen} bottom={store.bottom} lastMove={shown.lastMove} marks={store.marks} bind:board={boardEl} />
    <div class="gameinfo" bind:this={infoEl}>
      <span class="strong">{store.state.user_side ? "you play " + store.state.user_side : "—"}</span>
      <span>ply {store.ply}</span>
      <span>{following}</span>
    </div>
  </section>

  <div id="splitBoard" class="vsplit" bind:this={splitBoard} title="Drag to resize the board; double-click to reset" hidden={!panes.board}></div>

  <section id="right" bind:this={right}>
    {#if panes.moves}
      <div class="panel-title">Moves</div>
      <Moves />
      <div class="hsplit" bind:this={splitMoves} title="Drag to resize the move list; double-click to reset"></div>
    {/if}
    <div class="panel-title">Coach</div>
    <Coach />
    <Composer />
  </section>
</main>

<footer bind:this={footer} class:folded>
  <div class="tsplit" bind:this={splitTerm} title="Drag to resize the terminal; double-click to reset"></div>
  {#if panes.strip}<Strip {folded} onfold={toggleFold} />{/if}
  <Terminal />
</footer>
