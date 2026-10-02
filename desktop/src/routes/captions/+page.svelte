<script lang="ts">
  // The captions' box: the coach's words in a window of their own over the other windows, the
  // way Live Caption shows speech (the phone's `ui/Captions.kt`; Tai, 2026-09-29: "really
  // useful in fast games"). The words come up the moment the coach has them, newest at the
  // bottom; the box fades a few seconds after the voice stops. Always on screen, it is up all
  // the time, words or not, over every window (Tai, 2026-09-29: "is it always on screen or not????").
  // Drag it anywhere; drag an edge or a corner to size it, as a
  // window is (Tai, 2026-09-29: "give me the ability to tune the sizing both horizontal and
  // vertical"); double-click it for more lines; click it for the game's moves. Its buttons
  // -- mute, and the panel's three quick pills -- are always up (Tai, 2026-09-29: "they are
  // handy"). Drag it onto the cross at the foot of the screen to turn captions off. Its place
  // and size are kept between runs. It keeps off the chess board (the vision reads the board
  // from the screen, and a box over it would hide squares) and, unless always on screen, steps
  // aside while Sidecoach's own window is in front, which shows the words already. The shell
  // builds the window
  // (`open_captions`): over every other, frameless, and never focused, so a click here leaves
  // the keyboard with the game. Outside the shell (a browser, for a test) the box is drawn in
  // the page.
  import Badge from "$lib/Badge.svelte";
  import { onDestroy, onMount, tick } from "svelte";
  import { isTauri } from "@tauri-apps/api/core";
  import { connect, panel, put, type Captions, type PanelState } from "$lib/api";
  import {
    CONTROLS_MS, CROSS_LIFT, CROSS_REACH, DOUBLE_CLICK_MS, EDGE, EXPANDED_LINES, HIDE_AFTER_MS,
    LINES, MAX_LINES, MIN_WIDTH, PREVIEW, SLOP, accent, clear, fill, joined, runs, type Rect,
  } from "$lib/captions";

  const shell = isTauri();
  const BADGED = new Set(["brilliant", "great", "best", "inaccuracy", "mistake", "miss", "blunder"]);
  const PLAIN: Captions = { on: false, keep: false, text: "#FFFFFF", background: "#1F1F1F", opacity: 92, size: 17, preview: 0 };

  let live = $state<PanelState>({});
  const look = $derived(live.captions ?? PLAIN);

  let said = $state("");
  let shown = $state(false);
  let fading = $state(false);
  let movesShown = $state(false);
  let expanded = $state(false);
  const lineHeight = $derived(Math.round(look.size * 1.15));
  /** The box's width in CSS pixels (NaN until an edge sets it: two fifths of the monitor) and
   * the words' lines, as the edges last set them. */
  let boxWidth = $state(stored("width", NaN));
  let lines = $state(stored("lines", LINES, 1, MAX_LINES));
  const width = $derived(Number.isFinite(boxWidth) ? boxWidth : 560);
  const shownHeight = $derived(lineHeight * (expanded ? Math.max(lines, EXPANDED_LINES) : lines));

  let boxEl = $state<HTMLDivElement | null>(null);
  let wordsEl = $state<HTMLDivElement | null>(null);
  let stripEl = $state<HTMLDivElement | null>(null);

  let previewing = false; // the box up now is the sample line, which fades unless kept
  let front = false; // Sidecoach's own window in front
  let lastN = -1; // the last caption taken
  let lastPreview = -1;
  let lastAgain = -1; // the last caption_again seen
  let lastKeep: boolean | null = null;
  let turnedOff = false; // off from the box, until the saved setting comes back
  let busy = false; // a pointer on the box: it neither fades nor moves on its own meanwhile
  let hideTimer: ReturnType<typeof setTimeout> | null = null;
  let movesTimer: ReturnType<typeof setTimeout> | null = null;
  let clickTimer: ReturnType<typeof setTimeout> | null = null;
  let pollTimer: ReturnType<typeof setTimeout> | null = null;
  let cursor = 0;
  let alive = true;

  // ---- the words -------------------------------------------------------------------------

  /** One poll's state: the settings, a sample line when their look changed, the words given
   * to the voice since the last poll, and the fade that follows the voice. */
  function take(next: PanelState): void {
    live = next;
    const c = next.captions ?? PLAIN;
    const numbered = next.caption_lines ?? [];
    const newest = numbered.length ? numbered[numbered.length - 1][0] : 0;
    // Words from before this window (or while captions were off) are not owed; a change of
    // their look is, when it came in the last few seconds -- the window is built when they
    // are turned on, after the change that turned them on.
    if (lastN < 0 || !c.on) lastN = newest;
    const again = next.caption_again ?? 0;
    // A line captioned when it was held back, said now: the box comes back for it.
    const sayAgain = lastAgain >= 0 && again !== lastAgain && !front;
    lastAgain = again;
    if (lastPreview < 0) lastPreview = Date.now() - c.preview < 5000 ? 0 : c.preview;
    if (!c.on) turnedOff = false;
    if (!c.on || turnedOff) {
      if (shown) drop();
      return;
    }
    if (c.keep !== lastKeep) {
      // Kept on screen: back with the words, and no fade; not kept: the fade once quiet.
      lastKeep = c.keep;
      if (c.keep) {
        cancelHide();
        restore();
      } else if (front) {
        drop();
      } else if (shown && !next.speaking) {
        hideSoon();
      }
    }
    if (c.preview !== lastPreview) {
      lastPreview = c.preview;
      preview();
    }
    for (const [n, text, continues] of numbered) {
      if (n > lastN) {
        lastN = n;
        begin(text, false, !!continues);
      }
    }
    if (sayAgain && !shown && said) {
      cancelHide();
      void showBox();
      void scrollWords();
    }
    if (!shown) {
      restore();
      return;
    }
    // Up while the voice speaks; after it, HIDE_AFTER_MS, unless kept.
    if (next.speaking) cancelHide();
    else if (!hideTimer) hideSoon();
  }

  /** A line of the coach's, shown whole the moment it comes: its voice follows. While
   * Sidecoach is in front, which shows it already, it is kept for when Sidecoach is not: a box
   * kept on screen comes back with it (Tai, 2026-09-29: "I don't see shit on my screen" -- an
   * answer asked for in Sidecoach's window, the window minimised, and the kept box had nothing
   * to show). */
  function begin(text: string, force = false, continues = false): void {
    if (front && !force && !look.keep) {
      if (!previewing) said = joined(said, text, !continues);
      return;
    }
    // The sample line is no word of the coach's: the coach's first word takes its place.
    if (previewing && !force) said = "";
    previewing = force;
    said = joined(said, text, !continues);
    cancelHide();
    void showBox();
    void scrollWords();
  }

  /** A sample line over whatever is in front, Sidecoach's own settings window included: a
   * change to the captions shows the moment it is saved (Tai, 2026-09-23: "applies instantly
   * on save"). A fresh box, and it fades as any line does. */
  function preview(): void {
    drop();
    said = "";
    begin(PREVIEW, true);
  }

  /** Always on screen: the box up, with the words said so far or none yet, whatever window is
   * in front. */
  function restore(): void {
    if (!look.keep || !look.on || shown) return;
    void showBox();
    void scrollWords();
  }

  function hideSoon(): void {
    cancelHide();
    if (look.keep) return;
    hideTimer = setTimeout(hide, HIDE_AFTER_MS);
  }

  function cancelHide(): void {
    if (hideTimer) clearTimeout(hideTimer);
    hideTimer = null;
  }

  /** A pointer on the box let go: the fade waits its full time again. */
  function rest(): void {
    if (!live.speaking) hideSoon();
  }

  function hide(): void {
    hideTimer = null;
    if (busy) {
      hideSoon();
      return;
    }
    fading = true;
    setTimeout(() => { if (fading) drop(); }, 350);
  }

  function drop(): void {
    cancelHide();
    if (movesTimer) clearTimeout(movesTimer);
    movesShown = false;
    // The sample line is no word of the coach's: a box kept on screen does not bring it back.
    if (previewing) said = "";
    previewing = false;
    shown = false;
    fading = false;
    drops++;
    void win()?.then((w) => w?.hide());
    void hideCross();
  }

  async function scrollWords(): Promise<void> {
    await tick();
    if (wordsEl) wordsEl.scrollTop = wordsEl.scrollHeight;
  }

  // ---- the window ------------------------------------------------------------------------

  type Win = import("@tauri-apps/api/window").Window;
  let winCache: Promise<Win | null> | null = null;
  function win(): Promise<Win | null> | null {
    if (!shell) return null;
    winCache ??= import("@tauri-apps/api/window").then((m) => m.getCurrentWindow());
    return winCache;
  }

  /** Where the box is, in physical screen pixels, and the monitor it is on. */
  let pos = { x: 0, y: 0, w: 0, h: 0 };
  let screen: Rect = { left: 0, top: 0, right: 1920, bottom: 1080 };
  let scale = 1;

  const board = (): Rect | null => {
    const b = live.board_screen;
    return b ? { left: b[0], top: b[1], right: b[0] + b[2], bottom: b[1] + b[3] } : null;
  };

  /** The monitor under a point, its work area and scale; the first one when none is. */
  async function monitorAt(x: number, y: number): Promise<void> {
    const m = await import("@tauri-apps/api/window");
    const monitor = (await m.monitorFromPoint(x, y).catch(() => null)) ?? (await m.primaryMonitor().catch(() => null));
    if (!monitor) return;
    const area = monitor.workArea ?? { position: monitor.position, size: monitor.size };
    screen = {
      left: area.position.x,
      top: area.position.y,
      right: area.position.x + area.size.width,
      bottom: area.position.y + area.size.height,
    };
    scale = monitor.scaleFactor || 1;
  }

  /** The box dropped this many times: a show still on its way when it is dropped (Sidecoach
   * brought to the front meanwhile) stops there, and never leaves a window up with no box
   * drawn in it, taking the clicks meant for what is under it. */
  let drops = 0;

  async function showBox(): Promise<void> {
    if (shown && !fading) return;
    fading = false;
    const first = !shown;
    shown = true;
    const at = drops;
    const w = await win();
    if (!w || !first) return;
    const saved = { x: stored("x", NaN), y: stored("y", NaN) };
    await monitorAt(Number.isFinite(saved.x) ? saved.x : 0, Number.isFinite(saved.y) ? saved.y : 0);
    // Unsized, two fifths of its monitor, never under 560 px: the phone's four fifths of a
    // phone's screen would stretch across a desktop's.
    const across = (screen.right - screen.left) / scale;
    if (!Number.isFinite(boxWidth)) boxWidth = Math.min(across - 32, Math.max(560, across * 0.4));
    boxWidth = Math.min(boxWidth, across);
    await tick();
    const h = Math.round((boxEl?.getBoundingClientRect().height ?? 120) * scale);
    const width = Math.round((boxEl?.getBoundingClientRect().width ?? boxWidth) * scale);
    const x = Number.isFinite(saved.x) ? saved.x : screen.left + ((screen.right - screen.left) - width) / 2;
    const y = Number.isFinite(saved.y) ? saved.y : screen.bottom - Math.round(220 * scale);
    const [cx, cy] = clear(Math.round(x), Math.round(y), width, h, screen, board(), Math.round(8 * scale));
    pos = { x: cx, y: cy, w: width, h };
    await setBounds();
    if (at === drops) await w.show();
  }

  async function setBounds(): Promise<void> {
    const w = await win();
    if (!w) return;
    const { PhysicalPosition, PhysicalSize } = await import("@tauri-apps/api/dpi");
    await w.setSize(new PhysicalSize(pos.w, pos.h));
    await w.setPosition(new PhysicalPosition(Math.round(pos.x), Math.round(pos.y)));
  }

  /** The box's size changed -- an edge dragged, more lines, the moves, a text
   * size -- and the window follows it. Grown by what it shows, low on the screen it grows
   * upward, its foot where it was, as Live Caption's does; sized by an edge, it goes where the
   * edge put it. It stays on the screen either way. */
  async function reflow(): Promise<void> {
    if (!shown || moving || !boxEl) return;
    const box = boxEl.getBoundingClientRect();
    const w = Math.round(box.width * scale);
    const h = Math.round(box.height * scale);
    if (!w || !h || (w === pos.w && h === pos.h)) return;
    let y = pos.y;
    if (!sizing && y + pos.h / 2 > (screen.top + screen.bottom) / 2) y -= h - pos.h;
    y = Math.min(Math.max(y, screen.top), Math.max(screen.top, screen.bottom - h));
    pos = { ...pos, y, w, h };
    await setBounds();
  }

  // ---- the pointer: drag, click, double-click --------------------------------------------

  /** The pointer held by `el` until it is let go -- dragged past the window's edge, the box
   * still hears it; a pointer no longer down cannot be held, and nothing is lost. */
  function capture(el: HTMLElement | null, id: number, on: boolean): void {
    try {
      if (on) el?.setPointerCapture(id);
      else el?.releasePointerCapture(id);
    } catch { /* not down any more */ }
  }

  let moving = false;
  let dragged = false;
  let down = { x: 0, y: 0 };
  let start = { x: 0, y: 0 };
  /** The pointer on the screen, in physical pixels, while the box is carried: where it is let go
   * decides the cross, not the box. */
  let pointer = { x: 0, y: 0 };
  let grab = { x: 0, y: 0 };
  let moveFrame = 0;

  /** Which edges of the box a pointer is on: "n", "s", "e", "w" or two of them at a corner;
   * "" inside. */
  function edgeAt(e: PointerEvent): string {
    if (!boxEl) return "";
    const r = boxEl.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    return (y < EDGE ? "n" : y > r.height - EDGE ? "s" : "") + (x < EDGE ? "w" : x > r.width - EDGE ? "e" : "");
  }

  const CURSORS: Record<string, string> = {
    n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
    ne: "nesw-resize", sw: "nesw-resize", nw: "nwse-resize", se: "nwse-resize",
  };
  let hover = $state("");

  /** The edges being dragged, and the size and place they started from. */
  let sizing: { edges: string; width: number; lines: number; x: number; y: number } | null = null;

  function pressed(e: PointerEvent): void {
    if (e.button !== 0) return;
    busy = true;
    dragged = false;
    down = { x: e.screenX, y: e.screenY };
    start = { x: pos.x, y: pos.y };
    const ratio = window.devicePixelRatio || scale;
    grab = { x: pos.x + e.clientX * ratio, y: pos.y + e.clientY * ratio };
    const edges = edgeAt(e);
    sizing = edges ? { edges, width, lines: expanded ? Math.max(lines, EXPANDED_LINES) : lines, x: pos.x, y: pos.y } : null;
    if (sizing) cancelHide();
    capture(boxEl, e.pointerId, true);
  }

  function moved(e: PointerEvent): void {
    if (!busy) {
      hover = edgeAt(e);
      return;
    }
    const dx = e.screenX - down.x;
    const dy = e.screenY - down.y;
    if (sizing) {
      resize(dx, dy);
      return;
    }
    // A pointer that moves drags the box at once; a click stays where it is.
    if (!moving && Math.hypot(dx, dy) > SLOP) {
      moving = true;
      dragged = true;
      cancelHide();
      void showCross();
    }
    if (!moving || !shell) return;
    const ratio = window.devicePixelRatio || scale;
    pos = { ...pos, x: start.x + dx * ratio, y: start.y + dy * ratio };
    pointer = { x: grab.x + dx * ratio, y: grab.y + dy * ratio };
    if (!moveFrame) {
      moveFrame = requestAnimationFrame(() => {
        moveFrame = 0;
        void setBounds();
        void nearCross();
      });
    }
  }

  /** An edge dragged by (`dx`, `dy`) CSS pixels: the width follows it, the words' lines by
   * whole lines, and a west or north edge moves the box so its other side stays where it was. */
  function resize(dx: number, dy: number): void {
    if (!sizing) return;
    const { edges } = sizing;
    const across = shell ? (screen.right - screen.left) / scale : window.innerWidth - 32;
    const w = Math.min(across, Math.max(MIN_WIDTH, sizing.width + (edges.includes("e") ? dx : edges.includes("w") ? -dx : 0)));
    const grown = Math.round((edges.includes("s") ? dy : edges.includes("n") ? -dy : 0) / lineHeight);
    const n = Math.min(MAX_LINES, Math.max(1, sizing.lines + grown));
    boxWidth = w;
    lines = n;
    expanded = false;
    const ratio = window.devicePixelRatio || scale;
    pos = {
      ...pos,
      x: edges.includes("w") ? sizing.x + (sizing.width - w) * ratio : pos.x,
      y: edges.includes("n") ? sizing.y + (sizing.lines - n) * lineHeight * ratio : pos.y,
    };
    void scrollWords();
  }

  function released(e: PointerEvent): void {
    if (!busy) return;
    busy = false;
    capture(boxEl, e.pointerId, false);
    if (sizing) {
      sizing = null;
      store("width", width);
      store("lines", lines);
      store("x", pos.x);
      store("y", pos.y);
      rest();
      return;
    }
    if (moving) {
      moving = false;
      void dropped();
      return;
    }
    // Two clicks within DOUBLE_CLICK_MS: more lines, or fewer; one: the moves.
    if (clickTimer) {
      clearTimeout(clickTimer);
      clickTimer = null;
      expanded = !expanded;
      void scrollWords();
    } else {
      clickTimer = setTimeout(() => { clickTimer = null; toggleMoves(); }, DOUBLE_CLICK_MS);
    }
    rest();
  }

  // ---- the cross a box is dropped on to turn captions off ---------------------------------

  let cross: Win | null = null;
  let crossAt = { x: 0, y: 0 };
  let hot = false;
  /** A box being carried: the cross shows only while it still is, however quick the drop. */
  let carried = false;

  async function showCross(): Promise<void> {
    if (!shell) return;
    carried = true;
    const m = await import("@tauri-apps/api/window");
    cross ??= await m.Window.getByLabel("w-captions-cross");
    if (!cross) return;
    await monitorAt(pos.x + pos.w / 2, pos.y + pos.h / 2);
    crossAt = { x: (screen.left + screen.right) / 2, y: screen.bottom - CROSS_LIFT * scale };
    const size = Math.round(96 * scale);
    const { PhysicalPosition } = await import("@tauri-apps/api/dpi");
    await cross.setPosition(new PhysicalPosition(Math.round(crossAt.x - size / 2), Math.round(crossAt.y - size / 2)));
    hot = false;
    const { emitTo } = await import("@tauri-apps/api/event");
    await emitTo("w-captions-cross", "cross-hot", false).catch(() => {});
    if (carried) await cross.show();
  }

  async function nearCross(): Promise<void> {
    if (!cross) return;
    const now = Math.hypot(pointer.x - crossAt.x, pointer.y - crossAt.y) < CROSS_REACH * scale;
    if (now === hot) return;
    hot = now;
    const { emitTo } = await import("@tauri-apps/api/event");
    await emitTo("w-captions-cross", "cross-hot", hot).catch(() => {});
  }

  async function hideCross(): Promise<void> {
    carried = false;
    hot = false;
    await cross?.hide().catch(() => {});
  }

  /** A box let go: onto the cross, captions off; anywhere else, kept there, off the board. */
  async function dropped(): Promise<void> {
    const onCross = hot;
    await hideCross();
    if (dragged && onCross) {
      await turnOff();
      return;
    }
    await monitorAt(pos.x + pos.w / 2, pos.y + pos.h / 2);
    const [x, y] = clear(Math.round(pos.x), Math.round(pos.y), pos.w, pos.h, screen, board(), Math.round(8 * scale));
    pos = { ...pos, x, y };
    await setBounds();
    store("x", x);
    store("y", y);
    rest();
  }

  /** Captions off from the box itself: the setting follows, as the settings' switch would. */
  async function turnOff(): Promise<void> {
    turnedOff = true;
    drop();
    try {
      await put("/api/settings", { settings: { voice: { captions: false } }, secrets: {} });
    } catch (e) {
      console.warn("captions off:", e);
    }
  }

  // ---- the buttons, the moves --------------------------------------------------------------

  function toggleMoves(): void {
    movesShown = !movesShown;
    if (movesTimer) clearTimeout(movesTimer);
    if (movesShown) movesTimer = setTimeout(() => { movesShown = false; }, CONTROLS_MS);
    void scrollStrip();
    rest();
  }

  /** The moves touched: they stay their full time again, and the box its own. */
  function keepMoves(): void {
    if (movesTimer) clearTimeout(movesTimer);
    movesTimer = setTimeout(() => { movesShown = false; }, CONTROLS_MS);
    rest();
  }

  function toggleMute(): void {
    const now = !live.muted;
    live = { ...live, muted: now };
    void panel.setMuted(now).catch(() => {});
    rest();
  }

  /** A quick pill: what the panel's own asks, the answer spoken and captioned here. */
  function quick(name: "hint" | "move" | "line"): void {
    void panel[name]().catch(() => {});
    rest();
  }

  const moves = $derived.by(() => {
    const sans = live.moves ?? [];
    const labels = live.labels ?? {};
    return sans.map((san, i) => {
      const label = labels[String(i + 1)] ?? null;
      return { san, num: i % 2 === 0 ? `${i / 2 + 1}.` : "", last: i === sans.length - 1, label: label && BADGED.has(label) ? label : null };
    });
  });

  async function scrollStrip(): Promise<void> {
    await tick();
    if (stripEl) stripEl.scrollLeft = stripEl.scrollWidth;
  }

  // ---- what the box keeps between runs ----------------------------------------------------

  function stored(name: string, fallback: number, low = -Infinity, high = Infinity): number {
    try {
      const v = localStorage.getItem(`captions.${name}`);
      const n = v === null ? NaN : Number(v);
      return Number.isFinite(n) ? Math.min(high, Math.max(low, n)) : fallback;
    } catch {
      return fallback;
    }
  }

  function store(name: string, value: number): void {
    try { localStorage.setItem(`captions.${name}`, String(Math.round(value))); } catch { /* kept for this run only */ }
  }

  // ---- the poll ---------------------------------------------------------------------------

  async function pollOnce(): Promise<void> {
    if (!alive) return;
    try {
      const r = await panel.poll(cursor);
      cursor = r.cursor ?? cursor;
      if (r.state) take(r.state);
    } catch { /* the sidecar restarting: asked again */ }
    pollTimer = setTimeout(pollOnce, 250);
  }

  let unlisten: (() => void) | null = null;
  let observer: ResizeObserver | null = null;

  onMount(async () => {
    await connect();
    if (shell) {
      const { getCurrentWebviewWindow } = await import("@tauri-apps/api/webviewWindow");
      // Sidecoach's own window in front whenever that changes (the shell says), and now:
      // heard first, so a change while it is asked is not lost.
      let told = false;
      unlisten = await getCurrentWebviewWindow().listen<boolean>("panel-front", (e) => {
        told = true;
        front = !!e.payload;
        if (front && !look.keep) drop();
        else restore();
      });
      const m = await import("@tauri-apps/api/window");
      const focused = (await (await m.Window.getByLabel("main"))?.isFocused().catch(() => false)) ?? false;
      if (!told) front = focused;
    }
    observer = new ResizeObserver(() => void reflow());
    if (boxEl) observer.observe(boxEl);
    void pollOnce();
  });

  onDestroy(() => {
    alive = false;
    if (pollTimer) clearTimeout(pollTimer);
    cancelHide();
    unlisten?.();
    observer?.disconnect();
  });

</script>

<svelte:head><title>Sidecoach captions</title></svelte:head>

<div
  class="box"
  class:shown
  class:fading
  class:browser={!shell}
  bind:this={boxEl}
  style="--text: {look.text}; --accent: {accent(look.background)}; --fill: {fill(look.background, look.opacity)}; --size: {look.size}px; --line: {lineHeight}px; width: {width}px; cursor: {CURSORS[hover] ?? 'default'}"
  role="presentation"
  onpointerdown={pressed}
  onpointermove={moved}
  onpointerup={released}
  onpointercancel={released}
>
  <div class="buttons" role="toolbar" tabindex="-1" onpointerdown={(e) => e.stopPropagation()} onpointerup={(e) => e.stopPropagation()}>
    <button class="round" title={live.muted ? "Unmute the coach's voice" : "Mute the coach's voice"} onclick={toggleMute}>
      {#if live.muted}
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.6 3 2.7 2.7-1.4 1.4-2.7-2.7-2.7 2.7-1.4-1.4 2.7-2.7-2.7-2.7 1.4-1.4 2.7 2.7 2.7-2.7 1.4 1.4-2.7 2.7z"/></svg>
      {:else}
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3A4.5 4.5 0 0 0 14 8v8a4.5 4.5 0 0 0 2.5-4zM14 3.2v2.1a7 7 0 0 1 0 13.4v2.1a9 9 0 0 0 0-17.6z"/></svg>
      {/if}
    </button>
    <button class="chip" onclick={() => quick("hint")}>Hint</button>
    <button class="chip" onclick={() => quick("move")}>Best move</button>
    <button class="chip" onclick={() => quick("line")}>Line</button>
  </div>
  <div class="words" bind:this={wordsEl} style="height: {shownHeight}px">{#each runs(said) as r, i (i)}{#if r.kind}<span class="w-{r.kind}">{r.text}</span>{:else}{r.text}{/if}{/each}</div>
  {#if movesShown && moves.length}
    <div class="strip" bind:this={stripEl} onpointerdown={(e) => e.stopPropagation()} onpointerup={(e) => { e.stopPropagation(); keepMoves(); }} role="presentation">
      {#each moves as m, i (i)}
        {#if m.num}<span class="num">{m.num}</span>{/if}<span class="san" class:last={m.last}>{m.san}{#if m.label}<Badge kind={m.label} />{/if}</span>
      {/each}
    </div>
  {/if}
</div>

<style>
  :global(html), :global(body) { background: transparent !important; overflow: hidden; margin: 0; }
  :global(body::after) { display: none !important; }

  /* The box square, its fill rounded behind it: a pointer hits the corners it sizes by, which
     a rounded box would leave to the page under it. */
  .box {
    position: relative;
    isolation: isolate;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    padding: 12px 16px;
    color: var(--text);
    font-size: var(--size);
    line-height: var(--line);
    opacity: 0;
    transition: opacity 180ms ease-out;
    user-select: none;
    touch-action: none;
  }
  .box::before { content: ""; position: absolute; inset: 0; z-index: -1; border-radius: 20px; background: var(--fill); }
  .box.shown { opacity: 1; }
  .box.fading { opacity: 0; transition: opacity 350ms ease-in; }
  /* In a browser, for a test: drawn at the foot of the page. */
  .box.browser { position: fixed; left: 50%; bottom: 40px; transform: translateX(-50%); max-width: calc(100vw - 32px); }

  .words { overflow: hidden; white-space: pre-wrap; overflow-wrap: anywhere; }
  /* A move or a square the coach names, and a piece: what a player's eye looks for. Classes
     of their own: "piece" is the board's (panel.css, absolute at the top-left), and took every
     piece word out of the sentence into the box's corner (2026-10-01). */
  .words .w-move { color: var(--accent); font-weight: 700; }
  .words .w-piece { font-weight: 700; }

  .buttons { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
  .round {
    width: 34px; height: 34px; padding: 7px; border: 0; border-radius: 50%;
    background: rgba(255, 255, 255, 0.2); color: #fff; cursor: pointer;
  }
  .round svg { width: 100%; height: 100%; fill: currentColor; display: block; }
  .chip {
    padding: 7px 12px; border: 0; border-radius: 16px; background: rgba(255, 255, 255, 0.2);
    color: #fff; font-size: 14px; line-height: 1.2; cursor: pointer;
  }
  .round:hover, .chip:hover { background: rgba(255, 255, 255, 0.3); }

  .strip {
    margin-top: 8px; white-space: nowrap; overflow-x: auto; scrollbar-width: none;
    font-size: calc(var(--size) - 2px);
  }
  .strip::-webkit-scrollbar { display: none; }
  .num { color: #9a9a9a; margin: 0 4px 0 12px; }
  .num:first-child { margin-left: 0; }
  .san { margin-right: 8px; }
  .san.last { background: rgba(255, 255, 255, 0.2); font-weight: 700; border-radius: 3px; padding: 0 2px; }

</style>
