// The panel's layout: side by side (board left, panel right, nothing scrolls but the
// panes) whenever the window is landscape and a board sized by its height or its width
// would be at least 220 px beside the panel at its minimum width; otherwise stacked and
// scrolling (a phone, a tablet held upright, a narrow window). The stylesheet keys off
// html.stacked; the dividers and the sizes they set are desktop only.

import { prefs } from "./panel.svelte";

export function layoutMode(): boolean {
  const w = window.innerWidth, h = window.innerHeight;
  const side = Math.min(h - 140, w - 382);
  const fits = w >= h && side >= 220;
  document.documentElement.classList.toggle("stacked", !fits);
  return !fits;
}

export const stacked = (): boolean => document.documentElement.classList.contains("stacked");

/** Stacked, the most the board's side can be by height: the main area less everything that
 * shares the first screen with it -- the rows under the board (the handle, the game line, the
 * board's tools) and the panel under those (the moves' line, the ask box, the chips), with the
 * coach at its least (the stylesheet's min-height). Measured, not assumed: those rows wrap
 * with the width and the panes can be hidden. */
function stackedHeightRoom(mainEl: HTMLElement): number {
  const left = mainEl.querySelector<HTMLElement>("#left");
  const wrap = left?.querySelector<HTMLElement>(".board-wrap");
  const right = mainEl.querySelector<HTMLElement>("#right");
  const px = (v: string) => parseFloat(v) || 0;
  const cs = getComputedStyle(mainEl);
  let room = mainEl.clientHeight - px(cs.paddingTop) - px(cs.paddingBottom);
  if (left && wrap) room -= left.getBoundingClientRect().height - wrap.getBoundingClientRect().height;
  if (right && right.getBoundingClientRect().height > 0) {
    const rs = getComputedStyle(right);
    room -= px(cs.rowGap) + px(rs.paddingTop) + px(rs.paddingBottom) + px(rs.borderTopWidth) + px(rs.borderBottomWidth);
    const shown = [...right.children].filter((c) => getComputedStyle(c).display !== "none") as HTMLElement[];
    room -= px(rs.rowGap) * Math.max(0, shown.length - 1);
    for (const c of shown) {
      const m = getComputedStyle(c);
      room -= px(m.marginTop) + px(m.marginBottom);
      room -= c.classList.contains("coach") ? px(m.minHeight) : c.getBoundingClientRect().height;
    }
  }
  return room;
}

/** Stacked, the board's width: the size the player dragged it to on the panel, within the
 * room the main area has; the whole width (the stylesheet's default) when none was chosen and
 * it fits. Never so big that the coach, the ask box and the chips go under the fold. */
export function mobileBoard(mainEl: HTMLElement | null, wanted: number | null): void {
  const root = document.documentElement;
  if (!mainEl || !stacked()) { root.style.removeProperty("--mboard"); return; }
  const byWidth = mainEl.clientWidth - 20;
  // A few passes: the rows under the board wrap with its width, so a narrower board can make
  // them taller (and a wider one shorter); each pass measures them at the last pass's width.
  for (let pass = 0; pass < 4; pass++) {
    const most = Math.max(160, Math.min(byWidth, 640, stackedHeightRoom(mainEl)));
    const side = wanted === null && most >= Math.min(byWidth, 640) ? null : Math.round(Math.max(160, Math.min(most, wanted ?? most)));
    const now = root.style.getPropertyValue("--mboard");
    if (side === null) { if (!now) return; root.style.removeProperty("--mboard"); continue; }
    if (now === side + "px") return;
    root.style.setProperty("--mboard", side + "px");
  }
}

/** The board's side from the room the main area has, smaller still if the player dragged
 * the divider. Re-measured on every resize and whenever the footer changes. */
export function fitBoard(mainEl: HTMLElement | null, divider: HTMLElement | null, info: HTMLElement | null, wanted: number | null): void {
  const root = document.documentElement;
  if (!mainEl || layoutMode()) { root.style.removeProperty("--board"); return; }
  const cs = getComputedStyle(mainEl);
  const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom);
  const padX = parseFloat(cs.paddingLeft) + parseFloat(cs.paddingRight);
  const dividerW = divider ? divider.getBoundingClientRect().width : 14;
  const infoH = info ? info.getBoundingClientRect().height + 8 : 30;
  const panelMin = parseFloat(getComputedStyle(root).getPropertyValue("--panel-min")) || 340;
  const byHeight = mainEl.clientHeight - padY - infoH;
  const byWidth = mainEl.clientWidth - padX - dividerW - panelMin;
  const side = Math.max(200, Math.floor(Math.min(byHeight, byWidth, wanted ?? 1200)));
  root.style.setProperty("--board", side + "px");
}

/** Stacked, the handle under the board: dragged down the board grows, up it shrinks (`apply`
 * gets the width wanted); a double-tap restores the default. Off side by side, where the
 * dividers are on. */
export function boardHandle(handle: HTMLElement, apply: (width: number) => void, reset: () => void): () => void {
  const down = (e: PointerEvent) => {
    if (!stacked()) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const origin = e.clientY;
    const start = (handle.parentElement as HTMLElement).getBoundingClientRect().width;
    handle.classList.add("live");
    const move = (ev: PointerEvent) => apply(start + (ev.clientY - origin));
    const done = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", done);
      handle.removeEventListener("pointercancel", done);
      handle.classList.remove("live");
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", done);
    handle.addEventListener("pointercancel", done);
  };
  handle.addEventListener("pointerdown", down);
  handle.addEventListener("dblclick", reset);
  return () => { handle.removeEventListener("pointerdown", down); handle.removeEventListener("dblclick", reset); };
}

/** A divider the pointer drags: `size` reads the size it controls at the start of a drag,
 * `apply` gets that and the distance moved; a double-click restores the default. */
export function splitter(handle: HTMLElement, axis: "x" | "y", size: () => number, apply: (start: number, d: number) => void, reset: () => void): () => void {
  const down = (e: PointerEvent) => {
    if (e.button !== 0 || stacked()) return;
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const origin = axis === "x" ? e.clientX : e.clientY;
    const start = size();
    document.body.classList.add("dragging-" + axis);
    handle.classList.add("live");
    const move = (ev: PointerEvent) => apply(start, (axis === "x" ? ev.clientX : ev.clientY) - origin);
    const done = () => {
      handle.removeEventListener("pointermove", move);
      handle.removeEventListener("pointerup", done);
      handle.removeEventListener("pointercancel", done);
      document.body.classList.remove("dragging-" + axis);
      handle.classList.remove("live");
    };
    handle.addEventListener("pointermove", move);
    handle.addEventListener("pointerup", done);
    handle.addEventListener("pointercancel", done);
  };
  handle.addEventListener("pointerdown", down);
  handle.addEventListener("dblclick", reset);
  return () => { handle.removeEventListener("pointerdown", down); handle.removeEventListener("dblclick", reset); };
}

export { prefs };
