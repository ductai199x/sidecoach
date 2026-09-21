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
