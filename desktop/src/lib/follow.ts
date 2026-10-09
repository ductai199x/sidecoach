// The move shown, kept in view in a move list: Game Review's list and strip, the panel's strip.
// Each follows the move as it changes; the strip must follow it too when its own box changes
// size -- the page turned from the list into the one-line strip, or the strip got wider or
// narrower (a phone turned and back, a window resized). The browser keeps the old scroll offset
// then, clamped to the new room or dropped to 0, and the move shown was left off-screen (on a
// phone turned back upright: move 1 showing, the board on move 9).

type Rect = { left: number; right: number; top: number; bottom: number; width: number; height: number };
type Box = {
  scrollLeft: number;
  scrollTop: number;
  readonly scrollWidth: number;
  readonly clientWidth: number;
  readonly scrollHeight: number;
  readonly clientHeight: number;
  getBoundingClientRect(): Rect;
};

/** A box that scrolls sideways: the one-line strip. */
export const sideways = (box: Box): boolean => box.scrollWidth > box.clientWidth;

/** The item into view in its box: out of view, or at the very edge, it goes to the middle, so the
 *  moves around it show too -- along the strip, or down the list. Only the box scrolls (never the
 *  page around it). */
export function keepInView(box: Box, el: { getBoundingClientRect(): Rect }): void {
  const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
  if (sideways(box)) {
    if (r.left < b.left + r.width || r.right > b.right - r.width) box.scrollLeft += r.left + r.width / 2 - (b.left + b.width / 2);
  } else if (box.scrollHeight > box.clientHeight) {
    if (r.top < b.top + r.height || r.bottom > b.bottom - r.height) box.scrollTop += r.top + r.height / 2 - (b.top + b.height / 2);
  }
}

type Observer = { observe(el: unknown): void; disconnect(): void };
type ObserverClass = new (cb: (entries: { contentRect: { width: number; height: number } }[]) => void) => Observer;

/** `follow` again whenever the box changes size while it is the one-line strip (its first size
 *  is the follow's own business, as the move shown changes). The full list is left as it was.
 *  Returns the stop. */
export function followOnResize(box: Box & object, follow: () => void, Observer: ObserverClass | undefined = (globalThis as { ResizeObserver?: ObserverClass }).ResizeObserver): () => void {
  if (!Observer) return () => {};
  let last: string | null = null;
  const ro = new Observer((entries) => {
    const r = entries[entries.length - 1].contentRect;
    const size = `${Math.round(r.width)}x${Math.round(r.height)}`;
    const changed = last !== null && size !== last;
    last = size;
    if (changed && sideways(box)) follow();
  });
  ro.observe(box);
  return () => ro.disconnect();
}
