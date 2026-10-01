/** A strip that scrolls sideways shows there is more past an edge: `more-l` / `more-r` on it
 * while it can scroll that way (the stylesheet fades that edge), off at the ends, so the first
 * and last items are never dimmed once they are reached. A Svelte action. */
export function edgeFade(el: HTMLElement): { destroy: () => void } {
  const mark = () => {
    const room = el.scrollWidth - el.clientWidth;
    el.classList.toggle("more-l", el.scrollLeft > 2);
    el.classList.toggle("more-r", room - el.scrollLeft > 2);
  };
  mark();
  el.addEventListener("scroll", mark, { passive: true });
  const ro = window.ResizeObserver ? new ResizeObserver(mark) : null;
  ro?.observe(el);
  return {
    destroy() {
      el.removeEventListener("scroll", mark);
      ro?.disconnect();
    },
  };
}
