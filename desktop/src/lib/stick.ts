// Keep a scrolling list at its newest entry (Tai, 2026-09-25: "the coach messages box
// don't have auto scroll to bottom whenever a new message is sent"). The lists used to
// scroll when their length changed, which it does not when the coach's "..." is replaced by
// its answer, when an answer grows as it streams, or once a list is full and each new entry
// pushes the oldest out. This follows every change of the content and every resize instead.
// A reader who has scrolled up to read something is left there; after RELEASE_MS without
// scrolling, the next change brings the list back to its newest entry.

const NEAR_PX = 48; // this close to the end counts as at the end
const RELEASE_MS = 10_000;
// A scroll counts as the reader's only this soon after their wheel, touch, click or key:
// the browser also scrolls a list when its box resizes (the phone's stacked layout), and
// that is no reader leaving the end.
const INPUT_MS = 1_500;

export interface StickOptions {
  // Whether to follow at all just now (the terminal's autoscroll switch, a hidden pane).
  enabled?: () => boolean;
}

export function stickToBottom(el: HTMLElement, options: StickOptions = {}) {
  let opts = options;
  let pinned = true; // at the end, or following it
  let awayAt = -Infinity; // when the reader last scrolled away from the end
  let inputAt = -Infinity; // the reader's last wheel, touch, click or key on the list
  const atEnd = () => el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_PX;
  const onInput = () => {
    inputAt = performance.now();
  };
  const onScroll = () => {
    if (atEnd()) {
      pinned = true; // back at the end, however it got there
      return;
    }
    if (performance.now() - inputAt > INPUT_MS) return; // moved by the layout, not the reader
    pinned = false;
    awayAt = performance.now();
  };
  let queued = false;
  const follow = () => {
    if (queued) return;
    queued = true;
    // The content changes before the layout does: follow once it has been laid out.
    requestAnimationFrame(() => {
      queued = false;
      if (opts.enabled && !opts.enabled()) return;
      if (pinned || performance.now() - awayAt > RELEASE_MS) {
        el.scrollTop = el.scrollHeight;
        pinned = true;
      }
    });
  };
  const mutations = new MutationObserver(follow);
  mutations.observe(el, { childList: true, subtree: true, characterData: true });
  const resizes = new ResizeObserver(follow);
  resizes.observe(el);
  const inputs = ["wheel", "touchstart", "touchmove", "pointerdown", "keydown"] as const;
  for (const name of inputs) el.addEventListener(name, onInput, { passive: true });
  el.addEventListener("scroll", onScroll, { passive: true });
  follow();
  return {
    update(next: StickOptions = {}) {
      opts = next;
      follow();
    },
    destroy() {
      mutations.disconnect();
      resizes.disconnect();
      for (const name of inputs) el.removeEventListener(name, onInput);
      el.removeEventListener("scroll", onScroll);
    },
  };
}
