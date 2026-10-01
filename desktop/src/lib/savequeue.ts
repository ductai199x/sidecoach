// The Settings page's saves (routes/settings): every change is saved as it is made (Tai,
// 2026-10-01: "save each change as you make it, with no Save button"). One save at a time, in
// order: what changes while one is on its way goes with the next, so two quick changes can
// never land in the wrong order.

export type SaveQueue = {
  /** Save now, after the save on its way if any; `keepalive`: the page is closing, and the
   * request must outlive it. */
  flush: (keepalive?: boolean) => Promise<void>;
  /** One more round after the save on its way (what is left once refused fields are out). */
  more: () => void;
};

export function saveQueue(save: (keepalive: boolean) => Promise<void>): SaveQueue {
  let inFlight: Promise<void> | null = null;
  let again = false;

  function flush(keepalive = false): Promise<void> {
    if (inFlight) {
      again = true;
      // The page closing: what changed since the save on its way goes now, in a request of
      // its own that outlives the page. Left to the next round, it was lost: a page gone runs
      // no next round, and that round had no keepalive (the settings review, 2026-10-01).
      if (keepalive) void save(true).catch(() => {});
      return inFlight;
    }
    inFlight = (async () => {
      do {
        again = false;
        await save(keepalive);
      } while (again);
    })().finally(() => (inFlight = null));
    return inFlight;
  }

  return { flush, more: () => (again = true) };
}
