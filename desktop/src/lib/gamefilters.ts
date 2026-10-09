// The filters of Your games (/games) and what they hide, as Gmail's "no results -- clear search":
// a list the filters cut says how many games they hide, with Clear filters, and an import whose
// new games the filters hide says so. Clear filters puts back the page's own starting filters
// (NO_FILTERS) and only on the player's press; nothing here changes a filter by itself.
import type { LibraryGame } from "$lib/api";

export type Filters = {
  show: "all" | LibraryGame["source"];
  outcome: "all" | "won" | "lost" | "drawn";
  speed: string;
  /** Days back; 0 every game. */
  since: number;
  /** The search: a player's name. */
  query: string;
};

/** The filters the page starts with: every game shown. */
export const NO_FILTERS: Readonly<Filters> = { show: "all", outcome: "all", speed: "all", since: 0, query: "" };

export const playedAt = (g: LibraryGame): number => g.played_at ?? g.added_at;

/** The game's outcome for the player, when the library knows which side was theirs. */
export function outcomeOf(g: LibraryGame): "won" | "lost" | "drawn" | null {
  if (!g.user_side || !g.result || g.result === "*") return null;
  if (g.result === "1/2-1/2") return "drawn";
  return (g.result === "1-0") === (g.user_side === "white") ? "won" : "lost";
}

/** Whether the game shows under these filters (`now` in seconds). */
export function passes(g: LibraryGame, f: Filters, now: number): boolean {
  const needle = f.query.trim().toLowerCase();
  return (
    (f.show === "all" || g.source === f.show) &&
    (f.outcome === "all" || outcomeOf(g) === f.outcome) &&
    (f.speed === "all" || g.time_class === f.speed) &&
    (!f.since || playedAt(g) >= now - f.since * 86400) &&
    (!needle || `${g.white ?? ""} ${g.black ?? ""}`.toLowerCase().includes(needle))
  );
}

const games = (n: number) => `${n} game${n === 1 ? "" : "s"}`;

/** Of `keys` (the games just brought in), how many the library holds that the list shows and
 *  how many it hides (a game removed since is neither). */
export function seenOf(
  keys: readonly string[],
  list: readonly LibraryGame[],
  shown: readonly LibraryGame[],
): { shown: number; hidden: number } {
  const held = new Set(list.map((g) => g.key));
  const seen = new Set(shown.map((g) => g.key));
  const kept = keys.filter((k) => held.has(k));
  const showing = kept.filter((k) => seen.has(k)).length;
  return { shown: showing, hidden: kept.length - showing };
}

/** The line under a list the filters cut: "127 games hidden by your filters." */
export function hiddenWords(n: number): string {
  return `${games(n)} hidden by your filters.`;
}

/** An import's line while the filters hide some of its `added` new games, or null when they
 *  hide none (the line is then the import's own). `extras`: the " · 2 already here" parts;
 *  `next`: what to do next, said only while some new game shows. */
export function broughtInWords(
  added: number,
  seen: { shown: number; hidden: number },
  extras: string,
  next: string,
): string | null {
  if (!seen.hidden) return null;
  if (!seen.shown) return `${games(added)} brought in — ${added === 1 ? "hidden" : "all hidden"} by your filters${extras}.`;
  return `${games(added)} brought in — ${Math.min(seen.hidden, added)} hidden by your filters${extras}. ${next}`;
}
