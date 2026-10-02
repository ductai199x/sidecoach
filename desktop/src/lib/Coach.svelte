<script lang="ts">
  // The conversation with the coach, drawn as iMessage draws one (Tai, 2026-09-29: "similar to like a chat interface like one you find on iMessage"):
  // what you asked on the right, the coach's words on the left, a tail on the last bubble of
  // each run, the typing dots while a reply is on its way; the review of a game that ended and
  // the practice it picked as cards across the pane. A message with marks shows its own
  // position while hovered, so an arrow from an
  // earlier moment is never drawn over pieces that have since moved; a turning point
  // in the review shows its position too, and carries a menu to correct why it happened.
  import LinkedText from "./LinkedText.svelte";
  import { motifWords } from "./chess";
  import { panel } from "./api";
  import { store, type CoachItem } from "./panel.svelte";
  import { inShell, openWindow } from "./shell.svelte";
  import { stickToBottom } from "./stick";

  const CAUSES = [
    "missed opponent threat", "miscalculated line", "never considered the candidate",
    "positional misjudgment", "opening knowledge gap", "endgame technique", "time pressure",
  ];
  const ORDER = ["brilliant", "great", "best", "excellent", "good", "book", "inaccuracy", "mistake", "miss", "blunder"];

  function meta(m: Extract<CoachItem, { kind: "message" }>): string {
    // A notice about the board (trigger "board") has no disclosure level to show.
    const level = m.disclosure != null ? " · level " + m.disclosure : "";
    return m.trigger + level + (m.ply != null ? " · ply " + m.ply : "") + (m.fallback ? " · template" : "");
  }
  function notes(r: Record<string, any>): string {
    return ((r.lines || []) as string[]).filter((l) => /^(Expected score|Most ground|No move cost|The opening)/.test(l)).join(" ");
  }
  function tallies(r: Record<string, any>): [string, number][] {
    return ORDER.filter((k) => r.tallies && r.tallies[k]).map((k) => [k, r.tallies[k]]);
  }
  /** Whose side an item is on: a run of bubbles from one side ends in a tail. */
  function side(item: CoachItem | undefined): string {
    if (!item) return "";
    return item.kind === "you" ? "you" : item.kind === "message" || item.kind === "pending" ? "coach" : "card";
  }
  const tail = (i: number): boolean => side(store.coach[i]) !== side(store.coach[i + 1]);
  // A kind of move counted, in words: "2 inaccuracies", "1 blunder", "3 best".
  const PLURAL: Record<string, string> = { inaccuracy: "inaccuracies", mistake: "mistakes", miss: "misses", blunder: "blunders" };
  const kindWords = (k: string, n: number): string => (n === 1 ? k : PLURAL[k] ?? k);
  const capital = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);
  /** A card's accuracy: Game Review's once the game's review is done (the server's, with each
   * poll), the live one until then. */
  function accuracyOf(item: Extract<CoachItem, { kind: "review" }>): number | null {
    const reviewed = item.reviewKey ? store.state.review_accuracy?.[item.reviewKey] : undefined;
    return reviewed ?? item.review.accuracy ?? null;
  }
</script>

<div id="coach" class="coach" use:stickToBottom>
  {#if !store.coach.length}
    <!-- Before the coach has said anything: what goes here, and what to do. -->
    <div class="coach-empty">
      {#if store.running}
        <b>The coach is watching.</b>
        <span>It speaks up when a move teaches something. Ask it anything below, or press Hint.</span>
      {:else}
        <b>Press Start, then play.</b>
        <span>Open a game on chess.com or Lichess: the coach follows it from the screen.</span>
      {/if}
    </div>
  {/if}
  {#each store.coach as item, i (item.id)}
    {#if item.kind === "pending"}
      <div class="msg typing" class:tail={tail(i)} role="status" aria-label="The coach is writing"><span></span><span></span><span></span></div>
    {:else if item.kind === "divider"}
      <div class="coach-divider" role="separator"><span>{item.text}</span></div>
    {:else if item.kind === "you"}
      <div class="msg you" class:tail={tail(i)} title={item.spoken ? "you, spoken" : "you"}><LinkedText text={item.text} /></div>
    {:else if item.kind === "message"}
      <div
        class="msg"
        class:tail={tail(i)}
        class:linked={item.marks.length > 0}
        role="note"
        title={meta(item) + (item.marks.length ? ` · ${item.marks.length} mark${item.marks.length > 1 ? "s" : ""}` : "")}
        onmouseenter={() => store.hoverMessage(item)}
        onmouseleave={() => store.hoverMessage(null)}
      ><LinkedText text={item.full} /></div>
    {:else if item.kind === "practice"}
      {@const p = item.practice}
      <div class="msg review practice">
        <div class="meta">Practice · {capital(motifWords(p.motif))} · around {p.level}</div>
        <div class="rv-head">{p.reason ? p.reason.charAt(0).toUpperCase() + p.reason.slice(1) + "." : "Puzzles to practise."}</div>
        <div class="rv-puzzles">
          {#each p.puzzles || [] as z, n (z.id)}
            <span class="rv-puzzle-wrap">
              <a href="/puzzle/{encodeURIComponent(z.id)}" target="_blank" rel="noopener" class="rv-puzzle" title="Solve it in a window of its own: {((z.themes || []) as string[]).map(motifWords).join(', ') || 'practice'}" onclick={(e) => { if (inShell()) { e.preventDefault(); openWindow("puzzle/" + encodeURIComponent(z.id)); } }}>Puzzle {n + 1} · {z.rating}</a>
              <a href={z.url} target="_blank" rel="noopener" class="rv-puzzle-link" title="Open it on Lichess instead" aria-label="Open it on Lichess instead"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 19H5V5h7V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z" /></svg></a>
            </span>
          {/each}
        </div>
      </div>
    {:else}
      {@const r = item.review}
      {@const accuracy = accuracyOf(item)}
      <div class="msg review">
        <div class="meta">Game over · {r.result}{r.reason ? " by " + r.reason : ""}{accuracy != null ? " · accuracy " + accuracy.toFixed(1) : ""}</div>
        <div class="rv-head">{(r.lines && r.lines[0]) || "Game over."}</div>
        {#if tallies(r).length}
          <div class="rv-tally">
            {#each tallies(r) as [k, n] (k)}<span class="rv-chip {k}">{n} {kindWords(k, n)}</span>{/each}
          </div>
        {/if}
        {#each r.turning_points || [] as tp (tp.ply + ":" + (tp.segment || 0))}
          <div
            class="rv-turn"
            role="note"
            onmouseenter={() => tp.fen_before && store.hoverFen(tp.fen_before)}
            onmouseleave={() => tp.fen_before && store.hoverFen(null)}
          >
            <span class="rv-move">{tp.san}</span>
            <span class="rv-label {tp.label}">{tp.label} −{tp.expected_loss.toFixed(2)}</span>
            {#if tp.best_san}<span class="rv-better">better: {tp.best_san}</span>{/if}
            <select
              class="rv-cause"
              title="Why it happened, as the profile reads it. Change it if you know better."
              value={tp.cause}
              onchange={(e) => panel.correct({ game_id: r.game_id, segment: tp.segment || 0, ply: tp.ply, cause: (e.currentTarget as HTMLSelectElement).value })}
            >
              {#each CAUSES as cause (cause)}
                <option value={cause}>{capital(cause)}{tp.motif && cause === tp.cause ? " (" + motifWords(tp.motif) + ")" : ""}</option>
              {/each}
            </select>
          </div>
        {/each}
        {#if notes(r)}<div class="rv-notes">{notes(r)}</div>{/if}
        {#if item.insights.length}<div class="rv-insights">{item.insights.join(" ")}</div>{/if}
        {#if item.reviewKey}
          <!-- Straight into Game Review of the game just played (docs/game-review-plan.md, A). -->
          <a class="rv-open" href="/review/{encodeURIComponent(item.reviewKey)}" target="_blank" rel="noopener" title="Review this game move by move" onclick={(e) => { if (inShell()) { e.preventDefault(); openWindow("review/" + encodeURIComponent(item.reviewKey!)); } }}>Game Review</a>
        {/if}
      </div>
    {/if}
  {/each}
</div>
