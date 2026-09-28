<script lang="ts">
  // The coach's messages, the review of a game that ended, and the practice it picked.
  // A message with marks shows its own position while hovered, so an arrow from an
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
</script>

<div id="coach" class="coach" use:stickToBottom>
  {#each store.coach as item (item.id)}
    {#if item.kind === "pending"}
      <div class="msg pending"><div class="meta">coach</div><div class="dots"></div></div>
    {:else if item.kind === "message"}
      <div
        class="msg"
        class:linked={item.marks.length > 0}
        role="note"
        onmouseenter={() => store.hoverMessage(item)}
        onmouseleave={() => store.hoverMessage(null)}
      >
        <div class="meta">{meta(item)}{#if item.marks.length}<span class="marks"> · {item.marks.length} mark{item.marks.length > 1 ? "s" : ""}</span>{/if}</div>
        <div><LinkedText text={item.full} /></div>
      </div>
    {:else if item.kind === "practice"}
      {@const p = item.practice}
      <div class="msg review practice">
        <div class="meta">practice · {motifWords(p.motif)} · around {p.level}</div>
        <div class="rv-head">{p.reason ? p.reason.charAt(0).toUpperCase() + p.reason.slice(1) + "." : "Puzzles to practise."}</div>
        <div class="rv-puzzles">
          {#each p.puzzles || [] as z (z.id)}
            <span class="rv-puzzle-wrap">
              <a href="/puzzle/{encodeURIComponent(z.id)}" target="_blank" rel="noopener" class="rv-puzzle" title="Solve it in a window of its own: {((z.themes || []) as string[]).map(motifWords).join(', ') || 'practice'}" onclick={(e) => { if (inShell()) { e.preventDefault(); openWindow("puzzle/" + encodeURIComponent(z.id)); } }}>puzzle {z.id} · {z.rating}</a>
              <a href={z.url} target="_blank" rel="noopener" class="rv-puzzle-link" title="Open on Lichess instead">↗</a>
            </span>
          {/each}
        </div>
      </div>
    {:else}
      {@const r = item.review}
      <div class="msg review">
        <div class="meta">review · {r.result}{r.reason ? " by " + r.reason : ""}{r.accuracy != null ? " · accuracy " + r.accuracy.toFixed(1) : ""}</div>
        <div class="rv-head">{(r.lines && r.lines[0]) || "Game over."}</div>
        {#if tallies(r).length}
          <div class="rv-tally">
            {#each tallies(r) as [k, n] (k)}<span class="rv-chip {k}">{n} {k}</span>{/each}
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
                <option value={cause}>{cause}{tp.motif && cause === tp.cause ? " (" + motifWords(tp.motif) + ")" : ""}</option>
              {/each}
            </select>
          </div>
        {/each}
        {#if notes(r)}<div class="rv-notes">{notes(r)}</div>{/if}
        {#if item.insights.length}<div class="rv-insights">{item.insights.join(" ")}</div>{/if}
      </div>
    {/if}
  {/each}
</div>
