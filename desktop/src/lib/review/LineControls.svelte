<script lang="ts">
  // A stepped line's controls: back, play (pause while it runs; again at its end), on; and where
  // it is ("Move 2 of 5"). The review's lines and a puzzle's solution share them (lineplayer).
  // The colours are custom properties, so a dark page sets its own.
  import type { LinePlayer } from "./lineplayer.svelte";

  let { player, label = "the line", extra = "" }: { player: LinePlayer; label?: string; extra?: string } = $props();
</script>

<div class="line-play">
  <span class="pair">
    <button class="iconbtn" aria-label="Back a move of {label}" title="Back (←)" disabled={player.atStart} onclick={() => player.step(-1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg></button>
    <button class="iconbtn play" aria-label={player.autoplay ? `Pause ${label}` : `Play ${label}`} title={player.autoplay ? "Pause (Space)" : player.atEnd ? "Play it again (Space)" : "Play (Space)"} onclick={() => player.playPause()}>
      {#if player.autoplay}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>
      {:else if player.atEnd}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5V1L7 6l5 5V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8z"/></svg>
      {:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>{/if}
    </button>
    <button class="iconbtn" aria-label="On a move of {label}" title="On (→)" disabled={player.atEnd} onclick={() => player.step(1)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"/></svg></button>
  </span>
  <span class="where" aria-live="polite">{player.where}{extra ? ` · ${extra}` : ""}</span>
</div>

<style>
  .line-play { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .pair { display: inline-flex; gap: 2px; padding: 2px; border-radius: 7px; background: var(--line-pair-bg, #ebe9e6); flex: 0 0 auto; }
  .iconbtn { width: var(--line-btn-w, 36px); height: var(--line-btn-h, 30px); border-radius: 5px; color: var(--line-ink, #312e2b); }
  .iconbtn:hover { background: var(--line-hover-bg, #fff); box-shadow: 0 1px 2px #00000026; color: var(--line-ink, #312e2b); }
  .iconbtn:disabled { opacity: 0.3; }
  .iconbtn:disabled:hover { background: transparent; box-shadow: none; }
  .iconbtn svg { width: 24px; height: 24px; fill: currentColor; }
  .iconbtn.play svg { width: 20px; height: 20px; }
  .where { font-size: 12.5px; font-weight: 600; color: var(--line-muted, #8b8987); font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; min-width: 0; }
</style>
