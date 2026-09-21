<script lang="ts">
  // A coach message with the squares and moves it names wrapped, so hovering one lights
  // its square on the board.
  import { SQ_TOKEN } from "./chess";
  import { store } from "./panel.svelte";

  let { text }: { text: string } = $props();

  const parts = $derived.by(() => {
    const out: { text: string; square: string | null }[] = [];
    let last = 0;
    for (const m of text.matchAll(SQ_TOKEN)) {
      if (m.index! > last) out.push({ text: text.slice(last, m.index), square: null });
      const sq = m[0].match(/[a-h][1-8]/g);
      out.push({ text: m[0], square: sq ? sq[sq.length - 1] : null });
      last = m.index! + m[0].length;
    }
    if (last < text.length) out.push({ text: text.slice(last), square: null });
    return out;
  });
</script>

{#each parts as p}
  {#if p.square}
    <span class="tok" role="note" onmouseenter={() => (store.hoverSquare = p.square)} onmouseleave={() => (store.hoverSquare = null)}>{p.text}</span>
  {:else}{p.text}{/if}
{/each}
