<script lang="ts">
  // The move list. Each move carries the engine's label as a badge (the ones chess.com
  // marks: brilliant, great, best, inaccuracy, mistake, miss, blunder; good and book stay
  // plain). Hovering a move shows its position on the board; clicking pins it there, and
  // clicking it again (or Escape) comes back to the live board.
  import Badge from "$lib/Badge.svelte";
  import { store } from "./panel.svelte";
  import { stacked } from "./layout";
  import { tick } from "svelte";

  const BADGED = new Set(["brilliant", "great", "best", "inaccuracy", "mistake", "miss", "blunder"]);
  let wrap = $state<HTMLDivElement | null>(null);

  const rows = $derived.by(() => {
    const sans = store.state.moves || [];
    const labels = store.state.labels || {};
    const out: { num: string; cells: { san: string | null; ply: number; label: string | null }[] }[] = [];
    for (let i = 0; i < sans.length; i += 2) {
      const cell = (ply: number) => {
        const san = sans[ply - 1] ?? null;
        const label = labels[String(ply)] ?? null;
        return { san, ply, label: label && BADGED.has(label) ? label : null };
      };
      out.push({ num: i / 2 + 1 + ".", cells: [cell(i + 1), cell(i + 2)] });
    }
    return out;
  });
  const count = $derived((store.state.moves || []).length);
  const viewed = $derived(store.pinnedPly ?? store.viewPly);

  // Follow the game: the newest move stays in view unless a move is pinned. Stacked, the
  // list is one line and follows sideways; a pinned move is brought into view there.
  $effect(() => {
    void count;
    if (store.pinnedPly === null && wrap) {
      tick().then(() => {
        if (!wrap) return;
        if (stacked()) wrap.scrollLeft = wrap.scrollWidth;
        else wrap.scrollTop = wrap.scrollHeight;
      });
    }
  });
  $effect(() => {
    void viewed;
    if (!wrap || !stacked()) return;
    tick().then(() => {
      const el = wrap?.querySelector(".mv.view") as HTMLElement | null;
      el?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
    });
  });
</script>

<div class="moves-wrap" bind:this={wrap}>
  <div id="moves">
    {#each rows as row (row.num)}
      <div class="mrow">
        <span class="num">{row.num}</span>
        {#each row.cells as c (c.ply)}
          {#if c.san}
            <span
              class="mv"
              class:cur={c.ply === count}
              class:view={c.ply === viewed}
              role="button"
              tabindex="-1"
              onmouseenter={() => store.hoverPly(c.ply)}
              onmouseleave={() => store.hoverPly(null)}
              onclick={() => store.pinPly(c.ply)}
              onkeydown={(e) => { if (e.key === "Enter") store.pinPly(c.ply); }}
            >{c.san}{#if c.label}<Badge kind={c.label} />{/if}</span>
          {:else}
            <span class="mv"></span>
          {/if}
        {/each}
      </div>
    {/each}
  </div>
</div>
