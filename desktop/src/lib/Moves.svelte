<script lang="ts">
  // The move list. Each move carries the engine's label as a badge (the ones chess.com
  // marks: brilliant, great, best, inaccuracy, mistake, miss, blunder; good and book stay
  // plain). Hovering a move shows its position on the board; clicking pins it there, and
  // clicking it again (or Escape) comes back to the live board. Stacked, or side by side when
  // the panel is short or the list was dragged small (`oneline`), the moves run in one line.
  import Badge from "$lib/Badge.svelte";
  import { store } from "./panel.svelte";
  import { stacked } from "./layout";
  import { tick } from "svelte";

  let { oneline = false }: { oneline?: boolean } = $props();

  // "unjudged": the engine could not judge the move (its searches failed), told rather than blank.
  const BADGED = new Set(["brilliant", "great", "best", "inaccuracy", "mistake", "miss", "blunder", "unjudged"]);
  const TITLES: Record<string, string> = { unjudged: "not judged: the engine failed on this move" };
  let wrap = $state<HTMLDivElement | null>(null);
  const sideways = () => oneline || stacked();

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

  // Follow the game: the newest move stays in view unless a move is pinned. In one line the
  // list follows sideways, and a pinned move is brought into view there.
  $effect(() => {
    void count;
    void oneline;
    if (store.pinnedPly === null && wrap) {
      tick().then(() => {
        if (!wrap) return;
        if (sideways()) wrap.scrollLeft = wrap.scrollWidth;
        else wrap.scrollTop = wrap.scrollHeight;
      });
    }
  });
  $effect(() => {
    void viewed;
    void oneline;
    if (!wrap || !sideways()) return;
    tick().then(() => {
      const el = wrap?.querySelector(".mv.view") as HTMLElement | null;
      el?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
    });
  });
  // A mouse wheel turns the line, as it scrolls the list.
  $effect(() => {
    const el = wrap;
    if (!el || !oneline) return;
    const wheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || el.scrollWidth <= el.clientWidth) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    };
    el.addEventListener("wheel", wheel, { passive: false });
    return () => el.removeEventListener("wheel", wheel);
  });
</script>

<div class="moves-wrap" class:oneline bind:this={wrap}>
  {#if !count}<div class="moves-empty">No moves yet</div>{/if}
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
            >{c.san}{#if c.label}<Badge kind={c.label} title={TITLES[c.label]} />{/if}</span>
          {:else}
            <span class="mv"></span>
          {/if}
        {/each}
      </div>
    {/each}
  </div>
</div>
