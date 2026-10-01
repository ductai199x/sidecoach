<script lang="ts">
  // A paragraph of the AI coach's words with its links to the board (coach/review_links.py): a
  // move in the colour of its arrow (drawn with its squares lit), a piece or square marked on the
  // board, a file spanned end to end, a term with its meaning (and the pawns or squares it names). Hovering shows a link on the board (`onmark(link, false)`, `null` on leaving); a
  // click or tap pins it there (`onmark(link, true)`), again unpins. A term's meaning shows over
  // it while it is shown.
  import type { AiLink, AiPiece } from "$lib/api";

  let {
    pieces,
    shown = null,
    onmark,
  }: {
    pieces: AiPiece[];
    shown?: AiLink | null;
    onmark: (link: AiLink | null, pin: boolean) => void;
  } = $props();

  // The meaning over the term shown, placed on the page (the card scrolls, a tip inside it
  // would be cut at its edge).
  let tip = $state<{ text: string; x: number; y: number; below: boolean } | null>(null);
  let anchor: HTMLElement | null = null;
  function place(el: HTMLElement, link: AiLink): void {
    anchor = el;
    const r = el.getBoundingClientRect();
    const half = 130;
    const x = Math.max(half + 8, Math.min(window.innerWidth - half - 8, r.left + r.width / 2));
    const below = r.top < 90;
    tip = { text: link.meaning ?? "", x, y: below ? r.bottom + 8 : r.top - 8, below };
  }
  $effect(() => {
    const link = shown;
    if (link?.kind === "term" && anchor?.dataset.i && pieces[+anchor.dataset.i] === link) place(anchor, link);
    else tip = null;
  });

  function enter(e: PointerEvent, link: AiLink): void {
    if (e.pointerType !== "mouse") return;
    anchor = e.currentTarget as HTMLElement;
    onmark(link, false);
  }
  function leave(e: PointerEvent): void {
    if (e.pointerType === "mouse") onmark(null, false);
  }
  function pin(e: Event, link: AiLink): void {
    e.stopPropagation();
    anchor = e.currentTarget as HTMLElement;
    onmark(link, true);
  }
</script>

{#each pieces as p, i (i)}{#if typeof p === "string"}{p.replaceAll(" -- ", " — ")}{:else}<span
      class="lk k-{p.kind} {p.arrow ? `r-${p.arrow[2]}` : ''}"
      class:on={shown === p}
      data-i={i}
      role="button"
      tabindex="0"
      aria-label={p.kind === "term" ? `${p.text}: ${p.meaning}` : undefined}
      onpointerenter={(e) => enter(e, p)}
      onpointerleave={leave}
      onclick={(e) => pin(e, p)}
      onkeydown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pin(e, p); } }}
    >{p.text}</span>{/if}{/each}{#if tip}<span class="rv-tip" class:below={tip.below} style="left: {tip.x}px; top: {tip.y}px" role="tooltip">{tip.text}</span>{/if}
