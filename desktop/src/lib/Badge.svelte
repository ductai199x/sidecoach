<script lang="ts" module>
  // Each kind's sign for a 24-unit badge, drawn as shapes, not a font's glyphs, so every size is
  // crisp and centred alike: a glyph drawn once about the centre, a pair of them scaled down and
  // set side by side.
  type Part = { fill?: string; stroke?: string; dot?: [number, number] };
  type Glyph = Part[];

  const QUESTION: Glyph = [
    { stroke: "M9.05 9.1c0-1.75 1.35-3 3.05-3s3 1.15 3 2.65c0 1.4-.95 2.05-1.85 2.65-.75.5-1.2 1-1.2 2.05v.5" },
    { dot: [12.15, 17.75] },
  ];
  const EXCLAIM: Glyph = [{ fill: "M10.3 5.2h3.4l-.45 9.2h-2.5z" }, { dot: [12, 17.75] }];
  const STAR: Glyph = [{ fill: "M12 5.35l1.73 4.71 5.02.2-3.94 3.1 1.36 4.83L12 15.4l-4.17 2.79 1.36-4.83-3.94-3.1 5.02-.2z" }];
  // A thumb up: the cuff, and the hand.
  const THUMB: Glyph = [
    { fill: "M5.3 11h2.6v7.3H5.3zM9.2 18.3V11.2l3.6-4.7c.5-.6 1.5-.3 1.5.5v3h3.1c.95 0 1.6.9 1.35 1.8l-1.5 5.3c-.2.7-.8 1.2-1.5 1.2z" },
  ];
  const CHECK: Glyph = [{ stroke: "M6.9 12.3l3.5 3.5 6.8-7.2" }];
  // An open book: two pages meeting at the spine.
  const BOOK: Glyph = [
    { fill: "M11.3 8.1C9.6 6.9 7.3 6.5 5.4 6.8v9.9c1.9-.3 4.2.1 5.9 1.3zM12.7 8.1c1.7-1.2 4-1.6 5.9-1.3v9.9c-1.9-.3-4.2.1-5.9 1.3z" },
  ];
  const CROSS: Glyph = [{ stroke: "M8.3 8.3l7.4 7.4M15.7 8.3l-7.4 7.4" }];
  // A dash: no verdict.
  const DASH: Glyph = [{ stroke: "M7.6 12h8.8" }];

  // A glyph drawn about the centre, moved to `x` and scaled by `s`.
  const at = (g: Glyph, x: number, s: number) => ({ g, t: `translate(${x} 12) scale(${s}) translate(-12 -12)` });

  const SIGNS: Record<string, { g: Glyph; t?: string }[]> = {
    brilliant: [at(EXCLAIM, 8.6, 0.9), at(EXCLAIM, 15.4, 0.9)],
    great: [{ g: EXCLAIM }],
    best: [{ g: STAR }],
    excellent: [{ g: THUMB }],
    good: [{ g: CHECK }],
    book: [{ g: BOOK }],
    inaccuracy: [at(QUESTION, 9.1, 0.82), at(EXCLAIM, 16.1, 0.82)],
    mistake: [{ g: QUESTION }],
    miss: [{ g: CROSS }],
    blunder: [at(QUESTION, 8.35, 0.82), at(QUESTION, 15.65, 0.82)],
    // The engine could not judge the move (its searches failed): told, not left blank.
    unjudged: [{ g: DASH }],
  };
</script>

<script lang="ts">
  // A move's kind as a badge: a disc of the kind's colour (--cls-<kind>, panel.css) with a fine
  // darker rim and a soft light on its upper half, and the kind's sign in white -- everywhere a
  // kind is shown: the board, the move lists, the cards, the tables, the captions.
  let { kind, size = 18, title }: { kind: string | null | undefined; size?: number; title?: string } = $props();
  const sign = $derived(kind ? SIGNS[kind] : undefined);
</script>

{#if sign}
  <svg class="kb" width={size} height={size} viewBox="0 0 24 24" role="img" aria-label={title ?? kind}>
    <title>{title ?? kind}</title>
    <circle cx="12" cy="12" r="12" style="fill: var(--cls-{kind})" />
    <path d="M2.2 10.2A10 10 0 0 1 21.8 10.2 30 12 0 0 0 2.2 10.2z" fill="#fff" opacity="0.16" />
    <circle cx="12" cy="12" r="11.4" fill="none" stroke="#000" stroke-opacity="0.16" stroke-width="1.2" />
    {#each sign as { g, t }, i (i)}
      <g transform={t}>
        {#each g as part, j (j)}
          {#if part.fill}<path d={part.fill} fill="#fff" />{/if}
          {#if part.stroke}<path d={part.stroke} fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" />{/if}
          {#if part.dot}<circle cx={part.dot[0]} cy={part.dot[1]} r="1.62" fill="#fff" />{/if}
        {/each}
      </g>
    {/each}
  </svg>
{/if}

<style>
  .kb { display: inline-block; flex: 0 0 auto; vertical-align: middle; filter: drop-shadow(0 1px 1px rgb(0 0 0 / 0.28)); }
</style>
