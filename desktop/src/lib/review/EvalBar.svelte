<script lang="ts">
  // The eval bar beside the board, as chess.com's: White's share from White's end (the
  // bottom when White is at the bottom), the number at the leading side's end. Stacked (a
  // phone), it lies across the top of the board, White's share from the left (the stylesheet
  // turns it; the share and which end is White's are all this sets).
  import type { Eval } from "$lib/api";
  import { evalText } from "./words";

  let { score, value, bottom = "white" }: { score: number; value: Eval | null; bottom?: string } = $props();

  // A little of each colour always shows, as on chess.com.
  const share = $derived(Math.max(0.04, Math.min(0.96, score)));
  const text = $derived(evalText(value).replace(/^[+-]/, ""));
</script>

<div class="rv-evalbar" class:flipped={bottom !== "white"} style="--share: {share}" aria-label="Evaluation {evalText(value)}" role="img">
  <div class="fill"></div>
  <span class="num" class:on-fill={score >= 0.5}>{text}</span>
</div>
