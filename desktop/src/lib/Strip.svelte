<script lang="ts">
  // The status strip: what `watch` prints every few seconds, kept current. Health comes
  // with every poll; the tokens and the cost of the session's model calls, and the
  // amortised latencies, are part of it.
  import { clockText, kilo, latencyText } from "./chess";
  import { store } from "./panel.svelte";
  import { perf } from "./perf.svelte";

  let { folded, onfold }: { folded: boolean; onfold: () => void } = $props();

  const h = $derived(store.state.health || {});
  const s = $derived(store.state);
  const lockClass = $derived(h.lock === "no board" ? "bad" : h.lock === "lock in doubt" ? "warn" : "");
  const lagClass = $derived(h.lag_ms != null && h.lag_ms > 1500 ? "bad" : h.lag_ms != null && h.lag_ms > 500 ? "warn" : "");
  const l = $derived(h.latency || null);
  const speechClass = $derived(l?.move_to_speech_s != null ? (l.move_to_speech_s > 4 ? "bad" : l.move_to_speech_s > 2.5 ? "warn" : "") : "");
  const tokenClass = $derived(l?.first_token_s != null ? (l.first_token_s > 3 ? "bad" : l.first_token_s > 1.5 ? "warn" : "") : "");
  const u = $derived(h.llm || null);
  const cached = $derived(u && u.prompt ? Math.round((100 * u.cached) / u.prompt) : 0);
  const est = $derived(u && u.estimated ? ", " + u.estimated + " estimated" : "");
</script>

<div id="health" class="health">
  <div class="stats">
    <span class="hitem"><span class="hk">fps</span><span class="hv">{h.fps != null ? h.fps.toFixed(1) : "–"}</span></span>
    <span class="hitem"><span class="hk">board</span><span class="hv {lockClass}">{h.lock || "–"}</span></span>
    <span class="hitem"><span class="hk">stage</span><span class="hv">{(h.stage || "–") + (s.ply ? ", ply " + s.ply : "")}</span></span>
    <span class="hitem"><span class="hk">pipeline</span><span class="hv">{h.pipeline_ms ? h.pipeline_ms[0] + "/" + h.pipeline_ms[1] + " ms" : "–"}</span></span>
    <span class="hitem"><span class="hk">lag</span><span class="hv {lagClass}">{h.lag_ms != null ? h.lag_ms + " ms" : "–"}</span></span>
    <span class="hitem"><span class="hk">clock</span><span class="hv">{clockText(s.clock)}</span></span>
    <span class="hitem"><span class="hk">coach queue</span><span class="hv" class:warn={(h.coach_pending ?? 0) > 1}>{h.coach_pending != null ? String(h.coach_pending) : "–"}</span></span>
    <span class="hitem"><span class="hk">move→speech</span><span class="hv {speechClass}" title="From a move being read to the first sound of the coach's word on it: the session's mean, and the last one">{l ? latencyText(l.move_to_speech_s, l.move_to_speech_last_s, l.move_to_speech_n) : "–"}</span></span>
    <span class="hitem"><span class="hk">first token</span><span class="hv {tokenClass}" title="From the model call to its first token: the session's mean, and the last one">{l ? latencyText(l.first_token_s, l.first_token_last_s, l.first_token_n) : "–"}</span></span>
    {#if u && u.calls}
      <span class="hitem"><span class="hk">tokens</span><span class="hv" title="prompt {u.prompt} (cached {u.cached}), completion {u.completion} (reasoning {u.reasoning}), {u.calls} calls{est}">{kilo(u.prompt)} in ({cached}% cached) · {kilo(u.completion)} out · {u.calls} calls</span></span>
      {#if u.cost == null}
        <span class="hitem"><span class="hk">cost</span><span class="hv" title="Set llm.price_prompt_per_m, price_cached_per_m and price_completion_per_m (USD per million tokens) in sidecoach.toml for a dollar figure">set prices</span></span>
      {:else}
        <span class="hitem"><span class="hk">cost</span><span class="hv" title="At the prices in sidecoach.toml{est}">${u.cost < 0.1 ? u.cost.toFixed(3) : u.cost.toFixed(2)}</span></span>
      {/if}
    {:else}
      <span class="hitem"><span class="hk">tokens</span><span class="hv">–</span></span>
      <span class="hitem"><span class="hk">cost</span><span class="hv">–</span></span>
    {/if}
    <span class="hitem"><span class="hk">recording</span><span class="hv">{h.recording || "–"}</span></span>
    <span class="hitem"><span class="hk">voice</span><span class="hv">{h.tts_device || "–"}</span></span>
    <span class="hitem"><span class="hk">ear</span><span class="hv">{h.stt_device || "–"}</span></span>
    {#if perf.on}
      <span class="hitem"><span class="hk">ui</span><span class="hv" title="The page: frames per second (the minimum this run), frames over 50 ms, the poll round trip p50/p95">{perf.text}</span></span>
    {/if}
    <span class="hitem"><span class="hk">rss</span><span class="hv">{h.rss_mb != null ? h.rss_mb + " MB" : "–"}</span></span>
  </div>
  <button id="fold" class="ghost small" title={folded ? "Show the debug terminal" : "Hide the debug terminal"} onclick={onfold}>{folded ? "show terminal" : "hide terminal"}</button>
</div>
