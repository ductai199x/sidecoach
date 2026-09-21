<script lang="ts">
  // Talking to the coach: a text box, the listening-mode button, the round button that
  // is Send while there is text, the hold-to-talk button in push-to-talk mode, the
  // listening indicator in voice mode and just a mic otherwise; the quick chips; and what
  // was heard, corrected against the board.
  import { MODE_TITLES, store } from "./panel.svelte";

  let text = $state("");
  let input = $state<HTMLInputElement | null>(null);

  const sendState = $derived(
    text.trim().length > 0 ? "send" : store.listenMode === "ptt" ? "ptt" : store.listenMode === "vad" ? "vad" : "idle",
  );
  const sendTitle = $derived(
    sendState === "send" ? "Send (Enter)" : sendState === "ptt" ? "Hold to talk (or hold Space)" : sendState === "vad" ? "Listening" : "Type a question, or turn listening on",
  );

  function send() {
    const t = text.trim();
    if (!t) return;
    store.ask(t);
    text = "";
  }
  function onSendClick() {
    if (sendState === "send") send();
    else if (sendState === "idle") input?.focus();
  }
</script>

<div class="composer">
  <div class="box">
    <input
      bind:this={input}
      bind:value={text}
      type="text"
      placeholder="Ask the coach…"
      autocomplete="off"
      spellcheck="false"
      onkeydown={(e) => { if (e.key === "Enter") send(); }}
    />
    <button class="iconbtn" data-mode={store.listenMode} title={MODE_TITLES[store.listenMode] || ""} onclick={() => store.cycleListen()} aria-label="Listening mode">
      <svg class="ic ic-off" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-8.1 3.9l1.5-1.5A3 3 0 0 0 15 12h2zM5.3 3.9 20.1 18.7l-1.4 1.4-3.2-3.2A6.9 6.9 0 0 1 13 18.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 7.1 4.5l-1.5-1.5A3 3 0 0 1 9 12V9.4L3.9 5.3z"/></svg>
      <svg class="ic ic-ptt" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3h2a7 7 0 0 1-6 6.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 10 0z"/></svg>
      <svg class="ic ic-vad" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 10h2v4H3zm4-3h2v10H7zm4-4h2v18h-2zm4 4h2v10h-2zm4 3h2v4h-2z"/></svg>
    </button>
    <button
      class="sendbtn"
      class:live={sendState === "ptt" && store.pttDown}
      data-state={sendState}
      title={sendTitle}
      aria-label={sendTitle}
      onclick={onSendClick}
      onmousedown={() => { if (sendState === "ptt") store.setPtt(true); }}
      onmouseup={() => store.setPtt(false)}
      onmouseleave={() => store.setPtt(false)}
      ontouchstart={(e) => { if (sendState === "ptt") { e.preventDefault(); store.setPtt(true); } }}
      ontouchend={() => store.setPtt(false)}
    >
      <svg class="ic ic-send" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l7 7-1.4 1.4L13 7.8V20h-2V7.8l-4.6 4.6L5 11z"/></svg>
      <svg class="ic ic-mic" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3h2a7 7 0 0 1-6 6.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 10 0z"/></svg>
      <svg class="ic ic-wave" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 10h2v4H3zm4-3h2v10H7zm4-4h2v18h-2zm4 4h2v10h-2zm4 3h2v4h-2z"/></svg>
    </button>
  </div>
  <div class="chips-row">
    <button class="qchip" title="A nudge at the current disclosure level" onclick={() => store.quick("hint")}>Hint</button>
    <button class="qchip" title="The engine's move, then why" onclick={() => store.quick("move")}>Best move</button>
    <button class="qchip" title="The engine's line" onclick={() => store.quick("line")}>Line</button>
    {#if store.state.heard}<span class="heard">heard: <b>{store.state.heard}</b></span>{/if}
  </div>
</div>
