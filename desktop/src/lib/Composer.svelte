<script lang="ts">
  // Talking to the coach: a text box, the listening-mode button, the round button that
  // is Send while there is text, the hold-to-talk button in push-to-talk mode, the
  // listening indicator in voice mode and just a mic otherwise; and the quick chips.
  //
  // Dictation: what is said goes into the box as it is recognised (the sidecar streams
  // partial results while the key is held), and sends on its own a second after the
  // utterance ends -- unless more is said, which appends, or the person types, which makes
  // the words theirs to edit and send (Tai, 2026-09-22).
  import { panel } from "./api";
  import { MODE_TITLES, store } from "./panel.svelte";

  let text = $state("");
  let input = $state<HTMLInputElement | null>(null);

  // The dictation merged into the box so far, and what the box held before it began.
  let dictated = "";
  let typed = "";
  // After a cancel, the sidecar's old text may still arrive on a poll or two: ignored
  // until it reports an empty dictation once.
  let awaitingClear = false;

  const dictation = $derived(store.state.dictation);
  const keyHeld = $derived(!!store.state.ptt);
  const talking = $derived(keyHeld || !!dictation?.live);
  const armed = $derived(!!dictation?.sends_at && !dictation?.live);

  // While the key is held or words are still arriving, the round button stays the mic (lit),
  // whatever the box holds; otherwise text makes it Send.
  const sendState = $derived(
    talking && store.listenMode === "ptt" ? "ptt"
    : talking && store.listenMode === "vad" ? "vad"
    : text.trim().length > 0 ? "send"
    : store.listenMode === "ptt" ? "ptt"
    : store.listenMode === "vad" ? "vad"
    : "idle",
  );
  const keyName = $derived(prettyKey(store.state.ptt_key || "ctrl_r"));
  const sendTitle = $derived(
    sendState === "send" ? (armed ? "Sending in a moment (type to keep it)" : "Send (Enter)")
    : sendState === "ptt" ? (talking ? "Listening…" : `Hold to talk (or hold Space or ${keyName})`)
    : sendState === "vad" ? "Listening"
    : "Type a question, or turn listening on",
  );

  function prettyKey(k: string): string {
    const names: Record<string, string> = {
      ctrl_r: "Right Ctrl", ctrl_l: "Left Ctrl", alt_r: "Right Alt", alt_l: "Left Alt",
      shift_r: "Right Shift", shift_l: "Left Shift", cmd_r: "Right Cmd", cmd_l: "Left Cmd",
      ctrl: "Ctrl", alt: "Alt", shift: "Shift", cmd: "Cmd",
    };
    return names[k] || (k.length === 1 ? k.toUpperCase() : k.replace(/^f(\d+)$/i, "F$1"));
  }

  // Mirror the sidecar's dictation into the box.
  $effect(() => {
    const t = dictation?.text ?? "";
    if (awaitingClear) {
      if (!t) awaitingClear = false;
      return;
    }
    if (t === dictated) return;
    if (!dictated) typed = text; // the dictation begins: remember what was typed
    dictated = t;
    text = [typed, t].filter(Boolean).join(" ");
    if (!t) typed = ""; // sent (the sidecar asked the coach) -- the box is clear again
    // The newest words stay in view: a long ask scrolls the box, as typing would.
    requestAnimationFrame(() => { if (input) input.scrollLeft = input.scrollWidth; });
  });

  // Something was typed first: the words join it and the person sends the whole ask.
  $effect(() => {
    if (armed && typed) void takeOver();
  });

  async function takeOver() {
    if (!dictated && !dictation?.text) return;
    awaitingClear = true;
    dictated = "";
    typed = text;
    await panel.cancelDictation().catch(() => {});
  }

  function onInput() {
    // The person is editing: the dictation (if any) is theirs now, nothing sends itself.
    if (dictated || dictation?.text) void takeOver();
  }

  function send() {
    const t = text.trim();
    if (!t) return;
    if (dictated || dictation?.text) void takeOver();
    store.ask(t);
    text = "";
    typed = "";
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
      placeholder={talking ? "Listening…" : "Ask the coach…"}
      autocomplete="off"
      spellcheck="false"
      oninput={onInput}
      onkeydown={(e) => { if (e.key === "Enter") send(); }}
    />
    <button class="iconbtn" data-mode={store.listenMode} title={MODE_TITLES[store.listenMode] || ""} onclick={() => store.cycleListen()} aria-label="Listening mode">
      <svg class="ic ic-off" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-8.1 3.9l1.5-1.5A3 3 0 0 0 15 12h2zM5.3 3.9 20.1 18.7l-1.4 1.4-3.2-3.2A6.9 6.9 0 0 1 13 18.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 7.1 4.5l-1.5-1.5A3 3 0 0 1 9 12V9.4L3.9 5.3z"/></svg>
      <svg class="ic ic-ptt" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3h2a7 7 0 0 1-6 6.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 10 0z"/></svg>
      <svg class="ic ic-vad" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 10h2v4H3zm4-3h2v10H7zm4-4h2v18h-2zm4 4h2v10h-2zm4 3h2v4h-2z"/></svg>
    </button>
    <button
      class="sendbtn"
      class:live={(sendState === "ptt" && (store.pttDown || keyHeld)) || (sendState === "vad" && !!dictation?.live)}
      class:armed={sendState === "send" && armed}
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
  </div>
</div>
