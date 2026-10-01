<script lang="ts">
  // Talking to the coach: a text box; the round button that is Send while there is text,
  // the hold-to-talk button in push-to-talk mode and the listening indicator in voice mode;
  // beside it a caret for how the coach hears you (off, hold to talk, always listening), as a
  // call's mic has its ^ -- one mic, not a second one that cycled the mode (2026-10-01);
  // and the quick chips.
  //
  // Dictation: what is said goes into the box as it is recognised (the sidecar streams
  // partial results while the key is held), and sends on its own a second after the
  // utterance ends -- unless more is said, which appends, or the person types, which makes
  // the words theirs to edit and send (Tai, 2026-09-22).
  import { panel } from "./api";
  import { keyName as hotkeyName } from "./hotkey";
  import { store } from "./panel.svelte";

  let text = $state("");
  let input = $state<HTMLInputElement | null>(null);

  // The dictation merged into the box so far, and what the box held before it began.
  let dictated = "";
  let typed = "";
  // After a cancel, the sidecar's old text may still arrive on a poll or two: ignored
  // until it reports an empty dictation once.
  let awaitingClear = false;

  const dictation = $derived(store.state.dictation);
  // No speech recognition -- the models were not built or the load failed -- means no
  // listening button to press.
  const canListen = $derived(store.state.stt_ready !== false);
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
    : canListen ? "idle"
    : "quiet",
  );
  // The phone has no push-to-talk key (its ptt_key is empty): the button, or Space.
  const keyName = $derived(store.state.ptt_key === "" ? "" : hotkeyName(store.state.ptt_key || "ctrl_r"));
  const sendTitle = $derived(
    sendState === "send" ? (armed ? "Sending in a moment (type to keep it)" : "Send (Enter)")
    : sendState === "ptt" ? (talking ? "Listening…" : keyName ? `Hold to talk (or hold Space or ${keyName})` : "Hold to talk (or hold Space)")
    : sendState === "vad" ? "Listening"
    : canListen ? "Type a question (the arrow beside sets how the coach hears you)"
    : "Type a question",
  );

  // The listening menu: its three ways, the one in use ticked; any press elsewhere or Escape
  // closes it, as any menu.
  let modeOpen = $state(false);
  let modeEl = $state<HTMLDivElement | null>(null);
  let caretEl = $state<HTMLButtonElement | null>(null);
  const MODES = $derived([
    { id: "ptt", name: "Hold to talk", hint: keyName ? `Hold the round button, Space or ${keyName}` : store.state.ptt_key === "" ? "Hold the round button" : "Hold the round button or Space" },
    { id: "vad", name: "Always listening", hint: "The coach hears you whenever you speak" },
    { id: "off", name: "Off", hint: "Type your questions" },
  ]);
  const modeName = $derived(MODES.find((m) => m.id === store.listenMode)?.name ?? "Off");
  function pickMode(id: string): void {
    modeOpen = false;
    store.setListen(id);
  }
  function outside(e: PointerEvent): void {
    if (modeOpen && !modeEl?.contains(e.target as Node) && !caretEl?.contains(e.target as Node)) modeOpen = false;
  }
  function onKey(e: KeyboardEvent): void {
    if (modeOpen && e.key === "Escape") { modeOpen = false; caretEl?.focus(); }
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
    else if (sendState === "idle" || sendState === "quiet") input?.focus();
  }
</script>

<svelte:window onpointerdown={outside} onkeydown={onKey} />

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
    {#if canListen}
      <div class="lmode">
        <button
          bind:this={caretEl}
          class="lcaret"
          aria-haspopup="menu"
          aria-expanded={modeOpen}
          aria-label="How the coach hears you: {modeName}"
          title="How the coach hears you: {modeName}"
          onclick={() => (modeOpen = !modeOpen)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 14l5-5 5 5z" /></svg>
        </button>
        {#if modeOpen}
          <div class="lmenu" role="menu" aria-label="How the coach hears you" bind:this={modeEl}>
            <div class="lhead">How the coach hears you</div>
            {#each MODES as m (m.id)}
              <button role="menuitemradio" aria-checked={store.listenMode === m.id} onclick={() => pickMode(m.id)}>
                {#if m.id === "ptt"}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3h2a7 7 0 0 1-6 6.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 10 0z"/></svg>
                {:else if m.id === "vad"}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 10h2v4H3zm4-3h2v10H7zm4-4h2v18h-2zm4 4h2v10h-2zm4 3h2v4h-2z"/></svg>
                {:else}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-8.1 3.9l1.5-1.5A3 3 0 0 0 15 12h2zM5.3 3.9 20.1 18.7l-1.4 1.4-3.2-3.2A6.9 6.9 0 0 1 13 18.9V21h-2v-2.1A7 7 0 0 1 5 12h2a5 5 0 0 0 7.1 4.5l-1.5-1.5A3 3 0 0 1 9 12V9.4L3.9 5.3z"/></svg>{/if}
                <span>{m.name}<small>{m.hint}</small></span>
                <svg class="tick" viewBox="0 0 24 24" aria-hidden="true"><path d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z" /></svg>
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
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
