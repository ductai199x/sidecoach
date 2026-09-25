<script lang="ts">
  // The debug terminal and the conversation view, under one head with tabs. A record is
  // {seq, t, chan, level, text, data}: the line shows `mm:ss.s  CHAN  text`, a click
  // expands `data`, identical consecutive lines collapse into one with a ×N badge. The
  // filters (channel chips, verbose, search) only hide rows, so the whole session stays
  // and can be re-examined without polling again. The conversation shows every ask and
  // every coach message as a turn, and under a coach turn the exact turn the model was
  // given and what it wrote, for each call it took.
  import { tick } from "svelte";
  import { stickToBottom } from "./stick";
  import { stamp } from "./chess";
  import { CHANNELS, store, type TermRow } from "./panel.svelte";

  let term = $state<HTMLDivElement | null>(null);
  let convo = $state<HTMLDivElement | null>(null);

  function visible(row: TermRow): boolean {
    const f = store.filters;
    if (!f.chans[row.line.chan]) return false;
    if (row.line.level === "debug" && !f.verbose) return false;
    if (f.query && !(row.line.chan + " " + row.line.text).toLowerCase().includes(f.query)) return false;
    return true;
  }
  function pretty(data: Record<string, any>): string {
    return JSON.stringify(data, null, 1).replace(/\n\s*/g, (x) => (x.length > 1 ? "\n " : x));
  }
  function hasData(row: TermRow): boolean {
    return !!row.line.data && Object.keys(row.line.data).length > 0;
  }

  // Both panes follow their newest entry (`stick.ts`) while autoscroll is on and they show.
  const followTerminal = { enabled: () => store.autoscroll && store.pane === "terminal" };
  const followConversation = { enabled: () => store.autoscroll && store.pane === "conversation" };
  function showPane(p: "terminal" | "conversation") {
    store.pane = p;
    tick().then(() => { const el = p === "terminal" ? term : convo; if (el) el.scrollTop = el.scrollHeight; });
  }
</script>

<div class="term-head">
  <div class="tabs">
    <button class="tab" class:on={store.pane === "terminal"} onclick={() => showPane("terminal")}>debug terminal</button>
    <button class="tab" class:on={store.pane === "conversation"} title="What you and the coach said, and what the model was given each time" onclick={() => showPane("conversation")}>conversation</button>
  </div>
  <div class="chips">
    {#each CHANNELS as chan (chan)}
      <button class="chip" class:on={store.filters.chans[chan]} onclick={() => (store.filters.chans[chan] = !store.filters.chans[chan])}>{chan}</button>
    {/each}
  </div>
  <label class="ctl"><input type="checkbox" bind:checked={store.filters.verbose} /> verbose</label>
  <input type="text" class="search" placeholder="filter…" oninput={(e) => (store.filters.query = (e.currentTarget as HTMLInputElement).value.trim().toLowerCase())} />
  <span class="spacer"></span>
  <label class="ctl"><input type="checkbox" bind:checked={store.autoscroll} /> autoscroll</label>
  <button class="ghost small" onclick={() => store.clearTerminal()}>clear</button>
</div>
<div class="terminal" bind:this={term} hidden={store.pane !== "terminal"} use:stickToBottom={followTerminal}>
  {#each store.rows as row (row.key + row.line.t + (row.line.seq ?? 0))}
    {@const data = hasData(row)}
    <div
      class="row chan-{row.line.chan} lvl-{row.line.level}"
      class:has-data={data}
      class:sev-blunder={row.line.data?.severity === "blunder"}
      class:sev-mistake={row.line.data?.severity === "mistake"}
      class:sev-inaccuracy={row.line.data?.severity === "inaccuracy"}
      hidden={!visible(row)}
      role={data ? "button" : undefined}
      onclick={() => { if (data) row.open = !row.open; }}
    >
      <span class="t">{stamp(row.line.t || 0)}</span>
      <span class="c">{row.line.chan}</span>
      <span class="m">{#if data}<span class="caret">{row.open ? "▾" : "▸"}</span>{/if}{row.line.text}<span class="n">{row.count > 1 ? "×" + row.count : ""}</span></span>
      {#if data && row.open}<pre class="d">{pretty(row.line.data!)}</pre>{/if}
    </div>
  {/each}
</div>
<div class="conversation" bind:this={convo} hidden={store.pane !== "conversation"} use:stickToBottom={followConversation}>
  {#each store.convo as turn (turn.id)}
    <div class="turn {turn.who}" class:unspoken={turn.unspoken}>
      <div class="who">{turn.label}</div>{turn.text}
      {#if turn.body}
        <details><summary>the whole message</summary><pre>{turn.body}</pre></details>
      {/if}
      {#each turn.calls as call}
        <details>
          <summary>{call.title}</summary>
          {#if call.prompt}<pre>GIVEN:{"\n"}{call.prompt}</pre>{/if}
          {#if call.raw}<pre class="raw">WROTE:{"\n"}{call.raw}</pre>{/if}
        </details>
      {/each}
    </div>
  {/each}
</div>
