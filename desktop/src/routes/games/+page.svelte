<script lang="ts">
  // The games page (/games): every game Game Review can open, the latest played first --
  // the ones Sidecoach watched, and the ones brought in from chess.com (the official public
  // archive), Lichess, or a PGN -- each with its accuracies once reviewed, and Review. The
  // library lives on the sidecar (profile/library.py, the store's library table).
  import { onMount } from "svelte";
  import "$lib/styles/review.css";
  import { connect, games as gamesApi, type LibraryGame } from "$lib/api";
  import { inShell, openWindow } from "$lib/shell.svelte";

  let list = $state<LibraryGame[]>([]);
  let users = $state({ chesscom: "", lichess: "" });
  let loaded = $state(false);
  let source = $state<"chesscom" | "lichess" | "pgn">("chesscom");
  let months = $state(3);
  let pgn = $state("");
  let busy = $state(false);
  let said = $state("");
  let saidBad = $state(false);
  let show = $state<"all" | LibraryGame["source"]>("all");
  let outcome = $state<"all" | "won" | "lost" | "drawn">("all");

  const SOURCES: Record<LibraryGame["source"], string> = { watched: "watched", chesscom: "chess.com", lichess: "Lichess", pgn: "PGN" };

  async function load(): Promise<void> {
    const res = await gamesApi.list();
    list = res.games;
    users = res.users;
    loaded = true;
    if (!users.chesscom && users.lichess) source = "lichess";
  }

  async function bringIn(): Promise<void> {
    busy = true;
    said = source === "pgn" ? "Reading the PGN…" : `Asking ${source === "chesscom" ? "chess.com" : "Lichess"}…`;
    saidBad = false;
    const res = await gamesApi.bringIn(source, pgn, months).catch((e) => ({ ok: false, error: String(e) }) as { ok: boolean; added?: number; error?: string });
    busy = false;
    if (!res.ok) { said = res.error || "Nothing came in."; saidBad = true; return; }
    said = res.added ? `${res.added} game${res.added === 1 ? "" : "s"} in the library.` : "No games found.";
    if (source === "pgn" && res.added) pgn = "";
    await load();
  }

  function readFile(e: Event): void {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { pgn = String(reader.result || ""); };
    reader.readAsText(file);
  }

  async function remove(game: LibraryGame): Promise<void> {
    await gamesApi.remove(game.key);
    list = list.filter((g) => g.key !== game.key);
  }

  function openReview(e: MouseEvent, key: string): void {
    if (!inShell()) return; // a browser follows the link to a tab of its own
    e.preventDefault();
    openWindow("review/" + encodeURIComponent(key));
  }

  function openSettings(): void {
    if (inShell()) openWindow("settings"); else location.href = "/settings";
  }

  // The game's outcome for the player, when the library knows which side was theirs.
  function outcomeOf(g: LibraryGame): "won" | "lost" | "drawn" | null {
    if (!g.user_side || !g.result || g.result === "*") return null;
    if (g.result === "1/2-1/2") return "drawn";
    return (g.result === "1-0") === (g.user_side === "white") ? "won" : "lost";
  }

  const shown = $derived(
    list.filter((g) => (show === "all" || g.source === show) && (outcome === "all" || outcomeOf(g) === outcome)),
  );
  const sourcesHeld = $derived([...new Set(list.map((g) => g.source))]);

  function when(g: LibraryGame): string {
    const t = g.played_at ?? g.added_at;
    const d = new Date(t * 1000);
    const days = (Date.now() - d.getTime()) / 86400000;
    if (days < 1 && d.getDate() === new Date().getDate()) return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric", year: days > 300 ? "numeric" : undefined });
  }

  onMount(() => {
    document.body.classList.add("games-page");
    (async () => { await connect(); await load(); })();
    return () => document.body.classList.remove("games-page");
  });
</script>

<svelte:head><title>sidecoach · games</title></svelte:head>

<header id="bar">
  <div class="group">
    <img class="gm-mark" src="/logo.svg" alt="" draggable="false" />
    <div class="gm-head">
      <div class="gm-name">Your games</div>
      <div class="gm-sub">{loaded ? `${list.length} game${list.length === 1 ? "" : "s"} · review any of them move by move` : "…"}</div>
    </div>
  </div>
</header>

<main>
  <section class="gm-in" aria-label="Bring games in">
    <div class="seg" role="group" aria-label="Where from">
      <button class="segbtn" aria-pressed={source === "chesscom"} onclick={() => (source = "chesscom")}>chess.com</button>
      <button class="segbtn" aria-pressed={source === "lichess"} onclick={() => (source = "lichess")}>Lichess</button>
      <button class="segbtn" aria-pressed={source === "pgn"} onclick={() => (source = "pgn")}>PGN</button>
    </div>
    {#if source === "pgn"}
      <textarea class="gm-pgn" placeholder="Paste one game or many in PGN…" spellcheck="false" bind:value={pgn}></textarea>
      <div class="gm-row">
        <label class="ghost gm-file">Open a .pgn file<input type="file" accept=".pgn,text/plain" onchange={readFile} /></label>
        <button class="primary" disabled={busy || !pgn.trim()} onclick={bringIn}>Add to the library</button>
      </div>
    {:else}
      {@const user = source === "chesscom" ? users.chesscom : users.lichess}
      {#if user}
        <div class="gm-row">
          <span class="gm-as">Your games as <b>{user}</b></span>
          {#if source === "chesscom"}
            <label class="gm-months">from the last
              <select bind:value={months}>
                {#each [1, 3, 6, 12] as m (m)}<option value={m}>{m} month{m === 1 ? "" : "s"}</option>{/each}
              </select>
            </label>
          {:else}
            <span class="gm-as muted">the last 60</span>
          {/if}
          <button class="primary" disabled={busy} onclick={bringIn}>Bring them in</button>
        </div>
      {:else}
        <div class="gm-row">
          <span class="gm-as">Your {source === "chesscom" ? "chess.com" : "Lichess"} username is not set.</span>
          <button class="ghost" onclick={openSettings}>Set it in Settings</button>
        </div>
      {/if}
    {/if}
    {#if said}<div class="gm-said" class:bad={saidBad} role="status">{said}</div>{/if}
  </section>

  <section class="gm-list" aria-label="The library">
    <div class="gm-filters">
      <div class="seg small">
        <button class="segbtn" aria-pressed={show === "all"} onclick={() => (show = "all")}>All</button>
        {#each sourcesHeld as s (s)}
          <button class="segbtn" aria-pressed={show === s} onclick={() => (show = s)}>{SOURCES[s]}</button>
        {/each}
      </div>
      <div class="seg small">
        {#each ["all", "won", "lost", "drawn"] as o (o)}
          <button class="segbtn" aria-pressed={outcome === o} onclick={() => (outcome = o as typeof outcome)}>{o === "all" ? "Any result" : o[0].toUpperCase() + o.slice(1)}</button>
        {/each}
      </div>
    </div>
    {#if loaded && !shown.length}
      <div class="gm-empty">{list.length ? "No game matches." : "No games yet. Games Sidecoach watches come here on their own; bring in your chess.com or Lichess games above, or paste a PGN."}</div>
    {/if}
    <ol class="gm-games">
      {#each shown as g (g.key)}
        {@const o = outcomeOf(g)}
        <li class="gm-game">
          <span class="gm-kind" title={SOURCES[g.source]}>{g.time_class || "—"}<small>{SOURCES[g.source]}</small></span>
          <span class="gm-players">
            <span class="gm-player" class:me={g.user_side === "white"}><i class="sw w"></i>{g.white || "White"}{#if g.white_elo}<small>({g.white_elo})</small>{/if}</span>
            <span class="gm-player" class:me={g.user_side === "black"}><i class="sw b"></i>{g.black || "Black"}{#if g.black_elo}<small>({g.black_elo})</small>{/if}</span>
          </span>
          <span class="gm-result {o ?? ''}" title={o ?? ""}>{g.result && g.result !== "*" ? g.result.replaceAll("1/2", "½") : "—"}</span>
          <span class="gm-acc" title="Accuracy, White then Black">
            {#if g.accuracy}
              <b class="acc w">{g.accuracy.white?.toFixed(1) ?? "–"}</b><b class="acc b">{g.accuracy.black?.toFixed(1) ?? "–"}</b>
            {/if}
          </span>
          <span class="gm-when">{when(g)}</span>
          <span class="gm-go">
            <a class="primary linkbtn" href="/review/{encodeURIComponent(g.key)}" target="_blank" rel="noopener" onclick={(e) => openReview(e, g.key)}>{g.reviewed ? "Open review" : "Review"}</a>
            {#if g.source !== "watched"}
              <button class="iconbtn small gm-del" title="Take it out of the library" aria-label="Take it out of the library" onclick={() => remove(g)}>
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path fill="currentColor" d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
              </button>
            {/if}
          </span>
        </li>
      {/each}
    </ol>
  </section>
</main>
