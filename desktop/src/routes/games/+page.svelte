<script lang="ts">
  // The games page (/games): every game Game Review can open, the latest played first --
  // the ones Sidecoach watched, and the ones brought in from chess.com (the official public
  // archive), Lichess, or a PGN -- each with its accuracies once reviewed, and Review. The
  // library lives on the sidecar (profile/library.py, the store's library table). It follows
  // the other windows: a username saved in Settings shows at once, and the background
  // reviewer's progress fills the accuracies in as it goes.
  import { onMount } from "svelte";
  import { SvelteSet } from "svelte/reactivity";
  import "$lib/styles/review.css";
  import { connect, failureText, get, games as gamesApi, jobs as jobsApi, type Backlog, type LibraryGame } from "$lib/api";
  import { announce, inShell, onAnnounce, openSetting, openWindow } from "$lib/shell.svelte";

  let list = $state<LibraryGame[]>([]);
  let users = $state({ chesscom: "", lichess: "" });
  let backlog = $state<Backlog>(null);
  let loaded = $state(false);
  let loadError = $state("");
  let source = $state<"chesscom" | "lichess" | "pgn">("chesscom");
  let months = $state(3);
  let pgn = $state("");
  let busy = $state(false);
  let said = $state("");
  let saidBad = $state(false);
  // The setting to fix when the site knows no such player ("profile.chesscom_user").
  let saidField = $state<string | null>(null);
  let show = $state<"all" | LibraryGame["source"]>("all");
  let outcome = $state<"all" | "won" | "lost" | "drawn">("all");
  let speed = $state("all");
  let since = $state(0);

  const SOURCES: Record<LibraryGame["source"], string> = { watched: "watched", chesscom: "chess.com", lichess: "Lichess", pgn: "PGN" };
  const SITE = { chesscom: "chess.com", lichess: "Lichess" } as const;
  // While the background reviewer works, the list is asked again this often: its accuracies
  // come in game by game.
  const FOLLOW_MS = 4000;
  let following: ReturnType<typeof setTimeout> | null = null;
  // Just after games came in, followed until the reviewer takes them up (it answers "nothing
  // to do" for a moment before it has looked).
  const EXPECT_MS = 12000;
  let expectUntil = 0;

  // Each load is numbered, and a removal moves the number on: a list asked before the newest
  // load, or before a removal, never lands (it would bring the removed rows back a moment).
  let loadSeq = 0;
  async function load(): Promise<void> {
    const mine = ++loadSeq;
    let failed = "";
    const res = await gamesApi.list().catch((e) => { failed = failureText(e); return null; });
    if (mine !== loadSeq) return;
    // Not loaded (never yet): said in the list's place, with Try again. A later reload that
    // fails keeps the list shown.
    if (!res) { if (!loaded) loadError = failed; return; }
    loadError = "";
    const reviewedBefore = list.filter((g) => g.reviewed).length;
    // A username changed in Settings: what was said of the old one goes with it.
    if (loaded && (res.users.chesscom !== users.chesscom || res.users.lichess !== users.lichess)) {
      said = "";
      saidField = null;
    }
    list = res.games;
    // A game picked that is gone (removed in another window) is no longer picked.
    const held = new Set(list.map((g) => g.key));
    for (const key of [...picked]) if (!held.has(key)) picked.delete(key);
    users = res.users;
    backlog = res.backlog;
    // A game reviewed since: Stats has one more to count.
    if (loaded && list.filter((g) => g.reviewed).length > reviewedBefore) void announce("games");
    if (!loaded && !users.chesscom && users.lichess) source = "lichess";
    loaded = true;
    if (following) clearTimeout(following);
    const working = backlog?.status === "reviewing";
    if (working) expectUntil = 0;
    following = working || Date.now() < expectUntil ? setTimeout(load, working ? FOLLOW_MS : 1000) : null;
  }

  // The page's first load, and Try again's: the sidecar found, then the list.
  async function firstLoad(): Promise<void> {
    try {
      await connect();
    } catch (e) {
      loadError = failureText(e);
      return;
    }
    await load();
  }

  // The job's card follows the reviewer closely (a light ask, not the whole list).
  const JOB_MS = 1500;
  async function followJob(): Promise<void> {
    const got = await get<{ reviews: Backlog }>("/api/jobs").catch(() => null);
    if (got) backlog = got.reviews;
  }
  async function pauseReviews(paused: boolean): Promise<void> {
    const res = await jobsApi.pauseReviews(paused).catch(() => null);
    if (res?.reviews) backlog = res.reviews;
    void announce("games");
  }
  function minutesLeft(s: number): string {
    if (s < 90) return "under 2 min left";
    const m = Math.round(s / 60);
    return m < 60 ? `about ${m} min left` : `about ${Math.floor(m / 60)} h ${m % 60} min left`;
  }
  const job = $derived(backlog && (backlog.status === "reviewing" || backlog.status === "paused") ? backlog : null);
  const jobFrac = $derived(job && job.total ? Math.min(1, (job.done + (job.current && job.current.total ? job.current.done / job.current.total : 0)) / job.total) : 0);

  async function bringIn(): Promise<void> {
    busy = true;
    said = source === "pgn" ? "Reading the PGN…" : `Asking ${SITE[source]} for your games…`;
    saidBad = false;
    saidField = null;
    const res = await gamesApi.bringIn(source, pgn, months).catch(() => ({ ok: false, error: "Sidecoach did not answer. Try again in a moment." }) as Awaited<ReturnType<typeof gamesApi.bringIn>>);
    busy = false;
    if (!res.ok) {
      said = res.error || "Nothing came in.";
      saidBad = true;
      saidField = res.field ?? null;
      return;
    }
    const added = res.added ?? 0, already = res.already ?? 0, out = res.left_out ?? 0;
    const games = (n: number) => `${n} game${n === 1 ? "" : "s"}`;
    // The games the player removed before stay out (a site's; a pasted game comes back).
    const outWords = out ? `${games(out)} you removed left out` : "";
    if (!added && !already && !out) said = source === "pgn" ? "No games in that PGN." : `No games in the last ${months} month${months === 1 ? "" : "s"}.`;
    else if (!added && !already) said = `Nothing new: ${out === 1 ? "the game you removed stays" : `the ${out} games you removed stay`} out.`;
    else if (!added) said = `Nothing new: ${already === 1 ? "that game was" : `${out ? "" : "all "}${already} were`} already here${out ? ` · ${outWords}` : ""}.`;
    else {
      // No background review (the phone's, Tai 2026-09-30): the player reviews the games they pick.
      const next = !backlog
        ? "Press Review on one to review it; the review is kept and opens at once after."
        : backlog.paused_by_you
          ? "The background review is paused: resume it below to review them."
          : "They are reviewed in the background; each shows its accuracy when done.";
      said = `${games(added)} brought in${already ? ` · ${already} already here` : ""}${out ? ` · ${outWords}` : ""}. ${next}`;
    }
    if (source === "pgn" && added) pgn = "";
    if (added && backlog) expectUntil = Date.now() + EXPECT_MS;
    await load();
    if (added) void announce("games");
  }

  function readFile(e: Event): void {
    const file = (e.currentTarget as HTMLInputElement).files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { pgn = String(reader.result || ""); };
    reader.readAsText(file);
  }

  // ---- select mode, as Gmail's: a checkbox on every row, the toolbar turned into the action
  // bar (select all, how many, Delete, Cancel); a range by shift-click, a long press on a phone.
  let selecting = $state(false);
  const picked = new SvelteSet<string>();
  let anchor: string | null = null; // the row a shift-click selects from
  function startSelecting(first?: string): void {
    selecting = true;
    picked.clear();
    anchor = first ?? null;
    if (first) picked.add(first);
  }
  function stopSelecting(): void {
    selecting = false;
    picked.clear();
    anchor = null;
  }
  function togglePick(key: string, range: boolean): void {
    const keys = shown.map((g) => g.key);
    const from = anchor ? keys.indexOf(anchor) : -1;
    if (range && from >= 0) {
      const to = keys.indexOf(key);
      for (const k of keys.slice(Math.min(from, to), Math.max(from, to) + 1)) picked.add(k);
    } else if (picked.has(key)) picked.delete(key);
    else picked.add(key);
    anchor = key;
  }
  function toggleAll(): void {
    if (allPicked) picked.clear();
    else for (const g of shown) picked.add(g.key);
  }

  /** A long press on a row (a phone has no shift-click): select mode with that row picked; the
   *  tap that ends it does nothing else. */
  const LONG_MS = 450;
  function longPress(node: HTMLElement, key: string) {
    let timer: ReturnType<typeof setTimeout> | null = null;
    let at = { x: 0, y: 0 };
    let fired = false;
    let current = key;
    const cancel = () => {
      if (timer) clearTimeout(timer);
      timer = null;
    };
    const down = (e: PointerEvent) => {
      if (selecting || e.pointerType === "mouse") return;
      cancel();
      fired = false;
      at = { x: e.clientX, y: e.clientY };
      timer = setTimeout(() => {
        timer = null;
        fired = true;
        startSelecting(current);
      }, LONG_MS);
    };
    const move = (e: PointerEvent) => {
      if (timer && Math.hypot(e.clientX - at.x, e.clientY - at.y) > 10) cancel();
    };
    const click = (e: MouseEvent) => {
      if (!fired) return;
      fired = false;
      e.preventDefault();
      e.stopPropagation();
    };
    const menu = (e: Event) => {
      if (timer || fired) e.preventDefault();
    };
    node.addEventListener("pointerdown", down);
    node.addEventListener("pointermove", move);
    node.addEventListener("pointerup", cancel);
    node.addEventListener("pointercancel", cancel);
    node.addEventListener("click", click, true);
    node.addEventListener("contextmenu", menu);
    return {
      update(k: string) {
        current = k;
      },
      destroy() {
        cancel();
        node.removeEventListener("pointerdown", down);
        node.removeEventListener("pointermove", move);
        node.removeEventListener("pointerup", cancel);
        node.removeEventListener("pointercancel", cancel);
        node.removeEventListener("click", click, true);
        node.removeEventListener("contextmenu", menu);
      },
    };
  }

  // The toolbar is stuck at the top once the marker just above it has scrolled out of view.
  let stuck = $state(false);
  function stuckWatch(node: HTMLElement) {
    const seen = new IntersectionObserver(([e]) => (stuck = !e.isIntersecting && e.boundingClientRect.top < (e.rootBounds?.top ?? 0)), {
      root: node.closest("main"),
    });
    seen.observe(node);
    return { destroy: () => seen.disconnect() };
  }

  // The bar's back arrow, as Stats' and the review's: the window closes (the app's), or the
  // page goes back where it was opened from (a browser tab, the phone).
  function leave(): void {
    window.close();
    if (history.length > 1) history.back();
  }

  // Removed in one go, then a few seconds' Undo (the sidecar keeps their rows a while longer).
  const UNDO_MS = 8000;
  let undo = $state<{ keys: string[]; label: string } | null>(null);
  let undoTimer: ReturnType<typeof setTimeout> | null = null;
  async function removePicked(): Promise<void> {
    const games = pickedShown;
    if (!games.length) return;
    const keys = games.map((g) => g.key);
    const res = await gamesApi.remove(keys).catch(() => null);
    if (!res?.ok) {
      said = "Sidecoach did not answer: nothing was removed. Try again in a moment.";
      saidBad = true;
      return;
    }
    const gone = new Set(res.removed ?? keys);
    list = list.filter((g) => !gone.has(g.key));
    void load(); // supersedes any list asked before the removal
    stopSelecting();
    undo = { keys, label: games.length === 1 ? `${games[0].white || "White"} v ${games[0].black || "Black"}` : `${games.length} games` };
    if (undoTimer) clearTimeout(undoTimer);
    undoTimer = setTimeout(() => (undo = null), UNDO_MS);
    void announce("games");
  }
  async function undoRemove(): Promise<void> {
    if (!undo) return;
    const keys = undo.keys;
    undo = null;
    if (undoTimer) clearTimeout(undoTimer);
    const res = await gamesApi.restore(keys).catch(() => null);
    if (!res?.ok) {
      said = res?.error || "Sidecoach did not answer: the games stay removed.";
      saidBad = true;
    }
    await load();
    void announce("games");
  }

  function openReview(e: MouseEvent, key: string): void {
    if (!inShell()) return; // a browser follows the link to a tab of its own
    e.preventDefault();
    openWindow("review/" + encodeURIComponent(key));
  }

  /** Settings at the username for `site`: its tab, the field lit. */
  function toUsername(site: "chesscom" | "lichess"): void {
    void openSetting(`profile.${site}_user`);
  }

  // The game's outcome for the player, when the library knows which side was theirs.
  function outcomeOf(g: LibraryGame): "won" | "lost" | "drawn" | null {
    if (!g.user_side || !g.result || g.result === "*") return null;
    if (g.result === "1/2-1/2") return "drawn";
    return (g.result === "1-0") === (g.user_side === "white") ? "won" : "lost";
  }

  // The time controls the library holds, fastest first, and how far back the list goes (days;
  // 0 every game): the plan's G, beside where from and how it went.
  const SPEEDS = ["bullet", "blitz", "rapid", "classical", "daily"];
  const speedsHeld = $derived(SPEEDS.filter((t) => list.some((g) => g.time_class === t)));

  // The list's search (a player's name) and order; a game with nothing to sort by goes last.
  let query = $state("");
  type Order = "new" | "old" | "best" | "worst" | "strong";
  let order = $state<Order>("new");
  const ORDERS: [Order, string][] = [
    ["new", "Newest first"],
    ["old", "Oldest first"],
    ["best", "Your best accuracy"],
    ["worst", "Your worst accuracy"],
    ["strong", "Strongest opponent"],
  ];
  const playedAt = (g: LibraryGame) => g.played_at ?? g.added_at;
  const yours = (g: LibraryGame) => (g.user_side && g.accuracy ? g.accuracy[g.user_side] : null) ?? null;
  const theirElo = (g: LibraryGame) => (g.user_side === "white" ? g.black_elo : g.user_side === "black" ? g.white_elo : null) ?? null;
  function ordered(games: LibraryGame[]): LibraryGame[] {
    const last = (v: number | null, f: (n: number) => number) => (v == null ? Infinity : f(v));
    const by: Record<Order, (g: LibraryGame) => number> = {
      new: (g) => -playedAt(g),
      old: (g) => playedAt(g),
      best: (g) => last(yours(g), (n) => -n),
      worst: (g) => last(yours(g), (n) => n),
      strong: (g) => last(theirElo(g), (n) => -n),
    };
    const key = by[order];
    return [...games].sort((a, b) => key(a) - key(b) || playedAt(b) - playedAt(a));
  }
  const needle = $derived(query.trim().toLowerCase());
  const shown = $derived(
    ordered(
      list.filter(
        (g) =>
          (show === "all" || g.source === show) &&
          (outcome === "all" || outcomeOf(g) === outcome) &&
          (speed === "all" || g.time_class === speed) &&
          (!since || playedAt(g) >= Date.now() / 1000 - since * 86400) &&
          (!needle || `${g.white ?? ""} ${g.black ?? ""}`.toLowerCase().includes(needle)),
      ),
    ),
  );
  // What the action bar acts on: the games picked that are shown.
  const pickedShown = $derived(shown.filter((g) => picked.has(g.key)));
  const allPicked = $derived(shown.length > 0 && pickedShown.length === shown.length);
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
    void firstLoad();
    const jobTimer = setInterval(() => void followJob(), JOB_MS);
    // A username saved in Settings, games brought in or reviewed elsewhere: shown at once.
    const stops = [onAnnounce("settings", () => void load()), onAnnounce("games", () => void load())];
    const onFocus = () => void load();
    window.addEventListener("focus", onFocus);
    // Select mode's keys: Esc leaves it, Delete removes what is picked, Ctrl+A picks all shown.
    const onKey = (e: KeyboardEvent) => {
      if (!selecting || (e.target as HTMLElement | null)?.closest?.("input[type=search], textarea, select")) return;
      if (e.key === "Escape") {
        e.preventDefault();
        stopSelecting();
      } else if (e.key === "Delete" && pickedShown.length) {
        e.preventDefault();
        void removePicked();
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "a") {
        e.preventDefault();
        for (const g of shown) picked.add(g.key);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.classList.remove("games-page");
      for (const stop of stops) stop();
      window.removeEventListener("focus", onFocus);
      window.removeEventListener("keydown", onKey);
      clearInterval(jobTimer);
      if (following) clearTimeout(following);
    };
  });
</script>

<svelte:head><title>sidecoach · games</title></svelte:head>

<header id="bar">
  <div class="group">
    <button class="iconbtn rv-back" aria-label="Close Your games" title="Close Your games" onclick={leave}>
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <img class="gm-mark" src="/logo.svg" alt="" draggable="false" />
    <div class="gm-head">
      <div class="gm-name">Your games</div>
      <div class="gm-sub">{loaded ? `${list.length} game${list.length === 1 ? "" : "s"} · review any of them move by move` : loadError ? "" : "…"}</div>
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
          <span class="gm-as">Your games as <b>{user}</b> <button type="button" class="gm-link" onclick={() => toUsername(source as "chesscom" | "lichess")}>change</button></span>
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
      {:else if loaded}
        <div class="gm-row">
          <span class="gm-as">Your {SITE[source]} username is not set.</span>
          <button class="ghost" onclick={() => toUsername(source as "chesscom" | "lichess")}>Set it in Settings</button>
        </div>
      {/if}
    {/if}
    {#if said}
      <div class="gm-said" class:bad={saidBad} role="status">
        {#if busy}<i class="gm-spin" aria-hidden="true"></i>{/if}{said}
        {#if saidField}<button type="button" class="gm-link" onclick={() => void openSetting(saidField!)}>Fix it in Settings</button>{/if}
      </div>
    {/if}
  </section>

  <section class="gm-list" aria-label="The library">
    <!-- The list's toolbar, at the top while the list scrolls; in select mode the action bar. -->
    <div class="gm-stick" use:stuckWatch aria-hidden="true"></div>
    <div class="gm-tools" class:selecting class:stuck role="toolbar" aria-label={selecting ? "The games selected" : "Your games"}>
      {#if selecting}
        <input class="gm-box" type="checkbox" checked={allPicked} indeterminate={pickedShown.length > 0 && !allPicked} onchange={toggleAll} aria-label="Select every game shown" title="Select every game shown (Ctrl+A)" />
        <span class="gm-count picked"><b>{pickedShown.length}</b> selected</span>
        <button class="gm-tool danger" disabled={!pickedShown.length} onclick={removePicked} title="Remove the selected games, their reviews and their Stats (Delete)">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z" /></svg><span>Delete</span>
        </button>
        <button class="gm-tool gm-cancel" onclick={stopSelecting} title="Leave select mode (Esc)" aria-label="Cancel">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" /></svg><span>Cancel</span>
        </button>
      {:else}
        <button class="gm-tool" disabled={!shown.length} onclick={() => startSelecting()} title="Select games, to remove them">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M22 7h-9v2h9V7zm0 8h-9v2h9v-2zM5.54 11 2 7.46l1.41-1.41 2.12 2.12 4.24-4.24 1.41 1.41L5.54 11zm0 8L2 15.46l1.41-1.41 2.12 2.12 4.24-4.24 1.41 1.41L5.54 19z" /></svg><span>Select</span>
        </button>
        <label class="gm-search">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z" /></svg>
          <input type="search" placeholder="Search players" aria-label="Search players" bind:value={query} />
        </label>
        <label class="gm-sort" title="Order">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18h6v-2H3v2zM3 6v2h18V6H3zm0 7h12v-2H3v2z" /></svg>
          <select aria-label="Order" bind:value={order}>
            {#each ORDERS as [value, label] (value)}<option {value}>{label}</option>{/each}
          </select>
        </label>
        <span class="gm-count">{shown.length === list.length ? `${list.length} game${list.length === 1 ? "" : "s"}` : `${shown.length} of ${list.length}`}</span>
      {/if}
    </div>
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
      {#if speedsHeld.length > 1}
        <div class="seg small">
          <button class="segbtn" aria-pressed={speed === "all"} onclick={() => (speed = "all")}>Any time</button>
          {#each speedsHeld as t (t)}
            <button class="segbtn" aria-pressed={speed === t} onclick={() => (speed = t)}>{t[0].toUpperCase() + t.slice(1)}</button>
          {/each}
        </div>
      {/if}
      <select class="gm-since" aria-label="Played when" bind:value={since}>
        <option value={0}>Any date</option>
        <option value={7}>Last 7 days</option>
        <option value={30}>Last 30 days</option>
        <option value={90}>Last 3 months</option>
        <option value={365}>Last year</option>
      </select>
    </div>
    {#if job}
      <!-- The background review as a job the player sees and runs (ui/backlog.py). -->
      <div class="gm-job" class:still={job.status !== "reviewing"} role="status" aria-label="The background review">
        <div class="gm-job-head">
          <b>{job.status === "paused" ? "Background review paused" : "Reviewing your games in the background"}</b>
          <span class="gm-job-count">{job.done} of {job.total} games{job.eta_s != null ? ` · ${minutesLeft(job.eta_s)}` : ""}</span>
          {#if job.status === "paused"}
            <button class="primary gm-job-btn" onclick={() => pauseReviews(false)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>Resume</button>
          {:else}
            <button class="gm-job-btn" title="What is done is kept; it goes on from there when you resume" onclick={() => pauseReviews(true)}><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>Pause</button>
          {/if}
        </div>
        <span class="gm-job-bar" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(jobFrac * 100)}><i style:width="{Math.round(jobFrac * 100)}%"></i></span>
        <small class="gm-job-detail">
          {#if job.status === "paused"}
            Paused by you: nothing is lost, and Sidecoach stays paused after a restart. Stats counts the games reviewed so far.
          {:else if job.current}
            Now: {job.current.white ?? "White"} v {job.current.black ?? "Black"}{job.current.total ? ` · ${job.current.done} of ${job.current.total} positions` : ""}. One game at a time, below normal priority.
          {:else}
            Getting the next game ready…
          {/if}
        </small>
      </div>
    {/if}
    {#if loadError && !loaded}
      <div class="gm-empty gm-said bad" role="alert">Couldn't load your games. {loadError} <button type="button" class="gm-link" onclick={() => void firstLoad()}>Try again</button></div>
    {/if}
    {#if loaded && !shown.length}
      <div class="gm-empty">{list.length ? "No game matches." : "No games yet. Games Sidecoach watches come here on their own; bring in your chess.com or Lichess games above, or paste a PGN."}</div>
    {/if}
    {#snippet cells(g: LibraryGame)}
      {@const o = outcomeOf(g)}
      <span class="gm-kind" title={SOURCES[g.source]}>{g.time_class || "—"}<small>{SOURCES[g.source]}</small></span>
      <span class="gm-players">
        <span class="gm-player" class:me={g.user_side === "white"}><i class="sw w"></i>{g.white || "White"}{#if g.white_elo}<small>({g.white_elo})</small>{/if}</span>
        <span class="gm-player" class:me={g.user_side === "black"}><i class="sw b"></i>{g.black || "Black"}{#if g.black_elo}<small>({g.black_elo})</small>{/if}</span>
      </span>
      <span class="gm-result {o ?? ''}" title={o ?? ""}>{g.result && g.result !== "*" ? g.result.replaceAll("1/2", "½") : "—"}</span>
      <span class="gm-acc" title={job?.current?.key === g.key ? "Being reviewed in the background" : "Accuracy, White then Black"}>
        {#if g.accuracy}
          <b class="acc w">{g.accuracy.white?.toFixed(1) ?? "–"}</b><b class="acc b">{g.accuracy.black?.toFixed(1) ?? "–"}</b>
        {:else if job?.current?.key === g.key && job.current.total}
          <span class="gm-row-bar" class:still={job.status !== "reviewing"}><i style:width="{Math.round((100 * job.current.done) / job.current.total)}%"></i></span>
        {/if}
      </span>
      <span class="gm-when">{when(g)}</span>
    {/snippet}
    <ol class="gm-games" class:selecting>
      {#each shown as g (g.key)}
        <li use:longPress={g.key}>
          {#if selecting}
            {@const on = picked.has(g.key)}
            <!-- In select mode the whole row is its checkbox's label: a click anywhere picks it
                 (the label clicks the box, with the shift key of the click: a range). -->
            <label class="gm-game" class:picked={on}>
              <span class="gm-check">
                <input
                  class="gm-box"
                  type="checkbox"
                  checked={on}
                  aria-label="Select {g.white || 'White'} v {g.black || 'Black'}"
                  onclick={(e) => {
                    e.preventDefault();
                    togglePick(g.key, e.shiftKey);
                  }}
                />
              </span>
              {@render cells(g)}
            </label>
          {:else}
            <div class="gm-game">
              {@render cells(g)}
              <span class="gm-go">
                <a class="primary linkbtn" href="/review/{encodeURIComponent(g.key)}" target="_blank" rel="noopener" onclick={(e) => openReview(e, g.key)}>{g.reviewed ? "Open review" : "Review"}</a>
              </span>
            </div>
          {/if}
        </li>
      {/each}
    </ol>
  </section>
  {#if undo}
    <div class="gm-undo" role="status">
      <span>Removed <b>{undo.label}</b></span>
      <button class="gm-undo-btn" onclick={undoRemove}>Undo</button>
    </div>
  {/if}
</main>
