<script lang="ts">
  // The games page (/games): every game Game Review can open, the latest played first --
  // the ones Sidecoach watched, and the ones brought in from chess.com (the official public
  // archive), Lichess, or a PGN -- each with its accuracies once reviewed, and Review. The
  // library lives on the sidecar (profile/library.py, the store's library table). It follows
  // the other windows: a username saved in Settings shows at once, and the background
  // reviewer's progress fills the accuracies in as it goes.
  import { onMount } from "svelte";
  import "$lib/styles/review.css";
  import { connect, get, games as gamesApi, jobs as jobsApi, type Backlog, type LibraryGame } from "$lib/api";
  import { announce, inShell, onAnnounce, openSetting, openWindow } from "$lib/shell.svelte";

  let list = $state<LibraryGame[]>([]);
  let users = $state({ chesscom: "", lichess: "" });
  let backlog = $state<Backlog>(null);
  let loaded = $state(false);
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

  async function load(): Promise<void> {
    const res = await gamesApi.list().catch(() => null);
    if (!res) return;
    const reviewedBefore = list.filter((g) => g.reviewed).length;
    // A username changed in Settings: what was said of the old one goes with it.
    if (loaded && (res.users.chesscom !== users.chesscom || res.users.lichess !== users.lichess)) {
      said = "";
      saidField = null;
    }
    list = res.games;
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
  const job = $derived(backlog && (backlog.status === "reviewing" || backlog.status === "paused" || backlog.status === "waiting") ? backlog : null);
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
    const added = res.added ?? 0, already = res.already ?? 0;
    const games = (n: number) => `${n} game${n === 1 ? "" : "s"}`;
    if (!added && !already) said = source === "pgn" ? "No games in that PGN." : `No games in the last ${months} month${months === 1 ? "" : "s"}.`;
    else if (!added) said = `Nothing new: ${already === 1 ? "that game was" : `all ${already} were`} already here.`;
    else said = `${games(added)} brought in${already ? ` · ${already} already here` : ""}. ${backlog?.paused_by_you ? "The background review is paused: resume it below to review them." : "They are reviewed in the background; each shows its accuracy when done."}`;
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

  async function remove(game: LibraryGame): Promise<void> {
    await gamesApi.remove(game.key);
    list = list.filter((g) => g.key !== game.key);
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
    const jobTimer = setInterval(() => void followJob(), JOB_MS);
    // A username saved in Settings, games brought in or reviewed elsewhere: shown at once.
    const stops = [onAnnounce("settings", () => void load()), onAnnounce("games", () => void load())];
    const onFocus = () => void load();
    window.addEventListener("focus", onFocus);
    return () => {
      document.body.classList.remove("games-page");
      for (const stop of stops) stop();
      window.removeEventListener("focus", onFocus);
      clearInterval(jobTimer);
      if (following) clearTimeout(following);
    };
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
      {:else}
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
    {#if job}
      <!-- The background review as a job the player sees and runs (ui/backlog.py). -->
      <div class="gm-job" class:still={job.status !== "reviewing"} role="status" aria-label="The background review">
        <div class="gm-job-head">
          <b>{job.status === "paused" ? "Background review paused" : job.status === "waiting" ? "Background review waits for you" : "Reviewing your games in the background"}</b>
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
          {:else if job.status === "waiting"}
            It steps aside while you play a game or a review is analysing, and goes on after.
          {:else if job.current}
            Now: {job.current.white ?? "White"} v {job.current.black ?? "Black"}{job.current.total ? ` · ${job.current.done} of ${job.current.total} positions` : ""}. One game at a time, below normal priority.
          {:else}
            Getting the next game ready…
          {/if}
        </small>
      </div>
    {/if}
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
          <span class="gm-acc" title={job?.current?.key === g.key ? "Being reviewed in the background" : "Accuracy, White then Black"}>
            {#if g.accuracy}
              <b class="acc w">{g.accuracy.white?.toFixed(1) ?? "–"}</b><b class="acc b">{g.accuracy.black?.toFixed(1) ?? "–"}</b>
            {:else if job?.current?.key === g.key && job.current.total}
              <span class="gm-row-bar" class:still={job.status !== "reviewing"}><i style:width="{Math.round((100 * job.current.done) / job.current.total)}%"></i></span>
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
