<script lang="ts">
  // The Stats window (/stats; docs/stats-plan.md): the player's games added up, as chess.com's
  // Advanced Stats lays them out -- a menu of sections (the overview, openings, tactics,
  // strategy, endgames, time, brilliant moves), the filters on top, each section a trend and a
  // table of figures; a figure opens to what it means and the positions behind it, each opening
  // Game Review at that move. The numbers are the sidecar's (profile/stats.py).
  import Badge from "$lib/Badge.svelte";
  import { onMount } from "svelte";
  import "$lib/styles/review.css";
  import "$lib/styles/stats.css";
  import { connect, stats as statsApi, type Backlog, type Figure, type Rate, type StatPosition, type StatRating, type StatsFilter, type StatsNumbers } from "$lib/api";
  import Board from "$lib/Board.svelte";
  import { prefs } from "$lib/panel.svelte";
  import { className } from "$lib/review/words";
  import { inShell, openWindow } from "$lib/shell.svelte";
  import Sparkline from "$lib/stats/Sparkline.svelte";
  import TrendChart from "$lib/stats/TrendChart.svelte";

  type Section = "overview" | "openings" | "tactics" | "strategy" | "endgames" | "time" | "moves";
  // The menu's icons (Material Symbols' shapes).
  const SECTIONS: { id: Section; name: string; icon: string }[] = [
    { id: "overview", name: "Overview", icon: "M3 3h8v8H3zm10 0h8v8h-8zM3 13h8v8H3zm10 0h8v8h-8z" },
    { id: "openings", name: "Openings", icon: "M18 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V4a2 2 0 0 0-2-2zM6 4h5v8l-2.5-1.5L6 12V4z" },
    { id: "tactics", name: "Tactics", icon: "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm0-14a6 6 0 1 0 6 6 6 6 0 0 0-6-6zm0 10a4 4 0 1 1 4-4 4 4 0 0 1-4 4zm0-6a2 2 0 1 0 2 2 2 2 0 0 0-2-2z" },
    { id: "strategy", name: "Strategy", icon: "M19 22H5v-2h14v2zM17 10c-1.1 0-2 .9-2 2v5H9v-1.4c1.5-1.2 2.5-3 2.5-5.1V9.5L10 8V5.7L7.1 7.6 5.3 6.7 9 2h3.7l3.7 3.3c.4.4.6.9.6 1.5V10z" },
    { id: "endgames", name: "Endgames", icon: "M13 2h-2v2H9v2h2v2H8l1 3h1l-2 7h8l-2-7h1l1-3h-3V6h2V4h-2z" },
    { id: "time", name: "Time", icon: "M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm0 18a8 8 0 1 1 8-8 8 8 0 0 1-8 8zm.5-13H11v6l5.2 3.2.8-1.3-4.5-2.7z" },
    { id: "moves", name: "Brilliant Moves", icon: "" },
  ];
  const KINDS = ["checkmate", "fork", "pin", "discovery", "trapped piece", "removing the defender", "free piece", "sacrifice", "other"];
  const KIND_NAMES: Record<string, string> = {
    checkmate: "Checkmates",
    fork: "Forks",
    pin: "Pins",
    discovery: "Discoveries",
    "trapped piece": "Trapped pieces",
    "removing the defender": "Removing the defender",
    "free piece": "Free pieces",
    sacrifice: "Sacrifices",
    other: "Other tactics",
  };
  const SMALL = 5; // a figure of fewer is shown greyed
  const STACK_BELOW = 760; // px: the menu (210) beside a section of at least about 520

  let filter = $state<StatsFilter>(readFilter());
  let section = $state<Section>((prefs.getStr("statsSection") as Section) || "overview");
  let numbers = $state<StatsNumbers | null>(null);
  let backlog = $state<Backlog>(null);
  let failure = $state("");
  let tacticsView = $state<"found" | "allowed">("found");
  let structure = $state("all");
  let openingSide = $state<"white" | "black">("white");
  let gallery = $state("brilliant");
  // The figure opened, and the positions behind it (kept per figure while the filter stands).
  let opened = $state<string | null>(null);
  let behind = $state<Record<string, StatPosition[]>>({});

  function readFilter(): StatsFilter {
    try {
      return { time_class: "", days: 0, opponents: "all", ...JSON.parse(prefs.getStr("statsFilter") ?? "{}") };
    } catch {
      return { time_class: "", days: 0, opponents: "all" };
    }
  }

  async function load(): Promise<void> {
    prefs.setStr("statsFilter", JSON.stringify(filter));
    const got = await statsApi.get(filter).catch((e) => ({ ok: false as const, error: String(e) }));
    if (!got.ok || !("stats" in got)) { failure = "error" in got ? got.error : "Stats could not be read."; return; }
    numbers = got.stats;
    backlog = got.backlog;
    // The openings open on the side the player has games with.
    const sides = new Set(numbers.openings.by_opening.map((o) => o.side));
    if (!sides.has(openingSide) && sides.size) openingSide = [...sides][0] as "white" | "black";
    behind = {};
    if (section === "moves") void positions("gallery", gallery);
  }

  function setFilter(change: Partial<StatsFilter>): void {
    filter = { ...filter, ...change };
    void load();
  }

  function go(next: Section): void {
    section = next;
    opened = null;
    prefs.setStr("statsSection", next);
    if (next === "moves") void positions("gallery", gallery);
  }

  async function positions(what: string, arg: string): Promise<void> {
    const id = `${what}:${arg}`;
    if (behind[id]) return;
    const got = await statsApi.positions(what, arg, filter);
    behind = { ...behind, [id]: got.ok ? got.positions : [] };
  }

  function toggle(id: string, what: string, arg: string): void {
    opened = opened === id ? null : id;
    if (opened && what) void positions(what, arg);
  }

  function openReview(p: StatPosition, e: MouseEvent): void {
    if (!inShell()) return; // a browser follows the link to a tab of its own
    e.preventDefault();
    openWindow(`review/${encodeURIComponent(p.key)}#ply-${p.ply}`);
  }

  function leave(): void {
    window.close();
    if (history.length > 1) history.back();
  }

  // ---- how a figure reads ----
  const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;
  // A figure's value (or the peers') as the page writes it.
  type Show = (v: number | null | undefined) => string;
  const tenth: Show = (v) => (v == null ? "–" : v.toFixed(1));
  const percent: Show = (v) => (v == null ? "–" : `${Math.round(v)}%`);
  const seconds: Show = (v) => (v == null ? "–" : `${v.toFixed(1)} s`);
  const whole: Show = (v) => (v == null ? "–" : String(Math.round(v)));
  const pct = (f: Figure | Rate | undefined) => percent(f?.value);
  const num = (f: Figure | undefined, unit = "") => (f?.value == null ? "–" : `${f.value}${unit}`);
  // A grade's badge said in words: how far ahead of the peers or behind the figure is.
  const GRADE_WORDS: Record<string, string> = {
    brilliant: "Far ahead of your peers",
    great: "Well ahead of your peers",
    best: "Ahead of your peers",
    excellent: "A little ahead of your peers",
    good: "On a par with your peers",
    inaccuracy: "A little behind your peers",
    mistake: "Behind your peers",
    blunder: "Well behind your peers",
  };
  const small = (n: number | undefined) => (n ?? 0) < SMALL;
  const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleDateString([], { month: "short", day: "numeric" }) : "");
  const moveNo = (p: StatPosition) => `${Math.ceil(p.ply / 2)}${p.side === "white" ? "." : "..."}`;

  // The rating now (the last ten games' moves pooled, the trend's last point) and how it moved
  // against the ten games before them (said only once there are twenty).
  const rating = $derived.by(() => {
    const vs = (numbers?.trends.rating ?? []).map((p) => p.v);
    const now = vs.length ? vs[vs.length - 1] : null;
    const before = vs.length >= 20 ? vs[vs.length - 11] : null;
    return { now, change: now != null && before != null ? now - before : null, games: vs.length };
  });

  // The results: won, drawn, lost.
  const results = $derived(
    numbers
      ? [
          { kind: "won", name: "Won", n: numbers.overview.wins },
          { kind: "drawn", name: "Drawn", n: numbers.overview.draws },
          { kind: "lost", name: "Lost", n: numbers.overview.losses },
        ]
      : [],
  );

  // The overview's cards: each section's figure, its trend where it has one, else a line on it.
  const icon = (id: Section) => SECTIONS.find((s) => s.id === id)?.icon ?? "";
  const cards = $derived.by(() => {
    if (!numbers) return [];
    const n = numbers, tm = n.time.by_phase;
    // A category with a rating shows it, its key figure after; without one, the figure alone.
    const rated = (id: "openings" | "tactics" | "strategy" | "endgames", figure: string, unit: string) => {
      const r = n.ratings[id];
      return r ? { value: String(r.rating), unit: `rating · ${figure} ${unit}` } : { value: figure, unit };
    };
    return [
      { id: "openings" as Section, name: "Openings", ...rated("openings", num(n.openings.accuracy), "accuracy"), trend: n.trends.openings, sub: "" },
      { id: "tactics" as Section, name: "Tactics", ...rated("tactics", pct(n.tactics.overall), "found"), trend: n.trends.tactics, sub: "" },
      { id: "strategy" as Section, name: "Strategy", ...rated("strategy", num(n.strategy.all.overall), "accuracy"), trend: n.trends.strategy, sub: "" },
      { id: "endgames" as Section, name: "Endgames", ...rated("endgames", num(n.endgames.accuracy.overall), "accuracy"), trend: n.trends.endgames, sub: "" },
      { id: "time" as Section, name: "Time", value: pct(n.time.pressure), unit: "mistakes short of time", trend: null, sub: ["opening", "middlegame", "endgame"].map((p) => `${className(p)} ${num(tm[p], "s")}`).join(" · ") },
      { id: "moves" as Section, name: "Brilliant Moves", value: String(n.galleries.brilliant ?? 0), unit: "brilliant", trend: null, sub: `${n.galleries.great ?? 0} great · ${n.galleries.miss ?? 0} misses · ${n.galleries.blunder ?? 0} blunders` },
    ].map((c) => ({ ...c, icon: icon(c.id) }));
  });

  // What each figure means (ours: chess.com publishes none), shown when the row opens.
  const DEF: Record<string, string> = {
    accuracy: "Your average accuracy per game, over every move you played.",
    openingAccuracy: "Your average accuracy over the moves of the opening.",
    openingMistakes: "How many mistakes, misses and blunders you make in the opening, per game.",
    bookMoves: "In games where you left the opening book first: how many book moves you had played.",
    found: "Of the positions where one move was clearly best and it was a tactic of this kind you could see (a mate only in three or fewer), how often you played it.",
    allowed: "How often your move handed your opponent a tactic of this kind, per 1,000 of your moves.",
    overall: "Your accuracy in positions with no tactic in them, out of the opening book.",
    attacking: "Your accuracy in quiet positions where you stand better.",
    defending: "Your accuracy in quiet positions where you stand worse.",
    space: "Your accuracy in quiet positions where you have more space.",
    activity: "Your accuracy in quiet positions where a rook or queen of yours stands on an open file, or a knight or bishop on an outpost.",
    structure: "Your accuracy in quiet positions where either side has a weak pawn: isolated, backward, doubled or hanging.",
    win: "Your score from the moment the endgame began, by how you stood then (a draw counts half).",
    endgameAccuracy: "Your accuracy over the moves of the endgame, by what was left besides kings and pawns.",
    timePhase: "Your average time a move in this phase, from the game's clock.",
    pressure: "How often your move was a mistake, miss or blunder with little time left (under 10 seconds in bullet, 20 in blitz, a minute in rapid), against with more time.",
    lostOnTime: "Games you lost because your clock ran out.",
    lostOnTimeWinning: "Games you lost on time while you stood better at your last move.",
  };

  onMount(() => {
    document.body.classList.add("stats-page");
    // Stacked (the menu a row of tabs above the section) when the window is too narrow for the
    // menu beside a section: by width alone, there being no board.
    const fit = () => document.documentElement.classList.toggle("stacked", window.innerWidth < STACK_BELOW);
    fit();
    window.addEventListener("resize", fit);
    (async () => { await connect(); await load(); })();
    // While games are still being reviewed in the background, the figures grow: read them again.
    const again = setInterval(() => { if (backlog && (backlog.waiting || backlog.reviewing)) void load(); }, 30_000);
    return () => {
      clearInterval(again);
      window.removeEventListener("resize", fit);
      document.documentElement.classList.remove("stacked");
      document.body.classList.remove("stats-page");
    };
  });
</script>

<svelte:head><title>sidecoach · stats</title></svelte:head>

<!-- A figure's row: its name, the grade against the peers, yours, the peers', what it rests on;
     opened, what it means and the positions behind it. -->
{#snippet row(id: string, label: string, f: Figure | undefined, show: Show, basis: string, def: string, what = "", arg = "")}
  {@const few = small(f?.n)}
  <div class="st-row" class:small={few} class:open={opened === id}>
    <button class="st-row-head" aria-expanded={opened === id} onclick={() => toggle(id, what, arg)}>
      <span class="label">{label}</span>
      <span class="grade">{#if f?.grade && !few}<Badge kind={f.grade} size={20} title={GRADE_WORDS[f.grade]} />{/if}</span>
      <span class="value">{show(f?.value)}</span>
      <span class="peers">{f?.peers != null ? show(f.peers) : ""}</span>
      <span class="n">{f?.n ? basis : ""}</span>
      <svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z" /></svg>
    </button>
    {#if opened === id}
      <div class="st-more">
        <p>{def}</p>
        {#if f?.n}<p class="st-basis">Based on {basis}{#if f.peers != null}; your peers: {show(f.peers)}{/if}{#if f.grade && !few}. {GRADE_WORDS[f.grade]}{/if}.</p>{/if}
        {#if what}
          {@const found = behind[`${what}:${arg}`]}
          {#if !found}
            <p class="muted">…</p>
          {:else if !found.length}
            <p class="muted">No positions here yet.</p>
          {:else}
            <div class="st-boards">
              {#each found.slice(0, 8) as p (p.key + p.ply)}
                {@render mini(p)}
              {/each}
            </div>
          {/if}
        {/if}
      </div>
    {/if}
  </div>
{/snippet}

<!-- The columns of a card of rows. -->
{#snippet cols()}
  <div class="st-cols" aria-hidden="true"><span></span><span>Grade</span><span>You</span><span>Peers</span><span>Based on</span><span></span></div>
{/snippet}

<!-- A section's head: the category's rating (its figure when too few moves rate it), and the
     level the peers are read at. -->
{#snippet head(title: string, r: StatRating | undefined, figure: string, unit: string)}
  <div class="st-head">
    <h2>{title}</h2>
    {#if r}
      <span class="st-big" title="Most likely {r.rating}; from {r.low} to {r.high} on {plural(r.moves, 'move')}">{r.rating}</span><span class="muted">rating · {figure} {unit}</span>
    {:else}
      <span class="st-big">{figure}</span><span class="muted">{unit}</span>
    {/if}
  </div>
  {@render peersNote()}
{/snippet}

{#snippet peersNote()}
  {#if numbers?.peers}
    <p class="st-note">Your peers: players rated about {numbers.peers.rating}, {numbers.peers.from === "rating" ? "your rating in these games" : "the level of your moves in these games"}.</p>
  {/if}
{/snippet}

{#snippet mini(p: StatPosition)}
  <a class="st-mini" href="/review/{encodeURIComponent(p.key)}#ply-{p.ply}" target="_blank" rel="noopener" title="Open in Game Review at this move" onclick={(e) => openReview(p, e)}>
    <span class="st-miniboard"><Board fen={p.fen} bottom={p.side} marks={[{ kind: "arrow", squares: [p.uci.slice(0, 2), p.uci.slice(2, 4)], color: p.label === "brilliant" || p.label === "great" || p.label === "best" ? "good" : "bad" }]} /></span>
    <span class="cap">{#if p.label}<Badge kind={p.label} />{/if}{moveNo(p)} {p.san} <small>{when(p.played_at)}</small></span>
  </a>
{/snippet}

<header id="bar">
  <div class="group">
    <button class="iconbtn rv-back" aria-label="Close Stats" title="Close Stats" onclick={leave}>
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"><path fill="currentColor" d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"/></svg>
    </button>
    <img class="rv-logo" src="/logo.svg" alt="" draggable="false" />
    <div>
      <div class="rv-name">Your stats</div>
      <div class="rv-sub">
        {#if numbers}{plural(numbers.overview.games, "game")} reviewed{/if}{#if backlog && (backlog.waiting || backlog.reviewing)}{" · "}{backlog.waiting} more being reviewed{backlog.paused ? " (paused while you play)" : ""}{/if}
      </div>
    </div>
  </div>
</header>

<main>
  <nav class="st-menu" aria-label="Sections">
    {#each SECTIONS as s (s.id)}
      <button class="st-item" aria-current={section === s.id ? "page" : undefined} onclick={() => go(s.id)}>
        {#if s.icon}<svg viewBox="0 0 24 24" aria-hidden="true"><path d={s.icon} /></svg>{:else}<Badge kind="brilliant" />{/if}
        <span>{s.name}</span>
      </button>
    {/each}
  </nav>

  <section class="st-content">
    <div class="st-filters">
      <div class="seg small" role="group" aria-label="Time control">
        {#each [["", "All"], ["bullet", "Bullet"], ["blitz", "Blitz"], ["rapid", "Rapid"], ["daily", "Daily"]] as [value, name] (value)}
          <button class="segbtn" aria-pressed={filter.time_class === value} onclick={() => setFilter({ time_class: value })}>{name}</button>
        {/each}
      </div>
      <div class="seg small" role="group" aria-label="Opponents">
        {#each [["all", "Everyone"], ["people", "People"], ["bots", "Bots"]] as [value, name] (value)}
          <button class="segbtn" aria-pressed={filter.opponents === value} onclick={() => setFilter({ opponents: value as StatsFilter["opponents"] })}>{name}</button>
        {/each}
      </div>
      <div class="seg small" role="group" aria-label="Period">
        {#each [[30, "30 days"], [90, "90 days"], [365, "A year"], [0, "All time"]] as [value, name] (value)}
          <button class="segbtn" aria-pressed={filter.days === value} onclick={() => setFilter({ days: value as number })}>{name}</button>
        {/each}
      </div>
    </div>

    {#if failure}
      <div class="rv-failed">{failure}</div>
    {:else if !numbers}
      <div class="rv-progress">Adding up your games…</div>
    {:else if !numbers.overview.games}
      <div class="st-empty">No reviewed games {filter.time_class || filter.days || filter.opponents !== "all" ? "for these filters" : "yet"}. Games you bring in, and games Sidecoach watches, are reviewed in the background{backlog?.waiting ? ` — ${backlog.waiting} waiting` : ""}.</div>
    {:else if section === "overview"}
      {@const o = numbers.overview}
      <div class="st-head">
        <h2>Game rating</h2>
        <span class="st-big" title="The rating your games were played at, from their moves alone">{rating.now ?? "–"}</span>
        {#if rating.change}<span class="st-change" class:down={rating.change < 0}>{rating.change > 0 ? "↑" : "↓"} {Math.abs(rating.change)}</span>{/if}
        <span class="muted st-span">{rating.games >= 10 ? "your last 10 games" : `your ${plural(rating.games, "game")}`}</span>
      </div>
      <TrendChart points={numbers.trends.rating} />
      <div class="st-card st-results">
        <div class="st-card-title">Results <span class="muted">{plural(o.games, "game")}</span></div>
        <div class="st-bar" role="img" aria-label="{o.wins} won, {o.draws} drawn, {o.losses} lost">
          {#each results as r (r.kind)}
            {#if r.n}<i class={r.kind} style="flex: {r.n}"></i>{/if}
          {/each}
        </div>
        <div class="st-legend">
          {#each results as r (r.kind)}
            <span class={r.kind}><i></i>{r.name} <b>{r.n}</b> <small>{o.games ? Math.round((100 * r.n) / o.games) : 0}%</small></span>
          {/each}
        </div>
      </div>
      <div class="st-cats">
        {#each cards as c (c.id)}
          <button class="st-cat" onclick={() => go(c.id)}>
            <span class="name">
              {#if c.icon}<svg viewBox="0 0 24 24" aria-hidden="true"><path d={c.icon} /></svg>{:else}<Badge kind="brilliant" size={20} />{/if}
              {c.name}
              <svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" /></svg>
            </span>
            <span class="fig"><b>{c.value}</b> <small>{c.unit}</small></span>
            {#if c.trend}<Sparkline points={c.trend} />{:else}<span class="sub">{c.sub}</span>{/if}
          </button>
        {/each}
      </div>
      <div class="st-card">
        <div class="st-card-title">How your games ended</div>
        <div class="st-table ends">
          <span class="h first"></span><span class="h">Won</span><span class="h">Drawn</span><span class="h">Lost</span>
          {#each Object.entries(numbers.results) as [kind, r] (kind)}
            <span class="first kind">{kind}</span>
            <b class="won" class:zero={!r.won}>{r.won}</b>
            <b class:zero={!r.drawn}>{r.drawn}</b>
            <b class="lost" class:zero={!r.lost}>{r.lost}</b>
          {/each}
        </div>
      </div>
    {:else if section === "openings"}
      {@const op = numbers.openings}
      {@render head("Openings", numbers.ratings.openings, num(op.accuracy), "accuracy")}
      <TrendChart points={numbers.trends.openings} smooth={5} />
      <div class="st-card">
        <div class="st-card-title">Opening stats</div>
        {@render cols()}
        {@render row("op.acc", "Opening accuracy", op.accuracy, tenth, plural(op.accuracy.n, "move"), DEF.openingAccuracy)}
        {@render row("op.mis", "Mistakes in the opening, a game", op.mistakes, tenth, plural(op.mistakes.n, "game"), DEF.openingMistakes, "opening.mistakes")}
        {@render row("op.book", "Book moves before you leave book", op.book_moves, tenth, plural(op.book_moves.n, "game"), DEF.bookMoves)}
      </div>
      <div class="st-card">
        <div class="st-card-title">
          Performance by opening
          <div class="seg small">
            <button class="segbtn" aria-pressed={openingSide === "white"} onclick={() => (openingSide = "white")}>As White</button>
            <button class="segbtn" aria-pressed={openingSide === "black"} onclick={() => (openingSide = "black")}>As Black</button>
          </div>
        </div>
        <div class="st-table openings">
          <span class="h first">Opening</span><span class="h">Games</span><span class="h">Score</span><span class="h">Accuracy</span>
          {#each op.by_opening.filter((o) => o.side === openingSide) as o (o.opening)}
            <span class="first" class:small={small(o.games)} title={o.opening}>{o.opening}</span>
            <b class:zero={small(o.games)}>{o.games}</b>
            <b class:zero={o.score == null}>{o.score == null ? "–" : `${Math.round(o.score)}%`}</b>
            <b class:zero={o.accuracy == null}>{o.accuracy ?? "–"}</b>
          {:else}
            <span class="none">No games as {openingSide} here.</span>
          {/each}
        </div>
      </div>
    {:else if section === "tactics"}
      {@const t = numbers.tactics}
      {@render head("Tactics", numbers.ratings.tactics, pct(t.overall), "found")}
      <TrendChart points={numbers.trends.tactics} smooth={5} unit="%" />
      <div class="st-card">
        <div class="st-card-title">
          {tacticsView === "found" ? "Tactics you found, of the ones you had" : "Tactics you handed your opponent, per 1,000 moves"}
          <div class="seg small">
            <button class="segbtn" aria-pressed={tacticsView === "found"} onclick={() => (tacticsView = "found")}>Found</button>
            <button class="segbtn" aria-pressed={tacticsView === "allowed"} onclick={() => (tacticsView = "allowed")}>Allowed</button>
          </div>
        </div>
        {@render cols()}
        {#each KINDS as kind (kind)}
          {#if tacticsView === "found"}
            {@const r = t.found[kind]}
            {@render row(`tf.${kind}`, KIND_NAMES[kind], r, percent, r ? `${r.hits} of ${r.n}` : "", DEF.found, "tactics.missed", kind)}
          {:else}
            {@const r = t.allowed[kind]}
            {@render row(`ta.${kind}`, KIND_NAMES[kind], r, tenth, r ? plural(r.hits ?? 0, "time") : "", DEF.allowed, "tactics.allowed", kind)}
          {/if}
        {/each}
      </div>
    {:else if section === "strategy"}
      {@const s = numbers.strategy[structure]}
      {@render head("Strategy", numbers.ratings.strategy, num(numbers.strategy.all.overall), "accuracy in quiet positions")}
      <TrendChart points={numbers.trends.strategy} smooth={5} />
      <div class="st-card">
        <div class="st-card-title">
          Strategy stats
          <div class="seg small">
            {#each [["all", "All"], ["open", "Open"], ["semi-open", "Semi-open"], ["closed", "Closed"]] as [value, name] (value)}
              <button class="segbtn" aria-pressed={structure === value} onclick={() => (structure = value)}>{name}</button>
            {/each}
          </div>
        </div>
        {@render cols()}
        {@render row("st.all", "Strategic accuracy", s.overall, tenth, plural(s.overall.n, "move"), DEF.overall, "strategy.mistakes", structure)}
        {@render row("st.att", "Attacking accuracy", s.attacking, tenth, plural(s.attacking.n, "move"), DEF.attacking)}
        {@render row("st.def", "Defending accuracy", s.defending, tenth, plural(s.defending.n, "move"), DEF.defending)}
        {@render row("st.space", "With more space", s.space, tenth, plural(s.space.n, "move"), DEF.space)}
        {@render row("st.act", "With active pieces", s.activity, tenth, plural(s.activity.n, "move"), DEF.activity)}
        {@render row("st.pawns", "Around weak pawns", s.structure, tenth, plural(s.structure.n, "move"), DEF.structure)}
      </div>
    {:else if section === "endgames"}
      {@const e = numbers.endgames}
      {@render head("Endgames", numbers.ratings.endgames, num(e.accuracy.overall), "accuracy")}
      <TrendChart points={numbers.trends.endgames} smooth={5} />
      <div class="st-card">
        <div class="st-card-title">Your score from the endgame's start</div>
        {@render cols()}
        {@render row("eg.better", "From a better position", e.win.better, percent, plural(e.win.better.n, "game"), DEF.win)}
        {@render row("eg.equal", "From an equal position", e.win.equal, percent, plural(e.win.equal.n, "game"), DEF.win)}
        {@render row("eg.worse", "From a worse position", e.win.worse, percent, plural(e.win.worse.n, "game"), DEF.win)}
      </div>
      <div class="st-card">
        <div class="st-card-title">Endgame accuracy by kind</div>
        {@render cols()}
        {#each [["overall", "All endgames"], ["pawn", "King and pawn"], ["rook", "Rook"], ["queen", "Queen"], ["other", "Other"]] as [kind, name] (kind)}
          {@render row(`eg.${kind}`, name, e.accuracy[kind], tenth, plural(e.accuracy[kind]?.n ?? 0, "move"), DEF.endgameAccuracy, "endgame.mistakes", kind)}
        {/each}
      </div>
    {:else if section === "time"}
      {@const tm = numbers.time}
      <div class="st-head"><h2>Time</h2></div>
      {@render peersNote()}
      <div class="st-card">
        <div class="st-card-title">Time a move</div>
        {@render cols()}
        {#each ["opening", "middlegame", "endgame"] as phase (phase)}
          {@render row(`tm.${phase}`, className(phase), tm.by_phase[phase], seconds, plural(tm.by_phase[phase]?.n ?? 0, "move"), DEF.timePhase)}
        {/each}
      </div>
      <div class="st-card">
        <div class="st-card-title">Mistakes, by the clock</div>
        {@render cols()}
        {@render row("tm.press", "Mistakes when short of time", tm.pressure, percent, plural(tm.pressure.n, "move"), DEF.pressure, "time.pressure")}
        {@render row("tm.calm", "Mistakes with time to think", tm.calm, percent, plural(tm.calm.n, "move"), DEF.pressure)}
        {@render row("tm.lost", "Games lost on time", { value: tm.lost_on_time, n: numbers.overview.games }, whole, plural(numbers.overview.games, "game"), DEF.lostOnTime)}
        {@render row("tm.won", "… while winning", { value: tm.lost_on_time_winning, n: tm.lost_on_time }, whole, plural(tm.lost_on_time, "loss", "losses"), DEF.lostOnTimeWinning)}
      </div>
    {:else}
      <div class="st-head">
        <h2>Brilliant Moves</h2><span class="st-big">{numbers.galleries[gallery] ?? 0}</span>
        <div class="seg small st-head-seg">
          {#each [["brilliant", "Brilliant"], ["great", "Great"], ["miss", "Misses"], ["blunder", "Blunders"]] as [label, name] (label)}
            <button class="segbtn" aria-pressed={gallery === label} onclick={() => { gallery = label; void positions("gallery", label); }}>{name} · {numbers.galleries[label] ?? 0}</button>
          {/each}
        </div>
      </div>
      {@const found = behind[`gallery:${gallery}`]}
      <div class="st-boards st-gallery">
        {#each found ?? [] as p (p.key + p.ply)}
          {@render mini(p)}
        {:else}
          <div class="st-empty">{found ? `No ${gallery === "miss" ? "misses" : gallery === "blunder" ? "blunders" : `${gallery} moves`} here yet.` : "…"}</div>
        {/each}
      </div>
    {/if}
  </section>
</main>
