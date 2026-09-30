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
  import { connect, stats as statsApi, type Backlog, type Figure, type Rate, type StatPosition, type StatsFilter, type StatsNumbers } from "$lib/api";
  import Board from "$lib/Board.svelte";
  import { prefs } from "$lib/panel.svelte";
  import { className } from "$lib/review/words";
  import { inShell, openWindow } from "$lib/shell.svelte";
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

  // ---- how a figure reads ----
  const pct = (f: Figure | Rate | undefined) => (f?.value == null ? "–" : `${f.value}%`);
  const num = (f: Figure | undefined, unit = "") => (f?.value == null ? "–" : `${f.value}${unit}`);
  const small = (n: number | undefined) => (n ?? 0) < SMALL;
  const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleDateString([], { month: "short", day: "numeric" }) : "");
  const moveNo = (p: StatPosition) => `${Math.ceil(p.ply / 2)}${p.side === "white" ? "." : "..."}`;
  const mean = (xs: number[]) => (xs.length ? Math.round(xs.reduce((a, b) => a + b, 0) / xs.length) : null);

  // The game rating now (the last ten games' mean) and how it moved (against the ten before).
  const rating = $derived.by(() => {
    const vs = (numbers?.trends.rating ?? []).map((p) => p.v);
    const now = mean(vs.slice(-10));
    const before = mean(vs.slice(-20, -10));
    return { now, change: now != null && before != null ? now - before : null, games: vs.length };
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
    (async () => { await connect(); await load(); })();
    // While games are still being reviewed in the background, the figures grow: read them again.
    const again = setInterval(() => { if (backlog && (backlog.waiting || backlog.reviewing)) void load(); }, 30_000);
    return () => {
      clearInterval(again);
      document.body.classList.remove("stats-page");
    };
  });
</script>

<svelte:head><title>sidecoach · stats</title></svelte:head>

<!-- A figure's row: its name, its value, how many it counts; opened, what it means and the
     positions behind it. -->
{#snippet row(id: string, label: string, value: string, n: number | undefined, def: string, what = "", arg = "")}
  <div class="st-row" class:small={small(n)} class:open={opened === id}>
    <button class="st-row-head" aria-expanded={opened === id} onclick={() => toggle(id, what, arg)}>
      <span class="label">{label}</span>
      <span class="value">{value}</span>
      <span class="n" title="How many this counts">{n ?? ""}</span>
      <svg class="chev" viewBox="0 0 24 24" aria-hidden="true"><path d="M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z" /></svg>
    </button>
    {#if opened === id}
      <div class="st-more">
        <p>{def}</p>
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

{#snippet mini(p: StatPosition)}
  <a class="st-mini" href="/review/{encodeURIComponent(p.key)}#ply-{p.ply}" target="_blank" rel="noopener" title="Open in Game Review at this move" onclick={(e) => openReview(p, e)}>
    <span class="st-miniboard"><Board fen={p.fen} bottom={p.side} marks={[{ kind: "arrow", squares: [p.uci.slice(0, 2), p.uci.slice(2, 4)], color: p.label === "brilliant" || p.label === "great" || p.label === "best" ? "good" : "bad" }]} /></span>
    <span class="cap">{#if p.label}<Badge kind={p.label} />{/if}{moveNo(p)} {p.san} <small>{when(p.played_at)}</small></span>
  </a>
{/snippet}

<header id="bar">
  <div class="group">
    <img class="rv-logo" src="/logo.svg" alt="" draggable="false" />
    <div>
      <div class="rv-name">Your stats</div>
      <div class="rv-sub">
        {#if numbers}{numbers.overview.games} games{/if}{#if backlog && (backlog.waiting || backlog.reviewing)} · {backlog.waiting} more being reviewed in the background{backlog.paused ? " (paused while you play)" : ""}{/if}
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
        <span class="st-big" title="The rating your last ten games were played at, from their moves alone">{rating.now ?? "–"}</span>
        {#if rating.change}<span class="st-change" class:down={rating.change < 0}>{rating.change > 0 ? "↑" : "↓"} {Math.abs(rating.change)}</span>{/if}
      </div>
      <TrendChart points={numbers.trends.rating} smooth={10} />
      <div class="st-tiles">
        <div><span>Games</span><b>{o.games}</b></div>
        <div class="won"><span>Won {o.games ? Math.round((100 * o.wins) / o.games) : 0}%</span><b>{o.wins}</b></div>
        <div class="drawn"><span>Drawn {o.games ? Math.round((100 * o.draws) / o.games) : 0}%</span><b>{o.draws}</b></div>
        <div class="lost"><span>Lost {o.games ? Math.round((100 * o.losses) / o.games) : 0}%</span><b>{o.losses}</b></div>
      </div>
      <div class="st-card">
        <div class="st-card-title">Your play</div>
        {#each [["openings", "Openings", `${num(numbers.openings.accuracy)} accuracy`], ["tactics", "Tactics", `${pct(numbers.tactics.overall)} found`], ["strategy", "Strategy", `${num(numbers.strategy.all.overall)} accuracy`], ["endgames", "Endgames", `${num(numbers.endgames.accuracy.overall)} accuracy`], ["moves", "Brilliant Moves", `${numbers.galleries.brilliant ?? 0}`]] as [id, name, value] (id)}
          <button class="st-link-row" onclick={() => go(id as Section)}><span>{name}</span><b>{value}</b><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 6 8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z" /></svg></button>
        {/each}
      </div>
      <div class="st-card">
        <div class="st-card-title">How your games ended</div>
        <div class="st-grid ends">
          <span></span><span class="muted">Won</span><span class="muted">Drawn</span><span class="muted">Lost</span>
          {#each Object.entries(numbers.results) as [kind, r] (kind)}
            <span class="cap">{kind}</span><span>{r.won}</span><span>{r.drawn}</span><span>{r.lost}</span>
          {/each}
        </div>
      </div>
    {:else if section === "openings"}
      {@const op = numbers.openings}
      <div class="st-head"><h2>Openings</h2><span class="st-big">{num(op.accuracy)}</span><span class="muted">accuracy</span></div>
      <TrendChart points={numbers.trends.openings} smooth={5} />
      <div class="st-card">
        <div class="st-card-title">Opening stats</div>
        {@render row("op.acc", "Opening accuracy", num(op.accuracy), op.accuracy.n, DEF.openingAccuracy)}
        {@render row("op.mis", "Mistakes in the opening, a game", num(op.mistakes), op.mistakes.n, DEF.openingMistakes, "opening.mistakes")}
        {@render row("op.book", "Book moves before you leave book", num(op.book_moves), op.book_moves.n, DEF.bookMoves)}
      </div>
      <div class="st-card">
        <div class="st-card-title">
          Performance by opening
          <div class="seg small">
            <button class="segbtn" aria-pressed={openingSide === "white"} onclick={() => (openingSide = "white")}>As White</button>
            <button class="segbtn" aria-pressed={openingSide === "black"} onclick={() => (openingSide = "black")}>As Black</button>
          </div>
        </div>
        <div class="st-grid openings">
          <span class="muted">Opening</span><span class="muted">Games</span><span class="muted">Score</span><span class="muted">Accuracy</span>
          {#each op.by_opening.filter((o) => o.side === openingSide) as o (o.opening)}
            <span class:small={small(o.games)}>{o.opening}</span><span>{o.games}</span><span>{o.score == null ? "–" : `${o.score}%`}</span><span>{o.accuracy ?? "–"}</span>
          {:else}
            <span class="muted" style="grid-column: 1 / -1">No games as {openingSide} here.</span>
          {/each}
        </div>
      </div>
    {:else if section === "tactics"}
      {@const t = numbers.tactics}
      <div class="st-head"><h2>Tactics</h2><span class="st-big">{pct(t.overall)}</span><span class="muted">found</span></div>
      <TrendChart points={numbers.trends.tactics} smooth={5} unit="%" />
      <div class="st-card">
        <div class="st-card-title">
          {tacticsView === "found" ? "Tactics you found, of the ones you had" : "Tactics you handed your opponent, per 1,000 moves"}
          <div class="seg small">
            <button class="segbtn" aria-pressed={tacticsView === "found"} onclick={() => (tacticsView = "found")}>Found</button>
            <button class="segbtn" aria-pressed={tacticsView === "allowed"} onclick={() => (tacticsView = "allowed")}>Allowed</button>
          </div>
        </div>
        {#each KINDS as kind (kind)}
          {#if tacticsView === "found"}
            {@const r = t.found[kind]}
            {@render row(`tf.${kind}`, KIND_NAMES[kind], r?.n ? `${pct(r)} · ${r.hits} of ${r.n}` : "–", r?.n, DEF.found, "tactics.missed", kind)}
          {:else}
            {@const r = t.allowed[kind]}
            {@render row(`ta.${kind}`, KIND_NAMES[kind], num(r), r?.hits, DEF.allowed, "tactics.allowed", kind)}
          {/if}
        {/each}
      </div>
    {:else if section === "strategy"}
      {@const s = numbers.strategy[structure]}
      <div class="st-head"><h2>Strategy</h2><span class="st-big">{num(numbers.strategy.all.overall)}</span><span class="muted">accuracy in quiet positions</span></div>
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
        {@render row("st.all", "Strategic accuracy", num(s.overall), s.overall.n, DEF.overall, "strategy.mistakes", structure)}
        {@render row("st.att", "Attacking accuracy", num(s.attacking), s.attacking.n, DEF.attacking)}
        {@render row("st.def", "Defending accuracy", num(s.defending), s.defending.n, DEF.defending)}
        {@render row("st.space", "With more space", num(s.space), s.space.n, DEF.space)}
        {@render row("st.act", "With active pieces", num(s.activity), s.activity.n, DEF.activity)}
        {@render row("st.pawns", "Around weak pawns", num(s.structure), s.structure.n, DEF.structure)}
      </div>
    {:else if section === "endgames"}
      {@const e = numbers.endgames}
      <div class="st-head"><h2>Endgames</h2><span class="st-big">{num(e.accuracy.overall)}</span><span class="muted">accuracy</span></div>
      <TrendChart points={numbers.trends.endgames} smooth={5} />
      <div class="st-card">
        <div class="st-card-title">Your score from the endgame's start</div>
        {@render row("eg.better", "From a better position", pct(e.win.better), e.win.better.n, DEF.win)}
        {@render row("eg.equal", "From an equal position", pct(e.win.equal), e.win.equal.n, DEF.win)}
        {@render row("eg.worse", "From a worse position", pct(e.win.worse), e.win.worse.n, DEF.win)}
      </div>
      <div class="st-card">
        <div class="st-card-title">Endgame accuracy by kind</div>
        {#each [["overall", "All endgames"], ["pawn", "King and pawn"], ["rook", "Rook"], ["queen", "Queen"], ["other", "Other"]] as [kind, name] (kind)}
          {@render row(`eg.${kind}`, name, num(e.accuracy[kind]), e.accuracy[kind]?.n, DEF.endgameAccuracy, "endgame.mistakes", kind)}
        {/each}
      </div>
    {:else if section === "time"}
      {@const tm = numbers.time}
      <div class="st-head"><h2>Time</h2></div>
      <div class="st-card">
        <div class="st-card-title">Time a move</div>
        {#each ["opening", "middlegame", "endgame"] as phase (phase)}
          {@render row(`tm.${phase}`, className(phase), num(tm.by_phase[phase], " s"), tm.by_phase[phase]?.n, DEF.timePhase)}
        {/each}
      </div>
      <div class="st-card">
        <div class="st-card-title">Mistakes, by the clock</div>
        {@render row("tm.press", "Short of time", pct(tm.pressure), tm.pressure.n, DEF.pressure, "time.pressure")}
        {@render row("tm.calm", "With time to think", pct(tm.calm), tm.calm.n, DEF.pressure)}
        {@render row("tm.lost", "Games lost on time", `${tm.lost_on_time}`, tm.lost_on_time, DEF.lostOnTime)}
        {@render row("tm.won", "… while winning", `${tm.lost_on_time_winning}`, tm.lost_on_time, DEF.lostOnTimeWinning)}
      </div>
    {:else}
      <div class="st-head">
        <h2>Brilliant Moves</h2><span class="st-big">{numbers.galleries.brilliant ?? 0}</span>
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
