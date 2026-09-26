"""The coach's benchmark: players from 800 to 2500, a frozen split, fixed moments, numbers with
their uncertainty (Tai, 2026-09-25: "each round is different...so how tf are you able to know
if what you change is stable and generalize? There's no exclusive train and test set?").

    games   Nova against Nova at 800, 1200, 1600, 2000 and 2500, twelve games a level, the
            coached side alternating -> data/bench/games/L<level>.pgn
    facts   their engine facts at the app's settings, a game a core
            -> data/eval/facts/bench/ (tools/eval/facts_parallel.py bench)
    draw    the split, by game, half dev and half test within each level (seeded), and the
            moments, drawn once from the templates' play-through: per level, DEV_MOMENTS
            (dev) or TEST_MOMENTS (test), a third each from the moments after a move, the
            hints and the answers, spread over the classes and the games. Frozen in
            data/eval/bench/manifest.json: never drawn again (--force starts a new benchmark).
            The drawn moments as the templates meet them go to data/eval/bench/draw/, for
            the free checks (tools/eval/align.py) and the local-model smoke run.
    run     the coach on one split's moments: the model answers the drawn moments only, the
            history before each is the templates' (a coach call a moment, not a game's
            worth), the coach's rating each game's level. Every drawn moment must come back
            with its class, or the run stops: the code no longer meets the frozen moments.
            -> data/eval/bench/runs/<name>/items/ (tools/eval/parity.py and rubric.py judge it)
    report  a judged run's rates with 95% intervals, by level and by family; with --vs,
            paired against another run on the same moments (McNemar on pass, a bootstrap on
            the rubric).

The coach's rating is the game's level, +-100: the coach pitches at it (prompt.persona_text).
The ask rotation, the templates' bags and the engine's word on a named move are seeded or
cached, so a run meets the moments exactly as the draw did.

    uv run --no-sync python tools/eval/bench.py games --workers 16
    uv run --no-sync python tools/eval/facts_parallel.py bench --workers 20
    uv run --no-sync python tools/eval/bench.py draw
    uv run --no-sync python tools/eval/bench.py run --split dev --name dev-base \\
        --coach openai:gpt-5.6-luna@medium --budget-usd 1 [--dry-run]
    uv run --no-sync python tools/eval/parity.py --items data/eval/bench/runs/dev-base/items \\
        --run bench-dev-base --judge openai:gpt-5.6-luna@none --moments-per-game 0 --budget-usd 2
    uv run --no-sync python tools/eval/rubric.py --run bench-dev-base \\
        --items data/eval/bench/runs/dev-base/items --floor --judge openai:gpt-5.6-luna@none \\
        --budget-usd 1
    uv run --no-sync python tools/eval/bench.py report --run bench-dev-base [--vs bench-dev-x]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import multiprocessing as mp
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools" / "bots"))

import chess.pgn  # noqa: E402

LEVELS = (800, 1200, 1600, 2000, 2500)
GAMES_PER_LEVEL = 12
# Moments a level: a third from each family.
DEV_MOMENTS = 30
TEST_MOMENTS = 60
SEED = 2509
# What the moments are, by what the coach is doing: the word after a move, the hint ladder,
# and the answers to the player's other questions.
FAMILIES: dict[str, tuple[str, ...]] = {
    "after_move": ("verdict", "verdict_weaker", "lesson"),
    "hint": ("decisive_question", "stuck_hint", "ask_hint", "ask_plan", "ask_move", "ask_line"),
    "answer": (
        "ask_eval",
        "ask_threat",
        "ask_named_move",
        "ask_last_move",
        "ask_opponent_move",
        "ask_game_so_far",
        "ask_opening",
        "ask_material",
        "game_over",
    ),
}
FAMILY_OF = {cls: fam for fam, classes in FAMILIES.items() for cls in classes}

GAMES = ROOT / "data" / "bench" / "games"
FACTS = ROOT / "data" / "eval" / "facts" / "bench"
BENCH = ROOT / "data" / "eval" / "bench"
MANIFEST = BENCH / "manifest.json"
ENGINE_CACHE = BENCH / "engine_cache.json"


def level_of(stem: str) -> int:
    """The level of a bench game from its name: "L1600__004" is 1600."""
    return int(stem.split("__", 1)[0].removeprefix("L"))


# ---- games -------------------------------------------------------------------------------


def cmd_games(args: argparse.Namespace) -> None:
    import play as bots  # pyright: ignore[reportMissingImports]

    GAMES.mkdir(parents=True, exist_ok=True)
    for level in LEVELS:
        path = GAMES / f"L{level}.pgn"
        if path.exists() and not args.force:
            print(f"{path.name}: there already; --force plays it again")
            continue
        side = bots.parse_side(str(level), None)
        jobs = [(SEED * 1_000_003 + level * 1_000 + i, side, side) for i in range(GAMES_PER_LEVEL)]
        t0 = time.time()
        with mp.Pool(args.workers, initializer=bots._init) as pool:
            pgns = pool.map(bots.play, jobs)
        with path.open("w", encoding="utf-8") as fh:
            for i, text in enumerate(pgns):
                game = chess.pgn.read_game(io.StringIO(text))
                assert game is not None
                # The coached side alternates: half the games from each side of the board.
                game.headers["CoachedSide"] = "white" if i % 2 == 0 else "black"
                fh.write(str(game) + "\n\n")
        results = Counter(t.split('[Result "', 1)[1].split('"', 1)[0] for t in pgns)
        plies = [len(list(chess.pgn.read_game(io.StringIO(t)).mainline_moves())) for t in pgns]  # type: ignore[union-attr]
        print(
            f"L{level}: {len(pgns)} games, {dict(results)}, plies {min(plies)}-{max(plies)},"
            f" {time.time() - t0:.0f} s -> {path.name}",
            flush=True,
        )


# ---- the play-through ------------------------------------------------------------------------


def settings_for(level: int):
    """The app's settings, with the player's rating the game's level (+-100)."""
    from sidecoach.config import load_settings

    settings = load_settings().model_copy(deep=True)
    settings.profile.rating_low = max(100, level - 100)
    settings.profile.rating_high = level + 100
    return settings


def _engine_cache() -> dict:
    return json.loads(ENGINE_CACHE.read_text(encoding="utf-8")) if ENGINE_CACHE.exists() else {}


def play_game(
    stem: str,
    *,
    provider=None,
    ledger=None,
    only: set[tuple] | None = None,
    stop: int | None = None,
    engine_cache: dict | None = None,
    version: str | None = None,
) -> tuple[list[dict], dict]:
    """One bench game through the coach at its level: the items, and the engine answers new
    to the cache. The ask rotation is seeded by the game, so every run meets the same
    questions at the same plies."""
    from corpus import CachedEngine, build_game

    engine = CachedEngine(engine_cache if engine_cache is not None else _engine_cache())
    try:
        items = build_game(
            "bench",
            FACTS / f"{stem}.json",
            random.Random(f"bench:{SEED}:{stem}"),
            None,
            engine,
            provider=provider,
            ledger=ledger,
            settings=settings_for(level_of(stem)),
            version=version,
            only=only,
            stop_after_ply=stop,
        )
    finally:
        engine.close()
    return items, dict(engine.new)


def _play_templates(stem: str) -> tuple[str, list[dict], dict]:
    import stamp

    items, new = play_game(stem, version=stamp.code_version())
    return stem, items, new


def _save_engine_cache(new: dict) -> None:
    cache = _engine_cache()
    cache.update(new)
    BENCH.mkdir(parents=True, exist_ok=True)
    ENGINE_CACHE.write_text(json.dumps(cache, indent=0), encoding="utf-8")


# ---- draw -----------------------------------------------------------------------------------


def split_games(stems: list[str]) -> dict[str, str]:
    """Half of each level's games for dev, half for test: a seeded shuffle, by game, so the
    moments of one game never sit on both sides."""
    out: dict[str, str] = {}
    by_level: dict[int, list[str]] = defaultdict(list)
    for stem in sorted(stems):
        by_level[level_of(stem)].append(stem)
    for level, games in by_level.items():
        order = sorted(games)
        random.Random(f"{SEED}:split:{level}").shuffle(order)
        half = len(order) // 2
        for i, stem in enumerate(order):
            out[stem] = "dev" if i < half else "test"
    return out


def draw_moments(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    """`n` moments from one level's split: a third from each family, taking the classes in
    turn (shuffled) so none crowds out the rest, and at most an even share (+1) from any one
    game."""
    per_family = n // len(FAMILIES)
    games = {it["game"] for it in items}
    cap = math.ceil(per_family / max(1, len(games))) + 1
    chosen: list[dict] = []
    for classes in FAMILIES.values():
        pools: dict[str, list[dict]] = {}
        for cls in classes:
            pool = [it for it in items if it["class"] == cls]
            rng.shuffle(pool)
            if pool:
                pools[cls] = pool
        order = list(pools)
        rng.shuffle(order)
        taken: list[dict] = []
        from_game: Counter = Counter()
        while len(taken) < per_family and any(pools.values()):
            for cls in order:
                pool = pools[cls]
                while pool and from_game[pool[0]["game"]] >= cap:
                    pool.pop(0)
                if pool and len(taken) < per_family:
                    it = pool.pop(0)
                    taken.append(it)
                    from_game[it["game"]] += 1
        chosen += taken
    return chosen


def _play_all_templates(stems: list[str], workers: int) -> list[dict]:
    """Every game through the templates, in parallel; the engine answers kept."""
    items: list[dict] = []
    new_engine: dict = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_play_templates, stem) for stem in stems]
        for i, done in enumerate(as_completed(futures), 1):
            stem, got, new = done.result()
            items += got
            new_engine.update(new)
            print(f"  [{i}/{len(stems)}] {stem}: {len(got)} moments", flush=True)
    _save_engine_cache(new_engine)
    return items


def cmd_refresh(args: argparse.Namespace) -> None:
    """The frozen moments as the code now meets them, for the free checks: every game
    through the templates again, the drawn moments picked out by their keys (never drawn
    again), each checked to come back with its class."""
    manifest = load_manifest()
    stems = sorted(manifest["games"])
    items = _play_all_templates(stems, args.workers)
    by_key = {(it["game"], tuple(it["moment_key"])): it for it in items}
    for part, moments in manifest["moments"].items():
        out: list[dict] = []
        for m in moments:
            it = by_key.get((m["game"], tuple(m["key"])))
            if it is None or it["class"] != m["class"]:
                raise SystemExit(
                    f"{m['id']} ({m['class']}) did not come back as drawn: the code no longer"
                    " meets the frozen moments"
                )
            out.append(it)
        with (BENCH / "draw" / f"{part}.jsonl").open("w", encoding="utf-8") as fh:
            for it in out:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{part}: {len(out)} moments refreshed")


def cmd_draw(args: argparse.Namespace) -> None:
    import stamp

    if MANIFEST.exists() and not args.force:
        raise SystemExit(f"{MANIFEST} exists: the benchmark is frozen (--force starts a new one)")
    stems = sorted(p.stem for p in FACTS.glob("*.json"))
    if not stems:
        raise SystemExit(f"no facts in {FACTS}: run tools/eval/facts_parallel.py bench first")
    split = split_games(stems)
    version = stamp.code_version()
    items: list[dict] = []
    new_engine: dict = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(_play_templates, stem) for stem in stems]
        for i, done in enumerate(as_completed(futures), 1):
            stem, got, new = done.result()
            items += got
            new_engine.update(new)
            print(f"  [{i}/{len(stems)}] {stem}: {len(got)} moments", flush=True)
    _save_engine_cache(new_engine)
    games_meta = {}
    moments: dict[str, list[dict]] = {"dev": [], "test": []}
    drawn_items: dict[str, list[dict]] = {"dev": [], "test": []}
    for level in LEVELS:
        for part, n in (("dev", DEV_MOMENTS), ("test", TEST_MOMENTS)):
            pool = [
                it
                for it in items
                if level_of(it["game"]) == level
                and split[it["game"]] == part
                and it["class"] in FAMILY_OF
            ]
            chosen = draw_moments(pool, n, random.Random(f"{SEED}:draw:{level}:{part}"))
            drawn_items[part] += chosen
            moments[part] += [
                {
                    "id": it["id"],
                    "game": it["game"],
                    "key": it["moment_key"],
                    "class": it["class"],
                    "family": FAMILY_OF[it["class"]],
                    "level": level,
                }
                for it in chosen
            ]
    for stem in stems:
        facts = json.loads((FACTS / f"{stem}.json").read_text(encoding="utf-8"))
        games_meta[stem] = {
            "level": level_of(stem),
            "side": facts["user_side"],
            "split": split[stem],
        }
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "code": version,
        "seed": SEED,
        "levels": list(LEVELS),
        "games": games_meta,
        "moments": moments,
    }
    BENCH.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (BENCH / "draw").mkdir(exist_ok=True)
    for part, chosen in drawn_items.items():
        with (BENCH / "draw" / f"{part}.jsonl").open("w", encoding="utf-8") as fh:
            for it in chosen:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    for part in ("dev", "test"):
        got = moments[part]
        by = Counter((m["level"], m["family"]) for m in got)
        print(f"{part}: {len(got)} moments; {dict(sorted(by.items()))}")
        print(f"   classes: {dict(Counter(m['class'] for m in got).most_common())}")
    print(f"wrote {MANIFEST}")


# ---- run ------------------------------------------------------------------------------------


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"no {MANIFEST}: run `bench.py draw` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def cmd_run(args: argparse.Namespace) -> None:
    import estimate
    import stamp
    from judge import Ledger, settings_prices
    from providers import coach_provider

    manifest = load_manifest()
    wanted = manifest["moments"][args.split]
    by_game: dict[str, list[dict]] = defaultdict(list)
    for m in wanted:
        by_game[m["game"]].append(m)
    version = stamp.code_version()
    if args.dry_run:
        drawn = [
            json.loads(line)
            for line in (BENCH / "draw" / f"{args.split}.jsonl").open(encoding="utf-8")
        ]
        requests = [
            {"game": it["game"], "system": it["system"], "user": str(it["messages"][-1]["content"])}
            for it in drawn
        ]
        cost = estimate.coach_estimate(
            requests, settings_prices(), completion=estimate.coach_completion(args.coach)
        )
        print(f"{args.split}: {len(wanted)} moments, the coach: {cost.line()}")
        return
    out = BENCH / "runs" / args.name / "items"
    out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(
        f"bench-{args.name}",
        "bench",
        args.coach.split(":", 1)[-1].split("@", 1)[0],
        budget_usd=args.budget_usd,
        resume=True,
        version=version,
    )
    provider = coach_provider(args.coach, ledger)
    cache = _engine_cache()
    lock = threading.Lock()
    new_engine: dict = {}

    def one(stem: str) -> str:
        done = out / f"{stem}.jsonl"
        if done.exists():
            return f"{stem}: done before"
        moments = by_game[stem]
        keys = {tuple(m["key"]) for m in moments}
        items, new = play_game(
            stem,
            provider=provider,
            ledger=ledger,
            only=keys,
            stop=max(m["key"][0] for m in moments),
            engine_cache=dict(cache),
            version=version,
        )
        with lock:
            new_engine.update(new)
        got = {tuple(it["moment_key"]): it for it in items if it.get("reply")}
        for m in moments:
            it = got.get(tuple(m["key"]))
            if it is None or it["class"] != m["class"]:
                raise SystemExit(
                    f"{stem}: the moment {m['id']} ({m['class']}) did not come back as drawn --"
                    " the code no longer meets the frozen moments; draw a new benchmark"
                )
        with done.open("w", encoding="utf-8") as fh:
            for m in moments:
                it = got[tuple(m["key"])]
                it["bench"] = {"level": m["level"], "family": m["family"], "split": args.split}
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
        return f"{stem}: {len(moments)} moments"

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one, stem) for stem in sorted(by_game)]
        for i, done in enumerate(as_completed(futures), 1):
            print(f"  [{i}/{len(futures)}] {done.result()}", flush=True)
    if new_engine:
        _save_engine_cache(new_engine)
    print(f"wrote {out}; {ledger.summary()}")


# ---- report ---------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% interval of a rate of k in n (Wilson's)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_interval(xs: list[float], rng: random.Random, n: int = 2000) -> tuple[float, float]:
    """The 95% interval of a mean, by the bootstrap."""
    if not xs:
        return (math.nan, math.nan)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def mcnemar(b: int, c: int) -> float:
    """The exact two-sided p of McNemar's test: b pairs only the first passed, c only the
    second."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _judgments(run: str) -> dict[str, dict]:
    from judge import JUDGMENTS

    out: dict[str, dict] = {}
    for path in (JUDGMENTS / run).glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def _excellence(run: str, arm: str) -> dict[str, float]:
    """Each reply's rubric score: the mean yes of the criteria answered."""
    from judge import JUDGMENTS

    folder = JUDGMENTS / (f"{run}-rubric" + ("-floor" if arm == "floor" else ""))
    out: dict[str, float] = {}
    for path in folder.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            yes = [a.get("yes") for a in rec["answers"].values() if a.get("yes") is not None]
            if yes:
                out[rec["id"]] = sum(1.0 for y in yes if y) / len(yes)
    return out


def _rate(values: list[bool]) -> str:
    k, n = sum(values), len(values)
    if not n:
        return "-"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}] n={n}"


def _mean(values: list[float], rng: random.Random) -> str:
    if not values:
        return "-"
    lo, hi = mean_interval(values, rng)
    return f"{sum(values) / len(values):.2f} [{lo:.2f}, {hi:.2f}] n={len(values)}"


def cmd_report(args: argparse.Namespace) -> None:
    manifest = load_manifest()
    meta = {m["id"]: m for part in manifest["moments"].values() for m in part}
    rng = random.Random(SEED)
    ai, floor = _judgments(args.run), _judgments(f"{args.run}-floor")
    excel_ai, excel_floor = _excellence(args.run, "ai"), _excellence(args.run, "floor")
    ids = [i for i in ai if i in meta]
    groups: dict[str, list[str]] = {"all": ids}
    for i in ids:
        groups[f"level {meta[i]['level']}"] = groups.get(f"level {meta[i]['level']}", []) + [i]
        groups[f"family {meta[i]['family']}"] = groups.get(f"family {meta[i]['family']}", []) + [i]
    lines = [
        f"# Bench report: {args.run}",
        "",
        "Rates with 95% intervals (Wilson for pass; the bootstrap for means).",
        "",
        "| group | AI pass | template pass | AI completeness | AI excellence "
        "| template excellence |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(groups, key=lambda g: (g != "all", g)):
        members = groups[name]
        ai_pass = [bool(ai[i]["scores"].get("pass")) for i in members]
        fl_pass = [bool(floor[i]["scores"].get("pass")) for i in members if i in floor]
        compl = [
            float(ai[i]["scores"]["completeness"])
            for i in members
            if isinstance(ai[i]["scores"].get("completeness"), float)
        ]
        ex_ai = [excel_ai[i] for i in members if i in excel_ai]
        ex_fl = [excel_floor[i] for i in members if i in excel_floor]
        lines.append(
            f"| {name} | {_rate(ai_pass)} | {_rate(fl_pass)} | {_mean(compl, rng)} |"
            f" {_mean(ex_ai, rng)} | {_mean(ex_fl, rng)} |"
        )
    if args.vs:
        other = _judgments(args.vs)
        excel_other = _excellence(args.vs, "ai")
        both = [i for i in ids if i in other]
        b = sum(
            1 for i in both if ai[i]["scores"].get("pass") and not other[i]["scores"].get("pass")
        )
        c = sum(
            1 for i in both if not ai[i]["scores"].get("pass") and other[i]["scores"].get("pass")
        )
        diffs = [excel_ai[i] - excel_other[i] for i in both if i in excel_ai and i in excel_other]
        lo, hi = mean_interval(diffs, rng) if diffs else (math.nan, math.nan)
        lines += [
            "",
            f"## Paired against {args.vs} ({len(both)} moments)",
            "",
            f"- pass: {b} moments pass only here, {c} only there; McNemar p = {mcnemar(b, c):.3f}",
            f"- excellence, this run minus that: {sum(diffs) / max(1, len(diffs)):+.3f}"
            f" [{lo:+.3f}, {hi:+.3f}] over {len(diffs)} moments",
        ]
    text = "\n".join(lines) + "\n"
    path = BENCH / "reports" / f"{args.run}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games")
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--force", action="store_true")
    d = sub.add_parser("draw")
    d.add_argument("--workers", type=int, default=16)
    d.add_argument("--force", action="store_true")
    f = sub.add_parser("refresh", help="the frozen moments as the code now meets them")
    f.add_argument("--workers", type=int, default=16)
    r = sub.add_parser("run")
    r.add_argument("--split", choices=["dev", "test"], required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--coach", default="openai:gpt-5.6-luna@medium")
    r.add_argument("--budget-usd", type=float, required=True)
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--run", required=True, help="the judged run (parity.py's --run)")
    p.add_argument("--vs", help="another judged run on the same moments, for a paired comparison")
    args = ap.parse_args()
    commands = {
        "games": cmd_games,
        "draw": cmd_draw,
        "refresh": cmd_refresh,
        "run": cmd_run,
        "report": cmd_report,
    }
    commands[args.cmd](args)


if __name__ == "__main__":
    main()
