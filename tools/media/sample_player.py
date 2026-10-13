"""A sample player for the README's screenshots and GIFs: some sixty amateur games by a made-up
player, "you_sample", reviewed by the app itself, and a couple of dozen puzzles played -- Your
games, Game Review, Stats and Puzzles filled as a player's own games fill them, with nothing of
Tai's and nothing from any site.

The games are Nova's (tools/bots: the dev-only model of human moves; the games it makes are
ours): the player at about 1300-1400 on Nova's (Lichess rapid) scale against opponents of
1150-1650 under made-up names, White and Black in turn. Each is played on a clock -- mostly
10+0, some 3+2, a few 15+10 -- each move's time drawn from a simple model of how amateurs
spend it (quick in the opening, a share of what is left after, fast in time trouble), and ends
as such games do: a mate, a resignation once a side has been lost by a wide margin for two of
its turns, a flag fall (a draw when the other side cannot mate), a draw by the rules, or one
agreed in a dead level ending. The candidates are reviewed by the app's own review
(`review_room.review_row`, the background reviewer's) and the set is chosen from them so the
galleries have something to show (brilliant and great moves, misses, blunders), with a range
of openings and endgames, then spread over the 120 days to the end date with the player's
rating moving by its results. The ratings in the PGN are on the scale the app reads a pasted
PGN on (chess.com's: `performance.chesscom` of the Nova rating), so Stats' peers are the
player's own (--elo-scale lichess writes Nova's instead).

The data folder is a fresh one (SIDECOACH_DATA_DIR) with only what the pages read, copied from
the checkout's data/ and tools/: the opening names, the puzzle sample (the store is built from
it at the first open, as an installed app's), Stockfish; and a settings file
(SIDECOACH_CONFIG) with the player's Lichess username, no chess.com one, and no AI model
(provider "none": $0). A panel started on it (port 8799; nothing posts Start, so the screen is
never captured; voice models off) takes the PGN through its own import, its background
reviewer reviews every game, and puzzles are played through its puzzle API as a player of
about 1400 would: those Stats offers, then each one's Next; mostly solved, some missed, two
with a hint (unrated). The panel is stopped at the end; summary.json says what the pages have.

    uv run --no-sync python tools/media/sample_player.py --out data/media
    uv run --no-sync python tools/media/sample_player.py --out data/media --stage panel

Stages, each reading the one before's file under --out: games (candidates.pgn), screen
(screen.jsonl), select (sample.pgn), data (data/, sidecoach.toml), panel (the import, the
reviews, the puzzles, summary.json). `--stage all` runs them in turn, each stage skipped when
its file is there already; `--fresh` makes the data folder again from nothing. `--repo` is the
checkout whose data/ (Nova's weights, the openings, the puzzles) and tools/stockfish are used,
for a run from a worktree.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import multiprocessing as mp
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "bots"))

import chess  # noqa: E402
import chess.engine  # noqa: E402
import chess.pgn  # noqa: E402

USER = "you_sample"
# Made-up handles, the same few met again as on a site; "sample_" says what they are.
OPPONENTS = (
    "sample_bishop",
    "sample_castle",
    "sample_tempo",
    "sample_outpost",
    "sample_gambit",
    "sample_fianchetto",
    "sample_zugzwang",
    "sample_endgame",
    "sample_knight",
    "sample_pawnstorm",
    "sample_rookie",
    "sample_queenside",
    "sample_kingside",
    "sample_opposition",
    "sample_skewer",
    "sample_pin",
)
# The time controls and how many of the set each is: rapid most, some blitz, a few longer.
CONTROLS = {"600+0": 42, "180+2": 12, "900+10": 6}
GAMES = sum(CONTROLS.values())
END = dt.date(2026, 10, 12)
DAYS = 120
MAX_PLIES = 300
# A side resigns once it has been this far behind (centipawns, its view, Stockfish at
# EVAL_DEPTH) at two of its turns running, at each such turn with this chance; the rest
# play on, some to mate.
RESIGN_CP = 700
RESIGN_P = 0.35
EVAL_DEPTH = 10
# A level ending: a draw is agreed with this chance a ply, past DRAW_PLY with this few pieces.
DRAW_CP = 40
DRAW_PIECES = 8
DRAW_PLY = 80
DRAW_P = 0.1
PORT = 8799
# The puzzle strength the player solves at (Lichess puzzle scale, a little over their rapid).
PUZZLE_STRENGTH = 1600
PUZZLES = 25
HINTED = 2


@dataclass(frozen=True)
class Job:
    seed: int
    user_white: bool
    user_rating: float
    opp_rating: float
    opp_classical: float
    opp_aggression: float
    control: str


# ---- games: Nova against Nova, on a clock ----------------------------------------------------

_policy: Any = None
_stockfish: Path | None = None


def _init_games(repo: str, stockfish: str) -> None:
    global _policy, _stockfish
    from nova import NovaPolicy

    _policy = NovaPolicy(Path(repo) / "data" / "models" / "nova" / "nova_v3b.onnx")
    _stockfish = Path(stockfish)


def think_time(rng: random.Random, remaining: int, increment: int, move_no: int) -> int:
    """Whole seconds a move takes: quick in the opening, a share of what is left after, and a
    second or two in time trouble -- where it can run out."""
    if remaining < 10:
        return rng.choices((0, 1, 2, 3), weights=(3, 4, 2, 1))[0]
    target = remaining / max(15, 40 - move_no // 2) + 0.8 * increment
    if move_no <= 6:
        target *= 0.25
    t = target * rng.lognormvariate(0.0, 0.75)
    return max(1, round(min(t, 0.3 * remaining)))


def _score_for(engine: chess.engine.SimpleEngine, board: chess.Board) -> int:
    """The side to move's score, centipawns (a mate as +-10000)."""
    info = engine.analyse(board, chess.engine.Limit(depth=EVAL_DEPTH))
    return info["score"].pov(board.turn).score(mate_score=10_000) or 0


def play(job: Job) -> str:
    """One game, as PGN text, between the player and the opponent of `job`."""
    from nova import Style, temperature_for

    assert _policy is not None and _stockfish is not None
    rng = random.Random(job.seed)
    user = Style(job.user_rating, 0.5, 0.55)
    opp = Style(job.opp_rating, job.opp_classical, job.opp_aggression)
    base, inc = (int(x) for x in job.control.split("+"))
    clock = {chess.WHITE: base, chess.BLACK: base}
    behind = {chess.WHITE: 0, chess.BLACK: 0}
    board = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    result, termination = "1/2-1/2", "Normal"
    with chess.engine.SimpleEngine.popen_uci(str(_stockfish)) as engine:
        engine.configure({"Threads": 1, "Hash": 16})
        while True:
            outcome = board.outcome(claim_draw=True)
            if outcome is not None:
                result = outcome.result()
                break
            if board.ply() >= MAX_PLIES:
                break  # a long shuffle, agreed drawn
            mover = board.turn
            if board.ply() >= 20:
                score = _score_for(engine, board)
                behind[mover] = behind[mover] + 1 if score <= -RESIGN_CP else 0
                if behind[mover] >= 2 and rng.random() < RESIGN_P:
                    result = "0-1" if mover == chess.WHITE else "1-0"
                    break
                if (
                    board.ply() >= DRAW_PLY
                    and abs(score) <= DRAW_CP
                    and len(board.piece_map()) <= DRAW_PIECES
                    and rng.random() < DRAW_P
                ):
                    break
            style = user if (mover == chess.WHITE) == job.user_white else opp
            probs = _policy.probabilities(
                [board], [style], temperature=temperature_for(style.rating)
            )[0]
            move = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
            spent = think_time(rng, clock[mover], inc, board.fullmove_number)
            if spent >= clock[mover]:
                # The flag falls while the move is thought over: lost, or drawn when the other
                # side has no mate left in its material.
                termination = "Time forfeit"
                if board.has_insufficient_material(not mover):
                    result = "1/2-1/2"
                else:
                    result = "0-1" if mover == chess.WHITE else "1-0"
                break
            clock[mover] = clock[mover] - spent + inc
            node = node.add_variation(move)
            node.set_clock(clock[mover])
            board.push(move)
    white, black = (USER, "opponent") if job.user_white else ("opponent", USER)
    game.headers.update(
        {
            "Event": "candidate",
            "White": white,
            "Black": black,
            "Result": result,
            "TimeControl": job.control,
            "Termination": termination,
            "Annotator": json.dumps(asdict(job)),
        }
    )
    return str(game)


def stage_games(args: argparse.Namespace, out: Path) -> None:
    rng = random.Random(args.seed)
    jobs = []
    controls = [c for c, n in CONTROLS.items() for _ in range(n)]
    for i in range(args.candidates):
        user_rating = rng.uniform(1300, 1400)
        opp = min(1650.0, max(1150.0, rng.gauss(user_rating, 120)))
        jobs.append(
            Job(
                seed=args.seed * 1_000_003 + i,
                user_white=i % 2 == 0,
                user_rating=round(user_rating),
                opp_rating=round(opp),
                opp_classical=round(rng.uniform(0.3, 0.7), 2),
                opp_aggression=round(rng.uniform(0.3, 0.7), 2),
                control=controls[i % len(controls)],
            )
        )
    started = time.time()
    texts: list[str] = []
    pool = mp.Pool(
        args.workers, initializer=_init_games, initargs=(str(args.repo), str(_stockfish_of(args)))
    )
    try:
        for i, pgn in enumerate(pool.imap(play, jobs), 1):
            texts.append(pgn)
            print(f"  game {i}/{len(jobs)}  {time.time() - started:.0f} s", flush=True)
    finally:
        pool.close()
        pool.join()
    (out / "candidates.pgn").write_text("\n\n".join(texts) + "\n", encoding="utf-8", newline="\n")


# ---- screen: each candidate reviewed by the app's own review ---------------------------------

_screen: dict[str, Any] = {}


def _init_screen(config: str, data: str, store: str) -> None:
    os.environ["SIDECOACH_CONFIG"] = config
    os.environ["SIDECOACH_DATA_DIR"] = data
    from sidecoach.config import load_settings
    from sidecoach.profile.store import Store

    _screen["settings"] = load_settings()
    _screen["store"] = Store(Path(store) / f"screen-{os.getpid()}.sqlite")


def screen(item: tuple[int, str]) -> dict[str, Any]:
    """A candidate reviewed as the background reviewer reviews it (standard strength), and what
    the choice needs of it: each side's labels, whether an ending was reached, the opening."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.analysis.phase import game_phases
    from sidecoach.profile import library
    from sidecoach.ui.review_room import review_row

    index, pgn = item
    store = _screen["store"]
    row = library.from_pgn(pgn, user_names=[USER])[0]
    store.add_library_game(row)
    done = review_row(
        _screen["settings"], row, store, STRENGTHS["standard"], workers=3, low_priority=True
    )
    assert done is not None
    reviewed, summary = done
    labels: dict[str, Counter[str]] = {"white": Counter(), "black": Counter()}
    for move in reviewed.moves:
        if move.assessment.label is not None:
            labels[move.side.value][move.assessment.label.value] += 1
    phases = game_phases(chess.Board(m.fen_before) for m in reviewed.moves)
    return {
        "index": index,
        "user_side": row.user_side,
        "plies": len(reviewed.moves),
        "endgame": "endgame" in phases,
        "opening": summary.opening,
        "labels": {side: dict(c) for side, c in labels.items()},
    }


def stage_screen(args: argparse.Namespace, out: Path) -> None:
    from sidecoach.profile import library

    games = library.games_in((out / "candidates.pgn").read_text(encoding="utf-8"))
    items = [(i, str(g)) for i, g in enumerate(games)]
    path = out / "screen.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["index"] for line in path.read_text(encoding="utf-8").splitlines()}
    items = [it for it in items if it[0] not in done]
    (out / "screen").mkdir(exist_ok=True)
    started = time.time()
    pool = mp.Pool(
        args.workers,
        initializer=_init_screen,
        initargs=(str(out / "sidecoach.toml"), str(out / "screen-data"), str(out / "screen")),
    )
    try:
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            for i, found in enumerate(pool.imap_unordered(screen, items), 1):
                fh.write(json.dumps(found) + "\n")
                fh.flush()
                mine = found["labels"][found["user_side"]]
                print(
                    f"  reviewed {i}/{len(items)} (#{found['index']})"
                    f"  {time.time() - started:.0f} s  brilliant {mine.get('brilliant', 0)}"
                    f" great {mine.get('great', 0)}",
                    flush=True,
                )
    finally:
        pool.close()
        pool.join()


# ---- select: the set, its dates, names and ratings -------------------------------------------


def _family(opening: str | None) -> str:
    return (opening or "?").split(":")[0].strip()


def stage_select(args: argparse.Namespace, out: Path) -> None:
    from sidecoach.profile import library, performance

    rng = random.Random(args.seed + 1)
    games = library.games_in((out / "candidates.pgn").read_text(encoding="utf-8"))
    found = {
        f["index"]: f
        for f in map(json.loads, (out / "screen.jsonl").read_text(encoding="utf-8").splitlines())
    }
    jobs = {i: Job(**json.loads(g.headers["Annotator"])) for i, g in enumerate(games)}

    def mine(i: int, label: str) -> int:
        f = found[i]
        return int(f["labels"][f["user_side"]].get(label, 0))

    # The quotas: each time control's games, half of them as White.
    quota = {(c, w): n // 2 + (n % 2 if w else 0) for c, n in CONTROLS.items() for w in (1, 0)}
    chosen: list[int] = []
    families: Counter[str] = Counter()

    def take(i: int) -> bool:
        key = (jobs[i].control, int(jobs[i].user_white))
        if i in chosen or quota.get(key, 0) <= 0:
            return False
        quota[key] -= 1
        chosen.append(i)
        families[_family(found[i]["opening"])] += 1
        return True

    pool = [i for i in found if i in jobs]
    rng.shuffle(pool)
    # The galleries first: the player's brilliant moves (all there are, up to a handful), great
    # moves; then the rest by what is new to the set: an opening, an ending.
    for i in sorted(pool, key=lambda i: -mine(i, "brilliant")):
        if mine(i, "brilliant") and sum(mine(j, "brilliant") > 0 for j in chosen) < 6:
            take(i)
    for i in sorted(pool, key=lambda i: -mine(i, "great")):
        if mine(i, "great") and sum(mine(j, "great") > 0 for j in chosen) < 12:
            take(i)
    while len(chosen) < GAMES:
        left = [
            i
            for i in pool
            if i not in chosen and quota[(jobs[i].control, int(jobs[i].user_white))] > 0
        ]
        if not left:
            break
        need_end = sum(found[j]["endgame"] for j in chosen) < GAMES // 3
        best = max(
            left,
            key=lambda i: (
                2.0 / (1 + families[_family(found[i]["opening"])])
                + (1.0 if need_end and found[i]["endgame"] else 0.0)
                + rng.random()
            ),
        )
        take(best)
    if len(chosen) < GAMES:
        raise SystemExit(f"only {len(chosen)} games fit the quotas: play more candidates")

    # The order played: White and Black in turn; the days, in sessions of one to three games.
    whites = [i for i in chosen if jobs[i].user_white]
    blacks = [i for i in chosen if not jobs[i].user_white]
    rng.shuffle(whites)
    rng.shuffle(blacks)
    order = [x for pair in zip(whites, blacks, strict=True) for x in pair]
    sessions: list[int] = []
    while sum(sessions) < GAMES:
        sessions.append(min(rng.choice((1, 1, 2, 2, 3)), GAMES - sum(sessions)))
    days = sorted(rng.sample(range(DAYS - 1), len(sessions) - 1)) + [DAYS - 1]
    start = END - dt.timedelta(days=DAYS - 1)
    rating = 1320.0  # the player's, Nova's scale, moved by each result
    names = list(OPPONENTS)
    texts: list[str] = []
    k = 0
    for size, day in zip(sessions, days, strict=True):
        when = dt.datetime.combine(start + dt.timedelta(days=day), dt.time(17)) + dt.timedelta(
            minutes=rng.randrange(0, 240)
        )
        for _ in range(size):
            i = order[k]
            k += 1
            game, job = games[i], jobs[i]
            opp_rating = job.opp_rating
            user_side = "white" if job.user_white else "black"
            result = game.headers["Result"]
            termination = game.headers.get("Termination", "Normal")
            score = 0.5 if result == "1/2-1/2" else float((result == "1-0") == job.user_white)
            elo = performance.chesscom if args.elo_scale == "chesscom" else (lambda r: round(r))
            user_elo, opp_elo = elo(rating), elo(opp_rating)
            opponent = rng.choice(names[: 6 + k // 6])  # the circle widens as the player plays
            white, black = (USER, opponent) if job.user_white else (opponent, USER)
            minutes = {"600+0": "rapid", "180+2": "blitz", "900+10": "rapid"}[job.control]
            game.headers.clear()
            game.headers.update(
                {
                    "Event": f"Rated {minutes} game",
                    "Site": "Sidecoach sample",
                    "Date": when.strftime("%Y.%m.%d"),
                    "White": white,
                    "Black": black,
                    "Result": result,
                    "UTCDate": when.strftime("%Y.%m.%d"),
                    "UTCTime": when.strftime("%H:%M:%S"),
                    "WhiteElo": str(user_elo if user_side == "white" else opp_elo),
                    "BlackElo": str(opp_elo if user_side == "white" else user_elo),
                    "Variant": "Standard",
                    "TimeControl": job.control,
                    "Termination": termination,
                }
            )
            # The other headers python-chess keeps (Round, the Seven Tag Roster's) as a site
            # leaves them.
            texts.append(str(game))
            expected = 1 / (1 + 10 ** ((opp_rating - rating) / 400))
            rating += 24 * (score - expected)
            # The next game a few minutes after this one's clocks were spent.
            base, inc = (int(x) for x in job.control.split("+"))
            plies = sum(1 for _ in game.mainline_moves())
            when += dt.timedelta(seconds=1.4 * base + plies * inc / 2 + rng.randrange(60, 600))
    (out / "sample.pgn").write_text("\n\n".join(texts) + "\n", encoding="utf-8", newline="\n")
    print(f"  {len(texts)} games, {len(sessions)} sessions, the player ends at {rating:.0f} (Nova)")


# ---- data: the folder and the settings ---------------------------------------------------------


def _stockfish_of(args: argparse.Namespace) -> Path:
    found = sorted((Path(args.repo) / "tools" / "stockfish").glob("stockfish*.exe")) or sorted(
        p for p in (Path(args.repo) / "tools" / "stockfish").glob("stockfish-*") if p.is_file()
    )
    if not found:
        raise SystemExit(f"no Stockfish under {args.repo}/tools/stockfish")
    return found[0]


def _copy(src: Path, dst: Path) -> None:
    """A copy, checked by its size; the source is never touched."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if dst.stat().st_size != src.stat().st_size:
        raise SystemExit(f"the copy of {src} is not whole")


def write_config(out: Path) -> None:
    lines = [
        "# The sample player's settings (tools/media/sample_player.py): no AI model, a",
        "# Lichess username and no chess.com one.",
        "",
        "[llm]",
        'provider = "none"',
        "",
        "[profile]",
        f'lichess_user = "{USER}"',
        'chesscom_user = ""',
        "",
    ]
    (out / "sidecoach.toml").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def make_data(args: argparse.Namespace, data: Path) -> None:
    """A data folder with what the pages read: the opening names, the puzzle sample, Stockfish."""
    repo = Path(args.repo)
    # The tables and their compiled cache (made once; parallel screens raced to write it).
    for table in sorted((repo / "data" / "openings").glob("*.tsv")):
        _copy(table, data / "openings" / table.name)
    cache = repo / "data" / "openings" / "openings.cache.json"
    if cache.exists():
        _copy(cache, data / "openings" / cache.name)
    _copy(repo / "data" / "puzzles" / "sample.csv", data / "puzzles" / "sample.csv")
    exe = _stockfish_of(args)
    _copy(exe, data / "tools" / "stockfish" / exe.name)


def stage_data(args: argparse.Namespace, out: Path) -> None:
    data = out / "data"
    if data.exists() and args.fresh:
        # Only ever this run's own folder, under --out.
        assert data.resolve().parent == out.resolve()
        shutil.rmtree(data)
    if (data / "profile.sqlite").exists():
        raise SystemExit(f"{data} has a profile already: --fresh to make it again")
    write_config(out)
    make_data(args, data)
    print(f"  data folder {data}, settings {out / 'sidecoach.toml'}")


# ---- panel: the import, the reviews, the puzzles -----------------------------------------------


class Panel:
    """A panel of our own on the sample data: started, asked, stopped (its pid only)."""

    def __init__(self, out: Path, port: int) -> None:
        import httpx

        self.out = out
        self.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env.update(
            SIDECOACH_DATA_DIR=str(out / "data"),
            SIDECOACH_CONFIG=str(out / "sidecoach.toml"),
            SIDECOACH_VOICE_MODELS="0",
        )
        env.pop("SIDECOACH_PANEL_TOKEN", None)
        cmd = [sys.executable, "-m", "sidecoach", "panel", "--port", str(port)]
        cmd += ["--no-record", "--no-persist", "--listen", "off", "--no-browser"]
        self.log = (out / "panel.log").open("a", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=out, env=env, stdout=self.log, stderr=self.log)
        print(f"  panel pid {self.proc.pid} on {self.base}", flush=True)
        self.http = httpx.Client(base_url=self.base, timeout=120.0)
        deadline = time.time() + 300
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"the panel exited ({self.proc.returncode}); see panel.log")
            with contextlib.suppress(Exception):
                if self.http.get("/api/health").json().get("ok"):
                    return
            if time.time() > deadline:
                self.stop()
                raise SystemExit("the panel did not answer in 300 s")
            time.sleep(1.0)

    def get(self, path: str, **params: Any) -> Any:
        return self.http.get(path, params=params).json()

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        return self.http.post(path, json=body or {}).json()

    def stop(self) -> None:
        with contextlib.suppress(Exception):
            self.post("/api/shutdown")
        try:
            self.proc.wait(90)
        except subprocess.TimeoutExpired:
            self.proc.terminate()  # ours, by its handle
            self.proc.wait(30)
        self.http.close()
        self.log.close()
        print(f"  panel {self.proc.pid} stopped ({self.proc.returncode})", flush=True)


def wait_reviews(panel: Panel, total: int) -> dict[str, Any]:
    started = time.time()
    last = ""
    while True:
        reviews = panel.get("/api/jobs").get("reviews") or {}
        line = f"{reviews.get('done')}/{reviews.get('total')} {reviews.get('status')}"
        if line != last:
            print(f"  reviews {line}  {time.time() - started:.0f} s", flush=True)
            last = line
        if reviews.get("status") == "done" and reviews.get("total", 0) >= total:
            return reviews
        if reviews.get("status") == "paused":
            panel.post("/api/jobs/reviews", {"paused": False})
        time.sleep(10.0)


def _solution(data: Path, puzzle_id: str) -> tuple[list[str], str, int]:
    db = sqlite3.connect(f"file:{data / 'puzzles.sqlite'}?mode=ro", uri=True)
    try:
        fen, moves, rating = db.execute(
            "SELECT fen, moves, rating FROM puzzles WHERE id = ?", (puzzle_id,)
        ).fetchone()
    finally:
        db.close()
    return moves.split(), fen, int(rating)


def _wrong_move(board: chess.Board, right: str, rng: random.Random) -> str:
    """A plausible wrong move: a legal one that is not the line's and does not mate."""
    moves = [m for m in board.legal_moves if m.uci() != right]
    rng.shuffle(moves)
    for move in moves:
        board.push(move)
        mates = board.is_checkmate()
        board.pop()
        if not mates:
            return move.uci()
    return moves[0].uci()


def play_puzzle(
    panel: Panel, data: Path, puzzle_id: str, mode: str, rng: random.Random
) -> dict[str, Any] | None:
    """One puzzle on the panel's puzzle API: solved, missed (a wrong move, then given up or put
    right), or solved after a Hint."""
    opened = panel.post(f"/api/puzzle/{puzzle_id}/open")
    if not opened.get("ok"):
        return None
    moves, fen, rating = _solution(data, puzzle_id)
    board = chess.Board(fen)
    board.push_uci(moves[0])
    base = f"/api/puzzle/{puzzle_id}"
    if mode == "hint":
        panel.post(f"{base}/ask", {"text": "hint"})
        time.sleep(2.0)  # the room takes its commands on its own tick
    solver = list(range(1, len(moves), 2))
    miss_at = rng.choice([0, 0, 0, len(solver) - 1]) if mode == "miss" else -1
    for k, idx in enumerate(solver):
        if k == miss_at:
            panel.post(f"{base}/move", {"uci": _wrong_move(board, moves[idx], rng)})
            panel.post(f"{base}/retract")
            if rng.random() < 0.5:
                panel.post(f"{base}/giveup")
                break
        panel.post(f"{base}/move", {"uci": moves[idx]})
        board.push_uci(moves[idx])
        if idx + 1 < len(moves):
            board.push_uci(moves[idx + 1])
    state: dict[str, Any] = {}
    for _ in range(40):
        state = panel.get(f"{base}/poll").get("state") or {}
        if state.get("result"):
            break
        time.sleep(0.25)
    panel.post(f"{base}/close")
    return {
        "id": puzzle_id,
        "rating": rating,
        "mode": mode,
        "motif": state.get("motif"),
        "status": state.get("status"),
        "result": state.get("result"),
        "after": opened.get("after") or [],
    }


def play_puzzles(panel: Panel, data: Path, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 2)
    stats = panel.get("/api/stats").get("stats") or {}
    queue = [z["id"] for w in stats.get("work", []) for z in w.get("puzzles", [])]
    hinted = set(rng.sample(range(PUZZLES), HINTED))
    played: list[dict[str, Any]] = []
    tried: set[str] = set()
    while len(played) < PUZZLES and queue:
        puzzle_id = queue.pop(0)
        if puzzle_id in tried:
            continue
        tried.add(puzzle_id)
        moves, _, rating = _solution(data, puzzle_id)
        n = len(played)
        solve = 1 / (1 + 10 ** ((rating - PUZZLE_STRENGTH) / 400))
        mode = "hint" if n in hinted else "solve" if rng.random() < solve else "miss"
        done = play_puzzle(panel, data, puzzle_id, mode, rng)
        if done is None:
            continue
        played.append(done)
        r = done["result"] or {}
        print(
            f"  puzzle {len(played)}/{PUZZLES} {puzzle_id} ({rating}, {done['motif']}) {mode}:"
            f" {'solved' if r.get('solved') else 'missed'}, level {r.get('level')}"
            f"{' (unrated)' if r.get('unrated') else ''}",
            flush=True,
        )
        # The puzzles Stats offers first (themed), then each one's Next.
        queue += [p for p in done["after"] if p not in tried]
    return played


def _filled(section: Any) -> tuple[int, int]:
    """A section's figures (dicts with a `value`): how many have one, of how many."""
    from sidecoach.profile.stats import walk

    figures = [f for _, f in walk(section)]
    return sum(1 for f in figures if f.get("value") is not None), len(figures)


def summary(panel: Panel, puzzles: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"filters": {}}
    for name, params in {
        "all": {},
        "30 days": {"days": 30},
        "90 days": {"days": 90},
        "rapid": {"time_class": "rapid"},
        "blitz": {"time_class": "blitz"},
    }.items():
        stats = panel.get("/api/stats", **params)["stats"]
        sections = {
            k: _filled(stats[k])
            for k in ("overview", "openings", "tactics", "strategy", "endgames", "time")
        }
        out["filters"][name] = {
            "sections": sections,
            "results": stats["results"],
            "lost_on_time": stats["time"].get("lost_on_time"),
            "galleries": stats["galleries"],
            "ratings": {k: v and v.get("rating") for k, v in stats["ratings"].items()},
            "peers": stats["peers"],
            "trends": {k: len(v) for k, v in stats["trends"].items()},
            "shapes": len(stats["shapes"]),
            "work": [w.get("motif") or w.get("area") for w in stats["work"]],
            "unsided": stats["unsided"],
        }
    games = panel.get("/api/games")
    out["games"] = len(games.get("games", []))
    out["backlog"] = games.get("backlog")
    out["puzzles"] = puzzles
    return out


def stage_panel(args: argparse.Namespace, out: Path) -> None:
    from sidecoach.profile import library

    text = (out / "sample.pgn").read_text(encoding="utf-8")
    total = len(library.games_in(text))
    panel = Panel(out, args.port)
    try:
        added = panel.post("/api/games/import", {"source": "pgn", "text": text})
        print(f"  import: {added.get('added')} added, {added.get('already')} there already")
        if not added.get("ok"):
            raise SystemExit(f"the import failed: {added}")
        wait_reviews(panel, total)
        puzzles = play_puzzles(panel, out / "data", args.seed) if args.puzzles else []
        found = summary(panel, puzzles)
    finally:
        panel.stop()
    db = sqlite3.connect(f"file:{out / 'data' / 'profile.sqlite'}?mode=ro", uri=True)
    try:
        perf = db.execute("SELECT value FROM settings WHERE key = 'puzzle_perf'").fetchone()
    finally:
        db.close()
    found["puzzle_perf"] = json.loads(perf[0]) if perf else None
    (out / "summary.json").write_text(json.dumps(found, indent=1), encoding="utf-8", newline="\n")
    print(f"  summary.json written; puzzle rating {found['puzzle_perf']}")


# ---- the run -----------------------------------------------------------------------------------

STAGES = ("games", "screen", "select", "data", "panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "media")
    parser.add_argument(
        "--repo", type=Path, default=ROOT, help="the checkout with data/ and tools/"
    )
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--candidates", type=int, default=160)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--elo-scale", choices=("chesscom", "lichess"), default="chesscom")
    parser.add_argument("--no-puzzles", dest="puzzles", action="store_false")
    parser.add_argument("--fresh", action="store_true", help="make the data folder again")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)
    files = {
        "games": "candidates.pgn",
        "select": "sample.pgn",
        "panel": "summary.json",
    }
    for stage in stages:
        if args.stage == "all" and stage in files and (out / files[stage]).exists():
            print(f"{stage}: {files[stage]} is there, skipped")
            continue
        if args.stage == "all" and stage == "data" and (out / "data" / "profile.sqlite").exists():
            print("data: the data folder is there, skipped")
            continue
        print(f"{stage}:", flush=True)
        if stage == "screen":
            # The screen reads the openings and Stockfish as the panel will: a folder of its
            # own beside the sample's, never the sample's profile.
            write_config(out)
            make_data(args, out / "screen-data")
        {
            "games": stage_games,
            "screen": stage_screen,
            "select": stage_select,
            "data": stage_data,
            "panel": stage_panel,
        }[stage](args, out)


if __name__ == "__main__":
    main()
