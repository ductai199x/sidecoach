"""The live analyser on a dumped session, simulated: every search's real result and real time,
measured once (one thread: the same answer every time, so it is kept), and the analyser's own
scheduling -- what it picks, what it stops, what it promotes, when it judges -- run on a virtual
clock. A 38-minute session simulates in seconds; the searches it needs are computed once, in
parallel on the machine's cores, and cached under data/engine-cache/.

    uv run python tools/engine/simulate.py <dump folder> [--workers 3] [--look 2 3] [--json out]

Prints, per setting: Stockfish's processor time and average cores, positions searched more than
once, searches stopped and the time they had taken, look-ahead positions reached, and the waits
after a move (classification, first analysis, full analysis). (Tai, 2026-10-04: "why can't this
be a simulation that can finish in 1 minute or less".)"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.analyser import Analyser  # noqa: E402
from sidecoach.analysis.annotator import (  # noqa: E402
    EARLY_DEPTH,
    GameAnnotator,
    Measured,
    cache_key,
)
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.events import Analysis, Line, MoveAssessment  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
DEPTH, MULTIPV, PV_PLIES = 18, 6, 12
STOCKFISH = Path(os.path.expandvars(r"%LOCALAPPDATA%\Sidecoach\data\tools\stockfish"))


# ---- the session --------------------------------------------------------------------------


@dataclass
class Step:
    t: float
    kind: str  # start, resync, move, takeback
    board: chess.Board | None = None  # start/resync: the position; move: the board before
    move: chess.Move | None = None
    ply: int = 0
    by_user: bool = False
    plies: int = 0
    game_id: str = ""
    user: bool | None = None  # the player's colour as known at this step


def session(dump: Path) -> list[Step]:
    """The dump's game, as the engine glue (`cli._Assessing`) hands it to the analyser."""
    steps: list[Step] = []
    board: chess.Board | None = None
    game_id, ply = "", 0
    user: bool | None = None
    for line in (dump / "debug.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d["chan"] != "game":
            continue
        t, text, data = d["t"], d["text"], d.get("data") or {}
        if "start_fen" in data:
            game_id, board, ply = data["game_id"], chess.Board(data["start_fen"]), 0
            user = chess.BLACK if text.endswith("you play black") else chess.WHITE
            steps.append(Step(t, "start", board.copy(), game_id=game_id, user=user))
        elif text.startswith("resync:") and "fen" in data:
            board, ply = chess.Board(data["fen"]), 0
            steps.append(Step(t, "resync", board.copy(), game_id=game_id, user=user))
        elif text.startswith("takeback") and board is not None:
            plies = int(re.match(r"takeback (\d+)", text)[1])
            ply = max(0, ply - plies)
            for _ in range(min(plies, len(board.move_stack))):
                board.pop()
            if board.fen() != data.get("fen", board.fen()):
                board = chess.Board(data["fen"])
            steps.append(
                Step(t, "takeback", board.copy(), plies=plies, ply=ply, game_id=game_id, user=user)
            )
        elif "uci" in data and board is not None:
            move = chess.Move.from_uci(data["uci"])
            if move not in board.legal_moves and ply == 0:
                board.turn = not board.turn  # a position adopted with the side to move pending
                board.ep_square = None
            if move not in board.legal_moves:
                continue
            ply = data["ply"]
            by_user = bool(data["by_user"])
            user = board.turn if by_user else not board.turn
            steps.append(
                Step(t, "move", board.copy(), move, ply, by_user, game_id=game_id, user=user)
            )
            board.push(move)
    return steps


# ---- the searches, measured once ----------------------------------------------------------


def _root_and_moves(board: chess.Board) -> tuple[str, list[str]]:
    root = board.root()
    return root.fen(), [m.uci() for m in board.move_stack]


def _board(root: str, moves: list[str]) -> chess.Board:
    board = chess.Board(root)
    for uci in moves:
        board.push_uci(uci)
    return board


_ENGINE: dict[str, Any] = {}


def _search(task: tuple[str, list[str]]) -> tuple[str, dict]:
    """One position's search, in a worker process: each depth's lines with the time it came,
    the whole time, and its side searches with theirs."""
    if "annotator" not in _ENGINE:
        import atexit

        exe = next(STOCKFISH.glob("*.exe"))
        engine = Engine(path=exe, threads=1, hash_mb=256, multipv=MULTIPV, pv_plies=PV_PLIES)
        engine.open()
        # An engine left open keeps its process alive past the end of the work: closed at exit.
        atexit.register(engine.close)
        _ENGINE["annotator"] = GameAnnotator(engine=engine, depth=DEPTH, multipv=MULTIPV)
    annotator: GameAnnotator = _ENGINE["annotator"]
    board = _board(*task)
    depths: list[tuple[int, float, list[dict]]] = []
    begun = time.perf_counter()

    def on_depth(lines: list[Line]) -> None:
        if lines and lines[0].depth >= EARLY_DEPTH:
            depths.append(
                (lines[0].depth, time.perf_counter() - begun, [ln.model_dump() for ln in lines])
            )

    measured = annotator.search(board, on_depth=on_depth)
    main_s = time.perf_counter() - begun
    begun = time.perf_counter()
    whole = annotator.with_side(board, measured)
    side_s = time.perf_counter() - begun
    return cache_key(board), {
        "lines": [ln.model_dump() for ln in measured.lines],
        "depths": depths,
        "main_s": main_s,
        "side_s": side_s,
        "threats": whole.threats,
        "traps": [ln.model_dump() for ln in whole.traps],
        "defences": [ln.model_dump() for ln in whole.defences],
        "threatens": [ln.model_dump() for ln in whole.threatens],
    }


class SearchCache(dict):
    """The searches made, by position: read whole at the start, and each new one added to an
    SQLite file as a row of its own (a JSON file of them all, rewritten every 200 searches,
    was 147 MB a write)."""

    def __init__(self, path: Path) -> None:
        import sqlite3

        super().__init__()
        CACHE.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS search (key TEXT PRIMARY KEY, found TEXT)")
        for key, found in self.db.execute("SELECT key, found FROM search"):
            super().__setitem__(key, json.loads(found))

    def __setitem__(self, key: str, found: dict) -> None:
        super().__setitem__(key, found)
        self.db.execute("INSERT OR REPLACE INTO search VALUES (?, ?)", (key, json.dumps(found)))

    def commit(self) -> None:
        self.db.commit()


def _cache_path(dump: Path) -> Path:
    """One cache for every session: a position is the same search wherever it comes from."""
    tag = hashlib.sha1(f"{DEPTH}:{MULTIPV}:{PV_PLIES}".encode()).hexdigest()[:10]
    return CACHE / f"searches-{tag}.sqlite"


_OPEN: dict[Path, SearchCache] = {}


def load_cache(dump: Path) -> SearchCache:
    path = _cache_path(dump)
    if path in _OPEN:
        return _OPEN[path]
    cache = _OPEN[path] = SearchCache(path)
    if not cache:
        # The JSON files the searches were kept in before: taken in once.
        for old in sorted(CACHE.glob("*.json")):
            if old.name.startswith(("searches-", "20261004-150208-")):
                for key, found in json.loads(old.read_text(encoding="utf-8")).items():
                    if key not in cache:
                        cache[key] = found
        cache.commit()
    return cache


def save_cache(dump: Path, cache: dict[str, dict]) -> None:
    if isinstance(cache, SearchCache):
        cache.commit()


def game_boards(steps: list[Step]) -> dict[str, chess.Board]:
    """Every position the sessions' games stand on."""
    out: dict[str, chess.Board] = {}
    for step in steps:
        for board in [step.board] if step.kind != "move" else [step.board, _after(step)]:
            if board is not None and not board.is_game_over():
                out[cache_key(board)] = board
    return out


def search_missing(
    boards: dict[str, chess.Board], cache: dict[str, dict], procs: int, what: str
) -> None:
    """The searches `boards` need that the cache has not, made in parallel and kept."""
    todo = {k: b for k, b in boards.items() if k not in cache}
    if not todo:
        return
    pool = ProcessPoolExecutor(procs)
    try:
        futures = [pool.submit(_search, _root_and_moves(b)) for b in todo.values()]
        for n, fut in enumerate(
            tqdm(
                as_completed(futures), total=len(futures), file=sys.stderr, desc=what, mininterval=2
            ),
            start=1,
        ):
            key, found = fut.result()
            cache[key] = found
            if n % 200 == 0:
                save_cache(Path("all"), cache)  # kept as it goes: a stop loses little
        save_cache(Path("all"), cache)
    finally:
        # The workers keep their Stockfish open and never exit on their own: a pool waiting on
        # them waits for good. Ended here, their engines with them.
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:
            process.terminate()


def _after(step: Step) -> chess.Board:
    assert step.board is not None and step.move is not None
    after = step.board.copy()
    after.push(step.move)
    return after


# ---- the analyser on a virtual clock -------------------------------------------------------


class _Stub:
    """A worker's engine as the analyser sees it: `interrupt` marks the worker's job stopped."""

    def __init__(self, index: int, stops: set[int]) -> None:
        self.index, self.stops = index, stops

    def interrupt(self) -> None:
        self.stops.add(self.index)

    def close(self) -> None:
        return None


@dataclass
class _Slot:
    job: Any = None
    start: float = 0.0
    end: float = 0.0
    reports: list[tuple[float, list[Line]]] = field(default_factory=list)


def _lines(raw: list[dict]) -> list[Line]:
    return [Line(**ln) for ln in raw]


def simulate(
    steps: list[Step],
    cache: dict[str, dict],
    *,
    workers: int,
    look: int,
    trace: tuple[float, float] | None = None,
    scale: float = 1.0,
    raw: bool = False,
    reach: float | None = None,
) -> dict:
    """`trace`: a stretch of the session (from, to) whose every step, search and word is
    printed. `scale`: every search's time multiplied by it -- the precompute times its searches
    side by side on every core, slower than one alone (`--calibrate`)."""
    annotator = GameAnnotator(engine=None, depth=DEPTH, multipv=MULTIPV, thresholds=Thresholds())  # type: ignore[arg-type]
    stops: set[int] = set()
    analyser = Analyser(
        annotator, engines=[_Stub(i, stops) for i in range(workers)], look_ahead=look
    )  # type: ignore[list-item]
    if reach is not None:
        analyser.reach = reach
    now = 0.0
    said: list[tuple[float, Any]] = []

    def show(*what: object) -> None:
        if trace is not None and trace[0] <= now <= trace[1]:
            print(f"{now:9.2f}", *what)

    def put(event: Any) -> None:
        said.append((now, event))
        show(
            "  said",
            type(event).__name__,
            getattr(event, "ply", None),
            "provisional" if getattr(event, "provisional", False) else "",
        )

    analyser._put = put  # type: ignore[method-assign]
    slots = [_Slot() for _ in range(workers)]
    busy_s = 0.0
    spent: defaultdict[str, float] = defaultdict(float)  # by (kind, key): the time it took
    ever_game: set[str] = set()  # every position that was the game's at some ply
    stopped_s = 0.0
    searched: Counter[str] = Counter()
    missing = 0
    missing_boards: dict[str, chess.Board] = {}  # searches the cache lacked: made, then run again

    def assign() -> None:
        nonlocal missing
        for i, slot in enumerate(slots):
            if slot.job is not None:
                continue
            with analyser._cond:
                job = analyser._pick(i)
                if job is None or not analyser._wanted(job):
                    continue
                found = cache.get(job.key)
                if found is None:
                    missing += 1
                    missing_boards[job.key] = analyser._nodes[job.key].board.copy()
                    analyser._nodes[job.key].failed = True
                    continue
                analyser._claimed[i] = analyser._running[i] = job
            if job.kind == "main":
                searched[job.key] += 1
                slot.reports = [(now + s * scale, _lines(ln)) for _, s, ln in found["depths"]]
                slot.end = now + found["main_s"] * scale
            else:
                slot.reports = []
                slot.end = now + found["side_s"] * scale
            slot.job, slot.start = job, now
            role = analyser._ply_of.get(job.key, "ahead" if job.key in analyser._ahead else "?")
            show(f"  worker {i} starts {job.kind} of {role} ({slot.end - now:.2f}s)")

    def finish(i: int) -> None:
        nonlocal busy_s
        slot = slots[i]
        job, node = slot.job, analyser._nodes.get(slot.job.key)
        busy_s += slot.end - slot.start
        spent[f"{job.kind} {job.key}"] += slot.end - slot.start
        slot.job = None
        with analyser._cond:
            analyser._running.pop(i, None)
            analyser._claimed.pop(i, None)
            if node is None or job.era != analyser._era:
                return
            found = cache[job.key]
            if job.kind == "main":
                measured = annotator.measured_lines(node.board, _lines(found["lines"]))
                analyser._searched(job.key, measured)
            else:
                assert node.measured is not None
                whole = Measured(
                    **{
                        **node.measured.__dict__,
                        "threats": found["threats"],
                        "traps": _lines(found["traps"]),
                        "defences": _lines(found["defences"]),
                        "threatens": _lines(found["threatens"]),
                    }
                )
                analyser._sided(job.key, whole)

    def stop_flagged() -> None:
        nonlocal busy_s, stopped_s
        for i in list(stops):
            stops.discard(i)
            slot = slots[i]
            if slot.job is None:
                continue
            busy_s += now - slot.start
            spent[f"stopped {slot.job.key}"] += now - slot.start
            stopped_s += now - slot.start
            slot.job = None
            with analyser._cond:
                analyser._running.pop(i, None)
                analyser._claimed.pop(i, None)

    queue = list(steps)
    played: list[tuple[float, int, bool]] = []  # each move: when, its ply, whose
    while True:
        # The next thing that happens: a step of the game, a depth reported, a search ending.
        times = [queue[0].t] if queue else []
        for slot in slots:
            if slot.job is not None:
                times.append(slot.reports[0][0] if slot.reports else slot.end)
        if not times:
            break
        now = min(times)
        if queue and queue[0].t == now:
            step = queue.pop(0)
            show("STEP", step.kind, step.ply, step.move.uci() if step.move else "")
            if step.kind in ("start", "resync"):
                analyser.reset()
                user = step.user if step.kind == "start" else None  # a resync keeps the game's
                analyser.submit_position(step.game_id, step.board, 0, user=user)
            elif step.kind == "takeback":
                analyser.withdraw(step.game_id, step.ply)
                if step.ply == 0:
                    analyser.submit_position(step.game_id, step.board, 0)
            else:
                analyser.submit_move(
                    step.game_id, step.board, step.move, step.ply, by_user=step.by_user
                )
                played.append((now, step.ply, step.by_user))
        else:
            for i, slot in enumerate(slots):
                if slot.job is None:
                    continue
                if slot.reports and slot.reports[0][0] == now:
                    _, lines = slot.reports.pop(0)
                    analyser._depth(slot.job.key, slot.job.era, lines)
                    break
                if not slot.reports and slot.end == now:
                    finish(i)
                    break
        stop_flagged()
        assign()
        stop_flagged()
        ever_game.update(analyser._ply_of)

    # Where the time went: positions of the game, looked at ahead and reached or not, their
    # side searches, and searches stopped.
    split: defaultdict[str, float] = defaultdict(float)
    for what, secs in spent.items():
        kind, key = what.split(" ", 1)
        if kind == "stopped":
            split["stopped"] += secs
        elif kind == "side":
            split["side searches"] += secs
        elif key in ever_game:
            split["game positions"] += secs
        else:
            split["ahead, never reached"] += secs

    # From the first step to the last search's end: searches go on after the last move, and an
    # average over the moves' span alone ran past the number of workers (browsing: 3.4 of 2).
    span = max(now, steps[-1].t) - steps[0].t if steps else 1.0
    # The waits after each move, as the page sees them.
    waits: dict[str, list[float]] = defaultdict(list)
    slowest: list[tuple[float, float, int, str]] = []
    # A move's word counts until the game leaves it: a new game or re-read, or the same ply
    # played again after a takeback.
    resets = [s.t for s in steps if s.kind in ("start", "resync", "takeback")]
    for n, (t, ply, by_user) in enumerate(played):
        nxt = played[n + 1][0] if n + 1 < len(played) else float("inf")
        gone = min(
            [r for r in resets if r > t]
            + [pt for pt, pp, _ in played[n + 1 :] if pp <= ply]
            + [float("inf")]
        )
        who = "you" if by_user else "opponent"
        later = [(ts, e) for ts, e in said if t <= ts < gone and getattr(e, "ply", None) == ply]
        verdict = next((ts for ts, e in later if isinstance(e, MoveAssessment)), None)
        first = next((ts for ts, e in later if isinstance(e, Analysis)), None)
        full = next((ts for ts, e in later if isinstance(e, Analysis) and not e.provisional), None)
        if verdict is not None:
            waits[f"classification ({who})"].append(verdict - t)
            slowest.append((round(verdict - t, 2), round(t, 2), ply, who))
        else:
            slowest.append((float("inf"), round(t, 2), ply, who))
        if first is not None and first < nxt:
            waits[f"first analysis ({who})"].append(first - t)
        if full is not None and full < nxt:
            waits[f"full analysis ({who})"].append(full - t)
    return {
        "workers": workers,
        "look": look,
        "span_s": span,
        "stockfish_core_s": round(busy_s, 1),
        "mean_cores": round(busy_s / span, 2),
        "positions": len(searched),
        "split_s": {k: round(v, 1) for k, v in sorted(split.items())},
        "searched_twice": sum(1 for n in searched.values() if n > 1),
        "stopped": analyser.stopped,
        "stopped_s": round(stopped_s, 1),
        "looked": analyser.looked,
        "ahead_hits": analyser.ahead_hits,
        "missing": missing,
        "missing_boards": missing_boards,
        "unjudged": sum(1 for s in slowest if s[0] == float("inf")),
        "slowest": sorted((s for s in slowest if s[0] != float("inf")), reverse=True)[:6],
        "waits": {k: _summary(v) for k, v in sorted(waits.items())},
        # Every wait, for pooling over sessions (`raw`).
        **({"raw_waits": {k: list(v) for k, v in waits.items()}} if raw else {}),
    }


def _summary(values: list[float]) -> dict[str, float]:
    v = sorted(values)
    return {
        "median": round(statistics.median(v), 3),
        "p90": round(v[int(len(v) * 0.9)], 3),
        "max": round(v[-1], 3),
        "n": len(v),
    }


# ---- the sessions: a dump, the library's games at their clocks, browsing ------------------


def pgn_session(
    pgn: str, user: str, *, speed: float = 1.0, game_id: str = "g"
) -> list[Step] | None:
    """A game as it was played: each move at the time its clock says it was made (the time the
    mover spent, from the clocks and the increment), `speed` times faster; the player's side by
    their name. None for a game without clocks or without the player."""
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    white = game.headers.get("White", "").lower() == user.lower()
    black = game.headers.get("Black", "").lower() == user.lower()
    if white == black:
        return None
    tc = game.headers.get("TimeControl", "")
    inc = float(tc.split("+")[1]) if "+" in tc else 0.0
    board = game.board()
    colour = chess.WHITE if white else chess.BLACK
    steps = [Step(0.0, "start", board.copy(), game_id=game_id, user=colour)]
    t = 0.0
    last = {chess.WHITE: None, chess.BLACK: None}
    for ply, node in enumerate(game.mainline(), start=1):
        clock = node.clock()
        if clock is None:
            return None
        mover = board.turn
        spent = 0.0 if last[mover] is None else max(0.0, last[mover] - clock + inc)
        last[mover] = clock
        t += max(0.05, spent) / speed
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                node.move,
                ply,
                mover == colour,
                game_id=game_id,
                user=colour,
            )
        )
        board.push(node.move)
    return steps


def browse_session(pgn: str, *, step_s: float = 0.3, game_id: str = "b") -> list[Step] | None:
    """An analysis board browsed: a game stepped through a move every `step_s`, back ten plies
    one at a time, forward again: every position reached, the engine never ahead of it."""
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    moves = list(game.mainline_moves())[:40]
    board = game.board()
    steps = [Step(0.0, "start", board.copy(), game_id=game_id, user=chess.WHITE)]
    t = 0.0
    for ply, move in enumerate(moves[:30], start=1):
        t += step_s
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                move,
                ply,
                board.turn == chess.WHITE,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
        board.push(move)
    for back in range(10):  # back, one ply at a time
        t += step_s
        board.pop()
        steps.append(
            Step(
                t,
                "takeback",
                board.copy(),
                plies=1,
                ply=29 - back,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
    for ply, move in enumerate(moves[20:40], start=21):  # and forward again
        t += step_s
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                move,
                ply,
                board.turn == chess.WHITE,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
        board.push(move)
    return steps


def library_sessions(user: str, per_control: int) -> dict[str, list[Step]]:
    """The library's games with clocks, `per_control` of each time control (the longest first,
    for the most positions), from a copy of the profile; plus the blitz games at three times
    their speed (no bullet in the library), and one game browsed."""
    import shutil

    from sidecoach.profile.store import Store

    src = Path(os.path.expandvars(r"%LOCALAPPDATA%\Sidecoach\data\profile.sqlite"))
    copy = CACHE / "profile-copy.sqlite"
    CACHE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, copy)
    store = Store(copy)
    try:
        rows = store.library()
    finally:
        store.close()
    by_tc: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in rows:
        m = re.search(r'\[TimeControl "([^"]+)"\]', row.pgn)
        if m and "%clk" in row.pgn:
            by_tc[m.group(1)].append((row.pgn.count("%clk"), row.pgn))
    out: dict[str, list[Step]] = {}
    for tc, games in sorted(by_tc.items()):
        for n, (_, pgn) in enumerate(sorted(games, reverse=True)[:per_control]):
            steps = pgn_session(pgn, user, game_id=f"{tc}-{n}")
            if steps:
                out[f"{tc} #{n}"] = steps
            if tc.startswith("180"):
                fast = pgn_session(pgn, user, speed=3.0, game_id=f"bullet-{n}")
                if fast:
                    out[f"blitz x3 (bullet-like) #{n}"] = fast
    longest = max((g for games in by_tc.values() for g in games), default=None)
    if longest:
        browsed = browse_session(longest[1])
        if browsed:
            out["browsing a game"] = browsed
    return out


def chesscom_sessions(user: str, per_control: int = 1000) -> dict[str, list[Step]]:
    """The games `chesscom_games.py` fetched, at their clocks, at most `per_control` of each time
    control; the blitz ones at three times their speed too (bullet-like: there is no bullet in
    the archive)."""
    games = json.loads((CACHE / f"chesscom-{user}.json").read_text(encoding="utf-8"))
    out: dict[str, list[Step]] = {}
    taken: Counter[str] = Counter()
    for n, game in enumerate(games):
        tc = game.get("time_control", "?")
        if taken[tc] >= per_control:
            continue
        taken[tc] += 1
        steps = pgn_session(game["pgn"], user, game_id=f"{tc}-{n}")
        if steps:
            out[f"{tc} #{n}"] = steps
        if tc.startswith("180"):
            fast = pgn_session(game["pgn"], user, speed=3.0, game_id=f"bullet-{n}")
            if fast:
                out[f"blitz x3 #{n}"] = fast
    return out


def calibrate(cache: dict[str, dict], boards: dict[str, chess.Board], n: int = 20) -> float:
    """How much slower the precompute's times are than one search alone: `n` positions searched
    again, one at a time, their own search only; the median ratio."""
    import random

    keys = [k for k in boards if k in cache]
    random.Random(7).shuffle(keys)
    exe = next(STOCKFISH.glob("*.exe"))
    ratios = []
    with Engine(path=exe, threads=1, hash_mb=256, multipv=MULTIPV, pv_plies=PV_PLIES) as engine:
        annotator = GameAnnotator(engine=engine, depth=DEPTH, multipv=MULTIPV)
        for key in tqdm(keys[:n], desc="calibrating, one search alone", file=sys.stderr):
            engine.reconfigure(hash_mb=256)  # the hash cleared, as a fresh position finds it
            begun = time.perf_counter()
            annotator.search(boards[key])
            ratios.append((time.perf_counter() - begun) / cache[key]["main_s"])
    return statistics.median(ratios)


def main() -> None:
    import psutil

    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, nargs="?")
    parser.add_argument("--user", default="nosegazm")
    parser.add_argument("--games", type=int, default=4, help="games of each time control")
    parser.add_argument("--workers", type=int, nargs="+", default=[3])
    parser.add_argument("--look", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--reach",
        type=float,
        nargs="+",
        default=[0.15],
        help="the chance of being reached a position needs to be searched ahead",
    )
    # Every logical core: searches side by side run slower than one alone, and `calibrate`
    # measures by how much, so the times simulated are what the app's workers meet.
    parser.add_argument("--procs", type=int, default=max(1, (psutil.cpu_count() or 4) - 2))
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--chesscom",
        action="store_true",
        help="every game fetched by tools/engine/chesscom_games.py",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="measure again how much slower searches side by side run",
    )
    args = parser.parse_args()
    sessions: dict[str, list[Step]] = {}
    if args.dump:
        sessions[f"puzzles ({args.dump.name})"] = session(args.dump)
    if args.chesscom:
        sessions.update(chesscom_sessions(args.user, args.games))
    else:
        sessions.update(library_sessions(args.user, args.games))
    every = [s for steps in sessions.values() for s in steps]
    cache = load_cache(Path("all"))
    boards = game_boards(every)
    search_missing(boards, cache, args.procs, "the games' own positions")
    # The ratio is the machine's, measured once and kept; --recalibrate measures it again.
    kept = CACHE / "scale.json"
    if kept.exists() and not args.recalibrate:
        scale = json.loads(kept.read_text(encoding="utf-8"))["scale"]
    else:
        scale = calibrate(cache, boards)
        kept.write_text(json.dumps({"scale": scale}), encoding="utf-8")
    settings = [(w, la, r) for w in args.workers for la in args.look for r in args.reach]
    # The look-ahead picks positions no list made beforehand could: what a run lacks is searched
    # and the run made again, until nothing is lacking.
    for round_ in range(6):
        results, lacking = [], {}
        for workers, look, reach in settings:
            for name, steps in sessions.items():
                r = simulate(
                    steps, cache, workers=workers, look=look, scale=scale, reach=reach, raw=True
                )
                lacking.update(r.pop("missing_boards"))
                r.update(session=name, reach=reach)
                results.append(r)
        if not lacking:
            break
        search_missing(lacking, cache, args.procs, f"the look-ahead's, round {round_ + 1}")
    print(
        f"{len(sessions)} sessions, {len(cache)} positions searched; one search alone takes "
        f"{scale:.2f} of the time kept: times scaled by it"
    )
    _pooled(results)
    if args.json:
        for r in results:
            r.pop("raw_waits", None)
        args.json.write_text(json.dumps(results, indent=1), encoding="utf-8")


def _kind(session_name: str) -> str:
    return session_name.split(" #")[0].split(" (")[0]


def _pooled(results: list[dict]) -> None:
    """Each setting's numbers over each kind of session: the processor time summed, the waits
    after a move pooled."""
    groups: dict[tuple, dict] = defaultdict(
        lambda: {"core": 0.0, "span": 0.0, "never": 0.0, "waits": defaultdict(list), "n": 0}
    )
    for r in results:
        for kind in (_kind(r["session"]), "ALL"):
            g = groups[(kind, r["workers"], r["look"], r["reach"])]
            g["core"] += r["stockfish_core_s"]
            g["span"] += r["span_s"]
            g["never"] += r["split_s"].get("ahead, never reached", 0.0)
            g["n"] += 1
            for k, v in r["raw_waits"].items():
                g["waits"][k] += v

    def q(values: list[float], at: float) -> str:
        v = sorted(values)
        return f"{v[min(len(v) - 1, int(len(v) * at))]:.2f}" if v else "-"

    print(
        f"{'sessions':16} {'w':>2} {'la':>2} {'reach':>5} {'cores':>5} {'wasted':>6} "
        f"{'you verdict p50/p90':>19} {'opp verdict p50/p90':>19} "
        f"{'you full p50/p90':>16} {'opp full p90':>12}"
    )
    for (kind, w, la, reach), g in sorted(groups.items()):
        wv = g["waits"]
        print(
            f"{kind[:16]:16} {w:>2} {la:>2} {reach:5.2f} {g['core'] / g['span']:5.2f} "
            f"{g['never'] / max(g['core'], 1e-9):6.0%} "
            f"{q(wv['classification (you)'], 0.5):>9}/{q(wv['classification (you)'], 0.9):<9} "
            f"{q(wv['classification (opponent)'], 0.5):>9}/"
            f"{q(wv['classification (opponent)'], 0.9):<9} "
            f"{q(wv['full analysis (you)'], 0.5):>7}/{q(wv['full analysis (you)'], 0.9):<8} "
            f"{q(wv['full analysis (opponent)'], 0.9):>12}"
        )


if __name__ == "__main__":
    main()
