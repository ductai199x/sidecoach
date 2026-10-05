"""Every position the workloads need, searched cleanly on real Stockfish: one thread, depth 18,
cold (the hash cleared first), each of the searches the design may ask of it -- two lines, six
lines, the threats -- with the lines found and when each depth came. Kept in SQLite by the
position's cache key (its FEN and the moves a repetition can reach back over), so a position
is measured once, ever.

Six processes, each on a physical core of its own on the second die; the positions atoms.py
measured alone are measured here too, so what running six at once costs each search is read off
directly (`--check`), never assumed.

    uv run --no-sync python tools/engine/measure.py              # the workloads' positions
    uv run --no-sync python tools/engine/measure.py --check      # the six-at-once distortion
    Get-Content "D:\\chess-coaching-companion\\data\\engine-cache\\measure.log" -Wait -Tail 2
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import cache_key  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
DB = CACHE / "clean.sqlite"
STOCKFISH = next(
    (Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "data" / "tools" / "stockfish").glob("*.exe")
)
CORES = [12, 14, 16, 18, 20, 22]
DEPTH = 18

_W: dict[str, Any] = {}


def _worker_init(cores: list[int]) -> None:
    """Each process takes the next core of the list, it and its engine pinned there."""
    import atexit
    import multiprocessing

    import psutil

    index = (multiprocessing.current_process()._identity[0] - 1) % len(cores)
    core = cores[index]
    psutil.Process().cpu_affinity([core + 1])
    _W["core"] = core
    _worker_init_engine()
    atexit.register(lambda: _W["engine"].close())


def _worker_init_engine() -> None:
    """This process's engine, started and pinned to its core."""
    import psutil

    from sidecoach.analysis.engine import Engine

    engine = Engine(path=STOCKFISH, threads=1, hash_mb=256, multipv=6, pv_plies=12)
    engine.open()
    assert engine._engine is not None
    process = psutil.Process(engine._engine.transport.get_pid())
    process.cpu_affinity([_W["core"]])
    engine.analyse(chess.Board(), depth=1)
    _W.update(engine=engine, process=process)


def _board(root: str, moves: list[str]) -> chess.Board:
    board = chess.Board(root)
    for uci in moves:
        board.push_uci(uci)
    return board


def _measure(task: tuple[str, str, list[str]]) -> tuple[str, dict[str, Any] | None]:
    """A position measured; once more on a fresh engine if the engine fails (an engine whose
    event loop died took the whole round with it once, on a position that searches fine
    alone); None, and the failure on stderr, if it fails again."""
    import traceback

    for attempt in (1, 2):
        try:
            return _measure_once(task)
        except Exception:  # noqa: BLE001 - one position's failure must not end the round
            traceback.print_exc(file=sys.stderr)
            if attempt == 1:
                _W["engine"].close()
                _worker_init_engine()
    return task[0], None


def _measure_once(task: tuple[str, str, list[str]]) -> tuple[str, dict[str, Any]]:
    from sidecoach.analysis.threats import threat_moves

    key, root, moves = task
    engine, process = _W["engine"], _W["process"]
    board = _board(root, moves)
    out: dict[str, Any] = {}

    def cpu() -> float:
        t = process.cpu_times()
        return t.user + t.system

    for lines in (2, 6):
        engine._engine.configure({"Clear Hash": None})
        depths: list[tuple[int, float]] = []
        t0, c0 = time.perf_counter(), cpu()

        def on_depth(found: list, t0: float = t0, depths: list = depths) -> None:
            if found:
                depths.append((found[0].depth, time.perf_counter() - t0))

        found = engine.analyse(board, depth=DEPTH, multipv=lines, on_depth=on_depth)
        out[str(lines)] = {
            "wall": time.perf_counter() - t0,
            "cpu": cpu() - c0,
            "depths": depths,
            "lines": [ln.model_dump() for ln in found],
        }
    # One line to depth 18: the search the one-line design makes of every position.
    engine._engine.configure({"Clear Hash": None})
    marks: list[tuple[int, float, float]] = []
    t0, c0 = time.perf_counter(), cpu()

    def mark(found: list, t0: float = t0, marks: list = marks) -> None:
        if found and (not marks or found[0].depth > marks[-1][0]):
            marks.append((found[0].depth, time.perf_counter() - t0, found[0].expected_score))

    one = engine.analyse(board, depth=DEPTH, multipv=1, on_depth=mark)
    out["1"] = {
        "wall": time.perf_counter() - t0,
        "cpu": cpu() - c0,
        "lines": [ln.model_dump() for ln in one],
        "marks": marks,  # each depth: (depth, seconds, expected score) -- the decided rule's
    }
    engine._engine.configure({"Clear Hash": None})
    t0, c0 = time.perf_counter(), cpu()
    threats = threat_moves(engine, board, depth=DEPTH)
    out["threats"] = {"wall": time.perf_counter() - t0, "cpu": cpu() - c0, "moves": threats}
    return key, out


def workloads() -> dict[str, list[Any]]:
    """The sessions the design is measured on: the puzzle session, six games of each long time
    control and every blitz game at their clocks, the blitz again at three times its speed, and
    a game browsed."""
    from tools.engine.sessions import browse_session, chesscom_sessions, session

    dump = Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "data" / "dumps" / "20261004-150208"
    out: dict[str, list[Any]] = {"puzzles #0": session(dump)}
    games = chesscom_sessions("nosegazm", 1000)
    taken: dict[str, int] = {}
    for name, steps in games.items():
        kind = name.split(" #")[0]
        if kind in ("600", "900+10") and taken.get(kind, 0) >= 6:
            continue
        taken[kind] = taken.get(kind, 0) + 1
        out[name] = steps
    archive = json.loads((CACHE / "chesscom-nosegazm.json").read_text(encoding="utf-8"))
    browsed = browse_session(archive[0]["pgn"])
    if browsed:
        out["browsing #0"] = browsed
    return out


def positions(sessions: dict[str, list[Any]]) -> dict[str, chess.Board]:
    from tools.engine.sessions import game_boards

    out: dict[str, chess.Board] = {}
    for steps in sessions.values():
        out.update(game_boards(steps))
    return out


def open_db() -> sqlite3.Connection:
    CACHE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.execute("CREATE TABLE IF NOT EXISTS clean (key TEXT PRIMARY KEY, found TEXT)")
    return db


def measure(boards: dict[str, chess.Board], what: str) -> None:
    """The boards not yet in the store, measured and kept as each comes."""
    from tqdm import tqdm

    db = open_db()
    have = {k for (k,) in db.execute("SELECT key FROM clean")}
    todo = [
        (k, b.root().fen(), [m.uci() for m in b.move_stack])
        for k, b in boards.items()
        if k not in have
    ]
    if not todo:
        return
    pool = ProcessPoolExecutor(len(CORES), initializer=_worker_init, initargs=(CORES,))
    try:
        futures = [pool.submit(_measure, t) for t in todo]
        for n, fut in enumerate(
            tqdm(
                as_completed(futures), total=len(futures), desc=what, file=sys.stdout, mininterval=2
            ),
            start=1,
        ):
            key, found = fut.result()
            if found is None:
                print(f"\nnot measured (failed twice): {key}", file=sys.stderr)
                continue
            db.execute("INSERT OR REPLACE INTO clean VALUES (?, ?)", (key, json.dumps(found)))
            if n % 20 == 0:
                db.commit()
        db.commit()
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for process in processes:  # their engines close at exit; the pool never ends them
            process.terminate()
        db.close()


def check() -> None:
    """The positions atoms.py measured alone, measured six at once here: the distortion."""
    atoms = json.loads((CACHE / "atoms.json").read_text(encoding="utf-8"))["alone"]
    alone = {r["fen"]: r["6x1"]["wall"] for r in atoms if not r["repeat"]}
    boards = {cache_key(chess.Board(f)): chess.Board(f) for f in alone}
    measure(boards, "check (alone-measured positions)")
    db = open_db()
    ratios = []
    for key, board in boards.items():
        row = db.execute("SELECT found FROM clean WHERE key = ?", (key,)).fetchone()
        ratios.append(json.loads(row[0])["6"]["wall"] / alone[board.fen()])
    ratios.sort()
    n = len(ratios)
    print(
        f"six at once over alone, {n} positions: "
        f"p10 {ratios[n // 10]:.2f} p50 {ratios[n // 2]:.2f} "
        f"p90 {ratios[9 * n // 10]:.2f} max {ratios[-1]:.2f}; sum {sum(ratios) / n:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    boards = positions(workloads())
    print(f"{len(boards)} positions in the workloads")
    measure(boards, "the workloads' positions")


if __name__ == "__main__":
    main()
