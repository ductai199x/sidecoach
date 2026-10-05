"""The atoms of the engine's work, measured on real Stockfish: what one search costs, in wall
time and in CPU, by its lines and threads; when its depth 12 arrives; what the side searches
cost; and how much slower a search runs while others run beside it.

A, one search alone: positions from real sessions (the simulator's cache, a seeded sample),
each searched to depth 18 by every variant (lines x threads), each variant on an engine of its
own (its own hash, as a worker's), the engines pinned to physical cores of the second die and
this process to a core of its own; the variants' order turns with each position. CPU is read
from the engine's process, never computed. The first positions are searched again at the end:
the run-to-run noise.

B, side by side: k one-thread six-line searches at once (k = 1 .. 12), each engine on a
physical core of its own, all started together; each search's time over the same position's
time alone (from A).

    uv run --no-sync python tools/engine/atoms.py
    Get-Content "D:\\chess-coaching-companion\\data\\engine-cache\\atoms.log" -Wait -Tail 2
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
SEARCHES = CACHE / "searches-2f003a7edf.sqlite"
OUT = CACHE / "atoms.json"
STOCKFISH = next(
    (Path.home() / "AppData" / "Local" / "Sidecoach" / "data" / "tools" / "stockfish").glob("*.exe")
)
DEPTH, HASH_MB, PV_PLIES = 18, 256, 12
# (lines, threads)
VARIANTS = [(1, 1), (6, 1), (1, 2), (6, 2), (1, 4), (6, 4), (6, 8)]
# Logical CPUs 2k and 2k+1 are one physical core; 0-11 the first die, 12-23 the second.
SECOND_DIE = [12, 14, 16, 18, 20, 22]
ME = 23  # this process, beside the engines, on a core of its own
ALL_PHYSICAL = list(range(0, 24, 2))
KS = [1, 2, 3, 4, 6, 8, 10, 12]


def cores_for(threads: int) -> list[int]:
    """One physical core per thread: the second die's first, then the first die's."""
    return (SECOND_DIE + [0, 2, 4, 6, 8, 10])[:threads]


def engine_process(engine: Engine) -> Any:
    import psutil

    pid = engine._engine.transport.get_pid()  # type: ignore[union-attr]
    return psutil.Process(pid)


def cpu(process: Any) -> float:
    t = process.cpu_times()
    return t.user + t.system


def make(lines: int, threads: int, cores: list[int]) -> tuple[Engine, Any]:
    engine = Engine(
        path=STOCKFISH, threads=threads, hash_mb=HASH_MB, multipv=lines, pv_plies=PV_PLIES
    )
    engine.open()
    process = engine_process(engine)
    process.cpu_affinity(cores)
    engine.analyse(chess.Board(), depth=1)  # threads, hash and network loaded before timing
    return engine, process


def search(engine: Engine, process: Any, board: chess.Board, lines: int) -> dict[str, float | None]:
    """One search, cold: the hash cleared first, so no search is helped by an earlier one."""
    assert engine._engine is not None
    engine._engine.configure({"Clear Hash": None})
    at12: list[float] = []
    c0, t0 = cpu(process), time.perf_counter()

    def on_depth(found: list) -> None:
        if found and found[0].depth >= 12 and not at12:
            at12.append(time.perf_counter() - t0)

    found = engine.analyse(board, depth=DEPTH, multipv=lines, on_depth=on_depth)
    wall = time.perf_counter() - t0
    reached = found[0].depth if found else 0
    assert reached == DEPTH, f"searched to {reached}, not {DEPTH}"
    return {"wall": wall, "cpu": cpu(process) - c0, "at12": at12[0] if at12 else None}


def positions(n: int) -> list[str]:
    db = sqlite3.connect(SEARCHES)
    keys = sorted(k for (k,) in db.execute("SELECT key FROM search"))
    db.close()
    random.Random(2026_10_04).shuffle(keys)
    fens = []
    for key in keys:
        fen = " ".join(key.split(" ")[:6])
        board = chess.Board(fen)
        if not board.is_game_over() and fen not in fens:
            fens.append(fen)
        if len(fens) == n:
            break
    return fens


def alone(fens: list[str], repeat: int, bar: Any) -> list[dict[str, Any]]:
    """Experiment A."""
    engines: dict[tuple[int, int], tuple[Engine, Any]] = {}
    for lines, threads in VARIANTS:
        engines[(lines, threads)] = make(lines, threads, cores_for(threads))
    side_engine, side_process = make(6, 1, SECOND_DIE[:1])
    annotator = GameAnnotator(engine=side_engine, depth=DEPTH, multipv=6)
    rows = []
    try:
        for i, fen in enumerate(fens + fens[:repeat]):
            board = chess.Board(fen)
            row: dict[str, Any] = {"fen": fen, "repeat": i >= len(fens)}
            order = VARIANTS[i % len(VARIANTS) :] + VARIANTS[: i % len(VARIANTS)]
            for lines, threads in order:
                if row["repeat"] and (lines, threads) != (6, 1):
                    continue
                engine, process = engines[(lines, threads)]
                row[f"{lines}x{threads}"] = search(engine, process, board, lines)
                bar.update(1)
            if not row["repeat"]:
                measured = annotator.search(board)
                c0, t0 = cpu(side_process), time.perf_counter()
                annotator.with_side(board, measured)
                row["side"] = {"wall": time.perf_counter() - t0, "cpu": cpu(side_process) - c0}
                bar.update(1)
            rows.append(row)
    finally:
        for engine, _ in engines.values():
            engine.close()
        side_engine.close()
    return rows


def side_by_side(fens: list[str], bar: Any) -> list[dict[str, Any]]:
    """Experiment B: k searches at once, each on a physical core of its own."""
    out = []
    for k in KS:
        engines = [make(6, 1, [ALL_PHYSICAL[j]]) for j in range(k)]
        try:
            queue = list(fens)
            while queue:
                batch, queue = queue[:k], queue[k:]
                results: list[dict[str, Any]] = [{} for _ in batch]
                start = threading.Barrier(len(batch))

                def run(j: int, fen: str) -> None:
                    start.wait()
                    engine, process = engines[j]
                    results[j] = {
                        "fen": fen,
                        "k": k,
                        **search(engine, process, chess.Board(fen), 6),
                    }

                workers = [threading.Thread(target=run, args=(j, f)) for j, f in enumerate(batch)]
                for w in workers:
                    w.start()
                for w in workers:
                    w.join()
                out.extend(results)
                bar.update(len(batch))
        finally:
            for engine, _ in engines:
                engine.close()
    return out


def main() -> None:
    import psutil
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    psutil.Process().cpu_affinity([ME])
    fens = positions(args.n)
    total = args.n * (len(VARIANTS) + 1) + args.repeat + args.n * len(KS)
    with tqdm(total=total, desc="searches", file=sys.stdout, mininterval=2) as bar:
        a = alone(fens, args.repeat, bar)
        OUT.write_text(json.dumps({"alone": a}), encoding="utf-8")
        b = side_by_side(fens, bar)
    OUT.write_text(json.dumps({"alone": a, "side_by_side": b}), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
