"""How often the engine solves Lichess puzzles at each depth: every position of a puzzle where
the solver is to move searched once, one line, one thread, to depth `--depth` (20), and the best
move each depth on the way chose kept (a single-thread search to depth d is exactly the first d
iterations of a deeper one). A puzzle is solved at depth d when every solver move is the move
depth d chose -- or, on the last move, any move that mates (Lichess takes any mate in one).
Also each depth's time.

Twelve processes on the first die's cores (the second die is the timing measurements'):
correctness, not time, is what this measures.

    uv run --no-sync python tools/engine/quality_puzzles.py --n 30000
    Get-Content data\\engine-cache\\quality_puzzles.log -Wait -Tail 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "quality_puzzles.json"
PUZZLES = ROOT / "data" / "puzzles" / "full-11-100000.csv"
STOCKFISH = next(
    (Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "data" / "tools" / "stockfish").glob("*.exe")
)
FIRST_DIE = list(range(12))
_W: dict[str, Any] = {}


def _init(cores: list[int]) -> None:
    import psutil

    from sidecoach.analysis.engine import Engine

    FIRST_DIE[:] = cores
    psutil.Process().cpu_affinity(FIRST_DIE)
    engine = Engine(path=STOCKFISH, threads=1, hash_mb=64, multipv=1, pv_plies=4)
    engine.open()
    assert engine._engine is not None
    psutil.Process(engine._engine.transport.get_pid()).cpu_affinity(FIRST_DIE)
    import atexit

    atexit.register(engine.close)
    _W["engine"] = engine


def _puzzle(task: tuple[str, str, str, int]) -> dict[str, Any]:
    pid, fen, moves_text, depth = task
    engine = _W["engine"]
    moves = moves_text.split()
    board = chess.Board(fen)
    board.push_uci(moves[0])
    steps = []
    for i in range(1, len(moves), 2):
        solution = moves[i]
        last = i == len(moves) - 1
        engine._engine.configure({"Clear Hash": None})
        chosen: dict[int, tuple[str, float]] = {}
        t0 = time.perf_counter()

        def on_depth(found: list, t0: float = t0, chosen: dict = chosen) -> None:
            if found and found[0].moves:
                chosen[found[0].depth] = (found[0].moves[0], time.perf_counter() - t0)

        lines = engine.analyse(board, depth=depth, multipv=1, on_depth=on_depth)
        if lines and lines[0].moves:
            chosen.setdefault(lines[0].depth, (lines[0].moves[0], time.perf_counter() - t0))
        right = {}
        for d, (uci, _) in chosen.items():
            ok = uci == solution
            if not ok and last:
                after = board.copy()
                after.push_uci(uci)
                ok = after.is_checkmate()
            right[d] = ok
        steps.append({"right": right, "times": {d: t for d, (_, t) in chosen.items()}})
        board.push_uci(solution)
        if i + 1 < len(moves):
            board.push_uci(moves[i + 1])
    return {"id": pid, "steps": steps}


def main() -> None:
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30000)
    parser.add_argument("--offset", type=int, default=0, help="skip this many puzzles first")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--die", type=int, default=0, help="0: logical CPUs 0-11; 1: 12-23")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    FIRST_DIE[:] = list(range(12 * args.die, 12 * args.die + 12))
    with PUZZLES.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for _ in range(args.offset):
            next(reader)
        rows = [r for _, r in zip(range(args.n), reader, strict=False)]
    tasks = [(r["PuzzleId"], r["FEN"], r["Moves"], args.depth) for r in rows]
    out = []
    pool = ProcessPoolExecutor(len(FIRST_DIE), initializer=_init, initargs=(list(FIRST_DIE),))
    try:
        futures = [pool.submit(_puzzle, t) for t in tasks]
        for n, fut in enumerate(
            tqdm(
                as_completed(futures),
                total=len(futures),
                desc="puzzles",
                file=sys.stdout,
                mininterval=2,
            ),
            start=1,
        ):
            out.append(fut.result())
            if n % 2000 == 0:
                args.out.write_text(json.dumps(out), encoding="utf-8")
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    args.out.write_text(json.dumps(out), encoding="utf-8")
    report(out, args.depth)


def report(out: list[dict[str, Any]], depth: int) -> None:
    n = len(out)
    print(f"{n} puzzles, {sum(len(r['steps']) for r in out)} solver moves")
    for d in range(10, depth + 1):
        solved = sum(
            1
            for r in out
            if all(s["right"].get(str(d), s["right"].get(d, False)) for s in r["steps"])
        )
        print(f"  depth {d}: solved {solved}/{n} ({solved / n:.4%}); failed {n - solved}")


if __name__ == "__main__":
    main()
