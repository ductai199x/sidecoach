"""The puzzles depth 18 really missed, searched deeper: at the move it missed, one line to depth
`--depth` (26), the first depth whose best move is the solution (or mates) and stays so, and the
time it came at. What a puzzle position searched on while the user thinks would find.

    uv run --no-sync python tools/engine/puzzle_deep.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.engine.quality_puzzles import PUZZLES, STOCKFISH  # noqa: E402

C = ROOT / "data" / "engine-cache"
_W: dict = {}


def _init() -> None:
    from sidecoach.analysis.engine import Engine

    e = Engine(path=STOCKFISH, threads=1, hash_mb=256, multipv=1, pv_plies=4)
    e.open()
    import atexit

    atexit.register(e.close)
    _W["e"] = e


def _deep(task: tuple[str, str, str, int, int]) -> tuple[str, int | None, float | None]:
    pid, fen, moves_text, step, depth = task
    moves = moves_text.split()
    board = chess.Board(fen)
    board.push_uci(moves[0])
    for i in range(1, 2 * step + 1, 2):
        board.push_uci(moves[i])
        board.push_uci(moves[i + 1])
    solution = moves[2 * step + 1]
    last = 2 * step + 1 == len(moves) - 1
    chosen: dict[int, tuple[str, float]] = {}
    t0 = time.perf_counter()

    def on_depth(found: list) -> None:
        if found and found[0].moves:
            chosen[found[0].depth] = (found[0].moves[0], time.perf_counter() - t0)

    _W["e"].analyse(board, depth=depth, multipv=1, on_depth=on_depth)

    def right(uci: str) -> bool:
        if uci == solution:
            return True
        if last:
            b = board.copy()
            b.push_uci(uci)
            return b.is_checkmate()
        return False

    found_at = None
    for d in sorted(chosen):
        if right(chosen[d][0]):
            if found_at is None:
                found_at = d
        else:
            found_at = None
    return pid, found_at, chosen[found_at][1] if found_at else None


def main() -> None:
    out = json.loads((C / "quality_puzzles.json").read_text(encoding="utf-8"))
    misses = {}
    for r in out:
        for k, s in enumerate(r["steps"]):
            if not s["right"].get("18", False):
                misses[r["id"]] = k
                break
    rows = {}
    with PUZZLES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["PuzzleId"] in misses:
                rows[row["PuzzleId"]] = row
    tasks = [(pid, r["FEN"], r["Moves"], misses[pid], 26) for pid, r in rows.items()]
    results = []
    with ProcessPoolExecutor(12, initializer=_init) as pool:
        for fut in as_completed([pool.submit(_deep, t) for t in tasks]):
            results.append(fut.result())
    by = sorted(d for _, d, _ in results if d)
    print(f"{len(results)} puzzles depth 18 missed; found (and kept) by depth 26: {len(by)}")
    for d in range(19, 27):
        n = sum(1 for x in by if x <= d)
        print(f"  by depth {d}: {n}")
    times = sorted(t for _, d, t in results if t)
    if times:
        print(
            f"  the time those took, alone: "
            f"median {times[len(times) // 2]:.1f}s max {times[-1]:.1f}s"
        )


if __name__ == "__main__":
    main()
