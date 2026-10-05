"""The slowest positions of the workloads, looked at closely: when each depth of their search
arrives (one thread, cold), what the same search costs right after the position it came from
was searched on the same engine (warm, as in a game), and on 2, 4 and 8 threads (cold).

    uv run --no-sync python tools/engine/slow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.engine import Engine  # noqa: E402
from tools.engine.measure import STOCKFISH  # noqa: E402

GAME = (
    "e2e4 c7c5 g1f3 b8c6 f1b5 e7e5 c2c3 g8f6 b5c6 b7c6 d2d3 c8a6 f3e5 f8d6 e5c4 a6c4 d3c4 f6e4 "
    "e1g1 e8g8 f1e1 f8e8 f2f3 d6h2"
).split()
SECOND = "r1bk3r/ppp2Qp1/2np3q/2bB2p1/3pP3/8/PPPN1PPP/R1B2RK1 w - - 0 1"


def board_of(moves: list[str]) -> chess.Board:
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b


def timed(engine: Engine, board: chess.Board, lines: int, clear: bool = True) -> str:
    assert engine._engine is not None
    if clear:
        engine._engine.configure({"Clear Hash": None})
    marks: dict[int, float] = {}
    t0 = time.perf_counter()

    def on_depth(found: list) -> None:
        if found and found[0].depth not in marks:
            marks[found[0].depth] = time.perf_counter() - t0

    engine.analyse(board, depth=18, multipv=lines, on_depth=on_depth)
    total = time.perf_counter() - t0
    at = " ".join(f"d{d}:{marks[d]:.1f}" for d in (12, 14, 16, 17) if d in marks)
    return f"{total:5.1f}s  ({at})"


def main() -> None:
    slow = board_of(GAME)
    parent = board_of(GAME[:-1])
    second = chess.Board(SECOND)
    cases = [("in check, 3 moves", slow), ("second slowest (FEN only)", second)]
    for threads in (1, 2, 4, 8):
        engine = Engine(path=STOCKFISH, threads=threads, hash_mb=256, multipv=6)
        engine.open()
        try:
            for name, board in cases:
                for lines in (2, 6):
                    print(f"{threads}T {lines} lines, {name}: {timed(engine, board, lines)}", flush=True)
            if threads == 1:
                for lines in (2, 6):
                    engine._engine.configure({"Clear Hash": None})
                    engine.analyse(parent, depth=18, multipv=6)  # the position before, as in a game
                    warm = timed(engine, slow, lines, clear=False)
                    print(f"1T {lines} lines, in check, WARM after its parent: {warm}", flush=True)
        finally:
            engine.close()


if __name__ == "__main__":
    main()
