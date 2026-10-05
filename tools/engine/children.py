"""The slow positions' alternatives searched one by one: each legal move's position searched
with one line to depth 17 (depth 18 from the position before), cold, against the position's
own multi-line search. Raw UCI, one thread, hash 256.

    uv run --no-sync python tools/engine/children.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.engine.measure import STOCKFISH  # noqa: E402

MOVES = [
    "e2e4",
    "c7c5",
    "g1f3",
    "b8c6",
    "f1b5",
    "e7e5",
    "c2c3",
    "g8f6",
    "b5c6",
    "b7c6",
    "d2d3",
    "c8a6",
    "f3e5",
    "f8d6",
    "e5c4",
    "a6c4",
    "d3c4",
    "f6e4",
    "e1g1",
    "e8g8",
    "f1e1",
    "f8e8",
    "f2f3",
    "d6h2",
]
SECOND = "r1bk3r/ppp2Qp1/2np3q/2bB2p1/3pP3/8/PPPN1PPP/R1B2RK1 w - - 0 1"


def search(position: str, depth: int, lines: int) -> tuple[float, str]:
    proc = subprocess.Popen(
        [str(STOCKFISH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(
        "uci\nsetoption name Threads value 1\nsetoption name Hash value 256\n"
        f"setoption name MultiPV value {lines}\nisready\n{position}\n"
    )
    proc.stdin.flush()
    for text in proc.stdout:
        if text.startswith("readyok"):
            break
    t0 = time.perf_counter()
    proc.stdin.write(f"go depth {depth}\n")
    proc.stdin.flush()
    score = ""
    for text in proc.stdout:
        m = re.search(r"multipv 1 score (\w+ -?\d+)", text)
        if m:
            score = m.group(1)
        if text.startswith("bestmove"):
            break
    took = time.perf_counter() - t0
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()
    return took, score


def main() -> None:
    for name, root, moves in (
        ("in check, 3 moves", "startpos", MOVES),
        ("second slowest", f"fen {SECOND}", []),
    ):
        board = chess.Board() if root == "startpos" else chess.Board(SECOND)
        for m in moves:
            board.push_uci(m)

        def pos(extra: list[str], root=root, moves=moves) -> str:  # noqa: ANN001
            seq = moves + extra
            return f"position {root}" + (f" moves {' '.join(seq)}" if seq else "")

        for lines in (1, 2, 6):
            took, score = search(pos([]), 18, lines)
            print(f"{name}: the position, {lines} line(s), depth 18: {took:.2f}s ({score})")
        total = 0.0
        for move in board.legal_moves:
            took, score = search(pos([move.uci()]), 17, 1)
            total += took
            print(f"   after {board.san(move):6s} one line, depth 17: {took:.2f}s ({score})")
        print(f"   every move this way: {total:.2f}s for {board.legal_moves.count()} moves")


if __name__ == "__main__":
    main()
