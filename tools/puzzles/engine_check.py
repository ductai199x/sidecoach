"""Does our engine find the puzzle solution? The soundness check for the analysis layer.

A Lichess puzzle's first solution move is (bar filtered duals) the uniquely best move in the
position the solver faces. So we ask our Stockfish for the position and check whether its
best line starts with that move -- and, when it does not, whether the solution move is at
least among the lines it returns. A high top-1 rate means the analysis the coach is built on
actually sees the win.

Run:  uv run python tools/puzzles/engine_check.py [limit] [depth]
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.config import find_stockfish, load_settings

SAMPLE = Path("data/puzzles/sample.csv")


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    path = find_stockfish(load_settings())
    if path is None:
        print("no stockfish found")
        return 1

    rows = list(csv.DictReader(SAMPLE.open(encoding="utf-8", newline="")))
    random.Random(7).shuffle(rows)

    top1 = returned = total = 0
    misses: list[tuple[str, str, str, str]] = []
    with Engine(path=path, multipv=4, max_seconds=2.5) as engine:
        for row in rows:
            if total >= limit:
                break
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            try:
                board = chess.Board(row["FEN"])
                board.push(chess.Move.from_uci(moves[0]))
            except (ValueError, AssertionError):
                continue
            want = moves[1]
            if chess.Move.from_uci(want) not in board.legal_moves:
                continue
            total += 1
            lines = engine.analyse(board, depth=depth)
            if not lines:
                continue
            firsts = [ln.moves[0] for ln in lines if ln.moves]
            if firsts and firsts[0] == want:
                top1 += 1
            elif want in firsts:
                returned += 1
            else:
                themes = " ".join(t for t in row["Themes"].split() if t in _KEY)[:40]
                misses.append((row["PuzzleId"], want, firsts[0] if firsts else "-", themes))

    print(f"engine solution-match on {total} puzzles at depth {depth} (2.5s cap):")
    print(f"  best move IS the solution:      {top1:4d}/{total}  {top1 / total:5.0%}")
    print(f"  solution among the top lines:   {returned:4d}/{total}  {returned / total:5.0%}")
    other = total - top1 - returned
    print(f"  solution not in the top lines:  {other:4d}/{total}  {other / total:5.0%}")
    print("\na few where our first choice differed (puzzle, solution, ours, themes):")
    for pid, want, ours, themes in misses[:12]:
        print(f"  {pid:8s} want {want:6s} got {ours:6s}  {themes}")
    return 0


_KEY = {
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "hangingPiece",
    "trappedPiece",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "backRankMate",
    "sacrifice",
    "deflection",
    "attraction",
}

if __name__ == "__main__":
    raise SystemExit(main())
