"""The puzzles depth 18 'failed' (quality_puzzles.py), looked at: at the solver move it differed
on, the engine's move and the solution each searched to depth 24, one line -- is the engine's
move winning too (an alternative the puzzle does not list), or a real miss?

    uv run --no-sync python tools/engine/puzzle_misses.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sidecoach.analysis.engine import Engine  # noqa: E402
from tools.engine.quality_puzzles import PUZZLES, STOCKFISH  # noqa: E402

C = ROOT / "data" / "engine-cache"


def main() -> None:
    out = json.loads((C / "quality_puzzles.json").read_text(encoding="utf-8"))
    failed = {r["id"]: r for r in out if not all(s["right"].get("18", False) for s in r["steps"])}
    puzzles = {}
    with PUZZLES.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["PuzzleId"] in failed:
                puzzles[row["PuzzleId"]] = row
    engine = Engine(path=STOCKFISH, threads=1, hash_mb=256, multipv=1)
    engine.open()
    kinds = {"engine's move wins too": 0, "real miss": 0}
    try:
        for pid, row in puzzles.items():
            moves = row["Moves"].split()
            board = chess.Board(row["FEN"])
            board.push_uci(moves[0])
            steps = failed[pid]["steps"]
            for k, i in enumerate(range(1, len(moves), 2)):
                if not steps[k]["right"].get("18", False):
                    engine._engine.configure({"Clear Hash": None})
                    mine = engine.analyse(board, depth=18, multipv=1)[0].moves[0]
                    values = {}
                    for m in (moves[i], mine):
                        child = board.copy()
                        child.push_uci(m)
                        if child.is_checkmate():
                            values[m] = 1.0
                            continue
                        engine._engine.configure({"Clear Hash": None})
                        line = engine.analyse(child, depth=23, multipv=1)
                        values[m] = 1.0 - line[0].expected_score if line else 0.5
                    sol, eng = values[moves[i]], values[mine]
                    wins = eng >= 0.85 or eng >= sol - 0.02
                    kinds["engine's move wins too" if wins else "real miss"] += 1
                    print(f"{pid} rating {row['Rating']}: solution {moves[i]} {sol:.2f}, engine {mine} {eng:.2f} "
                          f"-> {'wins too' if wins else 'REAL MISS'}")
                    break
                board.push_uci(moves[i])
                if i + 1 < len(moves):
                    board.push_uci(moves[i + 1])
    finally:
        engine.close()
    print(kinds)


if __name__ == "__main__":
    main()
