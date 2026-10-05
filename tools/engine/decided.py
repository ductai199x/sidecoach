"""Whether a decided position's verdicts can be read before depth 18: on the 300 slowest
one-line searches of the 8,886 workload positions (every position slower than the budgets
below is among them), each depth's expected score recorded; the rule "after `budget` seconds,
if the deepest depth finished shows the position decided (expected score at most 0.15 or at
least 0.85), its score stands in for depth 18's" applied; and every legal move from each
position the rule fires on judged both ways -- its loss from the position's depth-18 score and
from the score the rule took, against the move's own position searched with one line to depth
17 (18 from here) -- and the two severities compared. Accepted only if no severity changes.

    uv run --no-sync python tools/engine/decided.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.score import Thresholds, severity  # noqa: E402
from tools.engine.measure import CORES, _W, _worker_init  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "decided.json"
SIX_AT_ONCE = 1.13
BUDGETS = (0.5, 1.0, 2.0)
DECIDED = (0.15, 0.85)


def _search(board: chess.Board, depth: int) -> list[tuple[int, float, float]]:
    """(depth, seconds, expected score for the side to move) for each depth finished."""
    engine = _W["engine"]
    engine._engine.configure({"Clear Hash": None})
    marks: list[tuple[int, float, float]] = []
    t0 = time.perf_counter()

    def on_depth(found: list) -> None:
        if found and (not marks or found[0].depth > marks[-1][0]):
            marks.append(
                (found[0].depth, (time.perf_counter() - t0) / SIX_AT_ONCE, found[0].expected_score)
            )

    lines = engine.analyse(board, depth=depth, multipv=1, on_depth=on_depth)
    if lines and (not marks or marks[-1][0] < lines[0].depth):
        marks.append((lines[0].depth, (time.perf_counter() - t0) / SIX_AT_ONCE, lines[0].expected_score))
    return marks


def _position(fen: str) -> dict[str, Any]:
    board = chess.Board(fen)
    marks = _search(board, 18)
    out: dict[str, Any] = {"fen": fen, "marks": marks, "children": {}}
    fires = any(_taken(marks, b) is not None for b in BUDGETS)
    if fires:
        for move in board.legal_moves:
            child = board.copy()
            child.push(move)
            if child.is_game_over():
                value = 0.5 if not child.is_checkmate() else 1.0  # mover's: mate is a win
            else:
                cm = _search(child, 17)
                value = 1.0 - cm[-1][2] if cm else 0.5
            out["children"][move.uci()] = value
    return out


def _taken(marks: list[tuple[int, float, float]], budget: float) -> tuple[int, float] | None:
    """The depth and score the rule takes in place of depth 18, or None when it does not fire."""
    if not marks or marks[-1][1] <= budget:
        return None  # depth 18 came within the budget: nothing to stand in for
    done = [m for m in marks if m[1] <= budget]
    if not done:
        return None
    depth, _, score = done[-1]
    if DECIDED[0] < score < DECIDED[1]:
        return None
    return depth, score


def main() -> None:
    from tqdm import tqdm

    found = json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    ranked = sorted(found, key=lambda r: -r["one"])
    slowest = ranked[:300]
    print(
        f"the 300 slowest of {len(found)}: every position slower than {ranked[299]['one']:.2f}s "
        f"at depth 18 is among them"
    )
    fens = [" ".join(r["key"].split(" ")[:6]) for r in slowest]
    out = []
    pool = ProcessPoolExecutor(len(CORES), initializer=_worker_init, initargs=(CORES,))
    try:
        futures = [pool.submit(_position, f) for f in fens]
        for fut in tqdm(as_completed(futures), total=len(futures), file=sys.stdout, mininterval=2):
            out.append(fut.result())
    finally:
        processes = list(pool._processes.values()) if pool._processes else []
        pool.shutdown(wait=False, cancel_futures=True)
        for p in processes:
            p.terminate()
    OUT.write_text(json.dumps(out), encoding="utf-8")
    th = Thresholds()
    for budget in BUDGETS:
        fired = changed = judged = 0
        worst = 0.0
        waits = []
        for r in out:
            marks = r["marks"]
            taken = _taken(marks, budget)
            final = marks[-1][2]
            if taken is None:
                waits.append(marks[-1][1])
                continue
            fired += 1
            waits.append(budget)
            worst = max(worst, abs(taken[1] - final))
            for value in r["children"].values():
                judged += 1
                if severity(max(0.0, final - value), th) != severity(max(0.0, taken[1] - value), th):
                    changed += 1
        waits.sort()
        print(
            f"budget {budget}s: fires on {fired}/300; score moved at most {worst:.3f}; "
            f"severities changed {changed}/{judged}; wait over these 300 then max "
            f"{waits[-1]:.1f}s p99 {waits[int(.99 * len(waits))]:.1f}s"
        )


if __name__ == "__main__":
    main()
