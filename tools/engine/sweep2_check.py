"""sweep2's clock checked by hand: tiny games whose every answer is worked out on paper. One
worker, no look-ahead, no candidates, the user White.

A. The start (U): its one line 0-0.2, its six lines to depth 12 0.2-0.3 (hint 0.3), its
   threats 0.3-0.5 (whole 0.5). 1.e4 at 10, the start's best: judged at once. The position
   after it (O): one line 10-10.2. 1...e5 at 11, that position's best: judged at once. Its
   position (U): one line 11-11.2, six lines 11.2-11.3 (hint 0.3), threats 11.3-11.5 (whole
   0.5). CPU 1.2.
B. As A, but 1.a3 at 10, not the start's best: judged when the position after it has its one
   line (10-10.3): 0.3.
C. As B, the position after 1.a3 taking 5 s for its one line but decided at 1 s: judged at 1.

    uv run --no-sync python tools/engine/sweep2_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import cache_key  # noqa: E402
from tools.engine.simulate import Step  # noqa: E402
from tools.engine.sweep2 import Atom, Policy, Run  # noqa: E402


def line(uci: str) -> dict:
    return {"moves": [uci], "depth": 18, "expected_score": 0.5}


def check(title: str, run: dict, want: dict) -> bool:
    ok = True
    print(title)
    for key, value in want.items():
        got = run[key]
        if isinstance(value, list):
            same = len(got) == len(value) and all(abs(a - b) < 1e-9 for a, b in zip(got, value, strict=True))
        else:
            same = abs(got - value) < 1e-9
        print(f"  {key}: {got} (by hand {value}) {'ok' if same else 'WRONG'}")
        ok &= same
    return ok


def main() -> None:
    start = chess.Board()
    e4, a3 = start.copy(), start.copy()
    e4.push_san("e4")
    a3.push_san("a3")
    e4e5, a3e5 = e4.copy(), a3.copy()
    e4e5.push_san("e5")
    a3e5.push_san("e5")
    policy = Policy(1, 0, 0, 0, 0.15, "newest")

    def atoms(slow_a3: bool = False) -> dict:
        return {
            cache_key(start): Atom(0.2, line("e2e4"), None, 0.1, [line("e2e4")], 0.2),
            cache_key(e4): Atom(0.2, line("e7e5"), None, 0.1, [line("e7e5")], 0.2),
            cache_key(a3): Atom(5.0 if slow_a3 else 0.3, line("e7e5"), 1.0 if slow_a3 else None, 0.1, [line("e7e5")], 0.2),
            cache_key(e4e5): Atom(0.2, line("g1f3"), None, 0.1, [line("g1f3")], 0.2),
            cache_key(a3e5): Atom(0.2, line("g1f3"), None, 0.1, [line("g1f3")], 0.2),
        }

    def game(first: chess.Board, move: str) -> list:
        return [
            Step(0.0, "start", start.copy(), game_id="g", user=chess.WHITE),
            Step(10.0, "move", start.copy(), chess.Move.from_uci(move), 1, True, game_id="g", user=chess.WHITE),
            Step(11.0, "move", first.copy(), chess.Move.from_uci("e7e5"), 2, False, game_id="g", user=chess.WHITE),
        ]

    ok = check(
        "A, the best moves:",
        Run(policy, atoms(), game(e4, "e2e4")).run(),
        {"verdict_user": [0.0], "verdict_opp": [0.0], "hint": [0.3, 0.3], "deep": [0.5, 0.5], "cpu": 1.2},
    )
    ok &= check(
        "B, 1.a3, not the best:",
        Run(policy, atoms(), game(a3, "a2a3")).run(),
        {"verdict_user": [0.3], "verdict_opp": [0.0]},
    )
    ok &= check(
        "C, the position after 1.a3 slow but decided at 1 s:",
        Run(policy, atoms(slow_a3=True), game(a3, "a2a3")).run(),
        {"verdict_user": [1.0]},
    )
    print("the clock agrees with the hand" if ok else "THE CLOCK DISAGREES")


if __name__ == "__main__":
    main()
