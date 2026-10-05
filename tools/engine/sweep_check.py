"""The sweep's clock checked by hand: a tiny game whose every answer is worked out on paper.

One worker, no look-ahead, the user White. Searches (seconds): the start, six lines 1.0 (depth 12
at 0.1), threats 0.2; after 1.e4, two lines 0.4; after 1...e5, six lines 1.0 (depth 12 at 0.1),
threats 0.2. 1.e4 at 10 s (among the start's lines), 1...e5 at 11 s (among the two lines).

By hand: the start told first at 0.1 and whole at 1.2; 1.e4 judged at once (0); the two-line
search of the position after it runs 10-10.4; 1...e5 judged at 11 at once (its line is there);
the position after it first at 11.1 and whole at 12.2. CPU 1.0+0.2+0.4+1.0+0.2 = 2.8.

    uv run --no-sync python tools/engine/sweep_check.py
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
from tools.engine.sweep import BESIDE, Atom, Policy, Run  # noqa: E402


def line(uci: str) -> dict:
    return {"moves": [uci], "depth": 18, "expected_score": 0.5}


def main() -> None:
    start = chess.Board()
    e4 = start.copy()
    e4.push_san("e4")
    e5 = e4.copy()
    e5.push_san("e5")
    atoms = {
        cache_key(start): Atom(0.3, 1.0, 0.1, 0.2, [line("e2e4")], [line("e2e4"), line("d2d4")]),
        cache_key(e4): Atom(0.4, 1.3, 0.1, 0.2, [line("e7e5"), line("c7c5")], [line("e7e5")]),
        cache_key(e5): Atom(0.3, 1.0, 0.1, 0.2, [line("g1f3")], [line("g1f3")]),
    }
    steps = [
        Step(0.0, "start", start.copy(), game_id="g", user=chess.WHITE),
        Step(10.0, "move", start.copy(), chess.Move.from_uci("e2e4"), 1, True, game_id="g", user=chess.WHITE),
        Step(11.0, "move", e4.copy(), chess.Move.from_uci("e7e5"), 2, False, game_id="g", user=chess.WHITE),
    ]
    assert BESIDE[1] == 1.0
    run = Run(Policy(1, 0, False, "2", 0, 0.15), atoms, steps).run()
    want = {
        "verdict_user": [0.0],
        "verdict_opp": [0.0],
        "first": [0.1, 0.1],
        "whole": [1.2, 1.2],
        "cpu": 2.8,
    }
    ok = True
    for key, value in want.items():
        got = run[key]
        same = (
            abs(got - value) < 1e-9
            if isinstance(value, float)
            else len(got) == len(value) and all(abs(a - b) < 1e-9 for a, b in zip(got, value, strict=True))
        )
        print(f"  {key}: {got} (by hand {value}) {'ok' if same else 'WRONG'}")
        ok &= same
    ok &= second()
    ok &= third()
    print("the clock agrees with the hand" if ok else "THE CLOCK DISAGREES")


def compare(title: str, run: dict, want: dict) -> bool:
    ok = True
    print(title)
    for key, value in want.items():
        got = run[key]
        if isinstance(value, list):
            same = len(got) == len(value) and all(
                abs(a - b) < 1e-9 for a, b in zip(got, value, strict=True)
            )
        else:
            same = abs(got - value) < 1e-9
        print(f"  {key}: {got} (by hand {value}) {'ok' if same else 'WRONG'}")
        ok &= same
    return ok


def third() -> bool:
    """Moves faster than the searches: one worker, no look-ahead. The start's six-line search
    runs 0-1.0 (depth 12 at 0.1). 1.e4 at 0.5, before it ends: its verdict waits for it (0.5);
    the start is left without its threats (left whole: 1). 1...e5 at 0.7, the worker busy until
    1.0. Then, the newest first: its six lines 1.0-2.0 (first at 1.1: 0.4), its threats 2.0-2.2
    (whole 1.5), the position after 1.e4 (two lines 2.2-2.6; 1...e5 judged: 1.9), the start's
    threats (2.6-2.8). The verdicts first: the position after 1.e4 1.0-1.4 (1...e5 judged: 0.7),
    the newest's six lines 1.4-2.4 (first at 1.5: 0.8), its threats 2.4-2.6 (whole 1.9), the
    start's 2.6-2.8. CPU 2.8 both ways."""
    start = chess.Board()
    e4 = start.copy()
    e4.push_san("e4")
    e5 = e4.copy()
    e5.push_san("e5")
    atoms = {
        cache_key(start): Atom(0.3, 1.0, 0.1, 0.2, [line("e2e4")], [line("e2e4")]),
        cache_key(e4): Atom(0.4, 1.0, 0.1, 0.2, [line("e7e5")], [line("e7e5")]),
        cache_key(e5): Atom(0.3, 1.0, 0.1, 0.2, [line("g1f3")], [line("g1f3")]),
    }
    steps = [
        Step(0.0, "start", start.copy(), game_id="g", user=chess.WHITE),
        Step(0.5, "move", start.copy(), chess.Move.from_uci("e2e4"), 1, True, game_id="g", user=chess.WHITE),
        Step(0.7, "move", e4.copy(), chess.Move.from_uci("e7e5"), 2, False, game_id="g", user=chess.WHITE),
    ]
    ok = True
    for order, opp, first, whole in (("newest", 1.9, 0.4, 1.5), ("verdicts", 0.7, 0.8, 1.9)):
        run = Run(Policy(1, 0, False, "2", 0, 0.15, order), atoms, steps).run()
        ok &= compare(
            f"third game, {order} first:",
            run,
            {
                "verdict_user": [0.5],
                "verdict_opp": [opp],
                "first": [0.1, first],
                "whole": [whole],
                "left_first": 0,
                "left_whole": 1,
                "cpu": 2.8,
            },
        )
    return ok


def second() -> bool:
    """Two workers, one kept for the game; a look-ahead of one ply. Every search 1.05 times its
    time (two at once). The start (six lines 1.0, depth 12 at 0.1, threats 0.2) has 1.e4 (0.60)
    and 1.d4 (0.58): both likely enough, so the positions after them are searched ahead, one at
    a time on the look-ahead's worker (two lines, 0.4 each): 1.05-1.47 and 1.47-1.89. The user
    plays 1.a3 at 10 -- among no line: its verdict waits for the position after it (two lines,
    10-10.42): 0.42. That position's line 1...e5 is likely: the position after it is searched
    ahead at once (six lines, 10.42-11.47, depth 12 at 10.525). 1...e5 at 11 is among the lines:
    judged at once; its position is the search under way, promoted -- its first answer came
    before it was reached (0), whole when its threats are in (11.47-11.68): 0.68. 1.e4 and 1.d4
    searched ahead in vain: 0.84 thrown away of 3.78."""
    start = chess.Board()
    e4, d4, a3 = start.copy(), start.copy(), start.copy()
    e4.push_san("e4")
    d4.push_san("d4")
    a3.push_san("a3")
    a3e5 = a3.copy()
    a3e5.push_san("e5")

    def scored(uci: str, score: float) -> dict:
        return {"moves": [uci], "depth": 18, "expected_score": score}

    atoms = {
        cache_key(start): Atom(0.3, 1.0, 0.1, 0.2, [], [scored("e2e4", 0.60), scored("d2d4", 0.58)]),
        cache_key(e4): Atom(0.4, 1.0, 0.1, 0.2, [scored("e7e5", 0.5)], []),
        cache_key(d4): Atom(0.4, 1.0, 0.1, 0.2, [scored("d7d5", 0.5)], []),
        cache_key(a3): Atom(0.4, 1.0, 0.1, 0.2, [scored("e7e5", 0.5)], []),
        cache_key(a3e5): Atom(0.3, 1.0, 0.1, 0.2, [], [scored("g1f3", 0.5)]),
    }
    steps = [
        Step(0.0, "start", start.copy(), game_id="g", user=chess.WHITE),
        Step(10.0, "move", start.copy(), chess.Move.from_uci("a2a3"), 1, True, game_id="g", user=chess.WHITE),
        Step(11.0, "move", a3.copy(), chess.Move.from_uci("e7e5"), 2, False, game_id="g", user=chess.WHITE),
    ]
    f = BESIDE[2]
    run = Run(Policy(2, 1, False, "2", 1, 0.15), atoms, steps).run()
    want = {
        "verdict_user": [0.4 * f],
        "verdict_opp": [0.0],
        "first": [0.1 * f, 0.0],
        "whole": [1.2 * f, 10.42 + 1.2 * f - 11.0],
        "cpu": 3.6 * f,
        "wasted": 0.8 * f,
    }
    ok = True
    print("second game:")
    for key, value in want.items():
        got = run[key]
        same = (
            abs(got - value) < 1e-9
            if isinstance(value, float)
            else len(got) == len(value) and all(abs(a - b) < 1e-9 for a, b in zip(got, value, strict=True))
        )
        print(f"  {key}: {got} (by hand {value}) {'ok' if same else 'WRONG'}")
        ok &= same
    return ok


if __name__ == "__main__":
    main()
