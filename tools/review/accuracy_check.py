"""Our game accuracy against chess.com's own (CAPS2) on games both of us reviewed
(docs/game-review-plan.md, phase 0): per game and side, chess.com's number from the official
API's `accuracies`, and ours in several forms -- as the app computes it today, without book
moves, Lichess's published formula, and our curve at other steepnesses -- with the mean error,
the bias and the correlation of each.

    uv run --no-sync python tools/review/accuracy_check.py data/review/runs/nosegazm
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path


def move_accuracy(loss: float, k: float) -> float:
    """Our per-move curve: Lichess's shape, `k` its steepness per unit of expected score (ours
    is 25: 3x Lichess's 0.04354 per win% point x 100 / ... as score.move_accuracy)."""
    return min(100.0, max(0.0, 103.1668 * math.exp(-k * loss) - 3.1669))


def lichess_game(moves: list[dict]) -> float | None:
    """Lichess's game accuracy (lila AccuracyPercent): per-move accuracy from the win% drop
    (+1 uncertainty bonus, 0..100), then the mean of a volatility-weighted mean (windows of
    plies/10 in 2..8, weight = the std of win% in the window, 0.5..12) and the harmonic mean."""
    if not moves:
        return None
    wins = [m["before"] * 100 for m in moves]
    accs = []
    for m in moves:
        drop = max(0.0, m["loss"] * 100)
        a = 103.1668 * math.exp(-0.04354 * drop) - 3.1669 + 1
        accs.append(min(100.0, max(0.0, a)))
    size = min(8, max(2, len(moves) // 10))
    weights = []
    for i in range(len(moves)):
        window = wins[max(0, i - size + 1) : i + 1] if i >= size - 1 else wins[:size]
        sd = statistics.pstdev(window) if len(window) > 1 else 0.0
        weights.append(min(12.0, max(0.5, sd)))
    weighted = sum(a * w for a, w in zip(accs, weights, strict=True)) / sum(weights)
    harmonic = len(accs) / sum(1 / max(a, 1e-6) for a in accs)
    return (weighted + harmonic) / 2


def sides(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    headers = data["headers"]
    out = []
    for colour, parity in (("White", 1), ("Black", 0)):
        theirs = headers.get(f"{colour}Accuracy")
        if theirs in (None, "None", ""):
            continue
        moves = [
            {
                "loss": m["loss"],
                "before": m["before_score"],
                "label": m["label"],
                "acc": m["accuracy"],
            }
            for m in data["moves"]
            if m["ply"] % 2 == parity
        ]
        if len(moves) < 5:
            continue
        out.append(
            {
                "game": path.stem,
                "side": colour,
                "theirs": float(theirs),
                "moves": moves,
                "elo": headers.get(f"{colour}Elo"),
                "time": headers.get("TimeClass"),
            }
        )
    return out


def report(name: str, pairs: list[tuple[float, float]]) -> None:
    diffs = [ours - theirs for ours, theirs in pairs]
    mae = statistics.fmean(abs(d) for d in diffs)
    bias = statistics.fmean(diffs)
    xs, ys = zip(*pairs, strict=True)
    r = statistics.correlation(xs, ys) if len(pairs) > 2 else float("nan")
    within5 = sum(abs(d) <= 5 for d in diffs) / len(diffs)
    print(
        f"{name:34} n={len(pairs):3}  MAE {mae:5.1f}  bias {bias:+5.1f}  r {r:.3f}"
        f"  within 5: {within5:.0%}"
    )


def main() -> None:
    rows = [s for p in sorted(Path(sys.argv[1]).glob("*.json")) for s in sides(p)]
    print(f"{len(rows)} game-sides from {sys.argv[1]}")
    report(
        "ours (app today, k=25)",
        [(statistics.fmean(m["acc"] for m in r["moves"]), r["theirs"]) for r in rows],
    )
    no_book = [
        (statistics.fmean(m["acc"] for m in r["moves"] if m["label"] != "book"), r["theirs"])
        for r in rows
        if any(m["label"] != "book" for m in r["moves"])
    ]
    report("ours without book moves", no_book)
    lich = [(lichess_game(r["moves"]), r["theirs"]) for r in rows]
    report("Lichess formula", [(a, t) for a, t in lich if a is not None])
    for k in (8, 12, 16, 20, 25, 30):
        report(
            f"our curve k={k}, no book",
            [
                (
                    statistics.fmean(
                        move_accuracy(m["loss"], k) for m in r["moves"] if m["label"] != "book"
                    ),
                    r["theirs"],
                )
                for r in rows
                if any(m["label"] != "book" for m in r["moves"])
            ],
        )
    # The worst disagreements, to read by hand.
    worst = sorted(
        rows, key=lambda r: -abs(statistics.fmean(m["acc"] for m in r["moves"]) - r["theirs"])
    )[:8]
    for r in worst:
        ours = statistics.fmean(m["acc"] for m in r["moves"])
        print(
            f"   {r['game']} {r['side']:5} {r['time']:6} elo {r['elo']:>4}"
            f"  theirs {r['theirs']:5.1f}  ours {ours:5.1f}  moves {len(r['moves'])}"
        )


if __name__ == "__main__":
    main()
