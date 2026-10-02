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


# Lichess's win% curve (scalachess WinPercent.fromCentiPawns): ours, with the centipawns
# capped at 1000 and a mate counted as the cap.
_LICHESS_K = 0.00368208
_CEILING = 100 / (1 + math.exp(-_LICHESS_K * 1000))
_INITIAL = 100 / (1 + math.exp(-_LICHESS_K * 15))  # Cp.initial: 15 centipawns


def _win_percent(score: float) -> float:
    """An expected score as Lichess's win%, mates at the 1000-centipawn ceiling."""
    return min(_CEILING, max(100 - _CEILING, score * 100))


def _lichess_move(before: float, after: float) -> float:
    """lila AccuracyPercent.fromWinPercents: 100 when the mover lost nothing."""
    if after >= before:
        return 100.0
    raw = 103.1668100711649 * math.exp(-0.04354415386753951 * (before - after)) - 3.166924740191411
    return min(100.0, max(0.0, raw + 1))


def lichess_game(game: list[dict]) -> dict[int, float]:
    """Lichess's game accuracy for each side (lila AccuracyPercent.gameAccuracy, read
    2026-10-02), from a whole game's moves in order (ply, before_score, loss): White's win%
    from the initial 15 centipawns through the position after every move; windows of
    plies/10 win% values (2 to 8), the first ones padded with the first window; each move
    weighted by its window's population standard deviation (0.5 to 12); per side the mean of
    the weighted mean and the harmonic mean (each accuracy floored at 1). The position after
    the last move is its mover's score less its loss. Keyed by parity: 1 White, 0 Black."""
    if not game:
        return {}
    whites = [_INITIAL]
    for i, m in enumerate(game):
        white_moves = m["ply"] % 2 == 1
        if i + 1 < len(game):
            after = 1.0 - game[i + 1]["before"]  # the next mover's score, turned round
        else:
            after = max(0.0, m["before"] - m["loss"])
        whites.append(_win_percent(after if white_moves else 1.0 - after))
    size = min(8, max(2, len(game) // 10))
    windows = [whites[:size]] * (min(size, len(whites)) - 2) + [
        whites[i : i + size] for i in range(max(1, len(whites) - size + 1))
    ]
    weights = [min(12.0, max(0.5, statistics.pstdev(w))) for w in windows]
    per: dict[int, list[tuple[float, float]]] = {1: [], 0: []}
    for i, (m, weight) in enumerate(zip(game, weights, strict=False)):
        prev, nxt = whites[i], whites[i + 1]
        white_moves = m["ply"] % 2 == 1
        accuracy = _lichess_move(prev, nxt) if white_moves else _lichess_move(100 - prev, 100 - nxt)
        per[m["ply"] % 2].append((accuracy, weight))
    out = {}
    for side, pairs in per.items():
        if not pairs:
            continue
        weighted = sum(a * w for a, w in pairs) / sum(w for _, w in pairs)
        harmonic = len(pairs) / sum(1 / max(1.0, a) for a, _ in pairs)
        out[side] = (weighted + harmonic) / 2
    return out


def sides(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    headers = data["headers"]
    game = [
        {"ply": m["ply"], "before": m["before_score"], "loss": m["loss"]}
        for m in sorted(data["moves"], key=lambda m: m["ply"])
    ]
    lichess = lichess_game(game)
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
                "lichess": lichess.get(parity),
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
    report("Lichess formula", [(r["lichess"], r["theirs"]) for r in rows if r["lichess"]])
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
