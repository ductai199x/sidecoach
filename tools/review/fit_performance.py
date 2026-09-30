"""The performance rating fitted on real players' games and tested on players it never saw
(docs/game-review-plan.md, phase 0). Reads the game sides `human_losses.py` judged, splits them
by player (a fixed hash: the same split every run), fits on one part and reports on the rest,
per band: how far the estimate is from the player's rating (mean absolute error), which way it
leans, and the correlation -- for a straight line through the side's accuracy (the baseline)
and the model under a few priors.

    uv run --no-sync python tools/review/fit_performance.py data/review/human/pilot.jsonl \
        --held data/review/human/paired.jsonl --runs data/review/runs/nosegazm

`--held`: more sides tested whole, their players kept out of the fit (the games the sample
holds both sides of, `human_losses.py --paired`). `--runs`: games reviewed by `sidecoach review
--out`, the player's chess.com games against people, on chess.com's scale. `--write` saves the
fitted model where the review reads it (`sidecoach/profile/performance.json`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from sidecoach.profile import performance

PRIORS: tuple[tuple[float, float] | None, ...] = ((1500.0, 400.0), (1500.0, 500.0), None)


def sides(paths: list[Path]) -> list[dict]:
    out = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["moves"]:
                out.append(row)
    return out


def reviewed(folder: Path) -> list[dict]:
    """The sides of games reviewed by `sidecoach review` (chess.com games through the official
    API) against people, shaped as the sample's: the side's chess.com rating, banded by 200,
    and its moves with the half-moves before each (the review counts White's first move as
    ply 1)."""
    out = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["headers"].get("Event") != "Live Chess":
            continue  # a game against a bot or the coach: its ratings are the bot's setting
        for colour, parity in (("White", 1), ("Black", 0)):
            elo = data["headers"].get(f"{colour}Elo", "")
            moves = [
                {
                    "ply": m["ply"] - 1,
                    "loss": m["loss"],
                    "score": m["before_score"],
                    "accuracy": m["accuracy"],
                }
                for m in data["moves"]
                if m["ply"] % 2 == parity
            ]
            if elo.isdigit() and len(moves) >= performance.MIN_MOVES:
                low = int(elo) // 200 * 200
                out.append({"band": f"{low}-{low + 200}", "rating": int(elo), "moves": moves})
    return out


def held_out(row: dict, share: float) -> bool:
    """Whether the side's player is in the test part: a hash of the player, so a player's
    games are all on one side of the split, and the split never changes."""
    digest = hashlib.sha1(str(row["player"]).encode()).digest()
    return digest[0] / 256 < share


def design(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every move as (context, bucket, the mover's rating as the model scales it)."""
    contexts, buckets, ratings = [], [], []
    for row in rows:
        for move in row["moves"]:
            contexts.append(performance.context(move["ply"], move["score"]))
            buckets.append(performance.bucket(move["loss"]))
            ratings.append(performance.scaled(row["rating"]))
    return np.array(contexts), np.array(buckets), np.array(ratings)


def fit(rows: list[dict], *, steps: int = 3000, rate: float = 0.05, ridge: float = 1e-3) -> dict:
    """The softmax over buckets per context, its logits quadratic in the scaled rating, fitted
    by full-batch Adam on the mean log-likelihood with a light ridge."""
    contexts, buckets, ratings = design(rows)
    n_ctx, n_b = performance.CONTEXTS, performance.BUCKETS
    powers = np.stack([np.ones_like(ratings), ratings, ratings**2], axis=1)  # (N, 3)
    onehot = np.zeros((len(buckets), n_b))
    onehot[np.arange(len(buckets)), buckets] = 1.0
    weights = np.zeros((n_ctx, n_b, 3))
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    probs = onehot
    for step in range(1, steps + 1):
        logits = np.einsum("nbk,nk->nb", weights[contexts], powers)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        error = probs - onehot  # d(-loglik)/d(logit), per move
        grad = np.zeros_like(weights)
        np.add.at(grad, contexts, np.einsum("nb,nk->nbk", error, powers))
        grad = grad / len(buckets) + ridge * weights
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        weights -= rate * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    nll = -np.log(probs[np.arange(len(buckets)), buckets] + 1e-12).mean()
    print(f"  fitted on {len(buckets)} moves: mean -loglik {nll:.4f}", flush=True)
    return {"weights": weights.round(5).tolist()}


def estimators(model: dict, train: list[dict]) -> dict[str, Callable[[list], float]]:
    """The ways a side's rating is estimated from its moves (Lichess rapid), by name."""
    x = np.array([np.mean([m["accuracy"] for m in r["moves"]]) for r in train])
    slope, intercept = np.polyfit(x, np.array([r["rating"] for r in train], dtype=float), 1)
    out: dict[str, Callable[[list], float]] = {
        "accuracy line": lambda s: float(slope * np.mean([m["accuracy"] for m in s]) + intercept)
    }
    for prior in PRIORS:
        name = f"model, prior {f'{prior[0]:.0f}±{prior[1]:.0f}' if prior else 'none'}"
        out[name] = lambda s, p=prior: performance.estimate(s, model=model, prior=p).rating
    return out


def report(name: str, rows: list[dict], estimates: Sequence[float]) -> None:
    by_band: dict[str, list[float]] = defaultdict(list)
    for row, estimate in zip(rows, estimates, strict=True):
        by_band[row["band"]].append(estimate - row["rating"])
    print(f"{name}:")
    everything = []
    for band in sorted(by_band, key=lambda b: int(b.split("-")[0])):
        errors = np.array(by_band[band])
        everything.extend(errors)
        print(
            f"  {band:>10}  n {len(errors):4d}  MAE {np.abs(errors).mean():5.0f}"
            f"  lean {errors.mean():+5.0f}"
        )
    errors = np.array(everything)
    truth = np.array([r["rating"] for r in rows])
    r = np.corrcoef(truth, np.array(estimates))[0, 1]
    print(f"  {'all':>10}  n {len(errors):4d}  MAE {np.abs(errors).mean():5.0f}  r {r:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--test-share", type=float, default=0.3)
    parser.add_argument("--held", type=Path)
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    held = sides([args.held]) if args.held else []
    kept_out = {r["player"] for r in held}
    rows = [r for r in sides(args.jsonl) if r["player"] not in kept_out]
    train = [r for r in rows if not held_out(r, args.test_share)]
    test = [r for r in rows if held_out(r, args.test_share)] + held
    print(f"{len(train)} sides to fit, {len(test)} to test")
    model = fit(train)
    ways = estimators(model, train)
    print("\n== held-out players (Lichess rapid)")
    for name, way in ways.items():
        report(name, test, [way(r["moves"]) for r in test])
    played = reviewed(args.runs) if args.runs else []
    if played:
        print("\n== the player's chess.com games (chess.com rapid)")
        for name, way in ways.items():
            report(name, played, [performance.chesscom(way(r["moves"])) for r in played])
    if args.write:
        performance.MODEL_PATH.write_text(json.dumps(model) + "\n", encoding="utf-8")
        print(f"written {performance.MODEL_PATH}")


if __name__ == "__main__":
    main()
