"""How far Nova's moves are from real players' of the same rating, on the same positions.

The bots are for making realistic amateur games (tests for the coach, simulated students), so
their mistakes must look like real players' at the rating they claim. The yardstick is the
600,000-position sample the Nova authors published (Lichess rapid, March 2026, 100,000 per
rating band, each with the move the human played; `data/models/nova/unified_sample_600k.jsonl`,
converted from their pickle after its opcodes were checked to import only pandas and numpy).

For positions of each band (ply 10 on), Stockfish measures what the human's move cost, in
expected score (the project's measure; never centipawns), and what Nova's moves cost when it
plays as that same player -- its rating, neutral style -- sampled at each temperature. The
report sets the two side by side: the mean loss and the share of inaccuracies, mistakes and
blunders by the coach's own thresholds (`analysis.score.Thresholds`), so a bot's setting can
be chosen to match the band.

    uv run --no-sync python tools/bots/calibrate.py --per-band 600 --workers 8

A level below the sample's bands (the students' 500) is played on the lowest band's
positions as a fixed rating, hotter than it, and set against a target extrapolated from the
bands (there is no human data under 800):

    uv run --no-sync python tools/bots/calibrate.py --bands 800-1100 --as-rating 500 \
        --temperatures 1.0,1.2,1.4,1.6 --workers 22
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess  # noqa: E402

SAMPLE = ROOT / "data" / "models" / "nova" / "unified_sample_600k.jsonl"
OUT = ROOT / "data" / "bots" / "calibration"
BANDS = ["800-1100", "1100-1400", "1400-1700", "1700-2000", "2000-2300", "2300-4000"]
TEMPERATURES = (0.5, 0.7, 0.85, 1.0)
SAMPLES = 4  # Nova moves drawn per position and temperature
DEPTH = 12

_engine = None
_policy = None


def _init() -> None:
    global _engine, _policy
    from nova import NovaPolicy

    from sidecoach.analysis.engine import Engine
    from sidecoach.config import find_stockfish, load_settings

    settings = load_settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    _engine = Engine(
        path=path,
        threads=1,
        hash_mb=32,
        multipv=1,
        pv_plies=2,
        score_model=settings.engine.score_model,
    )
    _engine.open()
    _policy = NovaPolicy()


def _score(board: chess.Board, cache: dict[str, float]) -> float:
    """The position's expected score for the side to move."""
    key = board.fen()
    if key in cache:
        return cache[key]
    if board.is_checkmate():
        value = 0.0
    elif board.is_game_over(claim_draw=False):
        value = 0.5
    else:
        assert _engine is not None
        lines = _engine.analyse(board, depth=DEPTH, multipv=1)
        value = lines[0].expected_score if lines else 0.5
    cache[key] = value
    return value


def _loss(board: chess.Board, move: chess.Move, best: float, cache: dict[str, float]) -> float:
    after = board.copy(stack=False)
    after.push(move)
    return max(0.0, best - (1.0 - _score(after, cache)))


def _measure(job: tuple[int, dict, tuple[float, ...], int | None]) -> dict:
    """One position: the human's loss, and Nova's at each temperature (SAMPLES draws each),
    as the row's player or as `as_rating`."""
    from nova import Style

    seed, row, temperatures, as_rating = job
    rng = random.Random(seed)
    board = chess.Board(row["fen"])
    cache: dict[str, float] = {}
    best = _score(board, cache)
    human = chess.Move.from_uci(row["actual"])
    out = {
        "band": row["band"],
        "rating": row["rating"],
        "human": _loss(board, human, best, cache) if human in board.legal_moves else None,
        "nova": {},
    }
    assert _policy is not None
    style = Style(rating=as_rating if as_rating is not None else row["rating"])
    for t in temperatures:
        probs = _policy.probabilities([board], [style], temperature=t)[0]
        drawn = rng.choices(list(probs), weights=list(probs.values()), k=SAMPLES)
        out["nova"][str(t)] = [_loss(board, m, best, cache) for m in drawn]
    return out


def _sample(per_band: int, seed: int, bands: list[str] = BANDS) -> list[dict]:
    by_band: dict[str, list[dict]] = defaultdict(list)
    with SAMPLE.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["ply"] >= 10:
                by_band[row["band"]].append(row)
    rng = random.Random(seed)
    chosen: list[dict] = []
    for band in bands:
        rows = by_band[band]
        chosen += rng.sample(rows, min(per_band, len(rows)))
    return chosen


def _stats(losses: list[float]) -> dict[str, float]:
    from sidecoach.analysis.score import Thresholds

    th = Thresholds()
    n = len(losses) or 1
    return {
        "mean": sum(losses) / n,
        "inaccuracy+": sum(x >= th.inaccuracy for x in losses) / n,
        "mistake+": sum(x >= th.mistake for x in losses) / n,
        "blunder": sum(x >= th.blunder for x in losses) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-band", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bands", default=",".join(BANDS), help="comma-separated bands")
    parser.add_argument(
        "--temperatures", default=",".join(str(t) for t in TEMPERATURES), help="comma-separated"
    )
    parser.add_argument("--as-rating", type=int, help="Nova plays as this rating, not the row's")
    args = parser.parse_args()
    bands = [b for b in args.bands.split(",") if b]
    temperatures = tuple(float(t) for t in args.temperatures.split(",") if t)

    rows = _sample(args.per_band, args.seed, bands)
    print(f"{len(rows)} positions ({args.per_band} a band, ply >= 10), depth {DEPTH}", flush=True)
    started = time.time()
    results: list[dict] = []
    with mp.Pool(args.workers, initializer=_init) as pool:
        jobs = [
            (args.seed * 1_000_003 + i, row, temperatures, args.as_rating)
            for i, row in enumerate(rows)
        ]
        for i, r in enumerate(pool.imap_unordered(_measure, jobs, chunksize=4), 1):
            results.append(r)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}  {time.time() - started:.0f} s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT / f"{stamp}.jsonl").write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")

    print("\nexpected-score loss per move (thresholds: inaccuracy .04, mistake .10, blunder .28)")
    print(f"{'band':>10} {'who':>10} {'mean':>6} {'inacc+':>7} {'mist+':>6} {'blund':>6}")
    summary: dict[str, dict] = {}
    for band in bands:
        rs = [r for r in results if r["band"] == band]
        human = _stats([r["human"] for r in rs if r["human"] is not None])
        summary[band] = {"human": human}
        print(
            f"{band:>10} {'human':>10} {human['mean']:.3f} {human['inaccuracy+']:7.1%} "
            f"{human['mistake+']:6.1%} {human['blunder']:6.1%}"
        )
        for t in temperatures:
            nova = _stats([x for r in rs for x in r["nova"][str(t)]])
            summary[band][f"nova@{t}"] = nova
            print(
                f"{'':>10} {'nova@' + str(t):>10} {nova['mean']:.3f} {nova['inaccuracy+']:7.1%} "
                f"{nova['mistake+']:6.1%} {nova['blunder']:6.1%}"
            )
    (OUT / f"{stamp}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{time.time() - started:.0f} s; records in {OUT / (stamp + '.jsonl')}")


if __name__ == "__main__":
    main()
