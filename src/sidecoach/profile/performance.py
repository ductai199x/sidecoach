"""The rating a game was played at, from its moves alone (the players' ratings are never read:
chess.com's Game Rating is criticised for echoing them).

Each move falls in a bucket by what it cost (best, a sliver, a little, an inaccuracy, a mistake,
a blunder: the coach's own lines), and how likely each bucket is was fitted on real players'
games at known ratings (Lichess rapid, 800 to 2300 and up; `tools/review/fit_performance.py`)
as a softmax whose logits are quadratic in the rating, one per context -- the opening or after
it, and whether the mover was lost, level or won, since a lost position forgives what a level
one punishes. A side's moves then say how likely each rating on a grid is to have played them,
and the estimate is the mean over the grid, weighted by that and a broad prior.

One game says little: a player's accuracy wanders ten points from game to game, and a rating
band's is a point or two above the band below's. On players the fit never saw the estimate is
about 300 from their rating (docs/game-review.md, phase 0), and it leans to the middle.
Reading the opponent's moves too (rated opponents are paired close) took that only from 298 to
290, and misleads in a game against a bot, so each side is read alone.

The model's scale is Lichess rapid; `chesscom` converts to chess.com's.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from sidecoach.analysis.score import DEFAULT_THRESHOLDS

MODEL_PATH = Path(__file__).with_name("performance.json")

# The loss lines of the buckets, in expected score: below each line is the bucket before it.
EDGES = (
    1e-9,
    0.02,
    DEFAULT_THRESHOLDS.inaccuracy,
    DEFAULT_THRESHOLDS.mistake,
    DEFAULT_THRESHOLDS.blunder,
)
BUCKETS = len(EDGES) + 1
OPENING_PLIES = 20
LOST, WON = 0.15, 0.85
CONTEXTS = 6
CENTRE, SPREAD = 1500.0, 500.0
GRID = np.arange(400.0, 3001.0, 5.0)
PRIOR = (1500.0, 400.0)
# Fewer of a side's moves than this say too little to name a rating.
MIN_MOVES = 8


def plies_before(fen: str) -> int:
    """The half-moves played before a position, by its FEN's move number and side to move: a
    game set up from a position starts where the position says, not at the opening (the review
    of 2026-10-01: such a game's moves counted as the opening's for the rating)."""
    fields = fen.split()
    try:
        number = int(fields[5]) if len(fields) > 5 else 1
    except ValueError:
        number = 1
    return 2 * (max(number, 1) - 1) + (1 if len(fields) > 1 and fields[1] == "b" else 0)


def bucket(loss: float) -> int:
    return bisect.bisect_right(EDGES, loss)


def context(ply: int, score: float) -> int:
    """The move's context: the opening or after it (`ply`, the half-moves before the move),
    times lost / level / won for the mover (`score`, their expected score before it)."""
    zone = 0 if score < LOST else 2 if score > WON else 1
    return (0 if ply < OPENING_PLIES else 3) + zone


def scaled(rating: float) -> float:
    return (rating - CENTRE) / SPREAD


@dataclass(frozen=True)
class Performance:
    rating: int  # Lichess rapid
    low: int  # the middle 80% of the weight
    high: int
    moves: int


@cache
def _default_table() -> np.ndarray:
    return _table(json.loads(MODEL_PATH.read_text(encoding="utf-8")))


def _table(model: dict) -> np.ndarray:
    """Each context's and bucket's log chance at every rating of the grid: (C, B, G)."""
    weights = np.asarray(model["weights"], dtype=float)
    x = (GRID - CENTRE) / SPREAD
    logits = np.einsum("cbk,kg->cbg", weights, np.stack([np.ones_like(x), x, x * x]))
    top = logits.max(axis=1, keepdims=True)
    return logits - (top + np.log(np.exp(logits - top).sum(axis=1, keepdims=True)))


def estimate(
    moves: Iterable[Mapping],
    *,
    model: dict | None = None,
    prior: tuple[float, float] | None = PRIOR,
) -> Performance:
    """The rating one side's moves were played at. `moves`: each with its ply, its loss and the
    mover's expected score before it. `prior`: (mean, spread) of a normal over the grid, or
    None. `model`: fitted weights, when not the shipped model's."""
    table = _table(model) if model is not None else _default_table()
    counts = np.zeros(table.shape[:2])
    for move in moves:
        counts[context(move["ply"], move["score"]), bucket(move["loss"])] += 1
    log_weight = np.einsum("cb,cbg->g", counts, table)
    if prior is not None:
        log_weight = log_weight - 0.5 * ((GRID - prior[0]) / prior[1]) ** 2
    mass = np.exp(log_weight - log_weight.max())
    mass /= mass.sum()
    cumulative = np.cumsum(mass)
    return Performance(
        rating=round(float((GRID * mass).sum())),
        low=int(GRID[np.searchsorted(cumulative, 0.1)]),
        high=int(GRID[np.searchsorted(cumulative, 0.9)]),
        moves=int(counts.sum()),
    )


# Lichess rapid against chess.com rapid for the same players (ChessGoals' survey, July 2026):
# (chess.com, Lichess), a straight line between the points and the end segments carried on.
CHESSCOM_LICHESS = (
    (815, 1290),
    (1000, 1430),
    (1255, 1615),
    (1500, 1795),
    (1995, 2165),
    (2430, 2615),
)


def chesscom(lichess: float) -> int:
    """A Lichess rapid rating on chess.com's rapid scale, no lower than its floor of 100."""
    points = CHESSCOM_LICHESS
    i = bisect.bisect_left([p[1] for p in points], lichess)
    i = min(max(i, 1), len(points) - 1)
    (c0, l0), (c1, l1) = points[i - 1], points[i]
    return max(100, round(c0 + (lichess - l0) * (c1 - c0) / (l1 - l0)))


def lichess(chesscom_rating: float) -> float:
    """A chess.com rapid rating on Lichess's rapid scale: `CHESSCOM_LICHESS` read the other way."""
    points = CHESSCOM_LICHESS
    i = bisect.bisect_left([p[0] for p in points], chesscom_rating)
    i = min(max(i, 1), len(points) - 1)
    (c0, l0), (c1, l1) = points[i - 1], points[i]
    return l0 + (chesscom_rating - c0) * (l1 - l0) / (c1 - c0)


def for_site(lichess: float, site: str | None) -> int:
    """The rating on the scale of the site the game was played on: Lichess's for a Lichess
    game, chess.com's otherwise (a game Sidecoach watched is on chess.com)."""
    return round(lichess) if site and "lichess" in site.lower() else chesscom(lichess)
