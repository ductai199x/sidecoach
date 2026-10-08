"""The puzzle rating as Lichess computes it: Glicko-2, line for line (Tai, 2026-10-08: "Ofc we
want to stick to how lichess does it and we want to match exactly").

A port of Lichess's own code, read at lila f5b261e3 and the scalachess it pins (18.0.0,
2ab0a403):

- the calculator: scalachess `rating/src/main/scala/glicko/impl/RatingCalculator.scala`
  (Glickman's steps 1-8, tau 0.75, the Illinois iteration to 1e-6) and `glicko/model.scala`
  (the Glicko triple, `average`, provisional at a deviation of 110, clueless at 230);
- Lichess's bounds: lila `modules/rating/src/main/Glicko.scala` (a new player 1500 / 500 /
  0.09; the rating at least 400; the deviation 45 to 500; the volatility at most 0.1; at most
  700 points a game; the deviation grows 0.21436 rating periods a day while the player is away)
  and `Perf.scala` (`addOrReset`, the deviation as of now on every read);
- a puzzle attempt: lila `modules/puzzle/src/main/PuzzleFinisher.scala`, a game the player
  plays as White against the puzzle as Black, won when solved, whose result is then weighed by
  the puzzle's theme (`ponder.player`) -- a themed puzzle moves the rating less than a mixed
  one, an obvious theme (a mate in one) least.

Ratings are doubles and shown truncated (`intRating`, Scala's `toInt`); the change shown after
a puzzle is the difference of the two shown ratings (lila `JsonView.scala`). The theme weights
are Scala Floats: they are rounded to single precision here as the JVM computes them.

Pure functions over values; nothing here reads or writes the profile.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, replace

# ---- scalachess glicko/model.scala -----------------------------------------------------------
TAU = 0.75
PROVISIONAL_DEVIATION = 110
CLUELESS_DEVIATION = 230

# ---- scalachess glicko/impl/RatingCalculator.scala -------------------------------------------
MULTIPLIER = 173.7178
DEFAULT_RATING = 1500.0
CONVERGENCE_TOLERANCE = 0.000001
ITERATION_MAX = 1000
DAYS_PER_MILLI = 1.0 / (1000 * 60 * 60 * 24)

# ---- lila modules/rating/src/main/Glicko.scala -----------------------------------------------
MIN_RATING = 400
MAX_RATING = 4000
MIN_DEVIATION = 45
MAX_DEVIATION = 500.0
MAX_VOLATILITY = 0.1
DEFAULT_VOLATILITY = 0.09
MAX_RATING_DELTA = 700
# "chosen so a typical player's RD goes from 60 -> 110 in 1 year"
PERIODS_PER_DAY = 0.21436

# ---- lila modules/puzzle/src/main/PuzzleFinisher.scala (ponder) ------------------------------
# Themes that do not hint at the solution.
NON_HINTING_THEMES = frozenset(
    {
        "opening",
        "middlegame",
        "endgame",
        "rookEndgame",
        "bishopEndgame",
        "pawnEndgame",
        "knightEndgame",
        "queenEndgame",
        "queenRookEndgame",
        "master",
        "masterVsMaster",
        "superGM",
    }
)
# PuzzleTheme.allMates: the visible themes whose key ends in "Mate" (PuzzleTheme.scala).
MATE_THEMES = frozenset(
    {
        "anastasiaMate",
        "arabianMate",
        "backRankMate",
        "balestraMate",
        "blindSwineMate",
        "bodenMate",
        "cornerMate",
        "doubleBishopMate",
        "dovetailMate",
        "epauletteMate",
        "hookMate",
        "killBoxMate",
        "pillsburysMate",
        "morphysMate",
        "operaMate",
        "swallowstailMate",
        "triangleMate",
        "vukovicMate",
        "smotheredMate",
    }
)
# Themes that make the solution very obvious.
OBVIOUS_THEMES = (
    frozenset({"enPassant", "attackingF2F7", "doubleCheck", "mateIn1", "castling"}) | MATE_THEMES
)
MIX = "mix"


def f32(x: float) -> float:
    """`x` as a Scala Float holds it (IEEE single precision)."""
    return struct.unpack("f", struct.pack("f", x))[0]


@dataclass(frozen=True)
class Glicko:
    rating: float
    deviation: float
    volatility: float

    @property
    def int_rating(self) -> int:
        """The rating as Lichess shows it: truncated (Scala's Double.toInt)."""
        return math.trunc(self.rating)

    @property
    def provisional(self) -> bool:
        return self.deviation >= PROVISIONAL_DEVIATION

    @property
    def clueless(self) -> bool:
        return self.deviation >= CLUELESS_DEVIATION

    def display(self) -> str:
        """The rating as Lichess writes it, "1500?" while provisional."""
        return f"{self.int_rating}{'?' if self.provisional else ''}"

    def average(self, other: Glicko, weight: float = 0.5) -> Glicko:
        """`weight` of the way from this one to `other` (a Scala Float)."""
        w = f32(weight)
        if w >= 1:
            return other
        if w <= 0:
            return self
        keep = f32(1 - w)
        return Glicko(
            rating=self.rating * keep + other.rating * w,
            deviation=self.deviation * keep + other.deviation * w,
            volatility=self.volatility * keep + other.volatility * w,
        )

    def cap(self) -> Glicko:
        """lila GlickoExt.cap: the rating at least 400, the deviation 45 to 500, the volatility
        at most 0.1."""
        return Glicko(
            rating=max(self.rating, MIN_RATING),
            deviation=min(max(self.deviation, MIN_DEVIATION), MAX_DEVIATION),
            volatility=min(self.volatility, MAX_VOLATILITY),
        )

    def sanity_check(self) -> bool:
        return (
            0 < self.rating < MAX_RATING
            and 0 < self.deviation < 1000
            and 0 < self.volatility < MAX_VOLATILITY * 2
        )


DEFAULT = Glicko(DEFAULT_RATING, MAX_DEVIATION, DEFAULT_VOLATILITY)


@dataclass(frozen=True)
class Perf:
    """A player's puzzle rating as Lichess keeps it (lila.core.perf.Perf): the Glicko triple,
    how many rated puzzles, and when the last one was (epoch milliseconds, None before any)."""

    glicko: Glicko = DEFAULT
    nb: int = 0
    latest: int | None = None


# ---- the calculator (RatingCalculator.scala) -------------------------------------------------


def _to_glicko2(rating: float) -> float:
    return (rating - DEFAULT_RATING) / MULTIPLIER


def _from_glicko2(rating: float) -> float:
    return (rating * MULTIPLIER) + DEFAULT_RATING


def _g(deviation: float) -> float:
    return 1.0 / math.sqrt(1.0 + (3.0 * (deviation * deviation) / (math.pi * math.pi)))


def _e(player: float, opponent: float, opponent_deviation: float) -> float:
    return 1.0 / (1.0 + math.exp(-1.0 * _g(opponent_deviation) * (player - opponent)))


def _f(x: float, delta: float, phi: float, v: float, a: float, tau: float) -> float:
    ex = math.exp(x)
    denom = phi * phi + v + ex
    return (ex * (delta * delta - phi * phi - v - ex) / (2.0 * (denom * denom))) - (
        (x - a) / (tau * tau)
    )


def _new_rd(phi: float, sigma: float, elapsed_periods: float) -> float:
    """Step 6, also the deviation of a player away for `elapsed_periods`."""
    squared = phi * phi + elapsed_periods * (sigma * sigma)
    # The JVM's Math.sqrt of a negative is NaN, where Python's raises.
    return math.sqrt(squared) if squared >= 0 else math.nan


def rate(
    player: Glicko,
    results: list[tuple[Glicko, float]],
    *,
    elapsed_periods: float = 1.0,
    tau: float = TAU,
) -> Glicko:
    """The player after one rating period of `results` (each an opponent as they stood before
    it and the score, 1 a win, 0.5 a draw, 0 a loss): calculateNewRating."""
    phi = player.deviation / MULTIPLIER
    sigma = player.volatility
    mu = _to_glicko2(player.rating)
    a = math.log(sigma * sigma)
    v_sum = 0.0
    outcome = 0.0
    for opponent, score in results:
        o_phi = opponent.deviation / MULTIPLIER
        expected = _e(mu, _to_glicko2(opponent.rating), o_phi)
        g = _g(o_phi)
        v_sum += (g * g) * expected * (1.0 - expected)
        outcome += g * (score - expected)
    v = 1 / v_sum
    delta = v * outcome

    # step 5.2
    big_a = a
    if delta * delta > phi * phi + v:
        big_b = math.log(delta * delta - phi * phi - v)
    else:
        k = 1.0
        big_b = a - (k * abs(tau))
        while _f(big_b, delta, phi, v, a, tau) < 0:
            k = k + 1
            big_b = a - (k * abs(tau))
    # step 5.3
    f_a = _f(big_a, delta, phi, v, a, tau)
    f_b = _f(big_b, delta, phi, v, a, tau)
    # step 5.4
    iterations = 0
    while abs(big_b - big_a) > CONVERGENCE_TOLERANCE and iterations < ITERATION_MAX:
        iterations += 1
        big_c = big_a + (((big_a - big_b) * f_a) / (f_b - f_a))
        f_c = _f(big_c, delta, phi, v, a, tau)
        if f_c * f_b <= 0:
            big_a = big_b
            f_a = f_b
        else:
            f_a = f_a / 2.0
        big_b = big_c
        f_b = f_c
    if iterations == ITERATION_MAX:
        raise RuntimeError("Convergence fail")
    new_sigma = math.exp(big_a / 2.0)
    # steps 6-7
    phi_star = _new_rd(phi, new_sigma, elapsed_periods)
    new_phi = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    return Glicko(
        rating=_from_glicko2(mu + (new_phi * new_phi) * outcome),
        deviation=new_phi * MULTIPLIER,
        volatility=new_sigma,
    )


def compute_game(
    white: Glicko, black: Glicko, white_score: float, *, skip_deviation_increase: bool = False
) -> tuple[Glicko, Glicko]:
    """Both players after one game (GlickoCalculator.computeGame, no colour advantage): each
    rated against the other as they stood before it."""
    elapsed = 0.0 if skip_deviation_increase else 1.0
    return (
        rate(white, [(black, white_score)], elapsed_periods=elapsed),
        rate(black, [(white, 1.0 - white_score)], elapsed_periods=elapsed),
    )


# ---- the deviation over time (Glicko.liveDeviation, previewDeviation) ------------------------


def live_deviation(perf: Perf, now_ms: int, *, reverse: bool = False) -> float:
    """The deviation as of `now_ms`: grown by the rating periods since the last rated puzzle
    (shrunk back when `reverse`, as Lichess stores it), within 45 to 500."""
    capped = perf.glicko.cap()
    elapsed = 0.0
    if perf.latest is not None:
        elapsed = (now_ms - perf.latest) * (PERIODS_PER_DAY * DAYS_PER_MILLI)
    if reverse:
        elapsed = -elapsed
    rd = _new_rd(capped.deviation / MULTIPLIER, capped.volatility, elapsed) * MULTIPLIER
    return min(max(rd, MIN_DEVIATION), MAX_DEVIATION)


def read(perf: Perf, now_ms: int) -> Perf:
    """The perf as Lichess reads it from its store: the deviation as of now."""
    return replace(perf, glicko=replace(perf.glicko, deviation=live_deviation(perf, now_ms)))


def stored(perf: Perf, now_ms: int) -> Perf:
    """The perf as Lichess writes it to its store: the growth since the last puzzle taken out."""
    deviation = live_deviation(perf, now_ms, reverse=True)
    return replace(perf, glicko=replace(perf.glicko, deviation=deviation))


# ---- a puzzle attempt (PuzzleFinisher.scala) -------------------------------------------------


def theme_weight(theme: str | None, win: bool) -> float:
    """ponder.weightOf: how much of the computed change a puzzle of this angle keeps (a Float).
    `theme` is the angle's theme, None for an angle without one (an opening)."""
    if theme is None or theme == MIX:
        return f32(1.0)
    if theme in OBVIOUS_THEMES:
        return f32(0.1) if win else f32(0.4)
    if theme not in NON_HINTING_THEMES:
        return f32(0.2) if win else f32(0.7)
    return f32(0.7) if win else f32(0.8)


def player_weight(theme: str | None, win: bool, puzzle: Glicko) -> float:
    """ponder.player's weight: the theme's, less for a provisional puzzle, at least 0.1."""
    provisional = (f32(-0.2) if win else f32(-0.7)) if puzzle.provisional else f32(0.0)
    return max(f32(theme_weight(theme, win) + provisional), f32(0.1))


def add_or_reset(perf: Perf, player: Glicko, now_ms: int) -> Perf:
    """Perf.addOrReset: at most 700 points either way; a result that fails the sanity check
    starts over from the default; then capped, counted, dated."""
    new = replace(
        player,
        rating=max(
            min(player.rating, perf.glicko.rating + MAX_RATING_DELTA),
            perf.glicko.rating - MAX_RATING_DELTA,
        ),
    )
    glicko = new if new.sanity_check() else DEFAULT
    return Perf(glicko=glicko.cap(), nb=perf.nb + 1, latest=now_ms)


def finish(perf: Perf, puzzle: Glicko, win: bool, now_ms: int, *, theme: str | None = None) -> Perf:
    """The player's puzzle perf after a rated first attempt at `puzzle` (its own Glicko, as
    Lichess holds it), `perf` as read now (see `read`): PuzzleFinisher.apply's rated branch."""
    white, _ = compute_game(perf.glicko.cap(), puzzle.cap(), 1.0 if win else 0.0)
    after = add_or_reset(perf, white, now_ms)
    weight = player_weight(theme, win, puzzle)
    return replace(after, glicko=perf.glicko.average(after.glicko, weight))


def rating_diff(before: Perf, after: Perf) -> int:
    """The change Lichess shows after a puzzle: the shown ratings' difference."""
    return after.glicko.int_rating - before.glicko.int_rating


def diff_text(diff: int) -> str:
    """The change as Lichess writes it beside the rating: "+12", "−9" (a minus sign), "" for
    none (ui/puzzle/src/view/side.ts)."""
    if diff > 0:
        return f"+{diff}"
    if diff < 0:
        return f"−{-diff}"
    return ""
