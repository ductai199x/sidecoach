"""The puzzle rating against Lichess's own numbers (sidecoach/profile/glicko.py).

The calculator cases are scalachess's reference tests, ported with their tolerances
(scalachess 18.0.0, test-kit/src/test/scala/rating/glicko/GlickoCalculatorTest.scala and
glicko/impl/RatingCalculatorTest.scala), and Glickman's worked example (glicko2.pdf, the
example of section 3). The puzzle cases follow lila's PuzzleFinisher.scala, Glicko.scala and
Perf.scala line by line: the weights, the bounds and the deviation over time.
"""

from __future__ import annotations

import pytest

from sidecoach.profile.glicko import (
    DEFAULT,
    Glicko,
    Perf,
    compute_game,
    diff_text,
    f32,
    finish,
    live_deviation,
    player_weight,
    rate,
    rating_diff,
    read,
    stored,
    theme_weight,
)

DAY_MS = 24 * 60 * 60 * 1000

WHITE, BLACK, DRAW = 1.0, 0.0, 0.5

# GlickoCalculatorTest: computeGame(..., skipDeviationIncrease = true) -- (white, black,
# outcome, white's rating/deviation/volatility, black's, tolerances rating/deviation/volatility).
CALCULATOR_CASES = [
    # default deviation
    ((1500, 350, 0.06), (1500, 350, 0.06), WHITE,
     (1662.21, 290.2305, 0.0599993), (1337.79, 290.2305, 0.0599993), (0.005, 0.00005, 0.0000001)),
    ((1500, 350, 0.06), (1500, 350, 0.06), BLACK,
     (1337.79, 290.2305, 0.0599993), (1662.21, 290.2305, 0.0599993), (0.005, 0.00005, 0.0000001)),
    ((1500, 350, 0.06), (1500, 350, 0.06), DRAW,
     (1500, 290.2305, 0.0599977), (1500, 290.2305, 0.0599977), (0.005, 0.00005, 0.0000001)),
    # mixed ratings and deviations
    ((1400, 79, 0.06), (1550, 110, 0.065), WHITE,
     (1422.63, 77.4956, 0.06), (1506.32, 105.8706, 0.065), (0.005, 0.00005, 0.00001)),
    ((1400, 79, 0.06), (1550, 110, 0.065), BLACK,
     (1389.99, 77.4956, 0.06), (1568.90, 105.8706, 0.065), (0.005, 0.00005, 0.00001)),
    ((1400, 79, 0.06), (1550, 110, 0.065), DRAW,
     (1406.31, 77.4956, 0.06), (1537.61, 105.8706, 0.065), (0.005, 0.00005, 0.00001)),
    # more mixed ratings and deviations
    ((1200, 60, 0.053), (1850, 200, 0.062), WHITE,
     (1216.73, 59.9006, 0.053013), (1635.99, 196.9873, 0.062028), (0.005, 0.00005, 0.000001)),
    ((1200, 60, 0.053), (1850, 200, 0.062), BLACK,
     (1199.29, 59.9006, 0.052999), (1855.42, 196.9873, 0.061999), (0.005, 0.00005, 0.000001)),
    ((1200, 60, 0.053), (1850, 200, 0.062), DRAW,
     (1208.01, 59.9006, 0.053002), (1745.71, 196.9873, 0.062006), (0.005, 0.00005, 0.000001)),
]  # fmt: skip

# RatingCalculatorTest: updateRatings(results, skipDeviationIncrease = true), Lichess's default
# player (1500 / 500 / 0.09) among them.
RATING_CALCULATOR_CASES = [
    ((1500, 500, 0.09), (1500, 500, 0.09), WHITE,
     (1741, 396, 0.0899983), (1258, 396, 0.0899983), (1, 1, 0.0000001)),
    ((1500, 500, 0.09), (1500, 500, 0.09), BLACK,
     (1258, 396, 0.0899983), (1741, 396, 0.0899983), (1, 1, 0.0000001)),
    ((1500, 500, 0.09), (1500, 500, 0.09), DRAW,
     (1500, 396, 0.0899954), (1500, 396, 0.0899954), (1, 1, 0.0000001)),
    ((1500, 80, 0.06), (1500, 80, 0.06), WHITE,
     (1517, 78, 0.06), (1482, 78, 0.06), (1, 1, 0.00001)),
    ((1500, 80, 0.06), (1500, 80, 0.06), BLACK,
     (1482, 78, 0.06), (1517, 78, 0.06), (1, 1, 0.00001)),
    ((1500, 80, 0.06), (1500, 80, 0.06), DRAW,
     (1500, 78, 0.06), (1500, 78, 0.06), (1, 1, 0.00001)),
    ((1400, 79, 0.06), (1550, 110, 0.065), WHITE,
     (1422, 77, 0.06), (1506, 105, 0.065), (1, 1, 0.00001)),
    ((1400, 79, 0.06), (1550, 110, 0.065), BLACK,
     (1389, 78, 0.06), (1568, 105, 0.065), (1, 1, 0.00001)),
    ((1400, 79, 0.06), (1550, 110, 0.065), DRAW,
     (1406, 78, 0.06), (1537, 105.87, 0.065), (1, 1, 0.00001)),
    ((1200, 60, 0.053), (1850, 200, 0.062), WHITE,
     (1216.7, 59.9, 0.053013), (1636, 196.9, 0.062028), (0.1, 0.1, 0.000001)),
    ((1200, 60, 0.053), (1850, 200, 0.062), BLACK,
     (1199.3, 59.9, 0.052999), (1855.4, 196.9, 0.061999), (0.1, 0.1, 0.000001)),
    ((1200, 60, 0.053), (1850, 200, 0.062), DRAW,
     (1208.0, 59.90056, 0.053002), (1745.7, 196.98729, 0.062006), (0.1, 0.1, 0.000001)),
]  # fmt: skip


def _close(got: Glicko, want: tuple[float, float, float], tol: tuple[float, float, float]) -> None:
    assert abs(got.rating - want[0]) <= tol[0], (got, want)
    assert abs(got.deviation - want[1]) <= tol[1], (got, want)
    assert abs(got.volatility - want[2]) <= tol[2], (got, want)


@pytest.mark.parametrize("case", CALCULATOR_CASES + RATING_CALCULATOR_CASES)
def test_a_game_rates_both_players_as_scalachess_does(case) -> None:
    white, black, outcome, want_w, want_b, tol = case
    w, b = compute_game(Glicko(*white), Glicko(*black), outcome, skip_deviation_increase=True)
    _close(w, want_w, tol)
    _close(b, want_b, tol)


def test_glickmans_worked_example() -> None:
    # glicko2.pdf: a 1500 / 200 / 0.06 player beats a 1400 / 30, loses to a 1550 / 100 and a
    # 1700 / 300, with tau 0.5: 1464.06 / 151.52 / 0.05999.
    player = rate(
        Glicko(1500, 200, 0.06),
        [(Glicko(1400, 30, 0.06), 1), (Glicko(1550, 100, 0.06), 0), (Glicko(1700, 300, 0.06), 0)],
        tau=0.5,
    )
    _close(player, (1464.06, 151.52, 0.05999), (0.01, 0.01, 0.00001))


def test_a_rated_game_grows_the_deviation_one_period_first() -> None:
    # computeGame without skipDeviationIncrease (as PuzzleFinisher calls it): step 6 adds one
    # period of volatility before the result narrows it, so the deviation ends wider.
    one, _ = compute_game(Glicko(1500, 80, 0.06), Glicko(1500, 80, 0.06), WHITE)
    skipped, _ = compute_game(
        Glicko(1500, 80, 0.06), Glicko(1500, 80, 0.06), WHITE, skip_deviation_increase=True
    )
    assert one.deviation > skipped.deviation and one.rating > skipped.rating > 1500


def test_the_deviation_grows_from_60_to_110_in_a_year_away() -> None:
    # Glicko.scala: periodsPerDay "chosen so a typical player's RD goes from 60 -> 110 in 1 year".
    perf = Perf(Glicko(1500, 60, 0.06), nb=100, latest=0)
    assert live_deviation(perf, 0) == pytest.approx(60)
    assert live_deviation(perf, 365 * DAY_MS) == pytest.approx(110, abs=0.5)
    # Within 45 to 500 whatever the time; none before any puzzle.
    assert live_deviation(Perf(Glicko(1500, 30, 0.06), latest=0), 0) == 45
    assert live_deviation(Perf(Glicko(1500, 450, 0.09), latest=0), 3650 * DAY_MS) == 500
    assert live_deviation(Perf(Glicko(1500, 300, 0.09)), 3650 * DAY_MS) == 300
    # Stored as Lichess writes it: the growth taken back out, read back the same.
    later = read(perf, 200 * DAY_MS)
    assert stored(later, 200 * DAY_MS).glicko.deviation == pytest.approx(60)


def test_the_theme_weighs_the_change_as_ponder_does() -> None:
    assert theme_weight(None, True) == 1 and theme_weight("mix", False) == 1
    # Scala Floats, as the JVM holds 0.2f and 0.7f, not the doubles 0.2 and 0.7.
    assert theme_weight("fork", True) == 0.20000000298023224
    assert theme_weight("fork", False) == 0.699999988079071
    assert theme_weight("mateIn1", True) == f32(0.1)
    assert theme_weight("backRankMate", False) == f32(0.4)
    assert theme_weight("fork", True) == f32(0.2) and theme_weight("mateIn2", False) == f32(0.7)
    assert theme_weight("endgame", True) == f32(0.7) and theme_weight("superGM", False) == f32(0.8)
    # A provisional puzzle counts less again, never under 0.1.
    shaky, firm = Glicko(1500, 110, 0.09), Glicko(1500, 109.9, 0.09)
    assert player_weight("mix", True, shaky) == f32(1 - f32(0.2))
    assert player_weight("mix", False, shaky) == f32(1 - f32(0.7))
    assert player_weight("fork", True, shaky) == f32(0.1)
    assert player_weight("mix", True, firm) == 1


def test_a_new_player_solving_a_mixed_puzzle_moves_as_computeGame_says() -> None:
    puzzle = Glicko(1500, 75, 0.09)
    after = finish(Perf(), puzzle, True, 1000)
    white, _ = compute_game(DEFAULT, puzzle, WHITE)
    assert after.glicko == white.cap() and after.nb == 1 and after.latest == 1000
    lost = finish(Perf(), puzzle, False, 1000)
    assert lost.glicko.rating < 1500 < after.glicko.rating
    # About 200 up or down, the first puzzle of a 500-deviation player.
    assert 150 < rating_diff(Perf(), after) < 300 and -300 < rating_diff(Perf(), lost) < -150


def test_a_themed_puzzle_keeps_only_its_weight_of_the_change() -> None:
    puzzle = Glicko(1400, 80, 0.09)
    start = Perf(Glicko(1450, 90, 0.06), nb=40, latest=0)
    full = finish(start, puzzle, True, 1000)
    fork = finish(start, puzzle, True, 1000, theme="fork")
    w = f32(0.2)
    want = start.glicko.rating * f32(1 - w) + full.glicko.rating * w
    assert fork.glicko.rating == pytest.approx(want, abs=1e-9)
    assert start.glicko.rating < fork.glicko.rating < full.glicko.rating


def test_the_rating_never_falls_under_400_nor_moves_700_at_once() -> None:
    low = finish(Perf(Glicko(420, 300, 0.09), nb=3, latest=0), Glicko(600, 75, 0.09), False, 1)
    assert low.glicko.rating == 400
    # A loss to a far weaker puzzle at the widest deviation: capped at 700 down.
    start = Perf(Glicko(2500, 500, 0.09))
    fell = finish(start, Glicko(400, 45, 0.09), False, 1)
    assert fell.glicko.rating == pytest.approx(1800)


def test_a_result_that_fails_the_sanity_check_starts_over() -> None:
    # Perf.addOrReset: a rating at 4000 or more is not sane, the default comes back.
    start = Perf(Glicko(3990, 300, 0.09), nb=5, latest=0)
    after = finish(start, Glicko(3990, 60, 0.09), True, 1)
    assert after.glicko == DEFAULT and after.nb == 6


def test_the_rating_shows_truncated_with_a_question_mark_while_provisional() -> None:
    assert Glicko(1499.99, 109, 0.09).display() == "1499"
    assert Glicko(1500.5, 110, 0.09).display() == "1500?"
    before, after = Perf(Glicko(1499.9, 80, 0.06)), Perf(Glicko(1512.1, 80, 0.06))
    assert rating_diff(before, after) == 13 and diff_text(13) == "+13"
    assert diff_text(-9) == "−9" and diff_text(0) == ""
