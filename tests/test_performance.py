"""The rating a game was played at, from its moves alone (profile/performance.py)."""

from __future__ import annotations

import json

from sidecoach.profile import performance


def test_a_move_falls_in_a_bucket_by_its_cost_and_a_context_by_where_it_was_played() -> None:
    assert [performance.bucket(x) for x in (0.0, 0.01, 0.03, 0.05, 0.2, 0.5)] == [0, 1, 2, 3, 4, 5]
    # The opening or after it, times lost / level / won for the mover.
    assert performance.context(0, 0.5) == 1
    assert performance.context(19, 0.1) == 0
    assert performance.context(20, 0.9) == 5


def _side(blunders: int, moves: int = 30) -> list[dict[str, float]]:
    """A side's moves after the opening from a level position: all best but `blunders`."""
    return [
        {"ply": 20 + 2 * i, "loss": 0.3 if i < blunders else 0.0, "score": 0.5}
        for i in range(moves)
    ]


def test_better_moves_read_as_a_higher_rating() -> None:
    ratings = [performance.estimate(_side(n)).rating for n in (0, 2, 5, 10)]
    assert ratings == sorted(ratings, reverse=True) and ratings[0] - ratings[-1] > 500
    one = performance.estimate(_side(2))
    assert one.low < one.rating < one.high and one.moves == 30
    # With no moves, the prior alone: its middle, as near as the grid's ends let it be.
    assert abs(performance.estimate([]).rating - performance.PRIOR[0]) < 10


def test_ratings_are_given_on_the_scale_of_the_games_site() -> None:
    assert performance.chesscom(1290) == 815 and performance.chesscom(1795) == 1500
    assert performance.chesscom(200) == 100, "chess.com's floor"
    assert performance.for_site(1500, "https://lichess.org/abcdefgh") == 1500
    assert performance.for_site(1500, "Chess.com") == performance.chesscom(1500)
    assert performance.for_site(1500, None) == performance.chesscom(1500), "watched: chess.com"


def test_the_shipped_model_has_a_softmax_for_every_context() -> None:
    weights = json.loads(performance.MODEL_PATH.read_text(encoding="utf-8"))["weights"]
    assert (len(weights), len(weights[0]), len(weights[0][0])) == (
        performance.CONTEXTS,
        performance.BUCKETS,
        3,
    )
