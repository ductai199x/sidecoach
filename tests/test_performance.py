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


def test_a_game_set_up_from_a_position_is_rated_from_where_it_starts() -> None:
    """A game set up from a position (Black to move at move 23) counted its first moves as the
    opening's, by their place in the game, for the summary's rating and Stats' (the review of
    2026-10-01): the plies before a move are its position's."""
    import chess

    from sidecoach.profile.stats import _rated

    fen = "4k3/8/8/8/8/8/4P3/4K3 b - - 0 23"
    assert performance.plies_before(fen) == 45
    assert performance.plies_before(chess.STARTING_FEN) == 0
    assert _rated([{"ply": 1, "fen": fen, "loss": 0.0, "score": 0.5}])[0]["ply"] == 45


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


def test_the_models_read_moves_by_the_lines_they_were_fitted_with() -> None:
    """The audit of 2026-10-07: the buckets took the labels' thresholds, and when those became
    chess.com's (0.05 / 0.10 / 0.20, 0.1.84) every model read moves through lines it was not
    fitted on -- every game rating, category rating and trend fell. Each fitted model keeps the
    lines it was fitted with (2026-09-30: 0.04 / 0.10 / 0.28) and reads moves by them."""
    from sidecoach.profile import stats

    fitted = (1e-9, 0.02, 0.04, 0.10, 0.28)
    shipped = json.loads(performance.MODEL_PATH.read_text(encoding="utf-8"))
    assert performance.edges_of(shipped) == fitted
    categories = json.loads(stats.CATEGORIES_PATH.read_text(encoding="utf-8"))
    assert {name: performance.edges_of(m) for name, m in categories.items()} == {
        name: fitted for name in stats.CATEGORIES
    }

    # Losses of 0.045 and 0.06 are one bucket to the models (0.04 to 0.10), whatever today's
    # labels call them (good and inaccuracy): the same moves to every model.
    def side(loss: float) -> list[dict[str, float]]:
        return [{"ply": 20 + 2 * i, "loss": loss, "score": 0.5} for i in range(20)]

    assert performance.estimate(side(0.045)) == performance.estimate(side(0.06))
    for model in categories.values():
        assert performance.estimate(side(0.045), model=model) == performance.estimate(
            side(0.06), model=model
        )


def test_a_fit_keeps_the_lines_it_read_the_moves_by() -> None:
    """A refit (tools/review/fit_performance.py, and fit_stats.py through it) writes the lines
    it bucketed the moves by with its weights, so the model it writes reads moves by them."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "tools" / "review" / "fit_performance.py"
    spec = importlib.util.spec_from_file_location("fit_performance", path)
    assert spec is not None and spec.loader is not None
    fitter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fitter)
    rows = [{"rating": 1000 + 400 * k, "moves": _side(k)} for k in range(3)]
    model = fitter.fit(rows, steps=3)
    assert performance.edges_of(model) == performance.EDGES
    assert performance.estimate(_side(1), model=model).moves == 30


def test_the_shipped_model_has_a_softmax_for_every_context() -> None:
    weights = json.loads(performance.MODEL_PATH.read_text(encoding="utf-8"))["weights"]
    assert (len(weights), len(weights[0]), len(weights[0][0])) == (
        performance.CONTEXTS,
        performance.BUCKETS,
        3,
    )
