"""The player's games added up (profile/stats.py): the filter, then each section's figures and
the positions behind them, on two games whose facts are set by hand."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from sidecoach.profile import performance
from sidecoach.profile.stats import (
    Filter,
    _Data,
    area,
    family,
    figures,
    grade,
    positions,
    stats,
    termination_kind,
    walk,
)
from sidecoach.profile.store import LibraryRow, Store


def _move(ply: int, side: str, **fields: Any) -> dict[str, Any]:
    fact = {
        "ply": ply,
        "side": side,
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "uci": "e2e4",
        "san": "e4",
        "phase": "middlegame",
        "loss": 0.0,
        "label": "best",
        "accuracy": 90.0,
        "score": 0.5,
        "chance": None,
        "found": False,
        "allowed": None,
        "structure": "semi-open",
        "themes": "",
        "endgame": None,
        "spent_ms": 5000,
        "clock_ms": 300_000,
    }
    return {**fact, **fields}


def _game(key: str, **fields: Any) -> dict[str, Any]:
    game = {
        "key": key,
        "source": "chesscom",
        "time_class": "rapid",
        "played_at": time.time() - 3600,
        "user_side": "white",
        "bot": False,
        "result": "1-0",
        "opening": "C50 Italian Game: Giuoco Piano",
        "termination": "me won by resignation",
    }
    return {**game, **fields}


def _store(tmp_path: Path) -> Store:
    store = Store(tmp_path / "profile.sqlite")
    # A rapid win against a person: a fork found, a pin missed, an endgame entered better.
    store.save_stats(
        "standard",
        _game("a"),
        [
            _move(1, "white", phase="opening", label="book", accuracy=100.0),
            _move(2, "black", phase="opening", label="book", accuracy=100.0),
            _move(3, "white", phase="opening", label="mistake", accuracy=40.0),
            _move(5, "white", chance="fork", found=True),
            _move(7, "white", chance="pin", found=False, label="miss", accuracy=30.0),
            _move(9, "white", allowed="free piece", label="blunder", accuracy=10.0),
            _move(10, "black", phase="endgame", endgame="rook", score=0.2),
            _move(11, "white", phase="endgame", endgame="rook", accuracy=80.0, clock_ms=30_000),
        ],
    )
    # A blitz loss on time to a bot, from a won position.
    store.save_stats(
        "standard",
        _game("b", time_class="blitz", bot=True, result="0-1", termination="bot won on time"),
        [_move(1, "white", score=0.9, label="brilliant", accuracy=100.0)],
    )
    return store


def test_the_games_are_added_up_section_by_section(tmp_path: Path) -> None:
    store = _store(tmp_path)
    everything = stats(store, Filter())
    overview = everything["overview"]
    assert (overview["games"], overview["wins"], overview["losses"]) == (2, 1, 1)
    openings = everything["openings"]
    assert openings["mistakes"]["value"] == 0.5, "one opening mistake over two games"
    # White left book first in both: after one book move of its own in a, at once in b.
    assert openings["book_moves"] == {"value": 0.5, "n": 2}
    assert openings["by_opening"][0]["opening"] == "Italian Game"
    tactics = everything["tactics"]
    assert tactics["found"]["fork"] == {"value": 100.0, "hits": 1, "n": 1}
    assert tactics["found"]["pin"]["value"] == 0.0
    assert tactics["allowed"]["free piece"]["hits"] == 1
    assert tactics["overall"]["value"] == 50.0
    # The endgame came with Black to move at 0.2 for Black: White entered it better, and won.
    assert everything["endgames"]["win"]["better"] == {"value": 100.0, "n": 1}
    assert everything["endgames"]["accuracy"]["rook"]["value"] == 80.0
    time_ = everything["time"]
    assert time_["pressure"]["n"] == 1, "the rook ending move at 30 s in a rapid game"
    assert (time_["lost_on_time"], time_["lost_on_time_winning"]) == (1, 1)
    assert everything["results"]["time"] == {"won": 0, "drawn": 0, "lost": 1}
    assert everything["galleries"]["brilliant"] == 1
    # Game by game: game a found one tactic of two; neither game is long enough for a rating.
    trends = everything["trends"]
    assert [p["v"] for p in trends["tactics"]] == [50.0] and trends["rating"] == []


def test_the_games_are_counted_by_their_shape(tmp_path: Path) -> None:
    """Which kinds of game suit the player (the plan's G): each game by the shape its review's
    summary gives it, won, drawn or lost; a game whose review has no shape is not counted, and
    the shapes come in game_summary's order."""
    store = _store(tmp_path)
    summary = {"white": {"side": "white"}, "black": {"side": "black"}}
    store.save_review("a", "standard", 1, 1.0, "{}", json.dumps({**summary, "shape": "sudden"}))
    store.save_review("b", "standard", 1, 1.0, "{}", json.dumps({**summary, "shape": "smooth"}))
    store.save_stats("standard", _game("c", result="1/2-1/2"), [_move(1, "white")])
    store.save_review("c", "standard", 1, 1.0, "{}", json.dumps({**summary, "shape": None}))
    shapes = stats(store, Filter())["shapes"]
    assert shapes == [
        {"shape": "smooth", "games": 1, "won": 0, "drawn": 0, "lost": 1},
        {"shape": "sudden", "games": 1, "won": 1, "drawn": 0, "lost": 0},
    ]
    assert stats(store, Filter(time_class="blitz"))["shapes"] == [shapes[0]]


def test_the_filter_takes_a_time_control_and_people_or_bots(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert stats(store, Filter(time_class="blitz"))["overview"]["games"] == 1
    assert stats(store, Filter(opponents="people"))["overview"]["games"] == 1
    assert stats(store, Filter(opponents="bots"))["overview"]["losses"] == 1
    assert stats(store, Filter(days=0))["overview"]["games"] == 2, "no limit"


def test_a_figure_opens_the_positions_behind_it(tmp_path: Path) -> None:
    store = _store(tmp_path)
    missed = positions(store, Filter(), "tactics.missed", "pin")
    assert [(p["key"], p["ply"]) for p in missed] == [("a", 7)]
    assert positions(store, Filter(), "gallery", "brilliant")[0]["key"] == "b"
    assert positions(store, Filter(), "nothing") == []


def test_the_peers_figures_come_from_the_same_code(tmp_path: Path) -> None:
    """The peers' sample is added up by the functions that add up the player's own games."""
    store = _store(tmp_path)
    mine = stats(store, Filter())
    games = store.stat_games()
    again = figures(_Data(games, store.stat_moves([g["key"] for g in games])))
    for section in ("overview", "openings", "tactics", "strategy", "endgames", "time"):
        assert again[section] == mine[section], section
    paths = dict(walk(again))
    assert paths["tactics.found.fork"] == {"value": 100.0, "hits": 1, "n": 1}
    assert "openings.by_opening" not in paths, "a list is not a figure"


def test_a_figure_is_graded_by_how_far_ahead_of_its_peers_it_is() -> None:
    # Accuracy rising two points a hundred rating points: 80 at 1500.
    rising = {"a": 50.0, "b": 0.02, "r2": 0.9, "lo": 900, "hi": 2400}
    assert grade(80.0, 1000, rising, 1500) == (pytest.approx(80.0), "good")
    assert grade(86.0, 1000, rising, 1500)[1] == "great", "about 300 points ahead"
    assert grade(70.0, 1000, rising, 1500)[1] == "blunder", "about 500 behind"
    # Past the sample's range the peers are read at its edge, not carried on.
    assert grade(98.0, 1000, rising, 3000)[0] == pytest.approx(98.0)
    # Mistakes a game, falling with the rating: fewer than the peers' is ahead.
    falling = {"a": 3.0, "b": -0.001, "r2": 0.8, "lo": 900, "hi": 2400}
    assert grade(1.0, 100, falling, 1500)[1] == "brilliant"
    # One game of it says little: pulled towards the peers' first.
    assert grade(1.0, 1, falling, 1500)[1] == "excellent"
    # A figure the rating hardly moves gets the peers' value but no grade.
    flat = {"a": 12.0, "b": 0.0001, "r2": 0.2, "lo": 900, "hi": 2400}
    assert grade(20.0, 500, flat, 1500) == (pytest.approx(12.15), None)


def test_the_peers_are_players_of_the_players_rating(tmp_path: Path) -> None:
    """With the games' ratings known, the peers are players of that rating (on the page's scale),
    every figure with the peers' value beside it; each category rated from enough of its moves."""
    store = Store(tmp_path / "profile.sqlite")
    moves = [
        _move(
            2 * i + 1,
            "white",
            phase="opening" if i < 12 else "middlegame",
            loss=0.01,
            accuracy=85.0,
        )
        for i in range(30)
    ]
    for key, elo in (("x", 1100), ("y", 1300)):
        store.add_library_game(
            LibraryRow(key=key, source="chesscom", added_at=0.0, pgn="", white_elo=elo)
        )
        store.save_stats("standard", _game(key), moves)
    everything = stats(store, Filter())
    assert everything["peers"] == {"rating": 1200, "from": "rating"}
    assert "peers" in everything["overview"]["accuracy"]
    ratings = everything["ratings"]
    assert ratings["overall"] is not None and ratings["openings"] is not None
    assert ratings["endgames"] is None, "no endgame moves"
    # Without the games' ratings, the peers are of the player's level of play.
    bare = Store(tmp_path / "bare.sqlite")
    bare.save_stats("standard", _game("z"), moves)
    assert stats(bare, Filter())["peers"]["from"] == "play"


def test_work_on_this_names_what_cost_most(tmp_path: Path) -> None:
    """Each costly move counts in one area (a missed tactic first, then one handed over, then the
    phase's), the dearest areas first, each with its puzzles' motif and its own positions."""
    store = Store(tmp_path / "profile.sqlite")
    for key in "abc":
        store.save_stats(
            "standard",
            _game(key),
            [
                _move(3, "white", phase="opening", label="blunder", loss=0.5),
                _move(21, "white", chance="fork", found=False, label="miss", loss=0.3),
                _move(23, "white", allowed="free piece", label="mistake", loss=0.2),
                _move(25, "white", phase="endgame", endgame="rook", label="mistake", loss=0.1),
                _move(27, "white", label="inaccuracy", loss=0.05),  # no area: not a mistake
                _move(29, "white", chance="pin", found=True, loss=0.0),  # found: costs nothing
            ],
        )
    everything = stats(store, Filter())
    assert everything["costs"]["opening"]["value"] == 0.5
    assert everything["costs"]["opening"]["moves"] == 3
    assert everything["costs"]["missed"]["pin"]["value"] == 0.0
    work = everything["work"]
    assert [w["area"] for w in work] == ["opening", "missed.fork", "allowed.free piece"]
    assert [w["motif"] for w in work] == ["opening", "fork", "hangingPiece"]
    assert [(w["moves"], w["games"]) for w in work] == [(3, 3)] * 3
    behind = positions(store, Filter(), "area", "missed.fork")
    assert [p["ply"] for p in behind] == [21, 21, 21]
    assert area(_move(1, "white", label="best")) is None


def test_work_on_this_waits_for_several_games(tmp_path: Path) -> None:
    """One game's blunders were "Work on this", and the cost read as points: 11.1 for one area
    when 9 were lost in all (the review of 2026-10-01). Nothing is named over fewer than three
    games, nor an area of fewer than three moves; each area says how many moves it counts."""
    store = Store(tmp_path / "profile.sqlite")
    moves = [
        _move(3 + 2 * i, "white", phase="opening", label="blunder", loss=0.5) for i in range(4)
    ]
    store.save_stats("standard", _game("a"), moves)
    assert stats(store, Filter())["work"] == []
    store.save_stats("standard", _game("b"), [_move(21, "white", label="best")])
    store.save_stats(
        "standard", _game("c"), [_move(21, "white", allowed="fork", label="mistake", loss=0.4)]
    )
    work = stats(store, Filter())["work"]
    assert [(w["area"], w["moves"], w["games"]) for w in work] == [("opening", 4, 3)]


def test_chesscom_and_lichess_ratings_convert_both_ways() -> None:
    for rating in (800, 1000, 1500, 2200):
        assert performance.chesscom(performance.lichess(rating)) == rating


def test_names_are_read_as_players_read_them() -> None:
    assert family("B12 Caro-Kann Defense: Advance Variation") == "Caro-Kann Defense"
    assert termination_kind("nosegazm won on time") == "time"
    assert termination_kind("Time forfeit") == "time"
    assert termination_kind("Game drawn by repetition") == "repetition"
