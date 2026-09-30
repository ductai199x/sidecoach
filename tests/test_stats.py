"""The player's games added up (profile/stats.py): the filter, then each section's figures and
the positions behind them, on two games whose facts are set by hand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sidecoach.profile.stats import Filter, family, positions, stats, termination_kind
from sidecoach.profile.store import Store


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


def test_names_are_read_as_players_read_them() -> None:
    assert family("B12 Caro-Kann Defense: Advance Variation") == "Caro-Kann Defense"
    assert termination_kind("nosegazm won on time") == "time"
    assert termination_kind("Time forfeit") == "time"
    assert termination_kind("Game drawn by repetition") == "repetition"
