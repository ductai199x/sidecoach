"""Wiping the profile and the caches (sidecoach/maintenance.py), from the CLI's helpers
and over the panel's API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import pytest
from fastapi.testclient import TestClient

from sidecoach.config import Settings
from sidecoach.events import GameStarted, Side
from sidecoach.maintenance import clear_caches, reset_profile
from sidecoach.profile.puzzles import PuzzleStore, open_store
from sidecoach.profile.store import Store
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app
from tests.fakes import puzzle_file

FORK = (
    "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5,"
    "1000,80,90,500,fork opening short,https://lichess.org/x#3,"
)


def _filled(tmp_path: Path) -> None:
    """A profile with a game and an attempt, a theme file, a puzzle store with its CSV."""
    store = Store(tmp_path / "profile.sqlite")
    store.start_game("g", 1.0, "white", chess.STARTING_FEN, "s")
    store.add_attempt("fork1", 2.0, "fork", True, ["Nxe5"])
    store.set_setting("puzzle_level", "1125")
    store.close()
    (tmp_path / "themes").mkdir()
    (tmp_path / "themes" / "theme-abc.pkl").write_bytes(b"x")
    puzzle_file(tmp_path / "puzzles" / "sample.csv", [FORK])
    puzzles = open_store(tmp_path)
    assert puzzles is not None and puzzles.count() == 1
    puzzles.close()


def test_reset_profile_empties_every_table_and_clear_caches_removes_the_files(
    tmp_path: Path,
) -> None:
    _filled(tmp_path)
    settings = Settings(data_dir=tmp_path)
    assert reset_profile(settings) == tmp_path / "profile.sqlite"
    store = Store(tmp_path / "profile.sqlite")
    assert (
        store.game("g") is None and store.attempts() == [] and store.setting("puzzle_level") is None
    )
    assert store.games_count() == 0
    store.close()
    done = clear_caches(settings)
    assert done == ["removed theme-abc.pkl", "removed puzzles.sqlite"]
    assert not (tmp_path / "themes" / "theme-abc.pkl").exists()
    assert not (tmp_path / "puzzles.sqlite").exists()
    assert clear_caches(settings) == [], "nothing left"
    # With a store held open, the puzzles are rebuilt from the CSV rather than unlinked.
    puzzles = PuzzleStore(tmp_path / "puzzles.sqlite")
    assert puzzles.count() == 0
    (tmp_path / "themes" / "theme-def.pkl").write_bytes(b"x")
    assert clear_caches(settings, puzzles) == [
        "removed theme-def.pkl",
        "puzzle store rebuilt: 1 puzzles",
    ]
    assert puzzles.count() == 1 and puzzles.get("fork1") is not None
    puzzles.close()
    # A missing profile is fine.
    (tmp_path / "profile.sqlite").unlink()
    assert reset_profile(settings) == tmp_path / "profile.sqlite"


def test_a_reset_keeps_the_pause_of_the_background_review(tmp_path: Path) -> None:
    """The review of 2026-10-01: Reset profile emptied the settings table, the player's Pause of
    the background review with it -- a choice about the app's work, not something the profile
    learned -- so the next start reviewed again. The puzzle level still goes."""
    from sidecoach.maintenance import KEPT_SETTINGS
    from sidecoach.ui.backlog import PAUSED_KEY

    assert PAUSED_KEY in KEPT_SETTINGS
    _filled(tmp_path)
    store = Store(tmp_path / "profile.sqlite")
    store.set_setting(PAUSED_KEY, "1")
    reset_profile(Settings(data_dir=tmp_path))
    assert store.setting(PAUSED_KEY) == "1" and store.setting("puzzle_level") is None
    store.close()


def test_a_review_under_way_saves_nothing_for_a_profile_reset_meanwhile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of 2026-10-01: Reset profile while a game was reviewed (the background review,
    or a review window): the review saved itself and its Stats facts for a game no longer in the
    library -- Stats counted a game nothing listed and nothing could remove. It saves nothing,
    keeps nothing part way, and the window says why."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.profile import library
    from sidecoach.ui import review_room
    from tests.fakes import FakeEngine
    from tests.test_review_server import GAME

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path / "profile.sqlite")
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    searched = [0]

    class Resetting(FakeEngine):
        """Reset profile pressed while the engine searches the review's fourth position."""

        def analyse(self, board: chess.Board, **kwargs: Any) -> list[Any]:
            searched[0] += 1
            if searched[0] == 4:
                reset_profile(settings)
            return super().analyse(board, **kwargs)

    done = review_room.review_row(
        settings, row, store, STRENGTHS["fast"], workers=1, make_engine=lambda **_: Resetting()
    )
    assert done is None
    assert store.review(row.key, "fast") is None and store.stat_games() == []
    assert store.progress(row.key, "fast", 14, 3) == []

    # A review window on it: the review is no longer there to save, and it says so.
    store.add_library_game(row)
    searched[0] = 0
    room = review_room.ReviewRoom(
        settings,
        row,
        store_path=tmp_path / "profile.sqlite",
        strength="fast",
        make_engine=lambda **_: Resetting(),
    )
    # Set by the room once the review is ready or failed; 60 s is only a safety net.
    assert room.settled.wait(60), "the review never settled"
    room.close()
    room.join()
    assert room.status == "failed" and room.error and "no longer" in room.error
    assert store.stat_games() == []
    store.close()


@pytest.mark.parametrize("lands", ["after the look", "after the save"])
def test_a_reset_between_the_reviews_look_and_its_save_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lands: str
) -> None:
    """The audit of 2026-10-02 (For Tai #14): Reset profile landing in the instant after the
    review asked whether the game is still in the record -- or after its review was saved, before
    its Stats facts -- still saved the review (and the facts) of a game no longer in the record.
    Staged exactly there: neither is saved."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.profile import library
    from sidecoach.ui import review_room
    from tests.fakes import FakeEngine
    from tests.test_review_server import GAME

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path / "profile.sqlite")
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    looked, saved = store.in_record, store.save_review

    def look(key: str) -> bool:
        answer = looked(key)
        if lands == "after the look":
            reset_profile(settings)  # another connection, as the panel's Reset profile
        return answer

    def save(*args: Any) -> bool:
        done = saved(*args)
        if lands == "after the save":
            reset_profile(settings)
        return done

    monkeypatch.setattr(store, "in_record", look)
    monkeypatch.setattr(store, "save_review", save)
    review_room.review_row(
        settings, row, store, STRENGTHS["fast"], workers=1, make_engine=lambda **_: FakeEngine()
    )
    assert store.review(row.key, "fast") is None, "no review of a game out of the record"
    assert store.stat_games() == [], "nor its Stats facts"
    store.close()


def test_the_panel_resets_the_profile_and_clears_the_caches_over_the_api(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    _filled(tmp_path)
    engine = make_panel()
    engine.practice_ids = ["fork1"]
    engine.puzzle_motifs["fork1"] = "fork"
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post("/api/clear_caches").json()["done"] == [
        "removed theme-abc.pkl",
        "removed puzzles.sqlite",
    ]
    res = client.post("/api/reset_profile").json()
    assert res["ok"] and res["path"].endswith("profile.sqlite")
    store = Store(tmp_path / "profile.sqlite")
    assert store.games_count() == 0 and store.attempts() == []
    store.close()
    assert engine.practice_ids == [] and engine.puzzle_motifs == {}
    lines = [ln["text"] for ln in client.get("/api/poll").json()["lines"]]
    assert any(ln.startswith("profile reset:") for ln in lines)
    assert any(ln.startswith("cache: removed theme-abc.pkl") for ln in lines)
    # A running profiler's game in progress is dropped with the reset.
    from sidecoach.profile.profiler import Profiler

    store = Store(tmp_path / "profile.sqlite")
    profiler = Profiler(store, session_id="s", ts=0.0)
    profiler.on_event(
        GameStarted(game_id="g2", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 1.0
    )
    engine._profiler = profiler
    client.post("/api/reset_profile")
    assert profiler._game is None and store.game("g2") is None
    store.close()
