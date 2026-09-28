"""Wiping the profile and the caches (sidecoach/maintenance.py), from the CLI's helpers
and over the panel's API."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chess
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


def test_the_panel_resets_the_profile_and_clears_the_caches_over_the_api(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    _filled(tmp_path)
    engine = make_panel()
    engine.practice_ids = ["fork1"]
    engine.puzzle_motifs["fork1"] = "fork"
    client = TestClient(build_app(engine))
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
