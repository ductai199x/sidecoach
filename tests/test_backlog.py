"""Every game of the library reviewed in the background for Stats (ui/backlog.py): a game
brought in is reviewed and its facts kept; a review saved before Stats has its facts kept without
an engine; nothing runs while the player is busy."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sidecoach.profile import library
from sidecoach.profile.store import Store
from sidecoach.ui.panel import PanelEngine
from tests.fakes import FakeEngine
from tests.test_review_server import GAME


def _until(check: Callable[[], bool], seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.05)
    return False


def test_a_game_brought_in_is_reviewed_in_the_background_and_its_facts_kept(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(backlog_module, "IDLE_S", 0.05)
    monkeypatch.setattr(backlog_module, "REST_S", 0.05)
    made: list[FakeEngine] = []

    def engine_for(**kwargs: Any) -> FakeEngine:
        assert kwargs["low_priority"], "the background's engines yield to everything else"
        made.append(FakeEngine())
        return made[-1]

    settings = make_panel().settings
    path = settings.data_dir / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    backlog = backlog_module.Backlog(settings, path, make_engine=engine_for)
    backlog.start()
    try:
        # It runs at once, beside whatever else runs: only the player's Pause stops it (Tai,
        # 2026-10-01: it waited on its own while a game was followed, which he never asked for).
        assert _until(lambda: any(g["key"] == row.key for g in store.stat_games()))
        game = next(g for g in store.stat_games() if g["key"] == row.key)
        assert (game["user_side"], game["bot"], game["strength"]) == ("black", False, "standard")
        facts = store.stat_moves([row.key])
        assert len(facts) == 12 and len(made) == 2
        assert store.best_review(row.key) is not None
        assert _until(lambda: backlog.state()["status"] == "done")
        done = backlog.state()
        assert (done["done"], done["total"], done["waiting"], done["reviewing"]) == (1, 1, 0, None)
        # A review saved before Stats (its facts gone): kept again, with no engine.
        with store._lock:  # noqa: SLF001 - a review from before Stats, as an upgrade finds it
            store._db.execute("DELETE FROM stat_games WHERE key = ?", (row.key,))  # noqa: SLF001
            store._db.execute("DELETE FROM stat_moves WHERE key = ?", (row.key,))  # noqa: SLF001
            store._db.commit()  # noqa: SLF001
        engines = len(made)
        assert _until(lambda: len(store.stat_moves([row.key])) == 12)
        assert len(made) == engines
    finally:
        backlog.close()
        backlog.join(10.0)
        store.close()


def test_a_game_that_failed_once_is_counted_and_tried_again(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of 2026-10-01: a game whose review raised once was skipped for the rest of the
    session and left out of the job's count, so the job said "empty" with games waiting; on a
    first run, before Setup had fetched Stockfish, every game failed until the app restarted.
    With no Stockfish yet no game has failed: they are reviewed once it is there. A game that
    failed is still counted, and tried again a while later."""
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    stockfish: list[Path | None] = [None]
    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: stockfish[0])
    monkeypatch.setattr(backlog_module, "IDLE_S", 0.05)
    monkeypatch.setattr(backlog_module, "REST_S", 0.05)
    monkeypatch.setattr(backlog_module, "FAILED_RETRY_S", 0.5)
    broken = [True]

    class Breaking(FakeEngine):
        def open(self) -> None:
            if broken[0]:
                raise RuntimeError("the engine would not start")

    settings = make_panel().settings
    path = settings.data_dir / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    backlog = backlog_module.Backlog(settings, path, make_engine=lambda **_: Breaking())
    backlog.start()
    try:
        # No Stockfish yet: counted and waiting, never "empty".
        assert _until(lambda: backlog.state()["total"] == 1)
        time.sleep(0.3)
        state = backlog.state()
        assert (state["status"], state["total"], state["waiting"]) == ("reviewing", 1, 1)
        # Stockfish fetched, but the engine would not start: failed, still counted.
        stockfish[0] = Path("stockfish")
        time.sleep(0.3)
        state = backlog.state()
        assert (state["status"], state["total"], state["waiting"]) == ("reviewing", 1, 1)
        assert store.best_review(row.key) is None
        # The engine starts again: the game is tried again, without a restart, and reviewed.
        broken[0] = False
        assert _until(lambda: backlog.state()["status"] == "done", 10.0)
        assert store.best_review(row.key) is not None
    finally:
        backlog.close()
        backlog.join(10.0)
        store.close()


def test_the_player_pauses_and_resumes_and_a_pause_outlives_a_restart(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-30: the background review is a job the player runs, as a download is --
    paused or resumed at will, paused staying paused when Sidecoach starts again."""
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(backlog_module, "IDLE_S", 0.05)
    monkeypatch.setattr(backlog_module, "REST_S", 0.05)
    made: list[FakeEngine] = []

    def engine_for(**_: Any) -> FakeEngine:
        made.append(FakeEngine())
        return made[-1]

    settings = make_panel().settings
    path = settings.data_dir / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    first = backlog_module.Backlog(settings, path, make_engine=engine_for)
    assert first.pause(True)["paused_by_you"]
    first.start()
    try:
        assert _until(lambda: first.state()["status"] == "paused" and first.state()["total"] == 1)
        time.sleep(0.3)
        assert not made and store.best_review(row.key) is None, "paused: nothing runs"
    finally:
        first.close()
        first.join(10.0)

    # Sidecoach started again: still paused, until the player resumes.
    again = backlog_module.Backlog(settings, path, make_engine=engine_for)
    assert again.paused_by_you
    again.start()
    try:
        time.sleep(0.3)
        assert not made
        assert again.pause(False)["paused_by_you"] is False
        assert _until(lambda: again.state()["status"] == "done")
        assert store.best_review(row.key) is not None and made
    finally:
        again.close()
        again.join(10.0)
        store.close()
