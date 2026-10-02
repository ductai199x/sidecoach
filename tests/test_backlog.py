"""Every game of the library reviewed in the background for Stats (ui/backlog.py): a game
brought in is reviewed and its facts kept; a review saved before Stats has its facts kept without
an engine; the player's Pause stops it. The job is stepped by hand (`Backlog.step`, what its
thread runs again and again) with a clock of the test's own: what it decides, never how soon."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sidecoach.profile import library
from sidecoach.profile.store import Store
from sidecoach.ui.panel import PanelEngine
from tests.fakes import FakeEngine
from tests.test_review_server import GAME


def test_a_game_brought_in_is_reviewed_in_the_background_and_its_facts_kept(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
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
    try:
        # It runs at once, beside whatever else runs: only the player's Pause stops it (Tai,
        # 2026-10-01: it waited on its own while a game was followed, which he never asked for).
        assert backlog.step() == "reviewed"
        game = next(g for g in store.stat_games() if g["key"] == row.key)
        assert (game["user_side"], game["bot"], game["strength"]) == ("black", False, "standard")
        facts = store.stat_moves([row.key])
        assert len(facts) == 12 and len(made) == 2
        assert store.best_review(row.key) is not None
        assert backlog.step() == "idle"
        done = backlog.state()
        assert (
            done["status"],
            done["done"],
            done["total"],
            done["waiting"],
            done["reviewing"],
        ) == (
            "done",
            1,
            1,
            0,
            None,
        )
        # A review saved before Stats (its facts gone): kept again, with no engine.
        with store._lock:  # noqa: SLF001 - a review from before Stats, as an upgrade finds it
            store._db.execute("DELETE FROM stat_games WHERE key = ?", (row.key,))  # noqa: SLF001
            store._db.execute("DELETE FROM stat_moves WHERE key = ?", (row.key,))  # noqa: SLF001
            store._db.commit()  # noqa: SLF001
        engines = len(made)
        assert backlog.step() == "kept"
        assert len(store.stat_moves([row.key])) == 12 and len(made) == engines
    finally:
        backlog.close()
        store.close()


def test_a_game_that_failed_once_is_counted_and_tried_again(
    make_panel: Callable[..., PanelEngine],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The review of 2026-10-01: a game whose review raised once was skipped for the rest of the
    session and left out of the job's count, so the job said "empty" with games waiting; on a
    first run, before Setup had fetched Stockfish, every game failed until the app restarted.
    With no Stockfish yet no game has failed: they are reviewed once it is there. A game that
    failed is still counted, and tried again FAILED_RETRY_S later."""
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    stockfish: list[Path | None] = [None]
    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: stockfish[0])
    broken = [True]
    now = [1000.0]

    class Breaking(FakeEngine):
        def open(self) -> None:
            if broken[0]:
                raise RuntimeError("the engine would not start")

    settings = make_panel().settings
    path = settings.data_dir / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    backlog = backlog_module.Backlog(
        settings, path, make_engine=lambda **_: Breaking(), clock=lambda: now[0]
    )
    try:
        # No Stockfish yet: counted and waiting, never "empty", and no game failed.
        assert backlog.step() == "no stockfish"
        state = backlog.state()
        assert (state["status"], state["total"], state["waiting"]) == ("reviewing", 1, 1)
        # Stockfish fetched, but the engine would not start: failed, still counted.
        stockfish[0] = Path("stockfish")
        assert backlog.step() == "reviewed"
        state = backlog.state()
        assert (state["status"], state["total"], state["waiting"]) == ("reviewing", 1, 1)
        assert store.best_review(row.key) is None
        # Said, never kept in silence (the audit of 2026-10-02): why a game stays waiting.
        assert f"the background review of {row.key} failed" in caplog.text
        assert "the engine would not start" in caplog.text
        # The engine starts again: the failed game rests until FAILED_RETRY_S has passed, then
        # it is tried again, without a restart, and reviewed.
        broken[0] = False
        assert backlog.step() == "idle", "resting"
        now[0] += backlog_module.FAILED_RETRY_S
        assert backlog.step() == "reviewed"
        assert backlog.state()["status"] == "done"
        assert store.best_review(row.key) is not None
    finally:
        backlog.close()
        store.close()


def test_the_player_pauses_and_resumes_and_a_pause_outlives_a_restart(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-30: the background review is a job the player runs, as a download is --
    paused or resumed at will, paused staying paused when Sidecoach starts again."""
    from sidecoach.ui import backlog as backlog_module
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
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
    try:
        assert first.step() == "paused"
        assert (first.state()["status"], first.state()["total"]) == ("paused", 1)
        assert not made and store.best_review(row.key) is None, "paused: nothing runs"
    finally:
        first.close()

    # Sidecoach started again: still paused, until the player resumes.
    again = backlog_module.Backlog(settings, path, make_engine=engine_for)
    assert again.paused_by_you
    try:
        assert again.step() == "paused" and not made
        assert again.pause(False)["paused_by_you"] is False
        assert again.step() == "reviewed"
        assert again.state()["status"] == "done"
        assert store.best_review(row.key) is not None and made
    finally:
        again.close()
        store.close()
