"""Every game of the library reviewed on its own, in the background (docs/stats-plan.md, decision
3), so Stats counts every game without the player opening each: the facts of reviews saved
before Stats first (no engine), then each game no review covers yet -- imported or watched, the
latest played first -- at the standard strength, one at a time, on two engines below normal
priority. It waits while the panel watches a game or a review room is analysing (the player's
own work comes first), and a review under way stops for them and starts again later.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.game_review import STRENGTHS
from sidecoach.config import Settings
from sidecoach.events import ReviewedGame
from sidecoach.profile import library
from sidecoach.profile.stat_facts import keep_stats
from sidecoach.profile.store import LibraryRow, Store
from sidecoach.ui.review_room import review_row

WORKERS = 2
STRENGTH = "standard"
IDLE_S = 5.0  # between looks while paused
REST_S = 30.0  # between looks when every game is reviewed


class Backlog:
    def __init__(
        self,
        settings: Settings,
        store_path: Path | None,
        *,
        busy: Callable[[], bool],
        make_engine: Callable[..., Engine] | None = None,
    ) -> None:
        self.settings = settings
        self.store_path = store_path
        self._busy = busy
        self._make_engine = make_engine
        self._stop = threading.Event()
        self._failed: set[str] = set()  # games that cannot be reviewed, skipped this session
        self.reviewing: str | None = None
        self.waiting = 0
        self.paused = False
        self._thread = threading.Thread(target=self._run, name="backlog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()

    def join(self, timeout: float) -> None:
        """Until the review under way has stopped (its engines closed), at most `timeout`."""
        if self._thread.is_alive():
            self._thread.join(timeout)

    def state(self) -> dict[str, Any]:
        return {"waiting": self.waiting, "reviewing": self.reviewing, "paused": self.paused}

    def _run(self) -> None:
        store = Store(self.store_path)
        try:
            while not self._stop.is_set():
                if self._busy():
                    self.paused = True
                    self._stop.wait(IDLE_S)
                    continue
                self.paused = False
                if self._keep_saved(store):
                    continue
                todo = self._todo(store)
                self.waiting = len(todo)
                if not todo:
                    self._stop.wait(REST_S)
                    continue
                self._review(store, todo[0])
        finally:
            store.close()

    def _keep_saved(self, store: Store) -> bool:
        """One saved review's facts kept for Stats; whether there was one."""
        for review in store.reviews_without_stats():
            row = library.find(store, review.key)
            if row is None or review.key in self._failed:
                continue
            try:
                keep_stats(
                    store, review.strength, ReviewedGame.model_validate_json(review.report), row
                )
            except Exception:  # noqa: BLE001 - a review that cannot be read is skipped
                self._failed.add(review.key)
            return True
        return False

    def _todo(self, store: Store) -> list[LibraryRow]:
        """The games no review covers, the latest played first; a watched game its import
        stands for is the import's to review."""
        twins = library.reconcile(store)
        watched = [
            r
            for r in library.from_watched(store)
            if r.key not in twins and store.latest_review(r.key) is None
        ]
        rows = [*store.unreviewed(), *watched]
        rows = [r for r in rows if r.key not in self._failed]
        return sorted(rows, key=lambda r: r.played_at or r.added_at, reverse=True)

    def _review(self, store: Store, row: LibraryRow) -> None:
        # Stopped part way by the player's own work (or the app closing): it starts again later.
        halt = threading.Event()

        def progress(_done: int, _total: int) -> None:
            if self._busy() or self._stop.is_set():
                halt.set()

        self.reviewing = row.key
        try:
            review_row(
                self.settings,
                row,
                store,
                STRENGTHS[STRENGTH],
                workers=WORKERS,
                make_engine=self._make_engine,
                low_priority=True,
                on_progress=progress,
                stop=halt,
            )
        except Exception:  # noqa: BLE001 - a game that cannot be reviewed is skipped
            self._failed.add(row.key)
        finally:
            self.reviewing = None
