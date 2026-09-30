"""Every game of the library reviewed on its own, in the background (docs/stats-plan.md, decision
3), so Stats counts every game without the player opening each: the facts of reviews saved
before Stats first (no engine), then each game no review covers yet -- imported or watched, the
latest played first -- at the standard strength, one at a time, on two engines below normal
priority.

A job the player sees and runs, as a download is (Tai, 2026-09-30: "we don't want to kick off a
long running intensive cpu job without the user's notice"): its progress -- games reviewed of
all, the game under way and its positions, the time left -- shown on the panel and in the Games
window, and paused or resumed there at will; paused stays paused across restarts. It also steps
aside by itself while the panel watches a game or a review room is analysing (the player's own
work first). A review stopped part way, by either or by the app closing, keeps its positions
(ui/review_room.py `review_row`) and goes on from there.
"""

from __future__ import annotations

import threading
import time
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
IDLE_S = 5.0  # between looks while waiting for the player's own work
REST_S = 30.0  # between looks when every game is reviewed
PAUSED_KEY = "backlog.paused"  # the store's settings: "1" while the player has it paused
# The estimate of the time left: the pace of this session's last games, once this many
# positions were measured; a game's positions until one is seen.
ETA_GAMES = 8
ETA_POSITIONS = 20
ETA_WARMUP = 5
TYPICAL_POSITIONS = 80


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
        self._wake = threading.Event()  # games came in, or the player resumed: look now
        self._failed: set[str] = set()  # games that cannot be reviewed, skipped this session
        self._store = Store(store_path)
        self._lock = threading.Lock()
        self.paused_by_you = self._store.setting(PAUSED_KEY) == "1"
        self.waiting_for_you = False  # stepped aside for the player's own game or review
        self.reviewing: str | None = None
        self._current: dict[str, Any] | None = None
        self.total = 0  # the library's games a review covers or will
        self.left = 0  # of those, the ones not reviewed yet
        # This session's reviews: positions measured and seconds each run took; each game's size.
        self._runs: list[tuple[int, float]] = []
        self._sizes: list[int] = []
        self._counted = 0.0  # when the job's games were last counted (monotonic)
        self._thread = threading.Thread(target=self._run, name="backlog", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._wake.set()

    def nudge(self) -> None:
        """Games came into the library: look for work now, not at the end of the rest (the
        Games window shows the progress from the first game on)."""
        self._counted = 0.0
        self._wake.set()

    def pause(self, paused: bool) -> dict[str, Any]:
        """Paused or resumed by the player; kept for the next start. A review under way stops
        within a position and keeps what it measured."""
        self.paused_by_you = paused
        self._store.set_setting(PAUSED_KEY, "1" if paused else "0")
        self._wake.set()
        return self.state()

    def join(self, timeout: float) -> None:
        """Until the review under way has stopped (its engines closed), at most `timeout`."""
        if self._thread.is_alive():
            self._thread.join(timeout)

    def state(self) -> dict[str, Any]:
        """What the panel and the Games window show: reviewing, paused (by the player), waiting
        (for the player's own game or review), done (every game reviewed), or empty (no games);
        games reviewed of all, the game under way with its positions, the seconds left."""
        with self._lock:
            current = dict(self._current) if self._current else None
            runs = list(self._runs[-ETA_GAMES:])
            sizes = list(self._sizes[-ETA_GAMES:])
        if current is not None:
            # The game under way counts once past its engines' start (a resumed one's first
            # positions come slow and would make the whole estimate jump).
            if current["done"] - current["start"] >= ETA_WARMUP:
                runs.append(
                    (current["done"] - current["start"], time.monotonic() - current["since"])
                )
            current = {k: v for k, v in current.items() if k not in ("start", "since")}
        if self.total == 0:
            status = "empty"
        elif self.left == 0:
            status = "done"
        elif self.paused_by_you:
            status = "paused"
        elif self.waiting_for_you:
            status = "waiting"
        else:
            status = "reviewing"
        # The time left: this session's pace a position measured (a game gone on from where it
        # was is quick, and would make a pace a game lie), times the positions left -- the game
        # under way's, and a typical game's for each of the others.
        eta = None
        measured = sum(n for n, _ in runs)
        if measured >= ETA_POSITIONS and self.left and status == "reviewing":
            per_position = sum(s for _, s in runs) / measured
            size = sum(sizes) / len(sizes) if sizes else TYPICAL_POSITIONS
            here = current["total"] - current["done"] if current and current["total"] else size
            eta = round(per_position * (here + (self.left - 1) * size))
        return {
            "status": status,
            "total": self.total,
            "done": self.total - self.left,
            "waiting": self.left,
            "reviewing": self.reviewing,
            "current": current,
            "eta_s": eta,
            "paused_by_you": self.paused_by_you,
            # The Games window's older reading: stepped aside for the player's own work.
            "paused": status == "waiting",
        }

    def _run(self) -> None:
        store = self._store
        try:
            while not self._stop.is_set():
                if self.paused_by_you or self._busy():
                    # Standing by, the counts kept fresh for the page now and then (reading the
                    # library is not free), at once when games come in.
                    self.waiting_for_you = not self.paused_by_you
                    if time.monotonic() - self._counted > REST_S:
                        self._todo(store)
                    self._wake.wait(IDLE_S)
                    self._wake.clear()
                    continue
                self.waiting_for_you = False
                if self._keep_saved(store):
                    continue
                todo = self._todo(store)
                if not todo:
                    self._wake.wait(REST_S)
                    self._wake.clear()
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
        stands for is the import's to review. Counts the job's games as it goes."""
        twins = library.reconcile(store)
        imported = store.library(limit=100_000)
        watched = [r for r in library.from_watched(store) if r.key not in twins]
        every = [*imported, *watched]
        unreviewed = {r.key for r in store.unreviewed()}
        todo = [
            r
            for r in every
            if (
                r.key in unreviewed if r.source != "watched" else store.latest_review(r.key) is None
            )
            and r.key not in self._failed
        ]
        self.total = len([r for r in every if r.key not in self._failed])
        self.left = len(todo)
        self._counted = time.monotonic()
        return sorted(todo, key=lambda r: r.played_at or r.added_at, reverse=True)

    def _review(self, store: Store, row: LibraryRow) -> None:
        # Stopped part way by the player (a pause, their own game or review) or the app closing:
        # it keeps what it measured and goes on from there later.
        halt = threading.Event()

        def progress(done: int, total: int) -> None:
            with self._lock:
                if self._current is not None:
                    self._current.update(done=done, total=total)
            if self.paused_by_you or self._busy() or self._stop.is_set():
                halt.set()

        strength = STRENGTHS[STRENGTH]
        # Positions kept from a run stopped before: counted as done, not in this run's pace.
        start = len(store.progress(row.key, strength.name, strength.depth, strength.multipv))
        self.reviewing = row.key
        with self._lock:
            self._current = {
                "key": row.key,
                "white": row.white,
                "black": row.black,
                "done": start,
                "total": 0,
                "start": start,
                "since": time.monotonic(),
            }
        try:
            done = review_row(
                self.settings,
                row,
                store,
                strength,
                workers=WORKERS,
                make_engine=self._make_engine,
                low_priority=True,
                on_progress=progress,
                stop=halt,
            )
            if done is not None:
                self.left = max(0, self.left - 1)
        except Exception:  # noqa: BLE001 - a game that cannot be reviewed is skipped
            self._failed.add(row.key)
        finally:
            self.reviewing = None
            with self._lock:
                run = self._current
                self._current = None
                if run is not None and run["done"] > run["start"]:
                    self._runs.append((run["done"] - run["start"], time.monotonic() - run["since"]))
                if run is not None and run["total"]:
                    self._sizes.append(run["total"])
