"""Frame-to-events driver: tracker, start-position bootstrap, recogniser, reconciler.

Used by replay and by the live watcher. The recogniser is rebuilt from every new start
position it sees, so a board theme or piece set changed between games is harmless, and the
orientation is decided per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sidecoach.events import Event, GameEnded, GameStarted, Side
from sidecoach.state.reconcile import Readings, Reconciler
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.pieces import (
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
)
from sidecoach.vision.squares import BoardGeometry
from sidecoach.vision.tracker import BoardTracker


@dataclass
class GamePipeline:
    start_frames_needed: int = 2
    tracker: BoardTracker = field(default_factory=BoardTracker)
    model: ThemeModel | None = None
    reconciler: Reconciler | None = None
    bottom: Side | None = None
    board: BoardCandidate | None = None
    bootstraps: int = 0
    _start_seen: int = 0
    _readings: Readings | None = None

    def process(self, frame: np.ndarray, ts: float, changed: bool = True) -> list[Event]:
        """Events for one frame. `changed` is False when the frame repeats the previous one."""
        self.board = self.tracker.update(frame)
        if self.board is None:
            return []
        geometry = BoardGeometry(self.board.region, self.board.cell_px)
        events: list[Event] = []
        rebuilt = False
        if changed or self._readings is None:
            if self._should_bootstrap(frame, geometry):
                events.extend(self._bootstrap(frame, geometry))
                rebuilt = True
            if self.model is None:
                return events
            self._readings = self.model.classify_frame(frame, geometry)
        if self.reconciler is None or self._readings is None:
            return events
        was_started = self.reconciler.started
        events += self.reconciler.update(self._readings, ts)
        if (
            was_started
            and not rebuilt
            and changed
            and any(isinstance(e, GameStarted) for e in events)
            and looks_like_start_position(frame, geometry)
        ):
            # The reconciler saw the start position come back with the same orientation.
            # The theme may still have changed between games, so rebuild the recogniser
            # from this frame and keep the reconciler it already reset.
            self._rebuild_model(frame, geometry)
        return events

    def _should_bootstrap(self, frame: np.ndarray, geometry: BoardGeometry) -> bool:
        if (
            self.model is not None
            and self.reconciler is not None
            and not self.reconciler.board.move_stack
        ):
            return False
        if not looks_like_start_position(frame, geometry):
            self._start_seen = 0
            return False
        self._start_seen += 1
        return self._start_seen >= self.start_frames_needed

    def _rebuild_model(self, frame: np.ndarray, geometry: BoardGeometry) -> None:
        self.bottom = orientation_from_start(frame, geometry)
        self.model = ThemeModel.from_start_position(frame, geometry, self.bottom)
        if self.reconciler is not None:
            self.reconciler.bottom = self.bottom
            self.reconciler.user_side = self.bottom
        self.bootstraps += 1
        self._start_seen = 0

    def _bootstrap(self, frame: np.ndarray, geometry: BoardGeometry) -> list[Event]:
        events: list[Event] = []
        if (
            self.reconciler is not None
            and self.reconciler.board.move_stack
            and not self.reconciler.finished
        ):
            events.append(
                GameEnded(
                    game_id=self.reconciler.game_id,
                    result="*",
                    reason="new game started",
                    pgn=self.reconciler.pgn(),
                    final_fen=self.reconciler.board.fen(),
                )
            )
        self._rebuild_model(frame, geometry)
        assert self.bottom is not None
        self.reconciler = Reconciler(bottom=self.bottom, user_side=self.bottom)
        return events
