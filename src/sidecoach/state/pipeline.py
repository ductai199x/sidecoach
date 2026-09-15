"""Frame-to-events driver: tracker, start-position bootstrap, recogniser, reconciler.

Used by replay and by the live watcher. The recogniser is rebuilt from every new start
position it sees, so a board theme or piece set changed between games is harmless, and the
orientation is decided per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from sidecoach.events import Event, GameEnded, GameStarted, Side
from sidecoach.state.reconcile import Readings, Reconciler, exact
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.pieces import (
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
)
from sidecoach.vision.squares import BoardGeometry, square_at
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
    _start_on_screen: bool = False
    _readings: Readings | None = None

    def process(
        self, frame: np.ndarray, ts: float, changed: bool = True, frame_id: int | None = None
    ) -> list[Event]:
        """Events for one frame, stamped with its time and id.

        `changed` is False when the gated part of the frame did not change since the last
        stored frame; readings are then reused, but every tick still counts for the
        reconciler's timers and for the start-position count.
        """
        self.board = self.tracker.update(frame)
        if self.board is None or not self.tracker.verified:
            # No board, or a lock the frame did not validate: readings taken from an
            # unverified geometry would be noise for the reconciler.
            return []
        geometry = BoardGeometry(self.board.region, self.board.cell_px)
        events: list[Event] = []
        rebuilt = False
        classified = False
        if changed or self._readings is None:
            self._start_on_screen = looks_like_start_position(frame, geometry)
            if self._start_on_screen and self.model is not None:
                # The current recogniser reading the frame as this game's start position
                # means the user browsed back to it, or the next game has the same theme and
                # orientation; the reconciler tells those apart. Only a frame it cannot read
                # that way, a new theme or the colours swapped, needs a fresh recogniser.
                self._readings = self.model.classify_frame(frame, geometry)
                classified = True
                if self._reads_as_start(self._readings):
                    self._start_on_screen = False
        if not self._start_on_screen:
            self._start_seen = 0
        else:
            self._start_seen += 1
        if changed or self._readings is None:
            if self._should_bootstrap():
                before = self.bootstraps
                events.extend(self._bootstrap(frame, geometry))
                rebuilt = self.bootstraps != before
            if self.model is None:
                return events
            if rebuilt or not classified:
                self._readings = self.model.classify_frame(frame, geometry)
        if self.reconciler is None or self._readings is None:
            return self._stamp(events, ts, frame_id)
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
        return self._stamp(events, ts, frame_id)

    def reset(self) -> None:
        """Forget the game and the recogniser; the next start position bootstraps again."""
        self.model = None
        self.reconciler = None
        self.bottom = None
        self._readings = None
        self._start_seen = 0
        self._start_on_screen = False

    @staticmethod
    def _stamp(events: list[Event], ts: float, frame_id: int | None) -> list[Event]:
        for event in events:
            event.ts = ts
            event.source_frame_id = frame_id
        return events

    def _reads_as_start(self, readings: Readings) -> bool:
        """Whether the current recogniser reads the frame as the start position it was built on."""
        assert self.bottom is not None
        start = chess.Board()
        off = 0
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                piece = start.piece_at(square_at(r, c, self.bottom))
                if piece is None:
                    good = reading.kind == "empty"
                else:
                    good = exact(reading) and reading.symbol == piece.symbol() and reading.settled
                if not good:
                    off += 1
        return off <= 1

    def _should_bootstrap(self) -> bool:
        if (
            self.model is not None
            and self.reconciler is not None
            and not self.reconciler.board.move_stack
            and self.reconciler.base_fen == chess.STARTING_FEN
        ):
            # This game has not started yet; the start position on screen is its own.
            return False
        return self._start_on_screen and self._start_seen >= self.start_frames_needed

    @property
    def readings(self) -> Readings | None:
        """What the recogniser read from the last changed frame."""
        return self._readings

    def _rebuild_model(self, frame: np.ndarray, geometry: BoardGeometry) -> bool:
        bottom = orientation_from_start(frame, geometry)
        model = ThemeModel.from_start_position(frame, geometry, bottom)
        self._start_seen = 0
        if model is None:
            return False  # something covered a piece; the next start frame gets a chance
        self.bottom = bottom
        self.model = model
        if self.reconciler is not None:
            self.reconciler.bottom = bottom
            self.reconciler.user_side = bottom
        self.bootstraps += 1
        return True

    def _bootstrap(self, frame: np.ndarray, geometry: BoardGeometry) -> list[Event]:
        ending = (
            self.reconciler is not None
            and self.reconciler.board.move_stack
            and not self.reconciler.finished
        )
        previous = self.reconciler
        if not self._rebuild_model(frame, geometry):
            return []
        assert self.bottom is not None
        events: list[Event] = []
        if ending and previous is not None:
            events.append(
                GameEnded(
                    game_id=previous.game_id,
                    result="*",
                    reason="new game started",
                    pgn=previous.pgn(),
                    final_fen=previous.board.fen(),
                )
            )
        self.reconciler = Reconciler(bottom=self.bottom, user_side=self.bottom)
        return events
