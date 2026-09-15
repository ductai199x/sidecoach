"""Frame-to-events driver: tracker, start-position bootstrap, recogniser, reconciler.

Used by replay and by the live watcher. The recogniser is rebuilt from every new start
position it sees, so a board theme or piece set changed between games is harmless, and the
orientation is decided per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from sidecoach.events import Event, GameEnded, GameStarted, Region, Side
from sidecoach.state.reconcile import Readings, Reconciler
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.pieces import (
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
    start_shaped,
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
    # The geometry the readings were taken with, and whether a changed frame went unread
    # because the lock was in doubt: either means the next verified frame is read again.
    _readings_key: tuple[Region, float] | None = None
    _unread_change: bool = False

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
            # unverified geometry would be noise for the reconciler. Whatever changed in
            # this frame is still unread.
            self._unread_change = self._unread_change or changed
            if self.reconciler is not None:
                self.reconciler.board_unseen()
            return []
        geometry = BoardGeometry(self.board.region, self.board.cell_px)
        key = (self.board.region, round(self.board.cell_px, 2))
        changed = (
            changed or self._unread_change or self._readings is None or key != self._readings_key
        )
        self._unread_change = False
        events: list[Event] = []
        rebuilt = False
        if changed:
            if self.model is not None:
                # The current recogniser reading the frame as this game's start position
                # means the user browsed back to it, or the next game has the same theme and
                # orientation; the reconciler tells those apart. Only a start-shaped frame
                # it cannot read that way, a new theme or the colours swapped, needs a
                # fresh recogniser.
                self._readings = self.model.classify_frame(frame, geometry)
                self._readings_key = key
                self._start_on_screen = start_shaped(self._readings) and not self._reads_as_start(
                    self._readings
                )
            else:
                self._start_on_screen = looks_like_start_position(frame, geometry)
        self._start_seen = self._start_seen + 1 if self._start_on_screen else 0
        if self._should_bootstrap():
            # Decided on every verified tick, changed or not, so that a start position
            # stored on one tick and left unchanged for the next few still counts.
            before = self.bootstraps
            events.extend(self._bootstrap(frame, geometry))
            rebuilt = self.bootstraps != before
            if rebuilt and self.model is not None:
                self._readings = self.model.classify_frame(frame, geometry)
                self._readings_key = key
                # The new recogniser reads this frame as its own start position.
                self._start_on_screen = False
        if self.model is None or self.reconciler is None or self._readings is None:
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

    def end_game(
        self, result: str, reason: str, ts: float, frame_id: int | None = None
    ) -> list[Event]:
        """End the game on the user's word, stamped like every other event."""
        if self.reconciler is None:
            return []
        return self._stamp(self.reconciler.end(result, reason), ts, frame_id)

    def reset(self) -> None:
        """Forget the game and the recogniser; the next start position bootstraps again."""
        self.model = None
        self.reconciler = None
        self.bottom = None
        self._readings = None
        self._readings_key = None
        self._unread_change = False
        self._start_seen = 0
        self._start_on_screen = False

    @staticmethod
    def _stamp(events: list[Event], ts: float, frame_id: int | None) -> list[Event]:
        for event in events:
            event.ts = ts
            event.source_frame_id = frame_id
        return events

    def _reads_as_start(self, readings: Readings) -> bool:
        """Whether the recogniser reads the frame as the start position it was built on.

        Nothing may contradict the start (a piece read as another, an empty square where a
        piece belongs, a piece where none belongs), bar one cell for a badge or a lifted
        piece; and most of the board must be readable at all, since a new theme reads as
        unknown almost everywhere without contradicting anything.
        """
        assert self.bottom is not None
        start = chess.Board()
        contradictions = 0
        unknown = 0
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                if reading.kind == "unknown":
                    unknown += 1
                    continue
                piece = start.piece_at(square_at(r, c, self.bottom))
                if piece is None:
                    bad = reading.kind == "piece" and reading.symbol is not None
                else:
                    bad = reading.kind == "empty" or (
                        reading.symbol is not None and reading.symbol != piece.symbol()
                    )
                contradictions += int(bad)
        return unknown < 32 and contradictions <= 1

    def _should_bootstrap(self) -> bool:
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
        previous = self.reconciler
        if not self._rebuild_model(frame, geometry):
            return []
        assert self.bottom is not None
        if previous is not None and not previous.in_progress and not previous.finished:
            # The game has not begun: the board was flipped, or re-themed, while the start
            # position was up (paired as black after watching the pre-game board). The
            # same game goes on, seen from the other side.
            return previous.reorient(self.bottom)
        events: list[Event] = []
        if previous is not None and previous.in_progress and not previous.finished:
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
