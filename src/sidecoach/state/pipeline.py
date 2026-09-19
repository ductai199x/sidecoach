"""Frame-to-events driver: tracker, start-position bootstrap, recogniser, reconciler.

Used by replay and by the live watcher. The recogniser is rebuilt from every new start
position it sees, so a board theme or piece set changed between games is harmless, and the
orientation is decided per game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from sidecoach.events import (
    ClockObservation,
    Event,
    GameEnded,
    GameStarted,
    Region,
    ScreenState,
    ScreenStateChanged,
    Side,
)
from sidecoach.state.reconcile import Readings, Reconciler
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import ClockReader, ClockReading
from sidecoach.vision.coordinates import learn_rank_labels, read_bottom_side
from sidecoach.vision.pieces import (
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
    start_shaped,
)
from sidecoach.vision.result_badges import read_result
from sidecoach.vision.squares import BoardGeometry, square_at
from sidecoach.vision.theme_cache import ThemeCache, sample_palette
from sidecoach.vision.tracker import BoardTracker

# Clocks. A reading is bottom/top relative to the board; the side at the bottom is the user,
# so the bottom box is the user's clock. chess.com draws white's box light and black's dark,
# but the board orientation is the game's own truth, so the colours are keyed from `bottom`
# rather than the box tint. Everything here tolerates a missing, hidden, or misread clock by
# producing nothing: a bad clock costs the clock, never the session.
CLOCK_MIN_CONFIDENCE = 0.4
# A cover of this many cells that stays this long over a live game is the game-over dialog.
# Chess.com's covers about thirty-five cells; a promotion picker four.
DIALOG_CELLS = 12
DIALOG_SECONDS = 2.0
# The site's result badges on the two kings end the game once they have been read on
# this many consecutive frames with the same result; a story of moves still pending is
# given this many more frames to settle first, so the record keeps the last move.
BADGE_FRAMES = 2
BADGE_PATIENCE = 10
# A loser whose clock read this little at the end lost on time.
FLAG_MS = 1000
CLOCK_MAX_MS = 3 * 60 * 60 * 1000  # three hours; anything above is a misread, not a clock


def _plausible_ms(ms: int | None) -> int | None:
    return ms if ms is not None and 0 <= ms <= CLOCK_MAX_MS else None


def clock_observation(reading: ClockReading | None, bottom: Side) -> ClockObservation | None:
    """Map a bottom-relative clock reading to a colour-keyed observation, or None.

    None whenever there is nothing trustworthy to report: no reading, confidence below the
    floor, or both boxes unreadable (an untimed game, the clocks scrolled off screen, a
    dialog drawn over them). A single unreadable box passes through as that side's time
    being unknown, which the consumer treats as no update for that side.
    """
    if reading is None or reading.confidence < CLOCK_MIN_CONFIDENCE:
        return None
    top = _plausible_ms(reading.top_ms)
    bottom_ms = _plausible_ms(reading.bottom_ms)
    if top is None and bottom_ms is None:
        return None
    if bottom is Side.WHITE:
        white_ms, black_ms = bottom_ms, top
    else:
        white_ms, black_ms = top, bottom_ms
    running: Side | None = None
    if reading.running == "bottom":
        running = bottom
    elif reading.running == "top":
        running = Side.BLACK if bottom is Side.WHITE else Side.WHITE
    return ClockObservation(white_ms=white_ms, black_ms=black_ms, running=running)


@dataclass
class GamePipeline:
    start_frames_needed: int = 2
    tracker: BoardTracker = field(default_factory=BoardTracker)
    model: ThemeModel | None = None
    reconciler: Reconciler | None = None
    bottom: Side | None = None
    board: BoardCandidate | None = None
    clock: ClockReader | None = None
    cache: ThemeCache | None = None
    bootstraps: int = 0
    _start_seen: int = 0
    _start_on_screen: bool = False
    _readings: Readings | None = None
    # The geometry the readings were taken with, and whether a changed frame went unread
    # because the lock was in doubt: either means the next verified frame is read again.
    _readings_key: tuple[Region, float] | None = None
    _unread_change: bool = False
    # The last clock observation emitted, so an unchanged reading is not repeated.
    _last_clock: tuple[int | None, int | None, Side | None] | None = None
    # The last screen state emitted, so a ScreenStateChanged fires only on a transition.
    _screen_state: ScreenState = ScreenState.UNKNOWN
    # When a dialog-sized cover first appeared over a live game's board, or None; and
    # whether one has stayed long enough to count as covering the board.
    _dialog_since: float | None = None
    _covered: bool = False
    # The result the site's badges showed on the last read frames, and for how many.
    _badges: tuple[str, int] | None = None

    def process(
        self, frame: np.ndarray, ts: float, changed: bool = True, frame_id: int | None = None
    ) -> list[Event]:
        """Events for one frame, stamped with its time and id.

        `changed` is False when the gated part of the frame did not change since the last
        stored frame; readings are then reused, but every tick still counts for the
        reconciler's timers and for the start-position count.
        """
        fresh = changed  # this tick carries new pixels: the moment to sample the clock
        self.board = self.tracker.update(frame)
        if self.board is None or not self.tracker.verified:
            # No board, or a lock the frame did not validate: readings taken from an
            # unverified geometry would be noise for the reconciler. Whatever changed in
            # this frame is still unread.
            self._unread_change = self._unread_change or changed
            if self.reconciler is not None:
                self.reconciler.board_unseen()
            return self._stamp(self._emit_screen_state(board_present=False), ts, frame_id)
        geometry = BoardGeometry(self.board.region, self.board.cell_px)
        key = (self.board.region, round(self.board.cell_px, 2))
        changed = (
            changed or self._unread_change or self._readings is None or key != self._readings_key
        )
        self._unread_change = False
        events: list[Event] = []
        if fresh:
            events += self._read_clock(frame, ts)
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
            events += self._emit_screen_state(board_present=True)
            return self._stamp(events, ts, frame_id)
        was_started = self.reconciler.started
        events += self.reconciler.update(self._readings, ts)
        if changed:
            events += self._end_on_badges(frame, geometry)
        events += self._dialog_over_the_board(self._readings, ts)
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
        events += self._emit_screen_state(board_present=True)
        return self._stamp(events, ts, frame_id)

    def _dialog_over_the_board(self, readings: Readings, ts: float) -> list[Event]:
        """A dialog that covers the board of a live game and stays.

        Chess.com puts its game-over dialog over the board the moment a game ends; other
        sites, and other moments, put other things there. So a cover is never taken as a
        result on its own. When the mating move was read once, clean, and the cover came
        before a second frame could confirm it, the cover is that confirmation (the third
        live game: five premoved plies in a second and a half, then the dialog). Otherwise
        the screen state becomes COVERED: the game stays open -- a new game, or the user's
        End game menu, ends it -- and the coach is quiet until the board is back. A cover
        that flashes past (a toast, a tooltip) counts for nothing.
        """
        rec = self.reconciler
        if rec is None or not rec.started or rec.finished:
            self._dialog_since = None
            self._covered = False
            return []
        covered = sum(1 for row in readings for reading in row if reading.covered)
        if covered < DIALOG_CELLS:
            self._dialog_since = None
            self._covered = False
            return []
        if self._dialog_since is None:
            self._dialog_since = ts
            return []
        if ts - self._dialog_since < DIALOG_SECONDS:
            return []
        if rec.pending_terminal() is not None:
            events = rec.commit_pending_terminal()
            if events:
                self._dialog_since = None
                return events
        self._covered = True
        return []

    def _end_on_badges(self, frame: np.ndarray, geometry: BoardGeometry) -> list[Event]:
        """The game ends on the site's result badges (vision/result_badges.py).

        Read at the two kings on every frame of a live game; the pair must agree with
        itself for BADGE_FRAMES frames. A mating story the badges confirm is committed
        first, so the record has the move; any other pending story is given a few frames
        to settle. The reason is what the pipeline knows: a mate or a stalemate on the
        board, a repetition or the material, a clock at zero; otherwise a resignation, or
        an agreed draw.
        """
        rec = self.reconciler
        if rec is None or not rec.started or rec.finished or self.bottom is None:
            self._badges = None
            return []
        result = read_result(frame, geometry, rec.board, self.bottom)
        if result is None:
            self._badges = None
            return []
        seen = self._badges[1] + 1 if self._badges is not None and self._badges[0] == result else 1
        self._badges = (result, seen)
        if seen < BADGE_FRAMES:
            return []
        pending = rec.pending_terminal()
        if pending is not None and pending[1].result() == result:
            events = rec.commit_pending_terminal()
            if events:
                return events
        if rec.pending_story and seen < BADGE_PATIENCE:
            return []
        return rec.end(result, self._end_reason(rec.board, result))

    def _end_reason(self, board: chess.Board, result: str) -> str:
        if result == "1/2-1/2":
            if board.is_stalemate():
                return "stalemate"
            if board.is_insufficient_material():
                return "insufficient material"
            if board.can_claim_threefold_repetition():
                return "repetition"
            if board.can_claim_fifty_moves():
                return "fifty-move rule"
            return "draw agreed"
        if board.is_checkmate():
            return "checkmate"
        if self._last_clock is not None:
            white_ms, black_ms, _ = self._last_clock
            loser_ms = black_ms if result == "1-0" else white_ms
            if loser_ms is not None and loser_ms <= FLAG_MS:
                return "lost on time"
        return "resignation"

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
        self._screen_state = ScreenState.UNKNOWN
        self._last_clock = None
        self._dialog_since = None
        self._covered = False
        self._badges = None
        if self.clock is not None:
            self.clock.reset()

    def _read_clock(self, frame: np.ndarray, ts: float) -> list[Event]:
        """A clock observation for this frame, or nothing.

        Never raises, never reports a clock the board is not oriented for, and repeats
        nothing: a missing, hidden, or misread clock simply produces no event, so the
        coach's cadence falls back to its untimed defaults rather than breaking.
        """
        if self.clock is None or self.bottom is None or self.board is None:
            return []
        try:
            reading = self.clock.read(frame, self.board, ts)
            observation = clock_observation(reading, self.bottom)
        except Exception:  # noqa: BLE001 - a clock must never take the pipeline down
            return []
        if observation is None:
            return []
        key = (observation.white_ms, observation.black_ms, observation.running)
        if key == self._last_clock:
            return []
        self._last_clock = key
        return [observation]

    def _derive_screen_state(self, board_present: bool) -> ScreenState:
        """The screen state from what the pipeline already knows: whether a board is
        locked, and where its game sits in its life. DESIGN's first slice reads it from
        board presence and the game lifecycle (running clocks refine live-versus-analysis
        later); there is no separate classifier.
        """
        if not board_present:
            return ScreenState.UNKNOWN  # no board: a menu, the lobby, or scrolled off
        rec = self.reconciler
        if rec is None or not rec.started:
            return ScreenState.UNKNOWN  # a board, but no game recognised on it yet
        if rec.finished:
            return ScreenState.GAME_OVER
        if self._covered:
            return ScreenState.COVERED
        return ScreenState.LIVE_GAME

    def _emit_screen_state(self, *, board_present: bool) -> list[Event]:
        """A ScreenStateChanged when the derived state differs from the last emitted."""
        state = self._derive_screen_state(board_present)
        if state == self._screen_state:
            return []
        self._screen_state = state
        return [ScreenStateChanged(state=state)]

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
        model.rank_labels = learn_rank_labels(frame, geometry, bottom)
        if self.cache is not None:
            self.cache.save(model)
        self._last_clock = None
        if self.clock is not None:
            self.clock.reset()
        if self.reconciler is not None:
            self.reconciler.bottom = bottom
            self.reconciler.user_side = bottom
        self.bootstraps += 1
        return True

    def cold_start(
        self, frame: np.ndarray, geometry: BoardGeometry, bottom: Side | None = None
    ) -> list[Event]:
        """Recognise the board cold from the theme cache and begin a game from the position on
        it -- a puzzle or a mid-game board, with no move-0 bootstrap. `bottom` may be supplied
        (the panel's flip control); otherwise it is read from the coordinate labels. Returns the
        GameStarted, or [] when no cached theme matches or the orientation cannot be determined.
        """
        if self.cache is None:
            return []
        sampled = sample_palette(frame, geometry)
        if sampled is None:
            return []
        model = self.cache.get(*sampled)
        if model is None:
            return []
        if bottom is None:
            bottom = read_bottom_side(frame, geometry, model.rank_labels)
        if bottom is None:
            return []  # coordinates are off; the caller supplies the orientation and retries
        readings = model.classify_frame(frame, geometry)
        placement = self._placement_from_readings(readings, bottom)
        if placement is None:
            return []
        reconciler = Reconciler(bottom=bottom, user_side=bottom)
        events = reconciler.start_from(placement)
        if not events:
            return []  # the placement is not a legal position; adopt nothing
        # Only now, with a legal position begun, is the pipeline switched onto this theme.
        self.model = model
        self.bottom = bottom
        self._readings = readings
        self._readings_key = (
            (self.board.region, round(self.board.cell_px, 2)) if self.board is not None else None
        )
        self.reconciler = reconciler
        return events

    @staticmethod
    def _placement_from_readings(readings: Readings, bottom: Side) -> chess.Board | None:
        """The classified board as a bare placement, or None unless it is a fully quiet,
        fully readable position with both kings on it.

        A cold read has no move history to lean on, so it must not adopt a board caught
        mid-move: a lifted piece empties its square and floats over another with the drag's
        hover outline, so a single frame read then would drop or misplace a piece (the c1
        bishop that read empty while it was in flight over f4). Every square must therefore
        be a settled piece or a settled empty with no hover and no unreadable cell -- the
        same quiet-board bar the reconciler holds a commit to -- or the read is refused and
        the caller retries on the next frame.
        """
        board = chess.Board(None)
        for row in range(8):
            for col in range(8):
                reading = readings[row][col]
                if reading.hover or reading.kind == "unknown" or not reading.settled:
                    return None  # a piece in flight, or a cell that cannot be read cleanly
                if reading.kind == "piece":
                    if reading.symbol is None:
                        return None  # a piece of no known kind: not a clean read
                    board.set_piece_at(
                        square_at(row, col, bottom), chess.Piece.from_symbol(reading.symbol)
                    )
        if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
            return None
        return board

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
