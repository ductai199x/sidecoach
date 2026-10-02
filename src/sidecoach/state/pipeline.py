"""Frame-to-events driver: tracker, start-position bootstrap, recogniser, reconciler.

Used by replay and by the live watcher. The recogniser is rebuilt from every new start
position it sees, so a board theme or piece set changed between games is harmless, and the
orientation is decided per game.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from typing import Literal

import chess
import numpy as np

from sidecoach.events import (
    BoardUnread,
    ClockObservation,
    Event,
    GameStarted,
    PositionChanged,
    Region,
    Resync,
    ScreenState,
    ScreenStateChanged,
    Side,
)
from sidecoach.state.reconcile import (
    Readings,
    Reconciler,
    by_square,
    highlighted_squares,
    loose_placement,
)
from sidecoach.vision.board_detect import BoardCandidate, refine, to_gray
from sidecoach.vision.clock import ClockReader, ClockReading
from sidecoach.vision.coordinates import (
    eights,
    learn_rank_labels,
    read_bottom_side,
    read_labels,
)
from sidecoach.vision.pieces import (
    PIECE_MIN_FRACTION,
    CellReading,
    ThemeModel,
    looks_like_start_position,
    mark_on_empty,
    occupancy,
    orientation_from_start,
    start_shaped,
)
from sidecoach.vision.result_badges import Badge, badge_on, result_of
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from sidecoach.vision.theme_cache import ThemeCache, same_theme, sample_palette
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
BADGE_PATIENCE = 12
# A loser whose clock read this little at the end lost on time.
FLAG_MS = 1000
# A resync or a cold start that left the side to move open takes it from the clock
# that is running, when the clock was read this recently (see Reconciler.turn_from_clock).
CLOCK_TURN_FRESH_S = 3.0
# A game joined from the middle -- Start pressed with a game already on the board -- is
# read cold from the theme cache: tried this often, and adopted once the same quiet,
# complete position has been read this many times, no two reads further apart than the
# expiry (a frame with a cell unreadable in between -- a tooltip, a label -- is skipped,
# not held against the position).
COLD_START_EVERY_S = 0.5
COLD_START_HOLD = 3
COLD_START_EXPIRE_S = 3.0
# A board that has gone this long without being read, with no game followed on it, is
# explained to the person once (`BoardUnread`) rather than left in silence.
COLD_NOTICE_S = 6.0
# A board no move explains has its coordinate labels read this often; two reads this far
# apart saying the other side is at the bottom turn the game round (`_flip_if_turned`).
FLIP_CHECK_EVERY_S = 0.25
CLOCK_MAX_MS = 3 * 60 * 60 * 1000  # three hours; anything above is a misread, not a clock
# A game whose board the recogniser no longer reads -- this many cells unknown (a new theme
# tints them all), or this many frames in a row no story explains (a new piece set read as
# other pieces) -- has its look learned again from its own position, tried this often
# (`_relearn_look`).
RELEARN_UNKNOWN_CELLS = 32
RELEARN_UNEXPLAINED_FRAMES = 2
RELEARN_EVERY_S = 0.5
RELEARN_HOLD_S = 0.5


def _fit(readings: Readings) -> float:
    """How surely a set read a board: the mean margin of its piece readings over the next
    nearest kind. The right set reads its own pieces well clear of any other kind."""
    margins = [cell.confidence for row in readings for cell in row if cell.kind == "piece"]
    return sum(margins) / len(margins) if margins else 0.0


def _misreads(readings: Readings, position: chess.Board, bottom: Side) -> int:
    """How many cells `readings` cannot show `position` on: unknown, empty where a piece
    stands, a piece where none does, or a piece of another known kind."""
    count = 0
    for r in range(8):
        for c in range(8):
            reading = readings[r][c]
            piece = position.piece_at(square_at(r, c, bottom))
            if reading.kind == "unknown":
                count += 1
            elif piece is None:
                count += int(reading.kind == "piece")
            else:
                count += int(
                    reading.kind == "empty"
                    or (reading.symbol is not None and reading.symbol != piece.symbol())
                )
    return count


def _on_its_board(frame: np.ndarray, board: BoardCandidate) -> bool:
    """Whether a re-fit of the lock finds the board where the lock is, to within the
    tracker's deadband, and validated."""
    refit = refine(to_gray(frame), board).board
    shift = max(
        abs(refit.region.x - board.region.x),
        abs(refit.region.y - board.region.y),
        abs(refit.region.x + refit.region.w - board.region.x - board.region.w),
        abs(refit.region.y + refit.region.h - board.region.y - board.region.h),
    )
    return refit.fit >= 0.85 and shift <= 2


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
        running = bottom.other
    return ClockObservation(white_ms=white_ms, black_ms=black_ms, running=running)


# A board read cold: the set that read it, the side at the bottom, the readings, the placement.
_ColdRead = tuple[ThemeModel, Side, Readings, chess.Board]
# What a read shows: the placement and the squares the site highlights.
_View = tuple[str, frozenset[chess.Square]]


def _view(read: _ColdRead) -> _View:
    """A board is adopted only once every read that confirms it shows the same view: the
    same pieces and the same highlighted squares. A site's move lights its squares before
    the piece leaves, so a read taken as the next puzzle's setup move began matched the one
    before it square for square; the board was adopted with the piece still on its origin,
    the highlight said the other side had just moved, and the setup move that followed
    could not be explained (Tai's desktop, 2026-09-24: 1...Qxh7 2.Qb1 invented, dump
    20260924-165745)."""
    _model, bottom, readings, placement = read
    return placement.board_fen(), highlighted_squares(by_square(readings, bottom))


# What `GamePipeline.reset` keeps: the settings, the tracker and its lock, the side the
# person told, the clock reader, the cache, and the count of recognisers built.
_RESET_KEEPS = frozenset(
    {"start_frames_needed", "tracker", "forced_bottom", "board", "clock", "cache", "bootstraps"}
)


@dataclass
class GamePipeline:
    start_frames_needed: int = 2
    tracker: BoardTracker = field(default_factory=BoardTracker)
    # The side at the bottom for a board read cold whose coordinate labels say nothing, when
    # set (the panel's Flip); labels that read win over it.
    forced_bottom: Side | None = None
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
    # The last clock observation emitted, so an unchanged reading is not repeated, and
    # when it was read.
    _last_clock: tuple[int | None, int | None, Side | None] | None = None
    _last_clock_ts: float | None = None
    # The automatic cold start: when it was last tried, and the placement it read then
    # with how many times it has read the same and when it last did.
    _cold_tried: float | None = None
    _cold_candidate: tuple[_View, int, float] | None = None
    # The last cold read and what it was read from (the lock and the side told), so pixels
    # that have not changed are not read again; since when no cold read has succeeded on
    # this lock, and whether the person has been told.
    _cold_last: tuple[tuple[Region, float, Side | None], _ColdRead | None] | None = None
    _cold_since: float | None = None
    _cold_noticed: bool = False
    # The lock the start check last ran on with no model: unchanged pixels on it look the same.
    _start_checked: tuple[Region, float] | None = None
    # The other side the labels last said was at the bottom of an unexplained board, and when.
    # The side the labels say, when, and the position then read that way round.
    _flip_seen: tuple[Side, float, _View | None] | None = None
    # The last screen state emitted, so a ScreenStateChanged fires only on a transition.
    _screen_state: ScreenState = ScreenState.UNKNOWN
    # When a dialog-sized cover first appeared over a live game's board, or None; and
    # whether one has stayed long enough to count as covering the board.
    _dialog_since: float | None = None
    _covered: bool = False
    # When the look was last learned again mid-game, or tried (`_relearn_look`).
    _relearn_tried: float | None = None
    # The lock and the readings the look was last read broken with, and since when: it must
    # read broken the same way for RELEARN_HOLD_S.
    _relearn_seen: tuple[object, float] | None = None
    # The result the site's badges showed on the last read frames, and for how many; and
    # what the last read frame showed: the result and the two kings' squares.
    _badges: tuple[str, int] | None = None
    _badge_frame: tuple[str, chess.Square, chess.Square] | None = None

    def process(
        self, frame: np.ndarray, ts: float, changed: bool = True, frame_id: int | None = None
    ) -> list[Event]:
        """Events for one frame, stamped with its time and id.

        `changed` is False when the gated part of the frame did not change since the last
        stored frame; readings are then reused, but every tick still counts for the
        reconciler's timers and for the start-position count.
        """
        fresh = changed  # this tick carries new pixels: the moment to sample the clock
        self.board = self.tracker.update(frame, changed=changed)
        if self.board is None or not self.tracker.verified:
            # No board, or a lock the frame did not validate: readings taken from an
            # unverified geometry would be noise for the reconciler. Whatever changed in
            # this frame is still unread.
            self._unread_change = self._unread_change or changed
            if self.reconciler is not None:
                self.reconciler.board_unseen()
            # A board that comes back is waited for, and explained, afresh.
            self._cold_last = None
            self._cold_since = None
            self._cold_noticed = False
            self._start_checked = None
            return self._stamp(self._emit_screen_state(board_present=False), ts, frame_id)
        geometry = BoardGeometry(self.board.region, self.board.cell_px)
        key = (self.board.region, round(self.board.cell_px, 2))
        new_pixels = fresh or self._unread_change
        if new_pixels:
            # These pixels are not the ones last read cold, whether or not this tick reads.
            self._cold_last = None
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
            elif new_pixels or key != self._start_checked:
                # With no model, unchanged pixels on the same lock look the same: not
                # checked again on every tick a board sits there unread (a quarter of a
                # second a frame on the phone).
                self._start_on_screen = looks_like_start_position(frame, geometry)
                self._start_checked = key
        if self.model is not None and not self._start_on_screen:
            # On every verified tick, changed or not: a new look must stay a moment.
            self._relearn_look(frame, geometry, ts)
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
        if self.model is None and changed and not self._start_on_screen:
            events += self._auto_cold_start(frame, geometry, ts)
        if self.model is None or self.reconciler is None or self._readings is None:
            events += self._emit_screen_state(board_present=True)
            return self._stamp(events, ts, frame_id)
        was_started = self.reconciler.started
        if changed:
            self._badge_frame = self._badged_kings(frame, geometry)
        events += self._resync_reoriented(
            self.reconciler.update(self._readings, ts), frame, geometry, ts
        )
        if not any(isinstance(e, Resync | GameStarted) for e in events):
            events += self._flip_if_turned(frame, geometry, ts)
        events += self._turn_from_clock(ts)
        if changed:
            events += self._end_on_badges()
        events += self._dialog_over_the_board(self._readings, ts)
        if (
            was_started
            and not rebuilt
            and changed
            and any(isinstance(e, GameStarted) for e in events)
            and start_shaped(self._readings)
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

    def _badged_kings(
        self, frame: np.ndarray, geometry: BoardGeometry
    ) -> tuple[str, chess.Square, chess.Square] | None:
        """The site's result badges on the kings (vision/result_badges.py): the result and
        the two squares they sit on, or None.

        The kings are looked for where the reconciler has them and, when the last frame
        left something unexplained, on the neighbouring squares as well: the mate often
        lands with the mated king's own move (a premoved king step, then the mate), and
        the site stamps the badge and the check glow on that king in the same frame, so
        its cell reads as an unsettled piece of no known kind and the reconciler holds
        as for a drag. A badge names the king's square: the reading there is replaced by
        a settled king, and the story confirms on the clean frames instead of waiting for
        the dialog. Covered cells are never read: a green badge under a half-faded dialog
        is a grey one.
        """
        rec = self.reconciler
        readings = self._readings
        if rec is None or not rec.started or rec.finished or self.bottom is None:
            return None
        if readings is None:
            return None
        bottom = self.bottom
        wide = rec.pending_story or rec.blocked or rec.unexplained

        def reading_at(square: chess.Square) -> CellReading:
            r, c = row_col(square, bottom)
            return readings[r][c]

        def candidates(colour: chess.Color) -> list[chess.Square]:
            king = rec.board.king(colour)
            if king is None:
                return []
            out = [king]
            if wide:
                out += [
                    sq
                    for sq in chess.SQUARES
                    if sq != king and chess.square_distance(sq, king) == 1
                ]
            return out

        def badged(colour: chess.Color) -> list[tuple[chess.Square, Badge]]:
            found = []
            for sq in candidates(colour):
                reading = reading_at(sq)
                if reading.covered or reading.kind == "empty":
                    continue  # a label runs over the neighbours; the king's square is not empty
                badge = badge_on(frame, geometry, sq, bottom)
                if badge is not None:
                    found.append((sq, badge))
            return found

        whites = badged(chess.WHITE)
        if not whites:
            return None
        blacks = badged(chess.BLACK)
        pairs = [
            (result, wk, bk)
            for wk, wb in whites
            for bk, bb in blacks
            if wk != bk and (result := result_of(wb, bb)) is not None
        ]
        if len(pairs) != 1:
            return None
        result, wk, bk = pairs[0]
        for square, symbol, colour in ((wk, "K", Side.WHITE), (bk, "k", Side.BLACK)):
            r, c = row_col(square, bottom)
            reading = readings[r][c]
            if reading.kind == "empty" or reading.covered:
                continue  # a badge beside an empty square belongs to a neighbour
            if reading.kind == "piece" and reading.symbol == symbol and reading.settled:
                continue
            # What is drawn on the square is the badge, and it names the king: not a
            # mark to wait out, not a drag.
            readings[r][c] = replace(
                reading,
                kind="piece",
                symbol=symbol,
                colour=colour,
                confidence=1.0,
                settled=True,
                hover=False,
                drawn=False,
                guess=symbol,
            )
        return result, wk, bk

    def _end_on_badges(self) -> list[Event]:
        """The game ends on the site's result badges when the board could not end it:
        the pair must agree with itself for BADGE_FRAMES frames. A mating story still
        pending that the badges confirm is committed first, so the record has the move;
        a frame the record has not caught up with is given a few frames to be told. The
        reason is what the pipeline knows: a mate on the board, a clock at zero;
        otherwise a resignation.
        """
        rec = self.reconciler
        if rec is None or not rec.started or rec.finished or self._badge_frame is None:
            self._badges = None
            return []
        result = self._badge_frame[0]
        seen = self._badges[1] + 1 if self._badges is not None and self._badges[0] == result else 1
        self._badges = (result, seen)
        if seen < BADGE_FRAMES:
            return []
        pending = rec.pending_terminal()
        if pending is not None and pending[1].result() == result:
            events = rec.commit_pending_terminal()
            if events:
                return events
        if not rec.agrees and seen < BADGE_PATIENCE:
            # The screen is ahead of the record -- the mating move is on the board and
            # not told yet: its story pending, the badge's spill on its cells still to
            # turn stale, or a premove's reply just committed with the mate behind it --
            # so the record gets a few frames to catch up before the badges end the
            # game without the move.
            return []
        return rec.end(result, self._end_reason(rec.board, result))

    def _end_reason(self, board: chess.Board, result: str) -> str:
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
        """Forget the game and the recogniser; the next start position bootstraps again.
        Every field returns to its default but what `_RESET_KEEPS` names."""
        for f in fields(self):
            if f.name not in _RESET_KEEPS:
                default = f.default_factory() if f.default_factory is not MISSING else f.default
                setattr(self, f.name, default)
        if self.clock is not None:
            self.clock.reset()

    def _forget_clock(self) -> None:
        """The last clock observation and the reader's memory dropped: the board's sides
        changed under them."""
        self._last_clock = None
        self._last_clock_ts = None
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
        self._last_clock_ts = ts
        if key == self._last_clock:
            return []
        self._last_clock = key
        return [observation]

    def _resync_reoriented(
        self, events: list[Event], frame: np.ndarray, geometry: BoardGeometry, ts: float
    ) -> list[Event]:
        """A position resync checked against the coordinate labels.

        A resync keeps the orientation the game had; a puzzle page shows each puzzle from
        the solver's side, so the next puzzle can come up the other way round and read as
        a position turned on its head (a white king on a7, 2026-09-20). When the labels
        say the other side is at the bottom, the board is read again that way and a new
        game begun from it, in place of the resync's events. A position adopted after a
        finished game is a game begun, with no Resync (`Reconciler._maybe_position_resync`),
        and is checked the same way."""
        if self.model is None or not any(
            isinstance(e, Resync) or (isinstance(e, GameStarted) and e.kind == "position")
            for e in events
        ):
            return events
        labelled = read_bottom_side(frame, geometry, self.model.rank_labels)
        if labelled is None or labelled == self.bottom:
            return events
        # A resync after a finished game began a new one on its head: that start goes
        # unsaid, and the game to end is the finished one, already ended.
        began = any(isinstance(e, GameStarted) for e in events)
        reoriented = self.reorient(frame, geometry, labelled, ts, end_previous=not began)
        if not reoriented:
            return events
        kept = [e for e in events if not isinstance(e, Resync | PositionChanged | GameStarted)]
        return kept + reoriented

    def _flip_if_turned(self, frame: np.ndarray, geometry: BoardGeometry, ts: float) -> list[Event]:
        """A board no move explains whose coordinate labels say the other side is at the
        bottom: the next puzzle, shown from its solver's side. It is read that way at once,
        as a new game, from the readings already taken (`reorient`).

        It used to wait for a resync to adopt the board on its head -- ten still frames,
        a second and a half on the phone -- and then read all 64 cells again the right way
        round, another second and a half; and a board that on its head was no legal
        position waited for the long unexplained-board note, which after it stopped being
        a Resync (the takeback fix) turned nothing round at all (Tai's puzzles on the Pixel,
        2026-09-23: five to ten seconds, and stuck). The labels cost a few milliseconds;
        two reads a moment apart must agree, so a piece dragged over the edge is not taken
        for the board turning. The labels are read even after a Flip: the side the person
        chose is for a board they say nothing about, and a Flip that stood over them read
        the next puzzle from the other side on its head, where no move was ever found
        (Tai, 2026-09-27: "The board white or black should be automatically read ... every
        time there's a sync or resync")."""
        rec = self.reconciler
        if rec is None or self.model is None or rec.unexplained_frames < 1:
            self._flip_seen = None
            return []
        seen = self._flip_seen
        if seen is not None and ts - seen[1] < FLIP_CHECK_EVERY_S:
            return []
        labelled = read_bottom_side(frame, geometry, self.model.rank_labels)
        if labelled is None or labelled == self.bottom:
            self._flip_seen = None
            return []
        if seen is None or seen[0] != labelled:
            self._flip_seen = (labelled, ts, None)
            return []
        # The position must read the same twice too: a piece still sliding in reads as two
        # empty squares, and a board adopted then lacks it for good -- the next puzzle's setup
        # move caught mid-slide at the phone's two frames a second, and nothing after it was
        # recorded (Tai, 2026-09-24: "sometimes there is no move list whatsoever").
        read = self._reread(labelled) or self._cold_read(frame, geometry, labelled)
        placed = _view(read) if read is not None else None
        if placed is None or placed != seen[2]:
            self._flip_seen = (labelled, ts, placed)
            return []
        self._flip_seen = None
        return self.reorient(frame, geometry, labelled, ts, read=read)

    def _reread(self, bottom: Side) -> _ColdRead | None:
        """The frame's own readings taken with `bottom` at the bottom: the recogniser read
        cells, not squares, so turning the board round needs no second read of its 64
        cells -- a second and a half on the phone. None when the readings are not this
        frame's, or not a quiet, complete, legal position that way round."""
        if self.model is None or self._readings is None or self.board is None:
            return None
        if self._readings_key != (self.board.region, round(self.board.cell_px, 2)):
            return None
        placement = self._placement_from_readings(self._readings, bottom)
        if placement is None or Reconciler._adopt_placement(placement) is None:
            return None
        return self.model, bottom, self._readings, placement

    def reorient(
        self,
        frame: np.ndarray,
        geometry: BoardGeometry,
        bottom: Side,
        ts: float,
        *,
        end_previous: bool = True,
        read: _ColdRead | None = None,
        told: bool = False,
    ) -> list[Event]:
        """Read the board again with `bottom` at the bottom and begin again from what it
        shows: the panel's Flip on a game joined from the middle, or a resync whose labels
        say the other side (see `_resync_reoriented`). A game under way ends first. Nothing
        happens when the board cannot be read cleanly now (a piece in flight): the caller
        may set `forced_bottom` and forget the game, and the automatic cold start reads it
        that way on the next quiet frame. The frame's own readings serve when they can
        (`_reread`); a cold read from the theme cache otherwise. `told`: the person said
        which side is at the bottom (Flip), and the labels learn from them where they did
        not say it themselves (`_learn_labels`)."""
        if read is None:
            read = self._reread(bottom) or self._cold_read(frame, geometry, bottom)
        if read is None:
            return []
        events = self._end_in_progress("board reoriented") if end_previous else []
        events += self._adopt_cold(*read, ts=ts)
        if events:
            self._learn_labels(frame, geometry, told=told)
        return events

    def _learn_labels(self, frame: np.ndarray, geometry: BoardGeometry, *, told: bool) -> None:
        """The coordinate labels learned from a board just adopted, when they say no side
        there -- learned where a start position was seen through a lock off its board, or
        not learned at all -- and the set saved with them, so the next board of the set reads
        its own side. Only a side known for this board teaches: told for it (`told`, a Flip),
        or read from its 8 (`eights`). A Flip left standing from the last puzzle would teach
        the next one, shown from the other side, upside down, and every read after it.
        Labels that read a side are kept, the Flip's or the other: learned as they read their
        start back, they are right, and a Flip by mistake would have taught every board of
        the set upside down."""
        model, bottom = self.model, self.bottom
        if model is None or bottom is None:
            return
        if read_labels(frame, geometry, model.rank_labels) is not None:
            return
        if not told and eights(frame, geometry) != bottom:
            return
        labels = learn_rank_labels(frame, geometry, bottom)
        if not labels:
            return
        model.rank_labels = labels
        vectors = model.exemplar_vectors
        if self.cache is None or model.light_bg is None or model.dark_bg is None or vectors is None:
            return
        for key, cached in self.cache.candidates(model.light_bg, model.dark_bg):
            if cached.exemplar_vectors is not None and np.array_equal(
                cached.exemplar_vectors, vectors
            ):
                self.cache.save(model, key=key)
                return

    def _end_in_progress(self, reason: str) -> list[Event]:
        previous = self.reconciler
        if previous is None or not previous.in_progress or previous.finished:
            return []
        return [previous.ended_event("*", reason)]

    def _turn_from_clock(self, ts: float) -> list[Event]:
        """A side to move left open by a resync or a cold start, settled by the clock
        that is running -- read on this tick or a moment ago."""
        rec = self.reconciler
        if rec is None or not rec.turn_unknown or self._last_clock is None:
            return []
        running = self._last_clock[2]
        if running is None or self._last_clock_ts is None:
            return []
        if ts - self._last_clock_ts > CLOCK_TURN_FRESH_S:
            return []
        return rec.turn_from_clock(running.color)

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

    def _reads_as_start(self, readings: Readings, bottom: Side | None = None) -> bool:
        """Whether the recogniser reads the frame as the start position it was built on,
        with `bottom` (the game's, by default) at the bottom.

        Nothing may contradict the start (a piece read as another, an empty square where a
        piece belongs, a piece where none belongs), bar one cell for a badge or a lifted
        piece; and most of the board must be readable at all, since a new theme reads as
        unknown almost everywhere without contradicting anything.
        """
        bottom = self.bottom if bottom is None else bottom
        assert bottom is not None
        start = chess.Board()
        contradictions = 0
        unknown = 0
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                if reading.kind == "unknown":
                    unknown += 1
                    continue
                piece = start.piece_at(square_at(r, c, bottom))
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
            self.cache.save(model, key=self._learned_set(model, frame, geometry, bottom))
        self._cold_since = None
        self._cold_noticed = False
        self._forget_clock()
        if self.reconciler is not None:
            self.reconciler.bottom = bottom
            self.reconciler.user_side = bottom
        self.bootstraps += 1
        return True

    def _relearn_look(self, frame: np.ndarray, geometry: BoardGeometry, ts: float) -> None:
        """The recogniser learned again, mid-game, when the board's look changed under it.

        A theme or a piece set changed mid-game left the rest of the game unread, with no
        event: the recogniser was rebuilt only from a start position, and a new theme reads
        as unknown almost everywhere, a new set as other pieces (round 2, finding 2). The
        game's own position says what is on the board: the record's position, or one legal
        move on from it (played before the new look was read). One whose occupied squares
        are the frame's, which the recogniser reads as neither it nor anything near it, is
        learned from (`ThemeModel.from_position`, validated before it is used), and these
        readings are taken again with it. A board showing anything else -- another game,
        a dialog, a piece in flight -- matches no such position and is left alone. The model
        learned is not cached: it lacks the kinds the position has lost, and the next start
        position rebuilds and caches the whole set."""
        rec, model, bottom, readings = self.reconciler, self.model, self.bottom, self._readings
        if rec is None or model is None or bottom is None or readings is None:
            return
        if not rec.started or rec.finished or start_shaped(readings):
            return  # a start position is the bootstrap's to learn from, and to cache
        unknown = sum(1 for row in readings for reading in row if reading.kind == "unknown")
        # Sampled only when it may matter: this runs on every still tick.
        sampled = sample_palette(frame, geometry) if unknown >= RELEARN_UNKNOWN_CELLS else None
        # A new theme is new square colours: half the board unknown on the colours the
        # recogniser knows is a lock a few pixels off its board, before the re-fit caught up
        # (the geometry sweep's seed 175), and a look learned there read nothing after.
        new_theme = (
            unknown >= RELEARN_UNKNOWN_CELLS
            and sampled is not None
            and model.light_bg is not None
            and model.dark_bg is not None
            and not same_theme(*sampled, model.light_bg, model.dark_bg)
        )
        dragging = any(reading.hover for row in readings for reading in row)
        if dragging or (not new_theme and rec.unexplained_frames < RELEARN_UNEXPLAINED_FRAMES):
            self._relearn_seen = None
            return
        # Only a look that stays broken, read the same on one lock for RELEARN_HOLD_S: a new
        # look stays, while a piece in flight, a premove's ghost or a lock a few pixels off
        # for a frame pass. Taken at two or three frames, a capture's animation was learned
        # as the new look, and the next moves went unread (the stress sweep's seeds 59, 60,
        # 194 and the geometry sweep's 133).
        seen = (
            self._readings_key,
            tuple((reading.kind, reading.symbol) for row in readings for reading in row),
        )
        if self._relearn_seen is None or self._relearn_seen[0] != seen:
            self._relearn_seen = (seen, ts)
            return
        if ts - self._relearn_seen[1] < RELEARN_HOLD_S:
            return
        if self._relearn_tried is not None and ts - self._relearn_tried < RELEARN_EVERY_S:
            return
        self._relearn_tried = ts
        # The recogniser reads the frame as a position of this game, earlier or one move on:
        # a move being taken back or browsed, a story pending -- not a new look (the stress
        # sweep's seeds 16, 22 and 23: a rook's capture shown undone for two frames).
        history = rec.board.copy()
        base = history.copy(stack=False)
        positions = [base]
        for move in base.legal_moves:
            if move.promotion not in (None, chess.QUEEN):
                continue
            after = base.copy(stack=False)
            after.push(move)
            positions.append(after)
        earlier: list[chess.Board] = []
        while history.move_stack:
            history.pop()
            earlier.append(history.copy(stack=False))
        if any(_misreads(readings, p, bottom) < 2 for p in positions + earlier):
            return
        # Learned only on a lock that sits on its board to the pixel: a board in the middle
        # of a nudge, the lock a few pixels off it, read its squares mixed and half the
        # board as a new theme, and a look learned there read nothing after (the geometry
        # sweep's seed 25).
        if self.board is None or not _on_its_board(frame, self.board):
            return
        occupied = occupancy(frame, geometry) >= PIECE_MIN_FRACTION
        if sampled is None:
            sampled = sample_palette(frame, geometry)
        for position in positions:
            expected = np.array(
                [[position.piece_at(square_at(r, c, bottom)) is not None for c in range(8)]
                 for r in range(8)]
            )  # fmt: skip
            if (occupied != expected).any():
                continue
            # A set learned before on these colours, whole, that reads the position.
            cached_reads = [
                (cached, cached_readings)
                for _key, cached in (
                    self.cache.candidates(*sampled)
                    if self.cache is not None and sampled is not None
                    else []
                )
                if _misreads(
                    cached_readings := cached.classify_frame(frame, geometry), position, bottom
                )
                <= 1
            ]
            if cached_reads:
                self.model, self._readings = cached_reads[0]
                return
            learned = ThemeModel.from_position(frame, geometry, bottom, position)
            if learned is None:
                continue
            learned.rank_labels = learn_rank_labels(frame, geometry, bottom) or model.rank_labels
            self.model = learned
            self._readings = learned.classify_frame(frame, geometry)
            return

    def _learned_set(
        self, model: ThemeModel, frame: np.ndarray, geometry: BoardGeometry, bottom: Side
    ) -> str | None:
        """The cached set this start position shows again -- the same pieces on the same
        board, learned before -- or None for a set not seen yet. Each set cached on these
        board colours reads the frame; one that reads it as the start position is this set,
        and the new model replaces it rather than standing beside it."""
        assert self.cache is not None
        if model.light_bg is None or model.dark_bg is None:
            return None
        for key, cached in self.cache.candidates(model.light_bg, model.dark_bg):
            if self._reads_as_start(cached.classify_frame(frame, geometry), bottom):
                return key
        return None

    def _cold_read(
        self,
        frame: np.ndarray,
        geometry: BoardGeometry,
        bottom: Side | None,
        *,
        asked: bool = False,
        fallback: Side | None = None,
    ) -> tuple[ThemeModel, Side, Readings, chess.Board] | None:
        """The board read cold from the theme cache: a cached recogniser for its colours,
        the orientation (`bottom`, else the coordinate labels, else `fallback` -- the side
        a Flip chose, for a board whose labels say nothing), the readings and the
        placement they make -- or None when no learned set gives all of those.

        One board's colours can carry several learned sets. Each reads the board; a set
        whose read is not a quiet, complete, legal position is passed over (a set for other
        pieces misreads them: a king taken for a bishop), and of those that give one the
        surest read wins, so a set learned for other pieces on the same colours never
        stands in for the right one (Tai's Pixel, 2026-09-23: a test board's set, cached
        under chess.com's green, read his puzzle's pawns as the other colour's)."""
        if self.cache is None:
            return None
        sampled = sample_palette(frame, geometry)
        if sampled is None:
            return None
        best: tuple[float, tuple[ThemeModel, Side, Readings, chess.Board]] | None = None
        for _key, model in self.cache.candidates(*sampled):
            side = bottom or read_bottom_side(frame, geometry, model.rank_labels) or fallback
            if side is None:
                continue  # coordinates off, or not this set's; the caller may say the side
            readings = model.classify_frame(frame, geometry)
            placement = self._placement_from_readings(readings, side, asked=asked)
            if placement is None or Reconciler._adopt_placement(placement) is None:
                continue
            fit = _fit(readings)
            if best is None or fit > best[0]:
                best = (fit, (model, side, readings, placement))
        return None if best is None else best[1]

    def _auto_cold_start(
        self, frame: np.ndarray, geometry: BoardGeometry, ts: float
    ) -> list[Event]:
        """A game joined from the middle, without being asked.

        Start pressed with a game already under way used to wait for a start position
        that was not coming (Tai, 2026-09-19: "shouldn't it just continue from the
        position on the board?"). With the theme in the cache the board reads cold; the
        position is adopted once it has read the same, quiet and complete, on
        COLD_START_HOLD tries COLD_START_EVERY_S apart, so a board mid-animation or a
        page passing by is never taken for a game. A start position is left to the
        bootstrap, which learns the theme from it. The side to move comes from the
        running clock (`_turn_from_clock`) or from the first move seen.

        Pixels that have not changed since the last read, on the same lock with the same
        side told, read the same, so that read stands for this try: a board left unread --
        a puzzle of a set not learned -- costs nothing while it sits there. New pixels drop
        it (`process`), on a tick that tries or not. A board that stays unread is explained
        once (`_cold_notice`).
        """
        if self.cache is None:
            return []
        if self._cold_tried is not None and ts - self._cold_tried < COLD_START_EVERY_S:
            return []
        self._cold_tried = ts
        assert self.board is not None
        seen = (self.board.region, round(self.board.cell_px, 2), self.forced_bottom)
        if self._cold_last is None or self._cold_last[0] != seen:
            read = self._cold_read(frame, geometry, None, fallback=self.forced_bottom)
            self._cold_last = (seen, read)
        else:
            read = self._cold_last[1]
        if read is None:
            # Nothing read cleanly this time; the candidate stands until it expires.
            return self._cold_notice(frame, geometry, ts)
        self._cold_since = None
        view = _view(read)
        candidate = self._cold_candidate
        if (
            candidate is not None
            and candidate[0] == view
            and ts - candidate[2] <= COLD_START_EXPIRE_S
        ):
            candidate = (view, candidate[1] + 1, ts)
        else:
            candidate = (view, 1, ts)
        self._cold_candidate = candidate
        if candidate[1] < COLD_START_HOLD:
            return []
        self._cold_candidate = None
        events = self._adopt_cold(*read, ts=ts)
        if events:
            self._learn_labels(frame, geometry, told=False)
        return events

    def _cold_notice(self, frame: np.ndarray, geometry: BoardGeometry, ts: float) -> list[Event]:
        """A board that has stayed unread for COLD_NOTICE_S is explained, once while it
        stays: the person saw nothing happen and could not know why (Tai's puzzle on the
        Pixel, 2026-09-23). Nothing is said when a learned set does read the board and only
        a passing thing stands in the way -- a piece held, a selected piece's dots -- and
        the wait starts again."""
        if self._cold_since is None:
            self._cold_since = ts
            return []
        if self._cold_noticed or ts - self._cold_since < COLD_NOTICE_S:
            return []
        reason = self._unread_reason(frame, geometry)
        if reason is None:
            self._cold_since = ts
            return []
        self._cold_noticed = True
        return [BoardUnread(reason=reason)]

    def _unread_reason(
        self, frame: np.ndarray, geometry: BoardGeometry
    ) -> Literal["unlearned", "orientation"] | None:
        """Why a board goes unread: "orientation" when a learned set reads it as a position
        but the coordinates say nothing (turned off), so Flip must; "unlearned" when no
        learned set reads it as one -- its colours never seen, or its pieces not the ones the
        sets on those colours learned -- so a start position shown once teaches it; None
        when a learned set reads it with its orientation and only a passing thing (a piece
        held, the dots of a selected piece) keeps the read from being clean."""
        if self.cache is None:
            return None
        sampled = sample_palette(frame, geometry)
        if sampled is None:
            return None  # the two square colours cannot be told apart: not a board to read
        for _key, model in self.cache.candidates(*sampled):
            readings = model.classify_frame(frame, geometry)
            told = read_bottom_side(frame, geometry, model.rank_labels) or self.forced_bottom
            for side in [told] if told is not None else [Side.WHITE, Side.BLACK]:
                shown = loose_placement(by_square(readings, side))
                if Reconciler._adopt_placement(shown) is not None:
                    return None if told is not None else "orientation"
        return "unlearned"

    def cold_start(
        self,
        frame: np.ndarray,
        geometry: BoardGeometry,
        bottom: Side | None = None,
        ts: float | None = None,
    ) -> list[Event]:
        """Recognise the board cold from the theme cache and begin a game from the position on
        it -- a puzzle or a mid-game board, with no move-0 bootstrap. `bottom` may be supplied
        (the panel's flip control); otherwise it is read from the coordinate labels. Returns the
        GameStarted, or [] when no cached theme matches or the orientation cannot be determined.
        """
        read = self._cold_read(frame, geometry, bottom, asked=True)
        if read is None:
            return []
        return self._adopt_cold(*read, ts=ts)

    def _adopt_cold(
        self,
        model: ThemeModel,
        bottom: Side,
        readings: Readings,
        placement: chess.Board,
        ts: float | None = None,
    ) -> list[Event]:
        reconciler = Reconciler(bottom=bottom, user_side=bottom)
        # The site's last-move highlight, for what a cold read cannot know: a pawn that has
        # just advanced two squares, and whose move it is (`reconcile.last_move_seen`).
        highlighted = highlighted_squares(by_square(readings, bottom))
        events = reconciler.start_from(placement, ts, highlighted)
        if not events:
            return []  # the placement is not a legal position; adopt nothing
        # Only now, with a legal position begun, is the pipeline switched onto this theme.
        if bottom != self.bottom:
            # The clock read the other way up named the other side as running, and the
            # side to move a cold read leaves open was settled from it on the next tick
            # (round 2, finding 3): read afresh, as a new recogniser's clock is.
            self._forget_clock()
        self.model = model
        self.bottom = bottom
        self._cold_since = None
        self._cold_noticed = False
        self._readings = readings
        self._readings_key = (
            (self.board.region, round(self.board.cell_px, 2)) if self.board is not None else None
        )
        self.reconciler = reconciler
        return events

    @staticmethod
    def _placement_from_readings(
        readings: Readings, bottom: Side, *, asked: bool = False
    ) -> chess.Board | None:
        """The classified board as a bare placement, or None unless it is a fully quiet,
        fully readable position with both kings on it.

        A cold read has no move history to lean on, so it must not adopt a board caught
        mid-move: a lifted piece empties its square and floats over another with the drag's
        hover outline, so a single frame read then would drop or misplace a piece (the c1
        bishop that read empty while it was in flight over f4). Every square must therefore
        be a settled piece or a settled empty with no hover and no unreadable cell -- the
        same quiet-board bar the reconciler holds a commit to -- or the read is refused and
        the caller retries on the next frame.

        `asked`: the player asked for this read ("Re-read the screen"), so a mark on an empty
        square -- a legal-move dot around a piece left selected -- is the empty square it is
        (`mark_on_empty`), as long as nothing on the board is being dragged. Reads nobody
        asked for keep treating it as no evidence: a phone may draw no outline around a
        dragged piece, and a drag held still must not be adopted.
        """
        if asked and any(reading.hover for row in readings for reading in row):
            return None
        board = chess.Board(None)
        for row in range(8):
            for col in range(8):
                reading = readings[row][col]
                if asked and mark_on_empty(reading):
                    continue
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
        events = self._end_in_progress("new game started")
        self.reconciler = Reconciler(bottom=self.bottom, user_side=self.bottom)
        return events
