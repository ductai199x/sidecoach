"""The clock reader wired into the pipeline.

A clock reading is bottom/top relative to the board; the pipeline turns it into a
colour-keyed `ClockObservation` using the board's own orientation, and every way a clock
can be absent, hidden, or misread produces nothing rather than a crash or a bad number.
The point mirrors the rest of the project: a broken input costs its own signal, never the
session.
"""

from __future__ import annotations

import numpy as np

from sidecoach.events import ClockObservation, Region, Side
from sidecoach.state.pipeline import CLOCK_MAX_MS, GamePipeline, clock_observation
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import ClockReader, ClockReading

FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
BOARD = BoardCandidate(region=Region(0, 0, 80, 80), cell_px=10.0, score=1.0, contrast=1.0)


def _reading(
    top: int | None,
    bottom: int | None,
    running: str | None = None,
    confidence: float = 0.9,
) -> ClockReading:
    return ClockReading(
        top_ms=top,
        bottom_ms=bottom,
        running=running,  # type: ignore[arg-type]
        confidence=confidence,
    )


class _StubReader(ClockReader):
    """A clock reader that returns a canned reading and counts its resets, no pixels read."""

    def __init__(self, out: ClockReading | None) -> None:
        super().__init__(templates=None)  # no template file is loaded
        self._out = out
        self.resets = 0

    def read(
        self, frame: np.ndarray, board: object, ts: float | None = None
    ) -> ClockReading | None:  # type: ignore[override]
        return self._out

    def reset(self) -> None:
        self.resets += 1


def _oriented(reader: ClockReader, bottom: Side = Side.WHITE) -> GamePipeline:
    pipe = GamePipeline(clock=reader)
    pipe.bottom = bottom
    pipe.board = BOARD
    return pipe


# The pure mapping ---------------------------------------------------------------------


def test_bottom_box_is_the_user_and_colours_key_off_the_board_orientation() -> None:
    # The user sits at the bottom. As White, the bottom box is White's clock, the top Black's.
    obs = clock_observation(_reading(top=120_000, bottom=95_000, running="bottom"), Side.WHITE)
    assert obs is not None
    assert (obs.white_ms, obs.black_ms, obs.running) == (95_000, 120_000, Side.WHITE)
    # The very same reading, but the user is playing Black from the bottom: the mapping flips.
    obs = clock_observation(_reading(top=120_000, bottom=95_000, running="bottom"), Side.BLACK)
    assert obs is not None
    assert (obs.white_ms, obs.black_ms, obs.running) == (120_000, 95_000, Side.BLACK)
    # A running top box belongs to the side that is not at the bottom.
    obs = clock_observation(_reading(top=120_000, bottom=95_000, running="top"), Side.WHITE)
    assert obs is not None and obs.running is Side.BLACK


def test_nothing_trustworthy_maps_to_nothing() -> None:
    assert clock_observation(None, Side.WHITE) is None
    # Confidence below the floor is dropped whole.
    assert clock_observation(_reading(120_000, 95_000, confidence=0.1), Side.WHITE) is None
    # Both boxes unreadable: an untimed game, or the clocks scrolled off screen.
    assert clock_observation(_reading(None, None), Side.WHITE) is None
    # An impossible value is a misread, dropped; a good box beside it still reports.
    obs = clock_observation(_reading(top=CLOCK_MAX_MS + 1, bottom=95_000), Side.WHITE)
    assert obs is not None and obs.black_ms is None and obs.white_ms == 95_000


# The pipeline plumbing ----------------------------------------------------------------


def test_pipeline_emits_one_observation_then_dedupes_the_repeat() -> None:
    pipe = _oriented(_StubReader(_reading(120_000, 95_000, running="bottom")))
    first = pipe._read_clock(FRAME, 1.0)
    assert len(first) == 1 and isinstance(first[0], ClockObservation)
    # The identical reading on the next tick is not repeated into the event stream.
    assert pipe._read_clock(FRAME, 1.1) == []
    # A changed reading (a second ticked off) is emitted again.
    pipe.clock = _StubReader(_reading(120_000, 94_000, running="bottom"))
    assert len(pipe._read_clock(FRAME, 2.0)) == 1


def test_silent_until_the_board_is_oriented_and_present() -> None:
    reader = _StubReader(_reading(120_000, 95_000))
    pipe = GamePipeline(clock=reader)
    pipe.board = BOARD
    pipe.bottom = None  # no game oriented yet
    assert pipe._read_clock(FRAME, 1.0) == []
    pipe.bottom = Side.WHITE
    pipe.board = None  # board lost this frame
    assert pipe._read_clock(FRAME, 1.0) == []


def test_a_broken_or_absent_clock_never_breaks_the_pipeline() -> None:
    class _Boom(ClockReader):
        def __init__(self) -> None:
            super().__init__(templates=None)

        def read(
            self, frame: np.ndarray, board: object, ts: float | None = None
        ) -> ClockReading | None:  # type: ignore[override]
            raise RuntimeError("a dialog is drawn over the clock")

    assert _oriented(_Boom())._read_clock(FRAME, 1.0) == []  # an exception is swallowed
    assert _oriented(_StubReader(None))._read_clock(FRAME, 1.0) == []  # no clock drawn


def test_default_pipeline_wires_no_clock_and_stays_inert() -> None:
    # Constructed without a reader (as every existing test and the detect command do), the
    # clock path does nothing at all, so nothing about the old behaviour changes.
    pipe = GamePipeline()
    pipe.bottom = Side.WHITE
    pipe.board = BOARD
    assert pipe.clock is None
    assert pipe._read_clock(FRAME, 1.0) == []
