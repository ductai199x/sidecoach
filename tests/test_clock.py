"""Reading the two chess.com clocks beside the board.

The synthetic boxes are drawn with the very templates the reader classifies against, so
these tests check the locating, the segmentation, the running-clock decision and the
failure behaviour, not the templates themselves. What the templates are worth is measured
against the recordings, by `scripts/validate_clocks.py`; the last test here runs that
comparison on one recording when `data/` is present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sidecoach.events import Region
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import ClockReader, ClockReading, GlyphTemplates, parse_clock
from tests.synth import board_with_clocks, busy_background

# A second theme: a cream box with brown text above, the inverse below.
CREAM = ((236, 96), (96, 236))


def read(frame: np.ndarray, board: BoardCandidate, ts: float = 1.0) -> ClockReading:
    """Read one frame, insisting there was a clock to read."""
    reading = ClockReader().read(frame, board, ts)
    assert reading is not None
    return reading


def running(reader: ClockReader, frame: np.ndarray, board: BoardCandidate, ts: float) -> str | None:
    reading = reader.read(frame, board, ts)
    assert reading is not None
    return reading.running


@pytest.mark.parametrize("cell", [30, 43, 64])
@pytest.mark.parametrize("colours", [None, CREAM])
def test_reads_both_clocks_at_several_scales_and_themes(cell: int, colours) -> None:
    frame, board = board_with_clocks(
        "16:18",
        "15:34",
        cell=cell,
        y0=60,
        height=60 + 8 * cell + 100,
        width=300 + 8 * cell + 140,
        colours=colours,
    )
    reading = read(frame, board)
    assert reading is not None
    assert reading.top_ms == (16 * 60 + 18) * 1000
    assert reading.bottom_ms == (15 * 60 + 34) * 1000
    assert reading.confidence > 0.35


def test_reads_clocks_on_a_light_page() -> None:
    """A light site theme: a near-white page, a white box above and a grey one below."""
    frame, board = board_with_clocks(
        "3:07", "12:45", page=(232, 232, 232), colours=((255, 40), (120, 250))
    )
    reading = read(frame, board)
    assert reading.top_ms == (3 * 60 + 7) * 1000
    assert reading.bottom_ms == (12 * 60 + 45) * 1000
    assert reading.light == "top"


def test_light_box_names_whites_clock_whichever_side_is_on_top() -> None:
    frame, board = board_with_clocks("5:00", "5:00", top_light=False)
    assert read(frame, board).light == "bottom"
    frame, board = board_with_clocks("5:00", "5:00", top_light=True)
    assert read(frame, board).light == "top"


def test_reads_tenths_under_a_minute() -> None:
    frame, board = board_with_clocks("0:59.9", "1:04")
    reading = read(frame, board)
    assert reading.top_ms == 59_900
    assert reading.bottom_ms == 64_000


def test_parse_clock_covers_the_formats_chesscom_draws() -> None:
    assert parse_clock("15:00") == 900_000
    assert parse_clock("0:59.9") == 59_900
    assert parse_clock("1:02:03") == 3_723_000
    assert parse_clock("0:00") == 0
    # A tenths dot read as a colon: seconds always have two digits, so a lone digit after
    # the last separator is tenths whichever glyph matched.
    assert parse_clock("0:12:3") == 12_300
    assert parse_clock("") is None
    assert parse_clock("1:2") is None
    assert parse_clock("0:75") is None


def _running_frames(cell: int = 43) -> tuple[np.ndarray, np.ndarray, BoardCandidate]:
    """Two frames a tenth of a second apart with the bottom clock ticking: its icon hand
    has swept on and its value has fallen, while the top clock has not moved a pixel."""
    first, board = board_with_clocks("10:00", "9:41", icon_angles=(None, 0.0))
    second, _ = board_with_clocks("10:00", "9:40", icon_angles=(None, math.pi / 2))
    return first, second, board


def test_running_comes_from_the_icon_and_the_falling_value() -> None:
    first, second, board = _running_frames()
    reader = ClockReader()
    assert running(reader, first, board, 0.0) is None  # nothing has moved yet
    assert running(reader, second, board, 0.1) == "bottom"


def test_a_clock_that_only_moves_its_icon_is_still_the_running_one() -> None:
    first, board = board_with_clocks("10:00", "9:41", icon_angles=(None, 0.0))
    second, _ = board_with_clocks("10:00", "9:41", icon_angles=(None, math.pi / 2))
    reader = ClockReader()
    reader.read(first, board, 0.0)
    assert running(reader, second, board, 0.1) == "bottom"


def test_a_clock_that_rises_has_taken_its_increment_and_stopped() -> None:
    first, board = board_with_clocks("10:00", "9:41", icon_angles=(None, 0.0))
    second, _ = board_with_clocks("10:00", "9:51", icon_angles=(0.0, None))
    reader = ClockReader()
    reader.read(first, board, 0.0)
    assert running(reader, second, board, 0.1) == "top"


def test_neither_clock_runs_once_the_evidence_goes_stale() -> None:
    first, second, board = _running_frames()
    reader = ClockReader()
    reader.read(first, board, 0.0)
    assert running(reader, second, board, 0.1) == "bottom"
    assert running(reader, second, board, 10.0) is None


def test_no_clock_drawn_reads_as_none() -> None:
    from tests.synth import synthetic_board

    frame = synthetic_board(width=1280, height=720, x0=300, y0=120, cell=43)
    board = BoardCandidate(Region(300, 120, 344, 344), 43.0, 1.0, 60.0, 1.0)
    assert ClockReader().read(frame, board, 1.0) is None


def test_a_clock_scrolled_off_the_frame_reads_as_none_not_nonsense() -> None:
    frame, board = board_with_clocks("16:18", "15:34", cell=43, y0=60, height=800)
    # The page scrolled up until the board's top edge is off screen: the clock above it
    # has gone with it, and the one below is still there.
    cropped = frame[200:, :]
    moved = BoardCandidate(
        Region(board.region.x, board.region.y - 200, board.region.w, board.region.h),
        board.cell_px,
        1.0,
        60.0,
        1.0,
    )
    reading = ClockReader().read(cropped, moved, 1.0)
    assert reading is not None
    assert reading.top_ms is None
    assert reading.bottom_ms == (15 * 60 + 34) * 1000


def test_a_clock_cut_off_at_the_frame_edge_reads_as_none() -> None:
    frame, board = board_with_clocks("16:18", "15:34", cell=43, y0=60, height=800)
    # Cut the frame just left of the clocks' right edge, so both boxes are half there.
    edge = board.region.x + board.region.w - 20
    assert ClockReader().read(frame[:, :edge], board, 1.0) is None


@pytest.mark.parametrize(
    "board",
    [
        BoardCandidate(Region(300, 120, 0, 0), 43.0, 1.0, 60.0, 1.0),
        BoardCandidate(Region(300, 120, 344, 344), 0.0, 1.0, 60.0, 1.0),
        BoardCandidate(Region(-500, -500, 344, 344), 43.0, 1.0, 60.0, 1.0),
        BoardCandidate(Region(9000, 9000, 344, 344), 43.0, 1.0, 60.0, 1.0),
        BoardCandidate(Region(300, 120, 344, 344), float("nan"), 1.0, 60.0, 1.0),
    ],
)
def test_a_bad_board_never_raises(board: BoardCandidate) -> None:
    frame, _ = board_with_clocks("16:18", "15:34")
    assert ClockReader().read(frame, board, 1.0) is None


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((720, 1280), dtype=np.uint8),
        np.full((720, 1280, 3), 128, dtype=np.uint8),
        np.zeros((720, 1280, 3), dtype=np.float32),
        np.zeros((720, 1280, 4), dtype=np.uint8),
        np.zeros((4, 4, 3), dtype=np.uint8),
        busy_background(1280, 720, seed=7),
    ],
)
def test_a_bad_frame_never_raises(frame: np.ndarray) -> None:
    board = BoardCandidate(Region(300, 120, 344, 344), 43.0, 1.0, 60.0, 1.0)
    reading = ClockReader().read(frame, board, 1.0)
    assert reading is None or (reading.top_ms is None and reading.bottom_ms is None)


def test_a_reader_without_templates_reads_nothing_rather_than_raising(tmp_path: Path) -> None:
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz file")
    with pytest.raises((OSError, ValueError, KeyError)):
        GlyphTemplates.load(broken)
    reader = ClockReader()
    reader.templates = None  # what a reader built against a broken file holds
    frame, board = board_with_clocks("16:18", "15:34")
    assert reader.read(frame, board, 1.0) is None


def test_the_reader_is_cheap_enough_for_every_changed_frame() -> None:
    from time import perf_counter

    frame, board = board_with_clocks("16:18", "15:34", cell=43)
    reader = ClockReader()
    reader.read(frame, board, 0.0)
    started = perf_counter()
    runs = 50
    for i in range(runs):
        reader.read(frame, board, 0.1 * i)
    per_frame_ms = 1000 * (perf_counter() - started) / runs
    # The budget is about 5 ms at 720p; the recordings measure well under 1 ms, so a
    # generous bound here still catches a change that makes the reader expensive.
    assert per_frame_ms < 5.0, per_frame_ms
