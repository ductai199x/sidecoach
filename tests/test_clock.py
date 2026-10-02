"""Reading the two chess.com clocks beside the board.

The synthetic boxes are drawn with the very templates the reader classifies against, so
these tests check the locating, the segmentation, the running-clock decision and the
failure behaviour, not the templates themselves. What the templates are worth is measured
against the recordings, by `scripts/validate_clocks.py`, outside the suite.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from sidecoach.events import Region
from sidecoach.vision import clock as clock_module
from sidecoach.vision.board_detect import BoardCandidate
from sidecoach.vision.clock import (
    BOX_WIDTH_CELLS,
    SEARCH_CELLS,
    ClockReader,
    ClockReading,
    GlyphTemplates,
    parse_clock,
)
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


# Clock strings, readable and not, and the milliseconds read (None: not a clock). The phone's
# parser is held to the cases of make_clock_fixtures.py's PARSE_CASES too, which the desktop
# did not pin (the parity audit, 2026-09-29).
PARSED: list[tuple[str, int | None]] = [
    ("15:00", 900_000),
    ("3:07", 187_000),
    ("0:09", 9_000),
    ("12:34", 754_000),
    ("59:59", 3_599_000),
    ("0:00", 0),
    ("0:59.9", 59_900),
    ("0:00.0", 0),
    ("1:02:03", 3_723_000),
    ("1:23:45", 5_025_000),
    # A tenths dot read as a colon: seconds always have two digits, so a lone digit after
    # the last separator is tenths whichever glyph matched...
    ("0:12:3", 12_300),
    # ...and what that leaves must still be a clock: 10:2.3 and 1:2.3 have one-digit seconds.
    ("10:2:3", None),
    ("1:2:3", None),
    # Out of range: sixty seconds, sixty minutes under an hour, three digits of minutes.
    ("9:60", None),
    ("0:75", None),
    ("1:60:00", None),
    ("100:00", None),
    # No clock at all.
    ("", None),
    ("::", None),
    ("5", None),
    ("5:", None),
    (":30", None),
    ("1:2", None),
    ("abc", None),
    ("1.2", None),
]


def test_parse_clock_covers_the_formats_chesscom_draws() -> None:
    wrong = [(text, got, ms) for text, ms in PARSED if (got := parse_clock(text)) != ms]
    assert not wrong, f"(text, read, wanted): {wrong}"


@pytest.mark.parametrize(
    ("clocks", "icon_angles", "runs"),
    [
        (("10:00", "9:40"), (None, math.pi / 2), "bottom"),
        (("10:00", "9:41"), (None, math.pi / 2), "bottom"),
        (("10:00", "9:51"), (0.0, None), "top"),
    ],
    ids=["icon and falling value", "icon alone", "a rise"],
)
def test_the_running_clock_is_read_from_what_moved(
    clocks: tuple[str, str], icon_angles: tuple[float | None, float | None], runs: str
) -> None:
    """Two frames a tenth of a second apart, the bottom clock's icon shown on the first: its
    hand swept on and its value fallen, or its hand alone, while the top clock has not moved
    a pixel, is the bottom clock running; a clock that rose has taken its increment and
    stopped, and the other runs. Nothing has moved on the first frame, and neither clock runs
    once that evidence goes stale."""
    first, board = board_with_clocks("10:00", "9:41", icon_angles=(None, 0.0))
    second, _ = board_with_clocks(*clocks, icon_angles=icon_angles)
    reader = ClockReader()
    assert running(reader, first, board, 0.0) is None
    assert running(reader, second, board, 0.1) == runs
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


def _read_or_raised(frame: np.ndarray, board: BoardCandidate) -> ClockReading | Exception | None:
    """A fresh reader's reading of one frame, or what it raised."""
    try:
        return ClockReader().read(frame, board, 1.0)
    except Exception as exc:  # noqa: BLE001 - raising is the failure these tests name
        return exc


def test_a_bad_board_never_raises() -> None:
    frame, _ = board_with_clocks("16:18", "15:34")
    boards = {
        "no width": BoardCandidate(Region(300, 120, 0, 0), 43.0, 1.0, 60.0, 1.0),
        "no cell": BoardCandidate(Region(300, 120, 344, 344), 0.0, 1.0, 60.0, 1.0),
        "above the frame": BoardCandidate(Region(-500, -500, 344, 344), 43.0, 1.0, 60.0, 1.0),
        "past the frame": BoardCandidate(Region(9000, 9000, 344, 344), 43.0, 1.0, 60.0, 1.0),
        "a NaN cell": BoardCandidate(Region(300, 120, 344, 344), float("nan"), 1.0, 60.0, 1.0),
    }
    wrong = [
        (name, got)
        for name, board in boards.items()
        if (got := _read_or_raised(frame, board)) is not None
    ]
    assert not wrong, wrong


def test_a_bad_frame_never_raises() -> None:
    board = BoardCandidate(Region(300, 120, 344, 344), 43.0, 1.0, 60.0, 1.0)
    frames = {
        "empty": np.zeros((0, 0, 3), dtype=np.uint8),
        "one channel": np.zeros((720, 1280), dtype=np.uint8),
        "flat grey": np.full((720, 1280, 3), 128, dtype=np.uint8),
        "float": np.zeros((720, 1280, 3), dtype=np.float32),
        "four channels": np.zeros((720, 1280, 4), dtype=np.uint8),
        "four pixels": np.zeros((4, 4, 3), dtype=np.uint8),
        "no clock": busy_background(1280, 720, seed=7),
    }
    wrong = []
    for name, frame in frames.items():
        got = _read_or_raised(frame, board)
        if isinstance(got, Exception) or (
            got is not None and (got.top_ms is not None or got.bottom_ms is not None)
        ):
            wrong.append((name, got))
    assert not wrong, wrong


def test_a_reader_without_templates_reads_nothing_rather_than_raising(tmp_path: Path) -> None:
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not an npz file")
    with pytest.raises((OSError, ValueError, KeyError)):
        GlyphTemplates.load(broken)
    reader = ClockReader()
    reader.templates = None  # what a reader built against a broken file holds
    frame, board = board_with_clocks("16:18", "15:34")
    assert reader.read(frame, board, 1.0) is None


def test_the_reader_is_cheap_enough_for_every_changed_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reader runs on every changed frame, so its budget is about 5 ms at 720p (the
    recordings measure well under 1 ms). The cost is proven by counting the work rather than
    by a stopwatch: per frame the reader looks at the two strips beside the board and the
    icons in them, never the frame, and classifies each glyph once. That work is the same
    for a frame four times the size and does not grow with the frames read before."""
    converted: list[int] = []  # pixels of every image the reader turns grey
    classified: list[int] = []  # one entry per glyph classification
    boxes: list[int] = []  # one entry per box search

    class CountingCv2:
        def __getattr__(self, name: str) -> object:
            return getattr(cv2, name)

        @staticmethod
        def cvtColor(image: np.ndarray, code: int) -> np.ndarray:  # noqa: N802 - cv2's name
            converted.append(int(image.shape[0] * image.shape[1]))
            return cv2.cvtColor(image, code)

    real_classify = GlyphTemplates.classify
    real_find_box = clock_module.find_box

    def counting_classify(self: GlyphTemplates, canvas: np.ndarray) -> tuple[str, float, float]:
        classified.append(1)
        return real_classify(self, canvas)

    def counting_find_box(*args: object, **kwargs: object) -> object:
        boxes.append(1)
        return real_find_box(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(clock_module, "cv2", CountingCv2())
    monkeypatch.setattr(GlyphTemplates, "classify", counting_classify)
    monkeypatch.setattr(clock_module, "find_box", counting_find_box)

    def work_per_frame(width: int, height: int, runs: int) -> list[tuple[int, int, int]]:
        frame, board = board_with_clocks("16:18", "15:34", cell=43, width=width, height=height)
        reader = ClockReader()
        work: list[tuple[int, int, int]] = []
        for i in range(runs):
            converted.clear()
            classified.clear()
            boxes.clear()
            reading = reader.read(frame, board, 0.1 * i)
            assert reading is not None and reading.top_ms == 978_000, reading
            work.append((sum(converted), len(classified), len(boxes)))
        return work

    runs = 50
    at_720p = work_per_frame(1280, 720, runs)
    pixels, glyphs, searches = at_720p[0]
    # The same work on every frame: nothing accumulates as frames go by.
    assert at_720p == [at_720p[0]] * runs, set(at_720p)
    # Two box searches, one per clock, and each glyph of "16:18" and "15:34" classified once.
    assert searches == 2
    assert glyphs == len("16:18") + len("15:34")
    # Only the strips beside the board are read: each at most the search reach high and the
    # widest box wide, plus the two icons (smaller than a cell each). A tiny part of a frame.
    cell = 43
    strip = int(round(SEARCH_CELLS * cell)) * (int(BOX_WIDTH_CELLS[1] * cell) + 4)
    assert pixels <= 2 * strip + 2 * cell * cell, pixels
    assert pixels < 1280 * 720 // 20, pixels
    # A frame four times the size costs exactly the same: the cost is the board's, not the
    # screen's.
    assert work_per_frame(2560, 1440, 3) == at_720p[:3]
