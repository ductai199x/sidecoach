from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import detect_boards, fit_score, square_colours, to_gray
from tests.synth import DARK, LIGHT, bulky_start, render_board, synthetic_board, with_overlays

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


def test_overlays_do_not_move_detection() -> None:
    clean = synthetic_board(x0=300, y0=120, cell=64)
    busy = with_overlays(clean, 300, 120, 64)
    clean_best = detect_boards(clean)[0]
    busy_best = detect_boards(busy)[0]
    assert abs(busy_best.region.x - clean_best.region.x) <= 1
    assert abs(busy_best.region.y - clean_best.region.y) <= 1
    assert abs(busy_best.cell_px - clean_best.cell_px) <= 0.3
    assert busy_best.fit >= 0.85


def test_detects_odd_cell_size_and_position() -> None:
    frame = synthetic_board(x0=17, y0=333, cell=37, seed=3)
    best = detect_boards(frame)[0]
    assert abs(best.region.x - 17) <= 2
    assert abs(best.region.y - 333) <= 2
    assert abs(best.cell_px - 37) <= 0.5


def test_prefers_largest_of_two_boards() -> None:
    frame = synthetic_board(x0=60, y0=60, cell=70, width=1600, height=900, seed=5)
    small = synthetic_board(x0=0, y0=0, cell=24, width=192, height=192, pieces=False, seed=6)
    frame[650:842, 1300:1492] = small
    boards = detect_boards(frame)
    assert len(boards) >= 2
    assert abs(boards[0].cell_px - 70) <= 0.5
    assert abs(boards[1].cell_px - 24) <= 0.5


def test_no_board_in_noise() -> None:
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
    assert detect_boards(frame) == []


def test_a_real_chesscom_page_yields_its_board() -> None:
    """A chess.com bot game as the page shows it (tests/fixtures/chesscom_page.png, a
    1440p screenshot cropped to the page): the board among the move list, the captured
    pieces, the coach's chat and a badge on the king's square. Only the board is found,
    where it is, at the size it is."""
    frame = cv2.imread(str(Path(__file__).parent / "fixtures" / "chesscom_page.png"))
    assert frame is not None
    boards = detect_boards(frame)
    assert boards, "no board found on the real page"
    best = boards[0]
    assert best.fit >= 0.85
    assert abs(best.cell_px - 86) <= 1
    assert abs(best.region.x - 290) <= 2 and abs(best.region.y - 65) <= 2


def test_whole_cell_shifts_are_not_a_fit() -> None:
    """Inside its 64 cells a shifted window is a perfect board; the cells outside tell."""
    frame = np.full((800, 1280, 3), 90, dtype=np.uint8)
    render_board(frame, 300, 120, 64, chess.Board())
    gray = to_gray(frame)
    assert fit_score(gray, Region(300, 120, 512, 512), 64.0)[0] >= 0.95
    for dx, dy in ((64, 0), (-64, 0), (0, 64), (0, -64), (128, 0), (64, 64)):
        fit, _ = fit_score(gray, Region(300 + dx, 120 + dy, 512, 512), 64.0)
        assert fit < 0.5, (dx, dy, fit)


def test_low_contrast_board_is_found_in_place() -> None:
    frame = synthetic_board(light=(141, 141, 141), dark=(128, 128, 128), pieces=False)
    boards = detect_boards(frame)
    assert boards, "no board found on the low-contrast theme"
    assert abs(boards[0].region.x - 300) <= 2 and abs(boards[0].region.y - 120) <= 2
    with_pieces = synthetic_board(light=(141, 141, 141), dark=(128, 128, 128))
    boards = detect_boards(with_pieces)
    assert boards and abs(boards[0].region.y - 120) <= 2


def _checkerboard(frame: np.ndarray, x0: int, y0: int, cell: int, n: int) -> None:
    for r in range(n):
        for c in range(n):
            colour = LIGHT if (r + c) % 2 == 0 else DARK
            frame[y0 + r * cell : y0 + (r + 1) * cell, x0 + c * cell : x0 + (c + 1) * cell] = colour


INVERTED: dict[str, Any] = {"light": (60, 60, 200), "dark": (200, 200, 60)}


@pytest.mark.parametrize("theme", [{}, INVERTED], ids=["light-is-brighter", "light-is-darker"])
def test_a_board_of_either_polarity_is_found_and_its_locked_colours_keep_it(
    theme: dict[str, Any],
) -> None:
    """Which square colour is the brighter one depends on the theme: a board with its pieces
    is found where it is, at its size, either way. A lock's palette names the colours by
    position, so the fit must follow its polarity: the board itself reads as a fit, and a
    whole-cell slide reads every cell wrong even before the outer ring is consulted."""
    frame = synthetic_board(**theme)
    boards = detect_boards(frame)
    assert boards, "no board found"
    best = boards[0]
    assert abs(best.region.x - 300) <= 2 and abs(best.region.y - 120) <= 2, best
    assert abs(best.cell_px - 64) <= 0.5 and best.fit >= 0.9, best
    gray = to_gray(frame)
    region = Region(300, 120, 512, 512)
    colours = square_colours(gray, region, 64.0)
    assert colours is not None
    light, dark, _ = colours
    assert (light > dark) == (not theme), colours
    assert fit_score(gray, region, 64.0, colours)[0] >= 0.95
    assert fit_score(gray, Region(364, 120, 512, 512), 64.0, colours)[0] < 0.5


def test_empty_board_does_not_hide_the_game_board() -> None:
    """An empty board's coarse score sits at 1.0 on a plateau of positions; taking the
    top scores per scale used to fill the quota with that plateau."""
    frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
    bulky_start(frame, 60, 100, 48)
    render_board(frame, 800, 100, 48, chess.Board(None))
    boards = detect_boards(frame)
    assert len(boards) == 2, boards
    assert abs(boards[0].region.x - 60) <= 2 and boards[0].occupied > 0, boards
    assert abs(boards[1].region.x - 800) <= 2 and boards[1].occupied == 0, boards

    larger = np.full((900, 1400, 3), 90, dtype=np.uint8)
    bulky_start(larger, 60, 100, 56)
    render_board(larger, 800, 100, 48, chess.Board(None))
    boards = detect_boards(larger)
    assert len(boards) == 2 and abs(boards[0].cell_px - 56) <= 0.5, boards


def test_a_larger_checkerboard_is_not_a_board_and_does_not_hide_the_game_board() -> None:
    """A 12x12 checkerboard has hundreds of positions scoring 1.0 at the game board's
    scale; each is rejected by the outer ring -- on its own it is no board at all -- but
    beside a game board only if the game board reaches refinement at all."""
    alone = np.full((900, 1000, 3), 90, dtype=np.uint8)
    _checkerboard(alone, 100, 100, 60, 12)
    assert detect_boards(alone) == []
    for game_x, checker_x in ((60, 700), (900, 100)):
        frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
        _checkerboard(frame, checker_x, 100, 48, 12)
        bulky_start(frame, game_x, 100, 48)
        boards = detect_boards(frame)
        assert len(boards) == 1 and abs(boards[0].region.x - game_x) <= 2, (game_x, boards)


def _two_boards(gap: float) -> tuple[np.ndarray, int]:
    """A game board and an empty board of the same cell size, `gap` cells apart."""
    frame = np.full((700, 1400, 3), 90, dtype=np.uint8)
    render_board(frame, 60, 100, 48, chess.Board())
    second_x = 60 + 8 * 48 + int(round(gap * 48))
    render_board(frame, second_x, 100, 48, chess.Board(None))
    return frame, second_x


@pytest.mark.parametrize("gap", [0.5, 1.0])
def test_boards_half_a_cell_or_more_apart_are_both_found(gap: float) -> None:
    """Half a cell is the edge: the outer ring between the boards reads half background,
    half the neighbour, and its median lands between the colours, so it is unreadable."""
    frame, second_x = _two_boards(gap)
    boards = detect_boards(frame)
    assert len(boards) == 2, boards
    assert abs(boards[0].region.x - 60) <= 2 and abs(boards[1].region.x - second_x) <= 2


@pytest.mark.parametrize("gap", [0.0, 0.25])
def test_boards_closer_than_half_a_cell_read_as_one_pattern(gap: float) -> None:
    """A known limitation of the detector, pinned so that a change to it is seen, not a
    requirement: the outer ring of each board reads the other's squares, so the checker
    pattern continues and both are rejected, as a larger checkerboard is. A detector that
    found both boards would be better, and this test would change with it."""
    frame, _ = _two_boards(gap)
    assert detect_boards(frame) == []


def test_board_flush_with_the_frame_corner() -> None:
    frame = np.full((600, 800, 3), 90, dtype=np.uint8)
    render_board(frame, 800 - 512, 600 - 512, 64, chess.Board())
    boards = detect_boards(frame)
    assert boards and abs(boards[0].cell_px - 64) <= 0.5
    assert abs(boards[0].region.x - 288) <= 2 and abs(boards[0].region.y - 88) <= 2


@pytest.mark.parametrize(
    ("style", "x", "y"),
    [
        ("chess7", 1920 - 8 * 136, 80),  # against the right edge: was not found at all
        ("pixel", 1920 - 8 * 136, 80),
        ("cardinal", 10, 1200 - 8 * 136),  # against the bottom edge: was found 10-19 px high
        ("governor", 10, 1200 - 8 * 136),
    ],
)
def test_a_board_flush_with_the_right_or_bottom_edge_is_found_where_it_is(
    style: str, x: int, y: int
) -> None:
    """At 136 px cells the coarse scale nearest the board's is a few percent larger, and its
    window on a board against the right or bottom edge ran past the frame; the boundary comb
    could not lay its last tooth on the frame's edge, where the board's boundary lies (the
    geometry sweep, 2026-09-27)."""
    frame = synthetic_board(width=1920, height=1200, x0=x, y0=y, cell=136, theme="ice", style=style)
    boards = detect_boards(frame)
    assert boards, "not found"
    best = boards[0]
    assert abs(best.region.x - x) <= 2 and abs(best.region.y - y) <= 2, best
    assert abs(best.cell_px - 136) <= 1.0


def test_a_candidate_off_a_board_by_a_fraction_is_refused_though_its_outer_row_is_cut() -> None:
    """A board two pixels above the frame's bottom and a candidate two thirds of a cell
    above it: the candidate's row of outer cells above it is off the frame and the row below
    it a third off, and neither was read, so the candidate validated with every cell across
    two squares (the geometry sweep, 2026-09-27, seed 260). The outer cells are read on the
    part the frame shows when that is half a cell or more."""
    board = chess.Board()
    for san in (
        "f3", "Nc6", "g4", "e5", "b4", "b6", "Bh3", "Na5",
        "Kf2", "Bxb4", "c4", "Nxc4", "Bf1", "Nb2", "a3", "d5",
    ):  # fmt: skip
        board.push_san(san)
    frame = synthetic_board(
        width=1920, height=1200, x0=583, y0=110, cell=136, theme="wood", style="pixel",
        bottom=Side.BLACK, board=board,
    )  # fmt: skip
    gray = to_gray(frame)
    assert fit_score(gray, Region(583, 110, 1088, 1088), 136.0)[0] >= 0.95
    assert fit_score(gray, Region(582, 24, 1090, 1090), 136.2)[0] == 0.0


def test_the_ring_cache_stays_within_its_budget() -> None:
    """A detection on a big board fits many geometries, and the ring cache held up to 64 of
    them in int64: 71 MB at 1440p, 103 MB at 4K (round 2 finding 10). Indices are int32,
    the cache keeps the newest used within RING_CACHE_BYTES, and a geometry read again is
    the same medians."""
    from sidecoach.vision import board_detect

    board_detect._RING_CACHE.clear()
    board_detect._ring_cache_bytes = 0
    frame = synthetic_board(width=2560, height=1440, x0=200, y0=100, cell=150)
    best = detect_boards(frame)[0]
    held = sum(i.nbytes for groups in board_detect._RING_CACHE.values() for _, i in groups)
    assert held <= board_detect.RING_CACHE_BYTES
    assert held == board_detect._ring_cache_bytes
    dtypes = {i.dtype for groups in board_detect._RING_CACHE.values() for _, i in groups}
    assert dtypes == {np.dtype(np.int32)}
    gray = to_gray(frame)
    first = board_detect._cells(gray, best.region, best.cell_px)
    board_detect._RING_CACHE.clear()
    board_detect._ring_cache_bytes = 0
    assert np.array_equal(first, board_detect._cells(gray, best.region, best.cell_px), True)
