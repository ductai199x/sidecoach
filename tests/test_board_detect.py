from pathlib import Path
from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region
from sidecoach.vision.board_detect import detect_boards, fit_score, square_colours, to_gray
from tests.synth import DARK, LIGHT, bulky_pieces, render_board, synthetic_board, with_overlays


def test_overlays_do_not_move_detection() -> None:
    clean = synthetic_board(x0=300, y0=120, cell=64)
    busy = with_overlays(clean, 300, 120, 64)
    clean_best = detect_boards(clean)[0]
    busy_best = detect_boards(busy)[0]
    assert abs(busy_best.region.x - clean_best.region.x) <= 1
    assert abs(busy_best.region.y - clean_best.region.y) <= 1
    assert abs(busy_best.cell_px - clean_best.cell_px) <= 0.3
    assert busy_best.fit >= 0.85


def test_detects_full_board_with_pieces() -> None:
    frame = synthetic_board()
    boards = detect_boards(frame)
    assert boards, "no board found"
    best = boards[0]
    assert abs(best.region.x - 300) <= 2
    assert abs(best.region.y - 120) <= 2
    assert abs(best.cell_px - 64) <= 0.5
    assert best.fit >= 0.9


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


REAL = sorted(Path("data/screenshots").glob("*_m1.png"))


@pytest.mark.skipif(not REAL, reason="no real screenshot in data/screenshots")
def test_real_chesscom_screenshot() -> None:
    frame = cv2.imread(str(REAL[0]))
    assert frame is not None
    boards = detect_boards(frame)
    assert boards, "no board found in the real screenshot"
    best = boards[0]
    assert best.fit >= 0.85
    assert 60 <= best.cell_px <= 120


def test_whole_cell_shifts_are_not_a_fit() -> None:
    """Inside its 64 cells a shifted window is a perfect board; the cells outside tell."""
    frame = np.full((800, 1280, 3), 90, dtype=np.uint8)
    render_board(frame, 300, 120, 64, chess_start())
    gray = to_gray(frame)
    assert fit_score(gray, Region(300, 120, 512, 512), 64.0)[0] >= 0.95
    for dx, dy in ((64, 0), (-64, 0), (0, 64), (0, -64), (128, 0), (64, 64)):
        fit, _ = fit_score(gray, Region(300 + dx, 120 + dy, 512, 512), 64.0)
        assert fit < 0.5, (dx, dy, fit)


def chess_start():
    import chess

    return chess.Board()


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


def test_a_larger_checkerboard_is_not_a_board() -> None:
    frame = np.full((900, 1000, 3), 90, dtype=np.uint8)
    _checkerboard(frame, 100, 100, 60, 12)
    assert detect_boards(frame) == []


INVERTED: dict[str, Any] = {"light": (60, 60, 200), "dark": (200, 200, 60)}


def test_inverted_luminance_theme_is_still_a_board() -> None:
    """Which square colour is the brighter one depends on the theme."""
    frame = synthetic_board(**INVERTED)
    boards = detect_boards(frame)
    assert boards and abs(boards[0].region.x - 300) <= 2 and abs(boards[0].region.y - 120) <= 2


@pytest.mark.parametrize("theme", [{}, INVERTED], ids=["light-is-brighter", "light-is-darker"])
def test_locked_colours_keep_the_theme_polarity(theme: dict[str, Any]) -> None:
    """A lock's palette names the colours by position, so the fit must follow its polarity:
    the board itself reads as a fit, and a whole-cell slide reads every cell wrong even
    before the outer ring is consulted."""
    gray = to_gray(synthetic_board(**theme))
    region = Region(300, 120, 512, 512)
    colours = square_colours(gray, region, 64.0)
    assert colours is not None
    light, dark, _ = colours
    assert (light > dark) == (not theme), colours
    assert fit_score(gray, region, 64.0, colours)[0] >= 0.95
    assert fit_score(gray, Region(364, 120, 512, 512), 64.0, colours)[0] < 0.5


def _game_board(frame: np.ndarray, x0: int, y0: int, cell: int) -> None:
    """A start position drawn with bulky pieces, whose coarse checker score is well under
    an empty board's, so the coarse ranking alone would leave it behind."""
    render_board(frame, x0, y0, cell, chess.Board())
    bulky_pieces(frame, x0, y0, cell, chess.Board())


def test_empty_board_does_not_hide_the_game_board() -> None:
    """An empty board's coarse score sits at 1.0 on a plateau of positions; taking the
    top scores per scale used to fill the quota with that plateau."""
    frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
    _game_board(frame, 60, 100, 48)
    render_board(frame, 800, 100, 48, chess.Board(None))
    boards = detect_boards(frame)
    assert len(boards) == 2, boards
    assert abs(boards[0].region.x - 60) <= 2 and boards[0].occupied > 0, boards
    assert abs(boards[1].region.x - 800) <= 2 and boards[1].occupied == 0, boards

    larger = np.full((900, 1400, 3), 90, dtype=np.uint8)
    _game_board(larger, 60, 100, 56)
    render_board(larger, 800, 100, 48, chess.Board(None))
    boards = detect_boards(larger)
    assert len(boards) == 2 and abs(boards[0].cell_px - 56) <= 0.5, boards


def test_larger_checkerboard_does_not_hide_the_game_board() -> None:
    """A 12x12 checkerboard has hundreds of positions scoring 1.0 at the game board's
    scale; each is rejected by the outer ring, but only if the game board reaches
    refinement at all."""
    for game_x, checker_x in ((60, 700), (900, 100)):
        frame = np.full((900, 1400, 3), 90, dtype=np.uint8)
        _checkerboard(frame, checker_x, 100, 48, 12)
        _game_board(frame, game_x, 100, 48)
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
    """Pins accepted behaviour: the outer ring of each board reads the other's squares, so
    the checker pattern continues and both are rejected, as a larger checkerboard is."""
    frame, _ = _two_boards(gap)
    assert detect_boards(frame) == []


def test_board_flush_with_the_frame_corner() -> None:
    frame = np.full((600, 800, 3), 90, dtype=np.uint8)
    render_board(frame, 800 - 512, 600 - 512, 64, chess_start())
    boards = detect_boards(frame)
    assert boards and abs(boards[0].cell_px - 64) <= 0.5
    assert abs(boards[0].region.x - 288) <= 2 and abs(boards[0].region.y - 88) <= 2
