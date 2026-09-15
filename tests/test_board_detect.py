from pathlib import Path

import cv2
import numpy as np
import pytest

from sidecoach.vision.board_detect import detect_boards
from tests.synth import synthetic_board, with_overlays


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
    from sidecoach.events import Region
    from sidecoach.vision.board_detect import fit_score, to_gray
    from tests.synth import render_board

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


def test_a_larger_checkerboard_is_not_a_board() -> None:
    from tests.synth import DARK, LIGHT

    frame = np.full((900, 1000, 3), 90, dtype=np.uint8)
    for r in range(12):
        for c in range(12):
            colour = LIGHT if (r + c) % 2 == 0 else DARK
            frame[100 + r * 60 : 160 + r * 60, 100 + c * 60 : 160 + c * 60] = colour
    assert detect_boards(frame) == []


def test_inverted_luminance_theme_is_still_a_board() -> None:
    """Which square colour is the brighter one depends on the theme."""
    frame = synthetic_board(light=(60, 60, 200), dark=(200, 200, 60))
    boards = detect_boards(frame)
    assert boards and abs(boards[0].region.x - 300) <= 2 and abs(boards[0].region.y - 120) <= 2


def test_board_flush_with_the_frame_corner() -> None:
    from tests.synth import render_board

    frame = np.full((600, 800, 3), 90, dtype=np.uint8)
    render_board(frame, 800 - 512, 600 - 512, 64, chess_start())
    boards = detect_boards(frame)
    assert boards and abs(boards[0].cell_px - 64) <= 0.5
    assert abs(boards[0].region.x - 288) <= 2 and abs(boards[0].region.y - 88) <= 2
