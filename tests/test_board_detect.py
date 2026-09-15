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
