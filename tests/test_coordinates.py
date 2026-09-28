"""Reading board orientation from the coordinate labels.

The real g01 recording proves the White direction (see the vision validation). Here a synthetic
board proves both directions and the no-labels fallback cleanly: rank digits are drawn in the
top-left of each left-edge cell -- 8..1 down for White at the bottom, 1..8 for Black -- exactly
as chess.com draws them, and the reader must recover the side from them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import detect_boards
from sidecoach.vision.coordinates import (
    GLYPH,
    eights,
    learn_rank_labels,
    read_bottom_side,
    read_labels,
)
from sidecoach.vision.squares import BoardGeometry
from tests.synth import synthetic_board

FIXTURES = Path(__file__).parent / "fixtures"
CELL = 48
GEOM = BoardGeometry(Region(0, 0, CELL * 8, CELL * 8), float(CELL))
LIGHT = (210, 220, 190)
DARK = (100, 140, 90)


def make_board(bottom: Side | None) -> np.ndarray:
    """An 8x8 board; when `bottom` is set, rank digits are drawn in the left-edge corners."""
    img = np.zeros((CELL * 8, CELL * 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            img[row * CELL : (row + 1) * CELL, col * CELL : (col + 1) * CELL] = (
                LIGHT if (row + col) % 2 == 0 else DARK
            )
    if bottom is not None:
        for row in range(8):
            rank = (8 - row) if bottom == Side.WHITE else (row + 1)
            square = LIGHT if row % 2 == 0 else DARK
            ink = (30, 30, 30) if sum(square) > 400 else (240, 240, 240)
            cv2.putText(
                img,
                str(rank),
                (3, row * CELL + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ink,
                1,
                cv2.LINE_AA,
            )
    return img


def test_orientation_is_read_from_the_rank_labels() -> None:
    white_board = make_board(Side.WHITE)
    black_board = make_board(Side.BLACK)
    # The theme is learned from the start position, where the orientation is known.
    labels = learn_rank_labels(white_board, GEOM, Side.WHITE)
    assert sorted(labels) == list(range(1, 9))
    # The same exemplars read either orientation from the edge digits alone.
    assert read_labels(white_board, GEOM, labels) == Side.WHITE
    assert read_labels(black_board, GEOM, labels) == Side.BLACK


def test_a_start_seen_through_a_lock_off_its_board_learns_no_labels() -> None:
    """A lock a fifth of a cell out cuts each edge cell partly from its neighbour: the
    square's colour is taken from the neighbour and the whole corner reads as ink. Kept,
    those solid blocks matched every digit alike, and no board of the set had its side read
    again (Tai's Pixel, 2026-09-27); none are learned, and the next start teaches them."""
    board = make_board(Side.WHITE)
    for dx, dy in ((CELL // 5, 0), (0, CELL // 5), (0, -CELL // 3)):
        off = BoardGeometry(Region(dx, dy, CELL * 8, CELL * 8), float(CELL))
        assert learn_rank_labels(board, off, Side.WHITE) == {}, (dx, dy)
    assert sorted(learn_rank_labels(board, GEOM, Side.WHITE)) == list(range(1, 9))


def test_no_coordinates_read_no_side_so_the_caller_can_ask() -> None:
    labels = learn_rank_labels(make_board(Side.WHITE), GEOM, Side.WHITE)
    # A board with coordinates turned off has nothing to read, labels learned or not.
    assert read_bottom_side(make_board(None), GEOM, labels) is None
    assert read_bottom_side(make_board(None), GEOM, {}) is None
    # And no labels learned read nothing themselves.
    assert read_labels(make_board(Side.WHITE), GEOM, {}) is None


def test_the_8_reads_the_side_with_nothing_learned() -> None:
    """Where the 8 is, at the top of the left edge or at its foot, says which side is at the
    bottom with nothing learned -- read through labels cached as solid blocks, as Tai's
    were: the board of his puzzle on the Pixel (84 px, Black at the bottom) and a chess.com
    start (80 px, White), both real (2026-09-28)."""
    solid = {rank: np.ones((GLYPH, GLYPH), dtype=bool) for rank in range(1, 9)}
    rendered = BoardGeometry(Region(300, 120, 512, 512), 64.0)  # synthetic_board's defaults
    for bottom in (Side.WHITE, Side.BLACK):
        board = synthetic_board(bottom=bottom, coordinates=True)
        assert eights(board, rendered) is bottom
        assert read_bottom_side(board, rendered, solid) is bottom
    assert eights(synthetic_board(coordinates=False), rendered) is None
    puzzle = cv2.imread(str(FIXTURES / "chesscom_puzzle_black_84.png"))
    assert puzzle is not None
    on_phone = BoardGeometry(Region(0, 0, 672, 672), 84.0)
    assert read_bottom_side(puzzle, on_phone, solid) is Side.BLACK
    start = cv2.imread(str(FIXTURES / "chesscom_start.png"))
    assert start is not None
    [board, *_] = detect_boards(start)
    assert read_bottom_side(start, BoardGeometry(board.region, board.cell_px), solid) is Side.WHITE
