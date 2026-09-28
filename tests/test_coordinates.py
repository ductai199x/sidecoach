"""Reading board orientation from the coordinate labels.

The real g01 recording proves the White direction (see the vision validation). Here a synthetic
board proves both directions and the no-labels fallback cleanly: rank digits are drawn in the
top-left of each left-edge cell -- 8..1 down for White at the bottom, 1..8 for Black -- exactly
as chess.com draws them, and the reader must recover the side from them.
"""

from __future__ import annotations

import cv2
import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.coordinates import learn_rank_labels, read_bottom_side
from sidecoach.vision.squares import BoardGeometry

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
    assert read_bottom_side(white_board, GEOM, labels) == Side.WHITE
    assert read_bottom_side(black_board, GEOM, labels) == Side.BLACK


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


def test_no_labels_returns_none_so_the_caller_can_ask() -> None:
    labels = learn_rank_labels(make_board(Side.WHITE), GEOM, Side.WHITE)
    # A board with coordinates turned off has nothing to read.
    assert read_bottom_side(make_board(None), GEOM, labels) is None
    # And a theme learned without labels cannot read them.
    assert read_bottom_side(make_board(Side.WHITE), GEOM, {}) is None
