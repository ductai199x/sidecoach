"""Cell geometry for a locked board: crops, and the mapping between cells and squares.

Rows and columns are screen order: row 0 is the top of the board on screen, column 0 the
left. Which chess square a cell holds depends on which side sits at the bottom, and that is
detected per game, never assumed (DESIGN.md section 5.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import cv2
import numpy as np

from sidecoach.events import Region, Side

CELL = 48


def square_at(row: int, col: int, bottom: Side) -> chess.Square:
    if bottom == Side.WHITE:
        return chess.square(col, 7 - row)
    return chess.square(7 - col, row)


def row_col(square: chess.Square, bottom: Side) -> tuple[int, int]:
    file, rank = chess.square_file(square), chess.square_rank(square)
    if bottom == Side.WHITE:
        return 7 - rank, file
    return rank, 7 - file


def is_light(row: int, col: int) -> bool:
    """Top-left cell of a full board is always a light square in either orientation."""
    return (row + col) % 2 == 0


# An edge this close to a pixel boundary is taken as on it: the pixel is nearly pure, and
# a lock measured a hair off must not drop a whole pixel of every cell.
_EDGE_SLACK = 0.1


@dataclass(frozen=True)
class BoardGeometry:
    region: Region
    cell_px: float

    def cell_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        """The pixels that lie wholly inside the cell, as (x0, y0, x1, y1), x1 and y1 past
        the end.

        A cell's true edges fall between pixels (641 px for eight squares is 80.125 a
        square), and a pixel straddling an edge is a blend of the cell and its neighbour --
        the other square colour. Rounding the edges to the nearest pixel let such blended
        pixels into the crop on both sides of a column and both sides of a row, and the
        one cell at their crossing then wore a dark line on all four edges: the hover
        outline's signature, permanently, so no move through it ever settled (1.e4 in the
        second live game). Taking only whole pixels costs at most one pixel a side.
        """
        x0 = int(math.ceil(self.region.x + col * self.cell_px - _EDGE_SLACK))
        y0 = int(math.ceil(self.region.y + row * self.cell_px - _EDGE_SLACK))
        x1 = int(math.floor(self.region.x + (col + 1) * self.cell_px + _EDGE_SLACK))
        y1 = int(math.floor(self.region.y + (row + 1) * self.cell_px + _EDGE_SLACK))
        return x0, y0, x1, y1

    def crop(self, frame: np.ndarray, row: int, col: int, size: int = CELL) -> np.ndarray:
        """The cell resized to `size` by `size`. Zeros if the cell is off the frame."""
        x0, y0, x1, y1 = self.cell_box(row, col)
        height, width = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
        # INTER_AREA averages pixels when shrinking but degenerates to nearest-neighbour
        # when enlarging; a small cell blown up to the working size needs interpolation.
        shrinking = x1 - x0 >= size and y1 - y0 >= size
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=interpolation)

    def crops(self, frame: np.ndarray, size: int = CELL) -> np.ndarray:
        """All 64 cells as an array of shape (8, 8, size, size, channels)."""
        out = np.zeros((8, 8, size, size, frame.shape[2]), dtype=frame.dtype)
        for r in range(8):
            for c in range(8):
                out[r, c] = self.crop(frame, r, c, size)
        return out
