"""Cell geometry for a locked board: crops, and the mapping between cells and squares.

Rows and columns are screen order: row 0 is the top of the board on screen, column 0 the
left. Which chess square a cell holds depends on which side sits at the bottom, and that is
detected per game, never assumed (DESIGN.md section 5.2).
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class BoardGeometry:
    region: Region
    cell_px: float

    def cell_box(self, row: int, col: int) -> tuple[int, int, int, int]:
        x0 = int(round(self.region.x + col * self.cell_px))
        y0 = int(round(self.region.y + row * self.cell_px))
        x1 = int(round(self.region.x + (col + 1) * self.cell_px))
        y1 = int(round(self.region.y + (row + 1) * self.cell_px))
        return x0, y0, x1, y1

    def crop(self, frame: np.ndarray, row: int, col: int, size: int = CELL) -> np.ndarray:
        """The cell resized to `size` by `size`. Zeros if the cell is off the frame."""
        x0, y0, x1, y1 = self.cell_box(row, col)
        height, width = frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
        return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)

    def crops(self, frame: np.ndarray, size: int = CELL) -> np.ndarray:
        """All 64 cells as an array of shape (8, 8, size, size, channels)."""
        out = np.zeros((8, 8, size, size, frame.shape[2]), dtype=frame.dtype)
        for r in range(8):
            for c in range(8):
                out[r, c] = self.crop(frame, r, c, size)
        return out
