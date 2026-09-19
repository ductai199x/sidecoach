"""Read the board's orientation from the coordinate labels chess.com draws on its edge.

A cached piece-style lets a puzzle or a mid-game board be classified, but turning that into a
position needs the orientation: which corner is a1. A static position is ambiguous under a
180-degree turn, so the pieces alone cannot say. The board's rank labels can: they run 8-to-1
down the left edge when White sits at the bottom, and 1-to-8 when Black does.

We read them with the same learned-exemplar idea the recogniser uses for pieces. When the theme
is first learned from the start position -- where the orientation is known -- the rank digit in
each left-edge cell is captured and labelled with its value. Later, on any board of that theme,
the edge digits are matched against those exemplars and the sequence gives the orientation
deterministically. A board whose owner has turned coordinates off has no labels to read, and the
reader says so (None) rather than guessing -- the caller then asks which side is at the bottom.
"""

from __future__ import annotations

import cv2
import numpy as np

from sidecoach.events import Side
from sidecoach.vision.squares import BoardGeometry

GLYPH = 24  # rank-digit exemplars are normalised to this square before matching
_CORNER_H = 0.40  # the digit sits in the top-left of the cell: this fraction of its height
_CORNER_W = 0.32  # ... and this fraction of its width
_INK = 60.0  # a pixel this far from the square colour is the digit, not the square
_MIN_INK = 4  # fewer foreground pixels than this is no label


def _corner_glyph(cell: np.ndarray) -> np.ndarray | None:
    """The rank digit in a cell's top-left corner as a normalised binary glyph, or None.

    The digit is drawn to contrast the square, so pixels far from the square's own colour are
    the glyph. It is cropped to its bounding box and scaled to a fixed size so digits read at
    slightly different pixel sizes still match.
    """
    if cell.ndim != 3 or cell.size == 0:
        return None
    height, width = cell.shape[:2]
    corner = cell[: max(1, int(height * _CORNER_H)), : max(1, int(width * _CORNER_W))]
    background = np.median(cell.reshape(-1, cell.shape[2]), axis=0)
    diff = np.linalg.norm(corner.astype(float) - background, axis=2)
    mask = diff > _INK
    if int(mask.sum()) < _MIN_INK:
        return None
    # The digit is one component of ink, inset from the cell's edge. A lock a pixel off
    # puts a strip of the neighbouring cell along the window's top or left edge, and a
    # pawn on a small cell reaches into the window from the lower right: neither is the
    # digit. Components touching the top or left edge go first, then the component nearest
    # the corner is the digit. Taken all together, the box stretched over a piece fragment
    # and the same digit no longer matched once the piece had moved on.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count > 2:
        inset = [
            k
            for k in range(1, count)
            if stats[k, cv2.CC_STAT_LEFT] > 0 and stats[k, cv2.CC_STAT_TOP] > 0
        ] or list(range(1, count))
        nearest = min(inset, key=lambda k: stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_TOP])
        mask = labels == nearest
        if int(mask.sum()) < _MIN_INK:
            return None
    rows, cols = np.where(mask)
    cropped = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].astype(np.uint8)
    resized = cv2.resize(cropped * 255, (GLYPH, GLYPH), interpolation=cv2.INTER_NEAREST)
    return resized > 127


def learn_rank_labels(
    frame: np.ndarray, geometry: BoardGeometry, bottom: Side
) -> dict[int, np.ndarray]:
    """Capture the rank-digit exemplar in each left-edge cell, labelled by its known value.

    With `bottom` known (the theme is learned from the start position), the left-edge cell in
    row r shows rank 8-r when White is at the bottom, r+1 when Black is.
    """
    labels: dict[int, np.ndarray] = {}
    for row in range(8):
        rank = (8 - row) if bottom == Side.WHITE else (row + 1)
        glyph = _corner_glyph(geometry.crop(frame, row, 0))
        if glyph is not None:
            labels[rank] = glyph
    return labels


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def read_bottom_side(
    frame: np.ndarray, geometry: BoardGeometry, labels: dict[int, np.ndarray]
) -> Side | None:
    """Which side sits at the bottom, read from the edge rank digits, or None if unreadable.

    Each left-edge digit is matched against the learned exemplars; the sequence down the column
    is 8..1 for White at the bottom and 1..8 for Black. A clear majority (at least six of the
    eight in one direction) decides it; anything less -- a missing label, coordinates turned off,
    an inconsistent read -- returns None so the caller can ask the user instead of guessing.
    """
    if len(labels) < 6:
        return None
    reads: list[int] = []
    for row in range(8):
        glyph = _corner_glyph(geometry.crop(frame, row, 0))
        if glyph is None:
            return None
        digit, score = max(
            ((value, _iou(glyph, exemplar)) for value, exemplar in labels.items()),
            key=lambda pair: pair[1],
        )
        if score < 0.3:
            return None
        reads.append(digit)
    descending = sum(reads[row] == 8 - row for row in range(8))
    ascending = sum(reads[row] == row + 1 for row in range(8))
    if descending >= 6 and descending > ascending:
        return Side.WHITE
    if ascending >= 6 and ascending > descending:
        return Side.BLACK
    return None
