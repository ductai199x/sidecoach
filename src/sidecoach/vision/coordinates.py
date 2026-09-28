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
from sidecoach.vision.squares import CELL, BoardGeometry

GLYPH = 24  # rank-digit exemplars are normalised to this square before matching
_CORNER_H = 0.30  # the digit sits in the top-left of the cell: this fraction of its height
_CORNER_W = 0.20  # ... and this fraction of its width (a rook reaches in from the lower right)
# A pixel this far from the square colour is the digit, not the square. The digits are
# drawn in the other square's colour, and on a dark board the two lie 48 levels apart:
# at 60 a 44-pixel digit, all anti-aliased blends, was a scatter of specks and nothing
# was learned but the 8. A wood grain strays 20 at most.
_INK = 30.0
_MIN_INK = 4  # fewer foreground pixels than this is no label
# A corner window more than this full of ink holds no digit: the square's colour was taken
# from its neighbour, so the cell is cut off its square -- a lock a tenth of a cell out is
# enough. A digit fills a quarter to a third of the window, a cut that far out nine tenths
# or all of it (Tai's Pixel, 2026-09-27: a start seen through a lock that far out cached
# all eight labels as solid blocks, and every edge digit then read as the same one).
_CORNER_FULL = 0.6
# A hole this share of a digit's box or more is a loop: an 8's are 2.4 to 14 % on chess.com's
# boards (80-86 px) and rendered ones (44-110 px), the pinholes in a fantasy piece's shading
# that reached the corner 0.7 % (2026-09-28).
_LOOP = 0.02


def _square_colour(cell: np.ndarray) -> np.ndarray:
    """The square's colour: the median of the bands along the cell's right and bottom
    edges, a twentieth of the cell in from the edge and as deep again."""
    n = min(cell.shape[:2])
    outer = max(1, int(round(n * 0.05)))
    inner = outer + max(1, int(round(n * 0.07)))
    bands = np.concatenate(
        [
            cell[:, -inner:-outer].reshape(-1, cell.shape[2]),
            cell[-inner:-outer, :].reshape(-1, cell.shape[2]),
        ]
    )
    return np.median(bands, axis=0)


def _corner_glyph(cell: np.ndarray) -> np.ndarray | None:
    """The rank digit in a cell's top-left corner as a normalised binary glyph, or None:
    `_digit` scaled to a fixed size, so digits read at slightly different pixel sizes still
    match."""
    digit = _digit(cell)
    if digit is None:
        return None
    resized = cv2.resize(
        digit.astype(np.uint8) * 255, (GLYPH, GLYPH), interpolation=cv2.INTER_NEAREST
    )
    return resized > 127


def _digit(cell: np.ndarray) -> np.ndarray | None:
    """The rank digit in a cell's top-left corner, its ink cropped to its box, or None.

    The digit is drawn to contrast the square, so pixels far from the square's own colour are
    the digit.
    """
    if cell.ndim != 3 or cell.size == 0:
        return None
    height, width = cell.shape[:2]
    corner = cell[: max(1, int(height * _CORNER_H)), : max(1, int(width * _CORNER_W))]
    # The square's colour from the bands just inside the cell's right and bottom edges,
    # which neither the digit (top left) nor a piece (the middle) reaches: the median of
    # the whole cell is the piece's colour when a rook fills half of it, and the whole
    # corner was ink then -- the "1" learned from the rook's cell at the start was a
    # solid block; the ring all round the cell has the digit in it.
    background = _square_colour(cell)
    diff = np.linalg.norm(corner.astype(float) - background, axis=2)
    mask = diff > _INK
    if int(mask.sum()) < _MIN_INK or mask.mean() > _CORNER_FULL:
        return None
    # The digit is one component of ink, inset from the cell's edge. A lock a pixel off
    # puts a strip of the neighbouring cell along the window's top or left edge, and a
    # pawn on a small cell reaches into the window from the lower right: neither is the
    # digit. Components touching the top or left edge go first, then the component nearest
    # the corner is the digit. Taken all together, the box stretched over a piece fragment
    # and the same digit no longer matched once the piece had moved on.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
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
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _loops(digit: np.ndarray) -> list[float]:
    """The closed loops in a digit's ink, each as the height of its middle in the digit (0 at
    the top, 1 at the foot): an 8 has two, one above the other, a 1 none, in any typeface. A
    hole under _LOOP of the digit's box is a pinhole in a piece's shading, not a loop."""
    ink = np.pad(digit, 1).astype(np.uint8)
    contours, hierarchy = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []
    heights: list[float] = []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0], strict=True):
        if parent < 0:
            continue
        inside = np.zeros_like(ink)
        cv2.drawContours(inside, [contour], -1, 1, -1)
        if int((inside & (1 - ink)).sum()) >= _LOOP * digit.size:
            heights.append((float(contour[:, 0, 1].mean()) - 1) / digit.shape[0])
    return heights


def _eight(digit: np.ndarray) -> bool:
    loops = _loops(digit)
    return len(loops) == 2 and min(loops) < 0.5 < max(loops)


def eights(frame: np.ndarray, geometry: BoardGeometry) -> Side | None:
    """Which side sits at the bottom, read from where the 8 is, with nothing learned: the
    corner digit with two loops at the top of the left edge and one with none at its foot is
    White at the bottom, the other way round Black; anything else (coordinates off, a piece
    over a digit) is None. Learned labels read more boards (a digit half covered still
    matches); this reads a board whose labels are not learned, or were learned wrong -- Tai's
    Pixel, 2026-09-28, the night after a start seen through an off lock cached them as solid
    blocks: every puzzle asked for Flip, and the 8 was there to read all along. A cell is
    never shrunk to read it: scaled down to the working size, the loops of an 8 closed up."""
    size = max(CELL, round(geometry.cell_px))
    top = _digit(geometry.crop(frame, 0, 0, size))
    foot = _digit(geometry.crop(frame, 7, 0, size))
    if top is None or foot is None:
        return None
    if _eight(top) and not _loops(foot):
        return Side.WHITE
    if _eight(foot) and not _loops(top):
        return Side.BLACK
    return None


def learn_rank_labels(
    frame: np.ndarray, geometry: BoardGeometry, bottom: Side
) -> dict[int, np.ndarray]:
    """Capture the rank-digit exemplar in each left-edge cell, labelled by its known value.

    With `bottom` known (the theme is learned from the start position), the left-edge cell in
    row r shows rank 8-r when White is at the bottom, r+1 when Black is. Labels that do not
    read `bottom` back from the frame they were learned from are none: kept, they would say
    nothing or the wrong side on every board after it.
    """
    labels: dict[int, np.ndarray] = {}
    for row in range(8):
        rank = (8 - row) if bottom == Side.WHITE else (row + 1)
        glyph = _corner_glyph(geometry.crop(frame, row, 0))
        if glyph is not None:
            labels[rank] = glyph
    return labels if read_labels(frame, geometry, labels) == bottom else {}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return int((a & b).sum()) / union if union else 0.0


def read_bottom_side(
    frame: np.ndarray, geometry: BoardGeometry, labels: dict[int, np.ndarray]
) -> Side | None:
    """Which side sits at the bottom, read from the edge rank digits -- the learned labels
    (`read_labels`), else where the 8 is (`eights`) -- or None if unreadable."""
    return read_labels(frame, geometry, labels) or eights(frame, geometry)


def read_labels(
    frame: np.ndarray, geometry: BoardGeometry, labels: dict[int, np.ndarray]
) -> Side | None:
    """Which side sits at the bottom, read with the learned labels, or None if unreadable.

    Each left-edge digit is matched against the learned exemplars; the sequence down the column
    is 8..1 for White at the bottom and 1..8 for Black. A clear majority (at least six of the
    eight in one direction) decides it; anything less -- coordinates turned off, an inconsistent
    read -- returns None so the caller can ask the user instead of guessing. A cell whose digit
    cannot be read (a piece over it, a piece whose ink touches it) is left out rather than
    ending the read: on a mid-game board the edge cells hold pieces, and one of them was
    enough to leave every cold start without an orientation.
    """
    if len(labels) < 6:
        return None
    reads: dict[int, int] = {}
    for row in range(8):
        glyph = _corner_glyph(geometry.crop(frame, row, 0))
        if glyph is None:
            continue
        digit, score = max(
            ((value, _iou(glyph, exemplar)) for value, exemplar in labels.items()),
            key=lambda pair: pair[1],
        )
        if score >= 0.3:
            reads[row] = digit
    descending = sum(digit == 8 - row for row, digit in reads.items())
    ascending = sum(digit == row + 1 for row, digit in reads.items())
    if descending >= 6 and descending > ascending:
        return Side.WHITE
    if ascending >= 6 and ascending > descending:
        return Side.BLACK
    return None
