"""The cookie-cutter matcher: an exemplar laid over the cell like a cutter.

An experiment beside the silhouette overlap in `pieces.ThemeModel._classify` (Tai,
2026-09-19), switched by `pieces.MATCHER` so the two can be run against each other on
the stress sweep and the recordings. The idea: the start position hands us every piece's
shape and, on the pixel variant, its very pixels; a cell is then read by laying each
exemplar over it like a cookie cutter and asking whether what lies under the cutter is
that piece and what lies just outside it is the square. The cutter is the exemplar's
silhouette widened by a few pixels; every pixel in it is expected to be one of two
things, and the score is the share of them that are:

- "pixels": a pixel that shows the piece in the reference (its silhouette, less the
  pockets its outline encloses, which show the square) must match the reference's
  pixel -- within PIXEL_TOLERANCE, and nearer to it than to the cell's own square
  colour; any other pixel -- the ring around the piece, the pockets -- must be near the
  cell's square colour, and nearer to it than to the piece's fill. The nearer-than
  clauses are what tell a white fill from a light square: the two lie within the
  tolerance of each other, and on the tolerance alone an empty light square was a
  white bishop. A piece's pixel is matched not to the reference's pixel alone but to
  the segment from it to where the same pixel would land on this cell's square (the
  reference's pixel moved by the difference of the two square colours): a set's
  shading is often translucent, so a shaded pixel takes on some of the square under
  it, and on a highlighted square half of a white queen read as something else. An
  opaque pixel sits at the segment's start, a translucent one along it, and no alpha
  need be known. The pixel just either side of the piece's boundary is its smoothed
  edge, blended with whichever square it stands on, and is left out. The reference is
  the piece's start cell, one per square colour it stood on (the median over the pawns
  of that colour), and the better reference counts, so a shaded side that hid in one
  square's tone is still read from the other.
- "binary": a pixel of the exemplar's fill, eroded by a pixel, must be in the cell's
  foreground; a pixel of the ring outside its full silhouette must not be.

Without the ring a pawn's cutter fits inside a queen as well as a queen's own. The badge
corner is left out as everywhere. The cutter is tried at every shift within SHIFT_MAX
pixels, and the best shift is where the piece sits: a lock a pixel off costs nothing,
and a piece in flight fits best at the window's edge or nowhere, which is what makes it
unsettled. So one measure answers both questions the silhouette overlap and the
centroid answered apart: which piece, and whether it has landed.

The pixel variant reads no foreground at all, which is what the overlap could not do
on a shaded set on a board of its own tone (see `pieces`): the piece's pixels are the
same on either square, only their contrast with the square differs.

The A/B (2026-09-19, 300 seeds of tests/test_stress.py, the six recordings), the
overlap against the two variants with the rest of the pipeline the same:

- overlap: 13 failures over the 900 sweep cases (seeds 34, 128, 212, 251 followed;
  34, 191, 170, 212, 238, 251 ending), about 4.5 minutes; the recordings as in
  DESIGN.md.
- pixels: 18 failures (128 and one 212 case gained; 27, 83, 168, 229 lost on the
  moves, 229, 274, 282, 283 on the endings), 14 minutes; g01 to g03 and g05 the
  same, g04 50 right and 32 wrong for the overlap's 73 and 19, g06 42 and 43 for
  66 and 22. Every case lost is a set with thin detail (chess7, cooke, pirouetti):
  each cell is cropped to whole pixels and resized, so the same piece's pixels sit
  up to a pixel apart from one square to the next, and the same white queen scored
  0.6 to 1.0 by the square it stood on. Box-blurring both sides (3 or 5 wide) or
  matching each pixel to its nearest within a step (SLACK) bought that back and
  cost the margin between pieces that differ in their detail: 2, 7 and 6 failures
  on 30 seeds against 1 without, 0 for the overlap.
- binary: 47 of 91 cases failed on 30 seeds. Its score is the share of a fat region
  that agrees, which every piece of the right colour nearly does; the overlap's
  intersection over union is the sharper measure of the same masks.

So the overlap stays the matcher (pieces.MATCHER "iou"); the variants stay behind
the switch, and the exemplars keep their references, for the next look at this. What
the experiment settled: matching pixels needs the pieces on the same pixel grid,
which a whole-pixel crop of a board of fractional cells does not give, and a
classifier that is to beat the overlap will have to be trained on the crops (DESIGN
section 5.2), not compared pixel for pixel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

# The search window for the cutter, in pixels of the working cell (48): a lock is good
# to a pixel or two, and four pixels is a twelfth of a cell, past the landing tolerance.
SHIFT_MAX = 4
# A cell pixel is the reference's, or the square's, when within this of it (Euclidean
# over BGR): the smoothing of a rendered piece moves an edge pixel a few tens of levels.
PIXEL_TOLERANCE = 40.0
# The cutter reaches this far outside the silhouette: the ring where the square must show.
RING = 3
# Binary variant: the fill eroded this much is where the foreground must be, and the ring
# starts this far outside the full silhouette.
CORE_ERODE = 1
RING_INNER = 1
# The score a reading must reach to name the piece, and the lead over the runner-up.
MIN_MATCH = 0.7
MIN_MATCH_MARGIN = 0.03
# Pixel variant: a cell pixel may match the reference's pixel this many working pixels
# away from its own place (the nearest of them counts). Each cell is cropped to whole
# pixels and resized, so one piece's pixels sit up to a pixel apart, and scaled a
# little differently, from one square to the next (and a browser draws a sprite at
# each square's own fraction of a pixel); a set's thin detail then never matches pixel
# for pixel, and the same white queen scored 0.6 to 1.0 by the square it stood on.
# Blurring both instead cost the margin between pieces that differ in their detail.
SLACK = int(os.environ.get("SIDECOACH_COOKIE_SLACK", "0"))
# The reference and the cell are box-blurred this wide before comparison (1: not at all).
BLUR = int(os.environ.get("SIDECOACH_COOKIE_BLUR", "1"))
_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class Reference:
    """A piece as it stood on one square colour at the start: the working patch (BGR
    uint8), the square's colour, the pixels that show the piece (its silhouette less the
    pockets), and the piece's fill colour."""

    patch: np.ndarray
    background: np.ndarray
    piece: np.ndarray
    fill: np.ndarray


def shifts(limit: int = SHIFT_MAX) -> np.ndarray:
    """Every (dx, dy) within the window, the smallest shifts first."""
    out = [(dx, dy) for dy in range(-limit, limit + 1) for dx in range(-limit, limit + 1)]
    out.sort(key=lambda s: (abs(s[0]) + abs(s[1]), s))
    return np.array(out, dtype=np.int32)


def _grid(moves: np.ndarray, limit: int = SHIFT_MAX) -> np.ndarray:
    """The index into `moves` of each (dy, dx) of the window, as a (2 limit + 1)^2 grid."""
    grid = np.zeros((2 * limit + 1, 2 * limit + 1), dtype=np.int64)
    for i, (dx, dy) in enumerate(moves):
        grid[int(dy) + limit, int(dx) + limit] = i
    return grid


def _nearest_within(values: np.ndarray, moves: np.ndarray, slack: int) -> np.ndarray:
    """`values` (n, size, size) over the shifts in `moves`: at each shift, the least
    value over the shifts within `slack` of it -- the distance to the nearest matching
    reference pixel a step or so away, computed once over the shift stack."""
    if slack <= 0:
        return values
    grid = _grid(moves)
    by_grid = values[grid]  # (2m+1, 2m+1, size, size)
    m = grid.shape[0]
    padded = np.full((m + 2 * slack, m + 2 * slack, *values.shape[1:]), np.inf, dtype=values.dtype)
    padded[slack : slack + m, slack : slack + m] = by_grid
    least = by_grid.copy()
    for dy in range(-slack, slack + 1):
        for dx in range(-slack, slack + 1):
            if dy or dx:
                np.minimum(
                    least,
                    padded[slack + dy : slack + dy + m, slack + dx : slack + dx + m],
                    out=least,
                )
    out = np.empty_like(values)
    out[grid] = least
    return out


def blurred(patch: np.ndarray) -> np.ndarray:
    """The patch box-blurred BLUR wide (see BLUR); the patch itself when BLUR is 1."""
    return cv2.blur(patch, (BLUR, BLUR)) if BLUR > 1 else patch


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask
    return cv2.dilate(mask.astype(np.uint8), _KERNEL, iterations=n).astype(bool)


def _erode(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return mask
    return cv2.erode(mask.astype(np.uint8), _KERNEL, iterations=n).astype(bool)


def _shifted(source: np.ndarray, moves: np.ndarray) -> np.ndarray:
    """`source` (h, w[, c]) moved by each (dx, dy) in `moves`, zero-filled at the edges."""
    h, w = source.shape[:2]
    out = np.zeros((len(moves), *source.shape), dtype=source.dtype)
    for i, (dx, dy) in enumerate(moves):
        dx, dy = int(dx), int(dy)
        out[i, max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)] = source[
            max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)
        ]
    return out


@dataclass(frozen=True)
class Cutter:
    """One reference's cutter at every shift in the window."""

    moves: np.ndarray  # (n, 2) of (dx, dy)
    # Pixel variant: where the piece must show (n, size, size), where the square must
    # (n, size, size), the reference's pixels (n, size, size, 3) uint8, its fill and
    # the square colour it stood on. Binary variant: the eroded fill and the ring.
    piece: np.ndarray
    square: np.ndarray
    pixels: np.ndarray | None
    fill: np.ndarray | None
    background: np.ndarray | None
    count: np.ndarray  # (n,) pixels judged at each shift


def _cutter(
    piece: np.ndarray,
    square: np.ndarray,
    keep: np.ndarray,
    reference: Reference | None = None,
) -> Cutter:
    moves = shifts()
    pieces = _shifted(piece & keep, moves) & keep
    squares = _shifted(square & keep, moves) & keep
    count = pieces.sum(axis=(1, 2)) + squares.sum(axis=(1, 2))
    if reference is None:
        return Cutter(moves, pieces, squares, None, None, None, np.maximum(count, 1))
    return Cutter(
        moves,
        pieces,
        squares,
        _shifted(blurred(reference.patch), moves),
        reference.fill,
        reference.background,
        np.maximum(count, 1),
    )


def build_cutters(
    mask: np.ndarray,
    tone: np.ndarray,
    references: tuple[Reference, ...],
    keep: np.ndarray,
    variant: str,
) -> tuple[Cutter, ...]:
    """The cutters for an exemplar: `mask` its full silhouette, `tone` its fill, and for
    the pixel variant `references` its start cells; `keep` is the set of pixels that
    take part (everything but the badge corner)."""
    region = _dilate(mask, RING)
    if variant != "pixels":
        core = _erode(tone, CORE_ERODE)
        ring = region & ~_dilate(mask, RING_INNER)
        return (_cutter(core, ring, keep),)
    cutters = []
    for ref in references:
        edge = _dilate(ref.piece, 1) & ~_erode(ref.piece, 1)
        judged = region & ~edge
        cutters.append(_cutter(judged & ref.piece, judged & ~ref.piece, keep, ref))
    return tuple(cutters)


def score(
    cutters: tuple[Cutter, ...],
    inner: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
    variant: str,
) -> tuple[float, tuple[int, int]]:
    """The best share of cutter pixels that agree with the cell over the exemplar's
    cutters and every shift, and the shift it was found at. `inner` is the cell's
    working patch (BGR uint8), `foreground` its silhouette mask, `background` its own
    square colour."""
    best = (-1.0, (0, 0))
    cell = blurred(inner).astype(np.float32)
    to_square = np.linalg.norm(cell - background.astype(np.float32), axis=2)
    for cutter in cutters:
        if variant == "pixels":
            assert cutter.pixels is not None
            assert cutter.fill is not None and cutter.background is not None
            to_fill = np.linalg.norm(cell - cutter.fill.astype(np.float32), axis=2)
            near_square = (to_square <= PIXEL_TOLERANCE) & (to_square < to_fill)
            # The distance from each cell pixel to the segment between the reference's
            # pixel and that pixel moved onto this cell's square colour (see the module).
            away = cell[None] - cutter.pixels.astype(np.float32)
            along = (background - cutter.background).astype(np.float32)
            length = float(along @ along)
            if length > 1.0:
                t = np.clip((away @ along) / length, 0.0, 1.0)
                away = away - t[..., None] * along
            to_piece = _nearest_within(np.sqrt((away**2).sum(axis=3)), cutter.moves, SLACK)
            near_piece = (to_piece <= PIXEL_TOLERANCE) & (to_piece < to_square[None])
            hits = (near_piece & cutter.piece).sum(axis=(1, 2)) + (
                cutter.square & near_square[None]
            ).sum(axis=(1, 2))
        else:
            hits = (cutter.piece & foreground[None]).sum(axis=(1, 2)) + (
                cutter.square & ~foreground[None]
            ).sum(axis=(1, 2))
        scores = hits / cutter.count
        at = int(np.argmax(scores))  # the smallest shift wins a tie: moves are sorted so
        if scores[at] > best[0]:
            best = (float(scores[at]), (int(cutter.moves[at][0]), int(cutter.moves[at][1])))
    return best
