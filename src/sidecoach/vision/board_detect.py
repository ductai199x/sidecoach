"""Classical board detection.

A 2D chess board is an axis-aligned square of 8 by 8 equal cells in two colours. The detector
works in three passes, see DESIGN.md section 5.2:

1. Coarse search. For every cell size in a geometric range, the frame is shrunk so that a cell
   is a few pixels, and every position is scored on how well the border-ring means of its 64
   cells alternate in a checkerboard: |sum(sign * mean)| / sum|mean - board mean|. That is 1
   for a perfect empty board, near 0 for anything else, and it is scale-selective because a
   doubled or halved cell size cancels the pattern. The rings rather than the whole cells, so
   that pieces do not drag the means. A position must also pass the amplitude gate,
   |sum(sign * mean)| / 64, half the gap between the two square colours, which background
   cannot inflate the way it inflates the deviation. Each scale keeps its local maxima, one
   per board-sized neighbourhood, so the plateau around an empty board and the many windows
   into a larger checkerboard cannot crowd out a board whose pieces lower its score.
2. Refinement. Around each surviving candidate the gradient profiles along x and y show peaks
   at the cell boundaries. A comb on the seven boundaries between the cells is fitted for
   sub-pixel offset and pitch; it places the board to within a whole cell, and of the
   placements a cell apart the one whose cells read most like a board is kept.
3. Validation. Each cell's colour is read as the median of a thin border ring, which pieces
   and highlights rarely disturb; a cell the frame cuts is read on the part it shows when
   that is half the cell or more. The two square colours are the medians of the even and the
   odd cells, or the colours a lock was taken with, and the fit score is the fraction of the
   readable cells that match the checker pattern in the polarity of those colours. A
   candidate whose pattern carries on in the ring of cells just outside it, a whole-cell
   shift or a window into a larger checkerboard, is rejected, and so is a board the frame
   does not show whole.

Pieces, the cursor, arrows, highlights and coordinate labels are local outliers against 64
cells of evidence, so they lower the scores a little and do not move the result.
"""

from __future__ import annotations

import os
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import NamedTuple

import cv2
import numpy as np

from sidecoach.events import Region

_SIGNS = np.array(
    [[1.0 if (r + c) % 2 == 0 else -1.0 for c in range(8)] for r in range(8)], dtype=np.float64
)
MIN_READABLE_CELLS = 24
# The smallest cell looked for: a 192 px board. Below it a set's pieces are a few pixels.
MIN_CELL_PX = 24
# Boards whose width is within this fraction of the largest are one size to the ranking.
SAME_SIZE = 0.95
# How far the fitted pitch may be from the candidate's: a coarse scale step, a small resize.
PITCH_SEARCH = 0.12

# (light, dark, tolerance): the square colours a board is read with. `light` is the colour of
# the cells the checker pattern puts a light square on, the top-left cell and its parity,
# `dark` the colour of the others. On a theme whose light squares are the darker ones in
# luminance, light < dark and the pattern is read inverted; see `expected_signs`. `tolerance`
# is how far a cell's median may lie from either colour and still be read as it.
Colours = tuple[float, float, float]


@dataclass(frozen=True)
class BoardCandidate:
    region: Region
    cell_px: float
    score: float  # the coarse checker score, only meaningful before refinement
    contrast: float
    fit: float = 0.0
    occupied: int = 0  # cells whose interior differs from their ring: pieces and marks
    readable: int = 0  # cells whose ring reads as one of the two square colours


@dataclass(frozen=True)
class Refinement:
    """A candidate after the boundary fit, with the comb strength at its new and old geometry.

    `strength` is the summed boundary gradient at the teeth of the comb (the boundaries
    between the cells) on both axes; `reference` is the same sum at the geometry the fit
    started from, over the same profiles, so the two compare: a re-fit that found nothing
    better than where it started scores about the same, one that corrected a drift or a
    resize scores clearly more.
    """

    board: BoardCandidate
    strength: float
    reference: float


def _window_sums(ii: np.ndarray, size: int) -> np.ndarray:
    """Sum of every size-by-size window at stride 1, from an integral image."""
    return ii[size:, size:] - ii[:-size, size:] - ii[size:, :-size] + ii[:-size, :-size]


def _cell_means(gray: np.ndarray, cell: int) -> np.ndarray:
    """Mean of every cell-sized window at stride 1. Shape (H - cell + 1, W - cell + 1)."""
    return _window_sums(cv2.integral(gray), cell) / float(cell * cell)


def _ring_means(gray: np.ndarray, cell: int, band: int = 1) -> np.ndarray:
    """Mean of the border ring of every cell-sized window, same shape as `_cell_means`.

    Pieces sit in the middle of their squares, so the ring reads the square's colour where
    the whole-cell mean would be dragged towards the piece.
    """
    if cell <= 2 * band + 1:
        return _cell_means(gray, cell)
    ii = cv2.integral(gray)
    outer = _window_sums(ii, cell)
    inner_size = cell - 2 * band
    inner = _window_sums(ii, inner_size)[band : band + outer.shape[0], band : band + outer.shape[1]]
    return (outer - inner) / float(cell * cell - inner_size * inner_size)


def _checker_response(
    gray: np.ndarray, cell: int, *, min_contrast: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(score, amplitude) maps for a board whose top-left cell starts at each position.

    The score is |sum(sign * mean)| / sum|mean - board mean|, 1 for a perfect empty board.
    The amplitude |sum(sign * mean)| / 64 is half the gap between the two square colours
    for a board and is only lowered by cells that are not board, unlike the deviation,
    which background cells inflate; it is the contrast gate.

    The signed and plain sums over the 8x8 grid of cell means are separable -- eight
    shifted adds along the columns, then eight along the rows -- so they cost sixteen
    passes over the map, not sixty-four. The deviation is not separable; with
    `min_contrast` it is summed only where the amplitude passes the gate (a few thousand
    positions of a frame's million), and the score is zero elsewhere.
    """
    means = _ring_means(gray, cell)
    h = means.shape[0] - 7 * cell
    w = means.shape[1] - 7 * cell
    if h <= 0 or w <= 0:
        empty = np.zeros((0, 0))
        return empty, empty
    # Along the columns: the signed and plain sums of the eight cells of each row of the
    # board, for every row position of the map.
    rows_h = means.shape[0]
    signed_cols = np.zeros((rows_h, w), dtype=np.float64)
    plain_cols = np.zeros((rows_h, w), dtype=np.float64)
    for c in range(8):
        block = means[:, c * cell : c * cell + w]
        plain_cols += block
        if c % 2 == 0:
            signed_cols += block
        else:
            signed_cols -= block
    acc = np.zeros((h, w), dtype=np.float64)
    total = np.zeros((h, w), dtype=np.float64)
    for r in range(8):
        acc += (1.0 if r % 2 == 0 else -1.0) * signed_cols[r * cell : r * cell + h]
        total += plain_cols[r * cell : r * cell + h]
    amplitude = np.abs(acc) / 64.0
    board_mean = total / 64.0
    score = np.zeros((h, w), dtype=np.float64)
    if min_contrast is None:
        deviation = np.zeros((h, w), dtype=np.float64)
        for r in range(8):
            for c in range(8):
                deviation += np.abs(
                    means[r * cell : r * cell + h, c * cell : c * cell + w] - board_mean
                )
        score = np.abs(acc) / (deviation + 1e-6)
        return score, amplitude
    ys, xs = np.nonzero(amplitude >= min_contrast)
    if ys.size:
        deviation = np.zeros(ys.size, dtype=np.float64)
        mean_at = board_mean[ys, xs]
        for r in range(8):
            for c in range(8):
                deviation += np.abs(means[ys + r * cell, xs + c * cell] - mean_at)
        score[ys, xs] = np.abs(acc[ys, xs]) / (deviation + 1e-6)
    return score, amplitude


def _local_maxima(
    score: np.ndarray, mask: np.ndarray, cell: int, radius: int, limit: int
) -> list[tuple[int, int]]:
    """(y, x) of the gated score's peaks, best first, at most `limit`, none within `radius`.

    A cell-sized dilation thins every plateau to the positions that equal their
    neighbourhood's maximum, then peaks are taken in score order and everything within
    `radius` of a taken peak is dropped. A plateau of equal scores therefore yields one
    position and the windows into a larger checkerboard, a cell apart, one per neighbourhood,
    while a second board at the same scale, at least eight cells away, keeps its own peak.
    """
    gated = np.where(mask, score, 0.0)
    dilated = cv2.dilate(gated, np.ones((cell, cell), dtype=np.uint8))
    ys, xs = np.nonzero(mask & (gated >= dilated))
    values = gated[ys, xs]
    remaining = np.ones(ys.size, dtype=bool)
    peaks: list[tuple[int, int]] = []
    while len(peaks) < limit:
        idx = np.flatnonzero(remaining)
        if idx.size == 0:
            break
        i = idx[int(np.argmax(values[idx]))]  # ties go to the first in raster order
        y, x = int(ys[i]), int(xs[i])
        peaks.append((y, x))
        remaining &= (np.abs(ys - y) > radius) | (np.abs(xs - x) > radius)
    return peaks


def _iou(a: Region, b: Region) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def _nms(
    candidates: list[BoardCandidate], key: str, iou_threshold: float = 0.3
) -> list[BoardCandidate]:
    kept: list[BoardCandidate] = []
    for cand in sorted(candidates, key=lambda c: getattr(c, key), reverse=True):
        if all(_iou(cand.region, k.region) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


def coarse_search(
    gray: np.ndarray,
    *,
    min_cell: int = MIN_CELL_PX,
    max_cell: int | None = None,
    work_cell: int = 8,
    scale_step: float = 1.06,
    min_score: float = 0.45,
    min_contrast: float = 6.0,
    per_scale: int = 10,
) -> list[BoardCandidate]:
    """Candidates from the multi-scale checker response, before refinement.

    Each scale contributes at most `per_scale` peaks of its gated score map, no two within
    half a board of each other, so a plateau of perfect scores around one board cannot fill
    the quota and hide a second board whose pieces lower its score.

    A board flush with the frame's right or bottom edge may be found a cell inward: at a
    scale a few percent larger than its cells, the window that sits on it runs past the
    frame. The refinement's choice of phase puts it back.
    """
    height, width = gray.shape
    limit = max_cell if max_cell is not None else min(height, width) // 8
    sizes: list[float] = []
    size = float(min_cell)
    while size <= limit:
        sizes.append(size)
        size *= scale_step

    def at_scale(size: float) -> list[BoardCandidate]:
        # Every caller searches from well above the work cell, so the frame always shrinks.
        factor = work_cell / size
        small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        cell = work_cell
        score, contrast = _checker_response(small, cell, min_contrast=min_contrast)
        found: list[BoardCandidate] = []
        if score.size:
            mask = (score >= min_score) & (contrast >= min_contrast)
            full_cell = cell / factor
            side = int(round(8 * full_cell))
            for y, x in _local_maxima(score, mask, cell, radius=4 * cell, limit=per_scale):
                region = Region(int(round(x / factor)), int(round(y / factor)), side, side)
                found.append(
                    BoardCandidate(region, full_cell, float(score[y, x]), float(contrast[y, x]))
                )
        return found

    # The scales are searched side by side, each on its own thread (OpenCV and numpy let go
    # of the interpreter for the frame-sized passes), and their boards gathered in scale
    # order, as they were one after another: the same boards, in a fraction of the time
    # (round 3: a full search at 4K was 0.6 s, each scale's shrunk frame 5 ms of it).
    found = [board for scale in _SCALES.map(at_scale, sizes) for board in scale]
    return _nms(found, "score")


# The threads `coarse_search` searches its scales on.
_SCALES = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1)))


def _gradient_profile(gray8: np.ndarray, axis: int, mask: np.ndarray) -> np.ndarray:
    """The summed step across every pixel edge; axis=1 gives a profile along x.

    Entry k is the edge between pixels k - 1 and k, where a board whose cells start at pixel
    k has its boundary, so a tooth lands on the boundary itself and not half a pixel to one
    side (entry 0 reads nothing). A step counts only where both its pixels take part
    (`mask`, 255 where a pixel does): the mask leaves out the cells' middles, where the
    pieces are, and within the board whatever is not the square's own colour.

    In whole grey levels (`gray8`, the grey image's own values) and OpenCV's passes: every
    step and every sum is a whole number, so these are the sums the float passes made, to
    the last bit, at a sixth of the cost -- the float passes, a temporary image each, were
    38 ms of a 4K re-fit's 83 (round 3).
    """
    if axis == 1:
        step = cv2.absdiff(gray8[:, 1:], gray8[:, :-1])
        both = cv2.bitwise_and(mask[:, 1:], mask[:, :-1])
    else:
        step = cv2.absdiff(gray8[1:], gray8[:-1])
        both = cv2.bitwise_and(mask[1:], mask[:-1])
    sums = cv2.reduce(cv2.bitwise_and(step, both), 1 - axis, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    return np.concatenate([[0.0], sums.ravel().astype(np.float64)])


# The comb's teeth: the seven boundaries between a board's cells, in cells from its edge.
_TEETH = np.arange(1, 8, dtype=np.float64)


def _comb_strength(profile: np.ndarray, offset: float, pitch: float) -> float:
    """Summed profile at the teeth of a board whose first cell starts at `offset`; a tooth
    past either end of the profile reads nothing."""
    return float(_on_grid(offset + pitch * _TEETH, profile).sum())


def _comb_fit(
    profile: np.ndarray, start: float, pitch: float, offset_cells: float
) -> tuple[float, float]:
    """The (offset, pitch) of the best comb over a gradient profile: the offset within
    `offset_cells` cells of `start`, the pitch within PITCH_SEARCH of `pitch`.

    The teeth are the boundaries between the cells, which show whatever lies around the
    board. Its outer edges are not teeth: against the frame's edge or past it, or on a
    background of one of the square colours, they show nothing, and a comb that counted them
    favoured the placement a cell over (a board flush with the frame's edge, or a cell from
    it, re-fitted a cell off: the geometry sweep, 2026-09-27). So the comb places the board
    to within a whole cell: of the placements a cell apart it returns the one nearest
    `start`, and `refine` picks the cell. (Its passes, each started where the last ended,
    walked two cells off a board past the frame's edge when each took the strongest.)

    Coarse to fine: whole pixels and one-percent pitch steps over the whole window first,
    then quarter pixels and 0.2 percent steps around the best, which is twenty times
    cheaper than a fine grid everywhere and lands on the same optimum. The fine pass
    searches offsets as far as a coarse pitch step moves the far teeth: a pitch a step
    off wants its offset a few pixels off to hold the middle of the board, and a window
    of one pixel kept the comb there, three pixels early and a quarter of a percent
    long at 144-pixel cells. A boundary's peak is a pixel wide, and each pass reads the
    profile blurred by as far as its grid lets a tooth fall from a boundary: the coarse
    pass's last tooth half a pitch step times seven, which at 105-pixel cells is four
    pixels, and a comb at the board's own pitch read its far boundaries as nothing, lost
    to one on the pieces' edges, and the re-fit settled seven percent short (lock_bench,
    2026-09-27); the fine pass's, a pixel.
    """
    step = max(0.25, pitch * 0.01)

    def best_over(
        seen: np.ndarray, pitches: np.ndarray, offsets: np.ndarray
    ) -> tuple[float, float, float]:
        # Every pitch at once, one interpolation (the same numbers as a pass a pitch); the
        # first strongest pitch wins, as it did a pitch at a time.
        best = (start, pitch, 0.0)
        positions = offsets[None, :, None] + pitches[:, None, None] * _TEETH[None, None, :]
        strengths = _on_grid(positions, seen).sum(axis=2)
        at = strengths.argmax(axis=1)
        for k, p in enumerate(pitches):
            i = int(at[k])
            if strengths[k, i] > best[2]:
                best = (float(offsets[i]), float(p), float(strengths[k, i]))
        return best

    reach = pitch * offset_cells
    o, p, _ = best_over(
        _blurred(profile, int(np.ceil(3.5 * step))),
        np.arange(pitch * (1 - PITCH_SEARCH), pitch * (1 + PITCH_SEARCH), step),
        np.arange(start - reach, start + reach, 1.0),
    )
    span = max(1.0, 4.0 * step)
    o, p, _ = best_over(
        _blurred(profile, 1),
        np.arange(p * 0.99, p * 1.01, max(0.05, p * 0.002)),
        np.arange(o - span, o + span + 0.01, 0.25),
    )
    return o - p * round((o - start) / p), p


def _on_grid(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`np.interp(positions, arange(len(values)), values, left=0, right=0)`, computed as it
    computes it -- the slope times the distance from the pixel below, plus that pixel's value;
    the last pixel's own value at it; nothing off either end -- without its search for each
    position's pixel, which a grid of whole pixels does not need: the same numbers, a third
    of the time (round 3: the comb's interpolation was 11 ms of a 4K re-fit's 36)."""
    last = len(values) - 1
    below = np.floor(positions).astype(np.intp)
    np.clip(below, 0, last - 1, out=below)
    out = (values[below + 1] - values[below]) * (positions - below) + values[below]
    out[positions == last] = values[last]
    out[(positions < 0) | (positions > last)] = 0.0
    return out


def _blurred(profile: np.ndarray, half: int) -> np.ndarray:
    """The profile under a triangle `half` pixels to either side, its peaks where they were."""
    kernel = np.bartlett(2 * half + 3)[1:-1]
    return np.convolve(profile, kernel / kernel.sum(), mode="same")


def cell_bounds(region: Region, cell: float, r: int, c: int) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of cell (r, c) of a board; r and c may lie outside 0..7."""
    x0 = int(round(region.x + c * cell))
    y0 = int(round(region.y + r * cell))
    x1 = int(round(region.x + (c + 1) * cell))
    y1 = int(round(region.y + (r + 1) * cell))
    return x0, y0, x1, y1


def _ring_band(cell: float) -> int:
    return max(1, int(round(cell * 0.12)))


def _ring_indices(
    shape: tuple[int, ...], region: Region, cell: float, pad: int
) -> list[tuple[list[int], np.ndarray]]:
    """The flat pixel indices of the border ring of every cell from row and column -pad to
    7 + pad, grouped by ring length so each group's medians come from one call: (the cells
    of the group, row by row, and an (n, length) index array). A cell the frame cuts keeps
    the part of its ring the frame shows, when that is half the cell or more both ways, and
    is in no group when it is less. The geometry holds for as long as a lock does, so the
    result is cached on it."""
    height, width = shape[:2]
    band = _ring_band(cell)
    side = 8 + 2 * pad
    groups: dict[int, tuple[list[int], list[np.ndarray]]] = {}
    for r in range(side):
        for c in range(side):
            x0, y0, x1, y1 = cell_bounds(region, cell, r - pad, c - pad)
            cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
            if 2 * (cx1 - cx0) < x1 - x0 or 2 * (cy1 - cy0) < y1 - y0:
                continue
            # The ring's rows of the band whole, and the other rows' band columns: the same
            # pixels as a mask over the whole cell, in another order a median does not see,
            # without the cell's every pixel twice over (a 4K re-fit's new geometry built
            # its rings in 73 ms that way: round 3).
            ys = np.arange(cy0, cy1, dtype=_RING_INDEX)
            xs = np.arange(cx0, cx1, dtype=_RING_INDEX)
            band_rows = (ys < y0 + band) | (ys >= y1 - band)
            band_cols = (xs < x0 + band) | (xs >= x1 - band)
            flat = np.concatenate(
                [
                    (ys[band_rows, None] * width + xs[None, :]).ravel(),
                    (ys[~band_rows, None] * width + xs[None, band_cols]).ravel(),
                ]
            )
            cells, rows = groups.setdefault(flat.size, ([], []))
            cells.append(r * side + c)
            rows.append(flat)
    return [(cells, np.stack(rows)) for cells, rows in groups.values()]


# The ring indices' type. numpy indexes with intp and casts int32 indices to it on every
# read: a lock's two reads a tick took 1 ms longer at 1440p and 2 ms at 4K (round 3).
_RING_INDEX = np.intp
# The rings by geometry, the least recently read dropped past RING_CACHE_KEYS: latency
# first, memory second (Tai, 2026-10-02). Over the geometry sweep's seeds 1-20, 64 keeps
# every geometry a search reads again (8 kept 187 of 274) and all but one of 5009 a lock
# does (8: 99.3%); a still lock reads four in all. At most 64 geometries: 445 MB at 1440p
# and 991 MB at 4K once a board has been dragged larger for 30 frames, a few MB a geometry
# otherwise. A 24 MB budget held three of a 4K board's, and every re-fit built the lock's
# own again, 35-43 ms each (round 3, finding 10).
RING_CACHE_KEYS = 64
_RING_CACHE: OrderedDict[
    tuple[tuple[int, ...], Region, float, int], list[tuple[list[int], np.ndarray]]
] = OrderedDict()


# The medians read from the last grey image, by geometry, with the most rings read there: a
# tick reads one geometry several times -- the lock's fit, its readable cells, its square
# colours, a re-fit's cells and its phases -- and each read was the whole median pass again,
# 6 ms at 4K (round 3). A geometry read with more rings around it holds the fewer. Kept for one
# image (by a weak reference, so an image let go is not held); `forget_medians` when an image
# is written over in place.
_Memo = tuple["weakref.ref[np.ndarray]", dict[tuple[Region, float], tuple[int, np.ndarray]]]
_MEDIANS: _Memo | None = None


def forget_medians(gray: np.ndarray) -> None:
    """The medians read from `gray` are of pixels no longer there: it was written over."""
    global _MEDIANS
    memo = _MEDIANS
    if memo is not None and memo[0]() is gray:
        _MEDIANS = None


def _cells(gray: np.ndarray, region: Region, cell: float, pad: int = 1) -> np.ndarray:
    """The border-ring medians of a board's cells and of `pad` rings of cells around it, an
    (8 + 2 pad)-square grid; NaN where the frame shows less than half a cell."""
    global _MEDIANS
    memo = _MEDIANS
    if memo is None or memo[0]() is not gray:
        memo = _MEDIANS = (weakref.ref(gray), {})
    held = memo[1].get((region, cell))
    if held is not None and held[0] >= pad:
        skip = held[0] - pad
        side = 8 + 2 * pad
        return held[1][skip : skip + side, skip : skip + side].copy()
    medians = _read_cells(gray, region, cell, pad)
    memo[1][(region, cell)] = (pad, medians)
    return medians.copy()


def _read_cells(gray: np.ndarray, region: Region, cell: float, pad: int) -> np.ndarray:
    key = (gray.shape, region, cell, pad)
    groups = _RING_CACHE.get(key)
    if groups is None:
        groups = _RING_CACHE[key] = _ring_indices(gray.shape, region, cell, pad)
        if len(_RING_CACHE) > RING_CACHE_KEYS:
            _RING_CACHE.popitem(last=False)
    else:
        _RING_CACHE.move_to_end(key)
    side = 8 + 2 * pad
    medians = np.full(side * side, np.nan)
    # Read from the image's bytes when it has them (`GrayBuffer`): the same whole levels and
    # so the same medians, from a quarter of the memory -- the float read was 45% slower.
    twin = _bytes_of(gray)
    flat = (gray if twin is None else twin).reshape(-1)
    for cells, indices in groups:
        medians[cells] = np.median(flat[indices], axis=1)
    return medians.reshape(side, side)


# The bytes of the grey images `GrayBuffer` keeps, by the image's id (checked by a weak
# reference): the image's own whole levels as uint8.
_BYTES: dict[int, tuple[weakref.ref[np.ndarray], np.ndarray]] = {}


def _bytes_of(gray: np.ndarray) -> np.ndarray | None:
    """The uint8 twin of a grey image `GrayBuffer` keeps, or None."""
    twin = _BYTES.get(id(gray))
    return twin[1] if twin is not None and twin[0]() is gray else None


# A board is read only when the frame shows it, to within this many pixels a side: a board
# flush with the frame's edge is fitted a pixel past it now and then. Of a board further off,
# the cells the frame cuts would reach the pieces' reader squeezed, so it is not locked; it is
# found when it is back (a partly shown board was locked a cell inward instead, on the window
# of part board and part background: lock_bench, 2026-09-27, 7 boards of 17).
EDGE_SLACK_PX = 2


def _on_frame(shape: tuple[int, ...], region: Region) -> bool:
    height, width = shape[:2]
    return (
        region.x >= -EDGE_SLACK_PX
        and region.y >= -EDGE_SLACK_PX
        and region.x + region.w <= width + EDGE_SLACK_PX
        and region.y + region.h <= height + EDGE_SLACK_PX
    )


def palette(medians: np.ndarray) -> Colours | None:
    """The square colours read from a grid of cell medians, with their tolerance; None when
    the frame shows fewer than MIN_READABLE_CELLS of the cells.

    The two colours are the medians of the cells the checker pattern expects to be light and
    dark, so which one is the brighter depends on the theme. A cell is readable when its
    median lies within the tolerance of either colour, and the tolerance follows how much the
    cells of each colour vary, so that a textured wood theme keeps its cells while a dialog,
    a tooltip, or a highlight drawn over a cell is left out of the fit. That is what lets a
    partly covered board keep its lock.
    """
    shown = np.isfinite(medians)
    if int(shown.sum()) < MIN_READABLE_CELLS:
        return None
    lights = medians[(_SIGNS > 0) & shown]
    darks = medians[(_SIGNS < 0) & shown]
    light = float(np.median(lights))
    dark = float(np.median(darks))
    spread = max(float(np.median(np.abs(lights - light))), float(np.median(np.abs(darks - dark))))
    return light, dark, min(max(6.0, 3.0 * spread), 0.35 * abs(light - dark))


def readable_against(medians: np.ndarray, colours: Colours) -> np.ndarray:
    """Cells that read as either square colour; a cell the frame does not show reads as
    neither."""
    light, dark, tolerance = colours
    return (np.abs(medians - light) <= tolerance) | (np.abs(medians - dark) <= tolerance)


def expected_signs(colours: Colours) -> np.ndarray:
    """The sign of (median - mid-grey) the checker pattern expects in each cell.

    Even cells carry the light colour, odd cells the dark one; which is brighter is the
    theme's business, so the pattern is inverted when the light squares are the darker ones.
    """
    light, dark, _ = colours
    return _SIGNS if light > dark else -_SIGNS


def _continues(cells: np.ndarray, colours: Colours) -> bool:
    """Whether the checker pattern carries on in the ring of cells around a board (`cells`,
    the board and its ring as `_cells` reads them) on any of its sides.

    A candidate shifted by whole cells, or a window of a larger checkerboard, reads a
    perfect board inside its 64 cells; only the cells just outside tell. A side continues
    the board when most of its eight outer cells read as square colours and alternate. A
    cell the frame does not show does not read, so a board flush with the frame's edge
    passes; one the frame cuts is read on what it shows, when that is half the cell or more:
    a candidate two thirds of a cell above a board that ended two pixels above the frame's
    bottom had its row of outer cells above it off the frame and the row below it a third
    off, so neither side was read, the board's last row went unseen below it, and the lock
    validated with every cell reading across two squares (the geometry sweep, 2026-09-27,
    seed 260).
    """
    light, dark, _ = colours
    readable = readable_against(cells, colours)
    bright = cells > 0.5 * (light + dark)
    for side in (np.s_[0, 1:9], np.s_[9, 1:9], np.s_[1:9, 0], np.s_[1:9, 9]):
        read, up = readable[side], bright[side]
        if read.sum() >= 6 and (read[1:] & read[:-1] & (up[1:] != up[:-1])).sum() >= 5:
            return True
    return False


class _Reading(NamedTuple):
    """How a board reads (`_judge`)."""

    fit: float
    contrast: float
    readable: int
    evidence: int
    colours: Colours | None  # the palette it was read with


def _judge(cells: np.ndarray, colours: Colours | None) -> _Reading:
    """How a board reads against `colours`, or in its own (`palette`) without them; `cells`
    is the board and its ring, as `_cells` reads them.

    The fit is the fraction of the readable cells that match the checker pattern: zero when
    fewer than MIN_READABLE_CELLS read, or when the pattern carries on outside the board. The
    evidence weighs placements a cell apart against each other: the matching cells less the
    cells the frame shows that do not match; a cell the frame does not show counts neither
    way. Read in its own colours, a board must also match on most of the cells the frame
    shows: a window of part board and part background reads a palette of blends from itself,
    and matched on the few cells that happened to fall within it (26 of 64, lock_bench case
    106, 2026-09-27). A lock's colours are exempt, so a dialog over most of the board does
    not cost it.
    """
    board = cells[1:9, 1:9]
    read = colours or palette(board)
    shown = int(np.isfinite(board).sum())
    if read is None:
        return _Reading(0.0, 0.0, 0, -shown, None)
    light, dark, _ = read
    # The threshold sits between the two square colours rather than at the board mean, which
    # a dark dialog over a light board would drag below the dark squares.
    mid = 0.5 * (light + dark)
    readable = readable_against(board, read)
    count = int(readable.sum())
    matched = int((readable & (np.sign(board - mid) == expected_signs(read))).sum())
    evidence = 2 * matched - shown
    if count < MIN_READABLE_CELLS:
        return _Reading(0.0, 0.0, count, evidence, read)
    contrast = float(np.abs(board[readable] - mid).mean())
    fit = matched / count
    if (colours is None and evidence <= 0) or (fit >= 0.5 and _continues(cells, read)):
        fit = 0.0
    return _Reading(fit, contrast, count, evidence, read)


def readable_cells(gray: np.ndarray, region: Region, cell: float, colours: Colours) -> int:
    """How many of the 64 cells at `region` read as one of `colours`' square colours.

    The fit counts only these, so a board that moved away by two cells and a fraction still
    fits where it was, on the cells it left behind; what gives it away is how few are left
    to read (the tracker's probe)."""
    if not _on_frame(gray.shape, region):
        return 0
    return int(readable_against(_cells(gray, region, cell, pad=0), colours).sum())


def fit_score(
    gray: np.ndarray,
    region: Region,
    cell: float,
    colours: Colours | None = None,
) -> tuple[float, float]:
    """(fraction of readable cells matching the checker pattern, contrast).

    With `colours` (the square colours a lock was taken with) cells are read against them,
    so a dialog over most of the board cannot rewrite the palette, and the pattern must keep
    the lock's polarity, so a board that slid by a whole cell reads every cell wrong. Without
    them the palette is read from the candidate itself, in whichever polarity the theme has,
    and the cells just outside the candidate decide whether it is shifted. A board the frame
    does not show whole scores zero.
    """
    if not _on_frame(gray.shape, region):
        return 0.0, 0.0
    reading = _judge(_cells(gray, region, cell), colours)
    return reading.fit, reading.contrast


def square_colours(gray: np.ndarray, region: Region, cell: float) -> Colours | None:
    """The square colours of a board as read now, with their tolerance, for a lock to keep;
    None for a board the frame does not show whole."""
    if not _on_frame(gray.shape, region):
        return None
    return palette(_cells(gray, region, cell, pad=0))


def occupied_grid(
    gray: np.ndarray, region: Region, cell: float, medians: np.ndarray, colours: Colours
) -> np.ndarray:
    """Which cells, as an 8x8 bool grid, are readable and have an interior mean that differs
    from their ring median by over the tolerance: the cells pieces and marks sit on."""
    band = _ring_band(cell)
    readable = readable_against(medians, colours)
    grid = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if not readable[r, c]:
                continue
            x0, y0, x1, y1 = cell_bounds(region, cell, r, c)
            inner = gray[y0 + band : y1 - band, x0 + band : x1 - band]
            if inner.size and abs(float(inner.mean()) - medians[r, c]) > colours[2]:
                grid[r, c] = True
    return grid


def occupied_cells(
    gray: np.ndarray, region: Region, cell: float, medians: np.ndarray, colours: Colours
) -> int:
    """Readable cells whose interior mean differs from their ring median by over the tolerance.

    Pieces and marks sit inside their squares, so on a game board this is well above zero
    while an empty board or a diagram reads near zero. Only readable cells count, so the
    background in a window that is part board does not pass for pieces. It only breaks
    ties: among boards of one size and fit the detector puts a game board before an empty
    one.
    """
    return int(occupied_grid(gray, region, cell, medians, colours).sum())


def board_occupancy(gray: np.ndarray, region: Region, cell: float) -> np.ndarray | None:
    """The occupied cells of the board at `region`, read in its own colours; None when the
    frame does not show it whole. Two boards of one size tell apart by it: a board moved
    keeps its pieces, another game's board does not."""
    colours = square_colours(gray, region, cell)
    if colours is None:
        return None
    return occupied_grid(gray, region, cell, _cells(gray, region, cell, pad=0), colours)


def _boundary_weight(
    crop8: np.ndarray,
    origin: tuple[int, int],
    board: BoardCandidate,
    medians: np.ndarray,
    colours: Colours | None,
    band: int,
) -> np.ndarray:
    """Which pixels of `crop8` (whose top-left is `origin` in the frame, in whole grey
    levels) take part in the boundary fit around `board`, as a mask of 0 or 255.

    Only the strips around the cell boundaries take part, and in them only the pixels of
    the square's own colour. Pieces sit in the middle of their squares and their outlines
    out-gradient faint boundaries on a low-contrast theme; a dialog over part of the board
    has straight edges and text that would pull the grid towards them. The mask repeats
    with the cell pitch, so a board that slid by a whole cell still has its boundaries in
    unmasked strips, and everything outside the board, such as its outer edge, stays. A real
    set's pieces reach into the strips (their bases sit at 0.87 of the cell), and a rank of
    them draws one line as strong as a boundary a tenth of a cell above it: at 144-pixel
    cells the comb settled between the two, a pitch two percent long and the top edge a
    quarter of a cell off. So within the board a pixel that is not its cell's own colour, to
    the palette's tolerance, takes no part: a piece's fill and outline are gone, so are the
    labels and a dialog. The blends at a boundary, between the two square colours, lie on
    the grid lines (within `band` of them) and are kept there: without them the comb drifted
    a third of a percent. A pixel of a piece's edge is not: eight bases in a rank, each one
    pixel of square kept beside black, summed to three boundaries' worth and pulled the comb
    a tenth of a cell.

    A cell that does not read as a square colour, under a dialog or a highlight, keeps only a
    thin strip along its boundaries. When too few cells read to tell the squares' own colour
    at all, the geometry is too far off to lay the mask on (a lock a quarter of a cell off on
    a textured theme reads most cells across two squares): its strips hid the boundaries
    themselves and kept the pieces' edges, and the comb settled nine percent long (lock_bench,
    2026-09-27). Then nothing is left out. The boundaries run the width of the board, the
    pieces' edges do not, and that pass finds the boundaries near enough for the next to lay
    the mask.
    """
    # Whole grey levels against the colours' bounds: a level is within a tolerance of a
    # colour when it lies between the bounds rounded inward, which OpenCV's inRange reads in
    # one pass a cell (round 3: the float comparisons, a temporary image each, were 17 ms of a
    # 4K re-fit).
    x_lo, y_lo = origin
    region, cell = board.region, board.cell_px
    height, width = crop8.shape
    mask = np.full(crop8.shape, 255, dtype=np.uint8)
    if colours is None:
        return mask
    readable = readable_against(medians, colours)
    if readable.sum() < MIN_READABLE_CELLS:
        return mask
    light, dark, tolerance = colours
    x0, y0 = max(0, region.x - x_lo), max(0, region.y - y_lo)
    x1 = min(width, region.x - x_lo + int(round(8 * cell)))
    y1 = min(height, region.y - y_lo + int(round(8 * cell)))
    if x1 > x0 and y1 > y0:
        board8 = crop8[y0:y1, x0:x1]
        own = np.zeros(board8.shape, dtype=np.uint8)
        for r in range(8):
            for c in range(8):
                median = float(medians[r, c])
                if np.isnan(median):
                    continue
                cx0, cy0, cx1, cy1 = cell_bounds(region, cell, r, c)
                cx0, cy0 = max(0, cx0 - x_lo) - x0, max(0, cy0 - y_lo) - y0
                cx1 = min(width, cx1 - x_lo) - x0
                cy1 = min(height, cy1 - y_lo) - y0
                cx0, cy0, cx1, cy1 = max(0, cx0), max(0, cy0), min(x1 - x0, cx1), min(y1 - y0, cy1)
                if cx1 > cx0 and cy1 > cy0:
                    own[cy0:cy1, cx0:cx1] = _between(
                        board8[cy0:cy1, cx0:cx1], median - tolerance, median + tolerance
                    )
        lines = np.zeros(board8.shape, dtype=np.uint8)
        for k in range(1, 8):
            gx = int(round(region.x - x_lo + k * cell)) - x0
            gy = int(round(region.y - y_lo + k * cell)) - y0
            lines[:, max(0, gx - band) : max(0, gx + band + 1)] = 255
            lines[max(0, gy - band) : max(0, gy + band + 1), :] = 255
        in_range = _between(board8, min(light, dark) - tolerance, max(light, dark) + tolerance)
        mask[y0:y1, x0:x1] = cv2.bitwise_or(own, cv2.bitwise_and(in_range, lines))
    for r in range(8):
        for c in range(8):
            inset = int(round(cell * (0.2 if readable[r, c] else 0.08)))
            ix0, iy0, ix1, iy1 = cell_bounds(region, cell, r, c)
            ix0, iy0 = ix0 - x_lo + inset, iy0 - y_lo + inset
            ix1, iy1 = ix1 - x_lo - inset, iy1 - y_lo - inset
            if ix1 > ix0 and iy1 > iy0:
                mask[max(0, iy0) : iy1, max(0, ix0) : ix1] = 0
    return mask


def _between(levels: np.ndarray, low: float, high: float) -> np.ndarray:
    """255 where a whole grey level lies within [low, high], else 0."""
    lo, hi = int(np.ceil(low)), int(np.floor(high))
    if hi < lo or hi < 0 or lo > 255:
        return np.zeros(levels.shape, dtype=np.uint8)
    return cv2.inRange(levels, max(0, lo), min(255, hi))


# The placements a cell apart that `refine` weighs, the comb's own first so a tie keeps it.
_PHASES: list[tuple[int, int]] = sorted(
    ((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)), key=lambda d: abs(d[0]) + abs(d[1])
)


def refine(
    gray: np.ndarray,
    cand: BoardCandidate,
    *,
    offset_cells: float = 0.6,
    colours: Colours | None = None,
) -> Refinement:
    """Sub-pixel boundaries from gradient peaks, the whole-cell phase from the cells, then
    validation.

    `offset_cells` is how far, in cells, the boundaries may move from the candidate. The
    detector needs less than a cell, since a coarse candidate is within a few pixels; the
    tracker's periodic re-fit uses a wider window and exists for what validation cannot see,
    sub-cell drift and a small resize (a whole-cell slide fails validation outright and goes
    through re-detection). With `colours` the cells are read against a lock's palette.

    The comb places the board to within a whole cell (`_comb_fit`). Of the placements a cell
    apart on each axis, the one kept validates and has the most evidence (`_judge`): a board
    partly off the frame and the window a cell inward, part board and part background, match
    as many cells, and the window's row of background is what tells them apart. The mask the
    boundaries are fitted through is laid out on a geometry, and a candidate a cell pitch or
    two percent off lays it a fraction of a cell off at the far side, where it then masks the
    boundaries themselves and keeps its own error: the comb settled two percent long there
    once. So the fit runs again, up to twice, while a pass moves it, each time with the mask
    laid out on the last pass's geometry.
    """
    height, width = gray.shape
    cell = cand.cell_px
    margin = cell * (offset_cells + 0.1)
    x_lo = int(max(0, np.floor(cand.region.x - margin)))
    y_lo = int(max(0, np.floor(cand.region.y - margin)))
    x_hi = int(min(width, np.ceil(cand.region.x + 8 * cell + margin)))
    y_hi = int(min(height, np.ceil(cand.region.y + 8 * cell + margin)))
    crop = gray[y_lo:y_hi, x_lo:x_hi]
    if crop.size == 0:
        return Refinement(cand, 0.0, 0.0)
    # The grey image holds whole levels (`to_gray`): the boundary fit runs on them as bytes.
    twin = _bytes_of(gray)
    crop8 = crop.astype(np.uint8) if twin is None else twin[y_lo:y_hi, x_lo:x_hi]
    fitted = cand
    origin = (float(cand.region.x), float(cand.region.y))
    profile_x = profile_y = np.zeros(0)
    for attempt in range(3):
        cell = fitted.cell_px
        medians = _cells(gray, fitted.region, cell, pad=0)
        # The first pass does not know where the lines are to within the search window
        # (the tracker's re-fit hands in a lock a third of a cell off), so its bands span
        # the window; the passes after it know to a few pixels.
        band = 3 if attempt else int(round(cell * offset_cells)) + 3
        read = colours or palette(medians)
        mask = _boundary_weight(crop8, (x_lo, y_lo), fitted, medians, read, band)
        profile_x = _gradient_profile(crop8, 1, mask)
        profile_y = _gradient_profile(crop8, 0, mask)
        ox, px = _comb_fit(profile_x, fitted.region.x - x_lo, cell, offset_cells)
        oy, py = _comb_fit(profile_y, fitted.region.y - y_lo, cell, offset_cells)
        pitch = (px + py) / 2.0
        origin = (x_lo + ox, y_lo + oy)
        side = int(round(8 * pitch))
        region = Region(int(round(origin[0])), int(round(origin[1])), side, side)
        moved = (
            abs(pitch - cell) > 0.005 * cell
            or abs(region.x - fitted.region.x) > 2
            or abs(region.y - fitted.region.y) > 2
        )
        fitted = replace(fitted, region=region, cell_px=pitch)
        if not moved:
            break
    pitch, side = fitted.cell_px, fitted.region.w
    grid = _cells(gray, fitted.region, pitch, pad=2)

    def weighed(phase: tuple[int, int]) -> tuple[bool, int, float]:
        dx, dy = phase
        reading = _judge(grid[1 + dy : 11 + dy, 1 + dx : 11 + dx], colours)
        return reading.fit > 0.0, reading.evidence, reading.fit

    dx, dy = max(_PHASES, key=weighed)
    x, y = origin[0] + dx * pitch, origin[1] + dy * pitch
    region = Region(int(round(x)), int(round(y)), side, side)
    fit = contrast = 0.0
    occupied = readable = 0
    if _on_frame(gray.shape, region):
        cells = _cells(gray, region, pitch)
        reading = _judge(cells, colours)
        fit, contrast, readable = reading.fit, reading.contrast, reading.readable
        if reading.colours is not None:
            occupied = occupied_cells(gray, region, pitch, cells[1:9, 1:9], reading.colours)
    # The comb's strength at the board kept and at the geometry the fit started from, over
    # the same profiles, for the tracker to weigh a re-fit against its lock.
    strength = _comb_strength(profile_x, region.x - x_lo, pitch) + _comb_strength(
        profile_y, region.y - y_lo, pitch
    )
    reference = _comb_strength(profile_x, cand.region.x - x_lo, cand.cell_px) + _comb_strength(
        profile_y, cand.region.y - y_lo, cand.cell_px
    )
    board = replace(
        cand,
        region=region,
        cell_px=pitch,
        fit=fit,
        contrast=contrast,
        occupied=occupied,
        readable=readable,
    )
    return Refinement(board, strength, reference)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


class GrayBuffer:
    """`to_gray` into images kept from one frame to the next: the same grey values, without
    allocating a whole frame twice a tick -- at 4K a new image cost 6 ms, written over one
    kept 2 (round 3). Each frame overwrites the last, so the medians read from it are forgotten
    (`forget_medians`), and what must outlive the tick is copied out of it."""

    def __init__(self) -> None:
        self._grey: np.ndarray | None = None
        self._image: np.ndarray | None = None

    @property
    def image(self) -> np.ndarray | None:
        """The last frame converted, or None."""
        return self._image

    @property
    def bytes(self) -> np.ndarray | None:
        """The last frame converted, as its uint8 grey levels, or None."""
        return self._grey

    def convert(self, frame_bgr: np.ndarray) -> np.ndarray:
        shape = frame_bgr.shape[:2]
        if self._image is None or self._image.shape != shape or self._grey is None:
            self._grey = np.empty(shape, dtype=np.uint8)
            self._image = np.empty(shape, dtype=np.float32)
            for key, (image, _bytes) in list(_BYTES.items()):
                if image() is None:
                    del _BYTES[key]
            _BYTES[id(self._image)] = (weakref.ref(self._image), self._grey)
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY, dst=self._grey)
        np.copyto(self._image, self._grey)
        forget_medians(self._image)
        return self._image


def detect_boards(
    frame_bgr: np.ndarray,
    *,
    top_k: int = 3,
    min_fit: float = 0.85,
    min_contrast: float = 6.0,
    min_cell: int = MIN_CELL_PX,
    max_cell: int | None = None,
) -> list[BoardCandidate]:
    """Boards in a frame, largest first, each validated against the checker pattern."""
    return detect_boards_gray(
        to_gray(frame_bgr),
        top_k=top_k,
        min_fit=min_fit,
        min_contrast=min_contrast,
        min_cell=min_cell,
        max_cell=max_cell,
    )


def detect_boards_gray(
    gray: np.ndarray,
    *,
    top_k: int = 3,
    min_fit: float = 0.85,
    min_contrast: float = 6.0,
    min_cell: int = MIN_CELL_PX,
    max_cell: int | None = None,
) -> list[BoardCandidate]:
    """Boards in a grey frame: largest first, then by fit, then the one with more pieces."""
    # The coarse amplitude is shaved by the work-scale misalignment of up to half a pixel,
    # so its gate sits below the one the refined ring contrast must pass.
    coarse = coarse_search(
        gray, min_cell=min_cell, max_cell=max_cell, min_contrast=0.6 * min_contrast
    )
    refined = [refine(gray, cand).board for cand in coarse[: top_k * 4]]
    good = [cand for cand in refined if cand.fit >= min_fit and cand.contrast >= min_contrast]
    good = _nms(good, "fit")
    # Size first; among boards of about the size of the largest (SAME_SIZE), the one that
    # reads more of its cells as squares -- a window 4 px wider, part board and part
    # background (24 cells readable against the board's 64), outranked a board moved two
    # cells away and the game was never read again (the parity audit's sweep, 2026-09-26);
    # then fit, and the occupied count between boards of one size and fit.
    widest = max((c.region.w for c in good), default=0)

    def rank(c: BoardCandidate) -> tuple[bool, int, int, float, int]:
        peer = c.region.w >= SAME_SIZE * widest
        return peer, c.readable if peer else 0, c.region.w, c.fit, c.occupied

    return sorted(good, key=rank, reverse=True)[:top_k]


def draw(frame_bgr: np.ndarray, candidates: list[BoardCandidate]) -> np.ndarray:
    """Annotated copy of the frame for debugging."""
    out = frame_bgr.copy()
    for i, cand in enumerate(candidates):
        r = cand.region
        colour = (0, 255, 0) if i == 0 else (0, 200, 255)
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), colour, 2)
        for k in range(1, 8):
            offset = int(round(k * cand.cell_px))
            cv2.line(out, (r.x + offset, r.y), (r.x + offset, r.y + r.h), colour, 1)
            cv2.line(out, (r.x, r.y + offset), (r.x + r.w, r.y + offset), colour, 1)
        label = f"cell {cand.cell_px:.1f}px fit {cand.fit:.2f} score {cand.score:.2f}"
        cv2.putText(out, label, (r.x, max(15, r.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out
