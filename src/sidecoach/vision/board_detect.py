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
   at the nine cell boundaries. A nine-tooth comb is fitted for sub-pixel offset and pitch.
3. Validation. Each cell's colour is read as the median of a thin border ring, which pieces
   and highlights rarely disturb. The two square colours are the medians of the even and the
   odd cells, or the colours a lock was taken with, and the fit score is the fraction of the
   readable cells that match the checker pattern in the polarity of those colours. A
   candidate whose pattern carries on in the ring of cells just outside it, a whole-cell
   shift or a window into a larger checkerboard, is rejected.

Pieces, the cursor, arrows, highlights and coordinate labels are local outliers against 64
cells of evidence, so they lower the scores a little and do not move the result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from sidecoach.events import Region

_SIGNS = np.array(
    [[1.0 if (r + c) % 2 == 0 else -1.0 for c in range(8)] for r in range(8)], dtype=np.float64
)
MIN_READABLE_CELLS = 24

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


@dataclass(frozen=True)
class Refinement:
    """A candidate after the boundary fit, with the comb strength at its new and old geometry.

    `strength` is the summed boundary gradient at the nine teeth of the fitted comb on both
    axes; `reference` is the same sum at the geometry the fit started from, over the same
    profiles, so the two compare: a re-fit that found nothing better than where it started
    scores about the same, one that corrected a drift or a resize scores clearly more.
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


def _checker_response(gray: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """(score, amplitude) maps for a board whose top-left cell starts at each position.

    The score is |sum(sign * mean)| / sum|mean - board mean|, 1 for a perfect empty board.
    The amplitude |sum(sign * mean)| / 64 is half the gap between the two square colours
    for a board and is only lowered by cells that are not board, unlike the deviation,
    which background cells inflate; it is the contrast gate.
    """
    means = _ring_means(gray, cell)
    h = means.shape[0] - 7 * cell
    w = means.shape[1] - 7 * cell
    if h <= 0 or w <= 0:
        empty = np.zeros((0, 0))
        return empty, empty
    blocks = [
        (_SIGNS[r, c], means[r * cell : r * cell + h, c * cell : c * cell + w])
        for r in range(8)
        for c in range(8)
    ]
    acc = np.zeros((h, w), dtype=np.float64)
    total = np.zeros((h, w), dtype=np.float64)
    for sign, block in blocks:
        acc += sign * block
        total += block
    board_mean = total / 64.0
    deviation = np.zeros((h, w), dtype=np.float64)
    for _, block in blocks:
        deviation += np.abs(block - board_mean)
    score = np.abs(acc) / (deviation + 1e-6)
    return score, np.abs(acc) / 64.0


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
    min_cell: int = 24,
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
    """
    height, width = gray.shape
    limit = max_cell if max_cell is not None else min(height, width) // 8
    found: list[BoardCandidate] = []
    size = float(min_cell)
    while size <= limit:
        factor = min(1.0, work_cell / size)
        if factor < 1.0:
            small = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            cell = work_cell
        else:
            small = gray
            cell = int(round(size))
        score, contrast = _checker_response(small, cell)
        if score.size:
            mask = (score >= min_score) & (contrast >= min_contrast)
            full_cell = cell / factor
            side = int(round(8 * full_cell))
            for y, x in _local_maxima(score, mask, cell, radius=4 * cell, limit=per_scale):
                region = Region(int(round(x / factor)), int(round(y / factor)), side, side)
                found.append(
                    BoardCandidate(region, full_cell, float(score[y, x]), float(contrast[y, x]))
                )
        size *= scale_step
    return _nms(found, "score")


def _gradient_profile(gray: np.ndarray, axis: int, weight: np.ndarray | None = None) -> np.ndarray:
    """Summed absolute gradient across the other axis. axis=1 gives a profile along x.

    `weight` (same shape as `gray`) zeroes out pixels that must not take part, such as cells
    covered by a dialog whose straight edges would otherwise attract the grid fit.
    """
    if axis == 1:
        grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    else:
        grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    if weight is not None:
        grad = grad * weight
    return grad.sum(axis=0) if axis == 1 else grad.sum(axis=1)


_TEETH = np.arange(9, dtype=np.float64)


def _comb_strength(profile: np.ndarray, offset: float, pitch: float) -> float:
    """Summed profile at the nine teeth of a comb; teeth past the profile's end read zero."""
    positions = offset + pitch * _TEETH
    inside = (positions >= 0) & (positions <= len(profile) - 1)
    xs = np.arange(len(profile), dtype=np.float64)
    return float(np.interp(positions[inside], xs, profile).sum())


def _comb_fit(
    profile: np.ndarray, start: float, pitch: float, search: float, offset_cells: float = 0.6
) -> tuple[float, float, float]:
    """Best (offset, pitch, strength) of a nine-tooth comb over a gradient profile.

    Coarse to fine: whole pixels and one-percent pitch steps over the whole window first,
    then quarter pixels and 0.2 percent steps around the best, which is twenty times
    cheaper than a fine grid everywhere and lands on the same optimum.
    """
    n = len(profile)
    xs = np.arange(n, dtype=np.float64)

    def best_over(pitches: np.ndarray, offsets: np.ndarray) -> tuple[float, float, float]:
        best = (start, pitch, -np.inf)
        if offsets.size == 0 or pitches.size == 0:
            return best
        for p in pitches:
            positions = offsets[:, None] + p * _TEETH[None, :]
            valid = positions[:, -1] <= n - 1
            if not valid.any():
                continue
            strength = np.interp(positions, xs, profile).sum(axis=1)
            strength[~valid] = -np.inf
            i = int(np.argmax(strength))
            if strength[i] > best[2]:
                best = (float(offsets[i]), float(p), float(strength[i]))
        return best

    reach = pitch * offset_cells
    coarse = best_over(
        np.arange(pitch * (1 - search), pitch * (1 + search), max(0.25, pitch * 0.01)),
        np.arange(max(0.0, start - reach), min(n - 1.0, start + reach), 1.0),
    )
    if coarse[2] == -np.inf:
        return start, pitch, 0.0
    o, p, _ = coarse
    fine = best_over(
        np.arange(p * 0.99, p * 1.01, max(0.05, p * 0.002)),
        np.arange(max(0.0, o - 1.0), min(n - 1.0, o + 1.0) + 0.01, 0.25),
    )
    return fine if fine[2] >= coarse[2] else coarse


def cell_bounds(region: Region, cell: float, r: int, c: int) -> tuple[int, int, int, int]:
    """Pixel box (x0, y0, x1, y1) of cell (r, c) of a board; r and c may lie outside 0..7."""
    x0 = int(round(region.x + c * cell))
    y0 = int(round(region.y + r * cell))
    x1 = int(round(region.x + (c + 1) * cell))
    y1 = int(round(region.y + (r + 1) * cell))
    return x0, y0, x1, y1


def _ring_band(cell: float) -> int:
    return max(1, int(round(cell * 0.12)))


def _ring_median(gray: np.ndarray, box: tuple[int, int, int, int], band: int) -> float | None:
    """Median of a cell's border ring, or None when the cell is not wholly in the frame."""
    x0, y0, x1, y1 = box
    height, width = gray.shape
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    patch = gray[y0:y1, x0:x1]
    if patch.shape[0] <= 2 * band or patch.shape[1] <= 2 * band:
        return None
    ring = np.concatenate(
        [
            patch[:band].ravel(),
            patch[-band:].ravel(),
            patch[band:-band, :band].ravel(),
            patch[band:-band, -band:].ravel(),
        ]
    )
    return float(np.median(ring))


def cell_medians(gray: np.ndarray, region: Region, cell: float) -> np.ndarray | None:
    """8x8 border-ring medians, or None when the board runs off the frame."""
    medians = np.zeros((8, 8), dtype=np.float64)
    band = _ring_band(cell)
    for r in range(8):
        for c in range(8):
            value = _ring_median(gray, cell_bounds(region, cell, r, c), band)
            if value is None:
                return None
            medians[r, c] = value
    return medians


def palette(medians: np.ndarray) -> Colours:
    """The square colours read from a grid of cell medians, with their tolerance.

    The two colours are the medians of the cells the checker pattern expects to be light and
    dark, so which one is the brighter depends on the theme. A cell is readable when its
    median lies within the tolerance of either colour, and the tolerance follows how much the
    cells of each colour vary, so that a textured wood theme keeps its cells while a dialog,
    a tooltip, or a highlight drawn over a cell is left out of the fit. That is what lets a
    partly covered board keep its lock.
    """
    lights = medians[_SIGNS > 0]
    darks = medians[_SIGNS < 0]
    light = float(np.median(lights))
    dark = float(np.median(darks))
    spread = max(float(np.median(np.abs(lights - light))), float(np.median(np.abs(darks - dark))))
    return light, dark, min(max(6.0, 3.0 * spread), 0.35 * abs(light - dark))


def readable_against(medians: np.ndarray, colours: Colours) -> np.ndarray:
    """Cells that read as either square colour."""
    light, dark, tolerance = colours
    return (np.abs(medians - light) <= tolerance) | (np.abs(medians - dark) <= tolerance)


def expected_signs(colours: Colours) -> np.ndarray:
    """The sign of (median - mid-grey) the checker pattern expects in each cell.

    Even cells carry the light colour, odd cells the dark one; which is brighter is the
    theme's business, so the pattern is inverted when the light squares are the darker ones.
    """
    light, dark, _ = colours
    return _SIGNS if light > dark else -_SIGNS


def outer_continues_board(gray: np.ndarray, region: Region, cell: float, colours: Colours) -> bool:
    """Whether the checker pattern carries on outside the candidate on any of its sides.

    A candidate shifted by whole cells, or a window of a larger checkerboard, reads a
    perfect board inside its 64 cells; only the cells just outside tell. A side continues
    the board when most of its eight outer cells read as square colours and alternate.
    Cells outside the frame are not counted, so a board flush with the screen edge passes.
    """
    light, dark, tolerance = colours
    band = _ring_band(cell)
    sides = (
        [(-1, c) for c in range(8)],
        [(8, c) for c in range(8)],
        [(r, -1) for r in range(8)],
        [(r, 8) for r in range(8)],
    )
    mid = 0.5 * (light + dark)
    for cells in sides:
        signs: list[float | None] = []
        for r, c in cells:
            value = _ring_median(gray, cell_bounds(region, cell, r, c), band)
            if value is None or (abs(value - light) > tolerance and abs(value - dark) > tolerance):
                signs.append(None)
            else:
                signs.append(1.0 if value > mid else -1.0)
        readable = [v for v in signs if v is not None]
        if len(readable) < 6:
            continue
        alternating = sum(
            1
            for a, b in zip(signs, signs[1:], strict=False)
            if a is not None and b is not None and a != b
        )
        if alternating >= 5:
            return True
    return False


def _score_cells(
    gray: np.ndarray, region: Region, cell: float, medians: np.ndarray, colours: Colours
) -> tuple[float, float]:
    """(fit, contrast) of a grid of cell medians read against `colours`."""
    light, dark, _ = colours
    readable = readable_against(medians, colours)
    # The threshold sits between the two square colours rather than at the board mean, which
    # a dark dialog over a light board would drag below the dark squares.
    mid = 0.5 * (light + dark)
    if int(readable.sum()) < MIN_READABLE_CELLS:
        return 0.0, float(np.abs(medians - mid).mean())
    contrast = float(np.abs(medians[readable] - mid).mean())
    expected = expected_signs(colours)
    match = float((np.sign(medians - mid)[readable] == expected[readable]).mean())
    if match >= 0.5 and outer_continues_board(gray, region, cell, colours):
        return 0.0, contrast
    return match, contrast


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
    and the cells just outside the candidate decide whether it is shifted.
    """
    medians = cell_medians(gray, region, cell)
    if medians is None:
        return 0.0, 0.0
    return _score_cells(gray, region, cell, medians, colours or palette(medians))


def square_colours(gray: np.ndarray, region: Region, cell: float) -> Colours | None:
    """The square colours of a board as read now, with their tolerance, for a lock to keep."""
    medians = cell_medians(gray, region, cell)
    return None if medians is None else palette(medians)


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
    band = _ring_band(cell)
    readable = readable_against(medians, colours)
    count = 0
    for r in range(8):
        for c in range(8):
            if not readable[r, c]:
                continue
            x0, y0, x1, y1 = cell_bounds(region, cell, r, c)
            inner = gray[y0 + band : y1 - band, x0 + band : x1 - band]
            if inner.size and abs(float(inner.mean()) - medians[r, c]) > colours[2]:
                count += 1
    return count


def refine(
    gray: np.ndarray,
    cand: BoardCandidate,
    *,
    search: float = 0.12,
    offset_cells: float = 0.6,
    colours: Colours | None = None,
) -> Refinement:
    """Sub-pixel boundaries from gradient peaks, then validation.

    `offset_cells` is how far, in cells, the boundaries may move from the candidate. The
    detector needs less than a cell, since a coarse candidate is within a few pixels; the
    tracker's periodic re-fit uses a wider window and exists for what validation cannot see,
    sub-cell drift and a small resize (a whole-cell slide fails validation outright and goes
    through re-detection). With `colours` the cells are read against a lock's palette.
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

    # Only the strips around the cell boundaries take part in the boundary fit. Pieces sit
    # in the middle of their squares and their outlines out-gradient faint boundaries on
    # a low-contrast theme; a dialog over part of the board has straight edges and text
    # that would pull the grid towards them. The mask repeats with the cell pitch, so a
    # board that slid by a whole cell still has its boundaries in unmasked strips, and
    # everything outside the candidate, such as the board's outer edge, stays.
    weight = np.ones(crop.shape, dtype=np.float32)
    medians = cell_medians(gray, cand.region, cell)
    readable = np.ones((8, 8), dtype=bool)
    if medians is not None:
        readable = readable_against(medians, colours or palette(medians))
        if int(readable.sum()) < MIN_READABLE_CELLS:
            # Too few cells to place the mask; the candidate stays where it is and carries
            # the fit and contrast measured there, which is a fit of zero.
            fit, contrast = fit_score(gray, cand.region, cell, colours)
            return Refinement(replace(cand, fit=fit, contrast=contrast), 0.0, 0.0)
    for r in range(8):
        for c in range(8):
            inset = int(round(cell * (0.2 if readable[r, c] else 0.08)))
            x0, y0, x1, y1 = cell_bounds(cand.region, cell, r, c)
            x0, y0 = x0 - x_lo + inset, y0 - y_lo + inset
            x1, y1 = x1 - x_lo - inset, y1 - y_lo - inset
            if x1 > x0 and y1 > y0:
                weight[max(0, y0) : y1, max(0, x0) : x1] = 0.0

    profile_x = _gradient_profile(crop, axis=1, weight=weight)
    profile_y = _gradient_profile(crop, axis=0, weight=weight)
    start_x = cand.region.x - x_lo
    start_y = cand.region.y - y_lo
    ox, px, sx = _comb_fit(profile_x, start_x, cell, search, offset_cells)
    oy, py, sy = _comb_fit(profile_y, start_y, cell, search, offset_cells)
    reference = _comb_strength(profile_x, start_x, cell) + _comb_strength(profile_y, start_y, cell)
    pitch = (px + py) / 2.0
    region = Region(
        int(round(x_lo + ox)),
        int(round(y_lo + oy)),
        int(round(8 * pitch)),
        int(round(8 * pitch)),
    )
    medians = cell_medians(gray, region, pitch)
    if medians is None:
        fit, contrast, occupied = 0.0, 0.0, 0
    else:
        read = colours or palette(medians)
        fit, contrast = _score_cells(gray, region, pitch, medians, read)
        occupied = occupied_cells(gray, region, pitch, medians, read)
    board = replace(
        cand, region=region, cell_px=pitch, fit=fit, contrast=contrast, occupied=occupied
    )
    return Refinement(board, sx + sy, reference)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def detect_boards(
    frame_bgr: np.ndarray,
    *,
    top_k: int = 3,
    min_fit: float = 0.85,
    min_contrast: float = 6.0,
    min_cell: int = 24,
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
    min_cell: int = 24,
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
    # Size first, then fit, so a window that is part board and part background, which can
    # pass the threshold on a palette read from itself, never outranks a clean board; the
    # occupied count only decides between boards of one size and fit.
    good.sort(key=lambda c: (c.region.w, c.fit, c.occupied), reverse=True)
    return good[:top_k]


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
