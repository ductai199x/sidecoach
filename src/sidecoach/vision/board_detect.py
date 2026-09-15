"""Classical board detection.

A 2D chess board is an axis-aligned square of 8 by 8 equal cells in two colours. The detector
works in three passes, see DESIGN.md section 5.2:

1. Coarse search. For every cell size in a geometric range, the frame is shrunk so that a cell
   is a few pixels, and every position is scored on how well its 64 cell means alternate in a
   checkerboard: |sum(sign * mean)| / sum|mean - board mean|. That is 1 for a perfect empty
   board, near 0 for anything else, and it is scale-selective because a doubled or halved cell
   size cancels the pattern.
2. Refinement. Around each surviving candidate the gradient profiles along x and y show peaks
   at the nine cell boundaries. A nine-tooth comb is fitted for sub-pixel offset and pitch.
3. Validation. Each cell's colour is read as the median of a thin border ring, which pieces
   and highlights rarely disturb, and the fraction of cells matching the checker pattern is
   the fit score.

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


@dataclass(frozen=True)
class BoardCandidate:
    region: Region
    cell_px: float
    score: float
    contrast: float
    fit: float = 0.0


def _cell_means(gray: np.ndarray, cell: int) -> np.ndarray:
    """Mean of every cell-sized window at stride 1. Shape (H - cell + 1, W - cell + 1)."""
    ii = cv2.integral(gray)
    s = cell
    return (ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]) / float(s * s)


def _checker_response(gray: np.ndarray, cell: int) -> tuple[np.ndarray, np.ndarray]:
    """(score, contrast) maps for a board whose top-left cell starts at each position."""
    means = _cell_means(gray, cell)
    h = means.shape[0] - 7 * cell
    w = means.shape[1] - 7 * cell
    if h <= 0 or w <= 0:
        empty = np.zeros((0, 0))
        return empty, empty
    acc = np.zeros((h, w), dtype=np.float64)
    total = np.zeros((h, w), dtype=np.float64)
    for r in range(8):
        for c in range(8):
            block = means[r * cell : r * cell + h, c * cell : c * cell + w]
            acc += _SIGNS[r, c] * block
            total += block
    board_mean = total / 64.0
    deviation = np.zeros((h, w), dtype=np.float64)
    for r in range(8):
        for c in range(8):
            block = means[r * cell : r * cell + h, c * cell : c * cell + w]
            deviation += np.abs(block - board_mean)
    score = np.abs(acc) / (deviation + 1e-6)
    return score, deviation / 64.0


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
    """Candidates from the multi-scale checker response, before refinement."""
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
            ys, xs = np.nonzero(mask)
            if ys.size:
                values = score[ys, xs]
                order = np.argsort(-values)[:per_scale]
                full_cell = cell / factor
                for i in order:
                    x0 = float(xs[i]) / factor
                    y0 = float(ys[i]) / factor
                    side = int(round(8 * full_cell))
                    found.append(
                        BoardCandidate(
                            Region(int(round(x0)), int(round(y0)), side, side),
                            full_cell,
                            float(values[i]),
                            float(contrast[ys[i], xs[i]]),
                        )
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


def _comb_fit(
    profile: np.ndarray, start: float, pitch: float, search: float, offset_cells: float = 0.6
) -> tuple[float, float, float]:
    """Best (offset, pitch, strength) of a nine-tooth comb over a gradient profile."""
    n = len(profile)
    xs = np.arange(n, dtype=np.float64)
    pitches = np.arange(pitch * (1 - search), pitch * (1 + search), max(0.05, pitch * 0.002))
    reach = pitch * offset_cells
    offsets = np.arange(max(0.0, start - reach), min(n - 1.0, start + reach), 0.25)
    if offsets.size == 0:
        return start, pitch, 0.0
    teeth = np.arange(9, dtype=np.float64)
    best = (start, pitch, -np.inf)
    for p in pitches:
        positions = offsets[:, None] + p * teeth[None, :]
        valid = positions[:, -1] <= n - 1
        if not valid.any():
            continue
        strength = np.interp(positions, xs, profile).sum(axis=1)
        strength[~valid] = -np.inf
        i = int(np.argmax(strength))
        if strength[i] > best[2]:
            best = (float(offsets[i]), float(p), float(strength[i]))
    return best


def cell_medians(gray: np.ndarray, region: Region, cell: float) -> np.ndarray | None:
    """8x8 border-ring medians, or None when the board runs off the frame."""
    medians = np.zeros((8, 8), dtype=np.float64)
    band = max(1, int(round(cell * 0.12)))
    for r in range(8):
        for c in range(8):
            x0 = int(round(region.x + c * cell))
            y0 = int(round(region.y + r * cell))
            x1 = int(round(region.x + (c + 1) * cell))
            y1 = int(round(region.y + (r + 1) * cell))
            patch = gray[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
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
            medians[r, c] = float(np.median(ring))
    return medians


def palette(medians: np.ndarray) -> tuple[float, float, np.ndarray]:
    """(light, dark, readable) for a grid of cell medians.

    The two square colours are the medians of the cells the checker pattern expects to be
    light and dark. A cell is readable when its median lies close to either colour, with a
    tolerance taken from how much the readable cells of each colour vary, so that a textured
    wood theme keeps its cells while a dialog, a tooltip, or a highlight drawn over a cell
    is left out of the fit. That is what lets a partly covered board keep its lock.
    """
    lights = medians[_SIGNS > 0]
    darks = medians[_SIGNS < 0]
    light = float(np.median(lights))
    dark = float(np.median(darks))
    spread = max(float(np.median(np.abs(lights - light))), float(np.median(np.abs(darks - dark))))
    tolerance = min(max(6.0, 3.0 * spread), 0.35 * abs(light - dark))
    readable = (np.abs(medians - light) <= tolerance) | (np.abs(medians - dark) <= tolerance)
    return light, dark, readable


def readable_cells(medians: np.ndarray) -> np.ndarray:
    return palette(medians)[2]


def fit_score(gray: np.ndarray, region: Region, cell: float) -> tuple[float, float]:
    """(fraction of readable cells matching the checker pattern, contrast)."""
    medians = cell_medians(gray, region, cell)
    if medians is None:
        return 0.0, 0.0
    light, dark, readable = palette(medians)
    # The threshold sits between the two square colours rather than at the board mean, which
    # a dark dialog over a light board would drag below the dark squares.
    mid = 0.5 * (light + dark)
    signs = np.sign(medians - mid)
    if int(readable.sum()) < MIN_READABLE_CELLS:
        return 0.0, float(np.abs(medians - mid).mean())
    contrast = float(np.abs(medians[readable] - mid).mean())
    # Either orientation of the checker pattern fits; a flat region, where no cell is above
    # or below the threshold, fits neither.
    match = float((signs[readable] == _SIGNS[readable]).mean())
    inverted = float((signs[readable] == -_SIGNS[readable]).mean())
    return max(match, inverted), contrast


def refine(
    gray: np.ndarray,
    cand: BoardCandidate,
    *,
    search: float = 0.12,
    offset_cells: float = 0.6,
) -> BoardCandidate:
    """Sub-pixel boundaries from gradient peaks, then validation.

    `offset_cells` is how far, in cells, the boundaries may move from the candidate. The
    tracker uses a wider window so a board that slid by a whole cell is re-locked.
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
        return cand

    # The insides of unreadable cells take no part in the boundary fit: a dialog over part
    # of the board has straight edges and text that would otherwise pull the grid towards
    # them. Cell boundaries and everything outside the candidate stay, so a board that slid
    # by a whole cell is still re-found from its outer edge.
    weight: np.ndarray | None = None
    medians = cell_medians(gray, cand.region, cell)
    if medians is not None:
        readable = readable_cells(medians)
        if int(readable.sum()) < MIN_READABLE_CELLS:
            return replace(cand, fit=0.0)
        if not readable.all():
            weight = np.ones(crop.shape, dtype=np.float32)
            pad = max(2, int(round(cell * 0.08)))
            for r in range(8):
                for c in range(8):
                    if readable[r, c]:
                        continue
                    x0 = int(round(cand.region.x + c * cell)) - x_lo + pad
                    y0 = int(round(cand.region.y + r * cell)) - y_lo + pad
                    x1 = int(round(cand.region.x + (c + 1) * cell)) - x_lo - pad
                    y1 = int(round(cand.region.y + (r + 1) * cell)) - y_lo - pad
                    if x1 > x0 and y1 > y0:
                        weight[max(0, y0) : y1, max(0, x0) : x1] = 0.0

    ox, px, _ = _comb_fit(
        _gradient_profile(crop, axis=1, weight=weight),
        cand.region.x - x_lo,
        cell,
        search,
        offset_cells,
    )
    oy, py, _ = _comb_fit(
        _gradient_profile(crop, axis=0, weight=weight),
        cand.region.y - y_lo,
        cell,
        search,
        offset_cells,
    )
    pitch = (px + py) / 2.0
    region = Region(
        int(round(x_lo + ox)),
        int(round(y_lo + oy)),
        int(round(8 * pitch)),
        int(round(8 * pitch)),
    )
    fit, contrast = fit_score(gray, region, pitch)
    return replace(cand, region=region, cell_px=pitch, fit=fit, contrast=contrast)


def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def detect_boards(
    frame_bgr: np.ndarray,
    *,
    top_k: int = 3,
    min_fit: float = 0.85,
    **coarse_kwargs: object,
) -> list[BoardCandidate]:
    """Boards in a frame, largest first, each validated against the checker pattern."""
    return detect_boards_gray(to_gray(frame_bgr), top_k=top_k, min_fit=min_fit, **coarse_kwargs)


def detect_boards_gray(
    gray: np.ndarray,
    *,
    top_k: int = 3,
    min_fit: float = 0.85,
    **coarse_kwargs: object,
) -> list[BoardCandidate]:
    coarse = coarse_search(gray, **coarse_kwargs)  # type: ignore[arg-type]
    refined = [refine(gray, cand) for cand in coarse[: top_k * 4]]
    good = [cand for cand in refined if cand.fit >= min_fit]
    good = _nms(good, "fit")
    good.sort(key=lambda c: c.region.w, reverse=True)
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
