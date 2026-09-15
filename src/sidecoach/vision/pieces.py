"""Per-theme piece recognition bootstrapped from the start position.

Every game starts from a known position, which hands us labelled examples of all twelve
piece kinds on whatever theme is on screen. Recognition then works on silhouettes:

1. Each cell's own background colour is the median of a ring just inside its border, so
   highlights need no special handling and the hover outline chess.com draws around the
   square under the cursor is skipped. A piece rarely reaches that ring.
2. Foreground is every pixel far from that background, with enclosed pockets filled (a white
   piece on a light square differs from the square mostly by its outline) and a small
   opening applied so labels and anti-aliasing do not count.
3. Each exemplar keeps its silhouette and the brightness profile of its core. A cell is
   matched against every exemplar of both colours on silhouette overlap (IoU, ignoring the
   top-right corner where chess.com draws move-quality badges in bot games) minus a penalty
   for brightness that differs from that exemplar's. Colour therefore comes from the nearest
   reference rendering, not from a global threshold: some piece sets draw a white queen
   whose interior is half black detail lines, which a threshold gets wrong.
4. When no exemplar overlaps well, the cell is still reported as occupied, with the colour
   of the exemplar whose brightness is closest, which is enough for the reconciler to
   explain a move.

A cell is also reported as highlighted when its background differs from the theme's light or
dark square colour, which is how the reconciler sees chess.com's last-move highlight.

This is theme-specific by construction and needs no training. The learned classifier planned
in DESIGN.md section 5.2 generalises across themes later and can be trained on the crops this
module labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import chess
import cv2
import numpy as np

from sidecoach.events import Side
from sidecoach.vision.squares import CELL, BoardGeometry, is_light, square_at

INNER = 0.08
RING_SKIP = 0.05
RING_BAND = 0.07
BG_TOLERANCE = 25.0
HIGHLIGHT_TOLERANCE = 30.0
EMPTY_MAX_FRACTION = 0.04
# Chess.com's legal-move hint dots cover about a tenth of a cell; the smallest piece, a pawn,
# covers a third. Anything in between is reported as unknown, which carries no evidence.
PIECE_MIN_FRACTION = 0.16
MIN_IOU = 0.35
MIN_MARGIN = 0.03
BRIGHTNESS_WEIGHT = 1.5
_KERNEL = np.ones((3, 3), dtype=np.uint8)
_CORE_KERNEL = np.ones((7, 7), dtype=np.uint8)

Kind = Literal["empty", "piece", "unknown"]


@dataclass(frozen=True)
class CellReading:
    kind: Kind
    symbol: str | None
    colour: Side | None
    confidence: float
    fg_fraction: float
    luminance: float
    highlighted: bool = False
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    brightness: float = 0.0
    # Geometry, independent of any colour: whether the piece sits where a settled piece of
    # its kind sits, and whether the square carries the hover outline drawn during a drag.
    settled: bool = True
    hover: bool = False
    offset: float = 0.0
    # Best symbol even when not confident enough to assert it; a position resync may use it
    # for the one or two cells a badge or ring obscures.
    guess: str | None = None
    # An opaque overlay confirmed by the neighbours hides this cell. The reconciler may
    # then carry the cell's last visible reading for a short while.
    covered: bool = False


def _reading(
    sil: Silhouette,
    kind: Kind,
    *,
    symbol: str | None = None,
    colour: Side | None = None,
    confidence: float = 0.0,
    highlighted: bool = False,
    settled: bool = False,
    hover: bool = False,
    offset: float = 0.0,
    guess: str | None = None,
    covered: bool = False,
) -> CellReading:
    bg = (float(sil.background[0]), float(sil.background[1]), float(sil.background[2]))
    return CellReading(
        kind,
        symbol,
        colour,
        confidence,
        sil.fraction,
        sil.luminance,
        highlighted,
        bg,
        sil.brightness,
        settled,
        hover,
        offset,
        guess,
        covered,
    )


@dataclass(frozen=True)
class Silhouette:
    mask: np.ndarray
    fraction: float
    luminance: float
    brightness: float
    background: np.ndarray
    centroid: tuple[float, float]
    hover: bool
    # The inner patch the mask was taken from, and the background read from a deeper ring,
    # which an overlay covers as well but a hover outline never reaches.
    inner: np.ndarray
    deep_background: np.ndarray


@dataclass(frozen=True)
class Exemplar:
    symbol: str
    mask: np.ndarray
    brightness: float
    centroid: tuple[float, float]


SETTLED_TOLERANCE = 0.08  # of a cell; a dragged piece follows the cursor, far off centre
# A white outline over a light square differs from it by only about 50 levels.
HOVER_TOLERANCE = 30.0


def _badge_blind(size: int) -> np.ndarray:
    """Pixels that take part in silhouette comparison: everything but the badge corner."""
    keep = np.ones((size, size), dtype=bool)
    keep[: int(size * 0.38), int(size * 0.62) :] = False
    return keep


_BADGE_BLIND = _badge_blind(CELL)


def ring_median_color(
    patch: np.ndarray, skip: float = RING_SKIP, band: float = RING_BAND
) -> np.ndarray:
    n = patch.shape[0]
    outer = max(1, int(round(n * skip)))
    inner = outer + max(1, int(round(n * band)))
    ring = np.concatenate(
        [
            patch[outer:inner, outer:-outer].reshape(-1, patch.shape[2]),
            patch[-inner:-outer, outer:-outer].reshape(-1, patch.shape[2]),
            patch[inner:-inner, outer:inner].reshape(-1, patch.shape[2]),
            patch[inner:-inner, -inner:-outer].reshape(-1, patch.shape[2]),
        ]
    )
    if ring.shape[0] == 0:  # a patch too small to have a ring at all
        ring = patch.reshape(-1, patch.shape[2])
    return np.median(ring, axis=0)


def _fill_enclosed(foreground: np.ndarray) -> np.ndarray:
    h, w = foreground.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = (~foreground).astype(np.uint8)
    padded[0, :] = padded[-1, :] = padded[:, 0] = padded[:, -1] = 1
    cv2.floodFill(padded, None, (0, 0), 2)
    return (padded[1:-1, 1:-1] != 2).astype(np.uint8)


def _hover_outline(patch: np.ndarray, background: np.ndarray) -> bool:
    """Whether the outermost border of the cell differs sharply from the cell's background.

    Chess.com draws a thin bright outline around the square under the cursor during a drag.
    The border ring used for the background skips this outer band, so it is measured here.
    """
    n = patch.shape[0]
    band = max(1, int(round(n * 0.04)))
    sides = (
        patch[:band].reshape(-1, patch.shape[2]),
        patch[-band:].reshape(-1, patch.shape[2]),
        patch[band:-band, :band].reshape(-1, patch.shape[2]),
        patch[band:-band, -band:].reshape(-1, patch.shape[2]),
    )
    # An outline runs along all four sides; a badge or a dialog edge crossing the cell
    # changes one or two sides only and leaves the others near zero. The weakest side of a
    # real outline can be thin, at small cells or where the outline is drawn a pixel
    # narrower, so it only has to clear half the tolerance while the median side clears it.
    if any(side.shape[0] == 0 for side in sides):
        return False  # a patch too small to have an outer band
    distances = sorted(
        float(
            np.median(
                np.linalg.norm(side.astype(np.float32) - background.astype(np.float32), axis=1)
            )
        )
        for side in sides
    )
    return distances[0] > HOVER_TOLERANCE / 2 and distances[2] > HOVER_TOLERANCE


def silhouette(patch: np.ndarray, size: int = CELL) -> Silhouette:
    """Silhouette mask, foreground fraction, colour statistics, centroid, hover, background."""
    background = ring_median_color(patch)
    border = int(round(patch.shape[0] * INNER))
    inner = patch[border : patch.shape[0] - border, border : patch.shape[1] - border]
    inner = cv2.resize(inner, (size, size), interpolation=cv2.INTER_AREA)
    distance = np.linalg.norm(inner.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = _fill_enclosed(distance > BG_TOLERANCE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    fraction = float(mask.mean())
    luminance = 0.0
    brightness = 0.0
    centroid = (0.5, 0.5)
    if fraction > 0:
        # Colour statistics come from the core of the piece, not its outline: white pieces
        # carry dark outlines and black pieces light ones, and thin pieces are mostly outline.
        core = cv2.erode(mask, _CORE_KERNEL)
        sample_mask = (core if core.any() else mask).astype(bool)
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
        sample = gray[sample_mask]
        luminance = float(np.median(sample))
        brightness = float((sample > 128).mean())
        ys, xs = np.nonzero(mask)
        centroid = (float(xs.mean()) / size, float(ys.mean()) / size)
    # Chess.com's hover outline is about 5% of the cell wide and sits at its edge, so the
    # ring the background is read from stays clean and serves as the reference. A deeper
    # ring is read as well, for telling a covered cell from a hovered one; it is not used as
    # the hover reference because a tall piece reaches into it and shifts its median.
    deep_background = ring_median_color(patch, skip=RING_SKIP + RING_BAND)
    hover = _hover_outline(patch, background)
    return Silhouette(
        mask.astype(bool),
        fraction,
        luminance,
        brightness,
        background,
        centroid,
        hover,
        inner,
        deep_background,
    )


COVER_TOLERANCE = 12.0  # two cells under one opaque overlay share a background
TINT_MIN_ALPHA = 0.2  # a translucent overlay this faint already spoils the pieces under it
TINT_ALPHA_SPREAD = 0.15  # the alpha read from each colour channel must agree
TINT_MIN_CELLS = 6  # highlights and marks tint a few cells; a fading dialog tints a block
TINT_BLOCK_FILL = 0.8  # of the block's bounding box; a dialog is a filled rectangle
OVERLAY_SIDE_FILL = 0.9  # an overlay edge fills a whole side of the inner patch
OVERLAY_BLOCK_FILL = 0.85  # an overlay corner is a solid rectangle


def opaque_cover(backgrounds: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    """Cells whose background matches a 4-neighbour's, both away from their square colours.

    A translucent tint keeps light and dark squares apart under it; only an opaque overlay
    makes two neighbouring squares the same colour. That marks a dialog, a tooltip, or the
    promotion picker as foreign whatever colour it has, while a badge or a highlight on a
    single cell is never mistaken for one.
    """
    covered = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            if not shifted[r, c]:
                continue
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < 8 and 0 <= cc < 8 and shifted[rr, cc]:
                    gap = float(np.linalg.norm(backgrounds[r, c] - backgrounds[rr, cc]))
                    if gap <= COVER_TOLERANCE:
                        covered[r, c] = True
                        break
    return covered


def _same_tint(x: np.ndarray, y: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> bool:
    """Whether two neighbouring cells sit under one translucent overlay.

    Under an overlay of colour T and opacity a, a cell of colour X shows (1 - a) X + a T,
    so the difference between two neighbours shrinks to (1 - a) of what the theme gives
    them. The opacity read from each colour channel must agree, be strong enough to hide a
    piece, and imply an overlay colour that exists.
    """
    alphas = []
    for c in range(3):
        gap = float(ex[c] - ey[c])
        if abs(gap) < 8.0:
            continue
        alphas.append(1.0 - float(x[c] - y[c]) / gap)
    if len(alphas) < 2:
        return False
    if max(alphas) - min(alphas) > TINT_ALPHA_SPREAD:
        return False
    alpha = float(np.mean(alphas))
    if not TINT_MIN_ALPHA <= alpha <= 1.05:
        return False
    alpha = min(alpha, 1.0)
    tint = (x - (1.0 - alpha) * ex) / alpha
    return bool(np.all(tint >= -15.0) and np.all(tint <= 270.0))


def translucent_cover(
    backgrounds: np.ndarray, expected: np.ndarray, shifted: np.ndarray
) -> np.ndarray:
    """Cells in a block of at least TINT_MIN_CELLS that share one translucent overlay.

    A dialog fades in over a few frames; while it is half transparent the pieces under
    it are unreadable but the light and dark squares still differ, so the opaque test
    does not fire and the cells would read as junk. A block of neighbours all consistent
    with one tint is that fading dialog. A move highlight or a selection tints one to
    three cells and a right-click mark a few, and stay below the block size.
    """
    parent = list(range(64))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(8):
        for c in range(8):
            if not shifted[r, c]:
                continue
            for rr, cc in ((r, c + 1), (r + 1, c)):
                if (
                    rr < 8
                    and cc < 8
                    and shifted[rr, cc]
                    and _same_tint(
                        backgrounds[r, c], backgrounds[rr, cc], expected[r, c], expected[rr, cc]
                    )
                ):
                    parent[find(r * 8 + c)] = find(rr * 8 + cc)
    members: dict[int, list[int]] = {}
    for i in range(64):
        if shifted[i // 8, i % 8]:
            members.setdefault(find(i), []).append(i)
    covered = np.zeros((8, 8), dtype=bool)
    for cells in members.values():
        if len(cells) < TINT_MIN_CELLS:
            continue
        # A dialog is a filled rectangle at least two cells wide and high. A file of
        # right-click marks or the squares of a queued premove make a line, and stay
        # readable.
        rows = [i // 8 for i in cells]
        cols = [i % 8 for i in cells]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        if height < 2 or width < 2 or len(cells) < TINT_BLOCK_FILL * height * width:
            continue
        for i in cells:
            covered[i // 8, i % 8] = True
    # A highlighted square or one with the dialog's text over its ring does not fit the
    # tint, yet it lies under the same overlay when its neighbours do.
    for _ in range(2):
        for r in range(8):
            for c in range(8):
                if covered[r, c] or not shifted[r, c]:
                    continue
                around = sum(
                    1
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                    if 0 <= rr < 8 and 0 <= cc < 8 and covered[rr, cc]
                )
                if around >= 2:
                    covered[r, c] = True
    return covered


def overlay_reaches(sil: Silhouette, colour: np.ndarray) -> bool:
    """Whether an overlay of `colour` reaches into this cell from a covered neighbour.

    The overlay's part of the cell is a rectangle anchored on a side or a corner: it fills
    a whole side of the inner patch, or a solid block holding a corner. A piece does
    neither, so a white piece next to a white dialog keeps reading as a piece.
    """
    distance = np.linalg.norm(sil.inner.astype(np.float32) - colour.astype(np.float32), axis=2)
    near = sil.mask & (distance <= BG_TOLERANCE)
    if not near.any():
        return False
    sides = (near[0], near[-1], near[:, 0], near[:, -1])
    if any(float(side.mean()) >= OVERLAY_SIDE_FILL for side in sides):
        return True
    if float(near.mean()) < EMPTY_MAX_FRACTION:
        return False
    ys, xs = np.nonzero(near)
    box = near[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if float(box.mean()) < OVERLAY_BLOCK_FILL:
        return False
    corners = (near[:3, :3], near[:3, -3:], near[-3:, :3], near[-3:, -3:])
    return any(bool(corner.all()) for corner in corners)


def iou(a: np.ndarray, b: np.ndarray, keep: np.ndarray = _BADGE_BLIND) -> float:
    a, b = a & keep, b & keep
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class ThemeModel:
    exemplars: list[Exemplar] = field(default_factory=list)
    light_bg: np.ndarray | None = None
    dark_bg: np.ndarray | None = None

    @property
    def symbols(self) -> set[str]:
        return {e.symbol for e in self.exemplars}

    def highlight_kind(self, background: np.ndarray, row: int, col: int) -> str:
        """ "none", "move", "red", or "foreign" for what changed a cell's background.

        Chess.com blends a translucent tint over the square, so the tint's hue is read from
        the direction of the colour shift rather than from the blended colour itself. A
        yellow-ish tint (last move, selection) never raises blue and never lowers green much;
        a red right-click mark raises red while lowering green; anything else, such as the
        white or grey of a dialog or tooltip drawn over the board, is foreign and the cell's
        content cannot be read. Arrows are drawn shapes and never change a background.
        """
        expected = self.light_bg if is_light(row, col) else self.dark_bg
        if expected is None:
            return "none"
        shift = background.astype(np.float32) - expected.astype(np.float32)
        if float(np.linalg.norm(shift)) <= HIGHLIGHT_TOLERANCE:
            return "none"
        blue, green, red = (float(v) for v in shift)
        if green < -10.0 and red > green + 25.0 and blue <= 5.0:
            return "red"
        if blue <= 5.0 and green >= -15.0:
            return "move"
        return "foreign"

    def _classify(self, sil: Silhouette, row: int, col: int, covered: bool) -> CellReading:
        kind_of_tint = self.highlight_kind(sil.background, row, col)
        highlighted = kind_of_tint == "move"
        if covered or kind_of_tint == "foreign":
            # Something is drawn over this square; whatever the silhouette shows is not the
            # board. No evidence either way. An opaque overlay confirmed by the neighbours
            # cannot be a hover outline, however its edges cross the cell; an odd colour on
            # a lone cell may still be a hovered square whose thick outline reached the
            # ring, and that outline must keep blocking commits.
            hover = (
                not covered
                and sil.hover
                and self.highlight_kind(sil.deep_background, row, col) != "foreign"
            )
            return _reading(sil, "unknown", hover=hover, covered=covered)
        if sil.fraction <= EMPTY_MAX_FRACTION:
            return _reading(
                sil, "empty", confidence=1.0, highlighted=highlighted, settled=True, hover=sil.hover
            )
        if sil.fraction < PIECE_MIN_FRACTION or not self.exemplars:
            return _reading(sil, "unknown", highlighted=highlighted, hover=sil.hover)

        best_by_symbol: dict[str, tuple[float, float, Exemplar]] = {}
        for exemplar in self.exemplars:
            overlap = iou(sil.mask, exemplar.mask)
            score = overlap - BRIGHTNESS_WEIGHT * abs(sil.brightness - exemplar.brightness)
            previous = best_by_symbol.get(exemplar.symbol)
            if previous is None or score > previous[0]:
                best_by_symbol[exemplar.symbol] = (score, overlap, exemplar)
        ranked = sorted(best_by_symbol.items(), key=lambda item: item[1][0], reverse=True)
        symbol, (score, overlap, exemplar) = ranked[0]
        runner_up = ranked[1][1][0] if len(ranked) > 1 else -1.0
        offset = float(
            np.hypot(sil.centroid[0] - exemplar.centroid[0], sil.centroid[1] - exemplar.centroid[1])
        )
        settled = offset <= SETTLED_TOLERANCE and not sil.hover
        if overlap >= MIN_IOU and score - runner_up >= MIN_MARGIN:
            colour = Side.WHITE if symbol.isupper() else Side.BLACK
            return _reading(
                sil,
                "piece",
                symbol=symbol,
                colour=colour,
                confidence=score - runner_up,
                highlighted=highlighted,
                settled=settled,
                hover=sil.hover,
                offset=offset,
            )
        nearest = min(self.exemplars, key=lambda e: abs(sil.brightness - e.brightness))
        colour = Side.WHITE if nearest.symbol.isupper() else Side.BLACK
        guess = symbol if symbol.isupper() == (colour == Side.WHITE) else None
        return _reading(
            sil,
            "piece",
            colour=colour,
            confidence=0.5,
            highlighted=highlighted,
            hover=sil.hover,
            offset=offset,
            guess=guess,
        )

    def covered_cells(self, sils: list[list[Silhouette]]) -> np.ndarray:
        """Cells under an opaque overlay, and the cells its edges reach into."""
        covered = np.zeros((8, 8), dtype=bool)
        if self.light_bg is None or self.dark_bg is None:
            return covered
        backgrounds = np.array([[sil.background for sil in row] for row in sils], dtype=np.float32)
        expected = np.zeros((8, 8, 3), dtype=np.float32)
        shifted = np.zeros((8, 8), dtype=bool)
        for r in range(8):
            for c in range(8):
                expected[r, c] = self.light_bg if is_light(r, c) else self.dark_bg
                shift = float(np.linalg.norm(backgrounds[r, c] - expected[r, c]))
                shifted[r, c] = shift > HIGHLIGHT_TOLERANCE
        covered = opaque_cover(backgrounds, shifted) | translucent_cover(
            backgrounds, expected, shifted
        )
        reached = np.zeros((8, 8), dtype=bool)
        for r in range(8):
            for c in range(8):
                if covered[r, c]:
                    continue
                for rr in range(max(0, r - 1), min(8, r + 2)):
                    for cc in range(max(0, c - 1), min(8, c + 2)):
                        if covered[rr, cc] and overlay_reaches(sils[r][c], backgrounds[rr, cc]):
                            reached[r, c] = True
                            break
                    if reached[r, c]:
                        break
        return covered | reached

    def classify_frame(self, frame: np.ndarray, geometry: BoardGeometry) -> list[list[CellReading]]:
        sils = [
            [silhouette(geometry.crop(frame, r, c, size=2 * CELL)) for c in range(8)]
            for r in range(8)
        ]
        covered = self.covered_cells(sils)
        readings = [
            [self._classify(sils[r][c], r, c, bool(covered[r, c])) for c in range(8)]
            for r in range(8)
        ]
        if int(covered.sum()) < TINT_MIN_CELLS:
            return readings
        # Next to a dialog-sized overlay, a cell that does not read cleanly is a cell the
        # overlay's edge cuts through: a strip of dialog over a piece reads as the wrong
        # piece, unsettled, with a hover outline that would keep every commit blocked
        # under the dialog. Nothing is being dragged while a dialog is up.
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                junk = reading.hover or (
                    reading.kind == "piece" and (not reading.settled or reading.symbol is None)
                )
                if not junk or covered[r, c]:
                    continue
                beside = any(
                    covered[rr, cc]
                    for rr in range(max(0, r - 1), min(8, r + 2))
                    for cc in range(max(0, c - 1), min(8, c + 2))
                )
                if beside:
                    readings[r][c] = replace(
                        reading,
                        kind="unknown",
                        symbol=None,
                        colour=None,
                        confidence=0.0,
                        highlighted=False,
                        settled=False,
                        hover=False,
                        offset=0.0,
                        guess=None,
                        covered=True,
                    )
        return readings

    @classmethod
    def from_start_position(
        cls, frame: np.ndarray, geometry: BoardGeometry, bottom: Side
    ) -> ThemeModel | None:
        """The recogniser for this game, or None when the start frame is not clean enough.

        A cursor, a hover outline, or a badge on one cell of the start frame would
        otherwise become that piece's exemplar for the whole game, and the king and the
        queen have one cell each. So every cell must show a plausible piece: enough
        foreground, centred, no hover outline. Pieces with several cells (the pawns, the
        pairs) are fused by majority vote, which keeps one odd cell from spoiling them.
        """
        start = chess.Board()
        light_bgs: list[np.ndarray] = []
        dark_bgs: list[np.ndarray] = []
        samples: dict[str, list[Silhouette]] = {}
        for r in range(8):
            for c in range(8):
                sil = silhouette(geometry.crop(frame, r, c, size=2 * CELL))
                piece = start.piece_at(square_at(r, c, bottom))
                if piece is None:
                    (light_bgs if is_light(r, c) else dark_bgs).append(sil.background)
                    continue
                if not _plausible_exemplar(sil):
                    return None
                samples.setdefault(piece.symbol(), []).append(sil)
        model = cls()
        for symbol, sils in samples.items():
            votes = np.mean([sil.mask for sil in sils], axis=0)
            mask = votes >= 0.5
            if float(mask.mean()) < PIECE_MIN_FRACTION:
                return None
            brightness = float(np.median([sil.brightness for sil in sils]))
            centroid = (
                float(np.median([sil.centroid[0] for sil in sils])),
                float(np.median([sil.centroid[1] for sil in sils])),
            )
            model.exemplars.append(Exemplar(symbol, mask, brightness, centroid))
        model.light_bg = np.median(np.array(light_bgs), axis=0)
        model.dark_bg = np.median(np.array(dark_bgs), axis=0)
        gap = float(np.linalg.norm(model.light_bg - model.dark_bg))
        if gap <= COVER_TOLERANCE:
            return None  # the two square colours cannot be told apart
        return model


EXEMPLAR_MAX_FRACTION = 0.9
# A piece is drawn on the middle of its file; how high it sits in the square is its own
# business (a pawn's centre of mass lies well below the centre, a king's near it).
EXEMPLAR_MAX_SIDEWAYS = 0.12  # of a cell
EXEMPLAR_MAX_VERTICAL = 0.28  # of a cell


def _plausible_exemplar(sil: Silhouette) -> bool:
    """A piece drawn on its own square: enough foreground, centred, nothing over it."""
    if sil.hover or not PIECE_MIN_FRACTION <= sil.fraction <= EXEMPLAR_MAX_FRACTION:
        return False
    return (
        abs(sil.centroid[0] - 0.5) <= EXEMPLAR_MAX_SIDEWAYS
        and abs(sil.centroid[1] - 0.5) <= EXEMPLAR_MAX_VERTICAL
    )


def occupancy(frame: np.ndarray, geometry: BoardGeometry) -> np.ndarray:
    """8x8 foreground fractions, the cheap occupancy signal."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            grid[r, c] = silhouette(geometry.crop(frame, r, c, size=2 * CELL)).fraction
    return grid


def looks_like_start_position(frame: np.ndarray, geometry: BoardGeometry) -> bool:
    """Rows 0, 1, 6, 7 occupied and rows 2 to 5 empty, allowing one stray cell.

    One move away from the start already differs in two cells, so the allowance is a single
    cell for a badge, a hover outline, or a cursor.
    """
    grid = occupancy(frame, geometry)
    occupied = grid >= PIECE_MIN_FRACTION
    expected = np.zeros((8, 8), dtype=bool)
    expected[[0, 1, 6, 7], :] = True
    return int((occupied != expected).sum()) <= 1


def orientation_from_start(frame: np.ndarray, geometry: BoardGeometry) -> Side:
    """Which colour sits at the bottom of a start position: the brighter pieces are white."""
    top: list[float] = []
    bottom: list[float] = []
    for c in range(8):
        for r in (0, 1):
            top.append(silhouette(geometry.crop(frame, r, c, size=2 * CELL)).luminance)
        for r in (6, 7):
            bottom.append(silhouette(geometry.crop(frame, r, c, size=2 * CELL)).luminance)
    return Side.WHITE if np.median(bottom) > np.median(top) else Side.BLACK
