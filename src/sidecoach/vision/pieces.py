"""Per-theme piece recognition bootstrapped from the start position.

Every game starts from a known position, which hands us labelled examples of all twelve
piece kinds on whatever theme is on screen. Recognition then works on silhouettes:

1. Each cell's own background colour is the median of a ring just inside its border, so
   highlights need no special handling and the thin hover outline chess.com draws around
   the square under the cursor is skipped. A piece rarely reaches that ring.
2. Foreground is every pixel far from that background, with enclosed pockets filled (a white
   piece on a light square differs from the square mostly by its outline) and a small
   opening applied so labels and anti-aliasing do not count.
3. Each exemplar keeps its silhouette and the brightness profile of its core. A cell is
   matched against every exemplar of both colours on silhouette overlap (IoU, ignoring the
   top-right corner where chess.com draws move-quality badges in bot games, after moving
   the cell's mask onto the exemplar's centroid so a lock a pixel or two off still matches
   by shape) minus a penalty for brightness that differs from that exemplar's. Colour
   therefore comes from the nearest reference rendering, not from a global threshold: some
   piece sets draw a white queen whose interior is half black detail lines, which a
   threshold gets wrong. Whether the piece has landed is judged separately, by how far its
   centroid sits from the exemplar's.
4. When no exemplar overlaps well, the cell is still reported as occupied, with the colour
   of the exemplar whose brightness is closest, which is enough for the reconciler to
   explain a move.

What sits on top of the board is told apart by geometry, never by colour:

- A tint (a last-move or selection highlight, the red of a check, a premove queue or a
  right-click mark, in whatever hue a site or theme uses) shifts the cell's ring and the
  deeper corner blocks of the cell in the same direction, leaving no corner pixel at the
  square's own colour. The cell stays readable, with `highlighted` set and `tint` naming
  the hue.
- A foreign cell has its ring shifted but corners that are clean, in part at least, or
  shifted another way: a hover outline thick enough to fill the ring, a capture ring, the
  edge of something drawn over the square. It reads as unknown with `hover` set, so
  nothing commits under it.
- A cover is an opaque or translucent overlay confirmed by the neighbours: two adjacent
  cells that share one background away from their square colours, or a block of six or
  more cells that all fit one translucent tint (a dialog fading in). The cells it reaches
  into read as covered too: unknown, never a hover, and the reconciler may keep their last
  stable reading for a while. Next to a dialog-sized cover (a block, never a strip), cells
  that read as junk are relabelled as covered, since nothing is dragged under a dialog.
- A tint over half the board or more is no highlight but a theme this recogniser was not
  built on; those cells read as unknown until the pipeline rebuilds it from a start
  position.
- A drawing -- an arrow, the edge of a mark -- is saturated colour that is neither the
  square's own colour nor a colour of the piece set (the fill and the outline of either
  side, learned from the start position) nor a blend of those where a piece's edge is
  smoothed onto the square. A cell with more than a sliver of such pixels reads as
  unknown with `drawn` set until the drawing is gone, since an arrowhead over a piece can
  look like a different piece, and a move must not commit through it. It is not a hover:
  nothing is in flight, and the rest of the board reads on. An arrow a player leaves on
  the board for several moves once held every one of them back.

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
# The four corner blocks between these insets sample the background deeper than the ring:
# a piece is round-ish and covers at most one or two of them partially, a badge one. Deeper
# is not an option: on the recorded games rooks and bishops reach blocks at 12% and beyond.
CORNER_INSET = (0.10, 0.20)
BG_TOLERANCE = 25.0
HIGHLIGHT_TOLERANCE = 30.0
# A tint shifts the ring and the corners the same way; an outline or an edge does not. Each
# corner pixel is read on its own: clean (at the square colour), shifted like the ring, or
# something else (a piece, a badge, a label), and a cell is foreign when a quarter of the
# corner pixels that are one or the other are clean, or when too few are either.
TINT_MIN_COSINE = 0.9
TINT_MIN_CORNER_SHIFT = HIGHLIGHT_TOLERANCE / 2
CORNER_CLEAN_MAX = 0.25
CORNER_MIN_KNOWN = 0.2
# Something drawn over a square -- an arrow, a mark's edge -- is saturated colour far from
# the square's own colour and from every colour of the piece set. The smoothed edge of a
# dark piece on a bright highlight is chromatic too, but it lies on the line from the
# square's colour to the piece's; a drawing is off every such line by more than
# OVERLAY_OFF_LINE. The lines to black and to white are always among them, and a set's
# own fills and outlines, learned from the start position, join them: a cream-and-brown
# set is chromatic everywhere and once had no rule at all, so a thin red arrow over an
# empty square read as a piece in flight and held the game. A cell with more than
# OVERLAY_MAX of its inner pixels like that cannot be read, so no move commits through it.
OVERLAY_DISTANCE = 60.0
OVERLAY_CHROMA = 70.0
OVERLAY_OFF_LINE = 40.0
OVERLAY_MAX = 0.03
# The move-classification badge sits on the top-right corner; a quarter-disc of this
# radius (as a share of the cell) around that corner is not measured for drawings.
OVERLAY_BADGE_RADIUS = 0.4
# An outline colour is read from a piece only when this many pixels of its silhouette
# lie outside its own tone.
EDGE_MIN_PIXELS = 8
EMPTY_MAX_FRACTION = 0.04
# Chess.com's legal-move hint dots cover about a tenth of a cell; the smallest piece, a pawn,
# covers a third. Anything in between is reported as unknown, which carries no evidence.
PIECE_MIN_FRACTION = 0.16
MIN_IOU = 0.35
MIN_MARGIN = 0.03
BRIGHTNESS_WEIGHT = 1.5
_KERNEL = np.ones((3, 3), dtype=np.uint8)
# The core of a piece, where its colour is read: the pixels at least this far inside the
# silhouette's edge, as a share of the deepest pixel's distance, and never nearer than
# CORE_MIN_DEPTH. A fixed erosion was three pixels, and a bold set's outline is wider
# than that at 56-pixel cells: the rim it left in the core made a white queen, whose
# fill is a thin star inside its outline, as grey as the black one.
CORE_DEPTH = 0.5
CORE_MIN_DEPTH = 3.0

Kind = Literal["empty", "piece", "unknown"]
TintKind = Literal["none", "tint", "foreign"]


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
    # The default is True so a reading built by hand (tests, the reconciler's memory) counts
    # as landed; `_reading` below defaults it to False and every branch of `_classify`
    # decides it explicitly.
    settled: bool = True
    hover: bool = False
    offset: float = 0.0
    # Best symbol even when not confident enough to assert it; a position resync may use it
    # for the one or two cells a badge or ring obscures.
    guess: str | None = None
    # An opaque overlay confirmed by the neighbours hides this cell. The reconciler may
    # then carry the cell's last visible reading for a short while.
    covered: bool = False
    # The hue of a translucent tint over the square, when the cell is still readable:
    # "yellow" (a last-move or selection highlight), "red" (a premove queue square, a
    # check, a right-click mark), or "other" (a green, blue or grey highlight on another
    # site or theme). None when the square shows its own colour.
    tint: str | None = None
    # Something drawn over the square -- an arrow, a mark's edge -- makes it unreadable
    # without anything being dragged: no move commits through it, and nothing else waits.
    drawn: bool = False


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
    tint: str | None = None,
    drawn: bool = False,
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
        tint,
        drawn,
    )


@dataclass(frozen=True)
class Silhouette:
    mask: np.ndarray
    # The mask restricted to the piece's own tone -- the dark fill of a black piece, the
    # bright fill of a white one -- with its enclosed detail filled in. A piece's outline
    # is the other tone, and whether it shows depends on the square under it: a light
    # outline vanishes on a light square and stands out on a dark one, so the full mask
    # of one piece is two shapes, and at small cells a black queen on a dark square
    # matched the pawn learned on a light one. Shapes are compared on this mask.
    tone: np.ndarray
    fraction: float
    luminance: float
    brightness: float
    background: np.ndarray
    centroid: tuple[float, float]
    hover: bool
    # The inner patch the mask was taken from, and the pixels of the four corner blocks,
    # deeper than the ring: a tint reaches them, an outline does not.
    inner: np.ndarray
    corner_pixels: np.ndarray
    # The share of inner pixels that are saturated colour far from the square's own colour
    # and the set's (an arrow drawn over the square).
    overlay: float = 0.0
    # The median colour of the piece's core (its fill) and of the rest of its silhouette
    # (its outline), when there is one: the set's colours, learned at the start.
    core_colour: np.ndarray | None = None
    edge_colour: np.ndarray | None = None


@dataclass(frozen=True)
class Exemplar:
    symbol: str
    mask: np.ndarray  # the full silhouette, fused over the piece's cells
    tone: np.ndarray  # its tone mask (see Silhouette.tone), fused the same way
    brightness: float
    centroid: tuple[float, float]
    # Centroid of the tone mask outside the badge corner, the reference a reading's mask is
    # moved onto before overlap is measured, so a badge on either side moves nothing.
    align_centroid: tuple[float, float]
    # The same for the full mask: where the piece's ink sits, which is what decides
    # whether it has landed. The fuller shape moves less under a lock a pixel off.
    landing_centroid: tuple[float, float]


SETTLED_TOLERANCE = 0.08  # of a cell; a dragged piece follows the cursor, far off centre
SETTLED_FLOOR_PX = 3.0  # source pixels: the lock itself is only good to about a pixel
# A white outline over a light square differs from it by only about 50 levels.
HOVER_TOLERANCE = 30.0
# The outer band an outline starts in: deep enough for one drawn a pixel inside the edge at
# 32-px cells, shallow enough that the base of a pawn on a small cell stays out of it.
HOVER_BAND = 0.05  # of the cell
# An outline covers a side end to end; the widest row of a piece that reaches the band
# covers 0.85 of it on the recorded games.
HOVER_LINE_FILL = 0.9
# The outermost edge, judged as a whole by its median: chess.com's outline fills most of it.
HOVER_EDGE = 0.04  # of the cell


def settle_tolerance(cell_px: float | None) -> float:
    """The landing tolerance in inner-patch units: a fraction of a cell, but never less
    than a few source pixels, so a small cell does not turn lock jitter into a piece in
    flight."""
    if cell_px is None:
        return SETTLED_TOLERANCE
    inner_px = cell_px * (1.0 - 2.0 * INNER)
    if inner_px <= 0:
        return SETTLED_TOLERANCE
    return max(SETTLED_TOLERANCE, SETTLED_FLOOR_PX / inner_px)


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


def corner_pixels(patch: np.ndarray, inset: tuple[float, float] = CORNER_INSET) -> np.ndarray:
    """The pixels of the four corner blocks between the two insets, as an N by 3 array.

    Deeper than the ring, so a thin hover outline never reaches them, yet where a piece
    rarely is: a piece covers at most one or two corners partially and a badge one.
    """
    n = patch.shape[0]
    outer = int(round(n * inset[0]))
    inner = max(outer + 1, int(round(n * inset[1])))
    blocks = (
        patch[outer:inner, outer:inner],
        patch[outer:inner, n - inner : n - outer],
        patch[n - inner : n - outer, outer:inner],
        patch[n - inner : n - outer, n - inner : n - outer],
    )
    pixels = np.concatenate([block.reshape(-1, patch.shape[2]) for block in blocks])
    if pixels.shape[0] == 0:  # a patch too small to have corners inside its border
        pixels = patch.reshape(-1, patch.shape[2])
    return pixels.astype(np.float32)


def _fill_enclosed(foreground: np.ndarray) -> np.ndarray:
    h, w = foreground.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = (~foreground).astype(np.uint8)
    padded[0, :] = padded[-1, :] = padded[:, 0] = padded[:, -1] = 1
    cv2.floodFill(padded, None, (0, 0), 2)
    return (padded[1:-1, 1:-1] != 2).astype(np.uint8)


# A tone mask smaller than this share of the full mask, or this share of the patch, is a
# piece of no clear tone (a mid-grey set, a glass set): the full mask serves. The share
# was 0.4, and a wooden queen's fill, a thin star inside a wide tan outline, was 0.39 of
# its silhouette on the dark squares (where the outline shows) and 0.87 on the light
# ones (where it does not), so the queen had one shape on each colour of square.
TONE_MIN_SHARE = 0.2
TONE_MIN_AREA = 0.08


def _tone_mask(
    mask: np.ndarray, inner: np.ndarray, core_colour: np.ndarray, background: np.ndarray
) -> np.ndarray:
    """The mask restricted to the piece's own tone (see `Silhouette.tone`): the pixels
    nearer the colour of the piece's core than the square's. The outline is the other
    tone, and it is nearer the square wherever the square can show it: a black piece's
    grey outline on a dark board lies between its fill and the squares, and a split at
    mid-grey kept it on light squares and lost it on dark ones, where its blend with the
    fill is the square's own colour and the opening takes the detached line."""
    pixels = inner.astype(np.float32)
    to_core = np.linalg.norm(pixels - core_colour.astype(np.float32), axis=2)
    to_square = np.linalg.norm(pixels - background.astype(np.float32), axis=2)
    core = mask & (to_core < to_square)
    if int(core.sum()) < max(TONE_MIN_SHARE * int(mask.sum()), TONE_MIN_AREA * core.size):
        return mask
    return _fill_enclosed(core).astype(bool) & mask


def _hover_outline(patch: np.ndarray, background: np.ndarray) -> bool:
    """Whether an outline differing from the cell's background runs along all four sides.

    Chess.com draws a thin bright outline around the square under the cursor during a
    drag, at the edge or a pixel inside it. It shows in one of two ways: the outermost
    edge of the cell differs from the background as a whole (its median, on every side),
    or one row of the outer band, at some depth, differs end to end on every side, which
    is how a one-pixel outline drawn a pixel inside the edge looks. The band stops before
    the ring the background is read from (a thicker outline fills that ring and is caught
    as foreign instead) and before the base of a pawn on a small cell. A badge, a dialog
    edge, a neighbour's colour let in by a lock a pixel off, or the blended boundary
    between two squares, all mark one or two sides only, never all four; a piece reaching
    the band never covers a side end to end. The weakest side of a real outline can be
    faint, at small cells or where it is drawn a pixel narrower, so it only has to clear
    half the tolerance while the others clear it.
    """
    n = patch.shape[0]
    band = max(1, int(round(n * HOVER_BAND)))
    edge = max(1, int(round(n * HOVER_EDGE)))
    if 2 * band >= n:
        return False  # a patch too small to have an outer band
    distance = np.linalg.norm(patch.astype(np.float32) - background.astype(np.float32), axis=2)
    sides = (distance, distance[::-1], distance.T, distance.T[::-1])
    strong: list[float] = []
    weak: list[float] = []
    medians: list[float] = []
    for side in sides:
        rows = side[:band, band : n - band]
        strong.append(float((rows > HOVER_TOLERANCE).mean(axis=1).max()))
        weak.append(float((rows > HOVER_TOLERANCE / 2).mean(axis=1).max()))
        medians.append(float(np.median(side[:edge])))
    strong.sort()
    weak.sort()
    medians.sort()
    # Either the outermost edge differs from the background on every side, taken as a
    # whole (an outline thick enough to fill most of it), or one row of the band does so
    # end to end on every side (a thin outline a pixel inside the edge).
    if medians[0] > HOVER_TOLERANCE / 2 and medians[2] > HOVER_TOLERANCE:
        return True
    return weak[0] >= HOVER_LINE_FILL and strong[1] >= HOVER_LINE_FILL


def _fill_colour(core: np.ndarray, boundary: np.ndarray | None) -> np.ndarray:
    """The colour of a piece's fill: the median of the half of its core pixels that lie
    farthest from its outline's colour (the median of the silhouette's outermost ring).
    A set draws detail inside its pieces in the outline's colour, and a queen whose
    centre is half such lines had a core as grey as the other side's queen; the half of
    the core least like the outline is the fill whatever is drawn on it."""
    if boundary is None or len(boundary) == 0 or len(core) < 2:
        return np.median(core, axis=0)
    edge = np.median(boundary.astype(np.float32), axis=0)
    away = np.linalg.norm(core - edge, axis=1)
    return np.median(core[away >= np.median(away)], axis=0)


def _core_mask(mask: np.ndarray) -> np.ndarray:
    """The deep inside of the silhouette (see CORE_DEPTH), or the whole of it when it is
    too thin to have one."""
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    core = depth >= max(CORE_MIN_DEPTH, CORE_DEPTH * float(depth.max()))
    return core if core.any() else mask.astype(bool)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    """Centre of mass of a mask in units of its own size; the middle when it is empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.5, 0.5)
    n = mask.shape[0]
    return (float(xs.mean()) / n, float(ys.mean()) / n)


def _align_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Where a piece sits for alignment: the centroid of the mask outside the badge corner.

    On the recorded games this lands within a pixel of the true shift of a moved lock
    (mean 0.9 px, worst 2.7); the strokes of a coordinate label can pull it a pixel off,
    which is why overlap is also measured in place and the better of the two is kept.
    """
    return _centroid(mask & _BADGE_BLIND)


def _overlay_blind(size: int) -> np.ndarray:
    """Pixels the overlay measure looks at: everything outside the badge's corner disc."""
    ys, xs = np.mgrid[0:size, 0:size]
    return np.hypot(xs - (size - 1), ys) / size > OVERLAY_BADGE_RADIUS


_OVERLAY_BLIND = _overlay_blind(CELL)


def _off_line(pixels: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """How far each pixel lies from the segment of colours between `start` and `end`."""
    direction = end - start
    length = float(direction @ direction) or 1.0
    along = np.clip(((pixels - start) @ direction) / length, 0.0, 1.0)
    nearest = start + along[:, None] * direction
    return np.linalg.norm(pixels - nearest, axis=1)


PieceColours = tuple[tuple[np.ndarray, np.ndarray | None], ...]  # (fill, outline) a side


def _overlay_fraction(
    inner: np.ndarray,
    background: np.ndarray,
    distance: np.ndarray,
    chroma_map: np.ndarray,
    colours: PieceColours,
) -> float:
    """The share of inner pixels that are a drawing: chromatic, far from the square's colour,
    and off the lines from that colour to black, to white and to each of the set's colours,
    and off the line from each side's fill to its outline, where piece edges blend.

    The badge's quarter-disc at the top-right corner is left out: a move-classification
    badge is saturated colour drawn over the square that stays until the next move, and the
    cell under it must still read. An arrowhead lands on the middle of its square and a mark
    rings the whole square, so both are seen without the corner; only the first inch of an
    arrow's tail leaving a square up and to the right is lost, and the piece it leaves from
    is genuinely still readable.
    """
    size = inner.shape[0]
    blind = _OVERLAY_BLIND if size == CELL else _overlay_blind(size)
    candidates = (distance > OVERLAY_DISTANCE) & (chroma_map > OVERLAY_CHROMA) & blind
    if not candidates.any():
        return 0.0
    pixels = inner[candidates].astype(np.float32)
    square = background.astype(np.float32)
    segments = [
        (square, np.zeros(3, dtype=np.float32)),
        (square, np.full(3, 255.0, dtype=np.float32)),
    ]
    for fill, edge in colours:
        segments.append((square, fill.astype(np.float32)))
        if edge is not None:
            segments.append((square, edge.astype(np.float32)))
            segments.append((fill.astype(np.float32), edge.astype(np.float32)))
    off = np.min([_off_line(pixels, start, end) for start, end in segments], axis=0)
    return float(np.count_nonzero(off > OVERLAY_OFF_LINE)) / candidates.size


def silhouette(patch: np.ndarray, size: int = CELL, colours: PieceColours = ()) -> Silhouette:
    """Silhouette mask, foreground fraction, colour statistics, centroid, hover, background.
    `colours` are the piece set's, for telling a drawing from a piece's edge."""
    background = ring_median_color(patch)
    border = int(round(patch.shape[0] * INNER))
    inner = patch[border : patch.shape[0] - border, border : patch.shape[1] - border]
    inner = cv2.resize(inner, (size, size), interpolation=cv2.INTER_AREA)
    distance = np.linalg.norm(inner.astype(np.float32) - background.astype(np.float32), axis=2)
    mask = _fill_enclosed(distance > BG_TOLERANCE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)
    fraction = float(mask.mean())
    chroma_map = inner.max(axis=2).astype(np.float32) - inner.min(axis=2).astype(np.float32)
    overlay = _overlay_fraction(inner, background, distance, chroma_map, colours)
    luminance = 0.0
    brightness = 0.0
    centroid = (0.5, 0.5)
    tone = mask.astype(bool)
    core_colour: np.ndarray | None = None
    edge_colour: np.ndarray | None = None
    if fraction > 0:
        # Colour statistics come from the core of the piece, not its outline: white pieces
        # carry dark outlines and black pieces light ones, and thin pieces are mostly outline.
        sample_mask = _core_mask(mask)
        # Not the badge corner: a classification badge's disc lies inside the mask, and its
        # bright ink swung a black queen's median luminance to a white one's.
        blinded = sample_mask & _BADGE_BLIND
        if blinded.any():
            sample_mask = blinded
        boundary = mask.astype(bool) & ~cv2.erode(mask, _KERNEL).astype(bool)
        core = _fill_colour(inner[sample_mask].astype(np.float32), inner[boundary])
        core_colour = core
        tone = _tone_mask(mask.astype(bool), inner, core, background)
        # The fill's luminance, 0 to 1. The share of core pixels over mid-grey was the
        # measure before: with anti-aliased outlines the core of a black piece keeps a ring
        # of its light outline on a dark square and none on a light one (0.17 against 0.03
        # at 48-pixel cells), and that square-dependent noise ate the margin between a pawn
        # and a bishop. A median does not move for a ring.
        luminance = float(0.114 * core[0] + 0.587 * core[1] + 0.299 * core[2])
        brightness = luminance / 255.0
        centroid = _centroid(mask)
        outline = mask.astype(bool) & ~tone
        if int(outline.sum()) >= EDGE_MIN_PIXELS:
            edge_colour = np.median(inner[outline], axis=0)
    # Chess.com's hover outline is about 5% of the cell wide and sits at its edge, so the
    # ring the background is read from stays clean and serves as the reference. The corner
    # blocks are kept as well, for telling a tint (which reaches them) from an outline or
    # an edge (which does not).
    hover = _hover_outline(patch, background)
    return Silhouette(
        mask.astype(bool),
        tone,
        fraction,
        luminance,
        brightness,
        background,
        centroid,
        hover,
        inner,
        corner_pixels(patch),
        overlay,
        core_colour,
        edge_colour,
    )


COVER_TOLERANCE = 12.0  # two cells under one opaque overlay share a background
TINT_MIN_ALPHA = 0.2  # a translucent overlay this faint already spoils the pieces under it
TINT_ALPHA_SPREAD = 0.15  # the alpha read from each colour channel must agree
TINT_MIN_CELLS = 6  # highlights and marks tint a few cells; a fading dialog tints a block
TINT_BLOCK_FILL = 0.8  # of the block's bounding box; a dialog is a filled rectangle
THEME_MIN_TINTED = 32  # one square colour re-themed tints half the board; no highlight does
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


def _has_block(cells: np.ndarray) -> bool:
    """Whether the marked cells hold a 2x2 block, as a dialog does and a strip does not."""
    return bool((cells[:-1, :-1] & cells[1:, :-1] & cells[:-1, 1:] & cells[1:, 1:]).any())


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The mask moved right by `dx` and down by `dy` pixels, zero-filled at the edges."""
    if dx == 0 and dy == 0:
        return mask
    h, w = mask.shape
    out = np.zeros_like(mask)
    if abs(dx) >= w or abs(dy) >= h:
        return out
    out[max(0, dy) : h + min(0, dy), max(0, dx) : w + min(0, dx)] = mask[
        max(0, -dy) : h + min(0, -dy), max(0, -dx) : w + min(0, -dx)
    ]
    return out


def iou(a: np.ndarray, b: np.ndarray, keep: np.ndarray = _BADGE_BLIND) -> float:
    a, b = a & keep, b & keep
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


@dataclass
class ThemeModel:
    exemplars: list[Exemplar] = field(default_factory=list)
    light_bg: np.ndarray | None = None
    dark_bg: np.ndarray | None = None
    # Rank-digit exemplars (value -> glyph) learned from the start position, where the
    # orientation is known; used to read the orientation cold from any board of this theme.
    rank_labels: dict[int, np.ndarray] = field(default_factory=dict)
    # The set's colours, a (fill, outline) pair for each side, learned from the start
    # position: what a saturated patch over a square may be before it is called a drawing.
    piece_colours: PieceColours = ()
    # The last frame's cells, by (row, col): the crop's pixels, its silhouette, and the
    # reading made of it, with the covered flag and cell size it was made with. Between
    # one frame and the next most cells do not change a pixel, and a cell that did not
    # change reads the same: its silhouette and reading are reused, most of a frame's cost.
    _cells: dict[tuple[int, int], tuple[np.ndarray, Silhouette, bool, float, CellReading]] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def symbols(self) -> set[str]:
        return {e.symbol for e in self.exemplars}

    def highlight_kind(self, background: np.ndarray, row: int, col: int) -> str:
        """The hue of a cell's background shift: "none", "move", "red", or "foreign".

        A translucent tint blended over the square keeps its hue whatever the square
        colour, so the hue is read from the direction of the shift rather than from the
        blended colour itself. A yellow-ish tint (chess.com's last move and selection)
        never raises blue and never lowers green much; a red one (a right-click mark, a
        check, a premove queue) raises red while lowering green; anything else is foreign
        to those two hues. This only names the tint. Whether the cell is readable at all
        is decided by `tint_kind`, which compares the ring with the cell's corners, so a
        green, blue or grey highlight on another site or theme reads fine as "other".
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

    def tint_kind(self, sil: Silhouette, row: int, col: int) -> tuple[TintKind, str | None]:
        """Whether what shifted a cell's background is a readable tint, and its hue.

        A tint blends one colour over the whole square, so the ring just inside the border
        and the deeper corner blocks move away from the square colour in the same direction
        (chess.com's radial check tint is stronger in the middle, still the same direction)
        and no corner pixel is left at the square colour. A hover outline thick enough to
        fill the ring, a capture ring, or the edge of something drawn over the square shifts
        the ring while the corners stay clean, at least in part, or shift another way: that
        is foreign, and the cell cannot be read. Corner pixels that are neither clean nor
        shifted like the ring belong to a piece, a badge or a label and are not counted, so
        a big piece on a highlighted square is still a highlighted square.
        """
        expected = self.light_bg if is_light(row, col) else self.dark_bg
        if expected is None:
            return "none", None
        expected = expected.astype(np.float32)
        shallow = sil.background.astype(np.float32) - expected
        shallow_norm = float(np.linalg.norm(shallow))
        if shallow_norm <= HIGHLIGHT_TOLERANCE:
            return "none", None
        shifts = sil.corner_pixels - expected
        magnitudes = np.linalg.norm(shifts, axis=1)
        clean = magnitudes <= TINT_MIN_CORNER_SHIFT
        cosines = (shifts @ shallow) / (np.maximum(magnitudes, 1e-6) * shallow_norm)
        shifted = ~clean & (cosines >= TINT_MIN_COSINE)
        known = int(clean.sum()) + int(shifted.sum())
        if known < CORNER_MIN_KNOWN * magnitudes.size:
            return "foreign", None  # the corners show something else again
        if int(clean.sum()) >= CORNER_CLEAN_MAX * known:
            return "foreign", None  # the shift stops before the corners
        hues = {"move": "yellow", "red": "red"}
        return "tint", hues.get(self.highlight_kind(sil.background, row, col), "other")

    def _classify(
        self, sil: Silhouette, row: int, col: int, covered: bool, cell_px: float | None = None
    ) -> CellReading:
        kind_of_tint, tint = self.tint_kind(sil, row, col)
        # Red is the colour of marks, checks and premove queues, never of a move, and the
        # reconciler reads it from `tint`; any other hue is a highlight.
        highlighted = kind_of_tint == "tint" and tint != "red"
        if covered:
            # An opaque overlay confirmed by the neighbours hides this cell: whatever the
            # silhouette shows is not the board, no evidence either way, and never a hover,
            # however the overlay's edges cross the cell.
            return _reading(sil, "unknown", covered=True)
        if kind_of_tint == "foreign":
            # Something that is not a tint sits on the square: a hover outline thick enough
            # to fill the ring, a capture ring, an edge. The silhouette is not the board,
            # and a commit must wait until it is gone.
            return _reading(sil, "unknown", hover=True)
        if sil.overlay > OVERLAY_MAX:
            # An arrow or a mark drawn over the square: saturated colour a white or black
            # piece never has. Whatever the silhouette says now is not the position -- an
            # arrowhead over a black pawn reads as a white piece -- so the cell waits.
            return _reading(sil, "unknown", drawn=True)
        if sil.fraction <= EMPTY_MAX_FRACTION:
            return _reading(
                sil,
                "empty",
                confidence=1.0,
                highlighted=highlighted,
                settled=True,
                hover=sil.hover,
                tint=tint,
            )
        if sil.fraction < PIECE_MIN_FRACTION or not self.exemplars:
            return _reading(sil, "unknown", highlighted=highlighted, hover=sil.hover, tint=tint)

        # Overlap is measured with the mask moved onto the exemplar's centroid as well as in
        # place, whichever fits better, so a lock a pixel or two off, or a piece drawn
        # slightly off its square, still matches by shape and a centroid pulled off by
        # something small on the square costs nothing; where the piece sits is judged by
        # `offset` on its own. Both shapes are tried and the better overlap counts: the
        # tone mask tells a piece from another whatever the square did to its outline,
        # the full mask, fatter, forgives a lock a pixel off on a small cell.
        align = _align_centroid(sil.tone)
        size = sil.tone.shape[0]
        best_by_symbol: dict[str, tuple[float, float, Exemplar]] = {}
        for exemplar in self.exemplars:
            overlap = max(iou(sil.tone, exemplar.tone), iou(sil.mask, exemplar.mask))
            dx = int(round((exemplar.align_centroid[0] - align[0]) * size))
            dy = int(round((exemplar.align_centroid[1] - align[1]) * size))
            if dx or dy:
                overlap = max(
                    overlap,
                    iou(shift_mask(sil.tone, dx, dy), exemplar.tone),
                    iou(shift_mask(sil.mask, dx, dy), exemplar.mask),
                )
            score = overlap - BRIGHTNESS_WEIGHT * abs(sil.brightness - exemplar.brightness)
            previous = best_by_symbol.get(exemplar.symbol)
            if previous is None or score > previous[0]:
                best_by_symbol[exemplar.symbol] = (score, overlap, exemplar)
        ranked = sorted(best_by_symbol.items(), key=lambda item: item[1][0], reverse=True)
        symbol, (score, overlap, exemplar) = ranked[0]
        runner_up = ranked[1][1][0] if len(ranked) > 1 else -1.0
        # Where the piece sits, judged outside the badge corner like the alignment is: a
        # classification badge on a small cell adds enough ink to pull the whole mask's
        # centroid a tenth of a cell toward the corner, and the piece under it then never
        # settled, so a move onto a badged square could not commit until the badge left.
        landing = _align_centroid(sil.mask)
        offset = float(
            np.hypot(
                landing[0] - exemplar.landing_centroid[0],
                landing[1] - exemplar.landing_centroid[1],
            )
        )
        settled = offset <= settle_tolerance(cell_px) and not sil.hover
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
                tint=tint,
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
            settled=False,  # a piece of no known kind is no evidence of a landed piece
            hover=sil.hover,
            offset=offset,
            guess=guess,
            tint=tint,
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
        crops = [[geometry.crop(frame, r, c, size=2 * CELL) for c in range(8)] for r in range(8)]
        sils: list[list[Silhouette]] = []
        for r in range(8):
            row: list[Silhouette] = []
            for c in range(8):
                last = self._cells.get((r, c))
                if last is not None and np.array_equal(last[0], crops[r][c]):
                    row.append(last[1])
                else:
                    row.append(silhouette(crops[r][c], colours=self.piece_colours))
            sils.append(row)
        covered = self.covered_cells(sils)
        cell_px = geometry.cell_px
        readings: list[list[CellReading]] = []
        for r in range(8):
            row_readings: list[CellReading] = []
            for c in range(8):
                last = self._cells.get((r, c))
                is_covered = bool(covered[r, c])
                if (
                    last is not None
                    and last[1] is sils[r][c]
                    and last[2] == is_covered
                    and last[3] == cell_px
                    and np.array_equal(last[0], crops[r][c])
                ):
                    row_readings.append(last[4])
                    continue
                reading = self._classify(sils[r][c], r, c, is_covered, cell_px)
                self._cells[(r, c)] = (crops[r][c], sils[r][c], is_covered, cell_px, reading)
                row_readings.append(reading)
            readings.append(row_readings)
        tinted = sum(1 for row in readings for reading in row if reading.tint is not None)
        if tinted >= THEME_MIN_TINTED:
            # Highlights, marks and queues tint a few squares; a tint over half the board
            # or more is a theme this recogniser was not built on (one or both square
            # colours changed), and the silhouettes it reads on it are not evidence. The
            # pipeline rebuilds the recogniser from the next start position it sees.
            for r in range(8):
                for c in range(8):
                    reading = readings[r][c]
                    if reading.tint is not None:
                        readings[r][c] = replace(
                            reading,
                            kind="unknown",
                            symbol=None,
                            colour=None,
                            confidence=0.0,
                            highlighted=False,
                            settled=False,
                            guess=None,
                            tint=None,
                        )
        if int(covered.sum()) < TINT_MIN_CELLS or not _has_block(covered):
            # A strip of covered cells is a tooltip over one rank or file, not a dialog:
            # the user may well be dragging next to it, and that drag must stay visible.
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
        foreground, centred, no hover outline. The eight pawns are fused by majority vote,
        which one odd pawn cannot sway. A pair (knights, bishops, rooks) has no majority:
        its two silhouettes must agree, on overlap and on centroid, to be fused (their
        union); when they do not, the member closer to the other colour's fused pair of
        the same type is kept on its own, since every piece set draws both colours from
        one silhouette, and when that pair disagrees as well the frame is refused.
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
        fused: dict[str, Exemplar] = {}
        disagreeing: list[str] = []
        for symbol, sils in samples.items():
            if len(sils) == 2 and not _pair_agrees(sils[0], sils[1]):
                disagreeing.append(symbol)
                continue
            fused[symbol] = _fuse(symbol, sils)
        for symbol in disagreeing:
            reference = fused.get(symbol.swapcase())
            if reference is None:
                return None  # both colours' pairs are spoiled; the next start frame gets a chance
            reference_tone = reference.tone
            best = max(samples[symbol], key=lambda sil: iou(sil.tone, reference_tone))
            fused[symbol] = _fuse(symbol, [best])
        model = cls()
        model.piece_colours = _piece_colours(samples)
        for symbol in samples:
            exemplar = fused[symbol]
            if float(exemplar.tone.mean()) < EXEMPLAR_MIN_TONE:
                return None
            model.exemplars.append(exemplar)
        model.light_bg = np.median(np.array(light_bgs), axis=0)
        model.dark_bg = np.median(np.array(dark_bgs), axis=0)
        gap = float(np.linalg.norm(model.light_bg - model.dark_bg))
        if gap <= COVER_TOLERANCE:
            return None  # the two square colours cannot be told apart
        return model


EXEMPLAR_MAX_FRACTION = 0.9
# A fused exemplar (a tone mask: the fill without the outline) must keep at least this much
# of the inner patch; a queen's star is 0.16 of it on the 64-pixel synthetic set.
EXEMPLAR_MIN_TONE = 0.08
# A piece is drawn on the middle of its file; how high it sits in the square is its own
# business (a pawn's centre of mass lies well below the centre, a king's near it).
EXEMPLAR_MAX_SIDEWAYS = 0.12  # of a cell
EXEMPLAR_MAX_VERTICAL = 0.28  # of a cell
# Two cells of one piece kind on a clean start frame overlap at 0.89 or better and sit
# within 0.043 of a cell of each other on the recorded games (rooks, whose cells carry the
# coordinate labels and a box a pixel narrower, are the worst); a white strip 18% of the
# cell tall over one of them moves its centroid by 0.06 or more and 0.075 at 24%.
PAIR_MIN_IOU = 0.8
PAIR_MAX_CENTROID = 0.06  # of a cell


def _piece_colours(samples: dict[str, list[Silhouette]]) -> PieceColours:
    """Each side's fill and outline: the medians over its sixteen pieces."""
    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    for white in (True, False):
        sils = [s for symbol, ss in samples.items() for s in ss if symbol.isupper() == white]
        fills = [s.core_colour for s in sils if s.core_colour is not None]
        edges = [s.edge_colour for s in sils if s.edge_colour is not None]
        if not fills:
            continue
        fill = np.median(np.array(fills), axis=0)
        edge = np.median(np.array(edges), axis=0) if len(edges) >= len(sils) // 2 else None
        out.append((fill, edge))
    return tuple(out)


def _plausible_exemplar(sil: Silhouette) -> bool:
    """A piece drawn on its own square: enough foreground, centred, nothing over it."""
    if sil.hover or not PIECE_MIN_FRACTION <= sil.fraction <= EXEMPLAR_MAX_FRACTION:
        return False
    return (
        abs(sil.centroid[0] - 0.5) <= EXEMPLAR_MAX_SIDEWAYS
        and abs(sil.centroid[1] - 0.5) <= EXEMPLAR_MAX_VERTICAL
    )


def _pair_agrees(a: Silhouette, b: Silhouette) -> bool:
    """Whether two cells of one piece kind show the same silhouette in the same place."""
    distance = float(np.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1]))
    overlap = max(iou(a.tone, b.tone), iou(a.mask, b.mask))
    return overlap >= PAIR_MIN_IOU and distance <= PAIR_MAX_CENTROID


def _fuse(symbol: str, sils: list[Silhouette]) -> Exemplar:
    """One exemplar from the silhouettes of the cells holding this piece kind: the pixels
    at least half of them share (the union of a pair), the median brightness and centroid."""
    mask = np.mean([sil.mask for sil in sils], axis=0) >= 0.5
    tone = np.mean([sil.tone for sil in sils], axis=0) >= 0.5
    brightness = float(np.median([sil.brightness for sil in sils]))
    centroid = (
        float(np.median([sil.centroid[0] for sil in sils])),
        float(np.median([sil.centroid[1] for sil in sils])),
    )
    return Exemplar(
        symbol, mask, tone, brightness, centroid, _align_centroid(tone), _align_centroid(mask)
    )


def occupancy(frame: np.ndarray, geometry: BoardGeometry) -> np.ndarray:
    """8x8 foreground fractions, the cheap occupancy signal."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            grid[r, c] = silhouette(geometry.crop(frame, r, c, size=2 * CELL)).fraction
    return grid


def _start_shaped(occupied: np.ndarray) -> bool:
    """Rows 0, 1, 6, 7 occupied and rows 2 to 5 empty, allowing one stray cell.

    One move away from the start already differs in two cells, so the allowance is a single
    cell for a badge, a hover outline, or a cursor.
    """
    expected = np.zeros((8, 8), dtype=bool)
    expected[[0, 1, 6, 7], :] = True
    return int((occupied != expected).sum()) <= 1


def looks_like_start_position(frame: np.ndarray, geometry: BoardGeometry) -> bool:
    """Whether the frame's occupancy has the shape of the start position."""
    return _start_shaped(occupancy(frame, geometry) >= PIECE_MIN_FRACTION)


def start_shaped(readings: list[list[CellReading]]) -> bool:
    """`looks_like_start_position` from readings already taken, by the same rule, so a
    frame that has been classified need not be silhouetted a second time."""
    occupied = np.array(
        [[reading.fg_fraction >= PIECE_MIN_FRACTION for reading in row] for row in readings],
        dtype=bool,
    )
    return _start_shaped(occupied)


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
