"""Per-theme piece recognition bootstrapped from the start position.

Every game starts from a known position, which hands us labelled examples of all twelve
piece kinds and both square colours on whatever theme is on screen. A cell is read in two
parts.

What is on the square, from its silhouette (`silhouette`): the square's colour is the
median of a ring just inside the cell's border; foreground is every pixel far from it,
enclosed pockets filled, a small opening applied so labels and anti-aliasing do not count.
How much of the cell that covers tells an empty square (EMPTY_MAX_FRACTION) from a mark
(below PIECE_MIN_FRACTION) from a piece. What sits over the square is told apart by
geometry, never by colour:

- A tint (a last-move or selection highlight, the red of a check, a premove queue or a
  right-click mark, in whatever hue a site or theme uses) shifts the cell's ring and the
  deeper corner blocks the same way, leaving no corner pixel at the square's own colour.
  The cell stays readable, with `highlighted` set and `tint` naming the hue.
- A foreign cell has its ring shifted but corners that are clean, at least in part, or
  shifted another way: a hover outline thick enough to fill the ring, a capture ring, the
  edge of something drawn over the square. It reads as unknown with `hover` set, so
  nothing commits under it.
- A cover is an opaque or translucent overlay confirmed by the neighbours: two adjacent
  cells that share one background away from their square colours, or a block of six or
  more cells that all fit one translucent tint (a dialog fading in), with the dialog's
  body -- cells off their colour that fill a rectangle with the block -- and the cells its
  edges reach into. They read as covered: unknown, never a hover, and the reconciler may
  keep their last stable reading for a while. Next to a dialog-sized cover, cells that read
  as junk are covered too, since nothing is dragged under a dialog.
- A tint over half the board or more is no highlight but a theme this recogniser was not
  built on; those cells read as unknown until the pipeline rebuilds it from a start
  position.
- A drawing -- an arrow, the edge of a mark -- is saturated colour off every line between
  the square's colour, black, white and the set's own colours (its palette, learned at the
  start). A cell with more than a sliver of it reads as unknown with `drawn` set until the
  drawing is gone: an arrowhead over a piece can look like another piece.

What the piece is, from a frozen pretrained net (`embedding`): the start position's 64
cells embedded once, a cell named by the nearest of them -- no threshold on the foreground,
so a shaded set on a board of its own tone and thin detail at small cells read like any
other. Every cell is brought to one resolution first (`canonical`), so a board resized
after its set was learned reads the same. The side is voted by three signals
(`_vote_side`): the embedding, the fill's brightness against each side's pieces, and the
fill's own colour under the kind's exemplar mask. Whether the piece has landed is the
phase correlation of the cell against that piece's start-position crop on a square of the
same colour (`_phase_shift`): a piece dragged or in flight is off its centre, and no move
commits on a cell that has not settled.

This is theme-specific by construction and needs no training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Literal

import chess
import cv2
import numpy as np

from sidecoach.events import Side
from sidecoach.vision.embedding import (
    MIN_MARGIN as EMBED_MIN_MARGIN,
)
from sidecoach.vision.embedding import (
    MIN_SIMILARITY,
    CellEmbedder,
)
from sidecoach.vision.squares import CELL, BoardGeometry, is_light, square_at

INNER = 0.08
RING_SKIP = 0.05
RING_BAND = 0.07
# The four corner blocks between these insets sample the background deeper than the ring:
# a piece is round-ish and covers at most one or two of them partially, a badge one. Deeper
# is not an option: on the recorded games rooks and bishops reach blocks at 12% and beyond.
CORNER_INSET = (0.10, 0.20)
BG_TOLERANCE = 25.0
# Once the set's fills are known, a piece whose fill has little contrast against its
# square is read again with a tolerance of half that contrast, within these bounds: a
# dark shaded set on a dark board, a greyish white on a pale one, had half of every
# piece within 25 of the square and read as two shapes. Only that piece's own fill
# counts: a lower tolerance for every piece let a label's anti-aliasing join a rook.
BG_TOLERANCE_MIN = 8.0
BG_CONTRAST_SHARE = 0.5
POCKET_TOLERANCE = 8.0
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
# A piece's side is read from its brightness first: when it lies within one side's
# exemplars' range of brightness, widened by this much, and not the other's, the shape
# is matched among that side's pieces alone. A brightness in both ranges or neither (a
# badge over the piece, a set whose sides are close, a queen whose fill read as dark as
# the other side's) matches every piece with a brightness penalty, as before; a
# pixel-art king, whose fill read a tenth brighter on the dark square than on the light
# one, was outscored there by the queen's shape on the penalty.
SIDE_MARGIN = 0.05
_KERNEL = np.ones((3, 3), dtype=np.uint8)
# The colour of a piece is its fill's. A silhouette's ink (its pixels that are not the
# square's colour; the pockets the outline encloses are not ink) is two tones, the fill
# and the outline with the detail drawn in it, and the fill is the tone that forms the
# widest region of its own: the tone whose deepest pixel lies farthest from any other
# tone, so a line however long weighs nothing against a body. Once the set's two fills
# are known
# (learned from the start position, each side's pooled over its sixteen pieces) the
# tones are the pixels nearer one fill or the other; before that, Otsu splits them. A
# fixed erosion was three pixels, and a bold set's outline is wider; the deepest tenth
# of the silhouette was tried next, and on a real queen the deepest band of the body is
# where the set draws its lines; then the weighted mode over the whole silhouette, and a
# real queen's crown, open to the square between its spikes, weighed black on a dark
# square and white on a light one.

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


def mark_on_empty(reading: CellReading) -> bool:
    """A little drawn on an empty square and nothing more: a legal-move dot, a label -- less
    than the smallest piece covers, no arrow over it, no drag outline on it, nothing hiding
    it. It is no evidence while pieces move (a dot is read as unknown for that: see
    `_classify`), but a board read on the player's word counts it as the empty square it
    is: a piece left selected put dots on five squares, and "Re-read the screen" failed
    every time (Tai's desktop, 2026-09-24, dump 20260924-165745)."""
    return (
        reading.kind == "unknown"
        and not reading.drawn
        and not reading.hover
        and not reading.covered
        and reading.fg_fraction < PIECE_MIN_FRACTION
    )


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
    return CellReading(
        kind=kind,
        symbol=symbol,
        colour=colour,
        confidence=confidence,
        fg_fraction=sil.fraction,
        luminance=sil.luminance,
        highlighted=highlighted,
        background=(
            float(sil.background[0]),
            float(sil.background[1]),
            float(sil.background[2]),
        ),
        brightness=sil.brightness,
        settled=settled,
        hover=hover,
        offset=offset,
        guess=guess,
        covered=covered,
        tint=tint,
        drawn=drawn,
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
    hover: bool
    # The inner patch the mask was taken from, and the pixels of the four corner blocks,
    # deeper than the ring: a tint reaches them, an outline does not.
    inner: np.ndarray
    corner_pixels: np.ndarray
    # The share of inner pixels that are saturated colour far from the square's own colour
    # and the set's (an arrow drawn over the square).
    overlay: float = 0.0
    # The colour of the piece's fill and of the rest of its silhouette (its outline),
    # when there is one: the set's colours, learned at the start.
    core_colour: np.ndarray | None = None
    edge_colour: np.ndarray | None = None
    # The two tones of the ink, each as (colour, weight), the fill first.
    tones: tuple[tuple[np.ndarray, float], ...] = ()
    # How far the square's own pixels stray from its colour (a wood grain's texture).
    spread: float = 0.0


@dataclass(frozen=True)
class Exemplar:
    """A piece kind as the start position showed it: its tone mask (see `Silhouette.tone`),
    fused over the piece's cells, for the fill vote (`_fill_side`), and the median
    brightness of its fill, for the brightness vote (`_side_of`)."""

    symbol: str
    tone: np.ndarray
    brightness: float


# The embedder sees every cell at this resolution, whatever the board's size: the crop brought
# down to CANONICAL_PX a side and back (canonical). A board drawn larger carries more detail,
# and the embeddings of a board resized away from the one its set was learned at drifted
# (bishops read as knights after a board was enlarged half again: the parity audit's sweep,
# 2026-09-26); at one resolution every size looks alike. 40 px, from the scale benchmark
# (44 sets, learned at one of 32-128 px and read at another): identity errors 161 -> 13, where
# 24 px lost the detail that tells some sets' pieces apart (a pirouetti queen read as a
# bishop). Landing is not brought down to it: a piece in flight read as landed (see
# LANDING_SAME_SIZE for how it meets a resize).
CANONICAL_PX = 40
SETTLED_TOLERANCE = 0.08  # of a cell; a dragged piece follows the cursor, far off centre
SETTLED_FLOOR_PX = 3.0  # source pixels: the lock itself is only good to about a pixel
# The embedding votes for a piece's side only when its nearest white and nearest black piece
# lie at least this far apart (`_vote_side`).
COLOUR_TIE = 0.05
# The fill read under the exemplar's mask (`_fill_side`): the mask eroded this many
# pixels so a piece a little off its place still shows its fill under it, at least this
# many pixels of it, and the nearer side's fill this many times nearer than the other's.
FILL_SIDE_ERODE = 2
FILL_SIDE_MIN_PIXELS = 30
FILL_SIDE_RATIO = 1.5
# Where a piece sits: the phase correlation of the cell against the named piece's
# start-position crop, which measures the shift itself (at 0.03, 0.06, 0.1 and 0.2 of a
# cell off it read 0.031, 0.062, 0.098 and 0.199 over eight sets, three sizes, both square
# colours, highlighted or not, and 0.033 at rest with a pixel of lock jitter; unmoved by a
# badge, the check glow, a hint ring, an arrow, a tint or a label). After a resize the cell
# and its reference are each a cell of another size blown up to the working size, and the
# correlation's peak was lost in the detail one has and the other has not (a rook at rest,
# learned at 40 px and read at 34, measured 0.15 of a cell off and never landed: the
# geometry sweep, 2026-09-27). So a board read at a size more than LANDING_SAME_SIZE from
# the one its set was learned at is also compared brought down to the coarser of the two,
# and the surer correlation (the higher peak) gives the shift: resting pieces read as not
# landed 486 -> 102 of 79096, pieces in flight read as landed 38 -> 26 of 15840 (the scale
# benchmark, 44 sets).
LANDING_SAME_SIZE = 0.02  # of the cell: a lock's pitch is good to a fraction of a pixel
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
# The portion at each end of a side where an outline's step is read: an outline runs
# through the corners, a piece's base and a label do not.
HOVER_CORNER = 0.2  # of the cell
# The change between one row and the next that makes a line: a thin outline upscaled
# from a small cell spreads over two or three rows and still steps this much; a check's
# glow ramps a few levels a row.
HOVER_STEP = 10.0


def settle_tolerance(cell_px: float | None) -> float:
    """The landing tolerance as a share of the cell: SETTLED_TOLERANCE, but never less than
    a few source pixels, so a small cell does not turn lock jitter into a piece in flight."""
    if not cell_px:
        return SETTLED_TOLERANCE
    return max(SETTLED_TOLERANCE, SETTLED_FLOOR_PX / cell_px)


def _badge_blind(size: int) -> np.ndarray:
    """Pixels that take part in silhouette comparison: everything but the badge corner."""
    keep = np.ones((size, size), dtype=bool)
    keep[: int(size * 0.38), int(size * 0.62) :] = False
    return keep


_BADGE_BLIND = _badge_blind(CELL)


def _ring_median_and_spread(
    patch: np.ndarray, skip: float = RING_SKIP, band: float = RING_BAND
) -> tuple[np.ndarray, float]:
    """The square's colour (the ring's median) and how far its own pixels stray from it
    (the median distance): nothing on a flat square, a dozen levels on a wood grain."""
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
    median = np.median(ring, axis=0)
    spread = float(np.median(np.linalg.norm(ring.astype(np.float32) - median, axis=1)))
    return median, spread


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


CORNER_BOX = 0.3  # of the patch: a component within one corner's box is no part of a piece


def _centred(mask: np.ndarray) -> np.ndarray:
    """The connected parts of the mask that are not confined to a corner: a coordinate
    label sits in the top-left or bottom-right corner, a badge in the top-right, and each
    lies wholly within its corner's box; a piece, whatever it is made of, does not. A
    label the closing above made solid enough to survive the opening moved the a-file
    pawns' centroids past the landing tolerance. Keeping only the parts that reached the
    middle of the patch was tried, and dropped the base of a set's bishop, drawn apart
    from its head."""
    n = mask.shape[0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    box = n * CORNER_BOX
    keep = np.ones(count, dtype=bool)
    keep[0] = False
    for k in range(1, count):
        left = int(stats[k, cv2.CC_STAT_LEFT])
        top = int(stats[k, cv2.CC_STAT_TOP])
        right = left + int(stats[k, cv2.CC_STAT_WIDTH])
        bottom = top + int(stats[k, cv2.CC_STAT_HEIGHT])
        in_x = left >= n - box or right <= box
        in_y = top >= n - box or bottom <= box
        if in_x and in_y:
            keep[k] = False
    return np.take(keep, np.asarray(labels, dtype=np.int64)).astype(np.uint8)


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
    colours = (patch, patch[::-1], patch.transpose(1, 0, 2), patch.transpose(1, 0, 2)[::-1])
    strong: list[float] = []
    weak: list[float] = []
    medians: list[float] = []
    steps: list[float] = []
    for side, colour in zip(sides, colours, strict=True):
        rows = side[:band, band : n - band]
        strong.append(float((rows > HOVER_TOLERANCE).mean(axis=1).max()))
        weak.append(float((rows > HOVER_TOLERANCE / 2).mean(axis=1).max()))
        medians.append(float(np.median(side[:edge])))
        # An outline is a step between one row and the next, and it runs through both
        # ends of the side; a tint that fades toward the edge (a check's glow) is a ramp
        # of a few levels a row, and a piece's base or a label reaches one end at most.
        # So the step is read at each end of the side, and the smaller counts.
        reach = max(2, int(round(n * HOVER_CORNER)))
        ends = []
        for lo, hi in ((band, band + reach), (n - band - reach, n - band)):
            lines = [
                np.median(colour[k, lo:hi].astype(np.float32), axis=0)
                for k in range(min(n, 2 * band + 1))
            ]
            ends.append(
                max(float(np.linalg.norm(lines[k + 1] - lines[k])) for k in range(len(lines) - 1))
            )
        steps.append(min(ends))
    strong.sort()
    weak.sort()
    medians.sort()
    steps.sort()
    if steps[1] < HOVER_STEP:
        return False  # no step on three sides: a ramp, not a line
    # Either the outermost edge differs from the background on every side, taken as a
    # whole (an outline thick enough to fill most of it), or one row of the band does so
    # end to end on every side (a thin outline a pixel inside the edge).
    if medians[0] > HOVER_TOLERANCE / 2 and medians[2] > HOVER_TOLERANCE:
        return True
    return weak[0] >= HOVER_LINE_FILL and strong[1] >= HOVER_LINE_FILL


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(
        values[order][min(len(order) - 1, int(np.searchsorted(cumulative, cumulative[-1] / 2)))]
    )


def _fill_colour(
    inner: np.ndarray,
    gray: np.ndarray,
    ink: np.ndarray,
    keep: np.ndarray,
    fills: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, float, tuple[tuple[np.ndarray, float], ...]]:
    """The fill's colour and luminance, and the ink's tones as (colour, weight), the fill
    first (see the note above `Kind`). `keep` is where the weights count (not the badge
    corner); `fills` are the set's, when known."""
    if len(fills) >= 2:
        pixels = inner.astype(np.float32)
        to = [np.linalg.norm(pixels - fill.astype(np.float32), axis=2) for fill in fills]
        nearest = np.argmin(np.stack(to), axis=0)
        classes = [ink & (nearest == i) for i in range(len(fills))]
    else:
        values = gray[ink].astype(np.uint8).reshape(-1, 1)
        split, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        classes = [ink & (gray > split), ink & (gray <= split)]
    limit = float(ink.shape[0])
    tones: list[tuple[np.ndarray, float, float]] = []
    for tone in classes:
        if not tone.any():
            continue
        # Measured from the patch's edge as well, or a piece reaching the edge (a bold
        # cross's arm) has a tone that runs off it and no edge to be shallow against.
        padded = cv2.copyMakeBorder(tone.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        depth = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        depth = np.minimum(depth, limit) * keep
        weights = (depth * depth)[tone]
        if float(weights.sum()) <= 0:
            weights = np.ones_like(weights)
        pixels = inner[tone].astype(np.float32)
        colour = np.array([_weighted_median(pixels[:, ch], weights) for ch in range(3)], np.float32)
        # A tone's weight is the depth of its deepest pixel: the widest region of one
        # tone is the fill. Summed depths favoured the outline, which runs round every
        # region of fill and is long, on a queen whose fill is cut into bands by its
        # lines.
        tones.append((colour, float(depth.max()), _weighted_median(gray[tone], weights)))
    tones.sort(key=lambda t: t[1], reverse=True)
    colour, _, luminance = tones[0]
    return colour, luminance, tuple((c, w) for c, w, _ in tones)


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
    palette: tuple[np.ndarray, ...],
) -> float:
    """The share of inner pixels that are a drawing: chromatic, far from the square's colour,
    and off the lines from that colour to black, to white and to each colour of the set's
    palette, and off the lines between any two of those, where piece edges and details
    blend.

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
    # The lines a piece's pixels may lie on: from the square to each colour of the
    # palette, and between any two of them -- a set draws one side's detail in the other
    # side's colour (a pink emblem on a navy piece), and the blend of those two is no
    # drawing.
    colours = [np.zeros(3, dtype=np.float32), np.full(3, 255.0, dtype=np.float32)]
    colours += [colour.astype(np.float32) for colour in palette]
    segments = [(square, colour) for colour in colours]
    segments += [(a, b) for i, a in enumerate(colours) for b in colours[i + 1 :]]
    off = np.min([_off_line(pixels, start, end) for start, end in segments], axis=0)
    return float(np.count_nonzero(off > OVERLAY_OFF_LINE)) / candidates.size


def _mask_and_fill(
    inner: np.ndarray,
    gray: np.ndarray,
    distance: np.ndarray,
    tolerance: float,
    pocket: float,
    fills: tuple[np.ndarray, ...],
) -> tuple[
    np.ndarray, float, tuple[np.ndarray, float, tuple[tuple[np.ndarray, float], ...]] | None
]:
    """The silhouette at `tolerance` and, unless it is empty, its fill's colour, luminance
    and tones (see `_fill_colour`).

    A piece whose fill is the square's own colour (a cream set on a light board) is only
    its outline, so what the outline encloses is filled in; the outline is closed first,
    since a thin stroke breaks at its anti-aliased joins, and a broken outline enclosed
    nothing: a wooden knight read as its bare outline on light squares.
    """
    foreground = cv2.morphologyEx((distance > tolerance).astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL)
    mask = _fill_enclosed(foreground.astype(bool))
    mask = _centred(cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL))
    fraction = float(mask.mean())
    if fraction <= 0:
        return mask, fraction, None
    # The fill's colour (see the note above `Kind`). Not the badge corner: a
    # classification badge's disc lies inside the mask, and its bright ink swung a
    # black queen to white. The ink: the silhouette less the pockets where the square
    # itself shows through (between a queen's crown spikes), to the square's own spread
    # (a wood grain). A pocket a shade off the square is fill: a cream set on a cream
    # board differs from it by a dozen levels, under what makes foreground, and its fill
    # is the pocket its outline encloses.
    on = mask.astype(bool)
    keep = _BADGE_BLIND if (on & _BADGE_BLIND).any() else np.ones_like(_BADGE_BLIND)
    ink = on & (distance > pocket)
    if not ink.any():
        ink = on
    return mask, fraction, _fill_colour(inner, gray, ink, keep.astype(np.float32), fills)


def silhouette(
    patch: np.ndarray,
    size: int = CELL,
    colours: PieceColours = (),
    palette: tuple[np.ndarray, ...] = (),
) -> Silhouette:
    """Silhouette mask, foreground fraction, colour statistics, centroid, hover, background.
    `colours` are the set's fills and outlines, for the tones; `palette` every colour the
    set draws with, for telling a drawing from a piece's detail."""
    background, spread = _ring_median_and_spread(patch)
    border = int(round(patch.shape[0] * INNER))
    inner = patch[border : patch.shape[0] - border, border : patch.shape[1] - border]
    inner = cv2.resize(inner, (size, size), interpolation=cv2.INTER_AREA)
    distance = np.linalg.norm(inner.astype(np.float32) - background.astype(np.float32), axis=2)
    chroma_map = inner.max(axis=2).astype(np.float32) - inner.min(axis=2).astype(np.float32)
    overlay = _overlay_fraction(inner, background, distance, chroma_map, palette)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fills = tuple(fill for fill, _ in colours)
    pocket = min(BG_TOLERANCE, max(POCKET_TOLERANCE, 3.0 * spread))
    mask, fraction, stats = _mask_and_fill(inner, gray, distance, BG_TOLERANCE, pocket, fills)
    if stats is not None and fills:
        # The piece's own fill against this square: with little contrast, read again with
        # a tolerance to match (see BG_TOLERANCE_MIN).
        core = stats[0]
        nearest = min(fills, key=lambda f: float(np.linalg.norm(core - f.astype(np.float32))))
        contrast = float(np.linalg.norm(nearest.astype(np.float32) - background.astype(np.float32)))
        tolerance = max(BG_TOLERANCE_MIN, BG_CONTRAST_SHARE * contrast)
        if tolerance < BG_TOLERANCE:
            mask, fraction, stats = _mask_and_fill(inner, gray, distance, tolerance, pocket, fills)
    luminance = 0.0
    brightness = 0.0
    tone = mask.astype(bool)
    core_colour: np.ndarray | None = None
    edge_colour: np.ndarray | None = None
    tones: tuple[tuple[np.ndarray, float], ...] = ()
    if stats is not None:
        core, luminance, tones = stats
        core_colour = core
        tone = _tone_mask(mask.astype(bool), inner, core, background)
        brightness = luminance / 255.0
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
        hover,
        inner,
        corner_pixels(patch),
        overlay,
        core_colour,
        edge_colour,
        tones,
        spread,
    )


COVER_TOLERANCE = 12.0  # two cells under one opaque overlay share a background
TINT_MIN_ALPHA = 0.2  # a translucent overlay this faint already spoils the pieces under it
TINT_ALPHA_SPREAD = 0.15  # the alpha read from each colour channel must agree
TINT_MIN_CELLS = 6  # highlights and marks tint a few cells; a fading dialog tints a block
TINT_BLOCK_FILL = 0.8  # of the block's bounding box; a dialog is a filled rectangle
THEME_MIN_TINTED = 32  # one square colour re-themed tints half the board; no highlight does
OVERLAY_SIDE_FILL = 0.9  # an overlay edge fills a whole side of the inner patch
OVERLAY_BLOCK_FILL = 0.85  # an overlay corner is a solid rectangle
# A cell this far off its square colour, joined to a covered block, is the dialog's body
# when the block and such cells fill a rectangle this well: the dialog's fringe fades.
COVER_BODY_SHIFT = 20.0
COVER_BODY_FILL = 0.8


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


def _blocks(
    cells: np.ndarray, joined: Callable[[int, int, int, int], bool], fill: float
) -> list[list[tuple[int, int]]]:
    """The groups of marked cells, neighbours joined when `joined(r, c, rr, cc)`, that are
    shaped like a dialog: at least TINT_MIN_CELLS, two cells or more each way, filling
    `fill` of their bounding box. A file of right-click marks or a queued premove's squares
    make a line, and a highlight or two a small group; neither is a dialog."""
    parent = list(range(64))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(8):
        for c in range(8):
            if not cells[r, c]:
                continue
            for rr, cc in ((r, c + 1), (r + 1, c)):
                if rr < 8 and cc < 8 and cells[rr, cc] and joined(r, c, rr, cc):
                    parent[find(r * 8 + c)] = find(rr * 8 + cc)
    groups: dict[int, list[tuple[int, int]]] = {}
    for i in range(64):
        if cells[i // 8, i % 8]:
            groups.setdefault(find(i), []).append((i // 8, i % 8))
    out = []
    for members in groups.values():
        rows = [r for r, _ in members]
        cols = [c for _, c in members]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        if (
            len(members) >= TINT_MIN_CELLS
            and height >= 2
            and width >= 2
            and len(members) >= fill * height * width
        ):
            out.append(members)
    return out


def translucent_cover(
    backgrounds: np.ndarray, expected: np.ndarray, shifted: np.ndarray
) -> np.ndarray:
    """Cells in a dialog-shaped block (`_blocks`) that share one translucent overlay.

    A dialog fades in over a few frames; while it is half transparent the pieces under
    it are unreadable but the light and dark squares still differ, so the opaque test
    does not fire and the cells would read as junk. A block of neighbours all consistent
    with one tint is that fading dialog. A move highlight or a selection tints one to
    three cells and a right-click mark a few, and stay below the block size.
    """

    def tinted(r: int, c: int, rr: int, cc: int) -> bool:
        return _same_tint(backgrounds[r, c], backgrounds[rr, cc], expected[r, c], expected[rr, cc])

    covered = np.zeros((8, 8), dtype=bool)
    for block in _blocks(shifted, tinted, TINT_BLOCK_FILL):
        for r, c in block:
            covered[r, c] = True
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


def cover_body(covered: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """The covered cells plus the body of each dialog: cells off their square colour that
    connect (4-neighbours) to a covered block and, with it, make a dialog-shaped block
    (`_blocks`). A highlight or two beside a tooltip does not fill a rectangle with it; a
    fading dialog's buttons and text do."""
    candidate = covered | (shifts > COVER_BODY_SHIFT)
    out = covered.copy()
    for block in _blocks(candidate, lambda r, c, rr, cc: True, COVER_BODY_FILL):
        if any(covered[r, c] for r, c in block):
            for r, c in block:
                out[r, c] = True
    return out


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
    # Every colour the set draws with (its ink clustered at the start), the palette a
    # drawing over a square is told from.
    palette: tuple[np.ndarray, ...] = ()
    # The last frame's cells, by (row, col): the crop's pixels, its silhouette, and the
    # reading made of it, with the covered flag and cell size it was made with. Between
    # one frame and the next most cells do not change a pixel, and a cell that did not
    # change reads the same: its silhouette and reading are reused, most of a frame's cost.
    _cells: dict[tuple[int, int], tuple[np.ndarray, Silhouette, bool, float, CellReading]] = field(
        default_factory=dict, repr=False, compare=False
    )
    # The start position's 64 cells as the frozen net embeds them (unit vectors, float16
    # for the cache) and what each held: a piece's symbol, or "empty". The exemplars of
    # the net (see `embedding`).
    exemplar_vectors: np.ndarray | None = None
    exemplar_labels: tuple[str, ...] = ()
    # Each piece kind's start-position crop in grey on each square colour, keyed by
    # (symbol, on a light square): the reference the phase landing measures a cell's shift
    # against (an instance off the labelled edge cells), and the cell size it was taken at.
    landing_crops: dict[tuple[str, bool], np.ndarray] = field(default_factory=dict)
    landing_px: float = 0.0

    def __getstate__(self) -> dict[str, object]:
        # The last frame's cells are this session's, not the theme's: pickled, they made
        # a 60 KB theme file 2.8 MB, and a cached theme would have begun its next game
        # believing it had read a frame.
        state = dict(self.__dict__)
        state["_cells"] = {}
        return state

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
        self,
        sil: Silhouette,
        row: int,
        col: int,
        covered: bool,
        cell_px: float | None = None,
        vector: np.ndarray | None = None,
        crop: np.ndarray | None = None,
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
        if vector is not None and sil.fraction >= PIECE_MIN_FRACTION:
            # The nearest of the start position's cells names what is drawn; the tint,
            # the outline, the cover and the drawing were the rules' above, and so are
            # an empty square and a square with a little on it: nothing is nothing,
            # whatever tint it wears, and a hint dot is no evidence (the embedding read
            # the dotted squares of a drag in progress as empty with such confidence
            # that a resync adopted the rook where the cursor held it).
            return self._from_embedding(
                sil, vector, highlighted, tint, cell_px, crop, is_light(row, col)
            )
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
        # A mark (a hint dot, a label's end) or, with no net to name it, a piece: no
        # evidence either way.
        return _reading(sil, "unknown", highlighted=highlighted, hover=sil.hover, tint=tint)

    def _from_embedding(
        self,
        sil: Silhouette,
        vector: np.ndarray,
        highlighted: bool,
        tint: str | None,
        cell_px: float | None,
        crop: np.ndarray | None = None,
        light: bool = True,
    ) -> CellReading:
        """The cell named by the nearest of the start position's cells in embedding space
        (see `embedding`). The side is voted (`_vote_side`); once it is, the kind may be
        named by either side's cells (`_kind_scores`, see CANONICAL_PX). Where the piece
        sits is the phase correlation against the piece's own start crop on a square of the
        cell's colour (`light`)."""
        assert self.exemplar_vectors is not None
        sims = self.exemplar_vectors.astype(np.float32) @ vector
        side = self._vote_side(sil, sims)
        scored = self._kind_scores(sims) if side is not None else sims
        allowed = [
            i
            for i, label in enumerate(self.exemplar_labels)
            if label == "empty" or side is None or label.isupper() == side.color
        ]
        order = sorted(allowed, key=lambda i: -float(scored[i]))
        best = order[0]
        label = self.exemplar_labels[best]
        similarity = float(scored[best])
        other = next(
            (float(scored[i]) for i in order[1:] if self.exemplar_labels[i] != label), -1.0
        )
        margin = similarity - other
        if similarity < MIN_SIMILARITY:
            # Nothing on the start position looks like this, yet a piece's worth of
            # foreground is there: a king under the check glow, a piece under a label or
            # an arrow. Occupied, by something of no sure kind, as the overlap read such
            # cells -- the reconciler's decorated-king rule wants the king's landing
            # square read as occupied, and a mate whose king had just moved never
            # committed while the square read as nothing at all.
            return _reading(
                sil,
                "piece",
                colour=side,
                confidence=0.5,
                highlighted=highlighted,
                settled=False,
                hover=sil.hover,
                guess=label if label != "empty" else None,
                tint=tint,
            )
        if label == "empty":
            if margin < EMBED_MIN_MARGIN:
                return _reading(sil, "unknown", highlighted=highlighted, hover=sil.hover, tint=tint)
            return _reading(
                sil,
                "empty",
                confidence=margin,
                highlighted=highlighted,
                settled=True,
                hover=sil.hover,
                tint=tint,
            )
        colour = Side.of(label.isupper())
        offset = 0.0
        reference = self.landing_crops.get((label, light))
        if crop is not None and reference is not None:
            offset = _phase_shift(reference, crop, self.landing_px, float(cell_px or 0.0))
        settled = offset <= settle_tolerance(cell_px) and not sil.hover
        runner = self.exemplar_labels[
            next(i for i in order[1:] if self.exemplar_labels[i] != label)
        ]
        if margin < EMBED_MIN_MARGIN and runner == "empty":
            # As near an empty square as a piece: something on the square that is
            # neither, a hint, a label, a mark's edge. No evidence, and no drag either:
            # a piece of no known kind would hold every move until it left. A near tie
            # between two pieces is not held against the nearer: a pixel-art set's
            # pieces all lie close, and a wrong kind is one stale cell the reconciler
            # carries and corrects.
            return _reading(
                sil, "unknown", highlighted=highlighted, hover=sil.hover, guess=label, tint=tint
            )
        return _reading(
            sil,
            "piece",
            symbol=label,
            colour=colour,
            confidence=margin,
            highlighted=highlighted,
            settled=settled,
            hover=sil.hover,
            offset=offset,
            tint=tint,
        )

    def _vote_side(self, sil: Silhouette, sims: np.ndarray) -> Side | None:
        """The piece's side by a vote of three signals, each cast only when sure: the
        embedding (its nearest white and nearest black piece COLOUR_TIE apart), the
        brightness of the silhouette's fill (`_side_of`), and the fill's own pixels
        under the named kind's exemplar mask (`_fill_side`). Each is wrong somewhere
        on its own -- the embedding on a flat white queen on a blue highlight, the
        brightness on a white queen of dark detail on a green one, the fill mask on
        sets whose tones the silhouette misreads -- and in every case seen the other
        two were right. A tie, or no sure voter, leaves the embedding to choose."""
        votes: list[Side] = []
        best_white = max(
            (float(sims[i]) for i, lab in enumerate(self.exemplar_labels) if lab.isupper()),
            default=-1.0,
        )
        best_black = max(
            # "empty".islower() is True: the empty squares are no side (the parity
            # audit, 2026-09-26: they voted as black).
            (
                float(sims[i])
                for i, lab in enumerate(self.exemplar_labels)
                if lab != "empty" and lab.islower()
            ),
            default=-1.0,
        )
        if abs(best_white - best_black) >= COLOUR_TIE:
            votes.append(Side.WHITE if best_white > best_black else Side.BLACK)
        brightness = self._side_of(sil)
        if brightness is not None:
            votes.append(brightness)
        nearest = self.exemplar_labels[int(np.argmax(sims))]
        fill = self._fill_side(sil, nearest) if nearest != "empty" else None
        if fill is not None:
            votes.append(fill)
        white = sum(1 for v in votes if v == Side.WHITE)
        black = len(votes) - white
        if white == black:
            return None
        return Side.WHITE if white > black else Side.BLACK

    def _kind_scores(self, sims: np.ndarray) -> np.ndarray:
        """Each cell's similarity raised to the best of its kind's, either side's: what a
        piece is, is its shape, once its side is voted. At CANONICAL_PX the square under a
        piece weighs more in its embedding, and the start position shows each side's king
        and queen on one square colour only: a black queen on a light square came nearer
        the black bishop learned on one than the black queen learned on a dark square, and
        a mated king under the check glow nearer a rook of its own side than its own king
        (stress seeds 469, 584, 766). The other side shows the king and queen on the other
        square colour. Learning them again on the other colour, their crops' squares
        recoloured, named a black queen on g2 (688), and a set never learned read as a
        position with two kings (tests/test_cold_start.py)."""
        scored = sims.copy()
        kinds: dict[str, list[int]] = {}
        for i, label in enumerate(self.exemplar_labels):
            if label != "empty":
                kinds.setdefault(label.lower(), []).append(i)
        for members in kinds.values():
            scored[members] = sims[members].max()
        return scored

    def _fill_side(self, sil: Silhouette, label: str) -> Side | None:
        """The side whose fill the pixels under the named kind's fill show: the median
        colour of the cell's inner patch under the exemplar's fill mask (eroded, the
        mask of the kind in both colours where both were learned), against the two
        sides' fills learned at the start. None when the mask is too small to read
        or the colour lies about as near one fill as the other. A white queen of dark
        detail on a green highlight, and a flat white queen on a blue one, each read
        as black by one of the other two signals; the fill itself is white in both."""
        if len(self.piece_colours) < 2:
            return None
        masks = [e.tone for e in self.exemplars if e.symbol.upper() == label.upper()]
        if not masks:
            return None
        mask = masks[0] if len(masks) == 1 else np.logical_and(masks[0], masks[1])
        core = cv2.erode(mask.astype(np.uint8), _KERNEL, iterations=FILL_SIDE_ERODE).astype(bool)
        if int(core.sum()) < FILL_SIDE_MIN_PIXELS:
            return None
        colour = np.median(sil.inner[core].astype(np.float32), axis=0)
        white = float(np.linalg.norm(colour - self.piece_colours[0][0].astype(np.float32)))
        black = float(np.linalg.norm(colour - self.piece_colours[1][0].astype(np.float32)))
        if min(white, black) * FILL_SIDE_RATIO > max(white, black):
            return None
        return Side.WHITE if white < black else Side.BLACK

    def _side_of(self, sil: Silhouette) -> Side | None:
        """The side whose exemplars' brightness the piece's falls among (see SIDE_MARGIN),
        or None when it falls among both or neither."""
        inside = []
        for side in (Side.WHITE, Side.BLACK):
            values = [e.brightness for e in self.exemplars if e.symbol.isupper() == side.color]
            if values and min(values) - SIDE_MARGIN <= sil.brightness <= max(values) + SIDE_MARGIN:
                inside.append(side)
        return inside[0] if len(inside) == 1 else None

    def covered_cells(self, sils: list[list[Silhouette]]) -> np.ndarray:
        """Cells under an opaque overlay, and the cells its edges reach into."""
        covered = np.zeros((8, 8), dtype=bool)
        if self.light_bg is None or self.dark_bg is None:
            return covered
        backgrounds = np.array([[sil.background for sil in row] for row in sils], dtype=np.float32)
        expected = np.zeros((8, 8, 3), dtype=np.float32)
        shifts = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for c in range(8):
                expected[r, c] = self.light_bg if is_light(r, c) else self.dark_bg
                shifts[r, c] = float(np.linalg.norm(backgrounds[r, c] - expected[r, c]))
        shifted = shifts > HIGHLIGHT_TOLERANCE
        covered = opaque_cover(backgrounds, shifted) | translucent_cover(
            backgrounds, expected, shifted
        )
        covered = cover_body(covered, shifts)
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
                    row.append(
                        silhouette(crops[r][c], colours=self.piece_colours, palette=self.palette)
                    )
            sils.append(row)
        covered = self.covered_cells(sils)
        cell_px = geometry.cell_px
        fresh: list[tuple[int, int]] = []  # the cells to read again
        for r in range(8):
            for c in range(8):
                last = self._cells.get((r, c))
                if not (
                    last is not None
                    and last[1] is sils[r][c]
                    and last[2] == bool(covered[r, c])
                    and last[3] == cell_px
                    and np.array_equal(last[0], crops[r][c])
                ):
                    fresh.append((r, c))
        vectors: dict[tuple[int, int], np.ndarray] = {}
        net = embedder()
        # The net embeds, in one batch, the cells that changed with a piece's worth of ink:
        # only those are named by it (`_classify`). An empty square or a hint dot is the
        # silhouette's to read, and embedding it anyway was most of a full read's cost --
        # a second on the phone for a board that was mostly empty squares.
        inked = [(r, c) for r, c in fresh if sils[r][c].fraction >= PIECE_MIN_FRACTION]
        if net is not None and inked:
            embedded = net.embed([canonical(crops[r][c]) for r, c in inked])
            vectors = dict(zip(inked, embedded, strict=True))
        readings: list[list[CellReading]] = []
        for r in range(8):
            row_readings: list[CellReading] = []
            for c in range(8):
                last = self._cells.get((r, c))
                is_covered = bool(covered[r, c])
                if (r, c) not in fresh and last is not None:
                    row_readings.append(last[4])
                    continue
                reading = self._classify(
                    sils[r][c], r, c, is_covered, cell_px, vectors.get((r, c)), crops[r][c]
                )
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
        """The recogniser for this game, or None when the start frame is not clean enough:
        a cursor's outline over a piece, a piece hidden (the embeddings say so,
        `_start_reads_as_start`), or square colours that cannot be told apart -- or when there
        is no net to embed with (a broken install). The kings and queens have one cell each,
        and what a start frame shows is every piece's reference for the game."""
        start = chess.Board()
        net = embedder()
        if net is None:
            return None
        light_bgs: list[np.ndarray] = []
        dark_bgs: list[np.ndarray] = []
        # Two passes: the first learns the set's colours from every piece, the second
        # reads every piece with those colours, as every later frame will be read.
        crops = [[geometry.crop(frame, r, c, size=2 * CELL) for c in range(8)] for r in range(8)]
        first: dict[str, list[Silhouette]] = {}
        for r in range(8):
            for c in range(8):
                piece = start.piece_at(square_at(r, c, bottom))
                if piece is not None:
                    first.setdefault(piece.symbol(), []).append(silhouette(crops[r][c]))
        colours = _piece_colours(first)
        palette = _set_palette(first)
        samples: dict[str, list[Silhouette]] = {}
        for r in range(8):
            for c in range(8):
                sil = silhouette(crops[r][c], colours=colours, palette=palette)
                piece = start.piece_at(square_at(r, c, bottom))
                if piece is None:
                    (light_bgs if is_light(r, c) else dark_bgs).append(sil.background)
                    continue
                if sil.hover:
                    return None  # the cursor's outline would be in the piece's reference
                samples.setdefault(piece.symbol(), []).append(sil)
        model = cls()
        model.piece_colours = colours
        model.palette = palette
        model.exemplars = [_fuse(symbol, sils) for symbol, sils in samples.items()]
        model.light_bg = np.median(np.array(light_bgs), axis=0)
        model.dark_bg = np.median(np.array(dark_bgs), axis=0)
        gap = float(np.linalg.norm(model.light_bg - model.dark_bg))
        if gap <= COVER_TOLERANCE:
            return None  # the two square colours cannot be told apart
        # The phase landing's references: a crop per piece kind and square colour, from a
        # cell off the labelled left column and bottom row where it can be. Against a crop on
        # the other colour a knight at rest read 0.138 of a cell off, against one on its own
        # 0.010 (the stress sweep's seed 138: the flat set on green, 2026-09-27). A king and a
        # queen start on one colour: that crop stands for both.
        for r in range(8):
            for c in range(8):
                piece = start.piece_at(square_at(r, c, bottom))
                if piece is None:
                    continue
                key = (piece.symbol(), is_light(r, c))
                if key in model.landing_crops and (c == 0 or r == 7):
                    continue
                model.landing_crops[key] = cv2.cvtColor(crops[r][c], cv2.COLOR_BGR2GRAY)
        for (symbol, light), crop in list(model.landing_crops.items()):
            model.landing_crops.setdefault((symbol, not light), crop)
        model.landing_px = float(geometry.cell_px)
        cells = [crops[r][c] for r in range(8) for c in range(8)]
        vectors = net.embed([canonical(cell) for cell in cells])
        labels = tuple(
            piece.symbol() if (piece := start.piece_at(square_at(r, c, bottom))) else "empty"
            for r in range(8)
            for c in range(8)
        )
        if not _start_reads_as_start(vectors, labels):
            return None  # a piece hidden or spoiled: the next start frame gets a chance
        model.exemplar_vectors = vectors.astype(np.float16)
        model.exemplar_labels = labels
        return model


def _piece_colours(samples: dict[str, list[Silhouette]]) -> PieceColours:
    """Each side's fill and outline, pooled over its sixteen pieces: every piece's ink is
    two tones, and pooled by luminance the heavier cluster is the fill. Pooled, because a
    piece whose fill is the square's own colour shows only its outline there, and read
    on its own would name the outline its fill."""
    out: list[tuple[np.ndarray, np.ndarray | None]] = []
    for white in (True, False):
        sils = [s for symbol, ss in samples.items() for s in ss if symbol.isupper() == white]
        tones = [tone for sil in sils for tone in sil.tones]
        if not tones:
            continue
        lum = np.array([0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2] for c, _ in tones])
        weight = np.array([w for _, w in tones])
        if len(tones) >= 2 and lum.max() - lum.min() > 1:
            split = float(_weighted_median(lum, weight))
            groups = [lum > split, lum <= split]
            if not all(g.any() for g in groups):
                groups = [lum >= split, lum < split]
        else:
            groups = [np.ones(len(tones), dtype=bool), np.zeros(len(tones), dtype=bool)]
        groups.sort(key=lambda g: float(weight[g].sum()) if g.any() else -1.0, reverse=True)
        fill_group, edge_group = groups[0], groups[1]
        colours = np.array([c for c, _ in tones], dtype=np.float32)
        fill = _pooled_colour(colours, weight, fill_group)
        edge = _pooled_colour(colours, weight, edge_group) if edge_group.any() else None
        out.append((fill, edge))
    return tuple(out)


# Clusters of the set's ink, and the share of the ink a cluster must hold to be a colour
# of the set: a queen's crown is one piece of thirty-two, a gold highlight a sliver of
# every gold piece, and both were drawings at eight clusters and one percent.
PALETTE_COLOURS = 16
PALETTE_MIN_SHARE = 0.002
PALETTE_SAMPLE = 30000  # ink pixels clustered, at most


def _set_palette(samples: dict[str, list[Silhouette]]) -> tuple[np.ndarray, ...]:
    """Every colour the set draws with: the ink of all its pieces on the start frame,
    clustered. A set's fills and outlines are two colours a side; its details may be
    more (a pink emblem and red lettering on a navy piece, a gold crown on a teal king),
    and each is a colour a drawing is told from."""
    ink = [sil.inner[sil.mask].reshape(-1, 3) for sils in samples.values() for sil in sils]
    if not ink:
        return ()
    pixels = np.concatenate(ink).astype(np.float32)
    if len(pixels) > PALETTE_SAMPLE:
        pixels = pixels[np.linspace(0, len(pixels) - 1, PALETTE_SAMPLE).astype(int)]
    k = min(PALETTE_COLOURS, len(pixels))
    if k < 1:
        return ()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    labels = np.zeros((len(pixels), 1), dtype=np.int32)
    _, labels, centres = cv2.kmeans(pixels, k, labels, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel().astype(np.int64), minlength=k)
    keep = counts >= PALETTE_MIN_SHARE * len(pixels)
    return tuple(centre.astype(np.float32) for centre in centres[keep])


def _pooled_colour(colours: np.ndarray, weight: np.ndarray, group: np.ndarray) -> np.ndarray:
    picked, weights = colours[group], weight[group]
    return np.array([_weighted_median(picked[:, ch], weights) for ch in range(3)], np.float32)


def _fuse(symbol: str, sils: list[Silhouette]) -> Exemplar:
    """One exemplar from the silhouettes of the cells holding this piece kind: the tone
    pixels at least half of them share (the union of a pair), the median brightness."""
    tone = np.mean([sil.tone for sil in sils], axis=0) >= 0.5
    return Exemplar(symbol, tone, float(np.median([sil.brightness for sil in sils])))


def occupancy(frame: np.ndarray, geometry: BoardGeometry) -> np.ndarray:
    """8x8 foreground fractions, the cheap occupancy signal."""
    grid = np.zeros((8, 8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            grid[r, c] = silhouette(geometry.crop(frame, r, c, size=2 * CELL)).fraction
    return grid


def canonical(crop: np.ndarray) -> np.ndarray:
    """A cell's crop as the embedder sees it: brought down to CANONICAL_PX a side and back,
    so a board of any size shows the embedder the same detail. Only the embedding goes
    through it; silhouettes, outlines, tints and landing keep the full-detail crop."""
    size = crop.shape[0]
    small = cv2.resize(crop, (CANONICAL_PX, CANONICAL_PX), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (size, size), interpolation=cv2.INTER_LINEAR)


def _start_shaped(occupied: np.ndarray) -> bool:
    """Rows 0, 1, 6, 7 occupied and rows 2 to 5 empty, allowing one stray cell.

    One move away from the start already differs in two cells, so the allowance is a single
    cell for a badge, a hover outline, or a cursor.
    """
    expected = np.zeros((8, 8), dtype=bool)
    expected[[0, 1, 6, 7], :] = True
    return int((occupied != expected).sum()) <= 1


def _phase_shift(
    reference: np.ndarray, crop: np.ndarray, reference_px: float = 0.0, crop_px: float = 0.0
) -> float:
    """How far the piece in `crop` (BGR) sits from where it sits in `reference` (grey),
    as a share of the cell, by phase correlation under a Hanning window. When the two
    were taken at cell sizes more than LANDING_SAME_SIZE apart, they are also compared
    brought down to the coarser of the two sizes, and the surer of the two correlations
    (the higher peak) gives the shift."""
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if grey.shape != reference.shape:
        grey = cv2.resize(grey, (reference.shape[1], reference.shape[0]))
    tries = [(reference, grey)]
    if reference_px > 0 and crop_px > 0:
        coarser, finer = sorted((reference_px, crop_px))
        size = int(round(coarser))
        if finer > coarser * (1.0 + LANDING_SAME_SIZE) and 8 <= size < reference.shape[0]:
            tries.append(
                (
                    cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA),
                    cv2.resize(grey, (size, size), interpolation=cv2.INTER_AREA),
                )
            )
    shift, surest = 0.0, -np.inf
    for ref, cell in tries:
        window = cv2.createHanningWindow((ref.shape[1], ref.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(
            ref.astype(np.float32), cell.astype(np.float32), window
        )
        if response > surest:
            shift, surest = float(np.hypot(dx, dy)) / ref.shape[0], response
    return shift


def _start_reads_as_start(vectors: np.ndarray, labels: tuple[str, ...]) -> bool:
    """Whether every piece cell of a start frame shows its piece, by the embeddings: a
    cell hidden under a cursor or a badge, or erased, is nearer an empty square than any
    piece of its own colour, and would be that piece's exemplar for the whole game."""
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -1.0)
    for i, label in enumerate(labels):
        if label == "empty":
            continue
        own = [
            j
            for j, other in enumerate(labels)
            if other != "empty" and other.isupper() == label.isupper()
        ]
        empties = [j for j, other in enumerate(labels) if other == "empty"]
        if not own or not empties:
            continue
        if float(sims[i, empties].max()) > float(sims[i, own].max()):
            return False
    return True


def embedder() -> CellEmbedder | None:
    """The embedding net, or None when its model file is missing (a broken install)."""
    return CellEmbedder.load()


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
