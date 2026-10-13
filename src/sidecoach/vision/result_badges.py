"""The site's result badges: a disc on each king when the game ends.

Chess.com marks the end of a game on the board itself, before and under its dialog: a
green disc with a white crown on the winner's king, a red disc on the loser's (the fallen
king, a flag or a clock, in the loser's colour). The disc
is about a third of a square across and sits just inside the square's top-right corner as
drawn on screen, pushed inward on the board's top row and right column so it never
leaves the board. It stays while the game-over dialog is up. Read at the two kings'
squares, the pair says who won without reading a word of the dialog, whose title names
the bot; what the pipeline already knows -- a mate on the board, a clock at zero -- says
why (state/pipeline.py).

A badge is a disc, not a colour. The winner's green is a cousin of the default theme's
dark squares, a check is a red glow, a highlight is a whole square: so the reader looks
for a disc near the corner -- a rim of one colour with a ring of other colours around
it, and not the colour of the square it sits on -- and only then asks which colour. The
last clause matters: a bulky piece can wall off the corner of its own square so that the
bare corner passes for a disc of the square's colour. The measurements below come from
the live games at 43 and 80 px squares (their crops kept locally under
data/fixtures/real/badges, tools/vision/real_crops.py; drawn for the tests by tests/scenes.py).
"""

from __future__ import annotations

from enum import StrEnum

import chess
import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.squares import BoardGeometry, row_col


class Badge(StrEnum):
    WIN = "win"
    LOSS = "loss"


# The disc's radius in squares, and where its centre may lie relative to the square's
# top-right corner: normally 0.04 in and 0.08 down, up to 0.21 in or down when pushed off
# the board's edge.
RADIUS = 0.17
SEARCH_X = (-0.27, 0.03)
SEARCH_Y = (-0.03, 0.27)
# The search is coarse then fine: every other step over the window, then the steps
# around the best. Rim and ring are sampled at most this many pixels each.
SEARCH_STEP = 0.03
SAMPLES = 240
# Fewer badge-coloured pixels than this share of a square's area near the corner, and
# there is no disc to look for.
PRESENCE = 0.02
# The rim sampled inside the disc (outside the icon at its centre) and the ring outside.
RIM = (0.08, 0.135)
RING = (0.21, 0.28)
# A pixel this close to the rim's median colour, per channel, is the disc's colour. The
# icon takes up to half the rim (the crown reaches almost to the edge); the ring outside
# a disc carries next to none of its colour, and that is the test a uniform square, a
# glow or an arrow's shaft fails.
TOLERANCE = 22
RIM_SHARE = 0.45
RING_SHARE = 0.15
# A disc closer than this, per channel, to the colour of the square it sits on is the
# square (the winner's green is 37 from the default theme's dark green).
DISTINCT = 25


def classes(colours: np.ndarray) -> np.ndarray:
    """For an array of BGR colours, the badge each is the colour of: 1 win, 2 loss, 0 none."""
    c = np.asarray(colours, dtype=np.int16).reshape(-1, 3)
    b, g, r = c[:, 0], c[:, 1], c[:, 2]
    win = (g >= 120) & (g - r >= 25) & (g - b >= 60)
    loss = (r >= 150) & (r - g >= 70) & (r - b >= 70)
    # The draw badge is unseen so far. A grey disc was tried for it and read the rounded
    # end of a white label (`result_of`).
    return np.select([win, loss], [1, 2], default=0)


_BADGES = {1: Badge.WIN, 2: Badge.LOSS}


def classify(colour: np.ndarray) -> Badge | None:
    """The result a disc of this BGR colour stands for, or None for any other colour."""
    return _BADGES.get(int(classes(colour)[0]))


def _offsets(cell: float, inner: float, outer: float) -> np.ndarray:
    """(dy, dx) offsets of the pixels between `inner` and `outer` squares of a centre,
    thinned evenly to at most SAMPLES of them."""
    reach = int(np.ceil(outer * cell))
    ys, xs = np.mgrid[-reach : reach + 1, -reach : reach + 1]
    d = np.hypot(xs, ys)
    keep = (d >= inner * cell) & (d < outer * cell)
    offsets = np.stack([ys[keep], xs[keep]], axis=1)
    if len(offsets) > SAMPLES:
        offsets = offsets[np.linspace(0, len(offsets) - 1, SAMPLES).astype(int)]
    return offsets


def _present(frame: np.ndarray, corner_x: float, corner_y: float, cell: float) -> bool:
    """Whether the window a badge could occupy holds enough badge-coloured pixels to be
    worth a search. Cheap, and it is the whole cost on most frames of a game."""
    reach = RADIUS + RING[0]
    x0 = int(corner_x + (SEARCH_X[0] - reach) * cell)
    x1 = int(np.ceil(corner_x + (SEARCH_X[1] + reach) * cell))
    y0 = int(corner_y + (SEARCH_Y[0] - reach) * cell)
    y1 = int(np.ceil(corner_y + (SEARCH_Y[1] + reach) * cell))
    window = frame[max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
    if window.size == 0:
        return False
    return int((classes(window) != 0).sum()) >= PRESENCE * cell * cell


def _score(
    frame: np.ndarray, centres: np.ndarray, rim: np.ndarray, ring: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each candidate centre: the rim's median colour, the share of the rim within
    tolerance of it, and the share of the ring. All candidates at once."""
    height, width = frame.shape[:2]

    def gather(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.rint(centres[:, None, :] + offsets[None, :, :]).astype(int)  # (K, N, 2)
        inside = (pts[..., 0] >= 0) & (pts[..., 0] < height) & (pts[..., 1] >= 0)
        inside &= pts[..., 1] < width
        ys = np.clip(pts[..., 0], 0, height - 1)
        xs = np.clip(pts[..., 1], 0, width - 1)
        return frame[ys, xs].astype(np.int16), inside  # (K, N, 3), (K, N)

    rim_px, rim_in = gather(rim)
    ring_px, ring_in = gather(ring)
    colour = np.median(rim_px, axis=1)  # (K, 3)
    rim_same = (np.abs(rim_px - colour[:, None, :]).max(axis=2) <= TOLERANCE) & rim_in
    ring_same = (np.abs(ring_px - colour[:, None, :]).max(axis=2) <= TOLERANCE) & ring_in
    rim_share = rim_same.sum(axis=1) / np.maximum(1, rim_in.sum(axis=1))
    ring_share = ring_same.sum(axis=1) / np.maximum(1, ring_in.sum(axis=1))
    # A rim mostly off the frame is no rim, and a rim of no badge's colour is no badge.
    usable = (rim_in.mean(axis=1) >= 0.8) & (classes(colour) != 0)
    return colour, np.where(usable, rim_share, 0.0), ring_share


def square_colour(frame: np.ndarray, corner_x: float, corner_y: float, cell: float) -> np.ndarray:
    """The colour of the square whose top-right corner is given, from small patches at
    its other three corners: what a badge at the fourth must differ from. A highlight
    tints all four alike, a piece seldom reaches a corner, a check's glow has faded there."""
    inset = 0.05 * cell
    size = max(2, int(round(0.08 * cell)))
    x_left, x_right = corner_x - cell + inset, corner_x - inset - size
    y_top, y_bottom = corner_y + inset, corner_y + cell - inset - size
    patches = []
    for x, y in ((x_left, y_top), (x_left, y_bottom), (x_right, y_bottom)):
        xi, yi = int(round(x)), int(round(y))
        patch = frame[max(0, yi) : yi + size, max(0, xi) : xi + size]
        if patch.size:
            patches.append(patch.reshape(-1, 3))
    if not patches:
        return np.zeros(3)
    return np.median(np.concatenate(patches), axis=0)


def find_disc(
    frame: np.ndarray,
    corner_x: float,
    corner_y: float,
    cell: float,
    background: np.ndarray | None = None,
    board: Region | None = None,
) -> tuple[np.ndarray, float, float] | None:
    """A disc near a square's top-right corner: its rim colour and the shares of the rim
    and the ring that carry it, or None when nothing disc-shaped is there.

    The candidate centres of the search window are scored together, coarsely first and
    then finely around the best; the disc is the centre whose rim is most uniform and
    whose ring is least like it. `background` is the square's own colour, which the
    disc must not be. With `board`, a centre whose disc would leave the board is none: the
    site keeps its discs on the board, and off it the page above a top-row king carried a
    red blob that read as a loss (the stress sweep's seed 699, 2026-09-27).
    """
    if cell < 16 or not _present(frame, corner_x, corner_y, cell):
        return None
    step = max(1.0, SEARCH_STEP * cell)
    rim = _offsets(cell, *RIM)
    ring = _offsets(cell, *RING)
    cxs = np.arange(corner_x + SEARCH_X[0] * cell, corner_x + SEARCH_X[1] * cell + 1e-6, 2 * step)
    cys = np.arange(corner_y + SEARCH_Y[0] * cell, corner_y + SEARCH_Y[1] * cell + 1e-6, 2 * step)
    coarse = _on_board(np.array([(cy, cx) for cy in cys for cx in cxs]), cell, board)
    if not len(coarse):
        return None
    colour, rim_share, ring_share = _score(frame, coarse, rim, ring)
    pick = int(np.argmax(rim_share - ring_share))
    if rim_share[pick] == 0.0:
        return None
    around = np.array([(dy, dx) for dy in (-step, 0.0, step) for dx in (-step, 0.0, step)])
    fine = _on_board(coarse[pick] + around, cell, board)
    colour, rim_share, ring_share = _score(frame, fine, rim, ring)
    best = int(np.argmax(rim_share - ring_share))
    if rim_share[best] < RIM_SHARE or ring_share[best] > RING_SHARE:
        return None
    if background is not None and np.abs(colour[best] - background).max() < DISTINCT:
        return None
    return colour[best], float(rim_share[best]), float(ring_share[best])


def _on_board(centres: np.ndarray, cell: float, board: Region | None) -> np.ndarray:
    """The (y, x) centres whose disc lies on `board`, to a pixel; all of them without one."""
    if board is None:
        return centres
    r = RADIUS * cell
    ys, xs = centres[:, 0], centres[:, 1]
    keep = (ys - r >= board.y - 1) & (ys + r <= board.y + board.h + 1)
    keep &= (xs - r >= board.x - 1) & (xs + r <= board.x + board.w + 1)
    return centres[keep]


def badge_at(
    frame: np.ndarray, corner_x: float, corner_y: float, cell: float, board: Region | None = None
) -> Badge | None:
    """The badge on the square whose top-right corner (on screen) is at the given point; with
    `board`, a disc on the board only."""
    background = square_colour(frame, corner_x, corner_y, cell)
    found = find_disc(frame, corner_x, corner_y, cell, background, board)
    if found is None:
        return None
    return classify(found[0])


def badge_on(
    frame: np.ndarray, geometry: BoardGeometry, square: chess.Square, bottom: Side
) -> Badge | None:
    """The badge on a square of a locked board, if the site has stamped one there."""
    row, col = row_col(square, bottom)
    corner_x = geometry.region.x + (col + 1) * geometry.cell_px
    corner_y = geometry.region.y + row * geometry.cell_px
    return badge_at(frame, corner_x, corner_y, geometry.cell_px, geometry.region)


def result_of(white: Badge | None, black: Badge | None) -> str | None:
    """The result a pair of badges on the two kings says, as a PGN result string, or None.

    Only the pair counts: a win on one king with a loss on the other. One badge alone, or
    two that disagree, is nothing -- the rim of a glow, an arrow's head, a mark the site
    draws for some other reason. A draw's badge has never been seen: a draw ends through
    the dialog's cover or the End game menu, which carries the reason.
    """
    if white is Badge.WIN and black is Badge.LOSS:
        return "1-0"
    if white is Badge.LOSS and black is Badge.WIN:
        return "0-1"
    return None
