"""Recognise the panel's own board on screen, so the vision never mistakes it for the game.

The panel draws a board too, and if it shares a monitor with the game the detector -- which
prefers the biggest, cleanest checkerboard -- would happily lock onto it. Screen coordinates
are a fragile way to exclude it (window chrome, DPI scaling, zoom, a moved window). A visual
signature is not: the panel paints its squares in two shades three units apart in blue,
alternating by row -- invisible to a person, and a pattern no real board has. A candidate
whose light squares AND dark squares both show that row-parity step is the panel's, and the
tracker drops it. A global colour shift (Night Light, a colour profile) moves every square
the same way and leaves the step intact.

The shades are set in the panel's stylesheet (`.sq.light`, `.sq.light.odd`, ...); this
module must agree with them.
"""

from __future__ import annotations

import numpy as np

from sidecoach.events import Region

# The odd display rows are this much bluer (BGR index 0) than the even ones.
SHADE_STEP = 3
_TOLERANCE = 1.0  # per channel, around the expected step


def _cell_colours(frame: np.ndarray, region: Region, cell_px: float) -> np.ndarray:
    """The median BGR colour of a clean patch of every cell: 8x8x3.

    The patch is the upper-right quarter of the cell, away from the rank labels (top-left)
    and the file labels (bottom-right), and off-centre enough that most pieces miss it.
    """
    out = np.zeros((8, 8, 3), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            y0 = region.y + r * cell_px
            x0 = region.x + c * cell_px
            patch = frame[
                int(y0 + 0.15 * cell_px) : int(y0 + 0.35 * cell_px),
                int(x0 + 0.65 * cell_px) : int(x0 + 0.85 * cell_px),
            ]
            if patch.size == 0:
                return np.full((8, 8, 3), np.nan, dtype=np.float32)
            out[r, c] = np.median(patch.reshape(-1, 3), axis=0)
    return out


def is_own_board(frame: np.ndarray, region: Region, cell_px: float) -> bool:
    """Whether the checkerboard at `region` carries the panel's row-parity signature."""
    if frame.ndim != 3 or frame.shape[2] < 3:
        return False
    if region.x < 0 or region.y < 0:
        return False
    if region.x + 8 * cell_px > frame.shape[1] or region.y + 8 * cell_px > frame.shape[0]:
        return False
    colours = _cell_colours(frame, region, cell_px)
    if np.isnan(colours).any():
        return False
    rows = np.arange(8)[:, None]
    cols = np.arange(8)[None, :]
    light = (rows + cols) % 2 == 1  # a1 is dark; display row 7 col 0 is a1 for White
    for mask in (light, ~light):
        even = colours[mask & (rows % 2 == 0)]
        odd = colours[mask & (rows % 2 == 1)]
        if len(even) < 4 or len(odd) < 4:
            return False
        step = np.median(odd, axis=0) - np.median(even, axis=0)
        expected = np.array([SHADE_STEP, 0.0, 0.0])
        if np.abs(step - expected).max() > _TOLERANCE:
            return False
    return True
