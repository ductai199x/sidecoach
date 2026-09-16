"""Arrows and marks drawn on the board are not moves.

From the first live trial: an orange chess.com arrow from d4 to e5 (with a red mark on e5)
read as the pawn capturing -- the arrowhead over the black pawn made a bright blob that
classified as a white piece -- and the reconciler committed a phantom 23.dxe5, then could
not reconcile the real board for the rest of the game. A square with saturated colour drawn
over it is now unreadable until the drawing is gone, and no move commits through it.
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import MovePlayed
from sidecoach.replay.recording import Recording
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Readings
from sidecoach.vision.pieces import OVERLAY_MAX, CellReading, silhouette
from sidecoach.vision.squares import BoardGeometry, square_at

REC = Path("data/recordings/20260916-000902")  # the trial game; local, not committed
ARROW_FRAMES = (640.0, 662.0)  # the d4->e5 arrow is on the board
CLEAN_FRAME = 590.0  # after 22...e5, before any arrow was drawn


def _frames_at(rec: Recording, wanted: list[float]) -> dict[float, np.ndarray]:
    ticks = [
        json.loads(line) for line in (REC / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    t0 = ticks[0]["ts"]
    got: dict[float, np.ndarray] = {}
    for tick, frame in rec.frames():
        t = tick.ts - t0
        for w in wanted:
            if w not in got and t >= w and tick.stored is not None:
                got[w] = frame.copy()
        if len(got) == len(wanted):
            break
    return got


def test_an_orange_arrow_is_a_saturated_overlay() -> None:
    """The measure itself: an orange stroke over a wooden square, with or without a piece."""
    cell = np.full((96, 96, 3), (99, 136, 181), dtype=np.uint8)  # chess.com brown, BGR
    assert silhouette(cell).overlay == 0.0
    with_piece = cell.copy()
    with_piece[24:72, 30:66] = (30, 30, 30)  # a black piece: no chroma
    assert silhouette(with_piece).overlay < OVERLAY_MAX
    arrowed = with_piece.copy()
    arrowed[40:56, 8:88] = (36, 179, 252)  # the arrow shaft, blended orange
    assert silhouette(arrowed).overlay > OVERLAY_MAX


def test_a_piece_edge_on_a_highlight_is_not_a_drawing() -> None:
    """On the green theme a black piece on the yellow last-move square has a smoothed edge
    of chromatic pixels far from the square's colour; it is the piece, not an arrow, and
    treating it as one hid every piece that landed on a highlight (g01 stalled at ply 16)."""
    cell = np.full((96, 96, 3), (67, 202, 185), dtype=np.uint8)  # highlighted green, BGR
    piece = cell.copy()
    cv2.circle(piece, (48, 48), 26, (30, 30, 30), -1, lineType=cv2.LINE_AA)
    # A wide anti-aliased edge: every blend between the piece and the square.
    for k in range(1, 7):
        alpha = k / 7
        colour = tuple(int(alpha * 30 + (1 - alpha) * v) for v in (67, 202, 185))
        cv2.circle(piece, (48, 48), 26 + k, colour, 1, lineType=cv2.LINE_AA)
    assert silhouette(piece).overlay < OVERLAY_MAX
    arrowed = piece.copy()
    arrowed[40:56, 8:88] = (36, 179, 252)
    assert silhouette(arrowed).overlay > OVERLAY_MAX


def test_a_badge_on_the_corner_is_not_a_drawing() -> None:
    """A move-classification badge stays on the square until the next move; the cell under
    it must still read, so the badge's corner is not measured."""
    cell = np.full((96, 96, 3), (181, 217, 240), dtype=np.uint8)  # light brown, BGR
    cv2.circle(cell, (95, 0), 20, (60, 60, 230), -1)  # a red badge on the top-right corner
    assert silhouette(cell).overlay < OVERLAY_MAX


@pytest.mark.skipif(not (REC / "index.jsonl").exists(), reason="the trial recording is local")
def test_cells_under_the_arrow_are_unreadable_and_the_move_is_not_committed() -> None:
    rec = Recording(REC)
    # Learn the theme from the game's own start, as the live engine did.
    pipe = GamePipeline()
    ticks = [
        json.loads(line) for line in (REC / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    t0 = ticks[0]["ts"]
    frames_iter = rec.frames()
    for tick, frame in frames_iter:
        pipe.process(frame, tick.ts, changed=tick.stored is not None)
        if pipe.model is not None and tick.ts - t0 > 30:
            break
    assert pipe.model is not None and pipe.board is not None and pipe.bottom is not None
    assert pipe.model.achromatic_pieces, "chess.com's pieces are white and black"
    grabbed = _frames_at(Recording(REC), [CLEAN_FRAME, *ARROW_FRAMES])
    # The window was moved during the game; let the tracker re-lock on the later frame
    # (it backs off for a few frames after a miss) so the geometry is that frame's.
    for _ in range(40):
        pipe.tracker.update(grabbed[CLEAN_FRAME])
    assert pipe.tracker.board is not None and pipe.tracker.verified
    geometry = BoardGeometry(pipe.tracker.board.region, pipe.tracker.board.cell_px)

    bottom = pipe.bottom

    def cell(readings: Readings, name: str) -> CellReading:
        target = chess.parse_square(name)
        for r in range(8):
            for c in range(8):
                if square_at(r, c, bottom) == target:
                    return readings[r][c]
        raise AssertionError(name)

    clean = pipe.model.classify_frame(grabbed[CLEAN_FRAME], geometry)
    assert cell(clean, "d4").symbol == "P" and cell(clean, "e5").symbol == "p"
    for t in ARROW_FRAMES:
        readings = pipe.model.classify_frame(grabbed[t], geometry)
        # The arrowhead over the black pawn on e5 is what read as a white piece.
        head = cell(readings, "e5")
        assert head.kind == "unknown" and head.hover, f"e5 at t={t} read as {head.kind}"
        # The tail leaves d4 by its top-right corner, where a badge would sit; the pawn
        # there is still readable, and read right, or the cell waits. Never anything else.
        tail = cell(readings, "d4")
        assert tail.symbol in (None, "P"), f"d4 at t={t} read as {tail.symbol}"

    # The whole pipeline over the arrow period: no phantom dxe5.
    moves: list[str] = []
    for tick, frame in frames_iter:
        t = tick.ts - t0
        if t < 575:
            continue
        if t > 700:
            break
        for event in pipe.process(frame, tick.ts, changed=tick.stored is not None):
            if isinstance(event, MovePlayed):
                moves.append(event.san)
    assert "dxe5" not in moves, moves
