"""The reading works at any reasonable board size, fractional cells included.

A site lays the board out at whatever width the window allows, so cells are rarely a whole
number of pixels (641 px for eight squares is 80.125 a square) and every cell edge falls at
a different sub-pixel phase. With the crop rounded to the nearest pixel, whether a cell wore
a phantom hover outline depended on that phase: at 641 px it was e4 alone and the second
live game never saw 1.e4; at 333 px four cells; at 555 and 851 px the start position could
not even be read. Each size here renders a board at a large integer cell, scales it so the
cells are fractional, and runs the whole pipeline: lock, theme, start position, a real drag,
and a fourteen-ply game.
"""

from __future__ import annotations

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import GameStarted, MovePlayed
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision

BASE_CELL = 96
SANS = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6"]


def frame_at(size: int, **kw: object) -> np.ndarray:
    """A 2560x1440 frame with the board `size` px wide, cells `size / 8` px, at an odd origin."""
    big = synthetic_board(width=2560, height=1440, x0=300, y0=120, cell=BASE_CELL, **kw)  # type: ignore[arg-type]
    board = big[120 : 120 + 8 * BASE_CELL, 300 : 300 + 8 * BASE_CELL]
    small = cv2.resize(board, (size, size), interpolation=cv2.INTER_AREA)
    out = big.copy()
    out[120 : 120 + 8 * BASE_CELL, 300 : 300 + 8 * BASE_CELL] = (90, 90, 90)
    x0, y0 = 411, 137
    out[y0 : y0 + size, x0 : x0 + size] = small
    return out


@pytest.mark.parametrize("size", [297, 333, 555, 641, 851, 1053])
def test_the_pipeline_follows_a_game_at_this_board_size(size: int) -> None:
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    events = []
    for _ in range(4):
        events += pipe.process(frame_at(size, board=board), ts)
        ts += 0.1
    assert any(isinstance(e, GameStarted) for e in events), "no game from the start position"
    assert pipe.board is not None and pipe.model is not None and pipe.bottom is not None
    assert abs(pipe.board.cell_px - size / 8) < 0.5
    geometry = BoardGeometry(pipe.board.region, pipe.board.cell_px)

    readings = pipe.model.classify_frame(frame_at(size, board=board), geometry)
    assert [(r, c) for r in range(8) for c in range(8) if readings[r][c].hover] == []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, pipe.bottom))
            assert readings[r][c].symbol == (piece.symbol() if piece else None), (r, c)

    # A real drag -- the e2 pawn lifted over e4 under the site's hover outline -- still reads
    # as a hover, so the fix did not blind the detector at small cells.
    dragged = frame_at(size, board=board, dragging=(chess.E2, chess.E4, 0.1, 0.1))
    r, c = row_col(chess.E4, pipe.bottom)
    assert pipe.model.classify_frame(dragged, geometry)[r][c].hover

    got: list[str] = []
    for san in SANS:
        move = board.push_san(san)
        for _ in range(3):
            for event in pipe.process(frame_at(size, board=board, highlight=move), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
            ts += 0.1
    assert got == SANS
