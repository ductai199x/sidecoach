"""Arrows and badges drawn over the board must not read as pieces.

In the trial game of 2026-09-16 an orange arrow from d4 to e5 put its head over the black
pawn on e5, and the arrowhead's saturated pixels read as a white piece: the recogniser saw
a white pawn on e5 for as long as the arrow was drawn, and the reconciler took it for dxe5
-- a move nobody played. The rule since: on a set whose pieces are white and black, a
cell with saturated colour over it is something drawn, and it waits -- that cell alone:
a drawing is no drag, and the moves elsewhere on the board go on committing under it.

The first three tests are the measure on hand-made patches; the next two render the
arrow on the brown theme the game was on and run the recogniser and the whole pipeline;
the last reads the real frames of that game (board crops under tests/fixtures).
"""

from __future__ import annotations

import json
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import MovePlayed, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Readings
from sidecoach.vision.pieces import OVERLAY_MAX, CellReading, ThemeModel, silhouette
from sidecoach.vision.squares import BoardGeometry, square_at
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision

FIXTURES = Path(__file__).parent / "fixtures"
# After 22...e5 in the trial game: the d4 pawn faces the e5 pawn, the arrow ran d4 -> e5.
POSITION = "r2q1rk1/1b3ppp/p1nbpn2/1p2p3/3P4/1BN1PN2/PP3PPP/R2Q1RK1 w - - 0 23"


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


def _real(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / name))
    assert frame is not None, name
    return frame


def _cell(readings: Readings, name: str, bottom: Side) -> CellReading:
    target = chess.parse_square(name)
    for r in range(8):
        for c in range(8):
            if square_at(r, c, bottom) == target:
                return readings[r][c]
    raise AssertionError(name)


def _assert_arrow_reads(readings: Readings, bottom: Side) -> None:
    # The arrowhead over the black pawn on e5 is what read as a white piece.
    head = _cell(readings, "e5", bottom)
    assert head.kind == "unknown" and head.drawn and not head.hover, (
        f"e5 read as {head.kind} {head.symbol}"
    )
    # The tail leaves d4 by its top-right corner, where a badge would sit; the pawn there
    # is still readable, and read right, or the cell waits. Never anything else.
    tail = _cell(readings, "d4", bottom)
    assert tail.symbol in (None, "P"), f"d4 read as {tail.symbol}"


@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_a_rendered_arrow_over_a_pawn_waits_and_the_badge_does_not(bottom: Side) -> None:
    geometry = BoardGeometry(Region(300, 120, 512, 512), 64.0)
    model = ThemeModel.from_start_position(
        synthetic_board(theme="brown", bottom=bottom), geometry, bottom
    )
    assert model is not None and len(model.piece_colours) == 2
    board = chess.Board(POSITION)
    arrow = chess.Move.from_uci("d4e5")
    # The badge sits on the moved piece's square: the knight that just came to f3.
    for badge in (None, (chess.F3, "best"), (chess.F3, "blunder")):
        frame = synthetic_board(
            board=board, theme="brown", bottom=bottom, arrows=(arrow,), badge=badge
        )
        readings = model.classify_frame(frame, geometry)
        _assert_arrow_reads(readings, bottom)
        if badge is not None:
            assert _cell(readings, "f3", bottom).symbol == "N", "the badge hid the knight"


def test_the_pipeline_commits_no_phantom_capture_under_a_rendered_arrow() -> None:
    pipe = GamePipeline()
    ts = 0.0
    for _ in range(3):
        pipe.process(synthetic_board(theme="brown"), ts)
        ts += 0.1
    assert pipe.reconciler is not None and pipe.model is not None
    board = chess.Board(POSITION)
    rec = pipe.reconciler
    rec.board = board.copy()
    rec._base_fen = POSITION
    moves: list[str] = []
    for _ in range(3):
        moves += [
            e.san
            for e in pipe.process(synthetic_board(board=board, theme="brown"), ts)
            if isinstance(e, MovePlayed)
        ]
        ts += 0.1
    arrow = chess.Move.from_uci("d4e5")
    for _ in range(12):
        frame = synthetic_board(board=board, theme="brown", arrows=(arrow,))
        moves += [e.san for e in pipe.process(frame, ts) if isinstance(e, MovePlayed)]
        ts += 0.1
    assert moves == [], f"phantom moves under the arrow: {moves}"
    # The arrow gone, the game goes on from where it was.
    move = board.push_san("dxe5")
    for _ in range(3):
        frame = synthetic_board(board=board, theme="brown", highlight=move)
        moves += [e.san for e in pipe.process(frame, ts) if isinstance(e, MovePlayed)]
        ts += 0.1
    assert moves == ["dxe5"]


def test_an_arrow_over_a_coloured_set_is_a_drawing_too() -> None:
    """A cream-and-brown set is chromatic everywhere, so the rule that a drawing is
    saturated colour a white or black piece never has did not apply, and a thin red
    arrow over an empty square read as a piece in flight that held the whole game. The
    set's own colours are learned at the start, and what is far from all of them is a
    drawing."""
    geometry = BoardGeometry(Region(300, 120, 512, 512), 64.0)
    start = synthetic_board(theme="wood", style="wooden")
    model = ThemeModel.from_start_position(start, geometry, Side.WHITE)
    assert model is not None and len(model.piece_colours) == 2
    (white_fill, white_edge), (black_fill, black_edge) = model.piece_colours
    assert white_edge is not None and black_edge is not None
    assert white_fill[2] > black_fill[2] and white_edge[2] < black_edge[2]  # BGR
    board = chess.Board(POSITION)
    for arrow, colour in (
        (chess.Move.from_uci("d4e5"), "orange"),
        (chess.Move.from_uci("c3d5"), "red"),
    ):
        frame = synthetic_board(
            board=board,
            theme="wood",
            style="wooden",
            arrows=(arrow,),
            arrow_colour=colour,
            arrow_width=0.12,
        )
        readings = model.classify_frame(frame, geometry)
        head = _cell(readings, chess.square_name(arrow.to_square), Side.WHITE)
        assert head.kind == "unknown" and head.drawn, f"{colour} arrowhead read as {head}"
        # The pieces beside the arrow read as themselves.
        assert _cell(readings, "f3", Side.WHITE).symbol == "N"
        assert _cell(readings, "b7", Side.WHITE).symbol == "b"
    # And without an arrow the set reads clean: no piece is mistaken for a drawing.
    readings = model.classify_frame(
        synthetic_board(board=board, theme="wood", style="wooden"), geometry
    )
    drawn = [
        chess.square_name(square_at(r, c, Side.WHITE))
        for r in range(8)
        for c in range(8)
        if readings[r][c].drawn
    ]
    assert not drawn, drawn


def test_moves_elsewhere_commit_under_a_standing_arrow() -> None:
    """An arrow is a drawing, not a drag: while it stands, the moves on the rest of the
    board commit as usual, and only a move onto or off its squares waits for it to go.
    An arrow that stood across four plies once held all four back, past what the
    missed-move search could bridge."""
    pipe = GamePipeline()
    ts = 0.0
    for _ in range(3):
        pipe.process(synthetic_board(theme="brown"), ts)
        ts += 0.1
    assert pipe.reconciler is not None
    board = chess.Board()
    arrow = chess.Move.from_uci("d2d4")  # standing over d2, d3 and d4 for the whole game
    played: list[str] = []
    for san in ("a3", "h6", "b3", "h5", "Nc3", "Nf6"):
        move = board.push_san(san)
        for _ in range(3):
            frame = synthetic_board(board=board, theme="brown", highlight=move, arrows=(arrow,))
            played += [e.san for e in pipe.process(frame, ts) if isinstance(e, MovePlayed)]
            ts += 0.1
    assert played == ["a3", "h6", "b3", "h5", "Nc3", "Nf6"], played
    # A move that lands on the arrow's squares waits until the arrow is gone, unless
    # the piece is read through the arrow (the embedding matcher reads a pawn under an
    # arrowhead; the overlap could not, and had to wait): either way nothing wrong
    # commits, and once the arrow is gone the move is there.
    move = board.push_san("d4")
    for _ in range(3):
        frame = synthetic_board(board=board, theme="brown", highlight=move, arrows=(arrow,))
        played += [e.san for e in pipe.process(frame, ts) if isinstance(e, MovePlayed)]
        ts += 0.1
    assert played[-1] in ("Nf6", "d4"), played
    for _ in range(3):
        frame = synthetic_board(board=board, theme="brown", highlight=move)
        played += [e.san for e in pipe.process(frame, ts) if isinstance(e, MovePlayed)]
        ts += 0.1
    assert played[-1] == "d4"


def test_the_real_arrow_frames_read_the_same() -> None:
    """The board crops of the trial game itself: the clean frame after 22...e5 and two
    frames with the arrow on the board, the recogniser learned from the game's start."""
    meta = json.loads((FIXTURES / "frames.json").read_text(encoding="utf-8"))
    margin = meta["margin"]
    start = _real("chesscom_start.png")
    pipe = GamePipeline()
    for i in range(4):
        pipe.process(start, 0.1 * i)
    assert pipe.model is not None and pipe.board is not None and pipe.bottom is not None
    (white_fill, _), (black_fill, _) = pipe.model.piece_colours
    assert white_fill.min() > 200 and black_fill.max() < 80, (
        "chess.com's pieces are white and black"
    )
    side = meta["chesscom_start"]["region"][2]
    geometry = BoardGeometry(Region(margin, margin, side, side), side / 8)
    clean = pipe.model.classify_frame(_real("chesscom_arrow_clean.png"), geometry)
    assert _cell(clean, "d4", pipe.bottom).symbol == "P"
    assert _cell(clean, "e5", pipe.bottom).symbol == "p"
    for name in ("chesscom_arrow_640.png", "chesscom_arrow_662.png"):
        readings = pipe.model.classify_frame(_real(name), geometry)
        _assert_arrow_reads(readings, pipe.bottom)
