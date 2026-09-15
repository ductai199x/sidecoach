import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region, Side
from sidecoach.vision.pieces import (
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
)
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from tests.synth import synthetic_board

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


def test_square_mapping_round_trips_both_orientations() -> None:
    for bottom in (Side.WHITE, Side.BLACK):
        for square in chess.SQUARES:
            r, c = row_col(square, bottom)
            assert square_at(r, c, bottom) == square
    assert square_at(7, 0, Side.WHITE) == chess.A1
    assert square_at(7, 0, Side.BLACK) == chess.H8
    assert square_at(0, 0, Side.BLACK) == chess.H1


def test_orientation_from_start_position() -> None:
    assert orientation_from_start(synthetic_board(), GEOM) == Side.WHITE
    assert orientation_from_start(synthetic_board(bottom=Side.BLACK), GEOM) == Side.BLACK


def test_start_position_detector() -> None:
    assert looks_like_start_position(synthetic_board(), GEOM)
    assert not looks_like_start_position(synthetic_board(pieces=False), GEOM)
    one_move = chess.Board()
    one_move.push_san("e4")
    assert not looks_like_start_position(synthetic_board(board=one_move), GEOM)
    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6"):
        later.push_san(san)
    assert not looks_like_start_position(synthetic_board(board=later), GEOM)


def _check_all_cells(model: ThemeModel, frame, board: chess.Board, bottom: Side) -> None:
    readings = model.classify_frame(frame, GEOM)
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, bottom))
            expected = piece.symbol() if piece else None
            got = readings[r][c]
            assert got.symbol == expected, f"row {r} col {c}: expected {expected}, got {got}"
            assert got.kind == ("empty" if expected is None else "piece")
            assert got.confidence > 0


def test_theme_model_classifies_start_and_later_positions() -> None:
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
    assert model is not None
    assert model.symbols == set("PNBRQKpnbrqk")
    _check_all_cells(model, frame, chess.Board(), Side.WHITE)

    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
        later.push_san(san)
    last = later.peek()
    frame2 = synthetic_board(board=later, highlight=last)
    _check_all_cells(model, frame2, later, Side.WHITE)
    readings = model.classify_frame(frame2, GEOM)
    highlighted = {(r, c) for r in range(8) for c in range(8) if readings[r][c].highlighted}
    assert highlighted == {
        row_col(last.from_square, Side.WHITE),
        row_col(last.to_square, Side.WHITE),
    }


def test_theme_model_black_at_bottom() -> None:
    frame = synthetic_board(bottom=Side.BLACK)
    model = ThemeModel.from_start_position(frame, GEOM, Side.BLACK)
    assert model is not None
    later = chess.Board()
    for san in ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"):
        later.push_san(san)
    _check_all_cells(model, synthetic_board(board=later, bottom=Side.BLACK), later, Side.BLACK)


def test_hover_outline_needs_all_four_sides() -> None:
    """A badge crossing one corner of a cell is not the outline drawn around a hovered square."""
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
    assert model is not None
    r, c = row_col(chess.E4, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)

    badged = frame.copy()
    cv2.rectangle(badged, (x1 - 24, y0 - 12), (x1 + 40, y0 + 22), (60, 60, 230), -1)
    reading = model.classify_frame(badged, GEOM)[r][c]
    assert not reading.hover

    hovered = frame.copy()
    cv2.rectangle(hovered, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (255, 255, 255), 3)
    reading = model.classify_frame(hovered, GEOM)[r][c]
    assert reading.hover


def test_bootstrap_refuses_a_start_frame_with_a_piece_hidden() -> None:
    """A cursor or badge over the queen at the start would be her exemplar for the game."""
    frame = synthetic_board()
    r, c = row_col(chess.D1, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)
    spoiled = frame.copy()
    spoiled[y0:y1, x0:x1] = (208, 236, 238)  # the queen erased: an empty light square
    assert looks_like_start_position(spoiled, GEOM)  # one stray cell is allowed here
    assert ThemeModel.from_start_position(spoiled, GEOM, Side.WHITE) is None

    hovered = frame.copy()
    cv2.rectangle(hovered, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (255, 255, 255), 3)
    assert ThemeModel.from_start_position(hovered, GEOM, Side.WHITE) is None


def test_bootstrap_fuses_pawns_so_one_odd_pawn_does_not_spoil_them() -> None:
    frame = synthetic_board()
    r, c = row_col(chess.E2, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)
    odd = frame.copy()
    cv2.circle(odd, (x1 - 10, y0 + 10), 7, (40, 40, 220), -1)  # a badge in the corner
    model = ThemeModel.from_start_position(odd, GEOM, Side.WHITE)
    assert model is not None
    clean = model.classify_frame(frame, GEOM)
    for square in (chess.A2, chess.E2, chess.H2):
        r, c = row_col(square, Side.WHITE)
        assert clean[r][c].symbol == "P" and clean[r][c].settled


def test_a_line_of_marked_squares_is_not_an_overlay() -> None:
    """Six right-click marks down a file, or a queued premove's squares, stay readable."""
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
    assert model is not None
    board = chess.Board()
    board.push_san("e4")
    marks = (chess.E3, chess.E4, chess.E5, chess.E6, chess.E7, chess.E8)
    readings = model.classify_frame(synthetic_board(board=board, red_marks=marks), GEOM)
    r, c = row_col(chess.E4, Side.WHITE)
    assert readings[r][c].symbol == "P" and not readings[r][c].covered
    r, c = row_col(chess.E8, Side.WHITE)
    assert readings[r][c].symbol == "k" and not readings[r][c].covered


def test_a_tiny_patch_reads_without_nan() -> None:
    import warnings

    from sidecoach.vision.pieces import silhouette

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sil = silhouette(np.full((2, 2, 3), 120, dtype=np.uint8))
    assert np.all(np.isfinite(sil.background)) and not sil.hover and sil.fraction == 0.0


@pytest.mark.parametrize("cell", [64, 48, 40, 32])
def test_recogniser_works_at_small_cell_sizes(cell: int) -> None:
    """A board on a small screen: every cell is upscaled to the working patch size.

    Thirty-two pixels a cell (a 256-pixel board) is the floor; below it the outline of a
    black piece outweighs its fill and colours flip. A 1440p session stored at 720p has
    43-pixel cells, a 1080p one 57.
    """
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    start = synthetic_board(cell=cell)
    assert looks_like_start_position(start, geometry)
    model = ThemeModel.from_start_position(start, geometry, Side.WHITE)
    assert model is not None, f"no recogniser at {cell} px cells"
    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
        later.push_san(san)
    readings = model.classify_frame(synthetic_board(board=later, cell=cell), geometry)
    wrong = []
    for r in range(8):
        for c in range(8):
            piece = later.piece_at(square_at(r, c, Side.WHITE))
            expected = piece.symbol() if piece else None
            if readings[r][c].symbol != expected:
                wrong.append((r, c, expected, readings[r][c].symbol))
    assert not wrong, wrong
