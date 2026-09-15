import chess
import cv2

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
    assert model.symbols == set("PNBRQKpnbrqk")
    assert model.white_luminance > model.black_luminance
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
    later = chess.Board()
    for san in ("d4", "Nf6", "c4", "e6", "Nc3", "Bb4"):
        later.push_san(san)
    _check_all_cells(model, synthetic_board(board=later, bottom=Side.BLACK), later, Side.BLACK)


def test_hover_outline_needs_all_four_sides() -> None:
    """A badge crossing one corner of a cell is not the outline drawn around a hovered square."""
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
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
