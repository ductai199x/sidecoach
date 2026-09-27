from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import detect_boards
from sidecoach.vision.pieces import (
    CELL,
    HIGHLIGHT_TOLERANCE,
    HOVER_TOLERANCE,
    CellReading,
    ThemeModel,
    iou,
    looks_like_start_position,
    orientation_from_start,
    overlay_reaches,
    silhouette,
    start_shaped,
    translucent_cover,
)
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from tests.pieceset import sheet_sets
from tests.synth import DARK, LIGHT, blend_outline, blend_rect, draw_piece, synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


def _start_model(**kwargs) -> ThemeModel:
    model = ThemeModel.from_start_position(synthetic_board(**kwargs), GEOM, Side.WHITE)
    assert model is not None
    return model


def _every_piece_reads(
    readings: list[list], board: chess.Board, bottom: Side, *, settled: bool = True
) -> list[str]:
    """Squares whose reading is not the piece there (exact and, if asked, settled)."""
    off = []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, bottom))
            expected = piece.symbol() if piece else None
            reading = readings[r][c]
            if reading.symbol != expected or (piece and settled and not reading.settled):
                off.append(chess.square_name(square_at(r, c, bottom)))
    return off


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
    # A badge on the pawn's upper left, touching it, so it is part of the silhouette
    # (a blob in a corner on its own is dropped by the silhouette itself) and in a
    # corner silhouette comparison does not ignore (only the top-right corner is
    # blind), so it is fusion that has to remove it.
    cv2.circle(odd, (x0 + 14, y0 + 14), 9, (40, 40, 220), -1)
    spoiled = silhouette(GEOM.crop(odd, r, c, size=2 * CELL)).mask
    clean_pawn = silhouette(GEOM.crop(frame, r, c, size=2 * CELL)).mask
    assert spoiled[:8, :8].any() and not clean_pawn[:8, :8].any()
    model = ThemeModel.from_start_position(odd, GEOM, Side.WHITE)
    assert model is not None
    exemplar = next(e for e in model.exemplars if e.symbol == "P")
    assert not exemplar.mask[:8, :8].any(), "the badge survived fusion"
    clean = model.classify_frame(frame, GEOM)
    for square in (chess.A2, chess.E2, chess.H2):
        r, c = row_col(square, Side.WHITE)
        assert clean[r][c].symbol == "P" and clean[r][c].settled


def _with_strip(frame: np.ndarray, square: chess.Square, height: float) -> np.ndarray:
    """A white strip `height` of the cell tall across the top of `square`'s cell."""
    r, c = row_col(square, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)
    out = frame.copy()
    out[y0 : y0 + int(round((y1 - y0) * height)), x0:x1] = (255, 255, 255)
    return out


@pytest.fixture
def overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """The silhouette overlap as the matcher: the tests of its exemplar hygiene."""
    from sidecoach.vision import pieces

    monkeypatch.setattr(pieces, "MATCHER", "iou")


@pytest.mark.usefixtures("overlap")
@pytest.mark.parametrize("height", [0.18, 0.24])
def test_bootstrap_does_not_fuse_a_spoiled_knight_with_its_pair(height: float) -> None:
    """A tooltip strip over b1 at bootstrap: refuse the frame, or keep g1's knight alone."""
    frame = synthetic_board()
    spoiled = _with_strip(frame, chess.B1, height)
    assert looks_like_start_position(spoiled, GEOM)
    model = ThemeModel.from_start_position(spoiled, GEOM, Side.WHITE)
    if model is None:
        return  # refused; the next start frame gets a chance
    exemplar = next(e for e in model.exemplars if e.symbol == "N")
    assert not exemplar.mask[:4].any(), "the strip became part of the knight"
    clean = model.classify_frame(frame, GEOM)
    for square in (chess.B1, chess.G1):
        r, c = row_col(square, Side.WHITE)
        assert clean[r][c].symbol == "N" and clean[r][c].settled, clean[r][c]


@pytest.mark.usefixtures("overlap")
def test_bootstrap_refuses_a_frame_where_both_colours_pairs_are_spoiled() -> None:
    frame = synthetic_board()
    spoiled = _with_strip(_with_strip(frame, chess.B1, 0.24), chess.G8, 0.24)
    assert ThemeModel.from_start_position(spoiled, GEOM, Side.WHITE) is None


def test_start_shaped_from_readings_matches_the_frame_rule() -> None:
    model = _start_model()
    one_move = chess.Board()
    one_move.push_san("e4")
    r, c = row_col(chess.D1, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)
    queen_erased = synthetic_board()
    queen_erased[y0:y1, x0:x1] = LIGHT
    for frame in (synthetic_board(), synthetic_board(board=one_move), queen_erased):
        assert start_shaped(model.classify_frame(frame, GEOM)) == looks_like_start_position(
            frame, GEOM
        )
    assert start_shaped(model.classify_frame(queen_erased, GEOM))
    assert not start_shaped(model.classify_frame(synthetic_board(board=one_move), GEOM))


@pytest.mark.usefixtures("overlap")
def test_degenerate_inputs() -> None:
    assert iou(np.zeros((CELL, CELL), dtype=bool), np.zeros((CELL, CELL), dtype=bool)) == 0.0

    frame = synthetic_board()
    r, c = row_col(chess.E2, Side.WHITE)
    pawn = silhouette(GEOM.crop(frame, r, c, size=2 * CELL))
    assert pawn.mask.any()
    assert not overlay_reaches(pawn, np.array([0.0, 255.0, 0.0], dtype=np.float32))
    r, c = row_col(chess.E4, Side.WHITE)
    empty = silhouette(GEOM.crop(frame, r, c, size=2 * CELL))
    assert not empty.mask.any()
    assert not overlay_reaches(empty, np.array(LIGHT, dtype=np.float32))

    backgrounds = np.zeros((8, 8, 3), dtype=np.float32)
    expected = np.zeros((8, 8, 3), dtype=np.float32)
    shifted = np.zeros((8, 8), dtype=bool)
    assert not translucent_cover(backgrounds, expected, shifted).any()

    # One member of a pair with no foreground at all: the frame is refused, not fused.
    r, c = row_col(chess.B1, Side.WHITE)
    x0, y0, x1, y1 = GEOM.cell_box(r, c)
    erased = frame.copy()
    erased[y0:y1, x0:x1] = LIGHT
    assert silhouette(GEOM.crop(erased, r, c, size=2 * CELL)).fraction == 0.0
    assert ThemeModel.from_start_position(erased, GEOM, Side.WHITE) is None


def test_a_board_retinted_all_over_is_a_theme_not_a_highlight() -> None:
    """The recogniser of the previous game reads a re-themed board as unknown, so the
    pipeline rebuilds it rather than taking the new theme for a highlighted start position."""
    model = _start_model()
    retinted = synthetic_board(light=(181, 217, 240), dark=(99, 136, 181))
    readings = model.classify_frame(retinted, GEOM)
    assert all(reading.kind == "unknown" for row in readings for reading in row)
    assert all(reading.tint is None and not reading.hover for row in readings for reading in row)
    # A few tinted cells stay highlights.
    later = chess.Board()
    move = later.push_san("e4")
    readings = model.classify_frame(synthetic_board(board=later, highlight=move), GEOM)
    r, c = row_col(chess.E4, Side.WHITE)
    assert readings[r][c].symbol == "P" and readings[r][c].tint == "yellow"


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


def test_recognition_survives_a_lock_a_pixel_off() -> None:
    """Exemplars built at one lock, readings taken at another a pixel or a tenth of a pixel
    of pitch away, which is what the tracker guarantees: every piece still reads exact and
    settled, because overlap is measured with the mask moved onto the exemplar and the
    settle tolerance has a floor of a few source pixels."""
    cell = 43
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    model = ThemeModel.from_start_position(synthetic_board(cell=cell), geometry, Side.WHITE)
    assert model is not None
    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
        later.push_san(san)
    frame = synthetic_board(board=later, cell=cell)
    for dx, dy, pitch in ((1, 1, 43.0), (0, 0, 43.1), (1, 1, 43.1), (-1, -1, 42.9)):
        off_lock = BoardGeometry(Region(300 + dx, 120 + dy, 8 * cell, 8 * cell), pitch)
        readings = model.classify_frame(frame, off_lock)
        assert not _every_piece_reads(readings, later, Side.WHITE), (dx, dy, pitch)
    # Twice the guaranteed error: still exact by shape, landing is another matter.
    for dx, dy, pitch in ((2, 2, 43.0), (0, 0, 43.3)):
        off_lock = BoardGeometry(Region(300 + dx, 120 + dy, 8 * cell, 8 * cell), pitch)
        readings = model.classify_frame(frame, off_lock)
        assert not _every_piece_reads(readings, later, Side.WHITE, settled=False), (dx, dy, pitch)


@pytest.mark.parametrize("cell", [43, 32])
@pytest.mark.parametrize("thickness", [1, 2, 3, 4])
@pytest.mark.parametrize("inset", [0, 1])
def test_hover_outline_of_any_thickness_is_seen(cell: int, thickness: int, inset: int) -> None:
    """A thin outline through the outer band test, a thick one through the foreign rule,
    on both square colours, with and without a piece; and a lock a pixel or two off does
    not turn a neighbour's edge into a hover."""
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    model = ThemeModel.from_start_position(synthetic_board(cell=cell), geometry, Side.WHITE)
    assert model is not None
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("N"))  # light square
    board.set_piece_at(chess.B4, chess.Piece.from_symbol("b"))  # dark square
    frame = synthetic_board(board=board, cell=cell)
    # Light with a piece, dark with a piece, light empty, dark empty; none adjacent, since
    # only one square is ever hovered and two outlined neighbours would look like a cover.
    outlined = (chess.E4, chess.B4, chess.G2, chess.C7)
    hovered = frame.copy()
    for square in outlined:
        x0, y0, x1, y1 = geometry.cell_box(*row_col(square, Side.WHITE))
        blend_outline(hovered, x0, y0, x1, y1, (255, 255, 255), 1.0, thickness, inset)
    readings = model.classify_frame(hovered, geometry)
    for square in outlined:
        r, c = row_col(square, Side.WHITE)
        assert readings[r][c].hover, (square, readings[r][c])
        if board.piece_at(square) is not None:
            assert not readings[r][c].settled, (square, readings[r][c])
    for dx, dy in ((1, 1), (2, 2), (-2, 1)):
        off_lock = BoardGeometry(Region(300 + dx, 120 + dy, 8 * cell, 8 * cell), float(cell))
        readings = model.classify_frame(frame, off_lock)
        false_hovers = [(r, c) for r in range(8) for c in range(8) if readings[r][c].hover]
        assert not false_hovers, (dx, dy, false_hovers)


TINTS = {
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "green": (76, 182, 129),
    "teal": (128, 128, 0),
}
OUTLINE_WHITE = (255, 255, 255)
RING_GREY = (40, 40, 40)


@pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("overlay", [*TINTS, "outline", "ring"])
@pytest.mark.parametrize("piece", [None, "N", "b"])
def test_tints_of_any_hue_read_through_and_outlines_read_foreign(
    overlay: str, alpha: float, piece: str | None
) -> None:
    """A translucent tint of any colour leaves the cell readable, named by hue when it is
    yellow or red; an outline or a ring at the edge, which shifts the ring but not the
    corners, reads as a hover, unknown when it is thick enough to fill the ring."""
    model = _start_model()
    for square in (chess.E4, chess.D4):  # a light and a dark square
        r, c = row_col(square, Side.WHITE)
        square_colour = LIGHT if (r + c) % 2 == 0 else DARK
        x0, y0, x1, y1 = GEOM.cell_box(r, c)
        frame = synthetic_board(board=chess.Board(None))
        if overlay in TINTS:
            colour = TINTS[overlay]
            blend_rect(frame, x0, y0, x1, y1, colour, alpha)
        elif overlay == "outline":
            colour = OUTLINE_WHITE
            blend_outline(frame, x0, y0, x1, y1, colour, alpha, thickness=3, inset=1)
        else:
            colour = RING_GREY
            blend_outline(frame, x0, y0, x1, y1, colour, alpha, thickness=8, inset=0)
        if piece is not None:
            draw_piece(frame, (x0 + x1) // 2, (y0 + y1) // 2, 64, piece)
        reading = model.classify_frame(frame, GEOM)[r][c]
        shift = alpha * float(np.linalg.norm(np.array(colour) - np.array(square_colour)))
        if overlay in TINTS:
            assert not reading.hover, reading
            if piece is None:
                assert reading.kind == "empty", reading
            else:
                assert reading.symbol == piece and reading.settled, reading
            if shift > HIGHLIGHT_TOLERANCE:
                hue = overlay if overlay in ("yellow", "red") else "other"
                assert reading.tint == hue, reading
                assert reading.highlighted == (overlay != "red")
            else:
                assert reading.tint is None and not reading.highlighted, reading
        elif overlay == "outline":
            # A thin outline leaves the ring clean, so the piece still reads, unsettled.
            if shift > HOVER_TOLERANCE:
                assert reading.hover, reading
                assert piece is None or not reading.settled, reading
            if piece is not None:
                assert reading.symbol == piece, reading
        else:
            # A ring thick enough to fill the shallow ring hides the cell's content.
            if shift > HIGHLIGHT_TOLERANCE:
                assert reading.kind == "unknown" and reading.hover, reading
            elif piece is not None:
                assert reading.symbol == piece, reading


@pytest.mark.parametrize("cell", [64, 48, 40])
def test_recogniser_works_at_small_cell_sizes(cell: int) -> None:
    """A board on a small screen: every cell is upscaled to the working patch size.

    Forty pixels a cell (a 320-pixel board) is the floor with anti-aliased pieces; below it
    the light outline of a black pawn outweighs its fifteen-pixel fill and the colour
    flips. A 1440p session stored at 720p has 43-pixel cells, a 1080p one 57.
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


@pytest.mark.parametrize(
    ("theme", "style", "cell"),
    [
        # A dark board and a set whose black pieces wear a grey outline: on the dark
        # squares that outline is a whisker from the square, its blend with the fill is
        # the square's own colour, and a split at mid-grey kept the outline on light
        # squares and lost it on dark ones, so a black pawn on a dark square matched the
        # queen as well as the pawn. The tone is now what lies nearer the core than the
        # square.
        ("dark", "flat", 44),
        ("grey", "flat", 44),
        # A bold set: an outline wider than the fixed erosion the core was read with,
        # whose rim made the white queen, a thin star inside its outline, as grey as
        # the black one. The core is now the deep inside of the silhouette.
        ("ice", "bold", 56),
        ("brown", "bold", 56),
        ("wood", "bold", 44),
        # A coloured set on a textured board: the black queen's fill is a thin star inside
        # a wide tan outline, 0.39 of its silhouette where the outline shows and 0.87 where
        # it does not, and a tone floor of 0.4 gave it one shape on each colour of square.
        ("wood", "wooden", 64),
        ("purple", "wooden", 48),
        ("blue", "thin", 44),
        # The real set (tests/pieceset.py): thin strokes, a knight's mane, a bishop's
        # slit, detail in the other side's colour inside every piece.
        ("green", "cburnett", 64),
        ("brown", "cburnett", 44),
        ("dark", "cburnett", 56),
        ("ice", "cburnett", 96),
        ("wood", "cburnett-wood", 64),
        ("grey", "cburnett-small", 44),
    ],
)
def test_every_theme_and_piece_style_reads_every_square(theme: str, style: str, cell: int) -> None:
    """The stress sweep's finds, kept as the cases they were found on: a start position
    and a middlegame on this theme and set, every square read as what it holds."""
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    width, height = 8 * cell + 600, 8 * cell + 240
    for bottom in (Side.WHITE, Side.BLACK):
        start = synthetic_board(
            width=width, height=height, cell=cell, theme=theme, style=style, bottom=bottom
        )
        model = ThemeModel.from_start_position(start, geometry, bottom)
        assert model is not None, f"no recogniser for {theme}/{style} at {cell} px"
        later = chess.Board()
        for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
            later.push_san(san)
        frame = synthetic_board(
            width=width,
            height=height,
            board=later,
            cell=cell,
            theme=theme,
            style=style,
            bottom=bottom,
        )
        off = _every_piece_reads(model.classify_frame(frame, geometry), later, bottom)
        assert not off, f"{theme}/{style} at {cell} px, {bottom.value} at the bottom: {off}"


@pytest.mark.parametrize("name", sheet_sets())
def test_every_lichess_set_reads_every_square(name: str) -> None:
    """Every set lichess draws (tests/fixtures/pieces): learned from a start position on
    the brown board at 64-pixel cells, both orientations, and a middlegame read square by
    square."""
    cell = 64
    geometry = BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))
    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
        later.push_san(san)
    for bottom in (Side.WHITE, Side.BLACK):
        start = synthetic_board(cell=cell, theme="brown", style=name, bottom=bottom)
        model = ThemeModel.from_start_position(start, geometry, bottom)
        assert model is not None, f"no recogniser for {name}"
        frame = synthetic_board(board=later, cell=cell, theme="brown", style=name, bottom=bottom)
        off = _every_piece_reads(model.classify_frame(frame, geometry), later, bottom)
        assert not off, f"{name}, {bottom.value} at the bottom: {off}"


def test_a_fractional_cell_never_wears_a_phantom_hover_outline() -> None:
    """641 px for eight squares is 80.125 a square: the edges fall between pixels and the
    straddling pixels blend the two square colours. Rounding the crop to the nearest pixel
    let those blends in on both sides of the e-file and both sides of the fourth rank, and
    e4 alone read as hovered on every frame -- so 1.e4 never settled and the second live
    game was never followed. A crop of whole pixels only has clean edges everywhere."""
    light, dark = (176, 214, 237), (98, 135, 184)
    exact = np.zeros((648, 648, 3), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            exact[r * 81 : (r + 1) * 81, c * 81 : (c + 1) * 81] = (
                light if (r + c) % 2 == 0 else dark
            )
    frame = np.full((900, 900, 3), 40, dtype=np.uint8)
    frame[100:741, 100:741] = cv2.resize(exact, (641, 641), interpolation=cv2.INTER_AREA)
    geometry = BoardGeometry(Region(100, 100, 641, 641), 641 / 8)
    hovered = [
        (r, c)
        for r in range(8)
        for c in range(8)
        if silhouette(geometry.crop(frame, r, c, size=2 * CELL)).hover
    ]
    assert hovered == []
    # Whole pixels only, and never more than a pixel lost a side.
    x0, y0, x1, y1 = geometry.cell_box(4, 4)
    assert (x0, y0, x1, y1) == (421, 421, 500, 500)
    assert all(
        79 <= geometry.cell_box(r, c)[2] - geometry.cell_box(r, c)[0] <= 81
        for r in range(8)
        for c in range(8)
    )


def _read_final(
    style: str,
    theme: str,
    cell: int,
    bottom: Side,
    sans: tuple[str, ...],
    square: str,
    coordinates: bool = True,
    **decorations: Any,
) -> tuple[CellReading, chess.Piece]:
    """A set learned from its start frame, then the position after `sans` read at the same
    place and size, each locked by the detector as the pipeline locks them (the stress
    sweep's frames), and what stands on `square` there."""
    size: dict[str, Any] = {
        "width": 8 * cell + 600, "height": 8 * cell + 240, "x0": 300, "y0": 120, "cell": cell,
        "theme": theme, "style": style, "bottom": bottom, "coordinates": coordinates,
    }  # fmt: skip
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    start = synthetic_board(**size)
    frame = synthetic_board(board=board, **size, **decorations)
    first, now = detect_boards(start)[0], detect_boards(frame)[0]
    model = ThemeModel.from_start_position(
        start, BoardGeometry(first.region, first.cell_px), bottom
    )
    assert model is not None
    r, c = row_col(chess.parse_square(square), bottom)
    piece = board.piece_at(chess.parse_square(square))
    assert piece is not None
    return model.classify_frame(frame, BoardGeometry(now.region, now.cell_px))[r][c], piece


@pytest.mark.parametrize(
    ("style", "theme", "cell", "bottom", "sans", "square"),
    [
        # A black king on a dark square, learned only on a light one: read as the queen, and
        # the final position was never read cold (stress seed 766).
        ("wooden", "wood", 64, Side.BLACK, ("d3", "e6", "Nd2", "Ke7"), "e7"),
        # A black queen on a light square, learned only on a dark one: read as a bishop, and
        # the game stalled (469).
        (
            "flat", "dark", 96, Side.WHITE,
            ("h3", "g5", "a4", "b6", "d4", "Bg7", "Bxg5", "d5", "c4", "Bf6", "cxd5", "Qxd5",
             "a5"),
            "d5",
        ),
    ],
)  # fmt: skip
def test_a_king_or_queen_is_named_on_the_other_square_colour(
    style: str,
    theme: str,
    cell: int,
    bottom: Side,
    sans: tuple[str, ...],
    square: str,
) -> None:
    """The start position shows each side's king and queen on one square colour only, and at
    the embedder's 40 px the square weighs in the embedding: once a piece's side is voted,
    its kind may be named by either side's cells and by the kings and queens recoloured onto
    the other square colour."""
    reading, piece = _read_final(style, theme, cell, bottom, sans, square)
    assert reading.kind == "piece" and reading.symbol == piece.symbol(), reading


def test_the_recoloured_kings_and_queens_never_choose_a_side() -> None:
    """A white queen on a dark square at the mate, its side vote split (its brightness white,
    the fill under the queen's mask black): the embedding chose, and the black queen
    recoloured onto a light square lay nearer it than its own white queen -- the mate was
    never committed (stress seed 372). The recoloured cells name a kind once the side is
    voted; a split vote is decided on the start position's own 64 cells."""
    sans = (
        "e3", "Nf6", "Be2", "g5", "f4", "Bh6", "fxg5", "Bxg5", "Nf3", "h5", "Bd3", "Ne4",
        "Nh4", "Nc5", "Qxh5", "Bxe3", "Qxh8#",
    )  # fmt: skip
    reading, piece = _read_final(
        "california", "grey", 44, Side.BLACK, sans, "h8",
        highlight=chess.Move.from_uci("h5h8"), check=True, result="1-0", badges=True,
    )  # fmt: skip
    assert reading.kind == "piece" and reading.symbol == piece.symbol(), reading
