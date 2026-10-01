import copy
import itertools
import random
import warnings
from functools import cache
from typing import Any

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import Region, Side
from sidecoach.vision.board_detect import detect_boards
from sidecoach.vision.embedding import CellEmbedder
from sidecoach.vision.pieces import (
    CELL,
    HIGHLIGHT_TOLERANCE,
    HOVER_TOLERANCE,
    CellReading,
    ThemeModel,
    looks_like_start_position,
    orientation_from_start,
    overlay_reaches,
    silhouette,
    start_shaped,
    translucent_cover,
)
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from tests.games import recogniser
from tests.pieceset import sheet_sets
from tests.synth import (
    DARK,
    GEOM,
    LIGHT,
    RED,
    YELLOW,
    blend_outline,
    blend_rect,
    draw_piece,
    synthetic_board,
    tint,
)

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision

# After 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.O-O d5 5.exd5 Nxd5: the middlegame the sets are read on.
MIDDLEGAME = ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5")


def _played(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _geometry(cell: int) -> BoardGeometry:
    """The lock on `synthetic_board(cell=cell)`'s board, which is drawn at (300, 120)."""
    return BoardGeometry(Region(300, 120, 8 * cell, 8 * cell), float(cell))


@cache
def _learned(cell: int) -> ThemeModel:
    model = ThemeModel.from_start_position(synthetic_board(cell=cell), _geometry(cell), Side.WHITE)
    assert model is not None
    return model


def _start_model(cell: int = 64) -> ThemeModel:
    """The default set learned from its start frame at `cell` px (White at the bottom), once
    per worker: each test gets a copy of its own, with no frame read yet."""
    return copy.deepcopy(_learned(cell))


def _every_piece_reads(
    readings: list[list], board: chess.Board, bottom: Side, *, settled: bool = True
) -> list[str]:
    """Squares whose reading is not what stands there: the piece, exact and (if asked)
    settled, or an empty square read as empty."""
    off = []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, bottom))
            reading = readings[r][c]
            if piece is None:
                wrong = reading.kind != "empty"
            else:
                wrong = reading.symbol != piece.symbol() or (settled and not reading.settled)
            if wrong:
                off.append(chess.square_name(square_at(r, c, bottom)))
    return off


def _queen_erased() -> np.ndarray:
    """The start frame with White's queen erased: an empty light square on d1."""
    frame = synthetic_board()
    x0, y0, x1, y1 = GEOM.cell_box(*row_col(chess.D1, Side.WHITE))
    frame[y0:y1, x0:x1] = LIGHT
    return frame


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


def test_the_start_position_rule_reads_the_same_from_the_frame_and_from_readings() -> None:
    """The start's shape, one stray cell allowed (a badge, an outline, a cursor), and nothing
    a move or more from it, nor an empty board: from the frame's occupancy before there is a
    recogniser (`looks_like_start_position`), and from a recogniser's readings by the same
    rule (`start_shaped`)."""
    model = _start_model()
    frames = {
        "start": (synthetic_board(), True),
        "queen erased": (_queen_erased(), True),
        "one move": (synthetic_board(board=_played("e4")), False),
        "later": (synthetic_board(board=_played("e4", "e5", "Nf3", "Nc6", "Bb5", "a6")), False),
        "empty": (synthetic_board(pieces=False), False),
    }
    for name, (frame, start) in frames.items():
        assert looks_like_start_position(frame, GEOM) == start, name
        assert start_shaped(model.classify_frame(frame, GEOM)) == start, name


def test_bootstrap_refuses_a_start_frame_with_a_piece_hidden() -> None:
    """A cursor or badge over the queen at the start would be her exemplar for the game; the
    start's shape allows the one stray cell, so the recogniser itself must refuse it."""
    assert ThemeModel.from_start_position(_queen_erased(), GEOM, Side.WHITE) is None
    hovered = synthetic_board()
    x0, y0, x1, y1 = GEOM.cell_box(*row_col(chess.D1, Side.WHITE))
    cv2.rectangle(hovered, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), (255, 255, 255), 3)
    assert ThemeModel.from_start_position(hovered, GEOM, Side.WHITE) is None


def _labelled(board: chess.Board, style: str, theme: str, cell: int, **kw: Any) -> np.ndarray:
    """`board` with its coordinate labels, drawn at (300, 120) on a frame 400 px wider and
    200 px taller than the board."""
    return synthetic_board(
        width=8 * cell + 400, height=8 * cell + 200, board=board, theme=theme, cell=cell,
        style=style, coordinates=True, **kw,
    )  # fmt: skip


def test_without_the_net_nothing_is_learned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The net ships in the package; a broken install without it learns no set, rather
    than read pieces some other way."""
    monkeypatch.setattr(CellEmbedder, "load", staticmethod(lambda path=None: None))
    cell = 64
    frame = _labelled(chess.Board(), "cburnett", "green", cell)
    assert ThemeModel.from_start_position(frame, _geometry(cell), Side.WHITE) is None


def test_degenerate_inputs_read_as_nothing() -> None:
    """A 2x2 patch reads without a NaN or a warning, an empty square has no mask for an
    overlay to reach (a pawn has one, out of a far colour's reach), and a board of zeros
    holds no translucent cover."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sil = silhouette(np.full((2, 2, 3), 120, dtype=np.uint8))
    assert np.all(np.isfinite(sil.background)) and not sil.hover and sil.fraction == 0.0

    frame = synthetic_board()
    pawn = silhouette(GEOM.crop(frame, *row_col(chess.E2, Side.WHITE), size=2 * CELL))
    assert pawn.mask.any()
    assert not overlay_reaches(pawn, np.array([0.0, 255.0, 0.0], dtype=np.float32))
    empty = silhouette(GEOM.crop(frame, *row_col(chess.E4, Side.WHITE), size=2 * CELL))
    assert not empty.mask.any()
    assert not overlay_reaches(empty, np.array(LIGHT, dtype=np.float32))

    zeros = np.zeros((8, 8, 3), dtype=np.float32)
    assert not translucent_cover(zeros, zeros, np.zeros((8, 8), dtype=bool)).any()


def test_hover_outline_needs_all_four_sides() -> None:
    """A badge crossing one corner of a cell is not the outline drawn around a hovered square."""
    frame = synthetic_board()
    model = _start_model()
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


def test_a_board_retinted_all_over_is_a_theme_not_a_highlight() -> None:
    """The recogniser of the previous game reads a re-themed board as unknown, so the
    pipeline rebuilds it rather than taking the new theme for a highlighted start position.
    (A few tinted cells stay highlights: test_every_theme_and_piece_style_reads_every_square.)"""
    model = _start_model()
    retinted = synthetic_board(light=(181, 217, 240), dark=(99, 136, 181))
    readings = model.classify_frame(retinted, GEOM)
    assert all(reading.kind == "unknown" for row in readings for reading in row)
    assert all(reading.tint is None and not reading.hover for row in readings for reading in row)


def test_a_line_of_marked_squares_is_not_an_overlay() -> None:
    """Six right-click marks down a file, or a queued premove's squares, stay readable."""
    model = _start_model()
    marks = (chess.E3, chess.E4, chess.E5, chess.E6, chess.E7, chess.E8)
    readings = model.classify_frame(synthetic_board(board=_played("e4"), red_marks=marks), GEOM)
    r, c = row_col(chess.E4, Side.WHITE)
    assert readings[r][c].symbol == "P" and not readings[r][c].covered
    r, c = row_col(chess.E8, Side.WHITE)
    assert readings[r][c].symbol == "k" and not readings[r][c].covered


def test_recognition_survives_a_lock_a_pixel_off() -> None:
    """Exemplars built at one lock, readings taken at another a pixel or a tenth of a pixel
    of pitch away, which is what the tracker guarantees: every piece still reads exact and
    settled. The embedding names the crop a pixel off as it names it in place, and the
    landing -- the shift the phase correlation measures against the piece's start crop --
    is held to a tolerance with a floor of a few source pixels (`settle_tolerance`)."""
    cell = 43
    model = _start_model(cell)
    later = _played(*MIDDLEGAME)
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


def _two_pieces(cell: int) -> tuple[chess.Board, np.ndarray]:
    """A knight on e4 (a light square) and a bishop on b4 (a dark one), and the frame."""
    board = chess.Board(None)
    board.set_piece_at(chess.E4, chess.Piece.from_symbol("N"))
    board.set_piece_at(chess.B4, chess.Piece.from_symbol("b"))
    return board, synthetic_board(board=board, cell=cell)


@pytest.mark.parametrize("cell", [43, 32])
def test_a_hover_outline_of_any_thickness_is_seen_and_a_lock_off_by_a_pixel_sees_none(
    cell: int,
) -> None:
    """A thin outline through the outer band test, a thick one through the foreign rule,
    on both square colours, with and without a piece; and a lock a pixel or two off does not
    turn a neighbour's edge into a hover. One recogniser reads every frame, as the live one
    does: only the cells that changed since the last are read again."""
    geometry = _geometry(cell)
    model = _start_model(cell)
    board, frame = _two_pieces(cell)
    # Light with a piece, dark with a piece, light empty, dark empty; none adjacent, since
    # only one square is ever hovered and two outlined neighbours would look like a cover.
    outlined = (chess.E4, chess.B4, chess.G2, chess.C7)
    for thickness, inset in itertools.product((1, 2, 3, 4), (0, 1)):
        hovered = frame.copy()
        for square in outlined:
            x0, y0, x1, y1 = geometry.cell_box(*row_col(square, Side.WHITE))
            blend_outline(hovered, x0, y0, x1, y1, (255, 255, 255), 1.0, thickness, inset)
        readings = model.classify_frame(hovered, geometry)
        for square in outlined:
            r, c = row_col(square, Side.WHITE)
            case = (thickness, inset, chess.square_name(square), readings[r][c])
            assert readings[r][c].hover, case
            if board.piece_at(square) is not None:
                assert not readings[r][c].settled, case
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


def test_tints_of_any_hue_read_through_and_outlines_read_foreign() -> None:
    """A translucent tint of any colour leaves the cell readable, named by hue when it is
    yellow or red; an outline or a ring at the edge, which shifts the ring but not the
    corners, reads as a hover, unknown when it is thick enough to fill the ring. Each
    overlay at three strengths over a light and a dark square, empty or under a white or a
    black piece; one recogniser reads them all, only the changed cell read again."""
    model = _start_model()
    empty = synthetic_board(board=chess.Board(None))
    overlays = [*TINTS, "outline", "ring"]
    for overlay, alpha, piece in itertools.product(overlays, (0.3, 0.5, 0.7), (None, "N", "b")):
        for square in (chess.E4, chess.D4):  # a light and a dark square
            case = (overlay, alpha, piece, chess.square_name(square))
            r, c = row_col(square, Side.WHITE)
            square_colour = LIGHT if (r + c) % 2 == 0 else DARK
            x0, y0, x1, y1 = GEOM.cell_box(r, c)
            frame = empty.copy()
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
                assert not reading.hover, (case, reading)
                if piece is None:
                    assert reading.kind == "empty", (case, reading)
                else:
                    assert reading.symbol == piece and reading.settled, (case, reading)
                if shift > HIGHLIGHT_TOLERANCE:
                    hue = overlay if overlay in ("yellow", "red") else "other"
                    assert reading.tint == hue, (case, reading)
                    assert reading.highlighted == (overlay != "red"), (case, reading)
                else:
                    assert reading.tint is None and not reading.highlighted, (case, reading)
            elif overlay == "outline":
                # A thin outline leaves the ring clean, so the piece still reads, unsettled.
                if shift > HOVER_TOLERANCE:
                    assert reading.hover, (case, reading)
                    assert piece is None or not reading.settled, (case, reading)
                if piece is not None:
                    assert reading.symbol == piece, (case, reading)
            else:
                # A ring thick enough to fill the shallow ring hides the cell's content.
                if shift > HIGHLIGHT_TOLERANCE:
                    assert reading.kind == "unknown" and reading.hover, (case, reading)
                elif piece is not None:
                    assert reading.symbol == piece, (case, reading)


# The piece styles the renderer draws besides the lichess sheets: its polygons -- a flat set
# (faint outlines, white on white and black on black), a bold one (an outline a twentieth of
# the cell), a wooden one (cream and brown, both sides chromatic) and a thin one -- and the
# real cburnett set in its colourings, each on themes and at sizes where the stress sweep
# once found them misread (the grey outline of a black piece on a dark board, a bold white
# queen as grey as the black one, a wooden queen's thin fill). The sweep plays these styles
# only in its random seeds; here each is learned and read every time.
STYLE_ROWS = [
    ("dark", "flat", 44),
    ("grey", "flat", 44),
    ("ice", "bold", 56),
    ("brown", "bold", 56),
    ("wood", "bold", 44),
    ("wood", "wooden", 64),
    ("purple", "wooden", 48),
    ("blue", "thin", 44),
    ("green", "cburnett", 64),
    ("brown", "cburnett", 44),
    ("dark", "cburnett", 56),
    ("ice", "cburnett", 96),
    ("wood", "cburnett-wood", 64),
    ("grey", "cburnett-small", 44),
    # Every set lichess draws (tests/fixtures/pieces), on the brown board at 64-pixel cells:
    # three of them by default, the rest with --sweep.
    *(("brown", name, 64) for name in sheet_sets()),
    # A board on a small screen, every cell upscaled to the working patch size: forty pixels
    # a cell (a 320-pixel board) is the floor with anti-aliased pieces; below it the light
    # outline of a black pawn outweighs its fifteen-pixel fill and the colour flips. A 1440p
    # session stored at 720p has 43-pixel cells, a 1080p one 57.
    ("green", "classic", 40),
]


def _alternating(rows: list) -> list[tuple]:
    """Each row with an orientation, White and Black at the bottom in turn. Turning the board
    round keeps every square's colour, so a set read one way up is read the other way up
    on the same squares; the orientation is covered across the rows, not twice in each."""
    return [(*row, (Side.WHITE, Side.BLACK)[i % 2]) for i, row in enumerate(rows)]


def _style_params() -> list:
    """Every row; past the first three lichess sheets, a sweep case (--sweep)."""
    sheets = set(sheet_sets())
    out, seen = [], 0
    for row in _alternating(STYLE_ROWS):
        theme, style, cell = row[:3]
        swept = style in sheets and (seen := seen + 1) > 3
        out.append(
            pytest.param(
                *row, id=f"{theme}-{style}-{cell}", marks=[pytest.mark.sweep] if swept else []
            )
        )
    return out


@pytest.mark.parametrize(("theme", "style", "cell", "bottom"), _style_params())
def test_every_theme_and_piece_style_reads_every_square(
    theme: str, style: str, cell: int, bottom: Side
) -> None:
    """A set learned from its start position on this theme at this size, and a middlegame
    read square by square: every piece exact and settled, every empty square empty, and the
    last move's two squares, and only those, highlighted."""
    geometry = _geometry(cell)
    kw: dict[str, Any] = {
        "width": 8 * cell + 600, "height": 8 * cell + 240, "cell": cell, "theme": theme,
        "style": style, "bottom": bottom,
    }  # fmt: skip
    model = ThemeModel.from_start_position(synthetic_board(**kw), geometry, bottom)
    assert model is not None, f"no recogniser for {theme}/{style} at {cell} px"
    later = _played(*MIDDLEGAME)
    readings = model.classify_frame(
        synthetic_board(board=later, highlight=later.peek(), **kw), geometry
    )
    off = _every_piece_reads(readings, later, bottom)
    lit = {
        chess.square_name(square_at(r, c, bottom))
        for r in range(8)
        for c in range(8)
        if readings[r][c].highlighted
    }
    where = f"{theme}/{style} at {cell} px, {bottom.value} at the bottom"
    assert not off, f"{where}: {off}"
    assert lit == {"f6", "d5"}, f"{where}: highlighted {lit}"


@pytest.mark.parametrize(
    ("style", "theme", "cell"),
    [
        pytest.param(style, theme, cell, id=f"{theme}-{cell}-{style}")
        for style, theme, cell in (
            ("pixel", "green", 48),
            ("pixel", "dark", 112),
            ("riohacha", "dark", 112),
            ("chess7", "green", 48),
            ("chess7", "dark", 112),
        )
    ],
)
def test_reads_a_set_from_its_own_start_position(style: str, theme: str, cell: int) -> None:
    """A random middlegame (thirty plies, seed 7) on a set learned from its own start
    position, its labels drawn and its last move highlighted: every piece named, every empty
    square empty. Pieces stand here on square colours they did not start on, where the
    embedding's nearest white and nearest black piece can lie too close for it to vote on
    the side (COLOUR_TIE) and the brightness and the fill decide it (pieces._vote_side): on
    the dark board a pixel queen and a riohacha pawn, which stand on their start squares'
    colours in STYLE_ROWS' middlegame."""
    rng = random.Random(7)
    board = chess.Board()
    for _ in range(30):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    geometry = _geometry(cell)
    model = ThemeModel.from_start_position(
        _labelled(chess.Board(), style, theme, cell), geometry, Side.WHITE
    )
    assert model is not None
    readings = model.classify_frame(
        _labelled(board, style, theme, cell, highlight=board.move_stack[-1]), geometry
    )
    wrong = []
    for r in range(8):
        for c in range(8):
            piece = board.piece_at(square_at(r, c, Side.WHITE))
            got = readings[r][c]
            want = piece.symbol() if piece else None
            if want is None and got.kind != "empty":
                wrong.append((r, c, "empty", got.kind, got.symbol))
            elif want is not None and (got.kind != "piece" or got.symbol != want):
                wrong.append((r, c, want, got.kind, got.symbol, got.guess))
    assert not wrong, wrong


@pytest.mark.slow  # -m slow: the default run reads the net on the CPU (tests/conftest.py)
def test_the_net_runs_on_the_gpu_and_reads_as_on_the_cpu() -> None:
    """The app takes the GPU when it has one, and the tests read the net on the CPU: this
    keeps the GPU path tested -- it loads there, and embeds the same cells the same way."""
    from sidecoach.voice.runtime import cuda_available

    if not cuda_available():
        pytest.skip("no CUDA on this machine")
    gpu, cpu = CellEmbedder(device="cuda"), CellEmbedder(device="cpu")
    assert gpu.device.name == "cuda", gpu.device.detail
    cell = 64
    frame = _labelled(chess.Board(), "cburnett", "green", cell)
    crops = [frame[120 + r * cell : 120 + (r + 1) * cell, 300 : 300 + cell] for r in range(8)]
    on_gpu, on_cpu = gpu.embed(crops), cpu.embed(crops)
    assert on_gpu.shape == on_cpu.shape == (8, 1280)
    assert float((on_gpu * on_cpu).sum(axis=1).min()) > 0.999, "the same vectors, to rounding"


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


def _read_after(
    style: str,
    theme: str,
    learned: tuple[int, int, int],
    read: tuple[int, int, int],
    sans: tuple[str, ...],
    square: str,
    bottom: Side = Side.WHITE,
    coordinates: bool = False,
    frame: tuple[int, int] = (1920, 1200),
    **decorations: object,
) -> CellReading:
    """A set learned from a start frame at one place and cell size (x, y, cell), and a later
    position read at another, on frames `frame` wide and high, each locked by the detector
    as the pipeline locks them."""
    kw: dict[str, Any] = {
        "width": frame[0], "height": frame[1], "theme": theme, "style": style, "bottom": bottom,
        "coordinates": coordinates,
    }  # fmt: skip
    start = synthetic_board(x0=learned[0], y0=learned[1], cell=learned[2], **kw)
    board = _played(*sans)
    now = synthetic_board(x0=read[0], y0=read[1], cell=read[2], board=board, **decorations, **kw)
    first, again = detect_boards(start)[0], detect_boards(now)[0]
    model = ThemeModel.from_start_position(
        start, BoardGeometry(first.region, first.cell_px), bottom
    )
    assert model is not None
    r, c = row_col(chess.parse_square(square), bottom)
    return model.classify_frame(now, BoardGeometry(again.region, again.cell_px))[r][c]


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
    the other square colour. Learned and read in place, on the stress sweep's frames."""
    at = (300, 120, cell)
    reading = _read_after(
        style, theme, at, at, sans, square, bottom, coordinates=True,
        frame=(8 * cell + 600, 8 * cell + 240),
    )  # fmt: skip
    piece = _played(*sans).piece_at(chess.parse_square(square))
    assert piece is not None
    assert reading.kind == "piece" and reading.symbol == piece.symbol(), reading


@pytest.mark.parametrize(
    ("style", "theme", "learned", "read", "sans", "square", "lit"),
    [
        # A rook just moved, on its lit square, learned at 40 px and read at 34: it measured
        # 0.15 of a cell off and never landed.
        (
            "wooden", "brown", (1124, 191, 40), (1648, 911, 34),
            ("b3", "h6", "c3", "Nf6", "Qc2", "g5", "Qe4", "Rh7"), "h7", "h8h7",
        ),
        # A king learned at 48 px and read at 132: 0.12 off.
        (
            "classic", "ice", (559, 284, 48), (607, 44, 132),
            ("a4", "e5", "Ra2", "Ba3", "Ra1", "f5", "Nh3", "Ke7", "d4", "Kd6"), "d6", "e7d6",
        ),
    ],
)  # fmt: skip
def test_a_piece_at_rest_lands_on_a_board_resized_since_its_set_was_learned(
    style: str,
    theme: str,
    learned: tuple[int, int, int],
    read: tuple[int, int, int],
    sans: tuple[str, ...],
    square: str,
    lit: str,
) -> None:
    """After a resize the cell and the piece's reference each carry detail the other has not,
    and the phase correlation's peak was lost in it; the two are also compared brought down
    to the coarser size, and the surer correlation counts (the geometry sweep, 2026-09-27,
    seeds 7 and 237)."""
    reading = _read_after(
        style, theme, learned, read, sans, square, highlight=chess.Move.from_uci(lit)
    )
    piece = _played(*sans).piece_at(chess.parse_square(square))
    assert piece is not None
    assert reading.kind == "piece" and reading.symbol == piece.symbol(), reading
    assert reading.settled, reading.offset


def test_a_piece_in_flight_on_a_resized_board_does_not_land_on_the_square_it_passes() -> None:
    """A pawn three quarters of the way from b2 to b4, learned at 40 px and read at 36: it
    measured 0.08 of a cell from b3's centre and was committed there, a square short (the
    geometry sweep, 2026-09-27, seed 87)."""
    move = chess.Move.from_uci("b2b4")
    reading = _read_after(
        "spatial", "dark", (1062, 299, 40), (600, 300, 36), ("h3", "f6"), "b3",
        highlight=move, animating=(move.from_square, move.to_square, 0.75),
    )  # fmt: skip
    assert not (reading.kind == "piece" and reading.settled), reading


def test_a_rook_on_a_labelled_corner_lands_at_the_size_its_set_was_learned_at() -> None:
    """Brought down to the size it was drawn at, a cell is at the edge of what its pixels
    hold: a rook on its labelled corner square measured half a cell off, and the cold read
    of every flat set at 44 px was refused (the stress sweep, 2026-09-27, seeds 113 and 138).
    At the size its set was learned at, a cell is compared at full detail only."""
    sans = (
        "Nh3", "d6", "Ng5", "e6", "Nc3", "Ne7", "Na4", "Nd7",
        "d3", "Nb6", "Kd2", "h6", "Ke3", "Rh7", "Nf3", "a5",
    )  # fmt: skip
    reading = _read_after(
        "flat", "green", (300, 120, 64), (300, 120, 64), sans, "h1", bottom=Side.BLACK,
        coordinates=True,
    )  # fmt: skip
    assert reading.kind == "piece" and reading.symbol == "R", reading
    assert reading.settled, reading.offset


def test_highlight_kinds() -> None:
    model = recogniser(Side.WHITE)
    light = np.array(LIGHT, dtype=np.float32)
    assert model.highlight_kind(light, 0, 0) == "none"
    assert model.highlight_kind(np.array(tint(LIGHT, YELLOW), dtype=np.float32), 0, 0) == "move"
    assert model.highlight_kind(np.array(tint(LIGHT, RED), dtype=np.float32), 0, 0) == "red"
    assert model.highlight_kind(np.array([245, 245, 245], dtype=np.float32), 0, 0) == "foreign"
    assert model.highlight_kind(np.array([200, 200, 200], dtype=np.float32), 0, 0) == "foreign"
    dark = np.array(DARK, dtype=np.float32)
    assert model.highlight_kind(dark, 0, 1) == "none"
    assert model.highlight_kind(np.array(tint(DARK, YELLOW), dtype=np.float32), 0, 1) == "move"
    assert model.highlight_kind(np.array([200, 200, 200], dtype=np.float32), 0, 1) == "foreign"
