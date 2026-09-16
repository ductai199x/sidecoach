"""A dialog over part of the board: keep the lock, read only the visible cells."""

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import GameEnded, MovePlayed, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Reconciler
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry, row_col
from sidecoach.vision.tracker import BoardTracker
from tests.synth import synthetic_board

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)
BOTTOMS = [Side.WHITE, Side.BLACK]


DIALOG_X0, DIALOG_Y0 = 300 - 40, 120 + 140  # columns a to e, rows 2 to 6 of the board
DIALOG_X1, DIALOG_Y1 = 300 + 330, 120 + 420


def _dialog_rect(bottom: Side) -> tuple[int, int, int, int]:
    """The dialog's rectangle, mirrored through the board's centre when black sits at the
    bottom so that it covers the same squares in either orientation."""
    if bottom == Side.WHITE:
        return DIALOG_X0, DIALOG_Y0, DIALOG_X1, DIALOG_Y1
    cx, cy = 300 + 256, 120 + 256
    return 2 * cx - DIALOG_X1, 2 * cy - DIALOG_Y1, 2 * cx - DIALOG_X0, 2 * cy - DIALOG_Y0


def with_dialog(
    frame: np.ndarray, dark: bool = False, alpha: float = 1.0, bottom: Side = Side.WHITE
) -> np.ndarray:
    """A game-over dialog with text covering the left half and middle of the board.

    `alpha` below 1 blends the dialog over the board, as chess.com does while it fades in.
    """
    if alpha < 1.0:
        full = with_dialog(frame, dark=dark, bottom=bottom)
        return cv2.addWeighted(full, alpha, frame, 1.0 - alpha, 0.0)
    fill = (50, 50, 50) if dark else (245, 245, 245)
    ink = (230, 230, 230) if dark else (40, 40, 40)
    width, height = DIALOG_X1 - DIALOG_X0 + 1, DIALOG_Y1 - DIALOG_Y0 + 1
    dialog = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(dialog, (0, 0), (width - 1, height - 1), fill, -1)
    cv2.rectangle(dialog, (0, 0), (width - 1, height - 1), (120, 120, 120), 2)
    for i in range(6):
        cv2.putText(
            dialog, "You won by checkmate", (15, 40 + 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 2
        )
    if bottom == Side.BLACK:
        # The same pixels over the same squares: the dialog turned with the board.
        dialog = cv2.rotate(dialog, cv2.ROTATE_180)
    out = frame.copy()
    x0, y0, x1, y1 = _dialog_rect(bottom)
    out[y0 : y1 + 1, x0 : x1 + 1] = dialog[: out.shape[0] - y0, : out.shape[1] - x0]
    return out


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_tracker_holds_lock_under_a_dialog(bottom: Side) -> None:
    for dark in (False, True):
        tracker = BoardTracker()
        clean = synthetic_board(bottom=bottom)
        assert tracker.update(clean) is not None
        covered = with_dialog(clean, dark=dark, bottom=bottom)
        board = None
        for _ in range(12):  # long enough for the periodic re-fit to run under the dialog
            board = tracker.update(covered)
            assert board is not None, "lock dropped under a partial dialog"
        assert tracker.detections == 1
        assert board is not None
        assert abs(board.region.x - 300) <= 2 and abs(board.cell_px - 64) <= 0.5


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_covered_cells_read_as_unknown_and_visible_cells_still_read(bottom: Side) -> None:
    clean = synthetic_board(bottom=bottom)
    model = ThemeModel.from_start_position(clean, GEOM, bottom)
    assert model is not None
    for dark in (False, True):
        readings = model.classify_frame(with_dialog(clean, dark=dark, bottom=bottom), GEOM)
        r, c = row_col(chess.C5, bottom)  # under the dialog
        assert readings[r][c].kind == "unknown"
        r, c = row_col(chess.A1, bottom)  # visible
        assert readings[r][c].symbol == "R"
        r, c = row_col(chess.H8, bottom)
        assert readings[r][c].symbol == "r"


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_mating_move_commits_while_the_game_over_dialog_covers_the_board(bottom: Side) -> None:
    """The last move is seen only with the dialog already up, as chess.com does it."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board, bottom=bottom), ts)
        ts += 0.1
    events = []
    for san in ("f3", "e5", "g4"):
        move = board.push_san(san)
        for _ in range(2):
            frame = synthetic_board(board=board, bottom=bottom, highlight=move)
            events += pipe.process(frame, ts)
            ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["f3", "e5", "g4"]

    mate = board.push_san("Qh4#")  # d8 and h4 both lie outside the dialog
    events = []
    for _ in range(3):
        frame = synthetic_board(board=board, bottom=bottom, highlight=mate)
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
        ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh4#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "0-1"


# White queen e5 to a5 is mate; e5, a5, and both kings lie under the dialog.
MIDDLE_MATE = "8/8/8/4Q3/8/k1K5/8/8 w - - 0 1"


def _reconciler_at(
    fen: str, bottom: Side = Side.WHITE
) -> tuple[ThemeModel, Reconciler, chess.Board]:
    start = synthetic_board(bottom=bottom)
    model = ThemeModel.from_start_position(start, GEOM, bottom)
    assert model is not None
    rec = Reconciler(bottom=bottom, user_side=bottom, settle_frames=2)
    rec.update(model.classify_frame(start, GEOM), 0.0)
    board = chess.Board(fen)
    rec.board = board.copy()
    rec._base_fen = fen
    for i in range(2):
        frame = synthetic_board(board=board, bottom=bottom)
        rec.update(model.classify_frame(frame, GEOM), 0.1 + 0.1 * i)
    return model, rec, board


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_mate_in_the_middle_commits_from_two_uncovered_frames(bottom: Side) -> None:
    """Two frames of the final position, then the dialog over the squares it came from.

    The covered cells keep their last stable reading, so the explanation found in the
    uncovered frames keeps settling under the dialog instead of turning ambiguous. Chess.com
    shows the final position for about ten frames before the dialog; two is the minimum,
    since a reading seen once may be the dialog fading in or a piece in flight.
    """
    model, rec, board = _reconciler_at(MIDDLE_MATE, bottom)
    mate = board.push_san("Qa5#")
    covered = with_dialog(
        synthetic_board(board=board, bottom=bottom, highlight=mate), dark=True, bottom=bottom
    )
    r, c = row_col(chess.E5, bottom)
    assert model.classify_frame(covered, GEOM)[r][c].covered
    final = model.classify_frame(synthetic_board(board=board, bottom=bottom, highlight=mate), GEOM)

    events = rec.update(final, 1.0)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.1 + 0.1 * i)
    assert not [e for e in events if isinstance(e, MovePlayed)], "one frame is not stable"

    model, rec, board = _reconciler_at(MIDDLE_MATE, bottom)
    board.push_san("Qa5#")
    events = rec.update(final, 1.0) + rec.update(final, 1.1)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qa5#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "1-0"


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_move_out_from_under_a_cover_waits_for_the_memory_to_expire(bottom: Side) -> None:
    """A remembered cell must not let a piece appear in two places; the memory is short."""
    model, rec, board = _reconciler_at(MIDDLE_MATE, bottom)
    rec.memory_frames = 4
    covered = with_dialog(synthetic_board(board=board, bottom=bottom), dark=True, bottom=bottom)
    for i in range(2):
        assert not rec.update(model.classify_frame(covered, GEOM), 1.0 + 0.1 * i)

    move = board.push_san("Qh5")  # e5 stays covered, h5 is visible
    covered = with_dialog(
        synthetic_board(board=board, bottom=bottom, highlight=move), dark=True, bottom=bottom
    )
    events = []
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert not [e for e in events if isinstance(e, MovePlayed)], "queen seen on e5 and h5"
    for i in range(4):
        events += rec.update(model.classify_frame(covered, GEOM), 1.5 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh5"]


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_fading_dialog_reads_as_covered_and_keeps_the_memory_clean(bottom: Side) -> None:
    """Half-transparent dialog frames hide the pieces without reading as junk.

    Chess.com fades the dialog in over a few frames. Those frames must count as covered,
    or the junk read through the fade becomes the remembered reading under the dialog.
    """
    model, rec, board = _reconciler_at(MIDDLE_MATE, bottom)
    mate = board.push_san("Qa5#")
    final = synthetic_board(board=board, bottom=bottom, highlight=mate)
    fading = model.classify_frame(with_dialog(final, dark=True, alpha=0.5, bottom=bottom), GEOM)
    r, c = row_col(chess.E5, bottom)
    assert fading[r][c].kind == "unknown" and fading[r][c].covered
    r, c = row_col(chess.H1, bottom)
    assert not fading[r][c].covered

    events = rec.update(model.classify_frame(final, GEOM), 0.9)
    events += rec.update(model.classify_frame(final, GEOM), 1.0)
    for i, alpha in enumerate((0.3, 0.6, 1.0, 1.0, 1.0)):
        frame = with_dialog(final, dark=True, alpha=alpha, bottom=bottom)
        events += rec.update(model.classify_frame(frame, GEOM), 1.1 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qa5#"]


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_nothing_reads_as_hovered_under_a_dialog(bottom: Side) -> None:
    """A dialog edge cutting through a pawn must not pass for a drag in progress."""
    clean = synthetic_board(bottom=bottom)
    model = ThemeModel.from_start_position(clean, GEOM, bottom)
    assert model is not None
    for dark in (False, True):
        readings = model.classify_frame(with_dialog(clean, dark=dark, bottom=bottom), GEOM)
        hovered = [(r, c) for r in range(8) for c in range(8) if readings[r][c].hover]
        assert hovered == []


def _opaque_cells(frame: np.ndarray, squares: list[chess.Square]) -> np.ndarray:
    """An opaque dark tooltip over exactly these squares."""
    out = frame.copy()
    for square in squares:
        x0, y0, x1, y1 = GEOM.cell_box(*row_col(square, Side.WHITE))
        out[y0:y1, x0:x1] = (60, 60, 60)
    return out


def test_junk_next_to_a_strip_is_not_relabelled_but_next_to_a_dialog_it_is() -> None:
    """The relabel of junk cells beside a cover arms on a dialog-sized block, never on a
    strip: an opaque tooltip over one rank must not hide a drag going on next to it."""
    model = ThemeModel.from_start_position(synthetic_board(), GEOM, Side.WHITE)
    assert model is not None
    board = chess.Board()
    drag = (chess.D7, chess.D6, 0.1, 0.1)
    strip = [chess.A5, chess.B5, chess.C5, chess.D5, chess.E5, chess.F5]
    r, c = row_col(chess.D6, Side.WHITE)

    readings = model.classify_frame(
        _opaque_cells(synthetic_board(board=board, dragging=drag), strip), GEOM
    )
    assert all(
        readings[row_col(sq, Side.WHITE)[0]][row_col(sq, Side.WHITE)[1]].covered for sq in strip
    )
    assert readings[r][c].hover and not readings[r][c].covered, readings[r][c]

    far = (chess.H7, chess.H6, 0.1, 0.1)
    readings = model.classify_frame(
        _opaque_cells(synthetic_board(board=board, dragging=far), strip), GEOM
    )
    r6, c6 = row_col(chess.H6, Side.WHITE)
    assert readings[r6][c6].hover and not readings[r6][c6].covered

    block = [
        chess.A4,
        chess.B4,
        chess.C4,
        chess.A5,
        chess.B5,
        chess.C5,
        chess.A6,
        chess.B6,
        chess.C6,
    ]
    beside = (chess.D7, chess.D5, 0.1, 0.1)  # hovering d5, beside the block
    readings = model.classify_frame(
        _opaque_cells(synthetic_board(board=board, dragging=beside), block), GEOM
    )
    r5, c5 = row_col(chess.D5, Side.WHITE)
    assert readings[r5][c5].covered and not readings[r5][c5].hover, readings[r5][c5]


@pytest.mark.parametrize("bottom", BOTTOMS)
def test_a_dialog_that_stays_over_a_live_board_ends_the_game(bottom: Side) -> None:
    """The third live game ended with premoves: five plies in a second and a half, then the
    game-over dialog, and the reconciler had explained the screen with a shorter path that
    left its board one move short of the mate. Nothing under the dialog could tell it
    more, so the coach thought the game was on. The dialog itself is the site's word:
    up for two seconds over a live board, it ends the game, with the result unknown when
    the board is not terminal -- and the user's word can still settle the result."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board, bottom=bottom), ts)
        ts += 0.1
    events = []
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, bottom=bottom, highlight=move), ts)
            ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["e4", "e5"]
    assert pipe.reconciler is not None and not pipe.reconciler.finished

    # A dialog that flashes past (a toast fading through) ends nothing.
    frame = synthetic_board(board=board, bottom=bottom)
    events = []
    for _ in range(5):
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
        ts += 0.1
    events += pipe.process(frame, ts)
    ts += 0.1
    assert not [e for e in events if isinstance(e, GameEnded)]
    assert not pipe.reconciler.finished

    # One that stays does, once, with the result not read; the game does not end twice.
    events = []
    for _ in range(30):
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
        ts += 0.1
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].result == "*" and "dialog" in ended[0].reason
    assert pipe.reconciler.finished

    # The user says who won, and the result is settled; a second word changes nothing.
    settled = pipe.end_game("1-0", "opponent resigned or lost on time", ts)
    assert [e.result for e in settled if isinstance(e, GameEnded)] == ["1-0"]
    assert pipe.end_game("0-1", "changed my mind", ts) == []
