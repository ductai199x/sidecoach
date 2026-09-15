"""A dialog over part of the board: keep the lock, read only the visible cells."""

import chess
import cv2
import numpy as np

from sidecoach.events import GameEnded, MovePlayed, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Reconciler
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry, row_col
from sidecoach.vision.tracker import BoardTracker
from tests.synth import synthetic_board

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


DIALOG_X0, DIALOG_Y0 = 300 - 40, 120 + 140  # columns a to e, rows 2 to 6 of the board
DIALOG_X1, DIALOG_Y1 = 300 + 330, 120 + 420


def with_dialog(frame: np.ndarray, dark: bool = False, alpha: float = 1.0) -> np.ndarray:
    """A game-over dialog with text covering the left half and middle of the board.

    `alpha` below 1 blends the dialog over the board, as chess.com does while it fades in.
    """
    if alpha < 1.0:
        full = with_dialog(frame, dark=dark)
        return cv2.addWeighted(full, alpha, frame, 1.0 - alpha, 0.0)
    out = frame.copy()
    fill = (50, 50, 50) if dark else (245, 245, 245)
    ink = (230, 230, 230) if dark else (40, 40, 40)
    cv2.rectangle(out, (DIALOG_X0, DIALOG_Y0), (DIALOG_X1, DIALOG_Y1), fill, -1)
    cv2.rectangle(out, (DIALOG_X0, DIALOG_Y0), (DIALOG_X1, DIALOG_Y1), (120, 120, 120), 2)
    for i in range(6):
        cv2.putText(
            out,
            "You won by checkmate",
            (DIALOG_X0 + 15, DIALOG_Y0 + 40 + 40 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            ink,
            2,
        )
    return out


def test_tracker_holds_lock_under_a_dialog() -> None:
    for dark in (False, True):
        tracker = BoardTracker()
        clean = synthetic_board()
        assert tracker.update(clean) is not None
        covered = with_dialog(clean, dark=dark)
        board = None
        for _ in range(12):  # long enough for the periodic re-fit to run under the dialog
            board = tracker.update(covered)
            assert board is not None, "lock dropped under a partial dialog"
        assert tracker.detections == 1
        assert board is not None
        assert abs(board.region.x - 300) <= 2 and abs(board.cell_px - 64) <= 0.5


def test_covered_cells_read_as_unknown_and_visible_cells_still_read() -> None:
    clean = synthetic_board()
    model = ThemeModel.from_start_position(clean, GEOM, Side.WHITE)
    assert model is not None
    for dark in (False, True):
        readings = model.classify_frame(with_dialog(clean, dark=dark), GEOM)
        r, c = row_col(chess.C5, Side.WHITE)  # under the dialog
        assert readings[r][c].kind == "unknown"
        r, c = row_col(chess.A1, Side.WHITE)  # visible
        assert readings[r][c].symbol == "R"
        r, c = row_col(chess.H8, Side.WHITE)
        assert readings[r][c].symbol == "r"


def test_mating_move_commits_while_the_game_over_dialog_covers_the_board() -> None:
    """The last move is seen only with the dialog already up, as chess.com does it."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    events = []
    for san in ("f3", "e5", "g4"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["f3", "e5", "g4"]

    mate = board.push_san("Qh4#")  # d8 and h4 both lie outside the dialog
    events = []
    for _ in range(3):
        frame = with_dialog(synthetic_board(board=board, highlight=mate), dark=True)
        events += pipe.process(frame, ts)
        ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh4#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "0-1"


# White queen e5 to a5 is mate; e5, a5, and both kings lie under the dialog.
MIDDLE_MATE = "8/8/8/4Q3/8/k1K5/8/8 w - - 0 1"


def _reconciler_at(fen: str) -> tuple[ThemeModel, Reconciler, chess.Board]:
    start = synthetic_board()
    model = ThemeModel.from_start_position(start, GEOM, Side.WHITE)
    assert model is not None
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE, settle_frames=2)
    rec.update(model.classify_frame(start, GEOM), 0.0)
    board = chess.Board(fen)
    rec.board = board.copy()
    rec._base_fen = fen
    for i in range(2):
        rec.update(model.classify_frame(synthetic_board(board=board), GEOM), 0.1 + 0.1 * i)
    return model, rec, board


def test_mate_in_the_middle_commits_from_two_uncovered_frames() -> None:
    """Two frames of the final position, then the dialog over the squares it came from.

    The covered cells keep their last stable reading, so the explanation found in the
    uncovered frames keeps settling under the dialog instead of turning ambiguous. Chess.com
    shows the final position for about ten frames before the dialog; two is the minimum,
    since a reading seen once may be the dialog fading in or a piece in flight.
    """
    model, rec, board = _reconciler_at(MIDDLE_MATE)
    mate = board.push_san("Qa5#")
    covered = with_dialog(synthetic_board(board=board, highlight=mate), dark=True)
    r, c = row_col(chess.E5, Side.WHITE)
    assert model.classify_frame(covered, GEOM)[r][c].covered
    final = model.classify_frame(synthetic_board(board=board, highlight=mate), GEOM)

    events = rec.update(final, 1.0)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.1 + 0.1 * i)
    assert not [e for e in events if isinstance(e, MovePlayed)], "one frame is not stable"

    model, rec, board = _reconciler_at(MIDDLE_MATE)
    board.push_san("Qa5#")
    events = rec.update(final, 1.0) + rec.update(final, 1.1)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qa5#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "1-0"


def test_move_out_from_under_a_cover_waits_for_the_memory_to_expire() -> None:
    """A remembered cell must not let a piece appear in two places; the memory is short."""
    model, rec, board = _reconciler_at(MIDDLE_MATE)
    rec.memory_frames = 4
    covered = with_dialog(synthetic_board(board=board), dark=True)
    for i in range(2):
        assert not rec.update(model.classify_frame(covered, GEOM), 1.0 + 0.1 * i)

    move = board.push_san("Qh5")  # e5 stays covered, h5 is visible
    covered = with_dialog(synthetic_board(board=board, highlight=move), dark=True)
    events = []
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert not [e for e in events if isinstance(e, MovePlayed)], "queen seen on e5 and h5"
    for i in range(4):
        events += rec.update(model.classify_frame(covered, GEOM), 1.5 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh5"]


def test_fading_dialog_reads_as_covered_and_keeps_the_memory_clean() -> None:
    """Half-transparent dialog frames hide the pieces without reading as junk.

    Chess.com fades the dialog in over a few frames. Those frames must count as covered,
    or the junk read through the fade becomes the remembered reading under the dialog.
    """
    model, rec, board = _reconciler_at(MIDDLE_MATE)
    mate = board.push_san("Qa5#")
    final = synthetic_board(board=board, highlight=mate)
    fading = model.classify_frame(with_dialog(final, dark=True, alpha=0.5), GEOM)
    r, c = row_col(chess.E5, Side.WHITE)
    assert fading[r][c].kind == "unknown" and fading[r][c].covered
    r, c = row_col(chess.H1, Side.WHITE)
    assert not fading[r][c].covered

    events = rec.update(model.classify_frame(final, GEOM), 0.9)
    events += rec.update(model.classify_frame(final, GEOM), 1.0)
    for i, alpha in enumerate((0.3, 0.6, 1.0, 1.0, 1.0)):
        frame = with_dialog(final, dark=True, alpha=alpha)
        events += rec.update(model.classify_frame(frame, GEOM), 1.1 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qa5#"]


def test_nothing_reads_as_hovered_under_a_dialog() -> None:
    """A dialog edge cutting through a pawn must not pass for a drag in progress."""
    clean = synthetic_board()
    model = ThemeModel.from_start_position(clean, GEOM, Side.WHITE)
    assert model is not None
    for dark in (False, True):
        readings = model.classify_frame(with_dialog(clean, dark=dark), GEOM)
        hovered = [(r, c) for r in range(8) for c in range(8) if readings[r][c].hover]
        assert hovered == []
