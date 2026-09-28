"""A dialog over part of the board: keep the lock, read only the visible cells.

The dialog is mirrored through the board's centre when black is at the bottom, so it covers
the same squares either way up: the reconciler is given the same squares to read, and its
cases run with white at the bottom. Reading the board under the dialog the other way up is
the recogniser's, and runs both ways."""

import chess
import numpy as np
import pytest

from sidecoach.events import GameEnded, Side
from sidecoach.state.pipeline import DIALOG_SECONDS, GamePipeline
from sidecoach.vision.squares import row_col
from sidecoach.vision.tracker import BoardTracker
from tests.games import recogniser, reconciler_at, sans_of
from tests.synth import GEOM, synthetic_board, with_dialog

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision

BOTTOMS = [Side.WHITE, Side.BLACK]


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
    """Under the dialog a cell is unknown; beside it the pieces read as themselves; and
    nothing reads as hovered, since a dialog edge cutting through a pawn must not pass for
    a drag in progress (test_nothing_reads_as_hovered_under_a_dialog)."""
    clean = synthetic_board(bottom=bottom)
    model = recogniser(bottom)
    for dark in (False, True):
        readings = model.classify_frame(with_dialog(clean, dark=dark, bottom=bottom), GEOM)
        r, c = row_col(chess.C5, bottom)  # under the dialog
        assert readings[r][c].kind == "unknown"
        r, c = row_col(chess.A1, bottom)  # visible
        assert readings[r][c].symbol == "R"
        r, c = row_col(chess.H8, bottom)
        assert readings[r][c].symbol == "r"
        hovered = [(r, c) for r in range(8) for c in range(8) if readings[r][c].hover]
        assert hovered == []


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
            frame = synthetic_board(board=board, highlight=move)
            events += pipe.process(frame, ts)
            ts += 0.1
    assert sans_of(events) == ["f3", "e5", "g4"]

    mate = board.push_san("Qh4#")  # d8 and h4 both lie outside the dialog
    events = []
    for _ in range(3):
        frame = synthetic_board(board=board, highlight=mate)
        events += pipe.process(with_dialog(frame, dark=True), ts)
        ts += 0.1
    assert sans_of(events) == ["Qh4#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "0-1"


# White queen e5 to a5 is mate; e5, a5, and both kings lie under the dialog.
MIDDLE_MATE = "8/8/8/4Q3/8/k1K5/8/8 w - - 0 1"


def test_mate_in_the_middle_commits_from_two_uncovered_frames() -> None:
    """Two frames of the final position, then the dialog over the squares it came from.

    The covered cells keep their last stable reading, so the explanation found in the
    uncovered frames keeps settling under the dialog instead of turning ambiguous. Chess.com
    shows the final position for about ten frames before the dialog; two is the minimum,
    since a reading seen once may be the dialog fading in or a piece in flight.
    """
    model, rec, board = reconciler_at(MIDDLE_MATE)
    mate = board.push_san("Qa5#")
    covered = with_dialog(synthetic_board(board=board, highlight=mate), dark=True)
    r, c = row_col(chess.E5, Side.WHITE)
    assert model.classify_frame(covered, GEOM)[r][c].covered
    final = model.classify_frame(synthetic_board(board=board, highlight=mate), GEOM)

    events = rec.update(final, 1.0)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.1 + 0.1 * i)
    assert not sans_of(events), "one frame is not stable"

    model, rec, board = reconciler_at(MIDDLE_MATE)
    board.push_san("Qa5#")
    events = rec.update(final, 1.0) + rec.update(final, 1.1)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert sans_of(events) == ["Qa5#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "1-0"


def test_move_out_from_under_a_cover_waits_for_the_memory_to_expire() -> None:
    """A remembered cell must not let a piece appear in two places; the memory is short."""
    model, rec, board = reconciler_at(MIDDLE_MATE)
    rec.memory_frames = 4
    covered = with_dialog(synthetic_board(board=board), dark=True)
    for i in range(2):
        assert not rec.update(model.classify_frame(covered, GEOM), 1.0 + 0.1 * i)

    move = board.push_san("Qh5")  # e5 stays covered, h5 is visible
    covered = with_dialog(synthetic_board(board=board, highlight=move), dark=True)
    events = []
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert not sans_of(events), "queen seen on e5 and h5"
    for i in range(4):
        events += rec.update(model.classify_frame(covered, GEOM), 1.5 + 0.1 * i)
    assert sans_of(events) == ["Qh5"]


def test_fading_dialog_reads_as_covered_and_keeps_the_memory_clean() -> None:
    """Half-transparent dialog frames hide the pieces without reading as junk.

    Chess.com fades the dialog in over a few frames. Those frames must count as covered,
    or the junk read through the fade becomes the remembered reading under the dialog.
    """
    model, rec, board = reconciler_at(MIDDLE_MATE)
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
    assert sans_of(events) == ["Qa5#"]


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
    model = recogniser(Side.WHITE)
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


def test_a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing() -> None:
    """A dialog is not a result: sites put dialogs over the board for many reasons. One
    that stays marks the screen COVERED, the game stays open, and the user's word or the
    next game ends it. One that flashes past counts for nothing."""
    from sidecoach.events import ScreenState, ScreenStateChanged

    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    events = []
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert sans_of(events) == ["e4", "e5"]
    assert pipe.reconciler is not None and not pipe.reconciler.finished

    frame = synthetic_board(board=board)
    events = []
    for _ in range(5):
        events += pipe.process(with_dialog(frame, dark=True), ts)
        ts += 0.1
    events += pipe.process(frame, ts)
    ts += 0.1
    assert not [e for e in events if isinstance(e, (GameEnded, ScreenStateChanged))]

    events = []
    for _ in range(30):
        events += pipe.process(with_dialog(frame, dark=True), ts)
        ts += 0.1
    states = [e.state for e in events if isinstance(e, ScreenStateChanged)]
    assert states == [ScreenState.COVERED]
    assert not [e for e in events if isinstance(e, GameEnded)]
    assert not pipe.reconciler.finished
    # The cover lifts: live again, same game.
    events = pipe.process(frame, ts)
    assert [e.state for e in events if isinstance(e, ScreenStateChanged)] == [ScreenState.LIVE_GAME]
    # The user says how it ended.
    settled = pipe.end_game("1-0", "opponent resigned or lost on time", ts)
    assert [e.result for e in settled if isinstance(e, GameEnded)] == ["1-0"]
    assert pipe.end_game("0-1", "changed my mind", ts) == []


def test_the_dialog_confirms_a_mate_read_once() -> None:
    """The opponent's move and the user's premoved mate land in one frame, read once, clean,
    and the dialog comes up over the opponent's move before a second frame: nothing under
    it can confirm the story, and what it leaves in sight -- the queen gone from h5 and
    standing on f7 -- reads as the premove still queued, so the story read once waits. The
    dialog that stays is the confirmation the settling needed
    (`GamePipeline._dialog_over_the_board`, after DIALOG_SECONDS), and the game ends on the
    board's own result. The third live game ended so: five premoved plies in a second and a
    half, then the dialog.

    The dialog over the mate's own squares confirms nothing: the memory of covered cells
    brings back every other square as it stood, the frame agrees with the position before
    the mate, and a story that nothing disagrees with is dropped (`Reconciler.update`). And
    the mate read twice, or with the opponent's move in sight, settles on its own frames
    (test_mating_move_commits_while_the_game_over_dialog_covers_the_board)."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    events = []
    for san in ("e4", "e5", "Qh5", "Nc6", "Bc4"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert sans_of(events) == ["e4", "e5", "Qh5", "Nc6", "Bc4"]
    board.push_san("Nd4")  # c6 and d4 lie under the dialog
    mate = board.push_san("Qxf7#")  # h5 and f7 do not
    frame = synthetic_board(board=board, highlight=mate)
    events = pipe.process(frame, ts)  # one clean frame
    ts += 0.1
    assert not sans_of(events)
    up = ts
    for _ in range(30):
        new = pipe.process(with_dialog(frame, dark=True), ts)
        if sans_of(new):
            assert ts - up >= DIALOG_SECONDS, "settled under the dialog, not confirmed by it"
        events += new
        ts += 0.1
    assert sans_of(events) == ["Nd4", "Qxf7#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "1-0"
