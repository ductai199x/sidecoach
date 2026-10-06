"""The desktop's reconciler tests the state and pipeline audit folded (2026-09-28, desktop
630f739..af1c0ba), each still recorded by `make_reconcile_fixtures.py` under the name it
had, so the fixture keeps every case.

A test folded into another, merged into a table, renamed, or played only with fewer of its
parameters there is played here as it stood: the same frames, the same assertions. The
desktop's own helpers drive them (the tests' `_setup`, `Session`, `_warm`, ..., and the
shared tests/games.py and tests/synth.py: `Game`, `reconciler_at`, `with_dialog`, ...), so a
helper renamed or moved there fails here at once.

`FOLDED` maps each desktop module to its folded cases: the name recorded (with the
parameter label it had), the function, and its arguments.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chess

from sidecoach.events import (
    GameEnded,
    GameStarted,
    MovePlayed,
    Resync,
    ScreenState,
    ScreenStateChanged,
    Side,
)
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.pieces import CellReading
from sidecoach.vision.squares import row_col
from sidecoach.vision.theme_cache import ThemeCache
from tests import test_cold_start as ct
from tests import test_occlusion as ot
from tests import test_pipeline as pt
from tests import test_premoves as qt
from tests import test_reconcile as rt
from tests import test_scenarios as st
from tests.games import Game, reconciler_at, sans_of
from tests.synth import GEOM, synthetic_board, with_dialog

# ---- tests.test_reconcile ---------------------------------------------------------------


def _selected_piece_does_not_block_the_opponents_move() -> None:
    """The bot's move is on the board and the user has already clicked their next piece."""
    model, rec = rt._setup()
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    rt._play(model, rec, board, ["e4", "e5", "Nf3"])
    reply = board.push_san("Nc6")
    frame = synthetic_board(board=board, highlight=reply)
    readings = model.classify_frame(frame, GEOM)
    r, c = row_col(chess.F1, Side.WHITE)  # the user selected the f1 bishop
    old = readings[r][c]
    readings[r][c] = CellReading(
        old.kind, old.symbol, old.colour, old.confidence, old.fg_fraction, old.luminance, True
    )
    events = rec.update(readings, 1.0) + rec.update(readings, 1.1)
    assert sans_of(events) == ["Nc6"]


def _black_at_bottom_user_black() -> None:
    model, rec = rt._setup(Side.BLACK)
    board = chess.Board()
    rt._feed(model, rec, board, Side.BLACK, 0.0)
    for san in ("e4", "c5", "Nf3"):
        move = board.push_san(san)
        rt._feed(model, rec, board, Side.BLACK, 0.1, highlight=move)
        assert sans_of(rt._feed(model, rec, board, Side.BLACK, 0.2, highlight=move)) == [san]
    assert rec.board.fen() == board.fen()


def _position_resync_after_a_finished_game_is_live_again() -> None:
    model, rec = rt._setup()
    rec.resync_hold_frames = 3
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    rt._play(model, rec, board, ["f3", "e5", "g4", "Qh4#"])
    assert rec.finished
    later = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    events = []
    for i in range(6):
        events += rt._feed(model, rec, later, Side.WHITE, 2.0 + 0.1 * i)
    # A new game, and no Resync of the same position after it (desktop 92d2b15).
    assert any(isinstance(e, GameStarted) for e in events)
    assert not any(isinstance(e, Resync) for e in events)
    assert rec.board.board_fen() == later.board_fen()
    assert not rec.finished


def _hover_outline_alone_blocks_a_commit() -> None:
    """A piece exactly centred but the hovered square still outlined: not dropped yet."""
    model, rec = rt._setup()
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    for i in range(4):
        drag = (chess.E2, chess.E4, 0.0, 0.0)
        assert not sans_of(rt._feed(model, rec, board, Side.WHITE, 0.1 * i, dragging=drag))
    move = board.push_san("e4")
    events = []
    for ts in (1.0, 1.1):
        events = rt._feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert sans_of(events) == ["e4"]


def _think_time_comes_from_the_frame_timestamps() -> None:
    model, rec = rt._setup()
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    events = []
    ts = 0.0
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            events += rt._feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert [e.think_ms for e in events if isinstance(e, MovePlayed)] == [200, 200]


def _checkmate_ends_the_game_once() -> None:
    model, rec = rt._setup()
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    events = []
    ts = 0.0
    for san in ("f3", "e5", "g4", "Qh4#"):
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            events += rt._feed(model, rec, board, Side.WHITE, ts, highlight=move)
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].result == "0-1" and ended[0].pgn == "1. f3 e5 2. g4 Qh4#"
    assert rec.finished
    # The next game starts: no second end event for the finished game.
    events = []
    for ts in (2.0, 2.1):
        events += rt._feed(model, rec, chess.Board(), Side.WHITE, ts)
    assert not any(isinstance(e, GameEnded) for e in events)
    assert any(isinstance(e, GameStarted) for e in events)
    assert not rec.finished


def _opponent_move_while_browsing_snaps_to_latest() -> None:
    model, rec = rt._setup()
    board = chess.Board()
    rt._feed(model, rec, board, Side.WHITE, 0.0)
    rt._play(model, rec, board, ["e4", "e5", "Nf3"])
    earlier = board.copy()
    earlier.pop()
    earlier.pop()
    for ts in (1.0, 1.1):
        rt._feed(model, rec, earlier, Side.WHITE, ts)
    assert rec.viewing == 1
    # The bot replies: the site jumps to the latest position plus the new move.
    reply = board.push_san("Nc6")
    events = rt._feed(model, rec, board, Side.WHITE, 2.0, highlight=reply)
    events += rt._feed(model, rec, board, Side.WHITE, 2.1, highlight=reply)
    assert sans_of(events) == ["Nc6"]
    assert rec.viewing is None
    assert rec.board.fen() == board.fen()


# ---- tests.test_pipeline ----------------------------------------------------------------


def _new_game_same_orientation_new_theme_rebuilds_recogniser() -> None:
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(2):
        pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert pipe.reconciler is not None and len(pipe.reconciler.board.move_stack) == 2

    board = chess.Board()
    events = []
    theme = {"light": pt.BROWN_LIGHT, "dark": pt.BROWN_DARK}
    for _ in range(3):
        events += pipe.process(synthetic_board(board=board, **theme), ts)
        ts += 0.1
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].pgn == "1. e4 e5"
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 2

    events = []
    for san in ("c4", "c5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move, **theme), ts)
            ts += 0.1
    assert sans_of(events) == ["c4", "c5"]


def _board_flipped_before_the_first_move_keeps_the_game() -> None:
    """Watching the pre-game board as white, then paired as black: the board flips."""
    pipe = GamePipeline()
    events, ts = pt._start_frames(pipe, 0.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 1 and pipe.bottom == Side.WHITE
    assert pipe.reconciler is not None
    game_id = pipe.reconciler.game_id
    events, ts = pt._start_frames(pipe, ts, frames=4, bottom=Side.BLACK)
    assert pipe.bootstraps == 2 and pipe.bottom == Side.BLACK
    assert not any(isinstance(e, GameEnded) for e in events)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert (
        len(started) == 1 and started[0].game_id == game_id and started[0].user_side == Side.BLACK
    )
    board = chess.Board()
    events = []
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(
                synthetic_board(board=board, bottom=Side.BLACK, highlight=move), ts
            )
            ts += 0.1
    assert sans_of(events) == ["e4", "e5"]
    assert pipe.reconciler.game_id == game_id


def _theme_changed_before_the_first_move_keeps_the_game() -> None:
    pipe = GamePipeline()
    _, ts = pt._start_frames(pipe, 0.0)
    assert pipe.reconciler is not None
    game_id = pipe.reconciler.game_id
    theme = {"light": pt.BROWN_LIGHT, "dark": pt.BROWN_DARK}
    events, ts = pt._start_frames(pipe, ts, frames=4, **theme)
    assert pipe.bootstraps == 2 and not any(isinstance(e, GameEnded) for e in events)
    board = chess.Board()
    events = []
    move = board.push_san("d4")
    for _ in range(2):
        events += pipe.process(synthetic_board(board=board, highlight=move, **theme), ts)
        ts += 0.1
    assert sans_of(events) == ["d4"] and pipe.reconciler.game_id == game_id


# ---- tests.test_scenarios ---------------------------------------------------------------


def _window_moved_mid_game() -> None:
    s = st.Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    s.x0, s.y0 = 520, 210  # the window was dragged elsewhere
    s.settle(6)  # the tracker loses the old lock and re-locks; no move is invented
    assert s.played == ["e4", "e5"]
    assert s.play("Nf3", "Nc6") == ["Nf3", "Nc6"]
    assert s.pipe.reconciler is not None and s.pipe.reconciler.board.fen() == s.board.fen()


def _board_scrolled_partly_off_screen_and_back() -> None:
    s = st.Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    s.y0 = -200  # the page scrolled: the top rows are off the screen
    s.settle(8)  # no crash, nothing invented
    assert s.played == ["e4", "e5"]
    s.y0 = 120
    s.settle(6)
    assert s.play("Nf3") == ["Nf3"]


def _catch_up_after_a_gap_of_a_few_plies(plies: int) -> None:
    """The board was not visible for a moment; up to four plies are searched for."""
    s = st.Session()
    s.start()
    assert s.play("e4", "e5") == ["e4", "e5"]
    line = ["Nf3", "Nc6", "Bb5", "a6"][:plies]
    for san in line:
        s.board.push_san(san)
    assert s.settle(6) == line


# ---- tests.test_occlusion: the reconciler's cases with black at the bottom, and the
# dialog test as it stood (its mate settled on its own frames: the test now plays the
# mate the dialog confirms) ---------------------------------------------------------------


def _mating_move_commits_while_the_game_over_dialog_covers_the_board(bottom: Side) -> None:
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


def _mate_in_the_middle_commits_from_two_uncovered_frames(bottom: Side) -> None:
    """Two frames of the final position, then the dialog over the squares it came from."""
    model, rec, board = reconciler_at(ot.MIDDLE_MATE, bottom)
    mate = board.push_san("Qa5#")
    covered = with_dialog(
        synthetic_board(board=board, bottom=bottom, highlight=mate), dark=True, bottom=bottom
    )
    r, c = row_col(chess.E5, bottom)
    assert model.classify_frame(covered, GEOM)[r][c].covered
    final = model.classify_frame(
        synthetic_board(board=board, bottom=bottom, highlight=mate), GEOM
    )

    events = rec.update(final, 1.0)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.1 + 0.1 * i)
    assert not [e for e in events if isinstance(e, MovePlayed)], "one frame is not stable"

    model, rec, board = reconciler_at(ot.MIDDLE_MATE, bottom)
    board.push_san("Qa5#")
    events = rec.update(final, 1.0) + rec.update(final, 1.1)
    for i in range(3):
        events += rec.update(model.classify_frame(covered, GEOM), 1.2 + 0.1 * i)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qa5#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "1-0"


def _move_out_from_under_a_cover_waits_for_the_memory_to_expire(bottom: Side) -> None:
    """A remembered cell must not let a piece appear in two places; the memory is short."""
    model, rec, board = reconciler_at(ot.MIDDLE_MATE, bottom)
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


def _fading_dialog_reads_as_covered_and_keeps_the_memory_clean(bottom: Side) -> None:
    """Half-transparent dialog frames hide the pieces without reading as junk."""
    model, rec, board = reconciler_at(ot.MIDDLE_MATE, bottom)
    mate = board.push_san("Qa5#")
    final = synthetic_board(board=board, bottom=bottom, highlight=mate)
    fading = model.classify_frame(
        with_dialog(final, dark=True, alpha=0.5, bottom=bottom), GEOM
    )
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


def _a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing(
    bottom: Side,
) -> None:
    """A dialog is not a result: one that stays marks the screen COVERED, the game stays
    open, and the user's word or the next game ends it. One that flashes past counts for
    nothing."""
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

    frame = synthetic_board(board=board, bottom=bottom)
    events = []
    for _ in range(5):
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
        ts += 0.1
    events += pipe.process(frame, ts)
    ts += 0.1
    assert not [e for e in events if isinstance(e, (GameEnded, ScreenStateChanged))]

    events = []
    for _ in range(30):
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
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


def _the_dialog_over_a_mate_read_once_as_it_stood(bottom: Side) -> None:
    """Fool's mate read in one clean frame, then the dialog: d8 and h4 lie outside it, and
    the mate settles on the dialog frames."""
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
            events += pipe.process(synthetic_board(board=board, bottom=bottom, highlight=move), ts)
            ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["f3", "e5", "g4"]
    mate = board.push_san("Qh4#")
    frame = synthetic_board(board=board, bottom=bottom, highlight=mate)
    events = pipe.process(frame, ts)  # one clean frame
    ts += 0.1
    assert not [e for e in events if isinstance(e, MovePlayed)]
    for _ in range(30):
        events += pipe.process(with_dialog(frame, dark=True, bottom=bottom), ts)
        ts += 0.1
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Qh4#"]
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].result == "0-1"


# ---- tests.test_premoves: as they stood, through the whole pipeline ---------------------


def _a_five_hop_knight_queue_is_undone_hop_by_hop() -> None:
    g = Game(Side.WHITE, fen="7k/pp4pp/8/8/8/8/8/KN4R1 w - - 0 1", pipeline=True)
    qt._run_queue(
        g,
        "Nc3",
        qt._uci("c3e4", "e4f6", "f6d5", "d5e7"),
        qt._uci("a6", "a5", "a4", "a3"),
        frames_per_state=2,
    )
    assert g.sans == ["Nc3", "a6", "Ne4", "a5", "Nf6", "a4", "Nd5", "a3", "Ne7"], g.sans


def _a_king_walk_queue_of_five_steps() -> None:
    """A king can only step, so a five-square queue is five red squares in a row."""
    g = Game(Side.WHITE, fen="7k/pp4pp/8/8/8/8/8/K5R1 w - - 0 1", pipeline=True)
    qt._run_queue(
        g,
        "Kb2",
        qt._uci("b2c3", "c3d4", "d4e5", "e5e6"),
        qt._uci("a6", "a5", "a4", "a3"),
        frames_per_state=2,
    )
    assert g.sans == ["Kb2", "a6", "Kc3", "a5", "Kd4", "a4", "Ke5", "a3", "Ke6"], g.sans


def _a_pawn_queue_up_the_board() -> None:
    g = Game(Side.WHITE, fen="7k/p5pp/8/8/8/8/3P4/K5R1 w - - 0 1", pipeline=True)
    qt._run_queue(
        g,
        "d4",
        qt._uci("d4d5", "d5d6", "d6d7"),
        qt._uci("a6", "a5", "a4"),
        frames_per_state=2,
    )
    assert g.sans == ["d4", "a6", "d5", "a5", "d6", "a4", "d7"], g.sans


def _the_opponent_moves_onto_the_queued_square_and_the_premove_captures_it() -> None:
    g = Game(Side.WHITE, pipeline=True)
    g.play("e4", "e5", "Nf3", "Nc6", "Bc4")
    g.show(frames=3, premoves=g.queue(qt._uci("f3g5")))
    g.board.push_san("Nf6")  # not onto g5; a plain reply
    executed = g.board.push_san("Ng5")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Nf6", "Ng5"], g.sans
    g.show(frames=3, highlight=executed, premoves=g.queue(qt._uci("g5e6")))  # Ne6 queued
    g.board.push_san("Be7")
    g.show(frames=1, highlight=chess.Move.from_uci("f8e7"), premoves=g.queue(qt._uci("g5e6")))
    executed = g.board.push_san("Ne6")
    g.show(frames=3, highlight=executed)
    assert g.sans[-2:] == ["Be7", "Ne6"], g.sans


def _the_users_king_in_check_on_a_red_square_is_not_a_ghost() -> None:
    """Chess.com paints the checked king's square red; with a queue on the board too, the
    king must not be mistaken for a queued piece and moved to some red square."""
    g = Game(Side.WHITE, pipeline=True)
    g.play("e4", "e5", "f3", "Nc6", "Nc3")
    g.show(frames=3, premoves=g.queue(qt._uci("c3d5")))  # Nd5 queued
    check = g.board.push_san("Qh4+")  # check clears the queue; the king's square is red
    g.show(frames=3, highlight=check, red_marks=(chess.E1,))
    assert g.sans[-1:] == ["Qh4+"], g.sans
    g.play("g3", red_marks=())
    assert g.sans[-2:] == ["Qh4+", "g3"], g.sans


def _black_runs_a_long_queue_to_mate(frames_per_state: int) -> None:
    """The same ladder from Black's side, black at the bottom."""
    g = Game(Side.BLACK, fen="3q2k1/8/8/8/8/8/P5PP/7K b - - 0 1", pipeline=True)
    qt._run_queue(
        g,
        "Qd7",
        qt._uci("d7d6", "d6d5", "d5d4", "d4d3", "d3d1"),
        qt._uci("a3", "a4", "a5", "a6", "a7"),
        frames_per_state=frames_per_state,
    )
    assert g.sans == ["Qd7", "a3", "Qd6", "a4", "Qd5", "a5", "Qd4", "a6", "Qd3", "a7", "Qd1#"], (
        g.sans
    )
    assert g.ended and g.ended[0].result == "0-1"


# ---- tests.test_cold_start --------------------------------------------------------------


def _cold_start_reads_a_rendered_mid_game_board(tmp_path: Path, theme: str, bottom: Side) -> None:
    cache = ThemeCache(tmp_path)
    ct._warm(cache, theme, bottom)
    truth = chess.Board(ct.MIDGAME)
    frame = synthetic_board(board=truth, theme=theme, bottom=bottom, coordinates=True)
    cold = GamePipeline(cache=cache)
    events = cold.cold_start(frame, GEOM)  # the orientation is read from the digits
    started = [e for e in events if isinstance(e, GameStarted)]
    assert started, "cold start did not begin a game"
    assert started[0].kind == "position", "a position: coached, not recorded"
    assert cold.reconciler is not None and cold.reconciler.kind == "position"
    assert cold.bottom == bottom, "the orientation was misread from the coordinates"
    board = cold.reconciler.board
    assert board.is_valid()
    assert not ct._squares_off(board, truth), f"cold read off on {ct._squares_off(board, truth)}"


def _a_stale_clock_does_not_settle_the_side_to_move(tmp_path: Path) -> None:
    cache = ThemeCache(tmp_path)
    ct._warm(cache, "green", Side.WHITE)
    frame = synthetic_board(board=chess.Board(ct.MIDGAME), coordinates=True)
    live = GamePipeline(cache=cache)
    joined = ct._join(live, frame)
    assert live.reconciler is not None
    live._last_clock = (300_000, 290_000, Side.BLACK)
    live._last_clock_ts = joined - 10.0  # read long before the position was adopted
    assert not any(isinstance(e, Resync) for e in ct._tick(live, frame, joined + 0.1))
    assert live.reconciler.turn_unknown


FOLDED: dict[str, list[tuple[str, Callable[..., None], dict]]] = {
    "tests.test_reconcile": [
        (
            "test_selected_piece_does_not_block_the_opponents_move",
            _selected_piece_does_not_block_the_opponents_move,
            {},
        ),
        ("test_black_at_bottom_user_black", _black_at_bottom_user_black, {}),
        (
            "test_position_resync_after_a_finished_game_is_live_again",
            _position_resync_after_a_finished_game_is_live_again,
            {},
        ),
        ("test_hover_outline_alone_blocks_a_commit", _hover_outline_alone_blocks_a_commit, {}),
        (
            "test_think_time_comes_from_the_frame_timestamps",
            _think_time_comes_from_the_frame_timestamps,
            {},
        ),
        ("test_checkmate_ends_the_game_once", _checkmate_ends_the_game_once, {}),
        (
            "test_opponent_move_while_browsing_snaps_to_latest",
            _opponent_move_while_browsing_snaps_to_latest,
            {},
        ),
    ],
    "tests.test_pipeline": [
        (
            "test_new_game_same_orientation_new_theme_rebuilds_recogniser",
            _new_game_same_orientation_new_theme_rebuilds_recogniser,
            {},
        ),
        (
            "test_board_flipped_before_the_first_move_keeps_the_game",
            _board_flipped_before_the_first_move_keeps_the_game,
            {},
        ),
        (
            "test_theme_changed_before_the_first_move_keeps_the_game",
            _theme_changed_before_the_first_move_keeps_the_game,
            {},
        ),
    ],
    "tests.test_scenarios": [
        ("test_window_moved_mid_game", _window_moved_mid_game, {}),
        (
            "test_board_scrolled_partly_off_screen_and_back",
            _board_scrolled_partly_off_screen_and_back,
            {},
        ),
        (
            "test_catch_up_after_a_gap_of_a_few_plies[2]",
            _catch_up_after_a_gap_of_a_few_plies,
            {"plies": 2},
        ),
        (
            "test_catch_up_after_a_gap_of_a_few_plies[4]",
            _catch_up_after_a_gap_of_a_few_plies,
            {"plies": 4},
        ),
    ],
    "tests.test_occlusion": [
        (
            "test_mating_move_commits_while_the_game_over_dialog_covers_the_board[black]",
            _mating_move_commits_while_the_game_over_dialog_covers_the_board,
            {"bottom": Side.BLACK},
        ),
        (
            "test_mate_in_the_middle_commits_from_two_uncovered_frames[black]",
            _mate_in_the_middle_commits_from_two_uncovered_frames,
            {"bottom": Side.BLACK},
        ),
        (
            "test_move_out_from_under_a_cover_waits_for_the_memory_to_expire[black]",
            _move_out_from_under_a_cover_waits_for_the_memory_to_expire,
            {"bottom": Side.BLACK},
        ),
        (
            "test_fading_dialog_reads_as_covered_and_keeps_the_memory_clean[black]",
            _fading_dialog_reads_as_covered_and_keeps_the_memory_clean,
            {"bottom": Side.BLACK},
        ),
        (
            "test_a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing[black]",
            _a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing,
            {"bottom": Side.BLACK},
        ),
        (
            "test_the_dialog_confirms_a_mate_read_once[white]",
            _the_dialog_over_a_mate_read_once_as_it_stood,
            {"bottom": Side.WHITE},
        ),
        (
            "test_the_dialog_confirms_a_mate_read_once[black]",
            _the_dialog_over_a_mate_read_once_as_it_stood,
            {"bottom": Side.BLACK},
        ),
    ],
    "tests.test_premoves": [
        (
            "test_a_five_hop_knight_queue_is_undone_hop_by_hop",
            _a_five_hop_knight_queue_is_undone_hop_by_hop,
            {},
        ),
        ("test_a_king_walk_queue_of_five_steps", _a_king_walk_queue_of_five_steps, {}),
        ("test_a_pawn_queue_up_the_board", _a_pawn_queue_up_the_board, {}),
        (
            "test_the_opponent_moves_onto_the_queued_square_and_the_premove_captures_it",
            _the_opponent_moves_onto_the_queued_square_and_the_premove_captures_it,
            {},
        ),
        (
            "test_the_users_king_in_check_on_a_red_square_is_not_a_ghost",
            _the_users_king_in_check_on_a_red_square_is_not_a_ghost,
            {},
        ),
        (
            "test_black_runs_a_long_queue_to_mate[3]",
            _black_runs_a_long_queue_to_mate,
            {"frames_per_state": 3},
        ),
    ],
    "tests.test_cold_start": [
        (
            "test_cold_start_reads_a_rendered_mid_game_board[brown-white]",
            _cold_start_reads_a_rendered_mid_game_board,
            {"theme": "brown", "bottom": Side.WHITE},
        ),
        (
            "test_cold_start_reads_a_rendered_mid_game_board[brown-black]",
            _cold_start_reads_a_rendered_mid_game_board,
            {"theme": "brown", "bottom": Side.BLACK},
        ),
        (
            "test_a_stale_clock_does_not_settle_the_side_to_move",
            _a_stale_clock_does_not_settle_the_side_to_move,
            {},
        ),
    ],
}
