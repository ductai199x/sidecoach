from typing import Any

import chess
import pytest

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


BROWN_LIGHT = (181, 217, 240)
BROWN_DARK = (99, 136, 181)


def _sans(events) -> list[str]:
    return [e.san for e in events if isinstance(e, MovePlayed)]


def test_pipeline_bootstraps_and_rebootstraps_on_new_theme() -> None:
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    events = []
    for _ in range(2):
        events += pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 1 and pipe.bottom == Side.WHITE

    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert _sans(events) == ["e4", "e5"]

    # A new game on a different theme with black at the bottom.
    board = chess.Board()
    events = []
    for _ in range(3):
        events += pipe.process(
            synthetic_board(board=board, bottom=Side.BLACK, light=BROWN_LIGHT, dark=BROWN_DARK),
            ts,
        )
        ts += 0.1
    assert any(isinstance(e, GameEnded) for e in events)
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 2 and pipe.bottom == Side.BLACK

    events = []
    for san in ("d4", "Nf6"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(
                synthetic_board(
                    board=board,
                    bottom=Side.BLACK,
                    light=BROWN_LIGHT,
                    dark=BROWN_DARK,
                    highlight=move,
                ),
                ts,
            )
            ts += 0.1
    assert _sans(events) == ["d4", "Nf6"]
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()


def _start_game(pipe: GamePipeline, ts: float, **theme) -> tuple[chess.Board, float]:
    board = chess.Board()
    for _ in range(2):
        pipe.process(synthetic_board(board=board, **theme), ts)
        ts += 0.1
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            pipe.process(synthetic_board(board=board, highlight=move, **theme), ts)
            ts += 0.1
    return board, ts


def test_browsing_to_the_start_position_keeps_the_game() -> None:
    """The move list's first entry shows the start position on the same theme for a while."""
    pipe = GamePipeline()
    board, ts = _start_game(pipe, 0.0)
    events = []
    for _ in range(5):
        events += pipe.process(synthetic_board(board=chess.Board()), ts)
        ts += 0.1
    assert not any(isinstance(e, GameEnded | GameStarted) for e in events)
    assert pipe.bootstraps == 1
    assert pipe.reconciler is not None and pipe.reconciler.viewing == 0
    move = board.push_san("Nf3")
    events = []
    for _ in range(2):
        events += pipe.process(synthetic_board(board=board, highlight=move), ts)
        ts += 0.1
    assert _sans(events) == ["Nf3"]
    assert len(pipe.reconciler.board.move_stack) == 3


def test_new_game_right_after_a_position_resync_is_a_new_game() -> None:
    pipe = GamePipeline()
    board, ts = _start_game(pipe, 0.0)
    assert pipe.reconciler is not None
    pipe.reconciler.resync_hold_frames = 3
    # Ten plies on: far beyond the move search, so only a position resync can follow.
    later = chess.Board("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6")
    for _ in range(6):
        pipe.process(synthetic_board(board=later), ts)
        ts += 0.1
    assert pipe.reconciler.base_fen.split()[:2] == later.fen().split()[:2]
    events = []
    for _ in range(4):
        events += pipe.process(
            synthetic_board(
                board=chess.Board(), bottom=Side.BLACK, light=BROWN_LIGHT, dark=BROWN_DARK
            ),
            ts,
        )
        ts += 0.1
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 2 and pipe.bottom == Side.BLACK


def test_bootstrap_counts_ticks_so_a_fast_first_move_does_not_starve_it() -> None:
    """A White bot moves within a second of the start; only one changed start frame exists."""
    pipe = GamePipeline()
    board = chess.Board()
    start = synthetic_board(board=board)
    events = pipe.process(start, 0.0, changed=True)
    events += pipe.process(start, 0.1, changed=False)
    events += pipe.process(start, 0.2, changed=False)
    assert any(isinstance(e, GameStarted) for e in events)
    move = board.push_san("d4")
    events = []
    for i in range(2):
        events += pipe.process(synthetic_board(board=board, highlight=move), 0.3 + 0.1 * i)
    assert _sans(events) == ["d4"]


def test_events_carry_the_tick_time_and_frame_id() -> None:
    pipe = GamePipeline()
    board = chess.Board()
    events = []
    for i in range(2):
        events += pipe.process(synthetic_board(board=board), 5.0 + 0.1 * i, frame_id=100 + i)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert started and started[0].source_frame_id in (100, 101)
    assert started[0].ts in (5.0, 5.1)


def test_reset_forgets_the_game_and_bootstraps_again() -> None:
    pipe = GamePipeline()
    board, ts = _start_game(pipe, 0.0)
    pipe.reset()
    assert pipe.model is None and pipe.reconciler is None
    events = []
    for _ in range(3):
        events += pipe.process(synthetic_board(board=chess.Board()), ts)
        ts += 0.1
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 2


def _start_frames(pipe: GamePipeline, ts: float, frames: int = 2, **kw) -> tuple[list, float]:
    events = []
    for _ in range(frames):
        events += pipe.process(synthetic_board(board=chess.Board(), **kw), ts)
        ts += 0.1
    return events, ts


@pytest.mark.parametrize(
    ("bottom", "theme", "sans"),
    [
        # Watching the pre-game board as white, then paired as black: the board flips.
        (Side.BLACK, "green", ("e4", "e5")),
        # The board's colours changed before the first move
        # (test_theme_changed_before_the_first_move_keeps_the_game).
        (Side.WHITE, "brown", ("d4",)),
    ],
)
def test_board_flipped_or_rethemed_before_the_first_move_keeps_the_game(
    bottom: Side, theme: str, sans: tuple[str, ...]
) -> None:
    """The start position seen anew before any move -- the other way up, or on other
    colours: a new recogniser, the same game announced again with the user's side, and
    its moves follow."""
    look: dict[str, Any] = {"bottom": bottom}
    if theme == "brown":
        look |= {"light": BROWN_LIGHT, "dark": BROWN_DARK}
    pipe = GamePipeline()
    events, ts = _start_frames(pipe, 0.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 1 and pipe.bottom == Side.WHITE
    assert pipe.reconciler is not None
    game_id = pipe.reconciler.game_id
    events, ts = _start_frames(pipe, ts, frames=4, **look)
    assert pipe.bootstraps == 2 and pipe.bottom == bottom
    assert not any(isinstance(e, GameEnded) for e in events)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert len(started) == 1 and started[0].game_id == game_id and started[0].user_side == bottom
    board = chess.Board()
    events = []
    for san in sans:
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move, **look), ts)
            ts += 0.1
    assert _sans(events) == list(sans)
    assert pipe.reconciler.game_id == game_id


def test_second_game_bootstraps_from_one_changed_tick() -> None:
    """A new theme's start position stored once, then unchanged ticks, then a fast first move.
    The game before it ends with its record, and the recogniser is built again for the new
    colours (test_new_game_same_orientation_new_theme_rebuilds_recogniser)."""
    pipe = GamePipeline()
    board = chess.Board()
    _, ts = _start_frames(pipe, 0.0)
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert pipe.reconciler is not None and len(pipe.reconciler.board.move_stack) == 2
    theme: dict[str, Any] = {"light": BROWN_LIGHT, "dark": BROWN_DARK}
    fresh = synthetic_board(board=chess.Board(), **theme)
    events = pipe.process(fresh, ts, changed=True)
    ts += 0.1
    for _ in range(3):
        events += pipe.process(fresh, ts, changed=False)
        ts += 0.1
    assert pipe.bootstraps == 2
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].pgn == "1. e4 e5"
    assert any(isinstance(e, GameStarted) for e in events)
    second = chess.Board()
    move = second.push_san("d4")
    events = []
    for _ in range(2):
        events += pipe.process(synthetic_board(board=second, highlight=move, **theme), ts)
        ts += 0.1
    assert _sans(events) == ["d4"]
    assert pipe.reconciler is not None and pipe.reconciler.pgn() == "1. d4"


def test_readings_are_retaken_after_the_lock_moves() -> None:
    """A move and a window move in one changed tick, then only unchanged ticks."""
    pipe = GamePipeline()
    board = chess.Board()
    _, ts = _start_frames(pipe, 0.0)
    move = board.push_san("e4")
    for _ in range(2):
        pipe.process(synthetic_board(board=board, highlight=move), ts)
        ts += 0.1
    move = board.push_san("e5")
    moved = synthetic_board(board=board, highlight=move, x0=330, y0=120)
    events = pipe.process(moved, ts, changed=True)
    ts += 0.1
    for _ in range(12):
        events += pipe.process(moved, ts, changed=False)
        ts += 0.1
    assert _sans(events) == ["e5"]


def test_browsing_back_with_a_stray_reading_is_not_a_new_game() -> None:
    """A junk reading on an empty square next to the start position: still this game."""
    from sidecoach.vision.pieces import CellReading
    from sidecoach.vision.squares import BoardGeometry

    pipe = GamePipeline()
    board = chess.Board()
    _, ts = _start_frames(pipe, 0.0)
    for san in ("e4", "e5", "Nf3"):
        move = board.push_san(san)
        for _ in range(2):
            pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert pipe.model is not None and pipe.board is not None
    geometry = BoardGeometry(pipe.board.region, pipe.board.cell_px)
    readings = pipe.model.classify_frame(synthetic_board(board=chess.Board()), geometry)
    readings[3][3] = CellReading("piece", None, Side.WHITE, 0.5, 0.4, 240.0, settled=False)
    readings[0][4] = CellReading("piece", "k", Side.BLACK, 0.9, 0.4, 60.0, settled=False)
    assert pipe._reads_as_start(readings)
