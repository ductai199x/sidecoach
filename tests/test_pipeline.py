import chess

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from tests.synth import synthetic_board

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


def test_new_game_same_orientation_new_theme_rebuilds_recogniser() -> None:
    """The reconciler spots the start position itself; the pipeline still rebuilds the model."""
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
    for _ in range(3):
        events += pipe.process(synthetic_board(board=board, light=BROWN_LIGHT, dark=BROWN_DARK), ts)
        ts += 0.1
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert ended and ended[0].pgn == "1. e4 e5"
    assert any(isinstance(e, GameStarted) for e in events)
    assert pipe.bootstraps == 2

    events = []
    for san in ("c4", "c5"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(
                synthetic_board(board=board, light=BROWN_LIGHT, dark=BROWN_DARK, highlight=move),
                ts,
            )
            ts += 0.1
    assert _sans(events) == ["c4", "c5"]


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
