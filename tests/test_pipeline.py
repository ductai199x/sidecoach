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
