from typing import Any

import chess
import pytest

from sidecoach.events import BoardUnread, GameEnded, GameStarted, Side
from sidecoach.state.pipeline import GamePipeline
from tests.games import sans_of
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


BROWN_LIGHT = (181, 217, 240)
BROWN_DARK = (99, 136, 181)


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
    assert sans_of(events) == ["Nf3"]
    assert len(pipe.reconciler.board.move_stack) == 3


def test_new_game_right_after_a_position_resync_is_a_new_game() -> None:
    """The next game's start on another theme, the other way up, is a new game on a new
    recogniser, even straight after a resync. (A new game on a new theme is also
    test_second_game_bootstraps_from_one_changed_tick's; the other way up, its moves
    followed, test_board_flipped_or_rethemed_before_the_first_move_keeps_the_game's:
    test_pipeline_bootstraps_and_rebootstraps_on_new_theme.)"""
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
    """A White bot moves within a second of the start; only one changed start frame exists.
    The game's start is stamped with the time and frame id of the tick that made it
    (test_events_carry_the_tick_time_and_frame_id)."""
    pipe = GamePipeline()
    board = chess.Board()
    start = synthetic_board(board=board)
    ticks = [(5.0 + 0.1 * i, 100 + i) for i in range(3)]
    events = []
    for i, (ts, frame_id) in enumerate(ticks):
        events += pipe.process(start, ts, changed=i == 0, frame_id=frame_id)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert len(started) == 1 and (started[0].ts, started[0].source_frame_id) in ticks
    move = board.push_san("d4")
    events = []
    for i in range(2):
        events += pipe.process(synthetic_board(board=board, highlight=move), 5.3 + 0.1 * i)
    assert sans_of(events) == ["d4"]


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
    assert sans_of(events) == list(sans)
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
    assert sans_of(events) == ["d4"]
    assert pipe.reconciler is not None and pipe.reconciler.pgn() == "1. d4"


def test_next_game_on_the_same_board_is_begun_by_the_reconciler() -> None:
    """A game mated on the board, then the next game's start position on the same theme,
    the same way up. The recogniser reads it as the start it was built on, so no bootstrap
    runs (`_reads_as_start`); the reconciler, its game finished, begins the next one
    itself, and the pipeline learns the recogniser again from that frame and keeps the
    reconciler it reset (`process`, the `_rebuild_model` after the reconciler's own
    GameStarted): the theme may have changed between games in a way the old recogniser
    still reads as the start. The game the mate ended is not ended again, and the next
    game's moves follow in a record of their own."""
    pipe = GamePipeline()
    board = chess.Board()
    _, ts = _start_frames(pipe, 0.0)
    for san in ("f3", "e5", "g4", "Qh4#"):
        move = board.push_san(san)
        for _ in range(2):
            pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    rec, model = pipe.reconciler, pipe.model
    assert rec is not None and rec.finished and rec.result == "0-1"
    first = rec.game_id
    events, ts = _start_frames(pipe, ts)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert len(started) == 1 and started[0].game_id != first
    assert not any(isinstance(e, GameEnded) for e in events)
    # The reconciler's own new game: the same reconciler, a recogniser learned again.
    assert pipe.reconciler is rec and pipe.model is not model and pipe.bootstraps == 2
    second = chess.Board()
    move = second.push_san("d4")
    events = []
    for _ in range(2):
        events += pipe.process(synthetic_board(board=second, highlight=move), ts)
        ts += 0.1
    assert sans_of(events) == ["d4"]
    assert rec.game_id == started[0].game_id and rec.pgn() == "1. d4"


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
    assert sans_of(events) == ["e5"]


def test_browsing_back_with_a_stray_reading_is_not_a_new_game() -> None:
    """A junk reading on an empty square next to the start position: still this game."""
    from sidecoach.vision.pieces import CellReading
    from sidecoach.vision.squares import BoardGeometry

    pipe = GamePipeline()
    _start_frames(pipe, 0.0)  # the recogniser the start is read with; moves change nothing
    assert pipe.model is not None and pipe.board is not None
    geometry = BoardGeometry(pipe.board.region, pipe.board.cell_px)
    readings = pipe.model.classify_frame(synthetic_board(board=chess.Board()), geometry)
    readings[3][3] = CellReading("piece", None, Side.WHITE, 0.5, 0.4, 240.0, settled=False)
    readings[0][4] = CellReading("piece", "k", Side.BLACK, 0.9, 0.4, 60.0, settled=False)
    assert pipe._reads_as_start(readings)


def _midgame(pipe: GamePipeline, look: dict[str, Any], ts: float) -> tuple[chess.Board, float]:
    board = chess.Board()
    for _ in range(3):
        pipe.process(synthetic_board(board=board, **look), ts)
        ts += 0.1
    for uci in ("e2e4", "e7e5", "g1f3"):
        move = chess.Move.from_uci(uci)
        board.push(move)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **look), ts)
            ts += 0.1
    return board, ts


@pytest.mark.parametrize(
    ("before", "after", "first_move_at_once"),
    [
        ({"theme": "green"}, {"theme": "brown"}, False),
        ({"theme": "brown"}, {"theme": "green"}, True),
        ({"style": "cburnett"}, {"style": "pixel"}, False),
        ({"style": "staunty"}, {"style": "papercut"}, True),
        ({"style": "cburnett"}, {"style": "kosal"}, False),
    ],
    ids=["green to brown", "brown to green, a move", "pixel set", "papercut, a move", "kosal"],
)
def test_a_look_changed_mid_game_is_learned_from_the_game_s_own_position(
    before: dict[str, Any], after: dict[str, Any], first_move_at_once: bool
) -> None:
    """The board's theme or the piece set changed mid-game: the recogniser read the new look
    as unknown (a theme) or as other pieces (a set), and the rest of the game went unread,
    with no event (round 2 finding 2). The look is learned again from the position the game
    is in -- or a legal move from it, played before the new look was read -- once the
    pieces on the frame are where that position puts them and its cells read back as their
    own labels."""
    pipe = GamePipeline()
    board, ts = _midgame(pipe, before, 0.0)
    assert pipe.reconciler is not None and pipe.reconciler.board.board_fen() == board.board_fen()
    plies = ["b8c6", "f1b5", "a7a6"]
    if first_move_at_once:
        move = chess.Move.from_uci(plies.pop(0))
        board.push(move)
    # The new look stays on a still board for a second (unchanged ticks after the first).
    changed = synthetic_board(board=board, highlight=board.peek(), **after)
    for i in range(10):
        pipe.process(changed, ts, changed=i == 0)
        ts += 0.1
    for uci in plies:
        move = chess.Move.from_uci(uci)
        board.push(move)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **after), ts)
            ts += 0.1
    assert pipe.reconciler.board.fen() == board.fen()
    assert pipe.bootstraps == 1


def test_a_theme_changed_mid_game_to_a_cached_one_takes_the_cached_set(tmp_path) -> None:
    """The new theme learned before, whole, in the cache: the cached set that reads the
    game's position is taken, not a partial one learned from the frame."""
    from sidecoach.vision.theme_cache import ThemeCache

    cache = ThemeCache(tmp_path)
    learner = GamePipeline(cache=cache)
    _start_frames(learner, 0.0, frames=3, theme="brown")
    assert learner.model is not None
    pipe = GamePipeline(cache=cache)
    board, ts = _midgame(pipe, {"theme": "green"}, 0.0)
    changed = synthetic_board(board=board, highlight=board.peek(), theme="brown")
    for i in range(10):
        pipe.process(changed, ts, changed=i == 0)
        ts += 0.1
    for uci in ("b8c6", "f1b5"):
        move = chess.Move.from_uci(uci)
        board.push(move)
        for _ in range(4):
            pipe.process(synthetic_board(board=board, highlight=move, theme="brown"), ts)
            ts += 0.1
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()
    assert pipe.model is not None and pipe.model.exemplar_labels == learner.model.exemplar_labels


def test_another_position_in_a_new_look_is_not_learned_as_the_game() -> None:
    """A board in a new theme showing another game's position: no legal move from the game
    puts its pieces there, so nothing is learned from it and the record stands."""
    pipe = GamePipeline()
    board, ts = _midgame(pipe, {"theme": "green"}, 0.0)
    model = pipe.model
    other = chess.Board("r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 4 8")
    for _ in range(12):
        pipe.process(synthetic_board(board=other, theme="brown"), ts)
        ts += 0.1
    assert pipe.model is model
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()


# The look-change sweep's set changes that stalled (round 3): the opening played in the old
# set, the moves played in the new one.
_SET_CHANGES = {
    "maestro to cburnett": (
        "maestro", "cburnett", "h2h3 b7b5 b1c3 f7f6 c3a4 h7h5", "g2g4 f6f5 c2c4 b5a4"
    ),
    "maestro to leipzig": (
        "maestro", "leipzig", "b2b4 g8f6 a2a3 h7h5 c2c4 h8g8", "b1c3 g7g5 d2d4 a7a6"
    ),
    # Read as the right pieces, unsettled: the still board explained, the next moves pending.
    "cburnett to chess7": (
        "cburnett", "chess7",
        "e2e3 a7a6 d1e2 h7h6 e2d3 g8f6 d3f5 f6g8 f5e4 c7c5 e4c4 h6h5 c4e6 b7b6 g2g3 a6a5 f1d3"
        " d7d6 f2f3 c8d7 g1e2 g8h6 b1c3 a8a6",
        "e6e7 d8e7 h1f1 f7f5",
    ),
    "cburnett to leipzig": (
        "cburnett", "leipzig", "g2g3 g7g5 a2a3 a7a5 f2f3 a5a4", "g3g4 h7h5 c2c4 f8h6"
    ),
}  # fmt: skip


@pytest.mark.parametrize("case", list(_SET_CHANGES))
def test_a_set_read_with_one_piece_wrong_is_learned_again(case: str) -> None:
    """The old set's recogniser read the new set with one piece wrong, and a frame within a
    cell of a position of the game was taken for "not a new look": the look was never
    learned, and the rest of the game went unread (round 3; about one set change in six of
    the look-change sweep). No story explains the frame, so the look is learned from the
    game's position while the board stands still, and the game is read on."""
    before, after, opening, rest = _SET_CHANGES[case]
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    for _ in range(3):
        pipe.process(synthetic_board(board=board, style=before), ts)
        ts += 0.1
    for uci in opening.split():
        move = board.push_uci(uci)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, style=before), ts)
            ts += 0.1
    model = pipe.model
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()
    changed = synthetic_board(board=board, highlight=board.peek(), style=after)
    for i in range(15):  # a second and a half, the board still
        pipe.process(changed, ts, changed=i == 0)
        ts += 0.1
    assert pipe.model is not model
    for uci in rest.split():
        move = board.push_uci(uci)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, style=after), ts)
            ts += 0.1
    assert pipe.reconciler.board.fen() == board.fen()


_OVERLAYS: dict[str, dict[str, Any]] = {
    "arrows on both rooks": {"arrows": (chess.Move.from_uci("a1b1"), chess.Move.from_uci("h1g1"))},
    "arrow from piece to piece": {"arrows": (chess.Move.from_uci("d1d2"),)},
    "red marks on the pawns": {
        "red_marks": tuple(chess.parse_square(f + "2") for f in "abcfgh") + (chess.E4, chess.D3)
    },
    "a selected knight": {"selected": chess.F3},
}


@pytest.mark.parametrize("overlay", list(_OVERLAYS))
def test_an_overlay_held_on_a_still_board_never_replaces_the_recogniser(overlay: str) -> None:
    """The look is learned again on a still board when a kind of which two or more pieces
    stand reads every one of them off its square (round 3): what a new set does, and what
    something drawn over the board does not. Held three seconds, an overlay does not even set
    a learning off (the overlay sweep: 192 of 192, six sets, two themes, sixteen overlays),
    and the game is read on after it."""
    from sidecoach.vision.pieces import ThemeModel

    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    look = {"style": "staunty", "theme": "green"}
    for _ in range(3):
        pipe.process(synthetic_board(board=board, **look), ts)
        ts += 0.1
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"):
        move = board.push_uci(uci)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **look), ts)
            ts += 0.1
    model = pipe.model
    learned: list[object] = []
    real = ThemeModel.from_position

    def counted(*args: Any) -> Any:
        learned.append(args)
        return real(*args)

    held = synthetic_board(board=board, highlight=board.peek(), **look, **_OVERLAYS[overlay])
    import unittest.mock

    with unittest.mock.patch.object(ThemeModel, "from_position", counted):
        for i in range(30):
            pipe.process(held, ts, changed=i == 0)
            ts += 0.1
    assert pipe.model is model and learned == []
    for uci in ("c2c3", "d7d6"):
        move = board.push_uci(uci)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **look), ts)
            ts += 0.1
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()


def test_a_look_that_cannot_be_learned_again_is_told_once() -> None:
    """A new theme first read with two plies already played on it: the frame is neither the
    game's position nor one move on, so nothing can be learned from it, and the game went
    unread in silence (round 3). The person is told once, after the wait a board unread at the
    start gets (COLD_NOTICE_S), and not again while it stays; with the old theme back, the
    game is read on, both plies with it."""
    from sidecoach.state.pipeline import COLD_NOTICE_S

    pipe = GamePipeline()
    board, ts = _midgame(pipe, {"theme": "green"}, 0.0)
    recorded = board.fen()
    board.push_uci("b8c6")
    last = board.push_uci("f1b5")
    changed = synthetic_board(board=board, highlight=last, theme="brown")
    told: list[tuple[float, BoardUnread]] = []
    start = ts
    for i in range(int((COLD_NOTICE_S + 3.0) / 0.1)):
        events = pipe.process(changed, ts, changed=i == 0)
        told += [(ts, e) for e in events if isinstance(e, BoardUnread)]
        ts += 0.1
    assert [e.reason for _, e in told] == ["look"], told
    assert told[0][0] - start >= COLD_NOTICE_S
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == recorded
    back = synthetic_board(board=board, highlight=last, theme="green")
    for i in range(8):
        pipe.process(back, ts, changed=i == 0)
        ts += 0.1
    assert pipe.reconciler.board.fen() == board.fen()
    assert pipe._look_lost is None
