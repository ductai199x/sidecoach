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


def test_a_frame_that_needs_confirming_is_said_so_until_it_is() -> None:
    """`awaiting_confirmation` is what brings the capture's next frame early: the start seen
    once, then a move seen once; never while nothing waits (Tai, 2026-10-02: one early frame
    per move, never a higher rate all the time)."""
    pipe = GamePipeline()
    board = chess.Board()
    assert not pipe.awaiting_confirmation
    pipe.process(synthetic_board(board=board), 0.0)
    assert pipe.awaiting_confirmation, "the start seen once"
    pipe.process(synthetic_board(board=board), 0.1)
    assert pipe.reconciler is not None and not pipe.awaiting_confirmation, "the game began"
    pipe.process(synthetic_board(board=board), 0.2)
    assert not pipe.awaiting_confirmation, "nothing new on the board"
    move = board.push_san("e4")
    pipe.process(synthetic_board(board=board, highlight=move), 0.3)
    assert pipe.awaiting_confirmation, "e4 seen once"
    events = pipe.process(synthetic_board(board=board, highlight=move), 0.333)
    assert sans_of(events) == ["e4"]
    assert not pipe.awaiting_confirmation, "e4 committed"


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


_LOOK_CHANGES: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "green to brown": ({"theme": "green"}, {"theme": "brown"}),
    "brown to green": ({"theme": "brown"}, {"theme": "green"}),
    "cburnett to pixel": ({"style": "cburnett"}, {"style": "pixel"}),
    "staunty to papercut": ({"style": "staunty"}, {"style": "papercut"}),
    "cburnett to kosal": ({"style": "cburnett"}, {"style": "kosal"}),
}


def _told_and_back(
    pipe: GamePipeline, board: chess.Board, ts: float, before: dict[str, Any], after: dict[str, Any]
) -> None:
    """The new look held still past the notice's wait: told once, nothing learned, the record
    as it stood; then the old look back and two more plies, read on."""
    from sidecoach.state.pipeline import COLD_NOTICE_S

    model, recorded = pipe.model, board.fen()
    changed = synthetic_board(board=board, highlight=board.peek(), **after)
    told: list[BoardUnread] = []
    for i in range(int((COLD_NOTICE_S + 2.0) / 0.3)):  # a tick every 0.3 s, as _watch's
        told += [e for e in pipe.process(changed, ts, changed=i == 0) if isinstance(e, BoardUnread)]
        ts += 0.3
    assert [e.reason for e in told] == ["look"], told
    assert pipe.model is model and pipe.bootstraps == 1, "nothing learned mid-game"
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == recorded
    for _ in range(2):
        move = next(iter(board.legal_moves))
        board.push(move)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **before), ts)
            ts += 0.1
    assert pipe.reconciler.board.fen() == board.fen(), "read on with the old look back"


@pytest.mark.parametrize("case", list(_LOOK_CHANGES))
def test_a_look_changed_mid_game_that_no_learned_set_reads_is_told_never_learned(
    case: str,
) -> None:
    """The board's theme or the piece set changed mid-game, to a look no learned set reads:
    the person is told, once, and the game goes unread until its look is back. Learned from
    the game's position, a set knew only the kinds the position held, and one board gathered
    several such sets, each disagreeing: a knight read as a bishop, whole puzzles unread, in
    silence (Tai's Pixel, 2026-10-03; Tai: "bring it up to the user as an error and do not try
    to get a partial read"). A look is learned whole, at a start position."""
    before, after = _LOOK_CHANGES[case]
    pipe = GamePipeline()
    board, ts = _midgame(pipe, before, 0.0)
    assert pipe.reconciler is not None and pipe.reconciler.board.board_fen() == board.board_fen()
    _told_and_back(pipe, board, ts, before, after)


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


# The look-change sweep's set changes that stalled before 0.1.69 (round 3): the opening played
# in the old set, the board then shown in the new one. The old set read the new one with a
# piece or two wrong, or right but off their squares.
_SET_CHANGES = {
    "maestro to cburnett": ("maestro", "cburnett", "h2h3 b7b5 b1c3 f7f6 c3a4 h7h5"),
    "maestro to leipzig": ("maestro", "leipzig", "b2b4 g8f6 a2a3 h7h5 c2c4 h8g8"),
    "cburnett to chess7": (
        "cburnett", "chess7",
        "e2e3 a7a6 d1e2 h7h6 e2d3 g8f6 d3f5 f6g8 f5e4 c7c5 e4c4 h6h5 c4e6 b7b6 g2g3 a6a5 f1d3"
        " d7d6 f2f3 c8d7 g1e2 g8h6 b1c3 a8a6",
    ),
    "cburnett to leipzig": ("cburnett", "leipzig", "g2g3 g7g5 a2a3 a7a5 f2f3 a5a4"),
}  # fmt: skip


@pytest.mark.parametrize("case", list(_SET_CHANGES))
def test_a_set_changed_mid_game_read_nearly_right_is_told_never_learned(case: str) -> None:
    """A piece set changed mid-game that the old set reads nearly right -- a piece or two
    wrong, or every piece off its square -- is a look no learned set reads all the same: told,
    never learned, and never followed as other pieces."""
    before, after, opening = _SET_CHANGES[case]
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
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()
    _told_and_back(pipe, board, ts, {"style": before}, {"style": after})


_M = chess.Move.from_uci
_S = chess.parse_square
# The overlay sweep of 8711b37 (scratchpad overlay_sweep.py), whole: arrows through pieces,
# piece to piece, along a file and a rank, on both rooks and both bishops, four at once, thick,
# green; red marks on pieces, on both rooks, on every pawn; a selected knight and bishop.
_OVERLAYS: dict[str, dict[str, Any]] = {
    "arrow f3xe5": {"arrows": (_M("f3e5"),)},
    "arrow c4xf7": {"arrows": (_M("c4f7"),)},
    "arrow d1-d2 piece to piece": {"arrows": (_M("d1d2"),)},
    "arrow a1-a2 piece to piece": {"arrows": (_M("a1a2"),)},
    "arrow e1-e8 along a file": {"arrows": (_M("e1e8"),)},
    "arrow a2-h2 along the pawns": {"arrows": (_M("a2h2"),)},
    "arrows on both rooks": {"arrows": (_M("a1b1"), _M("h1g1"))},
    "arrows on both bishops": {"arrows": (_M("c1d2"), _M("c4b5"))},
    "four arrows": {"arrows": (_M("c4f7"), _M("f3g5"), _M("d1h5"), _M("c5f2"))},
    "red marks on pieces": {"red_marks": (_S("f7"), _S("e5"), _S("c6"), _S("c4"), _S("f3"))},
    "red marks on both rooks": {"red_marks": (_S("a1"), _S("h1"))},
    "red marks on the pawns": {
        "red_marks": tuple(_S(f + "2") for f in "abcfgh") + (chess.E4, chess.D3)
    },
    "a selected knight": {"selected": chess.F3},
    "a selected bishop": {"selected": chess.C4},
    "a thick orange arrow": {"arrows": (_M("f3e5"),), "arrow_width": 0.3},
    "green arrows": {"arrows": (_M("c4f7"), _M("d1d2")), "arrow_colour": "green"},
}
_OVERLAY_SETS = ("cburnett", "maestro", "staunty", "leipzig", "alpha", "pixel")


_OVERLAY_OPENING = ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5")
_OVERLAY_REST = ("c2c3", "d7d6", "b1d2", "a7a6")


def _overlay_opening(style: str, theme: str) -> tuple[GamePipeline, chess.Board, float]:
    """A pipeline that has read the opening on this set and theme, three frames a position."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    look: dict[str, Any] = {"style": style, "theme": theme}
    for _ in range(3):
        pipe.process(synthetic_board(board=board, **look), ts)
        ts += 0.1
    for uci in _OVERLAY_OPENING:
        move = board.push_uci(uci)
        for _ in range(3):
            pipe.process(synthetic_board(board=board, highlight=move, **look), ts)
            ts += 0.1
    return pipe, board, ts


# The opening read once a set and theme in a test process, every overlay case of them handed a
# copy of the pipeline as it stands after it. The opening is the same pixels and the same steps
# for all sixteen overlays -- two openings read afresh give the same events, frame for frame --
# and it was two thirds of each case's time (2026-10-02: 0.35-0.74 s of about 0.6 s). A copy
# reads the overlay and the rest exactly as a pipeline that read the opening itself: every event
# the same for all sixteen overlays on eight of the twelve sets and themes, measured, and
# test_a_copied_opening_reads_on_as_one_read_afresh keeps checking it. The embedding net is
# the process's one (`CellEmbedder.load`), shared by the copies as by every pipeline.
_OPENED: dict[tuple[str, str], tuple[GamePipeline, chess.Board, float]] = {}


def _opened(style: str, theme: str) -> tuple[GamePipeline, chess.Board, float]:
    import copy

    from sidecoach.vision.embedding import CellEmbedder

    if (style, theme) not in _OPENED:
        pipe, board, ts = _overlay_opening(style, theme)
        assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()
        _OPENED.clear()  # one look at a time: its sixteen cases run together (xdist_group)
        _OPENED[(style, theme)] = (pipe, board, ts)
    pipe, board, ts = _OPENED[(style, theme)]
    shared = {id(net): net for net in (CellEmbedder.load(),) if net is not None}
    return copy.deepcopy(pipe, shared), board.copy(), ts


def _overlay_and_rest(
    pipe: GamePipeline, board: chess.Board, ts: float, overlay: str, look: dict[str, Any]
) -> tuple[list[str], bool, int]:
    """The overlay held three seconds on the still board, then the rest of the game: the
    events as said (the game's random id left out), whether the model was kept, and on how
    many frames the look was taken for changed (`_look_lost`)."""
    import re

    said: list[str] = []
    model = pipe.model
    judged = 0

    def say(events: list[Any]) -> None:
        said.append(re.sub(r"game_id='[0-9a-f]+'", "game_id=*", repr(events)))

    held = synthetic_board(board=board, highlight=board.peek(), **look, **_OVERLAYS[overlay])
    for i in range(30):
        say(pipe.process(held, ts, changed=i == 0))
        judged += pipe._look_lost is not None
        ts += 0.1
    kept = pipe.model is model
    for uci in _OVERLAY_REST:
        move = board.push_uci(uci)
        for _ in range(3):
            say(pipe.process(synthetic_board(board=board, highlight=move, **look), ts))
            ts += 0.1
    assert pipe.reconciler is not None and pipe.reconciler.board.fen() == board.fen()
    return said, kept, judged


def _overlay_cases() -> list[Any]:
    """Every overlay on every set and theme, a set and theme's cases one xdist group (their
    opening read once on the worker); the ids as the three stacked parametrisations gave them."""
    return [
        pytest.param(
            overlay,
            style,
            theme,
            id=f"{overlay}-{style}-{theme}",
            marks=pytest.mark.xdist_group(f"overlay-{style}-{theme}"),
        )
        for theme in ("green", "brown")
        for style in _OVERLAY_SETS
        for overlay in _OVERLAYS
    ]


@pytest.mark.parametrize(("overlay", "style", "theme"), _overlay_cases())
def test_an_overlay_held_on_a_still_board_never_replaces_the_recogniser(
    overlay: str, style: str, theme: str
) -> None:
    """Something drawn over the board -- arrows, red marks, a selected piece -- is never taken
    for a changed look, which would leave the game unread and tell the person their pieces
    changed (a look is told, never learned, since 0.1.69). Sixteen overlays, six sets, two
    themes, 192 cases (round 3 audit E13, Tai 2026-10-02: the overlay sweep in the suite):
    held three seconds after the opening, the look is never judged changed, the recogniser is
    kept, and the game is read on after it."""
    pipe, board, ts = _opened(style, theme)
    _said, kept, judged = _overlay_and_rest(
        pipe, board, ts, overlay, {"style": style, "theme": theme}
    )
    assert kept and judged == 0


def test_a_copied_opening_reads_on_as_one_read_afresh() -> None:
    """What lets the overlay cases share their opening: a copy of a pipeline that read it
    says, frame for frame, what a pipeline reading the opening itself says, through an
    overlay held and the game after it -- and the original is untouched by its copy's run."""
    look: dict[str, Any] = {"style": "staunty", "theme": "brown"}
    _OPENED.pop(("staunty", "brown"), None)
    copied = _overlay_and_rest(*_opened("staunty", "brown"), "four arrows", look)
    again = _overlay_and_rest(*_opened("staunty", "brown"), "red marks on pieces", look)
    fresh = _overlay_and_rest(*_overlay_opening("staunty", "brown"), "four arrows", look)
    assert copied == fresh
    fresh = _overlay_and_rest(*_overlay_opening("staunty", "brown"), "red marks on pieces", look)
    assert again == fresh, "the first copy's run left the kept opening as it was"


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
