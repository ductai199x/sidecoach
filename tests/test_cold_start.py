"""Cold start: a mid-game board recognised from the theme cache, no move-0 bootstrap.

A theme learned once from a start position is cached with its palette and its rank-digit
exemplars. Later, a board of that theme caught in the middle of a game -- a puzzle, a game
joined late, a restart -- is classified cold: the palette finds the cached recogniser, the
edge digits give the orientation, and a fully quiet, fully readable position is adopted as
the game's start. A board caught mid-move is refused, so a lifted piece is never read as an
empty square (the c1 bishop that read empty in flight over f4).

Rendered boards first, on two themes and both orientations, the orientation read from the
rendered coordinates; then the real frames of a recorded game (board crops under
tests/fixtures), where the digits are chess.com's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import GameStarted, Region, Side
from sidecoach.state.pipeline import COLD_NOTICE_S, GamePipeline
from sidecoach.vision.squares import BoardGeometry
from sidecoach.vision.theme_cache import ThemeCache
from tests.synth import GEOM, synthetic_board

pytestmark = pytest.mark.vision

FIXTURES = Path(__file__).parent / "fixtures"
# After 10 moves of a Ruy Lopez, with both sides castled: a quiet middlegame board.
MIDGAME = "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 4 8"

# The theme files a learning left in a cache, by the learning and what the cache held
# before it. Most tests begin by learning the same start position into an empty cache,
# and learning is a third of what they cost: each learning is done once and its files
# copied into the next cache that asks for it. The tests that are about learning itself
# learn afresh. A plain module dict, not a fixture: the phone's fixture maker calls these
# tests directly.
_LEARNED: dict[tuple, dict[str, bytes]] = {}


def _theme_files(cache: ThemeCache) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(cache.directory.glob("theme-*.pkl"))}


def _learned_once(cache: ThemeCache, step: tuple, learn: Callable[[], None]) -> None:
    """`learn` into `cache` the first time this learning meets these files; its files after."""
    before = _theme_files(cache)
    key = (step, tuple((name, hashlib.sha1(data).hexdigest()) for name, data in before.items()))
    after = _LEARNED.get(key)
    if after is None:
        learn()
        _LEARNED[key] = _theme_files(cache)
        return
    for name in before:
        (cache.directory / name).unlink()
    cache.directory.mkdir(parents=True, exist_ok=True)
    for name, data in after.items():
        (cache.directory / name).write_bytes(data)


def _warm(cache: ThemeCache, theme: str, bottom: Side) -> None:
    """Learn the theme from a rendered start position, as a live session does, into `cache`."""

    def learn() -> None:
        warm = GamePipeline(cache=cache)
        for i in range(3):
            warm.process(synthetic_board(theme=theme, bottom=bottom, coordinates=True), 0.1 * i)
        assert warm.model is not None and warm.model.rank_labels, "no theme learned"

    _learned_once(cache, ("warm", theme, bottom), learn)


def _squares_off(board: chess.Board, truth: chess.Board) -> list[str]:
    return [
        chess.square_name(sq) for sq in chess.SQUARES if board.piece_at(sq) != truth.piece_at(sq)
    ]


@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_cold_start_reads_a_rendered_mid_game_board(tmp_path: Path, bottom: Side) -> None:
    """Chess.com's green, either way up. Other themes are the stress sweep's: every seeded
    game's final position is read cold from its own cached theme, set and size."""
    theme = "green"
    cache = ThemeCache(tmp_path)
    _warm(cache, theme, bottom)
    truth = chess.Board(MIDGAME)
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
    # Every one of the 64 squares: a single misread piece would poison the analysis.
    assert not _squares_off(board, truth), f"cold read off on {_squares_off(board, truth)}"


@pytest.mark.parametrize("theme", ["green", "brown"])
def test_asked_to_reread_a_board_with_a_piece_left_selected(tmp_path: Path, theme: str) -> None:
    """Tai's desktop, 2026-09-24: a knight left selected put chess.com's legal-move dots on
    the empty squares it could go to, the dots read as unknown, and every "Re-read the
    screen" was refused. Asked for, a dot is the empty square it is; a read nobody asked
    for still treats it as no evidence."""
    cache = ThemeCache(tmp_path)
    _warm(cache, theme, Side.WHITE)
    truth = chess.Board(MIDGAME)
    knight = chess.F3  # the side to move's knight: dots on its empty destinations
    frame = synthetic_board(board=truth, theme=theme, coordinates=True, selected=knight)
    cold = GamePipeline(cache=cache)
    events = cold.cold_start(frame, GEOM)
    assert any(isinstance(e, GameStarted) for e in events), "the dots blocked the re-read"
    assert cold.reconciler is not None
    assert not _squares_off(cold.reconciler.board, truth)
    # Nobody asked: the same frame is not taken on its own.
    auto = GamePipeline(cache=cache)
    assert auto._cold_read(frame, GEOM, None) is None
    assert auto._cold_read(frame, GEOM, None, asked=True) is not None


def test_cold_start_without_coordinates_takes_the_side_it_is_told(tmp_path: Path) -> None:
    """Coordinates turned off: the digits cannot be read, the caller says which side is at
    the bottom (the panel's flip control), and the board is adopted with that."""
    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.BLACK)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, bottom=Side.BLACK)
    cold = GamePipeline(cache=cache)
    assert cold.cold_start(frame, GEOM) == [], "no digits, no orientation, nothing adopted"
    events = cold.cold_start(frame, GEOM, bottom=Side.BLACK)
    assert any(isinstance(e, GameStarted) for e in events)
    assert cold.reconciler is not None and not _squares_off(cold.reconciler.board, truth)


def _real(name: str) -> np.ndarray:
    frame = cv2.imread(str(FIXTURES / name))
    assert frame is not None, name
    return frame


def test_cold_start_reads_the_real_frames_the_same(tmp_path: Path) -> None:
    """The recorded game g01: the theme learned from its start frame, then a settled
    mid-game frame adopted and compared square by square with the tracked position, and
    a frame with a piece in flight refused."""
    meta = json.loads((FIXTURES / "frames.json").read_text(encoding="utf-8"))
    margin = meta["margin"]
    cache = ThemeCache(tmp_path)
    warm = GamePipeline(cache=cache)
    start = _real("g01_start.png")
    for i in range(4):
        warm.process(start, 0.1 * i)
    assert warm.model is not None and warm.model.rank_labels, "no theme learned from g01"
    side = meta["g01_settled"]["region"][2]
    geometry = BoardGeometry(Region(margin, margin, side, side), side / 8)

    cold = GamePipeline(cache=cache)
    events = cold.cold_start(_real("g01_settled.png"), geometry)
    assert any(isinstance(e, GameStarted) for e in events), "cold start did not begin a game"
    assert cold.bottom is not None and cold.bottom.value == meta["g01_settled"]["bottom"]
    assert cold.reconciler is not None
    truth = chess.Board(meta["g01_settled"]["placement"] + " w - - 0 1")
    off = _squares_off(cold.reconciler.board, truth)
    assert not off, f"cold read disagrees with the tracked position on {off}"

    refused = GamePipeline(cache=cache)
    assert refused.cold_start(_real("g01_in_flight.png"), geometry) == []


def _tick(pipe: GamePipeline, frame: np.ndarray, ts: float) -> list:
    return pipe.process(frame, ts, changed=True)


def _join(pipe: GamePipeline, frame: np.ndarray, ticks: int = 10) -> float:
    """Feed `frame` every 0.6 s until the pipeline adopts a game from it; the time it did."""
    for i in range(ticks):
        if any(isinstance(e, GameStarted) for e in _tick(pipe, frame, 0.6 * i)):
            return 0.6 * i
    raise AssertionError("the mid-game board was never adopted")


def test_a_game_joined_from_the_middle_is_adopted_without_being_asked(tmp_path: Path) -> None:
    """Start pressed with a game already on the board: the position is read cold from
    the cached theme and adopted once it has held still for a few reads, no Analyze
    click needed (Tai, 2026-09-19); then the game is followed from there."""
    from sidecoach.events import MovePlayed

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, coordinates=True)
    live = GamePipeline(cache=cache)
    assert _join(live, frame) >= 1.2, "adopted before the position had held still for three reads"
    assert live.reconciler is not None and not _squares_off(live.reconciler.board, truth)
    assert live.reconciler.turn_unknown, "both sides could move; the clock or a move decides"
    move = truth.push_san("d4")
    after = synthetic_board(board=truth, coordinates=True, highlight=move)
    played: list[MovePlayed] = []
    for j in range(3):
        played += [e for e in _tick(live, after, 10.0 + 0.1 * j) if isinstance(e, MovePlayed)]
    assert [e.san for e in played] == ["d4"], "the game is followed from the adopted position"
    assert not live.reconciler.turn_unknown and live.reconciler.board.turn == chess.BLACK
    # The first move's think time counts from the read, not from the dawn of time.
    assert played[0].think_ms is not None and played[0].think_ms < 12_000


def test_a_puzzle_read_cold_keeps_the_opening_move_played_before_it_was_adopted(
    tmp_path: Path,
) -> None:
    """The first puzzle of a session, read cold: its first board is read once before the site
    plays the opening move, and three reads of the board after it adopt the game. The first
    board is the start and the opening move is played (Tai's puzzles, 2026-10-03)."""
    from sidecoach.events import MovePlayed

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    puzzle = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1")
    live = GamePipeline(cache=cache)
    events = _tick(live, synthetic_board(board=puzzle, coordinates=True), 0.0)
    opening = puzzle.push_san("h6")
    after = synthetic_board(board=puzzle, coordinates=True, highlight=opening)
    for i in range(1, 6):
        events += _tick(live, after, 0.6 * i)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert len(started) == 1 and started[0].kind == "position"
    assert started[0].start_fen.split()[0] == "6k1/5ppp/8/8/8/8/5PPP/3R2K1", "the first board"
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["h6"]
    assert live.reconciler is not None and live.reconciler.board.turn == chess.WHITE


def test_the_side_to_move_of_an_adopted_position_comes_from_the_running_clock(
    tmp_path: Path,
) -> None:
    """The adopted position could be either side's move; the clock that is running says
    whose, and the position is re-announced with the turn so the coach and the engine
    look at the right side (the game where a takeback read as two moves and left the
    coach naming the opponent's best move on the player's own clock). A clock read long
    before the position was adopted settles nothing
    (test_a_stale_clock_does_not_settle_the_side_to_move)."""
    from sidecoach.events import PositionChanged, Resync

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, coordinates=True)
    live = GamePipeline(cache=cache)
    joined = _join(live, frame)
    rec = live.reconciler
    assert rec is not None and rec.turn_unknown and rec.board.turn == chess.WHITE
    # Black's clock seen running, but read long before the position was adopted.
    live._last_clock = (300_000, 290_000, Side.BLACK)
    live._last_clock_ts = joined - 10.0
    assert not any(isinstance(e, Resync) for e in _tick(live, frame, joined + 0.1))
    assert rec.turn_unknown
    # The clock reader saw black's clock running a moment ago.
    live._last_clock_ts = joined + 0.1
    events = _tick(live, frame, joined + 0.2)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert resyncs and "clock: black" in resyncs[0].reason
    assert [e.fen for e in events if isinstance(e, PositionChanged)] == [rec.board.fen()]
    assert not rec.turn_unknown and rec.board.turn == chess.BLACK
    assert rec.base_fen.split(" ")[1] == "b"
    assert not any(isinstance(e, Resync) for e in _tick(live, frame, joined + 0.3)), "said once"


def test_a_board_mid_move_is_never_adopted_and_a_start_position_bootstraps(
    tmp_path: Path,
) -> None:
    """Four reads of a board mid-move, the c1 bishop in flight over f4 under the hover
    outline, 0.6 s apart: a quiet one would have been adopted by the third (COLD_START_HOLD
    reads). Asked for, the read refuses it too
    (test_cold_start_refuses_a_rendered_board_caught_mid_move)."""
    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    dragged = synthetic_board(
        board=truth, coordinates=True, dragging=(chess.C1, chess.F4, 0.1, 0.1)
    )
    live = GamePipeline(cache=cache)
    for i in range(4):
        assert not any(isinstance(e, GameStarted) for e in _tick(live, dragged, 0.6 * i))
    assert live.cold_start(dragged, GEOM) == []
    assert live.model is None and live.reconciler is None
    # The start position is the bootstrap's, which learns the theme from it: no cold read.
    fresh = GamePipeline(cache=cache)
    start = synthetic_board(coordinates=True)
    for i in range(4):
        _tick(fresh, start, 0.6 * i)
    assert fresh.bootstraps == 1 and fresh.reconciler is not None
    assert fresh.reconciler.base_fen == chess.STARTING_FEN


def test_flip_re_reads_a_joined_position_the_other_way_up(tmp_path: Path) -> None:
    """Joined with the wrong side at the bottom (no labels to read, so the forced side
    was wrong): `reorient` reads the screen again the other way and begins again from
    that, ending the bogus game."""
    from sidecoach.events import GameEnded

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.BLACK)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, bottom=Side.BLACK)  # no coordinates drawn
    live = GamePipeline(cache=cache)
    assert not any(isinstance(e, GameStarted) for e in _tick(live, frame, 0.0))
    live.forced_bottom = Side.WHITE  # a Flip before any game: the wrong guess
    _join(live, frame)
    assert live.reconciler is not None and live.bottom is Side.WHITE
    assert _squares_off(live.reconciler.board, truth), "read upside down, as forced"
    first = live.reconciler.game_id
    events = live.reorient(frame, GEOM, Side.BLACK, 20.0)
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "GameEnded" and "GameStarted" in kinds, kinds
    assert isinstance(events[0], GameEnded) and events[0].reason == "board reoriented"
    assert live.reconciler.game_id != first and live.bottom is Side.BLACK
    assert not _squares_off(live.reconciler.board, truth)


def test_a_resync_onto_a_board_the_other_way_up_reads_the_labels(tmp_path: Path) -> None:
    """The next puzzle comes up from the other side: the resync would keep the old
    orientation and adopt the position on its head; the labels say otherwise, and the
    board is read again the right way, as a new game. The solver is Black, the side at the
    bottom, and so is the presumed side to move -- not White by default (a puzzle for Black
    was coached for White, 2026-09-20:
    test_a_position_for_black_is_read_from_the_labels_and_coached_for_black)."""
    from sidecoach.events import Resync

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)  # its digits read the board either way up
    first = chess.Board(MIDGAME)
    live = GamePipeline(cache=cache)
    _join(live, synthetic_board(board=first, coordinates=True))
    assert live.bottom is Side.WHITE
    # A different position, shown with Black at the bottom.
    second = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1")
    frame = synthetic_board(board=second, bottom=Side.BLACK, coordinates=True)
    events = []
    for i in range(12):
        events += _tick(live, frame, 30.0 + 0.1 * i)
        if any(isinstance(e, GameStarted) for e in events):
            break
    assert not any(isinstance(e, Resync) for e in events), "no resync on its head"
    assert any(isinstance(e, GameStarted) for e in events), "a new game, the right way up"
    rec = live.reconciler
    assert live.bottom is Side.BLACK and rec is not None and rec.user_side is Side.BLACK
    assert not _squares_off(rec.board, second)
    assert rec.turn_unknown and rec.board.turn == chess.BLACK, "the solver's move, meanwhile"


# The puzzle Tai had up on the Pixel (2026-09-23), just after Black's d5xc4.
PUZZLE = "6r1/pp3k2/4b2R/2p1b1p1/2p3P1/P5P1/1P2BP2/2K2N2 w - - 0 1"
LAST = chess.Move.from_uci("d5c4")


def _learn(
    cache: ThemeCache, style: str, bottom: Side = Side.WHITE, *, afresh: bool = False
) -> None:
    """A start position in `style` on chess.com's green, shown to a live pipeline; `afresh`
    when the learning is what the test is about, not its setting."""

    def learn() -> None:
        warm = GamePipeline(cache=cache)
        for i in range(3):
            warm.process(
                synthetic_board(theme="green", style=style, bottom=bottom, coordinates=True),
                0.1 * i,
            )
        assert warm.model is not None and warm.model.rank_labels, f"{style} not learned"

    if afresh:
        learn()
    else:
        _learned_once(cache, ("learn", style, bottom), learn)


def _puzzle(style: str, **kwargs) -> np.ndarray:
    return synthetic_board(
        board=chess.Board(PUZZLE),
        theme="green",
        style=style,
        coordinates=True,
        highlight=LAST,
        **kwargs,
    )


# Long enough for a board left unread to have been explained (`BoardUnread`), or not.
EXPLAINED_BY = COLD_NOTICE_S + 2.0


def _watch(pipe: GamePipeline, frame: np.ndarray, seconds: float, start: float = 0.0) -> list:
    """`frame` held on screen for `seconds`, a tick every 0.3 s, its pixels new only on the
    first: a puzzle sitting there while the person thinks."""
    events: list = []
    ts = start
    while ts < start + seconds:
        events += pipe.process(frame, ts, changed=ts == start)
        ts += 0.3
    return events


def test_two_piece_sets_on_one_board_are_both_kept_and_each_read(tmp_path: Path) -> None:
    """One board's colours, two piece sets learned from their start positions: both are
    kept, and a puzzle in either is read with its own (the cache held one set per board
    colours, and a test board's set cached under chess.com's green read Tai's puzzle's
    black pawns as white: nothing was ever adopted)."""
    from sidecoach.events import BoardUnread

    cache = ThemeCache(tmp_path)
    _learn(cache, "classic")
    _learn(cache, "staunty", afresh=True)
    assert len(list(tmp_path.glob("theme-*.pkl"))) == 2, "a second set stands beside the first"
    for style in ("classic", "staunty"):
        live = GamePipeline(cache=cache)
        events = _watch(live, _puzzle(style), 3.0)
        assert any(isinstance(e, GameStarted) for e in events), f"{style} puzzle not adopted"
        assert not any(isinstance(e, BoardUnread) for e in events)
        assert live.reconciler is not None
        assert not _squares_off(live.reconciler.board, chess.Board(PUZZLE)), style


def test_a_set_learned_again_replaces_itself(tmp_path: Path) -> None:
    """The same set learned again, here from the other side's start position, replaces
    the file it had rather than standing beside it."""
    cache = ThemeCache(tmp_path)
    _learn(cache, "staunty")
    _learn(cache, "staunty", Side.BLACK, afresh=True)
    assert len(list(tmp_path.glob("theme-*.pkl"))) == 1, "the same set learned again is one file"


def test_a_set_not_learned_is_explained_once_then_learned_from_a_start_position(
    tmp_path: Path,
) -> None:
    """A puzzle in pieces not learned yet, on board colours that are: nothing is adopted
    -- the other set's read is not the position -- and after a few seconds the person is
    told, once, that a start position teaches it. Shown one, the set is learned beside the
    other and the puzzle is read."""
    from sidecoach.events import BoardUnread

    cache = ThemeCache(tmp_path)
    _learn(cache, "classic")
    live = GamePipeline(cache=cache)
    events = _watch(live, _puzzle("staunty"), EXPLAINED_BY)
    assert not any(isinstance(e, GameStarted) for e in events), "adopted with the wrong set"
    notices = [e for e in events if isinstance(e, BoardUnread)]
    assert [n.reason for n in notices] == ["unlearned"], notices
    assert COLD_NOTICE_S <= notices[0].ts <= COLD_NOTICE_S + 1.0, f"told at {notices[0].ts:.1f} s"
    # Away from the board and back: explained again.
    away = EXPLAINED_BY + 10.0
    events = _watch(live, synthetic_board(pieces=False, x0=-2000), 1.0, start=away)
    events = _watch(live, _puzzle("staunty"), EXPLAINED_BY, start=away + 10.0)
    assert [n.reason for n in events if isinstance(n, BoardUnread)] == ["unlearned"]
    _learn(cache, "staunty", afresh=True)
    assert len(list(tmp_path.glob("theme-*.pkl"))) == 2
    fresh = GamePipeline(cache=cache)
    events = _watch(fresh, _puzzle("staunty"), 3.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert fresh.reconciler is not None
    assert not _squares_off(fresh.reconciler.board, chess.Board(PUZZLE))


def test_a_puzzle_left_still_is_read_once(tmp_path: Path, monkeypatch) -> None:
    """A still puzzle is adopted after the three reads a cold start waits for, and the
    board is read once, not every half second: pixels that did not change read the same."""
    cache = ThemeCache(tmp_path)
    _learn(cache, "staunty")
    reads = 0
    original = ThemeCache.candidates

    def counted(self, *args):
        nonlocal reads
        reads += 1
        return original(self, *args)

    monkeypatch.setattr(ThemeCache, "candidates", counted)
    live = GamePipeline(cache=cache)
    events = _watch(live, _puzzle("staunty"), 3.0)
    started = [e for e in events if isinstance(e, GameStarted)]
    assert started and 1.0 <= started[0].ts <= 2.0, "adopted after three reads, within two seconds"
    assert reads == 1, f"the still board was read {reads} times"


def test_a_selected_piece_is_not_explained(tmp_path: Path) -> None:
    """The person clicks a piece of a learned set and thinks: its dots and rings keep the
    read from being clean, but the set reads the board, so nothing is said; let go, the
    puzzle is adopted."""
    from sidecoach.events import BoardUnread

    cache = ThemeCache(tmp_path)
    _learn(cache, "staunty")
    live = GamePipeline(cache=cache)
    events = _watch(live, _puzzle("staunty", selected=chess.E2), EXPLAINED_BY)
    assert not any(isinstance(e, BoardUnread) for e in events), "a held piece explained"
    if not any(isinstance(e, GameStarted) for e in events):
        events = _watch(live, _puzzle("staunty"), 3.0, start=EXPLAINED_BY + 1.0)
        assert any(isinstance(e, GameStarted) for e in events)
    assert live.reconciler is not None
    assert not _squares_off(live.reconciler.board, chess.Board(PUZZLE))


def test_a_board_without_coordinates_is_explained_as_its_orientation(tmp_path: Path) -> None:
    """A learned set with the coordinates turned off: the pieces read, the side at the
    bottom cannot be; the person is told Flip says it, and Flip does."""
    from sidecoach.events import BoardUnread

    cache = ThemeCache(tmp_path)
    _learn(cache, "staunty")
    frame = synthetic_board(
        board=chess.Board(PUZZLE), theme="green", style="staunty", highlight=LAST
    )
    live = GamePipeline(cache=cache)
    events = _watch(live, frame, EXPLAINED_BY)
    assert [e.reason for e in events if isinstance(e, BoardUnread)] == ["orientation"]
    live.forced_bottom = Side.WHITE
    events = _watch(live, frame, 3.0, start=EXPLAINED_BY + 2.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert live.reconciler is not None
    assert not _squares_off(live.reconciler.board, chess.Board(PUZZLE))


def _cached_wrong(cache: ThemeCache) -> tuple[np.ndarray, np.ndarray]:
    """The staunty set learned, then its labels cached as solid blocks -- what a start seen
    through a lock off its board left on Tai's Pixel (2026-09-27); its board colours."""
    _learn(cache, "staunty")
    model_palette = _palette(cache)
    [(key, model)] = cache.candidates(*model_palette)
    model.rank_labels = {rank: np.ones((24, 24), dtype=bool) for rank in range(1, 9)}
    cache.save(model, key=key)
    return model_palette


BLACK_PUZZLE = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"


def test_labels_cached_wrong_are_read_past_by_the_8_and_learned_again(tmp_path: Path) -> None:
    """Tai's Pixel, 2026-09-27: the labels cached as solid blocks read no side, every puzzle
    asked for Flip, and the side flipped to stood over the next puzzle, shown from the other
    side, which was read on its head and no move of it was ever found. The 8 reads the side
    with nothing learned: the puzzle is taken up by itself, its labels are learned from it
    and cached with the set, and the next puzzle, from White's side, turns round."""
    from sidecoach.events import BoardUnread
    from sidecoach.vision.coordinates import read_labels

    cache = ThemeCache(tmp_path)
    model_palette = _cached_wrong(cache)
    black = chess.Board(BLACK_PUZZLE)
    frame = synthetic_board(
        board=black, theme="green", style="staunty", bottom=Side.BLACK, coordinates=True
    )
    live = GamePipeline(cache=cache)
    events = _watch(live, frame, EXPLAINED_BY)
    assert not [e for e in events if isinstance(e, BoardUnread)]
    assert any(isinstance(e, GameStarted) for e in events)
    assert live.bottom is Side.BLACK and live.reconciler is not None
    assert not _squares_off(live.reconciler.board, black)
    [(_, cached)] = cache.candidates(*model_palette)
    assert read_labels(frame, GEOM, cached.rank_labels) is Side.BLACK, "not learned again"
    events = _watch(live, _puzzle("staunty"), 3.0, start=EXPLAINED_BY + 12.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert live.bottom is Side.WHITE and live.reconciler is not None
    assert not _squares_off(live.reconciler.board, chess.Board(PUZZLE))


def test_a_flip_teaches_labels_that_read_no_side_where_the_8_cannot_be_read(
    tmp_path: Path, monkeypatch
) -> None:
    """Where the 8 cannot be read either (a piece over it, a typeface whose loops close at
    that size), the person says the side with Flip: the board is read that way at once, the
    labels learn from it and are cached with the set, and the next puzzle turns round by its
    labels, the Flip notwithstanding."""
    import sidecoach.state.pipeline as pipeline_module
    import sidecoach.vision.coordinates as coordinates_module
    from sidecoach.events import BoardUnread
    from sidecoach.vision.coordinates import read_labels

    monkeypatch.setattr(coordinates_module, "eights", lambda frame, geometry: None)
    monkeypatch.setattr(pipeline_module, "eights", lambda frame, geometry: None)
    cache = ThemeCache(tmp_path)
    model_palette = _cached_wrong(cache)
    black = chess.Board(BLACK_PUZZLE)
    frame = synthetic_board(
        board=black, theme="green", style="staunty", bottom=Side.BLACK, coordinates=True
    )
    live = GamePipeline(cache=cache)
    events = _watch(live, frame, EXPLAINED_BY)
    assert [e.reason for e in events if isinstance(e, BoardUnread)] == ["orientation"]
    assert live.reconciler is None or not live.reconciler.started
    # The panel's Flip: the side told, and the board read that way now.
    live.forced_bottom = Side.BLACK
    events = live.reorient(frame, GEOM, Side.BLACK, EXPLAINED_BY + 2.0, told=True)
    assert any(isinstance(e, GameStarted) for e in events)
    assert live.reconciler is not None and live.bottom is Side.BLACK
    assert not _squares_off(live.reconciler.board, black)
    [(_, cached)] = cache.candidates(*model_palette)
    assert read_labels(frame, GEOM, cached.rank_labels) is Side.BLACK, "not cached"
    events = _watch(live, _puzzle("staunty"), 3.0, start=EXPLAINED_BY + 12.0)
    assert any(isinstance(e, GameStarted) for e in events)
    assert live.bottom is Side.WHITE and live.reconciler is not None
    assert not _squares_off(live.reconciler.board, chess.Board(PUZZLE))


def test_a_flip_by_mistake_teaches_no_labels(tmp_path: Path) -> None:
    """Labels that read a board's side are kept whatever a Flip says: taught by a mistaken
    Flip, every later board of the set would be read upside down."""
    from sidecoach.vision.coordinates import read_bottom_side

    cache = ThemeCache(tmp_path)
    _learn(cache, "staunty")
    model_palette = _palette(cache)
    frame = _puzzle("staunty")
    live = GamePipeline(cache=cache)
    _join(live, frame)
    assert live.bottom is Side.WHITE
    live.forced_bottom = Side.BLACK
    live.reorient(frame, GEOM, Side.BLACK, 20.0, told=True)
    [(_, cached)] = cache.candidates(*model_palette)
    assert read_bottom_side(frame, GEOM, cached.rank_labels) is Side.WHITE


def _palette(cache: ThemeCache) -> tuple[np.ndarray, np.ndarray]:
    """The board colours of the one set in `cache`."""
    [path] = cache.directory.glob("theme-*.pkl")
    palette = cache._palette(path)
    assert palette is not None
    return palette


def test_the_next_puzzle_from_the_other_side_is_turned_round_at_once(
    tmp_path: Path, monkeypatch
) -> None:
    """A puzzle for White followed, then the next one shown from Black's side: the labels
    turn the game round within half a second of the board standing still, from the
    readings already taken -- no resync on its head first, and no second read of the board
    (five to ten seconds a puzzle on the Pixel, 2026-09-23)."""
    from sidecoach.events import Resync

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)  # its digits read the board either way up
    first = chess.Board(PUZZLE)
    live = GamePipeline(cache=cache)
    _join(live, synthetic_board(board=first, coordinates=True, highlight=LAST))
    assert live.bottom is Side.WHITE
    second = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1")
    frame = synthetic_board(board=second, bottom=Side.BLACK, coordinates=True)
    reads = 0
    original = ThemeCache.candidates

    def counted(self, *args):
        nonlocal reads
        reads += 1
        return original(self, *args)

    monkeypatch.setattr(ThemeCache, "candidates", counted)
    events: list = []
    ts = 40.0
    while ts < 42.0 and not any(isinstance(e, GameStarted) for e in events):
        events += live.process(frame, ts, changed=ts == 40.0)
        ts += 0.1
    started = [e for e in events if isinstance(e, GameStarted)]
    assert started, "the turned board was not taken up"
    assert started[0].ts is not None and started[0].ts - 40.0 <= 0.6, started[0].ts
    assert not any(isinstance(e, Resync) for e in events), "a resync on its head first"
    assert reads == 0, "the board was read a second time to turn it round"
    assert live.bottom is Side.BLACK and live.reconciler is not None
    assert not _squares_off(live.reconciler.board, second)


def test_a_new_position_on_a_slow_stream_is_taken_after_a_moment_not_ten_frames() -> None:
    """Frames a third of a second apart, as a phone reads when the frames are heavy: the
    next position is adopted once it has stood still for three frames and 0.4 s, not after
    ten frames -- three seconds."""
    from sidecoach.events import Resync

    live = GamePipeline()
    for i in range(3):
        live.process(synthetic_board(coordinates=True), 0.1 * i)
    assert live.reconciler is not None
    later = chess.Board("r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6")
    frame = synthetic_board(board=later, coordinates=True)
    events: list = []
    ts = 10.0
    while ts < 14.0 and not any(isinstance(e, Resync) for e in events):
        events += live.process(frame, ts, changed=ts == 10.0)
        ts += 0.33
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert resyncs and resyncs[0].ts is not None and resyncs[0].ts - 10.0 <= 1.1, events
    assert live.reconciler.base_fen.split()[0] == later.board_fen()
