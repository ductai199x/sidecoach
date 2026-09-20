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

import json
from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from sidecoach.events import GameStarted, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.squares import BoardGeometry
from sidecoach.vision.theme_cache import ThemeCache
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision

FIXTURES = Path(__file__).parent / "fixtures"
GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)
# After 10 moves of a Ruy Lopez, with both sides castled: a quiet middlegame board.
MIDGAME = "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 4 8"


def _warm(cache: ThemeCache, theme: str, bottom: Side) -> None:
    """Learn the theme from a rendered start position, as a live session does, into `cache`."""
    warm = GamePipeline(cache=cache)
    for i in range(3):
        warm.process(synthetic_board(theme=theme, bottom=bottom, coordinates=True), 0.1 * i)
    assert warm.model is not None and warm.model.rank_labels, "no theme learned"


def test_a_cached_theme_carries_no_frame(tmp_path: Path) -> None:
    """The theme file holds the exemplars, the palette and the digits, not the last
    frame's 64 cells: pickled with them it was 2.8 MB instead of 60 KB, and a theme read
    back would have started its next game believing it had read a frame."""
    cache = ThemeCache(tmp_path)
    warm = GamePipeline(cache=cache)
    for i in range(3):
        warm.process(synthetic_board(theme="green", coordinates=True), 0.1 * i)
    assert warm.model is not None and warm.model._cells, "the live model caches its cells"
    files = list(tmp_path.glob("theme-*.pkl"))
    # The embedding matcher's 64 exemplar vectors (float16) are 164 KB of the file.
    assert len(files) == 1 and files[0].stat().st_size < 400_000, files[0].stat().st_size
    light, dark = warm.model.light_bg, warm.model.dark_bg
    assert light is not None and dark is not None
    again = cache.get(light, dark)
    assert again is not None and again._cells == {}


def _squares_off(board: chess.Board, truth: chess.Board) -> list[str]:
    return [
        chess.square_name(sq) for sq in chess.SQUARES if board.piece_at(sq) != truth.piece_at(sq)
    ]


@pytest.mark.parametrize("theme", ["green", "brown"])
@pytest.mark.parametrize("bottom", [Side.WHITE, Side.BLACK])
def test_cold_start_reads_a_rendered_mid_game_board(
    tmp_path: Path, theme: str, bottom: Side
) -> None:
    cache = ThemeCache(tmp_path)
    _warm(cache, theme, bottom)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, theme=theme, bottom=bottom, coordinates=True)
    cold = GamePipeline(cache=cache)
    events = cold.cold_start(frame, GEOM)  # the orientation is read from the digits
    assert any(isinstance(e, GameStarted) for e in events), "cold start did not begin a game"
    assert cold.bottom == bottom, "the orientation was misread from the coordinates"
    assert cold.reconciler is not None
    board = cold.reconciler.board
    assert board.is_valid()
    # Every one of the 64 squares: a single misread piece would poison the analysis.
    assert not _squares_off(board, truth), f"cold read off on {_squares_off(board, truth)}"


def test_cold_start_refuses_a_rendered_board_caught_mid_move(tmp_path: Path) -> None:
    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    # The c1 bishop in flight over f4, under the hover outline.
    frame = synthetic_board(board=truth, coordinates=True, dragging=(chess.C1, chess.F4, 0.1, 0.1))
    cold = GamePipeline(cache=cache)
    assert cold.cold_start(frame, GEOM) == []
    assert cold.reconciler is None and cold.model is None


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


def test_the_side_to_move_of_an_adopted_position_comes_from_the_running_clock(
    tmp_path: Path,
) -> None:
    """The adopted position could be either side's move; the clock that is running says
    whose, and the position is re-announced with the turn so the coach and the engine
    look at the right side (the game where a takeback read as two moves and left the
    coach naming the opponent's best move on the player's own clock)."""
    from sidecoach.events import PositionChanged, Resync

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    frame = synthetic_board(board=truth, coordinates=True)
    live = GamePipeline(cache=cache)
    joined = _join(live, frame)
    rec = live.reconciler
    assert rec is not None and rec.turn_unknown and rec.board.turn == chess.WHITE
    # The clock reader saw black's clock running a moment ago.
    live._last_clock = (300_000, 290_000, Side.BLACK)
    live._last_clock_ts = joined
    events = _tick(live, frame, joined + 0.1)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert resyncs and "clock: black" in resyncs[0].reason
    assert [e.fen for e in events if isinstance(e, PositionChanged)] == [rec.board.fen()]
    assert not rec.turn_unknown and rec.board.turn == chess.BLACK
    assert rec.base_fen.split(" ")[1] == "b"
    assert not any(isinstance(e, Resync) for e in _tick(live, frame, joined + 0.2)), "said once"


def test_a_stale_clock_does_not_settle_the_side_to_move(tmp_path: Path) -> None:
    from sidecoach.events import Resync

    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    frame = synthetic_board(board=chess.Board(MIDGAME), coordinates=True)
    live = GamePipeline(cache=cache)
    joined = _join(live, frame)
    assert live.reconciler is not None
    live._last_clock = (300_000, 290_000, Side.BLACK)
    live._last_clock_ts = joined - 10.0  # read long before the position was adopted
    assert not any(isinstance(e, Resync) for e in _tick(live, frame, joined + 0.1))
    assert live.reconciler.turn_unknown


def test_a_board_mid_move_is_never_adopted_and_a_start_position_bootstraps(
    tmp_path: Path,
) -> None:
    cache = ThemeCache(tmp_path)
    _warm(cache, "green", Side.WHITE)
    truth = chess.Board(MIDGAME)
    dragged = synthetic_board(
        board=truth, coordinates=True, dragging=(chess.C1, chess.F4, 0.1, 0.1)
    )
    live = GamePipeline(cache=cache)
    for i in range(10):
        assert not any(isinstance(e, GameStarted) for e in _tick(live, dragged, 0.6 * i))
    assert live.model is None and live.reconciler is None
    # The start position is the bootstrap's, which learns the theme from it: no cold read.
    fresh = GamePipeline(cache=cache)
    start = synthetic_board(coordinates=True)
    for i in range(4):
        _tick(fresh, start, 0.6 * i)
    assert fresh.bootstraps == 1 and fresh.reconciler is not None
    assert fresh.reconciler.base_fen == chess.STARTING_FEN
