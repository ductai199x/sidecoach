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
    return cv2.imread(str(FIXTURES / name))


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
