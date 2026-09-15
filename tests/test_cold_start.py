"""Cold-start recognition: read an arbitrary position from the theme cache, no move-0 bootstrap.

Learn a theme from a real game's start (which caches the piece style and the rank labels), then,
with a fresh pipeline that has only the cache, read a mid-game board of that theme cold and begin
a game from the position on it -- the puzzle / mid-game-analysis capability.
"""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.events import GameStarted
from sidecoach.replay.recording import Recording
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.squares import BoardGeometry
from sidecoach.vision.theme_cache import ThemeCache

REC = Path("data/recordings/g01-david-green")


def test_cold_start_reads_a_mid_game_position_from_the_cache(tmp_path: Path) -> None:
    cache = ThemeCache(tmp_path)

    # Warm run: learn and cache the theme, then note a genuinely quiet mid-game frame -- one
    # the recogniser reads as a fully settled, fully readable board (no piece in flight) and
    # the reconciler has no move pending on, so the tracked position matches the pixels exactly.
    warm = GamePipeline(cache=cache)
    captured: tuple = ()
    for i, (tick, frame) in enumerate(Recording(REC).frames()):
        warm.process(frame, tick.ts, changed=tick.stored is not None)
        if not (
            warm.reconciler is not None
            and warm.model is not None
            and warm.bottom is not None
            and warm.board is not None
            and warm.tracker.verified
            and len(warm.reconciler.board.move_stack) >= 8
            and warm.reconciler._pending is None  # no move mid-commit: pixels == tracked board
            and i > 500
        ):
            continue
        geom = BoardGeometry(warm.board.region, warm.board.cell_px)
        readings = warm.model.classify_frame(frame, geom)
        # A fully quiet, fully readable board -- the same bar cold_start adopts on.
        if GamePipeline._placement_from_readings(readings, warm.bottom) is None:
            continue
        captured = (frame, geom, warm.reconciler.board.board_fen())
        break
    assert captured, "no settled mid-game frame found"
    frame, geom, true_placement = captured

    # Cold run: a fresh pipeline with only the cache, no start position ever seen.
    cold = GamePipeline(cache=cache)
    events = cold.cold_start(frame, geom)
    assert any(isinstance(e, GameStarted) for e in events), "cold start did not begin a game"
    assert cold.reconciler is not None
    cold_board = cold.reconciler.board
    assert cold_board.is_valid()
    assert cold_board.king(chess.WHITE) is not None and cold_board.king(chess.BLACK) is not None

    # On a quiet board the cold read must reproduce the move-by-move tracked position exactly:
    # every one of the 64 squares. A single misread piece would poison the coach's analysis,
    # so nothing less than a perfect read is adopted.
    true_board = chess.Board(true_placement + " w - - 0 1")
    disagree = [
        chess.square_name(sq)
        for sq in chess.SQUARES
        if cold_board.piece_at(sq) != true_board.piece_at(sq)
    ]
    assert not disagree, f"cold read disagrees with the tracked position on {disagree}"


def test_cold_start_refuses_a_board_caught_mid_move(tmp_path: Path) -> None:
    """A frame with a piece in flight must not be adopted: the cold read is refused so the
    caller retries, rather than a lifted piece being read as an empty square."""
    cache = ThemeCache(tmp_path)
    warm = GamePipeline(cache=cache)
    mid_move: tuple = ()
    for i, (tick, frame) in enumerate(Recording(REC).frames()):
        warm.process(frame, tick.ts, changed=tick.stored is not None)
        if not (
            warm.model is not None
            and warm.bottom is not None
            and warm.board is not None
            and warm.tracker.verified
            and i > 500
        ):
            continue
        geom = BoardGeometry(warm.board.region, warm.board.cell_px)
        readings = warm.model.classify_frame(frame, geom)
        # A frame the recogniser reads as not-fully-quiet: a piece is being dragged.
        if any(
            readings[r][c].hover or (readings[r][c].kind == "piece" and not readings[r][c].settled)
            for r in range(8)
            for c in range(8)
        ):
            mid_move = (frame, geom)
            break
    assert mid_move, "no mid-move frame found to exercise the refusal"
    frame, geom = mid_move

    cold = GamePipeline(cache=cache)
    events = cold.cold_start(frame, geom)
    assert events == [], "cold start adopted a board caught mid-move"
    assert cold.reconciler is None, "a refused cold start must leave the pipeline untouched"
