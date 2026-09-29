"""The screen-state detector: what page the pipeline thinks is on screen.

DESIGN's first slice derives it from board presence and the game lifecycle, with no separate
classifier: no board is UNKNOWN, a board whose game is under way is LIVE_GAME, and a finished
game is GAME_OVER. The pipeline emits a ScreenStateChanged only when the state transitions.
"""

from __future__ import annotations

import chess
import numpy as np
import pytest

from sidecoach.events import Event, ScreenState, ScreenStateChanged, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Reconciler
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


def _states(events: list[Event]) -> list[ScreenState]:
    return [e.state for e in events if isinstance(e, ScreenStateChanged)]


def test_derive_screen_state_maps_the_game_lifecycle() -> None:
    pipe = GamePipeline()
    # No board on screen at all.
    assert pipe._derive_screen_state(board_present=False) is ScreenState.UNKNOWN
    # A board, but no game recognised on it yet (a menu, a set-up board, the lobby).
    assert pipe._derive_screen_state(board_present=True) is ScreenState.UNKNOWN
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    pipe.reconciler = rec
    assert pipe._derive_screen_state(board_present=True) is ScreenState.UNKNOWN  # not started
    rec._started = True
    assert pipe._derive_screen_state(board_present=True) is ScreenState.LIVE_GAME
    rec._finished = True
    assert pipe._derive_screen_state(board_present=True) is ScreenState.GAME_OVER


def test_screen_state_goes_live_then_game_over_on_a_mate_then_unknown_without_a_board() -> None:
    """Live once the game is recognised, game over on a mate, each announced once; and
    unknown when the board is lost:
    test_screen_state_falls_back_to_unknown_when_the_board_is_lost."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    events: list[Event] = []
    for _ in range(2):
        events += pipe.process(synthetic_board(board=board), ts)
        ts += 0.1
    # Once the recogniser bootstraps a game, the screen reads as a live game.
    assert ScreenState.LIVE_GAME in _states(events)
    assert pipe._screen_state is ScreenState.LIVE_GAME
    every = events

    # Fool's mate ends the game from the board alone; the screen reads as game over.
    events = []
    for san in ("f3", "e5", "g4", "Qh4#"):
        move = board.push_san(san)
        for _ in range(2):
            events += pipe.process(synthetic_board(board=board, highlight=move), ts)
            ts += 0.1
    assert board.is_checkmate()
    assert ScreenState.GAME_OVER in _states(events)
    assert pipe._screen_state is ScreenState.GAME_OVER
    # A transition is announced once, not on every frame the state holds: the game was
    # live across the six frames before the mate.
    every += events
    assert _states(every).count(ScreenState.LIVE_GAME) == 1
    assert _states(every).count(ScreenState.GAME_OVER) == 1

    # A blank screen: no board locks, so the state falls back to unknown (a lobby, a menu).
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    events = []
    for _ in range(4):
        events += pipe.process(blank, ts)
        ts += 0.1
    assert _states(events) == [ScreenState.UNKNOWN]
    assert pipe._screen_state is ScreenState.UNKNOWN
