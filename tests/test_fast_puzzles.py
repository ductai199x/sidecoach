"""Puzzles at the phone's frame rate (Tai, 2026-09-24: "the move list is very very fragile,
sometimes there is no move list whatsoever"). Two ways the record lost every move, found by
playing puzzles as chess.com shows them sampled at two frames a second:

- a puzzle adopted with the side to move open: the solver's move and the instant reply,
  seen together, fit the same two moves in the other order too, and nothing was recorded;
  the site's last-move highlight now says who moved (`reconcile.last_move_seen`);
- the next puzzle shown from the other side, read while its setup move was still sliding
  in: the sliding piece read as two empty squares and the board was adopted without it;
  a turned board is now adopted only when it reads the same twice.
"""

from __future__ import annotations

import chess
import pytest

from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.pieces import Side
from tests.synth import synthetic_board

pytestmark = pytest.mark.vision


def _frame(board: chess.Board, bottom: bool, **kw: object):  # noqa: ANN202
    return synthetic_board(
        board=board,
        x0=300,
        y0=120,
        cell=64,
        width=1280,
        height=800,
        bottom=Side.WHITE if bottom == chess.WHITE else Side.BLACK,
        coordinates=True,
        **kw,  # type: ignore[arg-type]
    )


class _Feed:
    def __init__(self, fps: float) -> None:
        self.pipe = GamePipeline()
        self.ts = 0.0
        self.dt = 1.0 / fps

    def show(self, frame, times: int = 1) -> None:  # noqa: ANN001
        for _ in range(times):
            self.ts += self.dt
            self.pipe.process(frame, self.ts, frame_id=int(self.ts * 1000))

    @property
    def rec(self):  # noqa: ANN201
        assert self.pipe.reconciler is not None
        return self.pipe.reconciler


def test_the_solvers_move_and_the_instant_reply_are_both_recorded() -> None:
    # Lichess 000rO at two frames a second: the setup move a6-a5 highlighted, then Bc7 and
    # the reply Kb4 land between two frames.
    f = _Feed(2.0)
    f.show(_frame(chess.Board(), chess.WHITE), 3)
    board = chess.Board("3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58")
    solver = chess.BLACK
    f.show(_frame(board, solver), 1)
    setup = board.push_uci("a6a5")
    f.show(_frame(board, solver, highlight=setup), 4)
    board.push_uci("e5c7")
    reply = board.push_uci("a5b4")
    f.show(_frame(board, solver, highlight=reply), 4)
    assert f.rec.board.board_fen() == board.board_fen()
    assert [m.uci() for m in f.rec.board.move_stack][-2:] == ["e5c7", "a5b4"]


def test_a_turned_board_is_not_adopted_while_a_piece_slides_in() -> None:
    # A game from White's side, then the next puzzle from Black's: its setup move is caught
    # sliding on the first frames that show it turned round.
    f = _Feed(2.0)
    f.show(_frame(chess.Board(), chess.WHITE), 3)
    board = chess.Board("4rrk1/ppp2pp1/7p/3n4/3P3q/1P2p2P/P5P1/R1BQRBK1 w - - 2 23")
    solver = chess.BLACK
    setup = chess.Move.from_uci("c1b2")
    sliding = _frame(board, solver, animating=(setup.from_square, setup.to_square, 0.5))
    f.show(sliding, 2)  # the labels read turned twice, the bishop still in flight
    board.push(setup)
    f.show(_frame(board, solver, highlight=setup), 4)
    assert f.rec.board.board_fen() == board.board_fen()
    # The solver plays on, and the record follows.
    move = board.push_uci("h4f2")
    f.show(_frame(board, solver, highlight=move), 3)
    assert f.rec.board.board_fen() == board.board_fen()
    assert f.rec.board.move_stack and f.rec.board.move_stack[-1].uci() == "h4f2"
