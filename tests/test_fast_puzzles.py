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
from tests.synth import puzzle_frame

pytestmark = pytest.mark.vision


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
    f.show(puzzle_frame(chess.Board(), chess.WHITE), 3)
    board = chess.Board("3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58")
    solver = chess.BLACK
    f.show(puzzle_frame(board, solver), 1)
    setup = board.push_uci("a6a5")
    f.show(puzzle_frame(board, solver, highlight=setup), 4)
    board.push_uci("e5c7")
    reply = board.push_uci("a5b4")
    f.show(puzzle_frame(board, solver, highlight=reply), 4)
    assert f.rec.board.board_fen() == board.board_fen()
    assert [m.uci() for m in f.rec.board.move_stack][-2:] == ["e5c7", "a5b4"]


def test_a_turned_board_is_not_adopted_while_a_piece_slides_in() -> None:
    # A game from White's side, then the next puzzle from Black's: its setup move is caught
    # sliding on the first frames that show it turned round.
    f = _Feed(2.0)
    f.show(puzzle_frame(chess.Board(), chess.WHITE), 3)
    board = chess.Board("4rrk1/ppp2pp1/7p/3n4/3P3q/1P2p2P/P5P1/R1BQRBK1 w - - 2 23")
    solver = chess.BLACK
    setup = chess.Move.from_uci("c1b2")
    sliding = puzzle_frame(board, solver, animating=(setup.from_square, setup.to_square, 0.5))
    f.show(sliding, 2)  # the labels read turned twice, the bishop still in flight
    board.push(setup)
    f.show(puzzle_frame(board, solver, highlight=setup), 4)
    assert f.rec.board.board_fen() == board.board_fen()
    # The solver plays on, and the record follows.
    move = board.push_uci("h4f2")
    f.show(puzzle_frame(board, solver, highlight=move), 3)
    assert f.rec.board.board_fen() == board.board_fen()
    assert f.rec.board.move_stack and f.rec.board.move_stack[-1].uci() == "h4f2"


def test_the_next_puzzle_is_not_adopted_as_its_setup_move_lights_up() -> None:
    # Tai's desktop, 2026-09-24 (dump 20260924-165745): the next puzzle, turned round, was
    # read twice with White's queen still on d1 -- the second time with d1 and b1 already lit
    # for the setup move Qd1-b1 -- and adopted then. The highlight said White had just moved,
    # so Black was to move, the queen's arrival on b1 could not be explained, and the record
    # invented 1...Qxh7 2.Qb1 2...Qh6. The two reads must show the same view: the same
    # pieces and the same lit squares.
    f = _Feed(10.0)
    f.show(puzzle_frame(chess.Board(), chess.WHITE), 3)
    board = chess.Board("2k5/7N/3p3q/1n2p2P/4P3/3P4/2P1KPP1/3Q1B1R w - - 0 1")
    solver = chess.BLACK
    setup = chess.Move.from_uci("d1b1")
    f.show(puzzle_frame(board, solver), 4)  # the puzzle, nothing lit yet
    f.show(puzzle_frame(board, solver, highlight=setup), 3)  # lit, the queen not yet off d1
    for progress in (0.35, 0.8):
        f.show(
            puzzle_frame(
                board,
                solver,
                highlight=setup,
                animating=(setup.from_square, setup.to_square, progress),
            )
        )
    board.push(setup)
    f.show(puzzle_frame(board, solver, highlight=setup), 6)
    assert f.rec.board.board_fen() == board.board_fen()
    assert f.rec.board.turn == chess.BLACK, "the solver is to move"
    # The fork the puzzle is about, recorded as the solver's move and nothing else.
    fork = board.push_uci("b5c3")
    f.show(puzzle_frame(board, solver, highlight=fork), 6)
    assert f.rec.board.board_fen() == board.board_fen()
    played = [m.uci() for m in f.rec.board.move_stack]
    assert played[-1] == "b5c3" and "h6h7" not in played, played
