"""Premove queues against the reconciler, the way chess.com runs them: a seeded sweep.

The site is simulated (`_Site`): the true game, and the user's premove queue while the
opponent thinks. Queued moves are made on the premove board (the position with the queue
played, the opponent passing); the site draws each queued piece at the end of its queue --
as the piece a queued pawn promotes to -- leaves its real square empty and tints every
queued square red over the last-move highlight. When the opponent moves, the first queued
move executes at once, or, when it cannot be played, every premove is cleared and the board
looks as it normally does (Tai, 2026-09-28).

What must hold is Tai's rule: once the queue has resolved and the board is still, the
reconciler holds the true position and whose move it is, and follows on from there. The
moves' order may be off; a move nobody played is tolerable only when nothing else fits;
a wrong position or a wrong side to move is a broken game. A handful of seeds run in the
suite; `--seeds` runs a sweep:

    uv run pytest tests/test_premove_sweep.py -q -n auto --seeds 500

How closely the moves themselves were told (exact, reordered, invented) is measured
alongside, not asserted: 393 of 400 seeds exact on 2026-09-28.

Found by it and fixed (reconcile.py): a queued pawn capture, drawn over its victim; a
capture in the middle of a queue; a king queued the long way round (e1-d2-d1); a queue the
opponent has blocked since; a queue that promotes and goes on as the queen; two readings of
one picture told apart by the order the queue was built in; and a ghost taken for a move
played (Qe4+ where the queen had only got to h4), which left the game a piece off for good.
"""

from __future__ import annotations

import random

import chess
import pytest

from sidecoach.events import Side
from tests.games import Game, Queue, sans_of

SEEDS = range(1, 3)  # --seeds N for the sweep
# Frames of each picture. A bot takes a second or more, and a premove executes the instant
# it answers, in the same picture: at the capture's rate every picture is seen, mostly on
# several frames. One frame is a picture the reconciler cannot settle on alone. (A picture
# never seen at all leaves a king's route or two queues' order unknowable, and the
# reconciler tells one of the stories that fit: that is test_premoves' skip_states case,
# not this sweep's.)
FRAMES = (1, 2, 2, 3, 4)
# Frames of the still board at the end: two seconds.
STILL = 20

# Endgames and middlegames where the user can queue: pawn runs, king walks, a queen's or a
# rook's checks. White to move in each; the side at the bottom is the seed's.
POSITIONS = (
    "8/1p6/8/1P2K3/8/2pk4/8/8 w - - 0 1",  # Tai's: passed pawns for both, the kings apart
    "8/5k2/8/8/8/8/1P4K1/8 w - - 0 1",  # a lone pawn and the kings
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",  # a back-rank rook
    "8/8/4k3/8/2P1P3/8/5K2/8 w - - 0 1",  # two pawns against the king
    "r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PP3PPP/R2Q1RK1 w kq - 0 1",  # a middlegame
    "4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1",  # four pawns racing
)


class _Site:
    """The true game and the user's queue, as chess.com keeps and draws them."""

    def __init__(self, rng: random.Random, fen: str, bottom: Side) -> None:
        self.rng = rng
        self.user = bottom.color
        self.game = Game(bottom=bottom, fen=fen)
        self.queue: list[chess.Move] = []
        self.last: chess.Move | None = None
        self.played: list[str] = []
        # The positions of the game so far. A move back to one of them reads as the player
        # browsing the move list, which the reconciler cannot yet tell from a repetition --
        # a gap of its own, outside this sweep's premoves: no move here repeats a position.
        self.seen = {self._key(self.board)}

    @staticmethod
    def _key(board: chess.Board) -> str:
        return board.board_fen()  # a picture shows the pieces, not whose move it is

    def _repeats(self, *moves: chess.Move) -> bool:
        """Whether playing `moves` on the true board passes through a position seen before."""
        board = self.board.copy(stack=False)
        for move in moves:
            if move not in board.legal_moves:
                return False
            board.push(move)
            if self._key(board) in self.seen:
                return True
        return False

    def _push(self, move: chess.Move) -> None:
        self.played.append(self.board.san(move))
        self.board.push(move)
        self.last = move
        self.seen.add(self._key(self.board))

    @property
    def board(self) -> chess.Board:
        return self.game.board

    def premove_board(self) -> chess.Board:
        """The position with the queue played and the opponent passing: what the next
        premove is made on."""
        board = self.board.copy(stack=False)
        for move in self.queue:
            board.turn = self.user
            board.push(move)
        board.turn = self.user
        return board

    def queues(self) -> tuple[Queue, ...]:
        """The queue as the renderer takes it: one chain of moves per piece, in order."""
        chains: list[list[chess.Move]] = []
        for move in self.queue:
            chain = next((c for c in chains if c[-1].to_square == move.from_square), None)
            if chain is None:
                chains.append([move])
            else:
                chain.append(move)
        return tuple(tuple(c) for c in chains)

    def show(self, frames: int) -> None:
        """`frames` pictures of the board as the site draws it now."""
        self.game.show(frames=frames, highlight=self.last, premoves=self.queues())

    def pick(self, board: chess.Board, *, continuing: chess.Square | None) -> chess.Move | None:
        """A move for the user on `board`: the queued piece again when it can go on, a pawn
        push often (a run to promotion is what a queue is for), else any; a promotion is to a
        queen, chess.com's own choice. Never onto a square the queue already covers: a queue
        is a way somewhere, and one that comes back over itself (a rook c1-b1-c1) draws the
        same picture as a shorter one, which nobody could tell apart."""
        covered = {sq for m in self.queue for sq in (m.from_square, m.to_square)}
        moves = [
            m
            for m in board.legal_moves
            if m.promotion in (None, chess.QUEEN)
            and not board.is_castling(m)
            and m.to_square not in covered
            and (continuing is None or m.from_square == continuing or m.from_square not in covered)
        ]
        if not moves:
            return None
        if continuing is not None and self.rng.random() < 0.7:
            chain = [m for m in moves if m.from_square == continuing]
            if chain:
                return self.rng.choice(chain)
        pawns = [m for m in moves if board.piece_type_at(m.from_square) == chess.PAWN]
        if pawns and self.rng.random() < 0.5:
            return self.rng.choice(pawns)
        return self.rng.choice(moves)

    def user_moves(self) -> None:
        """The user's turn, nothing queued: a move, then its pictures."""
        move = self.pick(self.board, continuing=None)
        for _ in range(20):
            assert move is not None
            if not self._repeats(move):
                break
            move = self.pick(self.board, continuing=None)
        assert move is not None
        self._push(move)
        self.show(self.rng.choice(FRAMES))

    def queue_up(self) -> None:
        """While the opponent thinks: one to five premoves, shown as they are made."""
        for _ in range(self.rng.choice((1, 2, 3, 3, 4, 5))):
            board = self.premove_board()
            continuing = self.queue[-1].to_square if self.queue else None
            move = self.pick(board, continuing=continuing)
            if move is None:
                return
            self.queue.append(move)
            self.show(self.rng.choice(FRAMES))

    def opponent_moves(self) -> None:
        """The opponent's move and, the same instant, the first queued move -- or the queue
        cleared when that move has become illegal."""
        replies = list(self.board.legal_moves)
        fresh = [r for r in replies if not self._repeats(r, *self.queue[:1])]
        reply = self.rng.choice(fresh or replies)
        self._push(reply)
        if any(self.board.color_at(c[0].from_square) != self.user for c in self.queues()):
            # A queued piece taken: its queue is void at once (test_premoves'
            # test_the_opponent_captures_the_queued_piece), and with it every premove, as
            # when one cannot be played.
            self.queue = []
        if self.queue and not self.board.is_game_over():
            step = self.queue.pop(0)
            if step in self.board.legal_moves:
                self._push(step)
            else:
                # Drawn until its turn came; then chess.com clears every premove.
                self.queue = []
        self.show(self.rng.choice(FRAMES))


@pytest.mark.vision
def test_premove_queues_are_followed(seed: int) -> None:
    rng = random.Random(seed)
    bottom = rng.choice((Side.WHITE, Side.BLACK))
    fen = rng.choice(POSITIONS)
    site = _Site(rng, fen, bottom)
    if site.board.turn != site.user:
        site.opponent_moves()
    for _ in range(12):
        if site.board.is_game_over():
            break
        if site.board.turn == site.user and not site.queue:
            site.user_moves()
            continue
        if not site.queue and rng.random() < 0.8:
            site.queue_up()
        site.opponent_moves()
    # The queue resolved (or given up) and the board still for two seconds.
    site.queue = []
    site.show(STILL)
    rec, board = site.game.rec, site.board
    where = (seed, fen, bottom, site.played, sans_of(site.game.events))
    assert rec.board.board_fen() == board.board_fen(), where
    assert rec.board.turn == board.turn and not rec._turn_unknown, where
    # And it follows on: the next move is told.
    if not board.is_game_over():
        move = next(m for m in board.legal_moves if not site._repeats(m))
        san = board.san(move)
        site._push(move)
        told = site.game.show(frames=3, highlight=move)
        assert sans_of(told) == [san], (*where, san)
