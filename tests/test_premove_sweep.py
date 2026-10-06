"""Premove queues against the reconciler, the way the sites run them: a seeded sweep.

The site is simulated (`_Site`) from the sites' own code (read 2026-10-05; see
docs/reconciler-model.md, section 3):

- chess.com keeps a queue of any length. A step is any move the piece could make alone on
  an empty board -- through pieces, onto the user's own, a pawn's diagonal onto an empty
  square, castling -- made from where the piece is drawn. The board draws the queue played
  out over the position, each step over whatever stood on its square
  (`synth.premove_picture`). When the opponent moves, the first step is played at once if it
  is legal; if not, the whole queue is cleared. A right-click cancels it all.
- lichess keeps one premove, draws the pieces where they stand, and plays it the same way.

Every seed draws the premove tint and the last-move highlight in its own colours, or no
premove tint at all: nothing the reconciler decides may rest on a colour.

What must hold (docs/reconciler-model.md, section 6), on every picture and not only at the
end: every move told was played, from the position it was told from; the game the
reconciler holds is a position the true game passed through, with its side to move. And
once the board is still and one more move is played, the reconciler holds the true position
and whose move it is. A handful of seeds run in the suite; `--seeds` runs a sweep:

    uv run pytest tests/test_premove_sweep.py -q -n auto --seeds 500
"""

from __future__ import annotations

import random

import chess
import pytest

from sidecoach.events import MovePlayed, Side, TurnPending
from tests.games import Game, sans_of
from tests.synth import HIGHLIGHTS, PREMOVE_NAVY, PREMOVE_RED, premove_picture

# Seed 117 leaves the side to move open for a while: the suite always sees it told so. A few
# games by default; the sweep is `--seeds N` (Tai, 2026-10-06: sweeps are not run by default).
SEEDS = (*range(1, 5), 117)
# Frames of each picture. A bot takes a second or more, and a premove executes the instant
# it answers, in the same picture; a bot that premoves too answers within a frame.
FRAMES = (1, 2, 2, 3, 4)
# The longest queue a player makes here. Chess.com takes any length; a queue is typically three
# or four steps (Tai's games and streams), and queue-ups stacked one on another made a king
# walk eleven squares ahead of the game -- a picture of no game anyone plays.
MAX_QUEUE = 6
# Frames of the still board at the end: two seconds.
STILL = 20

# Endgames and middlegames where the user can queue: pawn runs, king walks, a queen's or a
# rook's checks, castling. White to move in each; the side at the bottom is the seed's.
POSITIONS = (
    "8/1p6/8/1P2K3/8/2pk4/8/8 w - - 0 1",  # Tai's: passed pawns for both, the kings apart
    "8/5k2/8/8/8/8/1P4K1/8 w - - 0 1",  # a lone pawn and the kings
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",  # a back-rank rook
    "8/8/4k3/8/2P1P3/8/5K2/8 w - - 0 1",  # two pawns against the king
    "r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PP3PPP/R2Q1RK1 w kq - 0 1",  # a middlegame
    "4k3/8/8/8/8/8/PPPP4/4K3 w - - 0 1",  # four pawns racing
    "r3k2r/pppq1ppp/2n5/8/8/2N5/PPPQ1PPP/R3K2R w KQkq - 0 1",  # both sides may castle
)


def lone_moves(board: chess.Board, square: chess.Square) -> list[chess.Move]:
    """The premoves chess.com offers for the piece drawn on `square`: its moves alone on an
    empty board, with an enemy piece on each of a pawn's forward diagonals (so a pawn may
    always premove a capture) and, for a king, the castling the position still allows (its
    generator `_`)."""
    piece = board.piece_at(square)
    assert piece is not None
    lone = chess.Board(None)
    lone.turn = piece.color
    lone.set_piece_at(square, piece)
    if piece.piece_type == chess.PAWN:
        rank = chess.square_rank(square) + (1 if piece.color == chess.WHITE else -1)
        for file in (chess.square_file(square) - 1, chess.square_file(square) + 1):
            if 0 <= file < 8 and 0 <= rank < 8:
                lone.set_piece_at(
                    chess.square(file, rank), chess.Piece(chess.KNIGHT, not piece.color)
                )
    if piece.piece_type == chess.KING:
        home = 0 if piece.color == chess.WHITE else 7
        for file in (7, 0):
            rook = chess.square(file, home)
            if square == chess.square(4, home) and board.castling_rights & chess.BB_SQUARES[rook]:
                lone.set_piece_at(rook, chess.Piece(chess.ROOK, piece.color))
                lone.castling_rights |= chess.BB_SQUARES[rook]
    out = []
    for move in lone.generate_pseudo_legal_moves(chess.BB_SQUARES[square]):
        if move.promotion not in (None, chess.QUEEN):
            continue  # the queen, chess.com's own choice
        out.append(move)
    return out


class _Site:
    """The true game and the user's queue, as the site keeps and draws them."""

    def __init__(self, rng: random.Random, fen: str, bottom: Side) -> None:
        self.rng = rng
        self.user = bottom.color
        self.chesscom = rng.random() < 0.75
        tints = [PREMOVE_RED if self.chesscom else PREMOVE_NAVY, None]
        tints.append(tuple(rng.randrange(256) for _ in range(3)))
        self.style = {
            "premove_drawn": self.chesscom,
            "premove_tint": rng.choice(tints),
            "highlight_colour": rng.choice(sorted(HIGHLIGHTS)),
        }
        self.game = Game(bottom=bottom, fen=fen)
        self.queue: list[chess.Move] = []
        self.last: chess.Move | None = None
        self.played: list[str] = []
        # The positions of the game so far, with the side to move, and the moves played from
        # each: what anything the reconciler tells must be among.
        self.history = {self._key(self.board)}
        self.moves: set[tuple[str, str]] = set()
        # A move back to a position seen before reads as the player browsing the move list,
        # which a picture cannot tell from a repetition: no move here repeats a placement.
        self.seen = {self.board.board_fen()}
        self.told = 0
        # What the reconciler told or held that the game never was (`check`).
        self.lies: list[tuple[str, str]] = []
        self.pending_seen = False

    @staticmethod
    def _key(board: chess.Board) -> str:
        return board.board_fen() + (" w" if board.turn else " b")

    def _repeats(self, *moves: chess.Move) -> bool:
        """Whether playing `moves` on the true board passes through a placement seen before."""
        board = self.board.copy(stack=False)
        for move in moves:
            if move not in board.legal_moves:
                return False
            board.push(move)
            if board.board_fen() in self.seen:
                return True
        return False

    def _push(self, move: chess.Move) -> None:
        self.moves.add((self._key(self.board), move.uci()))
        self.played.append(self.board.san(move))
        self.board.push(move)
        self.last = move
        self.seen.add(self.board.board_fen())
        self.history.add(self._key(self.board))

    @property
    def board(self) -> chess.Board:
        return self.game.board

    def show(self, frames: int) -> None:
        """`frames` pictures of the board as the site draws it now, each checked."""
        for _ in range(frames):
            self.game.show(
                frames=1, highlight=self.last, premoves=(tuple(self.queue),), **self.style
            )
            self.check()
        self.settled()

    def check(self) -> None:
        """Tai's order (2026-10-05), after every picture. First, the side to move is never
        wrong: whenever the reconciler has caught up with the screen, its side to move is the
        game's. That is asserted. Then the moves: one told that was not played, or a game held
        that the true game never passed through, is counted (`lies`), not asserted -- a few
        pictures cannot be told apart from the pieces alone (a premove that takes a piece back
        to the square it came from draws the move list stepped back)."""
        rec = self.game.rec
        for event in self.game.events[self.told :]:
            if isinstance(event, MovePlayed):
                before = chess.Board(event.fen_before)
                if (self._key(before), event.uci) not in self.moves:
                    self.lies.append(("told", event.san))
        told = [e.pending for e in self.game.events if isinstance(e, TurnPending)]
        self.told = len(self.game.events)
        # A side to move the stories disagree on is told pending, and settled when they agree.
        assert (told[-1] if told else False) == rec.turn_pending, ("pending", self.played)
        self.pending_seen |= rec.turn_pending
        if self._key(rec.board) not in self.history and not rec.turn_unknown:
            self.lies.append(("held", rec.board.fen()))

    def settled(self) -> None:
        """Tai (2026-10-05): "it could be misread briefly but it needs to be correct the
        immediate next move or moves or whenever the premoves resolve or cancelled". So at
        the end of each step of the game with no premove pending, once the reconciler says it
        has caught up with the screen, its position and side to move are the game's."""
        rec = self.game.rec
        if self.queue or not rec.agrees or rec.pending_story or rec.turn_unknown:
            return
        if any(story.queue for story in rec._stories):
            # The reconciler holds a premove still to play that the site has played or
            # dropped: a queued step lands over what it covers, so whether the king took the
            # pawn before or after it moved can be invisible. The opponent's next move plays
            # the step and the game is the site's again (`test_premove_queues_are_followed`).
            return
        assert self._key(rec.board) == self._key(self.board), (
            "turn" if rec.board.board_fen() == self.board.board_fen() else "position",
            rec.board.fen(),
            self.played,
        )

    def picture(self) -> chess.Board:
        return premove_picture(self.board, self.queue)

    def pick(self, board: chess.Board, square: chess.Square | None) -> chess.Move | None:
        """A legal move for the user on the true board: a pawn push often, else any."""
        moves = [
            m
            for m in board.legal_moves
            if (square is None or m.from_square == square) and m.promotion in (None, chess.QUEEN)
        ]
        if not moves:
            return None
        pawns = [m for m in moves if board.piece_type_at(m.from_square) == chess.PAWN]
        if pawns and self.rng.random() < 0.5:
            return self.rng.choice(pawns)
        return self.rng.choice(moves)

    def user_moves(self) -> None:
        """The user's turn, nothing queued: a move, then its pictures."""
        for _ in range(20):
            move = self.pick(self.board, None)
            assert move is not None
            if not self._repeats(move):
                break
        self._push(move)
        # A hand that has just played premoves on no sooner than a third of a second later.
        self.show(max(3, self.rng.choice(FRAMES)))

    def queue_up(self) -> None:
        """While the opponent thinks: premoves, shown as they are made. On chess.com one to
        five steps, the same piece again most often, each from where the piece is drawn; on
        lichess one premove, which replaces any before it."""
        steps = self.rng.choice((1, 2, 3, 3, 4, 5)) if self.chesscom else 1
        for _ in range(steps):
            if len(self.queue) >= MAX_QUEUE:
                return
            picture = self.picture()
            drawn = [sq for sq, p in picture.piece_map().items() if p.color == self.user]
            last = self.queue[-1].to_square if self.queue else None
            again = last in drawn and self.rng.random() < 0.7
            square = last if again and last is not None else self.rng.choice(drawn)
            # Never onto a square the queue already covers: a queue that comes back over
            # itself (a king e7-f8-g7-f8-e7) draws the same picture as a shorter one, or as
            # none, which no reader of pieces can tell apart -- the reconciler keeps both, and
            # the next move is all that could decide (seed 68 of the first faithful sweep).
            covered = {sq for m in self.queue for sq in (m.from_square, m.to_square)}
            moves = [
                m
                for m in lone_moves(picture, square)
                if m.to_square not in covered and not self._draws_the_past(m)
            ]
            if not moves:
                return
            pawns = [m for m in moves if picture.piece_type_at(square) == chess.PAWN]
            move = self.rng.choice(pawns if pawns and self.rng.random() < 0.5 else moves)
            self.queue = [*self.queue, move] if self.chesscom else [move]
            self.show(self.rng.choice(FRAMES))

    def _draws_the_past(self, step: chess.Move) -> bool:
        """Whether queueing `step` draws a placement the game has stood on: a piece premoved
        back to the square it came from draws the move list stepped back, or the user's own
        move taken back, which no reader of pieces can tell apart. Tai (2026-10-05): a
        takeback and browsing must always be read right, a premove may be read a bit wrong --
        so the reconciler reads such a picture as browsing, and test_premoves'
        test_a_premove_back_to_where_the_piece_came_from_is_put_right_by_the_queue covers
        the premove. This sweep leaves it out."""
        drawn = premove_picture(self.board, [*self.queue, step])
        return drawn.board_fen() in self.seen

    def cancel(self) -> None:
        """The user cancels the queue (a right-click clears every premove): the board looks
        as it normally does again (Tai, 2026-10-05: "I can premove but sometimes I can also
        cancel said premove")."""
        self.queue = []
        self.show(self.rng.choice(FRAMES))

    def opponent_moves(self) -> None:
        """The opponent's move and, the same instant, the first queued step if it is legal,
        else the whole queue cleared (chess.com's `consume`)."""
        replies = list(self.board.legal_moves)
        fresh = [r for r in replies if not self._repeats(r)]
        self._push(self.rng.choice(fresh or replies))
        if self.queue and not self.board.is_game_over():
            step = self.queue.pop(0)
            if step in self.board.legal_moves and not self._repeats(step):
                self._push(step)
            else:
                self.queue = []
        elif self.queue:
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
        if rng.random() < 0.8:
            site.queue_up()
            if site.queue and rng.random() < 0.3:
                site.cancel()  # thought better of it
                if rng.random() < 0.5:
                    site.queue_up()  # and queued something else
        site.opponent_moves()
    # The queue given up and the board still for two seconds; then moves until the opponent
    # has moved after the user. By then the game held is the site's, the side to move
    # included: a picture can leave the record a premove behind (`_Site.settled`), the
    # opponent's next move plays it a ply off, and the one after puts the turn right.
    site.queue = []
    site.show(STILL)
    rec, board = site.game.rec, site.board
    where = (seed, fen, bottom, site.style, site.played, sans_of(site.game.events), site.lies)
    for played in range(3):
        if board.is_game_over() or (played >= 2 and board.turn == site.user):
            break  # the opponent has moved after the user: the cycle a lag heals in
        move = next((m for m in board.legal_moves if not site._repeats(m)), None)
        if move is None:
            return
        site._push(move)
        site.show(4)
    assert site._key(rec.board) == site._key(board), ("position", *where)
    assert not (rec.turn_unknown or rec.turn_pending), ("turn", *where)
