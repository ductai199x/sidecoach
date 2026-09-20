"""A puzzle solved on the panel's own board (Tai, 2026-09-20: "a doing-puzzle mode where
you do the puzzle, moving the pieces, right in the panel, with hint/help by clicking a
chip; no auto hint or anything from the coach").

The panel's board turns interactive: the solver's position from their side, a click on
a piece and a click on a square (the legal targets shown), the line's replies played
back, a wrong move shown for a moment and taken back as Lichess does, the solution on
Give up. The coach is told the position as a game of kind "puzzle" -- the engine
measures it, so Hint, Best move and Line answer -- and says nothing unasked. The
outcome, solved through the last move or failed at the first wrong one, goes to the
profile beside the puzzle's motif (profile/puzzles.py) and moves the puzzle level.

The session is pure state over python-chess, driven by the panel engine on its loop
thread; the panel reads it for the page.
"""

from __future__ import annotations

from typing import Any

import chess

from sidecoach.events import GameStarted, MovePlayed, Side
from sidecoach.profile.puzzles import Puzzle


class PuzzleSession:
    def __init__(self, puzzle: Puzzle, motif: str | None) -> None:
        self.puzzle = puzzle
        self.motif = motif
        self.game_id = f"puzzle-{puzzle.id}"
        self.board = chess.Board(puzzle.fen)
        self.board.push_uci(puzzle.moves[0])  # the opponent's move: the position shown
        self.solver: chess.Color = self.board.turn
        self.side = Side.WHITE if self.solver == chess.WHITE else Side.BLACK
        self.start_fen = self.board.fen()
        self.index = 1  # the next move of the line, the solver's
        self.ply = 0  # plies played on the board since the start
        self.sans: list[str] = []
        self.last_move: str | None = None
        self.wrong: str | None = None  # a wrong move shown, to be taken back
        self.misses = 0
        self.status = "playing"  # playing | solved | given_up
        self.message = "Your move."
        self.recorded = False  # the outcome given to the profile (the panel sets it)

    # ---- what the page shows -------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.status != "playing"

    @property
    def failed(self) -> bool:
        return self.misses > 0 or self.status == "given_up"

    def state(self) -> dict[str, Any]:
        return {
            "id": self.puzzle.id,
            "url": self.puzzle.url,
            "rating": self.puzzle.rating,
            "themes": list(self.puzzle.themes),
            "motif": self.motif,
            "fen": self.board.fen(),
            "bottom": self.side.value,
            "turn": "white" if self.board.turn else "black",
            "your_move": not self.done and self.board.turn == self.solver and self.wrong is None,
            "check": (
                chess.square_name(king)
                if self.board.is_check() and (king := self.board.king(self.board.turn)) is not None
                else None
            ),
            "status": self.status,
            "message": self.message,
            "moves": list(self.sans),
            "last_move": self.last_move,
            "wrong": self.wrong,
            "misses": self.misses,
            "remaining": max(0, (len(self.puzzle.moves) - self.index + 1) // 2),
        }

    def targets(self, square: str) -> list[str]:
        """Where the piece on `square` may go, for the page's dots; empty unless it is the
        solver's turn and their piece."""
        if self.done or self.wrong is not None or self.board.turn != self.solver:
            return []
        try:
            sq = chess.parse_square(square)
        except ValueError:
            return []
        piece = self.board.piece_at(sq)
        if piece is None or piece.color != self.solver:
            return []
        legal = self.board.legal_moves
        return sorted({chess.square_name(m.to_square) for m in legal if m.from_square == sq})

    # ---- the moves -----------------------------------------------------------------------

    def started(self) -> GameStarted:
        return GameStarted(
            game_id=self.game_id, user_side=self.side, start_fen=self.start_fen, kind="puzzle"
        )

    def move(self, uci: str) -> list[MovePlayed]:
        """The solver's move: the line's next move, or any mate where the line mates, is
        right and brings the reply; a wrong one is shown and taken back by `retract`;
        anything else (not their turn, not legal) is ignored. Returns the events of the
        moves that stand."""
        if self.done or self.wrong is not None or self.board.turn != self.solver:
            return []
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return []
        if move not in self.board.legal_moves:
            # A pawn reaching the last rank without a promotion piece: a queen.
            promoted = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if move.promotion is None and promoted in self.board.legal_moves:
                move = promoted
            else:
                return []
        expected = self.puzzle.moves[self.index] if self.index < len(self.puzzle.moves) else None
        if move.uci() != expected and not self._mates(move):
            self.wrong = move.uci()
            self.misses += 1
            self.message = "Not that one. Have another look."
            return []
        events = [self._play(move, by_user=True)]
        self.index += 1
        if self.index >= len(self.puzzle.moves) or self.board.is_game_over():
            self.status = "solved"
            self.message = "Solved." if not self.misses else "Solved, after a miss."
            return events
        reply = chess.Move.from_uci(self.puzzle.moves[self.index])
        events.append(self._play(reply, by_user=False))
        self.index += 1
        if self.index >= len(self.puzzle.moves):
            self.status = "solved"
            self.message = "Solved."
        else:
            self.message = "Your move."
        return events

    def retract(self) -> None:
        """The wrong move shown is taken back; the position stands as it was."""
        self.wrong = None
        if not self.done:
            self.message = "Your move."

    def give_up(self) -> list[MovePlayed]:
        """The rest of the line, played out on the board."""
        if self.done:
            return []
        self.wrong = None
        events: list[MovePlayed] = []
        while self.index < len(self.puzzle.moves):
            move = chess.Move.from_uci(self.puzzle.moves[self.index])
            if move not in self.board.legal_moves:
                break
            events.append(self._play(move, by_user=self.board.turn == self.solver))
            self.index += 1
        self.status = "given_up"
        self.message = "The solution, played out."
        return events

    def _mates(self, move: chess.Move) -> bool:
        """Lichess takes any mate where the line's move mates."""
        expected = self.puzzle.moves[self.index] if self.index < len(self.puzzle.moves) else None
        if expected is None:
            return False
        trial = self.board.copy(stack=False)
        try:
            trial.push_uci(expected)
        except ValueError:
            return False
        if not trial.is_checkmate():
            return False
        trial = self.board.copy(stack=False)
        trial.push(move)
        return trial.is_checkmate()

    def _play(self, move: chess.Move, *, by_user: bool) -> MovePlayed:
        before = self.board.fen()
        san = self.board.san(move)
        self.board.push(move)
        self.ply += 1
        self.sans.append(san)
        self.last_move = move.uci()
        return MovePlayed(
            game_id=self.game_id,
            ply=self.ply,
            san=san,
            uci=move.uci(),
            fen_before=before,
            fen_after=self.board.fen(),
            by_user=by_user,
        )
