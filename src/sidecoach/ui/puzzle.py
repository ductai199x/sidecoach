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
        self.side = Side.of(self.solver)
        self.start_fen = self.board.fen()
        self.index = 1  # the next move of the line, the solver's
        self.ply = 0  # plies played on the board since the start
        self.sans: list[str] = []
        # The positions the page steps through: the start, then one per ply, each with the
        # move that made it, the king in check, and whether the solver played it.
        self.positions: list[dict[str, Any]] = [self._position(None, by_user=False)]
        self.wrong: str | None = None  # a wrong move shown, to be taken back
        # Counts every change, so the page can tell an older answer from a newer one: a poll
        # that left before a move and came back after it showed the move taken back.
        self.version = 0
        self.misses = 0
        self.status = "playing"  # playing | solved | given_up
        self.message = "Your move."
        self.recorded = False  # the outcome given to the profile (the panel sets it)
        # The outcome as the profile took it, for the page's result line once the puzzle has
        # ended (as Lichess shows it): None until the room has recorded it.
        self.result: dict[str, Any] | None = None

    # ---- what the page shows -------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.status != "playing"

    @property
    def failed(self) -> bool:
        return self.misses > 0 or self.status == "given_up"

    def state(self) -> dict[str, Any]:
        your_move = not self.done and self.board.turn == self.solver and self.wrong is None
        return {
            "id": self.puzzle.id,
            "url": self.puzzle.url,
            "rating": self.puzzle.rating,
            "themes": list(self.puzzle.themes),
            "motif": self.motif,
            "version": self.version,
            "positions": list(self.positions),
            "bottom": self.side.value,
            "your_move": your_move,
            "legal": self.legal() if your_move else {},
            "status": self.status,
            "message": self.message,
            "moves": list(self.sans),
            "wrong": self.wrong,
            "misses": self.misses,
            "remaining": max(0, (len(self.puzzle.moves) - self.index + 1) // 2),
            "result": self.result,
        }

    def recorded_as(self, solved: bool, before: int, level: int) -> None:
        """The outcome the profile took: solved or not, the puzzle level after it and how far
        it moved from `before` (the step is cut short at the ends of the range)."""
        self.result = {"solved": solved, "level": level, "change": level - before}

    def legal(self) -> dict[str, list[str]]:
        """Where each of the solver's pieces may go, for the page's dots: sent with the
        state, so a piece picked up shows its squares at once."""
        out: dict[str, set[str]] = {}
        for move in self.board.legal_moves:
            out.setdefault(chess.square_name(move.from_square), set()).add(
                chess.square_name(move.to_square)
            )
        return {square: sorted(targets) for square, targets in out.items()}

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
        self.version += 1
        if move.uci() != expected and not self._mates(move):
            self.wrong = move.uci()
            self.misses += 1
            self.message = "Not quite. Take another look."
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
        self.version += 1
        self.wrong = None
        if not self.done:
            self.message = "Your move."

    def give_up(self) -> list[MovePlayed]:
        """The rest of the line, played out on the board."""
        if self.done:
            return []
        self.version += 1
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

    def _position(self, move: chess.Move | None, *, by_user: bool) -> dict[str, Any]:
        king = self.board.king(self.board.turn)
        checked = self.board.is_check() and king is not None
        return {
            "fen": self.board.fen(),
            "uci": move.uci() if move is not None else None,
            "check": chess.square_name(king) if checked and king is not None else None,
            "by_user": by_user,
        }

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
        self.positions.append(self._position(move, by_user=by_user))
        return MovePlayed(
            game_id=self.game_id,
            ply=self.ply,
            san=san,
            uci=move.uci(),
            fen_before=before,
            fen_after=self.board.fen(),
            by_user=by_user,
        )
