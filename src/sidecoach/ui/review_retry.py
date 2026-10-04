"""Retry in a reviewed game (docs/game-review-plan.md, screen D): the player plays another move
in the position a move of the game was played from, and it is graded as the review graded the
game's own -- the same annotator and thresholds, the game's own measurements adopted so the
position is read as the review read it, the position the new move leads to searched at the
review's depth; and the coach's hints for the position, as in play (two hints, then the move).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import chess

from sidecoach.analysis.annotator import GameAnnotator, Measured
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.game_review import Strength
from sidecoach.analysis.score import Thresholds, position_score
from sidecoach.config import Settings, find_stockfish
from sidecoach.events import Analysis, DisclosureLevel, ReviewedGame


class Retries:
    """A reviewed game's positions, open to new moves and hints. The engine is made when a new
    move first needs a search, and closed with the room."""

    def __init__(
        self,
        settings: Settings,
        reviewed: ReviewedGame,
        strength: Strength,
        *,
        game_id: str,
        make_engine: Callable[..., Engine] | None = None,
    ) -> None:
        self.settings = settings
        self.reviewed = reviewed
        self.strength = strength
        self.game_id = game_id
        self._make_engine = make_engine
        self._engine: Engine | None = None
        self._annotator: GameAnnotator | None = None
        self._lock = threading.Lock()

    def position(self, ply: int) -> dict[str, Any]:
        """The position move `ply` was played from: its FEN, whose move, and every legal move
        as the board takes them (a square -> the squares its piece may go to)."""
        board = self._board(ply)
        legal: dict[str, list[str]] = {}
        for move in board.legal_moves:
            legal.setdefault(chess.square_name(move.from_square), []).append(
                chess.square_name(move.to_square)
            )
        return {
            "ply": ply,
            "fen": board.fen(),
            "side": "white" if board.turn == chess.WHITE else "black",
            "legal": {sq: sorted(set(to)) for sq, to in legal.items()},
        }

    def attempt(self, ply: int, uci: str) -> dict[str, Any]:
        """A new move from the position of move `ply`, graded: its label, what it cost, its
        accuracy, and the position it leads to."""
        board = self._board(ply)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"{uci} is not legal here")
        with self._lock:
            annotator = self._ready()
            # The move before tells the annotator what it needs of it: whether the opponent
            # had just erred (a miss), whether this is a recapture. Both its positions are the
            # game's, adopted, so this searches nothing.
            if ply > 1:
                before = self.reviewed.moves[ply - 2]
                annotator.assess_move(
                    self.game_id,
                    chess.Board(before.fen_before),
                    chess.Move.from_uci(before.assessment.uci),
                    ply - 1,
                )
            assessment = annotator.assess_move(self.game_id, board, move, ply)
        after = board.copy(stack=False)
        after.push(move)
        return {
            "ply": ply,
            "uci": uci,
            "san": board.san(move),
            "fen": after.fen(),
            "label": assessment.label.value if assessment.label else None,
            "loss": round(assessment.expected_loss, 4),
            "accuracy": assessment.accuracy,
            "only": self._analysis_before(ply).only_move,
        }

    def hint(self, ply: int, level: int) -> dict[str, Any]:
        """The coach's hint at `level` for the position of move `ply` -- 1 the idea, 2 where to
        look, 3 the move -- as the coach gives it in play (`coach.hint_template`)."""
        from sidecoach.coach.coach import hint_template
        from sidecoach.coach.facts import facts_from_analysis

        board = self._board(ply)
        analysis = self._analysis_before(ply)
        previous = self.reviewed.moves[ply - 2].assessment.uci if ply > 1 else None
        facts = facts_from_analysis(board, analysis, last_move=previous)
        level = DisclosureLevel(max(DisclosureLevel.IDEA, min(DisclosureLevel.MOVE, level)))
        best = analysis.lines[0].moves[0] if analysis.lines and analysis.lines[0].moves else None
        return {
            "ply": ply,
            "level": int(level),
            "text": hint_template(facts, level, board),
            "best": best if level >= DisclosureLevel.MOVE else None,
        }

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
            self._engine = None
            self._annotator = None

    # ---- inside ---------------------------------------------------------------------------------

    def _board(self, ply: int) -> chess.Board:
        if not 1 <= ply <= len(self.reviewed.moves):
            raise ValueError(f"no move {ply} in the game")
        return chess.Board(self.reviewed.moves[ply - 1].fen_before)

    def _analysis_before(self, ply: int) -> Analysis:
        return self.reviewed.start if ply == 1 else self.reviewed.moves[ply - 2].after

    def _ready(self) -> GameAnnotator:
        """The annotator, with an engine and the game's own measurements (held under the lock)."""
        if self._annotator is not None:
            return self._annotator
        from sidecoach.cli import _load_book, _load_tagger

        path = find_stockfish(self.settings)
        if path is None:
            raise RuntimeError("no Stockfish to grade the move with")
        engine = self.settings.engine
        self._engine = (self._make_engine or Engine)(
            path=path,
            threads=1,
            hash_mb=min(engine.hash_mb, 128),
            multipv=1,
            pv_plies=engine.pv_plies,
            score_model=engine.score_model,
        )
        self._engine.open()
        annotator = GameAnnotator(
            engine=self._engine,
            depth=self.strength.depth,
            multipv=self.strength.multipv,
            thresholds=Thresholds.from_settings(self.settings.coach),
            tagger=_load_tagger(),
            book=_load_book(self.settings),
            with_threats=False,
        )
        analyses = [self.reviewed.start, *(m.after for m in self.reviewed.moves)]
        annotator.adopt(measured(a) for a in analyses)
        self._annotator = annotator
        return annotator


def measured(analysis: Analysis) -> Measured:
    """A position of the game as the review measured it."""
    board = chess.Board(analysis.fen)
    return Measured(
        fen=analysis.fen,
        lines=analysis.lines,
        score=position_score(board, analysis.lines),
        decisive=analysis.decisive,
        only_move=analysis.only_move,
        threats=list(analysis.threats),
        traps=list(analysis.traps),
        defences=list(analysis.defences),
        threatens=list(analysis.threatens),
    )
