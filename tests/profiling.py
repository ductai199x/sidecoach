"""The profiler fed a game: the moves and the engine's verdicts as the pipeline and the
annotator emit them, with no engine. The phone's fixture maker records the profilers these
make (tools/make_profile_fixtures.py swaps `Profiler` here and in test_profile)."""

from __future__ import annotations

import chess

from sidecoach.events import (
    Analysis,
    GameEnded,
    GameStarted,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
)
from sidecoach.profile.profiler import Profiler
from sidecoach.profile.store import Store


class Feed:
    """Events of a game the way the pipeline and the annotator emit them."""

    def __init__(self, user_side: Side = Side.WHITE) -> None:
        self.store = Store(None)
        self.profiler = Profiler(self.store, session_id="s", ts=0.0)
        self.board = chess.Board()
        self.ply = 0
        self.ts = 0.0
        self.user_side = user_side
        self.out: list = []
        self.game_id = "g"
        self.send(GameStarted(game_id="g", user_side=user_side, start_fen=chess.STARTING_FEN))

    def next_game(self, game_id: str, *, kind: str = "game", fen: str = chess.STARTING_FEN) -> list:
        """The next game begun, from `fen` (`kind` "position": a game joined from the middle)."""
        self.game_id = game_id
        self.board = chess.Board(fen)
        self.ply = 0
        return self.send(
            GameStarted(game_id=game_id, user_side=self.user_side, start_fen=fen, kind=kind)
        )

    def send(self, event) -> list:
        self.ts += 1.0
        got = self.profiler.on_event(event, self.ts)
        self.out += got
        return got

    def unjudged(self, san: str) -> list:
        """A move as the pipeline reports it, before the engine has judged it."""
        move = self.board.parse_san(san)
        fen_before = self.board.fen()
        by_user = (self.board.turn == chess.WHITE) == (self.user_side is Side.WHITE)
        self.board.push(move)
        self.ply += 1
        return self.send(
            MovePlayed(
                game_id=self.game_id,
                ply=self.ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=self.board.fen(),
                by_user=by_user,
                think_ms=3000,
            )
        )

    def play(
        self,
        san: str,
        *,
        loss: float = 0.0,
        label: Label = Label.GOOD,
        tags: list[MotifTag] | None = None,
        after_tags: list[MotifTag] | None = None,
        reply: str | None = None,
        reply_mate: int | None = None,
        analysis: bool = True,
    ) -> list:
        got = self.unjudged(san)
        move = self.board.peek()
        severity = (
            Severity.BLUNDER
            if loss >= 0.28
            else Severity.MISTAKE
            if loss >= 0.10
            else Severity.INACCURACY
            if loss >= 0.04
            else Severity.NONE
        )
        got += self.send(
            MoveAssessment(
                game_id=self.game_id,
                ply=self.ply,
                uci=move.uci(),
                expected_loss=loss,
                severity=severity,
                best=["g1f3"] if loss else [move.uci()],
                tags=tags or [],
                depth=14,
                label=label,
                accuracy=max(0.0, 100.0 - loss * 200),
            )
        )
        if analysis:
            lines = (
                []
                if self.board.is_game_over()
                else [
                    Line(
                        moves=[reply] if reply else [],
                        depth=14,
                        expected_score=0.5,
                        mate=reply_mate,
                    )
                ]
            )
            got += self.send(
                Analysis(
                    game_id=self.game_id,
                    fen=self.board.fen(),
                    ply=self.ply,
                    lines=lines,
                    decisive=False,
                    only_move=False,
                    tags=after_tags or [],
                )
            )
        return got

    def end(self, result: str = "1-0", reason: str = "resignation") -> list:
        return self.send(
            GameEnded(
                game_id=self.game_id,
                result=result,
                reason=reason,
                pgn="1. e4",
                final_fen=self.board.fen(),
            )
        )
