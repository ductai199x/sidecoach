"""What the player says while the opponent is to move is about the move they just played (a
simulated student's session, 2026-09-26: "I didn't even notice they were aiming for my rook"
after 9.Nf3? was answered for the side to move -- "You can win a knight. Their knight on f3
can simply be taken" -- the student's own knight)."""

from __future__ import annotations

import chess

from sidecoach.events import GameStarted, Label, MoveAssessment, MovePlayed, Severity, Side
from tests.test_coach import _coach


def _played(coach, ucis: list[str], mistake_at: int) -> None:
    board = chess.Board()
    coach.on_event(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=board.fen()), 0.0)
    for ply, uci in enumerate(ucis, 1):
        move = chess.Move.from_uci(uci)
        before, san = board.fen(), board.san(move)
        board.push(move)
        by_user = ply % 2 == 1
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=uci,
                fen_before=before,
                fen_after=board.fen(),
                by_user=by_user,
            ),
            float(ply),
        )
        coach.on_event(
            MoveAssessment(
                game_id="g",
                ply=ply,
                uci=uci,
                expected_loss=0.3 if ply == mistake_at else 0.0,
                severity=Severity.BLUNDER if ply == mistake_at else Severity.NONE,
                best=[uci],
                label=Label.BLUNDER if ply == mistake_at else Label.BEST,
            ),
            float(ply) + 0.5,
        )


def test_chat_on_the_opponents_turn_is_about_the_players_move() -> None:
    coach, _ = _coach([])
    _played(coach, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"], mistake_at=7)
    assert not coach.user_to_move  # the player's 4.Ng5 is the last move
    own = coach._asked_own_move("I didn't even notice they were aiming for my rook")
    assert own is not None and own.move.ply == 7
    # About the opponent's side of the board: not the player's move.
    assert coach._asked_own_move("what are they threatening?") is None
    assert coach._asked_own_move("how did I do this game?") is None


def test_chat_on_the_players_own_turn_is_about_the_position() -> None:
    coach, _ = _coach([])
    _played(coach, ["e2e4", "e7e5", "g1f3", "b8c6"], mistake_at=3)
    assert coach.user_to_move
    assert coach._asked_own_move("I didn't even notice they were aiming for my rook") is None
    # Asked in so many words, it is still about the player's move.
    assert coach._asked_own_move("why was my last move bad?") is not None
