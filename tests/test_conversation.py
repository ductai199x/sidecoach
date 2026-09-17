"""The conversation renderer: what a ply message says about the move and its verdict."""

from __future__ import annotations

import chess

from sidecoach.coach.conversation import Conversation
from sidecoach.events import GameStarted, Label, MoveAssessment, MovePlayed, Severity, Side


def test_the_preferred_line_is_dropped_only_when_its_first_move_was_played() -> None:
    """After 10...O-O (a mistake) the engine's line was 10...Nxe4 11. Qxd4 Nf6 12. O-O:
    it ends on the move played, and the verdict lost the line entirely. The line is
    dropped only when its first move is the one played."""
    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    fen_before = "r1bqk2r/2p1bppp/p2p1n2/1p1P4/3pP3/1B6/PPP1NPPP/R1BQK2R b KQkq - 1 10"
    board = chess.Board(fen_before)
    move = board.parse_san("O-O")
    board.push(move)
    record = convo.move(
        MovePlayed(
            game_id="g",
            ply=20,
            san="O-O",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=False,
        )
    )
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=20,
            uci=move.uci(),
            expected_loss=0.13,
            severity=Severity.MISTAKE,
            best=["f6e4", "d1d4", "e4f6", "e1g1"],
            label=Label.MISTAKE,
        )
    )
    text = convo.render_ply(record)
    assert "the engine preferred 10... Nxe4 11. Qxd4 Nf6 12. O-O" in text
    # The line whose first move is the one played is not repeated back.
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=20,
            uci=move.uci(),
            expected_loss=0.01,
            severity=Severity.NONE,
            best=[move.uci(), "a2a4"],
            label=Label.GOOD,
        )
    )
    assert "the engine preferred" not in convo.render_ply(record)


def test_the_opponents_facts_are_reworded_whole_word_from_the_first_word() -> None:
    """A fact for the side to move that began with "you" kept it when reworded for the
    opponent: the old table matched " you " with a space on each side."""
    from sidecoach.coach.conversation import _theirs

    assert _theirs("you still have 2 pieces at home (b8, c8); they are not in the game yet") == (
        "they still have 2 pieces at home (b8, c8); they are not in the game yet"
    )
    assert _theirs("your king is still in the centre on e8 and can still castle") == (
        "their king is still in the centre on e8 and can still castle"
    )
    assert _theirs("Your opponent's rook holds the file; it attacks you.") == (
        "Their rook holds the file; it attacks them."
    )
    assert _theirs("yours is the better bishop") == "theirs is the better bishop"


def test_the_verdict_says_what_the_move_does_and_what_the_better_move_would_do() -> None:
    """The coach's commonest invention was the reason a move was good or bad. The verdict
    now carries the mechanism, computed on the board: what the move takes, attacks,
    defends, leaves undefended; the same for the move the engine preferred."""
    from sidecoach.coach.effects import describe_effects

    # After 1. e4 e5 2. Nf3 Nc6 3. Bb5: the bishop pins nothing yet but attacks the knight.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert describe_effects(board, board.parse_san("Bb5")) == ["attacks the knight on c6"]
    # A queen stepping onto an open file with nothing to attack: the file is named.
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert "takes the open d-file" in describe_effects(board, board.parse_san("Qd1"))
    # In the message: the played move's effects and the better move's.
    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    fen_before = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    board = chess.Board(fen_before)
    move = board.parse_san("Nc3")
    board.push(move)
    record = convo.move(
        MovePlayed(
            game_id="g",
            ply=5,
            san="Nc3",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        )
    )
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=5,
            uci=move.uci(),
            expected_loss=0.03,
            severity=Severity.INACCURACY,
            best=["f1b5"],
            label=Label.INACCURACY,
        )
    )
    text = convo.render_ply(record)
    assert "What Bb5 would do: attacks the knight on c6." in text
    assert "What Nc3 does:" not in text  # a quiet developing move: nothing the board can name
