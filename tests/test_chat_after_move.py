"""What the player says while the opponent is to move is about the move they just played (a
simulated student's session, 2026-09-26: "I didn't even notice they were aiming for my rook"
after 9.Nf3? was answered for the side to move -- "You can win a knight. Their knight on f3
can simply be taken" -- the student's own knight)."""

from __future__ import annotations

import chess

from sidecoach.events import (
    Analysis,
    GameStarted,
    Label,
    Line,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
)
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


def test_what_a_question_names_is_examined_aloud_and_what_a_statement_names_is_not() -> None:
    """Named in a statement or before the question, the pieces came back as the whole answer
    -- where each can go, what each attacks -- and the question went unanswered (the
    simulated students, 2026-09-26)."""
    from sidecoach.coach.coach import question_part
    from tests.test_coach import _messages

    assert question_part("I see it now: my rook on e1 stays active. Why is h5 their reply?") == (
        "Why is h5 their reply?"
    )
    assert question_part("where can my knight go") == "where can my knight go"
    assert question_part("I see the fork now: 19.Nf6 hits the queen and rook.") == ""
    assert question_part("After 12...h5 what then?") == "After 12...h5 what then?"

    coach, _ = _coach([])
    _played(coach, ["e2e4", "e7e5", "g1f3", "b8c6"], mistake_at=3)
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=4,
            lines=[Line(moves=["f1c4"], depth=12, expected_score=0.55)],
            decisive=False,
            only_move=False,
            tags=[],
        ),
        4.5,
    )
    _messages(coach)
    coach.ask("I see it now: my knight on f3 attacks the pawn on e5.", 5.0)
    assert not any("can go to" in m.text for m in _messages(coach))
    coach.ask("Where can my knight on f3 go?", 6.0)
    assert any("Your knight on f3 can go to" in m.text for m in _messages(coach))


def test_an_answer_to_the_coachs_question_closes_it_without_another() -> None:
    """Every answer to the coach's question had earned a new question and none was resolved
    (the simulated students: "what Nc4 was expected to do was asked but not answered")."""
    from tests.test_coach import _directive, _given, _messages

    coach, provider = _coach(["You have it: the knight was loose.", "Right."])
    _played(coach, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"], mistake_at=7)
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5"):
        board.push_uci(uci)
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=7,
            lines=[Line(moves=["d7d5"], depth=12, expected_score=0.45)],
            decisive=False,
            only_move=False,
            tags=[],
        ),
        7.5,
    )
    _messages(coach)
    with coach._lock:
        coach.conversation.coach_said("What were you expecting that move to do?")
    assert coach._answering_our_question("I wanted to attack f7 with the knight.")
    assert not coach._answering_our_question("Why is that bad?")
    coach.ask("I wanted to attack f7 with the knight.", 8.0)
    _messages(coach)
    directive = _directive(_given(provider))
    assert "They are answering the question you just asked" in directive
    assert "then one question" not in directive
    # Once answered, the thread is closed: words after the coach's reply answer nothing.
    assert not coach._answering_our_question("ok")


def test_a_piece_asked_about_is_read_on_the_board_as_it_stands() -> None:
    """Asked about their own last move, the coach answers from the position it was played
    from -- but "b5 is empty" was said with the player's bishop on b5 (the students' round
    6): pieces and squares are read on the board as it stands."""
    from tests.test_coach import _messages

    coach, _ = _coach([])
    ucis = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
    _played(coach, ucis, mistake_at=5)
    board = chess.Board()
    for ply, uci in enumerate(ucis, 1):
        board.push_uci(uci)
        if ply >= 4:
            coach.on_event(
                Analysis(
                    game_id="g",
                    fen=board.fen(),
                    ply=ply,
                    lines=[
                        Line(moves=["a7a6" if ply == 5 else "f1b5"], depth=12, expected_score=0.5)
                    ],
                    decisive=False,
                    only_move=False,
                    tags=[],
                ),
                ply + 0.7,
            )
    _messages(coach)
    assert not coach.user_to_move
    coach.ask("Is my bishop on b5 safe there?", 6.0)
    said = " ".join(m.text for m in _messages(coach))
    assert "Your bishop on b5" in said and "b5 is empty" not in said


def test_chat_on_the_players_own_turn_is_about_the_position() -> None:
    coach, _ = _coach([])
    _played(coach, ["e2e4", "e7e5", "g1f3", "b8c6"], mistake_at=3)
    assert coach.user_to_move
    assert coach._asked_own_move("I didn't even notice they were aiming for my rook") is None
    # Asked in so many words, it is still about the player's move.
    assert coach._asked_own_move("why was my last move bad?") is not None
