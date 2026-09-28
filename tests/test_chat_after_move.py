"""What the player says while the opponent is to move is, offline, about the move they just
played (a simulated student's session, 2026-09-26: "I didn't even notice they were aiming for
my rook" after 9.Nf3? was answered for the side to move -- "You can win a knight. Their knight
on f3 can simply be taken" -- the student's own knight); with a model, the model reads it."""

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
from tests.coaching import directive_of, drain, offline_coach, scripted_coach, sent


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


ITALIAN = ["e2e4", "e7e5", "g1f3", "b8c6"]
NG5 = [*ITALIAN, "f1c4", "g8f6", "f3g5"]


def test_chat_is_about_the_players_move_on_the_opponents_turn_and_the_position_on_theirs() -> None:
    """Offline, words said on the opponent's turn are about the player's last move unless they
    ask about something else. With a model only a question about the move in so many words
    is: the model reads the rest from the record, and the guess made for it answered "how
    would this position continues?" four times as a question about the move just made (Tai's
    puzzle, 2026-09-28)."""
    # (the moves, the player's mistake, what they say, the ply of the move it is about offline,
    # and with a model)
    rows = [
        # The player's 4.Ng5 is the last move: offline, their words are about it.
        (NG5, 7, "I didn't even notice they were aiming for my rook", 7, None),
        # About the opponent's side of the board, the game or how it goes on: not the move.
        (NG5, 7, "what are they threatening?", None, None),
        (NG5, 7, "how did I do this game?", None, None),
        (NG5, 7, "how would this position continue?", None, None),
        # About the move in so many words.
        (NG5, 7, "why was my last move bad?", 7, 7),
        # On the player's own turn their words are about the position...
        (ITALIAN, 3, "I didn't even notice they were aiming for my rook", None, None),
        # ...unless they ask about their move in so many words.
        (ITALIAN, 3, "why was my last move bad?", 3, 3),
    ]
    for ucis, mistake_at, said, offline, with_model in rows:
        for coach, about in ((offline_coach(), offline), (scripted_coach([])[0], with_model)):
            _played(coach, ucis, mistake_at=mistake_at)
            assert coach.user_to_move is (len(ucis) % 2 == 0)
            own = coach._asked_own_move(said)
            assert (own.move.ply if own is not None else None) == about, said


def test_what_a_question_names_is_examined_aloud_and_what_a_statement_names_is_not() -> None:
    """Named in a statement or before the question, the pieces came back as the whole answer
    -- where each can go, what each attacks -- and the question went unanswered (the
    simulated students, 2026-09-26)."""
    from sidecoach.coach.coach import question_part

    assert question_part("I see it now: my rook on e1 stays active. Why is h5 their reply?") == (
        "Why is h5 their reply?"
    )
    assert question_part("where can my knight go") == "where can my knight go"
    assert question_part("I see the fork now: 19.Nf6 hits the queen and rook.") == ""
    assert question_part("After 12...h5 what then?") == "After 12...h5 what then?"

    def italian(coach) -> None:
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
        drain(coach)

    # Offline the examination is the answer: a question's pieces are read out, a statement's not.
    coach = offline_coach()
    italian(coach)
    coach.ask("I see it now: my knight on f3 attacks the pawn on e5.", 5.0)
    assert not any("can go to" in m.text for m in drain(coach))
    coach.ask("Where can my knight on f3 go?", 6.0)
    assert any("Your knight on f3 can go to" in m.text for m in drain(coach))
    # With a model to answer in words, a piece's inventory said first is noise ("Your king on
    # g8 can go to f7, g7, h8; ..." before the AI's own answer).
    coach, _ = scripted_coach(["[p=1.0] It can go to d4, e5, g5 or h4."])
    italian(coach)
    coach.ask("Where can my knight on f3 go?", 6.0)
    assert not any("Your knight on f3 can go to" in m.text for m in drain(coach))


def test_an_answer_to_the_coachs_question_closes_it_without_another() -> None:
    """Every answer to the coach's question had earned a new question and none was resolved
    (the simulated students: "what Nc4 was expected to do was asked but not answered")."""

    coach, provider = scripted_coach(["You have it: the knight was loose.", "Right."])
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
    drain(coach)
    with coach._lock:
        coach.conversation.coach_said("What were you expecting that move to do?")
    assert coach._answering_our_question("I wanted to attack f7 with the knight.")
    assert not coach._answering_our_question("Why is that bad?")
    coach.ask("I wanted to attack f7 with the knight.", 8.0)
    drain(coach)
    directive = directive_of(sent(provider))
    assert "They are answering the question you just asked" in directive
    assert "then one question" not in directive
    # Once answered, the thread is closed: words after the coach's reply answer nothing.
    assert not coach._answering_our_question("ok")


def test_a_piece_asked_about_is_read_on_the_board_as_it_stands() -> None:
    """Asked about their own last move, the coach answers from the position it was played
    from -- but "b5 is empty" was said with the player's bishop on b5 (the students' round
    6): pieces and squares are read on the board as it stands."""

    coach = offline_coach()
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
    drain(coach)
    assert not coach.user_to_move
    coach.ask("Is my bishop on b5 safe there?", 6.0)
    said = " ".join(m.text for m in drain(coach))
    assert "Your bishop on b5" in said and "b5 is empty" not in said
