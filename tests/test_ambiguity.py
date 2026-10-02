"""Words that can mean two things (Tai, 2026-10-02: "Look for all of these weird ambiguous stuff
and make sure the AI coach have enough guidelines to be able to ask clarification questions and
the offline coach can still respond correctly"). The AI reads them for itself, from guidelines;
the offline coach (and the AI's fallback) reads the known two-reading patterns by its rules: the
likelier reading where the turn, the words or the verdicts settle it, one short question where
they do not and the readings get different answers. Logic on events, no timing."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach.coach import AI_GUIDELINES, Coach
from sidecoach.events import Label, MoveAssessment, Severity
from tests.coaching import (
    SCHOLAR,
    directive_of,
    drain,
    mate_analysis,
    offline_coach,
    play_to_scholar,
    scripted_coach,
    sent,
)
from tests.test_coach import _looked, _played

AFTER_BISHOP_CHECK = "r1bqkb1r/pppp1Bpp/2n2n2/4p2Q/4P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


def _verdict(coach: Coach, ply: int, uci: str, severity: Severity, label: Label, ts: float) -> None:
    loss = {Severity.NONE: 0.0, Severity.MISTAKE: 0.2, Severity.BLUNDER: 0.45}[severity]
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=ply,
            uci=uci,
            expected_loss=loss,
            severity=severity,
            best=["h5f7"] if ply == 7 else ["e8e7"],
            label=label,
        ),
        ts,
    )


def _bishop_check_and_ke7(coach: Coach, mine: Severity, theirs: Severity) -> None:
    """At SCHOLAR the player plays Bxf7+ (judged `mine`), the opponent Ke7 (judged `theirs`),
    each position read: the player to move, right after the reply."""
    play_to_scholar(coach)
    coach.quiet = True  # no word on the moves: the answers are the only messages
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    board = chess.Board(SCHOLAR)
    _played(coach, board, 7, "c4f7", True, 10.0)
    labels = {Severity.NONE: Label.BEST, Severity.MISTAKE: Label.MISTAKE}
    _verdict(coach, 7, "c4f7", mine, labels.get(mine, Label.MISS), 10.5)
    _looked(coach, board, 7, ["e8e7"], 10.6)
    _played(coach, board, 8, "e8e7", False, 11.0)
    _verdict(coach, 8, "e8e7", theirs, labels.get(theirs, Label.BLUNDER), 11.5)
    _looked(coach, board, 8, ["h5e5"], 11.6)
    drain(coach)


def _answer(coach: Coach, words: str, ts: float) -> str:
    coach.ask(words, ts)
    [message] = drain(coach)
    return message.text


@pytest.mark.parametrize(
    ("mine", "theirs", "said"),
    [
        # Only the player's move went wrong: "that" is it, as the lesson after it.
        (Severity.BLUNDER, Severity.NONE, "Qxf7# was better"),
        # Only the reply went wrong: "that" is the opponent's.
        (Severity.NONE, Severity.MISTAKE, "Their Ke7 was a mistake."),
        # Neither: both readings get the same answer, said of both, no question.
        (
            Severity.NONE,
            Severity.NONE,
            "Neither was a mistake: your Bxf7+ was best, and their Ke7 was best.",
        ),
        # Both: the readings differ and nothing settles it -- one short question.
        (Severity.BLUNDER, Severity.MISTAKE, "Your Bxf7+, or their Ke7?"),
    ],
)
def test_that_after_the_reply_is_the_move_that_went_wrong_or_asked(
    mine: Severity, theirs: Severity, said: str
) -> None:
    for words in ("why was that bad?", "was that a mistake?"):
        coach = offline_coach()
        _bishop_check_and_ke7(coach, mine, theirs)
        assert _answer(coach, words, 12.0).startswith(said), (words, mine, theirs)


def test_whose_move_said_in_the_words_settles_it() -> None:
    for words, said in (
        ("why was my move bad?", "Qxf7# was better"),
        ("was their move a mistake?", "Their Ke7 was a mistake."),
        ("why was Bxf7+ bad?", "Qxf7# was better"),
    ):
        coach = offline_coach()
        _bishop_check_and_ke7(coach, Severity.BLUNDER, Severity.MISTAKE)
        assert _answer(coach, words, 12.0).startswith(said), words


def test_the_answer_to_which_move_is_read_with_the_question() -> None:
    """Asked "your Bxf7+, or their Ke7?", the player's "mine" or "theirs" is the answer to it:
    the question they first asked is answered on that reading. A new question is not."""
    for reply, said in (
        ("mine", "Qxf7# was better"),
        ("theirs", "Their Ke7 was a mistake."),
        ("Ke7", "Their Ke7 was a mistake."),
    ):
        coach = offline_coach()
        _bishop_check_and_ke7(coach, Severity.BLUNDER, Severity.MISTAKE)
        assert _answer(coach, "why was that bad?", 12.0) == "Your Bxf7+, or their Ke7?"
        assert _answer(coach, reply, 13.0).startswith(said), reply
        assert coach._clarifying is None
    coach = offline_coach()
    _bishop_check_and_ke7(coach, Severity.BLUNDER, Severity.MISTAKE)
    _answer(coach, "why was that bad?", 12.0)
    assert not _answer(coach, "what are they threatening?", 13.0).startswith("Qxf7# was better")
    assert coach._clarifying is None, "a question of its own ends the wait for the answer"


def test_the_ai_reads_the_words_itself_with_a_guideline_to_ask() -> None:
    """With the AI on no rule picks the reading: the guideline says to let the conversation and
    the position decide, to say which reading it answers, and to ask one short question naming
    both when nothing settles it and the answers differ -- never when they are the same."""
    coach, provider = scripted_coach(["Your Bxf7+, or their Ke7?"])
    _bishop_check_and_ke7(coach, Severity.BLUNDER, Severity.MISTAKE)
    coach.ask("why was that bad?", 12.0)
    assert [m.text for m in drain(coach)] == ["Your Bxf7+, or their Ke7?"]
    directive = directive_of(sent(provider))
    assert AI_GUIDELINES in directive
    for words in (
        "First, what the words refer to.",
        "say in a few words which it is",
        "do not guess: reply with one short question that names the readings",
        "Never ask when every reading gets the same answer.",
    ):
        assert words in AI_GUIDELINES, words
    assert coach._clarifying is None, "the offline coach's question is the offline coach's"
