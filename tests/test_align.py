"""Inputs match the judge (Tai, 2026-09-25): every question of the judge's classes reaches the
directive of its class, and a directive that stops asking for what the judge demands is found
before a round is paid for (tools/eval/align.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

from sidecoach.events import DisclosureLevel, Label, MoveAssessment, Severity, Trigger
from tests.coaching import SCHOLAR, mate_analysis, mate_missed, play_to_scholar, scripted_coach

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "eval"))

import align  # noqa: E402
import rubric  # noqa: E402
from taxonomy import CLASSES  # noqa: E402

# What each class's directive says, and the level its question is asked at.
MARKERS = {
    "ask_hint": ("Give the first hint", DisclosureLevel.IDEA),
    "ask_plan": ("They asked what to aim for", DisclosureLevel.IDEA),
    "ask_eval": ("Who stands better, in the words of the Standing line", DisclosureLevel.IDEA),
    "ask_threat": ("They ask what the opponent threatens", DisclosureLevel.IDEA),
    "ask_named_move": ("whether it holds up", DisclosureLevel.IDEA),
    "ask_opponent_move": ("They ask about the opponent's last move", DisclosureLevel.IDEA),
    "ask_last_move": ("They ask about the move they already played", DisclosureLevel.LINE),
    "ask_game_so_far": ("They asked how they did overall", DisclosureLevel.IDEA),
    "ask_opening": ("The opening by the name the record gives it", DisclosureLevel.IDEA),
    "ask_material": ("The material is this", DisclosureLevel.IDEA),
    "ask_move": ("They asked what to play", DisclosureLevel.MOVE),
    "ask_line": ("They asked to see the line", DisclosureLevel.LINE),
}

QUESTIONS = [
    (cls.id, q.format(san="Nc3"))
    for cls in CLASSES.values()
    if cls.kind == "ask"
    for q in cls.questions
]
ASK_CLASSES = sorted({cls for cls, _ in QUESTIONS})


def _at_scholar():
    """The player to move after 3... Nf6, the position analysed and Nf6 judged."""
    coach, _ = scripted_coach([])
    play_to_scholar(coach)
    board = chess.Board(SCHOLAR)
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=6,
            uci="g8f6",
            expected_loss=0.4,
            severity=Severity.BLUNDER,
            best=["d8e7"],
            label=Label.BLUNDER,
        ),
        6.5,
    )
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    assert coach.board.fen() == board.fen()
    return coach


def _after_the_miss():
    """The player has just played 4. Bxf7+, a miss, and it is judged."""
    coach = _at_scholar()
    mate_missed(coach)
    return coach


def test_every_class_has_a_marker():
    assert {c for c, _ in QUESTIONS} == set(MARKERS)


@pytest.mark.parametrize("cls", ASK_CLASSES)
def test_every_question_reaches_the_directive_of_its_class(cls: str):
    marker, level = MARKERS[cls]
    # One game per class, its questions asked in turn: the directive is the question's, not
    # the order it came in.
    coach = _after_the_miss() if cls == "ask_last_move" else _at_scholar()
    for question in (q for c, q in QUESTIONS if c == cls):
        directive, _ = coach._instruction(
            Trigger.USER_ASK, coach.positions[6], level, question, (), None, None
        )
        assert marker in directive, (cls, question, directive[:300])


def _item(directive: str, required: list[dict], cls: str = "verdict_weaker") -> dict:
    coach, _ = scripted_coach([])
    return {
        "id": "t",
        "class": cls,
        "level": "LINE",
        "trigger": "post_move",
        "system": coach.system,
        "messages": [{"role": "user", "content": "# Setting\n...\n\n# Task\n" + directive}],
        "required": required,
        "unlisted": None,
        "must_name": [],
    }


def test_a_demand_the_directive_does_not_make_is_a_gap():
    from sidecoach.coach.coach import NO_NUMBERS, NOT_THE_MOVE

    criteria = rubric.load_rubric()
    asked = (
        "The player just played a3, which was an inaccuracy: playable, not the strongest -- Rc1 "
        "was better. One or two sentences: how it rated, then the better move, what it does, "
        f"and the idea behind it as a lesson to keep. No question. {NOT_THE_MOVE} {NO_NUMBERS}"
    )
    label = [{"key": "label", "value": "inaccuracy", "how": "sense"}]
    assert align.gaps(_item(asked, label), criteria) == []
    # Told only that it was fine: the judge's label and "why" are never asked for.
    vague = asked.replace("an inaccuracy", "fine").replace(", what it does,", "")
    problems = align.gaps(_item(vague, label), criteria)
    assert "fact label (inaccuracy): not asked" in problems
    assert "rubric why: not asked" in problems
    # A check the coach runs and the directive never states is a gap too.
    silent = asked.replace(NO_NUMBERS, "")
    assert "check figures: applied, never stated" in align.gaps(_item(silent, label), criteria)
    # And a fact with no rule is a gap until align.py has one.
    novel = [{"key": "brand_new", "value": "x", "how": "sense"}]
    assert "fact brand_new: no rule in align.py" in align.gaps(_item(asked, novel), criteria)


def test_a_rule_held_back_is_not_asked_for_and_not_judged():
    """While a rule of thumb cools down (2026-09-27) the directive asks for none, and the
    rubric's "transfer" is not asked of the reply: the inputs still match the judge."""
    from sidecoach.coach.coach import NO_NUMBERS, NOT_THE_MOVE, _hold_rule

    criteria = rubric.load_rubric()
    asked = (
        "The player just played a3, which was an inaccuracy: playable, not the strongest -- Rc1 "
        "was better. One or two sentences: how it rated, then the better move, what it does, "
        f"and the idea behind it as a lesson to keep. No question. {NOT_THE_MOVE} {NO_NUMBERS}"
    )
    held = _hold_rule(asked)
    assert "lesson to keep" not in held and "pattern to keep" not in held
    label = [{"key": "label", "value": "inaccuracy", "how": "sense"}]
    assert align.gaps(_item(held, label), criteria) == []
    transfer = next(c for c in criteria if c.key == "transfer")
    assert transfer.applies("lesson", "", asked) and not transfer.applies("lesson", "", held)
