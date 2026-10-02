"""Inputs match the judge (Tai, 2026-09-25): every question of the judge's classes reaches the
directive of its class, and a directive that stops asking for what the judge demands is found
before a round is paid for (tools/eval/align.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import pytest

from sidecoach.coach.coach import NO_FIGURES, NOT_THE_MOVE, _hold_rule
from sidecoach.events import DisclosureLevel, Label, MoveAssessment, Severity, Trigger
from tests.coaching import (
    SCHOLAR,
    mate_analysis,
    mate_missed,
    offline_coach,
    play_to_scholar,
    scripted_coach,
)

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


def _at_scholar(offline: bool = False):
    """The player to move after 3... Nf6, the position analysed and Nf6 judged."""
    coach = offline_coach() if offline else scripted_coach([])[0]
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


def _after_the_miss(offline: bool = False):
    """The player has just played 4. Bxf7+, a miss, and it is judged."""
    coach = _at_scholar(offline)
    mate_missed(coach)
    return coach


def test_every_class_has_a_marker():
    assert {c for c, _ in QUESTIONS} == set(MARKERS)


@pytest.mark.parametrize("cls", ASK_CLASSES)
def test_every_question_reaches_the_directive_of_its_class_offline(cls: str):
    """The offline coach keeps its keyword branches: a question's class picks its directive
    (the templates' and the fallback's)."""
    marker, level = MARKERS[cls]
    # One game per class, its questions asked in turn: the directive is the question's, not
    # the order it came in.
    coach = _after_the_miss(True) if cls == "ask_last_move" else _at_scholar(True)
    for question in (q for c, q in QUESTIONS if c == cls):
        directive, _ = coach._instruction(
            Trigger.USER_ASK, coach.positions[6], level, question, (), None, None
        )
        assert marker in directive, (cls, question, directive[:300])


# With the AI on no keyword picks the directive: one set of guidelines, whatever the question
# (Tai, 2026-10-02), and each class of question finds its guidance in it.
GUIDELINES = {
    "ask_hint": "A hint: one rung up the ladder",
    "ask_plan": "A hint: one rung up the ladder",
    "ask_eval": "How they stand, the material, the opening",
    "ask_threat": "The opponent's move or threat",
    "ask_named_move": "A move, square or piece they name",
    "ask_opponent_move": "The opponent's move or threat",
    "ask_last_move": "A move they already played",
    "ask_game_so_far": "How they did overall",
    "ask_opening": "How they stand, the material, the opening",
    "ask_material": "How they stand, the material, the opening",
    "ask_move": "The move -- ",
    "ask_line": "The line, or what happens next",
}


@pytest.mark.parametrize("cls", ASK_CLASSES)
def test_every_question_reaches_the_ai_guidelines_whatever_its_words(cls: str):
    coach = _after_the_miss() if cls == "ask_last_move" else _at_scholar()
    for question in (q for c, q in QUESTIONS if c == cls):
        directive, _ = coach._instruction(
            Trigger.USER_ASK, coach.positions[6], DisclosureLevel.IDEA, question, (), None, None
        )
        assert GUIDELINES[cls] in directive, (cls, question, directive[:300])
        assert "They asked" not in directive and "They ask " not in directive


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


# A directive after an inaccuracy that asks for all the judge demands, and the fact it is
# judged on.
ASKED = (
    "The player just played a3, which was an inaccuracy: sound, a little behind Rc1. One or "
    "two sentences: the better move, what it does, and the idea behind it as a lesson to "
    f"keep, with how theirs rated in passing. No question. {NOT_THE_MOVE} {NO_FIGURES}"
)
LABEL = [{"key": "label", "value": "inaccuracy", "how": "sense"}]


def test_a_demand_the_directive_does_not_make_is_a_gap():
    criteria = rubric.load_rubric()
    assert align.gaps(_item(ASKED, LABEL), criteria) == []
    # Told only that it was fine: the judge's label and "why" are never asked for.
    vague = ASKED.replace("an inaccuracy", "fine").replace(", what it does,", "")
    problems = align.gaps(_item(vague, LABEL), criteria)
    assert "fact label (inaccuracy): not asked" in problems
    assert "rubric why: not asked" in problems
    # A check the coach runs and the directive never states is a gap too.
    silent = ASKED.replace(NO_FIGURES, "")
    assert "check figures: applied, never stated" in align.gaps(_item(silent, LABEL), criteria)
    # And a fact with no rule is a gap until align.py has one.
    novel = [{"key": "brand_new", "value": "x", "how": "sense"}]
    assert "fact brand_new: no rule in align.py" in align.gaps(_item(ASKED, novel), criteria)


def test_a_rule_held_back_is_not_asked_for_and_not_judged():
    """While a rule of thumb cools down (2026-09-27) the directive asks for none, and the
    rubric's "transfer" is not asked of the reply: the inputs still match the judge."""
    criteria = rubric.load_rubric()
    held = _hold_rule(ASKED)
    assert "lesson to keep" not in held and "pattern to keep" not in held
    assert align.gaps(_item(held, LABEL), criteria) == []
    transfer = next(c for c in criteria if c.key == "transfer")
    assert transfer.applies("lesson", "", ASKED) and not transfer.applies("lesson", "", held)
