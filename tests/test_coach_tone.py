"""The coach teaches gently, offline and with a model (Tai, 2026-10-02: "make all messages a
welcoming experience where you learn"; "That is three times now" and "That one hurts" read as
scolding). The offline coach's fixed wordings hold no tally of failures and no blame; the AI
coach is told the same in its rules."""

from __future__ import annotations

import re

import pytest

from sidecoach.coach import coach as coach_mod
from sidecoach.coach.prompt import PERSONAS, corrections, system_message
from sidecoach.profile import causes
from sidecoach.profile.review import pattern_line

# What reads as scolding: counting the times something went wrong, "again", a verdict on the
# player rather than a lesson.
SCOLDING = re.compile(
    r"\b(?:times now|again|keep missing|kept missing|that one hurts|not that one|"
    r"careless|you always|you never)\b",
    re.IGNORECASE,
)


def _offline_wordings() -> list[str]:
    """Every fixed wording the offline coach says about a mistake or a pattern."""
    said = [*coach_mod.FINE_TIGHTER]
    for bag in coach_mod.COST_BY_SEVERITY.values():
        said.extend(bag)
    for skill, habit in causes.HABIT.values():
        said.extend([skill, habit])
    return said


@pytest.mark.parametrize("wording", _offline_wordings())
def test_no_offline_wording_scolds(wording: str) -> None:
    assert not SCOLDING.search(wording), wording


def test_a_pattern_is_a_skill_to_work_on_never_a_count() -> None:
    from collections import Counter

    line = pattern_line(Counter({causes.MISSED_THREAT: 3}), {causes.MISSED_THREAT: 0.5})
    skill, habit = causes.HABIT[causes.MISSED_THREAT]
    assert line == f"One thing to work on: {skill}. A habit that helps: {habit}."
    assert not re.search(r"\d", line), "no count of the mistakes"


@pytest.mark.parametrize("persona", sorted(PERSONAS))
def test_the_ai_coach_is_told_to_teach_gently_in_every_persona(persona: str) -> None:
    system = system_message(persona=persona)
    assert "Teach it gently and warmly" in system
    assert "never count how often something went wrong" in system
    assert "vary how you say it" in system
    assert "Never a question about them" in system, "never what they expected or saw"
    assert "tease" not in system, "no teasing about a blunder"


def test_the_ai_coachs_worked_examples_hold_no_scolding() -> None:
    """The corrections the model learns from (context/corrections.md) teach by example: none
    of them may show it a tally or a scolding."""
    assert not SCOLDING.search(corrections())
