"""No model at all (llm.provider = none, llm/offline.py): the coach speaks from the board
and its templates, so the app works free and offline (Tai, 2026-09-21)."""

from __future__ import annotations

from sidecoach.coach.coach import Trigger
from sidecoach.coach.verdicts import MATE
from sidecoach.events import Analysis, DisclosureLevel, Line
from sidecoach.llm.offline import OfflineProvider
from tests.coaching import (
    drain,
    mate_analysis,
    mate_played,
    offline_coach,
    play,
    play_to_scholar,
    start_game,
)


def test_a_decisive_moment_and_a_move_get_the_templates_with_no_call() -> None:
    coach = offline_coach()
    provider = coach.speaker.provider
    assert isinstance(provider, OfflineProvider)
    play_to_scholar(coach)
    # The decisive moment: the question template, no model.
    coach.on_event(mate_analysis(6), 7.0)
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
    assert out[0].text and "?" in out[0].text
    assert coach.speaker.fallbacks == 1
    # The player's best move: the verdict written from the board -- here the mate, and
    # never the move itself.
    mate_played(coach)
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MOVE
    assert out[0].text in MATE and "Qxf7#" not in out[0].text
    assert provider.usage.calls == 0, "nothing was ever called"


def test_the_first_hints_lead_is_drawn_when_the_hint_is_said() -> None:
    """The first hint is written into every position's facts and said in few: its lead
    ("There is a mate in one here") is peeked from its bag as it is written and taken only
    when it is said, so the hints never asked for do not use up the bag."""
    from sidecoach.coach.idea_words import _MATE_ONE_LEADS
    from sidecoach.coach.situation import spoken

    coach = offline_coach()
    play_to_scholar(coach)
    for ts in (7.0, 7.5):  # the engine's word twice over; nothing asked
        coach.on_event(mate_analysis(6, decisive=False), ts)
    drain(coach)
    position = coach.positions[6]
    lead = position.leads["lead:mate-one"]
    assert sorted(coach._bags._bags["lead:mate-one"]) == sorted(_MATE_ONE_LEADS)
    coach.ask("hint", 8.0)
    out = drain(coach)
    assert len(out) == 1, out
    assert out[0].text.lower().startswith(spoken(lead).lower()), (lead, out[0].text)
    left = coach._bags._bags["lead:mate-one"]
    assert lead not in left and len(left) == len(_MATE_ONE_LEADS) - 1
    assert position.leads == {}
    # The next hint is the focus rung's template, from the board as well.
    coach.ask("another hint", 9.0)
    out = drain(coach)
    assert len(out) == 1 and out[0].text and out[0].level is DisclosureLevel.FOCUS


def test_no_decisive_question_on_an_obvious_recapture() -> None:
    """9...Nxd4 in a Dragon drew "This is a moment to slow down" (a master-game pass,
    offline, 2026-09-21): when the opponent has just captured and the engine's move takes
    back on that square, the lines diverge only because everything else hangs a piece."""
    coach = offline_coach()
    start_game(coach)
    board = play(coach, ("e4", "c5", "Nf3", "d6", "d4", "cxd4"))
    lines = [
        Line(moves=["f3d4"], depth=12, expected_score=0.52),
        Line(moves=["c2c3"], depth=12, expected_score=0.35),
    ]
    coach.on_event(
        Analysis(game_id="g", fen=board.fen(), ply=6, lines=lines, decisive=True, only_move=False),
        7.0,
    )
    assert drain(coach) == [], "the recapture is the move; no question"
    # The same gap with a zwischenzug as the best move keeps the question.
    lines = [
        Line(moves=["f1b5"], depth=12, expected_score=0.52),
        Line(moves=["f3d4"], depth=12, expected_score=0.35),
    ]
    coach.on_event(
        Analysis(game_id="g", fen=board.fen(), ply=6, lines=lines, decisive=True, only_move=False),
        8.0,
    )
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
