"""No model at all (llm.provider = none, llm/offline.py): the coach speaks from the board
and its templates, so the app works free and offline (Tai, 2026-09-21)."""

from __future__ import annotations

import chess

from sidecoach.coach.coach import Coach, Trigger
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.coach.verdicts import MATE, OPENERS
from sidecoach.config import LLMSettings
from sidecoach.events import Analysis, Line, MovePlayed
from sidecoach.llm.factory import make_provider
from sidecoach.llm.offline import OfflineProvider
from tests.test_coach import (
    _mate_analysis,
    _mate_in_one_played,
    _messages,
    _play_to_scholar,
    _start,
)


def test_the_factory_gives_the_offline_provider_and_it_reports_itself() -> None:
    provider = make_provider(LLMSettings(provider="none"))
    assert isinstance(provider, OfflineProvider) and provider.offline
    ok, detail = provider.healthcheck()
    assert ok and "offline" in detail
    assert provider.with_effort("low") is provider
    assert provider.usage.calls == 0


def test_a_decisive_moment_and_a_move_get_the_templates_with_no_call() -> None:
    provider = OfflineProvider()
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    _play_to_scholar(coach)
    # The decisive moment: the question template, no model.
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
    assert out[0].text and "?" in out[0].text
    assert coach.speaker.fallbacks == 1
    # The player's best move: the verdict written from the board -- here the mate, and
    # never the move itself.
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MOVE
    assert out[0].text in MATE and "Qxf7#" not in out[0].text
    assert provider.usage.calls == 0, "nothing was ever called"


def test_a_typed_question_gets_the_hint_template() -> None:
    provider = OfflineProvider()
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    coach.ask("hint", 8.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text, "a hint from the templates"
    assert out[0].fen == chess.Board(out[0].fen).fen()


def test_the_templates_rotate_and_never_repeat_the_last() -> None:
    """Four decisive moments in one game got the same sentence four times (a master-game
    pass, 2026-09-21): each list is a bag, every wording once before any comes again (Tai,
    2026-09-25), and never the last one said."""
    from sidecoach.coach.coach import COST_YOU, FINE_TIGHTER, KEEP_LEADS

    coach = Coach(OfflineProvider(), policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    for key, options in (("cost", COST_YOU), ("fine", FINE_TIGHTER), ("keep", KEEP_LEADS)):
        said = [coach._bags.draw(key, options) for _ in range(len(options) * 8)]
        assert all(a != b for a, b in zip(said, said[1:], strict=False))
        for start in range(0, len(said), len(options)):
            assert sorted(said[start : start + len(options)]) == sorted(options)
    # The praise on a fine move too, the same bag whether a lesson follows or not.
    said = [coach._verdict("best") for _ in range(10)]
    assert sorted(said[:5]) == sorted(o + "." for o in OPENERS["best"])
    # A list of one is said again, rather than nothing.
    assert coach._bags.draw("one", ["only"]) == "only" == coach._bags.draw("one", ["only"])


def test_the_first_hints_lead_is_drawn_when_the_hint_is_said() -> None:
    """The first hint is written into every position's facts and said in few: its lead
    ("There is a mate in one here") is peeked from its bag as it is written and taken only
    when it is said, so the hints never asked for do not use up the bag."""
    from sidecoach.coach.idea_words import _MATE_ONE_LEADS
    from sidecoach.coach.situation import spoken

    coach = Coach(OfflineProvider(), policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    _play_to_scholar(coach)
    for ts in (7.0, 7.5):  # the engine's word twice over; nothing asked
        coach.on_event(_mate_analysis(6, decisive=False), ts)
    _messages(coach)
    position = coach.positions[6]
    lead = position.leads["lead:mate-one"]
    assert sorted(coach._bags._bags["lead:mate-one"]) == sorted(_MATE_ONE_LEADS)
    coach.ask("hint", 8.0)
    out = _messages(coach)
    assert len(out) == 1, out
    assert out[0].text.lower().startswith(spoken(lead).lower()), (lead, out[0].text)
    left = coach._bags._bags["lead:mate-one"]
    assert lead not in left and len(left) == len(_MATE_ONE_LEADS) - 1
    assert position.leads == {}


def test_no_decisive_question_on_an_obvious_recapture() -> None:
    """9...Nxd4 in a Dragon drew "This is a moment to slow down" (a master-game pass,
    offline, 2026-09-21): when the opponent has just captured and the engine's move takes
    back on that square, the lines diverge only because everything else hangs a piece."""
    coach = Coach(OfflineProvider(), policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "c5", "Nf3", "d6", "d4", "cxd4"), start=1):
        move = board.parse_san(san)
        before = board.fen()
        board.push(move)
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=(ply % 2 == 1),
            ),
            float(ply),
        )
    lines = [
        Line(moves=["f3d4"], depth=12, expected_score=0.52),
        Line(moves=["c2c3"], depth=12, expected_score=0.35),
    ]
    coach.on_event(
        Analysis(game_id="g", fen=board.fen(), ply=6, lines=lines, decisive=True, only_move=False),
        7.0,
    )
    assert _messages(coach) == [], "the recapture is the move; no question"
    # The same gap with a zwischenzug as the best move keeps the question.
    lines = [
        Line(moves=["f1b5"], depth=12, expected_score=0.52),
        Line(moves=["f3d4"], depth=12, expected_score=0.35),
    ]
    coach.on_event(
        Analysis(game_id="g", fen=board.fen(), ply=6, lines=lines, decisive=True, only_move=False),
        8.0,
    )
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
