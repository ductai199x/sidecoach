"""With the AI coach on, the model reads the question: no keyword picks its directive, forces or
removes its reply, or sets the rung (Tai, 2026-10-02). The keyword rules are the offline coach's.
Fake providers only; logic on the request the model is sent and on what is said."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach.coach import AI_GUIDELINES
from sidecoach.events import DisclosureLevel, Trigger
from tests.coaching import (
    directive_of,
    drain,
    harness_of,
    mate_analysis,
    mate_missed,
    offline_coach,
    play_to_scholar,
    scripted_coach,
    sent,
    stuck,
)
from tests.test_coach import _looked

# Words the keyword rules would have missed or read as something else.
PHRASINGS = (
    "whats the play here lol",
    "ok so what now",
    "gimme the good stuff",
    "thoughts?",
    "idk man",
)


@pytest.mark.parametrize("said", PHRASINGS)
def test_a_question_the_keywords_would_miss_reaches_the_model_with_the_whole_sheet(
    said: str,
) -> None:
    reply = "Qxf7 is mate: their king has no square. Bxf7+ looks as good and passes it up."
    coach, provider = scripted_coach([reply])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    assert drain(coach) == [], "nothing unasked in a quiet position"
    coach.ask(said, 8.0)
    out = drain(coach)
    # Nothing refuses or blanks it: the model's own words, once, whatever they asked.
    assert [m.text for m in out] == [reply]
    assert coach.spoken[-1].problems == [] and coach.speaker.fallbacks == 0
    assert len(provider.calls) == 1
    harness = harness_of(sent(provider))
    # The whole sheet, whatever the rung: every engine line, the facts, the answer among them.
    lines = harness.split("## Engine lines", 1)[1].split("\n\n", 1)[0]
    assert "Qxf7#" in lines and "Bxf7+" in lines
    assert "goal: the player can mate in one move" in harness
    assert f'PLAYER said: "{said}"' in harness
    directive = directive_of(sent(provider))
    # The guidelines, whatever the words, and the ladder as it stands -- never a branch
    # chosen by a keyword.
    assert AI_GUIDELINES in directive
    assert "nothing yet" in directive
    assert "They ask" not in directive and "They asked" not in directive


def test_the_ladder_follows_what_the_reply_said_not_what_the_question_was_guessed_to_mean() -> None:
    """ "what should I play?" opened the move level before the model answered; now the rung is
    what the reply said. A reply that names the move says it; one that only teaches the idea
    does not, and the next hint is the next rung -- each rung once, never lower after higher."""
    coach, provider = scripted_coach(
        [
            "Their king is nearly boxed in: look for a check that ends it.",
            "Qxf7# is mate.",
            "It is still Qxf7#, taking the pawn that guards their king.",
        ]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    position = coach._current
    assert position is not None
    coach.ask("what should I play?", 8.0)
    out = drain(coach)
    assert out[0].level is DisclosureLevel.IDEA, "the reply gave no move"
    assert position.disclosure.level is DisclosureLevel.IDEA
    coach.ask("hint", 9.0)
    out = drain(coach)
    assert "nothing yet" not in directive_of(sent(provider, 1)), "the idea was said"
    assert (
        out[0].level is DisclosureLevel.MOVE and position.disclosure.level is DisclosureLevel.MOVE
    )
    # Said, it stays said: the ladder never falls, and the move is given again with why.
    coach.ask("another hint", 10.0)
    drain(coach)
    third = directive_of(sent(provider, 2))
    assert "The move has been given already, in your turns above: it is Qxf7#" in third
    assert position.disclosure.level is DisclosureLevel.MOVE


def test_a_question_about_a_move_asked_the_model_is_not_withheld_by_the_rung() -> None:
    """The line asked for at the first rung is the model's to give: the keyword rules do not
    stand between the question and the answer."""
    coach, provider = scripted_coach(["Qxf7# ends it at once; nothing follows."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("show me the line", 8.0)
    out = drain(coach)
    assert out and out[0].text.startswith("Qxf7#")
    assert coach.spoken[-1].problems == []
    assert "The line, or what happens next" in directive_of(sent(provider))


def test_a_question_to_the_ai_charges_no_rule_its_fallback_holds() -> None:
    """The fallback's principle ("It's a mating attack: their king runs out of squares.") was
    counted as said for every hint asked of the AI, though the fallback is said only when the
    reply fails: the rules of thumb were held back for the next messages all the same. Offline,
    the template is the answer, and its principle is said."""
    coach, _ = scripted_coach(["Their king is nearly out of squares."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("hint", 8.0)
    assert [m.text for m in drain(coach)] == ["Their king is nearly out of squares."]
    assert coach._rules == {} and coach._last_rule is None
    offline = offline_coach()
    play_to_scholar(offline)
    offline.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(offline)
    offline.ask("hint", 8.0)
    [said] = drain(offline)
    assert "their king runs out of squares" in said.text and offline._rules


def test_the_offline_coach_keeps_its_keyword_rules() -> None:
    """Offline the words still pick the template and open the rung: the templates are the
    offline coach's. Words the rules miss get the plain hint, as before."""
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("whats the play here lol", 8.0)
    [plain] = drain(coach)
    assert plain.level is DisclosureLevel.IDEA and "Qxf7" not in plain.text
    coach.ask("what should I play?", 9.0)
    [move] = drain(coach)
    assert move.level is DisclosureLevel.MOVE and "Qxf7" in move.text


def test_asked_after_a_reply_the_model_has_both_positions_and_how_long_it_took() -> None:
    """ "That" may be the player's move or the opponent's reply: the sheet of the position they
    faced, the one after the reply and the reply's time go to the model, which decides --
    with no threshold in code (the two-second rule is gone)."""
    from tests.test_coach import _played

    for took in (0.5, 25.0):
        # The first reply is the lesson on the miss; the second answers the question.
        coach, provider = scripted_coach(["Qxf7 was mate.", "That was the bishop check."])
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 7.0)
        drain(coach)
        mate_missed(coach, 10.0)
        drain(coach)
        board = chess.Board(SCHOLAR_AFTER_BISHOP_CHECK)
        _played(coach, board, 8, "e8f7", False, 10.0 + took)
        _looked(coach, board, 8, ["h5f3", "h5h4"], 10.0 + took + 0.5)
        coach.ask("why was that bad?", 10.0 + took + 1.0)
        out = drain(coach)
        assert [m.text for m in out] == ["That was the bishop check."]
        harness = harness_of(sent(provider))
        assert "The position they faced, before Bxf7+" in harness and "# Position now" in harness
        assert f"came {took:.1f} seconds after the player's Bxf7+" in directive_of(sent(provider))


SCHOLAR_AFTER_BISHOP_CHECK = "r1bqkb1r/pppp1Bpp/2n2n2/4p2Q/4P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


def test_a_reply_after_the_player_moved_again_is_late_but_shown_in_order() -> None:
    """No reply is dropped. A reply whose moment is over -- the player moved again, a takeback
    or a new game -- comes out in its place, flagged late for the voice alone."""
    from sidecoach.events import Label, Severity
    from tests.coaching import _judged_at_scholar

    coach, _ = scripted_coach(["Qxf7# is mate at once."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("what should I play?", 8.0)  # the request is built; the model has not answered
    assert coach.moved_on(coach._moment()) is False
    _judged_at_scholar(coach, "Qxf7#", 0.0, Severity.NONE, Label.BEST, 9.0)  # the player moves
    out = [m for m in drain(coach) if m.trigger is Trigger.USER_ASK]
    assert [m.text for m in out] == ["Qxf7# is mate at once."], "shown, in its place"
    asked = next(s for s in coach.spoken if s.message.trigger is Trigger.USER_ASK)
    assert asked.late, "for the voice: not spoken"
    # The opponent's reply alone is no new moment.
    moment = coach._moment()
    from tests.test_coach import _played

    board = chess.Board(coach.board.fen())
    _played(coach, board, 8, "e8e7", False, 11.0)
    assert coach.moved_on(moment) is False


def test_a_reply_that_lands_after_the_players_move_is_the_records_not_said_here() -> None:
    """A reply asked for before the player moved and landing after it is shown (late), and it is
    this game's -- but about the position they moved from. Recorded as said at the position now,
    it was a turn "since the move": the next question was told "do not repeat the verdict or the
    better move you told them", and the "?" it ended with made the next words its answer."""
    reply = "Look at f7: what does their king have left?"
    coach, provider = scripted_coach([reply, "The bishop check gave the king a square."])
    play_to_scholar(coach)
    coach.quiet = True  # no word on the move: the late reply is the only coach turn
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("what should I aim for?", 8.0)  # built; the model has not answered yet
    mate_missed(coach, 9.0)  # the player moves first
    board = chess.Board(SCHOLAR_AFTER_BISHOP_CHECK)
    _looked(coach, board, 7, ["e8f7"], 9.5)
    out = [m for m in drain(coach) if m.trigger is Trigger.USER_ASK]
    assert [m.text for m in out] == [reply], "shown, in its place"
    assert coach.spoken[0].late
    assert coach._said_since_move() == [], "not said at the position now"
    assert not coach._last_turn_asked() and not coach._answering_our_question("the king")
    harness = coach.conversation.render()
    assert reply in harness.split("# Position now", 1)[0], "in the record"
    coach.ask("why was that bad?", 10.0)
    drain(coach)
    directive = directive_of(sent(provider))
    assert "You have spoken since the last move already" not in directive
    assert "Your last turn above ended with a question" not in directive


def test_a_question_held_for_the_engine_is_answered_after_the_player_moves() -> None:
    """ "One moment -- still looking" was a promise: the player moving before the engine's
    look came threw the question away unanswered (Tai, 2026-10-02: no reply is dropped). It
    is answered once the position they face now is read."""
    from tests.test_coach import _played

    coach, provider = scripted_coach(["Develop the knight and keep the centre."])
    play_to_scholar(coach)
    coach.ask("what's the plan here?", 7.0)  # no analysis yet: held
    [wait] = drain(coach)
    assert wait.text.startswith("One moment")
    board = chess.Board(coach.board.fen())
    _played(coach, board, 7, "b1c3", True, 8.0)
    _played(coach, board, 8, "a7a6", False, 9.0)
    assert not provider.calls, "nothing is answered before the position is read"
    _looked(coach, board, 8, ["d2d3"], 10.0)
    out = [m for m in drain(coach) if m.trigger is Trigger.USER_ASK]
    assert [m.text for m in out] == ["Develop the knight and keep the centre."]
    assert 'PLAYER said: "what\'s the plan here?"' in harness_of(sent(provider))


def test_the_rung_a_reply_said_is_read_from_its_words() -> None:
    """`said_level`: the move, the focus, the line; and the ladder never falls."""
    from sidecoach.coach.disclosure import Disclosure, PositionFacts, said_level

    facts = PositionFacts(
        fen=chess.STARTING_FEN,
        side_to_move="white",
        phase="opening",
        target="the opponent's king on e8",
        piece="the player's queen on h5",
        goal="the player can mate in one move",
        candidates=["Qxf7#"],
        line=["Qxf7#"],
    )
    assert said_level("Qxf7 is mate.", facts) is DisclosureLevel.MOVE
    assert said_level("Their king on e8 has no square left.", facts) is DisclosureLevel.FOCUS
    assert said_level("Look for a check.", facts) is DisclosureLevel.SILENT
    long = facts.model_copy(update={"line": ["Qxf7#", "Kd8"]})
    assert said_level("Qxf7 and then Kd8 is forced.", long) is DisclosureLevel.LINE
    # A pawn move is named only as a whole move: "e4" is not named by "Nxe4" or "Qe4" (the phone
    # read it as a square; the round-3 parity audit). The phone twin holds the same cases.
    pawn = facts.model_copy(
        update={"target": None, "piece": None, "candidates": ["e4"], "line": ["e4", "e5"]}
    )
    assert said_level("Nxe4 wins a pawn.", pawn) is DisclosureLevel.SILENT
    assert said_level("Qe4 is fine.", pawn) is DisclosureLevel.SILENT
    assert said_level("e4 grabs the centre.", pawn) is DisclosureLevel.MOVE
    assert said_level("After e4 e5 it is level.", pawn) is DisclosureLevel.LINE
    ladder = Disclosure()
    assert ladder.said(DisclosureLevel.IDEA) is DisclosureLevel.IDEA
    assert ladder.said(DisclosureLevel.SILENT) is None
    assert ladder.said(DisclosureLevel.IDEA) is None
    assert ladder.said(DisclosureLevel.MOVE) is DisclosureLevel.MOVE
    assert ladder.said(DisclosureLevel.FOCUS) is None, "never lower after higher"
    assert ladder.level is DisclosureLevel.MOVE


def test_the_rung_is_what_the_reply_names_never_a_count_of_shared_words() -> None:
    """Two words of six letters shared with the goal or theme read as the idea (a magic number,
    Tai 2026-10-02): now the reply's squares, moves, pieces, mate and promotion are read against
    what the idea's facts name, and words alone climb nothing."""
    from sidecoach.coach.disclosure import PositionFacts, said_level

    facts = PositionFacts(
        fen=chess.STARTING_FEN,
        side_to_move="white",
        phase="opening",
        goal="you can mate in one move",
        theme="a mating attack",
        mechanism="a mating attack: their king runs out of squares",
        key="Their king on e8 has one free square, and you have three checks",
        target="their king on e8, whose only free square is e7; you have three checks",
        piece="your queen on h5",
        candidates=["Qxf7#"],
        line=["Qxf7#"],
    )
    assert said_level("Every attack needs squares.", facts) is DisclosureLevel.SILENT
    assert said_level("You are winning, and the material is level.", facts) is (
        DisclosureLevel.SILENT
    )
    assert said_level("Their king is short of air.", facts) is DisclosureLevel.IDEA
    assert said_level("There is a mate here somewhere.", facts) is DisclosureLevel.IDEA
    assert said_level("Your queen can do it.", facts) is DisclosureLevel.FOCUS
    assert said_level("Your knight is fine where it is.", facts) is DisclosureLevel.SILENT
    trap = facts.model_copy(update={"key": None, "trap": "hxg4 looks like it wins, but Qh5 mates"})
    assert said_level("Careful with hxg4.", trap) is DisclosureLevel.IDEA


def test_an_answer_that_names_nothing_of_the_idea_leaves_the_ladder_where_it_was() -> None:
    """ "am I winning?" answered "You are winning; the material is level." raised the ladder to
    the idea, and the stuck hint 45 seconds later was the second hint, the idea never said (Tai,
    2026-10-02: the ladder rises only when the reply hints)."""
    coach, provider = scripted_coach(
        ["You are winning, and the material is level.", "Their king is nearly out of squares."]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    position = coach._current
    assert position is not None
    coach.ask("am I winning?", 8.0)
    [answer] = drain(coach)
    assert answer.text == "You are winning, and the material is level."
    assert position.disclosure.level is DisclosureLevel.SILENT, "nothing of the idea was said"
    [hint] = stuck(coach)
    assert hint.trigger is Trigger.STUCK and hint.level is DisclosureLevel.IDEA
    assert len(provider.calls) == 2
