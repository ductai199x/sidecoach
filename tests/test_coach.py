"""The coach's dialogue: what it says, when, from which facts, and what it may never say."""

from __future__ import annotations

import re

import chess

from sidecoach.coach.coach import Coach, unlock_for
from sidecoach.coach.disclosure import PositionFacts, Unlock
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.coach.verdicts import OPENERS
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameEnded,
    GameStarted,
    Label,
    Line,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
    Takeback,
    Trigger,
)
from sidecoach.llm.base import ChatMessage
from tests.coaching import (
    SCHOLAR,
    SCHOLAR_MOVES,
    directive_of,
    drain,
    harness_of,
    mate_analysis,
    mate_missed,
    mate_played,
    offline_coach,
    play,
    play_to_scholar,
    scripted_coach,
    sent,
    start_game,
    stuck,
)
from tests.fakes import ScriptedProvider


def _current_move_message(messages: list[ChatMessage]) -> str:
    """The 'Position now' section: the position the player faces, at the level in force."""
    text = harness_of(messages)
    assert "# Position now" in text, "no position in the harness"
    return text.split("# Position now", 1)[1].split("# Task", 1)[0]


def _record(messages: list[ChatMessage]) -> str:
    """The 'Game so far' section."""
    text = harness_of(messages)
    return (
        text.split("# Game so far", 1)[1].split("# Position now", 1)[0]
        if "# Game so far" in text
        else ""
    )


def test_stuck_at_a_decisive_moment_the_player_gets_the_idea_and_no_move() -> None:
    """A decisive moment says nothing of its own (Tai, 2026-10-01: nothing "until the user is
    stuck or actively request a hint or after they have made a move"); stuck, the player
    gets the idea."""
    coach, provider = scripted_coach(
        ["You can mate right now: their king on e8 is short of squares. Which check does it?"]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    assert drain(coach) == [] and provider.prompts == [], "nothing before the stuck limit"
    out = stuck(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.STUCK
    assert out[0].level is DisclosureLevel.IDEA
    # The model was handed the game as one message: the setting, the record one line per
    # move, the position the player faces in full -- everything, the answer included (Tai,
    # 2026-09-26: the model sees engine lines at hints; the reply is checked).
    messages = sent(provider, 0)
    harness = harness_of(messages)
    assert harness.startswith("# Setting\nYou coach White.")
    record = _record(messages)
    assert [line.split(" (")[0] for line in record.strip().splitlines()] == [
        "1. e4",
        "1... e5",
        "2. Bc4",
        "2... Nc6",
        "3. Qh5",
    ]
    current = _current_move_message(messages)
    assert "White to move." in current and "decisive: yes" in current
    assert "3... Nf6 by the opponent" in current
    assert "Pieces -- White: K e1, Q h5" in current, "the position is given piece by piece"
    assert "Legal moves" in current and "Qxf7#" in current, "the model sees every move"
    assert "line 1's first move is the answer to the position they face" in current
    assert "On the board (complete lists)" in current
    # The rung's facts -- the goal and the pattern -- stand apart from those past it.
    facts = current.split("## Facts (the player to move)")[1].split("## Beyond this level")[0]
    assert "goal: the player can mate in one move" in facts
    assert "pattern: a mating attack" in facts
    for secret in ("Qxf7", "queen", "h5f7", "candidates"):
        assert secret not in facts, secret
    beyond = current.split("## Beyond this level")[1].split("## On the board")[0]
    assert "piece: the player's queen on h5" in beyond and "Qxf7" in beyond
    # What the level keeps back is said once, in the Task, with why the model sees it all.
    assert "You see the answer so that the hint can be precise" in directive_of(messages)
    # The first hint's facts are on the sheet to teach from (parity, 2026-09-25; the prompt
    # audit, 2026-09-28), and the idea level names the key fact on the board.
    assert "what a first hint rests on" in current
    assert "Their king on e8 has one free square" in current
    assert "the idea: what is at stake and the key fact" in directive_of(messages)
    assert "no square" not in directive_of(messages)
    assert out[0].text.startswith("You can mate right now"), "named the key square: it stands"
    # The engine's word on it again says nothing either.
    coach.on_event(mate_analysis(6), 8.0)
    assert drain(coach) == []


def test_a_leaked_draft_is_spoken_and_the_leak_logged() -> None:
    """A reply that gives the answer away is logged, never refused (Tai, 2026-09-26: the
    leak check is "a debugging tool and not something used to reject response")."""
    coach, provider = scripted_coach(
        ["Their king on e8 is boxed in: your queen can end it on f7. Which check is it?"]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    out = stuck(coach)
    assert len(out) == 1 and out[0].text.startswith("Their king on e8 is boxed in")
    spoken = coach.spoken[-1]
    assert spoken.attempts == 1 and not spoken.problems and coach.speaker.fallbacks == 0
    assert "withheld square f7" in spoken.leaks and "withheld piece queen" in spoken.leaks
    assert len(provider.prompts) == 1, "no second call for a leak"


def test_two_hints_give_the_idea_then_the_focus_and_asking_for_the_move_gives_it() -> None:
    coach, provider = scripted_coach(
        [
            "You can mate in one: their king has no square. Which check is it?",
            "Their king on e8 is boxed in; your queen does it. Which check?",
            "Qxf7# ends it.",
        ]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    assert drain(coach) == [], "nothing unprompted in a quiet position"
    # The first hint in a fresh position is the idea: the goal and the pattern.
    coach.ask("give me a hint", 8.0)
    out = drain(coach)
    assert out[0].level is DisclosureLevel.IDEA
    current = _current_move_message(sent(provider, 0))
    facts = current.split("## Facts (the player to move)")[1].split("## Beyond this level")[0]
    assert "goal: the player can mate in one move" in facts and "Qxf7" not in facts
    assert "queen" not in facts and "target:" not in facts, "not the piece or the target yet"
    assert "target:" in current.split("## Beyond this level")[1], "the model knows them"
    # The player's words stand under the position, verbatim.
    assert '## Said here\n    PLAYER said: "give me a hint"' in current
    # The second, the focus: the target and the piece, never the move.
    coach.ask("another hint", 8.5)
    out = drain(coach)
    assert out[0].level is DisclosureLevel.FOCUS
    current = _current_move_message(sent(provider, 1))
    facts = current.split("## Facts (the player to move)")[1].split("## Beyond this level")[0]
    assert "target: the opponent's king on e8" in facts
    assert "piece: the player's queen on h5" in facts
    assert "Qxf7" not in facts, "the move is past the focus rung"
    assert "Qxf7" in current.split("## Beyond this level")[1], "and the model knows it"
    coach.ask("ok just tell me the move", 9.0)
    out = drain(coach)
    assert out[0].level is DisclosureLevel.MOVE and "Qxf7" in out[0].text
    # Asked of the AI coach, the move is the model's to give too (Tai, 2026-09-28); asked for
    # outright, the task states the move's level, not the ladder (2026-10-03).
    assert len(provider.prompts) == 3
    directive = directive_of(sent(provider, 2))
    assert "The move -- " in directive and "give it plainly, and why it works" in directive
    assert directive.startswith("Disclosure for the position the player faces: the move")
    # Every earlier coach turn stands under the position: the two hints, each the model's.
    current = _current_move_message(sent(provider, 2))
    said = current.split("## Said here\n", 1)[1]
    lines = said.strip().splitlines()
    assert sum(line.startswith("    YOU (the coach) said: ") for line in lines) == 2


def test_the_move_asked_for_first_in_a_fresh_position_is_given() -> None:
    """Tai's Pixel, 2026-10-03: the Best move button ("what should I play") on a fresh puzzle got
    a hint -- the AI coach was told the hint rung beside the words, and the rung won. The move
    or the line asked for outright opens it exactly, and the task says so, not the ladder; a
    real question is answered fully; a hint keeps the answer back."""
    for asked, level in (
        ("what should I play", DisclosureLevel.MOVE),
        ("show me the line", DisclosureLevel.LINE),
    ):
        coach, provider = scripted_coach(["Qxf7# ends it."])
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6, decisive=False), 7.0)
        coach.ask(asked, 8.0)
        out = drain(coach)
        assert out[0].level is level, asked
        directive = directive_of(sent(provider, 0))
        assert "Keep back" not in directive and "A hint they ask for" not in directive, asked
        assert directive.startswith("Disclosure for the position the player faces: "), asked
    # A question in the player's own words: the ladder is only for a hint they ask for.
    coach, provider = scripted_coach(["Qxf7# ends it."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("which move wins here?", 8.0)
    drain(coach)
    directive = directive_of(sent(provider, 0))
    assert directive.startswith("Their words decide what they get. A question is answered fully")
    # A hint asked for first is the idea, the answer kept back.
    coach, provider = scripted_coach(["Their king has no square."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 8.0)
    assert drain(coach)[0].level is DisclosureLevel.IDEA


def test_a_question_starts_the_stuck_spell_again() -> None:
    """A hint asked at 15 s was followed by the focus unasked at 16 s: the stuck spell ran
    from the position, not from the answer (the coach review, 2026-10-01). The next hint
    unasked waits a full spell from the answer, as from one said unasked."""
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 15.0)
    assert [m.level for m in drain(coach)] == [DisclosureLevel.IDEA]
    coach.tick(16.0)
    assert drain(coach) == [], "a second after the answer"
    assert [m.level for m in stuck(coach)] == [DisclosureLevel.FOCUS]


def test_naming_the_move_yourself_earns_the_move_level() -> None:
    """Offline the keyword rules (`unlock_for`) read the rung off the words. With the AI on the
    rung is what the answer said: it names the move, so the move is said."""
    coach, _ = scripted_coach(["Yes, Qxf7 is mate.", "What do you think?"])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    coach.ask("Is Qxf7 mate?", 8.0)
    out = drain(coach)
    assert out[0].level is DisclosureLevel.MOVE
    assert coach._current is not None and coach._current.disclosure.level is DisclosureLevel.MOVE
    board = chess.Board(SCHOLAR)
    assert unlock_for("Is Qxf7 mate?", board, ["Qxf7#"]) is Unlock.USER_NAMED_MOVE
    assert unlock_for("What about Bxf7?", board, ["Qxf7#"]) is None, "not the candidate"
    assert unlock_for("show me the line", board, []) is Unlock.ASK_LINE
    assert unlock_for("HINT please", board, []) is Unlock.HINT
    assert unlock_for("what should I play here", board, []) is Unlock.ASK_MOVE
    assert unlock_for("how are you", board, []) is None
    # Words, not a word: "open lines" opened the line, and "which move" or "the answer" about
    # a move already played opened the move, and the answer was given (the coach review,
    # 2026-10-01).
    for said in (
        "are there any open lines for my rooks?",
        "which move did they just play?",
        "which move did I miss there?",
        "what was the answer to their last threat?",
        "was that the best move?",
    ):
        assert unlock_for(said, board, ["Qxf7#"]) is None, said
    for said, unlock in (
        ("what's the line?", Unlock.ASK_LINE),
        ("what are the lines here?", Unlock.ASK_LINE),
        ("give me a line", Unlock.ASK_LINE),
        ("show me some lines", Unlock.ASK_LINE),
        ("what's the main variation?", Unlock.ASK_LINE),
        ("ok and after that?", Unlock.ASK_LINE),
        ("how would this position continues?", Unlock.ASK_LINE),
        ("which move should I play?", Unlock.ASK_MOVE),
        ("which move?", Unlock.ASK_MOVE),  # bare, the whole question (9416b99, 0da008b)
        ("ok, which move ?", Unlock.ASK_MOVE),
        ("which move is best?", Unlock.ASK_MOVE),
        ("just tell me the answer", Unlock.ASK_MOVE),
        ("what's the answer to their threat?", Unlock.ASK_MOVE),
    ):
        assert unlock_for(said, board, ["Qxf7#"]) is unlock, said


def test_feedback_after_the_users_move_may_say_everything_about_that_position() -> None:
    coach, provider = scripted_coach(
        [
            "Their king on e8 is nearly out of squares.",
            "Qxf7 was mate on the spot, sharper than the bishop. Do you see why?",
        ]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    stuck(coach)
    mate_missed(coach)  # Bxf7+ instead of the mate
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MISTAKE
    assert out[0].level is DisclosureLevel.LINE and "Qxf7" in out[0].text
    # The move's own message carries the verdict and what the engine preferred; the
    # position before it, now history, shows its lines in full.
    messages = sent(provider)
    played = _current_move_message(messages)
    assert "4. Bxf7+ by the player: miss; it cost 0.45 of expected score" in played
    assert "the engine preferred 4. Qxf7#" in played
    # The record is at full disclosure: the earlier verdicts stand in full.
    assert "3... Nf6 (the opponent)" in _record(messages)
    # The lesson is taught from the position they faced, which the record keeps as one line:
    # its sheet goes with the request (parity pass, 2026-09-25), and the better move the reply
    # must name is the directive's -- not the template's lesson, which came back word for word
    # (the prompt audit, 2026-09-28).
    content = str(messages[-1]["content"])
    faced = content.split("# The position they faced, before Bxf7+")[1].split("# Task")[0]
    assert "goal: the player can mate in one move" in faced
    assert "The better move was Qxf7#" in directive_of(messages)
    assert coach.spoken[-1].problems == [], coach.spoken[-1].problems
    # The first hint is for the side to move: never on the opponent's sheet, where labelled
    # "to the player" it was the opponent's (parity pass 2).
    now = played.split("# The position they faced")[0]
    assert "(to the player, as it stands)" not in now.split("## Facts", 1)[-1]


def test_a_question_about_the_plan_is_answered_with_the_first_hint() -> None:
    """ "What should I be aiming for?" earns the hint's rung and the hint's directive, not
    the generic answer (parity pass 2, 2026-09-25); asked of the AI coach, the model answers
    it, with nothing said before it (Tai, 2026-09-28)."""
    reply = "Their king has one square left. Which check takes it away?"
    coach, provider = scripted_coach([reply])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    assert drain(coach) == []
    coach.ask("what should I be aiming for here?", 8.0)
    out = drain(coach)
    assert [m.text for m in out] == [reply] and out[0].level is DisclosureLevel.IDEA
    directive = directive_of(sent(provider, 0))
    # No keyword picked a directive: the guidelines, and the ladder with nothing said yet, so
    # a hint now is the first rung.
    assert "A hint: one rung up the ladder" in directive
    assert "nothing yet" in directive and "the idea: what is at stake" in directive
    assert "They asked what to aim for" not in directive


def test_no_hint_unasked_comes_after_the_move_was_given() -> None:
    """Tai, 2026-09-29: "Each level can only be said once and if the level higher was said
    then the lower one can't be said." Once a question had the move, no stuck hint follows:
    the focus is below it (on 2026-09-28 one did, and both replies went for "d5: your bishop
    on c8 is still shut in", twice). A stuck template stays at the focus all the same."""
    coach, provider = scripted_coach(["Qxf7# is mate at once."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("what should I play?", 8.0)
    assert [m.text for m in drain(coach)] == ["Qxf7# is mate at once."]
    for ts in (17.0, 40.0, 90.0):
        coach.tick(ts)
    assert drain(coach) == [] and len(provider.prompts) == 1
    position = coach._current
    assert position is not None
    _, template = coach._instruction(
        Trigger.STUCK, position, DisclosureLevel.MOVE, None, (), None, None
    )
    assert "Qxf7" not in template, "the template is the focus's, not the move's"


def test_stuck_gives_the_idea_then_the_focus_and_the_opponents_turn_is_silent() -> None:
    coach, _ = scripted_coach(["You can mate here.", "Look at their king."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.tick(12.0)
    assert drain(coach) == [], "not stuck yet: ten seconds from the move at 6.0"
    coach.tick(17.0)
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.STUCK
    assert out[0].level is DisclosureLevel.IDEA
    coach.tick(22.0)
    assert drain(coach) == [], "the second waits a full spell from the first"
    coach.tick(28.0)
    out = drain(coach)
    assert len(out) == 1 and out[0].level is DisclosureLevel.FOCUS
    coach.tick(80.0)
    assert drain(coach) == [], "two unprompted messages per position; the move only asked"
    # With the opponent to move, nothing unprompted.
    quiet, _ = scripted_coach(["..."])
    start_game(quiet, Side.BLACK)
    quiet.on_event(mate_analysis(0), 1.0)
    quiet.tick(100.0)
    assert drain(quiet) == []


def test_a_good_move_gets_one_sentence_from_the_model_with_the_template_to_fall_back_on() -> None:
    """With local verdicts off (coach/verdicts.py writes them by default)."""
    # Every move of the player's is a turn for the coach: a fine move gets one short
    # sentence from the model, asked for as two candidates so the call stays short.
    coach, provider = scripted_coach(
        ["[p=0.6] Mate on the board, clean finish.\n[p=0.4] That ends it."]
    )
    coach.local_verdicts = False  # the model's word, not the board's
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    mate_played(coach)
    out = drain(coach)
    assert len(out) == 1 and out[0].text in ("Mate on the board, clean finish.", "That ends it.")
    assert out[0].trigger is Trigger.POST_MOVE and len(provider.prompts) == 1
    directive = directive_of(sent(provider, 0))
    assert "The player just played Qxf7#" in directive
    assert "best -- the engine's own choice" in directive
    assert "fifteen words at most" in directive and "give TWO different ways" in directive
    # When the reply fails, the two-word template stands in.
    coach, provider = scripted_coach(["", ""])
    coach.local_verdicts = False  # the model's word, not the board's
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    mate_played(coach)
    out = drain(coach)
    assert len(out) == 1 and out[0].text in [(o + ".").format(san="Qxf7#") for o in OPENERS["best"]]
    assert len(provider.prompts) == 2, "one call, one retry, then the template"


def test_in_blitz_a_good_move_earns_two_words_without_the_model() -> None:
    from sidecoach.coach.policy import TimeControl

    coach, provider = scripted_coach(["never used"])
    coach.local_verdicts = False
    play_to_scholar(coach)
    coach.policy.time_control = TimeControl.BLITZ
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    mate_played(coach)
    out = drain(coach)
    assert len(out) == 1 and out[0].text in [o + "." for o in OPENERS["best"]]
    assert provider.prompts == [], "no model call for a good move in blitz"
    # The wording is by the label, drawn from its bag: every wording once before any comes
    # again, and never the same twice running -- "good, good, good" for a whole game wore
    # thin. Never the move: the player just played it.

    fresh, _ = scripted_coach([])
    for label in ("good", "best"):
        wordings = sorted(o + "." for o in OPENERS[label])
        said = [fresh._verdict(label) for _ in range(len(wordings) * 4)]
        for start in range(0, len(said), len(wordings)):
            assert sorted(said[start : start + len(wordings)]) == wordings, label
        assert all(a != b for a, b in zip(said, said[1:], strict=False)), label
    assert fresh._verdict("book") in [o + "." for o in OPENERS["book"]]


def test_in_blitz_a_mistake_gets_a_word_that_does_not_call_it_fine() -> None:
    # Blitz keeps only the word after the move (the policy), so a blunder comes to the
    # directive as POST_MOVE: no lesson and no question, but not "fine" either.
    coach, _ = scripted_coach([])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 6.5)
    position = coach.positions[6]
    assert position.facts is not None
    board = chess.Board(SCHOLAR)
    move = board.parse_san("a3")
    board.push(move)
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="a3",
        uci=move.uci(),
        fen_before=SCHOLAR,
        fen_after=board.fen(),
        by_user=True,
    )
    blunder = MoveAssessment(
        game_id="g",
        ply=7,
        uci=move.uci(),
        expected_loss=0.5,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.BLUNDER,
    )
    instruction, fallback = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), blunder, played
    )
    from sidecoach.coach.coach import COST_BY_SEVERITY, FINE_TIGHTER

    # A blunder is said as one, never "that cost you something" (the facts audit, 2026-10-01).
    assert fallback in [c.format(best="Qxf7#") + "." for c in COST_BY_SEVERITY["blunder"]]
    assert "which was a blunder" in instruction and "fine" not in instruction
    assert "no lesson, no question" in instruction
    # A move a little behind the best is still fine.
    good = blunder.model_copy(
        update={"expected_loss": 0.03, "severity": Severity.NONE, "label": Label.GOOD}
    )
    from sidecoach.coach.coach import RULE_SPACING

    coach._messages += RULE_SPACING  # clear of any rule of thumb said before
    said_rules: list[str] = []
    record = coach._rule_said

    def spy(rule: str) -> None:
        said_rules.append(rule)
        record(rule)

    coach._rule_said = spy  # type: ignore[method-assign]
    _, fallback = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), good, played
    )
    # The better move and what it does -- and no rule of thumb: a fine move is not a lesson
    # (Tai, 2026-09-27: "the coach is a teacher, not a nagging annoying bot").
    assert any(fallback.startswith(f.format(best="Qxf7#")) for f in FINE_TIGHTER)
    assert "it is mate" in fallback
    assert not said_rules and not ANNOUNCED.search(fallback), (said_rules, fallback)
    # Said once, the lesson is not said again a move later: the verdict and the better move
    # stand alone (Nova bot games, 2026-09-25: 27% of lessons came again within three moves).
    coach.conversation.coach_said(fallback)
    _, again = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), good, played
    )
    assert any(again.startswith(f.format(best="Qxf7#")) for f in FINE_TIGHTER)
    assert "it is mate" not in again and not said_rules
    assert "tells them nothing: do not open with it" in instruction


# The leads that announced a rule, gone for good.
ANNOUNCED = re.compile(
    r"(keep this one|the lesson to keep|worth remembering|a rule to take with you|"
    r"the thing to remember|remember this):",
    re.IGNORECASE,
)


def _sentence_of(rule: str) -> str:
    return rule[0].upper() + rule[1:].rstrip(".") + "."


def test_a_rule_of_thumb_cools_down_across_games_and_puzzles() -> None:
    """Tai, 2026-09-27, after a run of puzzles: "it keeps on nagging me about some lessons
    like 'A rule to take with you' ... there has to be a cooldown of some number of messages
    before it can appear again". A rule is not said again for RULE_COOLDOWN of the coach's
    messages, whatever game or puzzle they fall in, and no two rules come within
    RULE_SPACING messages; the model is told to leave it out while it cools."""
    from sidecoach.coach.coach import RULE_COOLDOWN, RULE_HELD, RULE_SPACING

    coach, _ = scripted_coach([])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 6.5)
    stuck(coach)
    position = coach.positions[6]
    board = chess.Board(SCHOLAR)
    move = board.parse_san("a3")
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="a3",
        uci=move.uci(),
        fen_before=SCHOLAR,
        fen_after=SCHOLAR,
        by_user=True,
    )
    mistake = MoveAssessment(
        game_id="g",
        ply=7,
        uci=move.uci(),
        expected_loss=0.5,
        severity=Severity.MISTAKE,
        best=["h5f7"],
        label=Label.MISTAKE,
    )

    def lesson() -> tuple[str, str]:
        coach.conversation.entries.clear()  # a new puzzle: nothing said in this game yet
        return coach._instruction(
            Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, None, (), mistake, played
        )

    said_rules: list[str] = []
    record = coach._rule_said

    def spy(rule: str) -> None:
        said_rules.append(rule)
        record(rule)

    coach._rule_said = spy  # type: ignore[method-assign]

    def has_rule(text: str) -> bool:
        return any(rule.lower().rstrip(".") in text.lower() for rule in said_rules)

    # The stuck hint's idea said its principle: no rule for a few messages.
    instruction, soon = lesson()
    assert not has_rule(soon) and RULE_HELD in instruction
    coach._messages += RULE_SPACING
    instruction, first = lesson()
    assert has_rule(first) and RULE_HELD not in instruction
    # Said as a sentence of its own, never announced (Tai, 2026-09-27: "nagging stuff like
    # 'Remember this' 'Keep this one'").
    assert _sentence_of(said_rules[-1]) in first and not ANNOUNCED.search(first), first
    assert "pattern to remember" in instruction
    for _ in range(RULE_COOLDOWN - 1):
        coach._messages += 1
        instruction, again = lesson()
        assert not has_rule(again), again
        # Held, the directive asks for no rule at all, not for one and then for none: the
        # local model gave one anyway in 2 of 17 lessons (the free check, 2026-09-27).
        assert RULE_HELD in instruction
        assert "pattern to remember" not in instruction and "pattern to keep" not in instruction
    coach._messages += 1
    _, back = lesson()
    assert has_rule(back), "back after the cooldown"
    # Any two rules keep their distance, and a sound move's principle is a rule too.
    assert not coach._rule_ok("a different rule")
    assert "takeFile" in coach._verdict_taught()
    coach._messages += RULE_SPACING
    assert coach._rule_ok("a different rule")
    coach._taught_by_verdict("takeFile")
    assert "takeFile" in coach._verdict_taught() and not coach._rule_ok(None)
    coach._messages += RULE_SPACING
    assert "takeFile" in coach._verdict_taught() and "castle" not in coach._verdict_taught()
    coach._messages += RULE_COOLDOWN
    assert coach._rule_ok("takeFile")  # cooled; the last few lessons' memory is its own


def test_a_hints_principle_cools_down_too() -> None:
    """The first hint's principle ("A move that attacks something they must then defend keeps
    you a step ahead: ...") came 218 times in the master games' hints, 59 of them again in
    the same game: it is a rule of thumb like the lesson's, left out while it cools."""
    from sidecoach.coach.coach import RULE_COOLDOWN

    coach, _ = scripted_coach([])
    rule = "A move that attacks something they must then defend keeps you a step ahead."
    hint = f"You can take the initiative. {rule} Their pawn on d3 can be hit."
    facts = PositionFacts(
        fen=SCHOLAR, side_to_move="white", phase="middlegame", situation=hint, principle=rule
    )
    said, note = coach._cool_hint(facts, hint)
    assert said == hint and not note
    coach._messages += RULE_COOLDOWN - 1
    said, note = coach._cool_hint(facts, hint)
    assert said == "You can take the initiative. Their pawn on d3 can be hit."
    assert rule in note and "heard it lately" in note
    coach._messages += 1
    assert coach._cool_hint(facts, hint) == (hint, "")


def test_a_lesson_is_said_of_the_board_the_player_sees_and_not_twice() -> None:
    from sidecoach.coach.coach import _stale, fresh

    # 4.d3 had just let the bishop out: "still shut in behind the pawn on d2" was the
    # position before the move (Nova bot games, 2026-09-25).
    assert _stale("your bishop on c1 is still shut in behind the pawn on d2", "d2d3")
    assert not _stale("your bishop on c1 is still shut in behind the pawn on d2", "b1c3")
    # A line is about the moves to come, and stands.
    assert not _stale("after Qg2, Rdg1, and the queen on g2 is trapped", "g2g3")
    said = (
        "That works, though d4 was a touch better -- your bishop on c1 is still shut in behind "
        "the pawn on d2."
    )
    rule = "The centre is where pieces have the most room."
    text = (
        "Playable -- d4 was a little more precise -- your bishop on c1 is still shut in behind "
        f"the pawn on d2. {rule}"
    )
    # The reason said lately goes; a rule of thumb is not `fresh`'s to hold -- one said lately
    # is never written in (`Coach._rule_ok`), so a rule that is here stays.
    assert fresh(text, said) == f"Playable -- d4 was a little more precise. {rule}"
    assert fresh(text, "Nice.") == text
    opener = "Playable -- d4 was a little more precise"
    assert fresh(text, said, opener) == f"{opener}. {rule}"
    # An opener with its own " -- " and no reason after it: the better move is not a reason,
    # and stays when the same move was just said ("Not that one." -- Nova bot games).
    opener = "Not that one -- Ng3 was better"
    just = "That one cost you: Ng3 was better. After c6, they win a pawn."
    assert fresh(opener + ".", just.lower(), opener) == opener + "."
    assert fresh(f"{opener} -- it wins a pawn.", just + " It wins a pawn.", opener) == opener + "."


def test_the_claim_reader_lists_the_drafts_claims_and_the_board_refuses_a_false_one() -> None:
    """A model lists what the draft asserts; python-chess decides it (coach/draft_claims.py).
    "Their queen on c5" is refused where no queen stands; the second draft is spoken."""
    import json

    false_claims = json.dumps(
        [
            {
                "kind": "piece",
                "after": [],
                "quote": "their queen on c5",
                "side": "opponent",
                "piece": "queen",
                "square": "c5",
            }
        ]
    )
    reader = ScriptedProvider([false_claims, "[]"])
    fine = "Their king on e8 is short of squares. Which check takes the last one?"
    provider = ScriptedProvider(["Their queen on c5 is loose. Which check wins it?", fine])
    coach = Coach(
        provider,
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
        claim_reader=reader,
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    out = stuck(coach)
    assert out[0].text == fine
    spoken = coach.spoken[-1]
    assert spoken.attempts == 2
    assert "c5 is empty then" in provider.prompts[1][1][-1]["content"], "the reason is named"


def test_a_reply_thinner_than_the_template_is_said_and_noted() -> None:
    """A first hint that names nothing of the key fact is thinner than the free coach's; it
    was sent back and the template spoken (parity pass, 2026-09-25). Its form is the
    prompt's to ask for, and a slip of it is minor (Tai, 2026-09-28: "we don't need to throw
    anything out"): it is said, and noted for the log."""
    thin = "Your king is safe for now, is it not?"
    coach, provider = scripted_coach([thin])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    out = stuck(coach)
    assert out[0].text == thin
    spoken = coach.spoken[-1]
    assert spoken.attempts == 1 and coach.speaker.fallbacks == 0 and not spoken.problems
    assert any(note.startswith("left out") for note in spoken.notes), spoken.notes


def test_a_takeback_is_a_message_that_says_so_and_history_keeps_the_withdrawn_move() -> None:
    from sidecoach.events import Takeback

    coach, provider = scripted_coach(["Fine."])
    play_to_scholar(coach)
    fen_before = chess.Board(SCHOLAR).fen()
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Bxf7+")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Bxf7+",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        ),
        10.0,
    )
    coach.on_event(Takeback(game_id="g", plies=1, fen=fen_before), 11.0)
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Qxf7#")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Qxf7#",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        ),
        12.0,
    )
    coach.on_event(mate_analysis(6, decisive=False), 12.5)  # facts for the ask below
    coach.ask("what happened there?", 13.0)
    drain(coach)
    harness = harness_of(sent(provider))
    withdrawn = harness.index("4. Bxf7+ (the player)")
    takeback = harness.index("    EVENT: Takeback: 4. Bxf7+ withdrawn")
    replacement = harness.index("4. Qxf7# by the player")
    assert withdrawn < takeback < replacement
    assert "back at the position after 3... Nf6" in harness
    # The withdrawn move stays in the record as something that happened, marked so; the
    # position now is the replacement's.
    assert harness.count("4. Bxf7+") == 2, "the withdrawn move: its line and the takeback"


def test_a_reread_of_the_board_is_a_message_and_the_numbering_restarts() -> None:
    from sidecoach.events import Resync

    coach, provider = scripted_coach(["Understood."])
    play_to_scholar(coach)
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4"
    coach.on_event(
        Resync(game_id="g", fen=fen, reason="position adopted after 3 unexplained cells"), 20.0
    )
    board = chess.Board(fen)
    move = board.parse_san("d6")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=1,
            san="d6",
            uci=move.uci(),
            fen_before=fen,
            fen_after=board.fen(),
            by_user=False,
        ),
        21.0,
    )
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=1,
            lines=[Line(moves=["c3d5"], depth=12, expected_score=0.6)],
            decisive=False,
            only_move=False,
        ),
        22.0,
    )
    coach.ask("what now?", 23.0)
    drain(coach)
    harness = harness_of(sent(provider))
    assert "    EVENT: The board was re-read" in harness and "Moves may have gone unseen" in harness
    # The numbering follows the position the board was read at, not a ply count.
    assert "4... d6 by the opponent" in _current_move_message(sent(provider))


def test_asking_before_the_engine_has_looked_is_answered_when_it_has_even_early() -> None:
    """A question asked before the engine has looked at the position on the board is told
    to wait a moment and answered when the analysis lands, never from an older position
    (Tai, 2026-09-24: a hint pressed as a puzzle appeared got "I have not seen enough of
    this game yet to say. Give me a move or two."). The early analysis answers it, the full
    depth still to come (Tai, 2026-09-28: "first usable answer is depth 12 ... the
    computation continues until max depth of 18")."""
    for provisional in (False, True):
        coach, provider = scripted_coach([])
        play_to_scholar(coach)
        coach.ask("hint", 6.5)
        # The word while the engine has not looked: one message, the ask's, and nothing
        # else. Its words are the coach's to change.
        out = drain(coach)
        assert len(out) == 1 and out[0].trigger is Trigger.USER_ASK and out[0].text, out
        assert provider.prompts == []
        coach.ask("hint", 6.7)  # asked again while waiting: one "moment" is enough
        assert drain(coach) == []
        analysis = mate_analysis(6, decisive=False)
        coach.on_event(analysis.model_copy(update={"provisional": provisional}), 7.0)
        answered = drain(coach)
        assert answered and all(m.trigger is Trigger.USER_ASK for m in answered), provisional
        assert answered[0].text != out[0].text


def test_an_early_analysis_of_a_position_left_behind_answers_nothing_asked_now() -> None:
    """The engine behind the game: its early look at a position the game has moved past
    lands while the player waits on a question about the position now. It is kept for its
    own position, and the question waits for the engine's look at the board now -- never
    answered from the older one. (The takeback tests met this only when their engine was
    slow enough; here it is the step the test takes.)"""
    coach, provider = scripted_coach([])
    play_to_scholar(coach)
    coach.ask("hint", 6.5)
    asked = drain(coach)
    assert len(asked) == 1 and asked[0].trigger is Trigger.USER_ASK  # "a moment"
    behind = coach.positions[4].board
    move = next(iter(behind.legal_moves)).uci()
    late = Analysis(
        game_id="g",
        fen=behind.fen(),
        ply=4,
        lines=[Line(moves=[move], depth=12, expected_score=0.5)],
        decisive=False,
        only_move=False,
        provisional=True,
    )
    coach.on_event(late, 6.8)
    assert coach.positions[4].analysis == late, "kept for its own position"
    assert drain(coach) == [] and provider.prompts == [], "the question about now still waits"
    coach.on_event(mate_analysis(6, decisive=False).model_copy(update={"provisional": True}), 7.0)
    answered = drain(coach)
    assert answered and all(m.trigger is Trigger.USER_ASK for m in answered)


def test_an_analysis_early_or_full_says_nothing_of_its_own() -> None:
    """What the engine finds is no reason to speak while the player thinks: neither its
    early look nor the full depth says a word, a decisive moment included (Tai,
    2026-10-01); the stuck limit does."""
    coach, provider = scripted_coach(
        ["You can mate right now: their king on e8 is short of squares. Which check does it?"]
    )
    play_to_scholar(coach)
    early = mate_analysis(6).model_copy(update={"provisional": True})
    coach.on_event(early, 7.0)
    assert drain(coach) == []
    coach.on_event(mate_analysis(6), 8.0)
    assert drain(coach) == [] and provider.prompts == []
    assert [m.trigger for m in stuck(coach)] == [Trigger.STUCK]


# The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
QD1_LINE = ["c2d1", "d4d1", "d8d1", "a5e1", "d1e1"]


def _puzzle(coach: Coach, fen: str = QD1) -> chess.Board:
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.BLACK, start_fen=fen, kind="position"), 0.0
    )
    return chess.Board(fen)


def _analysed(
    coach: Coach, board: chess.Board, ply: int, line: list[str], ts: float, mate: int | None
) -> None:
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            lines=[
                Line(moves=line, depth=18, expected_score=1.0, mate=mate),
                Line(moves=["h6h5"], depth=18, expected_score=0.1),
            ],
            decisive=True,
            only_move=True,
        ),
        ts,
    )


def _played(coach: Coach, board: chess.Board, ply: int, uci: str, by_user: bool, ts: float) -> None:
    move = chess.Move.from_uci(uci)
    before = board.fen()
    san = board.san(move)
    board.push(move)
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
        ts,
    )


def _hint_then(coach: Coach, board: chess.Board) -> None:
    """The puzzle appears, the player presses Hint before the engine has looked, and the
    hint comes when it has: no puzzle question was asked, as on Tai's desktop."""
    coach.ask("hint", 0.5)
    _analysed(coach, board, 0, QD1_LINE, 1.0, 3)
    got = drain(coach)
    assert got and all(m.trigger is Trigger.USER_ASK for m in got)


def test_no_new_question_while_the_line_is_played_out_nor_a_hint_said_twice() -> None:
    """After 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 the coach asked "You can mate in one move ...
    Which check leaves their king no square at all?" -- a fresh question in the middle of
    the combination the player was carrying out (Tai's desktop, 2026-09-24): nothing unasked
    comes while the line is played. And a reply off the engine's line brought the idea back:
    the stuck hint's "a tactic that wins a knight ... a pin" at a puzzle's start came again
    after the move was found
    (Tai's phone, 2026-09-29: "Each level can only be said once and if the level higher was
    said then the lower one can't be said"). The next hint unasked is the focus."""
    coach, _ = scripted_coach([])
    board = _puzzle(coach)
    _hint_then(coach, board)
    for ply, uci in enumerate(QD1_LINE[:4], start=1):
        _played(coach, board, ply, uci, ply % 2 == 1, 2.0 + ply)
        if ply % 2 == 0:
            _analysed(coach, board, ply, QD1_LINE[ply:], 2.5 + ply, (5 - ply) // 2)
    unasked = [m.text for m in drain(coach) if m.trigger is not Trigger.POST_MOVE]
    assert unasked == [], unasked
    # The idea at the start, the move found, a reply off the line: the idea not again.
    coach, _ = scripted_coach([])
    board = _puzzle(coach)
    _analysed(coach, board, 0, QD1_LINE, 1.0, 3)
    assert drain(coach) == [], "a puzzle's first position says nothing of its own"
    first = stuck(coach)
    assert [(m.trigger, m.level) for m in first] == [(Trigger.STUCK, DisclosureLevel.IDEA)]
    _played(coach, board, 1, "c2d1", True, 3.0)
    _played(coach, board, 2, "a5e1", False, 4.0)  # blocks instead of taking: off the line
    _analysed(coach, board, 2, ["d1e1"], 4.5, 1)
    assert [m for m in drain(coach) if m.trigger is not Trigger.POST_MOVE] == []
    coach.tick(4.0 + coach.policy.stuck_threshold() + 1)
    focus = [m for m in drain(coach) if m.trigger is Trigger.STUCK]
    assert [m.level for m in focus] == [DisclosureLevel.FOCUS], focus


def test_a_new_puzzle_gets_its_hints_whatever_the_last_one_had() -> None:
    """The counts of what was said unasked were kept by ply, and every puzzle starts at ply
    0: after one puzzle's idea and focus there, the next puzzle's log said "gave the idea
    and the focus here" before a word of it (Tai's phone, 2026-09-29). Each comes after its
    own stuck spell, and nothing before the first (Tai, 2026-10-01)."""
    from sidecoach.events import Resync

    def both_hints() -> list[DisclosureLevel]:
        assert drain(coach) == [], "nothing before the stuck limit"
        levels = [m.level for m in stuck(coach)]
        levels += [m.level for m in stuck(coach)]  # a full stuck spell from the idea
        return levels

    coach, _ = scripted_coach([])
    board = _puzzle(coach)
    _analysed(coach, board, 0, QD1_LINE, 1.0, 3)
    assert both_hints() == [DisclosureLevel.IDEA, DisclosureLevel.FOCUS]
    coach.on_event(
        Resync(game_id="g", fen="6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1", reason="adopted"), 100.0
    )
    _analysed(coach, chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"), 0, ["g8f8"], 101.0, None)
    assert both_hints() == [DisclosureLevel.IDEA, DisclosureLevel.FOCUS]


def test_clock_scales_the_policy_and_a_new_game_resets_it() -> None:
    """A ClockObservation feeds the interjection cadence: it bands the time control and
    sets the user's remaining time; a new game clears it back to untimed until a clock is
    seen again; and never seeing a clock leaves the policy on its untimed defaults."""
    from sidecoach.coach.policy import TimeControl
    from sidecoach.events import ClockObservation

    coach, _ = scripted_coach([])
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 0.0
    )
    # Before any clock, the policy keeps its untimed default (every trigger fires).
    assert coach.policy.time_control is TimeControl.UNTIMED
    # A fresh 10+0 game: the peak clock bands it as rapid.
    coach.on_event(ClockObservation(white_ms=600_000, black_ms=600_000, running=Side.WHITE), 1.0)
    assert coach.policy.time_control is TimeControl.RAPID
    # The user is White, so their remaining is White's; a low clock shrinks the stuck hint
    # (down to the floor, which the test coach's ten-second setting already sits under).
    coach.policy.settings = PolicySettings(stuck_seconds=45)
    coach.on_event(ClockObservation(white_ms=50_000, black_ms=300_000, running=Side.WHITE), 2.0)
    assert coach.policy._remaining_ms == 50_000
    assert coach.policy.stuck_threshold() < coach.policy.settings.stuck_seconds
    # A new game clears the clock back to untimed until one is seen again.
    coach.on_event(
        GameStarted(game_id="g2", user_side=Side.BLACK, start_fen=chess.STARTING_FEN), 3.0
    )
    assert coach.policy.time_control is TimeControl.UNTIMED
    assert coach.policy._remaining_ms is None
    # Now the user plays Black: their remaining is the black clock, and ~3 min is blitz.
    coach.on_event(ClockObservation(white_ms=185_000, black_ms=180_000, running=Side.BLACK), 4.0)
    assert coach.policy._remaining_ms == 180_000
    assert coach.policy.time_control is TimeControl.BLITZ


def test_the_buttons_are_the_models_to_answer_and_the_templates_offline() -> None:
    """Hint, Best move and Line. Asked of the AI coach, the model answers, with its marks
    and nothing said before it: its first words come within a second or two (Tai,
    2026-09-28: "it's already almost instant"). Offline the template for the level goes out
    at once, the move and the line asked for bare are the template's alone, and no model is
    asked."""
    coach, _ = scripted_coach(["Qxf7# ends it: the queen takes with check and mate."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("what should I play", 8.0)
    assert coach.poll() == [], "nothing before the model's answer"
    out = drain(coach)
    assert [m.text for m in out] == ["Qxf7# ends it: the queen takes with check and mate."]
    assert out[0].trigger is Trigger.USER_ASK and out[0].level is DisclosureLevel.MOVE
    assert any(mark.kind == "arrow" for mark in out[0].annotations)
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("what should I play", 8.0)
    quick = coach.poll()  # at once
    assert len(quick) == 1 and quick[0].text.startswith("Qxf7#"), quick
    assert quick[0].level is DisclosureLevel.MOVE
    assert any(mark.kind == "arrow" for mark in quick[0].annotations)
    assert drain(coach) == []


def test_claims_are_checked_off_the_callers_thread_and_only_when_there_are_any() -> None:
    from collections.abc import Callable

    from sidecoach.coach.coach import may_carry_claims

    assert not may_carry_claims("hint")
    assert not may_carry_claims("what should I play")
    assert not may_carry_claims("show me the line")
    assert may_carry_claims("is Nf3 good here")
    assert may_carry_claims("my knight is hanging")
    assert may_carry_claims("can I castle")

    calls: list[str] = []

    def claims(text: str, board: chess.Board, facts: object) -> list[str]:
        calls.append(text)
        return ["'Nf3 is good': not the strongest, Qxf7# mates"]

    provider = ScriptedProvider(["Not quite. What is your most active piece already eyeing?"])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), claims=claims
    )
    # The work handed off is kept, and run when the test chooses: no threads, no waits
    # (rewritten from Events and timed joins, Tai's rule of 2026-10-01).
    spawned: list[tuple[Callable[[], None], str]] = []
    coach.spawn = lambda work, name: spawned.append((work, name))
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    # ask() comes back with the check not yet run: the caller never waits on it.
    coach.ask("is Nf3 good here", 8.0)
    assert calls == [] and [name for _, name in spawned] == ["sidecoach-claims"]
    spawned.pop()[0]()
    assert calls == ["is Nf3 good here"]
    out = drain(coach)
    assert [m.text for m in out] == ["Not quite. What is your most active piece already eyeing?"]
    assert "Their claims, checked against the board" in provider.prompts[0][1][-1]["content"]
    # A bare hint never runs the check.
    coach.hint(9.0)
    drain(coach)
    assert len(calls) == 1


def test_an_ask_overtakes_a_remark_still_waiting() -> None:
    from sidecoach.coach.coach import Speaker, _Request

    provider = ScriptedProvider([])
    speaker = Speaker(provider)
    coach, _ = scripted_coach([])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    position = coach.positions[6]
    assert position.facts is not None

    def request(trigger: Trigger) -> _Request:
        return _Request(
            trigger=trigger,
            level=DisclosureLevel.IDEA,
            system="",
            messages=[{"role": "user", "content": "x"}],
            facts=position.facts,  # type: ignore[arg-type]
            board=position.board,
            fallback="fallback",
            ply=6,
            game_id="g",
        )

    speaker.submit(request(Trigger.STUCK))
    speaker.submit(request(Trigger.POST_MOVE))
    speaker.submit(request(Trigger.USER_ASK))
    # Ahead of the remark, never instead of it (Tai, 2026-10-02: no reply is dropped); the
    # stuck hint, said first, stays first (`test_a_question_drops_nothing_queued`).
    assert speaker.pending == 3
    speaker.run_pending()
    assert [s.message.trigger for s in speaker.poll()] == [
        Trigger.STUCK,
        Trigger.USER_ASK,
        Trigger.POST_MOVE,
    ]


def test_material_is_always_from_the_users_side_and_the_review_draws_on_the_tally() -> None:
    """The first post-game question ("how did I do overall?") was answered "you were down
    a queen for much of the middlegame" to a player who had won a queen: the material was
    read from whichever side was to move, and the memory carried that into the verdict.
    Now the facts, the sheet and the memory all speak from the user's side, a finished
    game withholds nothing, and an overall verdict is drawn from the engine's tally. The
    model sees the game as it went: every move a line with the engine's word on it, and the
    running tally of the player's moves; nothing is summarised away."""
    coach, provider = scripted_coach(["You kept it clean: nothing lost, and the finish was sharp."])
    start_game(coach)
    # 1. e4 e5 2. Qh5 Nc6 3. Qxe5+ (White, the user, wins a pawn; then Black to move.)
    board = play(coach, ("e4", "e5", "Qh5", "Nc6", "Qxe5+"))
    # The engine looks at the position with Black to move.
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=5,
            lines=[Line(moves=["f8e7"], depth=12, expected_score=0.4)],
            decisive=False,
            only_move=False,
        ),
        6.0,
    )
    facts = coach.positions[5].facts
    assert facts is not None
    assert facts.material == "you are up a pawn", "from the user's side, not the mover's"
    assert facts.user_side == "white" and facts.side_to_move == "black"
    assert "Material: the player is up a pawn." in coach.conversation.render_ply(
        coach.conversation.records[-1], DisclosureLevel.LINE
    )
    # The user's moves get their labels tallied.
    for ply, (uci, label) in {
        1: ("e2e4", Label.BOOK),
        3: ("d1h5", Label.INACCURACY),
        5: ("h5e5", Label.BEST),
    }.items():
        coach.on_event(
            MoveAssessment(
                game_id="g",
                ply=ply,
                uci=uci,
                expected_loss=0.03 if label is Label.INACCURACY else 0.0,
                severity=Severity.INACCURACY if label is Label.INACCURACY else Severity.NONE,
                best=[uci],
                label=label,
            ),
            7.0,
        )
    drain(coach)
    # The game ends; an overall question is answered from the tally, at full disclosure.
    coach.on_event(
        GameEnded(game_id="g", result="1-0", reason="resigned", pgn="", final_fen=board.fen()), 8.0
    )
    coach.ask("how did I do overall this game?", 9.0)
    out = drain(coach)
    assert out and out[-1].level is DisclosureLevel.LINE
    messages = sent(provider)
    directive = directive_of(messages)
    assert "How they did overall: from the verdicts and the tally in the record" in directive
    assert "nothing inferred from the position on the board" in directive
    assert "1. e4 (the player): book." in _record(messages).splitlines()
    harness = harness_of(messages)
    assert "The player's moves so far (3 judged): 1 best, 1 book, 1 inaccuracy." in harness
    assert "Ground lost most at: 2. Qh5 (inaccuracy)" in harness
    assert "    EVENT: Game over: 1-0" in harness
    # Other ways of asking are reviews too (test_align.py has them), a plan is not, and when
    # the model's review fails the checks the template is the record, not "I have no line
    # for this yet".
    from sidecoach.coach.coach import _ASK_REVIEW

    assert not _ASK_REVIEW.search("what's the plan so far?")
    provider.replies = ["", ""]
    coach.ask("what do I keep missing?", 10.0)
    out = drain(coach)
    assert "1 of your 3 moves" in out[-1].text and "2. Qh5 (inaccuracy)" in out[-1].text
    # One of them: "was", not "were" (the audit, 2026-10-07: grammar in the templates).
    assert out[-1].text.startswith("1 of your 3 moves was best."), out[-1].text


def test_a_bare_hint_is_asked_as_a_hint_and_one_the_model_fails_twice_is_the_template() -> None:
    """ "hint" carries no claim; prompted to be honest about "their claim", the model answered
    its own previous message as though the player had said it ("Your suggestion to focus on
    opening lines is spot-on"). When every candidate fails, the second try carries the
    complaint inside the directive: appending the failed attempt as two more messages doubled
    the prompt on a long game, past the model's context. Failed twice, the template for the
    rung speaks in its place, once, and the failed calls are in the log."""
    from sidecoach.coach.coach import _ASK_HINT

    for said in ("hint", "Hint", "give me a hint", "a clue please", "help", "any tips?"):
        assert _ASK_HINT.match(said), said
    for said in ("what should I play", "is Nf3 a good idea", "help me with the knight"):
        assert not _ASK_HINT.match(said), said
    coach, provider = scripted_coach(["", ""])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 8.0)
    out = drain(coach)
    # The idea template: the goal and the pattern.
    assert len(out) == 1 and "mate in one" in out[0].text.split(".")[0]
    # Nothing was said before the model was asked: their words are the last turn.
    given = sent(provider, 0)
    said = _current_move_message(given).split("## Said here", 1)[1].strip(chr(10)).splitlines()
    assert said[-1] == '    PLAYER said: "hint"'
    directive = directive_of(given)
    # No keyword picked the hint's directive: the guidelines, with the ladder as it stands.
    assert "A hint: one rung up the ladder" in directive and "nothing yet" in directive
    assert "They asked for a hint." not in directive
    retry = sent(provider, 1)
    assert len(retry) == len(given)
    assert "Those replies are not allowed" in retry[-1]["content"]
    assert "Your first attempt was" in retry[-1]["content"]
    spoken = coach.spoken[-1]
    assert spoken.attempts == 2 and spoken.problems and coach.speaker.fallbacks == 1
    turns = [e for e in coach.conversation.entries if e.role == "assistant"]
    assert [e.text for e in turns] == [out[0].text]


def _bishop_check_instead_of_mate(coach: Coach) -> None:
    """The user plays Bxf7+ where Qxf7 was mate, and the engine has judged it."""
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_missed(coach)
    drain(coach)
    # The engine has looked at the position the move left (the opponent to move): with the AI
    # on a question waits for that look, as it reads the position now.
    board = chess.Board(SCHOLAR)
    board.push_san("Bxf7+")
    _looked(coach, board, 7, ["e8f7"], 11.5)
    drain(coach)


def test_a_question_about_the_last_move_gets_its_verdict_in_full() -> None:
    """ "Why was that bad?" is about a position that is past: the directive states the
    verdict, the cost, the move the engine preferred and the idea, and says nothing is
    withheld. The generic ask directive ("the better move is withheld") had the model
    answer with a hint about the position now instead."""
    coach, provider = scripted_coach(["...", "..."])
    _bishop_check_instead_of_mate(coach)
    provider.replies = ["", ""]
    coach.ask("why was that bad?", 12.0)
    out = drain(coach)
    # Taught from the position the move was played from, as the lesson after it was.
    assert "# The position they faced, before Bxf7+" in str(sent(provider)[-1]["content"])
    # With the AI on the guidelines say a move already played may be told in full, and the
    # record gives the verdict, the better move and the sheet of the position they faced.
    directive = directive_of(sent(provider))
    assert "A move they already played: its verdict is in the record" in directive
    assert "They ask about the move they already played" not in directive
    assert "45%" not in directive, "the figure is for when they ask: no word of it unasked"
    assert "no percentages, scores or move numbers unless they ask for the number" in directive
    assert "the engine preferred 4. Qxf7#" in str(sent(provider)[-1]["content"])
    # The fallback (a reply that failed its checks) is the verdict itself, not a hint -- and
    # without the figure.
    assert out and out[-1].text.startswith("Qxf7# was better")
    # Asked what it cost, the fallback has the figure, theirs to hear.
    coach, provider = scripted_coach(["...", "..."])
    _bishop_check_instead_of_mate(coach)
    provider.replies = ["", ""]
    coach.ask("how much did that cost me?", 12.0)
    out = drain(coach)
    assert out and out[-1].text.startswith("It cost you 45% of the game. Qxf7# was better")


def test_a_follow_up_gets_what_it_asks_and_a_nod_one_sentence() -> None:
    """The verdict and the better move came back in every reply, and "I understand: I should
    play Nxc3" got the lesson again and "Which White piece is still at home on g1?" (the AI's
    sessions, 2026-09-26). With a model the words are the model's to read, with the sheet of
    the position the move was played from beside them (Tai's puzzle, 2026-09-28). A reply
    that does not name the better move again, as told, is said: the check that a reply names
    it refused every reply that obeyed, and the template said the verdict again (the prompt
    audit, 2026-09-28)."""
    coach, provider = scripted_coach(["...", "...", "...", "..."])
    _bishop_check_instead_of_mate(coach)
    with coach._lock:
        coach.conversation.coach_said("That one cost you: Qxf7# was better -- it is mate.")
    provider.replies = ["", ""]
    coach.ask("I see, I should have seen the mate.", 12.0)
    drain(coach)
    directive = directive_of(sent(provider))
    # What the coach has said since the move is a fact of the conversation, not a keyword: the
    # guideline for a follow-up is there, and the model reads whether the words ask anything.
    assert "do not say it again in other words" in directive
    assert "do not repeat the verdict or the better move you told them" in directive
    reply = "A check that leads nowhere only hands their king a free move."
    provider.replies = [reply, reply]
    coach.ask("Why does the bishop check not work?", 13.0)
    out = drain(coach)
    request = str(sent(provider)[-1]["content"])
    assert "# The position they faced, before Bxf7+" in request
    directive = directive_of(sent(provider))
    assert "do not say it again" in directive
    # A question is no claim to agree with (Luna: "You are right that g1 is controlled by
    # your King", Tai's puzzle, 2026-09-29).
    assert "a question states nothing to agree with" in directive
    assert out and out[-1].text == reply


def test_a_question_about_a_fine_last_move_says_it_was_fine() -> None:
    coach, provider = scripted_coach(["...", ""])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_played(coach)
    drain(coach)
    coach.ask("was that a mistake?", 12.0)
    drain(coach)
    directive = directive_of(sent(provider))
    # The verdict is in the record the model reads; the guidelines say a sound move gets no
    # invented fault.
    assert "inventing no fault in a sound move" in directive
    assert "4. Qxf7# by the player: best" in harness_of(sent(provider))


def test_asked_why_the_answer_says_why_whole_and_a_bare_one_gets_the_line() -> None:
    """Tai, 2026-09-27: "I want to hear the like why was that bad or why was it good or what
    was the better move or what was the better idea". In the offline sweep's bot games 61 of
    95 answers to "why was that bad?" were the better move and nothing else: the answer was
    trimmed of the reason the lesson after the move had just given. Asked, the reason is said
    whole; where the coach has no words for the better move, the engine's line from the
    position the move was played from is the reason (f363ea0)."""
    from sidecoach.coach.coach import _with_line

    def why_was_that_bad(bare: bool) -> tuple[str, str]:
        """The Qd1+ puzzle, 1...h5?? played: the lesson after it, and the answer to the
        question, both the templates' (the offline coach's, and the model's fallback)."""
        coach = offline_coach()
        board = _puzzle(coach)
        _analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        drain(coach)
        facts = coach.positions[0].facts
        assert facts is not None
        if bare:  # no words for why Qd1+ works
            coach.positions[0].facts = facts.model_copy(update={"why": None, "theme": None})
        _played(coach, board, 1, "h6h5", True, 2.0)
        coach.on_event(
            MoveAssessment(
                game_id="g",
                ply=1,
                uci="h6h5",
                expected_loss=0.5,
                severity=Severity.BLUNDER,
                best=["c2d1"],
                label=Label.BLUNDER,
            ),
            2.5,
        )
        (lesson,) = [m.text for m in drain(coach)]
        coach.ask("why was that bad?", 3.0)
        return lesson, drain(coach)[-1].text

    reason = "it is a queen sacrifice, and it forces mate: Qd1+ Rxd1 Rxd1+ Qe1 Rxe1#"
    lesson, answer = why_was_that_bad(bare=False)
    assert reason in lesson
    assert answer == f"Qd1+ was better -- {reason}.", "said again, whole: they asked"
    lesson, answer = why_was_that_bad(bare=True)
    assert "the line runs" not in lesson, "the line is for the answer to their question"
    assert answer == "Qd1+ was better -- the line runs Qd1+ Rxd1 Rxd1+ Qe1 Rxe1#."
    # The line only where the answer names nothing but the better move, and only a line that
    # starts with it.
    facts = PositionFacts(
        fen=QD1, side_to_move="black", phase="middlegame", line=["Qd1+", "Rxd1", "Rxd1+"]
    )
    kept = "Qd1+ was better -- it forces mate."
    assert _with_line(kept, "Qd1+ was better", "Qd1+", facts) == kept
    assert _with_line("Rd1 was better.", "Rd1 was better", "Rd1", facts) == "Rd1 was better."
    assert _with_line("Qd1+ was better.", "Qd1+ was better", "Qd1+", None) == "Qd1+ was better."


class _ItalianEngine:
    """Before the move the best is Bb5 at 0.55; after 3. Bc4, 0.48 for Black to move (0.52
    for White), the reply ...Bc5."""

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        if board.turn == chess.WHITE:
            return [Line(moves=["f1b5"], depth=14, expected_score=0.55)]
        return [Line(moves=["f8c5", "d2d3"], depth=14, expected_score=0.48)]


def _italian(coach: Coach) -> None:
    """1. e4 e5 2. Nf3 Nc6, White to move, looked at by the engine."""
    start_game(coach)
    board = play(coach, ("e4", "e5", "Nf3", "Nc6"))
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=4,
            lines=[Line(moves=["f1b5"], depth=12, expected_score=0.55)],
            decisive=False,
            only_move=False,
            tags=[],
        ),
        5.0,
    )
    drain(coach)


def test_a_named_move_is_analysed_by_the_engine_and_put_before_the_coach() -> None:
    """ "What about Bc4?" once got only "not among the engine's best moves". With an engine
    the object carries the engine's word on that move: the score after it, how far behind
    the best, the reply, what it does -- and the directive says it is theirs to hear."""
    # The move named in the words is examined; the model answers from it.
    coach, provider = scripted_coach(
        ["[p=1.0] Bc4 is fine: a little behind the best. What does it aim at?"]
    )
    coach.engine = _ItalianEngine()
    coach.sync_claims = True  # a question with the engine runs on a worker otherwise
    _italian(coach)
    coach.ask("what about Bc4?", 6.0)
    out = drain(coach)
    # The model answers from the examination, with nothing said before it (Tai, 2026-09-28:
    # every question the AI coach's to answer); offline the examination is the answer.
    assert [m.text for m in out] == ["Bc4 is fine: a little behind the best. What does it aim at?"]
    said = (
        _current_move_message(sent(provider))
        .split("## Said here", 1)[1]
        .strip(chr(10))
        .splitlines()
    )
    assert said[-1] == '    PLAYER said: "what about Bc4?"'
    directive = directive_of(sent(provider))
    assert "theirs to hear in full" in directive
    assert (
        "Engine on 3. Bc4, which the player asked about: after 3. Bc4 the expected score "
        "is 0.52" in directive
    )
    assert "0.03 behind the engine's best;" in directive
    assert "the opponent's reply would be 3... Bc5 4. d3" in directive
    assert "what Bc4 does: attacks the pawn on f7" in directive
    # Below LINE the engine's own preferred move is not named.
    assert "Bb5" not in directive


def test_a_move_proposed_in_a_statement_is_examined_and_nothing_is_said_twice() -> None:
    """ "I'll play Qd1, blocking the check" answered the templates' hint, and got the hint
    back word for word, three times (the students' round 5): a move proposed is examined as
    a question about it is, and the model answers from the examination; and a template
    already said since the last move is not said again."""
    coach, provider = scripted_coach(["[p=1.0] Fine."])
    coach.engine = _ItalianEngine()
    coach.sync_claims = True  # a question with the engine runs on a worker otherwise
    _italian(coach)
    coach.ask("I'll play Bc4 and aim at f7.", 6.0)
    assert [m.text for m in drain(coach)] == ["Fine."]
    assert "Engine on 3. Bc4, which the player asked about" in directive_of(sent(provider))
    # Offline, a statement naming no move gets the template once, and not again.
    coach = offline_coach()
    _italian(coach)
    coach.ask("I think my position is fine.", 7.0)
    first = [m.text for m in drain(coach)]
    coach.ask("I think my position is fine.", 8.0)
    again = [m.text for m in drain(coach)]
    assert first and not set(again) & set(first)


def test_a_cause_that_comes_back_is_named_once_with_the_habit_that_fixes_it() -> None:
    """Each lesson taught its own moment and a principle of its own, and the threat-blind
    student's notebook held five unrelated rules (the simulated students, 2026-09-26): the
    third mistake of one cause in the session says the pattern once, with its habit. Said the
    second time and then in every game, "this keeps happening -- 5 times now" nagged (Tai's
    phone, 2026-09-27); the review sums it up after."""
    from sidecoach.events import ProfileEvidence
    from sidecoach.profile import causes

    coach, _ = scripted_coach([])
    start_game(coach)

    def placed(ply: int, cause: str, game_id: str = "g") -> list[str]:
        coach.on_event(
            ProfileEvidence(
                game_id=game_id, ply=ply, phase="middlegame", expected_loss=0.2, cause=cause
            ),
            float(ply),
        )
        return [m.text for m in drain(coach)]

    assert placed(9, causes.MISSED_THREAT) == []
    assert placed(13, causes.POSITIONAL) == []
    assert placed(21, causes.MISSED_THREAT, game_id="old") == []  # another game's
    assert placed(25, causes.MISSED_THREAT) == [], "twice is not a pattern yet"
    third = placed(31, causes.MISSED_THREAT)
    # Said as a skill to work on, never a count (Tai, 2026-10-02: "That is three times now"
    # read as scolding).
    assert len(third) == 1 and third[0].startswith("Something worth working on:")
    assert "times" not in third[0] and "again" not in third[0]
    assert "spotting what their moves threaten" in third[0]
    assert "A habit that helps: before each move, look at what their last move attacks" in third[0]
    # Once a session: not again in this game, nor in the next.
    assert placed(37, causes.MISSED_THREAT) == []
    coach.game_id = "g2"
    assert placed(9, causes.MISSED_THREAT, game_id="g2") == []
    assert placed(15, causes.MISSED_THREAT, game_id="g2") == []

    # A player of 1800 and up hears the pattern without the textbook habit.
    strong_coach, _ = scripted_coach([])
    start_game(strong_coach)
    strong_coach.strong_player = True
    for ply in (9, 13, 17):
        strong_coach.on_event(
            ProfileEvidence(
                game_id="g", ply=ply, phase="middlegame", expected_loss=0.2,
                cause=causes.MISSED_THREAT,
            ),
            float(ply),
        )  # fmt: skip
    strong = [m.text for m in drain(strong_coach)]
    assert len(strong) == 1 and "spotting what their moves threaten" in strong[0]
    assert "look at what" not in strong[0]


def test_a_strong_players_review_names_the_pattern_without_the_textbook_habit() -> None:
    """A 2000 player was told "you kept missing endgame technique: bring your King to the
    middle and push your passed pawns" (the AI's sessions, 2026-09-26)."""
    from sidecoach.events import GameReview

    coach, _ = scripted_coach([])
    coach.strong_player = True
    start_game(coach)
    pattern = (
        "One thing to work on: endgame technique. A habit that helps: in the endgame, bring "
        "your king to the middle and push your passed pawns."
    )
    coach.on_event(
        GameReview(
            game_id="g",
            user_side=Side.WHITE,
            result="1-0",
            moves=40,
            lines=["You played white and won in 40 moves.", pattern],
        ),
        1.0,
    )
    said = " ".join(m.text for m in drain(coach))
    assert "One thing to work on: endgame technique." in said and "passed pawns" not in said
    # What the model is shown of the review says it the same way.
    assert "passed pawns" not in coach.conversation.render()


def test_the_review_is_spoken_from_the_profiles_facts_and_the_next_game_knows_the_player() -> None:
    """The game ends and the profile's review arrives: it joins the conversation as a
    message of its own and the coach sums it up through the model, the review's sentences
    as the template. The profile's snapshot puts what earlier games say into the next
    game's setting, and the review stays with its game: game 2 answered "how did I do
    overall?" with game 1's sentences when the model failed (the parity audit, 2026-09-26)."""
    from sidecoach.events import GameReview, ProfileSnapshot, TurningPoint

    summary = (
        "You won by checkmate. Your one lapse was 3. Qh5, where Nf3 kept the edge; "
        "the rest was clean. Work on spotting what your opponent threatens."
    )
    # The same reply to every call: the earlier triggers (the verdict on the mate) take
    # their share of them; the review is the last.
    coach, provider = scripted_coach([summary] * 8)
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 6.5)  # facts for the position the review is checked on
    mate_played(coach)
    coach.on_event(GameEnded(game_id="g", result="1-0", reason="checkmate"), 12.0)
    review = GameReview(
        game_id="g",
        user_side=Side.WHITE,
        result="1-0",
        reason="checkmate",
        moves=4,
        accuracy=88.0,
        tallies={"best": 3, "mistake": 1},
        turning_points=[
            TurningPoint(
                ply=5,
                san="3. Qh5",
                label="mistake",
                expected_loss=0.11,
                best_san="Nf3",
                cause="missed opponent threat",
                motif="fork",
            )
        ],
        lines=[
            "You played white and won by checkmate in 4 moves.",
            "Accuracy 88.0. Your 4 moves: 3 best, 1 mistake.",
            "Ground lost at 3. Qh5, a mistake costing 0.11; Nf3 was better"
            " (missed opponent threat: fork).",
        ],
    )
    coach.on_event(review, 12.5)
    said = drain(coach)
    assert said and said[-1].trigger.value == "review"
    assert "Qh5" in said[-1].text and "Nf3" in said[-1].text
    given = sent(provider)
    assert "The review of the game:" in harness_of(given)
    assert "Sum it up from the review" in directive_of(given)
    assert coach._review_lines == review.lines
    # The snapshot's insights and last session frame the next game.
    coach.on_event(
        ProfileSnapshot(
            games=3,
            insights=[
                "Your biggest cost over 3 games is missed opponent threat: 4 mistakes,"
                " 0.70 expected score."
            ],
            last_session="2 games (1 won, 1 lost), accuracy 80.",
        ),
        13.0,
    )
    coach.on_event(
        GameStarted(game_id="g2", user_side=Side.BLACK, start_fen=chess.Board().fen()), 14.0
    )
    setting = coach.conversation.render()
    assert "What earlier games say about this player" in setting
    assert "missed opponent threat" in setting and "Last session: 2 games" in setting
    assert coach._review_lines == []


def test_a_question_about_the_player_over_their_games_goes_to_the_record() -> None:
    """The record's answerer is asked off the coach's thread; the answer is spoken as the
    coach's turn, the player's words and a note of the lookup join the conversation, and
    the queries are kept for the log. Questions about this game stay with the game."""
    from collections.abc import Callable

    from sidecoach.profile.ask import Answer, Exchange

    coach, provider = scripted_coach(["Keep looking."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 6.5)
    asked: list[str] = []

    def lookup(question: str) -> Answer:
        asked.append(question)
        return Answer(
            text="Over 9 games you left the opening ahead 7 times and were behind by move 25.",
            exchanges=[Exchange(sql="SELECT 1", columns=["n"], rows=[(1,)])],
            calls=2,
        )

    coach.profile_ask = lookup
    # Off the coach's thread: the work handed off, run here when the test chooses (rewritten
    # from a sleep-and-poll loop, Tai's rule of 2026-10-01).
    spawned: list[tuple[Callable[[], None], str]] = []
    coach.spawn = lambda work, name: spawned.append((work, name))
    coach.ask("why am I always losing in the middlegame?", 7.0)
    assert asked == [] and [name for _, name in spawned] == ["profile-ask"]
    spawned.pop()[0]()
    said = drain(coach)
    assert asked == ["why am I always losing in the middlegame?"]
    assert (
        said and said[-1].text.startswith("Over 9 games") and said[-1].trigger.value == "user_ask"
    )
    assert coach.profile_answers[0][0] == "answer"
    record = coach.conversation.render()
    assert 'PLAYER said: "why am I always losing in the middlegame?"' in record
    assert "looked the question up in the record (1 query)" in record
    # A question about this position never goes to the record.
    coach.ask("what should I play here?", 8.0)
    assert asked == ["why am I always losing in the middlegame?"]


ABOUT_THE_BOARD = (
    "Do I keep my bishop or trade it?",
    "Do I keep the queens on?",
    "Should I keep my knight on f3?",
    "Am I better here?",
    "Am I worse after that?",
    "Am I better or worse in this endgame?",
    "Why am I losing?",
    "Why am I so far behind?",
    "Why am I losing this game?",
    "What tactics do I have?",
    "Which tactic do I have here?",
    "What's the biggest problem in this position?",
    "What is the main weakness in their camp?",
    "What is the worst piece I have?",
    "What should I focus on here?",
    "Do I have time to castle?",
    "Am I winning?",
    "Do I take the pawn?",
    "Which phase am I in?",
)
ABOUT_THE_RECORD = (
    "What do I usually miss?",
    "Why do I keep losing in the endgame?",
    "Why do I keep hanging pieces?",
    "Do I always blunder when short of time?",
    "Do I tend to trade too early?",
    "What are my weaknesses?",
    "Am I getting better?",
    "Am I improving?",
    "How did I do over the last 10 games?",
    "Which openings do I play worst?",
    "What tactics do I miss most often?",
    "What's my most common mistake?",
    "What is the most common mistake I make?",
    "What should I study?",
    "What should I work on overall?",
    "Why am I so bad at endgames?",
    "How many games have I played?",
    "How were my last few games?",
    "What is my biggest weakness?",
    "Why am I always losing in the middlegame?",
)


def test_a_question_about_the_position_is_answered_about_the_position() -> None:
    """ "Do I keep my bishop or trade it?" went to the cross-game record, as did "Am I better
    here?" and "What tactics do I have?" (the audit, 2026-10-07): words of the position now
    -- "keep" a piece, "am I better", "do I have", "the main weakness" -- were read as words
    of a habit. The record is asked only about the player over time: how often (always,
    usually, keep doing), a trend (getting better), their games, their weaknesses."""
    from collections.abc import Callable

    from sidecoach.profile.ask import Answer

    def routed(asked: str) -> bool:
        coach, _ = scripted_coach([])
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 6.5)
        coach.profile_ask = lambda q: Answer(text="From the record.")
        spawned: list[tuple[Callable[[], None], str]] = []
        coach.spawn = lambda work, name: spawned.append((work, name))
        coach.ask(asked, 7.0)
        return "profile-ask" in [name for _, name in spawned]

    assert [q for q in ABOUT_THE_BOARD if routed(q)] == []
    assert [q for q in ABOUT_THE_RECORD if not routed(q)] == []


def test_offline_a_question_about_the_record_gets_what_the_record_says() -> None:
    """Offline, "what do I usually miss?" answered "I could not read the record just now; ask
    me again in a moment." every time: no model writes the queries, so the moment never came
    (the audit, 2026-10-07). The offline coach answers with what the record already says of
    the player (the profile's insights, the lines the AI is given about earlier games); with
    nothing there yet, it says so plainly."""
    from collections.abc import Callable

    from sidecoach.events import ProfileSnapshot
    from sidecoach.profile import ask as profile_ask

    insights = [
        "The thing to work on most over 12 games: forks you missed, which came up in 9 moves.",
        "Worth practising: the fork. You found it 3 of the 8 times you had one.",
    ]
    for given, expected in (
        (insights, " ".join(insights)),
        ([], "The record has nothing on that yet."),
    ):
        coach = offline_coach()
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 6.5)
        coach.on_event(ProfileSnapshot(games=12, insights=given), 6.6)
        # As the app wires it: the record's answerer with the coach's own (offline) provider.
        coach.profile_ask = lambda q, c=coach: profile_ask.answer(q, None, c.speaker.provider)  # type: ignore[arg-type]
        spawned: list[tuple[Callable[[], None], str]] = []
        coach.spawn = lambda work, name, kept=spawned: kept.append((work, name))
        coach.ask("what do I usually miss?", 7.0)
        while spawned:
            spawned.pop()[0]()
        said = [m.text for m in drain(coach)]
        assert said == [expected], said


def test_no_lesson_on_a_mistake_asks_what_the_player_expected() -> None:
    """ "What were you expecting that move to do?" ended every third lesson, for the profile's
    record; after a mistake it reads as a challenge (Tai, 2026-10-02: "very rude and super
    unwelcoming"). Neither the model's instruction nor the template asks it, however many
    mistakes came before."""
    coach, _ = scripted_coach([])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 6.5)
    position = coach.positions[6]
    assert position.facts is not None
    board = chess.Board(SCHOLAR)
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="a3",
        uci="a2a3",
        fen_before=SCHOLAR,
        fen_after=board.fen(),
        by_user=True,
    )
    assessment = MoveAssessment(
        game_id="g",
        ply=7,
        uci="a2a3",
        expected_loss=0.5,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.BLUNDER,
    )
    instruction, fallback = coach._instruction(
        Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, None, (), assessment, played
    )
    assert "expect" not in instruction and "expect" not in fallback

    # The offline coach, the same mistake six times over (taken back each time): every
    # lesson is a lesson, none asks what they expected or saw.
    offline = offline_coach()
    play_to_scholar(offline)
    offline.on_event(mate_analysis(6), 6.5)
    drain(offline)
    lessons = []
    for n in range(6):
        mate_missed(offline, 10.0 + 10 * n)
        lessons += [m.text for m in drain(offline) if m.trigger is Trigger.POST_MISTAKE]
        offline.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 15.0 + 10 * n)
        drain(offline)
    assert len(lessons) == 6, lessons
    for text in lessons:
        assert not re.search(r"expect|what did you|did you see|were you", text, re.I), text


def test_the_coach_is_told_a_position_is_not_a_game() -> None:
    """A puzzle or an analysis board: the coach hears it is a position with no history,
    the player the side at the bottom and, on a puzzle, to move."""
    coach, _ = scripted_coach([])
    puzzle = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"
    coach.on_event(
        GameStarted(game_id="p", user_side=Side.BLACK, start_fen=puzzle, kind="position"), 0.0
    )
    record = coach.conversation.render()
    assert "You coach Black." in record
    assert "This is a position, not a game from the start" in record
    assert "joined in progress" not in record


def test_a_puzzle_solved_by_mate_gets_no_word_and_no_summary() -> None:
    """The mating move on a position -- a puzzle -- gets nothing from the coach, and the
    end no summary: the site says the puzzle is solved and moves on (Tai). Other moves
    on a puzzle get their word as in a game."""
    coach, provider = scripted_coach([])
    puzzle = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    for n in range(2):
        coach.on_event(
            GameStarted(game_id=f"p{n}", user_side=Side.WHITE, start_fen=puzzle, kind="position"),
            10.0 * n,
        )
        coach.on_event(mate_analysis(0, game_id=f"p{n}"), 10.0 * n + 0.5)
        assert drain(coach) == []  # the position itself: nothing said unasked
        provider.calls.clear()
        board = chess.Board(puzzle)
        move = board.push_san("Qxf7#")
        coach.on_event(
            MovePlayed(
                game_id=f"p{n}",
                ply=1,
                san="Qxf7#",
                uci=move.uci(),
                fen_before=puzzle,
                fen_after=board.fen(),
                by_user=True,
            ),
            10.0 * n + 1.0,
        )
        coach.on_event(
            MoveAssessment(
                game_id=f"p{n}",
                ply=1,
                uci=move.uci(),
                expected_loss=0.0,
                severity=Severity.NONE,
                best=[move.uci()],
                label=Label.BEST,
            ),
            10.0 * n + 1.5,
        )
        assert drain(coach) == [], "the mating move is not commented on"
        coach.on_event(
            GameEnded(game_id=f"p{n}", result="1-0", reason="game over on board", pgn="1. Qxf7#"),
            10.0 * n + 2.0,
        )
        assert drain(coach) == [], "no summary of a puzzle"
    assert provider.prompts == [], "no model call at all"
    # A move on a puzzle that does not end it gets its word, as in a game.
    coach.on_event(
        GameStarted(game_id="q", user_side=Side.WHITE, start_fen=puzzle, kind="position"), 50.0
    )
    coach.on_event(mate_analysis(0, game_id="q"), 50.5)
    drain(coach)
    board = chess.Board(puzzle)
    move = board.push_san("Nf3")
    coach.on_event(
        MovePlayed(
            game_id="q",
            ply=1,
            san="Nf3",
            uci=move.uci(),
            fen_before=puzzle,
            fen_after=board.fen(),
            by_user=True,
        ),
        51.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="q",
            ply=1,
            uci=move.uci(),
            expected_loss=0.5,
            severity=Severity.BLUNDER,
            best=["h5f7"],
            label=Label.BLUNDER,
        ),
        51.5,
    )
    assert [m.trigger for m in drain(coach)] == [Trigger.POST_MISTAKE]


def test_a_fast_game_gets_the_templates_and_no_model_call() -> None:
    """The user's think times set the pace: under FAST_PACE_S a move, the coach's word
    on a move -- even a mistake's lesson -- is the template written from the board,
    at once, with no model call that could not keep up (Tai, 2026-09-22: a queen
    ending at two seconds a move, the coach silent from ply 80 to the mate). What the
    player asks still goes to the model."""
    from sidecoach.coach.coach import FAST_PACE_S

    coach, provider = scripted_coach(["never used for a move", "[p=1.0] Asked and answered."])
    coach.local_verdicts = False
    start_game(coach)
    # Three times faster than the line.
    board = play(coach, SCHOLAR_MOVES, think_ms=int(FAST_PACE_S * 1000 / 3))
    assert coach.fast_pace, "two quick moves of the user's and the game counts as fast"
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    mate_played(coach)
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MOVE and out[0].text
    assert provider.prompts == [], "no model call in a fast game"
    # The ask is the exception: the player is waiting for it.
    coach.ask("what was that about", 12.0)
    out = drain(coach)
    assert out and out[-1].trigger is Trigger.USER_ASK
    assert len(provider.prompts) >= 1, "the ask goes to the model"
    # Slow down and the model comes back.
    for ply in (8, 9):
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san="a3",
                uci="a2a3",
                fen_before=board.fen(),
                fen_after=board.fen(),
                by_user=True,
                think_ms=int(FAST_PACE_S * 3000),
            ),
            20.0 + ply,
        )
    assert not coach.fast_pace


def test_the_pace_is_the_last_think_times_and_the_games_own() -> None:
    """The pace is the last few think times; the player thinking longer than that in the
    position at hand has slowed down, and a hint or lesson may take the model's time (and
    the voice) again. And it is the game's own: a puzzle's last moves come quick -- the line
    found, played out -- and carried over they made the next puzzle fast from its first
    move: a find after two minutes' thought was only shown (Tai's puzzles, 2026-09-27:
    "sometimes it's just completely muted")."""
    from sidecoach.coach.coach import FAST_PACE_S

    coach, _provider = scripted_coach([])
    start_game(coach)
    play(coach, SCHOLAR_MOVES, think_ms=1000)
    assert coach.fast_pace and coach.user_to_move
    assert coach.fast_now(6.0 + 1.0), "a second into the position: still fast"
    assert not coach.fast_now(6.0 + FAST_PACE_S + 1), "thinking longer than the pace: slowed"
    coach.on_event(
        GameStarted(game_id="h", user_side=Side.WHITE, start_fen=chess.Board().fen()), 10.0
    )
    assert coach.pace_s is None and not coach.fast_pace
    # The next puzzle in the same orientation comes as a re-read of the board: the pace was
    # kept, and its first find was only shown (the game-state review, 2026-10-01).
    from sidecoach.events import Resync

    coach, _provider = scripted_coach([])
    start_game(coach)
    play(coach, SCHOLAR_MOVES, think_ms=1000)
    assert coach.fast_pace
    coach.on_event(Resync(game_id="g", fen=SCHOLAR, reason="position adopted"), 10.0)
    assert coach.pace_s is None and not coach.fast_pace


def _after_move_request(fen: str, san: str, best_uci: str):  # noqa: ANN202
    from sidecoach.coach.coach import _Request

    board = chess.Board(fen)
    facts = PositionFacts(fen=fen, side_to_move="white", phase="opening")
    return _Request(
        trigger=Trigger.POST_MISTAKE,
        level=DisclosureLevel.LINE,
        system="",
        messages=[],
        facts=facts,
        board=board,
        fallback="",
        ply=7,
        game_id="g",
        played_san=san,
        assessment=MoveAssessment(
            game_id="g",
            ply=7,
            uci=board.parse_san(san).uci(),
            expected_loss=0.2,
            severity=Severity.MISTAKE,
            best=[best_uci],
            label=Label.MISTAKE,
        ),
    )


RUY = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
F5 = "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"


def test_a_word_after_the_move_never_restates_it_nor_quotes_a_figure() -> None:
    """The owner's rules after a move, checked in code (the eval audit, 2026-09-25): never
    open with the move just played, never say what it plainly did unless the better move does
    the same, and no figure in a remark nobody asked for -- the last NO_FIGURES' note, where
    the directive carries it (`Speaker._notes`). The second audit: "ahead by about 1.2 pawns"
    was no figure to the old check; "you took on d5" and "that d5 pawn" said the capture back;
    "which was a inaccuracy"."""
    from sidecoach.coach.coach import _rated, figure_said, restated

    capture = "says what Bxc6 plainly did: the capture on c6"
    check = "says what Qh5 plainly did: the check"
    # (position, move played, the better move, the words, the faults: a list is all of them,
    # a string one among them)
    rows: list[tuple[str, str, str, str, list[str] | str]] = [
        (RUY, "Bxc6", "e1g1", "Bxc6 gives up the bishop pair.", [
            "opens with Bxc6, the move just played"
        ]),
        (RUY, "Bxc6", "e1g1", "Taking the knight on c6 gives up the bishop pair.", [capture]),
        (RUY, "Bxc6", "e1g1", "You took on c6, giving up the bishop pair.", capture),
        (RUY, "Bxc6", "e1g1", "Winning that c6 knight gives up the bishop pair.", capture),
        # The better move castles: saying so is the lesson, not a restatement.
        (RUY, "Bxc6", "e1g1",
         "Castling first was better: the king is safe, the rook joins in.", []),
        # The better move takes on the same square with check: the words are about it.
        (SCHOLAR, "Bxf7+", "h5f7",
         "Taking the pawn on f7 with check was right, but with the queen.", []),
        # The check said of the move played is a restatement; a check in the better line is not
        # (the benchmark's smoke run, 2026-09-25: "after Bg4 your knight jumps in with check").
        (F5, "Qh5+", "e4f5", "Sure, Qh5+ gives check, but exf5 wins a pawn.", check),
        (F5, "Qh5+", "e4f5", "That gave check, but exf5 wins a pawn.", check),
        (F5, "Qh5+", "e4f5", "exf5 wins a pawn, and after Nf6 your queen comes in with check.", []),
    ]  # fmt: skip
    for fen, played, best, text, faults in rows:
        found = restated(text, _after_move_request(fen, played, best))
        if isinstance(faults, str):
            assert faults in found, (text, found)
        else:
            assert found == faults, (text, found)
    for text, figure in (
        ("You are ahead by about 1.2 pawns.", True),
        ("It cost you 15% of the game.", True),
        ("Your rook on the 7th rank and Nf3 were strong.", False),
    ):
        assert figure_said(text) is figure, text
    assert [_rated(x) for x in ("inaccuracy", "miss", "best", "book")] == [
        "an inaccuracy",
        "a miss",
        "best",
        "a book move",
    ]


def test_the_draft_checker_sees_every_line_the_facts_were_read_from() -> None:
    """A trap the template names ("Bxf7+ looks natural, but after Kxf7 ...") is a line from
    the root: its boards are frames, so saying it is not faulted as unplayable."""
    from sidecoach.coach.coach import _Position

    analysis = mate_analysis(6)
    analysis.traps = [Line(moves=["c4f7", "e8e7"], depth=12, expected_score=0.4)]
    position = _Position(ply=6, board=chess.Board(SCHOLAR), analysis=analysis)
    frames = Coach._analysis_frames(position)
    assert any(f.piece_at(chess.E7) == chess.Piece(chess.KING, chess.BLACK) for f in frames)
    assert any(f.piece_at(chess.F7) == chess.Piece(chess.QUEEN, chess.WHITE) for f in frames)


def test_a_late_lesson_does_not_give_away_the_move_they_face_now() -> None:
    from sidecoach.coach.coach import Speaker

    request = _after_move_request(SCHOLAR, "Bxf7+", "h5f7")
    request.now_answer = ("Qxf7#",)
    speaker = Speaker(ScriptedProvider([]))
    said = "And now Qxf7# is mate on the spot."
    assert "gives away Qxf7# in the position they face now" in speaker._leaks(said, request)
    # Logged, never a reason to refuse the reply (Tai, 2026-09-26).
    assert not any(p.startswith("gives away") for p in speaker._problems(said, request))


def test_the_verdict_words_are_checked_clause_by_clause() -> None:
    """Every lesson names the better move, and skipping any reply that did let "That was a
    blunder: a3 was better" through for a good move (the runtime audit)."""
    from sidecoach.coach.coach import _verdict_problems

    good = _after_move_request(RUY, "O-O", "b5a4")
    assert good.assessment is not None
    good.assessment = good.assessment.model_copy(
        update={"severity": Severity.NONE, "label": Label.GOOD, "expected_loss": 0.04}
    )
    assert _verdict_problems("That was a blunder: Ba4 was better.", good) == [
        "called O-O a slip; its verdict is good"
    ]
    assert _verdict_problems("O-O was the best move here, though Ba4 was close.", good)
    assert _verdict_problems("Fine, but Ba4 was the best move here.", good) == []
    # A clause opening with "it" or "that" is about the move only where the verdict word is
    # said of it: these true sentences were refused (the deep prompt review, 2026-10-01).
    for said in (
        "It avoids the mistake of locking your bishop in.",
        "That keeps you clear of the classic blunder of moving the f-pawn.",
        "It was not a blunder, but Ba4 was a touch better.",
        "It is best to develop a piece before you push pawns.",
        "This is a position where the best move is quiet.",
    ):
        assert _verdict_problems(said, good) == [], said
    assert _verdict_problems("It was a clear mistake; Ba4 was better.", good)
    assert _verdict_problems("That is the best move here.", good)


def test_their_own_move_is_theirs_even_after_the_reply_and_a_threat_is_heard_as_one() -> None:
    # Where each reaches its directive: test_align.py.
    from sidecoach.coach.coach import _ASK_THREAT, _MY_MOVE

    for said in ("what should I have played?", "did I just blunder?", "was my move bad?"):
        assert _MY_MOVE.search(said), said
    assert not _MY_MOVE.search("why was that bad?"), "'that' is the last move played"
    for said in ("what is their threat?", "what does my opponent threaten?", "am I in danger?"):
        assert _ASK_THREAT.search(said), said


def test_a_question_only_about_a_fact_is_the_models_and_offline_the_templates() -> None:
    """The material, the opening by name, who stands better. Offline the template says them
    exactly, the fact itself, never the hint for the position ("How many pieces am I up?" was
    once answered "Something forcing is on..."). Asked of the AI coach, the model answers
    (Tai, 2026-09-28); and a question that only mentions the opening is not a question about
    its name: "what should I play against this opening? give me a line" got "The opening was
    the A40 Queen's Pawn Game.", and is the line's to answer."""
    from sidecoach.coach.coach import fact_question

    for said in ("how is my opening?", "why am I losing?", "am I winning? what should I do?"):
        assert not fact_question(said), said
    coach, provider = scripted_coach(["The model's words.", "Qxf7# -- it is mate at once."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("what opening is this?", 8.0)
    assert [m.text for m in drain(coach)] == ["The model's words."]
    # No keyword picks the question's directive: the same guidelines carry both questions, the
    # opening by its name and the line, and the model reads which is asked.
    assert "the opening by the name the record gives it" in directive_of(sent(provider, 0))
    coach.ask("what should I play against this opening? give me a line", 9.0)
    assert [m.text for m in drain(coach)] == ["Qxf7# -- it is mate at once."]
    directive = directive_of(sent(provider, 1))
    assert "The line, or what happens next: the engine's line move by move" in directive
    assert "They asked to see the line" not in directive
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    for ts, (said, opens) in enumerate(
        (
            ("who's up material?", "Material:"),
            ("how many pieces am I up?", "Material:"),
            ("am I winning?", "Right now you are winning"),
            ("what opening is this?", ("We left the opening book early.", "The opening was the")),
        ),
        start=8,
    ):
        assert fact_question(said), said
        coach.ask(said, float(ts))
        out = drain(coach)
        assert out and out[-1].text.startswith(opens), (said, out)


def test_their_threat_on_their_turn_is_their_best_move_not_the_players() -> None:
    """The sheet's threats are the side NOT to move's: on the opponent's turn, the player's
    own. "What are they threatening?" there is answered with their best move now."""
    coach, provider = scripted_coach(["", ""])
    start_game(coach, Side.BLACK)
    play(coach, SCHOLAR_MOVES, user_side=Side.BLACK)
    assert not coach.user_to_move
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("what are they threatening?", 8.0)
    out = drain(coach)
    assert "it is the opponent's move" in directive_of(sent(provider, 0))
    assert out and out[-1].text.startswith("It's their move, and their best is Qxf7#"), out
    # A hint asked on their turn is the same, never the player's hint on the opponent's
    # position: the model was asked for "the first hint" and to keep back "the player's move"
    # (the coach review, 2026-10-01).
    coach, provider = scripted_coach(["", ""])
    start_game(coach, Side.BLACK)
    play(coach, SCHOLAR_MOVES, user_side=Side.BLACK)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 9.0)
    out = drain(coach)
    directive = directive_of(sent(provider, 0))
    assert "it is the opponent's move" in directive
    assert "Give the first hint" not in directive and "Keep back only the answer" not in directive
    assert out and out[-1].text.startswith("It's their move, and their best is Qxf7#"), out


def test_what_to_play_on_their_turn_never_gives_their_move_as_the_players() -> None:
    """On the opponent's move the engine's best move and line are the opponent's: "what should
    I play?" told the model "tell them plainly it is Qxf7#", the opponent's mate given as the
    player's move (the deep prompt review, 2026-10-01)."""
    for said in ("what should I play?", "what is the best move?", "show me the line"):
        coach, provider = scripted_coach(["", "", ""])
        start_game(coach, Side.BLACK)
        play(coach, SCHOLAR_MOVES, user_side=Side.BLACK)
        coach.on_event(mate_analysis(6, decisive=False), 7.0)
        for ts in range(3):  # climb to the move rung
            coach.ask(said, 9.0 + ts)
        out = drain(coach)
        for i in range(len(provider.calls)):
            directive = directive_of(sent(provider, i))
            assert "it is the opponent's move" in directive, (said, directive)
            assert "tell them plainly it is" not in directive
            assert "They asked to see the line: give it" not in directive
        texts = [m.text for m in out if getattr(m, "text", None)]
        assert texts and texts[0].startswith("It's their move"), texts  # the empty reply's
        if said == "show me the line":
            # The line asked for is the line from their move on, each move named whose it is
            # (Tai, 2026-10-03: their next move alone was said).
            assert texts[0] == "It's their move, and the line runs their Qxf7#, checkmate.", texts
            assert "the line from here may be said" in directive_of(sent(provider, 0))


MOVES_INSTEAD = (
    "Qxf7# was the move there. Yours, Bxf7+, was a miss. "
    "Bxf7+ was clearly worse there: it passes up the mate."
)


def test_the_other_good_moves_on_their_turn_are_the_players_last_turns() -> None:
    """On the opponent's move the player has just moved: "any other good moves?", "does
    anything else work too?" and "what else could I have played?" ask what they could have
    played instead of it (Tai, 2026-10-02: "since they have not made a move yet (else it would
    have been your turn), that question should have been about other alternatives moves you
    could have taken during your previous turn"). Answered from the position they faced, with
    how theirs compares -- never with the opponent's near-equal moves named as the player's
    (the desktop's alternatives branch did that on their move) nor with the opponent's threat
    (the phone's). "What is their best move?" stays the opponent's."""
    from sidecoach.coach.coach import AI_GUIDELINES

    for said in (
        "any other good moves?",
        "does anything else work too?",
        "what else could I have played?",
    ):
        coach = offline_coach()
        play_to_scholar(coach)
        coach.quiet = True  # no word on the move: the answer is the only message
        coach.on_event(mate_analysis(6, decisive=False), 7.0)
        mate_missed(coach)  # Bxf7+, a miss: the opponent to move, their position not read
        coach.ask(said, 11.0)
        [answer] = drain(coach)
        assert answer.text == MOVES_INSTEAD, (said, answer.text)
        # The AI is told the same reading, and given the moves of that position as data.
        coach, provider = scripted_coach(["Qxf7# would have mated."])
        play_to_scholar(coach)
        coach.quiet = True
        coach.on_event(mate_analysis(6, decisive=False), 7.0)
        mate_missed(coach)
        board = chess.Board(SCHOLAR)
        board.push_san("Bxf7+")
        _looked(coach, board, 7, ["e8e7"], 10.5)  # the opponent's position read
        coach.ask(said, 11.0)
        assert [m.text for m in drain(coach)] == ["Qxf7# would have mated."]
        directive = directive_of(sent(provider))
        assert AI_GUIDELINES in directive and "what they could have played instead" in directive
        assert (
            "Before Bxf7+, the player's last move, the moves about equally good were: Qxf7#; "
            "Bxf7+ was a miss." in directive
        )
        assert "Clearly worse there, and what goes wrong with them: Bxf7+ passes up the mate" in (
            directive
        )
    # The opponent's move asked about is theirs, not the player's last turn's.
    coach = offline_coach()
    play_to_scholar(coach)
    coach.quiet = True
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    mate_missed(coach)
    board = chess.Board(SCHOLAR)
    board.push_san("Bxf7+")
    _looked(coach, board, 7, ["e8e7"], 10.5)
    coach.ask("what is their best move?", 11.0)
    [answer] = drain(coach)
    assert answer.text.startswith("It's their move"), answer.text
    assert "Ke7" in answer.text


def test_a_refused_answer_about_a_move_falls_back_to_the_engines_word_on_it() -> None:
    """Both drafts refused, "is Bxf7+ good?" got the mate-in-one hint, about another move
    (the deep prompt review, 2026-10-01). The fallback is the engine's word on Bxf7+."""
    from sidecoach.cli import _draft_check
    from tests.coaching import _policy
    from tests.test_engine_tools import FirstLegal

    bad = "[p=0.6] Your queen on e8 is in danger here. [p=0.4] Their bishop on h5 is loose."
    coach = Coach(
        ScriptedProvider([bad, bad]),
        policy=_policy(),
        draft_check=_draft_check(),
        engine=FirstLegal(),
    )
    coach.sync_claims = True
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("is Bxf7+ good?", 8.0)
    texts = [m.text for m in drain(coach) if getattr(m, "text", None)]
    assert coach.spoken[-1].problems, "both drafts were refused"
    assert texts and "after 4. Bxf7+" in texts[-1], texts
    assert "Qxf7" not in texts[-1], texts


def test_a_late_lesson_asks_nothing_while_they_think() -> None:
    """A verdict landing after the opponent's reply, with the player to move: "one question
    that sends them to find on the board" asked what the move allowed -- the threat they face
    now, a nudge while they think (the deep prompt review, 2026-10-01). The late lesson asks
    no question. What the move allowed is still said, as in any lesson: it happened on the
    board, and nothing of the lesson is held back (Tai, 2026-10-02: a late lesson had left it
    out -- the knight blundered and taken at once, the lesson never said the move hung it)."""
    coach, provider = scripted_coach(["[p=1.0] Qxf7 was mate at once."])
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    board = chess.Board(SCHOLAR)
    plays = []
    for ply, san in ((7, "d3"), (8, "Nxh5")):
        before = board.fen()
        move = board.parse_san(san)
        board.push(move)
        plays.append(move)
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply == 7,
            ),
            10.0 + ply,
        )
        if ply == 7:
            # The engine's look at the position d3 left: the knight takes the queen.
            reply = chess.Board(board.fen())
            line = []
            for step in ("Nxh5", "Nf3", "d6"):
                found = reply.parse_san(step)
                line.append(found.uci())
                reply.push(found)
            coach.on_event(
                Analysis(
                    game_id="g",
                    fen=board.fen(),
                    ply=7,
                    lines=[Line(moves=line, depth=12, expected_score=0.75)],
                    decisive=False,
                    only_move=False,
                ),
                17.5,
            )
    verdict = MoveAssessment(
        game_id="g",
        ply=7,
        uci=plays[0].uci(),
        expected_loss=0.45,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.MISS,
    )
    coach.on_event(verdict, 20.0)
    drain(coach)
    directives = [directive_of(sent(provider, i)) for i in range(len(provider.calls))]
    lesson = next(d for d in directives if "The player just played d3" in d)
    assert "no question" in lesson
    assert "one question that sends them to find" not in lesson
    assert "what their move missed or allowed" in lesson
    assert "What d3 allowed, from the engine's line: after d3, Nxh5 wins your queen" in lesson


def test_a_mistakes_lesson_waits_for_what_the_move_allowed() -> None:
    """The verdict comes first, from the best line alone; the analysis of the position the
    move left about a second later, and with it what the move allowed. The app holds the
    lesson for it (at most LESSON_WAIT_S); the game moving on releases it at once."""
    from sidecoach.coach.coach import LESSON_WAIT_S
    from sidecoach.events import Takeback

    coach = Coach(
        ScriptedProvider([]),
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
        lesson_wait_s=LESSON_WAIT_S,
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Bxf7+")
    board.push(move)
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="Bxf7+",
        uci=move.uci(),
        fen_before=SCHOLAR,
        fen_after=board.fen(),
        by_user=True,
    )
    verdict = MoveAssessment(
        game_id="g",
        ply=7,
        uci=move.uci(),
        expected_loss=0.45,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.MISS,
    )
    coach.on_event(played, 10.0)
    coach.on_event(verdict, 10.1)
    assert drain(coach) == [], "held for the analysis of the position it left"
    coach.tick(10.1 + LESSON_WAIT_S / 2)
    assert drain(coach) == [], "still inside the wait"
    coach.tick(10.1 + LESSON_WAIT_S)
    out = drain(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MISTAKE
    # The analysis landing releases it at once.
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 12.0)
    drain(coach)
    coach.on_event(played, 13.0)
    coach.on_event(verdict, 13.1)
    after = Analysis(
        game_id="g",
        fen=board.fen(),
        ply=7,
        lines=[Line(moves=["e8e7"], depth=12, expected_score=0.1)],
        decisive=False,
        only_move=True,
    )
    coach.on_event(after, 13.5)
    out = drain(coach)
    assert [m.trigger for m in out] == [Trigger.POST_MISTAKE]


def test_the_models_markdown_never_reaches_the_voice_or_the_checks() -> None:
    """The parity pilot: "**Rc1** was better", and a closing question wrapped in ** failed
    the question check twice, so the template spoke."""
    from sidecoach.coach.coach import _parse_candidates, plain

    assert (
        plain("Their threat is **...f5**, on `e4`. **Why?**")
        == "Their threat is ...f5, on e4. Why?"
    )
    assert plain("# Plan\n- push the pawn\n* trade") == "Plan\npush the pawn\ntrade"
    assert _parse_candidates("**Rc1** was better. **Do you see why?**") == [
        (1.0, "Rc1 was better. Do you see why?")
    ]


def test_the_coach_never_names_the_engine_unless_asked_about_it() -> None:
    """Round 2: "Stockfish gives White about 0.8 pawns" in seven answers."""
    from sidecoach.coach.coach import Speaker

    request = _after_move_request(SCHOLAR, "Bxf7+", "h5f7")
    speaker = Speaker(ScriptedProvider([]))
    # Noted, never refused: a reply's form is the prompt's to ask for (Tai, 2026-09-28).
    said = "Qxf7 was better; Stockfish liked it more."
    assert "named the engine: say it as the coach who saw the game" in speaker._notes(said, request)
    assert speaker._problems(said, request) == []
    request.user_text = "what does the engine say?"
    assert "named the engine: say it as the coach who saw the game" not in speaker._notes(
        "Qxf7 was better; the engine liked it more.", request
    )


def test_a_legal_move_called_illegal_is_a_fault() -> None:
    """The AI said "Qxf7+ is illegal here" and "Kf7 is illegal because f7 is controlled by
    the King on e8" of legal moves (its own sessions, checked against the boards)."""
    from sidecoach.coach.coach import false_illegal

    board = chess.Board("2r3k1/4bpp1/p3p2p/qp1bn1NQ/8/1P1BB3/P4PPP/3R2K1 w - - 0 20")
    assert false_illegal("No: Qxf7+ is illegal here.", board) == [
        "Qxf7: called illegal, but it is a legal move here"
    ]
    # A claim inside a line is about a later position; a true one is no fault.
    assert false_illegal("After Bh7+ Kh8, Qxf7+ is illegal.", board) == []
    assert false_illegal("Qh4 is not legal: the pawn blocks it.", chess.Board()) == []


def test_a_saved_rating_changes_the_templates_register() -> None:
    # Saving a new rating rewords the system message and the templates alike (the register
    # was kept until the coach was rebuilt).
    coach = Coach(ScriptedProvider([]), rating=(800, 1100))
    assert coach.plain_words and not coach.strong_player
    coach.reword(None, (2000, 2300))
    assert not coach.plain_words and coach.strong_player
    assert "2000" in coach.system
    # The rating is the system message's, never the game's setting.
    start_game(coach, Side.BLACK)
    setting = coach.conversation.messages()[0]["content"]
    assert setting.startswith("# Setting\nYou coach Black.") and "rated" not in setting


def test_a_reply_about_a_moment_the_game_has_left_is_no_turn_of_the_one_now() -> None:
    """A reply still being written when the game moved on -- a takeback, a re-read, the next
    game -- was recorded wherever the game was when it landed: under "Said here" of the
    position taken back to or of the new game, where it made the player's next words a
    follow-up, or the answer to its question (the coach review, 2026-10-01). And a lesson held
    for its analysis was said in the next game."""
    from sidecoach.coach.coach import LESSON_WAIT_S
    from sidecoach.events import Resync, Takeback

    for change in ("takeback", "re-read", "new game"):
        coach, _ = scripted_coach(["[p=1.0] The queen mates on f7. What did you miss?"])
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 7.0)
        mate_missed(coach)  # the lesson, asked of the model
        if change == "takeback":
            coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 12.0)
        elif change == "re-read":
            coach.on_event(Resync(game_id="g", fen=SCHOLAR, reason="re-read"), 12.0)
        else:
            start_game(coach)
        out = drain(coach)  # the reply lands now
        # No reply is dropped (Tai, 2026-10-02): the lesson on a move the board no longer has is
        # shown, in its place, and marked late for the voice -- but it is no turn of the moment
        # now (the deep coach.py review, 2026-10-01).
        assert [m.text for m in out] == ["The queen mates on f7. What did you miss?"], change
        assert coach.spoken[-1].late, change
        assert coach._said_since_move() == [], change
        assert not coach._answering_our_question("the bishop check"), change
        assert out[0].text not in coach.conversation.render(), change
    # A lesson held for the analysis of the position its move left is no exception: when the
    # moment ends first -- a takeback, a re-read, the next game -- it is said in its own moment,
    # shown, late for the voice, and never a turn of the moment after. Dropped, the lesson on a
    # mistake taken back at once was never seen (Tai, 2026-10-02: no reply or lesson is dropped
    # because its moment passed); kept, the next tick said it over the next game's first position.
    lesson = "The queen mates on f7 at once."
    for change in ("takeback", "re-read", "new game"):
        coach = Coach(
            ScriptedProvider([f"[p=1.0] {lesson}"]),
            policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
            lesson_wait_s=LESSON_WAIT_S,
        )
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 7.0)
        mate_missed(coach)
        assert drain(coach) == [], "held for the analysis of the position it left"
        if change == "takeback":
            coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 12.0)
        elif change == "re-read":
            coach.on_event(Resync(game_id="g", fen=SCHOLAR, reason="re-read"), 12.0)
        else:
            start_game(coach)
        out = drain(coach)
        assert [m.text for m in out] == [lesson], change
        assert out[0].trigger is Trigger.POST_MISTAKE and coach.spoken[-1].late, change
        assert lesson not in coach.conversation.render(), change
        assert coach._said_since_move() == [], change
        coach.tick(12.0 + LESSON_WAIT_S)
        assert drain(coach) == [], f"{change}: said once, not again over the next position"


def _bishop_check_then_ke7(coach: Coach) -> chess.Board:
    """At SCHOLAR, the player's Bxf7+ (a miss: Qxf7 mated) judged in quiet mode -- no word
    after it -- and the opponent's Ke7 played, its position not yet analysed."""
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    coach.quiet = True
    mate_missed(coach)
    coach.quiet = False
    board = chess.Board(SCHOLAR)
    board.push_san("Bxf7+")
    before = board.fen()
    board.push_san("Ke7")
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=8,
            san="Ke7",
            uci="e8e7",
            fen_before=before,
            fen_after=board.fen(),
            by_user=False,
        ),
        12.0,
    )
    assert drain(coach) == []
    return board


def test_the_coachs_one_moment_is_no_turn_of_the_conversation() -> None:
    """The coach's "One moment -- still looking at this position." was recorded as its turn, so
    the question it held, answered when the analysis landed, read as a follow-up: "what should
    I have played?" was answered "You have already told them the verdict on it and the better
    move ... do not repeat either" (the coach review, 2026-10-01)."""
    coach, provider = scripted_coach(["[p=1.0] Qxf7# was mate on the spot."])
    board = _bishop_check_then_ke7(coach)
    coach.ask("what should I have played?", 12.5)
    assert [m.trigger for m in drain(coach)] == [Trigger.USER_ASK], "one moment"
    assert coach._said_since_move() == []
    _looked(coach, board, 8, ["h5e5", "d2d3"], 13.0)
    drain(coach)
    directive = directive_of(sent(provider))
    assert "A move they already played: its verdict is in the record" in directive
    assert "# The position they faced, before Bxf7+" in harness_of(sent(provider))
    assert "do not repeat the verdict or the better move" not in directive


def _looked(
    coach: Coach,
    board: chess.Board,
    ply: int,
    ucis: list[str],
    ts: float,
    threats: list[str] | None = None,
) -> None:
    """The engine's look at `board`: `ucis` its lines, best first; `threats` (UCI) what the
    other side would play if the side to move passed."""
    lines = [Line(moves=[u], depth=18, expected_score=0.5 - i / 10) for i, u in enumerate(ucis)]
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            lines=lines,
            decisive=False,
            only_move=False,
            threats=threats or [],
        ),
        ts,
    )


def _judged(coach: Coach, ply: int, uci: str, ts: float) -> None:
    """The engine's verdict on the move at `ply`: its own best."""
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=ply,
            uci=uci,
            expected_loss=0.0,
            severity=Severity.NONE,
            best=[uci],
            label=Label.BEST,
        ),
        ts,
    )


# White to move; Black's queen and bishop threaten Qxf2#.
QH4 = "rnb1k1nr/pppp1ppp/8/2b1p3/2B1P2q/2N5/PPPP1PPP/R1BQK1NR w KQkq - 6 4"


def test_their_threat_after_the_players_move_waits_for_the_position_now() -> None:
    """Right after the player's move, its position not yet read, "what are they threatening?"
    was answered from the position before the move: "They threaten Qxf2#", the threat the
    move had just stopped (the coach review, 2026-10-01). It waits for the engine's look at
    the position on the board, where their threat is their best move."""
    coach = offline_coach()
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=QH4, kind="position"), 0.0
    )
    board = chess.Board(QH4)
    _looked(coach, board, 0, ["d1e2", "d1f3"], 0.5, threats=["h4f2"])
    assert coach.positions[0].facts is not None and coach.positions[0].facts.threats == ["Qxf2#"]
    _played(coach, board, 1, "d1e2", True, 3.0)  # Qe2 guards f2
    _judged(coach, 1, "d1e2", 3.2)
    drain(coach)
    coach.ask("what are they threatening?", 3.5)
    assert not any("Qxf2" in m.text for m in drain(coach))
    _looked(coach, board, 1, ["g8f6", "b8c6"], 4.0)
    said = [m.text for m in drain(coach)]
    assert said and said[-1].startswith("It's their move, and their best is"), said


def test_a_hint_before_the_verdict_is_answered_and_climbs_no_rung_of_the_position_left() -> None:
    """Pressed between the player's move and its verdict, "hint" got "Give me a moment; I am
    still reading this position." with nothing queued -- the promise never kept -- and the
    rung was climbed at the position the player had moved from: the combination's next stuck
    hint skipped the idea (the coach review, 2026-10-01)."""
    coach = offline_coach()
    start_game(coach, Side.BLACK)
    board = play(coach, ["e4"], user_side=Side.BLACK)
    _looked(coach, board, 1, ["e7e5", "c7c5"], 1.5)
    _played(coach, board, 2, "e7e5", True, 3.0)  # the engine's best
    coach.ask("hint", 3.05)
    assert [m.trigger for m in drain(coach)] == [Trigger.USER_ASK], "a moment"
    _judged(coach, 2, "e7e5", 3.1)
    drain(coach)
    _looked(coach, board, 2, ["g1f3", "b1c3"], 3.6)
    kept = [m.text for m in drain(coach) if m.trigger is Trigger.USER_ASK]
    assert len(kept) == 1 and "moment" not in kept[0], kept
    assert coach.positions[1].disclosure.hinted is DisclosureLevel.SILENT
    _played(coach, board, 3, "g1f3", False, 5.0)
    _looked(coach, board, 3, ["b8c6", "g8f6"], 5.5)
    assert [m.level for m in stuck(coach)] == [DisclosureLevel.IDEA]


def test_a_question_held_for_the_engine_is_answered_whatever_the_board_does() -> None:
    """The coach's "One moment" was a promise it did not keep when the board changed under it:
    a takeback neither answered nor dropped the question, a re-read and a new game dropped it,
    and the game's end left it waiting (the coach review, 2026-10-01). It is answered at the
    position the player faces once the coach has read it, and at the game's end at once."""
    from sidecoach.events import Resync, Takeback

    four = chess.Board()
    for san in SCHOLAR_MOVES[:4]:
        four.push_san(san)
    other = chess.Board(QH4)
    for change in ("takeback", "re-read", "new game", "game over"):
        coach = offline_coach()
        play_to_scholar(coach)
        _looked(coach, four, 4, ["d1h5", "g1f3"], 6.5)
        coach.ask("hint", 7.0)  # the position now not read yet
        assert [m.trigger for m in drain(coach)] == [Trigger.USER_ASK], "a moment"
        if change == "takeback":
            coach.on_event(Takeback(game_id="g", plies=2, fen=four.fen()), 8.0)
        elif change == "re-read":
            coach.on_event(Resync(game_id="g", fen=QH4, reason="re-read"), 8.0)
            _looked(coach, other, 0, ["d1e2", "d1f3"], 8.5)
        elif change == "new game":
            coach.on_event(
                GameStarted(game_id="g", user_side=Side.WHITE, start_fen=QH4, kind="position"),
                8.0,
            )
            _looked(coach, other, 0, ["d1e2", "d1f3"], 8.5)
        else:
            coach.on_event(GameEnded(game_id="g", result="1-0", reason="resigned"), 8.0)
        answered = drain(coach)
        assert [m.trigger for m in answered] == [Trigger.USER_ASK], change
        assert "moment" not in answered[0].text, change


def test_the_hints_said_at_a_withdrawn_ply_do_not_count_against_the_next_position_there() -> None:
    """What was said unasked at the plies taken back is not counted against the positions
    played to there after: counted by ply, both hints said after 1.e4 left 1.d4, played in its
    place, with no stuck hint at all (the coach review, 2026-10-01)."""
    from sidecoach.events import Takeback

    coach = offline_coach()
    start_game(coach, Side.BLACK)
    board = play(coach, ["e4"], user_side=Side.BLACK)
    _looked(coach, board, 1, ["e7e5", "c7c5"], 1.5)
    assert [m.level for m in [*stuck(coach), *stuck(coach)]] == [
        DisclosureLevel.IDEA,
        DisclosureLevel.FOCUS,
    ]
    coach.on_event(Takeback(game_id="g", plies=1, fen=chess.STARTING_FEN), 40.0)
    board = chess.Board()
    board.push_san("d4")
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=1,
            san="d4",
            uci="d2d4",
            fen_before=chess.STARTING_FEN,
            fen_after=board.fen(),
            by_user=False,
        ),
        41.0,
    )
    _looked(coach, board, 1, ["d7d5", "g8f6"], 41.5)
    assert [m.level for m in stuck(coach)] == [DisclosureLevel.IDEA]


def test_an_inaccuracy_is_never_called_sound() -> None:
    """An inaccuracy got the word for a sound move -- "Fine, but Qxf7# was a little tighter",
    and the model was told "an inaccuracy: sound, a little behind Qxf7#" -- while asked "was
    that a mistake?" right after, the coach answered it as the costly move it was (the coach
    review, 2026-10-01)."""
    from sidecoach.coach.coach import FINE_TIGHTER
    from tests.coaching import _judged_at_scholar

    fine = tuple(opener.split("{best}")[0] for opener in FINE_TIGHTER)
    for offline in (True, False):
        coach, provider = (offline_coach(), None) if offline else scripted_coach(["", ""])
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 7.0)
        _judged_at_scholar(coach, "Bxf7+", 0.07, Severity.INACCURACY, Label.INACCURACY, 10.0)
        out = drain(coach)
        assert len(out) == 1 and out[0].trigger is Trigger.POST_MOVE, out
        assert "Qxf7" in out[0].text and not out[0].text.startswith(fine), out[0].text
        if provider is not None:
            directive = directive_of(sent(provider, 0))
            assert "which was an inaccuracy" in directive and "sound" not in directive


def test_a_hint_after_the_move_was_given_does_not_promise_it_again() -> None:
    """A fourth "hint" was told "The next hint is the answer" after the move had been given
    (the deep prompt review, 2026-10-01)."""
    coach, provider = scripted_coach(
        ["[p=1.0] Look at f7.", "[p=1.0] Your queen and f7.", "[p=1.0] Qxf7# is mate.", ""]
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    for ts in range(4):
        coach.ask("hint", 8.0 + ts)
        drain(coach)
    directives = [directive_of(sent(provider, i)) for i in range(len(provider.calls))]
    # With the AI on, the ladder says what a hint would be, from what the answers said; once
    # the move was said, the guideline says to give it again, with why it beats worse ones.
    assert "has been given already" not in directives[2], directives[2]
    assert "The move has been given already, in your turns above: it is Qxf7#" in directives[3]
    assert "why it is the move to play instead of something that looks as good" in directives[3]
    assert "Bxf7+ passes up the mate" in directives[3]
    assert coach._current is not None and coach._current.disclosure.level is DisclosureLevel.MOVE


def test_the_second_hint_for_castling_says_what_the_king_goes_after() -> None:
    """The second hint's "the piece to use and what it goes after", the piece being the king
    and the answer O-O, invited "your king goes after their queen" (the deep prompt review)."""
    from sidecoach.events import Analysis, Line

    coach, provider = scripted_coach(["[p=1.0] One.", "[p=1.0] Two."])
    start_game(coach)
    board = play(coach, ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"])
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=6,
            lines=[Line(moves=["e1g1", "g8f6"], depth=12, expected_score=0.55)],
            decisive=False,
            only_move=False,
            tags=[],
        ),
        7.0,
    )
    drain(coach)
    coach.ask("hint", 8.0)
    drain(coach)
    coach.ask("hint", 9.0)
    drain(coach)
    directive = directive_of(sent(provider, len(provider.calls) - 1))
    assert "Here the move is castling" in directive, directive


def test_a_move_named_on_their_turn_is_not_called_unrated() -> None:
    """On the opponent's move a move the player names is never examined, yet the directive
    said "when the engine does not rate their move, say it is not the strongest" (the deep
    prompt review, 2026-10-01)."""
    coach, provider = scripted_coach(["[p=1.0] Fine.", "[p=1.0] Fine."])
    start_game(coach, Side.BLACK)
    play(coach, SCHOLAR_MOVES, user_side=Side.BLACK)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    coach.ask("what about Qxf7?", 8.0)
    drain(coach)
    directive = directive_of(sent(provider, len(provider.calls) - 1))
    assert "say it is not the strongest" not in directive, directive
    assert "rate one only from the engine's word on it" in directive
