"""The coach's core logic, from the deep review of coach.py (2026-10-01): what a takeback
leaves, what a combination is, and how the player's words and the model's replies are kept
to their moment. Logic only: time is the policy's `ts` and the Speaker's own hooks."""

from __future__ import annotations

from collections.abc import Callable

import chess
import pytest

from sidecoach.coach.coach import Coach, unlock_for
from sidecoach.coach.disclosure import Unlock
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameStarted,
    Line,
    Side,
    Takeback,
    Trigger,
)
from tests.coaching import (
    SCHOLAR,
    directive_of,
    drain,
    harness_of,
    mate_analysis,
    mate_missed,
    mate_played,
    offline_coach,
    play_to_scholar,
    sent,
    start_game,
    stuck,
)
from tests.fakes import ScriptedProvider
from tests.test_alternatives import load
from tests.test_coach import _judged, _looked, _played


def _levels(messages: list) -> list[DisclosureLevel]:
    return [m.level for m in messages]


# 1. A takeback leaves the position at the hints it had


def test_a_takeback_leaves_the_position_at_the_hints_it_had_before_the_withdrawn_move() -> None:
    """The move played from a position opens its line (`MOVE_PLAYED`); taken back, the position
    stayed at the line: no stuck hint, and the first hint asked gave the answer. The line
    belonged to the move withdrawn: the position is the player's again at the hints it had
    had, none said twice and none lower after a higher (Tai, 2026-09-29)."""
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]
    mate_missed(coach, 20.0)  # Bxf7+, a wrong move
    drain(coach)
    assert coach.positions[6].disclosure.level is DisclosureLevel.LINE
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 30.0)
    position = coach.positions[6]
    assert position is coach._current
    assert position.disclosure.level is DisclosureLevel.IDEA, "the idea was said, the rest not"
    # The next hint unasked climbs from there: the focus, not nothing and not the line.
    assert _levels(stuck(coach)) == [DisclosureLevel.FOCUS]


def test_a_takeback_with_no_hint_said_leaves_the_first_hint_the_idea() -> None:
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_missed(coach, 20.0)
    drain(coach)
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 30.0)
    assert coach.positions[6].disclosure.level is DisclosureLevel.SILENT
    coach.ask("hint", 31.0)
    answers = drain(coach)
    assert [m.trigger for m in answers] == [Trigger.USER_ASK]
    assert answers[0].level is DisclosureLevel.IDEA, "the first hint is the idea, not the answer"


def test_a_takeback_keeps_what_the_player_asked_for_before_the_move() -> None:
    """A level said on request is said: the move asked for before the wrong move stays the
    level after it is taken back, and the idea is not said again below it."""
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    coach.ask("just tell me the move", 8.0)
    drain(coach)
    assert coach.positions[6].disclosure.level is DisclosureLevel.MOVE
    mate_missed(coach, 20.0)
    drain(coach)
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 30.0)
    assert coach.positions[6].disclosure.level is DisclosureLevel.MOVE


# 2. Ordinary words open nothing: the move and the line only on an explicit ask


def test_ordinary_words_open_neither_the_move_nor_the_line_nor_a_hint() -> None:
    """ "Which move should I avoid?", "the answer is Nf3, right?" and "what do I do about
    that?" gave the move; "what's next for me after castling" the line; "I'm not stuck, just
    thinking" climbed a rung. Only explicit asks unlock; the rest goes to the model at the
    level in force (the deep coach.py review, 2026-10-01)."""
    board = chess.Board(SCHOLAR)
    for said in (
        "which move should I avoid?",
        "which move is the opponent threatening?",
        "the answer is Nf3, right?",
        "what do I do about their threat?",
        "what should I do about the bishop?",
        "what's next for me after castling?",
        "what comes next in the plan after I castle?",
        "I'm not stuck, just thinking",
        "I am not really stuck, I am looking at the queen",
        "I don't feel stuck",
    ):
        assert unlock_for(said, board, ["Qxf7#"]) is None, said
    for said, unlock in (
        ("show me the move", Unlock.ASK_MOVE),
        ("tell me the move", Unlock.ASK_MOVE),
        ("what's the best move?", Unlock.ASK_MOVE),
        ("what should I play?", Unlock.ASK_MOVE),
        ("what's the line?", Unlock.ASK_LINE),
        ("what comes next?", Unlock.ASK_LINE),
        ("I'm stuck", Unlock.HINT),
        ("give me a hint", Unlock.HINT),
    ):
        assert unlock_for(said, board, ["Qxf7#"]) is unlock, said


# 4. A reply is the moment it was asked in


def test_a_reply_checked_while_a_new_game_starts_is_still_the_old_games() -> None:
    """The words were checked on a worker, and the epoch and the game id read when the
    request was built: a new game starting meanwhile filed the answer under itself, a turn of
    the new game's dialogue (the deep coach.py review, 2026-10-01). Taken at ask()."""
    provider = ScriptedProvider(["Yes, Qxf7 is mate."])
    coach = Coach(
        provider,
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
        claims=lambda text, board, facts: [],
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    held: list[Callable[[], None]] = []
    coach.spawn = lambda work, name: held.append(work)
    coach.ask("Is Qxf7 mate?", 8.0)
    assert len(held) == 1, "the claims check is on a worker"
    coach.on_event(
        GameStarted(game_id="h", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 9.0
    )
    held[0]()
    answers = drain(coach)
    assert [m.text for m in answers] == ["Yes, Qxf7 is mate."]
    assert answers[0].game_id == "g", "the game it was asked in"
    assert [e for e in coach.conversation.entries if e.kind == "coach"] == [], (
        "no turn of the new game"
    )


# "That move" after a bot's instant reply is the player's own


def _asked_after_the_reply(reply_ts: float) -> tuple[Coach, ScriptedProvider]:
    """Bxf7+ from SCHOLAR (a miss), answered by Kxf7 at `reply_ts`, then "why was that bad?"."""
    provider = ScriptedProvider(
        ["Qxf7 was mate on the spot.", "It gave the bishop away for a pawn."]
    )
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_missed(coach, 10.0)
    drain(coach)
    board = chess.Board(SCHOLAR)
    board.push_san("Bxf7+")
    _played(coach, board, 8, "e8f7", False, reply_ts)
    _looked(coach, board, 8, ["h5f3", "h5h4"], reply_ts + 0.5)
    coach.ask("why was that bad?", reply_ts + 2.0)
    drain(coach)
    return coach, provider


def test_what_that_means_after_a_reply_is_the_models_to_decide_from_both_positions() -> None:
    """The opponent's reply was last on the board and "why was that bad?" went to the generic
    branch, with no sheet of the position the player faced: the model critiqued the
    opponent's move (Tai, 2026-10-01). Not decided by a threshold of two seconds in code
    any more (Tai, 2026-10-02: "why does instant mean 2 seconds"): the model gets the sheet
    of the position the player faced and the one after the reply, and how long the reply
    took, in the guidelines, fast or slow, and decides what "that" is."""
    for reply_ts, took in ((10.5, "0.5 seconds"), (40.0, "30.0 seconds")):
        coach, provider = _asked_after_the_reply(reply_ts)
        harness = harness_of(sent(provider))
        assert "The position they faced, before Bxf7+" in harness, reply_ts
        assert "# Position now" in harness
        directive = directive_of(sent(provider))
        assert f"The opponent's last move, Kxf7, came {took} after the player's Bxf7+" in (
            directive
        ), reply_ts
        assert '"That", "it" or "that move" may mean their own last move' in directive
        assert "They ask about the move they already played" not in directive
        assert coach.user_to_move


# The best moves, and a hint asked once the move was given: real engine analyses
# (tests/fixtures/alternatives)


def _coach_at(name: str, replies: list[str] | None = None) -> tuple[Coach, ScriptedProvider]:
    """The player to move in the fixture's position, the engine's look at it in."""
    board, analysis = load(name)
    provider = ScriptedProvider(replies or [])
    coach = (
        Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
        if replies is not None
        else offline_coach()
    )
    side = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
    coach.on_event(
        GameStarted(game_id="g", user_side=side, start_fen=board.fen(), kind="position"), 0.0
    )
    coach.on_event(analysis, 1.0)
    return coach, provider


def test_asked_for_the_best_moves_every_move_about_as_good_is_named() -> None:
    """ "what are the best moves?", "any other good moves?", "what else works?" and "which moves
    are fine here?" were answered with line 1, or with a hint (Tai, 2026-10-01). Graded as the
    move labels grade them (Tai, 2026-10-02): Bc5 and Nf6 about as good as each other (Best,
    Excellent), Be7 and d6 good, a little behind (Good) -- each named for what it is."""
    for said in (
        "what are the best moves?",
        "any other good moves?",
        "what else works?",
        "which moves are fine here?",
    ):
        assert unlock_for(said, chess.Board(load("italian")[1].fen), []) is Unlock.ASK_MOVE, said
        coach, _ = _coach_at("italian")
        coach.ask(said, 2.0)
        answer = drain(coach)
        assert len(answer) == 1, said
        assert answer[0].text == (
            "Bc5 and Nf6 are both fine here -- about equally good. "
            "Be7 and d6 are good too, a little behind."
        ), said
        assert answer[0].level is DisclosureLevel.MOVE


def test_the_model_is_told_to_read_every_line_and_to_name_the_moves_that_are_fine() -> None:
    coach, provider = _coach_at("italian", ["Bc5 and Nf6 are both fine."])
    coach.ask("what are the best moves?", 2.0)
    drain(coach)
    directive = directive_of(sent(provider))
    # The AI's guidelines, whatever the words: every line, every move about level named, and the
    # moves about as good as the facts list them (no keyword picks this).
    assert "read all of the engine lines" in directive
    assert "The moves about equally good here: Bc5, Nf6." in directive
    assert "Good too, a little behind the best: Be7, d6." in directive
    assert "name every move about level with the best" in directive
    assert "no percentages, scores or move numbers" in directive
    harness = harness_of(sent(provider))
    for san in ("Bc5", "Nf6", "Be7"):
        assert san in harness.split("## Engine lines", 1)[1].split("\n\n", 1)[0]


def test_asked_whether_a_move_is_also_good_the_model_gets_the_fine_moves() -> None:
    coach, provider = _coach_at("italian", ["Yes, Be7 is good too."])
    coach.ask("is Be7 also good?", 2.0)
    drain(coach)
    directive = directive_of(sent(provider))
    assert "or whether another move works too" in directive
    assert "The moves about equally good here: Bc5, Nf6." in directive
    assert "Good too, a little behind the best: Be7, d6." in directive
    assert "read all of the engine lines" in directive
    # Offline, at the move level, the template says what it is: good, a little behind.
    coach, _ = _coach_at("italian")
    coach.ask("what are the best moves?", 2.0)
    drain(coach)
    coach.ask("is Be7 also good?", 3.0)
    [said] = drain(coach)
    assert said.text == "Yes, Be7 is good too, a little behind Bc5.", said.text


def test_the_claim_check_says_a_good_move_is_good_and_not_the_best() -> None:
    """Be7 is 0.029 behind the best line, graded Good: "is Be7 a good move?" was refused as not
    among the engine's best moves while the same coach was asked to name it. It is a good move
    -- and called the best, it is not (Tai, 2026-10-02: the grading bands, one source)."""
    from sidecoach.coach.claims import Claim, ClaimKind, verify_claims
    from sidecoach.coach.facts import facts_from_analysis

    board, analysis = load("italian")
    facts = facts_from_analysis(board, analysis, last_move=None, user=board.turn)

    def verdict(move: str, text: str) -> tuple[str, str]:
        [v] = verify_claims([Claim(ClaimKind.BEST, None, None, move, text)], board, facts=facts)
        return v.result, v.reason

    assert verdict("Nf6", "Nf6 is a good move")[0] == "true"
    good = verdict("Be7", "Be7 is a good move")
    assert good == (
        "true",
        "Be7 is a good move here, a little behind the engine's best (Bc5 and Nf6).",
    )
    best = verdict("Be7", "Be7 is the best move")
    assert best[0] == "false" and "is a good move here, a little behind" in best[1]
    assert verdict("d6", "d6 is a good move")[0] == "true"
    assert verdict("h6", "h6 is a good move")[0] == "false", "an inaccuracy is no good move"

    class Lines:
        """The engine, asked when the facts are not this position's: the fixture's lines."""

        def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
            return list(reversed(analysis.lines))  # in any order: graded from the best

    claim = Claim(ClaimKind.BEST, None, None, "Be7", "Be7 is a good move")
    [v] = verify_claims([claim], board, engine=Lines())
    assert (v.result, v.reason) == good


def test_one_clear_best_move_is_named_with_what_goes_wrong_with_the_others() -> None:
    coach, _ = _coach_at("scholar")
    coach.ask("what are the best moves?", 2.0)
    answer = drain(coach)
    assert answer[0].text == (
        "Qxf7# is the move here. Qd1 is clearly worse: it passes up the mate. "
        "Qe2 is clearly worse: it passes up the mate."
    )
    coach, provider = _coach_at("scholar", ["Qxf7 is mate; the others pass it up."])
    coach.ask("what are the best moves?", 2.0)
    drain(coach)
    directive = directive_of(sent(provider))
    assert "The moves about equally good here" not in directive, "one clear best"
    assert "Clearly worse here, and what goes wrong with them: Qd1 passes up the mate" in (
        directive
    )
    assert "Qe2 passes up the mate" in directive


def test_the_best_move_asked_in_the_singular_names_the_near_equal_ones_too() -> None:
    coach, _ = _coach_at("italian")
    coach.ask("what's the best move?", 2.0)
    text = drain(coach)[0].text
    assert "Bc5" in text and "Nf6 is about as good." in text and "Be7" not in text
    coach, provider = _coach_at("italian", ["Bc5."])
    coach.ask("what's the best move?", 2.0)
    drain(coach)
    assert "The moves about equally good here: Bc5, Nf6." in directive_of(sent(provider))
    coach, provider = _coach_at("scholar", ["Qxf7 mates."])
    coach.ask("what's the best move?", 2.0)
    drain(coach)
    assert "about equally good here" not in directive_of(sent(provider)), "one clear best"


def test_a_hint_asked_after_the_move_was_given_says_why_it_beats_what_is_worse() -> None:
    """The fourth hint, with the move given by the third, was "the move again": it is now the
    move with why it is the one to play instead of the moves that are clearly worse, as the
    board says it (Tai, 2026-10-01)."""
    coach, _ = _coach_at("scholar")
    said = []
    for n in range(4):
        coach.ask("hint", 2.0 + n)
        said.append(drain(coach)[0])
    assert [m.level for m in said] == [
        DisclosureLevel.IDEA,
        DisclosureLevel.FOCUS,
        DisclosureLevel.MOVE,
        DisclosureLevel.MOVE,
    ]
    assert "Qxf7" in said[2].text and "clearly worse" not in said[2].text
    assert "Qd1 is clearly worse: it passes up the mate." in said[3].text
    # And to the model.
    # And to the model, as a guideline: said by what the reply gave (the third names the move).
    replies = ["a", "b", "Qxf7 is mate.", "Qxf7 is mate; Qd1 passes it up."]
    coach, provider = _coach_at("scholar", replies)
    for n in range(4):
        coach.ask("hint", 2.0 + n)
        drain(coach)
    directive = directive_of(sent(provider))
    assert "has been given already, in your turns above: it is Qxf7#" in directive
    assert "Clearly worse here, and what goes wrong with them: Qd1 passes up the mate" in directive
    assert "instead of something that looks as good but is clearly worse" in directive
    assert "has been given already" not in directive_of(sent(provider, 2))


# 5, 13. A lesson on a withdrawn move is shown all the same, and is no turn of the game now


def test_a_lesson_on_a_withdrawn_move_is_shown_in_its_place_not_recorded_not_spoken() -> None:
    """The deep coach.py review (2026-10-01) dropped the lesson on the move taken back. Tai,
    2026-10-02: "There is no need to drop any replies, the text message should come out in the
    same correct order. The only thing that might be good is to interrupt the tts": it is
    shown, marked late (the voice does not say it), and is no turn of the game now."""
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_missed(coach, 10.0)  # the lesson is queued
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 12.0)
    shown = drain(coach)
    assert len(shown) == 1 and shown[0].trigger is Trigger.POST_MISTAKE
    assert coach.spoken[-1].late, "for the voice: not spoken"
    assert coach._said_since_move() == [], "no turn of the moment now"


def test_a_late_lesson_that_ends_with_a_question_has_asked_it() -> None:
    """ "What did you expect?" is marked asked when its lesson is queued. The lesson, shown late
    after a takeback, still asks it: the player reads it, so it was asked (no reply is dropped,
    Tai, 2026-10-02)."""
    from sidecoach.coach.coach import ELICIT_EVERY

    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    coach._mistakes = ELICIT_EVERY - 1
    mate_missed(coach, 10.0)
    assert coach._elicit_asked is not None, "asked, in the lesson queued"
    coach.on_event(Takeback(game_id="g", plies=1, fen=SCHOLAR), 12.0)
    shown = drain(coach)
    assert shown and shown[0].text.endswith("?")
    assert coach._elicit_asked is not None, "the lesson was shown: so was the question"


# 6. A question does not delete the lesson on a mistake or the review


def _requests(coach: Coach, *triggers: Trigger) -> list:
    from sidecoach.coach.coach import _Request

    position = coach.positions[6]
    assert position.facts is not None
    return [
        _Request(
            trigger=trigger,
            level=DisclosureLevel.IDEA,
            system="",
            messages=[{"role": "user", "content": "x"}],
            facts=position.facts,
            board=position.board,
            fallback=trigger.value,
            ply=6,
            game_id="g",
        )
        for trigger in triggers
    ]


def test_a_question_drops_nothing_queued() -> None:
    """Any question deleted the queued lessons and the review (the noise budget was charged
    for a lesson never said), and then the remark and the nudge still: never seen (Tai,
    2026-10-02: "no need to drop any replies, the text message should come out in the same
    correct order"). The hint said unasked keeps its place ahead of the answer -- said first,
    counted by the ladder, and after the answer it would be a lower rung after a higher; the
    remark, the lesson and the review wait for the answer and go after it, in their order."""
    from sidecoach.coach.coach import Speaker
    from sidecoach.llm.offline import OfflineProvider

    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    speaker = Speaker(OfflineProvider())
    for request in _requests(
        coach,
        Trigger.STUCK,
        Trigger.POST_MISTAKE,
        Trigger.REVIEW,
        Trigger.POST_MOVE,
        Trigger.USER_ASK,
    ):
        speaker.submit(request)
    assert speaker.pending == 5
    speaker.run_pending()
    assert [s.message.trigger for s in speaker.poll()] == [
        Trigger.STUCK,
        Trigger.USER_ASK,
        Trigger.POST_MISTAKE,
        Trigger.REVIEW,
        Trigger.POST_MOVE,
    ]


def test_a_stuck_hint_a_question_came_behind_is_shown_first_and_not_spoken() -> None:
    """Tai, 2026-10-02 (T8): "the stuck hint does not need to be voiced but the order should
    still be preserved". The hint waiting when the question comes is shown first, in its place,
    and the voice goes to the answer: its sink reads it as not to be spoken, and the panel is
    told so (`Spoken.unvoiced`). A hint with no question behind it is spoken as ever."""
    coach = Coach(
        ScriptedProvider(["Look at f7.", "Qxf7 is mate."]),
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
    )
    sinks: dict[Trigger, Callable[[], bool]] = {}

    def stream_sink(trigger: Trigger, ply: int | None, moved_on: Callable[[], bool]) -> None:
        sinks[trigger] = moved_on  # what the voice will ask when the first sentence comes

    coach.stream_sink = stream_sink  # type: ignore[assignment]
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    since = coach.policy.stuck_from
    assert since is not None
    coach.tick(since + coach.policy.stuck_threshold())  # the stuck hint, queued, not yet run
    assert sinks[Trigger.STUCK]() is False, "nothing behind it yet: spoken"
    coach.ask("what should I play?", since + coach.policy.stuck_threshold() + 1)
    assert sinks[Trigger.STUCK]() is True, "a question came behind it: not spoken"
    assert sinks[Trigger.USER_ASK]() is False, "the answer is spoken"
    out = drain(coach)
    assert [m.trigger for m in out] == [Trigger.STUCK, Trigger.USER_ASK], "in order, none dropped"
    hint, answer = coach.spoken[-2:]
    assert hint.unvoiced and not answer.unvoiced


# 7. A failure while answering is counted, and the question is answered


def test_a_failure_building_a_reply_is_counted_and_the_template_is_said() -> None:
    """A speaker-thread error left the question with no answer and no template; run_pending
    did not catch, so no test could see it."""
    from sidecoach.coach.coach import Speaker
    from sidecoach.llm.offline import OfflineProvider

    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    speaker = Speaker(OfflineProvider())

    def boom(request: object) -> None:
        raise RuntimeError("boom")

    speaker.speak_now = boom  # type: ignore[method-assign]
    speaker.submit(_requests(coach, Trigger.USER_ASK)[0])
    speaker.run_pending()
    said = speaker.poll()
    assert [s.message.text for s in said] == ["user_ask"], "the template"
    assert speaker.failures == 1 and speaker.last_error == "reply: RuntimeError: boom"
    assert said[0].problems == ["the reply failed: RuntimeError: boom"]


def test_a_failure_on_the_claims_thread_is_counted_and_the_question_answered() -> None:
    provider = ScriptedProvider(["Yes, Qxf7 is mate."])
    coach = Coach(
        provider,
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
        claims=lambda text, board, facts: [],
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    held: list[Callable[[], None]] = []
    coach.spawn = lambda work, name: held.append(work)

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    coach._speak = boom  # type: ignore[method-assign]
    coach.ask("Is Qxf7 mate?", 8.0)
    held[0]()  # the worker's end of it: nothing raised
    assert coach.speaker.failures == 1
    assert coach.speaker.last_error == "claims check: RuntimeError: boom"
    answer = drain(coach)
    assert [m.trigger for m in answer] == [Trigger.USER_ASK]
    assert "could not work that out" in answer[0].text


def test_a_failed_claims_check_is_counted_and_the_question_answered_without_it() -> None:
    """The claims check's own failure was swallowed: the question went on unchecked, with no
    count and nothing in the log."""
    provider = ScriptedProvider(["Yes, Qxf7 is mate."])

    def broken(text: str, board: object, facts: object) -> list[str]:
        raise RuntimeError("no reader")

    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), claims=broken
    )
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    held: list[Callable[[], None]] = []
    coach.spawn = lambda work, name: held.append(work)
    coach.ask("Is Qxf7 mate?", 8.0)
    held[0]()
    assert coach.speaker.failures == 1
    assert coach.speaker.last_error == "claims check: RuntimeError: no reader"
    assert [m.text for m in drain(coach)] == ["Yes, Qxf7 is mate."], "answered all the same"


# 12. The summary's and the named analysis' failures are counted


def test_a_failed_summary_is_counted_like_a_failed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import sidecoach.coach.coach as module

    coach = offline_coach()

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("no model")

    monkeypatch.setattr(module, "summarise", boom)
    coach._summarise()
    assert coach.speaker.failures == 1
    assert coach.speaker.last_error == "summary: RuntimeError: no model"


# 8. The pattern and the practice are unprompted too


def test_the_pattern_and_the_practice_keep_quiet_in_quiet_mode_and_in_bullet() -> None:
    from sidecoach.events import PuzzleRef, Recommendation
    from sidecoach.profile import causes

    def evidence(coach: Coach) -> None:
        from sidecoach.events import ProfileEvidence

        for ply in (9, 13, 17):
            coach.on_event(
                ProfileEvidence(
                    game_id="g",
                    ply=ply,
                    phase="middlegame",
                    expected_loss=0.2,
                    cause=causes.MISSED_THREAT,
                ),
                float(ply),
            )

    recommended = Recommendation(
        game_id="g",
        motif="fork",
        motif_words="fork",
        reason="forks cost you most",
        level=1500,
        puzzles=[PuzzleRef(id="p1", url="https://example.org/p1", rating=1500)],
    )
    for how in ("quiet", "bullet"):
        coach = offline_coach()
        start_game(coach)
        if how == "quiet":
            coach.quiet = True
        else:
            from sidecoach.coach.policy import TimeControl

            coach.policy.time_control = TimeControl.BULLET
        evidence(coach)
        coach.on_event(recommended, 20.0)
        assert drain(coach) == [], how
    coach = offline_coach()
    start_game(coach)
    evidence(coach)
    coach.on_event(recommended, 20.0)
    said = [m.text for m in drain(coach)]
    assert len(said) == 2 and "three times" in said[0] and "For practice" in said[1]


# 9. Two questions while the position is read: both are answered


def test_two_questions_held_for_the_engine_are_both_answered_in_order() -> None:
    coach = offline_coach()
    start_game(coach)
    coach.ask("what's the best move?", 1.0)
    coach.ask("what is my opponent threatening?", 2.0)
    held = drain(coach)
    assert [m.text for m in held] == ["One moment -- still looking at this position."]
    board, analysis = load("scholar")
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=board.fen(), kind="position"), 3.0
    )
    coach.on_event(analysis, 4.0)
    answered = drain(coach)
    assert len(answered) == 2, "the last question did not overwrite the first"
    assert "Qxf7" in answered[0].text


# 10. What the log keeps is bounded


def test_the_log_keeps_the_prompts_of_the_last_replies_only() -> None:
    from sidecoach.coach.coach import SPOKEN_PROMPTS_KEPT, ModelCall, Spoken
    from sidecoach.events import CoachMessage

    coach = offline_coach()
    for n in range(SPOKEN_PROMPTS_KEPT + 10):
        coach.speaker._results.put(
            Spoken(
                message=CoachMessage(
                    text=f"m{n}", level=DisclosureLevel.IDEA, trigger=Trigger.USER_ASK
                ),
                level=DisclosureLevel.IDEA,
                problems=[],
                attempts=1,
                calls=[ModelCall("raw", {}, 1, 1, prompt=f"prompt {n}")],
            )
        )
    assert len(coach.poll()) == SPOKEN_PROMPTS_KEPT + 10
    assert len(coach.spoken) == SPOKEN_PROMPTS_KEPT + 10, "the list itself is the panel's index"
    kept = [s.calls[0].prompt for s in coach.spoken]
    assert kept[:10] == [""] * 10 and kept[10] == "prompt 10" and kept[-1] == "prompt 29"


def test_the_claim_readings_kept_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    from sidecoach.coach import draft_claims
    from sidecoach.coach.coach import CLAIMS_KEPT, Speaker
    from sidecoach.llm.offline import OfflineProvider

    monkeypatch.setattr(draft_claims, "extract", lambda *args, **kwargs: [])
    monkeypatch.setattr(draft_claims, "verify", lambda *args, **kwargs: [])
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    speaker = Speaker(OfflineProvider(), claim_reader=OfflineProvider())
    request = _requests(coach, Trigger.USER_ASK)[0]
    for n in range(CLAIMS_KEPT + 5):
        speaker._claim_problems(f"reply number {n}", request)
    assert len(speaker._claims_seen) == CLAIMS_KEPT
    assert (request.board.fen(), "reply number 0") not in speaker._claims_seen


# 14. The review's sentence counts what it says


def test_the_review_sentence_names_every_label_it_counts() -> None:
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6), 7.0)
    drain(coach)
    mate_played(coach)  # a best move
    assert coach._review_template().startswith("1 of your 1 moves were best.")


# 3. A combination is a forcing sequence or one plan, not any run of best moves


def _looked_line(coach: Coach, board: chess.Board, ply: int, moves: list[str], ts: float) -> None:
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            lines=[Line(moves=moves, depth=18, expected_score=0.5)],
            decisive=False,
            only_move=False,
        ),
        ts,
    )


def _two_hints_at_the_start_then_the_next_positions(plans: dict[int, str | None]) -> Coach:
    coach = offline_coach()
    start_game(coach)
    board = chess.Board()
    steps = list(enumerate(["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"]))
    for ply, uci in steps:
        mine = ply % 2 == 0
        if mine:
            _looked_line(coach, board, ply, [uci], 100.0 * ply)
            facts = coach.positions[ply].facts
            assert facts is not None
            facts.plan = plans.get(ply)
            facts.theme = None
            facts.nature = "quiet -- no tactic to find; the move improves your position"
            if ply == 0:
                assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]
                assert _levels(stuck(coach)) == [DisclosureLevel.FOCUS]
        _played(coach, board, ply + 1, uci, mine, 100.0 * ply + 50)
        if mine:
            _judged(coach, ply + 1, uci, 100.0 * ply + 51)
        drain(coach)
        if ply == 3:
            break
    # The player faces ply 4: the engine reads it.
    _looked_line(coach, board, 4, ["f1b5"], 500.0)
    facts = coach.positions[4].facts
    assert facts is not None
    facts.plan = plans.get(4)
    facts.nature = "quiet -- no tactic to find; the move improves your position"
    return coach


def test_a_best_move_on_another_plan_is_a_new_combination_and_gets_its_stuck_hint() -> None:
    """The hints said at the game's first move kept every stuck hint after it from a player
    who played best moves and was then stuck: any run of best moves was one combination
    (the deep coach.py review, 2026-10-01). A combination is one plan the facts name, or a
    forcing sequence, not any best moves."""
    coach = _two_hints_at_the_start_then_the_next_positions(
        {0: "fighting for the centre", 2: "finishing your development", 4: "getting your king safe"}
    )
    # Ply 2's plan was another: its own hints were given then; now at ply 4, on a third plan,
    # a stuck hint comes.
    assert coach._ladder(coach.positions[4]) is DisclosureLevel.SILENT
    assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]


def test_the_same_plan_across_best_moves_is_still_another_position() -> None:
    """A plan that holds for several quiet moves is no combination: the opponent moved, the
    position is another, and a player stuck in it gets a hint (Tai, 2026-10-02: "whatever
    the hint is should be dependent on the position ... you just won't be able to get any
    hint is ridiculous")."""
    coach = _two_hints_at_the_start_then_the_next_positions(
        {
            0: "finishing your development",
            2: "finishing your development",
            4: "finishing your development",
        }
    )
    assert coach._ladder(coach.positions[4]) is DisclosureLevel.SILENT
    assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]


def test_a_forcing_line_followed_is_one_combination() -> None:
    """A forcing sequence: the engine's line at the first move -- the move, the reply, the
    next move -- is played, and the next stuck hint is not the idea again."""
    coach = offline_coach()
    start_game(coach)
    board = chess.Board()
    _looked_line(coach, board, 0, ["e2e4", "e7e5", "g1f3"], 0.0)
    first = coach.positions[0].facts
    assert first is not None
    first.nature = "forcing -- the answer is a check, a capture or a promotion"
    first.plan = first.theme = None
    assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]
    assert _levels(stuck(coach)) == [DisclosureLevel.FOCUS]
    _played(coach, board, 1, "e2e4", True, 50.0)
    _judged(coach, 1, "e2e4", 50.5)
    _played(coach, board, 2, "e7e5", False, 51.0)
    drain(coach)
    _looked_line(coach, board, 2, ["g1f3"], 52.0)
    second = coach.positions[2].facts
    assert second is not None
    second.plan = second.theme = None
    assert coach._ladder(coach.positions[2]) is DisclosureLevel.FOCUS
    assert stuck(coach) == []


def test_a_forcing_line_left_is_a_new_combination() -> None:
    """The opponent's reply was not the line's: the sequence is over, and the next position
    is a new one with its own hints."""
    coach = offline_coach()
    start_game(coach)
    board = chess.Board()
    _looked_line(coach, board, 0, ["e2e4", "e7e5", "g1f3"], 0.0)
    first = coach.positions[0].facts
    assert first is not None
    first.nature = "forcing -- the answer is a check, a capture or a promotion"
    first.plan = first.theme = None
    stuck(coach)
    stuck(coach)
    _played(coach, board, 1, "e2e4", True, 50.0)
    _judged(coach, 1, "e2e4", 50.5)
    _played(coach, board, 2, "c7c5", False, 51.0)  # not the line's reply
    drain(coach)
    _looked_line(coach, board, 2, ["g1f3"], 52.0)
    second = coach.positions[2].facts
    assert second is not None
    second.plan = second.theme = None
    assert coach._ladder(coach.positions[2]) is DisclosureLevel.SILENT
    assert _levels(stuck(coach)) == [DisclosureLevel.IDEA]
