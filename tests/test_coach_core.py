"""The coach's core logic, from the deep review of coach.py (2026-10-01): what a takeback
leaves, what a combination is, and how the player's words and the model's replies are kept
to their moment. Logic only: time is the policy's `ts` and the Speaker's own hooks."""

from __future__ import annotations

import chess

from sidecoach.coach.coach import Coach, unlock_for
from sidecoach.coach.disclosure import Unlock
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    Line,
    Takeback,
    Trigger,
)
from tests.coaching import (
    SCHOLAR,
    drain,
    mate_analysis,
    mate_missed,
    offline_coach,
    play_to_scholar,
    start_game,
    stuck,
)
from tests.test_coach import _judged, _played


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


def test_the_same_plan_across_best_moves_is_one_combination() -> None:
    """One plan the facts name, kept across best moves: said at its first move, not again at
    the next -- each level once per combination."""
    coach = _two_hints_at_the_start_then_the_next_positions(
        {
            0: "finishing your development",
            2: "finishing your development",
            4: "finishing your development",
        }
    )
    assert coach._ladder(coach.positions[4]) is DisclosureLevel.FOCUS
    assert stuck(coach) == []


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
