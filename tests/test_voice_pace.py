"""The voice follows the player's pace (`voice/pace.py`): what is said, held to a fast line's
end, or left to the text, in the sessions `tools/voice_sim.py` tuned it on (Tai, 2026-10-03:
"simulate what you would hear ... this has to work with the user playing slow, playing fast,
finding the moves, not finding the moves"). Pure logic on a clock the test moves."""

from __future__ import annotations

from sidecoach.voice.pace import (
    HOLD,
    LESSON_FRESH_S,
    LINE_END_S,
    LINE_WORD_FRESH_S,
    PRAISE_FRESH_S,
    SAY,
    TEXT,
    PacePlan,
)
from tests.fakes import ManualClock


def plan_at() -> tuple[PacePlan, ManualClock]:
    clock = ManualClock(start=1000.0)
    plan = PacePlan(clock=clock.monotonic)
    plan.new_position()
    return plan, clock


def said(plan: PacePlan) -> list[str]:
    """Everything the plan hands the voice now, in order."""
    out = []
    while (line := plan.next_line()) is not None:
        out.append(line.text)
    return out


def test_slow_play_hears_every_word_whole() -> None:
    plan, clock = plan_at()
    clock.advance(20)
    plan.player_moved(1, 20.0)
    clock.advance(1.6)
    assert plan.offer(1, "post_move", 0, 0, "Excellent: the only move.", streamed=True) == SAY
    assert plan.offer(1, "post_move", 0, 1, "It pulls the defender away.", streamed=True) == SAY
    assert said(plan) == ["Excellent: the only move.", "It pulls the defender away."]


def test_a_fast_line_is_quiet_then_one_word_on_the_find_at_its_end() -> None:
    """P2: the find after 30 s, then the line played at 1.5 s a move. The model's word on the
    find comes after the next move: held, and said once the line is done -- its first sentence,
    made ahead. The words on the quick moves are text."""
    plan, clock = plan_at()
    clock.advance(30)
    plan.player_moved(1, 30.0)
    clock.advance(1.5)
    plan.opponent_moved(2)
    plan.player_moved(3, 1.5)
    clock.advance(0.1)
    assert plan.offer(1, "post_move", 0, 0, "Excellent: the only move.", streamed=True) == HOLD
    assert plan.offer(1, "post_move", 0, 1, "It pulls the defender.", streamed=True) == TEXT
    assert plan.offer(2, "post_move", 2, 0, "Exactly.", streamed=False) == TEXT
    assert plan.held_text() == "Excellent: the only move."
    plan.tick()
    assert said(plan) == [], "not while the line is played"
    clock.advance(LINE_END_S)
    plan.tick()
    line = plan.next_line()
    assert line is not None and line.text == "Excellent: the only move." and line.prepared
    assert said(plan) == []


def test_the_line_word_is_said_at_once_when_the_next_puzzle_comes() -> None:
    plan, clock = plan_at()
    clock.advance(30)
    plan.player_moved(1, 30.0)
    clock.advance(1.0)
    plan.player_moved(3, 1.0)
    clock.advance(0.6)
    assert plan.offer(1, "post_move", 0, 0, "Well found.", streamed=True) == HOLD
    clock.advance(1.0)
    plan.new_position()  # the next puzzle, before the line's end
    assert said(plan) == ["Well found."]


def test_a_line_word_too_old_is_dropped() -> None:
    plan, clock = plan_at()
    clock.advance(30)
    plan.player_moved(1, 30.0)
    clock.advance(1.0)
    plan.player_moved(3, 1.0)
    clock.advance(0.5)
    assert plan.offer(1, "post_move", 0, 0, "Well found.", streamed=True) == HOLD
    for ply in (5, 7, 9, 11, 13, 15):  # a long fast line
        clock.advance(1.5)
        plan.player_moved(ply, 1.5)
    clock.advance(LINE_END_S)
    plan.tick()
    assert said(plan) == [], f"older than {LINE_WORD_FRESH_S} s: never said"


def test_a_lesson_is_said_after_a_premove_its_first_sentence_only() -> None:
    """G4: a blunder, the next move premoved 0.8 s later, then the lesson comes (the case of
    Tai's lost exf5 lesson, 2026-10-02)."""
    plan, clock = plan_at()
    clock.advance(25)
    plan.player_moved(1, 25.0)
    clock.advance(0.8)
    plan.player_moved(3, 0.8)
    clock.advance(1.1)
    assert plan.offer(1, "post_mistake", 0, 0, "exf5 was the move.", streamed=True) == SAY
    assert plan.offer(1, "post_mistake", 0, 1, "It wins the queen.", streamed=True) == TEXT
    assert said(plan) == ["exf5 was the move."]


def test_a_lesson_on_a_move_taken_back_is_never_said() -> None:
    """P3: a wrong move in a puzzle, then a retry. The lesson names the answer: once the move
    is taken back it is not said, and one playing is stopped."""
    plan, clock = plan_at()
    clock.advance(40)
    plan.player_moved(1, 40.0)
    clock.advance(1.9)
    assert plan.offer(1, "post_mistake", 0, 0, "Rg8 was the move.", streamed=True) == SAY
    assert plan.next_line() is not None  # handed to the voice: playing
    plan.voice_state(busy=True, waiting=0)
    clock.advance(3)
    assert plan.taken_back(0) is True, "the lesson playing is cut: it names the answer"
    plan.stopped()
    assert plan.offer(1, "post_mistake", 0, 1, "It mates.", streamed=True) == TEXT


def test_the_answer_to_a_question_is_said_whole_whatever_is_played() -> None:
    """P7: asked, then the move played while the answer is said."""
    plan, clock = plan_at()
    clock.advance(30)
    assert plan.offer(1, "user_ask", 0, 0, "Play Qa8+.", streamed=True) == SAY
    assert plan.next_line() is not None
    plan.voice_state(busy=True, waiting=0)
    clock.advance(5)
    assert plan.player_moved(1, 35.0) is False, "the answer is not cut"
    assert plan.offer(1, "user_ask", 0, 1, "It deflects the rook.", streamed=True) == SAY
    assert said(plan) == ["It deflects the rook."]


def test_a_hint_is_said_only_while_its_position_is_on_the_board() -> None:
    plan, clock = plan_at()
    clock.advance(45)
    assert plan.offer(1, "stuck", 0, 0, "Only one move holds.", streamed=True) == SAY
    assert plan.next_line() is not None
    plan.voice_state(busy=True, waiting=0)
    clock.advance(1)
    assert plan.player_moved(1, 46.0) is True, "a hint whose position is gone stops"
    plan.stopped()
    assert plan.offer(1, "stuck", 0, 1, "Which check?", streamed=True) == TEXT
    # A hint that comes after the move is never said.
    assert plan.offer(2, "stuck", 0, 0, "Look at f7.", streamed=True) == TEXT


def test_blitz_says_the_lessons_and_leaves_the_praise_to_the_text() -> None:
    """G2: three seconds a move. The word on a move is text -- held for a next puzzle, never said
    at a pause in the game; the lesson on the mistake is said."""
    plan, clock = plan_at()
    ply = 1
    for _ in range(3):
        clock.advance(3)
        plan.player_moved(ply, 2.8)
        ply += 2
    clock.advance(0.15)
    assert plan.offer(1, "post_move", ply - 3, 0, "Exactly.", streamed=False) == HOLD
    clock.advance(LINE_END_S)
    plan.tick()
    assert said(plan) == [], "a pause in a game says no template word"
    clock.advance(2.8)
    plan.player_moved(ply, 2.8)
    clock.advance(0.15)
    assert plan.offer(2, "post_mistake", ply - 1, 0, "Rg8 was stronger.", streamed=False) == SAY


def test_the_word_on_a_retry_after_a_takeback_is_said() -> None:
    """Tai's Pixel, 2026-10-03: a takeback, the right move played at the same ply, and its word
    ("Just what the position wanted.") went to the text -- the ply was marked withdrawn, not the
    move. The lesson on the withdrawn move stays unsaid."""
    plan, clock = plan_at()
    clock.advance(16.5)
    plan.player_moved(1, 16.5)
    clock.advance(1.0)
    assert plan.offer(1, "post_mistake", 0, 0, "Bxd2+ was the move.", streamed=True) == SAY
    assert said(plan) == ["Bxd2+ was the move."]
    clock.advance(8.0)
    plan.taken_back(0)
    plan.stopped()
    clock.advance(4.0)
    plan.player_moved(1, 4.0)
    clock.advance(0.3)
    assert plan.offer(2, "post_move", 0, 0, "Just what the position wanted.", streamed=False) == SAY
    assert said(plan) == ["Just what the position wanted."]
    assert plan.offer(1, "post_mistake", 0, 1, "It wins a piece.", streamed=True) == TEXT


def test_a_quickly_solved_puzzle_hears_its_word_when_the_next_comes_up() -> None:
    """The puzzle solved in quick moves, no word of the model's: the last template word is said
    as the next puzzle comes up, not lost to the text."""
    plan, clock = plan_at()
    clock.advance(20.0)
    plan.player_moved(1, 20.0)
    for ply in (3, 5):
        clock.advance(1.2)
        plan.player_moved(ply, 1.2)
    clock.advance(0.2)
    assert plan.offer(1, "post_move", 4, 0, "Spot on.", streamed=False) == HOLD
    clock.advance(1.5)
    plan.new_position()
    assert said(plan) == ["Spot on."]


def test_a_moderate_pace_hears_its_short_words() -> None:
    """G5: four to six seconds a move -- the coach's fast pace -- and a half-second word is said
    on a move that took four seconds or more: it is over before the next lands."""
    plan, clock = plan_at()
    ply = 1
    for think in (5.0, 4.5, 5.5):
        clock.advance(think)
        plan.player_moved(ply, think)
        ply += 2
    clock.advance(0.15)
    assert plan.offer(1, "post_move", ply - 3, 0, "Exactly.", streamed=False) == SAY


def test_a_move_after_long_thought_is_slow_whatever_the_pace_was() -> None:
    """G3: a burst of recaptures, then 22 s on a move: its word is said."""
    plan, clock = plan_at()
    for ply in (1, 3, 5, 7):
        clock.advance(1.2)
        plan.player_moved(ply, 1.2)
    clock.advance(22)
    plan.player_moved(9, 22.0)
    clock.advance(1.6)
    assert plan.offer(1, "post_move", 8, 0, "Good square.", streamed=True) == SAY


def test_no_word_is_said_late() -> None:
    plan, clock = plan_at()
    clock.advance(20)
    plan.player_moved(1, 20.0)
    clock.advance(PRAISE_FRESH_S + 0.5)
    assert plan.offer(1, "post_move", 0, 0, "Exactly.", streamed=False) == TEXT
    clock.advance(20)
    plan.player_moved(3, 20.0)
    clock.advance(LESSON_FRESH_S + 1)
    assert plan.offer(2, "post_mistake", 2, 0, "Late lesson.", streamed=True) == TEXT


def test_what_waits_is_judged_again_when_the_player_moves() -> None:
    """Two sentences of a word wait; the player moves: the second is dropped, the first plays."""
    plan, clock = plan_at()
    clock.advance(20)
    plan.player_moved(1, 20.0)
    clock.advance(1.6)
    plan.offer(1, "post_move", 0, 0, "One.", streamed=True)
    plan.offer(1, "post_move", 0, 1, "Two.", streamed=True)
    assert plan.next_line() is not None  # "One." playing
    assert plan.next_line() is not None  # "Two." made ahead, waiting
    plan.voice_state(busy=True, waiting=1)
    clock.advance(4)
    assert plan.player_moved(3, 4.0) is False, "the sentence playing finishes"
    assert plan.waiting_stale() is True, "the one waiting is dropped"
