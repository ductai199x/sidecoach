"""When the coach may speak unprompted (coach/policy.py): by the clock, and by what has
happened in the position."""

from __future__ import annotations

from sidecoach.coach.policy import (
    InterjectionPolicy,
    PolicySettings,
    TimeControl,
    time_control_of,
)
from sidecoach.events import (
    Label,
    MoveAssessment,
    Severity,
    Trigger,
)


def _assessed(severity: Severity, label: Label | None = None) -> MoveAssessment:
    return MoveAssessment(
        game_id="g",
        ply=5,
        uci="e2e4",
        expected_loss=0.2 if severity is Severity.BLUNDER else 0.0,
        severity=severity,
        best=["d2d4"],
        label=label,
    )


def test_time_controls_follow_the_sites_bands() -> None:
    assert time_control_of(None) is TimeControl.UNTIMED
    assert time_control_of(15 * 60_000) is TimeControl.RAPID
    assert time_control_of(10 * 60_000) is TimeControl.RAPID
    assert time_control_of(5 * 60_000) is TimeControl.BLITZ
    assert time_control_of(3 * 60_000, 2_000) is TimeControl.BLITZ
    assert time_control_of(60_000, 1_000) is TimeControl.BULLET
    assert time_control_of(2 * 60_000, 12_000) is TimeControl.RAPID, "increment counts forty moves"


def test_two_unprompted_messages_per_position_the_idea_then_the_focus() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=10))
    policy.new_position(6, 100.0, user_to_move=True)
    assert policy.on_position_analysed(6, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(6, 101.0)
    assert policy.on_position_analysed(6, decisive=True) is None, "already spoke here"
    assert policy.tick(105.0) is None, "a full stuck spell from the idea"
    assert policy.tick(112.0) is Trigger.STUCK, "then the focus"
    policy.said(6, 112.0)
    assert policy.tick(200.0) is None, "and no third: the move only when asked"
    policy.new_position(8, 200.0, user_to_move=True)
    assert policy.tick(205.0) is None, "not stuck yet"
    assert policy.tick(211.0) is Trigger.STUCK
    assert policy.on_position_analysed(9, decisive=True) is None, "not the position on the board"


def test_nothing_unprompted_while_the_opponent_thinks_or_when_quiet() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=1))
    policy.new_position(7, 0.0, user_to_move=False)
    assert policy.on_position_analysed(7, decisive=True) is None
    assert policy.tick(50.0) is None
    quiet = InterjectionPolicy(PolicySettings(stuck_seconds=1, quiet=True))
    quiet.new_position(6, 0.0, user_to_move=True)
    assert quiet.on_position_analysed(6, decisive=True) is None
    assert quiet.tick(50.0) is None
    assert quiet.on_user_move_assessed(_assessed(Severity.BLUNDER)) is None


def test_a_mistake_earns_a_lesson_and_a_good_move_a_word() -> None:
    policy = InterjectionPolicy()
    assert policy.on_user_move_assessed(_assessed(Severity.NONE)) is Trigger.POST_MOVE
    assert policy.on_user_move_assessed(_assessed(Severity.INACCURACY)) is Trigger.POST_MOVE
    assert policy.on_user_move_assessed(_assessed(Severity.MISTAKE)) is Trigger.POST_MISTAKE
    assert policy.on_user_move_assessed(_assessed(Severity.BLUNDER)) is Trigger.POST_MISTAKE
    assert policy.on_user_move_assessed(_assessed(Severity.INACCURACY, Label.MISS)) is (
        Trigger.POST_MISTAKE
    ), "a missed chance is worth a lesson whatever it cost"


def test_blitz_keeps_only_the_post_move_word_and_bullet_nothing() -> None:
    blitz = InterjectionPolicy(PolicySettings(stuck_seconds=1), time_control=TimeControl.BLITZ)
    blitz.new_position(6, 0.0, user_to_move=True)
    assert blitz.on_position_analysed(6, decisive=True) is None
    assert blitz.tick(60.0) is None
    assert blitz.on_user_move_assessed(_assessed(Severity.BLUNDER)) is Trigger.POST_MOVE
    bullet = InterjectionPolicy(time_control=TimeControl.BULLET)
    bullet.new_position(6, 0.0, user_to_move=True)
    assert bullet.on_user_move_assessed(_assessed(Severity.BLUNDER)) is None
    assert bullet.on_position_analysed(6, decisive=True) is None


def test_the_stuck_threshold_shrinks_with_the_clock() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    assert policy.stuck_threshold() == 45
    policy.set_remaining(15 * 60_000)
    assert policy.stuck_threshold() == 45, "plenty of time: the default holds"
    policy.set_remaining(5 * 60_000)
    assert policy.stuck_threshold() == 24, "eight per cent of five minutes"
    policy.set_remaining(100_000)
    assert policy.stuck_threshold() == 20, "never below the floor"


def test_no_stuck_hints_in_a_time_scramble() -> None:
    # With 100 s left a hint still comes, at the floor; with 25 s left the user has no time
    # to hear a question, let alone act on it.
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    policy.new_position(6, 0.0, user_to_move=True)
    policy.set_remaining(100_000)
    assert policy.tick(21.0) is Trigger.STUCK
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    policy.new_position(6, 0.0, user_to_move=True)
    policy.set_remaining(25_000)
    assert policy.tick(60.0) is None
    assert policy.last_reason == "time scramble: no stuck hints"


def test_the_decisive_question_is_asked_once_per_decisive_stretch() -> None:
    """A won endgame is decisive at every move (the lines diverge at each turn), and the
    question was asked after every one of them: "before making a quiet move, what forcing
    possibilities ..." six times in a row. Once asked, it waits for a quiet position in
    between, or eight plies."""
    policy = InterjectionPolicy()
    policy.new_position(60, 0.0, user_to_move=True)
    assert policy.on_position_analysed(60, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(60)
    policy.new_position(62, 10.0, user_to_move=True)
    assert policy.on_position_analysed(62, decisive=True) is None
    assert policy.last_reason is not None and "already pointed at" in policy.last_reason
    policy.new_position(64, 20.0, user_to_move=True)
    assert policy.on_position_analysed(64, decisive=False) is None, "a quiet position between"
    policy.new_position(66, 30.0, user_to_move=True)
    assert policy.on_position_analysed(66, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(66)
    policy.new_position(74, 40.0, user_to_move=True)
    assert policy.on_position_analysed(74, decisive=True) is Trigger.DECISIVE_QUESTION, (
        "eight plies on, even with no quiet position between"
    )
