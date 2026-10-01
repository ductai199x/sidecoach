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
    """Nothing before the stuck limit, whatever the position (Tai, 2026-10-01: no "hints or
    questions or any thing whatsoever until the user is stuck"); then the idea, and after a
    full stuck spell more, the focus."""
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=10))
    policy.new_position(6, 100.0, user_to_move=True)
    assert policy.tick(101.0) is None and policy.tick(109.0) is None, "not stuck yet"
    assert policy.tick(110.5) is Trigger.STUCK, "the idea"
    policy.said(6, 110.5)
    assert policy.tick(115.0) is None, "a full stuck spell from the idea"
    assert policy.tick(121.0) is Trigger.STUCK, "then the focus"
    policy.said(6, 121.0)
    assert policy.tick(200.0) is None, "and no third: the move only when asked"
    policy.new_position(8, 200.0, user_to_move=True)
    assert policy.tick(205.0) is None, "not stuck yet"
    assert policy.tick(211.0) is Trigger.STUCK


def test_nothing_unprompted_while_the_opponent_thinks_or_when_quiet() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=1))
    policy.new_position(7, 0.0, user_to_move=False)
    assert policy.tick(50.0) is None
    quiet = InterjectionPolicy(PolicySettings(stuck_seconds=1, quiet=True))
    quiet.new_position(6, 0.0, user_to_move=True)
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
    assert blitz.tick(60.0) is None
    assert blitz.on_user_move_assessed(_assessed(Severity.BLUNDER)) is Trigger.POST_MOVE
    bullet = InterjectionPolicy(time_control=TimeControl.BULLET)
    bullet.new_position(6, 0.0, user_to_move=True)
    assert bullet.on_user_move_assessed(_assessed(Severity.BLUNDER)) is None


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
