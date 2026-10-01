"""When the coach speaks without being asked.

DESIGN.md section 5.5, in priority order: an explicit ask is always answered; a lesson after
a mistake; brief feedback after every move the user plays; a hint once the user has been
stuck for a while. Higher numbers yield to lower ones when they collide. While the user is
thinking the coach says nothing until they are stuck -- no question at a decisive moment
(Tai, 2026-10-01: nothing "until the user is stuck or actively request a hint or after they
have made a move") -- and then at most two messages per position: the idea, then, if they
are still stuck, the focus, each after its own wait, so the coach never nags and never gives
the move unasked. A quiet toggle silences everything but explicit asks.

Time control scales all of it: rapid and slower get every trigger, with the stuck threshold
shrinking as the clock runs down; blitz gets the brief post-move feedback only; bullet gets
nothing during the game. Untimed games and analysis boards get everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from sidecoach.events import MoveAssessment, Severity, Trigger


class TimeControl(StrEnum):
    UNTIMED = "untimed"
    RAPID = "rapid"
    BLITZ = "blitz"
    BULLET = "bullet"


def time_control_of(initial_ms: int | None, increment_ms: int = 0) -> TimeControl:
    """Chess.com's bands: ten minutes and up is rapid, three to ten blitz, under three bullet.

    The estimated game length counts forty moves of increment, as the sites do.
    """
    if initial_ms is None:
        return TimeControl.UNTIMED
    estimated = initial_ms + 40 * increment_ms
    if estimated >= 10 * 60_000:
        return TimeControl.RAPID
    if estimated >= 3 * 60_000:
        return TimeControl.BLITZ
    return TimeControl.BULLET


# Unprompted messages in one position while the user thinks: the idea, then the focus (Tai,
# 2026-09-23). The move itself only when asked.
UNPROMPTED_PER_POSITION = 2


@dataclass
class PolicySettings:
    stuck_seconds: float = 45.0
    quiet: bool = False
    # The stuck threshold never drops below this, even on a low clock. Twenty seconds on a
    # move is thinking, not being stuck; at eight, the first live game got a nudge on nearly
    # every endgame move, each one spoken over the next.
    stuck_floor_seconds: float = 20.0
    # With a clock, a hint comes after this share of the remaining time, at most.
    stuck_share_of_remaining: float = 0.08
    # Under this much clock the user is in a scramble: a two-sentence question takes longer
    # to hear than they have to think, so no stuck hints at all.
    stuck_scramble_seconds: float = 30.0


@dataclass
class InterjectionPolicy:
    """Decides which unprompted triggers fire, and remembers what it already said."""

    settings: PolicySettings = field(default_factory=PolicySettings)
    time_control: TimeControl = TimeControl.UNTIMED
    # The position the user is thinking in: its ply, when it appeared, how many times the
    # coach has spoken unprompted while they think in it, and when it last did.
    _ply: int | None = field(default=None, init=False)
    _since: float | None = field(default=None, init=False)
    _spoken: dict[int, int] = field(default_factory=dict, init=False)
    _said_ts: float | None = field(default=None, init=False)
    _user_to_move: bool = field(default=False, init=False)
    _remaining_ms: int | None = field(default=None, init=False)
    # The board is under a dialog: the game may be over, and no nudge makes sense.
    covered: bool = field(default=False, init=False)
    # Why the last unprompted trigger did not fire, in a few words, for the debug log --
    # the answer to "why is the coach quiet?". None once a trigger fires.
    last_reason: str | None = field(default=None, init=False)

    # What the board is doing

    def new_position(self, ply: int, ts: float, *, user_to_move: bool) -> None:
        """The board reached `ply`; the user is (or is not) the one to move."""
        if ply != self._ply:
            self._said_ts = None
        self._ply = ply
        self._since = ts
        self._user_to_move = user_to_move

    def restart(self) -> None:
        """A new game, or a position adopted: move numbers start again, so what was said in
        the last one says nothing about this one. Kept by ply, the last puzzles' words at
        their first positions left a new puzzle "gave the idea and the focus here" before a
        hint (Tai's phone, 2026-09-29)."""
        self._spoken.clear()
        self._said_ts = None

    def set_remaining(self, remaining_ms: int | None) -> None:
        """The user's clock, when the screen shows one."""
        self._remaining_ms = remaining_ms

    @property
    def ply(self) -> int | None:
        return self._ply

    @property
    def user_to_move(self) -> bool:
        return self._user_to_move

    @property
    def position_since(self) -> float | None:
        """When the position the user is thinking in appeared (the loop's clock)."""
        return self._since

    @property
    def stuck_from(self) -> float | None:
        """When the stuck spell being counted began: the position, or the last thing said
        in it."""
        if self._since is None or self._said_ts is None:
            return self._since
        return max(self._since, self._said_ts)

    # The triggers

    def on_user_move_assessed(self, assessment: MoveAssessment) -> Trigger | None:
        """After the user's move is measured: a lesson if it was a mistake, else a word."""
        if self.settings.quiet:
            return self._silent("quiet mode")
        if self.time_control is TimeControl.BULLET:
            return self._silent("bullet: no unprompted coaching")
        serious = assessment.severity in (Severity.MISTAKE, Severity.BLUNDER) or (
            assessment.label is not None and assessment.label.value == "miss"
        )
        self.last_reason = None
        if serious and self.time_control is not TimeControl.BLITZ:
            return Trigger.POST_MISTAKE
        return Trigger.POST_MOVE

    def tick(self, ts: float) -> Trigger | None:
        """Time passing while the user thinks; a hint once they have been stuck long enough,
        counted from the position or from the last thing said in it."""
        if self._ply is None or self._since is None:
            return self._silent("no position yet")
        if not self._may_speak_now(self._ply):
            return None
        if self.in_scramble:
            return self._silent("time scramble: no stuck hints")
        threshold = self.stuck_threshold()
        since = self.stuck_from
        if since is None or ts - since < threshold:
            return self._silent(f"waiting for a stuck hint at {threshold:.0f}s")
        self.last_reason = None
        return Trigger.STUCK

    @property
    def in_scramble(self) -> bool:
        """Whether the user's clock is down to the last seconds."""
        return (
            self._remaining_ms is not None
            and self._remaining_ms < self.settings.stuck_scramble_seconds * 1000
        )

    def _silent(self, reason: str) -> None:
        self.last_reason = reason
        return None

    def said(self, ply: int, ts: float | None = None) -> None:
        """One unprompted message went out for the position at `ply`, at `ts`: the next one
        waits a full stuck spell from it, and none comes after the second."""
        self._spoken[ply] = self._spoken.get(ply, 0) + 1
        if ply == self._ply and ts is not None:
            self._said_ts = ts

    def exhausted(self, ply: int) -> None:
        """The hints unasked for the position at `ply` are all given, here or earlier in the
        same combination: nothing more comes until they ask."""
        self._spoken[ply] = max(self._spoken.get(ply, 0), UNPROMPTED_PER_POSITION)

    def stuck_threshold(self) -> float:
        """Seconds on the move before a hint, shorter as the clock runs down."""
        threshold = self.settings.stuck_seconds
        if self._remaining_ms is not None:
            share = self._remaining_ms / 1000.0 * self.settings.stuck_share_of_remaining
            threshold = min(threshold, max(self.settings.stuck_floor_seconds, share))
        return threshold

    def _may_speak_now(self, ply: int) -> bool:
        reason = None
        if self.settings.quiet:
            reason = "quiet mode"
        elif self.covered:
            reason = "the board is covered"
        elif not self._user_to_move:
            reason = "opponent to move"
        elif ply != self._ply:
            reason = "position moved on"
        elif self.time_control in (TimeControl.BLITZ, TimeControl.BULLET):
            reason = f"{self.time_control.value}: no unprompted hints"
        elif self._spoken.get(ply, 0) >= UNPROMPTED_PER_POSITION:
            reason = "gave the idea and the focus here; the move only when asked"
        if reason is not None:
            self.last_reason = reason
            return False
        return True
