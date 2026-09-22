"""When the coach speaks without being asked.

DESIGN.md section 5.5, in priority order: an explicit ask is always answered; a lesson after
a mistake; brief feedback after every move the user plays; an orienting question at a
decisive moment; a hint once the user has been stuck for a while. Higher numbers yield to
lower ones when they collide. While the user is thinking, at most one unprompted message per
position, so the coach never nags. A quiet toggle silences everything but explicit asks.

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


# In a run of decisive positions (a won endgame: every move is "the" move), the decisive
# question is asked at most once in this many plies.
DECISIVE_EVERY = 8


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
    # The position the user is thinking in: its ply, when it appeared, and whether the
    # coach has spoken while they think in it.
    _ply: int | None = field(default=None, init=False)
    _since: float | None = field(default=None, init=False)
    _spoken_at: set[int] = field(default_factory=set, init=False)
    _user_to_move: bool = field(default=False, init=False)
    _remaining_ms: int | None = field(default=None, init=False)
    # The board is under a dialog: the game may be over, and no nudge makes sense.
    covered: bool = field(default=False, init=False)
    # Why the last unprompted trigger did not fire, in a few words, for the debug log --
    # the answer to "why is the coach quiet?". None once a trigger fires.
    last_reason: str | None = field(default=None, init=False)
    # The last ply the decisive question was asked at, while the positions stay decisive.
    _last_decisive_ply: int | None = field(default=None, init=False)

    # What the board is doing

    def new_position(self, ply: int, ts: float, *, user_to_move: bool) -> None:
        """The board reached `ply`; the user is (or is not) the one to move."""
        self._ply = ply
        self._since = ts
        self._user_to_move = user_to_move

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

    def on_position_analysed(self, ply: int, *, decisive: bool) -> Trigger | None:
        """The engine has looked at the position the user is thinking in. The decisive
        question is asked when a position turns decisive, not on every move of a won
        endgame, where the lines diverge at each turn: once a stretch of decisive
        positions has been pointed at, the next question waits for a quiet position in
        between, or `DECISIVE_EVERY` plies."""
        if not self._may_speak_now(ply):
            return None
        if not decisive:
            self._last_decisive_ply = None
            return self._silent("position not decisive")
        last = self._last_decisive_ply
        if last is not None and ply - last < DECISIVE_EVERY:
            return self._silent(f"decisive stretch already pointed at (ply {last})")
        self._last_decisive_ply = ply
        self.last_reason = None
        return Trigger.DECISIVE_QUESTION

    def tick(self, ts: float) -> Trigger | None:
        """Time passing while the user thinks; a hint once they have been stuck long enough."""
        if self._ply is None or self._since is None:
            return self._silent("no position yet")
        if not self._may_speak_now(self._ply):
            return None
        if self.in_scramble:
            return self._silent("time scramble: no stuck hints")
        threshold = self.stuck_threshold()
        if ts - self._since < threshold:
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

    def said(self, ply: int) -> None:
        """One unprompted message went out for the position at `ply`; no more while it lasts."""
        self._spoken_at.add(ply)

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
        elif ply in self._spoken_at:
            reason = "already spoke in this position"
        if reason is not None:
            self.last_reason = reason
            return False
        return True


# Lower is more important; when two triggers want the same moment, the coach speaks to the
# lower one and folds the other in or drops it.
PRIORITY: dict[Trigger, int] = {
    Trigger.USER_ASK: 1,
    Trigger.POST_MISTAKE: 2,
    Trigger.POST_MOVE: 3,
    Trigger.DECISIVE_QUESTION: 4,
    Trigger.STUCK: 5,
    Trigger.REVIEW: 6,
}


def in_priority(triggers: list[Trigger]) -> list[Trigger]:
    return sorted(dict.fromkeys(triggers), key=lambda trigger: PRIORITY[trigger])
