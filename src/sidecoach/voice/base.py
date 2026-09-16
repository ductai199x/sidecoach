"""Text-to-speech provider interface. Every backend implements this and nothing more.

The coach calls `say`; how the words are voiced -- which model, on which device -- is the
provider's business. Speaking must never take the session down: a provider that cannot load
its model reports it from `healthcheck` and turns `say` into a no-op, exactly as a missing
clock costs the clock and nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sidecoach.voice.notation import speak_notation


@runtime_checkable
class TTSProvider(Protocol):
    name: str

    def healthcheck(self) -> tuple[bool, str]:
        """(ready, detail): whether the voice can speak, and why not when it cannot."""
        ...

    def say(self, text: str) -> None:
        """Speak a line. Chess notation in it is read aloud in words (see `speak_notation`).

        Returns once the line is queued, not once it finishes: speaking is asynchronous so
        the coach is never blocked waiting on audio.
        """
        ...

    def stop(self) -> None:
        """Cut off anything now playing or queued -- for a barge-in or a new position."""
        ...

    @property
    def busy(self) -> bool:
        """Whether a line is playing or waiting to play."""
        ...


class NullTTS:
    """The silent provider: healthcheck says why, `say` does nothing. Used when voice is off
    or a real backend could not be built, so the caller always has something to call."""

    name = "null"

    def __init__(self, reason: str = "voice disabled") -> None:
        self._reason = reason

    def healthcheck(self) -> tuple[bool, str]:
        return False, self._reason

    def say(self, text: str) -> None:
        return None

    def stop(self) -> None:
        return None

    @property
    def busy(self) -> bool:
        return False


__all__ = ["NullTTS", "TTSProvider", "speak_notation"]
