"""No model at all: the coach speaks from the board and its templates (Tai, 2026-09-21:
the app must work, free and offline, for whoever wants it that way).

`llm.provider = "none"` gives the coach this provider. It answers no call; the Speaker
sees `offline` and takes each request's fallback -- the verdict written from the board
for a best, excellent or book move, the lesson and hint templates, the review's own
sentences -- without a model, a network or a key. The user's typed questions get the
template for their kind (the hint, the material, the standing) rather than a reply.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sidecoach.llm.base import ChatMessage, Completion
from sidecoach.llm.usage import Usage


class OfflineError(RuntimeError):
    """Raised by any call: there is no model to call."""


class OfflineProvider:
    name = "none"
    model = None
    offline = True

    def __init__(self) -> None:
        self.usage = Usage()

    def healthcheck(self) -> tuple[bool, str]:
        return True, "offline: the coach speaks from the board and its templates, no model"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        raise OfflineError("no model: llm.provider is none")

    def with_effort(self, effort: str | None) -> OfflineProvider:
        return self

    def __getattr__(self, name: str) -> Any:
        # No chat_stream, no client: the Speaker looks for those with getattr and falls
        # back to `chat`, which raises -- but it never gets there, `offline` comes first.
        raise AttributeError(name)
