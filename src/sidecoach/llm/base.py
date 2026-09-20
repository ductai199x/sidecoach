"""Provider interface. Every LLM backend implements this and nothing more."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    stats: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str

    def healthcheck(self) -> tuple[bool, str]: ...

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion: ...


# A provider may also stream: `chat_stream(system, messages, *, max_tokens, temperature,
# thinking, on_delta)` with the same result as `chat`, each piece of visible text going
# to `on_delta` as it arrives; a callback returning False ends the stream early (the text
# so far is the completion, the usage then unknown). The coach's Speaker uses it when it
# is there (`getattr(provider, "chat_stream", None)`) and `chat` otherwise.
StreamCallback = Callable[[str], bool]
