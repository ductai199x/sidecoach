"""Provider interface. Every LLM backend implements this and nothing more."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass
class ToolCall:
    """A function the model asked to have run, and its arguments as it wrote them."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    stats: dict[str, Any] = field(default_factory=dict)
    # The functions the model asked for instead of (or before) a reply; empty when it
    # replied, and always empty from a provider that takes no tools.
    tool_calls: list[ToolCall] = field(default_factory=list)


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
#
# A provider may also take tools (`tools_ok = True`): `chat` and `chat_stream` then accept
# `tools`, function specs in the OpenAI form, and `turns`, the wire messages that follow the
# conversation -- the model's earlier calls and what each returned -- and a completion
# carries the model's `tool_calls` when it asked for some.
