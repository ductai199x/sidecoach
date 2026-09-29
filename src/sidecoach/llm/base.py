"""Provider interface. Every LLM backend implements this and nothing more."""

from __future__ import annotations

import re
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
    # The reply as its wire takes it back in a later request, where that is more than the
    # text and the calls: the Responses API's output items, the model's reasoning among them.
    items: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolRound:
    """One round of the model's calls: the reply that made them, and what each returned, in
    the calls' order. Each provider puts it on its own wire."""

    reply: Completion
    results: list[str]


_STATUS = re.compile(r"^(\d{3}):")


def refused_for_good(exc: BaseException) -> bool:
    """A request the server refused as it stands -- a 4xx other than 429 (bad request, too
    long, no access): asking again costs time and changes nothing. Providers raise with the
    status code heading the message."""
    found = _STATUS.match(str(exc))
    return found is not None and found.group(1).startswith("4") and found.group(1) != "429"


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
# `tools`, function specs in the chat-completions form, `turns`, the rounds that follow the
# conversation (`ToolRound`: the model's earlier calls and what each returned), and
# `tool_choice`; a completion carries the model's `tool_calls` when it asked for some.
