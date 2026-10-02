"""Provider interface. Every LLM backend implements this and nothing more."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
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


class ProviderError(RuntimeError):
    """A call the server answered with an error status. The message starts with the status
    ("429: ..."), as the callers read it; `retry_after_s` is how long the server asked to be
    left alone before the next call (its Retry-After), None when it named no time."""

    def __init__(self, status: int, text: str, retry_after_s: float | None = None) -> None:
        super().__init__(f"{status}: {text[:200]}")
        self.status = status
        self.retry_after_s = retry_after_s


def retry_after(headers: Mapping[str, str], now: float | None = None) -> float | None:
    """The wait a response's headers ask for, in seconds: `retry-after-ms` (OpenAI's), else
    `Retry-After` as seconds or as an HTTP date (counted from `now`, the epoch time).
    None when they name none or nothing readable."""
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = (headers.get("retry-after") or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - (time.time() if now is None else now))


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
