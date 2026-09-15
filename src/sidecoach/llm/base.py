"""Provider interface. Every LLM backend implements this and nothing more."""

from __future__ import annotations

from collections.abc import Sequence
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
