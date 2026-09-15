"""Provider for any OpenAI-compatible chat endpoint: LM Studio locally, DashScope hosted.

Thinking control differs by server. Measured 2026-09-14:

- LM Studio serving the Qwen 3.8 GGUFs ignores `/no_think` and every reasoning parameter,
  but honours a trailing assistant message holding an empty think block: the server
  continues that message and the model skips reasoning. That is the "prefill" strategy.
- DashScope honours `enable_thinking` in the request body, the "param" strategy. With
  qwen3.8-max a turn took 9.7 s with thinking and 1.8 s without; qwen3.8-flash 6.1 s and
  1.1 s. Streaming first token about 1.1 s for both.
- "none" sends nothing and leaves the model to its default.

Reasoning text, when produced, comes back in `message.reasoning_content` on both servers.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Literal

import httpx

from sidecoach.llm.base import ChatMessage, Completion

ThinkingControl = Literal["prefill", "param", "none"]
THINKING_CONTROLS: tuple[str, ...] = ("prefill", "param", "none")
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"


def build_messages(
    system: str,
    messages: Sequence[ChatMessage],
    *,
    thinking: bool,
    control: ThinkingControl,
) -> list[dict[str, str]]:
    """The wire message list: system, the conversation, and the no-think prefill if used."""
    wire: list[dict[str, str]] = [{"role": "system", "content": system}]
    wire.extend({"role": m["role"], "content": m["content"]} for m in messages)
    if not thinking and control == "prefill":
        wire.append({"role": "assistant", "content": EMPTY_THINK_PREFILL})
    return wire


class OpenAICompatProvider:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        thinking_control: ThinkingControl = "none",
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking_control: ThinkingControl = thinking_control
        self.extra_body = dict(extra_body or {})
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout_s, headers=headers, transport=transport)

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self._client.get(f"{self.base_url}/models")
            response.raise_for_status()
            ids = [str(m.get("id")) for m in response.json().get("data", [])]
        except Exception as exc:  # noqa: BLE001
            return False, f"{self.base_url} unreachable: {exc}"
        if self.model in ids:
            return True, f"{self.model} at {self.base_url}"
        return False, f"model {self.model!r} not among {len(ids)} served models"

    def request_body(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        thinking: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(
                system, messages, thinking=thinking, control=self.thinking_control
            ),
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        if self.thinking_control == "param":
            body["enable_thinking"] = thinking
        return body

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        body = self.request_body(
            system, messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking
        )
        started = time.perf_counter()
        response = self._client.post(f"{self.base_url}/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()
        elapsed = time.perf_counter() - started

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        stats = {
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        return Completion(
            text=(message.get("content") or "").strip(),
            reasoning=(message.get("reasoning_content") or "").strip(),
            stats=stats,
        )

    def complete(self, system: str, user: str, **kwargs: Any) -> Completion:
        """Single-turn convenience over `chat`."""
        return self.chat(system, [{"role": "user", "content": user}], **kwargs)

    def close(self) -> None:
        self._client.close()
