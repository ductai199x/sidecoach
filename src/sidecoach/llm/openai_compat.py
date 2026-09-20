"""One provider for every OpenAI-compatible chat endpoint: LM Studio locally, DashScope,
OpenAI itself, or anything else that serves `/v1/chat/completions`.

What differs between servers is how reasoning is controlled, and that is the one thing a
provider is configured with (`thinking_control`). Measured 2026-09-14 on the first two:

- "prefill": LM Studio serving the Qwen 3.8 GGUFs ignores `/no_think` and every reasoning
  parameter, but honours a trailing assistant message holding an empty think block: the
  server continues that message and the model skips reasoning.
- "param": DashScope honours `enable_thinking` in the request body. With qwen3.8-max a
  turn took 9.7 s with thinking and 1.8 s without; qwen3.8-flash 6.1 s and 1.1 s.
- "effort": OpenAI's reasoning models take `reasoning_effort` (none, low, medium, high,
  ...) and reject `max_tokens` and `temperature`; the answer budget goes in
  `max_completion_tokens`, with headroom for the reasoning tokens that count against it.
  The effort is a property of the provider, not of the call: a coach at medium and a
  judge at none are two providers of the same model (`with_effort`).
- "none": nothing is sent and the model keeps its default.

Reasoning text, when the server returns it, comes back in `message.reasoning_content`;
the cached share of the prompt in `usage.prompt_tokens_details.cached_tokens`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any, Literal

import httpx

from sidecoach.llm.base import ChatMessage, Completion
from sidecoach.llm.usage import Usage

ThinkingControl = Literal["prefill", "param", "effort", "none"]
THINKING_CONTROLS: tuple[str, ...] = ("prefill", "param", "effort", "none")
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"
# Reasoning tokens count against max_completion_tokens: a caller's budget is for the
# answer, so the reasoning gets headroom of its own or a short answer comes back empty.
REASONING_HEADROOM = 4000


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
        reasoning_effort: str | None = None,
        timeout_s: float = 60.0,
        extra_body: dict[str, Any] | None = None,
        transport: httpx.BaseTransport | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.thinking_control: ThinkingControl = thinking_control
        self.reasoning_effort = reasoning_effort
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._transport = transport
        # The session's tokens over every call, shared with the provider's copies at
        # other efforts (llm.usage): one figure for the panel.
        self.usage = usage if usage is not None else Usage()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=timeout_s, headers=headers, transport=transport)

    def with_effort(self, effort: str | None) -> OpenAICompatProvider:
        """The same endpoint and model at another reasoning effort (an "effort" provider);
        any other kind of provider is returned as it is."""
        if self.thinking_control != "effort" or effort == self.reasoning_effort:
            return self
        return OpenAICompatProvider(
            self.name,
            self.base_url,
            self.model,
            api_key=self._api_key,
            thinking_control=self.thinking_control,
            reasoning_effort=effort,
            timeout_s=self.timeout_s,
            extra_body=self.extra_body,
            transport=self._transport,
            usage=self.usage,
        )

    @staticmethod
    def _chars(body: dict[str, Any]) -> int:
        """The prompt's size in characters, for the estimate when no usage comes back."""
        return sum(len(str(m.get("content") or "")) for m in body.get("messages") or [])

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
            **self.extra_body,
        }
        if self.thinking_control == "effort":
            body["max_completion_tokens"] = max_tokens + (
                REASONING_HEADROOM if self.reasoning_effort else 0
            )
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        else:
            body["max_tokens"] = max_tokens
            body["temperature"] = temperature
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
        if response.status_code != 200:
            # The status code stays in the message: the callers' backoff keys on "429".
            raise RuntimeError(f"{response.status_code}: {response.text[:200]}")
        data = response.json()
        elapsed = time.perf_counter() - started

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        stats = {
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        text = (message.get("content") or "").strip()
        reasoning = (message.get("reasoning_content") or "").strip()
        self.usage.record(
            stats, prompt_chars=self._chars(body), completion_chars=len(text) + len(reasoning)
        )
        return Completion(text=text, reasoning=reasoning, stats=stats)

    def chat_stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
        on_delta: Callable[[str], bool],
    ) -> Completion:
        """The reply streamed: every visible delta to `on_delta` as it arrives, the first
        one's latency and the stream's usage in the stats. A callback returning False
        closes the stream; the usage is then whatever the server had sent, usually
        nothing, and finish_reason is "stopped"."""
        body = self.request_body(
            system, messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking
        )
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        reasoning: list[str] = []
        usage: dict[str, Any] = {}
        finish: str | None = None
        stopped = False
        with self._client.stream(
            "POST", f"{self.base_url}/chat/completions", json=body
        ) as response:
            if response.status_code != 200:
                text = response.read().decode("utf-8", "replace")
                raise RuntimeError(f"{response.status_code}: {text[:200]}")
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("reasoning_content"):
                        reasoning.append(str(delta["reasoning_content"]))
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    text = delta.get("content")
                    if not text:
                        continue
                    if first is None:
                        first = time.perf_counter() - started
                    parts.append(text)
                    if not on_delta(text):
                        stopped = True
                        break
                if stopped:
                    break
        elapsed = time.perf_counter() - started
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        stats = {
            "elapsed_s": round(elapsed, 3),
            "first_token_s": round(first, 3) if first is not None else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "finish_reason": "stopped" if stopped else finish,
            "streamed": True,
        }
        text = "".join(parts).strip()
        thought = "".join(reasoning).strip()
        # A stream closed early has no usage chunk: the count is then an estimate, and
        # the reasoning that was never streamed goes uncounted.
        self.usage.record(
            stats, prompt_chars=self._chars(body), completion_chars=len(text) + len(thought)
        )
        return Completion(text=text, reasoning=thought, stats=stats)

    def complete(self, system: str, user: str, **kwargs: Any) -> Completion:
        """Single-turn convenience over `chat`."""
        return self.chat(system, [{"role": "user", "content": user}], **kwargs)

    def close(self) -> None:
        self._client.close()


__all__ = [
    "EMPTY_THINK_PREFILL",
    "REASONING_HEADROOM",
    "THINKING_CONTROLS",
    "OpenAICompatProvider",
    "ThinkingControl",
    "build_messages",
]
