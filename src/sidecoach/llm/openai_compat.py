"""One provider for every OpenAI-compatible endpoint: LM Studio locally, DashScope, OpenAI
itself, or anything else that serves `/v1/chat/completions`.

What differs between servers is how reasoning is controlled, and that is the one thing a
provider is configured with (`thinking_control`). Measured 2026-09-14 on the first two:

- "prefill": LM Studio serving the Qwen 3.8 GGUFs ignores `/no_think` and every reasoning
  parameter, but honours a trailing assistant message holding an empty think block: the
  server continues that message and the model skips reasoning.
- "param": DashScope honours `enable_thinking` in the request body. With qwen3.8-max a
  turn took 9.7 s with thinking and 1.8 s without; qwen3.8-flash 6.1 s and 1.1 s.
- "effort": OpenAI's reasoning models take a reasoning effort (none, low, medium, high,
  ...) and reject `max_tokens` and `temperature`; the answer budget has headroom for the
  reasoning tokens that count against it. The effort is a property of the provider, not
  of the call: a coach at medium and a judge at none are two providers of the same model
  (`with_effort`).
- "none": nothing is sent and the model keeps its default.

And which API it speaks (`wire`): "chat", the chat completions every such server serves, or
"responses", OpenAI's `/v1/responses` -- the only one on which its reasoning models take
function tools with an effort ("Function tools with reasoning_effort are not supported for
gpt-5.6-luna in /v1/chat/completions", 2026-09-29: at medium the coach's engine tools were
refused on every question). It is sent stateless (`store` false), and the model's reasoning
comes back encrypted in its output items, which go back with the tools' results so it
reasons on from where it was.

On the chat wire, reasoning text, when the server returns it, comes back in
`message.reasoning_content`; the cached share of the prompt in
`usage.prompt_tokens_details.cached_tokens`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from sidecoach.llm.base import (
    ChatMessage,
    Completion,
    ProviderError,
    ToolCall,
    ToolRound,
    retry_after,
)
from sidecoach.llm.usage import Usage

ThinkingControl = Literal["prefill", "param", "effort", "none"]
THINKING_CONTROLS: tuple[str, ...] = ("prefill", "param", "effort", "none")
Wire = Literal["chat", "responses"]
EMPTY_THINK_PREFILL = "<think>\n\n</think>\n\n"
# Reasoning tokens count against the answer's token budget: a caller's budget is for the
# answer, so the reasoning gets headroom of its own or a short answer comes back empty.
REASONING_HEADROOM = 4000
# The fewest output tokens the Responses API takes; fewer is refused.
MIN_OUTPUT_TOKENS = 16
# The failures that come after the request reached the server: the call is billed. A
# refused connection or a connect timeout never reached it, and is not.
_SENT_LOST = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError)


def chat_turns(turns: Sequence[ToolRound]) -> list[dict[str, Any]]:
    """The rounds of tool calls as chat messages: the reply that asked, then each result."""
    wire: list[dict[str, Any]] = []
    for turn in turns:
        wire.append(
            {
                "role": "assistant",
                "content": turn.reply.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in turn.reply.tool_calls
                ],
            }
        )
        wire.extend(
            {"role": "tool", "tool_call_id": call.id, "content": result}
            for call, result in zip(turn.reply.tool_calls, turn.results, strict=True)
        )
    return wire


def build_messages(
    system: str,
    messages: Sequence[ChatMessage],
    *,
    thinking: bool,
    control: ThinkingControl,
    turns: Sequence[ToolRound] = (),
) -> list[dict[str, Any]]:
    """The chat wire's message list: system, the conversation, the tool calls and their
    results (`turns`), and the no-think prefill if used."""
    wire: list[dict[str, Any]] = [{"role": "system", "content": system}]
    wire.extend({"role": m["role"], "content": m["content"]} for m in messages)
    wire.extend(chat_turns(turns))
    if not thinking and control == "prefill":
        wire.append({"role": "assistant", "content": EMPTY_THINK_PREFILL})
    return wire


def response_input(
    messages: Sequence[ChatMessage], turns: Sequence[ToolRound] = ()
) -> list[dict[str, Any]]:
    """The Responses wire's input: the conversation, then each round -- the reply's own
    output items (its reasoning with them), or its text and calls when it has none -- and
    what each call returned."""
    wire: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages]
    for turn in turns:
        if turn.reply.items:
            wire.extend(turn.reply.items)
        else:
            if turn.reply.text:
                wire.append({"role": "assistant", "content": turn.reply.text})
            wire.extend(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
                for call in turn.reply.tool_calls
            )
        wire.extend(
            {"type": "function_call_output", "call_id": call.id, "output": result}
            for call, result in zip(turn.reply.tool_calls, turn.results, strict=True)
        )
    return wire


def response_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Chat-completions function specs in the Responses form: the function's fields at the
    top, and `strict` off -- on, every parameter would have to be required."""
    return [{"type": "function", **tool.get("function", {}), "strict": False} for tool in tools]


def _unthought(text: str) -> tuple[str, str]:
    """(the thinking, the reply) of a reply that carries the model's thinking before a closing
    </think>, as a server that did not parse it hands it back; ("", text) otherwise."""
    head, mark, tail = text.rpartition("</think>")
    if not mark:
        return "", text
    return head.replace("<think>", "").strip(), tail.strip()


def _call(call_id: object, name: object, text: object, index: int) -> ToolCall:
    """One call of the model's; arguments that are no JSON object come through under
    "_raw", for the tool to say so."""
    raw = text or "{}"
    arguments: object = raw
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError:
            arguments = None
    if not isinstance(arguments, dict):
        arguments = {"_raw": str(raw)}
    return ToolCall(id=str(call_id or f"call_{index}"), name=str(name or ""), arguments=arguments)


def _tool_calls(raw: Sequence[dict[str, Any]]) -> list[ToolCall]:
    """The model's calls as the chat wire gives them."""
    out: list[ToolCall] = []
    for i, call in enumerate(raw):
        function = call.get("function") or {}
        out.append(_call(call.get("id"), function.get("name"), function.get("arguments"), i))
    return out


@dataclass
class _Output:
    """What a response's output items say: the reply, the reasoning's summary, the calls to
    run, and a refusal's words."""

    text: str
    summary: str
    calls: list[ToolCall]
    refusal: str


def _output_read(items: Iterable[dict[str, Any]]) -> _Output:
    """The Responses wire's output items read. A preamble -- a message of phase "commentary",
    written before tool calls in a tool-heavy turn -- is no part of the reply: it goes back
    with the items, and is never said. A call not "completed" (the token budget spent inside
    its arguments) is not run."""
    replies: list[str] = []
    summary: list[str] = []
    refusals: list[str] = []
    calls: list[ToolCall] = []
    for item in items:
        kind = item.get("type")
        if kind == "message":
            parts = item.get("content") or []
            refusals.extend(
                str(p.get("refusal") or "") for p in parts if p.get("type") == "refusal"
            )
            if item.get("phase") != "commentary":
                replies.append(
                    "".join(
                        str(p.get("text") or "") for p in parts if p.get("type") == "output_text"
                    )
                )
        elif kind == "reasoning":
            summary.extend(str(part.get("text") or "") for part in item.get("summary") or [])
        elif kind == "function_call" and item.get("status") in (None, "completed"):
            calls.append(
                _call(item.get("call_id"), item.get("name"), item.get("arguments"), len(calls))
            )
    return _Output(
        text="\n\n".join(r.strip() for r in replies if r.strip()),
        summary="\n".join(s for s in summary if s).strip(),
        calls=calls,
        refusal=" ".join(r for r in refusals if r).strip(),
    )


def _response_stats(response: dict[str, Any], out: _Output) -> dict[str, Any]:
    """A response's usage in the chat wire's terms, and its end as a finish reason: "stop",
    "tool_calls", "length" for the token budget, or what else cut it short; a refusal's
    words when the model refused."""
    usage = response.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    status = response.get("status")
    if status == "completed":
        finish = "tool_calls" if out.calls else "stop"
    elif status == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        finish = "length" if reason == "max_output_tokens" else reason
    else:
        finish = status
    stats = {
        "prompt_tokens": usage.get("input_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "cache_write_tokens": details.get("cache_write_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        "finish_reason": finish,
    }
    if out.refusal:
        stats["refusal"] = out.refusal
    return stats


class OpenAICompatProvider:
    # Function calling is part of both APIs.
    tools_ok = True

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        thinking_control: ThinkingControl = "none",
        reasoning_effort: str | None = None,
        wire: Wire = "chat",
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
        self.wire: Wire = wire
        self.extra_body = dict(extra_body or {})
        self.timeout_s = timeout_s
        self._api_key = api_key
        self._transport = transport
        # The session's tokens over every call, shared with the provider's copies at
        # other efforts (llm.usage): one figure for the panel.
        self.usage = usage if usage is not None else Usage()
        if self.usage.model is None:
            self.usage.model = model
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
            wire=self.wire,
            timeout_s=self.timeout_s,
            extra_body=self.extra_body,
            transport=self._transport,
            usage=self.usage,
        )

    @property
    def _endpoint(self) -> str:
        return f"{self.base_url}/{'responses' if self.wire == 'responses' else 'chat/completions'}"

    def _lost(self, body: dict[str, Any], completion_chars: int, exc: Exception) -> None:
        """A call that reached the server and then failed -- a read timeout, a dropped
        connection -- is billed all the same: counted, estimated from what was sent and
        what came back."""
        self.usage.record(
            {"lost": type(exc).__name__},
            prompt_chars=self._chars(body),
            completion_chars=completion_chars,
        )

    @staticmethod
    def _chars(body: dict[str, Any]) -> int:
        """The prompt's size in characters, for the estimate when no usage comes back."""
        parts = body.get("messages") or body.get("input") or []
        return len(str(body.get("instructions") or "")) + sum(
            len(str(m.get("content") or m.get("output") or m.get("arguments") or "")) for m in parts
        )

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
        tools: Sequence[dict[str, Any]] = (),
        turns: Sequence[ToolRound] = (),
        tool_choice: str | None = None,
    ) -> dict[str, Any]:
        if self.wire == "responses":
            return self._responses_body(
                system,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                turns=turns,
                tool_choice=tool_choice,
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": build_messages(
                system, messages, thinking=thinking, control=self.thinking_control, turns=turns
            ),
            **self.extra_body,
        }
        if tools:
            body["tools"] = list(tools)
            if tool_choice:
                body["tool_choice"] = tool_choice
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

    def _responses_body(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        temperature: float,
        tools: Sequence[dict[str, Any]],
        turns: Sequence[ToolRound],
        tool_choice: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "instructions": system,
            "input": response_input(messages, turns),
            # Nothing kept on the server: the reasoning travels encrypted with the items.
            "store": False,
            **self.extra_body,
        }
        if tools:
            body["tools"] = response_tools(tools)
            if tool_choice:
                body["tool_choice"] = tool_choice
        headroom = (
            REASONING_HEADROOM if self.thinking_control == "effort" and self.reasoning_effort else 0
        )
        body["max_output_tokens"] = max(MIN_OUTPUT_TOKENS, max_tokens + headroom)
        if self.thinking_control == "effort":
            if self.reasoning_effort:
                body["reasoning"] = {"effort": self.reasoning_effort}
                body["include"] = ["reasoning.encrypted_content"]
        else:
            body["temperature"] = temperature
        return body

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
        tools: Sequence[dict[str, Any]] = (),
        turns: Sequence[ToolRound] = (),
        tool_choice: str | None = None,
    ) -> Completion:
        body = self.request_body(
            system,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            tools=tools,
            turns=turns,
            tool_choice=tool_choice,
        )
        started = time.perf_counter()
        try:
            response = self._client.post(self._endpoint, json=body)
        except _SENT_LOST as exc:
            self._lost(body, 0, exc)
            raise
        if response.status_code != 200:
            # The status code stays in the message: the callers' backoff keys on "429".
            raise ProviderError(response.status_code, response.text, retry_after(response.headers))
        data = response.json()
        elapsed = time.perf_counter() - started
        if self.wire == "responses":
            if data.get("status") == "failed":
                error = data.get("error") or {}
                raise RuntimeError(f"failed: {error.get('code')}: {error.get('message')}")
            items = list(data.get("output") or [])
            out = _output_read(items)
            stats = {"elapsed_s": round(elapsed, 3), **_response_stats(data, out)}
            self.usage.record(
                stats,
                prompt_chars=self._chars(body),
                completion_chars=len(out.text) + len(out.summary),
            )
            return Completion(
                text=out.text, reasoning=out.summary, stats=stats, tool_calls=out.calls, items=items
            )

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        stats = {
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "cache_write_tokens": prompt_details.get("cache_write_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "finish_reason": choice.get("finish_reason"),
        }
        thought, text = _unthought((message.get("content") or "").strip())
        reasoning = "\n".join(
            x for x in ((message.get("reasoning_content") or "").strip(), thought) if x
        )
        self.usage.record(
            stats, prompt_chars=self._chars(body), completion_chars=len(text) + len(reasoning)
        )
        return Completion(
            text=text,
            reasoning=reasoning,
            stats=stats,
            tool_calls=_tool_calls(message.get("tool_calls") or []),
        )

    def chat_stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
        on_delta: Callable[[str], bool],
        tools: Sequence[dict[str, Any]] = (),
        turns: Sequence[ToolRound] = (),
        tool_choice: str | None = None,
    ) -> Completion:
        """The reply streamed: every visible delta to `on_delta` as it arrives, the first
        one's latency and the stream's usage in the stats. A callback returning False
        closes the stream; the usage is then whatever the server had sent, usually
        nothing, and finish_reason is "stopped". Tool calls arrive in pieces, by index,
        and are put together.

        With tools, LM Studio does not continue the prefilled message: the model thinks again,
        and its thinking comes back as the reply's first words, up to a closing </think> (Tai's
        puzzle replayed, 2026-09-28). Nothing of it may reach the voice, so under "prefill" a
        reply with tools is read whole, its thinking split off, and handed on at once."""
        if tools and not thinking and self.thinking_control == "prefill":
            completion = self.chat(
                system,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                tools=tools,
                turns=turns,
                tool_choice=tool_choice,
            )
            if completion.text:
                on_delta(completion.text)
            return completion
        body = self.request_body(
            system,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking,
            tools=tools,
            turns=turns,
            tool_choice=tool_choice,
        )
        if self.wire == "responses":
            return self._responses_stream({**body, "stream": True}, on_delta)
        body = {**body, "stream": True, "stream_options": {"include_usage": True}}
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        reasoning: list[str] = []
        calls: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish: str | None = None
        stopped = False
        try:
            with self._client.stream("POST", self._endpoint, json=body) as response:
                if response.status_code != 200:
                    text = response.read().decode("utf-8", "replace")
                    raise ProviderError(response.status_code, text, retry_after(response.headers))
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
                        for piece in delta.get("tool_calls") or []:
                            call = calls.setdefault(
                                int(piece.get("index") or 0),
                                {"id": "", "function": {"name": "", "arguments": ""}},
                            )
                            call["id"] = piece.get("id") or call["id"]
                            function = piece.get("function") or {}
                            call["function"]["name"] += function.get("name") or ""
                            call["function"]["arguments"] += function.get("arguments") or ""
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
        except _SENT_LOST as exc:
            self._lost(body, len("".join(parts)), exc)
            raise
        elapsed = time.perf_counter() - started
        completion_details = usage.get("completion_tokens_details") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        stats = {
            "elapsed_s": round(elapsed, 3),
            "first_token_s": round(first, 3) if first is not None else None,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": prompt_details.get("cached_tokens"),
            "cache_write_tokens": prompt_details.get("cache_write_tokens"),
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
        return Completion(
            text=text,
            reasoning=thought,
            stats=stats,
            tool_calls=_tool_calls([calls[i] for i in sorted(calls)]),
        )

    def _responses_stream(
        self, body: dict[str, Any], on_delta: Callable[[str], bool]
    ) -> Completion:
        """`chat_stream` on the Responses wire: the text deltas to `on_delta`, but for a
        preamble's (`_output_read`); the output items, the usage and the end from the completed
        response (or, stopped early, the items finished so far). An error in the stream
        raises, as a refused request does, and so does a stream that ends unfinished: its
        text is cut off where the connection was."""
        started = time.perf_counter()
        first: float | None = None
        parts: list[str] = []
        items: dict[int, dict[str, Any]] = {}
        preambles: set[int] = set()  # the output items that are preambles, by index
        final: dict[str, Any] = {}
        stopped = False
        try:
            with self._client.stream("POST", self._endpoint, json=body) as response:
                if response.status_code != 200:
                    text = response.read().decode("utf-8", "replace")
                    raise ProviderError(response.status_code, text, retry_after(response.headers))
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        event = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    if kind == "response.output_item.added":
                        if (event.get("item") or {}).get("phase") == "commentary":
                            preambles.add(int(event.get("output_index") or 0))
                    elif kind == "response.output_text.delta":
                        text = str(event.get("delta") or "")
                        if not text or int(event.get("output_index") or 0) in preambles:
                            continue
                        if first is None:
                            first = time.perf_counter() - started
                        parts.append(text)
                        if not on_delta(text):
                            stopped = True
                            break
                    elif kind == "response.output_item.done":
                        items[int(event.get("output_index") or 0)] = event.get("item") or {}
                    elif kind in ("response.completed", "response.incomplete"):
                        final = event.get("response") or {}
                        break
                    elif kind == "response.failed":
                        error = (event.get("response") or {}).get("error") or {}
                        raise RuntimeError(f"failed: {error.get('code')}: {error.get('message')}")
                    elif kind == "error":
                        raise RuntimeError(f"error: {event.get('code')}: {event.get('message')}")
        except _SENT_LOST as exc:
            self._lost(body, len("".join(parts)), exc)
            raise
        if not final and not stopped:
            raise RuntimeError("stream ended before the response completed")
        elapsed = time.perf_counter() - started
        output = list(final.get("output") or [items[i] for i in sorted(items)])
        out = _output_read(output)
        # Stopped early, the text is what was handed on: the message has no item yet.
        text = out.text if final else "".join(parts).strip()
        stats = {
            "elapsed_s": round(elapsed, 3),
            "first_token_s": round(first, 3) if first is not None else None,
            **_response_stats(final, out),
            "streamed": True,
        }
        if stopped:
            stats["finish_reason"] = "stopped"
        self.usage.record(
            stats, prompt_chars=self._chars(body), completion_chars=len(text) + len(out.summary)
        )
        return Completion(
            text=text, reasoning=out.summary, stats=stats, tool_calls=out.calls, items=output
        )

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
    "Wire",
    "build_messages",
    "chat_turns",
    "response_input",
    "response_tools",
]
