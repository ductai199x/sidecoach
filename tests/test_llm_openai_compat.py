import json
from unittest.mock import patch

import httpx
import pytest

from sidecoach.llm.base import ChatMessage, Completion, ProviderError, ToolCall, ToolRound
from sidecoach.llm.openai_compat import EMPTY_THINK_PREFILL, OpenAICompatProvider

USER: list[ChatMessage] = [{"role": "user", "content": "hi"}]


PREFILL = {"role": "assistant", "content": EMPTY_THINK_PREFILL}
LAST_USER = {"role": "user", "content": "hi"}


@pytest.mark.parametrize(
    ("control", "thinking", "last", "sent"),
    [
        # LM Studio skips reasoning only for a trailing empty think block; DashScope honours
        # enable_thinking in the body (and is sent true when thinking: the chat test below).
        ("prefill", False, PREFILL, None),
        ("prefill", True, LAST_USER, None),
        ("param", False, LAST_USER, False),
        ("none", False, LAST_USER, None),
    ],
)
def test_only_prefill_ends_on_an_empty_think_block_and_only_param_sends_enable_thinking(
    control: str, thinking: bool, last: dict[str, str], sent: bool | None
) -> None:
    provider = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control=control)  # type: ignore[arg-type]
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=thinking)  # type: ignore[arg-type]
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert body["messages"][-1] == last
    assert body.get("enable_thinking") is sent


def test_chat_parses_content_reasoning_and_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": " answer ", "reasoning_content": "why"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 30,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        )

    provider = OpenAICompatProvider(
        "x",
        "http://h/v1",
        "m",
        api_key="secret",
        thinking_control="param",
        transport=httpx.MockTransport(handler),
    )
    completion = provider.complete("sys", "hi", thinking=True)
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["enable_thinking"] is True  # type: ignore[index]
    assert completion.text == "answer"
    assert completion.reasoning == "why"
    assert completion.stats["reasoning_tokens"] == 20
    assert completion.stats["finish_reason"] == "stop"


def test_effort_strategy_uses_the_reasoning_request_shape() -> None:
    provider = OpenAICompatProvider(
        "x", "http://h/v1", "m", thinking_control="effort", reasoning_effort="medium"
    )
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=False)  # type: ignore[arg-type]
    assert body["reasoning_effort"] == "medium"
    assert body["max_completion_tokens"] == 10 + 4000
    assert "max_tokens" not in body and "temperature" not in body
    assert "enable_thinking" not in body
    # No effort configured: no reasoning, no headroom.
    plain = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control="effort")
    body = plain.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=False)  # type: ignore[arg-type]
    assert body["max_completion_tokens"] == 10 and "reasoning_effort" not in body


def test_an_error_keeps_its_status_for_the_callers_backoff() -> None:
    slow = httpx.MockTransport(lambda request: httpx.Response(429, text="slow down"))
    failing = OpenAICompatProvider("x", "http://h/v1", "m", transport=slow)
    with pytest.raises(RuntimeError, match="429"):
        failing.complete("sys", "hi")


@pytest.mark.parametrize(
    ("headers", "wait"),
    [
        ({"Retry-After": "20"}, 20.0),
        ({"retry-after-ms": "1500", "Retry-After": "2"}, 1.5),
        ({"Retry-After": "Wed, 21 Oct 2015 07:28:30 GMT"}, 30.0),
        ({"Retry-After": "soon"}, None),
        ({}, None),
    ],
)
def test_a_rate_limit_carries_the_servers_retry_after(
    headers: dict[str, str], wait: float | None
) -> None:
    """The wait the server asks for reaches the caller's backoff, on every wire."""
    busy = httpx.MockTransport(
        lambda request: httpx.Response(429, text="slow down", headers=headers)
    )
    now = 1445412480.0  # 07:28:00 GMT that day: the date form is 30 s ahead
    with patch("sidecoach.llm.base.time.time", return_value=now):
        for wire in ("chat", "responses"):
            failing = OpenAICompatProvider("x", "http://h/v1", "m", transport=busy, wire=wire)
            with pytest.raises(ProviderError, match="^429") as plain:
                failing.complete("sys", "hi")
            with pytest.raises(ProviderError, match="^429") as streamed:
                failing.chat_stream(
                    "sys", [{"role": "user", "content": "hi"}], on_delta=lambda _: True
                )
            assert plain.value.retry_after_s == wait == streamed.value.retry_after_s


TOOL = {"type": "function", "function": {"name": "analyse", "parameters": {"type": "object"}}}
CALL = {"id": "c1", "type": "function", "function": {"name": "analyse", "arguments": "{}"}}
ASKED = Completion(text="", tool_calls=[ToolCall(id="c1", name="analyse", arguments={})])
TURNS = [ToolRound(ASKED, ["From the board now: ..."])]


def test_tools_and_their_turns_go_on_the_wire_before_the_prefill() -> None:
    """The model's calls and their results follow the conversation; LM Studio's empty think
    block stays last, or the model would think again after every result."""
    provider = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control="prefill")
    body = provider.request_body(
        "sys",
        USER,
        max_tokens=10,
        temperature=0.1,
        thinking=False,  # type: ignore[arg-type]
        tools=[TOOL],
        turns=TURNS,
        tool_choice="none",
    )
    assert body["tools"] == [TOOL] and body["tool_choice"] == "none"
    assert body["messages"][2:4] == [
        {"role": "assistant", "content": None, "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "From the board now: ..."},
    ]
    assert body["messages"][-1] == PREFILL
    plain = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=False)  # type: ignore[arg-type]
    assert "tools" not in plain and "tool_choice" not in plain


def test_the_models_calls_are_read_whole_and_streamed_in_pieces() -> None:
    def whole(request: httpx.Request) -> httpx.Response:
        call = {**CALL, "function": {"name": "analyse", "arguments": '{"moves": ["Kg2"]}'}}
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                call,
                {**CALL, "id": "c2", "function": {"name": "square", "arguments": "not json"}},
            ],
        }
        return httpx.Response(
            200, json={"choices": [{"message": message, "finish_reason": "tool_calls"}]}
        )

    provider = OpenAICompatProvider("x", "http://h/v1", "m", transport=httpx.MockTransport(whole))
    got = provider.chat("sys", USER, tools=[TOOL])  # type: ignore[arg-type]
    assert got.text == "" and [(c.id, c.name, c.arguments) for c in got.tool_calls] == [
        ("c1", "analyse", {"moves": ["Kg2"]}),
        ("c2", "square", {"_raw": "not json"}),
    ]

    def pieces(request: httpx.Request) -> httpx.Response:
        deltas = [
            {
                "tool_calls": [
                    {"index": 0, "id": "c1", "function": {"name": "analyse", "arguments": ""}}
                ]
            },
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"moves": '}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": '["Kg2", "Rg1+"]}'}}]},
        ]
        lines = ["data: " + json.dumps({"choices": [{"delta": d}]}) for d in deltas]
        lines += [
            "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        return httpx.Response(200, content="\n".join(lines).encode())

    streaming = OpenAICompatProvider("x", "http://h/v1", "m", transport=httpx.MockTransport(pieces))
    heard: list[str] = []
    got = streaming.chat_stream(
        "sys", USER, on_delta=lambda t: heard.append(t) or True, tools=[TOOL]
    )  # type: ignore[arg-type]
    assert heard == [] and got.text == ""
    assert [(c.id, c.name, c.arguments) for c in got.tool_calls] == [
        ("c1", "analyse", {"moves": ["Kg2", "Rg1+"]})
    ]


def test_thinking_handed_back_before_a_closing_think_tag_never_reaches_the_reply() -> None:
    """With tools, LM Studio does not continue the prefilled message and the model's thinking
    comes back as content up to </think> (Tai's puzzle replayed, 2026-09-28): the reply is
    what follows it, and a stream with tools under "prefill" is read whole."""
    content = "Here's how it goes: Kg2, Rg1+. </think>\n\nKg2 is forced; then Rg1+ wins."

    def handler(request: httpx.Request) -> httpx.Response:
        assert "stream" not in json.loads(request.content), "read whole, not streamed"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    provider = OpenAICompatProvider(
        "x", "http://h/v1", "m", thinking_control="prefill", transport=httpx.MockTransport(handler)
    )
    heard: list[str] = []
    got = provider.chat_stream(
        "sys",
        USER,
        on_delta=lambda t: heard.append(t) or True,
        tools=[TOOL],  # type: ignore[arg-type]
    )
    assert got.text == "Kg2 is forced; then Rg1+ wins." and heard == [got.text]
    assert got.reasoning == "Here's how it goes: Kg2, Rg1+."


# ---- OpenAI's Responses API: the items as the API returned them (2026-09-29) --------------
REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "content": [],
    "encrypted_content": "gAAAAABqu-ri...",
    "summary": [],
}
FUNCTION_CALL = {
    "id": "fc_1",
    "type": "function_call",
    "status": "completed",
    "arguments": '{"moves":["e4","e5","Nf3"]}',
    "call_id": "call_1",
    "name": "analyse",
}
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": "About equal."}],
}
USAGE = {
    "input_tokens": 372,
    "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 100},
    "output_tokens": 38,
    "output_tokens_details": {"reasoning_tokens": 9},
    "total_tokens": 410,
}


def _responses(handler=None, effort: str | None = "medium") -> OpenAICompatProvider:  # noqa: ANN001
    return OpenAICompatProvider(
        "openai",
        "http://h/v1",
        "gpt-5.6-luna",
        thinking_control="effort",
        reasoning_effort=effort,
        wire="responses",
        transport=httpx.MockTransport(handler) if handler is not None else None,
    )


def test_the_responses_wire_sends_the_reasoning_back_with_the_tools_results() -> None:
    """Luna at an effort takes function tools only on /v1/responses (chat completions refused
    them, 2026-09-29): stateless, its reasoning comes back encrypted in its output items, which
    go back before what each call returned."""
    call = ToolCall(id="call_1", name="analyse", arguments={"moves": ["e4", "e5", "Nf3"]})
    asked = Completion(text="", tool_calls=[call], items=[REASONING, FUNCTION_CALL])
    body = _responses().request_body(
        "sys",
        USER,
        max_tokens=10,
        temperature=0.1,
        thinking=False,
        tools=[TOOL],
        turns=[ToolRound(asked, ["Standing: about equal."])],
        tool_choice="none",
    )
    assert body["instructions"] == "sys" and "messages" not in body
    assert body["input"] == [
        LAST_USER,
        REASONING,
        FUNCTION_CALL,
        {"type": "function_call_output", "call_id": "call_1", "output": "Standing: about equal."},
    ]
    # The function's fields at the top of the tool, strict off as on the chat wire.
    assert body["tools"] == [
        {"type": "function", "name": "analyse", "parameters": {"type": "object"}, "strict": False}
    ]
    assert body["tool_choice"] == "none" and body["store"] is False
    assert body["reasoning"] == {"effort": "medium"}
    assert body["include"] == ["reasoning.encrypted_content"]
    assert body["max_output_tokens"] == 10 + 4000 and "temperature" not in body
    # A reply with no items of its own (another provider's): its calls written out.
    plain = _responses().request_body(
        "sys", USER, max_tokens=10, temperature=0.1, thinking=False, turns=TURNS
    )
    assert plain["input"][1:] == [
        {"type": "function_call", "call_id": "c1", "name": "analyse", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "From the board now: ..."},
    ]
    assert "tools" not in plain and "tool_choice" not in plain


def test_the_responses_wire_reads_the_calls_the_reply_and_the_usage() -> None:
    replies = [
        {"status": "completed", "output": [REASONING, FUNCTION_CALL], "usage": USAGE},
        {"status": "completed", "output": [MESSAGE], "usage": USAGE},
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [REASONING],
            "usage": USAGE,
        },
    ]
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json=replies[len(paths) - 1])

    provider = _responses(handler)
    asked = provider.chat("sys", USER, tools=[TOOL])
    assert paths == ["/v1/responses"]
    assert [(c.id, c.name, c.arguments) for c in asked.tool_calls] == [
        ("call_1", "analyse", {"moves": ["e4", "e5", "Nf3"]})
    ]
    assert asked.text == "" and asked.items == [REASONING, FUNCTION_CALL]
    assert asked.stats | {"elapsed_s": 0} == {
        "elapsed_s": 0,
        "prompt_tokens": 372,
        "cached_tokens": 100,
        "cache_write_tokens": 0,
        "completion_tokens": 38,
        "reasoning_tokens": 9,
        "finish_reason": "tool_calls",
    }
    answered = provider.chat("sys", USER)
    assert answered.text == "About equal." and answered.stats["finish_reason"] == "stop"
    cut = provider.chat("sys", USER)
    assert cut.text == "" and cut.stats["finish_reason"] == "length"


def _events(*events: dict[str, object]) -> bytes:
    """A stream as the API writes it: each event's type, then its data."""
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


STREAMED = _events(
    {"type": "response.created", "response": {"status": "in_progress", "output": []}},
    *(
        {"type": "response.output_text.delta", "output_index": 0, "delta": piece}
        for piece in ("About", " equal", ".")
    ),
    {"type": "response.output_item.done", "output_index": 0, "item": MESSAGE},
    {
        "type": "response.completed",
        "response": {"status": "completed", "output": [MESSAGE], "usage": USAGE},
    },
)


def test_the_responses_wire_streams_the_text_and_ends_on_the_completed_response() -> None:
    provider = _responses(lambda request: httpx.Response(200, content=STREAMED))
    heard: list[str] = []
    got = provider.chat_stream("sys", USER, on_delta=lambda t: heard.append(t) or True)
    assert heard == ["About", " equal", "."] and got.text == "About equal."
    assert got.items == [MESSAGE] and got.stats["streamed"] and got.stats["first_token_s"] >= 0
    assert got.stats["finish_reason"] == "stop" and got.stats["prompt_tokens"] == 372
    # Closed after the first piece: the text so far, the usage unknown.
    stopped = provider.chat_stream("sys", USER, on_delta=lambda t: False)
    assert stopped.text == "About" and stopped.stats["finish_reason"] == "stopped"
    assert stopped.stats["prompt_tokens"] is None


def test_an_error_in_a_responses_stream_raises() -> None:
    failing = _events(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": "error", "code": "rate_limit_exceeded", "message": "slow down"},
    )
    provider = _responses(lambda request: httpx.Response(200, content=failing))
    with pytest.raises(RuntimeError, match="rate_limit_exceeded: slow down"):
        provider.chat_stream("sys", USER, on_delta=lambda t: True)
    refused = _responses(lambda request: httpx.Response(400, text="bad tools"))
    with pytest.raises(RuntimeError, match="^400: bad tools"):
        refused.chat_stream("sys", USER, on_delta=lambda t: True)
    # Cut off before the response completed: an error, not a reply that ends mid-sentence.
    cut = _events(
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "About"},
    )
    dropped = _responses(lambda request: httpx.Response(200, content=cut))
    with pytest.raises(RuntimeError, match="stream ended before the response completed"):
        dropped.chat_stream("sys", USER, on_delta=lambda t: True)
    # A failed response read whole raises too, as the stream's failure does.
    failed = {"status": "failed", "error": {"code": "server_error", "message": "try again"}}
    whole = _responses(lambda request: httpx.Response(200, json=failed))
    with pytest.raises(RuntimeError, match="failed: server_error: try again"):
        whole.chat("sys", USER)


PREAMBLE = {
    "id": "msg_0",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "phase": "commentary",
    "content": [{"type": "output_text", "annotations": [], "text": "Let me ask the engine."}],
}


def test_a_preamble_before_the_calls_is_never_said_and_goes_back_with_them() -> None:
    """A tool-heavy turn may open with a preamble (phase "commentary"): no part of the reply,
    and never to the voice -- but it goes back with the round, phase and all (OpenAI's
    reasoning guide)."""
    events = _events(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "phase": "commentary", "content": []},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": "Let me ask the engine.",
        },
        {"type": "response.output_item.done", "output_index": 0, "item": PREAMBLE},
        {"type": "response.output_item.done", "output_index": 1, "item": FUNCTION_CALL},
        {
            "type": "response.completed",
            "response": {
                "status": "completed",
                "output": [PREAMBLE, FUNCTION_CALL],
                "usage": USAGE,
            },
        },
    )
    provider = _responses(lambda request: httpx.Response(200, content=events))
    heard: list[str] = []
    got = provider.chat_stream("sys", USER, on_delta=lambda t: heard.append(t) or True)
    assert heard == [] and got.text == ""
    assert [c.name for c in got.tool_calls] == ["analyse"]
    assert got.items == [PREAMBLE, FUNCTION_CALL]
    whole = _responses(
        lambda request: httpx.Response(
            200, json={"status": "completed", "output": [PREAMBLE, MESSAGE], "usage": USAGE}
        )
    )
    assert whole.chat("sys", USER).text == "About equal."


def test_a_call_cut_short_is_not_run_and_a_refusal_is_on_the_record() -> None:
    cut_call = {**FUNCTION_CALL, "status": "incomplete", "arguments": '{"moves":["e4",'}
    refusal = {
        **MESSAGE,
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }
    replies = [
        {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [cut_call],
            "usage": USAGE,
        },
        {"status": "completed", "output": [refusal], "usage": USAGE},
    ]
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        return httpx.Response(200, json=replies[len(seen) - 1])

    provider = _responses(handler)
    cut = provider.chat("sys", USER, tools=[TOOL])
    assert cut.tool_calls == [] and cut.stats["finish_reason"] == "length"
    refused = provider.chat("sys", USER)
    assert refused.text == "" and refused.stats["refusal"] == "I can't help with that."


def test_the_responses_wire_asks_for_at_least_sixteen_tokens() -> None:
    """Fewer output tokens than 16 the API refuses; a provider with no effort gets no
    headroom, and a caller's tiny budget (a yes/no judge) is raised to the floor."""
    body = _responses(effort=None).request_body(
        "sys", USER, max_tokens=5, temperature=0.1, thinking=False
    )
    assert body["max_output_tokens"] == 16 and "reasoning" not in body
