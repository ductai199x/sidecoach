import json

import httpx
import pytest

from sidecoach.llm.openai_compat import EMPTY_THINK_PREFILL, OpenAICompatProvider

USER = [{"role": "user", "content": "hi"}]


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


TOOL = {"type": "function", "function": {"name": "analyse", "parameters": {"type": "object"}}}
CALL = {"id": "c1", "type": "function", "function": {"name": "analyse", "arguments": "{}"}}
TURNS = [
    {"role": "assistant", "content": None, "tool_calls": [CALL]},
    {"role": "tool", "tool_call_id": "c1", "content": "From the board now: ..."},
]


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
    assert body["messages"][2:4] == TURNS and body["messages"][-1] == PREFILL
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
