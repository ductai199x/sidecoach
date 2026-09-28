import json

import httpx
import pytest

from sidecoach.llm.openai_compat import EMPTY_THINK_PREFILL, OpenAICompatProvider, build_messages

USER = [{"role": "user", "content": "hi"}]


PREFILL = {"role": "assistant", "content": EMPTY_THINK_PREFILL}
LAST_USER = {"role": "user", "content": "hi"}


@pytest.mark.parametrize(
    ("control", "thinking", "last"),
    [
        # LM Studio skips reasoning only for a trailing empty think block.
        ("prefill", False, PREFILL),
        ("prefill", True, LAST_USER),
        ("param", False, LAST_USER),
        ("none", False, LAST_USER),
    ],
)
def test_only_the_prefill_strategy_ends_on_an_empty_think_block(
    control: str, thinking: bool, last: dict[str, str]
) -> None:
    wire = build_messages("sys", USER, thinking=thinking, control=control)  # type: ignore[arg-type]
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[-1] == last


@pytest.mark.parametrize(
    ("control", "thinking", "sent"),
    [
        # DashScope honours enable_thinking in the body; nothing else is sent it.
        ("param", False, False),
        ("param", True, True),
        ("prefill", False, None),
    ],
)
def test_only_the_param_strategy_sends_enable_thinking(
    control: str, thinking: bool, sent: bool | None
) -> None:
    provider = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control=control)  # type: ignore[arg-type]
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=thinking)  # type: ignore[arg-type]
    if sent is None:
        assert "enable_thinking" not in body
    else:
        assert body["enable_thinking"] is sent


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


def test_chat_reports_cached_tokens_and_keeps_the_status_in_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-fail"):
            return httpx.Response(429, text="slow down")
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 50, "prompt_tokens_details": {"cached_tokens": 40}},
            },
        )

    provider = OpenAICompatProvider("x", "http://h/v1", "m", transport=httpx.MockTransport(handler))
    assert provider.complete("sys", "hi").stats["cached_tokens"] == 40
    failing = OpenAICompatProvider("x", "http://h/v1", "m", transport=httpx.MockTransport(handler))
    failing._client.headers["x-fail"] = "1"
    with pytest.raises(RuntimeError, match="429"):
        failing.complete("sys", "hi")
