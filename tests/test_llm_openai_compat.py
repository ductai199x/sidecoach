import json

import httpx

from sidecoach.llm.openai_compat import EMPTY_THINK_PREFILL, OpenAICompatProvider, build_messages

USER = [{"role": "user", "content": "hi"}]


def test_prefill_strategy_appends_empty_think_block() -> None:
    wire = build_messages("sys", USER, thinking=False, control="prefill")  # type: ignore[arg-type]
    assert wire[0] == {"role": "system", "content": "sys"}
    assert wire[-1] == {"role": "assistant", "content": EMPTY_THINK_PREFILL}


def test_prefill_omitted_when_thinking_on() -> None:
    wire = build_messages("sys", USER, thinking=True, control="prefill")  # type: ignore[arg-type]
    assert wire[-1] == {"role": "user", "content": "hi"}


def test_param_and_none_strategies_never_prefill() -> None:
    for control in ("param", "none"):
        wire = build_messages("sys", USER, thinking=False, control=control)  # type: ignore[arg-type]
        assert wire[-1] == {"role": "user", "content": "hi"}


def test_param_strategy_sets_enable_thinking() -> None:
    provider = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control="param")
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=False)  # type: ignore[arg-type]
    assert body["enable_thinking"] is False
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=True)  # type: ignore[arg-type]
    assert body["enable_thinking"] is True


def test_prefill_strategy_does_not_send_enable_thinking() -> None:
    provider = OpenAICompatProvider("x", "http://h/v1", "m", thinking_control="prefill")
    body = provider.request_body("sys", USER, max_tokens=10, temperature=0.1, thinking=False)  # type: ignore[arg-type]
    assert "enable_thinking" not in body


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
    import pytest

    with pytest.raises(RuntimeError, match="429"):
        failing.complete("sys", "hi")
