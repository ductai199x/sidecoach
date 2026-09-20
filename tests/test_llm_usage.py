"""The session's model calls counted on the provider: tokens from the server's usage, an
estimate when a stream was closed before it came, one figure across the provider's
copies at other efforts, and dollars only at known prices."""

from __future__ import annotations

import json

import httpx

from sidecoach.config import LLMSettings, Settings
from sidecoach.llm.openai_compat import OpenAICompatProvider
from sidecoach.llm.usage import Prices, Usage, prices_for


def _provider(handler) -> OpenAICompatProvider:
    return OpenAICompatProvider(
        "test",
        "http://llm.test/v1",
        "m",
        thinking_control="effort",
        reasoning_effort="medium",
        transport=httpx.MockTransport(handler),
    )


def test_usage_adds_up_over_calls_and_effort_copies() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Fine."}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_tokens_details": {"cached_tokens": 800},
                    "completion_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            },
        )

    provider = _provider(handler)
    routine = provider.with_effort("none")
    assert routine is not provider and routine.usage is provider.usage
    provider.chat("s", [{"role": "user", "content": "hi"}])
    routine.chat("s", [{"role": "user", "content": "hi"}])
    u = provider.usage
    assert (u.calls, u.prompt, u.cached, u.completion, u.reasoning) == (2, 2000, 1600, 100, 60)
    assert u.estimated == 0
    assert u.cost(Prices()) is None, "no prices, no dollars"
    # 400 fresh at $1, 1600 cached at $0.1, 100 completion at $8.
    assert abs((u.cost(Prices(1.0, 0.1, 8.0)) or 0) - (0.0004 + 0.00016 + 0.0008)) < 1e-9
    assert abs((u.cost(Prices(1.0, None, 8.0)) or 0) - (0.002 + 0.0008)) < 1e-9, (
        "cached at the fresh price"
    )
    summary = u.summary(Prices(1.0, 0.1, 8.0))
    assert summary["calls"] == 2 and summary["cost"] is not None


def test_a_stream_closed_early_is_estimated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        lines = []
        for word in ["One.", " Two.", " Three."]:
            chunk = {"choices": [{"delta": {"content": word}}]}
            lines.append("data: " + json.dumps(chunk))
        lines.append(
            "data: "
            + json.dumps({"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 9}})
        )
        lines.append("data: [DONE]")
        return httpx.Response(200, content="\n".join(lines).encode())

    provider = _provider(handler)
    heard: list[str] = []
    provider.chat_stream(
        "s",
        [{"role": "user", "content": "x" * 400}],
        on_delta=lambda t: heard.append(t) or len(heard) < 1,
    )
    u = provider.usage
    assert u.calls == 1 and u.estimated == 1, "the usage chunk never came"
    assert u.prompt == 100, "four characters a token over the messages"
    assert u.completion == 1  # "One." is four characters
    provider.chat_stream("s", [{"role": "user", "content": "x"}], on_delta=lambda t: True)
    assert u.calls == 2 and u.estimated == 1 and u.prompt == 600 and u.completion == 10


def test_prices_come_from_the_settings_or_the_known_table() -> None:
    luna = prices_for(Settings(), "gpt-5.6-luna")
    assert luna.known and (luna.prompt, luna.cached, luna.completion) == (0.20, 0.02, 1.20)
    assert not prices_for(Settings(), "some-other-model").known
    settings = Settings(llm=LLMSettings(price_prompt_per_m=1.25, price_completion_per_m=10.0))
    prices = prices_for(settings, "anything")
    assert prices.known and prices.cached is None
    usage = Usage()
    usage.record({"prompt_tokens": 1_000_000, "completion_tokens": 100_000})
    assert abs((usage.cost(prices) or 0) - 2.25) < 1e-9
