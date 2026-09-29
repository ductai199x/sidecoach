"""Record what the desktop's LLM layer does with a request and a reply, for the Kotlin.

`llm/openai_compat.py` is a provider over any OpenAI-compatible endpoint and `llm/usage.py`
the ledger behind it; everything about them except the socket is pure. So the wire body is
built and canonicalised here, the server's replies are canned and fed through the real
`chat` and `chat_stream` with an httpx MockTransport, and the ledger is driven by hand.
`llm/factory.py` is resolved against a fake environment.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_llm_fixtures.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import httpx  # noqa: E402

from sidecoach.config import LLMSettings  # noqa: E402
from sidecoach.llm import factory as F  # noqa: E402
from sidecoach.llm import usage as U  # noqa: E402
from sidecoach.llm.openai_compat import OpenAICompatProvider, build_messages  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"

# Timings are wall-clock; everything else about a call is compared.
TIMINGS = ("elapsed_s", "first_token_s")


def canon(value: Any) -> str:
    """A body as one string, keys sorted, so two JSON writers agree."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stats_without_timings(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in stats.items() if k not in TIMINGS}


# --- the message lists every control is asked to build ---------------------------------------

CONVERSATIONS: list[list[dict[str, str]]] = [
    [],
    [{"role": "user", "content": "what about Nf3?"}],
    [
        {"role": "user", "content": "is my knight safe?"},
        {"role": "assistant", "content": "Which knight do you mean?"},
        {"role": "user", "content": "the one on f6"},
    ],
    [{"role": "user", "content": ""}],
]


# --- the servers' replies --------------------------------------------------------------------

CHAT_REPLIES: list[tuple[str, int, str]] = [
    (
        "full",
        200,
        json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "  Nf3 develops and eyes e5.  ",
                            "reasoning_content": " the knight is the piece ",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 812,
                    "completion_tokens": 47,
                    "prompt_tokens_details": {"cached_tokens": 768},
                    "completion_tokens_details": {"reasoning_tokens": 320},
                },
            }
        ),
    ),
    (
        "no usage",
        200,
        json.dumps(
            {"choices": [{"message": {"content": "A short answer."}, "finish_reason": "stop"}]}
        ),
    ),
    (
        "half usage",
        200,
        json.dumps(
            {
                "choices": [{"message": {"content": "Half counted."}}],
                "usage": {"prompt_tokens": 100},
            }
        ),
    ),
    (
        "empty content",
        200,
        json.dumps(
            {
                "choices": [{"message": {"content": "", "reasoning_content": "thought only"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 0},
            }
        ),
    ),
    ("no choices", 200, json.dumps({"usage": {"prompt_tokens": 5, "completion_tokens": 5}})),
    ("empty object", 200, "{}"),
    (
        "nulls",
        200,
        json.dumps(
            {
                "choices": [{"message": {"content": None, "reasoning_content": None}}],
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": None},
                },
            }
        ),
    ),
    ("rate limited", 429, '{"error": {"message": "slow down"}}'),
    ("server error", 500, "upstream exploded, and the body is long " + "x" * 400),
]

STREAM_REPLIES: list[tuple[str, int, str, int]] = [
    (
        "full",
        200,
        "\n".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}',
                'data: {"choices":[{"delta":{"content":"Nf3 "}}]}',
                "",
                "not a data line at all",
                'data: {"choices":[{"delta":{"content":"develops."},"finish_reason":"stop"}]}',
                'data: {"usage":{"prompt_tokens":812,"completion_tokens":47,'
                '"prompt_tokens_details":{"cached_tokens":768},'
                '"completion_tokens_details":{"reasoning_tokens":320}},"choices":[]}',
                "data: [DONE]",
            ]
        ),
        99,
    ),
    (
        "stopped early",
        200,
        "\n".join(
            [
                'data: {"choices":[{"delta":{"content":"one "}}]}',
                'data: {"choices":[{"delta":{"content":"two "}}]}',
                'data: {"choices":[{"delta":{"content":"three"}}]}',
                'data: {"usage":{"prompt_tokens":10,"completion_tokens":3},"choices":[]}',
                "data: [DONE]",
            ]
        ),
        2,
    ),
    (
        "broken chunks",
        200,
        "\n".join(
            [
                "data: {not json}",
                'data: {"choices":[{"delta":{}}]}',
                'data: {"choices":[{"delta":{"content":""}}]}',
                'data: {"choices":[{"delta":{"content":"  kept  "}}]}',
                "data: [DONE]",
                'data: {"choices":[{"delta":{"content":"after done"}}]}',
            ]
        ),
        99,
    ),
    ("nothing at all", 200, "", 99),
    ("rate limited", 429, '{"error": "too fast"}', 99),
]


# What a server answers `GET /models`, for the healthcheck: the model served or not, and the
# statuses it refuses with (worded by httpx's `raise_for_status`).
HEALTH_REPLIES: list[tuple[str, int, str]] = [
    ("served", 200, json.dumps({"data": [{"id": "b-model"}, {"id": "a-model"}]})),
    ("not served", 200, json.dumps({"data": [{"id": "b-model"}]})),
    ("no data", 200, "{}"),
    ("unauthorised", 401, '{"error": "bad key"}'),
    ("not found", 404, "nope"),
    ("rate limited", 429, "slow down"),
    ("server error", 500, "boom"),
    ("unavailable", 503, ""),
]


def provider_for(body_of, *, control: str, effort: str | None = None, extra=None):
    """A provider whose transport answers with `body_of(request)`."""

    def handler(request: httpx.Request) -> httpx.Response:
        status, text = body_of(request)
        return httpx.Response(status, content=text.encode("utf-8"))

    return OpenAICompatProvider(
        "test",
        "https://example.test/v1/",
        "a-model",
        api_key="k",
        thinking_control=control,  # type: ignore[arg-type]
        reasoning_effort=effort,
        extra_body=extra,
        transport=httpx.MockTransport(handler),
    )


def usage_json(usage: U.Usage) -> dict[str, Any]:
    return {
        "calls": usage.calls,
        "estimated": usage.estimated,
        "prompt": usage.prompt,
        "cached": usage.cached,
        "completion": usage.completion,
        "reasoning": usage.reasoning,
    }


def prices_json(prices: U.Prices) -> dict[str, Any]:
    return {
        "prompt": prices.prompt,
        "cached": prices.cached,
        "completion": prices.completion,
        "known": prices.known,
    }


# --- the settings the factory is asked to resolve --------------------------------------------

FAKE_ENV = {
    "OPENAI_API_KEY": "sk-openai",
    "DASHSCOPE_API_KEY": "sk-dashscope",
    "CUSTOM_API_KEY": "sk-custom",
    "OPENAI_BASE_URL": "https://proxy.test/v1",
}

SETTINGS: list[tuple[str, dict[str, Any]]] = [
    ("openai default", {"provider": "openai"}),
    ("openai at high", {"provider": "openai", "reasoning_effort": "high"}),
    ("openai other model", {"provider": "openai", "model": "gpt-5.6-mini"}),
    ("openai own url", {"provider": "openai", "base_url": "https://mine.test/v1"}),
    ("local", {"provider": "local"}),
    ("local other model", {"provider": "local", "model": "qwen3.8-4b"}),
    ("dashscope", {"provider": "dashscope"}),
    ("dashscope flash", {"provider": "dashscope", "model": "qwen3.8-flash"}),
    ("none", {"provider": "none"}),
    (
        "custom",
        {
            "provider": "groq",
            "base_url": "https://api.groq.test/openai/v1",
            "model": "llama-3.3-70b",
        },
    ),
    (
        "custom keyless",
        {
            "provider": "groq",
            "base_url": "https://api.groq.test/openai/v1",
            "model": "llama-3.3-70b",
            "api_key_env": "",
        },
    ),
    (
        "custom with control",
        {
            "provider": "groq",
            "base_url": "https://api.groq.test/openai/v1",
            "model": "llama-3.3-70b",
            "thinking_control": "param",
            "timeout_s": 12.5,
            "extra_body": {"top_p": 0.9, "seed": 7},
        },
    ),
    ("unknown provider", {"provider": "groq"}),
    ("missing key", {"provider": "dashscope", "api_key_env": "NOT_SET_ANYWHERE"}),
    ("bad control", {"provider": "openai", "thinking_control": "telepathy"}),
]

SPECS = [
    "local",
    "dashscope",
    "flash",
    "openai",
    "openai:gpt-5.6-mini",
    "openai:gpt-5.6-luna@none",
    "openai:@high",
    "dashscope:qwen3.8-flash",
    "nonsense",
]


def resolved(settings: LLMSettings) -> dict[str, Any]:
    try:
        provider = F.make_provider(settings)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if getattr(provider, "offline", False):
        return {"name": provider.name, "offline": True}
    return {
        "name": provider.name,
        "base_url": provider.base_url,
        "model": provider.model,
        "thinking_control": provider.thinking_control,
        "reasoning_effort": provider.reasoning_effort,
        "timeout_s": provider.timeout_s,
        "extra_body": canon(provider.extra_body),
    }


TEST_PROVIDER = {
    "name": "test",
    "base_url": "https://example.test/v1/",
    "model": "a-model",
    "api_key": "k",
}


def main() -> None:
    payload: dict[str, Any] = {"provider": TEST_PROVIDER}

    # 1. The wire message list.
    built = []
    for control in ("prefill", "param", "effort", "none"):
        for thinking in (True, False):
            for index, conversation in enumerate(CONVERSATIONS):
                built.append(
                    {
                        "control": control,
                        "thinking": thinking,
                        "conversation": index,
                        "wire": canon(
                            build_messages(
                                "SYSTEM", conversation, thinking=thinking, control=control
                            )
                        ),
                    }
                )
    payload["conversations"] = CONVERSATIONS
    payload["build_messages"] = built

    # 2. The request body.
    bodies = []
    cases = [
        ("prefill", None, None),
        ("param", None, None),
        ("effort", "medium", None),
        ("effort", None, None),
        ("none", None, None),
        ("none", None, {"top_p": 0.9, "seed": 7}),
        ("effort", "none", {"model": "overridden", "extra": [1, 2]}),
    ]
    for control, effort, extra in cases:
        provider = provider_for(lambda r: (200, "{}"), control=control, effort=effort, extra=extra)
        for thinking in (True, False):
            for max_tokens, temperature in ((1500, 0.7), (64, 0.0)):
                body = provider.request_body(
                    "SYSTEM",
                    CONVERSATIONS[2],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )
                bodies.append(
                    {
                        "conversation": 2,
                        "control": control,
                        "effort": effort,
                        "extra": canon(extra or {}),
                        "thinking": thinking,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "body": canon(body),
                        "chars": OpenAICompatProvider._chars(body),
                    }
                )
    payload["request_bodies"] = bodies

    # 3. What `chat` makes of each reply.
    chats = []
    for label, status, text in CHAT_REPLIES:
        provider = provider_for(lambda r, s=status, t=text: (s, t), control="none")
        entry: dict[str, Any] = {
            "label": label,
            "status": status,
            "reply": text,
            "conversation": 1,
            "max_tokens": 100,
        }
        try:
            completion = provider.chat("SYSTEM", CONVERSATIONS[1], max_tokens=100)
            entry["text"] = completion.text
            entry["reasoning"] = completion.reasoning
            entry["stats"] = canon(stats_without_timings(completion.stats))
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
        entry["usage"] = usage_json(provider.usage)
        chats.append(entry)
    payload["chats"] = chats

    # 4. What `chat_stream` makes of each stream, including one closed early.
    streams = []
    for label, status, text, keep in STREAM_REPLIES:
        provider = provider_for(lambda r, s=status, t=text: (s, t), control="none")
        seen: list[str] = []

        def on_delta(piece: str, seen=seen, keep=keep) -> bool:
            seen.append(piece)
            return len(seen) < keep

        entry = {
            "label": label,
            "status": status,
            "reply": text,
            "keep": keep,
            "conversation": 1,
            "max_tokens": 100,
        }
        try:
            completion = provider.chat_stream(
                "SYSTEM", CONVERSATIONS[1], max_tokens=100, on_delta=on_delta
            )
            entry["text"] = completion.text
            entry["reasoning"] = completion.reasoning
            entry["stats"] = canon(stats_without_timings(completion.stats))
            entry["deltas"] = list(seen)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["deltas"] = list(seen)
        entry["usage"] = usage_json(provider.usage)
        streams.append(entry)
    payload["streams"] = streams

    # 4c. With tools (desktop c80b64b): the request with the engine tools' specs, the model's
    # earlier calls and their results, and tool_choice; its calls read whole and streamed in
    # pieces; the thinking a server hands back before </think> split off; and, under "prefill",
    # a reply with tools read whole and handed on at once.
    from sidecoach.coach.engine_tools import SPECS as TOOL_SPECS

    tool_turns = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "analyse", "arguments": '{"moves": ["Kg2"]}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "After 4. Kg2 (from the board now): ..."},
    ]
    tool_bodies = []
    for control, effort in (("prefill", None), ("effort", "none"), ("none", None)):
        provider = provider_for(lambda r: (200, "{}"), control=control, effort=effort)
        for choice in (None, "none"):
            body = provider.request_body(
                "SYSTEM",
                CONVERSATIONS[1],
                max_tokens=100,
                temperature=0.1,
                thinking=False,
                tools=TOOL_SPECS,
                turns=tool_turns,
                tool_choice=choice,
            )
            tool_bodies.append(
                {"control": control, "effort": effort, "tool_choice": choice, "body": canon(body)}
            )
    tool_chats = []
    for label, text in (
        (
            "two calls, one unreadable",
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {
                                            "name": "analyse",
                                            "arguments": '{"moves": ["Kg2", "Rg1+"]}',
                                        },
                                    },
                                    {"function": {"name": "square", "arguments": "not json"}},
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            ),
        ),
        (
            "thinking before </think>",
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "<think>draft Kg2.</think>\n\nKg2 is forced; Rg1+ wins.",
                                "reasoning_content": "earlier",
                            }
                        }
                    ]
                }
            ),
        ),
    ):
        provider = provider_for(lambda r, t=text: (200, t), control="none")
        completion = provider.chat("SYSTEM", CONVERSATIONS[1], tools=TOOL_SPECS)
        tool_chats.append(
            {
                "label": label,
                "reply": text,
                "text": completion.text,
                "reasoning": completion.reasoning,
                "calls": [[c.id, c.name, canon(c.arguments)] for c in completion.tool_calls],
            }
        )
    pieces = [
        {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "analyse", "arguments": ""}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '{"moves": '}}]},
        {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "square", "arguments": '{"square"'}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '["Kg2"]}'}}]},
        {"tool_calls": [{"index": 1, "function": {"arguments": ': "g1"}'}}]},
    ]
    stream_text = "\n".join(
        ["data: " + json.dumps({"choices": [{"delta": d}]}) for d in pieces]
        + ["data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}), "data: [DONE]"]
    )
    tool_streams = []
    for control, text in (("none", stream_text), ("prefill", tool_chats[1]["reply"])):
        provider = provider_for(lambda r, t=text: (200, t), control=control)
        heard: list[str] = []
        completion = provider.chat_stream(
            "SYSTEM",
            CONVERSATIONS[1],
            on_delta=lambda piece, heard=heard: heard.append(piece) or True,
            tools=TOOL_SPECS,
        )
        tool_streams.append(
            {
                "control": control,
                "reply": text,
                "text": completion.text,
                "deltas": heard,
                "calls": [[c.id, c.name, canon(c.arguments)] for c in completion.tool_calls],
            }
        )
    payload["tools"] = {
        "specs": canon(TOOL_SPECS),
        "turns": canon(tool_turns),
        "bodies": tool_bodies,
        "chats": tool_chats,
        "streams": tool_streams,
    }

    # 4b. What the healthcheck says of each answer to `GET /models`.
    healths = []
    for label, status, text in HEALTH_REPLIES:
        provider = provider_for(lambda r, s=status, t=text: (s, t), control="none")
        ok, detail = provider.healthcheck()
        healths.append({"label": label, "status": status, "reply": text, "ok": ok, "detail": detail})
    payload["healthchecks"] = healths

    # 5. The ledger by hand: every shape of stats a server has sent.
    records = [
        ({"prompt_tokens": 100, "completion_tokens": 20}, 0, 0),
        ({"prompt_tokens": 100, "cached_tokens": 80, "completion_tokens": 20}, 0, 0),
        ({"prompt_tokens": 1, "completion_tokens": 1, "reasoning_tokens": 500}, 0, 0),
        ({}, 4000, 400),
        ({"prompt_tokens": 7}, 4000, 401),
        ({"completion_tokens": 7}, 4002, 400),
        ({"prompt_tokens": None, "cached_tokens": None}, 3, 3),
        ({"prompt_tokens": 10.7, "completion_tokens": 3.2, "cached_tokens": 2.9}, 0, 0),
    ]
    price_sets = [
        U.Prices(),
        U.Prices(0.20, 0.02, 1.20),
        U.Prices(1.0, None, 2.0),
        U.Prices(None, 0.5, 2.0),
    ]
    ledger = U.Usage()
    steps = []
    for stats, prompt_chars, completion_chars in records:
        ledger.record(stats, prompt_chars=prompt_chars, completion_chars=completion_chars)
        steps.append(
            {
                "stats": canon(stats),
                "prompt_chars": prompt_chars,
                "completion_chars": completion_chars,
                "after": usage_json(ledger),
                "summaries": [canon(ledger.summary(p)) for p in price_sets],
            }
        )
    payload["prices"] = [prices_json(p) for p in price_sets]
    payload["ledger"] = steps

    # 6. Several ledgers as one.
    combos = [
        [],
        [canon({"calls": 1, "prompt": 10, "cost": 0.5})],
        [
            canon({"calls": 1, "estimated": 1, "prompt": 10, "cached": 2, "completion": 3,
                   "reasoning": 1, "cost": 0.5}),
            canon({"calls": 2, "estimated": 0, "prompt": 20, "cached": 0, "completion": 6,
                   "reasoning": 0, "cost": 1.25}),
        ],
        [
            canon({"calls": 1, "prompt": 10, "cost": 0.5}),
            canon({"calls": 1, "prompt": 10, "cost": None}),
        ],
        [canon({"calls": 1, "prompt": 10})],
    ]
    payload["combines"] = [
        {"parts": parts, "out": canon(U.combine([json.loads(one) for one in parts]))}
        for parts in combos
    ]

    # 7. Prices from the settings.
    price_settings = [
        {"provider": "openai"},
        {"provider": "openai", "model": "gpt-5.6-luna"},
        {"provider": "openai", "model": "something-else"},
        {"provider": "openai", "price_prompt_per_m": 3.0},
        {"provider": "openai", "price_prompt_per_m": 3.0, "price_cached_per_m": 0.3,
         "price_completion_per_m": 9.0},
        {"provider": "openai", "price_completion_per_m": 9.0},
    ]

    class Holder:
        def __init__(self, llm: LLMSettings) -> None:
            self.llm = llm

    payload["prices_for"] = [
        {
            "settings": canon(one),
            "model": model,
            "prices": prices_json(U.prices_for(Holder(LLMSettings(**one)), model)),
        }
        for one in price_settings
        for model in (None, "gpt-5.6-luna", "unknown-model")
    ]

    # 8. What the factory resolves.
    # Overlaid, not swapped: clearing the environment takes httpx's SSL settings with it.
    touched = [*FAKE_ENV, "NOT_SET_ANYWHERE"]
    saved = {key: os.environ.get(key) for key in touched}
    os.environ.update(FAKE_ENV)
    os.environ.pop("NOT_SET_ANYWHERE", None)
    try:
        payload["env"] = FAKE_ENV
        payload["settings"] = [
            {"label": label, "settings": canon(one), "resolved": resolved(LLMSettings(**one))}
            for label, one in SETTINGS
        ]
        specs = []
        for spec in SPECS:
            try:
                provider = F.provider_from_spec(spec)
                specs.append(
                    {
                        "spec": spec,
                        "resolved": {
                            "name": provider.name,
                            "base_url": provider.base_url,
                            "model": provider.model,
                            "thinking_control": provider.thinking_control,
                            "reasoning_effort": provider.reasoning_effort,
                            "timeout_s": provider.timeout_s,
                            "extra_body": canon(provider.extra_body),
                        },
                    }
                )
            except Exception as exc:  # noqa: BLE001
                specs.append({"spec": spec, "resolved": {"error": f"{type(exc).__name__}: {exc}"}})
        payload["specs"] = specs
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "llm.json").write_text(json.dumps(payload), encoding="utf-8")
    size = (OUT / "llm.json").stat().st_size
    print(
        f"{len(built)} message lists, {len(bodies)} bodies, {len(chats)} chats, "
        f"{len(streams)} streams, {len(steps)} ledger steps, {len(SETTINGS)} settings "
        f"-> {OUT / 'llm.json'} ({size / 1e3:.0f} kB)"
    )


if __name__ == "__main__":
    main()
