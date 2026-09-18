"""The models the evaluation tools call, by a short spec, and the wrapper that puts every
call on the ledger.

  local                      the app's own server (sidecoach.toml)
  dashscope | flash          DashScope qwen3.8-max | qwen3.8-flash
  openai:<model>[@effort]    an OpenAI reasoning model, e.g. openai:gpt-5.6-luna@medium;
                             the effort applies to every call (none turns reasoning off)
"""

from __future__ import annotations

import os
import time
from typing import Any

from sidecoach.config import LLMSettings, load_dotenv, load_settings
from sidecoach.llm.base import Completion
from sidecoach.llm.factory import make_provider

REASONING_HEADROOM = 4000  # tokens of reasoning allowed on top of the caller's answer budget


class OpenAIReasoning:
    """OpenAI's chat completions for a reasoning model: `max_completion_tokens` instead of
    `max_tokens`, no temperature, `reasoning_effort` when set. Reasoning text is not
    returned; its token count is, and so is the cached share of the prompt."""

    def __init__(self, model: str, effort: str | None, timeout_s: float = 900.0) -> None:
        os.environ.update(load_dotenv())
        self.name, self.model, self.effort, self.timeout_s = "openai", model, effort, timeout_s
        self.key = os.environ["OPENAI_API_KEY"]

    def healthcheck(self) -> tuple[bool, str]:
        return True, self.model

    def chat(self, system, messages, *, max_tokens=1500, temperature=0.1, thinking=False):
        import httpx

        # Reasoning tokens count against max_completion_tokens: a caller's budget is for the
        # answer, so the reasoning gets headroom of its own or a short answer comes back empty.
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens + (REASONING_HEADROOM if self.effort else 0),
        }
        if self.effort:
            body["reasoning_effort"] = self.effort
        t0 = time.time()
        r = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": "Bearer " + self.key},
            json=body,
            timeout=httpx.Timeout(self.timeout_s, connect=30),
        )
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
        j = r.json()
        msg = j["choices"][0]["message"]
        usage = j.get("usage", {})
        stats = {
            "elapsed_s": round(time.time() - t0, 3),
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            ),
            "finish_reason": j["choices"][0].get("finish_reason"),
        }
        return Completion(text=msg.get("content") or "", reasoning="", stats=stats)


class Metered:
    """A provider whose every call is written to the ledger, under `label`."""

    def __init__(self, inner, ledger, label: str = "coach") -> None:
        self.inner, self.ledger, self.label = inner, ledger, label
        self.name, self.model = inner.name, inner.model

    def healthcheck(self):
        return self.inner.healthcheck()

    def chat(self, system, messages, **kw):
        t0 = time.time()
        completion = self.inner.chat(system, messages, **kw)
        self.ledger.record(self.label, completion.stats, time.time() - t0)
        return completion


def provider_for(kind: str, *, timeout_s: float):
    """The model a spec names (see the module docstring)."""
    if kind == "dashscope":
        return make_provider(LLMSettings(provider="dashscope"), timeout_s=timeout_s)
    if kind == "flash":
        return make_provider(
            LLMSettings(provider="dashscope", model="qwen3.8-flash"), timeout_s=timeout_s
        )
    if kind.startswith("openai:"):
        model, _, effort = kind.split(":", 1)[1].partition("@")
        return OpenAIReasoning(model, effort or None, timeout_s=timeout_s)
    return make_provider(load_settings().llm, timeout_s=timeout_s)


def coach_provider(kind: str, ledger=None):
    """The coach's (or the writer's, or the summariser's) model for a run; on the ledger
    when one is given."""
    inner = provider_for(kind, timeout_s=600 if kind == "local" else 300)
    return Metered(inner, ledger) if ledger is not None else inner


def judge_provider(kind: str) -> tuple[Any, bool]:
    """The judge's model and whether it thinks (only the local model does)."""
    if kind == "local":
        return provider_for(kind, timeout_s=240), True
    return provider_for(kind, timeout_s=300 if kind.startswith("openai:") else 120), False
