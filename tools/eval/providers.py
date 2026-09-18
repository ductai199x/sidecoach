"""The models the evaluation tools call, by a short spec, and the wrapper that puts every
call on the ledger. One provider class serves every endpoint (sidecoach.llm); the specs:

  local                      the app's own server (sidecoach.toml)
  dashscope | flash          DashScope qwen3.8-max | qwen3.8-flash
  openai[:<model>[@effort]]  OpenAI, e.g. openai:gpt-5.6-luna@medium; the effort is the
                             model's setting (none turns reasoning off)
  <preset>:<model>           any preset with another model
"""

from __future__ import annotations

import time
from typing import Any

from sidecoach.llm.factory import provider_from_spec
from sidecoach.llm.openai_compat import OpenAICompatProvider


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


def provider_for(kind: str, *, timeout_s: float) -> OpenAICompatProvider:
    """The model a spec names (see the module docstring)."""
    return provider_from_spec(kind, timeout_s=timeout_s)


def coach_provider(kind: str, ledger=None):
    """The coach's (or the writer's, or the summariser's) model for a run; on the ledger
    when one is given."""
    inner = provider_for(kind, timeout_s=600 if kind == "local" else 300)
    return Metered(inner, ledger) if ledger is not None else inner


def judge_provider(kind: str) -> tuple[Any, bool]:
    """The judge's model and whether it thinks (only the local model does)."""
    if kind == "local":
        return provider_for(kind, timeout_s=240), True
    return provider_for(kind, timeout_s=300 if kind.startswith("openai") else 120), False


def without_reasoning(provider) -> Any:
    """The same model with reasoning off, for the yes/no checks: an "effort" provider at
    effort none; any other provider as it is (its calls pass thinking=False)."""
    inner = getattr(provider, "inner", provider)
    if isinstance(inner, OpenAICompatProvider):
        return inner.with_effort("none")
    return provider
