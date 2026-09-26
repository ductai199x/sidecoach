"""The models the evaluation tools call, by a short spec, and the wrapper that puts every
call on the ledger. One provider class serves every endpoint (sidecoach.llm); the specs:

  none                       no model: the coach's templates (free)
  local                      the app's own server (sidecoach.toml)
  dashscope | flash          DashScope qwen3.8-max | qwen3.8-flash
  openai[:<model>[@effort]]  OpenAI, e.g. openai:gpt-5.6-luna@medium; the effort is the
                             model's setting (none turns reasoning off)
  <preset>:<model>           any preset with another model

A paid model is only ever called through `Metered`: the ledger's budget is checked before
each call (judge.BudgetExceeded stops the run) and each call is written down with its
tokens and dollars.
"""

from __future__ import annotations

import time
from typing import Any

from sidecoach.llm.factory import provider_from_spec
from sidecoach.llm.openai_compat import OpenAICompatProvider


class Metered:
    """A provider whose every call is checked against the ledger's budget first and written
    to the ledger after, under `label` (coach, coach-routine, claims, summariser, ...).

    It has no `chat_stream`: a stream would go around the ledger, and the Speaker falls back
    to `chat` without one. `offline` is passed through, so a metered "none" is still the
    offline coach."""

    def __init__(self, inner, ledger, label: str = "coach") -> None:
        if ledger is None:
            raise ValueError("Metered needs a ledger")
        self.inner, self.ledger, self.label = inner, ledger, label
        self.name, self.model = inner.name, inner.model
        self.offline = bool(getattr(inner, "offline", False))

    @property
    def reasoning_effort(self) -> str | None:
        return getattr(self.inner, "reasoning_effort", None)

    def healthcheck(self):
        return self.inner.healthcheck()

    def with_effort(self, effort: str | None) -> Metered:
        """The same model at another effort, on the same ledger."""
        with_effort = getattr(self.inner, "with_effort", None)
        if with_effort is None:
            return self
        other = with_effort(effort)
        return self if other is self.inner else Metered(other, self.ledger, self.label)

    def chat(self, system, messages, **kw):
        self.ledger.check()
        t0 = time.time()
        completion = self.inner.chat(system, messages, **kw)
        self.ledger.record(
            self.label, completion.stats, time.time() - t0, model=spec_of(self.inner)
        )
        return completion


def spec_of(provider: Any) -> str:
    """The model and its effort, as the ledger names it: 'gpt-5.6-luna@medium'."""
    model = str(getattr(provider, "model", None) or getattr(provider, "name", "") or "")
    effort = getattr(provider, "reasoning_effort", None)
    return f"{model}@{effort}" if effort else model


def is_offline(kind: str) -> bool:
    return kind.strip().lower() in ("none", "offline")


def provider_for(kind: str, *, timeout_s: float) -> Any:
    """The model a spec names (see the module docstring)."""
    if is_offline(kind):
        from sidecoach.llm.offline import OfflineProvider

        return OfflineProvider()
    # The keys from .env, whichever tool asks first: parity.py on its own had never loaded
    # the settings, and its judge found no key (the benchmark's dev baseline, 2026-09-25).
    from sidecoach.config import load_dotenv

    load_dotenv()
    return provider_from_spec(kind, timeout_s=timeout_s)


def coach_provider(kind: str, ledger=None, label: str = "coach"):
    """The coach's (or the writer's, or the summariser's) model for a run, on the ledger.
    A paid model without a ledger is refused: every paid call is on the record."""
    inner = provider_for(kind, timeout_s=600 if kind == "local" else 300)
    if getattr(inner, "offline", False):
        return inner
    if ledger is None:
        raise ValueError(f"{kind}: a model is only called through a ledger (judge.Ledger)")
    return Metered(inner, ledger, label)


def judge_provider(kind: str) -> tuple[Any, bool]:
    """The judge's model and whether it thinks (only the local model does). Its calls go
    through judge._chat_with_backoff, which checks and records them on the ledger."""
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
