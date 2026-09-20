"""What the session's model calls have cost: tokens, and dollars when the prices are known.

One `Usage` per endpoint, shared by every provider object that speaks to it (the coach's
model at its several reasoning efforts, the verifier, the profile's query writer), so the
panel can show one figure for the session. A call whose usage the server did not report
-- a stream closed early, before the usage chunk -- is estimated from the text at four
characters a token, and counted as such.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

# The estimate for a call the server sent no usage for.
CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class Prices:
    """USD per million tokens: fresh prompt, cached prompt, completion (reasoning included).
    None for any of them means the cost cannot be given."""

    prompt: float | None = None
    cached: float | None = None
    completion: float | None = None

    @property
    def known(self) -> bool:
        return self.prompt is not None and self.completion is not None


def prices_for(settings: Any, model: str | None = None) -> Prices:
    """The prices from the LLM settings (config.LLMSettings), or the known ones for
    `model` (the provider's, once the preset has resolved it); unknown otherwise, and
    the cost is not given."""
    llm = settings.llm
    if llm.price_prompt_per_m is not None or llm.price_completion_per_m is not None:
        return Prices(llm.price_prompt_per_m, llm.price_cached_per_m, llm.price_completion_per_m)
    return KNOWN_PRICES.get(str(model or llm.model or ""), Prices())


# Prices the presets know, USD per million tokens (fresh prompt, cached prompt,
# completion, reasoning billed as completion), as OpenAI's model page gave them on
# 2026-09-20 -- the standard tier; a prompt over 272K tokens costs 2x/1.5x, which the
# coach never sends. Settings in sidecoach.toml override these.
KNOWN_PRICES: dict[str, Prices] = {
    "gpt-5.6-luna": Prices(0.20, 0.02, 1.20),
}


class Usage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.estimated = 0  # calls whose tokens are an estimate
        self.prompt = 0
        self.cached = 0  # of the prompt tokens, the ones the server served from its cache
        self.completion = 0
        self.reasoning = 0  # of the completion tokens, the reasoning ones

    def record(
        self, stats: dict[str, Any], *, prompt_chars: int = 0, completion_chars: int = 0
    ) -> None:
        """Add one call. `stats` is the provider's (prompt_tokens, cached_tokens,
        completion_tokens, reasoning_tokens); missing counts are estimated from the
        character counts."""
        prompt = stats.get("prompt_tokens")
        completion = stats.get("completion_tokens")
        estimated = prompt is None or completion is None
        if prompt is None:
            prompt = int(prompt_chars / CHARS_PER_TOKEN)
        if completion is None:
            completion = int(completion_chars / CHARS_PER_TOKEN)
        with self._lock:
            self.calls += 1
            self.estimated += int(estimated)
            self.prompt += int(prompt)
            self.cached += int(stats.get("cached_tokens") or 0)
            self.completion += int(completion)
            self.reasoning += int(stats.get("reasoning_tokens") or 0)

    def cost(self, prices: Prices) -> float | None:
        """Dollars at `prices`, or None when they are not known."""
        if not prices.known:
            return None
        assert prices.prompt is not None and prices.completion is not None
        with self._lock:
            fresh = max(0, self.prompt - self.cached)
            cached_price = prices.cached if prices.cached is not None else prices.prompt
            return (
                fresh / 1e6 * prices.prompt
                + self.cached / 1e6 * cached_price
                + self.completion / 1e6 * prices.completion
            )

    def summary(self, prices: Prices) -> dict[str, Any]:
        """For the panel: the counts, and the cost when it can be given."""
        with self._lock:
            out: dict[str, Any] = {
                "calls": self.calls,
                "estimated": self.estimated,
                "prompt": self.prompt,
                "cached": self.cached,
                "completion": self.completion,
                "reasoning": self.reasoning,
            }
        out["cost"] = self.cost(prices)
        return out
