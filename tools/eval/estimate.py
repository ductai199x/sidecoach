"""What a run would cost, before any call: the dry run's arithmetic.

The requests and items come from the games played with no model (corpus.build_game with
no provider: the coach's own requests, answered by its templates -- the same requests the
model would get, its own words aside). Prompt sizes are measured on them; the rest are
assumptions from the ledger's history (data/eval/usage.jsonl, 2026-09-17/18): the coach at
luna medium wrote ~200 completion tokens a call; a judged reply took ~13 judge calls, ~1.9
claims and ~1.4 required facts. The estimate is printed with its assumptions, and it is an
estimate: the ledger's budget, checked before every call, is what stops a run.
"""

from __future__ import annotations

from dataclasses import dataclass

# Roughly four characters a token for English and chess notation.
CHARS_PER_TOKEN = 4
# The provider's prompt cache serves a repeated prefix only from this many tokens.
CACHE_MIN_TOKENS = 1024

# From the ledger's history (see the module docstring).
COACH_COMPLETION = 250  # tokens a coach call writes, reasoning included
RETRY_SHARE = 0.15  # the share of coach calls sent back once by the checks
CLAIMS_PER_REPLY = 2.0
CHECKER_CALLS_PER_REPLY = 0.4
VOTES = 3
DECOMPOSE_COMPLETION = 200
CHECKER_COMPLETION = 200
VOTE_COMPLETION = 2
POINT_COMPLETION = 60
QUESTION_TOKENS = 40  # the user message of one yes/no check
# A yes/no check stops once two votes agree (judge._yes_no): three votes only on a split.
EXPECTED_VOTES = 2.2
# The player's words, where they may carry a claim (coach.may_carry_claims): the claims
# extractor at the routine effort (~560-token system prompt, the FEN and the words) and the
# named-move planner at the coach's effort (~290-token system prompt, the FEN and the legal
# moves), one call each.
CLAIMS_PROMPT = 700
CLAIMS_COMPLETION = 200
PLANNER_PROMPT = 700
PLANNER_COMPLETION = 400


@dataclass
class Estimate:
    calls: float = 0.0
    prompt: float = 0.0
    cached: float = 0.0
    completion: float = 0.0
    usd: float = 0.0

    def add(self, other: Estimate) -> Estimate:
        return Estimate(
            self.calls + other.calls,
            self.prompt + other.prompt,
            self.cached + other.cached,
            self.completion + other.completion,
            self.usd + other.usd,
        )

    def line(self) -> str:
        return (
            f"~{self.calls:,.0f} calls, ~{self.prompt:,.0f} prompt tokens "
            f"(~{self.cached:,.0f} cached) + ~{self.completion:,.0f} completion: ~${self.usd:.2f}"
        )


def _usd(prompt: float, cached: float, completion: float, prices: tuple[float, ...]) -> float:
    price_in, price_cached, price_out = prices
    return ((prompt - cached) * price_in + cached * price_cached + completion * price_out) / 1e6


def _tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def coach_estimate(
    requests: list[dict],
    prices: tuple[float, float, float],
    *,
    completion: int = COACH_COMPLETION,
    retry_share: float = RETRY_SHARE,
) -> Estimate:
    """The coach's calls for `requests` (in order within each game; each with its game,
    system, user message and template_only): the cached share is the prefix a request shares
    with the one before it in the same game, once past the cache's minimum."""
    out = Estimate()
    previous: dict[str, str] = {}
    for r in requests:
        if r.get("template_only"):
            continue
        text = str(r.get("system") or "") + "\n" + str(r.get("user") or "")
        prompt = _tokens(text)
        shared = _tokens(text[: _common_prefix(previous.get(r["game"], ""), text)])
        cached = shared if shared >= CACHE_MIN_TOKENS else 0
        previous[r["game"]] = text
        calls = 1 + retry_share
        out = out.add(
            Estimate(
                calls,
                prompt * calls,
                cached * calls,
                completion * calls,
                _usd(prompt * calls, cached * calls, completion * calls, prices),
            )
        )
    return out


def words_estimate(asks: int, prices: tuple[float, float, float]) -> Estimate:
    """The calls the player's words cost: the claims check and the planner, per ask whose
    words may carry a claim."""
    prompt = asks * (CLAIMS_PROMPT + PLANNER_PROMPT)
    completion = asks * (CLAIMS_COMPLETION + PLANNER_COMPLETION)
    return Estimate(2 * asks, prompt, 0, completion, _usd(prompt, 0, completion, prices))


def judge_calls(item: dict) -> tuple[float, float]:
    """(calls, completion tokens) the judge spends on one reply of `item`."""
    votes = EXPECTED_VOTES
    calls = 1.0 + votes * CLAIMS_PER_REPLY + CHECKER_CALLS_PER_REPLY
    completion = (
        DECOMPOSE_COMPLETION
        + votes * CLAIMS_PER_REPLY * VOTE_COMPLETION
        + CHECKER_CALLS_PER_REPLY * CHECKER_COMPLETION
    )
    for fact in item.get("required") or []:
        if fact.get("key") == "opening":
            continue  # decided in code
        asked = 2 * votes if fact.get("key") in ("label", "named_verdict") else votes
        calls += asked
        completion += asked * VOTE_COMPLETION
    if item.get("reference"):
        calls += 1 + 2 * votes
        completion += POINT_COMPLETION + 2 * votes * VOTE_COMPLETION
    return calls, completion


def judge_estimate(items: list[dict], prices: tuple[float, float, float], *, record_of) -> Estimate:
    """The judge's calls for one reply to each of `items`. `record_of(item)` is the record
    the judge puts in its system prompt (judge.evidence_block with the reply): the first
    call reads it fresh, every later one from the cache."""
    out = Estimate()
    for item in items:
        record = _tokens(record_of(item))
        calls, completion = judge_calls(item)
        prompt = calls * (record + QUESTION_TOKENS)
        cached = (calls - 1) * record if record >= CACHE_MIN_TOKENS else 0
        out = out.add(
            Estimate(calls, prompt, cached, completion, _usd(prompt, cached, completion, prices))
        )
    return out
