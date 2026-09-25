"""What a run would cost, before any call: the dry run's arithmetic.

The requests and items come from the games played with no model (corpus.build_game with
no provider: the coach's own requests, answered by its templates -- the same requests the
model would get, its own words aside). Prompt sizes are measured on them; the rest are
assumptions from the ledger's history (data/eval/usage.jsonl, 2026-09-17/18) and from the
parity pilot (2026-09-25: $0.50 against an estimate of $0.36, the coach's cache hit and the
token count too optimistic; both are calibrated below): the coach at luna medium wrote ~210
completion tokens a call; a judged reply took ~9-13 judge calls, ~1.9 claims and ~1.4
required facts. The estimate is printed with its assumptions, and it is an estimate: the
ledger's budget, checked before every call, is what stops a run.
"""

from __future__ import annotations

from dataclasses import dataclass

# Characters a token, measured on the parity pilot (2026-09-25, 284 coach calls): the
# coach's prompts came to 6,734 tokens a call where four characters a token said 5,500.
CHARS_PER_TOKEN = 3.3
# The provider's prompt cache serves a repeated prefix only from this many tokens.
CACHE_MIN_TOKENS = 1024
# The share of a coach prompt's shared prefix the cache actually served in the pilot: 37%
# of the prompt was cached where the prefix each request shares with the one before it
# came to ~83% (games played side by side, the cache's own routing and eviction). The
# judge's record, asked again at once, was served as predicted (86%).
COACH_CACHE_HIT = 0.45

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
# The player's words, where they may carry a claim (coach.may_carry_claims): two calls at
# the routine effort, the claims extractor and the named-move planner -- as measured in the
# pilot (42 asks: 634 + 24 and 454 + 14 tokens a call).
CLAIMS_PROMPT = 650
CLAIMS_COMPLETION = 30
PLANNER_PROMPT = 460
PLANNER_COMPLETION = 20


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
    return int(len(text) / CHARS_PER_TOKEN)


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
        cached = shared * COACH_CACHE_HIT if shared >= CACHE_MIN_TOKENS else 0
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


def single_call_estimate(
    prompts: list[str], prices: tuple[float, float, float], *, completion: int = 2
) -> Estimate:
    """One uncached call per prompt (the preference: one A-or-B question a moment)."""
    prompt = sum(_tokens(p) for p in prompts)
    total = completion * len(prompts)
    return Estimate(len(prompts), prompt, 0, total, _usd(prompt, 0, total, prices))


# What a coach call writes, reasoning included, by the model's effort. Medium is the
# ledger's history (~200-250); none and low are assumptions until a run measures them
# (the effort bench's dry run says so).
COMPLETION_BY_EFFORT = {"none": 150, "minimal": 150, "low": 200, "medium": 250, "high": 600}


def coach_completion(spec: str) -> int:
    """The completion a coach call is assumed to write, from its spec's effort
    ('openai:gpt-5.6-luna@low' -> low)."""
    effort = spec.rpartition("@")[2] if "@" in spec else "medium"
    return COMPLETION_BY_EFFORT.get(effort or "medium", COACH_COMPLETION)


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
