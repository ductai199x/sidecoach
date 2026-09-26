"""The evaluation tools (tools/eval): the coach they build is the app's, the judge checks a
reply as the coach does, money is checked before every call, a failed call is never
scored, and old results are never read as new ones. No model, no network, no data/ files:
the game here is three plies written by hand."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import chess
import pytest

EVAL = Path(__file__).resolve().parents[1] / "tools" / "eval"
sys.path.insert(0, str(EVAL))

# tools/eval is scripts, not a package: found through the path above, not by pyright.
import corpus  # noqa: E402  # pyright: ignore[reportMissingImports]
import estimate  # noqa: E402  # pyright: ignore[reportMissingImports]
import judge  # noqa: E402  # pyright: ignore[reportMissingImports]
import parity  # noqa: E402  # pyright: ignore[reportMissingImports]
import providers  # noqa: E402  # pyright: ignore[reportMissingImports]
import stamp  # noqa: E402  # pyright: ignore[reportMissingImports]
import taxonomy  # noqa: E402  # pyright: ignore[reportMissingImports]
from coach_build import build_coach  # noqa: E402  # pyright: ignore[reportMissingImports]

from sidecoach.coach import coach as coach_module  # noqa: E402
from sidecoach.coach.disclosure import PositionFacts  # noqa: E402
from sidecoach.config import Settings  # noqa: E402
from sidecoach.events import Analysis, DisclosureLevel, MoveAssessment, MovePlayed  # noqa: E402
from sidecoach.llm.base import Completion  # noqa: E402

PRICES = (0.20, 0.02, 1.20)


@pytest.fixture(autouse=True)
def _ledger_file(tmp_path, monkeypatch):
    """Every ledger line of a test goes to a scratch file, never data/eval/usage/, and a
    ledger resuming a run reads only the scratch files."""
    monkeypatch.setattr(judge, "USAGE_MINE", tmp_path / "usage-test.jsonl")
    monkeypatch.setattr(judge, "USAGE", tmp_path / "legacy-usage.jsonl")


class FakeModel:
    """A paid model that answers from a script: a string, an exception, or a callable."""

    name = "fake"
    model = "fake-model"
    offline = False

    def __init__(self, answer, usage=(1000, 0, 100)) -> None:
        self.answer, self.usage, self.calls = answer, usage, 0

    def chat(self, system, messages, **kw):
        self.calls += 1
        answer = self.answer(self.calls) if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        prompt, cached, completion = self.usage
        stats = {"prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": completion}
        return Completion(text=str(answer), reasoning="", stats=stats)


# ---- a three-ply game, written by hand ------------------------------------------------


def _line(*moves: str, score: float = 0.5) -> dict:
    return {"moves": list(moves), "depth": 12, "expected_score": score}


def _analysis(board: chess.Board, ply: int, *lines: dict) -> dict:
    return {
        "game_id": "g",
        "fen": board.fen(),
        "ply": ply,
        "lines": list(lines),
        "decisive": False,
        "only_move": False,
        "traps": [],
        "defences": [],
        "threatens": [],
        "in_book": False,
    }


def _facts_file(tmp_path: Path) -> Path:
    """1. e4 e5 2. f3?: a mistake by the player (White), so the coach teaches a lesson."""
    board = chess.Board()
    facts = {
        "name": "hand/three-plies",
        "user_side": "white",
        "start_fen": board.fen(),
        "headers": {},
        "asks": [],
        "start_analysis": _analysis(board, 0, _line("e2e4", "e7e5", score=0.54)),
        "script": [],
    }
    moves = [
        ("e2e4", True, "none", "best", 0.0, ["e2e4", "e7e5"], _line("e7e5", "g1f3")),
        ("e7e5", False, "none", "best", 0.0, ["e7e5", "g1f3"], _line("g1f3", "b8c6")),
        ("f2f3", True, "mistake", "mistake", 0.12, ["g1f3", "b8c6"], _line("b8c6", "d2d4")),
    ]
    for ply, (uci, by_user, severity, label, loss, best, after) in enumerate(moves, start=1):
        move = chess.Move.from_uci(uci)
        before = board.fen()
        san = board.san(move)
        board.push(move)
        facts["script"].append(
            {
                "kind": "move",
                "ply": ply,
                "san": san,
                "uci": uci,
                "fen_before": before,
                "fen_after": board.fen(),
                "by_user": by_user,
                "think_ms": 8000,
                "analysis": _analysis(board, ply, after),
                "assessment": {
                    "game_id": "g",
                    "ply": ply,
                    "uci": uci,
                    "expected_loss": loss,
                    "severity": severity,
                    "best": best,
                    "label": label,
                },
            }
        )
    path = tmp_path / "hand__three-plies.json"
    path.write_text(json.dumps(facts), encoding="utf-8")
    return path


def _build(tmp_path, provider=None, ledger=None, **kw) -> list[dict]:
    return corpus.build_game(
        "hand",
        _facts_file(tmp_path),
        random.Random(1),
        None,
        None,
        provider=provider,
        ledger=ledger,
        version="v-test",
        **kw,
    )


def _lesson(items: list[dict]) -> dict:
    return next(it for it in items if it["trigger"] == "post_mistake")


# ---- B2: the coach is the app's ------------------------------------------------------


def test_the_eval_coach_is_built_from_the_apps_settings():
    settings = Settings()
    settings.coach.verbalized_sampling = True
    settings.coach.summarise_every = 7
    coach = build_coach(None, settings=settings)
    assert coach.verbalized_sampling is True
    assert coach.summarise_every == 7
    assert coach.local_verdicts is settings.coach.local_verdicts
    assert coach.policy.settings.stuck_seconds == settings.coach.stuck_seconds
    # The shipped system prompt: the persona with the rating, and the shipped C.
    rating = f"{settings.profile.rating_low} to {settings.profile.rating_high}"
    assert rating in coach.system
    assert "Lessons from earlier games" in coach.system
    assert coach.system == build_coach(None).system


def test_a_paid_model_is_only_built_on_a_ledger():
    with pytest.raises(ValueError):
        build_coach(FakeModel("hi"))
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    coach = build_coach(FakeModel("hi"), ledger=ledger)
    assert isinstance(coach.speaker.provider, providers.Metered)
    with pytest.raises(ValueError):
        providers.coach_provider("openai:gpt-5.6-luna@medium")  # no ledger: refused


def test_the_eval_sends_the_verdict_before_the_analysis(tmp_path, monkeypatch):
    seen: list[str] = []
    real = corpus.build_coach

    def spying(*a, **kw):
        coach = real(*a, **kw)
        on_event = coach.on_event

        def record(event, ts):
            seen.append(type(event).__name__)
            on_event(event, ts)

        coach.on_event = record  # type: ignore[method-assign]
        return coach

    monkeypatch.setattr(corpus, "build_coach", spying)
    _build(tmp_path)
    after_moves = [seen[i + 1 : i + 3] for i, name in enumerate(seen) if name == "MovePlayed"]
    assert after_moves and all(
        pair == [MoveAssessment.__name__, Analysis.__name__] for pair in after_moves
    )
    assert MovePlayed.__name__ in seen


def test_templates_are_the_same_on_every_rebuild(tmp_path):
    first = [it["fallback"] for it in _build(tmp_path)]
    second = [it["fallback"] for it in _build(tmp_path)]
    assert first == second


# ---- the item carries what the coach's checks read -----------------------------------


def test_an_item_keeps_the_request_the_coach_built(tmp_path):
    item = _lesson(_build(tmp_path))
    assert item["played_san"] == "f3"
    assert item["must_name"] == ["Nf3"]  # the better move the lesson's template names
    assert item["frames"], "the boards after the move, for the draft checker"
    assert item["assessment"]["severity"] == "mistake"
    assert item["commit"] == "v-test"
    request = judge.request_for(item, item["system"])
    assert request.must_name == ("Nf3",)
    assert [b.fen() for b in request.frames] == item["frames"]
    # The lesson's cost is owed in words: its directive forbids figures.
    cost = next(f for f in item["required"] if f["key"] == "cost")
    assert "%" not in cost["value"]


def test_the_judges_record_holds_the_whole_request_save_the_coachs_words(tmp_path):
    item = _lesson(_build(tmp_path))
    record = judge.evidence_block(item, with_reference=False)
    shown = item["messages"][-1]["content"]
    assert judge.without_coach_words(shown) in record
    said = [x for x in shown.splitlines() if "YOU (the coach) said: " in x]
    assert said, "the lesson's message carries the coach's word on 1. e4"
    assert not any(x.strip() in record for x in said)


# ---- B4, B5, M3: the judge checks as the coach checks --------------------------------


def test_the_played_moves_are_read_bare():
    record = "13. Qe2 (the player): inaccuracy, cost 0.07\n14... Rxe2 by the opponent"
    item = {"messages": [{"role": "user", "content": record}]}
    assert judge._played_sans(item) == ["Qe2", "Rxe2"]
    # The review's ground truth keeps its numbers and labels.
    assert judge.record_from_harness(item) == ["13. Qe2 (the player, inaccuracy)"]


def _with_task(task: str) -> dict:
    return {"messages": [{"role": "user", "content": "# Game so far\n...\n\n# Task\n" + task}]}


def test_form_comes_from_the_items_own_directive():
    hint = _with_task(coach_module.IDEA_DIRECTIVE)
    _, problems = judge.form_problems(hint, "Only one move holds. Their knight on e5 is loose.")
    assert problems == []  # the first hint asks for no question
    answer = _with_task(coach_module.MOVE_DIRECTIVE)
    _, problems = judge.form_problems(answer, "Play Nf3. Why does it work?")
    assert problems == ["a question where the task allows none"]


def test_the_templates_pass_the_judges_checks(tmp_path):
    draft_check = judge._draft_check()
    for item in _build(tmp_path):
        if not item["fallback"]:
            continue
        det = judge.deterministic(item, item["fallback"], draft_check, floor=True)
        assert det["form"]["problems"] == [], (item["id"], det)
        assert det["board"] == [], (item["id"], det)
        assert det["rules"] == [] and det["verdict"] == [] and det["floor"] == []


def test_the_owners_rules_fail_a_lesson_that_restates_or_quotes(tmp_path):
    item = _lesson(_build(tmp_path))
    draft_check = judge._draft_check()
    reply = "f3 cost you 12% of the game: Nf3 was better. Which piece should come out first?"
    det = judge.deterministic(item, reply, draft_check)
    assert det["rules"], det
    scores = judge.score(item, det, None)
    assert scores["rules"] and not scores["pass"]


def test_a_reply_that_leaves_out_what_the_template_names_is_not_a_pass(tmp_path):
    item = _lesson(_build(tmp_path))
    det = judge.deterministic(item, "That was a slip. What would you do instead?", None)
    assert det["floor"]
    assert not judge.score(item, det, None)["pass"]


# ---- M2 -------------------------------------------------------------------------------


def _moment(**facts) -> taxonomy.Moment:
    base = {"fen": chess.STARTING_FEN, "side_to_move": "white", "phase": "opening"}
    return taxonomy.Moment(
        facts=PositionFacts(**base, **facts),
        level=DisclosureLevel.IDEA,
        board=chess.Board(),
        user_to_move=True,
    )


def test_the_first_hint_requires_its_stakes_key_fact_and_trap():
    m = _moment(
        situation="Only one move holds. They threaten Nxe5. Bxf7 looks free but loses.",
        stakes="only one move holds",
        key="they threaten Nxe5",
        trap="Bxf7 looks free but loses",
        goal="a threat to meet",
        theme="a pin",
    )
    keys = [f.key for f in taxonomy.required(taxonomy.CLASSES["stuck_hint"], m)]
    assert keys == ["stakes", "key_fact", "trap"]
    keys = [f.key for f in taxonomy.required(taxonomy.CLASSES["decisive_question"], m)]
    assert keys == ["stakes", "key_fact", "trap"]
    plain = _moment(goal="a threat to meet", theme="a pin")
    keys = [f.key for f in taxonomy.required(taxonomy.CLASSES["stuck_hint"], plain)]
    assert keys == ["goal", "pattern"]


def test_the_cost_is_owed_without_a_figure_where_figures_are_forbidden():
    assert taxonomy._loss_words(0.12) == "a good deal (about 12% of the game)"
    assert taxonomy._loss_words(0.12, figure=False) == "a good deal"


# ---- B6: money -------------------------------------------------------------------------


def test_the_ledger_counts_dollars_and_stops_before_the_call_past_the_budget(tmp_path):
    ledger = judge.Ledger("t", "test", "fake", budget_usd=0.0005, prices=PRICES)
    model = FakeModel("yes", usage=(1000, 0, 100))  # $0.0002 + $0.00012 a call
    metered = providers.Metered(model, ledger)
    metered.chat("s", [])
    assert ledger.usd == pytest.approx(0.00032)
    metered.chat("s", [])  # 0.00032 < 0.0005: allowed
    with pytest.raises(judge.BudgetExceeded):
        metered.chat("s", [])
    assert model.calls == 2  # checked before the call: the third was never made
    lines = [json.loads(x) for x in judge.USAGE_MINE.read_text(encoding="utf-8").splitlines()]
    assert [x["usd"] for x in lines] == [pytest.approx(0.00032)] * 2
    assert not issubclass(judge.BudgetExceeded, Exception)  # no `except Exception` eats it


def test_cached_tokens_are_priced_as_cached():
    ledger = judge.Ledger("t", "test", "fake", prices=PRICES)
    usd = ledger.record("x", {"prompt_tokens": 10_000, "cached_tokens": 8_000}, 0.1)
    assert usd == pytest.approx((2_000 * 0.20 + 8_000 * 0.02) / 1e6)


def test_a_refused_request_is_not_retried(monkeypatch):
    monkeypatch.setattr(judge.time, "sleep", lambda s: None)
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    bad = FakeModel(RuntimeError("400: context too long"))
    with pytest.raises(judge.JudgeFailed):
        judge._chat_with_backoff(bad, "s", [], ledger, "x")
    assert bad.calls == 1
    flaky = FakeModel(lambda n: RuntimeError("429: slow down") if n == 1 else "yes")
    assert judge._chat_with_backoff(flaky, "s", [], ledger, "x").text == "yes"
    assert flaky.calls == 2


def test_the_budget_stops_the_game_through_the_coachs_own_handlers(tmp_path):
    ledger = judge.Ledger("t", "test", "fake", budget_usd=0.0, prices=PRICES)
    model = FakeModel("Fine.")
    with pytest.raises(judge.BudgetExceeded):
        _build(tmp_path, provider=model, ledger=ledger)
    assert model.calls == 0


def test_the_dry_run_estimate_reads_the_cache_prefix():
    system = "S" * 8000  # ~2,400 tokens, shared by both requests of the game
    requests = [
        {"game": "g", "system": system, "user": "a" * 400, "template_only": False},
        {"game": "g", "system": system, "user": "a" * 400 + "b" * 400, "template_only": False},
        {"game": "g", "system": system, "user": "x", "template_only": True},
    ]
    est = estimate.coach_estimate(requests, PRICES, completion=100, retry_share=0.0)
    assert est.calls == 2
    shared = estimate._tokens(system + "\n" + "a" * 400)
    # Of the shared prefix, the share the cache served in the pilot.
    assert est.cached == pytest.approx(shared * estimate.COACH_CACHE_HIT)


# ---- B7: a failed call is never scored ------------------------------------------------


def test_a_failed_call_is_marked_and_never_judged(tmp_path):
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    items = _build(tmp_path, provider=FakeModel(RuntimeError("500: down")), ledger=ledger)
    reply = _lesson(items)["reply"]
    assert reply["model_failed"] and reply["call_errors"] >= 1
    assert judge.reply_status(reply) == "model_failed"


def test_an_empty_answer_is_a_failed_call_and_a_refused_one_is_not(tmp_path):
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    silent = _lesson(_build(tmp_path, provider=FakeModel(""), ledger=ledger))["reply"]
    assert judge.reply_status(silent) == "model_failed"
    refused = _lesson(_build(tmp_path, provider=FakeModel("Blah."), ledger=ledger))["reply"]
    assert judge.reply_status(refused) == "rejected"  # the checks said no; the template spoke
    assert not refused["model_failed"]


# ---- M6: old results are not read as new ------------------------------------------------


def test_results_from_other_code_are_refused_unless_asked():
    stamp.check("abc1234", "x", reuse_stale=False, current="abc1234")
    with pytest.raises(stamp.StaleData):
        stamp.check("abc1234", "x", reuse_stale=False, current="def5678")
    with pytest.raises(stamp.StaleData):
        stamp.check(None, "x", reuse_stale=False, current="def5678")
    stamp.check("abc1234", "x", reuse_stale=True, current="def5678")
    assert stamp.code_version()


def test_the_judge_refuses_a_stale_judgment(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "JUDGMENTS", tmp_path)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "a.jsonl").write_text(
        json.dumps({"id": "i", "commit": "old"}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(stamp, "code_version", lambda: "new")
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    with pytest.raises(stamp.StaleData):
        judge.judge_all({}, [], "run", FakeModel("yes"), ledger, thinking=False, quiet=True)


# ---- the floor arm -------------------------------------------------------------------


def test_floor_coverage_is_decided_in_code():
    item = {"fallback": "That one cost you: Nf3 was better -- after Nc6, d4.", "must_name": ["Nf3"]}
    full = parity.coverage(item, "Nf3 develops and meets Nc6 with d4.")
    assert full["must_name_ok"] and full["share"] == 1.0
    thin = parity.coverage(item, "Develop your knights first.")
    assert not thin["must_name_ok"] and thin["share"] == 0.0


def _judged(i: str, ok: bool, completeness: float | None = 1.0) -> dict:
    return {
        "id": i,
        "class": "lesson",
        "level": "LINE",
        "reply": "Nf3.",
        "scores": {"pass": ok, "completeness": completeness},
        "deterministic": {},
    }


def test_the_pairing_counts_each_outcome_and_the_required_facts():
    items = {i: {"fallback": "Nf3.", "must_name": []} for i in "abcd"}
    ai = [
        _judged("a", True, 1.0),
        _judged("b", True, 0.5),
        _judged("c", False),
        _judged("d", False),
    ]
    floor = [
        _judged("a", True, 0.5),
        _judged("b", False, 1.0),
        _judged("c", True),
        _judged("d", False),
    ]
    rows = parity.pair(items, ai, floor)
    assert [r["outcome"] for r in rows] == ["both", "ai_only", "template_only", "neither"]
    assert [r["facts"] for r in rows] == ["more", "fewer", "same", "same"]
    text = parity.summary(rows, parity.Counter({"rejected": 2}), seed=7001, per_game=8)
    # Pass and completeness side by side; a reply the checks refused is a template win.
    assert "| AI | 0.50 | 0.88 |" in text and "| template | 0.50 | 0.88 |" in text
    assert "the template 3 (1 judged, 2 where the checks refused" in text
    assert "fewer than the template in 1, the same in 2, more in 1" in text
    assert "seed 7001, 8 a game" in text


def _drawn(i: str, *, fallback: str = "Nf3 was better.", quick: bool = False, **reply) -> dict:
    return {
        "id": i,
        "fallback": fallback,
        "quick_followup": quick,
        "reply": {"fallback_used": False, **reply},
    }


def test_quick_followups_and_refused_replies_are_not_paired():
    drawn = [
        _drawn("own"),
        _drawn("quick", fallback="", quick=True),
        _drawn("no-template", fallback=""),
        _drawn("refused", fallback_used=True),
        _drawn("failed", fallback_used=True, model_failed=True),
        _drawn("fast", template_only=True),
    ]
    pairable, left_out = parity.sort_out(drawn)
    assert [it["id"] for it in pairable] == ["own"]
    assert left_out == {
        parity.QUICK: 2,
        "rejected": 1,
        "model_failed": 1,
        "template_only": 1,
    }


def test_quick_followups_owe_nothing_their_template_said():
    from types import SimpleNamespace

    from sidecoach.events import Trigger

    quick = SimpleNamespace(trigger=Trigger.USER_ASK, fallback="", must_name=())
    assert corpus.quick_followup(quick, "the quick answer they have already been shown")
    assert corpus.quick_followup(quick, "")
    hint = SimpleNamespace(trigger=Trigger.USER_ASK, fallback="Only one move holds.", must_name=())
    assert not corpus.quick_followup(hint, "They asked for a hint.")
    assert {"stakes", "key_fact", "trap"} <= corpus.QUICK_NOT_REQUIRED


def test_one_seed_draws_a_rounds_moments_in_both_tools():
    assert parity.round_seed(7, 1) == 7001
    from icr_play import sample_moments  # pyright: ignore[reportMissingImports]

    items = [
        {"id": f"g{g}/p{p:03d}", "game": f"g{g}", "ply": p, "class": "lesson"}
        for g in range(2)
        for p in range(30)
    ]
    drawn = [it["id"] for it in sample_moments(items, 8, parity.round_seed(7, 1))]
    shuffled = list(reversed(items))
    assert [it["id"] for it in sample_moments(shuffled, 8, parity.round_seed(7, 1))] == drawn
    assert len(drawn) == 16


def test_latency_is_reported_by_trigger():
    items = [
        {"trigger": "post_mistake", "reply": {"elapsed_s": s, "fallback_used": False}}
        for s in (2.0, 4.0, 3.0, 10.0)
    ] + [{"trigger": "stuck", "reply": {"elapsed_s": 9.0, "model_failed": True}}]
    assert parity.latency_by_trigger(items) == [("post_mistake", 4, 3.0, 10.0)]


# ---- the second audit --------------------------------------------------------------------


def test_the_version_follows_only_what_shapes_a_reply():
    assert "src/sidecoach/coach" in stamp.WATCHED and "tools/eval" in stamp.WATCHED
    assert "src/sidecoach" not in stamp.WATCHED
    for elsewhere in ("ui", "voice", "vision", "capture"):
        assert not any(w.startswith(f"src/sidecoach/{elsewhere}") for w in stamp.WATCHED)


def test_the_coachs_own_words_are_not_evidence():
    harness = (
        "# Game so far\n1. e4 (the player): best\n"
        '    YOU (the coach) said: "Their queen on d8 is trapped."\n'
        '    PLAYER said: "why?"\n'
        "Earlier dialogue, summarised: the coach said the queen was lost.\n"
        "# Task\nAnswer."
    )
    cleaned = judge.without_coach_words(harness)
    assert "trapped" not in cleaned and "queen was lost" not in cleaned
    assert 'PLAYER said: "why?"' in cleaned and "YOU (the coach) said:" in cleaned
    assert "Nothing the coach itself said earlier in the game is evidence" in judge.BINARY_SYSTEM


def test_the_claims_check_runs_at_the_routine_effort(monkeypatch):
    class EffortModel(FakeModel):
        reasoning_effort = "medium"

        def with_effort(self, effort):
            other = EffortModel(self.answer)
            other.reasoning_effort = effort
            return other

    seen = []

    def claims_check(provider):
        seen.append(provider)
        return None

    import coach_build  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(coach_build.cli, "_claims_check", claims_check)
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    build_coach(EffortModel("hi"), ledger=ledger)
    assert isinstance(seen[0], providers.Metered)
    assert seen[0].inner.reasoning_effort == Settings().llm.routine_effort == "none"


def test_the_review_owes_the_move_that_cost_the_most():
    m = _moment()
    m.tally = {"best": 3, "inaccuracy": 1, "mistake": 1}
    m.lost = ["3. f3 (mistake; Nf3 was better)", "9. h3 (inaccuracy; O-O was better)"]
    m.lost_loss = [0.10, 0.20]  # the inaccuracy cost more
    facts = taxonomy.required(taxonomy.CLASSES["ask_game_so_far"], m)
    worst = next(f for f in facts if f.key == "lost_worst")
    assert worst.value.startswith("9. h3")


def test_a_restarted_run_keeps_its_budget(tmp_path):
    first = judge.Ledger("run-a", "test", "fake", budget_usd=0.0005, prices=PRICES, resume=True)
    providers.Metered(FakeModel("yes"), first).chat("s", [])
    again = judge.Ledger("run-a", "test", "fake", budget_usd=0.0005, prices=PRICES, resume=True)
    assert again.before == pytest.approx(0.00032) and again.calls == 1
    model = FakeModel("yes")
    providers.Metered(model, again).chat("s", [])  # 0.00032 < 0.0005: allowed
    with pytest.raises(judge.BudgetExceeded):
        providers.Metered(model, again).chat("s", [])
    other = judge.Ledger("run-b", "test", "fake", budget_usd=0.0005, prices=PRICES, resume=True)
    assert other.before == 0.0


def test_a_vote_stops_once_two_agree():
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    agree = FakeModel("yes")
    assert judge._yes_no(agree, ledger, "x", "record", "question?") is True
    assert agree.calls == 2
    split = FakeModel(lambda n: "yes" if n % 2 else "no")
    assert judge._yes_no(split, ledger, "x", "record", "question?") is True
    assert split.calls == 3


def test_the_estimate_counts_the_calls_the_players_words_cost():
    est = estimate.words_estimate(3, PRICES)
    assert est.calls == 6 and est.usd > 0


# ---- the pilot's board-check fails ------------------------------------------------------


def _position_item(fen: str, sheet: str, **facts) -> dict:
    """An item of the player's own position: the sheet the coach was shown, its facts."""
    board = chess.Board(fen)
    side = "white" if board.turn == chess.WHITE else "black"
    base = {"fen": fen, "side_to_move": side, "user_side": side, "phase": "middlegame"}
    return {
        "id": "hand/x/p001",
        "game": "hand/x",
        "ply": 1,
        "fen": fen,
        "fen_now": fen,
        "trigger": "stuck",
        "level": "IDEA",
        "system": "",
        "fallback": "",
        "frames": [],
        "messages": [{"role": "user", "content": sheet + "\n\n# Task\nThe first hint."}],
        "moment": {"facts": PositionFacts(**base, **facts).model_dump()},
    }


def test_a_threat_the_facts_give_is_supported():
    # bots/...__004/p032: "your opponent threatens ...h5", h5 among the facts' threats.
    fen = "2kr3r/1pp1q1pp/p1bp1b2/4p3/2PP4/6NP/PP3PP1/R1BQR1K1 w - - 1 17"
    sheet = "what the other side threatens, if the side to move passed: h5"
    item = _position_item(fen, sheet, threats=["h5"])
    assert judge.code_check("your opponent threatens ...h5", item) is True
    assert judge.code_check("Your opponent threatens ...c4", item) is None  # not given
    # The checker's misreading of the move as a target square is no contradiction.
    from sidecoach.coach.claims import Claim, ClaimKind

    misread = Claim(ClaimKind.THREATENS, "they", "h5", None, "they threaten h5")
    ruling, _, _ = judge.rule_claims("your opponent threatens ...h5", item, [misread])
    assert ruling != "false"


def test_a_threat_on_the_players_own_piece_is_the_opponents():
    # bots/...__002/p054: the key fact is their threat to the rook on c6 (Nb4xc6).
    fen = "r5k1/5p1p/p1R3p1/4P1B1/1n6/8/PP3PP1/6K1 w - - 1 28"
    item = _position_item(fen, "key: They threaten Nxc6, winning your rook on c6", threats=["Nxc6"])
    text = "The plan is to keep the rook on c6 active while removing the threat to it."
    assert judge.code_check(text, item) is True
    from sidecoach.coach.claims import Claim, ClaimKind

    # The extractor read the threat as the player's own: its twin reads it as theirs.
    own = Claim(ClaimKind.THREATENS, "I", "my rook on c6", None, "I threaten my rook on c6")
    ruling, reason, _ = judge.rule_claims(text, item, [own])
    assert ruling == "true", reason
    # A piece of theirs said to be under threat is not decided this way.
    assert judge.code_check("the threat to the pawn on a6", item) is None


def test_an_attack_a_move_uncovers_is_the_moves():
    # bots/...__004/p058: Ba7+ moves the bishop off the e-file; the rook on e1 attacks e5.
    fen = "1k1r1q2/1pp5/p4p2/4p3/2P5/4Bb1P/PP4r1/R1Q1RK2 w - - 4 30"
    item = _position_item(fen, "")
    assert judge.code_check("Ba7+ keeps the attack on e5", item) is True
    assert judge.code_check("Bd4 keeps the attack on h8", item) is None  # the vote decides
    from sidecoach.coach.claims import Claim, ClaimKind

    bishop = Claim(ClaimKind.ATTACKS, "my bishop", "their pawn on e5", "Ba7+", "Ba7+ attacks e5")
    ruling, reason, _ = judge.rule_claims("Ba7+ keeps the attack on e5", item, [bishop])
    assert ruling == "true", reason


# ---- the preference and the effort bench -------------------------------------------------


def _pair_row(i: str) -> dict:
    return {"id": i, "class": "lesson", "ai": f"AI reply {i}.", "template": f"Template {i}."}


def _pair_item(i: str) -> dict:
    shown = (
        "# Game so far\n1. e4 (the player): best\n"
        '    YOU (the coach) said: "An old word of the coach."\n\n# Task\nThe lesson.'
    )
    return {"id": i, "game": "hand/x", "messages": [{"role": "user", "content": shown}]}


def test_the_preference_shows_the_record_and_both_replies_in_a_seeded_order():
    item = _pair_item("m1")
    system, user, first = parity.prefer_prompt(
        item, "AI.", "Template.", seed=7001, rating=(1000, 1400)
    )
    assert "An old word of the coach" not in system and "The lesson." in system
    assert "rated about 1000 to 1400" in user and user.endswith("Answer A or B.")
    assert ('Reply A:\n"""\nAI.' in user) == first
    assert parity.ai_first("m1", 7001) == first  # the same coin every time
    orders = {parity.ai_first(f"m{k}", 7001) for k in range(20)}
    assert orders == {True, False}  # both orders occur


def test_the_preference_is_one_question_a_moment_asked_once(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "JUDGMENTS", tmp_path)
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    rows = [_pair_row(f"m{k}") for k in range(6)]
    by_id = {r["id"]: _pair_item(r["id"]) for r in rows}
    model = FakeModel("A")
    prefs = parity.judge_prefer(by_id, rows, "run", model, ledger, seed=7001, workers=1)
    assert model.calls == 6 and len(prefs) == 6
    for p in prefs:
        assert p["ai_preferred"] == (p["ai_is"] == "A")  # the judge always said A
    again = parity.judge_prefer(by_id, rows, "run", model, ledger, seed=7001, workers=1)
    assert model.calls == 6 and len(again) == 6  # kept, not asked again
    text = "\n".join(parity.prefer_summary(prefs))
    assert "the judge answered A in 6 of 6" in text


def test_an_answer_that_is_neither_a_nor_b_is_no_preference(tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "JUDGMENTS", tmp_path)
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    rows = [_pair_row("m1")]
    prefs = parity.judge_prefer(
        {"m1": _pair_item("m1")}, rows, "run", FakeModel("Both."), ledger, seed=7001, workers=1
    )
    assert prefs == []


def test_the_effort_table_says_when_arms_are_not_comparable():
    import effort_table  # pyright: ignore[reportMissingImports]

    def arm(name: str, drawn: list[str], commit: str = "abc") -> dict:
        return {
            "name": name,
            "run": name,
            "effort": name.split("-")[1],
            "commit": commit,
            "seed": 7001,
            "drawn": drawn,
            "ids": drawn,
            "scores": {"pass": 0.9, "completeness": 0.8, "fabrication": 0.05},
            "preferred": 0.6,
            "prefs": 10,
            "latency": {"stuck": (4, 2.0, 5.0)},
            "usd": 0.4,
            "coach_usd": 0.2,
        }

    same = effort_table.table([arm("effort-none", ["a", "b"]), arm("effort-low", ["a", "b"])])
    assert "NOT COMPARABLE" not in same and "| effort-none | none | 2 | 0.90 |" in same
    assert "2.0/5.0 (n=4)" in same
    other = effort_table.table([arm("effort-none", ["a"]), arm("effort-low", ["b"], "def")])
    assert "NOT COMPARABLE" in other
    assert estimate.coach_completion("openai:gpt-5.6-luna@low") == 200


# ---- the excellence rubric ---------------------------------------------------------------


def test_the_rubric_file_reads_as_criteria():
    import rubric  # pyright: ignore[reportMissingImports]

    criteria = rubric.load_rubric(rating=(1000, 1400))
    keys = [c.key for c in criteria]
    # "economy" is gone (Tai, 2026-09-25): conciseness is the prompt's and the length check's.
    assert keys == ["why", "transfer", "question", "connection", "actionable", "tone"]
    by = {c.key: c for c in criteria}
    assert by["tone"].applies("ask_material", "Level.")  # "*": every class
    assert not by["connection"].applies("stuck_hint", "Look at e5.")
    # The question criterion is asked only of a reply that ends with a question -- and not
    # of "what were you expecting?", the coach listening rather than teaching.
    assert by["question"].applies("lesson", "Nf3 was better. Which piece guards e5?")
    assert not by["question"].applies("lesson", "Nf3 was better.")
    assert not by["question"].applies(
        "lesson", "Nf3 was better. What were you expecting that move to do?"
    )


def test_a_rubric_file_with_a_repeated_key_or_an_unknown_condition_is_refused(tmp_path):
    import rubric  # pyright: ignore[reportMissingImports]

    twice = tmp_path / "twice.toml"
    twice.write_text(
        '[[criterion]]\nkey = "a"\nquestion = "Q?"\napplies_to = ["*"]\n' * 2, encoding="utf-8"
    )
    with pytest.raises(ValueError):
        rubric.load_rubric(twice, rating=(1000, 1400))
    odd = tmp_path / "odd.toml"
    odd.write_text(
        '[[criterion]]\nkey = "a"\nquestion = "Q?"\napplies_to = ["*"]\nonly_if = "sunny"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        rubric.load_rubric(odd, rating=(1000, 1400))


def test_a_rubric_vote_stops_at_two_and_an_unreadable_answer_is_not_scored():
    import rubric  # pyright: ignore[reportMissingImports]

    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    agree = FakeModel("No.")
    assert rubric.ask(agree, ledger, "x", "record", "Q?") is False and agree.calls == 2
    with pytest.raises(judge.JudgeFailed):
        rubric.ask(FakeModel("Perhaps."), ledger, "x", "record", "Q?")


def test_the_rubric_asks_each_question_once_and_scores_the_replies(tmp_path, monkeypatch):
    import rubric  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(judge, "JUDGMENTS", tmp_path / "judgments")
    item = _lesson(_build(tmp_path))
    ai_reply = "Nf3 was better: it develops a piece and keeps f3 free. Which piece is still home?"
    tasks = [rubric.Task("ai", item, ai_reply), rubric.Task("floor", item, item["fallback"])]
    criteria = rubric.load_rubric(rating=(1000, 1400))
    ledger = judge.Ledger("t", "test", "fake", budget_usd=1.0, prices=PRICES)
    model = FakeModel(lambda n: "no" if n <= 2 else "yes")  # the first question: no
    answers, failed = rubric.run_rubric("run", tasks, criteria, model, ledger, workers=1)
    asked = model.calls
    assert not failed and asked > 0
    ai_keys = set(answers["ai"][item["id"]])
    assert "question" in ai_keys and "tone" in ai_keys  # the AI reply ends with a question
    assert "question" not in answers["floor"][item["id"]]  # the template does not
    # Asked again: nothing to pay for.
    rubric.run_rubric("run", tasks, criteria, model, ledger, workers=1)
    assert model.calls == asked
    # An edited question is asked again, and only it.
    edited = [
        rubric.Criterion(c.key, c.question + " Be strict.", c.applies_to, c.only_if)
        if c.key == "tone"
        else c
        for c in criteria
    ]
    rubric.run_rubric("run", tasks, edited, model, ledger, workers=1)
    assert model.calls == asked + 2 * 2  # two replies, two votes each
    kept = {arm: rubric.load_answers("run", arm, reuse_stale=False) for arm in ("ai", "floor")}
    rows = rubric.scores_of(tasks, edited, kept)
    assert rows[0]["excellence"] is not None and rows[0]["answers"]["tone"] is True
    text = rubric.report("run", rows, edited, failed)
    assert "| tone |" in text and "Excellence" in text


def test_the_rubric_finds_a_runs_items_and_judges_the_models_own_replies(tmp_path, monkeypatch):
    import rubric  # pyright: ignore[reportMissingImports]

    path = rubric.items_dir("effort-medium-47138dd-r1")
    assert path.parts[-2:] == ("effort-medium", "items-47138dd-r1")
    assert rubric.items_dir("a-b-1234567-d89abcde-r2").name == "items-1234567-d89abcde-r2"
    monkeypatch.setattr(judge, "JUDGMENTS", tmp_path)

    def item(i: str, **reply) -> dict:
        return {
            "id": i,
            "game": "g",
            "class": "lesson",
            "fallback": "Nf3 was better.",
            "reply": {"reply": f"AI {i}.", "fallback_used": False, **reply},
        }

    items = {i: item(i) for i in ("a", "b")} | {"c": item("c", fallback_used=True)}
    for run, ids in (("run", ["a", "b", "c"]), ("run-floor", ["a"])):
        (tmp_path / run).mkdir()
        (tmp_path / run / "g.jsonl").write_text(
            "".join(json.dumps({"id": i}) + "\n" for i in ids), encoding="utf-8"
        )
    tasks = rubric.tasks_of("run", items, floor=True)
    assert [(t.arm, t.id) for t in tasks] == [("ai", "a"), ("ai", "b"), ("floor", "a")]
    work = [("R" * 8000, 3)]
    est = estimate.repeated_record_estimate(work, PRICES, votes=2)
    assert est.calls == 6 and est.cached == pytest.approx(5 * estimate._tokens("R" * 8000))
