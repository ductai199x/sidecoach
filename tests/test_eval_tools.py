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
    """Every ledger line of a test goes to a scratch file, never data/eval/usage/."""
    monkeypatch.setattr(judge, "USAGE_MINE", tmp_path / "usage.jsonl")


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


def test_the_judges_record_holds_the_whole_request(tmp_path):
    item = _lesson(_build(tmp_path))
    record = judge.evidence_block(item, with_reference=False)
    assert item["messages"][-1]["content"] in record


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
    system = "S" * 8000  # 2,000 tokens, shared by both requests of the game
    requests = [
        {"game": "g", "system": system, "user": "a" * 400, "template_only": False},
        {"game": "g", "system": system, "user": "a" * 400 + "b" * 400, "template_only": False},
        {"game": "g", "system": system, "user": "x", "template_only": True},
    ]
    est = estimate.coach_estimate(requests, PRICES, completion=100, retry_share=0.0)
    assert est.calls == 2
    assert est.cached == pytest.approx(len(system + "\n" + "a" * 400) // 4)


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


def test_the_pairing_counts_each_outcome():
    items = {i: {"fallback": "Nf3.", "must_name": []} for i in "abcd"}

    def judged(i: str, ok: bool) -> dict:
        return {
            "id": i,
            "class": "lesson",
            "level": "LINE",
            "reply": "Nf3.",
            "scores": {"pass": ok},
            "deterministic": {},
        }

    ai = [judged("a", True), judged("b", True), judged("c", False), judged("d", False)]
    floor = [judged("a", True), judged("b", False), judged("c", True), judged("d", False)]
    rows = parity.pair(items, ai, floor)
    assert [r["outcome"] for r in rows] == ["both", "ai_only", "template_only", "neither"]
    assert "template_only 1" in parity.summary(rows, parity.excluded([]))
