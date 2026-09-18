"""A game played through the live coach's own code from its cached facts.

Each game is replayed through a real Coach -- the same Conversation, disclosure machine,
policy and directives the panel runs -- with the model replaced by a capture: every request
the coach would have sent is one item (a coach-initiated moment), and at chosen plies the
player's questions are put to it, each of those a request too (a user-ask item). The
coach's own turns in the history are the templates it falls back on, so the history is
deterministic and free of model calls. An item holds exactly what a request holds --
the system message, the messages, the directive (the sampling format is added when the
model is called, as live) -- plus what the
judge needs: the class, the required facts, the moment's evidence, and -- in a master's
game, for the items about a move the book annotates -- the master's note as `reference`.
Both sources take exactly the same path; the note is the only difference.

Used by icr_play.py (`build_game`, with the model answering) and playthrough.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from taxonomy import (  # noqa: E402
    ABOUT_LAST_MOVE,
    CLASSES,
    Moment,
    as_json,
    classify_trigger,
    pick_named_move,
    question_for,
    required,
)

from sidecoach.coach.coach import Coach, CoachMessage, Spoken, _Request  # noqa: E402
from sidecoach.coach.facts import material_text  # noqa: E402
from sidecoach.coach.outcome import line_outcome  # noqa: E402
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings  # noqa: E402
from sidecoach.events import (  # noqa: E402
    Analysis,
    GameEnded,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Side,
    Takeback,
)

FACTS = ROOT / "data" / "eval" / "facts"
MASTERS = ROOT / "data" / "eval" / "masters"
STUCK_SECONDS = 30

# The player's questions, by where they fit. Each user-to-move position gets at most one
# question; the rotation keeps the classes balanced across a game.
ROTATION_TO_MOVE = [
    "ask_hint",
    "ask_plan",
    "ask_eval",
    "ask_named_move",
    "ask_threat",
    "ask_move",
    "ask_material",
    "ask_line",
    "ask_opening",
]
ASK_EVERY = 3  # of the player's turns


class Capture:
    """On the speaker: records each request and what was spoken for it. Without a model it
    answers with the request's own template, so the conversation's history is the coach's
    fallbacks; with one (`inner`, the speaker's own speak_now) the model answers and its
    words go into the history, as they do live."""

    def __init__(self, inner=None) -> None:
        self.inner = inner
        self.requests: list[_Request] = []
        self.spoken: list[Spoken] = []

    def __call__(self, request: _Request) -> Spoken:
        if self.inner is not None:
            spoken = self.inner(request)
        else:
            message = CoachMessage(
                text=request.fallback,
                level=request.level,
                trigger=request.trigger,
                game_id=request.game_id,
                ply=request.ply,
                fen=request.board.fen(),
                annotations=list(request.annotations),
            )
            spoken = Spoken(message=message, level=request.level, problems=["captured"], attempts=0)
        self.requests.append(request)
        self.spoken.append(spoken)
        return spoken


def reply_record(item_id: str, cls_id: str, spoken: Spoken) -> dict[str, Any]:
    """What the model said for one moment, in the shape generate.py writes for a reply."""
    calls = spoken.calls
    return {
        "id": item_id,
        "class": cls_id,
        "reply": spoken.message.text,
        "raw": calls[0].raw if calls else "",
        "candidates": sum(c.candidates for c in calls),
        "passed": sum(c.passed for c in calls),
        "problems": spoken.problems,
        "fallback_used": bool(spoken.problems),
        "attempts": spoken.attempts,
        "verified": spoken.verified,
        "repaired": spoken.repaired,
        "elapsed_s": round(sum((c.stats.get("elapsed_s") or 0) for c in calls), 2),
        "tokens": {
            "prompt": sum((c.stats.get("prompt_tokens") or 0) for c in calls),
            "completion": sum((c.stats.get("completion_tokens") or 0) for c in calls),
        },
    }


def _label(a: MoveAssessment | None) -> str | None:
    if a is None:
        return None
    return a.label.value if a.label else a.severity.value


def moment_for(
    coach: Coach,
    request: _Request,
    *,
    named: tuple[str, int, float] | None = None,
    result: str | None = None,
) -> Moment:
    convo = coach.conversation
    user_colour = chess.WHITE if coach.user_side == Side.WHITE else chess.BLACK
    records = convo.active_records()
    last = records[-1] if records else None
    # The request's board is the position the coach's facts are for (before the player's
    # move, for a verdict on it); the evidence is read from the position as it stands now.
    board = chess.Board(last.move.fen_after) if last is not None else request.board
    with_leads = (
        _current_lines_with_leads(coach) if last is not None and last.analysis is not None else []
    )
    lines = [(san, score) for san, score, _ in with_leads]
    leads = {san: text for san, _, text in with_leads if text}
    last_best_leads = None
    if (
        last is not None
        and last.assessment is not None
        and last.assessment.best
        and last.assessment.best[0] != last.move.uci
    ):
        try:
            last_best_leads = (
                line_outcome(
                    chess.Board(last.move.fen_before), list(last.assessment.best[:8]), user_colour
                )
                or None
            )
        except (ValueError, AssertionError):
            last_best_leads = None
    tally: dict[str, int] = {}
    lost: list[str] = []
    for r in convo.judged_records():
        if not r.move.by_user or r.assessment is None:
            continue
        lab = _label(r.assessment) or ""
        tally[lab] = tally.get(lab, 0) + 1
        if r.assessment.severity.value != "none":
            better = ""
            if r.assessment.best:
                try:
                    better = (
                        "; "
                        + chess.Board(r.move.fen_before).san(
                            chess.Move.from_uci(r.assessment.best[0])
                        )
                        + " was better"
                    )
                except (ValueError, AssertionError):
                    better = ""
            lost.append(f"{convo._numbered(r.move)} ({lab}{better})")
    best_san = None
    if last is not None and last.assessment is not None and last.assessment.best:
        try:
            best_san = chess.Board(last.move.fen_before).san(
                chess.Move.from_uci(last.assessment.best[0])
            )
        except (ValueError, AssertionError):
            best_san = None
    previous = records[-2] if len(records) > 1 else None
    # The facts of the position the last move was played from: the previous record
    # carries them (they are for the position its move led to); for the first move of
    # the game, the start position's facts.
    facts_before = None
    if last is not None and last.move.by_user:
        if previous is not None:
            facts_before = previous.facts
        else:
            start = coach.positions.get(0)
            facts_before = start.facts if start is not None else None
    return Moment(
        previous_san=previous.move.san if previous else None,
        facts_before=facts_before
        if request.board.fen() != (last.move.fen_before if last else None)
        else None,
        fen_before_last=last.move.fen_before if last else None,
        material_arc=_material_arc(records, user_colour),
        facts=request.facts,
        level=request.level,
        board=board,
        user_to_move=board.turn == user_colour,
        last_san=last.move.san if last else None,
        last_by_user=last.move.by_user if last else None,
        last_label=_label(last.assessment) if last else None,
        last_loss=last.assessment.expected_loss if last and last.assessment else None,
        last_best_san=best_san,
        lines=lines,
        leads=leads,
        last_best_leads=last_best_leads,
        record=_record_lines(convo, user_colour),
        tally=tally,
        lost=lost[-6:],
        opening=_opening_so_far(records),
        material=material_text(board, user_colour),
        standing=(
            convo._standing_text(last.analysis.lines[0], board.turn, user_colour)
            if last is not None and last.analysis is not None and last.analysis.lines
            else None
        ),
        result=result,
        named_san=named[0] if named else None,
        named_rank=named[1] if named else None,
        named_loss=named[2] if named else None,
    )


def _material_arc(records, user_colour: chess.Color) -> str | None:
    """The material balance as it changed over the game, in the coach's words: the
    change points only."""
    arc: list[str] = []
    current = None
    texts = [material_text(chess.Board(r.move.fen_after), user_colour) for r in records]
    # A balance counts once it has held for two plies (or the game ended on it), so a
    # capture answered at once by a recapture does not show as a swing.
    for i, r in enumerate(records):
        text = texts[i]
        settled = i == len(records) - 1 or texts[i + 1] == text
        if settled and text != current:
            who = "you" if r.move.by_user else "they"
            arc.append(f"after {r.move.san} ({who}, ply {r.move.ply}): {text}")
            current = text
    if not arc:
        return None
    return "; ".join(arc[-8:])


def _opening_so_far(records) -> str | None:
    """The opening the game was in: the last one the book named (it names none once the
    game has left theory, and the name still answers "what opening was that?")."""
    for r in reversed(records):
        if r.analysis is not None and r.analysis.opening:
            return r.analysis.opening
    return None


def _current_lines(coach: Coach) -> list[tuple[str, float]]:
    """The engine's lines for the position the player faces, (san, score from their side)."""
    return [(san, score) for san, score, _ in _current_lines_with_leads(coach)]


def _current_lines_with_leads(coach: Coach) -> list[tuple[str, float, str]]:
    """The lines with where each leads, as the harness words it."""
    records = coach.conversation.active_records()
    if not records or records[-1].analysis is None:
        return []
    analysis = records[-1].analysis
    b = chess.Board(analysis.fen)
    user_colour = chess.WHITE if coach.user_side == Side.WHITE else chess.BLACK
    out = []
    for line in analysis.lines:
        if not line.moves:
            continue
        score = line.expected_score if b.turn == user_colour else 1.0 - line.expected_score
        try:
            san = b.san(chess.Move.from_uci(line.moves[0]))
        except (ValueError, AssertionError):
            continue
        out.append((san, round(score, 3), line_outcome(b, list(line.moves[:8]), user_colour)))
    return out


def _record_lines(convo, user_colour: chess.Color) -> list[str]:
    """Every move of the game with its label, as the record shows them: the player's
    marked as theirs, the opponent's as theirs."""
    out = []
    for r in convo.judged_records():
        who = "the player" if r.move.by_user else "the opponent"
        label = _label(r.assessment) if r.assessment is not None else "not judged"
        out.append(f"{convo._numbered(r.move)} ({who}, {label})")
    return out


def live_class(text: str) -> str:
    """The class of a question the player really asked, by the coach's own reading of it."""
    from sidecoach.coach.coach import _ASK_HINT, _ASK_LINE, _ASK_MOVE, _ASK_REVIEW

    if _ASK_REVIEW.search(text):
        return "ask_game_so_far"
    if _ASK_LINE.search(text):
        return "ask_line"
    if _ASK_MOVE.search(text):
        return "ask_move"
    if _ASK_HINT.match(text):
        return "ask_hint"
    return "ask_free"


ENGINE_CACHE = ROOT / "data" / "eval" / "engine_cache.json"


class CachedEngine:
    """Stockfish behind a cache keyed by position and depth, for the moves a player
    names: the same question gets the same lines on every rebuild, and Stockfish runs
    only for a position it has not seen."""

    def __init__(self, cache: dict[str, list[dict]] | None = None) -> None:
        self.cache: dict[str, list[dict]] = dict(cache) if cache is not None else {}
        if cache is None and ENGINE_CACHE.exists():
            self.cache = json.loads(ENGINE_CACHE.read_text(encoding="utf-8"))
        self.new: dict[str, list[dict]] = {}
        self._engine = None

    def _open(self):
        if self._engine is None:
            from sidecoach.analysis.engine import Engine
            from sidecoach.config import load_settings, require_stockfish

            settings = load_settings()
            path = require_stockfish(settings)
            self._engine = Engine(
                path=path,
                threads=2,
                hash_mb=64,
                multipv=3,
                pv_plies=settings.engine.pv_plies,
                score_model=settings.engine.score_model,
                max_seconds=3.0,
            )
            self._engine.open()
        return self._engine

    def close(self) -> None:
        """Stockfish closed between games: an open engine keeps a worker process alive at
        the pool's shutdown (its reader thread is not a daemon), and the build never ends."""
        if self._engine is not None:
            with contextlib.suppress(Exception):
                self._engine.close()
            self._engine = None

    def analyse(self, board: chess.Board, *, depth: int | None = None):
        from sidecoach.events import Line

        key = f"{board.fen()}|{depth}"
        if key not in self.cache:
            lines = self._open().analyse(board, depth=depth)
            self.cache[key] = [line.model_dump() for line in lines]
            self.new[key] = self.cache[key]
        return [Line.model_validate(x) for x in self.cache[key]]


def build_game(
    source: str,
    facts_path: Path,
    rng: random.Random,
    annotations: dict[int, str] | None,
    engine: CachedEngine | None = None,
    *,
    provider=None,
    system: str | None = None,
    draft_check=None,
    summarise_every: int = 0,
    summariser=None,
    stop_after_ply: int | None = None,
    compact_record: bool = False,
) -> list[dict[str, Any]]:
    """The game played through the coach, moment by moment: every request the coach made
    of its model as an item (the harness as it stood, the class, the required facts, the
    master's note). With a `provider` the model answers each moment and its words go
    forward into the harness of the next; each item then carries its "reply"."""
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    game = facts["name"]
    # The coach's own templates draw their wording from the global random module; seeded
    # per game, a rebuilt corpus renders the same histories and keeps its replies valid.
    random.seed(f"sidecoach-eval:{game}")
    # The questions the player really asked, by the ply they came at (after the game, at
    # the last ply); put to the coach verbatim, before any synthetic question at that ply.
    live_by_ply: dict[int, list[dict]] = {}
    for a in facts.get("asks", []):
        live_by_ply.setdefault(a["ply"], []).append(a)
    coach = Coach(
        provider=provider,  # type: ignore[arg-type] -- None: the capture answers instead
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=STUCK_SECONDS)),
        engine=engine,
        draft_check=draft_check,
        summarise_every=summarise_every,
        summariser=summariser,
        compact_record=compact_record,
    )  # type: ignore[arg-type]
    coach.sync_summaries = True
    if system is not None:
        coach.system = system
    capture = Capture(coach.speaker.speak_now if provider is not None else None)
    coach.speaker.speak_now = capture  # type: ignore[method-assign]
    coach.sync_claims = True
    items: list[dict[str, Any]] = []
    key_counts: dict[str, int] = {}
    seen = 0
    ts = 0.0
    user_turns = 0

    def drain() -> None:
        coach.speaker.run_pending()
        coach.poll()

    def take(
        kind: str,
        question: str | None,
        named: tuple[str, int, float] | None = None,
        result: str | None = None,
        reference: str | None = None,
        live: bool = False,
    ) -> None:
        nonlocal seen
        for request, spoken in zip(capture.requests[seen:], capture.spoken[seen:], strict=True):
            m = moment_for(coach, request, named=named, result=result)
            if kind == "coach":
                best_played = bool(m.last_best_san) and m.last_best_san == m.last_san
                cls_id = classify_trigger(
                    request.trigger.value, m.last_label, m.last_loss, best_played
                )
            else:
                cls_id = kind
            if cls_id is None or cls_id not in CLASSES:
                continue
            cls = CLASSES[cls_id]
            reqs = required(cls, m)
            # A quick follow-up ("They asked what to play and have just been shown that it
            # is Nf3...") is asked only for the idea; the move or line it must not restate
            # is not required of it.
            directive_text = str(request.messages[-1]["content"]) if request.messages else ""
            if "have just been shown" in directive_text:
                reqs = [f for f in reqs if f.key not in ("candidate", "line")]
            # The engine's word on a move they named is spoken first, as the coach's own
            # turn (the standing after it, how far behind the best, and at LINE the best
            # move); the reply that builds on it is not asked to say those again.
            if "have just been shown the engine's word" in directive_text:
                reqs = [f for f in reqs if f.key not in ("named_verdict", "better_move")]
            # The master's note is about the move just played: it is the reference for the
            # items about that move and for nothing else asked in the same position.
            note_for_item = reference if cls_id in ABOUT_LAST_MOVE else None
            # A stable id: the moment (ply, trigger, class, the question) and a counter for
            # a repeat of the same question at the same ply, so a rebuilt corpus keeps the
            # ids of items that did not change.
            digest = hashlib.sha1((question or "").encode()).hexdigest()[:6]
            key = f"p{request.ply:03d}-{request.trigger.value}-{cls_id}-{digest}"
            key_counts[key] = key_counts.get(key, 0) + 1
            items.append(
                {
                    "id": f"{source}/{game}/{key}-{key_counts[key]}",
                    "source": source,
                    "game": game,
                    "ply": request.ply,
                    "fen": request.board.fen(),
                    "fen_now": m.board.fen(),
                    "level": request.level.name,
                    "class": cls_id,
                    "trigger": request.trigger.value,
                    "question": question if kind != "coach" else None,
                    "live": live,
                    "system": request.system,
                    "messages": list(request.messages),
                    "candidates": request.candidates,
                    "fallback": request.fallback,
                    "required": [f.as_dict() for f in reqs],
                    "quick_followup": "have just been shown" in directive_text,
                    "form": {
                        "max_sentences": cls.max_sentences,
                        "ends_with_question": cls.ends_with_question,
                    },
                    "moment": as_json(m),
                    "reference": note_for_item,
                    **(
                        {
                            "reply": reply_record(
                                f"{source}/{game}/{key}-{key_counts[key]}", cls_id, spoken
                            )
                        }
                        if provider is not None
                        else {}
                    ),
                }
            )
        seen = len(capture.requests)

    user_side = Side.WHITE if facts["user_side"] == "white" else Side.BLACK
    coach.on_event(GameStarted(game_id=game, user_side=user_side, start_fen=facts["start_fen"]), ts)
    coach.on_event(Analysis.model_validate(facts["start_analysis"]), ts)
    drain()
    take("coach", None)
    ply_annotation = annotations or {}
    for step in facts["script"]:
        if (
            stop_after_ply is not None
            and step.get("ply") is not None
            and step["ply"] > stop_after_ply
        ):
            break  # every moment wanted from this game has been played (an "ended" step
            # carries no ply and is kept: the moments after the game are drawn too)
        if step["kind"] == "takeback":
            ts += 5
            coach.on_event(Takeback(game_id=game, plies=step["plies"], fen=step["fen"]), ts)
            drain()
            take("coach", None)
            continue
        if step["kind"] == "ended":
            ts += 2
            coach.on_event(
                GameEnded(
                    game_id=game,
                    result=step["result"],
                    reason=step.get("reason"),
                    pgn="",
                    final_fen=coach.board.fen(),
                ),
                ts,
            )
            drain()
            take("coach", None, result=step["result"])
            for a in [x for xs in live_by_ply.values() for x in xs if x.get("after_end")]:
                ts += 3

                coach.ask(a["text"], ts)
                drain()
                take(live_class(a["text"]), a["text"], result=step["result"], live=True)
            ts += 3
            q = question_for(CLASSES["ask_game_so_far"], rng)

            coach.ask(q, ts)
            drain()
            take("ask_game_so_far", q, result=step["result"])
            continue
        ts += (step.get("think_ms") or 5000) / 1000
        move = MovePlayed(
            game_id=game,
            ply=step["ply"],
            san=step["san"],
            uci=step["uci"],
            fen_before=step["fen_before"],
            fen_after=step["fen_after"],
            by_user=step["by_user"],
            think_ms=step.get("think_ms"),
        )
        coach.on_event(move, ts)
        coach.on_event(Analysis.model_validate(step["analysis"]), ts)
        coach.on_event(MoveAssessment.model_validate(step["assessment"]), ts)
        drain()
        note = ply_annotation.get(step["ply"])
        take("coach", None, reference=note)
        for a in live_by_ply.get(step["ply"], []):
            if a.get("after_end"):
                continue
            ts += 4

            coach.ask(a["text"], ts)
            drain()
            take(live_class(a["text"]), a["text"], live=True)
        if step["by_user"]:
            # After the player's own mistake: sometimes they ask why.
            a = step["assessment"]
            if a.get("severity") not in (None, "none") and rng.random() < 0.5:
                ts += 4
                q = question_for(CLASSES["ask_last_move"], rng)

                coach.ask(q, ts)
                drain()
                take("ask_last_move", q, reference=note)
            continue
        # The opponent moved: the player is to move. Sometimes a question about their move,
        # sometimes one about the position, sometimes a long think (a stuck hint).
        if rng.random() < 0.08:
            ts += 4
            q = question_for(CLASSES["ask_opponent_move"], rng)

            coach.ask(q, ts)
            drain()
            take("ask_opponent_move", q, reference=note)
        user_turns += 1
        if user_turns % ASK_EVERY == 0:
            cls_id = ROTATION_TO_MOVE[(user_turns // ASK_EVERY) % len(ROTATION_TO_MOVE)]
            named = None
            if cls_id == "ask_named_move":
                named = pick_named_move(_current_lines(coach), rng)
                if named is None:
                    cls_id = "ask_hint"
            q = question_for(CLASSES[cls_id], rng, san=named[0] if named else None)
            ts += 6

            coach.ask(q, ts)
            drain()
            take(cls_id, q, named=named)
        elif user_turns % ASK_EVERY == 1 and rng.random() < 0.25:
            ts += STUCK_SECONDS + 15
            coach.tick(ts)
            drain()
            take("coach", None)
    return items
