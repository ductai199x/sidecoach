"""Record what the desktop coach says, and every prompt it says it from, for the Kotlin.

`coach/coach.py` is the orchestrator: it decides what to say and when, builds the turn's
instruction and its template, calls the model, checks the draft for leaks and for claims the
board contradicts, retries once, falls back, and hands the message out with its board marks.
Everything but the model is deterministic, so the model here is a recorded one -- a fixed
list of replies answered by call index -- and every call's system message and user turn goes
into the fixture along with everything that came out.

A real game is played against the real Stockfish, the real tagger and the real opening book,
with the player asking questions, going quiet, taking a move back and reviewing at the end.
The draws (which template of a rotating list, which candidate is spoken) are made
deterministic by patching `random.Random.choice` and `.choices` with a counter, and the
Kotlin injects the same chooser -- so the order of the draws is compared too.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_session_fixtures.py
"""

from __future__ import annotations

import itertools
import json
import random
import sys
import threading
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

# --- the draws, made deterministic before anything imports random -----------------------------

class Draws:
    """The draws, in the order they are made: the Kotlin injects the same counter, so a
    template chosen out of turn shows up as a drift in the count rather than as prose."""

    def __init__(self) -> None:
        self.n = 0

    def take(self) -> int:
        out = self.n
        self.n += 1
        return out


_counter = Draws()


def _choice(self, seq):  # noqa: ANN001, ANN201
    items = list(seq)
    return items[_counter.take() % len(items)]


def _choices(self, population, weights=None, *, cum_weights=None, k=1):  # noqa: ANN001, ANN201
    items = list(population)
    return [items[_counter.take() % len(items)] for _ in range(k)]


def _shuffle(self, x):  # noqa: ANN001, ANN201
    """`shuffle`, pinned like `choice`: the list turned round by the next draw (the coach's
    bags, `coach/bag.py`, desktop 6f70e2e); the Kotlin injects the same."""
    if x:
        k = _counter.take() % len(x)
        x[:] = x[k:] + x[:k]


# The class, for the `random.Random()` a verdict makes of its own and the coach's bags; and the
# module's own names, which were bound to the hidden instance's methods at import and do not
# follow the class.
random.Random.choice = _choice  # type: ignore[method-assign]
random.Random.choices = _choices  # type: ignore[method-assign]
random.Random.shuffle = _shuffle  # type: ignore[method-assign]
random.choice = lambda seq: _choice(None, seq)  # type: ignore[assignment]
random.choices = lambda population, weights=None, *, cum_weights=None, k=1: _choices(  # type: ignore[assignment]
    None, population, weights, cum_weights=cum_weights, k=k
)

import chess  # noqa: E402

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.coach.claims import check_draft  # noqa: E402
from sidecoach.coach.coach import Coach  # noqa: E402
from sidecoach.config import Settings, find_stockfish  # noqa: E402
from sidecoach.events import (  # noqa: E402
    ClockObservation,
    GameEnded,
    GameReview,
    GameStarted,
    MovePlayed,
    ProfileSnapshot,
    Side,
    Takeback,
)
from sidecoach.llm.base import Completion  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
GAME_ID = "parity"
DEPTH = 12

# The replies the recorded model gives, in order, cycling. Between them they exercise every
# path: a clean distribution, one whose candidates are all too long, one that leaks a move,
# one with no marker at all, an empty one, one cut off mid-sentence, and a refusal.
REPLIES = [
    "[p=0.40] Nice -- that keeps the centre. What is your opponent's plan now?\n"
    "[p=0.35] Good, the centre is yours. What do they want next?\n"
    "[p=0.25] Solid. Where does their counterplay come from?",
    "[p=0.50] That move costs you material. The knight was hanging on f6, and after their "
    "reply the bishop drops too, and then the rook, and the game is gone, and there is "
    "nothing left to hold, and the position falls apart entirely. Do you see it?\n"
    "[p=0.50] It drops a piece. The knight on f6 was hanging, the bishop follows, the rook "
    "follows that, the queen has no squares, and the whole thing collapses on the next "
    "move. What did you expect?",
    "The centre is the thing here. Which of your pieces is doing the least?",
    "[p=0.60] Play Qh5 and it is over.\n[p=0.40] Look at the kingside. What is loose there?",
    "",
    "[p=0.70] That was fine. The centre holds, and the bishop is doing real work. Now",
    "[p=0.90] Good.\n[p=0.10] Nice one.",
    "[p=0.33] What is their most dangerous idea right now?\n"
    "[p=0.33] Before you move: what do they threaten?\n"
    "[p=0.34] Which of your pieces is worst placed?",
]

# What the verifier says. Keyed on what it was asked about, never on when: its calls run in
# a thread pool and their order is not fixed. A repair request is the one that lists the
# unsupported statements; one candidate is faulted, so the repair path runs.
VERIFY_CLEAN = '{"unsupported": []}'
VERIFY_FAULT = '{"unsupported": ["the bishop is doing real work"]}'
VERIFY_REPAIR = '{"reply": "That was fine. The centre holds."}'
VERIFY_FAULTED_PHRASE = "the bishop is doing real work"


def verifier_reply(content: str) -> str:
    """What the recorded verifier answers, from the call's own text."""
    if "Unsupported statements:" in content:
        return VERIFY_REPAIR
    return VERIFY_FAULT if VERIFY_FAULTED_PHRASE in content else VERIFY_CLEAN

# What the claims check hands back for an ask that carries a claim, in order, cycling.
CLAIM_VERDICTS = [
    ["Your knight on f3 attacks their pawn on e5, though it is defended by the c6 knight."],
    [],
    ["Your rook on a1 is not attacked at all."],
]


class RecordedProvider:
    """A model that answers by call index, and writes down every call it was given."""

    name = "recorded"
    offline = False

    def __init__(self, replies: list[str], calls: list[dict[str, Any]], tag: str) -> None:
        self.replies = replies
        self.calls = calls
        self.tag = tag
        self.n = 0

    def healthcheck(self) -> tuple[bool, str]:
        return True, "recorded"

    def chat(self, system, messages, *, max_tokens=1500, temperature=0.7, thinking=False):
        index = self.n
        self.n += 1
        text = self.replies[index % len(self.replies)]
        self.calls.append(
            {
                "kind": self.tag,
                "index": index,
                "system": system,
                "messages": [dict(m) for m in messages],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "reply": text,
            }
        )
        return Completion(text=text, stats={"recorded": True})


class RecordedVerifier:
    """The verifier: answers by what it was asked, since its calls run in a thread pool."""

    name = "recorded-verifier"
    offline = False

    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self._lock = threading.Lock()

    def healthcheck(self) -> tuple[bool, str]:
        return True, "recorded"

    def chat(self, system, messages, *, max_tokens=1500, temperature=0.7, thinking=False):
        content = "\n".join(str(m.get("content") or "") for m in messages)
        text = verifier_reply(content)
        with self._lock:
            self.calls.append(
                {
                    "kind": "verify",
                    "index": -1,
                    "system": system,
                    "messages": [dict(m) for m in messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "reply": text,
                }
            )
        return Completion(text=text, stats={"recorded": True})


def annotation_json(a: Any) -> dict[str, Any]:
    return {
        "kind": a.kind,
        "squares": list(a.squares),
        "color": a.color,
        "opacity": a.opacity,
        "label": a.label,
    }


def message_json(m: Any) -> dict[str, Any]:
    return {
        "text": m.text,
        "level": int(m.level),
        "trigger": str(m.trigger),
        "game_id": m.game_id,
        "ply": m.ply,
        "fen": m.fen,
        "annotations": [annotation_json(a) for a in m.annotations],
    }


def spoken_json(s: Any) -> dict[str, Any]:
    return {
        "text": s.message.text,
        "level": int(s.level),
        "problems": list(s.problems),
        "attempts": s.attempts,
        "recorded": s.recorded,
        "streamed": s.streamed,
        "repaired": s.repaired,
        # What the reply gave away of the answer, logged, never refused (desktop 9e2e8e2).
        "leaks": list(getattr(s, "leaks", [])),
        # What the reply did against the form asked of it, logged, never refused (desktop
        # 3ff672b).
        "notes": list(getattr(s, "notes", [])),
        "verified": [list(v) for v in s.verified],
        "calls": [
            {"candidates": c.candidates, "passed": c.passed, "prompt": c.prompt} for c in c_list
        ]
        if (c_list := list(s.calls))
        else [],
    }


class Script:
    """The steps, and everything the coach did after each one."""

    def __init__(self, coach: Coach, calls: list[dict[str, Any]]) -> None:
        self.coach = coach
        self.calls = calls
        self.steps: list[dict[str, Any]] = []
        self._spoken = 0
        self._calls = 0

    def record(self, step: dict[str, Any]) -> None:
        self.coach.speaker.run_pending()
        out = self.coach.poll()
        step["messages"] = [message_json(m) for m in out]
        step["spoken"] = [spoken_json(s) for s in self.coach.spoken[self._spoken :]]
        self._spoken = len(self.coach.spoken)
        fresh = self.calls[self._calls :]
        self._calls = len(self.calls)
        step["calls"] = [one for one in fresh if one["kind"] != "verify"]
        step["verify_calls"] = sorted(
            (one for one in fresh if one["kind"] == "verify"),
            key=lambda one: "\n".join(str(m["content"]) for m in one["messages"]),
        )
        step["fallbacks"] = self.coach.speaker.fallbacks
        step["failures"] = self.coach.speaker.failures
        step["dropped"] = self.coach.speaker.dropped
        step["last_reason"] = self.coach.policy.last_reason
        step["level"] = int(self.coach.conversation.level)
        step["draws"] = _counter.n
        step["mistakes_seen"] = self.coach.speaker.fallbacks  # a counter that only grows
        self.steps.append(step)


def play_game(
    annotator: GameAnnotator,
    coach: Coach,
    script: Script,
    moves: list[str],
    board: chess.Board,
    start_ply: int,
    start_ts: float,
    asks: dict[int, str],
) -> tuple[chess.Board, int, float]:
    ply = start_ply
    ts = start_ts
    for san in moves:
        ply += 1
        ts += 9.0
        move = board.parse_san(san)
        played_san = board.san(move)
        before_fen = board.fen()
        after = board.copy()
        after.push(move)
        by_user = board.turn == chess.WHITE
        event = MovePlayed(
            game_id=GAME_ID,
            ply=ply,
            san=played_san,
            uci=move.uci(),
            fen_before=before_fen,
            fen_after=after.fen(),
            by_user=by_user,
            think_ms=9000,
        )
        coach.on_event(event, ts)
        script.record({"op": "move", "ply": ply, "ts": ts, "san": played_san,
                       "uci": move.uci(), "fen_before": before_fen, "fen_after": after.fen(),
                       "by_user": by_user, "think_ms": 9000})

        assessment, analysis = annotator.assess(GAME_ID, board, move, ply)
        ts += 0.5
        coach.on_event(analysis, ts)
        script.record({"op": "analysis", "ply": ply, "ts": ts, "analysis": analysis_json(analysis)})
        ts += 0.5
        coach.on_event(assessment, ts)
        script.record(
            {"op": "assessment", "ply": ply, "ts": ts, "assessment": assessment_json(assessment)}
        )
        if ply in asks:
            ts += 1.0
            coach.ask(asks[ply], ts)
            script.record({"op": "ask", "ts": ts, "text": asks[ply]})
        board = after
    return board, ply, ts


def line_json(one: Any) -> dict[str, Any]:
    return {
        "moves": list(one.moves),
        "depth": one.depth,
        "expected_score": one.expected_score,
        "score_cp": one.score_cp,
        "mate": one.mate,
        "wdl": list(one.wdl) if one.wdl else None,
    }


def analysis_json(a: Any) -> dict[str, Any]:
    return {
        "fen": a.fen,
        "ply": a.ply,
        "lines": [line_json(one) for one in a.lines],
        "decisive": a.decisive,
        "only_move": a.only_move,
        "threats": list(a.threats),
        # The captures that look free and are not, and the book (desktop fc30fa7).
        "traps": [line_json(one) for one in a.traps],
        # The recaptures the best line declines (desktop a1f7f81).
        "defences": [line_json(one) for one in a.defences],
        # What a quiet best move threatens (desktop 29ae423).
        "threatens": [line_json(one) for one in a.threatens],
        "tags": [{"name": t.name, "squares": list(t.squares), "move": t.move} for t in a.tags],
        "opening": a.opening,
        "in_book": a.in_book,
    }


def assessment_json(a: Any) -> dict[str, Any]:
    return {
        "ply": a.ply,
        "uci": a.uci,
        "expected_loss": a.expected_loss,
        "severity": str(a.severity),
        "best": list(a.best),
        "tags": [{"name": t.name, "squares": list(t.squares), "move": t.move} for t in a.tags],
        "depth": a.depth,
        "label": str(a.label) if a.label else None,
        "accuracy": a.accuracy,
    }


def draft_check(text: str, board: chess.Board, after=(), us=None) -> list[str]:
    """The board check over the coach's draft, exactly as cli._draft_check wires it: since
    desktop 2ec1747 with the player's colour, whose "your" (None from an older coach)."""
    found = check_draft(text, board, after=after) if us is None else check_draft(
        text, board, after=after, us=us
    )
    return [f"{item.text}: {item.reason}" for item in found]


# The Blackburne Shilling trap: 4. Nxe5? and 5. Nxf7?? lose to the queen's raid and a mate.
TRAP = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4", "Nxe5", "Qg5", "Nxf7", "Qxg2", "Rf1", "Qxe4+",
        "Be2", "Nf3#"]


def run_blitz(annotator: GameAnnotator) -> dict[str, Any]:
    """A 5+0 game: the coach's blitz manners -- a word after each move and no lesson, the
    board's own sentence for a routine one -- through two mistakes, a mate and the review."""
    calls: list[dict[str, Any]] = []
    provider = RecordedProvider(REPLIES, calls, "coach")
    coach = Coach(
        provider,
        draft_check=draft_check,
        claims=lambda text, board, facts: [],
        summarise_every=0,
        compact_record=False,
        persona="warm_direct",
        rating=(1000, 1400),
    )
    coach.sync_claims = True
    coach.sync_summaries = True
    script = Script(coach, calls)

    ts = 1000.0
    started = GameStarted(game_id=GAME_ID, user_side=Side.WHITE, start_fen=chess.STARTING_FEN)
    coach.on_event(started, ts)
    script.record({"op": "new_game", "ts": ts, "user_side": "white",
                   "start_fen": chess.STARTING_FEN, "kind": "game"})
    ts += 1.0
    coach.on_event(ClockObservation(white_ms=300000, black_ms=300000), ts)
    script.record({"op": "clock", "ts": ts, "white_ms": 300000, "black_ms": 300000})

    board, ply, ts = play_game(annotator, coach, script, TRAP, chess.Board(), 0, ts, {})

    ts += 1.0
    coach.on_event(GameEnded(game_id=GAME_ID, result="0-1", reason="checkmate"), ts)
    script.record({"op": "ended", "ts": ts, "result": "0-1", "reason": "checkmate"})
    review_lines = [
        "You played white and lost by checkmate in 8 moves.",
        "Ground lost at 5. Nxf7, a blunder costing 0.38; O-O was better.",
    ]
    ts += 1.0
    coach.on_event(
        GameReview(game_id=GAME_ID, user_side=Side.WHITE, result="0-1", lines=review_lines), ts
    )
    script.record({"op": "review", "ts": ts, "lines": review_lines})
    ts += 1.0
    coach.ask("why was Nxf7 bad?", ts)
    script.record({"op": "ask", "ts": ts, "text": "why was Nxf7 bad?"})
    return {
        "with_verifier": False,
        "blitz": True,
        "replies": REPLIES,
        "claim_verdicts": [[]],
        "steps": script.steps,
    }


def run(annotator: GameAnnotator, with_verifier: bool) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    provider = RecordedProvider(REPLIES, calls, "coach")
    verifier = RecordedVerifier(calls) if with_verifier else None
    claim_index = itertools.count()

    def claims(text: str, board: chess.Board, facts: Any) -> list[str]:
        return list(CLAIM_VERDICTS[next(claim_index) % len(CLAIM_VERDICTS)])

    coach = Coach(
        provider,
        draft_check=draft_check,
        claims=claims,
        verifier=verifier,
        summarise_every=0,
        compact_record=False,
        persona="warm_direct",
        rating=(1000, 1400),
    )
    coach.sync_claims = True
    coach.sync_summaries = True
    script = Script(coach, calls)

    ts = 1000.0
    coach.on_event(
        ProfileSnapshot(
            games=12,
            insights=["You lose most from hanging a piece after a capture."],
            last_session="three games, two losses",
        ),
        ts,
    )
    script.record(
        {
            "op": "profile",
            "ts": ts,
            "games": 12,
            "insights": ["You lose most from hanging a piece after a capture."],
            "last_session": "three games, two losses",
        }
    )

    started = GameStarted(game_id=GAME_ID, user_side=Side.WHITE, start_fen=chess.STARTING_FEN)
    coach.on_event(started, ts)
    script.record({"op": "new_game", "ts": ts, "user_side": "white",
                   "start_fen": chess.STARTING_FEN, "kind": "game"})

    ts += 1.0
    coach.on_event(ClockObservation(white_ms=600000, black_ms=600000), ts)
    script.record({"op": "clock", "ts": ts, "white_ms": 600000, "black_ms": 600000})

    board = chess.Board()
    asks = {
        2: "what should I play",
        4: "is my knight on f3 safe?",
        6: "hint",
        8: "show me the line",
        10: "why was that bad?",
        12: "how did I do overall?",
        14: "what is the opening?",
        16: "am I winning?",
        18: "what does that move do?",
    }
    board, ply, ts = play_game(
        annotator,
        coach,
        script,
        ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"],
        board,
        0,
        ts,
        asks,
    )

    # A long think: the stuck hint, and a second one a full stuck spell later -- the idea,
    # then the focus, and never more unasked -- with hints asked between: the rungs up to
    # the move.
    for op in ("tick", "hint", "tick", "tick", "another hint"):
        if op == "tick":
            ts += 200.0
            coach.tick(ts)
            script.record({"op": "tick", "ts": ts})
        else:
            ts += 1.0
            coach.ask(op, ts)
            script.record({"op": "ask", "ts": ts, "text": op})

    # A takeback of the last two plies, and two replacements.
    back = chess.Board(coach.conversation.active_records()[-3].move.fen_after)
    ts += 5.0
    coach.on_event(Takeback(game_id=GAME_ID, plies=2, fen=back.fen()), ts)
    script.record({"op": "takeback", "ts": ts, "plies": 2, "fen": back.fen()})
    # The replacements are numbered from the position the board is back at, as the pipeline
    # numbers them: the lesson on the first one, a mistake, is spoken from the position
    # before it.
    board, ply, ts = play_game(
        annotator, coach, script, ["Nxd4", "Nxd4"], back, ply - 2, ts, {}
    )

    ts += 1.0
    coach.ask("I think Qb3 wins a pawn", ts)
    script.record({"op": "ask", "ts": ts, "text": "I think Qb3 wins a pawn"})

    ts += 1.0
    coach.on_event(GameEnded(game_id=GAME_ID, result="1-0", reason="the opponent resigned"), ts)
    script.record({"op": "ended", "ts": ts, "result": "1-0", "reason": "the opponent resigned"})

    review_lines = [
        "You played 7 moves: 1 mistake, the rest sound.",
        "The game turned at 6. Nxd4.",
        "Accuracy 82.",
    ]
    ts += 1.0
    coach.on_event(
        GameReview(
            game_id=GAME_ID,
            user_side=Side.WHITE,
            result="1-0",
            lines=review_lines,
        ),
        ts,
    )
    script.record({"op": "review", "ts": ts, "lines": review_lines})

    ts += 1.0
    coach.ask("what should I work on?", ts)
    script.record({"op": "ask", "ts": ts, "text": "what should I work on?"})

    return {
        "with_verifier": with_verifier,
        "replies": REPLIES,
        "claim_verdicts": CLAIM_VERDICTS,
        "steps": script.steps,
    }


def main() -> None:
    settings = Settings()
    path = find_stockfish(settings)
    if path is None:
        raise SystemExit("no stockfish under tools/")
    book = None
    try:
        from sidecoach.analysis.openings import OpeningBook

        book = OpeningBook.load(settings.data_dir / "openings")
    except Exception as exc:  # noqa: BLE001
        print(f"no opening book ({exc})")
    engine = Engine(path=path, threads=4, hash_mb=256, multipv=6, pv_plies=8, max_seconds=5.0)
    engine.open()
    try:
        runs = []
        for with_verifier in (False, True):
            global _counter
            _counter = Draws()
            annotator = GameAnnotator(
                engine=engine,
                depth=DEPTH,
                multipv=6,
                thresholds=Thresholds(),
                tagger=motifs,
                book=book,
            )
            runs.append(run(annotator, with_verifier))
        _counter = Draws()
        runs.append(
            run_blitz(
                GameAnnotator(
                    engine=engine,
                    depth=DEPTH,
                    multipv=6,
                    thresholds=Thresholds(),
                    tagger=motifs,
                    book=book,
                )
            )
        )
    finally:
        engine.close()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "session.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    size = (OUT / "session.json").stat().st_size
    steps = sum(len(one["steps"]) for one in runs)
    calls = sum(len(s["calls"]) for one in runs for s in one["steps"])
    print(
        f"{len(runs)} runs, {steps} steps, {calls} model calls "
        f"-> {OUT / 'session.json'} ({size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
