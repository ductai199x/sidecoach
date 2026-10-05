"""Record what the desktop's panel writes down, for the Kotlin.

`ui/panel.py`'s `PanelEngine` turns every event of a session into a line of its log -- the
game's moves, the engine's verdicts and signals, the notes on the board -- and keeps the
page's state from them. That part is pure: the events go in through `_dispatch_game` and
`_dispatch_engine` and the records come out. So a scripted game is fed through the real
engine here and every record it wrote, and the state it kept, goes into the fixture; the
phone feeds the same events to `PanelEngine.onGameEvent` / `onEngineEvents` and must write
the same lines. The coach's messages too (`_coach_meta`): each put in an offline coach's
record, with the board's ply and the voice as the case sets them, and the lines written for
it; the phone puts them in its `CoachRunner`'s coach and drains it.

With them: `log.elide` over awkward whitespace, `outcome_result` and `end_game`'s refusal,
what `Settings.model_validate` says of a save that does not fit (the settings page shows
its messages by field), and what the server answers a route it does not serve.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android-wt/ui/tools/make_panel_fixtures.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# No voice models: the panel is silent and deaf, as in the desktop's tests.
os.environ["SIDECOACH_VOICE_MODELS"] = "0"

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
from fastapi.encoders import jsonable_encoder  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from sidecoach.config import Settings, VoiceSettings  # noqa: E402
from sidecoach.events import (  # noqa: E402
    Analysis,
    BoardUnread,
    ClockObservation,
    GameEnded,
    GameStarted,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    MovePlayed,
    Resync,
    ScreenState,
    ScreenStateChanged,
    Severity,
    Side,
    Takeback,
    Unexplained,
)
from sidecoach.ui.log import elide  # noqa: E402
from sidecoach.ui.panel import OUTCOMES, PanelEngine  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "app" / "src" / "androidTest" / "assets" / "panel"
PHONE_DEFAULTS = HERE.parent / "app" / "src" / "main" / "assets" / "settings" / "defaults.json"
PHONE_SCHEMA = HERE.parent / "app" / "src" / "main" / "assets" / "settings" / "schema.json"

GAME = "g1"


def event_json(event: Any) -> dict[str, Any]:
    """An event as the Kotlin test builds it back: its type and its fields, in JSON."""
    out = {"type": type(event).__name__}
    out.update(json.loads(event.model_dump_json(exclude={"ts", "source_frame_id"})))
    return out


def moves_game() -> list[Any]:
    """A short game with everything the panel writes a line for: the moves with and without
    think times, a verdict of every kind, the engine's signals, a takeback and a resync, the
    notes on the board, a clock and the dialog over it, and the ends a game can have."""
    steps: list[Any] = [GameStarted(game_id=GAME, user_side=Side.WHITE, start_fen=chess.STARTING_FEN)]
    board = chess.Board()
    sans = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]
    thinks = [2350, None, 12050, 400, 0, 999, 61000, 1450]
    befores: list[str] = []
    for ply, (san, think) in enumerate(zip(sans, thinks, strict=True), start=1):
        before = board.fen()
        befores.append(before)
        move = board.push_san(san)
        steps.append(
            MovePlayed(
                game_id=GAME,
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
                think_ms=think,
            )
        )
    # The engine's word on each, in the order it comes: the verdicts of every kind.
    verdicts = [
        (1, "e2e4", 0.0, Severity.NONE, ["e2e4", "e7e5", "g1f3"], [], 18, Label.BOOK),
        (
            2,
            "e7e5",
            0.0456,
            Severity.INACCURACY,
            ["c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
            ["fork", "pin", "quietMove", "skewer", "attraction", "fork"],
            16,
            Label.INACCURACY,
        ),
        (3, "g1f3", 0.125, Severity.MISTAKE, ["f1c4"], ["quietMove"], 12, Label.MISTAKE),
        (4, "b8c6", 0.3, Severity.BLUNDER, ["g8f6"], ["hangingPiece", "pin"], 9, None),
        (5, "f1b5", 0.005, Severity.NONE, [], ["discoveredAttack"], 20, Label.BEST),
        # A ply the panel never saw played: the move stays in UCI, the line too.
        (42, "d2d4", 0.0625, Severity.MISTAKE, ["c2c4", "e7e5"], [], 7, Label.MISS),
        (6, "a7a6", 0.015, Severity.NONE, ["g8f6"], [], 18, Label.EXCELLENT),
    ]
    for ply, uci, loss, severity, best, tags, depth, label in verdicts:
        steps.append(
            MoveAssessment(
                game_id=GAME,
                ply=ply,
                uci=uci,
                expected_loss=loss,
                severity=severity,
                best=best,
                tags=[MotifTag(name=t) for t in tags],
                depth=depth,
                label=label,
            )
        )
    # The engine's signals on positions: the opening named once, the decisive moment and the
    # only move with their threats, a quiet position at debug, and one with no lines at all.
    after_two = chess.Board()
    for san in sans[:2]:
        after_two.push_san(san)
    after_four = chess.Board()
    for san in sans[:4]:
        after_four.push_san(san)
    after_five = chess.Board()
    for san in sans[:5]:
        after_five.push_san(san)
    steps += [
        Analysis(
            game_id=GAME,
            fen=after_two.fen(),
            ply=2,
            lines=[Line(moves=["g1f3", "b8c6", "f1b5"], depth=18, expected_score=0.625)],
            decisive=True,
            only_move=False,
            threats=["d8h4", "f8c5", "d7d5", "a7a5"],
            opening="C20 King's Pawn Game",
        ),
        Analysis(
            game_id=GAME,
            fen=after_four.fen(),
            ply=4,
            lines=[Line(moves=["f1b5", "a7a6"], depth=16, expected_score=0.5549)],
            decisive=False,
            only_move=False,
            threats=[],
            opening="C20 King's Pawn Game",
        ),
        Analysis(
            game_id=GAME,
            fen=after_four.fen(),
            ply=4,
            lines=[Line(moves=["f1b5", "a7a6"], depth=16, expected_score=0.5549)],
            decisive=True,
            only_move=True,
            # The opponent's idea, read after a null move and then one after the other, as
            # `_sans_of` reads them; a move from an empty square stops the list.
            threats=["c6d4", "d4f3", "h3h4", "g8f6"],
            opening="C60 Ruy Lopez",
        ),
        Analysis(
            game_id=GAME,
            fen=after_five.fen(),
            ply=5,
            lines=[Line(moves=["a7a6", "b5a4", "g8f6"], depth=20, expected_score=0.4375)],
            decisive=False,
            only_move=False,
            threats=["b5c6"],
        ),
        Analysis(
            game_id=GAME,
            fen=after_five.fen(),
            ply=5,
            lines=[],
            decisive=False,
            only_move=True,
            threats=[],
            opening="C60 Ruy Lopez",
        ),
        # Measured early, the full depth to follow: its "early ply" line, at debug, whatever
        # its signals (the parity audit, 2026-09-29).
        Analysis(
            game_id=GAME,
            fen=after_five.fen(),
            ply=5,
            lines=[Line(moves=["a7a6", "b5a4"], depth=12, expected_score=0.4625)],
            decisive=True,
            only_move=True,
            threats=["b5c6"],
            provisional=True,
        ),
    ]
    steps += [
        Takeback(game_id=GAME, plies=2, fen=befores[6]),
        MoveAssessment(
            game_id=GAME, ply=7, uci="b5a4", expected_loss=0.0, severity=Severity.NONE,
            best=["b5a4"], depth=14, label=Label.BEST,
        ),
        Unexplained(game_id=GAME, fen=befores[6], reason="e4 unexplained for 40 frames"),
        BoardUnread(reason="unlearned"),
        BoardUnread(reason="orientation"),
        BoardUnread(reason="look"),  # round 3: a look changed mid-game and not learned again
        ClockObservation(white_ms=180000, black_ms=175500, running=Side.BLACK),
        ClockObservation(white_ms=None, black_ms=5000, running=None),
        ScreenStateChanged(state=ScreenState.COVERED),
        ScreenStateChanged(state=ScreenState.LIVE_GAME),
        Resync(game_id=GAME, fen=befores[3], reason="position adopted"),
        GameEnded(game_id=GAME, result="*", reason="dialog", pgn="1. e4 e5 *"),
        GameEnded(game_id=GAME, result="1-0", reason="checkmate", pgn="1. e4 e5 1-0"),
        GameStarted(
            game_id="g2", user_side=Side.BLACK, start_fen=befores[4], kind="position"
        ),
        GameStarted(game_id="g3", user_side=Side.WHITE, start_fen=chess.STARTING_FEN, kind="puzzle"),
    ]
    return steps


def dispatch_fixture() -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = PanelEngine(
            Settings(data_dir=Path(tmp), voice=VoiceSettings()),
            analysis=False,
            coach=False,
            frame_source=iter(()),
        )

        class Glue:
            def submit(self, event: object) -> None:
                return None

        glue = Glue()
        opening: str | None = None
        out = []
        for event in moves_game():
            before = engine._log.cursor
            if isinstance(event, (MoveAssessment, Analysis)):
                opening = engine._dispatch_engine([event], opening, chess)
            else:
                engine._dispatch_game(event, glue, chess)
            records = engine._log.since(before)
            state = engine.poll(0)["state"]
            out.append(
                {
                    "event": event_json(event),
                    "records": [
                        {k: r[k] for k in ("chan", "level", "text", "data")} for r in records
                    ],
                    "state": {
                        k: state.get(k)
                        for k in ("labels", "game_id", "user_side", "kind", "clock")
                    },
                }
            )
        return json.loads(json.dumps(out))


class _Voice:
    """A voice that writes down what it is asked to do (test_panel_voice_policy's FakeTTS)."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.busy = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def say(self, text: str) -> None:
        self.calls.append("say:" + text)

    def stop(self) -> None:
        self.calls.append("stop")


LONG = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"

# The coach's messages as the panel logs them (`_coach_meta`): each with the board's ply, the
# voice busy or not and what it is saying, a model failure to report, and the message itself
# with what it was checked against and the calls behind it. What the voice then does decides
# the line's tail: spoken, streamed, "not spoken: board moved on", "not spoken: voice busy",
# "cut in".
COACH_CASES: list[dict[str, Any]] = [
    {
        "label": "the model's reply rejected, the quick answer standing",
        "ply_now": 6, "busy": False, "voicing": None, "failed": None,
        "spoken": {
            "text": "", "trigger": "user_ask", "level": 1, "ply": 6, "fen": None,
            "problems": ["empty reply"], "attempts": 2, "streamed": False, "leaks": [], "notes": [],
            "calls": [
                {"raw": "", "stats": {"finish_reason": "stop", "completion_tokens": 0, "elapsed_s": 1.25},
                 "candidates": 0, "passed": 0, "prompt": "# Task\nThey asked for a hint."},
                {"raw": "[p=1.0] Given", "stats": {"finish_reason": "length", "completion_tokens": 400,
                                                   "elapsed_s": 3.456, "first_token_s": 0.75},
                 "candidates": 0, "passed": 0, "prompt": "x" * 7000},
            ],
        },
    },
    {
        "label": "a draft rejected, the template said",
        "ply_now": 7, "busy": False, "voicing": None, "failed": None,
        "spoken": {
            "text": "Qxf7# was better -- it is mate, the king has no square.", "trigger": "post_mistake",
            "level": 4, "ply": 6, "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "problems": ["Qxf7: not on the board"], "attempts": 2, "streamed": False, "leaks": [], "notes": [],
            "calls": [
                {"raw": "[p=1.0] Qxf7 wins.", "stats": {"finish_reason": "stop", "completion_tokens": 12},
                 "candidates": 1, "passed": 0, "prompt": ""},
            ],
        },
    },
    {
        "label": "a stream cut short",
        "ply_now": 7, "busy": False, "voicing": None, "failed": None,
        "spoken": {
            "text": "A fine move.", "trigger": "post_move", "level": 4, "ply": 7, "fen": None,
            "problems": ["the rook on f7 is not there"], "attempts": 1, "streamed": True,
            "leaks": [], "notes": [],
            "calls": [
                {"raw": "[p=0.5] A fine move. Your rook on f7 is strong.",
                 "stats": {"finish_reason": "stopped", "first_token_s": 0.4,
                           "stream_problems": ["the rook on f7 is not there"]},
                 "candidates": 1, "passed": 0, "prompt": ""},
            ],
        },
    },
    {
        "label": "a hint the board has left",
        "ply_now": 6, "busy": False, "voicing": None, "failed": None,
        "spoken": {
            "text": "Look at the long diagonal.", "trigger": "stuck", "level": 1, "ply": 4, "fen": None,
            "problems": [], "attempts": 1, "streamed": False, "leaks": [], "notes": [], "calls": [],
        },
    },
    {
        "label": "a short word while the voice is busy",
        "ply_now": 8, "busy": True, "voicing": ["post_move", 7], "failed": None,
        "spoken": {
            "text": "Good.", "trigger": "post_move", "level": 4, "ply": 8, "fen": None,
            "problems": [], "attempts": 0, "streamed": False, "leaks": [], "notes": [], "calls": [],
        },
    },
    {
        "label": "a lesson cutting in over a stale word, its leak and its form noted",
        "ply_now": 8, "busy": True, "voicing": ["post_move", 5], "failed": None,
        "spoken": {
            "text": LONG, "trigger": "post_mistake", "level": 4, "ply": 8, "fen": None,
            "problems": [], "attempts": 1, "streamed": False,
            "leaks": ["withheld square f7"], "notes": ["too long: 5 sentences, 3 asked for"],
            "calls": [],
        },
    },
    {
        "label": "a question spoken, and a model call that failed",
        "ply_now": 8, "busy": False, "voicing": None, "failed": "RuntimeError: 503: down",
        "spoken": {
            "text": "Their king has one square left. Which check takes it?", "trigger": "decisive_question",
            "level": 1, "ply": 8, "fen": None, "problems": [], "attempts": 1, "streamed": False,
            "leaks": [], "notes": [], "calls": [],
        },
    },
    {
        "label": "the review, with no ply",
        "ply_now": 9, "busy": False, "voicing": None, "failed": None,
        "spoken": {
            "text": "You won in 4 moves.", "trigger": "review", "level": 4, "ply": None, "fen": None,
            "problems": [], "attempts": 1, "streamed": False, "leaks": [], "notes": [], "calls": [],
        },
    },
]


def coach_meta_fixture() -> list[dict[str, Any]]:
    """The coach's per-message lines (`_coach_meta`, the phone's `CoachRunner.drain`): each
    case's message put in an offline coach's record, with the panel's ply and voice as the case
    sets them, and the "coach" records it wrote -- "model reply rejected", "draft rejected ->
    template", "cut short", "not spoken: ...", "cut in", the notes, each model reply's figures,
    "LLM call failed" (the parity audit, 2026-09-29)."""
    from types import SimpleNamespace

    from sidecoach.coach.coach import Coach, ModelCall, Spoken
    from sidecoach.events import CoachMessage, DisclosureLevel, Trigger
    from sidecoach.llm.offline import OfflineProvider

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = PanelEngine(
            Settings(data_dir=Path(tmp), voice=VoiceSettings()),
            analysis=False,
            coach=False,
            frame_source=iter(()),
        )
        voice = _Voice()
        engine._tts = voice  # type: ignore[assignment]
        coach = Coach(OfflineProvider())
        coaching = SimpleNamespace(coach=coach)
        engine._coaching = coaching  # type: ignore[assignment]
        seen, failures = 0, 0
        out = []
        for case in COACH_CASES:
            engine._state["ply"] = case["ply_now"]
            voice.busy = case["busy"]
            engine._voicing = tuple(case["voicing"]) if case["voicing"] else None  # type: ignore[assignment]
            s = case["spoken"]
            message = CoachMessage(
                text=s["text"],
                level=DisclosureLevel(s["level"]),
                trigger=Trigger(s["trigger"]),
                game_id=GAME,
                ply=s["ply"],
                fen=s["fen"],
            )
            coach.spoken.append(
                Spoken(
                    message=message,
                    level=DisclosureLevel(s["level"]),
                    problems=list(s["problems"]),
                    attempts=s["attempts"],
                    calls=[
                        ModelCall(c["raw"], dict(c["stats"]), c["candidates"], c["passed"], c["prompt"])
                        for c in s["calls"]
                    ],
                    streamed=s["streamed"],
                    leaks=list(s["leaks"]),
                    notes=list(s["notes"]),
                )
            )
            if case["failed"]:
                coach.speaker.failures += 1
                coach.speaker.last_error = case["failed"]
            before = engine._log.cursor
            engine._coach_meta(coaching)
            records = [r for r in engine._log.since(before) if r["chan"] == "coach"]
            out.append(
                {
                    **case,
                    "records": [{k: r[k] for k in ("chan", "level", "text", "data")} for r in records],
                }
            )
        return json.loads(json.dumps(out))


ELIDES = [
    ("short", 80),
    ("word " * 40, 20),
    ("a\n  b\t c", 80),
    ("x" * 80, 80),
    ("x" * 81, 80),
    ("ends with a space here " + "y" * 60, 30),
    ("no\u00a0break\u2003em\u3000ideographic\u2028line\u0085next", 80),
    ("zero\u200bwidth is not space", 80),
    ("emoji \U0001f600\U0001f600\U0001f600 count as one each", 10),
    ("\U0001f600" * 12, 10),
    ("\x1cfile\x1dgroup\x1erecord\x1funit", 80),
    ("", 80),
    ("   ", 80),
    ("trailing cut   " + "z" * 40, 15),
]


def outcome_fixture() -> dict[str, Any]:
    table = [
        {"outcome": o, "side": s, "result": list(PanelEngine.outcome_result(o, s))}
        for o in (*OUTCOMES, "rage-quit")
        for s in ("white", "black", None)
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = PanelEngine(
            Settings(data_dir=Path(tmp), voice=VoiceSettings()),
            analysis=False,
            coach=False,
            frame_source=iter(()),
        )
        try:
            engine.end_game("rage-quit")
            refused = ""
        except ValueError as exc:
            refused = str(exc)
    return {"table": table, "refused": refused}


def settings_fixture() -> list[dict[str, Any]]:
    """What a save that does not fit is told, field by field; and what one that fits is saved
    as. Only fields the phone's page shows are changed: the phone checks against its schema."""
    base = json.loads(PHONE_DEFAULTS.read_text(encoding="utf-8"))
    schema = json.loads(PHONE_SCHEMA.read_text(encoding="utf-8"))
    shown = {
        section: set(schema["$defs"][prop["$ref"].rsplit("/", 1)[-1]]["properties"])
        for section, prop in schema["properties"].items()
        if "$ref" in prop
    }
    cases: list[tuple[str, dict[str, Any]]] = [
        ("depth too deep", {"engine.depth": 999}),
        ("depth too shallow", {"engine.depth": 3}),
        ("depth words", {"engine.depth": "deep"}),
        ("depth a fraction", {"engine.depth": 12.5}),
        ("depth a whole float", {"engine.depth": 12.0}),
        ("depth as text", {"engine.depth": "12"}),
        ("depth as padded text", {"engine.depth": " 14 "}),
        ("depth with a plus", {"engine.depth": "+15"}),
        ("depth as float text", {"engine.depth": "12.0"}),
        ("depth with underscores", {"engine.depth": "1_0"}),
        ("depth with a dot", {"engine.depth": "12."}),
        ("depth a fraction as text", {"engine.depth": "12.5"}),
        ("depth padded with a newline", {"engine.depth": " 7\n"}),
        ("depth an exponent", {"engine.depth": "1e1"}),
        ("depth null", {"engine.depth": None}),
        ("depth true", {"engine.depth": True}),
        ("depth a list", {"engine.depth": [12]}),
        ("workers too many", {"engine.workers": 100}),
        ("workers none", {"engine.workers": 0}),
        ("workers words", {"engine.workers": "many"}),
        ("workers as text", {"engine.workers": "3"}),
        ("workers a fraction", {"engine.workers": 2.5}),
        ("workers a whole float", {"engine.workers": 2.0}),
        ("workers empty", {"engine.workers": ""}),
        ("workers null", {"engine.workers": None}),
        ("workers an exponent", {"engine.workers": "1e1"}),
        ("workers true", {"engine.workers": True}),
        ("workers hex", {"engine.workers": "0x1"}),
        ("workers padded", {"engine.workers": " 2 "}),
        ("quiet yes", {"coach.quiet": "yes"}),
        ("quiet OFF", {"coach.quiet": "OFF"}),
        ("quiet maybe", {"coach.quiet": "maybe"}),
        ("quiet one", {"coach.quiet": 1}),
        ("quiet two", {"coach.quiet": 2}),
        ("quiet a float", {"coach.quiet": 1.0}),
        ("quiet null", {"coach.quiet": None}),
        ("quiet padded", {"coach.quiet": " yes "}),
        ("quiet one and a half", {"coach.quiet": 1.5}),
        ("quiet t", {"coach.quiet": "t"}),
        ("quiet zero text", {"coach.quiet": "0"}),
        ("persona a number", {"coach.persona": 5}),
        ("persona null", {"coach.persona": None}),
        ("persona unknown", {"coach.persona": "grumpy"}),
        ("price negative", {"llm.price_prompt_per_m": -1}),
        ("price words", {"llm.price_prompt_per_m": "cheap"}),
        ("price null", {"llm.price_prompt_per_m": None}),
        ("price as text", {"llm.price_prompt_per_m": "0.25"}),
        ("model a number", {"llm.model": 7}),
        ("model null", {"llm.model": None}),
        ("provider null", {"llm.provider": None}),
        ("listen anything", {"voice.listen": "shout"}),
        ("speed too fast", {"voice.speed": 3}),
        ("fps zero", {"capture.fps": 0}),
        ("rating too low", {"profile.rating_low": 50}),
        ("inaccuracy over one", {"coach.inaccuracy": 1.5}),
        ("timeout empty", {"llm.timeout_s": ""}),
        (
            "two at once",
            {"engine.depth": 999, "coach.stuck_seconds": 1, "profile.rating_high": "high"},
        ),
        ("section not a dictionary", {"coach": 5}),
        ("section null", {"engine": None}),
        ("unknown field", {"coach.bogus": 1}),
        ("verify model null", {"coach.verify_model": None}),
        # The board's own validators (BoardSettings._known_pieces, _known_theme).
        ("pieces unknown", {"board.pieces": "neo"}),
        ("pieces an own set", {"board.pieces": "own:neo-2"}),
        ("pieces an own set of no such id", {"board.pieces": "own:../x"}),
        ("pieces the one own set from before", {"board.pieces": "custom"}),
        ("pieces with a quote", {"board.pieces": "it's"}),
        ("theme unknown", {"board.theme": "plaid"}),
        ("theme known", {"board.theme": "night"}),
    ]
    out = []
    for label, changes in cases:
        body = copy.deepcopy(base)
        for dotted, value in changes.items():
            if "." in dotted:
                section, key = dotted.split(".", 1)
                assert key in shown[section] or key == "bogus", f"{dotted} is not on the phone's page"
                body[section][key] = value
            else:
                body[dotted] = value
        try:
            new = Settings.model_validate(body)
        except ValidationError as exc:
            out.append(
                {
                    "label": label,
                    "changes": changes,
                    "errors": jsonable_encoder(exc.errors(include_url=False)),
                }
            )
            continue
        dumped = new.model_dump(mode="json")
        saved = {}
        for dotted in changes:
            if "." in dotted:
                section, key = dotted.split(".", 1)
                saved[dotted] = dumped[section].get(key, "<dropped>")
        out.append({"label": label, "changes": changes, "saved": saved})
    return json.loads(json.dumps(out))


def served_fixture() -> list[dict[str, str]]:
    """Every API route the desktop's server serves, by method: the phone must serve each."""
    from fastapi.routing import APIRoute

    from sidecoach.ui.server import build_app

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = PanelEngine(
            Settings(data_dir=Path(tmp), voice=VoiceSettings()),
            analysis=False,
            coach=False,
            frame_source=iter(()),
        )
        app = build_app(engine)
        return sorted(
            (
                {"method": method, "path": route.path}
                for route in app.routes
                if isinstance(route, APIRoute) and route.path.startswith("/api/")
                for method in sorted(route.methods)
            ),
            key=lambda r: (r["path"], r["method"]),
        )


def routes_fixture() -> list[dict[str, Any]]:
    """What the server answers a route it does not serve, or a method a route does not take."""
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    probes = [
        ("GET", "/api/nope"),
        ("POST", "/api/nope"),
        ("GET", "/api/start"),
        ("GET", "/api/flip"),
        ("POST", "/api/poll"),
        ("POST", "/api/health"),
        ("DELETE", "/api/poll"),
        ("PUT", "/api/start"),
        ("POST", "/api/settings"),
        ("DELETE", "/api/settings"),
        ("GET", "/api/setup/start"),
        ("POST", "/api/setup/status"),
        ("GET", "/api/puzzle/abc/open"),
        ("POST", "/api/puzzle/abc/poll"),
        ("GET", "/api/puzzle/abc/nope"),
        ("GET", "/api/pieces/own"),
        ("GET", "/api/pieces/own/abc"),
        ("POST", "/api/pieces/own/abc"),
        ("PUT", "/api/pieces/own/abc/images"),
        ("POST", "/api/pieces/own/abc/nope"),
        ("DELETE", "/api/pieces/own/abc/restore"),
    ]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        engine = PanelEngine(
            Settings(data_dir=Path(tmp), voice=VoiceSettings()),
            analysis=False,
            coach=False,
            frame_source=iter(()),
        )
        client = TestClient(build_app(engine), base_url="http://127.0.0.1", raise_server_exceptions=False)
        out = []
        for method, path in probes:
            reply = client.request(method, path)
            try:
                body: Any = reply.json()
            except ValueError:
                body = reply.text
            out.append({"method": method, "path": path, "status": reply.status_code, "body": body})
        # A body that fits the route but names no outcome the panel knows.
        reply = client.post("/api/end", json={"outcome": "rage-quit"})
        out.append(
            {
                "method": "POST",
                "path": "/api/end",
                "json": {"outcome": "rage-quit"},
                "status": reply.status_code,
                "body": reply.text,
            }
        )
        return out


def main() -> None:
    payload = {
        "dispatch": dispatch_fixture(),
        "coach_meta": coach_meta_fixture(),
        "elide": [{"text": t, "limit": n, "out": elide(t, n)} for t, n in ELIDES],
        "outcomes": outcome_fixture(),
        "settings": settings_fixture(),
        "routes": routes_fixture(),
        "served": served_fixture(),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "panel.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size = (OUT / "panel.json").stat().st_size
    print(
        f"{len(payload['dispatch'])} dispatch steps, {len(payload['elide'])} elides, "
        f"{len(payload['settings'])} settings cases, {len(payload['routes'])} route probes "
        f"-> {OUT / 'panel.json'} ({size / 1e3:.0f} kB)"
    )


if __name__ == "__main__":
    main()
