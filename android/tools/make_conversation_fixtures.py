"""Record the harness the desktop builds, move by move, for the Kotlin to be held to.

`coach/conversation.py` turns the game's events into the one user message the model reads,
and `coach/prompt.py` builds the system message that goes with it. Both are pure functions
of the events, so a real game is played here against the real Stockfish, the real tagger
and the real opening book -- the engine's lines, verdicts, motifs and facts all go into the
fixture -- and the Kotlin replays the same events and must write the same text, after every
single step.

The game is played twice, once plain and once with the compact record, and takes a takeback,
a re-read of the board, a dialog, notes, a review and the end of the game on the way.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_conversation_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.coach import prompt as PROMPT  # noqa: E402
from sidecoach.coach import summarise as SUM  # noqa: E402
from sidecoach.coach.conversation import Conversation  # noqa: E402
from sidecoach.coach.facts import facts_from_analysis  # noqa: E402
from sidecoach.config import Settings, find_stockfish  # noqa: E402
from sidecoach.events import (  # noqa: E402
    DisclosureLevel,
    GameEnded,
    GameStarted,
    Line,
    MovePlayed,
    Resync,
    Side,
    Takeback,
)

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
GAME_ID = "parity"
DEPTH = 12


def line_json(line: Line) -> dict[str, Any]:
    return {
        "moves": list(line.moves),
        "depth": line.depth,
        "expected_score": line.expected_score,
        "score_cp": line.score_cp,
        "mate": line.mate,
        "wdl": list(line.wdl) if line.wdl else None,
    }


def tag_json(tag: Any) -> dict[str, Any]:
    return {"name": tag.name, "squares": list(tag.squares), "move": tag.move}


def facts_json(facts: Any) -> dict[str, Any]:
    out = facts.model_dump()
    return {key: out[key] for key in out}


class Script:
    """The steps, and what the desktop's conversation said after each one."""

    def __init__(self, convo: Conversation) -> None:
        self.convo = convo
        self.steps: list[dict[str, Any]] = []

    def record(self, step: dict[str, Any]) -> None:
        new = self.convo.drain_new()
        step["render"] = self.convo.render()
        step["drained"] = [e.kind for e in new]
        step["active"] = len(self.convo.active_records())
        step["judged"] = len(self.convo.judged_records())
        self.steps.append(step)


def play(
    annotator: GameAnnotator,
    script: Script,
    board: chess.Board,
    san: str,
    ply: int,
    by_user: bool,
    *,
    think_ms: int | None = None,
    clock: tuple[int | None, int | None] | None = None,
    user_colour: chess.Color,
) -> chess.Board:
    """One move, with its verdict, its analysis and its facts: the three events a real
    game sends, in the order the coach gets them."""
    move = board.parse_san(san)
    played_san = board.san(move)
    before_fen = board.fen()
    after = board.copy()
    after.push(move)

    event = MovePlayed(
        game_id=GAME_ID,
        ply=ply,
        san=played_san,
        uci=move.uci(),
        fen_before=before_fen,
        fen_after=after.fen(),
        by_user=by_user,
        think_ms=think_ms,
    )
    script.convo.move(event)
    script.record(
        {
            "op": "move",
            "ply": ply,
            "san": played_san,
            "uci": move.uci(),
            "fen_before": before_fen,
            "fen_after": after.fen(),
            "by_user": by_user,
            "think_ms": think_ms,
        }
    )
    if clock is not None:
        script.convo.clock(clock[0], clock[1])
        script.record({"op": "clock", "white_ms": clock[0], "black_ms": clock[1]})

    assessment, analysis = annotator.assess(GAME_ID, board, move, ply)
    script.convo.assess(assessment)
    script.record(
        {
            "op": "assess",
            "ply": ply,
            "uci": assessment.uci,
            "expected_loss": assessment.expected_loss,
            "severity": str(assessment.severity),
            "best": list(assessment.best),
            "tags": [tag_json(t) for t in assessment.tags],
            "depth": assessment.depth,
            "label": str(assessment.label) if assessment.label else None,
            "accuracy": assessment.accuracy,
        }
    )

    facts = facts_from_analysis(after, analysis, last_move=played_san, user=user_colour)
    analysed(script, analysis, facts, after)
    return after


def analyse_start(
    annotator: GameAnnotator, script: Script, board: chess.Board, *, user_colour: chess.Color
) -> None:
    """The start's analysis (ply 0) -- a puzzle's, a joined game's, a re-read board's -- as
    the analyser measures it before any move (`cli` submit_position)."""
    analysis = annotator.analysis_event(GAME_ID, board, 0)
    facts = facts_from_analysis(board, analysis, last_move=None, user=user_colour)
    analysed(script, analysis, facts, board)


def analysed(script: Script, analysis: Any, facts: Any, after: chess.Board) -> None:
    """The analysis of the position `after` and its facts, into the conversation and the
    script."""
    script.convo.analysed(analysis, facts)
    # The position's facts as the lesson after the engine's move would read them, past: the
    # board once the move (and the reply) is played leaves out an idea about a piece gone
    # from its square, and the board right after it says a threat the move stopped too
    # (`faced_text`; the parity audit TG-09).
    played_board = then_board = None
    if analysis.lines and analysis.lines[0].moves:
        first = chess.Move.from_uci(analysis.lines[0].moves[0])
        if first in after.legal_moves:
            played_board = after.copy(stack=False)
            played_board.push(first)
            then_board = played_board
            if len(analysis.lines[0].moves) > 1:
                reply = chess.Move.from_uci(analysis.lines[0].moves[1])
                if reply in played_board.legal_moves:
                    then_board = played_board.copy(stack=False)
                    then_board.push(reply)
    script.record(
        {
            "op": "analysed",
            "ply": analysis.ply,
            "fen": analysis.fen,
            "lines": [line_json(one) for one in analysis.lines],
            "decisive": analysis.decisive,
            "only_move": analysis.only_move,
            "threats": list(analysis.threats),
            "tags": [tag_json(t) for t in analysis.tags],
            "opening": analysis.opening,
            "facts": facts_json(facts),
            "faced": {
                "after": None if then_board is None else then_board.fen(),
                "played": None if played_board is None else played_board.fen(),
                "text": script.convo.faced_text(facts, after, then_board, played_board),
                "text_bare": script.convo.faced_text(facts, after),
            },
        }
    )


def run(annotator: GameAnnotator, compact: bool) -> dict[str, Any]:
    """The whole game, with everything that can happen to a record on the way."""
    convo = Conversation()
    convo.compact_record = compact
    script = Script(convo)
    user_colour = chess.WHITE

    started = GameStarted(game_id=GAME_ID, user_side=Side.WHITE, start_fen=chess.STARTING_FEN)
    convo.new_game(
        started,
        control="5+0",
        profile=[
            "You lose most from hanging a piece after a capture.",
            "Your last session was three games, two losses.",
        ],
    )
    script.record(
        {
            "op": "new_game",
            "user_side": "white",
            "start_fen": chess.STARTING_FEN,
            "kind": "game",
            "control": "5+0",
            "profile": [
                "You lose most from hanging a piece after a capture.",
                "Your last session was three games, two losses.",
            ],
        }
    )

    board = chess.Board()
    ply = 0
    opening = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"]
    for index, san in enumerate(opening):
        ply += 1
        by_user = ply % 2 == 1
        board = play(
            annotator,
            script,
            board,
            san,
            ply,
            by_user,
            think_ms=1250 + index * 400,
            clock=(300000 - ply * 4000, 300000 - ply * 3000) if index % 3 == 0 else None,
            user_colour=user_colour,
        )
        if index == 1:
            convo.coach_said("Good start. What does the centre look like now?")
            script.record({"op": "coach_said", "text": "Good start. What does the centre look "
                                                        "like now?", "nod": False})
        if index == 3:
            convo.user_said("my knight is attacking e5", ["My knight on f3 attacks their pawn "
                                                          "on e5: true."])
            script.record(
                {
                    "op": "user_said",
                    "text": "my knight is attacking e5",
                    "verdicts": ["My knight on f3 attacks their pawn on e5: true."],
                }
            )
        if index in (5, 7, 9):
            convo.coach_said("Good.", nod=True)
            script.record({"op": "coach_said", "text": "Good.", "nod": True})

    # The disclosure level rises as the coach lets more through: two hints, then the answer.
    for level in (
        DisclosureLevel.SILENT,
        DisclosureLevel.IDEA,
        DisclosureLevel.FOCUS,
        DisclosureLevel.MOVE,
        DisclosureLevel.LINE,
    ):
        convo.level = level
        script.record({"op": "level", "level": int(level)})

    convo.level = DisclosureLevel.IDEA
    script.record({"op": "level", "level": int(DisclosureLevel.IDEA)})

    convo.note("The player switched to the analysis board and came back.")
    script.record({"op": "note", "text": "The player switched to the analysis board and came "
                                          "back."})

    # A takeback of the last two plies, then two different moves.
    taken = 2
    back = chess.Board(convo.active_records()[-(taken + 1)].move.fen_after)
    convo.takeback(Takeback(game_id=GAME_ID, plies=taken, fen=back.fen()))
    script.record({"op": "takeback", "plies": taken, "fen": back.fen()})
    board = back
    for san in ("Nxd4", "Nxd4"):
        ply += 1
        board = play(
            annotator, script, board, san, ply, ply % 2 == 1, user_colour=user_colour
        )

    convo.user_said("is my bishop on c4 safe?")
    script.record({"op": "user_said", "text": "is my bishop on c4 safe?", "verdicts": None})
    convo.coach_said("Count the attackers and the defenders on c4; what do you get?")
    script.record(
        {
            "op": "coach_said",
            "text": "Count the attackers and the defenders on c4; what do you get?",
            "nod": False,
        }
    )

    # What can be folded, and the fold.
    pending = convo.foldable(4)
    summary_before = convo.summary
    summary = (
        "The coach asked about the centre after 2. Nf3 and nodded at the player's sound "
        "moves; the player claimed the knight hit e5, which held."
    )
    convo.fold([e for _, e in pending], summary)
    script.record(
        {
            "op": "fold",
            "keep": 4,
            "pending": [{"after": after, "kind": e.kind, "text": e.text} for after, e in pending],
            # The desktop's own request (summarise.summary_request, desktop c5c488f).
            "summary_request": [dict(m) for m in SUM.summary_request(pending, summary_before)],
            "summary": summary,
        }
    )

    # A re-read of the board: the numbering restarts from a middlegame position.
    fresh = chess.Board("r1bq1rk1/pppp1ppp/2n2n2/8/1bBPP3/2N2N2/PP3PPP/R1BQ1RK1 b - - 0 8")
    convo.resync(Resync(game_id=GAME_ID, fen=fresh.fen(), reason="the board moved"), fresh)
    script.record({"op": "resync", "fen": fresh.fen(), "reason": "the board moved"})
    # The re-read position's own analysis, before any move: the position now until one.
    analyse_start(annotator, script, fresh, user_colour=user_colour)
    convo.user_said("what is going on here?")
    script.record({"op": "user_said", "text": "what is going on here?", "verdicts": None})
    board = fresh
    for san in ("Bxc3", "bxc3", "Nxe4", "Qd3"):
        ply += 1
        board = play(
            annotator, script, board, san, ply, board.turn == chess.WHITE, user_colour=user_colour
        )

    convo.covered()
    script.record({"op": "covered"})
    convo.review(
        [
            "The player played 24 moves: 1 blunder, 2 inaccuracies, the rest sound.",
            "The game turned at 9... Bxc3.",
        ]
    )
    script.record(
        {
            "op": "review",
            "lines": [
                "The player played 24 moves: 1 blunder, 2 inaccuracies, the rest sound.",
                "The game turned at 9... Bxc3.",
            ],
        }
    )
    convo.ended(GameEnded(game_id=GAME_ID, result="1-0", reason="the opponent resigned"))
    script.record({"op": "ended", "result": "1-0", "reason": "the opponent resigned"})

    # The last record rendered at every level, and as the record's own one line.
    last = convo.active_records()[-1]
    plies = {
        "short": convo.render_ply(last, full=False),
        "levels": {
            str(int(level)): convo.render_ply(last, level)
            for level in DisclosureLevel
        },
    }
    return {"compact": compact, "steps": script.steps, "last_ply": plies}


# Tai's puzzle (2026-10-01), Black to move: the stuck hint was asked of the model with no
# position now at all, and it named Black's own Qxf2+ as the opponent's threat.
PUZZLE = "4r1k1/p6p/6p1/2Q5/8/N1P2q1P/PP1K1P2/R6R b - - 0 1"


def run_puzzle(annotator: GameAnnotator) -> dict[str, Any]:
    """A puzzle, the player Black: the position now is the start's -- its board, its lines,
    its facts at every level -- until a move; again after a takeback to the start."""
    convo = Conversation()
    script = Script(convo)
    user_colour = chess.BLACK
    convo.new_game(
        GameStarted(game_id=GAME_ID, user_side=Side.BLACK, start_fen=PUZZLE, kind="puzzle")
    )
    script.record(
        {
            "op": "new_game",
            "user_side": "black",
            "start_fen": PUZZLE,
            "kind": "puzzle",
            "control": "",
            "profile": [],
        }
    )
    start = chess.Board(PUZZLE)
    analyse_start(annotator, script, start, user_colour=user_colour)
    for level in (
        DisclosureLevel.SILENT,
        DisclosureLevel.IDEA,
        DisclosureLevel.FOCUS,
        DisclosureLevel.MOVE,
        DisclosureLevel.LINE,
        DisclosureLevel.IDEA,
    ):
        convo.level = level
        script.record({"op": "level", "level": int(level)})
    convo.user_said("what is my opponent threatening?")
    script.record({"op": "user_said", "text": "what is my opponent threatening?", "verdicts": None})
    convo.coach_said("Look at what their queen on c5 can reach.")
    script.record(
        {"op": "coach_said", "text": "Look at what their queen on c5 can reach.", "nod": False}
    )
    board = start
    for ply, san in enumerate(("Qxf2+", "Kd3"), start=1):
        board = play(annotator, script, board, san, ply, ply == 1, user_colour=user_colour)
    convo.takeback(Takeback(game_id=GAME_ID, plies=2, fen=PUZZLE))
    script.record({"op": "takeback", "plies": 2, "fen": PUZZLE})
    convo.user_said("and now?")
    script.record({"op": "user_said", "text": "and now?", "verdicts": None})
    board = play(annotator, script, start, "Re2+", 1, True, user_colour=user_colour)
    last = convo.active_records()[-1]
    plies = {
        "short": convo.render_ply(last, full=False),
        "levels": {str(int(level)): convo.render_ply(last, level) for level in DisclosureLevel},
    }
    return {"compact": False, "steps": script.steps, "last_ply": plies}


def prompts() -> dict[str, Any]:
    """The system message, every persona and both with and without C."""
    out: dict[str, Any] = {
        "corrections": PROMPT.corrections(),
        "contract": PROMPT.CONTRACT,
        "rules": PROMPT.RULES,
        "default_persona": PROMPT.DEFAULT_PERSONA,
        "personas": {k: list(v) for k, v in PROMPT.PERSONAS.items()},
        # The order too, as an array: a JSON object's key order is the parser's to keep.
        "persona_order": list(PROMPT.PERSONAS),
        "cases": [],
    }
    names = [None, "", *PROMPT.PERSONAS, "no such persona"]
    ratings = [None, (1000, 1200), (1800, 2000)]
    for persona in names:
        for rating in ratings:
            for with_c in (True, False):
                out["cases"].append(
                    {
                        "persona": persona,
                        "rating": list(rating) if rating else None,
                        "with_corrections": with_c,
                        "persona_text": PROMPT.persona_text(persona, rating),
                        "system": PROMPT.system_message(
                            with_corrections=with_c, persona=persona, rating=rating
                        ),
                    }
                )
    return out


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
        print(f"no opening book ({exc}); the opening lines will be absent")
    engine = Engine(path=path, threads=1, hash_mb=256, multipv=6, pv_plies=8)
    engine.open()
    try:
        runs = []
        for compact in (False, True):
            annotator = GameAnnotator(
                engine=engine,
                depth=DEPTH,
                multipv=6,
                thresholds=Thresholds(),
                tagger=motifs,
                book=book,
            )
            runs.append(run(annotator, compact))
        annotator = GameAnnotator(
            engine=engine,
            depth=DEPTH,
            multipv=6,
            thresholds=Thresholds(),
            tagger=motifs,
            book=book,
        )
        runs.append(run_puzzle(annotator))
    finally:
        engine.close()

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"runs": runs, "prompt": prompts()}
    (OUT / "conversation.json").write_text(json.dumps(payload), encoding="utf-8")
    size = (OUT / "conversation.json").stat().st_size
    steps = sum(len(one["steps"]) for one in runs)
    print(
        f"{len(runs)} runs, {steps} steps, {len(payload['prompt']['cases'])} prompts "
        f"-> {OUT / 'conversation.json'} ({size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
