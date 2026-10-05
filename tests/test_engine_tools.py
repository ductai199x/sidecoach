"""The engine at the coach model's call while it answers a question (Tai, 2026-09-28: "the AI
coach should be trying to answer from all the info or possibly even REQUEST info from the
engine"): the tools' answers, and the Speaker's rounds of calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import chess

from sidecoach.coach.coach import Coach
from sidecoach.coach.engine_tools import TOOL_ROUNDS, TOOL_WORDS, EngineTools
from sidecoach.events import DisclosureLevel, Line
from sidecoach.llm.base import ChatMessage, Completion, ToolCall
from tests.coaching import _policy, drain, mate_analysis, play_to_scholar

# Tai's puzzle after 3...Rxf1+: White to move, Kg2 the only move; his bishop on a7 guards g1.
AFTER_RXF1 = "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4"
BEFORE_RXF1 = "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3NrB1K b - - 2 3"


class FirstLegal:
    """An engine that plays the first legal moves, a line of four, the mover a little ahead."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        self.seen.append(board.fen())
        replay = board.copy(stack=False)
        moves: list[str] = []
        for _ in range(4):
            legal = sorted(replay.legal_moves, key=lambda m: m.uci())
            if not legal:
                break
            moves.append(legal[0].uci())
            replay.push(legal[0])
        return [Line(moves=moves, depth=depth or 14, expected_score=0.6)] if moves else []


def test_analyse_plays_the_moves_and_gives_the_standing_the_lines_and_the_board() -> None:
    tools = EngineTools(chess.Board(AFTER_RXF1), None, FirstLegal(), chess.BLACK)
    out = tools.run("analyse", {"moves": ["4. Kg2", "4...Rg1+"]})
    assert out.startswith("After 4. Kg2 Rg1+ (from the board now): White to move, in check.")
    # The first move rated against the engine's best from the board now: Kg2 is the only move.
    # Said as the first move's, never "best there": printed after the line's end, that read
    # as the best move at its end ("White's best is 8. Kg2", Tai's puzzle).
    assert "The first move given, 4. Kg2, is the engine's best move from the board now." in out
    # White to move and a little ahead: from the player's (Black's) side, a little behind.
    assert "Standing: it is about equal (0.40 from the player's side)." in out
    assert "The engine's lines, best first:" in out and "move by move:" in out
    # The lists one a line, as in the harness's section.
    assert "\nOn the board there (complete lists):\n  The " in out
    # The boards it went through, for the draft check: after each move and along line 1.
    assert chess.Board(AFTER_RXF1).fen() not in [b.fen() for b in tools.frames]
    assert len(tools.frames) >= 2
    assert tools.log == ["analyse(Kg2 Rg1+)"]


def test_a_call_the_tools_cannot_answer_is_told_what_was_wrong() -> None:
    """A move that cannot be played stops the analysis and is named; so are a square that is
    none, a start before a last move there is not, and a tool that does not exist."""
    tools = EngineTools(chess.Board(AFTER_RXF1), None, FirstLegal(), chess.BLACK)
    for name, arguments, said in (
        ("analyse", {"moves": ["Kg2", "Qd8"]}, "Qd8 is not a legal move there"),
        ("square", {"square": "z9"}, "is not a square"),
        ("analyse", {"moves": [], "start": "before_last_move"}, "no last move"),
        ("fly", {}, "No tool named 'fly'"),
    ):
        assert said in tools.run(name, arguments), (name, arguments)


def test_before_the_last_move_is_where_the_move_was_played_from() -> None:
    tools = EngineTools(
        chess.Board(AFTER_RXF1), chess.Board(BEFORE_RXF1), FirstLegal(), chess.BLACK
    )
    out = tools.run("analyse", {"moves": ["Rxf1+"], "start": "before_last_move"})
    assert out.startswith("After 3... Rxf1+ (from the position before the last move):")


def test_square_says_what_stands_there_and_who_attacks_it() -> None:
    # After Kg2 Rg1+: the rook on g1 is the player's, defended by the bishop on a7 -- the
    # point of the line the coach had missed ("gets the rook out of attack").
    tools = EngineTools(chess.Board(AFTER_RXF1), None, FirstLegal(), chess.BLACK)
    out = tools.run("square", {"square": "g1", "moves": ["Kg2", "Rg1+"]})
    assert "On g1: the player's rook." in out
    assert "Defended by the player's: bishop on a7." in out
    assert "Attacked by the opponent's: king on g2." in out
    # And whether the side to move can take it: the king cannot, the bishop guards g1.
    assert out.endswith("None of the opponent's pieces can move there.")


class _Asked:
    """An engine that notes the moves each position came with and the depth asked for, and
    answers with one line at that depth."""

    def __init__(self) -> None:
        self.histories: list[list[str]] = []
        self.depths: list[int | None] = []
        self.best_only: list[bool] = []

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        self.histories.append([m.uci() for m in board.move_stack])
        self.depths.append(depth)
        self.best_only.append(best_only)
        move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        return [Line(moves=[move.uci()], depth=depth or 0, expected_score=0.55)]


def test_the_tools_search_with_the_moves_that_led_there() -> None:
    """Stockfish reads a repetition from the moves before a position; the tools dropped
    them, and a move that repeats the position a third time was searched as any other."""
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6"):
        board.push_san(san)
    engine = _Asked()
    tools = EngineTools(board, None, engine, chess.WHITE)
    out = tools.run("analyse", {"moves": ["Ng1"]})
    assert engine.histories[-1][:7] == [m.uci() for m in board.move_stack] + ["f3g1"]
    # Ng8 there is the third time: a draw the game ends on, said as such.
    assert "The game is over there" not in out
    assert "The game is over there: a draw" in tools.run("analyse", {"moves": ["Ng1", "Ng8"]})


def test_a_move_off_the_lines_is_rated_at_the_lines_depth() -> None:
    """The first move given is rated against the best line shown; a move no line plays was
    searched at the tools' own depth (14) and set against a depth-18 line."""
    engine = _Asked()
    known = [Line(moves=["e2e4"], depth=18, expected_score=0.55)]
    tools = EngineTools(chess.Board(), None, engine, chess.WHITE, known_now=known)
    tools.run("analyse", {"moves": ["a3"]})
    assert engine.depths == [14, 18], "the position searched, then the move rated at 18"
    assert engine.best_only == [False, True], "the reply read for its first line alone"


def test_a_first_answer_shown_is_not_set_against_the_full_depth() -> None:
    """The first answer shown (the hint's lines, at EARLY_DEPTH) is not what a move off it is
    rated against: the reply is searched to the full depth, so the best comes from the full
    depth too -- else a move as good as the best reads as worse by what the depths differ by
    (the audit of 2026-10-05, round 2)."""
    engine = _Asked()
    known = [Line(moves=["e2e4"], depth=12, expected_score=0.6)]
    tools = EngineTools(chess.Board(), None, engine, chess.WHITE, known_now=known, depth=18)
    tools.run("analyse", {"moves": ["a3"]})
    assert engine.depths[-2:] == [18, 18], engine.depths


def test_a_move_on_the_first_answers_lines_is_not_read_from_them_against_the_full_depth() -> None:
    """The sheet still on the first answer (the hint's lines at EARLY_DEPTH): a move one of them
    begins is searched, not read from that line, and set against the full depth's best -- read
    from the line, it was "0.54, better than the engine's first line" for a move the full depth
    puts behind (the audit of 2026-10-05, round 3)."""
    from sidecoach.coach.named import analyse_named

    engine = _Asked()
    known = [
        Line(moves=["e2e4"], depth=12, expected_score=0.55),
        Line(moves=["d2d4"], depth=12, expected_score=0.54),
    ]
    out = analyse_named(
        "what if I play 1. d4 instead?",
        chess.Board(),
        engine,
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
        known=known,
    )
    assert out and "better than the engine's first line" not in out[0], out
    assert engine.best_only == [True, True], "the best and the move's reply, each one line"


class ToolModel:
    """A model that takes tools: it makes the calls of each round in `rounds` in turn, then
    answers with what it got."""

    name = "tool-model"
    tools_ok = True

    def __init__(self, rounds: Sequence[Sequence[ToolCall]], answer: str) -> None:
        self.script = [list(calls) for calls in rounds]
        self.answer = answer
        self.seen: list[dict[str, Any]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "tools"

    def chat(self, system: str, messages: Sequence[ChatMessage], **kwargs: Any) -> Completion:
        self.seen.append(kwargs)
        if self.script and kwargs.get("tool_choice") != "none":
            return Completion(text="", tool_calls=self.script.pop(0))
        return Completion(text=self.answer)


def _at_scholar(model: Any) -> Coach:
    """The coach with `model` and the engine, the player to move at SCHOLAR, analysed."""
    coach = Coach(model, policy=_policy(), engine=FirstLegal())
    coach.sync_claims = True  # the request exists when ask() returns
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    return coach


def test_the_model_asks_the_engine_and_answers_from_what_it_returns() -> None:
    model = ToolModel(
        [[ToolCall(id="c1", name="analyse", arguments={"moves": ["Qxf7#"]})]],
        "Qxf7# is mate: the queen takes on f7 and the king has no square.",
    )
    coach = _at_scholar(model)
    model.seen.clear()
    coach.ask("what happens if I take on f7 with the queen?", 8.0)
    out = drain(coach)
    assert out and out[-1].text.startswith("Qxf7# is mate")
    first, second = model.seen[0], model.seen[1]
    assert first["tools"] and not first["turns"]
    # The engine's answer went back to the model as the call's result.
    (turn,) = second["turns"]
    assert [call.name for call in turn.reply.tool_calls] == ["analyse"]
    (result,) = turn.results
    assert "The game is over there: checkmate, White wins." in result
    # The Task says what the tools are for; the call is on the record for the log.
    assert TOOL_WORDS in coach.spoken[-1].calls[-1].prompt
    assert coach.spoken[-1].calls[-1].stats["tools"] == ["analyse(Qxf7#)"]


def test_a_model_that_keeps_asking_answers_after_the_last_round() -> None:
    calls = [ToolCall(id=f"c{i}", name="square", arguments={"square": "f7"}) for i in range(9)]
    model = ToolModel([[c] for c in calls], "The pawn on f7 is defended only by the king.")
    coach = _at_scholar(model)
    model.seen.clear()
    coach.ask("is f7 weak?", 8.0)
    out = drain(coach)
    assert out and out[-1].text == "The pawn on f7 is defended only by the king."
    assert len(model.seen) == TOOL_ROUNDS + 1
    assert model.seen[-1]["tool_choice"] == "none"


def test_the_line_asked_for_again_brings_the_engine_on_its_end() -> None:
    """Asked "ok and after that?" with the line already given, the engine is asked about the
    line's end before the model answers (Luna at no effort gave the same line back, Tai's
    puzzle)."""
    from sidecoach.coach.coach import LINE_GIVEN

    coach = _at_scholar(ToolModel([], "Qxf7# -- it is mate at once."))
    coach.ask("show me the line", 8.0)
    drain(coach)
    assert LINE_GIVEN not in coach.spoken[-1].calls[-1].prompt
    coach.ask("ok and after that?", 9.0)
    drain(coach)
    prompt = coach.spoken[-1].calls[-1].prompt
    assert LINE_GIVEN in prompt
    assert "The engine, asked about the line's end:" in prompt
    assert "The game is over there: checkmate, White wins." in prompt


def test_a_server_that_refuses_the_tools_is_asked_without_them() -> None:
    """Luna with a reasoning effort takes no tools on chat completions: the 400 left the coach
    silent on every question. A refused request is asked again without the tools; a server
    that is down is not asked twice."""

    class Refusing(ToolModel):
        def __init__(self, error: str) -> None:
            super().__init__([], "The pawn on f7 is defended only by the king.")
            self.error = error

        def chat(self, system: str, messages: Sequence[ChatMessage], **kwargs: Any) -> Completion:
            self.seen.append(kwargs)
            if "tools" in kwargs:
                raise RuntimeError(self.error)
            return Completion(text=self.answer)

    for error, answered in (("400: tools not supported", True), ("503: down", False)):
        model = Refusing(error)
        coach = _at_scholar(model)
        model.seen.clear()
        coach.ask("is f7 weak?", 8.0)
        said = [m.text for m in drain(coach)]
        assert ("The pawn on f7 is defended only by the king." in said) is answered, error
        assert any("tools" not in k for k in model.seen) is answered, error
        if answered:
            # The refusal on the record: asked without the tools, the reply had none of them.
            stats = coach.spoken[-1].calls[-1].stats
            assert stats["tools_refused"] == "400: tools not supported"


def test_a_model_without_tools_is_asked_as_before() -> None:
    """A provider that takes no tools (no `tools_ok`) is never sent them, the engine there or
    not: its `chat` takes no `tools`, and the question is answered as before the tools."""
    from tests.fakes import ScriptedProvider

    coach = _at_scholar(ScriptedProvider(["It is a fine square for the knight."]))
    coach.ask("is f7 weak?", 8.0)
    assert drain(coach)[-1].text == "It is a fine square for the knight."


def test_a_question_s_engine_searches_run_off_the_callers_thread() -> None:
    """The engine's searches for a question -- the move named, the line's end -- ran inside
    ask(), on the panel's loop: up to seconds each, and the vision stalled (the deep prompt
    review, 2026-10-01). ask() hands that work to the coach's spawner and returns; here the
    spawner keeps it, and the test runs it after."""
    engine = FirstLegal()
    coach = Coach(ToolModel([], "Bxf7+ gives a pawn for little."), policy=_policy(), engine=engine)
    spawned: list[tuple[Callable[[], None], str]] = []
    coach.spawn = lambda work, name: spawned.append((work, name))
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    drain(coach)
    engine.seen.clear()
    coach.ask("is Bxf7+ good?", 8.0)
    assert engine.seen == [], "no search on the caller's thread"
    assert [name for _, name in spawned] == ["sidecoach-ask"]
    spawned[0][0]()
    assert engine.seen, "the move named was examined by the work handed off"
    assert [m.text for m in drain(coach)] == ["Bxf7+ gives a pawn for little."]


def test_a_call_the_tools_cannot_read_is_told_so_never_analysed_as_now() -> None:
    """Arguments that were no JSON object, or an unknown start, were analysed silently as
    the board now (the deep prompt review, 2026-10-01): the model is told instead."""
    engine = FirstLegal()
    tools = EngineTools(chess.Board(AFTER_RXF1), None, engine, chess.BLACK)
    out = tools.run("analyse", {"_raw": "moves: Kg2"})
    assert out.startswith("The arguments were not a JSON object ('moves: Kg2')"), out
    out = tools.run("analyse", {"moves": ["Kg2"], "start": "before"})
    assert out.startswith("No start named 'before'"), out
    assert engine.seen == [], "nothing was analysed"
