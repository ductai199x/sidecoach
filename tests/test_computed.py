"""What the engine has computed, read before a search (`analysis/computed.py`).

Tai, 2026-09-29: "our harness must first look at what was already computed, then if that's
not computed only then invoke a call to stockfish to ask for analysis over that specific
line" -- the analyser is the producer, the coach's engine one of its readers.
"""

from __future__ import annotations

from typing import Any

import chess

from sidecoach.analysis.annotator import EARLY_DEPTH, GameAnnotator
from sidecoach.analysis.computed import Computed, ComputedFirst, line_after, position_key
from sidecoach.coach.engine_tools import EngineTools
from sidecoach.events import Line
from tests.fakes import FakeEngine

START = chess.Board()
LINES = [
    Line(moves=["e2e4", "e7e5", "g1f3", "b8c6"], depth=18, expected_score=0.55),
    Line(moves=["d2d4", "d7d5", "c2c4"], depth=18, expected_score=0.54),
]


def _after(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


class _Recording:
    """The coach's own engine: every position it searches, and a line of its own for each."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        self.seen.append(board.fen())
        move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        return [Line(moves=[move.uci()], depth=depth or 14, expected_score=0.5)]

    def close(self) -> None:
        return


def test_the_rest_of_a_line_is_read_from_the_side_to_move_there() -> None:
    line = Line(
        moves=["a", "b", "c", "d"], depth=18, expected_score=0.8, score_cp=300, wdl=(700, 200, 100)
    )
    once = line_after(line, 1)
    assert once.moves == ["b", "c", "d"] and once.depth == 17
    assert round(once.expected_score, 6) == 0.2
    assert once.score_cp == -300 and once.wdl == (100, 200, 700)
    twice = line_after(line, 2)
    assert twice.expected_score == 0.8 and twice.score_cp == 300 and twice.wdl == (700, 200, 100)


def test_a_mate_is_counted_in_the_moves_still_to_come() -> None:
    # The mover mates in 2 (A1 B1 A2#): after A1 the side to move is mated in 1; after B1 the
    # mover mates in 1.
    mating = Line(moves=["a1", "b1", "a2"], depth=18, expected_score=1.0, mate=2)
    assert line_after(mating, 1).mate == -1
    assert line_after(mating, 2).mate == 1
    # The mover is mated in 2 (A1 B1 A2 B2#): after A1 the other side mates in 2, after B1
    # the mover is mated in 1, after A2 the other side mates in 1.
    mated = Line(moves=["a1", "b1", "a2", "b2"], depth=18, expected_score=0.0, mate=-2)
    assert [line_after(mated, k).mate for k in (1, 2, 3)] == [2, -1, 1]


def test_a_position_is_the_same_whatever_its_clocks() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    later = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 37 80")
    assert position_key(board) == position_key(later)
    computed = Computed()
    computed.add(board, [Line(moves=["h1h8"], depth=10, expected_score=0.9)])
    assert computed.lines(later)[0].moves == ["h1h8"]


def test_the_positions_own_lines_come_first_then_a_verdicts_then_a_line_through_it() -> None:
    computed = Computed()
    after_e4 = _after("e4")
    computed.add(START, LINES)
    # Through it only: the rest of the first line, one ply in.
    (through,) = computed.lines(after_e4)
    assert through.moves == ["e7e5", "g1f3", "b8c6"] and through.depth == 17
    thin = [Line(moves=["c7c5"], depth=18, expected_score=0.47)]
    computed.add(after_e4, thin, thin=True)
    assert computed.lines(after_e4) == thin
    whole = [
        Line(moves=["e7e5"], depth=12, expected_score=0.46),
        Line(moves=["c7c5"], depth=12, expected_score=0.45),
    ]
    computed.add(after_e4, whole)
    assert computed.lines(after_e4) == whole
    # Nothing reaches a position no line plays.
    assert computed.lines(_after("a3")) == []
    computed.clear()
    assert computed.lines(START) == [] and computed.lines(after_e4) == []


def test_a_shallower_search_leaves_the_deeper_word_standing() -> None:
    computed = Computed()
    computed.add(START, LINES)
    computed.add(START, [Line(moves=["g1f3"], depth=12, expected_score=0.52)])
    assert computed.lines(START) == LINES
    deeper = [Line(moves=["c2c4"], depth=20, expected_score=0.53)]
    computed.add(START, deeper)
    assert computed.lines(START) == deeper


def test_of_two_lines_through_a_position_the_deeper_one_answers() -> None:
    # 1. e4 e5 is reached from the start, two plies in, and from after 1. e4, one ply in.
    computed = Computed()
    computed.add(START, LINES)
    computed.add(
        _after("e4"), [Line(moves=["e7e5", "b1c3", "g8f6"], depth=18, expected_score=0.45)]
    )
    (line,) = computed.lines(_after("e4", "e5"))
    assert line.moves == ["b1c3", "g8f6"] and line.depth == 17


def test_the_coachs_engine_searches_only_what_is_not_computed() -> None:
    computed = Computed()
    computed.add(START, LINES)
    searcher = _Recording()
    engine = ComputedFirst(searcher, computed)
    assert engine.analyse(START, depth=14) == LINES
    assert engine.analyse(_after("e4", "e5"), depth=14)[0].moves == ["g1f3", "b8c6"]
    assert searcher.seen == [] and engine.searches == 0
    # A move no line plays: searched once, and known from then on.
    engine.analyse(_after("a3"), depth=14)
    engine.analyse(_after("a3"), depth=14)
    assert searcher.seen == [_after("a3").fen()] and engine.searches == 1


def test_the_tools_answer_from_the_analysers_lines_and_search_a_move_they_do_not_play() -> None:
    """Tai, 2026-09-29: "it only launches IF the move doesn't exist in the cached line or in
    the line already computed". The analyser measured the board; the model asks the tools."""
    annotator = GameAnnotator(engine=FakeEngine({START.fen(): LINES}), depth=18, with_threats=False)
    annotator.measure(START)
    searcher = _Recording()
    tools = EngineTools(START, None, ComputedFirst(searcher, annotator.computed), chess.WHITE)
    tools.run("analyse", {"moves": []})
    tools.run("analyse", {"moves": ["e4", "e5"]})
    tools.run("analyse", {"moves": ["d4"]})
    assert searcher.seen == [], "a position the lines reach was searched"
    # The rest of the line, as the sheet has it.
    assert "1... e5 2. Nf3 Nc6" in tools.run("analyse", {"moves": ["e4"]})
    # A move no line plays: its position searched once, the start's best read from the lines.
    out = tools.run("analyse", {"moves": ["a3"]})
    assert searcher.seen == [_after("a3").fen()]
    assert "The first move given, 1. a3, is" in out and "behind the engine's best" in out
    tools.run("analyse", {"moves": ["a3"]})
    assert len(searcher.seen) == 1


class _Streaming(FakeEngine):
    """An engine that hands each depth on as it finishes it, and notes what a reader had
    of the position at each."""

    def __init__(self, annotator_of: list[GameAnnotator]) -> None:
        super().__init__({})
        self.annotator_of = annotator_of
        self.read: list[int] = []

    def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
        super().analyse(board, **kwargs)
        computed = self.annotator_of[0].computed
        on_depth = kwargs.get("on_depth")
        for depth in (EARLY_DEPTH - 2, EARLY_DEPTH, EARLY_DEPTH + 2):
            if on_depth is not None:
                on_depth([Line(moves=["e2e4"], depth=depth, expected_score=0.5)])
            known = computed.lines(board)
            self.read.append(known[0].depth if known else 0)
        return [Line(moves=["e2e4"], depth=18, expected_score=0.5)]


def test_the_analyser_puts_in_each_usable_depth_as_it_finishes_it() -> None:
    """A reader gets the newest depth from the first usable one (Tai, 2026-09-28: "if the
    consumers read after stockfish reaches depth > 12 then they will get whatever lines that
    is available at the time"); a verdict's line counts too, and a new game forgets them."""
    holder: list[GameAnnotator] = []
    engine = _Streaming(holder)
    annotator = GameAnnotator(engine=engine, depth=18, with_threats=False)
    holder.append(annotator)
    annotator.measure(START)
    assert engine.read == [0, EARLY_DEPTH, EARLY_DEPTH + 2]
    assert annotator.computed.lines(START)[0].depth == 18
    after_e4 = _after("e4")
    annotator.measure_for_verdict(after_e4)
    assert annotator.computed.lines(after_e4)[0].depth == 18
    annotator.reset()
    assert annotator.computed.lines(START) == [] and annotator.computed.lines(after_e4) == []
