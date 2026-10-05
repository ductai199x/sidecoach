"""What the engine has computed, read before a search (`analysis/computed.py`).

Tai, 2026-09-29: "our harness must first look at what was already computed, then if that's
not computed only then invoke a call to stockfish to ask for analysis over that specific
line" -- the analyser is the producer, the coach's engine one of its readers, and a move on a
line the model was shown is read from that line.
"""

from __future__ import annotations

import chess

from sidecoach.analysis.annotator import GameAnnotator
from sidecoach.analysis.computed import (
    Computed,
    ComputedFirst,
    line_after,
    position_key,
    rest_of_line,
)
from sidecoach.coach.engine_tools import EngineTools
from sidecoach.events import Line
from tests.fakes import FakeEngine, GatedEngine

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
    # A line carried on past its mate (a slower one found from its end) has no count left.
    carried = Line(moves=["a1", "b1", "a2", "b2", "a3"], depth=18, expected_score=1.0, mate=1)
    assert line_after(carried, 2).mate is None


def test_a_move_on_a_line_is_read_from_it() -> None:
    (rest,) = [rest_of_line(LINES, ["e2e4", "e7e5"])]
    assert rest is not None and rest.moves == ["g1f3", "b8c6"] and rest.expected_score == 0.55
    assert rest_of_line(LINES, ["d2d4"]) == line_after(LINES[1], 1)
    # Not on a line, a whole line (nothing after it), or nothing asked: none.
    assert rest_of_line(LINES, ["a2a3"]) is None
    assert rest_of_line(LINES, ["d2d4", "d7d5", "c2c4"]) is None
    assert rest_of_line(LINES, []) is None


def test_a_position_is_the_same_whatever_its_clocks() -> None:
    board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
    later = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 37 80")
    assert position_key(board) == position_key(later)
    computed = Computed()
    computed.add(board, [Line(moves=["h1h8"], depth=10, expected_score=0.9)])
    assert computed.lines(later)[0].moves == ["h1h8"]


def test_a_position_measured_whole_comes_before_its_verdict_and_nothing_else_is_known() -> None:
    computed = Computed()
    after_e4 = _after("e4")
    computed.add(START, LINES)
    # A line passing through a position is no measure of it: a move refuted deeper still
    # read off its line at depth 12 (the code review, 2026-09-29).
    assert computed.lines(after_e4) == []
    thin = [Line(moves=["c7c5"], depth=18, expected_score=0.47)]
    computed.add(after_e4, thin, thin=True)
    assert computed.lines(after_e4) == thin
    whole = [
        Line(moves=["e7e5"], depth=12, expected_score=0.46),
        Line(moves=["c7c5"], depth=12, expected_score=0.45),
    ]
    computed.add(after_e4, whole)
    assert computed.lines(after_e4) == whole
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


def test_the_coachs_engine_searches_only_a_position_not_computed_and_never_once_closed() -> None:
    computed = Computed()
    computed.add(START, LINES)
    searcher = _Recording()
    engine = ComputedFirst(searcher, computed)
    assert engine.analyse(START, depth=14) == LINES
    assert searcher.seen == [] and engine.searches == 0
    # Not computed: searched once, and known from then on.
    engine.analyse(_after("a3"), depth=14)
    engine.analyse(_after("a3"), depth=14)
    assert searcher.seen == [_after("a3").fen()] and engine.searches == 1
    # Closed with a question still being answered: what is known, and no new Stockfish.
    engine.close()
    assert engine.analyse(_after("h3"), depth=14) == [] and engine.searches == 1
    assert engine.analyse(START, depth=14) == LINES


def test_the_tools_read_the_sheets_lines_and_search_only_a_position_nothing_has() -> None:
    """Tai, 2026-09-29: "it only launches IF the move doesn't exist in the cached line or in
    the line already computed". The analyser measured the board, the sheet shows its lines,
    and the model asks the tools."""
    annotator = GameAnnotator(engine=FakeEngine({START.fen(): LINES}), depth=18, with_threats=False)
    annotator.measure(START)
    searcher = _Recording()
    engine = ComputedFirst(searcher, annotator.computed)
    tools = EngineTools(START, None, engine, chess.WHITE, known_now=LINES)
    assert "The engine's lines, best first:" in tools.run("analyse", {"moves": []})
    on_line = tools.run("analyse", {"moves": ["e4"]})
    assert "The rest of the engine's line through it, as given above:" in on_line
    assert "1... e5 2. Nf3 Nc6" in on_line
    assert "The first move given, 1. e4, is the engine's best move" in on_line
    # A move the second line plays: rated by that line's score, as the sheet shows it.
    assert "0.01 behind the engine's best from the board now, 1. e4." in tools.run(
        "analyse", {"moves": ["d4"]}
    )
    assert searcher.seen == [], "a move on the sheet's lines was searched"
    # A move no line plays: its position searched once.
    out = tools.run("analyse", {"moves": ["a3"]})
    assert searcher.seen == [_after("a3").fen()]
    assert "The first move given, 1. a3, is" in out and "behind the engine's best" in out
    tools.run("analyse", {"moves": ["a3"]})
    assert len(searcher.seen) == 1


def test_the_sheets_line_wins_over_a_separate_search_of_the_position_after_it() -> None:
    """The look-ahead measures the position after each of the two best moves on its own, and
    its search can answer the move differently from the sheet's line through it: the model,
    shown both, believed each in turn (Bd2 answered with Kh6 where the lines had Rxh3, the dev
    audit, 2026-09-26). A move on the sheet's line is read from the line."""
    computed = Computed()
    computed.add(START, LINES)
    computed.add(_after("e4"), [Line(moves=["c7c5", "g1f3"], depth=18, expected_score=0.44)])
    tools = EngineTools(
        START, None, ComputedFirst(_Recording(), computed), chess.WHITE, known_now=LINES
    )
    out = tools.run("analyse", {"moves": ["e4"]})
    assert "1... e5 2. Nf3 Nc6" in out and "c5" not in out.split("On the board there")[0]


def test_a_reader_gets_the_best_line_and_then_the_position_whole() -> None:
    """A reader gets what is there from the first usable answer on (Tai, 2026-09-28: "if the
    consumers read after stockfish reaches depth > 12 then they will get whatever lines that
    is available at the time"): the best line at the full depth as its search ends, then the
    position whole -- never the hint's lines at EARLY_DEPTH, which a reader would set against
    the full depth's (the audit of 2026-10-05); a verdict's line counts too, and a new game
    forgets them."""
    engine = GatedEngine({})  # its lines at the depth asked for
    annotator = GameAnnotator(engine=engine, depth=18, with_threats=False)
    best = annotator.best_line(START)
    assert annotator.computed.lines(START) == best.lines
    assert annotator.computed.whole(START) == []
    annotator.hint_lines(START)
    assert annotator.computed.lines(START) == best.lines, "the hint kept for a reader"
    annotator.measure(START)
    whole = annotator.computed.whole(START)
    assert whole and {line.depth for line in whole} == {18}
    after_e4 = _after("e4")
    annotator.measure_for_verdict(after_e4)
    assert annotator.computed.lines(after_e4)[0].depth == 18
    annotator.reset()
    assert annotator.computed.lines(START) == [] and annotator.computed.lines(after_e4) == []


def test_the_coachs_engine_asks_for_a_whole_it_lacks_and_files_only_what_it_found_itself() -> None:
    """The coach's engine (`ComputedFirst`) answers from a position measured whole; with only
    a best line known it asks its searcher for the whole. What a searcher that keeps its
    answers itself returns (the analyser's pool: a whole, or a best line when its wait ran out)
    is not filed again as the position whole."""
    best = [Line(moves=["e2e4"], depth=18, expected_score=0.55)]
    whole = [*best, Line(moves=["d2d4"], depth=18, expected_score=0.5)]

    class Pool:
        fills_computed = True

        def __init__(self) -> None:
            self.asked = 0

        def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
            self.asked += 1
            return list(best)  # its wait ran out: the best line alone

        def close(self) -> None:
            return

    computed = Computed()
    computed.add(START, best, thin=True)
    pool = Pool()
    engine = ComputedFirst(pool, computed)
    assert engine.analyse(START) == best and pool.asked == 1, "the whole asked for"
    assert computed.whole(START) == [], "a best line filed as the position whole"
    computed.add(START, whole)
    assert engine.analyse(START) == whole and pool.asked == 1, "the whole known: no asking"
    engine.close()
    assert engine.analyse(_after("d4")) == [], "closed: nothing searched"
