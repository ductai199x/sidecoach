"""The engine service for a game followed live (`analysis/analyser.py`): every search one line,
every position searched once, on a pool of workers; a position told as it comes (its first
answer, then whole, then final); a move judged as soon as what it needs is in, in the order of
the moves; the coach's questions ahead of the game's other positions; nothing a worker does
after a search ends it. Gated fake engines stage every race, a manual clock times the decided
rule; nothing waits on a clock."""

from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import chess
import pytest

from sidecoach.analysis import analyser as analyser_module
from sidecoach.analysis.analyser import (
    BEST,
    DECIDED_AFTER_S,
    HINT,
    Analyser,
    PoolSearcher,
    _Job,
    open_analyser,
)
from sidecoach.analysis.annotator import EARLY_DEPTH, GameAnnotator, cache_key
from sidecoach.analysis.engine import Engine
from sidecoach.events import Analysis, Label, Line, MoveAssessment
from tests.fakes import HANG_S, AnalyserWatch, FakeEngine, GatedEngine, ManualClock
from tests.test_analysis import SCHOLAR, STOCKFISH, needs_engine

START = chess.Board().fen()


@pytest.fixture(scope="module", name="engine")
def _real_engine() -> Iterator[Engine]:
    """The real Stockfish, one thread as every search (`needs_engine`'s tests)."""
    assert STOCKFISH is not None
    with Engine(path=STOCKFISH, threads=1, hash_mb=64, multipv=4) as running:
        yield running


def _line(*moves: str, score: float = 0.5, depth: int = 18) -> Line:
    return Line(moves=list(moves), depth=depth, expected_score=score)


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _after(*sans: str) -> str:
    return _board(*sans).fen()


def _analyser(
    engines: Sequence[FakeEngine], *, threats: bool = False, clock: ManualClock | None = None
) -> Analyser:
    annotator = GameAnnotator(engine=engines[0], depth=18, multipv=6, with_threats=threats)
    if clock is None:
        return Analyser(annotator, engines=engines)
    return Analyser(annotator, engines=engines, clock=clock.monotonic, timer=clock.Timer)


def _searches(engines: Sequence[FakeEngine]) -> list[tuple[str, int, int | None]]:
    """Every search the engines made: the position, the lines asked for and the depth."""
    return [
        (fen, lines, depth)
        for e in engines
        for fen, lines, depth in zip(e.calls, e.multipvs, e.depths, strict=True)
    ]


def _play(analyser: Analyser, sans: Sequence[str], *, game_id: str = "g") -> chess.Board:
    """The moves submitted one after another from the start, as the glue submits them."""
    board = chess.Board()
    for ply, san in enumerate(sans, start=1):
        move = board.parse_san(san)
        analyser.submit_move(game_id, board, move, ply)
        board.push(move)
    return board


def test_every_search_is_one_line_but_the_hint_and_none_is_made_twice() -> None:
    """The engine audit (2026-10-05): several lines to the full depth explode on decided and
    forcing positions, one line does not. A game played, a move taken back and played again:
    every search to the full depth is one line, the hint's alone several at EARLY_DEPTH, and
    no position is searched twice either way."""
    lines = {
        START: [_line("e2e4"), _line("d2d4")],
        _after("e4"): [_line("e7e5"), _line("c7c5")],
        _after("e4", "e5"): [_line("g1f3"), _line("b1c3")],
    }
    engines = [FakeEngine(lines), FakeEngine(lines)]
    analyser = _analyser(engines)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = chess.Board()
        analyser.submit_position("g", board, 0)
        for ply, san in enumerate(["e4", "e5", "Nf3"], start=1):
            move = board.parse_san(san)
            analyser.submit_move("g", board, move, ply)
            board.push(move)
            watch.idle()
        board.pop()
        analyser.withdraw("g", 2)
        analyser.submit_move("g", board, board.parse_san("Nf3"), 3)
        watch.idle()
    searched = _searches(engines)
    assert len(searched) == len(set(searched)), sorted(searched)
    assert {(m, d) for _, m, d in searched} == {(1, 18), (6, EARLY_DEPTH)}
    assert analyser.searches == sum(1 for _, m, _ in searched if m == 1)


def test_a_position_is_whole_from_its_best_line_and_its_candidates_own_searches() -> None:
    """The best line is the position's own search, first -- it picks the best move more often
    than the candidates' searches (231 against 222 of 300 against a depth-24 truth), and the
    engine's own move is never charged a loss for a candidate scoring higher -- then each
    other candidate from its own position's one-line search, turned to the side to move, best
    first."""
    lines = {
        START: [_line("e2e4", "e7e5", score=0.55), _line("d2d4", score=0.5)],
        _after("e4"): [_line("e7e5", score=0.40)],  # higher than the best line says: not taken
        _after("d4"): [_line("d7d5", score=0.48)],
    }
    engine = FakeEngine(lines)
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        told = [e for e in watch.settled() if isinstance(e, Analysis)]
    whole = told[-1]
    assert not whole.provisional
    assert [(line.moves, line.expected_score) for line in whole.lines] == [
        (["e2e4", "e7e5"], pytest.approx(0.55)),
        (["d2d4", "d7d5"], pytest.approx(0.52)),
    ]


def test_a_position_is_told_at_once_and_then_whole() -> None:
    """The first answer comes at once, one search's lines -- the best line while the hint is
    not in, then the hint's at EARLY_DEPTH, never the two set against each other -- each
    provisional; then the position whole; with the side searches, whole once they are in too.
    Each stage once."""
    lines = {START: [_line("e2e4"), _line("d2d4")]}
    for threats in (False, True):
        engine = GatedEngine(lines)  # its lines at the depth asked for
        analyser = _analyser([engine], threats=threats)
        watch = AnalyserWatch(analyser)
        with analyser:
            analyser.submit_position("g", chess.Board(), 0)
            told = [e for e in watch.settled() if isinstance(e, Analysis)]
        assert [line.depth for line in told[0].lines] == [18], "the best line alone, first"
        assert {line.depth for line in told[1].lines} == {EARLY_DEPTH}, "then the hint's"
        assert [e.provisional for e in told] == [True] * (len(told) - 1) + [False]
        assert len(told) == (4 if threats else 3), [e.lines for e in told]
        assert all(line.depth == 18 for line in told[-1].lines)


def test_a_move_not_among_the_lines_waits_for_the_position_after() -> None:
    """A move none of the candidates is judged from the position after's best line -- the
    game's next position, searched anyway -- so its verdict waits for that search."""
    lines = {
        START: [_line("e2e4", score=0.60)],
        _after("a3"): [_line("e7e5", score=0.55)],
    }
    engine = GatedEngine(lines, gated=[_after("a3")])
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.idle()
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("a2a3"), 1)
        assert engine.entered[_after("a3")].wait(HANG_S)
        assert analyser._next_verdict == 1, "judged before the position after was searched"
        engine.release(_after("a3"))
        got = watch.settled()
    verdict = next(e for e in got if isinstance(e, MoveAssessment))
    assert verdict.expected_loss == pytest.approx(0.60 - (1 - 0.55))
    assert verdict.label is not Label.BEST


def test_the_engines_own_move_is_judged_without_waiting_for_the_position_after() -> None:
    """The best line's move is judged from that line, the position's own search: the search
    of the position it leads to -- the game's likeliest next, searched all the same -- is not
    waited for."""
    engine = GatedEngine({START: [_line("e2e4", score=0.6)]}, gated=[_after("e4")])
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.until(lambda: analyser._nodes[cache_key(chess.Board())].whole is not None)
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        # Judged on the move, the position after it still held.
        verdicts = [e for e in analyser.poll() if isinstance(e, MoveAssessment)]
        assert [(v.ply, v.expected_loss, v.label) for v in verdicts] == [(1, 0.0, Label.BEST)]
        engine.release(_after("e4"))
        watch.idle()


def test_a_decided_search_running_long_stands_in_for_the_verdict() -> None:
    """Tai, 2026-10-05: a best-line search still running DECIDED_AFTER_S in whose deepest
    finished depth (EARLY_DEPTH at least) the position is decided has that depth's score judge
    the move, once; the full depth, when it comes, judges nothing again and is what is told.
    Undecided, or too shallow, the move waits for the full depth."""
    for reported, stands in (
        (_line("d8h4", score=0.9, depth=15), True),  # decided: the stand-in judges
        (_line("e7e5", score=0.5, depth=15), False),  # undecided: waited for
        (_line("d8h4", score=0.9, depth=EARLY_DEPTH - 1), False),  # too shallow: waited for
    ):
        clock = ManualClock()
        engine = GatedEngine(
            {START: [_line("e2e4", score=0.6)], _after("h3"): [_line("e7e5", score=0.45)]},
            gated=[_after("h3")],
            reports={_after("h3"): [reported]},
        )
        analyser = _analyser([engine], clock=clock)
        watch = AnalyserWatch(analyser)
        with analyser:
            analyser.submit_position("g", chess.Board(), 0)
            watch.idle()
            analyser.submit_move("g", chess.Board(), chess.Move.from_uci("h2h3"), 1)
            # Held: its depths reported, the rule's timer started before the search began.
            assert engine.entered[_after("h3")].wait(HANG_S)
            clock.advance(DECIDED_AFTER_S)
            judged = [e for e in analyser.poll() if isinstance(e, MoveAssessment)]
            assert len(judged) == (1 if stands else 0), reported
            engine.release(_after("h3"))
            got = watch.settled()
        verdicts = judged + [e for e in got if isinstance(e, MoveAssessment)]
        assert [v.ply for v in verdicts] == [1], "judged once"
        loss = 0.60 - (1 - (0.9 if stands else 0.45))
        assert verdicts[0].expected_loss == pytest.approx(loss)
        told = [e for e in got if isinstance(e, Analysis) and e.ply == 1]
        assert told[-1].lines[0].expected_score == pytest.approx(0.45), "the full depth told"


def test_moves_are_judged_in_the_order_they_were_played() -> None:
    """A verdict reads the move before it ("the opponent just erred", "a recapture"): the
    second move's is held until the first's is out, though its own searches were done first."""
    lines = {START: [_line("e2e4", score=0.55)], _after("e4"): [_line("e7e5", score=0.5)]}
    engines = [GatedEngine(lines, gated=[START]) for _ in range(3)]
    analyser = _analyser(engines)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = _play(analyser, ["e4", "e5"])
        board.pop()
        watch.until(lambda: analyser._nodes[cache_key(board)].whole is not None, "e4 whole")
        assert analyser._next_verdict == 1, "e5 judged before e4"
        for e in engines:
            e.release(START)
        got = watch.settled()
    assert [e.ply for e in got if isinstance(e, MoveAssessment)] == [1, 2]


def test_a_takeback_says_nothing_more_of_the_withdrawn_moves() -> None:
    """Tai, 2026-09-23: "takebacks make the program lag behind by a lot". The moves taken back
    get no verdict and no analysis, and nothing is owed for them; a search made only for them
    stops; the move kept is judged and its positions told."""
    gated = [_after("e4", "e5")]
    engine = GatedEngine({}, gated=gated)
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    # The moves in before the worker starts: the first verdict's needs, then the second's,
    # whose position after e5 holds.
    _play(analyser, ["e4", "e5", "Nf3"])
    with analyser:
        assert engine.entered[gated[0]].wait(HANG_S), "the position after e5 never searched"
        analyser.withdraw("g", 1)
        watch.until(lambda: engine.stopped == gated, "the withdrawn move's search stopped")
        got = watch.settled()
    assert [e for e in got if getattr(e, "ply", 0) >= 2] == [], got
    assert [e.ply for e in got if isinstance(e, MoveAssessment)] == [1]
    assert {e.ply for e in got if isinstance(e, Analysis) and not e.provisional} == {0, 1}
    assert analyser.withdrawn == 2 and analyser.pending == 0


def test_a_position_seen_again_is_not_searched_again() -> None:
    """A puzzle tried again, a board re-read: a position searched before is answered from what
    was found, after the game's reset too."""
    engine = FakeEngine({START: [_line("e2e4")]})
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.quiet()
        analyser.poll()
        searched = len(engine.calls)
        analyser.reset()
        analyser.submit_position("h", chess.Board(), 0)
        watch.quiet()
        again = analyser.poll()
    assert len(engine.calls) == searched
    assert [e.game_id for e in again if isinstance(e, Analysis)] == ["h"]


class _Broken(FakeEngine):
    """An engine whose searches of the positions in `broken` (FENs; every position when none
    is given) fail as a dead engine's do."""

    def __init__(self, broken: Sequence[str] = ()) -> None:
        super().__init__({})
        self.broken = set(broken)

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        if not self.broken or board.fen() in self.broken:
            raise RuntimeError("engine gone")
        return super().analyse(board, **kwargs)


def test_a_failed_search_does_not_end_the_session_nor_hold_the_game() -> None:
    """An engine that fails every search: each failure counted, the moves go unjudged, and
    nothing is owed -- the game is never held waiting on a search that cannot come."""
    analyser = Analyser(GameAnnotator(engine=_Broken(), depth=6, with_threats=False))
    watch = AnalyserWatch(analyser, failures_ok=True)
    with analyser:
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        got = watch.settled()
        failed = analyser.failures
        assert failed == 4, "each position's best line and hint, of both positions"
        assert analyser.last_error is not None and "engine gone" in analyser.last_error
        assert not [e for e in got if isinstance(e, MoveAssessment)]
        analyser.submit_position("h", _board("d4"), 0)
        watch.idle()
    assert analyser.failures == failed + 2, "the next position tried, and failed alike"


def test_a_candidate_that_fails_is_left_out_of_the_whole() -> None:
    """One candidate's own search fails: the position is whole without it, told final, and
    nothing is owed."""
    engine = _Broken([_after("d4")])
    engine.answers = {START: [_line("e2e4", score=0.6), _line("d2d4", score=0.5)]}
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser, failures_ok=True)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        told = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert analyser.failures == 1
    assert not told[-1].provisional and [line.moves[0] for line in told[-1].lines] == ["e2e4"]


def test_side_searches_that_fail_leave_the_position_whole_and_final() -> None:
    """With threats on, the side searches failing: the position is told final with its lines,
    kept for the review, and nothing is owed."""
    engine = FakeEngine({START: [_line("e2e4"), _line("d2d4")]})
    analyser = _analyser([engine], threats=True)

    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("side searches gone")

    analyser.annotator.with_side = broken  # type: ignore[method-assign]
    watch = AnalyserWatch(analyser, failures_ok=True)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        told = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert analyser.failures == 1
    assert not told[-1].provisional and told[-1].lines
    assert len(analyser.measured_path("g")) == 1


def test_a_position_with_one_legal_move_is_never_searched() -> None:
    """In check with one way out: its best line is that move's, from the position it leads
    to; the position itself is never searched, and it is told and judged as any other."""
    one = "k7/p7/8/8/8/8/6q1/7K w - - 0 1"  # the queen gives check; only Kxg2
    engine = FakeEngine({})
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(one), 0)
        told = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert one not in engine.calls
    assert told[-1].lines and told[-1].lines[0].moves[0] == "h1g2" and not told[-1].provisional
    assert analyser.searches == sum(1 for m in engine.multipvs if m == 1)


# Black to move; Qg2+ leaves White one legal move, Kxg2.
QG2 = "k7/p7/8/8/8/6q1/8/7K b - - 0 1"


def test_a_candidate_giving_check_with_one_reply_is_searched_through() -> None:
    """A candidate check with a single reply: its position takes its best line from the
    reply's, two moves down the game -- wanted as the plan wants it, never refused and asked
    again (the audit of 2026-10-05, round 2: a worker spun on it and every verdict after it
    waited for good). For the game's position, a verdict on a move from it, and a question."""
    lines = {QG2: [_line("a7a6", score=0.5), _line("g3g2", score=0.4)]}
    asked = QG2.replace("p7/8/8/8/6q1", "1p6/8/8/8/6q1")  # the same check, the pawn on b7
    lines[asked] = [_line("b7b6", score=0.5), _line("g3g2", score=0.4)]
    engines = [FakeEngine(lines), FakeEngine(lines)]
    analyser = _analyser(engines)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = chess.Board(QG2)
        analyser.submit_position("g", board, 0)
        whole = [e for e in watch.settled() if isinstance(e, Analysis)][-1]
        analyser.submit_move("g", board, chess.Move.from_uci("a7a6"), 1)
        verdicts = [e for e in watch.settled() if isinstance(e, MoveAssessment)]
        answer = analyser.request(chess.Board(asked), timeout=None)
        watch.quiet()
    assert not whole.provisional and "g3g2" in [line.moves[0] for line in whole.lines]
    assert [v.ply for v in verdicts] == [1]
    assert answer is not None and "g3g2" in [line.moves[0] for line in answer.lines]


def test_a_decided_stand_in_reaches_a_move_into_a_forced_reply() -> None:
    """A move into a check with one reply, the position after the reply decided while its
    search runs on: the stand-in is carried back up the forced move, as its best line would
    be, and the move is judged at DECIDED_AFTER_S rather than at the full depth."""
    clock = ManualClock()
    after_reply = _board_from(QG2, "g3g2", "h1g2")
    engine = GatedEngine(
        {QG2: [_line("a7a6", score=0.5)]},  # Qg2+ not among the candidates: judged from after
        gated=[after_reply],
        reports={after_reply: [_line("a6a5", score=0.1, depth=15)]},  # Black lost
    )
    analyser = _analyser([engine], clock=clock)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = chess.Board(QG2)
        analyser.submit_position("g", board, 0)
        watch.idle()
        analyser.submit_move("g", board, chess.Move.from_uci("g3g2"), 1)
        assert engine.entered[after_reply].wait(HANG_S)
        clock.advance(DECIDED_AFTER_S)
        judged = [e for e in analyser.poll() if isinstance(e, MoveAssessment)]
        engine.release(after_reply)
        watch.settled()
    assert [v.ply for v in judged] == [1], "not judged at the stand-in"
    assert judged[0].expected_loss == pytest.approx(0.5 - 0.1)


def _board_from(fen: str, *ucis: str) -> str:
    board = chess.Board(fen)
    for uci in ucis:
        board.push_uci(uci)
    return board.fen()


def test_a_mate_on_the_board_is_judged_told_and_owes_nothing() -> None:
    """Scholar's mate played through, threats on: the mating move judged, the mated position
    told final with no lines, nothing owed."""
    analyser = _analyser([FakeEngine({})], threats=True)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = _play(analyser, ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"])
        got = watch.settled()
    assert board.is_checkmate()
    assert [e.ply for e in got if isinstance(e, MoveAssessment)] == list(range(1, 8))
    mated = [e for e in got if isinstance(e, Analysis) and e.ply == 7]
    assert mated[-1].lines == [] and not mated[-1].provisional


def test_an_illegal_move_is_refused_and_owes_nothing() -> None:
    analyser = _analyser([FakeEngine({})])
    watch = AnalyserWatch(analyser, failures_ok=True)
    with analyser:
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e5"), 1)
        got = watch.settled()
    assert analyser.failures == 1 and got == [] and analyser.pending == 0


def test_the_positions_kept_are_trimmed_and_the_workers_go_on() -> None:
    """Past KEEP_NODES the oldest positions go -- never the game's, its moves', a worker's or
    their candidates -- and a candidate let go is made again when wanted: no worker ever dies
    on a position it no longer finds (the audit of 2026-10-05)."""
    keep = 40
    analyser = _analyser([FakeEngine({}), FakeEngine({})])
    watch = AnalyserWatch(analyser)
    sans = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(analyser_module, "KEEP_NODES", keep)
        with analyser:
            board = chess.Board()
            for ply, san in enumerate(sans, start=1):
                move = board.parse_san(san)
                analyser.submit_move("g", board, move, ply)
                board.push(move)
                watch.idle()
                assert len(analyser._nodes) < 2 * keep
            board.pop()
            analyser.withdraw("g", len(sans) - 1)  # a takeback, and a move played again
            analyser.submit_move("g", board, board.parse_san("Bb6"), len(sans))
            got = watch.settled()
            assert all(t.is_alive() for t in analyser._threads)
    assert [e.ply for e in got if isinstance(e, MoveAssessment)][-1] == len(sans)
    assert analyser.pending == 0


def test_the_workers_take_the_verdicts_then_the_coach_then_the_game() -> None:
    """Who waits on a search, first to last: the next verdict (the word on the move just
    played), the coach's question (the player waiting on it), the position the game is on,
    the game's other positions."""
    analyser = _analyser([FakeEngine({START: [_line("e2e4")]})])
    asked = _board("d4")
    asking = threading.Thread(target=analyser.request, args=(asked,), daemon=True)
    try:
        analyser.submit_position("g", chess.Board(), 0)
        asking.start()
        with analyser._cond:
            while not analyser._requests:
                assert analyser._cond.wait(HANG_S), "the coach never asked"
            assert analyser._pick() == _Job(BEST, cache_key(asked), 0), "the coach's first"
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        with analyser._cond:
            assert analyser._pick() == _Job(BEST, cache_key(chess.Board()), 0)
            jobs = analyser._agenda()
        coach_job, game_job = (
            _Job(BEST, cache_key(asked), 0),
            _Job(HINT, cache_key(_board("e4")), 0),
        )
        assert jobs.index(coach_job) < jobs.index(game_job), "the coach's after the game's"
    finally:
        analyser.close()
        asking.join(HANG_S)
    assert not asking.is_alive()


def test_the_coach_is_answered_from_the_pool_and_never_searches_twice() -> None:
    """The coach's engine (`PoolSearcher`): a question on a position gets it whole, from the
    analyser's workers -- no Stockfish of its own -- and asked again, at once."""
    lines = {
        _after("d4"): [_line("d7d5", score=0.5), _line("g8f6", score=0.45)],
    }
    engine = FakeEngine(lines)
    analyser = _analyser([engine])
    searcher = PoolSearcher(analyser)
    with analyser:
        first = searcher.analyse(_board("d4"))
        searched = len(engine.calls)
        again = searcher.analyse(_board("d4"), depth=12)
    assert first == again and [line.moves[0] for line in first] == ["d7d5", "g8f6"]
    assert len(engine.calls) == searched
    assert analyser._requests == []


def test_the_coach_asking_for_one_line_does_not_wait_for_the_whole() -> None:
    """`best_only`: a probe that reads the first line alone gets the position's best line
    while its candidates are still held -- the whole is not waited for."""
    lines = {_after("d4"): [_line("d7d5", score=0.5), _line("g8f6", score=0.45)]}
    held = _after("d4", "g8f6")
    engine = GatedEngine(lines, gated=[held])
    analyser = _analyser([engine])
    try:
        analyser.start()
        got = PoolSearcher(analyser, timeout=None).analyse(_board("d4"), best_only=True)
        assert [line.moves[0] for line in got] == ["d7d5"]
        with analyser._cond:
            assert analyser._nodes[cache_key(_board("d4"))].whole is None, "the whole waited for"
    finally:
        engine.release(held)
        analyser.close()


def test_a_question_is_never_left_hanging() -> None:
    """The coach's wait is bounded however the search goes: past its time it gets what is
    known (here nothing) and its question is dropped; a question whose search fails is
    answered at once; one waiting when the analyser closes is let go; one asked of a closed
    analyser is answered at once."""
    gated = _after("c4")
    engine = GatedEngine({}, gated=[gated])
    analyser = _analyser([engine])
    analyser.start()
    try:
        assert analyser.request(_board("c4"), timeout=0) is None
        assert analyser._requests == []
        answers: list[object] = []
        waiting = threading.Thread(
            target=lambda: answers.append(analyser.request(_board("c4"))), daemon=True
        )
        waiting.start()
        with analyser._cond:
            while not analyser._requests:
                assert analyser._cond.wait(HANG_S), "the coach never asked"
    finally:
        engine.release(gated)
        analyser.close()
    waiting.join(HANG_S)
    assert not waiting.is_alive() and len(answers) == 1, "let go at the close"
    assert analyser.request(_board("e4"), timeout=None) is None

    broken = Analyser(GameAnnotator(engine=_Broken(), depth=6, with_threats=False))
    with broken:
        assert broken.request(_board("d4"), timeout=None) is None
    assert broken.failures == 2


def test_the_workers_engines_search_on_one_thread_each() -> None:
    """At a fixed depth more threads widen the search instead of speeding it: four took 7.7 times
    the processor time of one, and twice as long, for an answer no better (the engine audit,
    2026-10-04). Each worker's engine has one, and the pool its size."""
    made: list[dict[str, object]] = []

    def make(**kwargs: object) -> Engine:
        made.append(kwargs)
        return FakeEngine({})

    from sidecoach.analysis.score import Thresholds

    analyser = open_analyser(
        Path("stockfish"),
        depth=18,
        multipv=6,
        workers=4,
        hash_mb=64,
        pv_plies=12,
        thresholds=Thresholds(),
        make_engine=make,
    )
    analyser.close()
    assert [m["threads"] for m in made] == [1, 1, 1, 1]
    assert len(analyser.engines) == 4


def test_settings_saved_mid_session_reach_every_workers_engine() -> None:
    """Hash, the lines' length and the scoring saved: on each worker's engine, hash through
    the engine's own `reconfigure` (set between searches), the rest set (the audit of
    2026-10-05, round 2: the route ran in no test)."""
    engines = [FakeEngine({}), FakeEngine({})]
    analyser = _analyser(engines)
    hashed: list[int | None] = []
    for engine in engines:
        engine.reconfigure = lambda *, threads=None, hash_mb=None: hashed.append(hash_mb)  # type: ignore[method-assign]
    analyser.reconfigure(hash_mb=128, multipv=3, pv_plies=8)
    assert hashed == [128, 128]
    assert [(e.multipv, e.pv_plies) for e in engines] == [(3, 8), (3, 8)]


def test_settings_changed_mid_search_keep_nothing_measured_the_old_way() -> None:
    """The depth or the lines saved while a search runs: what it finds is not kept -- even a
    search that runs on to its end -- and the game's position is measured again and told
    from the new search."""

    class Stubborn(GatedEngine):
        """A search no stop reaches: it runs to its end, the old way -- scoring as the settings
        were when it began, so what it found is told apart from the new search's."""

        def interrupt(self) -> None:
            self._wake.set()

        def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
            began = [line.expected_score for line in self.answers.get(board.fen(), [])]
            lines = super().analyse(board, **kwargs)
            if not began:
                return lines
            return [line.model_copy(update={"expected_score": began[0]}) for line in lines]

    engine = Stubborn({START: [_line("e2e4", score=0.3)]}, gated=[START])
    analyser = _analyser([engine])
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        assert engine.entered[START].wait(HANG_S)
        analyser.forget_measured()
        engine.answers[START] = [_line("e2e4", score=0.7)]  # the new settings' answer
        engine.release(START)
        told = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert sum(1 for fen, m, _ in _searches([engine]) if fen == START and m == 1) == 2
    assert told and not told[-1].provisional
    assert told[-1].lines[0].expected_score == pytest.approx(0.7), "the old search kept"
    assert analyser.searches == sum(1 for m in engine.multipvs if m == 1) - 1


def test_a_watched_games_review_searches_nothing_the_game_measured(tmp_path: Path) -> None:
    """A game followed live had every position measured whole -- its candidates and its side
    searches, at the review's standard -- and its review measured them all again (the engine
    audit, 2026-10-04). Kept as the game ends, they are the review's: it searches none of them."""
    import io

    import chess.pgn

    from sidecoach.analysis.game_review import measured_from_json, review_game
    from sidecoach.analysis.score import Thresholds
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Side
    from sidecoach.profile.store import Store

    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    analyser = Analyser(GameAnnotator(engine=FakeEngine({}), depth=18, multipv=6))
    assessing.analyser = analyser
    watch = AnalyserWatch(analyser)
    analyser.start()
    assessing.submit(GameStarted(game_id="g", start_fen=START, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3"], start=1):
        before = board.fen()
        move = board.push_san(san)
        assessing.submit(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )
    watch.idle()
    store = Store(tmp_path / "profile.sqlite")
    try:
        assert assessing.keep_for_review(store, "g") == 4, "the four positions, whole"
        known = {
            m.fen: m
            for m in map(measured_from_json, store.progress("watched:g", "standard", 18, 6))
        }
    finally:
        store.close()
        analyser.close()
    review_engine = FakeEngine({})
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 *"))
    assert game is not None
    reviewed = review_game(
        game, [review_engine], depth=18, multipv=6, thresholds=Thresholds(), known=known
    )
    assert len(reviewed.moves) == 3
    assert review_engine.calls == [], "the review searched what the game had measured"
    fresh = review_game(game, [FakeEngine({})], depth=18, multipv=6, thresholds=Thresholds())

    def verdicts(review: object) -> list[tuple[str, object, float]]:
        moves = review.moves  # type: ignore[attr-defined]
        return [(m.assessment.uci, m.assessment.label, m.assessment.expected_loss) for m in moves]

    assert verdicts(reviewed) == verdicts(fresh), "the kept positions are the review's own"


def test_a_live_game_stays_in_the_book_past_its_first_move(tmp_path: Path) -> None:
    """The live path hands the analyser each move's board; the book needs the moves that led
    to it. An Italian Game submitted the way the panel submits it -- a takeback in the middle --
    is book to its last move and named as the whole line names it, not as its first move does."""
    import shutil

    from sidecoach.analysis.openings import OpeningBook
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Side, Takeback

    folder = tmp_path / "openings"
    folder.mkdir()
    shutil.copy(Path(__file__).parent / "openings_sample.tsv", folder / "c.tsv")
    book = OpeningBook.load(folder)
    assert book is not None
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    assessing.analyser = Analyser(GameAnnotator(engine=FakeEngine({}), book=book))
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()

    def play(board: chess.Board, san: str, ply: int) -> None:
        before = board.fen()
        move = board.push_san(san)
        assessing.submit(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )

    assessing.submit(GameStarted(game_id="g", start_fen=chess.STARTING_FEN, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d4"], start=1):
        play(board, san, ply)
    board.pop()  # a takeback of 3.d4: the next move is played from the position before it
    assessing.submit(Takeback(game_id="g", plies=1, fen=board.fen()))
    play(board, "d3", 7)
    labels: dict[int, object] = {}
    openings: dict[int, object] = {}
    for event in watch.settled():
        if isinstance(event, MoveAssessment):
            labels[event.ply] = event.label
        elif isinstance(event, Analysis):
            openings[event.ply] = event.opening
    assessing.analyser.close()
    assert [labels[p] for p in range(1, 8)] == [Label.BOOK] * 7, labels
    assert openings[1] == "B00 King's Pawn Game"
    assert openings[5] == "C50 Italian Game"
    assert openings[7] == "C50 Italian Game: Giuoco Pianissimo", openings


def test_what_the_engine_says_of_a_withdrawn_move_is_not_passed_on(tmp_path: Path) -> None:
    """A verdict on a move taken back came back at its ply and was read as the verdict on the
    move that replaced it; `_Assessing.drain` holds each result against the game as it stands."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Severity, Side, Takeback

    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    assessing.analyser = Analyser(GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False))

    def played(board: chess.Board, san: str, ply: int) -> MovePlayed:
        before = board.fen()
        move = board.push_san(san)
        return MovePlayed(
            game_id="g",
            ply=ply,
            san=san,
            uci=move.uci(),
            fen_before=before,
            fen_after=board.fen(),
            by_user=ply % 2 == 1,
        )

    assessing.submit(GameStarted(game_id="g", start_fen=chess.STARTING_FEN, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1):
        assessing.submit(played(board, san, ply))
    # Nothing searched yet (the workers are not started): 3.Nf3 Nc6 taken back, 3.Bc4 played.
    board.pop()
    board.pop()
    assessing.submit(Takeback(game_id="g", plies=2, fen=board.fen()))
    assessing.submit(played(board, "Bc4", 3))
    # And a verdict on the withdrawn 3.Nf3, as if the engine had finished it just before.
    stale = MoveAssessment(
        game_id="g", ply=3, uci="g1f3", expected_loss=0.0, severity=Severity.NONE, best=[]
    )
    assessing.analyser._results.put(stale)
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()
    watch.idle()
    got: list[object] = list(assessing.drain())
    assessing.analyser.close()
    verdicts = [e for e in got if isinstance(e, MoveAssessment)]
    assert [(e.ply, e.uci) for e in verdicts] == [(1, "e2e4"), (2, "e7e5"), (3, "f1c4")], verdicts
    assert all(e.ply <= 3 for e in got if isinstance(e, Analysis))


def test_a_position_re_read_mid_search_gets_no_word_on_the_old_reading(tmp_path: Path) -> None:
    """The side to move settled by the clock while the joined position is searched: the reset
    lets go of the search under way, and the glue passes on nothing of a position the game no
    longer stands on, nor of another game (the state review, 2026-10-01). The white reading's
    search is held until something stops it: the test acts mid-search by the engine's state."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, Resync, Side

    white = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    black = white.replace(" w ", " b ")
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    engine = GatedEngine({}, gated=[white])
    analyser = Analyser(GameAnnotator(engine=engine, depth=18, with_threats=False))
    assessing.analyser = analyser
    watch = AnalyserWatch(analyser)
    analyser.start()
    try:
        assessing.submit(
            GameStarted(game_id="g", user_side=Side.WHITE, start_fen=white, kind="position")
        )
        assert engine.entered[white].wait(HANG_S), "the search never began"
        assessing.submit(Resync(game_id="g", fen=black, reason="side to move settled by the clock"))
        watch.until(lambda: engine.stopped == [white], "the reset let go of the search")
        watch.idle()
        assert engine.calls.count(white) == 0, "the old reading searched to an answer"
        told = [e.fen for e in assessing.drain() if isinstance(e, Analysis)]
        assert told and set(told) == {black}, told
        word = {"ply": 0, "lines": [], "decisive": False, "only_move": False}
        analyser._results.put(Analysis(game_id="g", fen=white, **word))
        analyser._results.put(Analysis(game_id="h", fen=black, **word))
        assert assessing.drain() == []
    finally:
        assessing.close()


@needs_engine
def test_a_real_position_is_told_at_once_and_then_whole(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real engine: the best line to the full depth first, the hint's moves at
    EARLY_DEPTH beside it, then the position whole -- every line to the full depth, best
    first, the hint's best move among them."""
    monkeypatch.setattr(engine, "close", lambda: None)  # the module's Stockfish stays up
    board = chess.Board(SCHOLAR)
    annotator = GameAnnotator(engine=engine, depth=16, multipv=3, with_threats=False)
    analyser = Analyser(annotator)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", board, 0)
        got = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert [e.provisional for e in got] == [True, True, False]
    assert [line.depth for line in got[0].lines] == [16], "the best line, first"
    assert {line.depth for line in got[1].lines} == {EARLY_DEPTH}, "then the hint's lines"
    whole = got[-1].lines
    scores = [line.expected_score for line in whole]
    assert scores == sorted(scores, reverse=True) and len(whole) <= 3
    assert whole[0].moves[0] in {line.moves[0] for line in got[1].lines}
