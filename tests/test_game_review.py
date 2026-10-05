"""A game reviewed whole (analysis/game_review.py): several engines measure its positions side
by side, one walk judges the moves in order, and the result is the same as one engine judging
the game move by move."""

from __future__ import annotations

import io
import threading

import chess
import chess.pgn

from sidecoach.analysis.annotator import GameAnnotator, Measured
from sidecoach.analysis.game_review import (
    measured_from_json,
    measured_json,
    positions,
    review_game,
    workers_for,
)
from sidecoach.analysis.score import Thresholds
from sidecoach.events import Line, ReviewedGame, Side
from tests.fakes import FakeEngine, scripted

PGN = """[Event "Live Chess"]
[TimeControl "600+5"]
[Result "1-0"]

1. e4 {[%clk 0:10:00]} e5 {[%clk 0:09:58]} 2. Nf3 {[%clk 0:09:57]} Nc6 {[%clk 0:09:40]}
3. Bb5 {[%clk 0:09:50]} a6 {[%clk 0:09:31]} 1-0
"""


def _game(text: str = PGN) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(text))
    assert game is not None
    return game


def _answers(game: chess.pgn.Game) -> dict[str, list[Line]]:
    """A different best line in each position, so the verdicts differ move to move."""
    out: dict[str, list[Line]] = {}
    for i, board in enumerate(positions(game)):
        legal = sorted(m.uci() for m in board.legal_moves)
        if legal:
            lines = [
                Line(moves=[legal[0]], depth=18, expected_score=0.5 + 0.01 * (i % 3)),
                Line(moves=[legal[-1]], depth=18, expected_score=0.45),
            ]
            out |= scripted(board, lines)
    return out


def test_side_by_side_the_verdicts_are_one_engines() -> None:
    game = _game()
    answers = _answers(game)
    engines = [FakeEngine(answers) for _ in range(3)]
    reviewed = review_game(game, engines, depth=18, multipv=2, thresholds=Thresholds())

    # The same game judged move by move by one engine, as the live analyser does.
    alone = GameAnnotator(engine=FakeEngine(answers), depth=18, multipv=2, thresholds=Thresholds())
    board = game.board()
    expected = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        assessment, _ = alone.assess("review", board, move, ply)
        expected.append((assessment.uci, assessment.label, round(assessment.expected_loss, 6)))
        board.push(move)

    got = [
        (m.assessment.uci, m.assessment.label, round(m.assessment.expected_loss, 6))
        for m in reviewed.moves
    ]
    assert got == expected
    assert [m.san for m in reviewed.moves] == ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
    assert [m.side for m in reviewed.moves][:2] == [Side.WHITE, Side.BLACK]
    assert reviewed.start.fen == chess.STARTING_FEN


def test_the_walk_searches_nothing_once_the_positions_are_measured() -> None:
    game = _game()
    engines = [FakeEngine(_answers(game)) for _ in range(3)]
    at_the_end: list[list[int]] = []

    def progress(done: int, total: int) -> None:
        if done == total:
            at_the_end.append([len(e.calls) for e in engines])

    review_game(game, engines, depth=18, multipv=2, thresholds=Thresholds(), on_progress=progress)
    # Every position measured, by whichever engine took it; then the walk searched nothing.
    assert at_the_end and [len(e.calls) for e in engines] == at_the_end[-1]
    assert {b.fen() for b in positions(game)} <= {fen for e in engines for fen in e.calls}


def test_a_review_stopped_part_way_goes_on_from_where_it_was() -> None:
    """The app closed in the middle of a review: the positions measured so far were kept (in
    their saved form), and the review opened again measures only the rest, counts from where
    it was, and comes out as a review never stopped."""
    game = _game()
    answers = _answers(game)
    kept: dict[str, Measured] = {}
    stop = threading.Event()

    def keep(m: Measured) -> None:
        kept[m.fen] = measured_from_json(measured_json(m))

    def progress(done: int, _total: int) -> None:
        if done >= 3:
            stop.set()

    review_game(
        game,
        [FakeEngine(answers)],
        depth=18,
        multipv=2,
        thresholds=Thresholds(),
        on_progress=progress,
        stop=stop,
        on_measured=keep,
    )
    total = len(positions(game))
    assert 3 <= len(kept) < total
    engine = FakeEngine(answers)
    counted: list[int] = []
    resumed = review_game(
        game,
        [engine],
        depth=18,
        multipv=2,
        thresholds=Thresholds(),
        known=kept,
        on_progress=lambda done, _total: counted.append(done),
    )
    assert counted[0] == len(kept) and counted[-1] == total, "counted on from where it was"
    assert not set(engine.calls) & set(kept), "nothing measured twice"
    whole = review_game(game, [FakeEngine(answers)], depth=18, multipv=2, thresholds=Thresholds())

    def verdicts(r: ReviewedGame) -> list[tuple[str, str | None, float]]:
        return [(m.assessment.uci, m.assessment.label, m.assessment.expected_loss) for m in r.moves]

    assert verdicts(resumed) == verdicts(whole)


def test_an_engine_that_fails_leaves_its_share_to_the_others() -> None:
    """Side by side, one engine dies on its first search -- its best line another engine may
    be waiting on: the others finish the game, nothing waits for good, and the review's
    verdicts are a single engine's."""
    game = _game()
    answers = _answers(game)

    class Dead(FakeEngine):
        def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
            raise RuntimeError("engine gone")

    reviewed = review_game(
        game,
        [Dead(answers), FakeEngine(answers), FakeEngine(answers)],
        depth=18,
        multipv=2,
        thresholds=Thresholds(),
    )
    alone = review_game(game, [FakeEngine(answers)], depth=18, multipv=2, thresholds=Thresholds())

    def verdicts(review: ReviewedGame) -> list[tuple[str, object, float]]:
        return [
            (m.assessment.uci, m.assessment.label, m.assessment.expected_loss) for m in review.moves
        ]

    assert verdicts(reviewed) == verdicts(alone)


def test_a_review_stopped_before_it_measured_anything_searches_nothing() -> None:
    """Stopped before the engines took a position: the walk searches nothing more -- a review
    the room throws away -- where it used to measure the start position whole."""
    engine = FakeEngine({})
    stop = threading.Event()
    stop.set()
    reviewed = review_game(
        _game(), [engine], depth=18, multipv=2, thresholds=Thresholds(), stop=stop
    )
    assert engine.calls == [] and reviewed.moves == []


def test_the_clock_and_the_time_each_move_took() -> None:
    reviewed = review_game(_game(), [FakeEngine()], depth=18, multipv=2, thresholds=Thresholds())
    clocks = [(m.clock_ms, m.spent_ms) for m in reviewed.moves]
    # 600+5: White 10:00 -> 10:00 after e4 took 5 s back (0 s thought); Black 10:00 -> 9:58
    # plus 5 s: 7 s; White 10:00 -> 9:57: 8 s; Black 9:58 -> 9:40: 23 s.
    assert clocks[:4] == [(600_000, 5_000), (598_000, 7_000), (597_000, 8_000), (580_000, 23_000)]
    # A correspondence game's clock is days: no time spent.
    daily = PGN.replace('[TimeControl "600+5"]', '[TimeControl "1/86400"]')
    reviewed = review_game(
        _game(daily), [FakeEngine()], depth=18, multipv=2, thresholds=Thresholds()
    )
    assert all(m.spent_ms is None for m in reviewed.moves)


def test_engines_side_by_side_leave_the_machine_a_third() -> None:
    """One thread an engine (the engine audit, 2026-10-04): two thirds of the cores, at most
    eight engines."""
    assert workers_for(24) == 8
    assert workers_for(8) == 5
    assert workers_for(2) == 1
