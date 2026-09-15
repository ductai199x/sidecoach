"""The engine wrapper, the expected-score arithmetic, and the annotator that joins them."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import chess
import pytest

from sidecoach.analysis.annotator import Analyser, GameAnnotator, _mate_in
from sidecoach.analysis.engine import Engine, san_line
from sidecoach.analysis.score import (
    Thresholds,
    best_gap,
    engine_expected_score,
    eval_loss,
    expected_score,
    human_expected_score,
    is_decisive,
    is_only_move,
    position_score,
    severity,
    terminal_expected_score,
)
from sidecoach.analysis.threats import threat_moves
from sidecoach.config import find_stockfish, load_settings
from sidecoach.events import Analysis, Line, MotifTag, MoveAssessment, Severity

STOCKFISH = find_stockfish(load_settings())
needs_engine = pytest.mark.skipif(STOCKFISH is None, reason="no Stockfish binary")
# Deep enough to be right about the positions below, shallow enough to keep the suite quick.
DEPTH = 10

# 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, where 4. Qxf7# is mate.
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# Black to move at queen odds: winning, but with no mate in sight to hide the numbers.
BLACK_WINS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
# White to move and not in check, but ...Rd1 would be mate: a threat, not a position.
THREATENED = "3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"


@pytest.fixture(scope="module")
def engine():
    assert STOCKFISH is not None
    with Engine(path=STOCKFISH, threads=2, hash_mb=64, multipv=4) as running:
        yield running


class FakeEngine(Engine):
    """An engine that answers from a script, so the annotator can be tested without one."""

    def __init__(self, answers: dict[str, list[Line]]) -> None:
        super().__init__(path=Path("fake"))
        self.answers = answers
        self.calls: list[str] = []

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        self.calls.append(board.fen())
        if board.is_game_over(claim_draw=True):
            return []
        return self.answers.get(board.fen(), [Line(moves=["a2a3"], depth=1, expected_score=0.5)])

    def open(self) -> None:  # never starts a process
        return

    def close(self) -> None:
        return


class RecordingTagger:
    def __init__(self) -> None:
        self.move_calls: list[tuple[str, int | None]] = []
        self.position_calls: list[tuple[str, int | None]] = []

    def tags_for_move(
        self,
        before: chess.Board,
        move: chess.Move,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]:
        self.move_calls.append((move.uci(), mate_in))
        return [MotifTag(name="fork", squares=["e5"], move=move.uci())]

    def tags_for_position(
        self,
        board: chess.Board,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]:
        self.position_calls.append((board.fen(), mate_in))
        return [MotifTag(name="hangingPiece", squares=["d1"])]


# Expected score and the arithmetic over it


def test_terminal_positions_score_from_the_side_to_move() -> None:
    mated = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert mated.is_checkmate() and terminal_expected_score(mated) == 0.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate() and terminal_expected_score(stalemate) == 0.5
    assert terminal_expected_score(chess.Board()) is None


def test_severity_names_the_thresholds() -> None:
    t = Thresholds(inaccuracy=0.05, mistake=0.10, blunder=0.20)
    assert severity(0.0, t) is Severity.NONE
    assert severity(0.049, t) is Severity.NONE
    assert severity(0.05, t) is Severity.INACCURACY
    assert severity(0.10, t) is Severity.MISTAKE
    assert severity(0.20, t) is Severity.BLUNDER
    assert severity(0.9, t) is Severity.BLUNDER


def test_eval_loss_is_measured_from_the_movers_side() -> None:
    before = [Line(moves=["e2e4"], depth=18, expected_score=0.60)]
    # The opponent is left with 0.55, so the mover kept 0.45 of a position worth 0.60.
    assert eval_loss(before, "d2d4", 0.55) == pytest.approx(0.15)
    # The engine's own choice costs nothing, whatever the search says afterwards.
    assert eval_loss(before, "e2e4", 0.55) == 0.0
    # A move that improves on the search is not negative loss.
    assert eval_loss(before, "d2d4", 0.30) == 0.0
    assert eval_loss([], "d2d4", 0.30) == 0.0


def test_only_move_and_decisive_moments() -> None:
    t = Thresholds(decisive_gap=0.10, only_move_gap=0.30)
    one_legal = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    assert one_legal.legal_moves.count() == 1
    assert is_only_move(one_legal, [], t) and is_decisive(one_legal, [], t)

    board = chess.Board()
    wide = [
        Line(moves=["e2e4"], depth=18, expected_score=0.90),
        Line(moves=["d2d4"], depth=18, expected_score=0.40),
    ]
    narrow = [
        Line(moves=["e2e4"], depth=18, expected_score=0.55),
        Line(moves=["d2d4"], depth=18, expected_score=0.54),
    ]
    assert best_gap(wide) == pytest.approx(0.50) and best_gap(narrow) == pytest.approx(0.01)
    assert best_gap(wide[:1]) is None
    assert is_only_move(board, wide, t) and is_decisive(board, wide, t)
    assert not is_only_move(board, narrow, t) and not is_decisive(board, narrow, t)
    middling = [
        Line(moves=["e2e4"], depth=18, expected_score=0.60),
        Line(moves=["d2d4"], depth=18, expected_score=0.45),
    ]
    assert is_decisive(board, middling, t) and not is_only_move(board, middling, t)


def test_position_score_prefers_the_result_over_the_lines() -> None:
    mated = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert position_score(mated, [Line(moves=[], depth=1, expected_score=0.9)]) == 0.0
    assert position_score(chess.Board(), []) == 0.5


# The engine itself


@needs_engine
def test_lines_come_back_from_the_movers_point_of_view(engine: Engine) -> None:
    board = chess.Board(BLACK_WINS)
    assert board.turn == chess.BLACK
    lines = engine.analyse(board, depth=DEPTH)
    assert lines and lines[0].expected_score > 0.5, "black is to move and better"
    assert lines[0].wdl is not None and sum(lines[0].wdl) == 1000
    assert lines[0].score_cp is not None and lines[0].score_cp > 0
    assert lines[0].depth >= DEPTH


def test_the_human_curve_leaves_room_above_a_won_position() -> None:
    """A knight up is not a hundred per cent, which is the whole reason for the curve."""
    assert human_expected_score(0, None) == pytest.approx(0.5)
    assert human_expected_score(100, None) == pytest.approx(0.591, abs=0.002)
    knight_up = human_expected_score(545, None)
    assert 0.8 < knight_up < 0.92, "still room to throw it away"
    assert human_expected_score(545, None) < human_expected_score(900, None) < 1.0
    assert human_expected_score(-545, None) == pytest.approx(1 - knight_up)
    assert human_expected_score(None, 3) == 1.0 and human_expected_score(None, -3) == 0.0
    # The engine's own numbers answer a different question and do saturate.
    assert engine_expected_score((1000, 0, 0)) == 1.0
    assert engine_expected_score((60, 935, 5)) == pytest.approx(0.5275)
    assert engine_expected_score(None) == 0.5
    assert expected_score("engine", 545, None, (1000, 0, 0)) == 1.0
    assert expected_score("human", 545, None, (1000, 0, 0)) == pytest.approx(knight_up)


@needs_engine
def test_the_engine_model_can_be_asked_for_instead(engine: Engine) -> None:
    board = chess.Board(BLACK_WINS)
    human = engine.analyse(board, depth=DEPTH)[0]
    assert engine.score_model == "human" and human.expected_score < 1.0
    engine.score_model = "engine"
    try:
        strict = engine.analyse(board, depth=DEPTH)[0]
    finally:
        engine.score_model = "human"
    assert strict.wdl is not None
    assert strict.expected_score == pytest.approx(
        (strict.wdl[0] + strict.wdl[1] / 2) / 1000, abs=1e-9
    )
    assert strict.expected_score > human.expected_score


@needs_engine
def test_multipv_is_ordered_best_first_and_bounded_by_the_legal_moves(engine: Engine) -> None:
    lines = engine.analyse(chess.Board(), depth=DEPTH, multipv=4)
    assert len(lines) == 4
    assert [line.expected_score for line in lines] == sorted(
        (line.expected_score for line in lines), reverse=True
    )
    forced = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    assert len(engine.analyse(forced, depth=DEPTH, multipv=4)) == 1


@needs_engine
def test_mate_is_reported_as_a_distance_and_a_certain_score(engine: Engine) -> None:
    lines = engine.analyse(chess.Board(SCHOLAR), depth=DEPTH)
    assert lines[0].moves[0] == "h5f7", "Qxf7 is mate"
    assert lines[0].mate == 1 and lines[0].expected_score == 1.0


@needs_engine
def test_a_finished_position_has_no_lines(engine: Engine) -> None:
    mated = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    assert engine.analyse(mated, depth=DEPTH) == []


@needs_engine
def test_the_engine_is_restarted_when_it_dies(engine: Engine) -> None:
    assert engine._engine is not None
    engine._engine.quit()  # the process is gone, the wrapper does not know yet
    lines = engine.analyse(chess.Board(), depth=DEPTH)
    assert lines, "a dead engine is restarted rather than ending the session"


@needs_engine
def test_threats_are_what_the_opponent_would_play(engine: Engine) -> None:
    # Nothing is attacking white this move, which is exactly why the threat must be asked for.
    board = chess.Board(THREATENED)
    assert not board.is_check()
    assert threat_moves(engine, board, depth=DEPTH) == ["d8d1"], "...Rd1 is mate"
    in_check = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    assert in_check.is_check() and not in_check.is_game_over()
    assert threat_moves(engine, in_check, depth=DEPTH) == [], "you cannot pass out of check"


def test_san_line_reads_like_a_move_list() -> None:
    board = chess.Board()
    assert san_line(board, ["e2e4", "e7e5", "g1f3"]) == "1. e4 e5 2. Nf3"
    board.push_san("e4")
    assert san_line(board, ["e7e5", "g1f3"]) == "1... e5 2. Nf3"
    assert san_line(board, ["e2e4"]) == "", "a line that does not fit the board stops"
    assert san_line(board, ["nonsense"]) == ""


# The annotator


def test_the_mate_distance_counts_the_move_being_judged() -> None:
    before = chess.Board(SCHOLAR)
    mate_now = chess.Move.from_uci("h5f7")
    assert _mate_in(before, mate_now, []) == 1
    quiet = chess.Move.from_uci("d2d3")
    # After the move the opponent is to move and is mated in two of the mover's moves.
    assert _mate_in(before, quiet, [Line(moves=[], depth=9, expected_score=0.0, mate=-2)]) == 3
    # The opponent mates instead.
    assert _mate_in(before, quiet, [Line(moves=[], depth=9, expected_score=1.0, mate=2)]) == -2
    assert _mate_in(before, quiet, [Line(moves=[], depth=9, expected_score=0.5)]) is None


def test_each_position_is_measured_once() -> None:
    board = chess.Board()
    fake = FakeEngine({})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    e2e4 = chess.Move.from_uci("e2e4")
    annotator.assess("g", board, e2e4, 1)
    after = board.copy(stack=False)
    after.push(e2e4)
    annotator.assess("g", after, chess.Move.from_uci("e7e5"), 2)
    assert len(fake.calls) == 3, "start, after e4, after e5: each measured once"
    annotator.reset()
    annotator.assess("g", board, e2e4, 1)
    assert len(fake.calls) == 5, "a new game measures again"


def test_assessment_carries_the_loss_the_best_line_and_the_tags() -> None:
    board = chess.Board()
    after = board.copy(stack=False)
    after.push(chess.Move.from_uci("a2a3"))
    fake = FakeEngine(
        {
            board.fen(): [
                Line(moves=["e2e4", "e7e5"], depth=6, expected_score=0.60),
                Line(moves=["d2d4"], depth=6, expected_score=0.45),
            ],
            after.fen(): [
                Line(moves=["e7e5"], depth=6, expected_score=0.55, mate=-3),
                Line(moves=["d7d5"], depth=6, expected_score=0.20),
            ],
        }
    )
    tagger = RecordingTagger()
    annotator = GameAnnotator(engine=fake, depth=6, tagger=tagger, with_threats=False)
    assessment, analysis = annotator.assess("game1", board, chess.Move.from_uci("a2a3"), 1)
    assert isinstance(assessment, MoveAssessment) and isinstance(analysis, Analysis)
    assert assessment.expected_loss == pytest.approx(0.15)
    assert assessment.severity is Severity.MISTAKE
    assert assessment.best == ["e2e4", "e7e5"]
    assert [tag.name for tag in assessment.tags] == ["fork"]
    assert tagger.move_calls == [("a2a3", 4)], "mated in three more moves, four counting this"
    assert analysis.ply == 1 and analysis.fen == after.fen()
    assert [tag.name for tag in analysis.tags] == ["hangingPiece"]
    assert analysis.decisive and analysis.only_move, "the replies differ by 0.35"


def test_the_assessment_says_how_deep_the_search_got() -> None:
    """A position that hits the engine's time cap is measured shallower than asked."""
    board = chess.Board()
    after = board.copy()
    after.push(chess.Move.from_uci("e2e4"))
    fake = FakeEngine(
        {
            board.fen(): [Line(moves=["e2e4"], depth=18, expected_score=0.55)],
            after.fen(): [Line(moves=["e7e5"], depth=11, expected_score=0.45)],
        }
    )
    annotator = GameAnnotator(engine=fake, depth=18, with_threats=False)
    assessment, _ = annotator.assess("g", board, chess.Move.from_uci("e2e4"), 1)
    assert assessment.depth == 11, "the shallower of the two searches"


@needs_engine
def test_no_single_position_may_hold_up_a_game(engine: Engine) -> None:
    """A knight endgame that takes minutes to reach depth 14 with four lines is cut short."""
    slow = chess.Board("8/k7/2p3NP/8/1P1K2P1/2P5/3n4/8 b - - 2 50")
    engine.max_seconds = 0.5
    started = time.monotonic()
    try:
        lines = engine.analyse(slow, depth=40, multipv=4)
    finally:
        engine.max_seconds = 5.0
    assert lines and time.monotonic() - started < 5.0
    assert lines[0].depth < 40, "the depth asked for was not reached, and that is reported"


def test_an_illegal_move_is_refused_rather_than_measured() -> None:
    annotator = GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False)
    with pytest.raises(ValueError, match="not legal"):
        annotator.assess("g", chess.Board(), chess.Move.from_uci("e2e5"), 1)


def test_the_analyser_answers_off_the_caller_s_thread() -> None:
    board = chess.Board()
    fake = FakeEngine({})
    with Analyser(GameAnnotator(engine=fake, depth=6, with_threats=False)) as analyser:
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1)
        events = _drain(analyser)
    assert [type(event).__name__ for event in events] == ["MoveAssessment", "Analysis"]
    assert analyser.failures == 0


def test_a_failing_analysis_does_not_end_the_session() -> None:
    class Broken(FakeEngine):
        def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
            raise RuntimeError("engine gone")

    with Analyser(GameAnnotator(engine=Broken({}), depth=6, with_threats=False)) as analyser:
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        deadline = time.monotonic() + 5.0
        while analyser.failures == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert analyser.failures == 1
    assert analyser.last_error is not None and "engine gone" in analyser.last_error


def test_the_queue_drops_the_oldest_work_when_it_cannot_keep_up() -> None:
    annotator = GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False)
    analyser = Analyser(annotator, max_pending=2)  # never started: nothing consumes
    for _ in range(5):
        analyser.submit_position("g", chess.Board(), 0)
    assert analyser.dropped == 3 and analyser.pending == 2


@needs_engine
def test_a_real_game_is_assessed_end_to_end(engine: Engine) -> None:
    """The scholar's mate: the losing move is a blunder, the mate is not."""
    annotator = GameAnnotator(engine=engine, depth=DEPTH, multipv=3, with_threats=False)
    board = chess.Board()
    losses: dict[str, float] = {}
    final: Analysis | None = None
    for ply, san in enumerate(["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"], start=1):
        move = board.parse_san(san)
        assessment, final = annotator.assess("scholar", board, move, ply)
        losses[san] = assessment.expected_loss
        board.push(move)
    assert losses["Nf6"] > 0.5, "the move that allows mate is a blunder"
    assert losses["Qxf7#"] == 0.0 and losses["e4"] < 0.05
    assert final is not None and final.lines == [] and board.is_checkmate()


def _drain(analyser: Analyser, timeout: float = 10.0) -> list[object]:
    deadline = time.monotonic() + timeout
    events: list[object] = []
    while time.monotonic() < deadline:
        events += analyser.poll()
        if events and analyser.pending == 0:
            return events
        time.sleep(0.01)
    return events
