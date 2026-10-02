"""The engine wrapper, the expected-score arithmetic, and the annotator that joins them."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
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
    game_accuracy,
    human_expected_score,
    is_decisive,
    is_only_move,
    label_move,
    move_accuracy,
    position_score,
    severity,
    terminal_expected_score,
)
from sidecoach.config import CoachSettings, find_stockfish, load_settings
from sidecoach.events import Analysis, Label, Line, MotifTag, MoveAssessment, Severity
from tests.coaching import SCHOLAR
from tests.fakes import FakeEngine

STOCKFISH = find_stockfish(load_settings())


def needs_engine[F: Callable[..., None]](test: F) -> F:
    """A test of the real engine. They share one worker (`--dist loadgroup`), so the
    module's one Stockfish is started once, not once on every worker one lands on."""
    test = pytest.mark.skipif(STOCKFISH is None, reason="no Stockfish binary")(test)
    return pytest.mark.xdist_group("stockfish")(test)


# Deep enough to be right about the positions below, shallow enough to keep the suite quick.
DEPTH = 10

# Black to move at queen odds: winning, but with no mate in sight to hide the numbers.
BLACK_WINS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
# The scholar's mate, played: black to move and mated.
MATED = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"


@pytest.fixture(scope="module")
def engine():
    assert STOCKFISH is not None
    with Engine(path=STOCKFISH, threads=2, hash_mb=64, multipv=4) as running:
        yield running


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
        # A fork the line never cashes in, and a sacrifice the payoff rule does not judge:
        # the annotator keeps a move's tactic only when the line after it wins with it.
        return [
            MotifTag(name="fork", squares=["a3", "b4", "c4"], move=move.uci()),
            MotifTag(name="sacrifice", squares=["a3"], move=move.uci()),
        ]

    def tags_for_position(
        self,
        board: chess.Board,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]:
        self.position_calls.append((board.fen(), mate_in))
        return [MotifTag(name="hangingPiece", squares=["d1"])]


def test_a_live_game_stays_in_the_book_past_its_first_move(tmp_path: Path) -> None:
    """The live path hands the analyser each move's board; the book needs the moves that
    led to it. An Italian Game submitted the way the panel submits it -- a takeback in the
    middle -- is book to its last move and named as the whole line names it, not as its
    first move does."""
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
    # A takeback of 3.d4: the next move is played from the position before it.
    board.pop()
    assessing.submit(Takeback(game_id="g", plies=1, fen=board.fen()))
    play(board, "d3", 7)

    labels: dict[int, object] = {}
    openings: dict[int, object] = {}
    deadline = time.monotonic() + 10
    # Ply 7 comes twice, d4 then d3: waited for until d3's own name arrives.
    while time.monotonic() < deadline and openings.get(7) != "C50 Italian Game: Giuoco Pianissimo":
        for event in assessing.analyser.poll():
            if isinstance(event, MoveAssessment):
                labels[event.ply] = event.label
            elif isinstance(event, Analysis):
                openings[event.ply] = event.opening
        time.sleep(0.02)
    assessing.analyser.close()
    assert [labels[p] for p in range(1, 7)] == [Label.BOOK] * 6, labels
    assert labels[7] == Label.BOOK, "the move after a takeback is still in the book"
    assert openings[1] == "B00 King's Pawn Game"
    assert openings[5] == "C50 Italian Game"
    assert openings[7] == "C50 Italian Game: Giuoco Pianissimo", openings


# Expected score and the arithmetic over it


def test_a_finished_position_scores_its_result_from_the_side_to_move() -> None:
    mated = chess.Board(MATED)
    assert mated.is_checkmate() and terminal_expected_score(mated) == 0.0
    stalemate = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert stalemate.is_stalemate() and terminal_expected_score(stalemate) == 0.5
    assert terminal_expected_score(chess.Board()) is None
    # The result is preferred over whatever the lines say; no lines and no result is even.
    assert position_score(mated, [Line(moves=[], depth=1, expected_score=0.9)]) == 0.0
    assert position_score(chess.Board(), []) == 0.5


def _shuffled(*sans: str) -> chess.Board:
    """White a queen up, the kings shuffling: K+Q against K, black to move first."""
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1")
    for san in sans:
        board.push_san(san)
    return board


# White, a queen up, to move, and Ke1 would repeat the position a third time.
CLAIMABLE = ("Kd8", "Ke2", "Ke8", "Ke1", "Kd8", "Ke2", "Ke8")


def test_a_draw_that_could_be_claimed_is_not_a_finished_game() -> None:
    """A repetition the side to move could claim is not one played: the sites end the game
    on the third occurrence itself, so until then the position is played on, and a queen up
    is a queen up (it was 0.5, "no lines", and the repeating move "excellent")."""
    claimable = _shuffled(*CLAIMABLE)
    assert claimable.can_claim_threefold_repetition() and not claimable.is_repetition(3)
    assert terminal_expected_score(claimable) is None
    # The third occurrence, played, is the end of the game.
    repeated = _shuffled(*CLAIMABLE, "Ke1")
    assert repeated.is_repetition(3) and terminal_expected_score(repeated) == 0.5
    # Fifty moves without a capture or a pawn move, played, is the end too.
    fifty = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 100 80")
    assert terminal_expected_score(fifty) == 0.5
    near = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 99 80")
    assert near.can_claim_fifty_moves() and terminal_expected_score(near) is None


@needs_engine
def test_a_claimable_draw_is_searched_like_any_position(engine: Engine) -> None:
    lines = engine.analyse(_shuffled(*CLAIMABLE), depth=DEPTH)
    assert lines, "the side to move has moves to play"
    assert lines[0].expected_score > 0.8, "a queen up, white plays on"


def test_severity_names_the_thresholds() -> None:
    t = Thresholds(inaccuracy=0.05, mistake=0.10, blunder=0.20)
    assert severity(0.0, t) is Severity.NONE
    assert severity(0.049, t) is Severity.NONE
    assert severity(0.05, t) is Severity.INACCURACY
    assert severity(0.10, t) is Severity.MISTAKE
    assert severity(0.20, t) is Severity.BLUNDER
    assert severity(0.9, t) is Severity.BLUNDER
    # The fitted lines are the settings' defaults: a re-fit changes both or neither.
    assert Thresholds() == Thresholds.from_settings(CoachSettings())


def _label(
    loss: float,
    *,
    is_best: bool = False,
    in_book: bool = False,
    only_move: bool = False,
    sacrifice: bool = False,
    before_score: float = 0.5,
    opponent_erred: bool = False,
    obvious: bool = False,
) -> Label:
    return label_move(
        loss,
        Thresholds(),
        is_best=is_best,
        in_book=in_book,
        only_move=only_move,
        sacrifice=sacrifice,
        before_score=before_score,
        opponent_erred=opponent_erred,
        obvious=obvious,
    )


def test_labels_follow_the_review_vocabulary() -> None:
    assert _label(0.0, in_book=True) is Label.BOOK
    assert _label(0.14, in_book=True) is Label.MISTAKE, "a gambit in the tables still loses"
    assert _label(0.0, is_best=True) is Label.BEST
    assert _label(0.01) is Label.EXCELLENT
    assert _label(0.03) is Label.GOOD
    assert _label(0.06) is Label.INACCURACY
    assert _label(0.12) is Label.MISTAKE
    assert _label(0.30) is Label.BLUNDER
    # The opponent handed over a chance: taking it with the best move is great, letting it
    # go is a miss however large the loss, and a decent move that is neither is just good.
    assert _label(0.0, opponent_erred=True, is_best=True) is Label.GREAT
    assert _label(0.0, opponent_erred=True, only_move=True) is Label.GREAT
    assert _label(0.03, opponent_erred=True) is Label.GOOD
    assert _label(0.06, opponent_erred=True) is Label.MISS
    assert _label(0.90, opponent_erred=True) is Label.MISS
    # A sound sacrifice is brilliant, unless the position was already decided either way.
    assert _label(0.0, sacrifice=True) is Label.BRILLIANT
    assert _label(0.0, sacrifice=True, before_score=0.97, is_best=True) is Label.BEST
    assert _label(0.0, sacrifice=True, before_score=0.05, is_best=True) is Label.BEST
    # The only move is great whether or not it was the engine's first choice, unless it is
    # obvious (a recapture, the only legal move) or the game was already decided.
    assert _label(0.0, only_move=True) is Label.GREAT
    assert _label(0.0, only_move=True, obvious=True) is Label.EXCELLENT
    assert _label(0.0, only_move=True, obvious=True, is_best=True) is Label.BEST
    assert _label(0.0, opponent_erred=True, is_best=True, obvious=True) is Label.BEST
    # A great move in a won position is still great; only brilliant needs a live game.
    assert _label(0.0, only_move=True, before_score=0.95) is Label.GREAT
    assert _label(0.0, opponent_erred=True, is_best=True, before_score=0.98) is Label.GREAT
    # In a lost position the best move changes nothing: neither great nor brilliant.
    assert _label(0.0, only_move=True, before_score=0.02, is_best=True) is Label.BEST
    assert _label(0.0, sacrifice=True, before_score=0.02) is Label.EXCELLENT
    # A move as good as the engine's choice but not that choice is excellent, not best:
    # measured against chess.com's tallies, "best" is the engine's first choice only.
    assert _label(0.0) is Label.EXCELLENT
    # An unsound sacrifice is judged by its loss like any move.
    assert _label(0.3, sacrifice=True) is Label.BLUNDER


def test_accuracy_is_a_hundred_for_no_loss_and_falls_with_it() -> None:
    assert move_accuracy(0.0) == 100.0
    assert 75 < move_accuracy(0.01) < 80
    assert 24 < move_accuracy(0.05) < 29, "an inaccuracy on chess.com's scale"
    assert move_accuracy(0.30) == 0.0 and move_accuracy(1.0) == 0.0
    assert game_accuracy([]) is None
    assert game_accuracy([100.0, 100.0]) == 100.0
    assert game_accuracy([100.0, 50.0]) == 75.0
    assert game_accuracy([100.0, 0.0]) == 50.0, "one lost game does not halve the number"


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


# The engine itself


@needs_engine
def test_lines_come_back_from_the_movers_point_of_view_on_the_curve_asked_for(
    engine: Engine,
) -> None:
    board = chess.Board(BLACK_WINS)
    assert board.turn == chess.BLACK
    lines = engine.analyse(board, depth=DEPTH)
    assert lines and lines[0].expected_score > 0.5, "black is to move and better"
    assert lines[0].wdl is not None and sum(lines[0].wdl) == 1000
    assert lines[0].score_cp is not None and lines[0].score_cp > 0
    assert lines[0].depth >= DEPTH
    # The human curve by default; the engine's own numbers when asked, which saturate.
    human = lines[0]
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
def test_multipv_is_ordered_best_first_and_bounded_by_the_legal_moves(engine: Engine) -> None:
    lines = engine.analyse(chess.Board(), depth=DEPTH, multipv=4)
    assert len(lines) == 4
    assert [line.expected_score for line in lines] == sorted(
        (line.expected_score for line in lines), reverse=True
    )
    forced = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    assert len(engine.analyse(forced, depth=DEPTH, multipv=4)) == 1


def test_a_finished_position_has_no_lines() -> None:
    """Answered before any process is asked: a finished game has nothing to play."""
    assert Engine(path=Path("never-started")).analyse(chess.Board(MATED), depth=DEPTH) == []


@needs_engine
def test_the_engine_is_restarted_when_it_dies(engine: Engine) -> None:
    assert engine._engine is not None
    engine._engine.quit()  # the process is gone, the wrapper does not know yet
    lines = engine.analyse(chess.Board(), depth=DEPTH)
    assert lines, "a dead engine is restarted rather than ending the session"


@needs_engine
def test_the_best_line_is_carried_on_for_the_coach(engine: Engine) -> None:
    """`carry_best`: the best line is kept to MIN_BEST_PLIES, searched on from its end when
    Stockfish's own stops short, every move of it legal where it stands; the other lines are
    cut as always, and a verdict's search (no `carry_best`) like the rest (the parity audit,
    2026-09-29: the phone's EngineTest held the only test of it)."""
    from sidecoach.analysis.engine import MIN_BEST_PLIES

    board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    carried = engine.analyse(board, depth=DEPTH, pv_plies=4, carry_best=True)
    plain = engine.analyse(board, depth=DEPTH, pv_plies=4)
    assert carried and plain
    best = carried[0].moves
    end = board.copy(stack=False)
    for uci in best:
        move = chess.Move.from_uci(uci)
        assert move in end.legal_moves, f"{uci} is not legal where the carried line has it"
        end.push(move)
    assert len(best) == MIN_BEST_PLIES or end.is_game_over(claim_draw=True), best
    assert all(len(line.moves) <= 4 for line in carried[1:]), "another line was carried on"
    assert len(plain[0].moves) <= 4, "a verdict's line was carried on"


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
    """Measured whole once each; the verdict on a move searches the position it led to
    for its best line alone first (a call of its own, one line), then the whole."""
    board = chess.Board()
    fake = FakeEngine({})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    e2e4 = chess.Move.from_uci("e2e4")
    assessment, _ = annotator.assess("g", board, e2e4, 1)
    after = board.copy(stack=False)
    after.push(e2e4)
    assert assessment.uci == "e2e4"
    # The verdict from one line of the position after the move, before it is measured whole.
    assert list(zip(fake.calls, fake.multipvs, strict=True)) == [
        (board.fen(), 1),
        (after.fen(), 1),
        (after.fen(), 6),
    ]
    annotator.assess("g", after, chess.Move.from_uci("e7e5"), 2)
    whole = [fen for fen, multipv in zip(fake.calls, fake.multipvs, strict=True) if multipv != 1]
    assert len(whole) == 2, "after e4, after e5: each measured whole once"
    # The start is measured for the verdict alone (its whole analysis is the pipeline's
    # own position submit); each position after a move once thin, then whole.
    assert fake.multipvs.count(1) == 3, "one verdict search per position"
    annotator.reset()
    annotator.assess("g", board, e2e4, 1)
    whole = [fen for fen, multipv in zip(fake.calls, fake.multipvs, strict=True) if multipv != 1]
    assert len(whole) == 3, "a new game measures again"


def test_assessment_carries_the_loss_the_best_line_the_tags_and_the_depth() -> None:
    board = chess.Board()
    after = board.copy(stack=False)
    after.push(chess.Move.from_uci("a2a3"))
    # The position after the move hit the engine's time cap: measured shallower than asked.
    fake = FakeEngine(
        {
            board.fen(): [
                Line(moves=["e2e4", "e7e5"], depth=6, expected_score=0.60),
                Line(moves=["d2d4"], depth=6, expected_score=0.45),
            ],
            after.fen(): [
                Line(moves=["e7e5"], depth=4, expected_score=0.55, mate=-3),
                Line(moves=["d7d5"], depth=4, expected_score=0.20),
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
    assert assessment.depth == 4, "the shallower of the two searches"
    assert [tag.name for tag in assessment.tags] == ["sacrifice"], "the unpaid fork is dropped"
    assert tagger.move_calls == [("a2a3", 4)], "mated in three more moves, four counting this"
    assert analysis.ply == 1 and analysis.fen == after.fen()
    assert [tag.name for tag in analysis.tags] == ["hangingPiece"]
    assert analysis.decisive and analysis.only_move, "the replies differ by 0.35"


def test_a_capture_guarded_only_by_a_pinned_piece_is_a_free_grab() -> None:
    """A capture nobody can take back is obvious, never great, and taking back is read by
    the rules: the c3 knight guarding e4 is pinned to its king by the b4 bishop, so ...Nxe4
    is a grab. With the knight free to take back, the one move that holds is great."""

    def label(fen: str) -> Label | None:
        board = chess.Board(fen)
        fake = FakeEngine(
            {
                board.fen(): [
                    Line(moves=["f6e4"], depth=6, expected_score=0.80),
                    Line(moves=["e8d7"], depth=6, expected_score=0.40),
                ]
            }
        )
        annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
        annotator.measure(board)  # measured whole, so the only-move call is made
        return annotator.assess_move("g", board, chess.Move.from_uci("f6e4"), 1).label

    assert label("4k3/8/5n2/8/1b2B3/2N5/8/4K3 b - - 0 1") is Label.BEST
    assert label("4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1") is Label.GREAT


@needs_engine
def test_no_single_position_may_hold_up_a_game(engine: Engine) -> None:
    """A knight endgame that takes minutes to reach depth 14 with four lines is cut short."""
    slow = chess.Board("8/k7/2p3NP/8/1P1K2P1/2P5/3n4/8 b - - 2 50")
    engine.max_seconds = 0.1
    started = time.monotonic()
    try:
        lines = engine.analyse(slow, depth=40, multipv=4)
    finally:
        engine.max_seconds = 5.0
    assert lines and time.monotonic() - started < 5.0
    assert lines[0].depth < 40, "the depth asked for was not reached, and that is reported"


@needs_engine
def test_a_search_cut_short_gives_lines_of_one_depth(engine: Engine) -> None:
    """Cut by the time cap mid-depth, Stockfish's best line is a depth further than the
    rest, and the gap between them (the only-move and decisive-moment calls) set a deeper
    search against a shallower one. The lines are those of the last depth finished whole."""
    slow = chess.Board("8/k7/2p3NP/8/1P1K2P1/2P5/3n4/8 b - - 2 50")
    quiet = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 9")
    engine.max_seconds = 0.15
    try:
        cuts = [
            engine.analyse(board, depth=40, multipv=4) for board in (slow, quiet) for _ in range(4)
        ]
    finally:
        engine.max_seconds = 5.0
    for lines in cuts:
        assert len(lines) == 4
        assert len({line.depth for line in lines}) == 1, [line.depth for line in lines]


# A UCI engine that answers the handshake and then never answers a search: alive, and silent.
_SILENT_ENGINE = """
import sys
for line in sys.stdin:
    word = line.split()[0] if line.split() else ""
    if word == "uci":
        print("id name silent")
        print("option name MultiPV type spin default 1 min 1 max 500")
        print("uciok", flush=True)
    elif word == "isready":
        print("readyok", flush=True)
    elif word == "quit":
        break
"""


@needs_engine
def test_an_engine_that_stops_answering_is_replaced(tmp_path: Path) -> None:
    """An engine that hangs without dying held the analyser's thread for good: the search
    waited on it forever. A search running well past its time cap has its process ended,
    and the search is made again on a new one."""
    import sys

    import chess.engine

    assert STOCKFISH is not None
    stub = tmp_path / "silent.py"
    stub.write_text(_SILENT_ENGINE, encoding="utf-8")
    running = Engine(path=STOCKFISH, threads=1, hash_mb=16, multipv=2, max_seconds=0.3)
    running._engine = chess.engine.SimpleEngine.popen_uci([sys.executable, str(stub)])
    running.watchdog_s = 2.0  # room for the new engine to start on a busy machine
    try:
        got: dict[str, list[Line]] = {}
        worker = threading.Thread(
            target=lambda: got.setdefault("lines", running.analyse(chess.Board(), depth=30))
        )
        started = time.monotonic()
        worker.start()
        worker.join(15)
        assert not worker.is_alive(), "the search still waits on the silent engine"
        assert got["lines"], "searched again on a new engine"
        assert time.monotonic() - started < 10
    finally:
        running.close()


@needs_engine
def test_settings_saved_mid_search_wait_for_the_search_to_end() -> None:
    """Threads or hash saved while a search runs used to reach Stockfish at once: the
    `setoption` stopped the search, and the cut one was kept as if whole (0.6 s and depth 14
    of a 4 s search, the engine review 2026-10-01). They now wait for the search to end."""
    assert STOCKFISH is not None
    quiet = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N2N2/PP2BPPP/R1BQ1RK1 w - - 0 9")
    with Engine(path=STOCKFISH, threads=2, hash_mb=64, multipv=4, max_seconds=1.5) as running:
        got: dict[str, float] = {}

        def search() -> None:
            started = time.monotonic()
            running.analyse(quiet, depth=60)
            got["took"] = time.monotonic() - started

        worker = threading.Thread(target=search)
        worker.start()
        time.sleep(0.4)
        saved = time.monotonic()
        running.reconfigure(threads=1, hash_mb=32)
        assert time.monotonic() - saved < 0.2, "the save does not wait on the search"
        worker.join(10)
        assert got["took"] > 1.2, f"the search was cut short at {got['took']:.2f} s"
        assert running.threads == 1 and running.hash_mb == 32
        running.analyse(quiet, depth=4)  # the next search starts with them
        assert running._engine is not None
        config = vars(running._engine.protocol)["config"]  # what was last set on the process
        assert config.get("Threads") == 1 and config.get("Hash") == 32


def test_an_illegal_move_is_refused_rather_than_measured() -> None:
    annotator = GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False)
    with pytest.raises(ValueError, match="not legal"):
        annotator.assess("g", chess.Board(), chess.Move.from_uci("e2e5"), 1)


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
    """The scholar's mate: the losing move is a blunder, the mate is not, and the mate is
    reported as a distance and a certain score."""
    annotator = GameAnnotator(engine=engine, depth=DEPTH, multipv=3, with_threats=False)
    board = chess.Board()
    losses: dict[str, float] = {}
    analyses: dict[str, Analysis] = {}
    for ply, san in enumerate(["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#"], start=1):
        move = board.parse_san(san)
        assessment, analyses[san] = annotator.assess("scholar", board, move, ply)
        losses[san] = assessment.expected_loss
        board.push(move)
    assert losses["Nf6"] > 0.5, "the move that allows mate is a blunder"
    assert losses["Qxf7#"] == 0.0 and losses["e4"] < 0.04
    mate = analyses["Nf6"].lines[0]
    assert analyses["Nf6"].fen == SCHOLAR and mate.moves[0] == "h5f7", "Qxf7 is mate"
    assert mate.mate == 1 and mate.expected_score == 1.0
    assert analyses["Qxf7#"].lines == [] and board.is_checkmate()


def _drain(analyser: Analyser, timeout: float = 10.0) -> list[object]:
    deadline = time.monotonic() + timeout
    events: list[object] = []
    while time.monotonic() < deadline:
        events += analyser.poll()
        if events and analyser.pending == 0:
            return events
        time.sleep(0.01)
    return events


def test_behind_a_fast_game_the_analyser_keeps_the_verdicts_and_drops_stale_analyses() -> None:
    """Moves arriving faster than the engine measures them: every ply still gets its
    verdict (the best line at the full depth, both positions), and the whole analysis --
    candidates, threats -- is run only for the position the player faces (2026-09-22: a
    queen ending at two seconds a move ran forty plies behind, every word on a move
    'not spoken: board moved on', and the review waited on the backlog)."""
    fake = FakeEngine({})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    analyser = Analyser(annotator)
    board = chess.Board()
    # Five plies queued before the worker starts: it is behind from the first one.
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bb5"], start=1):
        move = board.parse_san(san)
        analyser.submit_move("g", board, move, ply)
        board.push(move)
    analyser.start()
    deadline = time.monotonic() + 5
    got: list = []
    while time.monotonic() < deadline and len([e for e in got if isinstance(e, Analysis)]) < 5:
        got += analyser.poll()
        time.sleep(0.02)
    time.sleep(0.1)
    got += analyser.poll()
    analyser.close()
    assessments = [e for e in got if isinstance(e, MoveAssessment)]
    analyses = [e for e in got if isinstance(e, Analysis)]
    assert [a.ply for a in assessments] == [1, 2, 3, 4, 5], "a verdict for every ply"
    # Every ply still gets an Analysis -- a thin one from the verdict's own line for the
    # plies the game has left (the coach needs the facts of the position each move was
    # played from; the review its score) -- and the latest one the whole measurement.
    assert [a.ply for a in analyses] == [1, 2, 3, 4, 5]
    assert all(len(a.lines) == 1 and not a.threats for a in analyses[:4]), "thin for stale plies"
    assert analyser.skipped == 4
    assert fake.multipvs.count(6) == 1, "one whole measurement: the position the player faces"


class _HeldEngine(FakeEngine):
    """A fake engine whose searches wait to be let go, so a test can act mid-search."""

    def __init__(self) -> None:
        super().__init__({})
        import threading

        self.searching = threading.Event()
        self.go = threading.Event()

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        self.searching.set()
        self.go.wait(5)
        return super().analyse(board, **kwargs)


def test_a_takeback_lets_the_engine_go_of_the_withdrawn_moves() -> None:
    """Tai, 2026-09-23: "takebacks make the program lag behind by a lot". The moves taken
    back were measured one after another before the replacement was: now what is queued for
    them is dropped, and the one being measured delivers nothing and is not followed by the
    whole analysis of the position after it."""
    engine = _HeldEngine()
    analyser = Analyser(GameAnnotator(engine=engine, depth=6, with_threats=False))
    board = chess.Board()
    moves = []
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1):
        move = board.parse_san(san)
        moves.append((board.copy(), move, ply))
        board.push(move)
    analyser.submit_move("g", *moves[0])
    analyser.start()
    assert engine.searching.wait(5), "the worker never started on ply 1"
    for before, move, ply in moves[1:]:
        analyser.submit_move("g", before, move, ply)
    # Plies 3 and 4 taken back while ply 1 is measured and 2 waits.
    analyser.withdraw("g", 2)
    assert analyser.withdrawn == 2 and analyser.pending == 2
    # Ply 1 taken back too, mid-search: its verdict is not delivered.
    analyser.withdraw("g", 0)
    assert analyser.withdrawn == 4
    engine.go.set()
    # Nothing is delivered, so wait for the work to be settled, not for events: a result is
    # put before its request is settled, so none can come after `pending` is 0.
    assert _until(lambda: analyser.pending == 0, 3), analyser.pending
    got = analyser.poll()
    analyser.close()
    assert [e for e in got if getattr(e, "ply", None) in (1, 3, 4)] == [], got
    assert analyser.pending == 0


def _line(*moves: str) -> Line:
    return Line(moves=list(moves), depth=6, expected_score=0.5)


def _after(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def _until(condition, timeout: float = 5.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


def test_while_no_move_waits_the_engine_looks_ahead_and_a_move_there_is_answered_at_once() -> None:
    """Tai, 2026-09-27: "why is it not possible to do some form of SMART caching of lines".
    With the position on the board measured and nothing queued, the positions after its
    best moves, and after the best reply to each, are measured and cached; the move played
    there is judged, and the position after it known, with no search of its own. The reply
    into a position looked at ahead is told whole at once, with no early word."""
    start = chess.Board()
    fake = FakeEngine(
        {
            start.fen(): [_line("e2e4"), _line("d2d4")],
            _after("e4"): [_line("e7e5")],
            _after("d4"): [_line("d7d5")],
        }
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    with Analyser(annotator, look_ahead=2) as analyser:
        analyser.submit_position("g", start, 0)
        assert _until(lambda: analyser.looked == 4), analyser.looked
        searched = set(fake.calls)
        assert {_after("e4"), _after("e4", "e5"), _after("d4"), _after("d4", "d5")} <= searched
        before = len(fake.calls)
        analyser.submit_move("g", start, chess.Move.from_uci("e2e4"), 1, by_user=True)
        got = _drain(analyser)
        assert _after("e4") not in fake.calls[before:], "the position after e4 was searched again"
        assert annotator.ahead_hits == 1
        after = chess.Board()
        after.push_san("e4")
        analyser.submit_move("g", after, chess.Move.from_uci("e7e5"), 2, by_user=False)
        reply = _drain(analyser)
    assert _kinds(got) == ["Analysis(early)", "Analysis", "MoveAssessment", "Analysis"]
    assert _kinds(reply) == ["MoveAssessment", "Analysis"]
    assert annotator.ahead_hits == 2


def test_a_position_looked_at_ahead_carries_the_moves_that_lead_there() -> None:
    """The engine reads a repetition from the moves before a position, so a position looked
    at ahead is searched with them, as the game's own are: measured without them it was
    cached, and the game arriving there judged its moves on a search that could not see the
    repetition (the engine review, 2026-10-01)."""
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1"):
        board.push_san(san)
    fake = FakeEngine({board.fen(): [_line("f6g8")]}, default=[_line("g1f3")])
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    with Analyser(annotator, look_ahead=1) as analyser:
        before = board.copy()
        before.pop()
        analyser.submit_move("g", before, chess.Move.from_uci("f3g1"), 3, by_user=False)
        assert _until(lambda: analyser.looked == 2), analyser.looked
    ahead = [
        h
        for fen, h in zip(fake.calls, fake.histories, strict=True)
        if fen == _after("Nf3", "Nf6", "Ng1", "Ng8")
    ]
    assert ahead and ahead[-1] == ["g1f3", "g8f6", "f3g1", "f6g8"], fake.histories


def test_one_position_reached_by_two_histories_is_measured_for_each() -> None:
    """The same placement after different quiet moves is not the same position to the
    engine: which positions repeat differs. Cached by its FEN alone, the first history's
    search answered the second."""
    one = chess.Board()
    for san in ("Nf3", "Nc6", "Nc3", "Nf6"):
        one.push_san(san)
    two = chess.Board()
    for san in ("Nc3", "Nf6", "Nf3", "Nc6"):
        two.push_san(san)
    assert one.fen() == two.fen()
    fake = FakeEngine()
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    annotator.measure(one)
    annotator.measure(two)
    annotator.measure(two.copy())
    assert fake.calls.count(one.fen()) == 2, "each history searched once"
    # After a capture or a pawn move nothing before can repeat: one search serves both.
    for board in (one, two):
        board.push_san("e4")
    annotator.measure(one)
    annotator.measure(two)
    assert fake.calls.count(one.fen()) == 1


class _DepthEngine(FakeEngine):
    """A fake engine that answers with the depth it was asked for."""

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        lines = super().analyse(board, **kwargs)
        depth = int(kwargs.get("depth") or 0)  # type: ignore[call-overload]
        return [line.model_copy(update={"depth": depth}) for line in lines]


def _kinds(events: list[object]) -> list[str]:
    out = []
    for e in events:
        name = type(e).__name__
        out.append(f"{name}(early)" if getattr(e, "provisional", False) else name)
    return out


def test_the_position_the_player_faces_is_told_early_and_then_whole() -> None:
    """Tai, 2026-09-28: "first usable answer is depth 12 ... the computation continues until
    max depth of 18". After the opponent's move, and for a position begun from, the position
    the player faces is measured at EARLY_DEPTH and delivered, provisional, before the verdict
    and the full depth; after the player's own move the verdict comes first, as it did."""
    from sidecoach.analysis.annotator import EARLY_DEPTH

    board = chess.Board()
    fake = _DepthEngine({})
    with Analyser(GameAnnotator(engine=fake, depth=18, with_threats=False)) as analyser:
        analyser.submit_position("g", board, 0)
        started = _drain(analyser)
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
        own = _drain(analyser)
        board.push_san("e4")
        analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
        reply = _drain(analyser)
    assert _kinds(started) == ["Analysis(early)", "Analysis"]
    # Off the caller's thread, the verdict first: the player's own move.
    assert _kinds(own) == ["MoveAssessment", "Analysis"]
    assert analyser.failures == 0
    assert _kinds(reply) == ["Analysis(early)", "MoveAssessment", "Analysis"]
    early, full = reply[0], reply[2]
    assert isinstance(early, Analysis) and isinstance(full, Analysis)
    assert early.lines[0].depth == EARLY_DEPTH and full.lines[0].depth == 18
    assert early.fen == full.fen


def test_behind_a_fast_game_the_position_faced_is_told_early_and_still_measured_whole() -> None:
    """Behind a fast game -- the player's move and the reply submitted together, as the
    panel's puzzle room does -- the position the player then faces is told at EARLY_DEPTH
    first, and then measured at the full depth and looked
    ahead from, as every position the player faces is. It was measured at EARLY_DEPTH alone
    and kept as the whole: never deepened, nothing looked ahead at, and the next move's
    verdict set depth 12 against depth 18 (the state review, 2026-10-01; Tai, 2026-09-28:
    "first usable answer is depth 12 ... the computation continues until max depth of 18")."""
    from sidecoach.analysis.annotator import EARLY_DEPTH

    fake = _DepthEngine({})
    annotator = GameAnnotator(engine=fake, depth=18)
    board = chess.Board()
    analyser = Analyser(annotator, look_ahead=2)
    analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
    board.push_san("e4")
    analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
    board.push_san("e5")
    analyser.start()
    try:
        got = _drain(analyser)
        assert _until(lambda: analyser.looked >= 1), "nothing looked ahead at"
    finally:
        analyser.close()
    faced = [e for e in got if isinstance(e, Analysis) and e.ply == 2]
    assert faced and faced[0].provisional and faced[0].lines[0].depth == EARLY_DEPTH
    assert not faced[-1].provisional and faced[-1].lines[0].depth == 18
    assert faced[-1].fen == board.fen()
    assert annotator._cache[board.fen()].depth == 18, "kept at the early depth"


class _YieldOnceEngine(FakeEngine):
    """A fake engine whose first search of the candidates (several lines: a whole measure,
    not a verdict's one line) runs until `interrupt` stops it; the ones after answer at once."""

    def __init__(self, answers: dict[str, list[Line]]) -> None:
        super().__init__(answers)
        self.held = False

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        several = int(kwargs.get("multipv") or 1) > 1  # type: ignore[call-overload]
        if self._interruptible and several and not self.held:
            self.held = True
            self._stopped.wait(5)
            raise Interrupted
        return super().analyse(board, **kwargs)


def test_the_position_the_player_left_gives_way_to_the_one_they_face() -> None:
    """Tai's puzzles, 2026-09-28: the site replies 0.6 s after the player's move, and the
    position they then faced waited behind the full measure of the one they had left. That
    measure gives way to the reply, whose position is told first, and is taken up again
    after it -- still owed until it is."""
    fake = _YieldOnceEngine({})
    board = chess.Board()
    with Analyser(GameAnnotator(engine=fake, depth=18, with_threats=False)) as analyser:
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
        assert _until(lambda: analyser._looking), "the full measure never started"
        board.push_san("e4")
        analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
        got = _drain(analyser)
    order = [
        (type(e).__name__, getattr(e, "ply", None), getattr(e, "provisional", False)) for e in got
    ]
    assert order == [
        ("MoveAssessment", 1, False),
        ("Analysis", 2, True),
        ("MoveAssessment", 2, False),
        ("Analysis", 2, False),
        ("Analysis", 1, False),
    ], order
    assert analyser.pending == 0


@needs_engine
def test_each_depth_is_delivered_as_the_search_finishes_it(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-28: "if the consumers read after stockfish reaches depth > 12 then they
    will get whatever lines that is available at the time". The position the player faces is
    delivered at the early depth, then at every depth the full search finishes on its way,
    provisional, then whole at the full depth -- the same position, deeper each time."""
    from sidecoach.analysis.annotator import EARLY_DEPTH

    # The analyser closes its engine as it closes; the module's Stockfish stays up for the
    # tests after this one.
    monkeypatch.setattr(engine, "close", lambda: None)
    board = chess.Board(SCHOLAR)
    annotator = GameAnnotator(engine=engine, depth=16, multipv=3, with_threats=False)
    with Analyser(annotator) as analyser:
        analyser.submit_position("g", board, 0)
        got = [e for e in _drain(analyser, timeout=30) if isinstance(e, Analysis)]
    depths = [e.lines[0].depth for e in got]
    assert [e.provisional for e in got] == [True] * (len(got) - 1) + [False]
    assert depths[0] == EARLY_DEPTH and depths[-1] == 16
    assert depths == sorted(depths) and len(set(depths)) == len(depths), depths
    assert {e.fen for e in got} == {board.fen()}


class _StoppableEngine(FakeEngine):
    """A fake engine whose searches of the positions ahead (`ahead`) run until `interrupt`
    stops them; the game's own are answered at once."""

    def __init__(self, answers: dict[str, list[Line]], ahead: set[str]) -> None:
        super().__init__(answers)
        self.ahead = ahead

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        if self._interruptible and board.fen() in self.ahead:
            self._stopped.wait(5)
            raise Interrupted
        return super().analyse(board, **kwargs)


def test_a_move_played_stops_the_look_ahead_at_once() -> None:
    """A search ahead of the game is let go the moment a move of the game's own comes: the
    move never waits for a position it did not go to."""
    fake = _StoppableEngine(
        {chess.Board().fen(): [_line("e2e4"), _line("d2d4")]}, {_after("e4"), _after("d4")}
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    with Analyser(annotator, look_ahead=2) as analyser:
        analyser.submit_position("g", chess.Board(), 0)
        assert _until(lambda: analyser._looking), "never looked ahead"
        t = time.monotonic()
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("g1f3"), 1, by_user=False)
        assert _until(lambda: not analyser._looking and analyser.pending == 0, 1.0)
        assert time.monotonic() - t < 0.5
        got = _drain(analyser)
    assert any(isinstance(e, MoveAssessment) and e.uci == "g1f3" for e in got)
    assert analyser.looked == 0 and annotator.ahead_hits == 0


@needs_engine
def test_the_engine_stops_a_search_it_is_interrupted_in(engine: Engine) -> None:
    import threading

    from sidecoach.analysis.engine import Interrupted

    raised: list[BaseException] = []

    def search() -> None:
        try:
            with engine.interruptible():
                engine.analyse(chess.Board(), depth=60, multipv=1)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    worker = threading.Thread(target=search)
    worker.start()
    assert _until(lambda: engine._search is not None), "the search never started"
    t = time.monotonic()
    engine.interrupt()
    worker.join(5)
    assert not worker.is_alive() and time.monotonic() - t < 1.0
    assert raised and isinstance(raised[0], Interrupted)
    # The engine answers as before afterwards.
    assert engine.analyse(chess.Board(), depth=6, multipv=1)


def test_what_the_engine_says_of_a_withdrawn_move_is_not_passed_on(tmp_path: Path) -> None:
    """A verdict on a move taken back came back at its ply and was read as the verdict on
    the move that replaced it; `_Assessing.drain` holds each result against the game as it
    stands now."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Side, Takeback

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
    # Nothing measured yet (the worker is not started): 3.Nf3 Nc6 taken back, 3.Bc4 played.
    board.pop()
    board.pop()
    assessing.submit(Takeback(game_id="g", plies=2, fen=board.fen()))
    assessing.submit(played(board, "Bc4", 3))
    # And a verdict on the withdrawn 3.Nf3, as if the engine had finished it just before.
    stale = MoveAssessment(
        game_id="g",
        ply=3,
        uci="g1f3",
        expected_loss=0.0,
        severity=Severity.NONE,
        best=[],
    )
    assessing.analyser._results.put(stale)
    assessing.analyser.start()
    got: list[object] = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not any(
        isinstance(e, MoveAssessment) and e.ply == 3 for e in got
    ):
        got += assessing.drain()
        time.sleep(0.02)
    assessing.analyser.close()
    verdicts = [e for e in got if isinstance(e, MoveAssessment)]
    assert [(e.ply, e.uci) for e in verdicts] == [(1, "e2e4"), (2, "e7e5"), (3, "f1c4")], verdicts
    analyses = [e for e in got if isinstance(e, Analysis)]
    assert all(e.ply <= 3 for e in analyses)


class _SlowEngine(FakeEngine):
    """A fake engine that takes a moment per search and stops when interrupted, as a real
    one does (`Engine.interrupt`)."""

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        end = time.monotonic() + 0.3
        while time.monotonic() < end:
            if self._interruptible and self._stopped.is_set():
                raise Interrupted
            time.sleep(0.005)
        return super().analyse(board, **kwargs)


def test_a_position_re_read_mid_measure_gets_no_word_on_the_old_reading(tmp_path: Path) -> None:
    """The side to move settled by the clock while the joined position is measured whole:
    the measure, given way to the re-read position's, was put off after the reset had
    cleared what was put off, measured again after the new reading and delivered last, and
    on the same game at ply 0 it passed as the new position's -- the profile filed White's
    analysis for a position with Black to move (the state review, 2026-10-01). The engine
    lets go of what it is on at a reset, and the glue passes on nothing of a position the
    game no longer stands on, nor of another game."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, Resync, Side

    white = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    black = white.replace(" w ", " b ")
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    analyser = Analyser(GameAnnotator(engine=_SlowEngine({}), depth=18, with_threats=False))
    assessing.analyser = analyser
    analyser.start()
    try:
        assessing.submit(
            GameStarted(game_id="g", user_side=Side.WHITE, start_fen=white, kind="position")
        )
        assert _until(lambda: analyser._looking, 5), "the full measure never began"
        before = assessing.drain()
        assessing.submit(Resync(game_id="g", fen=black, reason="side to move settled by the clock"))
        after = assessing.wait(timeout=10)
        assert [e.fen for e in before if isinstance(e, Analysis)] == [white]
        told = [e.fen for e in after if isinstance(e, Analysis)]
        assert told and set(told) == {black}, told
        # The glue's own check, whatever the engine sends: a word at ply 0 on the old
        # reading, and one on another game, are not passed on.
        word = {"ply": 0, "lines": [], "decisive": False, "only_move": False}
        stale = Analysis(game_id="g", fen=white, **word)
        other = Analysis(game_id="h", fen=black, **word)
        analyser._results.put(stale)
        analyser._results.put(other)
        assert assessing.drain() == []
    finally:
        assessing.close()


def test_the_move_that_replaces_a_withdrawn_one_knows_the_move_before_it() -> None:
    """After a takeback the move before the replacement is the one the game kept; the last
    move judged was the withdrawn one, a ply too far, and the replacement was judged as if
    nothing came before it: the opponent's mistake unseen, a loss after it a blunder, not
    the miss it is (the state review, 2026-10-01)."""
    start = chess.Board()
    fake = FakeEngine(
        {
            start.fen(): [Line(moves=["d2d4"], depth=6, expected_score=0.55)],
            _after("e4"): [Line(moves=["e7e5"], depth=6, expected_score=0.70)],
            _after("e4", "a6"): [Line(moves=["d2d4"], depth=6, expected_score=0.80)],
            _after("e4", "h6"): [Line(moves=["d2d4"], depth=6, expected_score=0.80)],
        }
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    board = chess.Board()
    first = annotator.assess_move("g", board, board.parse_san("e4"), 1)
    assert first.severity is Severity.MISTAKE, "the opponent erred"
    board.push_san("e4")
    withdrawn = annotator.assess_move("g", board, board.parse_san("a6"), 2)
    replacement = annotator.assess_move("g", board, board.parse_san("h6"), 2)
    assert withdrawn.label is Label.MISS
    assert replacement.label is Label.MISS, replacement.label


# Tai's chess.com puzzle of 2026-09-24: 2.exf7+ Kh8 3.Qxb8, and the rook on f8 cannot take back.
OVERLOADED = "1r3rk1/pp3ppp/2pQP3/1P1pP1q1/3Nn3/6P1/P5KP/2R2R2 w - - 1 2"
OVERLOADED_LINE = ["e6f7", "g8h8", "d6b8", "g5d2", "g2h1", "d2b4"]


def test_the_recaptures_a_line_declines_are_found_without_an_engine() -> None:
    from sidecoach.analysis.defences import declined

    board = chess.Board(OVERLOADED)
    found = declined(board, Line(moves=OVERLOADED_LINE, depth=18, expected_score=0.975))
    assert [(i, m.uci()) for i, m in found] == [(0, "f8f7"), (2, "f8b8")]
    queen = chess.Board("2kr1b1r/ppq2p2/2p1pp2/8/2B5/3P2Q1/PPPB1PP1/R4RK1 b - - 1 1")
    line = Line(moves=["c7g3", "f1e1", "g3h4"], depth=18, expected_score=0.975)
    assert [(i, m.uci()) for i, m in declined(queen, line)] == [(0, "f2g3")]
    taken_back = Line(moves=["c7g3", "f2g3", "f8c5"], depth=18, expected_score=0.94)
    assert declined(queen, taken_back) == [], "the line takes back itself"


def test_the_solution_played_out_of_order_is_the_first_temptation() -> None:
    from sidecoach.analysis.traps import tempting

    board = chess.Board(OVERLOADED)
    lines = [Line(moves=OVERLOADED_LINE, depth=18, expected_score=0.975)]
    assert tempting(board, lines)[0].uci() == "d6b8"


@needs_engine
def test_a_declined_recapture_is_played_out_by_the_engine(engine) -> None:  # noqa: ANN001
    from sidecoach.analysis.defences import defence_lines

    board = chess.Board(OVERLOADED)
    lines = [Line(moves=OVERLOADED_LINE, depth=18, expected_score=0.975)]
    defences = defence_lines(engine, board, lines, depth=12)
    rook = [d for d in defences if d.moves[:4] == ["e6f7", "g8h8", "d6b8", "f8b8"]]
    assert rook and rook[0].mate is not None and rook[0].mate > 0, defences
    assert rook[0].moves[4].startswith("f7f8"), "the pawn queens where the rook stood"


# 32.Qg4 in Lichess 06VoP: a quiet queen move whose point is Qxg7#.
QUIET_MATE = "2rr2k1/5pp1/p1qp3p/4pP1N/1p6/1P4PK/PP5P/R2Q4 w - - 5 32"


def test_a_quiet_best_move_is_asked_what_it_threatens() -> None:
    from sidecoach.analysis.threats import aim_lines

    board = chess.Board(QUIET_MATE)
    passed = board.copy(stack=False)
    passed.push_uci("d1g4")
    passed.push(chess.Move.null())
    threat = [Line(moves=["g4g7"], depth=14, expected_score=1.0, mate=1)]
    engine = FakeEngine({chess.Board(passed.fen()).fen(): threat})
    assert aim_lines(engine, board, [Line(moves=["d1g4"], depth=18, expected_score=0.92)]) == threat
    # A capture is its own point, and a line that only holds is not a win to explain.
    assert aim_lines(engine, board, [Line(moves=["h5g7"], depth=18, expected_score=0.9)]) == []
    assert aim_lines(engine, board, [Line(moves=["d1g4"], depth=18, expected_score=0.5)]) == []


@needs_engine
def test_the_engine_finds_what_a_quiet_move_threatens(engine) -> None:  # noqa: ANN001
    from sidecoach.analysis.threats import aim_lines

    board = chess.Board(QUIET_MATE)
    aims = aim_lines(engine, board, [Line(moves=["d1g4"], depth=18, expected_score=0.92)], depth=10)
    assert aims and aims[0].moves[0] == "g4g7" and aims[0].mate == 1, aims
