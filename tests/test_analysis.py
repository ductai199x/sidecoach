"""The engine wrapper, the expected-score arithmetic, and the annotator that joins them."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chess
import chess.engine
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
from tests.fakes import HANG_S, FakeEngine

STOCKFISH = find_stockfish(load_settings())


class _Poking:
    """The analyser's results queue, poking a watcher at every result put."""

    def __init__(self, inner: Any, poke: Callable[[], None]) -> None:
        self._inner, self._poke = inner, poke

    def put(self, item: object) -> None:
        self._inner.put(item)
        self._poke()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AnalyserWatch:
    """An analyser's worker, watched without a clock: every result it puts, every request it
    settles and every position it looks ahead at wakes `until`, which returns once its
    condition holds. HANG_S only bounds a hang; it is never the expected path. Made before the
    worker starts. (Shared: tests/test_takebacks.py.)"""

    def __init__(self, analyser: Analyser) -> None:
        self.analyser = analyser
        self._cond = threading.Condition()
        settle, look_ahead = analyser._settle, analyser._look_ahead

        def settled() -> None:
            settle()
            self.poke()

        def looked(*args: Any) -> None:
            try:
                look_ahead(*args)
            finally:
                self.poke()

        analyser._settle = settled  # type: ignore[method-assign]
        analyser._look_ahead = looked  # type: ignore[method-assign]
        analyser._results = _Poking(analyser._results, self.poke)  # type: ignore[assignment]

    def poke(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def until(self, condition: Callable[[], bool], what: str = "the condition") -> None:
        with self._cond:
            assert self._cond.wait_for(condition, HANG_S), f"never came: {what}"

    def idle(self) -> None:
        """Every request answered. A result is put before its request settles, so none can
        come after `pending` is 0 (`Analyser._outstanding`); what was said stays queued for
        whoever drains it (the engine glue's `drain`)."""
        self.until(lambda: self.analyser.pending == 0, "the analyser settled")

    def settled(self) -> list[object]:
        """Every request answered (`idle`), and what the worker said, polled."""
        self.idle()
        return list(self.analyser.poll())


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
    # A takeback of 3.d4: the next move is played from the position before it.
    board.pop()
    assessing.submit(Takeback(game_id="g", plies=1, fen=board.fen()))
    play(board, "d3", 7)

    labels: dict[int, object] = {}
    openings: dict[int, object] = {}
    # Ply 7 can come twice, d4 (measured before the takeback was told) then d3: everything
    # is waited for, in the order it was said, so d3's own word is the last at ply 7.
    for event in watch.settled():
        if isinstance(event, MoveAssessment):
            labels[event.ply] = event.label
        elif isinstance(event, Analysis):
            openings[event.ply] = event.opening
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
    decisive: bool = False,
    sacrifice: bool = False,
    before_score: float = 0.5,
    second_score: float | None = None,
    opponent_blundered: bool = False,
    chance: bool = False,
    mate_missed: bool = False,
    mate_allowed: bool = False,
    obvious: bool = False,
) -> Label:
    return label_move(
        loss,
        Thresholds(),
        is_best=is_best,
        in_book=in_book,
        decisive=decisive,
        sacrifice=sacrifice,
        before_score=before_score,
        second_score=second_score,
        opponent_blundered=opponent_blundered,
        chance=chance,
        mate_missed=mate_missed,
        mate_allowed=mate_allowed,
        obvious=obvious,
    )


def test_labels_follow_chess_coms_definitions() -> None:
    """Chess.com's review, as its help pages define each word (2026-10-01): best, "the
    engine's top choice"; excellent, "almost as good as the best move"; good, "a decent move
    but not the best"; book, "a conventional opening move"; inaccuracy, mistake and blunder
    by what the move lost; great, "a move that altered the course of the game"; brilliant,
    "the best move -- and tricky to find too", a sound sacrifice; miss, "a move that missed
    a tactical opportunity or a chance to punish the opponent"."""
    assert _label(0.0, is_best=True) is Label.BEST
    assert _label(0.01) is Label.EXCELLENT
    assert _label(0.0) is Label.EXCELLENT, "as good as the top choice is not the top choice"
    assert _label(0.03) is Label.GOOD
    assert _label(0.06) is Label.INACCURACY
    assert _label(0.12) is Label.MISTAKE
    assert _label(0.30) is Label.BLUNDER
    # Book: the tables' move is never flagged for a little ground; a move that loses a
    # tenth of the game is no conventional move, whatever a table holds (1. e4 f5).
    assert _label(0.0, in_book=True) is Label.BOOK
    assert _label(0.06, in_book=True) is Label.BOOK
    assert _label(0.14, in_book=True) is Label.MISTAKE


def test_great_is_the_move_that_altered_the_course_of_the_game() -> None:
    # The only good move: the best, every other move losing ground (a decisive moment).
    assert _label(0.0, is_best=True, decisive=True) is Label.GREAT
    assert _label(0.0, is_best=True, decisive=True, before_score=0.95) is Label.GREAT
    # The opponent's blunder cashed in with the best move.
    assert _label(0.0, is_best=True, opponent_blundered=True) is Label.GREAT
    # Not merely the best move after any error, nor a move that is not the best.
    assert _label(0.0, is_best=True) is Label.BEST
    assert _label(0.0, decisive=True) is Label.EXCELLENT
    # Never an obvious move (a recapture, a free capture, the only legal move), nor in a
    # lost position, where the best move changes nothing.
    assert _label(0.0, is_best=True, decisive=True, obvious=True) is Label.BEST
    assert _label(0.0, is_best=True, opponent_blundered=True, obvious=True) is Label.BEST
    assert _label(0.0, is_best=True, decisive=True, before_score=0.02) is Label.BEST


def test_brilliant_is_a_sound_sacrifice_that_was_needed() -> None:
    assert _label(0.0, is_best=True, sacrifice=True) is Label.BRILLIANT
    assert _label(0.01, sacrifice=True) is Label.BRILLIANT, "as good as the best"
    # Not a sacrifice that gives ground: chess.com's brilliant is the best move.
    assert _label(0.03, sacrifice=True) is Label.GOOD
    assert _label(0.3, sacrifice=True) is Label.BLUNDER
    # Not when the game was won without it -- read from the second-best line, since the
    # best is the one the sacrifice makes. The only win is brilliant however won it leaves.
    assert _label(0.0, is_best=True, sacrifice=True, before_score=0.97, second_score=0.95) is (
        Label.BEST
    )
    assert _label(0.0, is_best=True, sacrifice=True, before_score=1.0, second_score=0.3) is (
        Label.BRILLIANT
    )
    # With no second line known, the position before is what is known.
    assert _label(0.0, is_best=True, sacrifice=True, before_score=0.97) is Label.BEST
    # Nor in a lost position.
    assert _label(0.0, is_best=True, sacrifice=True, before_score=0.05) is Label.BEST


def test_a_miss_is_a_chance_to_punish_let_go() -> None:
    # The opponent's move before was an error, and the move played let what it gave go: a
    # miss, whatever the size of the loss, from an inaccuracy to a blunder.
    assert _label(0.06, chance=True) is Label.MISS
    assert _label(0.15, chance=True) is Label.MISS
    assert _label(0.90, chance=True) is Label.MISS
    # A small slip with the chance still there is no miss; nor a loss with no chance.
    assert _label(0.01, chance=True) is Label.EXCELLENT
    assert _label(0.15) is Label.MISTAKE
    # A forced mate let go is a miss even when the move played still wins easily.
    assert _label(0.02, chance=True, mate_missed=True) is Label.MISS
    # A move that walks into a forced mate loses the game: a blunder whatever the curve
    # says it cost -- unless the game was lost already, or every move allows it.
    assert _label(0.02, mate_allowed=True, before_score=0.3) is Label.BLUNDER
    assert _label(0.02, mate_allowed=True, before_score=0.03) is Label.GOOD
    assert _label(0.0, mate_allowed=True, before_score=0.3, is_best=True) is Label.BEST


def _judged(before: chess.Board, move: str, before_line: Line, after_line: Line) -> MoveAssessment:
    after = before.copy()
    after.push_uci(move)
    fake = FakeEngine({before.fen(): [before_line], after.fen(): [after_line]})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    return annotator.assess_move("g", before, chess.Move.from_uci(move), 1)


def _answered(fen: str, theirs: str, mine: str, scores: tuple[Line, Line, Line]) -> MoveAssessment:
    """The opponent's move `theirs` from `fen`, then the player's `mine`, judged in order by one
    annotator: the lines of the three positions are `scores`."""
    first = chess.Board(fen)
    second = first.copy()
    second.push_uci(theirs)
    third = second.copy()
    third.push_uci(mine)
    fake = FakeEngine(
        {b.fen(): [line] for b, line in zip((first, second, third), scores, strict=True)}
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    annotator.assess_move("g", first, chess.Move.from_uci(theirs), 1)
    return annotator.assess_move("g", second, chess.Move.from_uci(mine), 2)


def test_a_miss_is_failing_to_punish_the_opponents_mistake() -> None:
    """Chess.com's miss, Tai's reading (2026-10-02): the opponent's move before was an error
    and the reply let what it gave go -- whatever the best line would have won. A loss after
    a sound move is labelled by its size, though the best line wins a piece: no "a tactic is
    two pawns within eight plies" rule."""
    # Pawn ending: black's ...e6 throws away an eighth of the game, white's best is the quiet
    # Kd2, and Kf1 gives it back. No material in the line: a miss all the same.
    pawns = "4k3/4p3/8/8/8/8/4P3/4K3 b - - 0 1"
    erred = (
        Line(moves=["e8d7"], depth=6, expected_score=0.47),
        Line(moves=["e1d2"], depth=6, expected_score=0.65),
        Line(moves=["e8d7"], depth=6, expected_score=0.47),
    )
    assert _answered(pawns, "e7e6", "e1f1", erred).label is Label.MISS
    # Black's move costs nothing (the knight was lost anyway); white's best takes the knight,
    # and Ke2 lets it go: a mistake by its size, not a miss.
    knight = "4k3/8/8/3n4/8/8/8/3QK3 b - - 0 1"
    sound = (
        Line(moves=["e8f8"], depth=6, expected_score=0.35),
        Line(moves=["d1d5", "f8e7"], depth=6, expected_score=0.65),
        Line(moves=["d5f4"], depth=6, expected_score=0.5),
    )
    assert _answered(knight, "e8f8", "e1e2", sound).label is Label.MISTAKE


def test_a_forced_mate_let_go_and_one_walked_into() -> None:
    """A mate is 1.0 on the curve and a thousand centipawns 0.975: a mate in four let go for
    a won ending read as an inaccuracy at most, and a move into a forced mate in a lost
    position as good (the engine review, 2026-10-01). Let go after the opponent's error it is
    a miss; with no error before it, it is no chance to punish (Tai, 2026-10-02) and is
    labelled by its loss, flagged however small."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    mating = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=1)
    still_won = Line(moves=["g8f8"], depth=6, expected_score=0.03, score_cp=-1000)
    let_go = _judged(board, "h2h3", mating, still_won)
    assert let_go.label is Label.INACCURACY, let_go
    # Black's Kg8 walks into the mate (a blunder), and h3 lets it go.
    walked_in = (
        Line(moves=["f8e8"], depth=6, expected_score=0.4),
        mating,
        still_won,
    )
    missed = _answered("5k2/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1", "f8g8", "h2h3", walked_in)
    assert missed.label is Label.MISS, missed
    # The other side: worse but holding, and the move lets a mate in.
    lost = Line(moves=["g8f8"], depth=6, expected_score=0.2, score_cp=-400)
    mates = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=1)
    black = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
    walked = _judged(black, "h7h6", lost, mates)
    assert walked.label is Label.BLUNDER, walked


def test_a_search_under_way_when_the_settings_change_is_not_kept() -> None:
    """The settings' save forgets what was measured (`forget_measured`) from another thread,
    and the search under way then finishes the old way. Kept, it was the depth-12 position a
    depth-18 one is set against -- the mix the forgetting is for (the facts audit,
    2026-10-02). It is handed to its caller, and searched again next time."""

    class ChangedMidSearch(FakeEngine):
        annotator: GameAnnotator | None = None

        def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
            lines = super().analyse(board, **kwargs)
            if self.annotator is not None and len(self.calls) == 1:
                self.annotator.forget_measured()  # the save, while this search ran
            return lines

    for measure in ("measure", "measure_for_verdict"):
        fake = ChangedMidSearch({})
        annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
        fake.annotator = annotator
        board = chess.Board()
        first = getattr(annotator, measure)(board)
        assert first.lines, "the caller still gets what it asked for"
        assert not annotator.measured(board), measure
        assert annotator.computed.lines(board) == [], measure
        getattr(annotator, measure)(board)
        assert len(fake.calls) == 2, f"{measure}: searched again the new way"
        getattr(annotator, measure)(board)
        assert len(fake.calls) == 2, f"{measure}: and that one is kept"


def test_the_best_move_of_a_mating_line_is_never_a_mate_let_go() -> None:
    """The line before shows a long mate, the engine's own first move is played, and the
    search after it does not report the mate (past its horizon, or cut by the time cap): the
    best move let nothing go. It was labelled an inaccuracy -- `mate_missed` flagged it, and
    a loss of nothing fell through to the lowest flag (the facts audit, 2026-10-02)."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    long_mate = Line(moves=["h2h3", "g8f8"], depth=6, expected_score=1.0, mate=9)
    seen_as_won = Line(moves=["g8f8"], depth=6, expected_score=0.03, score_cp=-1000)
    best = _judged(board, "h2h3", long_mate, seen_as_won)
    assert best.expected_loss == 0.0
    assert best.label is Label.BEST, best


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


def test_no_single_position_may_hold_up_a_game() -> None:
    """A knight endgame that takes minutes to reach depth 14 with four lines is cut short:
    every search asked for a depth carries the time cap too, and a search the cap ended
    reports the depth it reached."""
    uci = _ScriptedUci([*_D13], [*_D13])
    running = _scripted_engine(uci)
    running.max_seconds = 0.1
    lines = running.analyse(chess.Board(), depth=40, multipv=4)
    assert uci.limits[-1].depth == 40 and uci.limits[-1].time == 0.1
    assert lines[0].depth == 13, "the depth asked for was not reached, and that is reported"
    running.max_seconds = None
    running.analyse(chess.Board(), depth=40, multipv=4)
    assert uci.limits[-1].time is None, "no cap, no time limit"
    running.analyse(chess.Board(), depth=40, time_s=0.05)
    assert uci.limits[-1].time == 0.05


def _info(rank: int, depth: int, move: str, cp: int) -> dict[str, object]:
    return {
        "multipv": rank,
        "depth": depth,
        "pv": [chess.Move.from_uci(move)],
        "score": chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE),
    }


class _ScriptedSearch:
    """python-chess's analysis, scripted: the infos it reports, its lines when it ends, and,
    with `hold`, a search that waits on the test before it ends."""

    def __init__(
        self,
        infos: list[dict[str, object]],
        multipv: list[dict[str, object]],
        started: threading.Event,
        hold: threading.Event | None,
    ) -> None:
        self.infos, self.multipv = infos, multipv
        self.started, self.hold = started, hold

    def __enter__(self) -> _ScriptedSearch:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self):  # noqa: ANN204
        yield from self.infos
        self.started.set()
        if self.hold is not None:
            assert self.hold.wait(HANG_S), "the scripted search was never let go"

    stopped = False

    def stop(self) -> None:
        self.stopped = True
        if self.hold is not None:
            self.hold.set()


class _ScriptedUci:
    """A stand-in for python-chess's engine: each search scripted, every option set kept."""

    def __init__(self, infos: list[dict[str, object]], multipv: list[dict[str, object]]) -> None:
        self.options = {"Threads": None, "Hash": None}
        self.configured: list[dict[str, object]] = []
        self.infos, self.multipv = infos, multipv
        self.started = threading.Event()
        self.hold: threading.Event | None = None
        self.limits: list[chess.engine.Limit] = []
        self.search: _ScriptedSearch | None = None

    def configure(self, options: dict[str, object]) -> None:
        self.configured.append(dict(options))

    def analysis(
        self, board: chess.Board, limit: chess.engine.Limit, multipv: int
    ) -> _ScriptedSearch:
        self.limits.append(limit)
        self.search = _ScriptedSearch(self.infos, self.multipv, self.started, self.hold)
        return self.search

    def quit(self) -> None:
        return None


# Four lines at depth 13, then the first two of depth 14 before the time cap cut it.
_D13 = [
    _info(1, 13, "e2e4", 40),
    _info(2, 13, "d2d4", 30),
    _info(3, 13, "g1f3", 20),
    _info(4, 13, "c2c4", 10),
]
_D14 = [_info(1, 14, "d2d4", 45), _info(2, 14, "e2e4", 35)]


def _scripted_engine(uci: _ScriptedUci) -> Engine:
    running = Engine(path=Path("scripted"), multipv=4)
    running._engine = uci  # type: ignore[assignment]  # open() keeps a running engine
    return running


def test_a_search_cut_short_gives_lines_of_one_depth() -> None:
    """Cut by the time cap mid-depth, Stockfish's first lines are a depth further than the
    rest ([14, 14, 13, 13]), and the gap between them (the only-move and decisive-moment
    calls) set two depths against each other. The lines are the last depth finished whole."""
    uci = _ScriptedUci([*_D13, *_D14], [*_D14, *_D13[2:]])
    lines = _scripted_engine(uci).analyse(chess.Board(), depth=40, multipv=4)
    assert [line.depth for line in lines] == [13, 13, 13, 13]
    assert [line.moves[0] for line in lines] == ["e2e4", "d2d4", "g1f3", "c2c4"]
    # A search that finished its last depth whole is answered with it, as before.
    whole = _ScriptedUci([*_D13], [*_D13])
    assert [line.depth for line in _scripted_engine(whole).analyse(chess.Board(), depth=13)] == [
        13,
        13,
        13,
        13,
    ]


def test_settings_saved_mid_search_wait_for_the_search_to_end() -> None:
    """Threads or hash saved while a search runs used to reach Stockfish at once: the
    `setoption` stopped the search, and the cut one was kept as if whole (0.6 s and depth 14
    of a 4 s search, the engine review 2026-10-01). They now wait for the search to end, and
    the save does not wait for it."""
    uci = _ScriptedUci([*_D13], [*_D13])
    uci.hold = threading.Event()
    running = _scripted_engine(uci)
    worker = threading.Thread(target=lambda: running.analyse(chess.Board(), depth=13))
    worker.start()
    assert uci.started.wait(HANG_S)  # the search is under way, holding the engine
    running.reconfigure(threads=1, hash_mb=32)  # returns: the save does not wait
    assert uci.configured == [], "set during the search"
    assert running.threads == 1 and running.hash_mb == 32
    uci.hold.set()
    worker.join()
    assert uci.configured == [], "set as the search ended, with nothing to search"
    uci.hold = None
    running.analyse(chess.Board(), depth=13)
    assert uci.configured == [{"Threads": 1, "Hash": 32}], "set before the next search"
    # With no search running they are set at once.
    running.reconfigure(threads=2)
    assert uci.configured[-1] == {"Threads": 2, "Hash": 32}


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
    waited on it forever. A search the watchdog finds still running past its patience has its
    process ended, and is made again on a new one. The watchdog's wait is driven here: its
    patience runs out at once on the silent engine, and never on the real one."""
    import sys

    import chess.engine

    assert STOCKFISH is not None
    stub = tmp_path / "silent.py"
    stub.write_text(_SILENT_ENGINE, encoding="utf-8")
    waits: list[float] = []

    def wait(done: threading.Event, seconds: float) -> bool:
        waits.append(seconds)
        return False if len(waits) == 1 else done.wait()

    running = Engine(path=STOCKFISH, threads=1, hash_mb=16, multipv=2, max_seconds=5.0)
    silent = chess.engine.SimpleEngine.popen_uci([sys.executable, str(stub)])
    running._engine = silent
    running.watchdog_wait = wait
    try:
        lines = running.analyse(chess.Board(), depth=6)
        assert lines, "searched again on a new engine"
        assert running._engine is not silent, "the silent engine was replaced"
        assert waits[0] == 5.0 + running.watchdog_s, "patience: the time limit and the grace"
    finally:
        running.close()


def test_an_illegal_move_is_refused_rather_than_measured() -> None:
    annotator = GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False)
    with pytest.raises(ValueError, match="not legal"):
        annotator.assess("g", chess.Board(), chess.Move.from_uci("e2e5"), 1)


def test_a_failing_analysis_does_not_end_the_session() -> None:
    class Broken(FakeEngine):
        def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
            raise RuntimeError("engine gone")

    analyser = Analyser(GameAnnotator(engine=Broken({}), depth=6, with_threats=False))
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        # The failure is counted before the request settles: settled, it is in.
        assert watch.settled() == []
        assert analyser.failures == 1
        assert analyser.last_error is not None and "engine gone" in analyser.last_error
        # The worker is still there: the next request is served (and fails alike).
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("d2d4"), 1)
        assert watch.settled() == []
    assert analyser.failures == 2
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


def test_behind_a_fast_game_the_analyser_keeps_the_verdicts_and_drops_stale_analyses() -> None:
    """Moves arriving faster than the engine measures them: every ply still gets its
    verdict (the best line at the full depth, both positions), and the whole analysis --
    candidates, threats -- is run only for the position the player faces (2026-09-22: a
    queen ending at two seconds a move ran forty plies behind, every word on a move
    'not spoken: board moved on', and the review waited on the backlog)."""
    fake = FakeEngine({})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    analyser = Analyser(annotator)
    watch = AnalyserWatch(analyser)
    board = chess.Board()
    # Five plies queued before the worker starts: it is behind from the first one.
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bb5"], start=1):
        move = board.parse_san(san)
        analyser.submit_move("g", board, move, ply)
        board.push(move)
    analyser.start()
    # Everything owed answered: nothing more can come after this (not merely five analyses
    # seen, then a moment's grace for anything extra).
    got = watch.settled()
    analyser.close()
    assert analyser.poll() == [], "nothing said after the work was settled"
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
        self.searching = threading.Event()
        self.go = threading.Event()

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        self.searching.set()
        assert self.go.wait(HANG_S), "the test never let the search go"
        return super().analyse(board, **kwargs)


def test_a_takeback_lets_the_engine_go_of_the_withdrawn_moves() -> None:
    """Tai, 2026-09-23: "takebacks make the program lag behind by a lot". The moves taken
    back were measured one after another before the replacement was: now what is queued for
    them is dropped, and the one being measured delivers nothing and is not followed by the
    whole analysis of the position after it."""
    engine = _HeldEngine()
    analyser = Analyser(GameAnnotator(engine=engine, depth=6, with_threats=False))
    watch = AnalyserWatch(analyser)
    board = chess.Board()
    moves = []
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1):
        move = board.parse_san(san)
        moves.append((board.copy(), move, ply))
        board.push(move)
    analyser.submit_move("g", *moves[0])
    analyser.start()
    assert engine.searching.wait(HANG_S), "the worker never started on ply 1"
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
    got = watch.settled()
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
    analyser = Analyser(annotator, look_ahead=2)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", start, 0)
        watch.until(lambda: analyser.looked == 4, "four positions looked at ahead")
        searched = set(fake.calls)
        assert {_after("e4"), _after("e4", "e5"), _after("d4"), _after("d4", "d5")} <= searched
        before = len(fake.calls)
        analyser.submit_move("g", start, chess.Move.from_uci("e2e4"), 1, by_user=True)
        got = watch.settled()
        assert _after("e4") not in fake.calls[before:], "the position after e4 was searched again"
        assert annotator.ahead_hits == 1
        after = chess.Board()
        after.push_san("e4")
        analyser.submit_move("g", after, chess.Move.from_uci("e7e5"), 2, by_user=False)
        reply = watch.settled()
    assert _kinds(got) == ["Analysis(early)", "Analysis", "MoveAssessment", "Analysis"]
    assert _kinds(reply) == ["MoveAssessment", "Analysis"]
    assert annotator.ahead_hits == 2


def test_a_position_looked_at_ahead_carries_the_moves_that_lead_there() -> None:
    """The engine reads a repetition from the moves before a position, so a position looked
    at ahead is searched with them, as the game's own are: measured without them it was
    cached, and the game arriving there judged its moves on a search that could not see the
    repetition (the engine review, 2026-10-01). The look ahead is driven here a step at a
    time, as the worker drives it."""
    board = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1"):
        board.push_san(san)
    fake = FakeEngine({board.fen(): [_line("f6g8")]}, default=[_line("g1f3")])
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    analyser = Analyser(annotator, look_ahead=1)
    analyser._plan_ahead(board, [_line("f6g8")])
    while analyser._ahead:
        analyser._look_ahead(*analyser._ahead.pop(0), analyser._ahead)
    assert analyser.looked == 2
    ahead = [
        h
        for fen, h in zip(fake.calls, fake.histories, strict=True)
        if fen == _after("Nf3", "Nf6", "Ng1", "Ng8")
    ]
    assert ahead == [["g1f3", "g8f6", "f3g1", "f6g8"]], fake.histories


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
    analyser = Analyser(GameAnnotator(engine=fake, depth=18, with_threats=False))
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", board, 0)
        started = watch.settled()
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
        own = watch.settled()
        board.push_san("e4")
        analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
        reply = watch.settled()
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
    watch = AnalyserWatch(analyser)
    analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
    board.push_san("e4")
    analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
    board.push_san("e5")
    analyser.start()
    try:
        got = watch.settled()
        watch.until(lambda: analyser.looked >= 1, "a position looked at ahead")
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
        self.holding = threading.Event()  # the held search is under way

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        several = int(kwargs.get("multipv") or 1) > 1  # type: ignore[call-overload]
        if self._interruptible and several and not self.held:
            self.held = True
            self.holding.set()
            assert self._stopped.wait(HANG_S), "the full measure was never interrupted"
            raise Interrupted
        return super().analyse(board, **kwargs)


def test_the_position_the_player_left_gives_way_to_the_one_they_face() -> None:
    """Tai's puzzles, 2026-09-28: the site replies 0.6 s after the player's move, and the
    position they then faced waited behind the full measure of the one they had left. That
    measure gives way to the reply, whose position is told first, and is taken up again
    after it -- still owed until it is."""
    fake = _YieldOnceEngine({})
    board = chess.Board()
    analyser = Analyser(GameAnnotator(engine=fake, depth=18, with_threats=False))
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
        assert fake.holding.wait(HANG_S), "the full measure never started"
        assert analyser._looking, "a full measure a request interrupts"
        board.push_san("e4")
        analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
        got = watch.settled()
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


class _HoldTheResumed(_YieldOnceEngine):
    """The give-way engine above, whose search of `fen` taken up again (its second whole
    measure) waits at `gate` -- an interrupt does not cut it short -- and then answers."""

    def __init__(self, fen: str) -> None:
        super().__init__({})
        self.fen = fen
        self.measures = 0
        self.resumed = threading.Event()  # the measure taken up again is under way
        self.gate = threading.Event()

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        several = int(kwargs.get("multipv") or 1) > 1  # type: ignore[call-overload]
        if several and board.fen() == self.fen:
            self.measures += 1
            if self.measures == 2:
                self.resumed.set()
                assert self.gate.wait(HANG_S), "the resumed measure was never let go"
                return [Line(moves=["e7e5"], depth=18, expected_score=0.5)]
        return super().analyse(board, **kwargs)


def test_a_measure_taken_up_again_and_then_taken_back_says_nothing() -> None:
    """The position the player left gave way to the one they faced and was taken up again
    (above); then both moves are taken back while it is measured. Its search ends with lines
    all the same, and they are dropped: nothing about a withdrawn move is passed on, and the
    request is settled. (The takeback tests met this only when the timing fell so.)"""
    board = chess.Board()
    board.push_san("e4")
    fake = _HoldTheResumed(board.fen())
    board = chess.Board()
    analyser = Analyser(GameAnnotator(engine=fake, depth=18, with_threats=False))
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_move("g", board, chess.Move.from_uci("e2e4"), 1, by_user=True)
        assert fake.holding.wait(HANG_S), "the full measure never started"
        board.push_san("e4")
        analyser.submit_move("g", board, chess.Move.from_uci("e7e5"), 2, by_user=False)
        assert fake.resumed.wait(HANG_S), "the measure was never taken up again"
        before = [
            (type(e).__name__, getattr(e, "ply", None), getattr(e, "provisional", False))
            for e in analyser.poll()
        ]
        analyser.withdraw("g", 0)  # both moves taken back, the resumed measure under way
        fake.gate.set()
        after = watch.settled()
    assert before == [
        ("MoveAssessment", 1, False),
        ("Analysis", 2, True),
        ("MoveAssessment", 2, False),
        ("Analysis", 2, False),
    ], before
    assert after == [], f"a withdrawn move's measure passed on: {after}"
    assert analyser.pending == 0 and fake.measures == 2


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
    analyser = Analyser(annotator)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", board, 0)
        got = [e for e in watch.settled() if isinstance(e, Analysis)]
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
        self.entered = threading.Event()

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        if self._interruptible and board.fen() in self.ahead:
            self.entered.set()
            # Until `interrupt`: a look ahead not let go never ends (HANG_S bounds a hang).
            assert self._stopped.wait(HANG_S), "the look ahead was never let go"
            raise Interrupted
        return super().analyse(board, **kwargs)


def test_a_move_played_stops_the_look_ahead_at_once() -> None:
    """A search ahead of the game is let go the moment a move of the game's own comes: the
    move never waits for a position it did not go to. The fake's search ahead ends only when
    it is interrupted, so a move that did not interrupt it would never be judged."""
    fake = _StoppableEngine(
        {chess.Board().fen(): [_line("e2e4"), _line("d2d4")]}, {_after("e4"), _after("d4")}
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    analyser = Analyser(annotator, look_ahead=2)
    analyser.start()
    analyser.submit_position("g", chess.Board(), 0)
    assert fake.entered.wait(HANG_S), "the worker never searched ahead"
    analyser.submit_move("g", chess.Board(), chess.Move.from_uci("g1f3"), 1, by_user=False)
    judged = threading.Event()
    got: list[object] = []
    while not judged.is_set():
        event = analyser._results.get(timeout=HANG_S)  # waits for the worker's next word
        got.append(event)
        if isinstance(event, MoveAssessment):
            judged.set()
    analyser.close()
    assert any(isinstance(e, MoveAssessment) and e.uci == "g1f3" for e in got)
    assert analyser.looked == 0 and annotator.ahead_hits == 0


def test_the_engine_stops_a_search_it_is_interrupted_in() -> None:
    """`interrupt` stops the search under way inside `interruptible`, which then raises
    `Interrupted`. The scripted search runs until it is stopped: one not stopped never ends."""
    from sidecoach.analysis.engine import Interrupted

    uci = _ScriptedUci([*_D13], [*_D13])
    uci.hold = threading.Event()
    running = _scripted_engine(uci)
    raised: list[BaseException] = []

    def search() -> None:
        try:
            with running.interruptible():
                running.analyse(chess.Board(), depth=60)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    worker = threading.Thread(target=search)
    worker.start()
    assert uci.started.wait(HANG_S)
    running.interrupt()
    worker.join(HANG_S)
    assert not worker.is_alive()
    assert uci.search is not None and uci.search.stopped
    assert raised and isinstance(raised[0], Interrupted)
    # The engine answers as before afterwards.
    uci.hold = None
    assert running.analyse(chess.Board(), depth=13)


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
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()
    # Every request answered (the stale verdict first in the queue, put before them all);
    # then the glue holds what came against the game as it stands.
    watch.idle()
    got: list[object] = list(assessing.drain())
    assessing.analyser.close()
    verdicts = [e for e in got if isinstance(e, MoveAssessment)]
    assert [(e.ply, e.uci) for e in verdicts] == [(1, "e2e4"), (2, "e7e5"), (3, "f1c4")], verdicts
    analyses = [e for e in got if isinstance(e, Analysis)]
    assert all(e.ply <= 3 for e in analyses)


class _WholeMeasureHeld(FakeEngine):
    """A fake engine that answers at once, except the first whole measure (the full `depth`)
    of `fen`: that search runs until it is interrupted, as a real one stops
    (`Engine.interrupt`), so the test acts while it is under way, never on a clock."""

    def __init__(self, fen: str, depth: int) -> None:
        super().__init__({})
        self.fen, self.depth = fen, depth
        self.holding = threading.Event()  # the held search is under way
        self.interrupted = threading.Event()  # and was stopped

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
        from sidecoach.analysis.engine import Interrupted

        whole = kwargs.get("depth") == self.depth
        if self._interruptible and whole and board.fen() == self.fen and not self.holding.is_set():
            self.holding.set()
            assert self._stopped.wait(HANG_S), "the whole measure was never let go"
            self.interrupted.set()
            raise Interrupted
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
    engine = _WholeMeasureHeld(white, 18)
    analyser = Analyser(GameAnnotator(engine=engine, depth=18, with_threats=False))
    assessing.analyser = analyser
    watch = AnalyserWatch(analyser)
    analyser.start()
    try:
        assessing.submit(
            GameStarted(game_id="g", user_side=Side.WHITE, start_fen=white, kind="position")
        )
        assert engine.holding.wait(HANG_S), "the full measure never began"
        assert analyser._looking, "the whole measure, under way"
        before = assessing.drain()
        assessing.submit(Resync(game_id="g", fen=black, reason="side to move settled by the clock"))
        assert engine.interrupted.wait(HANG_S), "the reset let go of the measure under way"
        watch.idle()
        after = assessing.drain()
        assert [e.fen for e in before if isinstance(e, Analysis)] == [white]
        # The old reading's whole measure was never taken up again after the reset: its only
        # search that ran to an answer is the early one.
        from sidecoach.analysis.annotator import EARLY_DEPTH

        white_depths = [d for f, d in zip(engine.calls, engine.depths, strict=True) if f == white]
        assert white_depths == [EARLY_DEPTH], white_depths
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
    nothing came before it: the opponent's blunder unseen, the best reply to it merely
    best, not the great move it is (the state review, 2026-10-01)."""
    start = chess.Board()
    fake = FakeEngine(
        {
            start.fen(): [Line(moves=["d2d4"], depth=6, expected_score=0.55)],
            _after("e4"): [Line(moves=["e7e5"], depth=6, expected_score=0.90)],
            _after("e4", "a6"): [Line(moves=["d2d4"], depth=6, expected_score=0.80)],
        }
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    board = chess.Board()
    first = annotator.assess_move("g", board, board.parse_san("e4"), 1)
    assert first.severity is Severity.BLUNDER, "the opponent blundered"
    board.push_san("e4")
    withdrawn = annotator.assess_move("g", board, board.parse_san("a6"), 2)
    replacement = annotator.assess_move("g", board, board.parse_san("e5"), 2)
    # The withdrawn a6 let the opponent's blunder go: a miss (Tai, 2026-10-02).
    assert withdrawn.label is Label.MISS
    assert replacement.label is Label.GREAT, replacement.label


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
