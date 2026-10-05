"""The engine wrapper, the expected-score arithmetic, and the annotator that joins them."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pytest

from sidecoach.analysis.annotator import GameAnnotator, _mate_in
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
    standing,
    terminal_expected_score,
)
from sidecoach.config import CoachSettings, find_stockfish, load_settings
from sidecoach.events import Analysis, Label, Line, MotifTag, MoveAssessment, Severity
from tests.coaching import SCHOLAR
from tests.fakes import HANG_S, FakeEngine, scripted

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
    mate_let_go: bool = False,
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
        mate_let_go=mate_let_go,
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


def test_a_miss_is_a_winning_chance_let_go() -> None:
    # The position offered a winning chance and the move played gave ground: a miss,
    # whatever the size of the loss, from an inaccuracy to a blunder.
    assert _label(0.06, chance=True) is Label.MISS
    assert _label(0.15, chance=True) is Label.MISS
    assert _label(0.90, chance=True) is Label.MISS
    # A small slip with the chance still there is no miss; nor a loss with no chance.
    assert _label(0.01, chance=True) is Label.EXCELLENT
    assert _label(0.15) is Label.MISTAKE
    # A chance with no ground given is labelled by the loss, never an inaccuracy.
    assert _label(0.025, chance=True) is Label.GOOD
    # A forced mate let go is a miss whatever it cost (Tai, 2026-10-02 12:16); the engine's
    # own move never is.
    assert _label(0.0, mate_let_go=True) is Label.MISS
    assert _label(0.025, mate_let_go=True) is Label.MISS
    assert _label(0.0, mate_let_go=True, is_best=True) is Label.BEST
    # A move that walks into a forced mate loses the game: a blunder whatever the curve
    # says it cost -- unless the game was lost already, or every move allows it.
    assert _label(0.02, mate_allowed=True, before_score=0.3) is Label.BLUNDER
    assert _label(0.02, mate_allowed=True, before_score=0.03) is Label.GOOD
    assert _label(0.0, mate_allowed=True, before_score=0.3, is_best=True) is Label.BEST


def test_the_standing_bands_are_the_coachs() -> None:
    """The bands a winning chance is read in are the ones the coach says how a side stands
    in (`facts.standing_words`): losing, worse, about equal, better, winning."""
    from sidecoach.coach.facts import standing_words

    for score, band, words in (
        (0.1, 0, "losing"),
        (0.2, 0, "losing"),
        (0.3, 1, "worse"),
        (0.38, 1, "worse"),
        (0.5, 2, "about equal"),
        (0.62, 3, "better"),
        (0.79, 3, "better"),
        (0.8, 4, "winning"),
    ):
        assert standing(score) == band, score
        assert words in standing_words(score), score


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


def test_a_miss_is_a_winning_chance_let_go_wherever_it_came_from() -> None:
    """Chess.com's miss, Tai's reading (2026-10-02): the move let go a winning chance the
    position offered -- the opponent's error or a tactic already on the board -- read off the
    engine's lines: the best line stood better or winning (0.62 up, `standing` 3), and the
    move played leaves a lower band. No "two pawns within eight plies" rule, and an
    opponent's error that left the player no better is no chance."""
    # Pawn ending: black's ...e6 throws away an eighth of the game, white's best, the quiet
    # Kd2, stands better (0.65), and Kf1 gives it back to equal. No material in the line.
    pawns = "4k3/4p3/8/8/8/8/4P3/4K3 b - - 0 1"
    erred = (
        Line(moves=["e8d7"], depth=6, expected_score=0.47),
        Line(moves=["e1d2"], depth=6, expected_score=0.65),
        Line(moves=["e8d7"], depth=6, expected_score=0.47),
    )
    assert _answered(pawns, "e7e6", "e1f1", erred).label is Label.MISS
    # A tactic on the board with no error before it: black's move is its best, white's Qxd5
    # stands better, and Ke2 lets the knight go -- a miss too.
    knight = "4k3/8/8/3n4/8/8/8/3QK3 b - - 0 1"
    tactic = (
        Line(moves=["e8f8"], depth=6, expected_score=0.35),
        Line(moves=["d1d5", "f8e7"], depth=6, expected_score=0.65),
        Line(moves=["d5f4"], depth=6, expected_score=0.5),
    )
    assert _answered(knight, "e8f8", "e1e2", tactic).label is Label.MISS
    # Black's ...Kf8 is a mistake, but white's best line, a knight up on the board, stands
    # only about equal (0.55): Ke2 for 0.43 is a mistake by its size, no chance let go.
    no_better = (
        Line(moves=["e8e7"], depth=6, expected_score=0.58),
        Line(moves=["d1d5", "f8e7"], depth=6, expected_score=0.55),
        Line(moves=["d5f4"], depth=6, expected_score=0.57),
    )
    assert _answered(knight, "e8f8", "e1e2", no_better).label is Label.MISTAKE


def test_a_forced_mate_let_go_and_one_walked_into() -> None:
    """Tai, 2026-10-02 12:16: "if I can mate in 5 but I made a move that still good, keep the
    advantage but no more mate in 5 then that's a miss. if I can mate in 1 but I made a move
    that is now mate in 2 then that's also a miss. if i can mate in 3 but I made a different
    move that's ALSO mate in 3 then that is fine". The mate is counted in the player's own
    moves on both sides of the move; the engine's own first move is never a miss for it. A
    move into a forced mate from a position not yet lost is a blunder."""
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    # Mate in 5; h3 keeps the advantage (a thousand centipawns) but no mate: a miss, though it
    # cost less than an inaccuracy.
    in_five = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=5)
    still_won = Line(moves=["g8f8"], depth=6, expected_score=0.03, score_cp=-1000)
    no_mate = _judged(board, "h2h3", in_five, still_won)
    assert no_mate.expected_loss < 0.04 and no_mate.label is Label.MISS, no_mate
    # Mate in 1; h3 leaves a mate in 2 (the opponent mated in 1 of theirs): a miss.
    in_one = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=1)
    mated_next = Line(moves=["g8f8"], depth=6, expected_score=0.0, mate=-1)
    longer = _judged(board, "h2h3", in_one, mated_next)
    assert longer.expected_loss == 0.0 and longer.label is Label.MISS, longer
    # Mate in 3; h3 is another mate in 3 (the opponent mated in 2 of theirs): fine.
    in_three = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=3)
    mated_in_two = Line(moves=["g8f8"], depth=6, expected_score=0.0, mate=-2)
    as_short = _judged(board, "h2h3", in_three, mated_in_two)
    assert as_short.label is Label.EXCELLENT, as_short
    # The engine's own first move, the search after it seeing only a longer mate (its
    # horizon): never a miss.
    own = Line(moves=["h2h3", "g8f8"], depth=6, expected_score=1.0, mate=3)
    seen_longer = Line(moves=["g8f8"], depth=6, expected_score=0.0, mate=-4)
    best = _judged(board, "h2h3", own, seen_longer)
    assert best.label is Label.BEST, best
    # The other side: worse but holding, and the move lets a mate in.
    lost = Line(moves=["g8f8"], depth=6, expected_score=0.2, score_cp=-400)
    mates = Line(moves=["a1a8"], depth=6, expected_score=1.0, mate=1)
    black = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
    walked = _judged(black, "h7h6", lost, mates)
    assert walked.label is Label.BLUNDER, walked
    # Like with like: the search before the move has a line for h3, mate in 3 as the best's;
    # the search after it, alone, sees the mate a move longer. The same search and depth as
    # the mate it is set against says h3 kept it: no miss.
    before = [in_three, Line(moves=["h2h3", "g8f8"], depth=6, expected_score=1.0, mate=3)]
    after = board.copy()
    after.push_uci("h2h3")
    fake = FakeEngine(
        {board.fen(): before, after.fen(): [mated_in_two.model_copy(update={"mate": -3})]}
    )
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    kept = annotator.assess_move("g", board, chess.Move.from_uci("h2h3"), 1)
    assert kept.label is Label.EXCELLENT, kept


def _mate_judged(fen: str, best_san: str, n: int, played_san: str, m: int | None) -> Label:
    """The label of `played_san` from `fen`, where the best line (`best_san` first) mates in
    `n`, and the move played leaves a mate in `m` of the player's moves (None: no mate, the
    position still a thousand centipawns up)."""
    board = chess.Board(fen)
    best = board.parse_san(best_san).uci()
    played = board.parse_san(played_san).uci()
    before = Line(moves=[best], depth=6, expected_score=1.0, mate=n)
    if m is None:
        after = Line(moves=[], depth=6, expected_score=0.03, score_cp=-1000)
    else:
        after = Line(moves=[], depth=6, expected_score=0.0, mate=1 - m)
    label = _judged(board, played, before, after).label
    assert label is not None
    return label


def test_a_mate_let_go_is_a_miss_in_one_none_left_or_four_longer_with_a_piece() -> None:
    """Tai, 2026-10-02 12:50: a missed forced mate is a Miss when (1) it was a mate in 1 and
    the move does not mate; (2) the move leaves no forced mate; (3) the mate gets MATE_LONGER
    (4) or more of the player's moves longer while the opponent still has a piece. Any other
    longer mate is labelled by its loss. The three games' cases of the analysis
    (findings/mate-miss-analysis.md) among them."""
    from sidecoach.analysis.score import MATE_LONGER

    assert MATE_LONGER == 4
    # (1) g01 ply 77: 39. Qg7# was mate in 1; 39. Qf2 leaves a mate in 2.
    g01 = "5Q2/7k/1pp1p3/p7/P2P4/1PK1P3/2P5/6R1 w - - 2 39"
    assert _mate_judged(g01, "Qg7#", 1, "Qf2", 2) is Label.MISS
    # (2) no mate left, the opponent a bare king or not.
    assert _mate_judged(g01, "Qg7#", 1, "Qf2", None) is Label.MISS
    # g04 ply 91: mate in 4 with 46. Qf6+; 46. Qh7 is a mate in 7, the opponent with a
    # bishop: three moves longer, under four -- labelled by its loss, not a miss.
    g04 = "2b2k2/8/3P2Q1/p4p2/2P2P1P/1P6/P6P/4K3 w - - 1 46"
    assert _mate_judged(g04, "Qf6+", 4, "Qh7", 7) is Label.EXCELLENT
    # (3) the same position, the mate four moves longer: a miss.
    assert _mate_judged(g04, "Qf6+", 4, "Qh7", 8) is Label.MISS
    # g02 ply 137: mate in 8 with 69. Nc4+ against a bare king; 69. Ne8 is a mate in 10 --
    # the technique, fine however much longer.
    g02 = "8/8/1kPN4/1P1K4/8/8/8/8 w - - 3 69"
    assert _mate_judged(g02, "Nc4+", 8, "Ne8", 10) is Label.EXCELLENT
    assert _mate_judged(g02, "Nc4+", 8, "Ne8", 14) is Label.EXCELLENT
    # King and pawns are technique too; a mate gone there is still a miss.
    pawns = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"
    assert _mate_judged(pawns, "Ra8#", 1, "h3", 2) is Label.MISS
    assert _mate_judged(pawns, "Ra8#", 1, "h3", None) is Label.MISS
    assert _mate_judged(pawns, "Ra8#", 3, "h3", 8) is Label.EXCELLENT
    assert _mate_judged(pawns, "Ra8#", 3, "h3", None) is Label.MISS


def test_a_search_under_way_when_the_settings_change_is_not_kept() -> None:
    """The settings' save forgets what was measured (`forget_measured`) from another thread,
    and the search under way then finishes the old way. Kept, it was the depth-12 position a
    depth-18 one is set against -- the mix the forgetting is for (the facts audit,
    2026-10-02). It is handed to its caller, and searched again next time."""

    class ChangedMidSearch(FakeEngine):
        annotator: GameAnnotator | None = None

        def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
            lines = super().analyse(board, **kwargs)
            if self.annotator is not None:
                self.annotator.forget_measured()  # the save, while this search ran
            return lines

    for measure in ("measure", "measure_for_verdict"):
        fake = ChangedMidSearch({})
        annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
        fake.annotator = annotator
        board = chess.Board()
        first = getattr(annotator, measure)(board)  # every search of it changed under it
        fake.annotator = None
        searched = fake.calls.count(board.fen())
        assert first.lines, "the caller still gets what it asked for"
        assert not annotator.measured(board), measure
        assert annotator.computed.lines(board) == [], measure
        getattr(annotator, measure)(board)
        assert fake.calls.count(board.fen()) == 2 * searched, f"{measure}: searched again"
        getattr(annotator, measure)(board)
        assert fake.calls.count(board.fen()) == 2 * searched, f"{measure}: and that one is kept"


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


def test_each_position_is_searched_once_each_way() -> None:
    """A verdict reads the position a move led to by its best line alone (one line, the full
    depth); measured whole after, that search is its best line there, not made again -- each
    position searched at most once for its best line and once for its hint. A new game
    searches again."""
    board = chess.Board()
    fake = FakeEngine({})
    annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
    e2e4 = chess.Move.from_uci("e2e4")
    assessment, _ = annotator.assess("g", board, e2e4, 1)
    after = board.copy(stack=False)
    after.push(e2e4)
    assert assessment.uci == "e2e4"
    searched = list(zip(fake.calls, fake.multipvs, strict=True))
    assert searched[:3] == [(board.fen(), 1), (after.fen(), 1), (after.fen(), 6)]
    annotator.assess("g", after, chess.Move.from_uci("e7e5"), 2)
    searched = list(zip(fake.calls, fake.multipvs, strict=True))
    assert len(searched) == len(set(searched)), searched
    annotator.reset()
    annotator.assess("g", board, e2e4, 1)
    assert fake.calls.count(board.fen()) == 2, "a new game measures again"


def test_assessment_carries_the_loss_the_best_line_the_tags_and_the_depth() -> None:
    board = chess.Board()
    after = board.copy(stack=False)
    after.push(chess.Move.from_uci("a2a3"))
    # The position after the move hit the engine's time cap: measured shallower than asked.
    fake = FakeEngine(
        scripted(
            board,
            [
                Line(moves=["e2e4", "e7e5"], depth=6, expected_score=0.60),
                Line(moves=["d2d4"], depth=6, expected_score=0.45),
            ],
        )
        | scripted(
            after,
            [
                Line(moves=["e7e5"], depth=4, expected_score=0.55, mate=-3),
                Line(moves=["d7d5"], depth=4, expected_score=0.20),
            ],
        )
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
            scripted(
                board,
                [
                    Line(moves=["f6e4"], depth=6, expected_score=0.80),
                    Line(moves=["e8d7"], depth=6, expected_score=0.40),
                ],
            )
        )
        annotator = GameAnnotator(engine=fake, depth=6, with_threats=False)
        annotator.measure(board)  # measured whole, so the only-move call is made
        return annotator.assess_move("g", board, chess.Move.from_uci("f6e4"), 1).label

    assert label("4k3/8/5n2/8/1b2B3/2N5/8/4K3 b - - 0 1") is Label.BEST
    assert label("4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1") is Label.GREAT


def test_a_depth_asked_for_is_never_cut_by_a_time_limit() -> None:
    """A five-second cap answered 63 of 1,642 searches of a session below depth 18, some at
    depth 11, read as depth 18 (the engine audit, 2026-10-04; Tai: no lower depths). A search
    asked for a depth goes to it; only a search asked for a time is limited by one."""
    uci = _ScriptedUci([*_D13], [*_D13])
    running = _scripted_engine(uci)
    running.analyse(chess.Board(), depth=40, multipv=4)
    assert uci.limits[-1].depth == 40 and uci.limits[-1].time is None
    running.analyse(chess.Board(), time_s=0.05)
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
    waited on it forever. A search that reports nothing for a whole patience has its process
    ended, and is made again on a new one; one still reporting is never cut, however long it
    takes (the depth asked for is the depth answered: the engine audit, 2026-10-04). The
    watchdog's wait is driven here: its patience runs out on the silent engine once its search
    is under way (as the real patience only can), and never on the real one. (It ran out at
    once: the process was then ended while the search was still being started, and the
    start's own CancelledError was raised -- a race the real watchdog never meets, seen once
    in a loaded run, 2026-10-02.)"""
    import sys

    import chess.engine

    assert STOCKFISH is not None
    stub = tmp_path / "silent.py"
    stub.write_text(_SILENT_ENGINE, encoding="utf-8")
    waits: list[float] = []
    searching = threading.Event()

    def wait(done: threading.Event, seconds: float) -> bool:
        waits.append(seconds)
        if len(waits) == 1:
            assert searching.wait(HANG_S), "the search on the silent engine never began"
            return False  # its patience has run out, the search under way and unanswered
        return done.wait()

    running = Engine(path=STOCKFISH, threads=1, hash_mb=16, multipv=2)
    silent = chess.engine.SimpleEngine.popen_uci([sys.executable, str(stub)])
    start_search = silent.analysis

    def analysis(*args: Any, **kwargs: Any) -> Any:
        search = start_search(*args, **kwargs)  # `go` sent, the search under way
        searching.set()
        return search

    silent.analysis = analysis  # type: ignore[method-assign]
    running._engine = silent
    running.watchdog_wait = wait
    try:
        lines = running.analyse(chess.Board(), depth=6)
        assert lines, "searched again on a new engine"
        assert running._engine is not silent, "the silent engine was replaced"
        assert waits[0] == running.watchdog_s, "patience: a whole watchdog_s of silence"
    finally:
        running.close()


def test_a_long_search_that_keeps_reporting_is_never_cut() -> None:
    """The watchdog ends only an engine that has gone silent: a search reporting between two of
    its looks -- however long it runs -- is left to reach its depth. The watchdog's waits are
    driven by the search's own reports here, never by a clock: each look comes after a report,
    and the process is never ended."""
    uci = _ScriptedUci([*_D13, *_D13], [*_D13])
    uci.hold = threading.Event()
    running = _scripted_engine(uci)
    ended: list[bool] = []
    running._end_process = lambda: ended.append(True)  # type: ignore[method-assign]
    looks = 0

    def wait(done: threading.Event, seconds: float) -> bool:
        nonlocal looks
        looks += 1
        if looks <= 3:
            assert uci.started.wait(HANG_S)  # the reports are in: the search talked
            running._heard += 1  # and talks on between this look and the next
            return False
        uci.hold.set()  # long enough: the search ends
        return done.wait(HANG_S)

    running.watchdog_wait = wait
    lines = running.analyse(chess.Board(), depth=13)
    assert lines and lines[0].depth == 13
    assert looks >= 4 and not ended, "a search still reporting was taken for a hung engine"


def test_a_search_working_in_silence_is_never_cut_and_an_idle_silent_one_is() -> None:
    """A real position (three moves, in check) went 48 s at depth 18 without a word -- one root
    move searched all that time -- and the 30 s silence watchdog killed it, twice, so the
    position was never answered (the engine audit, 2026-10-04). Silence with the processor
    working is a search; silence with the processor idle is a hung engine. Driven, never
    timed: each look of the watchdog sees no report, and the engine's processor time either
    moves on by a whole patience or stands still."""
    for working, killed in ((True, False), (False, True)):
        uci = _ScriptedUci([*_D13], [*_D13])
        uci.hold = threading.Event()
        running = _scripted_engine(uci)
        ended: list[bool] = []
        running._end_process = lambda ended=ended: ended.append(True)  # type: ignore[method-assign]
        spent = [0.0]
        running.cpu_seconds = lambda spent=spent: spent[0]
        looks = 0

        def wait(done: threading.Event, seconds: float, uci=uci, spent=spent, w=working) -> bool:
            nonlocal looks
            looks += 1
            if looks <= 3:
                assert uci.started.wait(HANG_S)  # under way, and then not a word
                if w:
                    spent[0] += seconds  # a whole patience of processor time
                return False
            uci.hold.set()
            return done.wait(HANG_S)

        running.watchdog_wait = wait
        running.analyse(chess.Board(), depth=13)
        assert bool(ended) is killed, (working, ended)
    annotator = GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False)
    with pytest.raises(ValueError, match="not legal"):
        annotator.assess("g", chess.Board(), chess.Move.from_uci("e2e5"), 1)


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


def _line(*moves: str) -> Line:
    return Line(moves=list(moves), depth=6, expected_score=0.5)


def _after(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


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
    # Each history's best line and its hint, once.
    assert fake.calls.count(one.fen()) == 2 * 2, "each history searched once"
    # After a capture or a pawn move nothing before can repeat: one search serves both.
    for board in (one, two):
        board.push_san("e4")
    annotator.measure(one)
    annotator.measure(two)
    assert fake.calls.count(one.fen()) == 2


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


def test_the_best_line_stays_first_and_mates_are_ordered_by_length() -> None:
    """A position whole: its best line, its own search, first -- a candidate whose own
    position's search scores it higher stays behind it, so the engine's move is never charged
    a loss (the audit of 2026-10-05) -- and among the others a shorter mate ahead of a longer
    one, every mate scoring alike."""
    board = chess.Board()
    annotator = GameAnnotator(engine=FakeEngine({}), depth=18, with_threats=False)
    best = annotator.measured_lines(board, [Line(moves=["e2e4"], depth=18, expected_score=0.55)])

    def child(score: float, mate: int | None = None) -> object:
        line = Line(moves=["a7a6"], depth=18, expected_score=score, mate=mate)
        return annotator.measured_lines(board, [line])

    children = [
        (chess.Move.from_uci("e2e4"), child(0.45)),
        (chess.Move.from_uci("d2d4"), child(0.40)),  # 0.60: higher than the best line
        (chess.Move.from_uci("g1f3"), child(0.0, mate=-5)),  # mates a move after: in 6
        (chess.Move.from_uci("c2c4"), child(0.0, mate=-1)),  # mates in 2
    ]
    whole = annotator.whole(board, best, children)  # type: ignore[arg-type]
    assert [line.moves[0] for line in whole.lines] == ["e2e4", "c2c4", "g1f3", "d2d4"]
    assert [line.mate for line in whole.lines[1:3]] == [2, 6]
    loss = annotator.assess_from("g", board, chess.Move.from_uci("e2e4"), 1, whole)
    assert loss.expected_loss == 0.0 and loss.label is Label.BEST


def test_the_lines_view_never_searches_several_lines_at_the_full_depth() -> None:
    """`WholeSearcher` (a review's line view, a coach made on demand): a position's lines
    made the one-line way -- one line to the depth, the hint at EARLY_DEPTH -- never several
    lines to the depth at once, which explode on decided and forcing positions."""
    from sidecoach.analysis.annotator import EARLY_DEPTH, WholeSearcher

    engine = FakeEngine({chess.Board().fen(): [Line(moves=["e2e4"], depth=18, expected_score=0.5)]})
    lines = WholeSearcher(engine, lines=3, depth=18).analyse(chess.Board())
    assert lines and lines[0].moves[0] == "e2e4"
    asked = set(zip(engine.multipvs, engine.depths, strict=True))
    assert asked <= {(1, 18), (3, EARLY_DEPTH)}, asked
