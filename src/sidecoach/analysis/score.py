"""Expected score, the loss a move costs, and what makes a moment decisive.

Everything here is in expected score, never centipawns: 0.5 is a dead draw, 1.0 is a win
for the side to move. A 50-centipawn slip at +8 costs nothing and the same slip at level
costs the game, so a threshold in centipawns says nothing about how much a move mattered.

Pure functions over engine lines, so they are cheap to test and carry no engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange
from sidecoach.config import CoachSettings
from sidecoach.events import Label, Line, Severity

DRAW = 0.5
# Lichess's curve from centipawns to winning chances, fitted to games between people.
# Stockfish's own win-draw-loss numbers answer a different question: how often Stockfish
# beats Stockfish from here, which is 100% from a knight up and says nothing about the rest
# of a club game. A coach needs the human curve, where a knight up is about 0.88 and the
# moves that follow can still cost something.
HUMAN_K = 0.00368208
CP_CAP = 1000


def human_expected_score(score_cp: int | None, mate: int | None) -> float:
    """Expected score for the side to move, on the curve fitted to human games."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    if score_cp is None:
        return DRAW
    capped = max(-CP_CAP, min(CP_CAP, score_cp))
    return 1.0 / (1.0 + math.exp(-HUMAN_K * capped))


def engine_expected_score(wdl: tuple[int, int, int] | None) -> float:
    """Expected score from the engine's own win-draw-loss numbers: (W + D/2) / 1000."""
    if wdl is None or sum(wdl) == 0:
        return DRAW
    wins, draws, _ = wdl
    return (wins + draws / 2) / sum(wdl)


def expected_score(
    model: str, score_cp: int | None, mate: int | None, wdl: tuple[int, int, int] | None
) -> float:
    """Expected score under the named model: "human" (the default) or "engine"."""
    if model == "engine":
        return engine_expected_score(wdl)
    return human_expected_score(score_cp, mate)


@dataclass(frozen=True)
class Thresholds:
    """How much loss earns a name, and what makes a position a decisive moment.

    The severity lines are chess.com's published cut-offs in expected points lost
    (support.chess.com 8572705, Classification V2): best 0, excellent up to 0.02, good up to
    0.05, inaccuracy up to 0.10, mistake up to 0.20, blunder beyond (Tai, 2026-10-06: "use
    their definition"). They had been fitted to six of the user's games instead, at 0.04
    and 0.28.
    """

    inaccuracy: float = 0.05
    mistake: float = 0.10
    blunder: float = 0.20
    decisive_gap: float = 0.10
    # An only move: everything else throws away this much. Wider than a decisive moment,
    # since "there is one move here" is a stronger claim than "this move matters".
    only_move_gap: float = 0.30
    # Below this loss a move is "excellent" rather than merely "good".
    excellent: float = 0.02
    # A position this far gone, either way, is decided: a sacrifice in it is not brilliant.
    decided: float = 0.90

    @classmethod
    def from_settings(cls, coach: CoachSettings) -> Thresholds:
        return cls(
            inaccuracy=coach.inaccuracy,
            mistake=coach.mistake,
            blunder=coach.blunder,
            decisive_gap=coach.decisive_gap,
            only_move_gap=coach.only_move_gap,
            excellent=coach.excellent,
            decided=coach.decided,
        )


DEFAULT_THRESHOLDS = Thresholds()


def game_over(board: chess.Board) -> bool:
    """Whether the game is over on the board as the sites play it: mate, stalemate, no mating
    material, or a draw by rule that has happened -- the third occurrence of a position, or
    fifty moves without a capture or a pawn move. Chess.com and Lichess end the game there
    themselves. A draw the side to move could merely claim with its next move is not one:
    that position is played on, and a queen up is still a queen up. The repetition needs the
    board's moves; a board without them knows no history and repeats nothing."""
    return (
        board.outcome(claim_draw=False) is not None
        or board.is_repetition(3)
        or board.is_fifty_moves()
    )


def terminal_expected_score(board: chess.Board) -> float | None:
    """The value of a finished position to the side to move, or None if it is not finished."""
    if not game_over(board):
        return None
    outcome = board.outcome(claim_draw=False)
    if outcome is None or outcome.winner is None:
        return DRAW  # a draw by rule (`game_over`)
    return 1.0 if outcome.winner == board.turn else 0.0


def position_score(board: chess.Board, lines: Sequence[Line]) -> float:
    """The position's expected score to the side to move, whether or not it is finished."""
    finished = terminal_expected_score(board)
    if finished is not None:
        return finished
    if not lines:
        return DRAW  # nothing was measured; claim nothing
    return lines[0].expected_score


def eval_loss(before: Sequence[Line], played_uci: str, after_score: float) -> float:
    """What the played move cost its mover, in expected score.

    `after_score` is the position after the move, scored for the player who is to move
    there, which is the opponent; the mover's own prospects are one minus that. Comparing
    the two needs both to have been measured at the same depth, which is why the annotator
    analyses every position of a game the same way.

    The engine's own first choice costs nothing by definition. Without that rule a search
    that returns a slightly different number for the same line at the same depth would have
    the coach calling the best move on the board an inaccuracy.
    """
    if not before:
        return 0.0
    best = before[0]
    if best.moves and played_uci == best.moves[0]:
        return 0.0
    return max(0.0, best.expected_score - (1.0 - after_score))


def about_as_good(loss: float, thresholds: Thresholds | None = None) -> bool:
    """A move this far behind the best is about as good as it: chess.com's Best or Excellent
    band, as `label_move` grades it. The coach's "about equally good" moves are these (Tai,
    2026-10-02: the bands, not an arbitrary margin)."""
    return loss < (thresholds or DEFAULT_THRESHOLDS).excellent


def graded_good(loss: float, thresholds: Thresholds | None = None) -> bool:
    """A move this far behind the best is graded good or better -- Best, Excellent or Good --
    and no inaccuracy (`label_move`'s flag)."""
    return loss < (thresholds or DEFAULT_THRESHOLDS).inaccuracy


def clearly_worse(loss: float, thresholds: Thresholds | None = None) -> bool:
    """A move this far behind the best is clearly worse: a mistake or a blunder, as `severity`
    grades it."""
    return loss >= (thresholds or DEFAULT_THRESHOLDS).mistake


def severity(loss: float, thresholds: Thresholds | None = None) -> Severity:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if loss >= thresholds.blunder:
        return Severity.BLUNDER
    if clearly_worse(loss, thresholds):
        return Severity.MISTAKE
    if loss >= thresholds.inaccuracy:
        return Severity.INACCURACY
    return Severity.NONE


# How a side stands, by its expected score: the bands the coach speaks in (`facts.standing_words`:
# losing, worse, about equal, better, winning). "Better" starts at 0.62 and "winning" at 0.8, as
# the idea reader counts a level and a won position (`idea.LEVEL`, `idea.WON`).
WORSE_FROM = 0.2
EQUAL_FROM = 0.38
BETTER_FROM = 0.62
WINNING_FROM = 0.8


def standing(score: float) -> int:
    """The band an expected score falls in: 0 losing, 1 worse, 2 about equal, 3 better, 4
    winning."""
    if score >= WINNING_FROM:
        return 4
    if score >= BETTER_FROM:
        return 3
    if score > EQUAL_FROM:
        return 2
    if score > WORSE_FROM:
        return 1
    return 0


def label_move(
    loss: float,
    thresholds: Thresholds,
    *,
    is_best: bool,
    in_book: bool,
    decisive: bool,
    sacrifice: bool,
    before_score: float,
    opponent_blundered: bool,
    second_score: float | None = None,
    chance: bool = False,
    mate_let_go: bool = False,
    mate_allowed: bool = False,
    obvious: bool = False,
) -> Label:
    """What kind of move it was, in the words chess.com's review uses, as its help pages
    define them (2026-10-01) and as far as six of the user's reviewed games bear them out.

    - Best, "the engine's top choice" (`is_best`); excellent, "almost as good as the best
      move" (a loss under `excellent`); good, "a decent move but not the best".
    - Inaccuracy, mistake, blunder: by the loss. A move that walks into a forced mate
      (`mate_allowed`) "loses the game" and is a blunder, unless the game was lost already
      (the curve puts a mate at 0.0 and a thousand centipawns at 0.025).
    - Book, "a conventional opening move": the tables' move, unless it threw away a tenth of
      the game -- the tables hold dubious gambits too (1. e4 f5), and chess.com's book does
      not, calling 1...f5 a mistake.
    - Miss, "a move that missed a tactical opportunity or a chance to punish the opponent":
      the position offered a winning chance (`chance`, read by `verdict` from the engine's
      lines) and the move played gave ground (a loss of an inaccuracy or more), whatever
      the size of the loss (Tai, 2026-10-02). And a forced mate let go (`mate_let_go`,
      read by `verdict`): a mate in 1 not given, no mate left, or one MATE_LONGER moves
      longer while the opponent still has a piece -- a miss whatever it cost, even while it
      still wins; any other longer mate is labelled by its loss (Tai, 2026-10-02 12:50).
    - Great, "a move that altered the course of the game": the best move where every other
      lost ground (`decisive`, the only good move), or the best reply to the opponent's
      blunder. Merely the best move after any error was great 30 times in the six games
      where chess.com said so 20 times.
    - Brilliant, chess.com's own definition and nothing more: "always the best or nearly
      best move" (best or excellent), "a good piece sacrifice" (`sacrifice`, a piece given
      or left to be taken),
      "you should not be in a bad position after" it (about equal or better once it is
      played: `standing` 2 or more), and "you should not be completely winning even if you
      hadn't found the move" (the second-best line, `second_score`, short of winning).

    Great applies to no `obvious` move (a recapture, a free capture nobody can take back,
    the only legal move), nor in a lost position, where the best move changes nothing.
    """
    lost = before_score <= 1.0 - thresholds.decided
    # Walking into a mate from a lost position is not the move that lost: chess.com leaves
    # those unmarked (eleven of them in the six games, the best move among them).
    mate_allowed = mate_allowed and not lost and not is_best
    # A mate let go is a miss of its own (`mate_let_go`), never flagged by what it cost: the
    # engine's own first move, or one that keeps a mate as short, is never an inaccuracy.
    flagged = not graded_good(loss, thresholds) or mate_allowed
    if in_book and loss < thresholds.mistake:
        return Label.BOOK
    if mate_let_go and not is_best:
        return Label.MISS
    if flagged and chance and not is_best:
        return Label.MISS
    if mate_allowed:
        return Label.BLUNDER
    if not flagged:
        # Brilliant, as chess.com defines it and nothing more (Tai, 2026-10-06: "use their
        # definition instead"; support.chess.com 8572705): "always the best or nearly best
        # move" (best or excellent), "a good piece sacrifice" (`sacrifice`, a piece), after
        # which "you should not be in a bad position" (about equal or better: worse is bad)
        # and without which you "should not be completely winning" (the best other line short
        # of the coach's "winning"). Tai, 2026-10-07, on his 135 reviews: the looser reading
        # (not lost after, short of decided without) gave a brilliant that left the mover at
        # 0.12, and seven where the best other move already won. Its generosity to newer
        # players is not published, and is not guessed at.
        without = second_score if second_score is not None else before_score
        after = before_score - loss
        nearly_best = is_best or about_as_good(loss, thresholds)
        if (
            sacrifice
            and nearly_best
            and standing(after) >= 2
            and without < WINNING_FROM
        ):
            return Label.BRILLIANT
        if not obvious and not lost and is_best and (decisive or opponent_blundered):
            return Label.GREAT
        if is_best:
            return Label.BEST
        return Label.EXCELLENT if about_as_good(loss, thresholds) else Label.GOOD
    verdict = severity(loss, thresholds)
    if verdict is Severity.BLUNDER:
        return Label.BLUNDER
    return Label.MISTAKE if verdict is Severity.MISTAKE else Label.INACCURACY


@dataclass(frozen=True)
class Previous:
    """The move before, as far as the next move's label needs it: what it cost its mover
    (the opponent), where it landed and whether it captured."""

    loss: float
    to_square: chess.Square
    captured: bool


def winning_chance(
    after: chess.Board,
    before_lines: Sequence[Line],
    after_lines: Sequence[Line],
    before_score: float,
) -> bool:
    """Whether the move from the position measured by `before_lines` to `after` let a
    winning chance go, read off the engine's own lines (Tai, 2026-10-02: chess.com's miss --
    the opponent's error or a tactic already on the board, whichever gave it). The chance:
    the best line stood better or winning (`standing` 3 or more, from 0.62), and the move
    played leaves a lower band. The engine's evaluation decides, not where the chance came
    from: an opponent's error that left the player no better was no chance to win. A
    position with a forced mate is read by its mate instead (`mate_let_go`)."""
    top = before_lines[0] if before_lines else None
    if top is not None and top.mate is not None and top.mate > 0:
        return False
    left = 1.0 - position_score(after, after_lines)
    return standing(before_score) >= standing(BETTER_FROM) and standing(left) < standing(
        before_score
    )


def mate_after(
    move: chess.Move, after: chess.Board, before_lines: Sequence[Line], after_lines: Sequence[Line]
) -> int | None:
    """The player's shortest forced mate once `move` is played, counted as the engine counts a
    mate before it -- in the player's own moves, the one played included -- or None when the
    move leaves none. Like is compared with like: when the search before the move has a line
    for the move played (its candidates, `multipv`), that line's mate is the answer, the same
    search and depth the mate it is set against comes from; else the search after the move,
    whose mate against the opponent (in their moves) is one of the player's longer."""
    if after.is_checkmate():
        return 1
    for line in before_lines:
        if line.moves[:1] == [move.uci()]:
            return line.mate if line.mate is not None and line.mate > 0 else None
    top = after_lines[0] if after_lines else None
    if top is not None and top.mate is not None and top.mate < 0:
        return 1 - top.mate
    return None


# How much longer a forced mate may get, in the player's moves, before letting it go is a
# Miss -- with a piece still on the opponent's side. Tai, 2026-10-02 12:45: "instead of 1 move
# longer, change it to 3-4 moves longer"; 12:50: four, the rule "mate in 1 missed, or the mate
# gone, or 4+ moves longer" with technique endgames exempt. One move longer gave 41 Misses in
# the six recorded games, nearly all in won endgames chess.com left unmarked.
MATE_LONGER = 4


def mate_let_go(
    move: chess.Move, after: chess.Board, before_lines: Sequence[Line], after_lines: Sequence[Line]
) -> bool:
    """The player had a forced mate in N and `move` let it go (Tai, 2026-10-02 12:50):
    1. it was a mate in 1, and the move does not mate ("missing mate in 1 is a miss regardless
       - it's a special rule", 12:46);
    2. the move leaves no forced mate at all;
    3. or the mate gets MATE_LONGER or more of the player's moves longer while the opponent
       still has a piece -- against a bare king, or a king and pawns, the technique mate (the
       box, the ladder, queening first) is the method to learn, and a longer route is fine.
    Any other longer mate is labelled by its loss. The engine's own first move is never one
    (`label_move` checks it): its line is the mate. No deeper search for "no mate left": the
    analyses the labels use decide."""
    top = before_lines[0] if before_lines else None
    if top is None or top.mate is None or top.mate <= 0:
        return False
    kept = mate_after(move, after, before_lines, after_lines)
    if kept is None:
        return True
    if top.mate == 1:
        return kept != 1
    return kept >= top.mate + MATE_LONGER and _has_a_piece(after, after.turn)


def _has_a_piece(board: chess.Board, color: chess.Color) -> bool:
    """Whether `color` has anything but its king and pawns."""
    return any(
        p.color == color and p.piece_type not in (chess.KING, chess.PAWN)
        for p in board.piece_map().values()
    )


def verdict(
    before: chess.Board,
    move: chess.Move,
    before_lines: Sequence[Line],
    after_lines: Sequence[Line],
    *,
    loss: float,
    before_score: float,
    decisive: bool,
    in_book: bool,
    sacrifice: bool,
    previous: Previous | None,
    thresholds: Thresholds,
) -> Label:
    """The label of `move` from `before`, read off the two positions' measured lines: the
    annotator's as it judges a move, and a saved review's made again from what it kept
    (`game_review.relabel`), so the two cannot drift. `previous`: the move before, None when
    it is not known (the game's first move, or a move that did not lead here)."""
    after = before.copy(stack=False)
    after.push(move)
    best = before_lines[0].moves if before_lines else []
    erred = previous.loss if previous is not None else 0.0
    captures = before.is_capture(move)
    # Taking back on the square the opponent just captured on. Taking a piece that merely
    # moved there is not obvious: that is how a blunder gets punished.
    recapture = (
        previous is not None
        and previous.captured
        and captures
        and move.to_square == previous.to_square
    )
    # A capture nobody can take back is a grab, not a find; a king's capture always is.
    # Taking back is read by the rules: a piece guarded only by one pinned to its own king
    # is taken for nothing.
    free_grab = captures and not exchange.can_retake(before, move)
    obvious = recapture or free_grab or before.legal_moves.count() == 1
    top = before_lines[0] if before_lines else None
    after_top = after_lines[0] if after_lines else None
    # A forced mate the move walks into. The mates are the lines' own: the curve puts a
    # thousand centipawns a fortieth of a game below a mate.
    mate_allowed = (
        after_top is not None
        and after_top.mate is not None
        and after_top.mate > 0
        and not (top is not None and top.mate is not None and top.mate < 0)
    )
    return label_move(
        loss,
        thresholds,
        is_best=bool(best) and best[0] == move.uci(),
        in_book=in_book,
        decisive=decisive,
        sacrifice=sacrifice,
        before_score=before_score,
        second_score=before_lines[1].expected_score if len(before_lines) > 1 else None,
        opponent_blundered=erred >= thresholds.blunder,
        chance=winning_chance(after, before_lines, after_lines, before_score),
        mate_let_go=mate_let_go(move, after, before_lines, after_lines),
        mate_allowed=mate_allowed,
        obvious=obvious,
    )


# Per-move accuracy: 100 for no loss, falling off exponentially with the drop in winning
# chances. The shape is Lichess's; the steepness is not. Lichess uses 0.04354, on which an
# inaccuracy scores about 80, and the plain mean of that ran fifteen points above
# chess.com's accuracy on every one of six games. At 0.25 an inaccuracy scores 26, a
# mistake 5, and the mean lands within 4.4 points of chess.com's number over twelve
# game-sides with no bias to speak of: chess.com's accuracy is, in effect, how often you
# played a near-best move.
ACCURACY_A = 103.1668
ACCURACY_K = 0.25
ACCURACY_C = 3.1669


def move_accuracy(loss: float) -> float:
    """0 to 100 for one move, from its loss in expected score."""
    raw = ACCURACY_A * math.exp(-ACCURACY_K * loss * 100.0) - ACCURACY_C
    return round(max(0.0, min(100.0, raw)), 2)


def game_accuracy(accuracies: Sequence[float]) -> float | None:
    """One number for a side's game: the mean of the per-move accuracies.

    Lichess averages a volatility-weighted mean with a harmonic mean; the harmonic mean
    collapses on a single move that flipped the game, which club games have. Chess.com's
    own formula is not published; the plain mean on the steepened curve is what fitted.
    """
    if not accuracies:
        return None
    return sum(accuracies) / len(accuracies)


def best_gap(lines: Sequence[Line]) -> float | None:
    """How much better the best line is than the second best, or None without a second."""
    if len(lines) < 2:
        return None
    return max(0.0, lines[0].expected_score - lines[1].expected_score)


def is_only_move(board: chess.Board, lines: Sequence[Line], thresholds: Thresholds) -> bool:
    """One move holds the position: either it is the only legal one, or the rest throw it away."""
    legal = board.legal_moves.count()
    if legal == 0:
        return False
    if legal == 1:
        return True
    gap = best_gap(lines)
    return gap is not None and gap >= thresholds.only_move_gap


def is_decisive(board: chess.Board, lines: Sequence[Line], thresholds: Thresholds) -> bool:
    """A moment where the choice matters: the lines diverge, or there is only one move.

    This is what the coach may say out loud before a move is played, because it points at
    the position without naming the move. Which move it is stays behind the disclosure
    filter until the user finds it, plays it, or asks.
    """
    if is_only_move(board, lines, thresholds):
        return True
    gap = best_gap(lines)
    return gap is not None and gap >= thresholds.decisive_gap
