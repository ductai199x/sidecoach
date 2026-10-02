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

    The three severity lines were fitted to chess.com's review tallies over six of the
    user's games (671 plies, 2026-09-15): they put our inaccuracy, mistake, miss and blunder
    counts at 55, 33, 29 and 9 against their 55, 38, 24 and 10. A blunder throws away more
    than a quarter of the game.
    """

    inaccuracy: float = 0.04
    mistake: float = 0.10
    blunder: float = 0.28
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


def severity(loss: float, thresholds: Thresholds | None = None) -> Severity:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    if loss >= thresholds.blunder:
        return Severity.BLUNDER
    if loss >= thresholds.mistake:
        return Severity.MISTAKE
    if loss >= thresholds.inaccuracy:
        return Severity.INACCURACY
    return Severity.NONE


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
    mate_missed: bool = False,
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
      the best move won material or mated (`chance`, read off its line by the annotator)
      and the move played gave ground. It names the move whatever the size of the loss; a
      forced mate let go (`mate_missed`) is one even when the move still wins.
    - Great, "a move that altered the course of the game": the best move where every other
      lost ground (`decisive`, the only good move), or the best reply to the opponent's
      blunder. Merely the best move after any error was great 30 times in the six games
      where chess.com said so 20 times.
    - Brilliant, "the best move -- and tricky to find too": a sound sacrifice, as good as
      the best, in a game still to be won. Whether it was won already is read from the
      second-best line (`second_score`): the best is the line the sacrifice makes, and a
      sacrifice that is the only win was never brilliant.

    Neither great nor brilliant applies to an `obvious` move (a recapture, a free capture
    nobody can take back, the only legal move), nor in a lost position, where the best move
    changes nothing.
    """
    lost = before_score <= 1.0 - thresholds.decided
    # Walking into a mate from a lost position is not the move that lost: chess.com leaves
    # those unmarked (eleven of them in the six games, the best move among them).
    mate_allowed = mate_allowed and not lost and not is_best
    # The engine's own first move lets no mate go, whatever the search after it reports (a
    # long mate past its horizon): it was flagged, and with no loss labelled an inaccuracy.
    mate_missed = mate_missed and not is_best
    flagged = loss >= thresholds.inaccuracy or mate_missed or mate_allowed
    if in_book and loss < thresholds.mistake:
        return Label.BOOK
    if flagged and chance and not is_best:
        return Label.MISS
    if mate_allowed:
        return Label.BLUNDER
    if not flagged:
        if not obvious and not lost:
            without = second_score if second_score is not None else before_score
            if sacrifice and loss < thresholds.excellent and without < thresholds.decided:
                return Label.BRILLIANT
            if is_best and (decisive or opponent_blundered):
                return Label.GREAT
        if is_best:
            return Label.BEST
        return Label.EXCELLENT if loss < thresholds.excellent else Label.GOOD
    verdict = severity(loss, thresholds)
    if verdict is Severity.BLUNDER:
        return Label.BLUNDER
    return Label.MISTAKE if verdict is Severity.MISTAKE else Label.INACCURACY


# What a line wins, in pawns: the piece values every club player counts by.
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
# A line that wins this much is a tactic to find (the coach's facts count the same).
TACTIC_GAIN = 2
# How far into a line a tactic is looked for: four moves each.
TACTIC_PLIES = 8


def _balance(board: chess.Board, us: chess.Color) -> int:
    return sum(
        PIECE_VALUES.get(piece.piece_type, 0) * (1 if piece.color == us else -1)
        for piece in board.piece_map().values()
    )


def material_won(board: chess.Board, moves: Sequence[str], plies: int = TACTIC_PLIES) -> int:
    """What the side to move wins over the first `plies` of `moves` (UCI), in pawns, counted
    after the opponent's replies; a line that mates wins everything (99). Material the
    opponent has just taken, which the side to move is owed back, is not won: taking back is
    a trade (read from the board's moves; a board without them owes nothing)."""
    us = board.turn
    base = _balance(board, us)
    back = board.copy()
    while back.move_stack:
        move = back.pop()
        if not back.is_capture(move):
            break
        base = max(base, _balance(back, us))
    at = board.copy(stack=False)
    best = 0
    for i, uci in enumerate(moves[:plies]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in at.legal_moves:
            break
        at.push(move)
        if at.is_checkmate():
            return 99 if at.turn != us else best
        if i % 2 == 1:
            best = max(best, _balance(at, us) - base)
    return best


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
