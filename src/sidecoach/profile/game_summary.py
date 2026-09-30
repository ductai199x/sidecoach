"""A reviewed game summed up (docs/game-review-plan.md, the summary screen): each side's
accuracy and count of every kind of move, how each side played each phase, the moves the
review stops at, and the graph of the game -- all read off the review's verdicts, nothing
guessed.

The phase grade is chess.com's report card in our own vocabulary: the phase's accuracy drawn
as the kind of move it amounts to -- a phase played as a best move is "best", one played as an
inaccuracy is "inaccuracy" -- with a great or brilliant move in a well-played phase lifting it
to that. Accuracy is the review's own (analysis/score.py); against chess.com's on 144 of the
player's game sides it is 5.8 points apart on average, with no lean either way (phase 0,
2026-09-29).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import fmean

import chess

from sidecoach.analysis.phase import phase_of
from sidecoach.analysis.score import game_accuracy, position_score
from sidecoach.events import (
    GameSummary,
    KeyMoment,
    Label,
    Line,
    PhaseGrade,
    ReviewedGame,
    Side,
    SideSummary,
)

PHASES = ("opening", "middlegame", "endgame")
# A phase needs this many of a side's moves to be graded: a king move or two into an
# ending grades nothing (chess.com leaves the endgame out of a game that never reached it).
MIN_PHASE_MOVES = 3
# The moves the review stops at, besides the last book move.
KEY_LABELS = (
    Label.BRILLIANT,
    Label.GREAT,
    Label.INACCURACY,
    Label.MISTAKE,
    Label.MISS,
    Label.BLUNDER,
)
# A phase's accuracy as the kind of move it amounts to, from the top.
GRADES = (
    (90.0, Label.BEST),
    (80.0, Label.EXCELLENT),
    (70.0, Label.GOOD),
    (55.0, Label.INACCURACY),
    (40.0, Label.MISTAKE),
)


SHAPES = ("smooth", "balanced", "sudden", "giveaway", "wild", "intense", "sharp")
# White's expected score from which a side is winning, and from which it was clearly winning
# (a giveaway throws that away); a game is close between the first two.
WINNING = 0.7
CLEARLY = 0.85
CLOSE = (0.35, 0.65)


def shape_of(graph: Sequence[float], result: str | None) -> str | None:
    """The game's shape, chess.com's "game shapes" read off White's expected score after each
    ply: **wild**, both sides winning by turns three times or more; **intense**, twice;
    **giveaway**, one side clearly winning and then not winning; **sudden**, close for three
    quarters of the game and decided late; **smooth**, one side took the lead and kept it;
    **sharp**, big swings that did not change who led; **balanced**, close throughout.
    Too short a game (under 10 plies) has no shape."""
    if len(graph) < 11:
        return None
    leaders: list[int] = []  # +1 White winning, -1 Black, in order of the changes
    for score in graph:
        now = 1 if score >= WINNING else -1 if score <= 1 - WINNING else 0
        if now and (not leaders or leaders[-1] != now):
            leaders.append(now)
    changes = len(leaders) - 1 if leaders else 0
    winner = 1 if result == "1-0" else -1 if result == "0-1" else 0
    if changes >= 3:
        return "wild"
    if changes == 2:
        return "intense"
    clearly = [1 if s >= CLEARLY else -1 if s <= 1 - CLEARLY else 0 for s in graph]
    if any(c and c != winner for c in clearly):
        return "giveaway"
    close = [CLOSE[0] <= s <= CLOSE[1] for s in graph]
    three_quarters = len(graph) * 3 // 4
    if all(close[:three_quarters]) and winner and not close[-1]:
        return "sudden"
    if leaders and winner == leaders[-1] and changes == 0:
        swings = sum(abs(b - a) for a, b in zip(graph, graph[1:], strict=False))
        return "sharp" if swings > 2.0 else "smooth"
    if sum(close) >= 0.7 * len(graph):
        return "balanced"
    return "sharp"


def phase_grade(accuracy: float, labels: list[Label]) -> Label:
    """A phase's grade: its accuracy as a kind of move; played that well with a brilliant or
    great move in it, that move's name."""
    if accuracy >= GRADES[0][0]:
        if Label.BRILLIANT in labels:
            return Label.BRILLIANT
        if Label.GREAT in labels:
            return Label.GREAT
    for floor, grade in GRADES:
        if accuracy >= floor:
            return grade
    return Label.BLUNDER


def summarize(game: ReviewedGame) -> GameSummary:
    """The summary of a reviewed game, all but the performance ratings (`profile/performance.py`
    adds them)."""
    per_side: dict[Side, list[tuple[str, float, Label | None]]] = {Side.WHITE: [], Side.BLACK: []}
    graph = [_white_score(chess.Board(game.start.fen), game.start.lines)]
    in_book = game.start.in_book
    last_book: int | None = None
    key: list[KeyMoment] = []
    opening: str | None = game.start.opening
    for move in game.moves:
        before = chess.Board(move.fen_before)
        phase = phase_of(before, in_book=in_book)
        label = move.assessment.label
        per_side[move.side].append((phase, move.assessment.accuracy or 0.0, label))
        if label is Label.BOOK:
            last_book = move.ply
        elif label is not None and label in KEY_LABELS:
            key.append(KeyMoment(ply=move.ply, side=move.side, kind=label.value))
        after = before.copy(stack=False)
        after.push_uci(move.assessment.uci)
        graph.append(_white_score(after, move.after.lines))
        in_book = move.after.in_book
        opening = move.after.opening or opening
    if last_book is not None:
        key.append(
            KeyMoment(ply=last_book, side=game.moves[last_book - 1].side, kind=Label.BOOK.value)
        )
    key.sort(key=lambda k: k.ply)
    headers = game.headers
    return GameSummary(
        white=_side(Side.WHITE, per_side[Side.WHITE], headers),
        black=_side(Side.BLACK, per_side[Side.BLACK], headers),
        opening=opening,
        last_book_ply=last_book,
        key_moments=key,
        graph=[round(g, 4) for g in graph],
        shape=shape_of(graph, headers.get("Result")),
    )


def _side(
    side: Side, moves: list[tuple[str, float, Label | None]], headers: dict[str, str]
) -> SideSummary:
    colour = "White" if side is Side.WHITE else "Black"
    counts: dict[str, int] = {}
    for _, _, label in moves:
        if label is not None:
            counts[label.value] = counts.get(label.value, 0) + 1
    phases = []
    for phase in PHASES:
        inside = [(accuracy, label) for p, accuracy, label in moves if p == phase]
        if len(inside) < MIN_PHASE_MOVES:
            continue
        accuracy = round(fmean(a for a, _ in inside), 1)
        labels = [label for _, label in inside if label is not None]
        phases.append(
            PhaseGrade(
                phase=phase,
                moves=len(inside),
                accuracy=accuracy,
                grade=phase_grade(accuracy, labels),
            )
        )
    accuracy = game_accuracy([a for _, a, _ in moves])
    return SideSummary(
        side=side,
        name=headers.get(colour) or None,
        rating=_int(headers.get(f"{colour}Elo")),
        accuracy=round(accuracy, 1) if accuracy is not None else None,
        counts=counts,
        phases=phases,
    )


def _white_score(board: chess.Board, lines: Sequence[Line]) -> float:
    """White's expected score in a position, from the lines measured for the side to move."""
    score = position_score(board, lines)
    return score if board.turn == chess.WHITE else 1.0 - score


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "?") else None
    except ValueError:
        return None
