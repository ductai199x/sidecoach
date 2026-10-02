"""The moves besides the best one: which are about as good, which are clearly worse, and what
goes wrong with the worse, from the engine's lines for the position.

The best move is not always one move: in a quiet position several are within a few hundredths of
the best line's score and any of them is fine. A coach that names only line 1 tells a player
who found another that they were wrong. And the move that is clearly better than the one the
player considered is better for a reason the board shows: the mate it passes up, the piece the
line gives away. Only what the board and the engine's line settle is said (a line cut off in
the middle of an exchange says nothing of material), and never a figure.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from sidecoach.coach.outcome import line_outcome, play_out
from sidecoach.events import Analysis, Line

# A line this far behind the best, in expected score, starts a clearly worse move.
WORSE_MARGIN = 0.10
# How far a line is read: the sheet's own (`conversation.LINE_PLIES`).
READ_PLIES = 8
# The most worse moves named in an answer.
MOST_NAMED = 2


@dataclass(frozen=True)
class Worse:
    """A move clearly behind the best, with what goes wrong with it when the board says: a
    verb phrase of the move -- "gives up a rook for a pawn", "passes up the mate"."""

    san: str
    drop: float
    wrong: str


def _san(board: chess.Board, line: Line) -> str | None:
    try:
        move = chess.Move.from_uci(line.moves[0])
    except (ValueError, IndexError):
        return None
    return board.san(move) if move in board.legal_moves else None


def _settled(board: chess.Board, moves: list[str], player: chess.Color) -> list[str] | None:
    """What the line changes, from the player's side, when it is settled: it ends on a move
    that takes nothing and gives no check, so no exchange is cut off in the middle. None
    when it is not, and nothing is said of it then."""
    after, played, _ = play_out(board, moves)
    if played == 0:
        return None
    last = board.copy(stack=False)
    taking = False
    for uci in moves[:played]:
        move = chess.Move.from_uci(uci)
        taking = last.is_capture(move)
        last.push(move)
    if taking or after.is_check():
        return None
    return [p for p in line_outcome(board, moves[:played], player).split("; ") if p]


def _part(parts: list[str] | None, lead: str) -> str | None:
    """The rest of the outcome's part that begins `lead`, if any."""
    for part in parts or []:
        if part.startswith(lead):
            return part.removeprefix(lead)
    return None


def wrong_with(board: chess.Board, best: Line, line: Line, player: chess.Color) -> str | None:
    """What goes wrong with the move `line` starts, against the best line, or None when the
    board does not say: the mate it passes up, the mate it walks into, what the best line wins
    that this one does not, what this one gives up."""
    if best.mate is not None and best.mate > 0 and not (line.mate is not None and line.mate > 0):
        return "passes up the mate"
    if line.mate is not None and line.mate < 0:
        return "walks into a mate"
    ours = _settled(board, line.moves[:READ_PLIES], player)
    theirs = _settled(board, best.moves[:READ_PLIES], player)
    won = _part(theirs, "the player wins ")
    if won is not None and _part(ours, "the player wins ") != won and ours is not None:
        return f"passes up winning {won}"
    lost = _part(ours, "the player gives up ")
    return f"gives up {lost}" if lost is not None else None


def worse_moves(
    board: chess.Board, analysis: Analysis, player: chess.Color, *, limit: int = MOST_NAMED
) -> list[Worse]:
    """The moves clearly behind the best that something can be said of, the nearest to the
    best first -- the tempting ones -- at most `limit`. Empty when the player is not the one
    to move, or no line is clearly behind."""
    lines = [line for line in analysis.lines if line.moves]
    if board.turn != player or len(lines) < 2:
        return []
    best = lines[0]
    named: list[Worse] = []
    seen = {_san(board, best)}
    for line in lines[1:]:
        drop = best.expected_score - line.expected_score
        san = _san(board, line)
        if drop < WORSE_MARGIN or san is None or san in seen:
            continue
        seen.add(san)
        wrong = wrong_with(board, best, line, player)
        if wrong:
            named.append(Worse(san, drop, wrong))
        if len(named) == limit:
            break
    return named


def worse_text(worse: list[Worse]) -> str:
    """The worse moves for the model: "Bd3 gives up a bishop; Qd1 passes up the mate"."""
    return "; ".join(f"{w.san} {w.wrong}" for w in worse)


def worse_said(worse: list[Worse], *, past: bool = False) -> str:
    """The same as the coach says it to the player, for the template: a sentence per move.
    `past`: of the position the player has moved from ("was clearly worse there")."""
    verb = "was clearly worse there" if past else "is clearly worse"
    return " ".join(f"{w.san} {verb}: it {w.wrong}." for w in worse)


def fine_said(fine: list[str], *, past: bool = False) -> str:
    """ "Nf6, Bc5 and Be7 are all fine here -- about equally good.", or the one move that is.
    `past`: of the position the player has moved from ("were all fine there")."""
    if len(fine) == 1:
        return f"{fine[0]} was the move there." if past else f"{fine[0]} is the move here."
    names = ", ".join(fine[:-1]) + f" and {fine[-1]}"
    if past:
        return f"{names} were all fine there -- about equally good."
    return f"{names} are all fine here -- about equally good."
