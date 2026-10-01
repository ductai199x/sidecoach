"""Which phase of the game a position is in: opening, middlegame or endgame -- divided the way
Lichess divides a game (scalachess `Divider`, read 2026-10-01; Tai: "do openings the same way
chess.com or lichess does it"; chess.com does not publish its own).

The middlegame begins at the first position where 10 pieces or fewer are left (kings and pawns
aside), or either side has fewer than 4 of its men on its own back rank (its pieces developed),
or the two sides' men are mixed together on the board (`mixedness` over 150). The endgame
begins at the first position with 6 pieces or fewer. A game is divided once: a phase reached is
never left (`game_phases`). Neither the move number nor the opening book decides it: a board
read from the screen starts at move 1 whatever was played, and the move number called Tai's
puzzle -- two queens, three rooks and a knight -- "the opening" (2026-10-01)."""

from __future__ import annotations

from collections.abc import Iterable

import chess

PHASES = ("opening", "middlegame", "endgame")

_RANK_1 = chess.BB_RANK_1
_RANK_8 = chess.BB_RANK_8
# The 49 two-by-two squares of the board, a1-b2 first, rank by rank, each with its row
# counted from 1 (the y `Divider.score` reads).
_REGIONS = [(0x0303 << (x + 8 * y), y + 1) for y in range(7) for x in range(7)]


def majors_and_minors(board: chess.Board) -> int:
    """The pieces on the board, kings and pawns aside."""
    return chess.popcount(board.occupied & ~(board.kings | board.pawns))


def backrank_sparse(board: chess.Board) -> bool:
    """Fewer than 4 of White's men on the first rank, or of Black's on the eighth: the pieces
    have been developed."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return chess.popcount(_RANK_1 & white) < 4 or chess.popcount(_RANK_8 & black) < 4


def _score(y: int, white: int, black: int) -> int:
    """`Divider.score`: how mixed one two-by-two square is, by how many of each side's men
    stand in it and how far up the board it is."""
    if white == 0:
        if black == 1:
            return 1 + y
        if black == 2:
            return 2 + (6 - y) if y < 6 else 0
        if black in (3, 4):
            return 3 + (7 - y) if y < 7 else 0
        return 0
    if white == 1:
        if black == 0:
            return 1 + (8 - y)
        if black == 1:
            return 5 + abs(4 - y)
        if black == 2:
            return 4 + (7 - y)
        if black == 3:
            return 5 + (7 - y)
        return 0
    if white == 2:
        if black == 0:
            return 2 + (y - 2) if y > 2 else 0
        if black == 1:
            return 4 + (y - 1)
        if black == 2:
            return 7
        return 0
    if white in (3, 4) and black == 0:
        return 3 + (y - 1) if y > 1 else 0
    if white == 3 and black == 1:
        return 5 + (y - 1)
    return 0


def mixedness(board: chess.Board) -> int:
    """How far the two sides' men are mixed together, over every two-by-two square."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    return sum(
        _score(y, chess.popcount(region & white), chess.popcount(region & black))
        for region, y in _REGIONS
    )


def phase_of(board: chess.Board) -> str:
    """The phase this position reads as, on its own."""
    pieces = majors_and_minors(board)
    if pieces <= 6:
        return "endgame"
    if pieces <= 10 or backrank_sparse(board) or mixedness(board) > 150:
        return "middlegame"
    return "opening"


def later(a: str, b: str) -> str:
    """The later of two phases: the game's phase so far, and a position's."""
    return a if PHASES.index(a) >= PHASES.index(b) else b


def game_phases(boards: Iterable[chess.Board]) -> list[str]:
    """Each position's phase in a game, in order: a phase reached is never left, as Lichess's
    division has the middlegame from the first position that reads as one and the endgame
    from the first with 6 pieces or fewer."""
    out: list[str] = []
    now = "opening"
    for board in boards:
        now = later(now, phase_of(board))
        out.append(now)
    return out
