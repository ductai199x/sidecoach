"""Which phase of the game a position is in: opening, middlegame or endgame."""

from __future__ import annotations

import chess

_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def phase_of(board: chess.Board, in_book: bool = False) -> str:
    """Opening, middlegame or endgame, from the book and what is left on the board."""
    if in_book or board.fullmove_number <= 8:
        return "opening"
    material = sum(
        _VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.piece_type not in (chess.PAWN, chess.KING)
    )
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if material <= 13 or (queens == 0 and material <= 20):
        return "endgame"
    return "middlegame"
