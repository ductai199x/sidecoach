"""Where a line leads: what is different on the board at the end of it, in a coach's words.

The engine's line is a move sequence with a score; the *point* of the line is what it
changes -- a pawn given up for a passed pawn, a file opened, a king brought to safety, a
piece won. Those are computed by playing the line out and comparing the two boards, so
the coach can say what the line is for without inventing it.
"""

from __future__ import annotations

import chess

from sidecoach.analysis.positional import is_passed

_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_FILES = "abcdefgh"


def _material(board: chess.Board, us: chess.Color) -> int:
    total = 0
    for piece in board.piece_map().values():
        value = _VALUES.get(piece.piece_type, 0)
        total += value if piece.color == us else -value
    return total


def _passed(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    return {s for s in board.pieces(chess.PAWN, colour) if is_passed(board, s)}


def _open_files(board: chess.Board) -> set[int]:
    pawn_files = {chess.square_file(s) for c in chess.COLORS for s in board.pieces(chess.PAWN, c)}
    return set(range(8)) - pawn_files


def _amount(diff: int) -> str:
    """Material in pawns' worth: what a line wins is often a mix, so no piece is named
    for a sum ("a knight and a pawn" would be a guess at which pieces)."""
    n = abs(diff)
    if n == 1:
        return "a pawn"
    if n == 2:
        return "two pawns"
    if n == 3:
        return "a piece's worth of material"
    return f"material worth {n} pawns"


def play_out(
    board: chess.Board, moves: list[str]
) -> tuple[chess.Board, int, dict[chess.Color, bool]]:
    """The board after `moves` (UCI), as far as they are legal; how many were played; and
    who castled along the way."""
    after = board.copy(stack=False)
    castled = {chess.WHITE: False, chess.BLACK: False}
    played = 0
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in after.legal_moves:
            break
        if after.is_castling(move):
            castled[after.turn] = True
        after.push(move)
        played += 1
    return after, played, castled


def line_outcome(board: chess.Board, moves: list[str], user_colour: chess.Color) -> str:
    """What the line changes, from the player's side: "you win a pawn; you get a passed
    pawn on d5; the e-file opens; your king castles". Empty when nothing of note changes."""
    after, played, castled = play_out(board, moves)
    if played == 0:
        return ""
    us, them = user_colour, not user_colour
    parts: list[str] = []
    diff = _material(after, us) - _material(board, us)
    if diff > 0:
        parts.append(f"you win {_amount(diff)}")
    elif diff < 0:
        parts.append(f"you give up {_amount(diff)}")
    new_ours = _passed(after, us) - _passed(board, us)
    new_theirs = _passed(after, them) - _passed(board, them)
    if new_ours:
        parts.append(
            "you get a passed pawn on "
            + " and ".join(chess.square_name(s) for s in sorted(new_ours))
        )
    if new_theirs:
        parts.append(
            "they get a passed pawn on "
            + " and ".join(chess.square_name(s) for s in sorted(new_theirs))
        )
    opened = _open_files(after) - _open_files(board)
    if opened:
        parts.append("the " + " and ".join(f"{_FILES[f]}-file" for f in sorted(opened)) + " opens")
    for colour, whose in ((us, "your"), (them, "their")):
        if castled[colour]:
            parts.append(f"{whose} king castles")
        else:
            before_king, after_king = board.king(colour), after.king(colour)
            if before_king is not None and after_king is not None and before_king != after_king:
                parts.append(f"{whose} king goes to {chess.square_name(after_king)}")
    both_queens = board.pieces(chess.QUEEN, us) and board.pieces(chess.QUEEN, them)
    if both_queens and not after.pieces(chess.QUEEN, us) and not after.pieces(chess.QUEEN, them):
        parts.append("queens come off")
    if after.is_checkmate():
        parts.append("it is mate" if after.turn == them else "you are mated")
    return "; ".join(parts)
