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


_ORDER = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
_NAMES = {
    chess.QUEEN: "queen",
    chess.ROOK: "rook",
    chess.BISHOP: "bishop",
    chess.KNIGHT: "knight",
    chess.PAWN: "pawn",
}
_COUNT = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def _pieces_said(counts: dict[chess.PieceType, int]) -> str:
    """ "a rook and two pawns"."""
    parts = [
        f"{_COUNT.get(n, str(n))} {_NAMES[kind]}{'s' if n > 1 else ''}"
        for kind in _ORDER
        if (n := counts.get(kind, 0)) > 0
    ]
    if not parts:
        return ""
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"


def _trade(board: chess.Board, after: chess.Board, us: chess.Color, diff: int) -> str:
    """What the line takes and gives, piece by piece: "the exchange: a rook for a bishop",
    "a knight and a pawn". A sum in pawns' worth said a rook for a bishop as "two pawns",
    and the model added the two pawns to the exchange (the test audit, 2026-09-26). The
    pawns' worth stands where a pawn promotes, when the pieces no longer add up."""
    them = not us
    lost: dict[chess.Color, dict[chess.PieceType, int]] = {}
    for colour in (us, them):
        counts = {}
        for kind in _ORDER:
            change = len(board.pieces(kind, colour)) - len(after.pieces(kind, colour))
            if change < 0:
                return _amount(diff)  # a piece made on the way: the sum, not the pieces
            counts[kind] = change
        lost[colour] = counts
    won, given = lost[them], lost[us]
    for kind in _ORDER:  # like for like is no gain
        n = min(won[kind], given[kind])
        won[kind] -= n
        given[kind] -= n
    for a, b in ((chess.BISHOP, chess.KNIGHT), (chess.KNIGHT, chess.BISHOP)):
        n = min(won[a], given[b])  # a bishop for a knight is a trade too
        won[a] -= n
        given[b] -= n
    taken, paid = _pieces_said(won), _pieces_said(given)
    if not taken:
        return _amount(diff)
    minors = given[chess.BISHOP] + given[chess.KNIGHT]
    if taken == "a rook" and minors == 1 and not given[chess.PAWN]:
        return f"the exchange: a rook for {paid}"
    return f"{taken} for {paid}" if paid else taken


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
    """What the line changes, from the player's side, written for the harness: "the player
    wins a pawn; the player gets a passed pawn on d5; the e-file opens; the player's king
    castles". Empty when nothing of note changes."""
    after, played, castled = play_out(board, moves)
    if played == 0:
        return ""
    us, them = user_colour, not user_colour
    parts: list[str] = []
    diff = _material(after, us) - _material(board, us)
    if diff > 0:
        parts.append(f"the player wins {_trade(board, after, us, diff)}")
    elif diff < 0:
        parts.append(f"the player gives up {_trade(board, after, them, -diff)}")
    new_ours = _passed(after, us) - _passed(board, us)
    new_theirs = _passed(after, them) - _passed(board, them)
    if new_ours:
        parts.append(
            "the player gets a passed pawn on "
            + " and ".join(chess.square_name(s) for s in sorted(new_ours))
        )
    if new_theirs:
        parts.append(
            "the opponent gets a passed pawn on "
            + " and ".join(chess.square_name(s) for s in sorted(new_theirs))
        )
    opened = _open_files(after) - _open_files(board)
    if opened:
        parts.append("the " + " and ".join(f"{_FILES[f]}-file" for f in sorted(opened)) + " opens")
    for colour, whose in ((us, "the player's"), (them, "the opponent's")):
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
        parts.append("the opponent is mated" if after.turn == them else "the player is mated")
    return "; ".join(parts)
