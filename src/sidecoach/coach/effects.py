"""What a move does on the board, computed, so the coach has a true mechanism to say.

The coach's commonest invention was the reason a move was good or bad: "Qf5 hits the
knight", "b3 locks the queenside", "taking on e6 cleared the diagonal" -- claims the board
contradicts. The messages gave it the better move and a plan word, never what the move
does. This module states that, from the board alone: what the move takes, checks, attacks,
defends, uncovers, pins or forks, whether it gets a piece out of attack or leaves one
undefended, which file it takes. Every phrase is checked against the position the move
leads to, so a coach that repeats it is right.
"""

from __future__ import annotations

import chess

from sidecoach.coach.facts import PIECE_NAMES

VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _name(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    word = PIECE_NAMES[piece.piece_type] if piece else "piece"
    return f"the {word} on {chess.square_name(square)}"


def _attacked_by(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    """The enemy pieces `colour` attacks, as squares."""
    out: set[chess.Square] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color != colour and board.is_attacked_by(colour, square):
            out.add(square)
    return out


def _undefended(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    """`colour`'s pieces (not the king, not pawns) with no defender."""
    out: set[chess.Square] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color == colour
            and piece.piece_type not in (chess.KING, chess.PAWN)
            and not board.is_attacked_by(colour, square)
        ):
            out.add(square)
    return out


def _pinned_enemies(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    """Enemy pieces pinned to their king."""
    out: set[chess.Square] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color != colour and board.is_pinned(not colour, square):
            out.add(square)
    return out


def describe_effects(board: chess.Board, move: chess.Move, *, limit: int = 3) -> list[str]:
    """The move's effects on the board, most telling first, at most `limit` phrases.

    Computed on the position before and after the move; nothing here is inferred.
    """
    if move not in board.legal_moves:
        return []
    us = board.turn
    them = not us
    after = board.copy(stack=False)
    after.push(move)
    mover = board.piece_at(move.from_square)
    mover_word = PIECE_NAMES[mover.piece_type] if mover else "piece"
    out: list[str] = []

    if after.is_checkmate():
        return ["checkmate"]
    if board.is_castling(move):
        out.append("castles, the king out of the centre")
    if board.is_capture(move):
        taken = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            out.append("takes the pawn en passant")
        elif taken is not None:
            out.append(
                f"takes the {PIECE_NAMES[taken.piece_type]} on {chess.square_name(move.to_square)}"
            )
    if after.is_check():
        out.append("gives check")

    # What the moved piece attacks from its new square, and what the move uncovers for the
    # rest of the army: enemy pieces attacked after that were not attacked before.
    before_attacked = _attacked_by(board, us)
    after_attacked = _attacked_by(after, us)
    def value_at(square: chess.Square) -> int:
        piece = after.piece_at(square)
        return VALUE[piece.piece_type] if piece else 0

    new = sorted(after_attacked - before_attacked, key=lambda s: -value_at(s))
    by_mover = [s for s in new if move.to_square in after.attackers(us, s)]
    by_others = [s for s in new if s not in by_mover]
    worth = [s for s in by_mover if value_at(s) >= 3]
    if len(worth) >= 2:
        out.append("forks " + " and ".join(_name(after, s) for s in worth[:2]))
    elif by_mover:
        out.append("attacks " + _name(after, by_mover[0]))
    if by_others:
        s = by_others[0]
        attacker = next(iter(after.attackers(us, s) - {move.to_square}), None)
        if attacker is not None:
            out.append(f"uncovers an attack on {_name(after, s)} by {_name(after, attacker)}")

    pins = _pinned_enemies(after, us) - _pinned_enemies(board, us)
    for s in sorted(pins):
        out.append(f"pins {_name(after, s)} to the king")
        break

    # The moved piece: was it attacked, and is it now?
    if mover is not None and mover.piece_type != chess.KING:
        was_attacked = board.is_attacked_by(them, move.from_square)
        is_attacked = after.is_attacked_by(them, move.to_square)
        if was_attacked and not is_attacked:
            out.append(f"gets the {mover_word} out of attack")
        elif is_attacked and not after.is_attacked_by(us, move.to_square):
            out.append(f"puts the {mover_word} where it is attacked and undefended")

    # Own pieces left undefended by the move, and attacked ones newly defended.
    before_loose = _undefended(board, us)
    after_loose = _undefended(after, us)
    left = [s for s in after_loose - before_loose if s != move.to_square]
    for s in sorted(left, key=lambda s: -value_at(s)):
        tail = ", and it is attacked" if after.is_attacked_by(them, s) else ""
        out.append(f"leaves {_name(after, s)} undefended{tail}")
        break
    covered = [
        s
        for s in before_loose - after_loose
        if s != move.from_square and after.is_attacked_by(them, s)
    ]
    for s in sorted(covered, key=lambda s: -value_at(s)):
        out.append(f"defends {_name(after, s)}, which was attacked")
        break

    # A rook or queen onto a file with no pawn of its own on it.
    if mover is not None and mover.piece_type in (chess.ROOK, chess.QUEEN):
        file_index = chess.square_file(move.to_square)
        own_pawn = any(
            board.piece_at(chess.square(file_index, r)) == chess.Piece(chess.PAWN, us)
            for r in range(8)
        )
        their_pawn = any(
            board.piece_at(chess.square(file_index, r)) == chess.Piece(chess.PAWN, them)
            for r in range(8)
        )
        if chess.square_file(move.from_square) != file_index and not own_pawn:
            kind = "open" if not their_pawn else "half-open"
            out.append(f"takes the {kind} {chess.FILE_NAMES[file_index]}-file")

    return out[:limit]


def effects_text(board: chess.Board, move: chess.Move, *, limit: int = 3) -> str:
    """The effects joined for a message, or an empty string when the move does nothing
    the board can name (a quiet developing move)."""
    return "; ".join(describe_effects(board, move, limit=limit))
