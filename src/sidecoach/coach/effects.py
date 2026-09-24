"""What a move does on the board, computed, so the coach has a true mechanism to say.

The coach's commonest invention was the reason a move was good or bad: "Qf5 hits the
knight", "b3 locks the queenside", "taking on e6 cleared the diagonal" -- claims the board
contradicts. The messages gave it the better move and a plan word, never what the move
does. This module states that, from the board alone: what the move takes, checks, attacks,
defends, uncovers, pins or forks, whether it gets a piece out of attack or leaves one
undefended, which file it takes. Every phrase is checked against the position the move
leads to, so a coach that repeats it is right.

Attack and defence are read off legal moves (`sidecoach.analysis.exchange`), not off the
lines the pieces cover: a knight pinned to its king attacks and defends nothing, a king
takes nothing the other side still guards. An audit on 2026-09-23 found "attacks the knight
on e4" said of a pinned knight and of a king that could not take, "defends the knight,
which was attacked" when a pawn still won it, and "forks" two knights that pawns defended.
The king alone is attacked by the lines: a pinned piece still gives check.
"""

from __future__ import annotations

import chess

from sidecoach.analysis.exchange import VALUE, can_be_won, capturers, defenders, exchange
from sidecoach.coach.facts import PIECE_NAMES


def _name(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    word = PIECE_NAMES[piece.piece_type] if piece else "piece"
    return f"the {word} on {chess.square_name(square)}"


def _hitters(board: chess.Board, colour: chess.Color, square: chess.Square) -> list[chess.Square]:
    """`colour`'s pieces that attack the enemy piece on `square`: those that could take it by
    a legal move (were it `colour`'s move); on the king, those that give it check.

    A check `colour` is in is set aside: it limits the move that answers it, not what stands
    attacked. Read through legal moves, a side in check attacked nothing but the checker, and
    every attack it had came back after the answer as new ("uncovers an attack on the bishop
    by the knight" after taking the checking queen). So then a piece attacks what it could
    take once the check is gone: a pinned one only along its line, the king only onto a
    square nothing guards once it has stepped off its own."""
    if board.piece_type_at(square) == chess.KING:
        return sorted(board.attackers(colour, square))
    if board.turn != colour or not board.is_check():
        return capturers(board, colour, square)
    out: list[chess.Square] = []
    for a in board.attackers(colour, square):
        if board.piece_type_at(a) == chess.KING:
            stepped = board.copy(stack=False)
            stepped.remove_piece_at(a)
            if not stepped.is_attacked_by(not colour, square):
                out.append(a)
        elif square in board.pin(colour, a):
            out.append(a)
    return sorted(out)


def _attacked_by(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    """The enemy pieces `colour` attacks, as squares. The lines are read first, the legal
    moves only where a piece bears on the square: nothing takes where nothing bears."""
    out: set[chess.Square] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color != colour
            and board.is_attacked_by(colour, square)
            and _hitters(board, colour, square)
        ):
            out.add(square)
    return out


def _undefended(board: chess.Board, colour: chess.Color) -> set[chess.Square]:
    """`colour`'s pieces (not the king, not pawns) with no legal defender."""
    out: set[chess.Square] = set()
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if (
            piece is not None
            and piece.color == colour
            and piece.piece_type not in (chess.KING, chess.PAWN)
            and not defenders(board, colour, square)
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


def describe_effects(
    board: chess.Board, move: chess.Move, *, limit: int = 3, hazards: bool = True
) -> list[str]:
    """The move's effects on the board, most telling first, at most `limit` phrases.

    Computed on the position before and after the move; nothing here is inferred. The
    static hazards -- a piece put or left where it is attacked and undefended -- are named
    only with `hazards`: for a move the engine judged sound (its own line, a threat it
    found, a move it called best) the tactics resolve them, and naming them would mislead.
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
    # rest of the army: enemy pieces attacked after that were not attacked before. Attacked
    # means takeable by a legal move, as though it were the mover's turn again.
    before_attacked = _attacked_by(board, us)
    after_attacked = _attacked_by(after, us)

    def value_at(square: chess.Square) -> int:
        piece = after.piece_at(square)
        return VALUE[piece.piece_type] if piece else 0

    # By value, then by square: the operands are sets, so equal-valued pieces came out
    # in CPython's hash order and the coach named a different one on another machine.
    new = sorted(after_attacked - before_attacked, key=lambda s: (-value_at(s), s))
    by_mover = [s for s in new if move.to_square in _hitters(after, us, s)]
    by_others = [s for s in new if s not in by_mover]
    worth = [s for s in by_mover if value_at(s) >= 3]
    # A fork wins something: two targets it can win -- the king, or a piece that can be taken
    # for profit (worth more than the forker, or not defended enough) -- from a square where
    # the forker cannot be taken for profit. A queen touching two knights that pawns guard
    # forks nothing; it attacks one.
    targets = [s for s in worth if value_at(s) == VALUE[chess.KING] or can_be_won(after, s)]
    if len(targets) >= 2 and not can_be_won(after, move.to_square):
        out.append("forks " + " and ".join(_name(after, s) for s in targets[:2]))
    elif by_mover:
        out.append("attacks " + _name(after, by_mover[0]))
    if by_others:
        s = by_others[0]
        attacker = next((a for a in _hitters(after, us, s) if a != move.to_square), None)
        if attacker is not None:
            out.append(f"uncovers an attack on {_name(after, s)} by {_name(after, attacker)}")

    pins = _pinned_enemies(after, us) - _pinned_enemies(board, us)
    for s in sorted(pins):
        # A pinned piece that takes its pinner for profit is not held: its only captures run
        # along the line, onto the pinner (Re1 against a rook on e7, with e1 unguarded).
        if any(
            exchange(after, m) > 0
            for m in after.legal_moves
            if m.from_square == s and after.is_capture(m)
        ):
            continue
        out.append(f"pins {_name(after, s)} to the king")
        break

    # The moved piece: was it attacked, and is it now? Attacked is a legal capture onto its
    # square, defended a legal recapture: a pinned piece neither threatens nor guards it.
    if mover is not None and mover.piece_type != chess.KING:
        was_attacked = bool(capturers(board, them, move.from_square))
        is_attacked = bool(capturers(after, them, move.to_square))
        if was_attacked and not is_attacked:
            out.append(f"gets the {mover_word} out of attack")
        elif hazards and is_attacked and not defenders(after, us, move.to_square):
            out.append(f"puts the {mover_word} where it is attacked and undefended")

    # Own pieces left undefended by the move, and attacked ones newly defended -- only when
    # the defence saves the piece: still attacked, and no longer to be won by the exchange
    # (a bishop guarding a knight that a pawn attacks saves nothing).
    before_loose = _undefended(board, us)
    after_loose = _undefended(after, us)
    left = [s for s in after_loose - before_loose if s != move.to_square] if hazards else []
    for s in sorted(left, key=lambda s: (-value_at(s), s)):
        tail = ", and it is attacked" if capturers(after, them, s) else ""
        out.append(f"leaves {_name(after, s)} undefended{tail}")
        break
    covered = [
        s
        for s in before_loose - after_loose
        if s != move.from_square and capturers(after, them, s) and not can_be_won(after, s)
    ]
    for s in sorted(covered, key=lambda s: (-value_at(s), s)):
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


def effects_text(
    board: chess.Board, move: chess.Move, *, limit: int = 3, hazards: bool = True
) -> str:
    """The effects joined for a message, or an empty string when the move does nothing
    the board can name (a quiet developing move)."""
    return "; ".join(describe_effects(board, move, limit=limit, hazards=hazards))
