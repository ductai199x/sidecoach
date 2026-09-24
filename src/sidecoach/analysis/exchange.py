"""Who can take on a square, who can take back, and what a capture nets: by the rules.

Every word the coach says about attack and defence -- "hanging", "defended", "safe", "can
be won", "attacks", "a real threat", "a fork", "a pin" -- comes down to these questions, and
`board.attackers()` answers them wrongly for a player: it counts a piece pinned to its own
king, which cannot take, and a king taking on a square the other side still guards. An
audit on 2026-09-23 reproduced 38 false sentences from that one gap ("their knight on e4 is
hanging, attacked by the knight on c3" with the knight pinned; "defended by the e8 king" of
f7 with the queen guarding it). So the answers here are read off legal moves: a capture is
a legal move, a recapture is a legal move on the board after the capture, and an exchange
is played out move by move, least valuable piece first -- which settles pins, checks, a
king that cannot take back, and the x-ray piece behind a capturer, with no special cases.

`board.attackers()` stays right for king safety: a pinned piece still keeps the enemy king
off the squares it covers.
"""

from __future__ import annotations

import chess

VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _as_mover(board: chess.Board, color: chess.Color) -> chess.Board:
    """`board` with `color` to move: the same position, the move handed over (no en passant
    right carried across, no castling taken as a capture)."""
    if board.turn == color:
        return board
    view = board.copy(stack=False)
    view.push(chess.Move.null())
    return view


def captures_on(board: chess.Board, color: chess.Color, square: chess.Square) -> list[chess.Move]:
    """The legal moves of `color` that take on `square` (were it `color`'s move)."""
    view = _as_mover(board, color)
    target = view.piece_at(square)
    if target is None or target.color == color:
        return []
    return list(_onto(view, square))


def capturers(board: chess.Board, color: chess.Color, square: chess.Square) -> list[chess.Square]:
    """The pieces of `color` that can legally take on `square`: none pinned off the line to
    their own king, the king only where it would not be taken back."""
    return sorted({m.from_square for m in captures_on(board, color, square)})


def defenders(board: chess.Board, color: chess.Color, square: chess.Square) -> list[chess.Square]:
    """The pieces of `color` that could legally take back on `square` were an enemy piece to
    take there: none pinned to its own king off the line, and the king only when the capture
    is the only enemy piece bearing on the square (a second one would guard it)."""
    enemy = board.attackers(not color, square)
    out: list[chess.Square] = []
    for a in board.attackers(color, square):
        if board.piece_type_at(a) == chess.KING:
            if len(enemy) <= 1:
                out.append(a)
        elif square in board.pin(color, a):
            out.append(a)
    return sorted(out)


def exchange(board: chess.Board, move: chess.Move) -> int:
    """What the capture `move` nets its side, in pawns, after the recaptures on its square
    played out with legal moves, least valuable piece first, each side free to stop. A
    non-capture scores 0. Promotions are counted as the pawn they were."""
    square = move.to_square
    if board.is_en_passant(move):
        taken = chess.PAWN
    else:
        victim = board.piece_at(square)
        if victim is None:
            return 0
        taken = victim.piece_type
    on_square = board.piece_type_at(move.from_square) or chess.PAWN
    gains = [VALUE[taken]]
    b = board.copy(stack=False)
    b.push(move)
    while True:
        replies = list(_onto(b, square))
        if not replies:
            break
        reply = min(replies, key=lambda m: VALUE[b.piece_type_at(m.from_square) or chess.PAWN])
        gains.append(VALUE[on_square] - gains[-1])
        on_square = b.piece_type_at(reply.from_square) or chess.PAWN
        b.push(reply)
    while len(gains) > 1:
        last = gains.pop()
        gains[-1] = -max(-gains[-1], last)
    return gains[0]


def best_capture(board: chess.Board, color: chess.Color, square: chess.Square) -> int:
    """The most `color` nets by taking on `square` (were it `color`'s move); 0 when it cannot
    take or would lose by it."""
    view = _as_mover(board, color)
    return max([0, *(exchange(view, m) for m in captures_on(board, color, square))])


def can_be_won(board: chess.Board, square: chess.Square) -> bool:
    """The piece on `square` can be taken for profit by the other side."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return False
    return best_capture(board, not piece.color, square) > 0


def hanging(board: chess.Board, square: chess.Square) -> bool:
    """The piece on `square` is attacked by a legal capturer and nothing can take back."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return False
    for move in captures_on(board, not piece.color, square):
        after = _as_mover(board, not piece.color).copy(stack=False)
        after.push(move)
        if not any(True for _ in _onto(after, square)):
            return True
    return False


def can_retake(board: chess.Board, capture: chess.Move) -> bool:
    """After `capture`, the other side can take back on its square."""
    after = board.copy(stack=False)
    after.push(capture)
    return any(True for _ in _onto(after, capture.to_square))


def _onto(board: chess.Board, square: chess.Square):  # noqa: ANN202
    """The legal moves of the side to move onto `square`, generated for that square only."""
    return board.generate_legal_moves(chess.BB_ALL, chess.BB_SQUARES[square])
