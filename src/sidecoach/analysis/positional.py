"""Positional facts: the strategic ideas a coach teaches, grounded on the board.

The motif tagger answers "what tactic is here". This module answers the other half of the
teaching goal: "what is the plan". Every fact is computed from the board (and, for the plan,
the engine's own best move), never guessed, and carries the squares it stands on, so the
coach can point at them and the draft checker can verify the claim -- exactly the discipline
the tactical layer follows.

Two kinds. `features` are static truths about the position (an open file, a passed pawn, a
knight on an outpost, an uncastled king). `best_move_idea` reads the strategic intent of the
engine's recommended move (castle, develop, take the file, plant a knight, push a passer,
simplify), which is the highest-value fact because it teaches why the move is good.

Nothing here evaluates: the engine already did. These functions only name, in a coach's
words, what is on the board and what the recommended move is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess

PIECE_NAMES: dict[chess.PieceType, str] = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
_FILES = "abcdefgh"
_HOME_MINORS = {
    chess.WHITE: (chess.B1, chess.C1, chess.F1, chess.G1),
    chess.BLACK: (chess.B8, chess.C8, chess.F8, chess.G8),
}


@dataclass(frozen=True)
class PositionalFact:
    """One strategic truth about the position, in a coach's words, with its evidence squares.

    `kind` is a stable slug (used to rank and to verify); `text` is the sentence a coach would
    say; `squares` are the squares it rests on, for focus and for the draft checker.
    """

    kind: str
    text: str
    squares: tuple[str, ...] = ()


def _name(square: chess.Square) -> str:
    return chess.square_name(square)


def _forward(color: chess.Color) -> int:
    return 1 if color == chess.WHITE else -1


# --- static features -----------------------------------------------------------------------


def is_passed(board: chess.Board, square: chess.Square) -> bool:
    """No enemy pawn can stop or capture the pawn on `square` on its way to promotion."""
    pawn = board.piece_at(square)
    if pawn is None or pawn.piece_type != chess.PAWN:
        return False
    enemy = not pawn.color
    step = _forward(pawn.color)
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    for df in (-1, 0, 1):
        f = file_ + df
        if not 0 <= f <= 7:
            continue
        for other in board.pieces(chess.PAWN, enemy):
            if chess.square_file(other) != f:
                continue
            ahead = (chess.square_rank(other) - rank) * step
            if ahead > 0:
                return False
    return True


def passed_pawns(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    facts: list[PositionalFact] = []
    for square in sorted(board.pieces(chess.PAWN, us)):
        if is_passed(board, square):
            file_ = _FILES[chess.square_file(square)]
            facts.append(
                PositionalFact(
                    "passedPawn",
                    f"your {file_}-pawn on {_name(square)} is a passed pawn",
                    (_name(square),),
                )
            )
    return facts


def _pawns_on_file(board: chess.Board, color: chess.Color, file_: int) -> list[chess.Square]:
    return [sq for sq in board.pieces(chess.PAWN, color) if chess.square_file(sq) == file_]


def is_isolated(board: chess.Board, square: chess.Square) -> bool:
    """A pawn with no friendly pawn on either adjacent file: it can never be defended by one."""
    pawn = board.piece_at(square)
    if pawn is None or pawn.piece_type != chess.PAWN:
        return False
    file_ = chess.square_file(square)
    return not any(
        _pawns_on_file(board, pawn.color, f) for f in (file_ - 1, file_ + 1) if 0 <= f <= 7
    )


def is_backward(board: chess.Board, square: chess.Square) -> bool:
    """A pawn its neighbours have advanced past, whose advance square an enemy pawn guards.

    It cannot advance safely and no friendly pawn is placed to defend it forward, so it is a
    lasting target on its half-open file, the way a coach means "backward pawn".
    """
    pawn = board.piece_at(square)
    if pawn is None or pawn.piece_type != chess.PAWN:
        return False
    color = pawn.color
    step = _forward(color)
    file_, rank = chess.square_file(square), chess.square_rank(square)
    if not 0 <= rank + step <= 7:
        return False
    advance = chess.square(file_, rank + step)
    if not any(board.piece_type_at(a) == chess.PAWN for a in board.attackers(not color, advance)):
        return False
    neighbours = [
        other
        for f in (file_ - 1, file_ + 1)
        if 0 <= f <= 7
        for other in _pawns_on_file(board, color, f)
    ]
    if not neighbours:
        return False  # no neighbours at all is isolated, a different weakness
    # If any neighbour is behind or level, it can come up to defend, so the pawn is not backward.
    return all((chess.square_rank(other) - rank) * step > 0 for other in neighbours)


def _doubled_files(board: chess.Board, color: chess.Color) -> list[tuple[int, list[chess.Square]]]:
    out = []
    for file_ in range(8):
        pawns = _pawns_on_file(board, color, file_)
        if len(pawns) >= 2:
            out.append((file_, sorted(pawns)))
    return out


def pawn_weaknesses(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Structural pawn weaknesses, the enemy's first (targets), then your own (cautions).

    Isolated, backward and doubled pawns, in a coach's words with the pawn's square, so a
    weakness can be named instead of hand-waved at. Enemy weaknesses come first because a
    target to attack is a plan; your own are a caution.
    """
    them = not us
    facts: list[PositionalFact] = []

    def note(color: chess.Color, kind: str) -> None:
        target = color == them
        for square in sorted(board.pieces(chess.PAWN, color)):
            letter = _FILES[chess.square_file(square)]
            if is_isolated(board, square):
                text = (
                    f"their isolated {letter}-pawn on {_name(square)} is a target to pile up on"
                    if target
                    else f"your isolated {letter}-pawn on {_name(square)} needs pieces to defend it"
                )
                facts.append(PositionalFact(kind, text, (_name(square),)))
            elif is_backward(board, square):
                text = (
                    f"their backward {letter}-pawn on {_name(square)} is a lasting target"
                    if target
                    else f"your backward {letter}-pawn on {_name(square)} is a long-term weakness"
                )
                facts.append(PositionalFact(kind, text, (_name(square),)))
        for file_, squares in _doubled_files(board, color):
            letter = _FILES[file_]
            text = (
                f"their doubled {letter}-pawns are a lasting weakness to work against"
                if target
                else f"your doubled {letter}-pawns are a long-term weakness"
            )
            facts.append(PositionalFact(kind, text, tuple(_name(s) for s in squares)))

    note(them, "targetPawn")
    note(us, "weakPawn")
    return facts


def open_files(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Open and half-open files that one of your rooks or queens already stands on."""
    facts: list[PositionalFact] = []
    heavy = {
        square
        for piece_type in (chess.ROOK, chess.QUEEN)
        for square in board.pieces(piece_type, us)
    }
    for file_ in range(8):
        squares = [chess.square(file_, rank) for rank in range(8)]
        pawns = [board.piece_at(sq) for sq in squares]
        our_pawns = any(p and p.piece_type == chess.PAWN and p.color == us for p in pawns)
        their_pawns = any(p and p.piece_type == chess.PAWN and p.color != us for p in pawns)
        on_file = [sq for sq in heavy if chess.square_file(sq) == file_]
        if our_pawns or not on_file:
            continue
        letter = _FILES[file_]
        where = _name(on_file[0])
        piece = PIECE_NAMES[board.piece_type_at(on_file[0])]  # type: ignore[index]
        if their_pawns:
            facts.append(
                PositionalFact(
                    "halfOpenFile",
                    f"your {piece} on {where} has the half-open {letter}-file",
                    (where,),
                )
            )
        else:
            facts.append(
                PositionalFact(
                    "openFile",
                    f"your {piece} on {where} holds the open {letter}-file",
                    (where,),
                )
            )
    return facts


def _pawn_safe(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """No enemy pawn can ever advance to attack `square` (the outpost pawn-safety test)."""
    enemy = not us
    step = _forward(us)
    file_, rank = chess.square_file(square), chess.square_rank(square)
    for other in board.pieces(chess.PAWN, enemy):
        if abs(chess.square_file(other) - file_) != 1:
            continue
        # an enemy pawn ahead of the square on an adjacent file can come down onto it
        if (chess.square_rank(other) - rank) * step >= 1:
            return False
    return True


def _is_outpost(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """A square in enemy territory, defended by one of your pawns, no enemy pawn can hit."""
    rank = chess.square_rank(square)
    reach = rank >= 4 if us == chess.WHITE else rank <= 3
    if not reach:
        return False
    if not any(
        board.piece_type_at(defender) == chess.PAWN for defender in board.attackers(us, square)
    ):
        return False
    return _pawn_safe(board, square, us)


def outposts(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """A knight or bishop of yours already sitting on an outpost -- a durable strong square."""
    facts: list[PositionalFact] = []
    for piece_type in (chess.KNIGHT, chess.BISHOP):
        for square in sorted(board.pieces(piece_type, us)):
            if _is_outpost(board, square, us):
                facts.append(
                    PositionalFact(
                        "outpost",
                        f"your {PIECE_NAMES[piece_type]} on {_name(square)} sits on an outpost, "
                        "a square no enemy pawn can challenge",
                        (_name(square),),
                    )
                )
    return facts


def bishop_pair(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """You keep both bishops while the opponent does not: an edge in an open position."""
    ours = board.pieces(chess.BISHOP, us)
    theirs = board.pieces(chess.BISHOP, not us)
    if len(ours) >= 2 and len(theirs) < 2:
        return [
            PositionalFact(
                "bishopPair",
                "you have the bishop pair, so keep the position open where you can",
                tuple(_name(sq) for sq in sorted(ours)),
            )
        ]
    return []


def bad_bishop(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """A bishop hemmed in by a wall of your own pawns fixed on its colour of square."""
    facts: list[PositionalFact] = []
    for square in sorted(board.pieces(chess.BISHOP, us)):
        light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        blockers = sum(
            1
            for pawn in board.pieces(chess.PAWN, us)
            if ((chess.square_file(pawn) + chess.square_rank(pawn)) % 2 == 1) == light
        )
        mobility = sum(1 for target in board.attacks(square) if board.color_at(target) != us)
        if blockers >= 4 and mobility <= 3:
            facts.append(
                PositionalFact(
                    "badBishop",
                    f"your bishop on {_name(square)} is a bad bishop, blocked in by your own "
                    "pawns; trading it or freeing it is worth thinking about",
                    (_name(square),),
                )
            )
    return facts


def development(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Minor pieces still on their home squares, and whether the king has castled."""
    home = [sq for sq in _HOME_MINORS[us] if board.piece_at(sq) and board.color_at(sq) == us]
    home = [sq for sq in home if board.piece_type_at(sq) in (chess.KNIGHT, chess.BISHOP)]
    facts: list[PositionalFact] = []
    if home:
        names = ", ".join(_name(sq) for sq in home)
        count = "a piece" if len(home) == 1 else f"{len(home)} pieces"
        facts.append(
            PositionalFact(
                "undeveloped",
                f"you still have {count} at home ({names}); they are not in the game yet",
                tuple(_name(sq) for sq in home),
            )
        )
    return facts


def uncastled_king(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """The king still in the centre with pieces about, an idea worth naming past the opening."""
    king = board.king(us)
    if king is None:
        return []
    file_ = chess.square_file(king)
    home_rank = 0 if us == chess.WHITE else 7
    in_centre = 2 <= file_ <= 5 and chess.square_rank(king) == home_rank
    has_rights = board.has_kingside_castling_rights(us) or board.has_queenside_castling_rights(us)
    if in_centre and has_rights:
        return [
            PositionalFact(
                "uncastled",
                f"your king is still in the centre on {_name(king)} and can still castle",
                (_name(king),),
            )
        ]
    return []


def worst_piece(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """A developed minor or rook of yours that is nearly stuck: worth naming to bring to life.

    A piece still on its home square is undeveloped, not passive, so it is left to
    `development`; and the bar is one square or none, so a rook merely waiting behind its pawns
    on a full board is not called out every move.
    """
    back = {chess.A1, chess.H1} if us == chess.WHITE else {chess.A8, chess.H8}
    home = set(_HOME_MINORS[us]) | back
    worst: tuple[int, chess.Square] | None = None
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        for square in board.pieces(piece_type, us):
            if square in home:
                continue
            mobility = sum(1 for target in board.attacks(square) if board.color_at(target) != us)
            if worst is None or mobility < worst[0]:
                worst = (mobility, square)
    if worst is None or worst[0] > 1:
        return []
    square = worst[1]
    return [
        PositionalFact(
            "passivePiece",
            f"your {PIECE_NAMES[board.piece_type_at(square)]} on {_name(square)} is your least "  # type: ignore[index]
            "active piece, with almost nowhere to go",
            (_name(square),),
        )
    ]


def _material_diff(board: chess.Board, us: chess.Color) -> int:
    total = 0
    for piece in board.piece_map().values():
        value = _VALUE.get(piece.piece_type, 0)
        total += value if piece.color == us else -value
    return total


def material_plan(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Up a piece means simplify; down a piece means keep the pieces on. A real plan, grounded."""
    diff = _material_diff(board, us)
    if diff >= 3:
        return [
            PositionalFact(
                "convert",
                "you are up material, so trade pieces where you can and steer for the endgame",
            )
        ]
    if diff <= -3:
        return [
            PositionalFact(
                "complicate",
                "you are down material, so keep pieces on and look for complications",
            )
        ]
    return []


def _center_distance(square: chess.Square) -> int:
    """Chebyshev distance from a square to the central d4-e5 box: 0 in the centre, 3 on the rim."""
    file_, rank = chess.square_file(square), chess.square_rank(square)
    return max(min(abs(file_ - 3), abs(file_ - 4)), min(abs(rank - 3), abs(rank - 4)))


def king_activity(
    board: chess.Board, us: chess.Color, *, phase: str = "middlegame"
) -> list[PositionalFact]:
    """In the endgame, a King left on the rim: it should be marching toward the centre."""
    if phase != "endgame":
        return []
    king = board.king(us)
    if king is None or _center_distance(king) < 3:
        return []
    return [
        PositionalFact(
            "passiveKing",
            f"your King on {_name(king)} is passive; in the endgame the King is a fighting "
            "piece, so bring it toward the centre",
            (_name(king),),
        )
    ]


def features(
    board: chess.Board, us: chess.Color, *, phase: str = "middlegame"
) -> list[PositionalFact]:
    """The static positional facts that hold in the position, most instructive first.

    `phase` gates the ones that only make sense at a stage: development and an uncastled king
    are opening and middlegame ideas, not endgame ones.
    """
    facts: list[PositionalFact] = []
    if phase != "endgame":
        facts += development(board, us)
        facts += uncastled_king(board, us)
    facts += pawn_weaknesses(board, us)
    facts += passed_pawns(board, us)
    facts += outposts(board, us)
    facts += open_files(board, us)
    facts += bad_bishop(board, us)
    facts += bishop_pair(board, us)
    facts += king_activity(board, us, phase=phase)
    facts += material_plan(board, us)
    facts += worst_piece(board, us)
    return facts


# --- the idea behind the best move ---------------------------------------------------------


def _is_central_break(board: chess.Board, move: chess.Move, us: chess.Color) -> bool:
    """A pawn move that strikes at or occupies the centre."""
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return False
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    central_file = 2 <= to_file <= 5
    central_rank = 3 <= to_rank <= 4
    return central_file and (central_rank or board.is_capture(move))


def _rook_to_file_is_open(board: chess.Board, move: chess.Move, us: chess.Color) -> str | None:
    """If a rook moves onto a file with no friendly pawns, the file letter, else None."""
    if board.piece_type_at(move.from_square) != chess.ROOK:
        return None
    file_ = chess.square_file(move.to_square)
    if chess.square_file(move.from_square) == file_:
        return None  # already on this file
    squares = [chess.square(file_, rank) for rank in range(8)]
    if any(
        board.piece_at(sq)
        and board.piece_at(sq).piece_type == chess.PAWN  # type: ignore[union-attr]
        and board.color_at(sq) == us
        for sq in squares
    ):
        return None
    return _FILES[file_]


def _is_central_strong(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """A square in the central box (c3-f6) that no enemy pawn attacks now, so a piece holds it."""
    file_, rank = chess.square_file(square), chess.square_rank(square)
    if not (2 <= file_ <= 5 and 2 <= rank <= 5):
        return False
    return not any(
        board.piece_type_at(attacker) == chess.PAWN for attacker in board.attackers(not us, square)
    )


def _rook_behind_passer(
    after: chess.Board, rook_square: chess.Square, us: chess.Color
) -> str | None:
    """The coaching line if the rook has landed behind a passed pawn on its file, else None.

    Behind means on the side the pawn came from -- the square the classic rule wants a rook,
    to escort your own passer or to hold up the enemy's.
    """
    file_ = chess.square_file(rook_square)
    rook_rank = chess.square_rank(rook_square)
    for rank in range(8):
        pawn = after.piece_at(chess.square(file_, rank))
        if pawn is None or pawn.piece_type != chess.PAWN:
            continue
        if not is_passed(after, chess.square(file_, rank)):
            continue
        promotes_up = pawn.color == chess.WHITE
        behind = rook_rank < rank if promotes_up else rook_rank > rank
        if not behind:
            continue
        letter = _FILES[file_]
        if pawn.color == us:
            return f"put the rook behind your passed {letter}-pawn to escort it forward"
        return f"put the rook behind their passed {letter}-pawn to hold it back"
    return None


def _piece_is_attacked(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """Whether an enemy piece attacks `square` and it is worth more than the cheapest attacker."""
    attackers = board.attackers(not us, square)
    if not attackers:
        return False
    mine = _VALUE.get(board.piece_type_at(square) or chess.PAWN, 0)
    cheapest = min(_VALUE.get(board.piece_type_at(a) or chess.PAWN, 0) for a in attackers)
    defended = bool(board.attackers(us, square))
    return cheapest < mine or not defended


def _prevents_threat(
    board: chess.Board, after: chess.Board, threat: chess.Move, us: chess.Color
) -> bool:
    """Whether the move that led to `after` stops the opponent's threatened move.

    Stopped when the threat is no longer a legal move at all, or when it was a capture of one
    of your pieces that you now guard, so it can be met.
    """
    if threat not in after.legal_moves:
        return True
    if after.is_capture(threat):
        target = threat.to_square
        victim = after.piece_at(target)
        if victim is not None and victim.color == us and after.attackers(us, target):
            return True
    return False


def best_move_idea(
    board: chess.Board,
    move: chess.Move,
    us: chess.Color,
    *,
    phase: str = "middlegame",
    worst: chess.Square | None = None,
    threat: chess.Move | None = None,
) -> PositionalFact | None:
    """The strategic point of the engine's best move, in a coach's words, or None.

    Priority order, most decisive idea first. The move is the engine's; this only names what
    kind of idea it is, from the board geometry, so it can never recommend a move of its own.
    `worst` is the least-active piece if already computed, so "this activates your worst piece"
    can be recognised without recomputing it.
    """
    piece_type = board.piece_type_at(move.from_square)
    if piece_type is None:
        return None
    after = board.copy(stack=False)
    after.push(move)
    to = _name(move.to_square)

    if board.is_castling(move):
        return PositionalFact(
            "castle", "castle to tuck your king away and connect the rooks", (to,)
        )

    if _piece_is_attacked(board, move.from_square, us) and not board.is_capture(move):
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "rescue", f"your {name} was in danger; moving to {to} brings it to safety", (to,)
        )

    if (
        threat is not None
        and not board.is_capture(move)
        and not after.is_check()
        and _prevents_threat(board, after, threat, us)
    ):
        return PositionalFact(
            "prophylaxis",
            "prophylaxis: this quietly stops your opponent's idea before you push your own plan",
            (to,),
        )

    if (
        phase == "endgame"
        and piece_type == chess.KING
        and _center_distance(move.to_square) < _center_distance(move.from_square)
    ):
        return PositionalFact(
            "activateKing",
            f"march your King to {to}, toward the centre; in the endgame it is a strong piece",
            (to,),
        )

    if phase != "endgame" and piece_type in (chess.KNIGHT, chess.BISHOP):
        home = _HOME_MINORS[us]
        if move.from_square in home and not board.is_capture(move):
            name = PIECE_NAMES[piece_type]
            return PositionalFact(
                "develop", f"finish developing: the {name} comes into the game on {to}", (to,)
            )

    if piece_type == chess.KNIGHT and _is_outpost(after, move.to_square, us):
        return PositionalFact(
            "outpostMove",
            f"the knight lands on {to}, an outpost no enemy pawn can chase it from",
            (to,),
        )

    if piece_type in (chess.KNIGHT, chess.BISHOP) and _is_central_strong(after, move.to_square, us):
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "centralise", f"centralise the {name} on {to}, a strong square in the middle", (to,)
        )

    file_letter = _rook_to_file_is_open(board, move, us)
    if file_letter is not None:
        return PositionalFact(
            "takeFile", f"your rook takes the {file_letter}-file, where it has room to work", (to,)
        )

    if piece_type == chess.ROOK:
        behind = _rook_behind_passer(after, move.to_square, us)
        if behind is not None:
            return PositionalFact("rookBehindPasser", behind, (to,))

    if piece_type == chess.PAWN and not board.is_capture(move):
        enemy = not us
        if any(
            after.piece_type_at(target) == chess.PAWN and after.color_at(target) == enemy
            for target in after.attacks(move.to_square)
        ):
            letter = _FILES[chess.square_file(move.to_square)]
            return PositionalFact(
                "break", f"the {letter}-pawn break challenges their pawns and opens lines", (to,)
            )

    if _is_central_break(board, move, us):
        return PositionalFact("centre", f"strike in the centre with the pawn to {to}", (to,))

    if piece_type == chess.PAWN and is_passed(board, move.from_square):
        file_ = _FILES[chess.square_file(move.from_square)]
        return PositionalFact("pushPasser", f"push your passed {file_}-pawn toward {to}", (to,))

    if worst is not None and move.from_square == worst:
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "activate", f"this brings your most passive piece, the {name}, to life on {to}", (to,)
        )

    if phase != "opening" and piece_type == chess.PAWN:
        file_ = chess.square_file(move.to_square)
        if file_ <= 1 or file_ >= 6:
            side = "queenside" if file_ <= 1 else "kingside"
            return PositionalFact("space", f"gain space on the {side} with the pawn to {to}", (to,))

    if board.is_capture(move) and _material_diff(board, us) >= 3:
        return PositionalFact(
            "simplify", "you are up material, so this trade simplifies toward the win", (to,)
        )

    return None


# How instructive each static feature is, most first, so the sheet leads with the one that
# teaches the most and never turns into a laundry list of every open file on the board.
_FEATURE_RANK: dict[str, int] = {
    "uncastled": 0,
    "undeveloped": 1,
    "passiveKing": 2,
    "targetPawn": 3,
    "passedPawn": 4,
    "convert": 5,
    "complicate": 5,
    "outpost": 6,
    "badBishop": 7,
    "openFile": 8,
    "bishopPair": 9,
    "weakPawn": 10,
    "passivePiece": 11,
    "halfOpenFile": 12,
}


def top_features(
    board: chess.Board, us: chess.Color, *, phase: str = "middlegame", limit: int = 2
) -> list[PositionalFact]:
    """The few most instructive static features, one per kind, ranked. The coach's context."""
    seen: set[str] = set()
    unique: list[PositionalFact] = []
    for fact in features(board, us, phase=phase):
        if fact.kind in seen:
            continue
        seen.add(fact.kind)
        unique.append(fact)
    unique.sort(key=lambda fact: _FEATURE_RANK.get(fact.kind, 99))
    return unique[:limit]


def worst_piece_square(board: chess.Board, us: chess.Color) -> chess.Square | None:
    """The square of the least-active minor or rook, or None; shared by features and the idea."""
    facts = worst_piece(board, us)
    if not facts or not facts[0].squares:
        return None
    return chess.parse_square(facts[0].squares[0])


def facts_text(facts: Sequence[PositionalFact]) -> list[str]:
    return [fact.text for fact in facts]
