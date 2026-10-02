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

from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange

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
# Which minor piece starts on each of those files.
_HOME_TYPE = {1: chess.KNIGHT, 2: chess.BISHOP, 5: chess.BISHOP, 6: chess.KNIGHT}


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


# The square of the pawn (endgame): can the defending king catch a running passed pawn on its
# own? The rule matches the Lichess tablebase on pure races: the king catches it iff its
# distance to the promotion square is within the pawn's moves to promote, plus one when the
# defender is to move. Only the decisive verdicts are named -- a pawn that queens untouched, or
# one that outruns the king -- so the coach never over-claims in a crowded middlegame.
def _king_catches_pawn(
    pawn: chess.Square, pawn_color: chess.Color, king: chess.Square, defender_to_move: bool
) -> bool:
    pf, pr = chess.square_file(pawn), chess.square_rank(pawn)
    kf, kr = chess.square_file(king), chess.square_rank(king)
    if pawn_color == chess.WHITE:
        moves = (7 - pr) - (1 if pr == 1 else 0)
        promotion_rank = 7
    else:
        moves = pr - (1 if pr == 6 else 0)
        promotion_rank = 0
    king_distance = max(abs(kf - pf), abs(kr - promotion_rank))
    return king_distance <= moves + (1 if defender_to_move else 0)


def _clear_run(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """Every square ahead of the pawn on its file is empty: a clean run to promotion."""
    step = _forward(us)
    rank = chess.square_rank(square) + step
    while 0 <= rank <= 7:
        if board.piece_at(chess.square(chess.square_file(square), rank)) is not None:
            return False
        rank += step
    return True


def square_of_the_pawn(
    board: chess.Board, us: chess.Color, *, phase: str = "middlegame"
) -> list[PositionalFact]:
    """Decisive passed-pawn races in the endgame: a pawn that queens untouched, or one that
    outruns the enemy king. Gated to the endgame, where a lone king really must chase a pawn."""
    if phase != "endgame":
        return []
    facts: list[PositionalFact] = []
    them = not us
    enemy_king, our_king = board.king(them), board.king(us)
    if enemy_king is not None:
        for sq in sorted(board.pieces(chess.PAWN, us)):
            if (
                is_passed(board, sq)
                and _clear_run(board, sq, us)
                and not _king_catches_pawn(sq, us, enemy_king, board.turn == them)
            ):
                letter = _FILES[chess.square_file(sq)]
                facts.append(
                    PositionalFact(
                        "runningPawn",
                        f"your passed {letter}-pawn on {_name(sq)} runs to queen on its own -- "
                        "the enemy king is outside its square",
                        (_name(sq),),
                    )
                )
    if our_king is not None:
        for sq in sorted(board.pieces(chess.PAWN, them)):
            if (
                is_passed(board, sq)
                and _clear_run(board, sq, them)
                and not _king_catches_pawn(sq, them, our_king, board.turn == us)
            ):
                letter = _FILES[chess.square_file(sq)]
                facts.append(
                    PositionalFact(
                        "outrunPawn",
                        f"their passed {letter}-pawn on {_name(sq)} is outside your king's "
                        "square -- you cannot catch it alone, so stop it another way",
                        (_name(sq),),
                    )
                )
    return facts


def _only_kings_and_pawns(board: chess.Board) -> bool:
    return all(p.piece_type in (chess.KING, chess.PAWN) for p in board.piece_map().values())


def opposition(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Who holds the opposition in a king-and-pawn endgame.

    The kings facing off with an odd number of squares between them force the other side to
    give way -- a true geometric fact, stated neutrally, since the opposition only wins in the
    right pawn structure (a king far enough in front of its pawn wins without it). Fires only
    with kings and pawns alone on the board, where the opposition is genuinely the battle.
    """
    if not _only_kings_and_pawns(board) or not board.pawns:
        return []
    white_king, black_king = board.king(chess.WHITE), board.king(chess.BLACK)
    if white_king is None or black_king is None:
        return []
    files = abs(chess.square_file(white_king) - chess.square_file(black_king))
    ranks = abs(chess.square_rank(white_king) - chess.square_rank(black_king))
    aligned = (
        (files == 0 and ranks in (2, 4, 6))
        or (ranks == 0 and files in (2, 4, 6))
        or (files == ranks and files in (2, 4, 6))
    )
    if not aligned:
        return []
    label = "opposition" if max(files, ranks) == 2 else "distant opposition"
    squares = (_name(white_king), _name(black_king))
    holder = not board.turn  # the side to move must give way; the other holds the opposition
    if holder == us:
        return [
            PositionalFact(
                "opposition", f"you have the {label}: the enemy king must give way first", squares
            )
        ]
    return [
        PositionalFact(
            "oppositionAgainst",
            f"your opponent has the {label}, so your king must give way -- look to seize it back",
            squares,
        )
    ]


def _key_squares_of(pawn: chess.Square, us: chess.Color) -> list[chess.Square]:
    """The king's key squares for a non-rook pawn: reach one and the pawn queens. The ranks
    were checked against the Lichess tablebase -- two in front for a pawn on its 2nd-4th rank,
    and both one and two in front once it reaches the 5th. A rook pawn has none (usually a draw)."""
    pf, pr = chess.square_file(pawn), chess.square_rank(pawn)
    if pf in (0, 7):
        return []
    if us == chess.WHITE:
        ranks = [pr + 2] if pr <= 3 else [pr + 1, pr + 2]
    else:
        ranks = [pr - 2] if pr >= 4 else [pr - 1, pr - 2]
    return [
        chess.square(f, kr)
        for kr in ranks
        for f in (pf - 1, pf, pf + 1)
        if 0 <= f <= 7 and 0 <= kr <= 7
    ]


def key_squares(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """The king's key squares in a king-and-pawn-versus-king ending: reach one and the pawn
    promotes. Only the clean textbook case -- your lone non-rook pawn, bare kings -- where the
    rule is exact."""
    if not _only_kings_and_pawns(board):
        return []
    ours = list(board.pieces(chess.PAWN, us))
    if len(ours) != 1 or board.pieces(chess.PAWN, not us):
        return []
    keys = _key_squares_of(ours[0], us)
    king = board.king(us)
    if not keys or king is None:
        return []
    letter = _FILES[chess.square_file(ours[0])]
    if king in keys:
        return [
            PositionalFact(
                "keySquare",
                f"your king holds a key square on {_name(king)} -- your {letter}-pawn promotes, so "
                "push it through",
                (_name(king), _name(ours[0])),
            )
        ]
    names = ", ".join(_name(k) for k in sorted(keys))
    return [
        PositionalFact(
            "keySquareGoal",
            f"march your king to a key square ({names}) to force your {letter}-pawn through",
            (_name(ours[0]), *(_name(k) for k in sorted(keys))),
        )
    ]


def _rook_and_pawn_vs_rook(board: chess.Board, us: chess.Color) -> bool:
    """Exactly a rook-and-pawn versus rook ending: your K+R+one pawn against their K+R."""
    them = not us

    def n(piece_type: chess.PieceType, color: chess.Color) -> int:
        return len(board.pieces(piece_type, color))

    return (
        n(chess.PAWN, us) == 1
        and n(chess.PAWN, them) == 0
        and n(chess.ROOK, us) == 1
        and n(chess.ROOK, them) == 1
        and n(chess.QUEEN, chess.WHITE) == 0
        and n(chess.QUEEN, chess.BLACK) == 0
        and n(chess.BISHOP, chess.WHITE) == 0
        and n(chess.BISHOP, chess.BLACK) == 0
        and n(chess.KNIGHT, chess.WHITE) == 0
        and n(chess.KNIGHT, chess.BLACK) == 0
    )


def lucena(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """The Lucena position: a won rook-and-pawn ending. Your king stands on the queening square
    in front of a pawn on the seventh, the enemy king cut off, and you win by building a bridge.
    Only for a non-rook pawn -- a rook pawn is won, if at all, by another method."""
    if not _rook_and_pawn_vs_rook(board, us):
        return []
    pawn = next(iter(board.pieces(chess.PAWN, us)))
    pf, pr = chess.square_file(pawn), chess.square_rank(pawn)
    seventh = 6 if us == chess.WHITE else 1
    promotion = chess.square(pf, 7 if us == chess.WHITE else 0)
    king, defender = board.king(us), board.king(not us)
    if pf in (0, 7) or pr != seventh or king is None or king != promotion or defender is None:
        return []
    if abs(chess.square_file(defender) - pf) < 1 or chess.square_distance(defender, pawn) < 2:
        return []  # the enemy king is not cut off from the pawn
    return [
        PositionalFact(
            "lucena",
            f"a Lucena position: this rook-and-pawn ending is won -- build a bridge with your rook "
            f"to shelter your king from the checks, then bring it out from in front of the "
            f"{_FILES[pf]}-pawn",
            (_name(king), _name(pawn)),
        )
    ]


def philidor(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """The Philidor position: the drawing defence in a rook-and-pawn ending. Your king stands in
    front of the enemy pawn and your rook waits on your own third rank, keeping the enemy king
    off the sixth; once the pawn advances there, you check it from behind. Recognised while the
    third-rank defence still holds -- the pawn not yet on its sixth rank."""
    them = not us
    if not _rook_and_pawn_vs_rook(board, them):  # they own the pawn, we defend with K + R
        return []
    pawn = next(iter(board.pieces(chess.PAWN, them)))
    pf, pr = chess.square_file(pawn), chess.square_rank(pawn)
    king = board.king(us)
    rook = next(iter(board.pieces(chess.ROOK, us)))
    if king is None or chess.square_file(king) != pf:
        return []
    in_front = chess.square_rank(king) > pr if them == chess.WHITE else chess.square_rank(king) < pr
    third_rank = 2 if us == chess.WHITE else 5
    on_third = chess.square_rank(rook) == third_rank
    not_yet_sixth = pr <= 4 if them == chess.WHITE else pr >= 3
    if in_front and on_third and not_yet_sixth:
        return [
            PositionalFact(
                "philidor",
                "hold the Philidor draw: keep your rook on the third rank to stop their "
                "king reaching the sixth, then check from behind once the pawn advances",
                (_name(king), _name(rook)),
            )
        ]
    return []


# Space (Silman's rules of space): central manoeuvring room. Measured the way an engine counts
# it -- the safe central squares behind your front, files c-f on your side, that no enemy pawn
# attacks and your own pawn does not block -- which separates a real space edge from a symmetric
# middlegame, where a naive "advanced pawns" count cannot (both sides' pawns cancel out).
_CENTRAL_FILES = (2, 3, 4, 5)  # c, d, e, f
SPACE_MARGIN = 3


def _space_count(board: chess.Board, color: chess.Color) -> int:
    ranks = (1, 2, 3) if color == chess.WHITE else (6, 5, 4)
    enemy_pawn_attacks: set[chess.Square] = set()
    for sq in board.pieces(chess.PAWN, not color):
        enemy_pawn_attacks |= set(board.attacks(sq))
    count = 0
    for file_ in _CENTRAL_FILES:
        for rank in ranks:
            sq = chess.square(file_, rank)
            piece = board.piece_at(sq)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                continue
            if sq in enemy_pawn_attacks:
                continue
            count += 1
    return count


def space_plan(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """A space advantage (avoid trades) or a cramp (seek them), when it is pronounced.

    More room means keep pieces on and improve them -- a lasting plus; less room means trade a
    pair to free the rest. Named only past a clear margin, so a roughly balanced middlegame is
    never called cramped, and the evidence is your own advanced central pawns.
    """
    advantage = _space_count(board, us) - _space_count(board, not us)
    advanced = (lambda r: r >= 3) if us == chess.WHITE else (lambda r: r <= 4)
    pawns = sorted(
        sq
        for sq in board.pieces(chess.PAWN, us)
        if chess.square_file(sq) in _CENTRAL_FILES and advanced(chess.square_rank(sq))
    )
    squares = tuple(_name(sq) for sq in pawns)
    if advantage >= SPACE_MARGIN:
        return [
            PositionalFact(
                "spaceAdvantage",
                "you have more space: keep pieces on and improve your worst-placed one -- a "
                "space edge is a lasting plus, so avoid trades",
                squares,
            )
        ]
    if advantage <= -SPACE_MARGIN:
        return [
            PositionalFact(
                "spaceCramp",
                "you are cramped for space: trade a pair of pieces to give the rest room",
                squares,
            )
        ]
    return []


# Wings for a pawn majority: the flanks, where a majority makes an outside passed pawn. The
# centre files (d, e) are left out -- a central plus is about breaks and the passers the
# other detectors already name, not about which wing to play on.
_QUEENSIDE = (0, 1, 2)  # files a, b, c
_KINGSIDE = (5, 6, 7)  # files f, g, h


def pawn_majority(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """A wing where you outnumber the opponent's pawns and can make a passed pawn.

    Silman's "which side of the board to play on": a wing majority is your side, and the plan
    is to advance it into a passed pawn. Only a healthy majority is named -- one with more
    advancing files than the enemy has pawns to blockade -- so a majority crippled by doubled
    pawns, which cannot make a passer, is never mistaken for a plan.
    """
    facts: list[PositionalFact] = []
    them = not us
    for name, files in (("queenside", _QUEENSIDE), ("kingside", _KINGSIDE)):
        ours = [sq for sq in board.pieces(chess.PAWN, us) if chess.square_file(sq) in files]
        theirs = sum(1 for sq in board.pieces(chess.PAWN, them) if chess.square_file(sq) in files)
        distinct = len({chess.square_file(sq) for sq in ours})
        if theirs >= 1 and len(ours) > theirs and distinct > theirs:
            squares = tuple(_name(sq) for sq in sorted(ours))
            facts.append(
                PositionalFact(
                    "pawnMajority",
                    f"your {name} pawn majority: advance it to make a passed pawn",
                    squares,
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
        attacker = not color
        for square in sorted(board.pieces(chess.PAWN, color)):
            file_ = chess.square_file(square)
            letter = _FILES[file_]
            # A lone d-pawn is the isolated queen's pawn, its own structure (`isolated_d_pawn`).
            if is_isolated(board, square) and file_ != 3:
                text = (
                    f"their isolated {letter}-pawn on {_name(square)} is a target to pile up on"
                    if target
                    else f"your isolated {letter}-pawn on {_name(square)} needs pieces to defend it"
                )
                facts.append(PositionalFact(kind, text, (_name(square),)))
            elif is_backward(board, square) and not _pawns_on_file(board, attacker, file_):
                # A backward pawn is a real weakness only on a half-open file, where the enemy's
                # rooks can get at it; behind an enemy pawn on the file it cannot be attacked, so
                # it is not named (Silman: nothing is weak if you cannot hit it).
                text = (
                    f"their backward {letter}-pawn on {_name(square)} is a lasting target"
                    if target
                    else f"your backward {letter}-pawn on {_name(square)} is a long-term weakness"
                )
                facts.append(PositionalFact(kind, text, (_name(square),)))
        for file_, squares in _doubled_files(board, color):
            letter = _FILES[file_]
            # Doubled pawns are not purely a weakness: the file the capture vacated is half-open
            # for the owner's rooks. Name that compensation on your own; the enemy's stay targets.
            half_open = next(
                (
                    _FILES[f]
                    for f in (file_ - 1, file_ + 1)
                    if 0 <= f <= 7 and not _pawns_on_file(board, color, f)
                ),
                None,
            )
            if target:
                text = f"their doubled {letter}-pawns are a lasting weakness to work against"
            elif half_open is not None:
                text = (
                    f"your doubled {letter}-pawns are a long-term weakness, though they give "
                    f"your rooks the half-open {half_open}-file"
                )
            else:
                text = f"your doubled {letter}-pawns are a long-term weakness"
            facts.append(PositionalFact(kind, text, tuple(_name(s) for s in squares)))

    note(them, "targetPawn")
    note(us, "weakPawn")
    return facts


def isolated_d_pawn(
    board: chess.Board, us: chess.Color, *, phase: str = "middlegame"
) -> list[PositionalFact]:
    """The isolated queen's pawn, named for both sides and by phase.

    For its owner the IQP is double-edged. In the middlegame its energy is an asset to attack
    with -- the outposts the pawn supports, the c- and e-files for the rooks, the d-file break
    -- and the plan is to keep pieces on. In an endgame that energy is spent and it is a target
    to hold. For the other side the plan is the classic formula: blockade the square in front,
    trade the minor pieces (which ends the owner's attack) and steer for the endgame.
    """
    facts: list[PositionalFact] = []
    for color in (not us, us):  # the enemy's IQP is the plan, so it comes first
        for square in _pawns_on_file(board, color, 3):
            if not is_isolated(board, square):
                continue
            name = _name(square)
            file_, rank = chess.square_file(square), chess.square_rank(square)
            ahead = rank + _forward(color)
            front = _name(chess.square(file_, ahead))  # the square in front of the pawn
            outposts = [
                _name(chess.square(f, ahead)) for f in (file_ - 1, file_ + 1) if 0 <= f <= 7
            ]
            if color == us:
                if phase == "endgame":
                    facts.append(
                        PositionalFact(
                            "iqp",
                            f"your isolated d-pawn on {name} is a target now: its middlegame "
                            "activity is spent, so hold it and trade rooks to ease the pressure",
                            (name,),
                        )
                    )
                else:
                    facts.append(
                        PositionalFact(
                            "iqp",
                            f"your isolated d-pawn on {name}, an IQP: play on its energy -- the "
                            f"{' and '.join(outposts)} outposts it supports, the c- and e-files "
                            f"for your rooks, the break to {front} -- keep pieces on and go for "
                            "the attack; it turns into a target only in an endgame",
                            (name, *outposts),
                        )
                    )
            else:
                facts.append(
                    PositionalFact(
                        "iqpTarget",
                        f"their isolated d-pawn on {name}, an IQP, is a long-term target: "
                        f"blockade it on {front}, trade the minor pieces (that ends its attack) "
                        "and steer for an endgame where it is just a weakness",
                        (name, front),
                    )
                )
    return facts


def hanging_pawns(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Two friendly pawns abreast on half-open files (the c- and d- or d- and e-pawns).

    Mobile and space-giving while they can advance, a target once they are blockaded; named
    for both sides. Only the central pairs count, not two flank pawns that happen to line up.
    """
    facts: list[PositionalFact] = []
    for color in (not us, us):
        rank = 3 if color == chess.WHITE else 4
        pawns = sorted(
            sq for sq in board.pieces(chess.PAWN, color) if chess.square_rank(sq) == rank
        )
        for first, second in zip(pawns, pawns[1:], strict=False):
            fa, fb = chess.square_file(first), chess.square_file(second)
            if fb != fa + 1 or not (fa >= 1 and fb <= 6):
                continue
            if _pawns_on_file(board, color, fa - 1) or _pawns_on_file(board, color, fb + 1):
                continue
            pair = f"{_FILES[fa]}- and {_FILES[fb]}-pawns"
            flanks = f"{_FILES[fa - 1]}- and {_FILES[fb + 1]}-files"
            if color == us:
                facts.append(
                    PositionalFact(
                        "hangingPawns",
                        f"your {pair} are hanging pawns: they hold the centre and give your "
                        f"rooks the {flanks}; keep them mobile -- advancing one can fix an "
                        "enemy pawn -- but once blockaded they are targets",
                        (_name(first), _name(second)),
                    )
                )
            else:
                facts.append(
                    PositionalFact(
                        "hangingPawnsTarget",
                        f"their {pair} are hanging pawns: press them until one must advance and "
                        "leaves a hole, or blockade the pair and work against the fixed pawn",
                        (_name(first), _name(second)),
                    )
                )
    return facts


def _closed_center(board: chess.Board) -> bool:
    """A locked centre: a central pawn ram, with both sides' d- and e-pawns committed forward.

    The ram alone is not enough -- 1.e4 e5 rams on the e-file but the centre is wide open with
    the d-pawns still home. A closed centre is a full, fixed one, so both sides must have
    advanced both their d- and e-pawns off their home rank (the King's Indian, the French, the
    Benoni), which is what forces the play onto the wings.
    """
    ram = any(
        chess.square_rank(black_pawn) == chess.square_rank(white_pawn) + 1
        for file_ in (3, 4)
        for white_pawn in _pawns_on_file(board, chess.WHITE, file_)
        for black_pawn in _pawns_on_file(board, chess.BLACK, file_)
    )
    if not ram:
        return False
    for color in (chess.WHITE, chess.BLACK):
        home = 1 if color == chess.WHITE else 6
        for file_ in (3, 4):  # both the d- and the e-pawn must be committed forward
            pawns = _pawns_on_file(board, color, file_)
            if not pawns or all(chess.square_rank(sq) == home for sq in pawns):
                return False
    return True


def pawn_pointing(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """With a closed centre, which wing to play on -- the way your pawns point.

    Silman's rule of the closed centre: a leaning central pawn chain points to the wing where
    you have space, and that is where you attack; push the pawn beside your most advanced
    central pawn. It fires only with a locked centre and a chain that clearly leans, so a broad
    front (the d- and e-pawns abreast, pointing nowhere) says nothing. On the textbook cases it
    gives the right wing: the French Advance to the kingside, the King's Indian to the kingside
    for Black (...f5), the Czech Benoni to the queenside.
    """
    if not _closed_center(board):
        return []
    d_pawns = _pawns_on_file(board, us, 3)
    e_pawns = _pawns_on_file(board, us, 4)
    if not d_pawns or not e_pawns:
        return []
    d_advance = max(chess.square_rank(sq) * _forward(us) for sq in d_pawns)
    e_advance = max(chess.square_rank(sq) * _forward(us) for sq in e_pawns)
    if e_advance > d_advance:
        wing, push = "kingside", "f"
    elif d_advance > e_advance:
        wing, push = "queenside", "c"
    else:
        return []  # a broad front, leaning neither way
    squares = tuple(_name(sq) for sq in sorted([*d_pawns, *e_pawns]))
    return [
        PositionalFact(
            "pawnPointing",
            f"the centre is closed, so play where your pawns point -- the {wing}; advance your "
            f"{push}-pawn to open lines there",
            squares,
        )
    ]


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
        # Sorted, not set order: with two heavy pieces on one file, which one the coach
        # names came out of CPython's set hashing -- a queen on a1 and a rook on a4 named
        # the rook, the same pair one file over named the queen. The lowest square is the
        # one a reader meets first.
        on_file = sorted(sq for sq in heavy if chess.square_file(sq) == file_)
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


def pawn_safe(board: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """No pawn of the other side can ever attack `square`, from where the pawns stand now."""
    return _pawn_safe(board, square, us)


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


def _center_pawn_count(board: chess.Board) -> int:
    """Pawns of both colours on the d- and e-files: a simple read of how open the centre is."""
    return sum(
        1
        for color in (chess.WHITE, chess.BLACK)
        for sq in board.pieces(chess.PAWN, color)
        if chess.square_file(sq) in (3, 4)
    )


def bishop_pair(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """You keep both bishops while the opponent does not.

    Silman's minor-piece rules: two bishops cover both colours, so the pair is an asset in an
    open position; in a closed one it is dormant and the plan is to open lines to bring it to
    life. The advice therefore reads the openness of the centre, it is not a fixed slogan.
    """
    ours = board.pieces(chess.BISHOP, us)
    theirs = board.pieces(chess.BISHOP, not us)
    if not (len(ours) >= 2 and len(theirs) < 2):
        return []
    centre = _center_pawn_count(board)
    if centre <= 1:
        text = (
            "you have the bishop pair and the position is open -- the two bishops cover both "
            "colours here, so keep lines open and don't trade one off"
        )
    elif centre >= 3:
        text = (
            "you have the bishop pair but the centre is closed -- open lines to bring the two "
            "bishops to life"
        )
    else:
        text = "you have the bishop pair, so keep the position open where you can"
    return [PositionalFact("bishopPair", text, tuple(_name(sq) for sq in sorted(ours)))]


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
                    f"your bishop on {_name(square)} is a bad bishop, blocked by your own "
                    "pawns on its colour: trade it, push those pawns off its colour, or get it "
                    "outside the pawn chain",
                    (_name(square),),
                )
            )
    return facts


# Past this move a minor piece on its home square is as likely to have gone back there as
# never to have left (a bishop retreating to f1 at move 35), so it is a piece on its home
# square, ranked with the small features, not "not yet in the game" at the top.
DEVELOPMENT_MOVES = 20


def at_home(board: chess.Board, us: chess.Color) -> list[chess.Square]:
    """The player's knights on b and g, and bishops on c and f, of the home rank: a knight
    back on f1 is not at home (39.Ne3 in Johner-Nimzowitsch)."""
    return [
        sq
        for sq in _HOME_MINORS[us]
        if board.color_at(sq) == us and board.piece_type_at(sq) == _HOME_TYPE[chess.square_file(sq)]
    ]


def development(board: chess.Board, us: chess.Color) -> list[PositionalFact]:
    """Minor pieces still on their home squares, and whether the king has castled."""
    home = at_home(board, us)
    facts: list[PositionalFact] = []
    if home:
        names = " and ".join(
            f"{chess.piece_name(board.piece_type_at(sq) or chess.PAWN)} on {_name(sq)}"
            for sq in home
        )
        verb = "is" if len(home) == 1 else "are"
        if board.fullmove_number <= DEVELOPMENT_MOVES:
            kind = "undeveloped"
            text = f"your {names} {verb} still at home, not yet in the game"
        else:
            kind = "homeSquare"
            where = "its home square" if len(home) == 1 else "their home squares"
            text = f"your {names} {verb} on {where}"
        facts.append(PositionalFact(kind, text, tuple(_name(sq) for sq in home)))
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
        facts += space_plan(board, us)
        facts += pawn_pointing(board, us)
    facts += isolated_d_pawn(board, us, phase=phase)
    facts += hanging_pawns(board, us)
    facts += pawn_weaknesses(board, us)
    facts += passed_pawns(board, us)
    facts += pawn_majority(board, us)
    facts += outposts(board, us)
    facts += open_files(board, us)
    facts += bad_bishop(board, us)
    facts += bishop_pair(board, us)
    facts += king_activity(board, us, phase=phase)
    facts += square_of_the_pawn(board, us, phase=phase)
    facts += opposition(board, us)
    facts += key_squares(board, us)
    facts += lucena(board, us)
    facts += philidor(board, us)
    facts += material_plan(board, us)
    facts += worst_piece(board, us)
    return facts


# --- the idea behind the best move ---------------------------------------------------------


def _is_central_break(board: chess.Board, move: chess.Move, us: chess.Color) -> bool:
    """A pawn move that strikes at or occupies the centre. A capture counts when it takes
    toward the centre: fxe5 does, exf5 goes away from it ("the fight is for the centre, and
    you joined it" was said of 2.exf5, Tai's game, 2026-09-24)."""
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return False
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    central_file = 2 <= to_file <= 5
    central_rank = 3 <= to_rank <= 4
    if board.is_capture(move):
        toward = abs(2 * to_file - 7) < abs(2 * chess.square_file(move.from_square) - 7)
        return central_file and toward
    return central_file and central_rank


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


def _wing(file_: int) -> str | None:
    """The wing a file belongs to: a to c the queenside, f to h the kingside, else None."""
    return "queenside" if file_ <= 2 else "kingside" if file_ >= 5 else None


def _relative_rank(square: chess.Square, us: chess.Color) -> int:
    """The square's rank counted from your side of the board, 1 to 8."""
    rank = chess.square_rank(square)
    return rank + 1 if us == chess.WHITE else 8 - rank


def _pawn_storm_side(board: chess.Board, move: chess.Move, us: chess.Color) -> str | None:
    """The wing being stormed if the move pushes a pawn at the enemy King, else None.

    A pawn advance -- not a capture -- toward an enemy King that has castled to a wing, on the
    files around it: the classic way to prise open a castled position. Only with your own King
    off that wing, and only a pawn that reaches its fourth rank: with both Kings behind the same
    pawns, h3 or g6 is a flight square or a guard, not a storm (the master games' notes, where
    "storming the pawns in front of their King" was said of h3 at move 15, 2026-09-24).
    """
    if board.piece_type_at(move.from_square) != chess.PAWN or board.is_capture(move):
        return None
    enemy_king, own_king = board.king(not us), board.king(us)
    if enemy_king is None:
        return None
    king_file = chess.square_file(enemy_king)
    if not (king_file <= 2 or king_file >= 5):
        return None  # a King in the centre is not a wing to storm
    if own_king is not None and _wing(chess.square_file(own_king)) == _wing(king_file):
        return None  # storming your own King's cover
    if abs(chess.square_file(move.to_square) - king_file) > 2:
        return None
    if _relative_rank(move.to_square, us) < 4:
        return None
    advancing = (
        chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
    ) * _forward(us) > 0
    return ("queenside" if king_file <= 2 else "kingside") if advancing else None


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
    """Whether your piece on `square` is in danger: the opponent can win it, taking with a
    legal capture and coming out ahead once the recaptures are played out.

    Read by the rules (`exchange.can_be_won`), not off `board.attackers()`, which called a
    knight guarded by a pawn "in danger" from the enemy king beside it (the king was worth
    nothing, and it cannot take a guarded piece anyway), and a knight attacked only by a piece
    pinned to its own king.
    """
    piece = board.piece_at(square)
    return piece is not None and piece.color == us and exchange.can_be_won(board, square)


def _prevents_threat(
    board: chess.Board, after: chess.Board, threat: chess.Move, us: chess.Color
) -> bool:
    """Whether the move that led to `after` stops the opponent's threatened move.

    Stopped when the threat is no longer a legal move at all, or when it was a capture of one
    of your pieces that won material and no longer does: played out by the rules, recaptures
    included, it now nets nothing (`exchange.exchange`). A guard alone is not enough -- a
    bishop guarding a knight a pawn attacks leaves the pawn winning it, and a guard pinned to
    its own king takes nothing back.
    """
    if threat not in after.legal_moves:
        return True
    victim = after.piece_at(threat.to_square)
    if victim is None or victim.color != us:
        return False
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())  # the position before the move, the opponent to move
    return (
        threat in passed.legal_moves
        and exchange.exchange(passed, threat) > 0
        and exchange.exchange(after, threat) <= 0
    )


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
    if piece_type is None or move.promotion is not None:
        # A new queen needs no idea read into it ("prophylaxis" was said of c8=Q).
        return None
    if board.is_check():
        # Out of check is the way out, not a plan: "in the endgame the king is a fighting
        # piece" was said of a king driven off by a check (the lesson audit, 2026-09-27).
        return None
    after = board.copy(stack=False)
    after.push(move)
    to = _name(move.to_square)

    if board.is_castling(move):
        return PositionalFact(
            "castle", "castle to tuck your king away and connect the rooks", (to,)
        )

    # A rescue needs the square the piece goes to to be safe as well: a knight that runs from
    # one pawn onto a square another pawn takes is not brought to safety. And the piece must
    # be what their threat goes after (`threat`, the engine's best move for them were it
    # theirs): a piece that could be taken but is not what they would take was "a piece in
    # danger comes first" of a rook check that led to mate (the lesson audit, 2026-09-27).
    if (
        threat is not None
        and threat.to_square == move.from_square
        and piece_type != chess.PAWN  # a pawn that steps on from an attack is a push
        and _piece_is_attacked(board, move.from_square, us)
        and not board.is_capture(move)
        and not _piece_is_attacked(after, move.to_square, us)
    ):
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "rescue", f"your {name} was in danger; moving to {to} brings it to safety", (to,)
        )

    # A passed pawn that runs is the move's point, before whatever it also stops.
    if piece_type == chess.PAWN and is_passed(board, move.from_square):
        file_ = _FILES[chess.square_file(move.from_square)]
        return PositionalFact("pushPasser", f"push your passed {file_}-pawn toward {to}", (to,))

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

    # The king fights for something: a bare king against a piece is only running from mate
    # ("the king is a fighting piece" of a lone king before a queen and a knight, the lesson
    # audit, 2026-09-27).
    men = board.piece_map().values()
    bare = sum(p.color == us for p in men) == 1 and any(
        p.color != us and p.piece_type not in (chess.KING, chess.PAWN) for p in men
    )
    if (
        phase == "endgame"
        and piece_type == chess.KING
        and not bare
        and not board.is_capture(move)
        and _center_distance(move.to_square) < _center_distance(move.from_square)
    ):
        return PositionalFact(
            "activateKing",
            f"march your King to {to}, toward the centre; in the endgame it is a strong piece",
            (to,),
        )

    # A minor piece leaving its home square is development while the game is young; past
    # that it may have gone back there, and "finish your development" at move 40 was
    # nonsense.
    if (
        phase != "endgame"
        and board.fullmove_number <= DEVELOPMENT_MOVES
        and piece_type in (chess.KNIGHT, chess.BISHOP)
    ):
        home = _HOME_MINORS[us]
        if move.from_square in home and not board.is_capture(move):
            name = PIECE_NAMES[piece_type]
            return PositionalFact(
                "develop", f"finish developing: the {name} comes into the game on {to}", (to,)
            )

    # What a piece move is for, when it takes nothing: a capture's point is the exchange, and
    # "centralise the knight" was said of Nxe5 winning a pawn (the master games, 2026-09-24).
    quiet = not board.is_capture(move)

    if quiet and piece_type == chess.KNIGHT and _is_outpost(after, move.to_square, us):
        return PositionalFact(
            "outpostMove",
            f"the knight lands on {to}, an outpost no enemy pawn can chase it from",
            (to,),
        )

    if (
        quiet
        and piece_type in (chess.KNIGHT, chess.BISHOP)
        and _is_central_strong(after, move.to_square, us)
    ):
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "centralise", f"centralise the {name} on {to}, a strong square in the middle", (to,)
        )

    file_letter = _rook_to_file_is_open(board, move, us) if quiet else None
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

    if quiet and worst is not None and move.from_square == worst:
        name = PIECE_NAMES[piece_type]
        return PositionalFact(
            "activate", f"this brings your most passive piece, the {name}, to life on {to}", (to,)
        )

    # A storm and space are middlegame ideas: in an ending with pieces a pawn goes forward to
    # queen or to make a passer ("pawns thrown at their king" of h4 with a queen against a
    # bishop, the lesson audit, 2026-09-27). A pawn ending keeps them: the coach's facts read
    # them there as the passer's race (`facts._PAWN_ENDING_PLAN`).
    pawn_ending = all(p.piece_type in (chess.KING, chess.PAWN) for p in board.piece_map().values())
    middle = phase != "endgame" or pawn_ending
    storm_side = _pawn_storm_side(board, move, us) if middle else None
    if storm_side is not None:
        return PositionalFact(
            "pawnStorm",
            f"storm their King with the pawn to {to}, prising open the {storm_side}",
            (to,),
        )

    # Space is a pawn that goes forward and stays: not a capture, and past its third rank (a3
    # and h3 make room for the King or keep a piece out; they take no space).
    if (
        phase != "opening"
        and middle
        and piece_type == chess.PAWN
        and quiet
        and _relative_rank(move.to_square, us) >= 4
    ):
        file_ = chess.square_file(move.to_square)
        if file_ <= 1 or file_ >= 6:
            side = "queenside" if file_ <= 1 else "kingside"
            return PositionalFact("space", f"gain space on the {side} with the pawn to {to}", (to,))

    # A trade is a capture that is taken back for as much: one that wins something is the
    # win, not a simplification ("trades bring the win closer" was said of taking a loose
    # bishop, the lesson audit, 2026-09-27). And a trade of pieces: ahead, the pawns are
    # what you keep.
    if (
        board.is_capture(move)
        and board.piece_type_at(move.to_square) not in (None, chess.PAWN)
        and _material_diff(board, us) >= 3
        and exchange.exchange(board, move) == 0
        and exchange.can_retake(board, move)
    ):
        return PositionalFact(
            "simplify", "you are up material, so this trade simplifies toward the win", (to,)
        )

    return None


# How instructive each static feature is, most first, so the sheet leads with the one that
# teaches the most and never turns into a laundry list of every open file on the board.
_FEATURE_RANK: dict[str, int] = {
    "uncastled": 0,
    "undeveloped": 1,
    "lucena": 1,
    "philidor": 1,
    "runningPawn": 1,
    "outrunPawn": 1,
    "passiveKing": 2,
    "keySquare": 2,
    "keySquareGoal": 2,
    "opposition": 2,
    "oppositionAgainst": 3,
    "iqpTarget": 3,
    "hangingPawnsTarget": 4,
    "targetPawn": 5,
    "passedPawn": 6,
    "iqp": 7,
    "hangingPawns": 8,
    "pawnMajority": 9,
    "pawnPointing": 9,
    "convert": 9,
    "complicate": 9,
    "spaceAdvantage": 10,
    "spaceCramp": 11,
    "homeSquare": 12,
    "outpost": 10,
    "badBishop": 11,
    "openFile": 12,
    "bishopPair": 13,
    "weakPawn": 14,
    "passivePiece": 15,
    "halfOpenFile": 16,
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
