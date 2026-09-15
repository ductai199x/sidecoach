"""Motif tagger: pure functions over (board, move, principal variation).

No engine calls, no I/O, no global state, and the boards handed in are never mutated. Every
tag is justified by evidence on the board or in the PV, never by a guess, because the coach
reads the evidence squares out loud ("look at f7 and the knight on g5"). Tag names are the
Lichess puzzle theme names, camelCase, spelled exactly as Lichess spells them, so the profile
and the recommendation engine can key on them.

Two questions, two functions. `tags_for_move` answers "what does this move do", from the
mover's point of view. `tags_for_position` answers "what is in this position" for the side to
move: the static features plus the tags of the engine's best move.

The theme names are Lichess's, and so are the definitions: a motif is what Lichess would tag,
not a stricter thing of our own. Two rules keep the tags honest:

1. A tag that claims material has to gain material once defenders are counted: a knight
   forking two rooks while a pawn can take it is not a fork, and a loose piece must really be
   undefended. But a pin, a skewer's front piece, or a discovered attack is a restriction or a
   revealed line, not a material claim, so those follow the geometry the way Lichess does --
   a piece is pinned when it cannot leave the line without exposing a more valuable one behind
   it, whether or not that piece can be won.
2. `tags_for_move` reports only what the move brought about. A pin the bishop was already
   making before the move is not something the move did, so castling does not inherit it.
   Those features still show up in `tags_for_position`, where they belong.

`mate_in` counts the mover's own moves including the one being judged: 1 means this move is
checkmate, 3 means this move plus two more, and a negative value means the mover gets mated.
A negative `mate_in` never produces a `mateIn*` tag.

Output order is fixed by `TAG_ORDER` below: what the move is, then what it creates, then what
the line does, then the mating pattern. Tags are deduplicated by name and a repeated name
merges its evidence squares in first-seen order.

Definitions that needed a judgement call are documented on the private detector that
implements them. Not implemented, and why:

- `equality`, `crushing`, `advantage`: engine score, not motifs.
- `zugzwang`, `intermezzo`: both need an engine to show that every move loses, or that the
  in-between move is better than the recapture. Out of scope for a pure tagger.
- `doubleAttack`: Lichess has no such theme, and under our definition it is exactly `fork`
  when one piece makes both attacks and `discoveredAttack` when two do. Nothing is left over.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import chess

from sidecoach.events import MotifTag

__all__ = ["TAG_ORDER", "tags_for_move", "tags_for_position"]


TAG_ORDER: tuple[str, ...] = (
    # A. What the move is: read off the move and the board.
    "capture",
    "enPassant",
    "promotion",
    "underPromotion",
    "castling",
    "advancedPawn",
    "check",
    "doubleCheck",
    "quietMove",
    "defensiveMove",
    # B. What the move creates: read off the attack maps after the move.
    "fork",
    "pin",
    "absolutePin",
    "skewer",
    "discoveredAttack",
    "xRayAttack",
    "interference",
    "hangingPiece",
    "trappedPiece",
    "attackingF2F7",
    "exposedKing",
    # C. What the line does: read off the PV and the mate distance.
    "sacrifice",
    "clearance",
    "capturingDefender",
    "deflection",
    "attraction",
    "mate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
    "mateIn4",
    "mateIn5",
    # D. The mating pattern, only when the line actually ends in mate.
    "backRankMate",
    "smotheredMate",
    "arabianMate",
    "anastasiaMate",
    "hookMate",
    "killBoxMate",
    "vukovicMate",
    "bodenMate",
    "doubleBishopMate",
    "dovetailMate",
)

_ORDER_INDEX = {name: i for i, name in enumerate(TAG_ORDER)}

# Standard 1/3/3/5/9. The king is priceless; 100 keeps every comparison the right way round.
_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}

_MINOR = 3
_ROOK = 5

# A sacrifice must give up at least this much: a pawn gambit is not a sacrifice.
_SACRIFICE_THRESHOLD = 2

_CORNERS = (chess.A1, chess.A8, chess.H1, chess.H8)


# --- small helpers ---------------------------------------------------------------------


def _value(piece_type: chess.PieceType | None) -> int:
    return 0 if piece_type is None else _VALUE[piece_type]


def _tag(name: str, squares: Iterable[chess.Square], move: chess.Move | None = None) -> MotifTag:
    names: list[str] = []
    for square in squares:
        label = chess.square_name(square)
        if label not in names:
            names.append(label)
    return MotifTag(name=name, squares=names, move=None if move is None else move.uci())


def _sliders(board: chess.Board, color: chess.Color) -> chess.SquareSet:
    bb = (board.bishops | board.rooks | board.queens) & board.occupied_co[color]
    return chess.SquareSet(bb)


def _sign(n: int) -> int:
    if n > 0:
        return 1
    return -1 if n < 0 else 0


def _next_beyond(board: chess.Board, near: chess.Square, far: chess.Square) -> chess.Square | None:
    """First occupied square past `far`, continuing along the `near` -> `far` ray."""
    df = _sign(chess.square_file(far) - chess.square_file(near))
    dr = _sign(chess.square_rank(far) - chess.square_rank(near))
    file_, rank_ = chess.square_file(far) + df, chess.square_rank(far) + dr
    while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
        square = chess.square(file_, rank_)
        if board.occupied & chess.BB_SQUARES[square]:
            return square
        file_ += df
        rank_ += dr
    return None


def _ep_victim(move: chess.Move) -> chess.Square:
    return chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))


def _balance(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, value in _VALUE.items():
        if piece_type == chess.KING:
            continue
        total += value * len(board.pieces(piece_type, color))
        total -= value * len(board.pieces(piece_type, not color))
    return total


def _line_boards(
    before: chess.Board, moves: Sequence[chess.Move]
) -> tuple[list[chess.Board], list[chess.Move]]:
    """Walk the line, stopping at the first move that is not legal.

    `boards[i]` is the position before `played[i]`; `boards[-1]` is the end of the line.
    """
    boards = [before.copy(stack=False)]
    played: list[chess.Move] = []
    for move in moves:
        current = boards[-1]
        if move not in current.legal_moves:
            break
        nxt = current.copy(stack=False)
        nxt.push(move)
        boards.append(nxt)
        played.append(move)
    return boards, played


def _finish(tags: Iterable[MotifTag]) -> list[MotifTag]:
    """Deduplicate by name, merging evidence squares in first-seen order, then sort."""
    merged: dict[str, MotifTag] = {}
    for tag in tags:
        current = merged.get(tag.name)
        if current is None:
            merged[tag.name] = MotifTag(name=tag.name, squares=list(tag.squares), move=tag.move)
            continue
        for square in tag.squares:
            if square not in current.squares:
                current.squares.append(square)
        if current.move is None:
            current.move = tag.move
    fallback = len(_ORDER_INDEX)
    return sorted(merged.values(), key=lambda t: (_ORDER_INDEX.get(t.name, fallback), t.name))


# --- group A: what the move is ---------------------------------------------------------


def _defensive_evidence(
    before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color
) -> list[chess.Square] | None:
    """A quiet move that saves material: it runs from an attack, or it defends an attacked man.

    Running away means the piece could have been taken for profit where it stood and its new
    square is not attacked at all. Defending means the move newly covers a friendly piece the
    opponent could have won before the move and is still attacking after it. Adding a second
    guard to a piece that was already holding is not a rescue, so it does not count.
    """
    enemy = not mover
    ran = not _is_safe(before, move.from_square) and not after.is_attacked_by(enemy, move.to_square)
    if ran:
        return [move.from_square, move.to_square]
    covered_before = before.attacks(move.from_square)
    for square in after.attacks(move.to_square):
        if square in covered_before:
            continue
        piece = after.piece_at(square)
        if piece is None or piece.color != mover:
            continue
        if _is_safe(before, square) or not after.is_attacked_by(enemy, square):
            continue
        return [move.to_square, square]
    return None


def _group_a(
    before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color
) -> list[MotifTag]:
    tags: list[MotifTag] = []
    frm, to = move.from_square, move.to_square
    is_ep = before.is_en_passant(move)
    is_capture = before.is_capture(move)
    gives_check = after.is_check()

    if is_capture:
        squares = [frm, to, _ep_victim(move)] if is_ep else [frm, to]
        tags.append(_tag("capture", squares, move))
    if is_ep:
        tags.append(_tag("enPassant", [frm, to, _ep_victim(move)], move))
    if move.promotion is not None:
        tags.append(_tag("promotion", [frm, to], move))
        if move.promotion != chess.QUEEN:
            tags.append(_tag("underPromotion", [frm, to], move))
    if before.is_castling(move):
        rank = chess.square_rank(frm)
        kingside = before.is_kingside_castling(move)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        tags.append(_tag("castling", [frm, to, rook_from, rook_to], move))
    if before.piece_type_at(frm) == chess.PAWN:
        rank = chess.square_rank(to)
        if (mover == chess.WHITE and rank >= 5) or (mover == chess.BLACK and rank <= 2):
            tags.append(_tag("advancedPawn", [frm, to], move))
    if gives_check:
        checkers = list(after.checkers())
        king = after.king(not mover)
        squares = checkers if king is None else [*checkers, king]
        tags.append(_tag("check", squares, move))
        if len(checkers) >= 2:
            tags.append(_tag("doubleCheck", squares, move))
    if not is_capture and not gives_check and move.promotion is None:
        tags.append(_tag("quietMove", [frm, to], move))
        evidence = _defensive_evidence(before, after, move, mover)
        if evidence is not None:
            tags.append(_tag("defensiveMove", evidence, move))
    return tags


# --- group B: what the move creates -----------------------------------------------------


def _winnable(
    view: chess.Board, square: chess.Square, cost: int, ignore: chess.Square | None = None
) -> bool:
    """Would taking the piece on `square` with a `cost`-valued piece gain material?

    True when the piece is worth more than what takes it, or when nothing defends it.
    `ignore` drops one defender, which is how a skewered piece stops counting as the guard of
    the piece behind it: it is the one being driven off the line.
    """
    piece = view.piece_at(square)
    if piece is None:
        return False
    if piece.piece_type == chess.KING or _value(piece.piece_type) > cost:
        return True
    defenders = view.attackers(piece.color, square)
    if ignore is not None:
        defenders.discard(ignore)
    return not defenders


def _is_safe(view: chess.Board, square: chess.Square) -> bool:
    """Can the piece on `square` be taken for profit? Safe means no, so it can stay."""
    piece = view.piece_at(square)
    if piece is None:
        return True
    mine = _value(piece.piece_type)
    attackers = view.attackers(not piece.color, square)
    if not attackers:
        return True
    if any(_value(view.piece_type_at(a)) < mine for a in attackers):
        return False
    return bool(view.attackers(piece.color, square))


def _mover_captures(
    boards: list[chess.Board], played: Sequence[chess.Move], mover: chess.Color, plies: int = 6
) -> frozenset[chess.Square]:
    """Squares the mover captures an enemy piece on early in the line the engine expects.

    This lets a fork that wins its piece a move or two later (once a check has been answered)
    be recognised on the board where the double attack is set up, using the engine's own
    continuation rather than a guess -- the same line the multi-move motifs already read.
    """
    squares: set[chess.Square] = set()
    for i, mv in enumerate(played[:plies]):
        board = boards[i]
        if board.turn != mover:
            continue
        victim = board.piece_at(mv.to_square)
        if victim is not None and victim.color != mover:
            squares.add(mv.to_square)
    return frozenset(squares)


def _fork(
    after: chess.Board,
    move: chess.Move,
    mover: chess.Color,
    *,
    won_in_line: frozenset[chess.Square] = frozenset(),
) -> MotifTag | None:
    """The moved piece attacks two or more targets it could profitably take.

    A target is the enemy king, an enemy piece worth more than the moved piece, or an enemy
    piece worth at least as much and undefended. Equal value plus a defender is a trade, not
    a fork, which is what keeps pawn pushes out of this. Unless the enemy king is one of the
    targets, the forking piece must also be safe where it stands: a knight forking two rooks
    while a pawn can take it wins nothing.

    A check is the exception to value: when the move gives check, a loose enemy piece the
    forker also hits counts however cheap it is, because the enemy must answer the check and
    cannot save the piece too. That single case is verified move by move (`_check_fork_wins`)
    so a check that the enemy can meet while rescuing the piece is not called a fork. This is
    the common queen-check-and-grab that plain value comparison misses.
    """
    piece = after.piece_at(move.to_square)
    if piece is None or piece.color != mover:
        return None
    enemy = not mover
    mine = _value(piece.piece_type)
    gives_check = after.is_check()
    targets: list[chess.Square] = []
    forks_king = False
    for square in after.attacks(move.to_square):
        victim = after.piece_at(square)
        if victim is None or victim.color != enemy:
            continue
        if victim.piece_type == chess.KING:
            forks_king = True
            targets.append(square)
        elif _fork_target(after, move.to_square, square, mine, mover, gives_check, won_in_line):
            targets.append(square)
    if len(targets) < 2:
        return None
    if not forks_king and not _is_safe(after, move.to_square):
        return None
    return _tag("fork", [move.to_square, *targets], move)


def _fork_target(
    after: chess.Board,
    forker: chess.Square,
    square: chess.Square,
    mine: int,
    mover: chess.Color,
    gives_check: bool,
    won_in_line: frozenset[chess.Square],
) -> bool:
    """Whether the enemy piece on `square` is one a fork from `forker` would win.

    A cheap loose piece counts when the move gives check and either the piece cannot be
    saved against any reply (`_check_fork_wins`) or the engine's own line actually captures
    it -- the latter settles the cases where the enemy has a defence that only fails to
    deeper calculation, which is the engine's job, not the geometry's.
    """
    enemy = not mover
    theirs = _value(after.piece_type_at(square))
    if theirs > mine or (theirs >= mine and not after.attackers(enemy, square)):
        return True
    return (
        gives_check
        and not after.attackers(enemy, square)
        and (square in won_in_line or _check_fork_wins(after, forker, square, mover))
    )


def _check_fork_wins(
    after: chess.Board, forker: chess.Square, target: chess.Square, mover: chess.Color
) -> bool:
    """After a checking move, is the loose enemy piece on `target` truly won?

    True only when no legal reply to the check rescues it: for every reply the piece is still
    there, still attacked by the forker, and still undefended. A reply that captures the
    forker, guards the piece, or moves it to safety means the check-fork does not win it.
    """
    replies = list(after.legal_moves)
    if not replies:  # checkmate: the mate tags cover this, it is not a fork win
        return False
    enemy = not mover
    cost = _value(after.piece_type_at(forker))
    for reply in replies:
        nxt = after.copy(stack=False)
        nxt.push(reply)
        victim = nxt.piece_at(target)
        if victim is None or victim.color != enemy:
            return False  # the piece moved, was traded, or the square changed hands
        if forker not in nxt.attackers(mover, target):
            return False  # the forker no longer hits it (captured, pinned, blocked)
        if not _winnable(nxt, target, cost):
            return False  # it is defended now
    return True


def _hanging_tags(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Enemy pieces attacked by the mover with no defender at all, one tag per piece.

    Pawns are excluded: loose pawns are everywhere and a coach calling one "hanging" is noise.
    One tag per piece so that a piece that was already loose before a move can be told apart
    from one the move just left en prise.
    """
    enemy = not mover
    tags: list[MotifTag] = []
    for square in sorted(view.piece_map()):
        piece = view.piece_at(square)
        if piece is None or piece.color != enemy:
            continue
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if view.attackers(mover, square) and not view.attackers(enemy, square):
            tags.append(_tag("hangingPiece", [square]))
    return tags


def _ray_motifs(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Pins, skewers and x-rays: one mover slider, one enemy piece, one enemy piece behind it.

    A pin is geometric, the way Lichess tags it: an enemy piece with a more valuable enemy
    piece (or the king) directly behind it on the slider's line is pinned, whether or not the
    piece behind can be won, because the front piece cannot leave the line without exposing
    it. Skewers and x-rays instead name a piece to go after, so those still require the piece
    behind to be winnable, and a front piece that can simply take the slider for profit leaves
    nothing to act on.
    """
    enemy = not mover
    tags: list[MotifTag] = []
    for slider in _sliders(view, mover):
        cost = _value(view.piece_type_at(slider))
        for front_sq in view.attacks(slider):
            front = view.piece_at(front_sq)
            if front is None or front.color != enemy:
                continue
            back_sq = _next_beyond(view, slider, front_sq)
            if back_sq is None:
                continue
            back = view.piece_at(back_sq)
            if back is None or back.color != enemy:
                continue
            squares = [slider, front_sq, back_sq]
            front_value, back_value = _value(front.piece_type), _value(back.piece_type)
            if back_value > front_value:
                tags.append(_tag("pin", squares))
                if back.piece_type == chess.KING:
                    tags.append(_tag("absolutePin", squares))
                continue
            if slider in view.attacks(front_sq) and _winnable(view, slider, front_value):
                continue
            behind_is_loose = _winnable(view, back_sq, cost, ignore=front_sq)
            if front_value > back_value:
                must_move = _winnable(view, front_sq, cost)
                if must_move and behind_is_loose:
                    tags.append(_tag("skewer", squares))
            elif front_value >= _MINOR and behind_is_loose:
                tags.append(_tag("xRayAttack", squares))
    return tags


def _discovered_attack(
    before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color
) -> MotifTag | None:
    """The move steps off a line and a friendly slider now bears on something it did not.

    The proof is geometric, the way Lichess tags a discovery: the vacated square lies
    strictly between the slider and an enemy piece it now attacks and did not before. Whether
    that piece can be won is not the test -- revealing the attack is the motif -- so a
    discovery onto a defended piece, or onto a pawn, still counts.
    """
    enemy = not mover
    squares: list[chess.Square] = []
    for slider in _sliders(after, mover):
        if slider == move.to_square or before.piece_at(slider) != after.piece_at(slider):
            continue
        was = before.attacks(slider)
        for target in after.attacks(slider):
            if target in was:
                continue
            piece = after.piece_at(target)
            if piece is None or piece.color != enemy:
                continue
            if not chess.BB_SQUARES[move.from_square] & chess.between(slider, target):
                continue
            if slider not in squares:
                squares.append(slider)
            squares.append(target)
    if not squares:
        return None
    return _tag("discoveredAttack", [move.from_square, *squares], move)


def _interference(
    before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color
) -> MotifTag | None:
    """The moved piece lands on the line between an enemy slider and the man it defended.

    Crisp because it is geometric plus a before/after test: the defended piece must lose its
    last defender and come under attack from the mover.
    """
    enemy = not mover
    for slider in _sliders(before, enemy):
        if slider == move.to_square or before.piece_at(slider) != after.piece_at(slider):
            continue
        for guarded in before.attacks(slider):
            if guarded == move.to_square:
                continue
            piece = before.piece_at(guarded)
            if piece is None or piece.color != enemy:
                continue
            if _value(piece.piece_type) < _MINOR or piece.piece_type == chess.KING:
                continue
            if not chess.BB_SQUARES[move.to_square] & chess.between(slider, guarded):
                continue
            if guarded in after.attacks(slider) or after.attackers(enemy, guarded):
                continue
            if not after.attackers(mover, guarded):
                continue
            return _tag("interference", [move.to_square, slider, guarded], move)
    return None


def _trapped_tags(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """An enemy piece attacked by something cheaper with nowhere safe to go.

    Minor pieces count, not only rooks and queens: a knight or bishop hemmed in and hunted by
    a pawn is trapped just as a rook is. A piece merely left undefended is caught as a loose
    piece instead; trapped is the one with no square to flee to. Needs the enemy to move, so
    it is skipped whenever either king is in check: check evasions would make every piece look
    trapped. One tag per piece, so a piece that was already trapped before a move can be told
    apart from one the move just shut in.
    """
    enemy = not mover
    enemy_king = view.king(enemy)
    mover_king = view.king(mover)
    if enemy_king is None or mover_king is None:
        return []
    if view.attackers(mover, enemy_king) or view.attackers(enemy, mover_king):
        return []
    escapes: dict[chess.Square, list[chess.Move]] = {}
    for move in view.legal_moves:
        escapes.setdefault(move.from_square, []).append(move)
    trapped: list[MotifTag] = []
    for square in sorted(view.piece_map()):
        piece = view.piece_at(square)
        if piece is None or piece.color != enemy or _value(piece.piece_type) < _MINOR:
            continue
        mine = _value(piece.piece_type)
        hunters = view.attackers(mover, square)
        if not any(_value(view.piece_type_at(h)) < mine for h in hunters):
            continue
        safe = False
        for move in escapes.get(square, []):
            victim = view.piece_at(move.to_square)
            if victim is not None and _value(victim.piece_type) >= mine:
                safe = True
                break
            if not view.is_attacked_by(mover, move.to_square):
                safe = True
                break
        if not safe:
            trapped.append(_tag("trappedPiece", [square, *hunters]))
    return trapped


def _attacking_f2f7(
    before: chess.Board, after: chess.Board, move: chess.Move, mover: chess.Color
) -> MotifTag | None:
    """The move takes aim at the beginner's square: f7 as white, f2 as black."""
    target = chess.F7 if mover == chess.WHITE else chess.F2
    piece = after.piece_at(target)
    if piece is None or piece.color == mover:
        return None
    if piece.piece_type not in (chess.PAWN, chess.KING):
        return None
    if target not in after.attacks(move.to_square):
        return None
    if target in before.attacks(move.from_square):
        return None
    return _tag("attackingF2F7", [move.to_square, target], move)


def _exposed_king(view: chess.Board, mover: chess.Color) -> MotifTag | None:
    """The enemy king has lost its pawn shelter and the mover has force bearing on it.

    Shelter is counted one pawn per file over the three files around the king, one or two
    ranks in front. At most one such pawn, at least two of the mover's pieces actually
    attacking the king's ring (one of them a queen or rook), and at least two ring squares
    under fire. Counting pieces on the ring rather than pieces on the board is what keeps an
    opening king that just pushed a pawn from looking besieged. The theme ranks low, so a
    sharper motif in the same position wins out.
    """
    enemy = not mover
    king = view.king(enemy)
    if king is None:
        return None
    king_file, king_rank = chess.square_file(king), chess.square_rank(king)
    step = 1 if enemy == chess.WHITE else -1
    shelter = 0
    for df in (-1, 0, 1):
        file_ = king_file + df
        if not 0 <= file_ <= 7:
            continue
        for distance in (1, 2):
            rank_ = king_rank + step * distance
            if not 0 <= rank_ <= 7:
                continue
            piece = view.piece_at(chess.square(file_, rank_))
            if piece is not None and piece.color == enemy and piece.piece_type == chess.PAWN:
                shelter += 1
                break
    if shelter > 1:
        return None
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king])
    hit = [s for s in ring if view.is_attacked_by(mover, s)]
    if len(hit) < 2:
        return None
    # The force has to be bearing on the king's ring, not merely present on the board: an
    # opening king that has just lost a pawn is not exposed while every piece is still at home.
    ring_attackers: set[chess.Square] = set()
    has_heavy = False
    for square in ring:
        for attacker in view.attackers(mover, square):
            piece_type = view.piece_type_at(attacker)
            if piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
                ring_attackers.add(attacker)
                has_heavy = has_heavy or piece_type in (chess.QUEEN, chess.ROOK)
    if len(ring_attackers) < 2 or not has_heavy:
        return None
    return _tag("exposedKing", [king, *hit])


def _static_tags(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Everything readable from the attack maps alone, from `mover`'s point of view.

    `view` must have the enemy to move, which is what the position after a move looks like.
    """
    tags = _ray_motifs(view, mover)
    tags.extend(_hanging_tags(view, mover))
    tags.extend(_trapped_tags(view, mover))
    exposed = _exposed_king(view, mover)
    if exposed is not None:
        tags.append(exposed)
    return tags


def _enemy_to_move(board: chess.Board, mover: chess.Color) -> chess.Board:
    if board.turn != mover:
        return board
    view = board.copy(stack=False)
    view.turn = not mover
    view.ep_square = None
    return view


_RAY_NAMES = ("pin", "absolutePin", "skewer", "xRayAttack")


def _static_key(tag: MotifTag) -> tuple[str, tuple[str, ...]]:
    """Identity of a static feature, so the same feature before and after a move matches.

    Only the part that makes the feature what it is goes into the key. A pin is which piece is
    shielding which, not which of my pieces happens to be doing the pinning, so sliding a rook
    along the rank it was already pinning on does not count as creating a pin. Likewise the
    king being out in the open is the feature, not which ring squares are hit this ply, and a
    trapped piece is the piece with nowhere to go, not the current list of hunters.
    """
    if tag.name in _RAY_NAMES:
        return (tag.name, tuple(tag.squares[1:]))
    if tag.name in ("exposedKing", "trappedPiece"):
        return (tag.name, tuple(tag.squares[:1]))
    return (tag.name, tuple(tag.squares))


def _new_static_tags(before: chess.Board, after: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Static features the move brought about, and only those.

    A pin the bishop was already making before the move is not something the move did, so it
    does not belong in the move's tags. It is still reported by `tags_for_position`.
    """
    prior = {_static_key(tag) for tag in _static_tags(_enemy_to_move(before, mover), mover)}
    return [tag for tag in _static_tags(after, mover) if _static_key(tag) not in prior]


# --- group C: what the line does --------------------------------------------------------


def _sacrifice(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color, mate_in: int | None
) -> MotifTag | None:
    """This move gives up at least two points to the opponent's reply, and the line still wins.

    The judgement is about the move played, not about the line it leads to. Taking the
    lowest point anywhere in a twelve-ply variation would call almost everything a
    sacrifice, since the opponent captures first in half of all trades and the material
    comes straight back: measured that way, `2. Nf3` in the game of 2026-09-14 20:06 was a
    sacrifice. So the balance is read after the move and the opponent's answer to it.

    Two points, so a pawn gambit does not count. The deficit must survive the mover's own
    next move as well: a reply that captures and is taken straight back is a trade, and
    measured after the reply alone, 5...d6 in a real game was a sacrifice because 6.Bxc6+
    bxc6 followed. "Still wins" is a mate for the mover, or the material coming back by the
    end of the line: a piece given for nothing at all is a blunder, and the assessment
    already says so.
    """
    if len(played) < 2:
        return None
    start = _balance(boards[0], mover)
    answered = _balance(boards[2], mover) if len(boards) > 2 else _balance(boards[-1], mover)
    if start - answered < _SACRIFICE_THRESHOLD:
        return None
    if len(boards) > 3 and start - _balance(boards[3], mover) < _SACRIFICE_THRESHOLD:
        return None
    mating = mate_in is not None and mate_in > 0
    if not mating and _balance(boards[-1], mover) < start:
        return None
    return _tag(
        "sacrifice", [played[0].from_square, played[0].to_square, played[1].to_square], played[0]
    )


def _clearance(boards: list[chess.Board], played: list[chess.Move]) -> MotifTag | None:
    """The move vacates a square with tempo and a later move of the mover's uses that square.

    Tempo means the move is a capture or a check, which is what separates a clearance from a
    piece that merely happened to be in the way.
    """
    if len(played) < 3:
        return None
    first = played[0]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    vacated = first.from_square
    for i in range(2, len(played), 2):
        later = played[i]
        if later.to_square == vacated:
            return _tag("clearance", [vacated, later.from_square, later.to_square], first)
        if chess.BB_SQUARES[vacated] & chess.between(later.from_square, later.to_square):
            return _tag("clearance", [vacated, later.from_square, later.to_square], first)
    return None


def _capturing_defender(boards: list[chess.Board], played: list[chess.Move]) -> MotifTag | None:
    """The move captures a piece and a later move of the mover's lands on a square it guarded."""
    first = played[0]
    if not boards[0].is_capture(first):
        return None
    victim_square = _ep_victim(first) if boards[0].is_en_passant(first) else first.to_square
    guarded = boards[0].attacks(victim_square)
    for i in range(2, len(played), 2):
        later = played[i]
        if later.to_square not in guarded:
            continue
        position = boards[i]
        if position.is_capture(later) or position.gives_check(later):
            return _tag("capturingDefender", [victim_square, later.to_square], first)
    return None


def _deflection(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """A forcing move pulls the only defender of a square away, and the mover takes the square.

    Forcing means the move is a capture or a check, so the reply is the engine's, not a
    choice. "Only defender" is checked on the position after the move.
    """
    if len(played) < 3:
        return None
    first, reply, follow = played[0], played[1], played[2]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    defender = reply.from_square
    guards = boards[1].attackers(not mover, follow.to_square)
    if len(guards) != 1 or defender not in guards:
        return None
    if follow.to_square not in boards[1].attacks(defender):
        return None
    if follow.to_square in boards[2].attacks(reply.to_square):
        return None
    if not boards[2].is_capture(follow) and not boards[2].gives_check(follow):
        return None
    return _tag("deflection", [defender, reply.to_square, follow.to_square], first)


def _attraction(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """A forcing move drags an enemy piece onto a square where the next move hits it.

    The enemy is dragged by capturing, so the reply must be a capture; the piece pulled has
    to be worth at least a knight (the king included); and the mover's follow-up must land on
    that square or attack it while capturing or checking.
    """
    if len(played) < 3:
        return None
    first, reply, follow = played[0], played[1], played[2]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    if not boards[1].is_capture(reply):
        return None
    landed = reply.to_square
    attracted = boards[2].piece_at(landed)
    if attracted is None or attracted.color == mover or _value(attracted.piece_type) < _MINOR:
        return None
    if not boards[2].is_capture(follow) and not boards[2].gives_check(follow):
        return None
    hits = follow.to_square == landed or landed in boards[3].attacks(follow.to_square)
    if not hits:
        return None
    return _tag("attraction", [reply.from_square, landed, follow.to_square], first)


def _mate_squares(position: chess.Board) -> list[chess.Square]:
    king = position.king(position.turn)
    checkers = list(position.checkers())
    return checkers if king is None else [*checkers, king]


def _mate_tags(
    after: chess.Board, final: chess.Board, mover: chess.Color, mate_in: int | None
) -> list[MotifTag]:
    distance = 1 if after.is_checkmate() else mate_in
    if distance is None or distance <= 0:
        return []
    if after.is_checkmate():
        squares = _mate_squares(after)
    elif final.is_checkmate() and final.turn != mover:
        squares = _mate_squares(final)
    else:
        king = after.king(not mover)
        squares = [] if king is None else [king]
    tags = [_tag("mate", squares)]
    if 1 <= distance <= 5:
        tags.append(_tag(f"mateIn{distance}", squares))
    return tags


# --- group D: mating patterns -----------------------------------------------------------


def _diagonal_kind(a: chess.Square, b: chess.Square) -> str | None:
    fa, ra = chess.square_file(a), chess.square_rank(a)
    fb, rb = chess.square_file(b), chess.square_rank(b)
    if fa - ra == fb - rb:
        return "main"
    if fa + ra == fb + rb:
        return "anti"
    return None


def _defenders(position: chess.Board, color: chess.Color, square: chess.Square) -> chess.SquareSet:
    return position.attackers(color, square)


def _mating_pattern_tags(position: chess.Board) -> list[MotifTag]:
    """Each pattern is emitted only when its shape matches exactly.

    `dovetailMate` and the rest are defined on the mated king, the single checker, and the
    pieces that support it, so two patterns can only both fire when the position genuinely
    contains both shapes.
    """
    loser = position.turn
    winner = not loser
    king = position.king(loser)
    if king is None:
        return []
    checkers = list(position.checkers())
    ring = chess.SquareSet(chess.BB_KING_ATTACKS[king])
    own_ring = [s for s in ring if (p := position.piece_at(s)) is not None and p.color == loser]
    king_file, king_rank = chess.square_file(king), chess.square_rank(king)
    tags: list[MotifTag] = []

    back_rank = 0 if loser == chess.WHITE else 7
    if king_rank == back_rank:
        forward = back_rank + (1 if loser == chess.WHITE else -1)
        front = [
            chess.square(f, forward)
            for f in (king_file - 1, king_file, king_file + 1)
            if 0 <= f <= 7
        ]
        blocked = all((p := position.piece_at(s)) is not None and p.color == loser for s in front)
        heavy = [
            c
            for c in checkers
            if position.piece_type_at(c) in (chess.ROOK, chess.QUEEN)
            and chess.square_rank(c) == back_rank
        ]
        if blocked and heavy:
            tags.append(_tag("backRankMate", [king, *heavy, *front]))

    if len(checkers) == 1:
        checker = checkers[0]
        kind = position.piece_type_at(checker)
        adjacent = chess.square_distance(checker, king) == 1
        knights = [
            s
            for s in _defenders(position, winner, checker)
            if position.piece_type_at(s) == chess.KNIGHT
        ]

        if kind == chess.KNIGHT and len(own_ring) == len(ring):
            tags.append(_tag("smotheredMate", [king, checker, *own_ring]))

        if kind == chess.ROOK and adjacent and knights and king in _CORNERS:
            tags.append(_tag("arabianMate", [king, checker, knights[0]]))

        if kind == chess.ROOK and adjacent and knights:
            pawns = [
                s
                for s in _defenders(position, winner, knights[0])
                if position.piece_type_at(s) == chess.PAWN
            ]
            if pawns:
                tags.append(_tag("hookMate", [king, checker, knights[0], pawns[0]]))

        if kind == chess.ROOK and adjacent:
            queens = [
                s
                for s in _defenders(position, winner, checker)
                if position.piece_type_at(s) == chess.QUEEN
            ]
            box = [q for q in queens if chess.square_knight_distance(q, king) == 1]
            if box:
                tags.append(_tag("killBoxMate", [king, checker, box[0]]))

        on_edge = king_file in (0, 7) or king_rank in (0, 7)
        if kind == chess.ROOK and adjacent and on_edge:
            others = [
                s
                for s in _defenders(position, winner, checker)
                if position.piece_type_at(s) != chess.KNIGHT
            ]
            covers = [
                s
                for s in ring
                if position.piece_at(s) is None
                and any(s in position.attacks(n) for n in position.pieces(chess.KNIGHT, winner))
            ]
            if others and covers:
                tags.append(_tag("vukovicMate", [king, checker, others[0], *covers]))

        if kind == chess.ROOK and king_file in (0, 7) and chess.square_file(checker) == king_file:
            inner = 1 if king_file == 0 else 6
            above, below = king_rank + 1, king_rank - 1
            if below >= 0 and above <= 7:
                beside = chess.square(inner, king_rank)
                blocker = position.piece_at(beside)
                flights = (chess.square(inner, above), chess.square(inner, below))
                covered = any(
                    all(f in position.attacks(n) for f in flights)
                    for n in position.pieces(chess.KNIGHT, winner)
                )
                if blocker is not None and blocker.color == loser and covered:
                    tags.append(_tag("anastasiaMate", [king, checker, beside, *flights]))

        if kind == chess.BISHOP:
            check_kind = _diagonal_kind(checker, king)
            for other in position.pieces(chess.BISHOP, winner):
                if other == checker:
                    continue
                for square in ring:
                    if square not in position.attacks(other):
                        continue
                    other_kind = _diagonal_kind(other, square)
                    if other_kind is None or check_kind is None:
                        continue
                    evidence = [king, checker, other, square]
                    if other_kind == check_kind and on_edge:
                        tags.append(_tag("doubleBishopMate", evidence))
                    elif other_kind != check_kind and len(own_ring) >= 2:
                        tags.append(_tag("bodenMate", evidence))
                    break

        if kind == chess.QUEEN and len(ring) == 8 and _defenders(position, winner, checker):
            diagonal = abs(chess.square_file(checker) - king_file) == 1
            diagonal = diagonal and abs(chess.square_rank(checker) - king_rank) == 1
            orthogonal = [
                s
                for s in own_ring
                if chess.square_file(s) == king_file or chess.square_rank(s) == king_rank
            ]
            if diagonal and len(own_ring) == 2 and len(orthogonal) == 2:
                tags.append(_tag("dovetailMate", [king, checker, *own_ring]))

    return tags


# --- public API --------------------------------------------------------------------------


def tags_for_move(
    before: chess.Board,
    move: chess.Move,
    *,
    pv: Sequence[chess.Move] = (),
    mate_in: int | None = None,
) -> list[MotifTag]:
    """What the move does, from the mover's point of view.

    `pv` is the continuation the engine expects AFTER `move` (it does not include `move`).
    `mate_in` counts the mover's own moves including this one: 1 means this move mates, 3
    means this move plus two more, negative means the mover gets mated, None means no mate.

    An illegal move returns an empty list rather than raising: the caller feeds this from a
    vision pipeline that can misread a board, and a bad read should not crash a live game.
    """
    if move not in before.legal_moves:
        return []
    mover = before.turn
    boards, played = _line_boards(before, [move, *pv])
    after = boards[1]

    tags = _group_a(before, after, move, mover)
    tags.extend(_new_static_tags(before, after, mover))
    for tag in (
        _fork(after, move, mover, won_in_line=_mover_captures(boards, played, mover)),
        _discovered_attack(before, after, move, mover),
        _interference(before, after, move, mover),
        _attacking_f2f7(before, after, move, mover),
        _sacrifice(boards, played, mover, mate_in),
        _clearance(boards, played),
        _capturing_defender(boards, played),
        _deflection(boards, played, mover),
        _attraction(boards, played, mover),
    ):
        if tag is not None:
            tags.append(tag)
    tags.extend(_mate_tags(after, boards[-1], mover, mate_in))

    if after.is_checkmate():
        tags.extend(_mating_pattern_tags(after))
    elif boards[-1].is_checkmate() and boards[-1].turn != mover:
        tags.extend(_mating_pattern_tags(boards[-1]))
    return _finish(tags)


def tags_for_position(
    board: chess.Board,
    *,
    pv: Sequence[chess.Move] = (),
    mate_in: int | None = None,
) -> list[MotifTag]:
    """What is in the position for the side to move.

    The static features (pins, skewers, x-rays, loose pieces, trapped pieces, an exposed
    king) plus the tags of the engine's best move, which is `pv[0]` when a PV is given.
    `mate_in` uses the same convention as `tags_for_move`, counting the side to move's own
    moves starting with `pv[0]`.
    """
    mover = board.turn
    tags = _static_tags(_enemy_to_move(board, mover), mover)

    if pv and pv[0] in board.legal_moves:
        tags.extend(tags_for_move(board, pv[0], pv=pv[1:], mate_in=mate_in))
    elif mate_in is not None and mate_in > 0:
        king = board.king(not mover)
        squares: list[chess.Square] = [] if king is None else [king]
        tags.append(_tag("mate", squares))
        if mate_in <= 5:
            tags.append(_tag(f"mateIn{mate_in}", squares))
    return _finish(tags)
