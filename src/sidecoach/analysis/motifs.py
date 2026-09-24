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

1. A tactic is named only when it is tactically real: it wins something, never as geometry
   alone (Tai, 2026-09-23). A tag that claims material has to gain it once defenders are
   counted: a knight forking two rooks while a pawn can take it is not a fork, and a loose
   piece must really be undefended. A pin is a claim too: a piece is pinned when leaving
   the line would lose what stands behind it -- the king, or a piece the pinner could take
   for profit -- and a pinner the enemy simply wins pins nothing. A queen "pinning" a knight
   to a defended rook pins nothing, whatever the geometry; Lichess tags it, the coach must
   not teach it. The same goes for the rest: a discovered attack reveals something the
   slider can win (or the king), a skewer still wins the piece behind whatever the front
   piece does, and the line motifs (deflection, attraction, removing the defender,
   interference, clearance) end in material won or mate. "Can take" and "defends" are read
   off legal moves (`exchange`): a piece pinned to its own king takes nothing.
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

from sidecoach.analysis import exchange, mates
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
    "swallowstailMate",
    "epauletteMate",
    "triangleMate",
    "blindSwineMate",
    "cornerMate",
    "operaMate",
    "pillsburysMate",
    "morphysMate",
    "balestraMate",
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


def _as_side(board: chess.Board, color: chess.Color) -> chess.Board:
    """`board` with `color` to move, the way `exchange` reads a threat: a null move hands the
    turn over, so no en passant right is carried across."""
    if board.turn == color:
        return board
    view = board.copy(stack=False)
    view.push(chess.Move.null())
    return view


# away, was the headline of a promotion (Tai's puzzles, 2026-09-23).
TARGETED = (
    "fork",
    "skewer",
    "discoveredAttack",
    "absolutePin",
    "pin",
    "trappedPiece",
    "hangingPiece",
    "capturingDefender",
)
# A fork or a skewer is the point only when what it wins is worth a minor piece: the king
# and a pawn on a2 forked five moves after the queen was won was named as the idea.
MIN_TARGET_VALUE = 3
# How far on the line takes it: three of the player's moves.
PAYOFF_PLIES = 6


def pays_off(
    boards: Sequence[chess.Board], pv: Sequence[chess.Move], i: int, tag: MotifTag
) -> bool:
    """Whether the line cashes in the tactic at ply `i` within `PAYOFF_PLIES`: takes a
    piece it goes after -- by the piece that does it, where the pattern says so -- with
    every piece followed as it moves. The squares are the tagger's, per pattern: a fork's
    forker then its targets; a skewer's and a pin's slider, front, back; a discovered
    attack's mover, slider, target; a trapped piece then its hunters; a loose piece; a
    captured defender then what it guarded."""
    us = boards[i].turn
    sq = [chess.parse_square(s) for s in tag.squares]
    if not sq:
        return False
    name = tag.name
    after = boards[i + 1]

    def theirs(square: chess.Square) -> bool:
        piece = after.piece_at(square)
        return piece is not None and piece.color != us and piece.piece_type != chess.KING

    def worth(square: chess.Square) -> int:
        return exchange.VALUE.get(after.piece_type_at(square) or chess.PAWN, 0)

    actors: set[chess.Square] | None = None  # None: any of the player's pieces
    # The piece the pattern stands on: a pin is gone once its pinner is taken (a rook
    # that pins as it takes the queen, and is taken back at once, pinned nothing).
    holder: chess.Square | None = None
    # A fork's piece taken by anything but one of its targets refutes it: "Nf3+ forks the
    # king, queen and rook" was said of a knight gxf3 simply took (Lichess 026RF).
    forker: chess.Square | None = None
    if name == "fork":
        # Whoever collects it: the queen that takes back on e6 after Bxe6+ Qxe6 cashed in
        # the bishop's fork (Lichess 0BD4O).
        aimed = [s for s in sq[1:] if theirs(s) and worth(s) >= MIN_TARGET_VALUE]
        forker = sq[0]
        # A bystander that can take the forking piece for profit where it lands refutes
        # the fork, whatever the line then does (Nf2+ with the rook on f1 able to take it:
        # the discovered attack behind it was the point, Lichess 01q4A). A pinned piece
        # cannot take; a target that takes and is taken back is the fork working.
        for capture in exchange.captures_on(after, not us, forker):
            if capture.from_square not in sq[1:] and exchange.exchange(after, capture) > 0:
                return False
    elif name == "skewer":
        aimed = [s for s in sq[2:3] if theirs(s) and worth(s) >= MIN_TARGET_VALUE]
        actors = {sq[0]}
    elif name == "discoveredAttack" and len(sq) >= 3:
        king = after.piece_at(sq[2])
        if king is not None and king.piece_type == chess.KING:
            # A discovered check: the point is what the piece that moved goes on to take.
            aimed = [s for s in chess.SQUARES if theirs(s)]
            actors = {sq[0]}
        else:
            aimed = [sq[2]] if theirs(sq[2]) else []
            actors = {sq[0], sq[1]}
    elif name in ("pin", "absolutePin"):
        aimed = [s for s in sq[1:3] if theirs(s)]
        holder = sq[0]
    elif name in ("trappedPiece", "hangingPiece"):
        aimed = [sq[0]] if theirs(sq[0]) else []
    elif name == "capturingDefender":
        # Taking their queen is its own point, not the removal of the pawn's guard: what
        # was guarded must be worth what was taken.
        taken = boards[i].piece_at(sq[0])
        taken_worth = exchange.VALUE.get(taken.piece_type, 0) if taken is not None else 0
        aimed = [s for s in sq[1:2] if theirs(s) and worth(s) >= taken_worth]
    else:
        return True
    if pv[i].to_square in aimed and boards[i].is_capture(pv[i]):
        return True  # the move itself takes it
    watched = set(aimed)
    for j in range(i + 1, min(len(pv), i + 1 + PAYOFF_PLIES)):
        move = pv[j]
        if boards[j].turn == us:
            if (
                move.to_square in watched
                and boards[j].is_capture(move)
                and (actors is None or move.from_square in actors)
            ):
                return True
            if actors is not None and move.from_square in actors:
                actors = (actors - {move.from_square}) | {move.to_square}
            if holder is not None and move.from_square == holder:
                holder = move.to_square
            if forker is not None and move.from_square == forker:
                forker = move.to_square
        else:
            if forker is not None and move.to_square == forker:
                if move.from_square not in watched:
                    return False  # the fork's piece taken by a bystander: refuted
                forker = None
            if move.from_square in watched:
                watched = (watched - {move.from_square}) | {move.to_square}
            if actors is not None:
                actors.discard(move.to_square)  # a piece of the player's taken
            if holder is not None and move.to_square == holder:
                return False  # the pinner taken: the pin is gone
    return False


def _capture_gain(
    board: chess.Board, color: chess.Color, frm: chess.Square, to: chess.Square
) -> int | None:
    """What the piece of `color` on `frm` nets by taking on `to`, the recaptures played out
    with legal moves (`exchange.exchange`); None when it cannot legally take there at all --
    pinned to its own king, blocked, gone, or the king in check with the capture no answer."""
    view = _as_side(board, color)
    gains = [
        exchange.exchange(view, m)
        for m in exchange.captures_on(view, color, to)
        if m.from_square == frm
    ]
    return max(gains) if gains else None


def _mates(boards: Sequence[chess.Board], mover: chess.Color) -> bool:
    """The line ends with the mover's opponent checkmated."""
    return boards[-1].is_checkmate() and boards[-1].turn != mover


def _wins_material(
    boards: Sequence[chess.Board],
    played: Sequence[chess.Move],
    mover: chess.Color,
    *,
    square: chess.Square | None = None,
    upto: int | None = None,
) -> bool:
    """Whether one of the mover's later captures (from its second move on) leaves it ahead of
    where it stood before the first move: the balance before that capture plus what the
    capture nets by static exchange. Counting the exchange rather than the bare balance at
    the end of the line means a line cut off mid-trade neither gains nor loses a piece.
    `square` keeps to captures on that square; `upto` stops at that ply."""
    start = _balance(boards[0], mover)
    last = len(played) if upto is None else min(len(played), upto + 1)
    for k in range(2, last, 2):
        later = played[k]
        if square is not None and later.to_square != square:
            continue
        if not boards[k].is_capture(later):
            continue
        if _balance(boards[k], mover) + exchange.exchange(boards[k], later) > start:
            return True
    return False


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

    Running away means the piece could have been taken for profit where it stood and cannot
    be on its new square. Defending means the move newly covers a friendly piece the opponent
    could have won before the move, is still attacking after it, and can no longer win. A
    guard that does not stop the loss is no rescue: a knight covering a rook that a knight
    attacks still loses the exchange. Nor is one that is itself lost where it lands, and
    adding a second guard to a piece that was already holding does not count either.
    """
    enemy = not mover
    if not is_safe(after, move.to_square):
        return None
    if not is_safe(before, move.from_square):
        return [move.from_square, move.to_square]
    covered_before = before.attacks(move.from_square)
    for square in after.attacks(move.to_square):
        if square in covered_before:
            continue
        piece = after.piece_at(square)
        if piece is None or piece.color != mover:
            continue
        if is_safe(before, square) or not exchange.capturers(after, enemy, square):
            continue
        if not is_safe(after, square):
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
        # Bookkeeping, not a teaching claim: the move is not a capture, a check or a
        # promotion, and that is all. Lichess's quietMove is narrower (a quiet move that
        # makes no threat and still wins); the coach never names this one as a theme and the
        # panel hides it. It gates `defensiveMove` and records what kind of move it was.
        tags.append(_tag("quietMove", [frm, to], move))
        evidence = _defensive_evidence(before, after, move, mover)
        if evidence is not None:
            tags.append(_tag("defensiveMove", evidence, move))
    return tags


# --- group B: what the move creates -----------------------------------------------------


def winnable(
    view: chess.Board, square: chess.Square, cost: int, ignore: chess.Square | None = None
) -> bool:
    """Would taking the piece on `square` with a `cost`-valued piece gain material?

    True when the piece is worth more than what takes it, or when nothing can legally take
    back (`exchange.defenders`: a defender pinned to its own king does not count, nor a
    king that would step onto a guarded square). `ignore` drops one piece from the board
    first, which is how a pinned or skewered piece stops counting as the guard of the piece
    behind it: it is the one being driven off the line.
    """
    piece = view.piece_at(square)
    if piece is None:
        return False
    if piece.piece_type == chess.KING or _value(piece.piece_type) > cost:
        return True
    probe = view
    if ignore is not None:
        probe = view.copy(stack=False)
        probe.remove_piece_at(ignore)
    return not exchange.defenders(probe, piece.color, square)


def is_safe(view: chess.Board, square: chess.Square) -> bool:
    """Can the piece on `square` be taken for profit? Safe means no, so it can stay. Read
    off the static exchange with legal moves (`exchange.can_be_won`): an attacker pinned to
    its own king cannot take -- the knight forking king and queen is safe from the pawn
    that is pinned (the "pin" of Lichess's fork puzzles) -- and an x-ray piece behind the
    first attacker counts."""
    return not exchange.can_be_won(view, square)


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
    a fork, which is what keeps pawn pushes out of this. The forker has to be able to take a
    target by the rules: a piece pinned to its own king bears on the pieces around it but may
    take none of them. And the forking piece must be safe where it stands, a check included:
    a knight forking two rooks while a pawn can take it wins nothing, and a knight checking
    king and queen that the g-pawn simply takes wins nothing either -- taking the checker is
    a legal answer to the check, which the exchange with legal moves counts.

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
    for square in after.attacks(move.to_square):
        victim = after.piece_at(square)
        if victim is None or victim.color != enemy:
            continue
        king = victim.piece_type == chess.KING
        if king or _fork_target(
            after, move.to_square, square, mine, mover, gives_check, won_in_line
        ):
            targets.append(square)
    if len(targets) < 2:
        return None
    if not is_safe(after, move.to_square):
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

    Only a piece the forker can legally take (`exchange.capturers`), and "undefended" means
    nothing can legally take back (`exchange.defenders`). A cheap loose piece counts when the
    move gives check and either the piece cannot be saved against any reply
    (`_check_fork_wins`) or the engine's own line actually captures it -- the latter settles
    the cases where the enemy has a defence that only fails to deeper calculation, which is
    the engine's job, not the geometry's.
    """
    if forker not in exchange.capturers(after, mover, square):
        return False
    enemy = not mover
    theirs = _value(after.piece_type_at(square))
    loose = not exchange.defenders(after, enemy, square)
    if theirs > mine or (theirs >= mine and loose):
        return True
    return (
        gives_check
        and loose
        and (square in won_in_line or _check_fork_wins(after, forker, square, mover))
    )


def _check_fork_wins(
    after: chess.Board, forker: chess.Square, target: chess.Square, mover: chess.Color
) -> bool:
    """After a checking move, is the loose enemy piece on `target` truly won?

    True only when no legal reply to the check rescues it: for every reply the piece is still
    there, the forker can still legally take it, and taking it still gains material. A reply
    that captures the forker, guards the piece, moves it to safety, or blocks with a piece
    that pins the forker to its own king means the check-fork does not win it.
    """
    replies = list(after.legal_moves)
    if not replies:  # checkmate: the mate tags cover this, it is not a fork win
        return False
    enemy = not mover
    for reply in replies:
        nxt = after.copy(stack=False)
        nxt.push(reply)
        victim = nxt.piece_at(target)
        if victim is None or victim.color != enemy:
            return False  # the piece moved, was traded, or the square changed hands
        gain = _capture_gain(nxt, mover, forker, target)
        if gain is None:
            return False  # the forker cannot take it now (captured, pinned, blocked)
        if gain <= 0:
            return False  # it is defended now
    return True


def _hanging_tags(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Enemy pieces the mover can take with nothing to take back, one tag per piece.

    Read off legal moves (`exchange.hanging`): a piece attacked only by a piece pinned to its
    own king is not hanging, nor is one the mover cannot take because it has a check to
    answer first. Pawns are excluded: loose pawns are everywhere and a coach calling one
    "hanging" is noise. One tag per piece so that a piece that was already loose before a
    move can be told apart from one the move just left en prise.
    """
    enemy = not mover
    tags: list[MotifTag] = []
    for square in sorted(view.piece_map()):
        piece = view.piece_at(square)
        if piece is None or piece.color != enemy:
            continue
        if piece.piece_type in (chess.PAWN, chess.KING):
            continue
        if exchange.hanging(view, square):
            tags.append(_tag("hangingPiece", [square]))
    return tags


def _ray_motifs(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """Pins and skewers: one mover slider, one enemy piece, one enemy piece behind it.

    A pin is tactical, not only geometric: an enemy piece with a more valuable enemy piece
    behind it on the slider's line is pinned when leaving the line would lose that piece --
    the king always, another piece when the slider could legally take it for profit once the
    front piece has gone. A pin against a defended piece worth no more than the pinner wins
    nothing, and the coach said such pins made no sense (Tai, 2026-09-23). A pin also has to
    hold something back: a pawn on the file whose only move is straight ahead is not stopped
    from anything (`_leaves_line`).

    Neither holds when the enemy simply wins the slider (`is_safe`): a rook "pinning" a rook
    that takes it with check, or a bishop pinning a knight while a pawn takes the bishop.
    A slider pinned to its own king can never take what is behind, so it pins nothing but
    the king: a pin to the king stays, because the front piece still cannot move by the
    rules, but a relative pin needs the slider to be able to cash it in.

    A skewer names a piece to go after, so it requires the piece behind to be winnable, and
    it must still be winnable whatever the front piece does (`_skewer_holds`): a queen that
    steps off the line to guard the rook behind has answered it. The x-ray is a line motif,
    not a static one, and lives in `_xray`.
    """
    enemy = not mover
    tags: list[MotifTag] = []
    for slider in _sliders(view, mover):
        if not is_safe(view, slider):
            continue
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
                if not _leaves_line(view, slider, front_sq, back_sq):
                    continue
                if back.piece_type == chess.KING:
                    tags.append(_tag("pin", squares))
                    tags.append(_tag("absolutePin", squares))
                    continue
                probe = view.copy(stack=False)
                probe.remove_piece_at(front_sq)
                gain = _capture_gain(probe, mover, slider, back_sq)
                if gain is not None and gain > 0:
                    tags.append(_tag("pin", squares))
                continue
            if front_value > back_value:
                must_move = winnable(view, front_sq, cost)
                behind_is_loose = winnable(view, back_sq, cost, ignore=front_sq)
                if must_move and behind_is_loose and _skewer_holds(view, mover, squares):
                    tags.append(_tag("skewer", squares))
    return tags


def _leaves_line(
    view: chess.Board, slider: chess.Square, front_sq: chess.Square, back_sq: chess.Square
) -> bool:
    """Whether the piece on `front_sq` has a move off the slider's line, the pin ignored.

    A pin restricts only a piece that had somewhere else to go: the pawn in front of the
    queen on a file whose only move is a step down that file is held back from nothing.
    Pseudo-legal moves are what the piece could do were it not pinned.
    """
    front = view.piece_at(front_sq)
    if front is None:
        return False
    line = chess.ray(slider, back_sq)
    side = _as_side(view, front.color)
    moves = side.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[front_sq])
    return any(not chess.BB_SQUARES[m.to_square] & line for m in moves)


def _skewer_holds(view: chess.Board, mover: chess.Color, squares: Sequence[chess.Square]) -> bool:
    """Whether every way the front piece can step off the line still loses the piece behind.

    For each legal move of the front piece off the line, the slider must still be able to
    take the piece behind for a gain (`exchange.exchange` > 0). A queen that answers a rook
    skewer by stepping to a square that guards the rook behind has met it: Rxb8 Qxb8 is only
    a trade. A front piece with no way off the line is not skewered but simply lost, which
    the loose and trapped pieces cover.
    """
    slider, front_sq, back_sq = squares
    front = view.piece_at(front_sq)
    if front is None:
        return False
    line = chess.ray(slider, back_sq)
    side = _as_side(view, front.color)
    escapes = [
        m
        for m in side.legal_moves
        if m.from_square == front_sq and not chess.BB_SQUARES[m.to_square] & line
    ]
    if not escapes:
        return False
    for escape in escapes:
        nxt = side.copy(stack=False)
        nxt.push(escape)
        gain = _capture_gain(nxt, mover, slider, back_sq)
        if gain is None or gain <= 0:
            return False
    return True


def _discovered_attack(
    boards: list[chess.Board], played: Sequence[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """The move steps off a line and a friendly slider now bears on something it can win.

    The geometry first: the vacated square lies strictly between the slider and an enemy
    piece it now attacks and did not before. Then the claim: the revealed piece is the king
    (a discovered check), or the slider could take it for a gain (`exchange.exchange` > 0),
    or the slider does take it later in the line -- Lichess tags a discovery only when that
    capture happens. Revealing a bishop onto a pawn the g-pawn guards, as 2.d4 does to h6,
    is a line opening, not a discovered attack.
    """
    before, after, move = boards[0], boards[1], played[0]
    enemy = not mover
    later = {
        (played[k].from_square, played[k].to_square)
        for k in range(2, len(played), 2)
        if boards[k].is_capture(played[k])
    }
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
            if piece.piece_type != chess.KING and (slider, target) not in later:
                gain = _capture_gain(after, mover, slider, target)
                if gain is None or gain <= 0:
                    continue
            if slider not in squares:
                squares.append(slider)
            squares.append(target)
    if not squares:
        return None
    return _tag("discoveredAttack", [move.from_square, *squares], move)


def _interference(
    boards: list[chess.Board], played: Sequence[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """Cutting a line of defence, the way Lichess tags it, read over the line and by either side.

    Two shapes. The mover interposes between an enemy slider and a piece it was defending,
    cutting the guard and winning that piece (`_mover_interference`). Or the mover forces the
    enemy -- with a check, or by taking so that the reply is the recapture -- to answer with
    a move that blocks one of the enemy's own sliders off a square it was guarding, a square
    the mover takes or mates on with its very next move (`_self_interference`). The forced
    self-block is the common case; the mover's own interposition is the textbook one.
    """
    for i, move in enumerate(played):
        before, after = boards[i], boards[i + 1]
        if i % 2 == 0:
            taken = {
                played[k].to_square
                for k in range(i + 2, len(played), 2)
                if boards[k].is_capture(played[k])
            }
            tag = _mover_interference(before, after, move, mover, taken)
        else:
            recapture = before.is_capture(move) and move.to_square == played[i - 1].to_square
            if not before.is_check() and not recapture:
                continue  # a reply the enemy chose freely did not interfere under force
            exploited: set[chess.Square] = set()
            if i + 1 < len(played):
                nxt = played[i + 1]
                if boards[i + 1].is_capture(nxt) or boards[i + 2].is_checkmate():
                    exploited.add(nxt.to_square)
            tag = _self_interference(before, after, move, not mover, exploited)
        if tag is not None:
            return tag
    return None


def _mover_interference(
    before: chess.Board,
    after: chess.Board,
    move: chess.Move,
    mover: chess.Color,
    taken: set[chess.Square],
) -> MotifTag | None:
    """The mover lands on the line between an enemy slider and the piece it was defending.

    Geometric plus a before/after test with legal moves: the defended piece loses its last
    defender (`exchange.defenders`) and the mover can now take it for a gain -- which a piece
    pinned to its own king cannot, even standing on the line. And the interposer must hold
    its square (`is_safe`), or the line must go on to take the piece it cut off (`taken`,
    the squares the mover captures on later): a knight dropped between rook and pawn that
    the rook simply takes has cut nothing.
    """
    enemy = not mover
    holds = is_safe(after, move.to_square)
    for slider in _sliders(before, enemy):
        if slider == move.to_square or before.piece_at(slider) != after.piece_at(slider):
            continue
        for guarded in before.attacks(slider):
            if guarded == move.to_square:
                continue
            piece = before.piece_at(guarded)
            if piece is None or piece.color != enemy or piece.piece_type == chess.KING:
                continue
            if not chess.BB_SQUARES[move.to_square] & chess.between(slider, guarded):
                continue
            if guarded in after.attacks(slider) or exchange.defenders(after, enemy, guarded):
                continue
            if exchange.best_capture(after, mover, guarded) <= 0:
                continue
            if not holds and guarded not in taken:
                continue
            return _tag("interference", [move.to_square, slider, guarded], move)
    return None


def _self_interference(
    before: chess.Board,
    after: chess.Board,
    reply: chess.Move,
    defender: chess.Color,
    exploited: set[chess.Square],
) -> MotifTag | None:
    """A forced reply that blocks one of the replier's own sliders off a square it was guarding.

    The block lands on `reply.to_square`; a `defender` slider that reached that square cleanly
    and guarded a square the mover takes or mates on next (`exploited`) no longer guards it
    afterwards. That is the enemy interfering with its own defence. `_interference` hands in
    only forced replies -- a check evasion or a recapture -- and only the next move's square:
    a knight that retreats of its own accord in front of its queen interferes with nothing.
    """
    block = reply.to_square
    for slider in _sliders(before, defender):
        if slider == block or not _slider_reaches(before.piece_type_at(slider), slider, block):
            continue
        if any(before.piece_at(sq) for sq in chess.SquareSet(chess.between(slider, block))):
            continue
        for target in _beyond(slider, block):
            if before.piece_at(target) is not None and target not in exploited:
                break
            if (
                target in exploited
                and slider in exchange.defenders(before, defender, target)
                and slider not in after.attackers(defender, target)
            ):
                return _tag("interference", [target, block, slider], reply)
            if before.piece_at(target) is not None:
                break
    return None


def _beyond(frm: chess.Square, through: chess.Square) -> Iterable[chess.Square]:
    """Squares continuing along the frm -> through ray, past `through` to the board edge."""
    df = _sign(chess.square_file(through) - chess.square_file(frm))
    dr = _sign(chess.square_rank(through) - chess.square_rank(frm))
    file_, rank_ = chess.square_file(through) + df, chess.square_rank(through) + dr
    while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
        yield chess.square(file_, rank_)
        file_ += df
        rank_ += dr


def _trapped_tags(view: chess.Board, mover: chess.Color) -> list[MotifTag]:
    """An enemy piece the mover can win that no move of the enemy's saves.

    The piece can be taken for profit now (`exchange.can_be_won`, legal moves: a hunter
    pinned to its own king hunts nothing), and after every legal enemy move it still can
    (`_rescues`): not by fleeing, not by guarding it, not by taking the hunter with something
    else, not by a check. A flight square that is attacked but held is a flight square: the
    knight on h6 hit by a pawn is not trapped while f5 and g4 are guarded. Minor pieces count,
    not only rooks and queens: a knight or bishop hemmed in and hunted by a pawn is trapped
    just as a rook is. Needs the enemy to move, so it is skipped whenever either king is in
    check: check evasions would make every piece look trapped. One tag per piece, so a piece
    that was already trapped before a move can be told apart from one the move just shut in.
    """
    enemy = not mover
    enemy_king = view.king(enemy)
    mover_king = view.king(mover)
    if enemy_king is None or mover_king is None:
        return []
    if view.attackers(mover, enemy_king) or view.attackers(enemy, mover_king):
        return []
    replies = list(view.legal_moves)
    trapped: list[MotifTag] = []
    for square in sorted(view.piece_map()):
        piece = view.piece_at(square)
        if piece is None or piece.color != enemy or _value(piece.piece_type) < _MINOR:
            continue
        if not exchange.can_be_won(view, square):
            continue
        # The piece's own moves first: fleeing is the usual rescue, and the search stops there.
        ordered = sorted(replies, key=lambda m, sq=square: m.from_square != sq)
        if any(_rescues(view, reply, square, mover) for reply in ordered):
            continue
        hunters = exchange.capturers(view, mover, square)
        trapped.append(_tag("trappedPiece", [square, *hunters]))
    return trapped


def _rescues(
    view: chess.Board, reply: chess.Move, square: chess.Square, mover: chess.Color
) -> bool:
    """Whether the enemy `reply` saves the piece on `square` from being won.

    It does when, after the reply, the mover can no longer take the piece (wherever it went)
    for more than the reply itself took: a piece that flees to a held square, a hunter taken,
    a check to answer first, or a capture worth at least what is lost -- the old rule that a
    trapped piece taking something as valuable on its way out is not trapped, and a queen that
    wins a rook elsewhere has not lost her knight for nothing. A queen taking a rook on a1
    that the bishop takes back has still lost herself.
    """
    nxt = view.copy(stack=False)
    nxt.push(reply)
    where = reply.to_square if reply.from_square == square else square
    taken = chess.PAWN if view.is_en_passant(reply) else view.piece_type_at(reply.to_square)
    gained = _value(taken)
    if reply.promotion is not None:
        gained += _value(reply.promotion) - _value(chess.PAWN)
    return exchange.best_capture(nxt, mover, where) <= gained


def _attacking_f2f7(
    before: chess.Board,
    after: chess.Board,
    move: chess.Move,
    mover: chess.Color,
    *,
    won_in_line: frozenset[chess.Square] = frozenset(),
) -> MotifTag | None:
    """The move takes aim at the beginner's weak pawn: f7 as white, f2 as black.

    The pawn itself, not whatever stands on the square: a rook checking a king that has
    walked to f7 in the endgame is a check, not this. And the aim must be real: the pawn can
    be won (`exchange.best_capture` > 0) or the engine's line takes it (`won_in_line`). A
    bishop eyeing f7 in a book Italian, with the king and queen holding it, only looks.
    """
    target = chess.F7 if mover == chess.WHITE else chess.F2
    piece = after.piece_at(target)
    if piece is None or piece.color == mover or piece.piece_type != chess.PAWN:
        return None
    if target not in after.attacks(move.to_square):
        return None
    if target in before.attacks(move.from_square):
        return None
    if exchange.best_capture(after, mover, target) <= 0 and target not in won_in_line:
        return None
    return _tag("attackingF2F7", [move.to_square, target], move)


def _exposed_king(view: chess.Board, mover: chess.Color) -> MotifTag | None:
    """The enemy king has lost its pawn shelter and the mover has force bearing on it.

    Shelter is counted one pawn per file over the three files around the king, one or two
    ranks in front. At most one such pawn, at least two of the mover's pieces actually
    attacking the king's ring (one of them a queen or rook), and at least two ring squares
    under fire. Counting pieces on the ring rather than pieces on the board is what keeps an
    opening king that just pushed a pawn from looking besieged. The theme ranks low, so a
    sharper motif in the same position wins out. Without a queen of the mover's there is no
    attack to speak of: a king walking up the board in a rook and knight ending is active,
    not exposed.
    """
    enemy = not mover
    king = view.king(enemy)
    if king is None or not view.pieces(chess.QUEEN, mover):
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


_RAY_NAMES = ("pin", "absolutePin", "skewer")


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
    """The move vacates a square with tempo and another piece soon uses that square with force.

    Tempo means the move is a capture or a check, which is what separates a clearance from a
    piece that merely happened to be in the way. The square is cleared for another piece: a
    queen that grabs a pawn and comes back to where it stood cleared nothing. Soon means one
    of the mover's next two moves, and the use must itself be a capture, a check or mate, or
    set up what the line goes on to win or mate with (Bxe5 Bxe5 Qd5, the queen's fork that
    takes e5 a move later) -- a rook strolling down a file the knight left long ago is not
    what the knight's move was for.
    """
    mover = boards[0].turn
    if len(played) < 3:
        return None
    first = played[0]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    vacated = first.from_square
    cleared_piece = first.to_square
    for i in range(2, min(len(played), 5), 2):
        later = played[i]
        if later.from_square == cleared_piece:
            cleared_piece = later.to_square  # the piece that cleared the square, moving on
            continue
        uses = later.to_square == vacated or bool(
            chess.BB_SQUARES[vacated] & chess.between(later.from_square, later.to_square)
        )
        if not uses:
            continue
        forcing = boards[i].is_capture(later) or boards[i + 1].is_check()
        if forcing or _mates(boards, mover) or _wins_material(boards, played, mover):
            return _tag("clearance", [vacated, later.from_square, later.to_square], first)
    return None


def _capturing_defender(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """The move captures a defender, and a later move of the mover's wins what it defended.

    The captured piece must have been a legal defender of the later square at the start
    (`exchange.defenders`), and the piece there must be one the mover could not win before
    (`exchange.best_capture` is 0) but wins at that point (`exchange.exchange` > 0), with the
    line ahead of where it started once that capture is counted: removing a knight that
    guarded f6 when the g-pawn guards it too wins nothing, and giving a rook for the knight
    that guarded a pawn loses more than the pawn. A move onto a square the defender guarded
    counts without material only when it is a check and the line ends in mate.
    """
    first = played[0]
    if not boards[0].is_capture(first):
        return None
    enemy = not mover
    victim_square = _ep_victim(first) if boards[0].is_en_passant(first) else first.to_square
    start = _balance(boards[0], mover)
    mates = _mates(boards, mover)
    for i in range(2, len(played), 2):
        later = played[i]
        square = later.to_square
        if victim_square not in exchange.defenders(boards[0], enemy, square):
            continue
        position = boards[i]
        if mates and boards[i + 1].is_check():
            return _tag("capturingDefender", [victim_square, square], first)
        if not position.is_capture(later):
            continue
        if exchange.best_capture(boards[0], mover, square) > 0:
            continue  # it could be won before: the defender was not what held it
        gain = exchange.exchange(position, later)
        if gain <= 0 or _balance(position, mover) + gain <= start:
            continue
        return _tag("capturingDefender", [victim_square, square], first)
    return None


def _deflection(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """A forcing move pulls the only defender of a square away, and the mover wins there.

    Forcing means the move is a capture or a check, and the reply is what it forces: the
    recapture on the move's square, or an answer to the check. A queen that walks away of
    her own accord was not deflected. "Only defender" is checked on the position after the
    move, by legal moves (`exchange.defenders`). And the follow-up must win: a capture that
    nets material (`exchange.exchange` > 0) with the line ahead of where it started, or
    mate. A rook check on the square the queen left, answered by a king step, wins nothing.
    """
    if len(played) < 3:
        return None
    first, reply, follow = played[0], played[1], played[2]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    recapture = boards[1].is_capture(reply) and reply.to_square == first.to_square
    if not recapture and not boards[1].is_check():
        return None
    defender = reply.from_square
    guards = exchange.defenders(boards[1], not mover, follow.to_square)
    if len(guards) != 1 or defender not in guards:
        return None
    if follow.to_square not in boards[1].attacks(defender):
        return None
    if follow.to_square in boards[2].attacks(reply.to_square):
        return None
    if not boards[3].is_checkmate() and not _mates(boards, mover):
        gain = exchange.exchange(boards[2], follow)
        if gain <= 0 or _balance(boards[2], mover) + gain <= _balance(boards[0], mover):
            return None
    return _tag("deflection", [defender, reply.to_square, follow.to_square], first)


def _attraction(
    boards: list[chess.Board], played: list[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """A forcing move drags an enemy piece onto a square where the next move hits it.

    The enemy is dragged by taking the piece the move put there, so the reply must capture on
    the move's square; the piece pulled is the king, a queen or a rook, as Lichess has it (a
    knight taking back a knight is a trade, not a lure); and the mover's follow-up must land
    on that square or attack it while capturing or checking. Then the lure has to pay: the
    line ends in mate, or wins material -- by taking the lured queen or rook on its square
    within the mover's next two moves, or, for a king, anywhere in the line. Material is
    counted from before the first move, so the piece given as bait is paid back first.
    """
    if len(played) < 3:
        return None
    first, reply, follow = played[0], played[1], played[2]
    if not boards[0].is_capture(first) and not boards[1].is_check():
        return None
    if not boards[1].is_capture(reply) or reply.to_square != first.to_square:
        return None
    landed = reply.to_square
    attracted = boards[2].piece_at(landed)
    if attracted is None or attracted.color == mover:
        return None
    if attracted.piece_type not in (chess.KING, chess.QUEEN, chess.ROOK):
        return None
    if not boards[2].is_capture(follow) and not boards[2].gives_check(follow):
        return None
    hits = follow.to_square == landed or landed in boards[3].attacks(follow.to_square)
    if not hits:
        return None
    if attracted.piece_type == chess.KING:
        pays = _wins_material(boards, played, mover)
    else:
        pays = _wins_material(boards, played, mover, square=landed, upto=4)
    if not pays and not _mates(boards, mover):
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


def _slider_reaches(
    piece_type: chess.PieceType | None, frm: chess.Square, to: chess.Square
) -> bool:
    """Whether a slider of this type on `frm` moves along the straight line to `to`."""
    same_rank = chess.square_rank(frm) == chess.square_rank(to)
    same_file = chess.square_file(frm) == chess.square_file(to)
    on_diag = abs(chess.square_file(frm) - chess.square_file(to)) == abs(
        chess.square_rank(frm) - chess.square_rank(to)
    )
    if piece_type == chess.ROOK:
        return same_rank or same_file
    if piece_type == chess.BISHOP:
        return on_diag
    if piece_type == chess.QUEEN:
        return same_rank or same_file or on_diag
    return False


def _xray_to(
    board: chess.Board, mover: chess.Color, target: chess.Square
) -> tuple[chess.Square, chess.Square] | None:
    """A mover slider bearing on `target` through exactly one enemy piece: (slider, blocker)."""
    for slider in _sliders(board, mover):
        if not _slider_reaches(board.piece_type_at(slider), slider, target):
            continue
        between = chess.SquareSet(chess.between(slider, target))
        blockers = [sq for sq in between if board.piece_at(sq) is not None]
        if len(blockers) != 1:
            continue
        blocker = blockers[0]
        if board.piece_at(blocker).color != mover:  # type: ignore[union-attr]
            return slider, blocker
    return None


def _xray(
    boards: list[chess.Board], played: Sequence[chess.Move], mover: chess.Color
) -> MotifTag | None:
    """Lichess's x-ray: a piece attacks a square through an enemy piece.

    Read over the line, the way the sacrifice and clearance motifs are: a square the mover
    captures or mates on was, at some earlier point in the line, borne on by a mover slider
    through exactly one enemy piece, and by the time the square is taken the slider is still
    there with its ray to the square open, and either makes the capture itself or stands
    behind the piece that does, guarding it. The ray opening is what turns a latent x-ray
    into a real one, so a slider that merely looks through a piece that never moves is not
    called one -- nor a rook whose own knight took the blocker's place on the file.
    """
    for k in range(0, len(played), 2):  # the mover's own moves
        move = played[k]
        board_k = boards[k]
        after_k = boards[k + 1]
        if not (board_k.is_capture(move) or after_k.is_checkmate()):
            continue
        target = move.to_square
        for j in range(0, k + 1, 2):  # a mover position at or before this move
            found = _xray_to(boards[j], mover, target)
            if found is None:
                continue
            slider, blocker = found
            if board_k.piece_at(slider) != boards[j].piece_at(slider):
                continue  # the slider has moved on or been taken
            takes = move.from_square == slider
            if not takes and target not in after_k.attacks(slider):
                continue  # the ray is still shut, by the blocker or by a piece of ours
            return _tag("xRayAttack", [target, blocker, slider], move)
    return None


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

    won_in_line = _mover_captures(boards, played, mover)
    tags = _group_a(before, after, move, mover)
    tags.extend(_new_static_tags(before, after, mover))
    for tag in (
        _fork(after, move, mover, won_in_line=won_in_line),
        _discovered_attack(boards, played, mover),
        _xray(boards, played, mover),
        _interference(boards, played, mover),
        _attacking_f2f7(before, after, move, mover, won_in_line=won_in_line),
        _sacrifice(boards, played, mover, mate_in),
        _clearance(boards, played),
        _capturing_defender(boards, played, mover),
        _deflection(boards, played, mover),
        _attraction(boards, played, mover),
    ):
        if tag is not None:
            tags.append(tag)
    tags.extend(_mate_tags(after, boards[-1], mover, mate_in))

    if after.is_checkmate():
        tags.extend(mates.mating_pattern_tags(after, move))
    elif boards[-1].is_checkmate() and boards[-1].turn != mover:
        tags.extend(mates.mating_pattern_tags(boards[-1], played[-1]))
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
