"""The plan in the best line: what the player's next few moves are for, read as a whole.

The positional layer reads one move -- "this develops a piece", "this takes the file" -- and a
grandmaster explains a game in plans: "White begins operations on the h-file", "Black wants
to double on the c-file", "the knight is heading for c6". Against the master games' notes
(tools/eval/masters_notes.py, 2026-09-24) the coach named an attack on the king at 12% of
the moves the note did: it read the best move alone.

So the player's first four moves in the engine's best line are read together, on the boards
the line passes through, each piece followed from where it stands now. A plan counts only
from quiet moves -- a capture or a recapture is an exchange, not a plan -- and the best move
must be part of it, so the coach never names a plan the move in hand does not start:

  attack   pieces coming at their king: each move leaves the piece bearing on more of the
           squares around their king than before (a rook lifted on the way counts), two
           moves at least, and more pieces bearing there at the end than now
  storm    two pawn advances at their king, castled on the wing away from yours
  double   two heavy pieces lined up on a file with none of the player's pawns, not so now
  outpost  a knight taken in two moves or more to an outpost it could not reach in one
  passer   the same passed pawn pushed twice
  break    a pawn advance in the centre that meets one of theirs, the best move a piece
           brought to bear on that square first
  wing     two pawn advances on the wing away from both kings

The first found in that order is the plan. An attack or a storm is not read for a side the
engine calls worse (below LEVEL_SCORE), where it would be a hope, not a plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange, positional

PLAN_MOVES = 4  # the player's moves read together
LEVEL_SCORE = 0.45  # an attack or storm is read only for a side at least this well off
_HEAVY = (chess.ROOK, chess.QUEEN)
_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


@dataclass(frozen=True)
class LinePlan:
    """A plan the best line carries out: its kind, the square it is about (their king, the
    outpost, the passed pawn, the break), the file or wing, the player's pieces that carry it
    out as they stand now, and the moves that do, in the order played."""

    kind: str
    square: chess.Square | None = None
    where: str | None = None
    pieces: tuple[chess.Square, ...] = ()
    sans: tuple[str, ...] = ()
    detail: str | None = None  # the doubled file: "open" or "half-open"


@dataclass(frozen=True)
class _Line:
    board: chess.Board
    boards: list[chess.Board]
    line: list[chess.Move]
    ours: list[int]  # indices of the player's moves in `line`
    origin: dict[int, chess.Square]  # the player's move -> where its piece stands now
    us: chess.Color


def _walk(board: chess.Board, pv: Sequence[chess.Move]) -> _Line:
    us = board.turn
    boards = [board.copy(stack=False)]
    line: list[chess.Move] = []
    for move in pv[: 2 * PLAN_MOVES]:
        if move not in boards[-1].legal_moves:
            break
        nxt = boards[-1].copy(stack=False)
        nxt.push(move)
        boards.append(nxt)
        line.append(move)
    ours = list(range(0, len(line), 2))
    # Each of the player's pieces followed from where it stands now.
    at: dict[chess.Square, chess.Square] = {sq: sq for sq in chess.SquareSet(board.occupied_co[us])}
    origin: dict[int, chess.Square] = {}
    for j in ours:
        move = line[j]
        start = at.pop(move.from_square, move.from_square)
        origin[j] = start
        at[move.to_square] = start
        if boards[j].is_castling(move):
            rook_from, rook_to = _castle_rook(boards[j], move)
            at[rook_to] = at.pop(rook_from, rook_from)
        if j + 1 < len(line):
            at.pop(line[j + 1].to_square, None)  # taken
    return _Line(board, boards, line, ours, origin, us)


def _castle_rook(board: chess.Board, move: chess.Move) -> tuple[chess.Square, chess.Square]:
    rank = chess.square_rank(move.from_square)
    if chess.square_file(move.to_square) > chess.square_file(move.from_square):
        return chess.square(7, rank), chess.square(5, rank)
    return chess.square(0, rank), chess.square(3, rank)


def _quiet(b: chess.Board, move: chess.Move) -> bool:
    return not b.is_capture(move) and move.promotion is None


def _zone(king: chess.Square) -> int:
    return chess.BB_KING_ATTACKS[king] | chess.BB_SQUARES[king]


def _bearing(b: chess.Board, us: chess.Color, king: chess.Square) -> int:
    """How many of the player's pieces (not the king) bear on their king's zone, pawns too."""
    zone = _zone(king)
    return sum(
        1
        for sq, piece in b.piece_map().items()
        if piece.color == us and piece.piece_type != chess.KING and b.attacks_mask(sq) & zone
    )


def read(
    board: chess.Board, pv: Sequence[chess.Move], score: float | None = None
) -> LinePlan | None:
    """The plan the player's first moves of `pv` carry out, or None. `score` is the player's
    expected score in the line, when known."""
    w = _walk(board, pv)
    if len(w.ours) < 2:
        return None
    level = score is None or score >= LEVEL_SCORE
    for check in (
        _attack if level else None,
        _storm if level else None,
        _double,
        _outpost,
        _passer,
        _break,
        _wing,
    ):
        if check is None:
            continue
        plan = check(w)
        if plan is not None:
            return plan
    return None


def _sans(w: _Line, js: Sequence[int]) -> tuple[str, ...]:
    return tuple(w.boards[j].san(w.line[j]) for j in js)


def _plan(w: _Line, kind: str, js: Sequence[int], **kw) -> LinePlan:  # noqa: ANN003
    pieces = tuple(dict.fromkeys(w.origin[j] for j in js))
    return LinePlan(kind, pieces=pieces, sans=_sans(w, js), **kw)


def _wing_of(file_: int) -> str | None:
    return "queenside" if file_ <= 2 else "kingside" if file_ >= 5 else None


def _storm_push(w: _Line, j: int) -> bool:
    """A pawn advance on the wing where their king stands, the kings apart (the positional
    layer's storm, kept to the files of that wing)."""
    b, move = w.boards[j], w.line[j]
    theirs = w.board.king(not w.us)
    if theirs is None or b.piece_type_at(move.from_square) != chess.PAWN or not _quiet(b, move):
        return False
    wing = _wing_of(chess.square_file(theirs))
    return (
        _wing_of(chess.square_file(move.to_square)) == wing
        and positional._pawn_storm_side(b, move, w.us) is not None
    )


def _rank(square: chess.Square, us: chess.Color) -> int:
    """The square's rank counted from the player's side, 1 to 8."""
    rank = chess.square_rank(square)
    return rank + 1 if us == chess.WHITE else 8 - rank


def _attack(w: _Line) -> LinePlan | None:
    king = w.board.king(not w.us)
    if king is None:
        return None
    joins: list[int] = []
    lifts: set[int] = set()
    for j in w.ours:
        b, move = w.boards[j], w.line[j]
        if not _quiet(b, move):
            continue
        piece = b.piece_type_at(move.from_square)
        their_king = b.king(not w.us)
        if their_king is None or piece in (None, chess.KING):
            continue
        if piece == chess.PAWN:
            if _storm_push(w, j):
                joins.append(j)
            continue
        zone = _zone(their_king)
        before = chess.popcount(b.attacks_mask(move.from_square) & zone)
        after = chess.popcount(w.boards[j + 1].attacks_mask(move.to_square) & zone)
        if after > before:
            joins.append(j)
        elif piece == chess.ROOK and chess.square_file(move.from_square) == chess.square_file(
            move.to_square
        ):
            # A rook lift: up its file, to swing across to their king later.
            lifts.add(j)
            joins.append(j)
    # A lift counts only when the same rook swings in afterwards; the swing itself always
    # counts (Qd3 Re3 Rh3 was read as "Qd3 and Re3", the Kotlin port, 2026-09-24).
    joins = [
        j
        for j in joins
        if j not in lifts
        or any(k > j and k not in lifts and w.origin[k] == w.origin[j] for k in joins)
    ]
    pieces = {
        w.origin[j] for j in joins if w.boards[j].piece_type_at(w.line[j].from_square) != chess.PAWN
    }
    if 0 not in joins or not pieces or len({w.origin[j] for j in joins}) < 2:
        return None
    end = w.boards[w.ours[-1] + 1]
    end_king = end.king(not w.us)
    if end_king is None or _bearing(end, w.us, end_king) <= _bearing(w.board, w.us, king):
        return None
    return _plan(w, "attack", joins, square=king)


def _storm(w: _Line) -> LinePlan | None:
    king = w.board.king(not w.us)
    if king is None:
        return None
    pushes = [j for j in w.ours if _storm_push(w, j)]
    if len(pushes) < 2 or pushes[0] != 0:
        return None
    wing = "kingside" if chess.square_file(king) >= 4 else "queenside"
    return _plan(w, "storm", pushes, square=king, where=wing)


def _heavy_on(b: chess.Board, us: chess.Color, file_: int) -> list[chess.Square]:
    return [
        sq
        for sq in chess.SquareSet(chess.BB_FILES[file_])
        if (p := b.piece_at(sq)) is not None and p.color == us and p.piece_type in _HEAVY
    ]


def _double(w: _Line) -> LinePlan | None:
    end = w.boards[w.ours[-1] + 1]
    first = w.line[0]
    if not _quiet(w.board, first) or w.board.piece_type_at(first.from_square) not in _HEAVY:
        return None
    file_ = chess.square_file(first.to_square)
    heavy = _heavy_on(end, w.us, file_)
    if len(heavy) < 2 or len(_heavy_on(w.board, w.us, file_)) >= 2:
        return None
    if end.pieces_mask(chess.PAWN, w.us) & chess.BB_FILES[file_]:
        return None
    theirs = end.pieces_mask(chess.PAWN, not w.us) & chess.BB_FILES[file_]
    detail = "half-open" if theirs else "open"
    # Lined up: nothing between the two.
    if chess.SquareSet(chess.between(heavy[0], heavy[1])) & chess.SquareSet(end.occupied):
        return None
    onto = [
        j
        for j in w.ours
        if _quiet(w.boards[j], w.line[j])
        and w.boards[j].piece_type_at(w.line[j].from_square) in _HEAVY
        and chess.square_file(w.line[j].to_square) == file_
        and chess.square_file(w.line[j].from_square) != file_
    ]
    if 0 not in onto:
        return None
    # The pieces that end up doubled, as they stand now.
    now = []
    for sq in heavy:
        landed = [j for j in w.ours if w.line[j].to_square == sq]
        now.append(w.origin[landed[-1]] if landed else sq)
    return LinePlan(
        "double", where="abcdefgh"[file_], pieces=tuple(now), sans=_sans(w, onto), detail=detail
    )


def _outpost(w: _Line) -> LinePlan | None:
    first = w.line[0]
    if w.board.piece_type_at(first.from_square) != chess.KNIGHT or not _quiet(w.board, first):
        return None
    knight = w.origin[0]
    route = [j for j in w.ours if w.origin[j] == knight]
    if len(route) < 2 or not all(_quiet(w.boards[j], w.line[j]) for j in route):
        return None
    for n, j in enumerate(route[1:], 2):
        to = w.line[j].to_square
        if to == knight or chess.BB_KNIGHT_ATTACKS[knight] & chess.BB_SQUARES[to]:
            continue
        if positional._is_outpost(w.boards[j + 1], to, w.us):
            return _plan(w, "outpost", route[:n], square=to)
    return None


def _passer(w: _Line) -> LinePlan | None:
    first = w.line[0]
    if w.board.piece_type_at(first.from_square) != chess.PAWN or not _quiet(w.board, first):
        return None
    if not positional.is_passed(w.board, first.from_square):
        return None
    pawn = w.origin[0]
    pushes = [j for j in w.ours if w.origin[j] == pawn and _quiet(w.boards[j], w.line[j])]
    if len(pushes) < 2:
        return None
    return _plan(w, "passer", pushes, square=pawn)


def _lever(b: chess.Board, move: chess.Move, us: chess.Color) -> bool:
    """A pawn advance that meets one of theirs: it attacks a pawn or a pawn attacks it."""
    after = b.copy(stack=False)
    after.push(move)
    to = move.to_square
    theirs = after.pieces_mask(chess.PAWN, not us)
    return bool(after.attacks_mask(to) & theirs) or bool(chess.BB_PAWN_ATTACKS[us][to] & theirs)


def _break(w: _Line) -> LinePlan | None:
    first = w.line[0]
    if w.board.piece_type_at(first.from_square) in (None, chess.PAWN, chess.KING):
        return None
    if not _quiet(w.board, first):
        return None
    for j in w.ours[1:]:
        b, move = w.boards[j], w.line[j]
        if b.piece_type_at(move.from_square) != chess.PAWN or not _quiet(b, move):
            continue
        if not 2 <= chess.square_file(move.to_square) <= 5:
            continue
        if _rank(move.to_square, w.us) not in (4, 5) or not _lever(b, move, w.us):
            continue
        # The best move prepared it: its piece bears on the square the pawn goes to.
        if w.boards[1].attacks_mask(first.to_square) & chess.BB_SQUARES[move.to_square]:
            return _plan(w, "break", [0, j], square=move.to_square)
        return None
    return None


def _wing(w: _Line) -> LinePlan | None:
    theirs, ours = w.board.king(not w.us), w.board.king(w.us)
    if theirs is None or ours is None:
        return None
    king_side = chess.square_file(theirs) >= 4
    if (chess.square_file(ours) >= 4) != king_side:
        return None  # kings on opposite wings: advancing is a storm, not expansion
    if 3 <= chess.square_file(theirs) <= 4:
        return None
    far = range(0, 3) if king_side else range(5, 8)
    pushes = [
        j
        for j in w.ours
        if w.boards[j].piece_type_at(w.line[j].from_square) == chess.PAWN
        and _quiet(w.boards[j], w.line[j])
        and chess.square_file(w.line[j].to_square) in far
        and _rank(w.line[j].to_square, w.us) >= 4
    ]
    if len(pushes) < 2 or pushes[0] != 0:
        return None
    return _plan(w, "wing", pushes, where="queenside" if king_side else "kingside")


# The words, per plan: its name (square-free, for the idea rung), why it works, the question
# that sets the player looking, and the focus rung's nudge.
PLAN_NAME = {
    "attack": "bringing more pieces against their king",
    "storm": "a pawn storm against their king",
    "double": "doubling your heavy pieces on {detail} file",
    "outpost": "rerouting a knight to an outpost",
    "passer": "pushing your passed pawn",
    "break": "preparing a pawn break in the centre",
    "wing": "gaining space with your pawns on the {where}",
}
PLAN_REASON = {
    "attack": "an attack breaks through when more of your pieces bear on their king than they "
    "can bring to defend it",
    "storm": "with your own king off that wing, your pawns can roll at theirs without "
    "exposing your own, and they pry open lines for your pieces",
    "double": "two heavy pieces on the same file hit twice as hard and break into their position",
    "outpost": "a knight on a square no pawn of theirs can ever attack stays there for good",
    "passer": "a passed pawn ties their pieces down, and every step makes it stronger",
    "break": "with your pieces trained on it, the pawn break opens the centre on your terms",
    "wing": "on the wing away from the kings you can take space without weakening yours",
}
PLAN_QUESTION = {
    "attack": "Which of your pieces is not yet helping against their king, and how does it "
    "get there?",
    "storm": "Which pawn starts the storm, and which file does it open?",
    "double": "Which file has none of your pawns on it, and how do two of your heavy pieces "
    "get onto it?",
    "outpost": "Which square could no pawn of theirs ever drive a knight from, and how does "
    "one of yours get there?",
    "passer": "What stops your passed pawn, and how do you get it moving?",
    "break": "Which pawn move would open the centre, and which piece should support it first?",
    "wing": "Where can your pawns take space without weakening your king?",
}
PLAN_NUDGE = {
    "attack": "Which piece joins the attack first?",
    "storm": "Which pawn goes first?",
    "double": "Which heavy piece goes to the file first?",
    "outpost": "Which first step starts the knight on its way?",
    "passer": "Which move pushes it, or clears its way?",
    "break": "Which piece supports the break first?",
    "wing": "Which pawn goes first?",
}


def name(plan: LinePlan) -> str:
    detail = "a half-open" if plan.detail == "half-open" else "an open"
    return PLAN_NAME[plan.kind].format(where=plan.where or "", detail=detail)


def mechanism(plan: LinePlan) -> str:
    return f"{name(plan)}: {PLAN_REASON[plan.kind]}"


def target(plan: LinePlan) -> str | None:
    """The focus rung's target: their king, the outpost, the file, the passed pawn."""
    sq = chess.square_name(plan.square) if plan.square is not None else None
    if plan.kind in ("attack", "storm") and sq:
        return f"their king on {sq}"
    if plan.kind == "outpost" and sq:
        return f"the {sq} square, an outpost for a knight"
    if plan.kind == "double" and plan.where:
        return f"the {plan.detail or 'open'} {plan.where}-file"
    if plan.kind == "passer" and sq:
        return f"your passed pawn on {sq}"
    if plan.kind == "break" and sq:
        return f"the {sq} square, where the break comes"
    if plan.kind == "wing" and plan.where:
        return f"your pawns on the {plan.where}"
    return None


def _listed(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


def pieces(plan: LinePlan, board: chess.Board) -> str | None:
    """The player's pieces that carry it out, where they stand now: "your rook on a1 and
    your queen on d1"."""
    named = [
        f"your {_NAMES[p.piece_type]} on {chess.square_name(sq)}"
        for sq in plan.pieces
        if (p := board.piece_at(sq)) is not None and p.color == board.turn
    ][:3]
    return _listed(named) if named else None


def why(plan: LinePlan, board: chess.Board) -> str:
    """The plan played out, for the move rung: the moves that carry it out and what for."""
    moves = _listed(list(plan.sans))
    one = len(plan.sans) == 1

    def verb(plural: str, singular: str) -> str:
        return singular if one else plural

    kinds = [
        _NAMES[p.piece_type]
        for sq in plan.pieces
        if (p := board.piece_at(sq)) is not None and p.color == board.turn
    ]
    # "your knights" for two knights, "your queen and rook" for one of each.
    unique = list(dict.fromkeys(kinds))
    what = (
        f"{unique[0]}s"
        if len(unique) == 1 and len(kinds) > 1
        else _listed(unique)
        if unique
        else "pieces"
    )
    sq = chess.square_name(plan.square) if plan.square is not None else ""
    if plan.kind == "attack":
        return f"{moves} {verb('bring', 'brings')} your {what} against their king"
    if plan.kind == "storm":
        return f"{moves} {verb('roll', 'rolls')} your pawns at their king on the {plan.where}"
    if plan.kind == "double":
        return (
            f"{moves} {verb('double', 'doubles')} your {what} on the {plan.detail or 'open'} "
            f"{plan.where}-file"
        )
    if plan.kind == "outpost":
        return f"{moves} take your knight to {sq}, an outpost no pawn of theirs can attack"
    if plan.kind == "passer":
        return f"{moves} push your passed pawn"
    if plan.kind == "break":
        first, last = plan.sans[0], plan.sans[-1]
        return f"{first} brings a piece to bear on {sq}, then {last} breaks open the centre"
    return f"{moves} gain space on the {plan.where}"


# The threat the best move makes. Where the line wins nothing -- they answer it -- the move
# can still be a threat they must answer: "threatening f5", "hitting the loose knight",
# "now Black threatens ...Bb3" (the master games' notes). The coach said "a quiet move is
# wanted here" at 18% of their annotated moves; a threat is what many of those moves are.


@dataclass(frozen=True)
class MoveThreat:
    """What the best move threatens, were they to pass: mate (`square` their king), winning
    what stands on `square` (`gain` pawns), or a trade of the moved piece for the like piece on
    `square`, which the line then makes; `follow` is the move that would do it. `why` says what
    a trade is for: "ahead" in material, "queens" off when the player is worse, else "plain"."""

    kind: str  # "mate", "win" or "trade"
    square: chess.Square
    follow: chess.Move
    gain: int = 0
    why: str = "plain"


def threat(
    board: chess.Board,
    move: chess.Move,
    pv: Sequence[chess.Move] = (),
    score: float | None = None,
) -> MoveThreat | None:
    """The threat `move` makes that was not there before it: a mate in one, or the most a
    capture would net on a square that could not be won before, or a trade it offers that the
    line `pv` (starting with `move`) takes up. None for a check (that is forcing, not a
    threat), a capture (its point is the exchange) or a move that leaves its piece to be won
    (an offer, not a threat)."""
    us = board.turn
    if board.is_capture(move) or move.promotion is not None:
        return None
    after = board.copy(stack=False)
    after.push(move)
    if after.is_check() or after.is_game_over():
        return None
    if exchange.can_be_won(after, move.to_square):
        return None
    passed = after.copy(stack=False)
    passed.push(chess.Move.null())
    king = after.king(not us)
    for follow in passed.legal_moves:
        passed.push(follow)
        mate = passed.is_checkmate()
        passed.pop()
        if mate and king is not None:
            return MoveThreat("mate", king, follow)
    mover = after.piece_at(move.to_square)
    if mover is None:
        return None
    to_bb = chess.BB_SQUARES[move.to_square]
    view = exchange._as_mover(after, us)
    best: tuple[int, chess.Square] | None = None
    trade: chess.Square | None = None
    for sq in chess.SquareSet(after.occupied_co[not us]):
        piece = after.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        # Face to face with a like piece: they take first, or trade, and "if they let it be"
        # is no threat at all (Bf6 against a bishop on h4, the master games, 2026-09-24).
        # Two pawns face to face are a pawn break, which the positional layer names.
        facing = bool(after.attacks_mask(sq) & to_bb) and bool(
            after.attacks_mask(move.to_square) & chess.BB_SQUARES[sq]
        )
        like = exchange.VALUE[piece.piece_type] == exchange.VALUE[mover.piece_type]
        if facing and like and mover.piece_type != chess.PAWN:
            if trade is None:
                trade = sq
            continue
        gain = exchange.best_capture(after, us, sq)
        if gain <= 0 or gain <= exchange.best_capture(board, us, sq):
            continue
        if best is None or gain > best[0]:
            best = (gain, sq)
    if best is not None:
        gain, sq = best
        follow = max(exchange.captures_on(after, us, sq), key=lambda m: exchange.exchange(view, m))
        return MoveThreat("win", sq, follow, gain)
    if trade is None:
        return None
    # A trade only when the line makes it, on either square, within the next two moves.
    squares = {move.to_square, trade}
    line = list(pv[1:3]) if pv and pv[0] == move else []
    if not any(m.to_square in squares for m in line):
        return None
    follow = next(
        (m for m in exchange.captures_on(after, us, trade)), chess.Move(move.to_square, trade)
    )
    material = sum(
        exchange.VALUE[p.piece_type] * (1 if p.color == us else -1)
        for p in board.piece_map().values()
        if p.piece_type != chess.KING
    )
    if material >= 2:
        why = "ahead"
    elif mover.piece_type == chess.QUEEN and score is not None and score < LEVEL_SCORE:
        why = "queens"
    else:
        why = "plain"
    return MoveThreat("trade", trade, follow, 0, why)


THREAT_GOAL = {
    "mate": "you can threaten mate",
    "win": "you can take the initiative",
    "trade": "you can offer a trade",
}
# The same, when the engine calls the game won: said first, as the plan goal says it.
THREAT_GOAL_WON = {
    "mate": "you are winning, and you can threaten mate",
    "win": "you are winning: a threat keeps them on the back foot",
    "trade": "you are winning, and a trade brings the win closer",
}
THREAT_NAME = {
    "mate": "a threat of mate",
    "win": "a threat",
    "trade": "a trade",
}
THREAT_REASON = {
    "mate": "they must spend their move stopping it, and have no time for their own plans",
    "win": "a move that attacks something they must then defend keeps you a step ahead: "
    "they answer you instead of playing their own moves",
}
TRADE_REASON = {
    "ahead": "when you are ahead, every trade brings the win closer",
    "queens": "with the queens off, their attack loses its sting",
    "plain": "every trade changes the fight, and here it is the best way on",
}
THREAT_QUESTION = {
    "mate": "Which move would leave you threatening mate next move?",
    "win": "Which of their pieces could you attack with something worth less, or where it "
    "is short of defenders?",
    "trade": "Which of their pieces does the most for them, and which of yours could trade "
    "itself off for it?",
}
THREAT_NUDGE = {
    "mate": "Which move sets up the mate?",
    "win": "Which move attacks it?",
    "trade": "Which move puts your piece face to face with it?",
}


def threat_mechanism(t: MoveThreat) -> str:
    reason = TRADE_REASON[t.why] if t.kind == "trade" else THREAT_REASON[t.kind]
    return f"{THREAT_NAME[t.kind]}: {reason}"


def threat_target(t: MoveThreat, board: chess.Board) -> str | None:
    """The focus rung's target: their king, or what the threat goes after, as it stands."""
    piece = board.piece_at(t.square)
    if t.kind == "mate":
        return f"their king on {chess.square_name(t.square)}"
    if piece is None or piece.color == board.turn:
        return None
    return f"their {_NAMES[piece.piece_type]} on {chess.square_name(t.square)}"


def threat_why(t: MoveThreat, board: chess.Board, move: chess.Move) -> str:
    """The move and its threat, for the move rung: "Nc4 goes after their pawn on e5: if they
    let it be, Nxe5 wins it"."""
    san = board.san(move)
    after = board.copy(stack=False)
    after.push(move)
    after.push(chess.Move.null())
    follow = after.san(t.follow)
    if t.kind == "mate":
        return f"{san} threatens {follow.rstrip('#')} mate"
    piece = board.piece_at(t.square)
    if piece is None:
        return f"{san} threatens {follow}"
    what = f"their {_NAMES[piece.piece_type]} on {chess.square_name(t.square)}"
    if t.kind == "trade":
        mover = board.piece_at(move.from_square)
        mine = _NAMES[mover.piece_type] if mover else "piece"
        return f"{san} puts your {mine} face to face with {what}, to trade it off"
    won = "wins it" if t.gain >= exchange.VALUE[piece.piece_type] else "wins material"
    return f"{san} goes after {what}: if they let it be, {follow} {won}"
