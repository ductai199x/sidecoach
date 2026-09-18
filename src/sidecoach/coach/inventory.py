"""What stands on the board right now, in the words a coach uses: which pieces hang,
which are attacked or pinned, the checks and captures on offer, the batteries and the
forks in place. Computed on the board, worded from the player's side, so the model reads
the tactical state instead of working it out square by square -- the one thing a language
model does worst with a chess position.

Everything here is a static fact of the position: no search, no judgement of whether a
capture is good. The engine's lines say what to do; this says what is there.
"""

from __future__ import annotations

import chess

from sidecoach.coach.facts import PIECE_NAMES

_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)
LIMIT = 8  # entries per list, most valuable first


def name_of(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    word = PIECE_NAMES[piece.piece_type] if piece else "piece"
    return f"{word} on {chess.square_name(square)}"


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def pin_target(
    board: chess.Board, square: chess.Square
) -> tuple[chess.Square, chess.Square] | None:
    """(pinner, shielded) when the piece on `square` is pinned: an enemy slider attacks it
    along a line and the next piece past it is a friendly piece worth more (the king most)."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return None
    for attacker in board.attackers(not piece.color, square):
        slider = board.piece_at(attacker)
        if slider is None or slider.piece_type not in _SLIDERS:
            continue
        df = _sign(chess.square_file(square) - chess.square_file(attacker))
        dr = _sign(chess.square_rank(square) - chess.square_rank(attacker))
        file_, rank_ = chess.square_file(square) + df, chess.square_rank(square) + dr
        while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
            behind = chess.square(file_, rank_)
            occupant = board.piece_at(behind)
            if occupant is not None:
                if (
                    occupant.color == piece.color
                    and _VALUE[occupant.piece_type] > _VALUE[piece.piece_type]
                ):
                    return attacker, behind
                break
            file_, rank_ = file_ + df, rank_ + dr
    return None


def skewer_target(
    board: chess.Board, square: chess.Square
) -> tuple[chess.Square, chess.Square] | None:
    """(front, back) when the slider on `square` skewers: it attacks an enemy piece and,
    past it on the same line, a less valuable enemy piece."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type not in _SLIDERS:
        return None
    for front in board.attacks(square):
        target = board.piece_at(front)
        if target is None or target.color == piece.color:
            continue
        df = _sign(chess.square_file(front) - chess.square_file(square))
        dr = _sign(chess.square_rank(front) - chess.square_rank(square))
        file_, rank_ = chess.square_file(front) + df, chess.square_rank(front) + dr
        while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
            behind = chess.square(file_, rank_)
            occupant = board.piece_at(behind)
            if occupant is not None:
                if (
                    occupant.color != piece.color
                    and _VALUE[occupant.piece_type] < _VALUE[target.piece_type]
                ):
                    return front, behind
                break
            file_, rank_ = file_ + df, rank_ + dr
    return None


def batteries(board: chess.Board, colour: chess.Color) -> list[list[chess.Square]]:
    """Aligned sliders of one colour with nothing between: a rook behind a rook on a file,
    a queen behind a bishop on a diagonal. Each battery once, squares in board order."""
    found: list[list[chess.Square]] = []
    seen: set[frozenset[chess.Square]] = set()
    sliders = [
        s
        for s in chess.SQUARES
        if (p := board.piece_at(s)) and p.color == colour and p.piece_type in _SLIDERS
    ]
    for a in sliders:
        pa = board.piece_at(a)
        assert pa is not None
        for b in sliders:
            if b <= a:
                continue
            pb = board.piece_at(b)
            assert pb is not None
            df = _sign(chess.square_file(b) - chess.square_file(a))
            dr = _sign(chess.square_rank(b) - chess.square_rank(a))
            straight = df == 0 or dr == 0
            diagonal = abs(chess.square_file(b) - chess.square_file(a)) == abs(
                chess.square_rank(b) - chess.square_rank(a)
            )
            if not (straight or diagonal):
                continue
            kinds_ok = all(
                (p.piece_type == chess.QUEEN)
                or (straight and p.piece_type == chess.ROOK)
                or (diagonal and p.piece_type == chess.BISHOP)
                for p in (pa, pb)
            )
            if not kinds_ok:
                continue
            clear = True
            file_, rank_ = chess.square_file(a) + df, chess.square_rank(a) + dr
            while chess.square(file_, rank_) != b:
                if board.piece_at(chess.square(file_, rank_)) is not None:
                    clear = False
                    break
                file_, rank_ = file_ + df, rank_ + dr
            if clear and frozenset((a, b)) not in seen:
                seen.add(frozenset((a, b)))
                found.append([a, b])
    return found


def forks_in_place(
    board: chess.Board, colour: chess.Color
) -> list[tuple[chess.Square, list[chess.Square]]]:
    """Each piece of `colour` that attacks two or more enemy units right now, the most
    valuable targets first; the double attacks worth most first."""
    out = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != colour:
            continue
        hits = [t for t in board.attacks(square) if (o := board.piece_at(t)) and o.color != colour]
        if len(hits) >= 2:
            out.append(
                (square, sorted(hits, key=lambda t: -_VALUE[board.piece_type_at(t) or chess.PAWN]))
            )
    out.sort(key=lambda x: -sum(_VALUE[board.piece_type_at(t) or chess.PAWN] for t in x[1]))
    return out


def _pieces(board: chess.Board, colour: chess.Color) -> list[chess.Square]:
    return sorted(
        (
            s
            for s in chess.SQUARES
            if (p := board.piece_at(s)) and p.color == colour and p.piece_type != chess.KING
        ),
        key=lambda s: -_VALUE[board.piece_type_at(s) or chess.PAWN],
    )


def _hits_by(board: chess.Board, square: chess.Square, colour: chess.Color) -> list[chess.Square]:
    return sorted(
        board.attackers(colour, square), key=lambda s: _VALUE[board.piece_type_at(s) or chess.PAWN]
    )


def side_lines(board: chess.Board, colour: chess.Color, *, own: str, their: str) -> list[str]:
    """The state of `colour`'s pieces: hanging (attacked, undefended), attacked by a lesser
    piece, pinned; and `colour`'s batteries and forks in place. `own`/`their` are the
    possessives from the reader's side ("your"/"their")."""
    out: list[str] = []
    hanging: list[str] = []
    attacked: list[str] = []
    pinned: list[str] = []
    for s in _pieces(board, colour):
        piece = board.piece_at(s)
        assert piece is not None
        attackers = _hits_by(board, s, not colour)
        defenders = board.attackers(colour, s)
        if attackers and not defenders:
            hanging.append(f"{name_of(board, s)} (attacked by the {name_of(board, attackers[0])})")
        elif (
            attackers
            and _VALUE[board.piece_type_at(attackers[0]) or chess.PAWN] < _VALUE[piece.piece_type]
        ):
            attacked.append(f"{name_of(board, s)} by the {name_of(board, attackers[0])}")
        pin = pin_target(board, s)
        if pin is not None:
            pinned.append(
                f"{name_of(board, s)} to the {name_of(board, pin[1])} "
                f"by the {name_of(board, pin[0])}"
            )
    if hanging:
        out.append(f"{own} hanging: " + "; ".join(hanging[:LIMIT]))
    if attacked:
        out.append(f"{own} attacked by a lesser piece: " + "; ".join(attacked[:LIMIT]))
    if pinned:
        out.append(f"{own} pinned: " + "; ".join(pinned[:LIMIT]))
    forks = forks_in_place(board, colour)
    if forks:
        out.append(
            f"{own} double attacks: "
            + "; ".join(
                f"{name_of(board, s)} hits the "
                + " and the ".join(name_of(board, t) for t in hits[:3])
                for s, hits in forks[:4]
            )
        )
    skewers = [
        (s, sk) for s in _pieces(board, colour) if (sk := skewer_target(board, s)) is not None
    ]
    if skewers:
        out.append(
            f"{own} skewers: "
            + "; ".join(
                f"{name_of(board, s)} through the {name_of(board, f)} to the {name_of(board, b)}"
                for s, (f, b) in skewers[:LIMIT]
            )
        )
    bats = batteries(board, colour)
    if bats:
        out.append(
            f"{own} batteries: "
            + "; ".join(" and ".join(name_of(board, s) for s in b) for b in bats[:LIMIT])
        )
    return out


def to_move_lines(board: chess.Board, *, own: str) -> list[str]:
    """What the side to move can do at once: checks and captures, in SAN, with what is taken."""
    out: list[str] = []
    checks: list[str] = []
    for m in board.legal_moves:
        if not board.gives_check(m):
            continue
        after = board.copy(stack=False)
        after.push(m)
        checkers = after.checkers()
        kind = ""
        if len(checkers) > 1:
            kind = " (double check)"
        elif m.to_square not in checkers:
            kind = " (discovered check)"
        checks.append(board.san(m) + kind)
    if checks:
        out.append(f"{own} checks: " + ", ".join(checks[:10]))
    caps = []
    for m in board.legal_moves:
        if board.is_capture(m):
            victim = board.piece_at(m.to_square)
            if victim is None and board.is_en_passant(m):
                caps.append((1, f"{board.san(m)} (a pawn, en passant)"))
                continue
            assert victim is not None
            defended = board.attackers(not board.turn, m.to_square)
            caps.append(
                (
                    _VALUE[victim.piece_type],
                    f"{board.san(m)} (the {PIECE_NAMES[victim.piece_type]}, "
                    f"{'defended' if defended else 'undefended'})",
                )
            )
    if caps:
        caps.sort(key=lambda x: -x[0])
        out.append(f"{own} captures: " + ", ".join(c for _, c in caps[:10]))
    return out


def inventory_lines(board: chess.Board, user_colour: chess.Color) -> list[str]:
    """The player's pieces, the opponent's, and what the side to move can do, one list per
    line; one line saying so when nothing hangs, is pinned or is on offer."""
    mine = side_lines(board, user_colour, own="The player's", their="the opponent's")
    theirs = side_lines(board, not user_colour, own="The opponent's", their="the player's")
    mover = "The player's" if board.turn == user_colour else "The opponent's"
    lines = [*mine, *theirs, *to_move_lines(board, own=mover)]
    return lines or ["Nothing hangs, nothing is pinned, no check or capture is on."]


def inventory_text(board: chess.Board, user_colour: chess.Color) -> str:
    """The inventory as one block, for a message that is not sectioned."""
    lines = inventory_lines(board, user_colour)
    if len(lines) == 1 and lines[0].startswith("Nothing"):
        return "On the board now: nothing hangs, nothing is pinned, no check or capture is on."
    return "On the board now (complete lists): " + " | ".join(lines) + "."
