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

from sidecoach.analysis import exchange, motifs
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
    along a line and the next piece past it is a friendly piece worth more -- the king, or a
    piece the slider could win once the front one moves: worth more than the slider, or
    undefended. A knight in front of a defended rook, with a queen behind it, pins nothing
    (Tai, 2026-09-23: a pin is tactical, "if that piece is defended then there is no winning
    that piece hence it's not a valid pin")."""
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
                # A pin to the king holds whatever the pinner: a pinned piece still gives
                # check. A pin to anything else needs a pinner that can legally take.
                to_king = occupant.piece_type == chess.KING
                takes = to_king or attacker in exchange.capturers(board, not piece.color, square)
                if (
                    occupant.color == piece.color
                    and _VALUE[occupant.piece_type] > _VALUE[piece.piece_type]
                    and takes
                    and motifs.winnable(board, behind, _VALUE[slider.piece_type], ignore=square)
                ):
                    return attacker, behind
                break
            file_, rank_ = file_ + df, rank_ + dr
    return None


def skewer_target(
    board: chess.Board, square: chess.Square
) -> tuple[chess.Square, chess.Square] | None:
    """(front, back) when the slider on `square` skewers: it attacks an enemy piece and,
    past it on the same line, a less valuable enemy piece the slider could win once the
    front one steps aside -- worth more than the slider, or undefended (the same rule as a
    pin: a defended piece behind is not won)."""
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
                # The front piece has to move: the king, something worth more than the
                # slider, or something it can win; and the slider must be able to take.
                must_move = (
                    target.piece_type == chess.KING
                    or _VALUE[target.piece_type] > _VALUE[piece.piece_type]
                    or not exchange.defenders(board, target.color, front)
                )
                if (
                    occupant.color != piece.color
                    and _VALUE[occupant.piece_type] < _VALUE[target.piece_type]
                    and must_move
                    and square in exchange.capturers(board, piece.color, front)
                    and motifs.winnable(board, behind, _VALUE[piece.piece_type], ignore=front)
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
    """Each piece of `colour` that forks right now: it attacks two or more enemy units it
    could win -- the king, a piece worth more than itself, or one nothing defends -- and it
    cannot be taken for profit where it stands. A knight touching two defended pawns, or a
    bishop hitting a queen and a rook while a pawn can take it, forks nothing (Tai,
    2026-09-23: forks, like pins, are tactical). The most valuable targets first; the forks
    worth most first."""
    out = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None or piece.color != colour:
            continue
        mine = _VALUE[piece.piece_type]
        hits = [
            t
            for t in board.attacks(square)
            if (o := board.piece_at(t))
            and o.color != colour
            and (o.piece_type == chess.KING or square in exchange.capturers(board, colour, t))
            and (
                o.piece_type == chess.KING
                or _VALUE[o.piece_type] > mine
                or not exchange.defenders(board, not colour, t)
            )
        ]
        safe = motifs.is_safe(board, square)
        if len(hits) >= 2 and safe:
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
    """The pieces of `colour` that can legally take on `square`, cheapest first: never one
    pinned to its own king, nor a king onto a guarded square (`analysis/exchange.py`)."""
    return sorted(
        exchange.capturers(board, colour, square),
        key=lambda s: _VALUE[board.piece_type_at(s) or chess.PAWN],
    )


def side_lines(board: chess.Board, colour: chess.Color, *, own: str, their: str) -> list[str]:
    """The state of `colour`'s pieces: hanging (a legal capturer, nothing to take back),
    attacked by a lesser piece, lost to the exchange (attacked more often than defended),
    pinned; and `colour`'s batteries and forks in place. Every capture is a legal one: a
    piece pinned to its own king neither attacks nor defends here. `own`/`their` are the
    possessives from the reader's side ("your"/"their")."""
    out: list[str] = []
    hanging: list[str] = []
    attacked: list[str] = []
    outnumbered: list[str] = []
    pinned: list[str] = []
    for s in _pieces(board, colour):
        piece = board.piece_at(s)
        assert piece is not None
        attackers = _hits_by(board, s, not colour)
        if attackers and exchange.hanging(board, s):
            hanging.append(f"{name_of(board, s)} (attacked by the {name_of(board, attackers[0])})")
        elif (
            attackers
            and _VALUE[board.piece_type_at(attackers[0]) or chess.PAWN] < _VALUE[piece.piece_type]
        ):
            attacked.append(f"{name_of(board, s)} by the {name_of(board, attackers[0])}")
        elif attackers and exchange.can_be_won(board, s):
            outnumbered.append(name_of(board, s))
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
    if outnumbered:
        out.append(
            f"{own} attacked more often than defended (lost to the exchange): "
            + "; ".join(outnumbered[:LIMIT])
        )
    if pinned:
        out.append(f"{own} pinned: " + "; ".join(pinned[:LIMIT]))
    forks = forks_in_place(board, colour)
    if forks:
        out.append(
            f"{own} forks: "
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
    """The check it is in, and what the side to move can do at once: checks and captures, in
    SAN, with what is taken. The check in force was missing, and the empty list said "no check
    is on" with the King in check (Tai's puzzle, 2026-09-28)."""
    out: list[str] = []
    if board.is_check():
        by = " and the ".join(name_of(board, s) for s in board.checkers())
        out.append(f"{own} King in check, from the {by}")
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
            # Taken back by a legal move on the board after the capture: the king cannot
            # take a piece its square is still covered by, nor a pinned piece recapture.
            defended = exchange.can_retake(board, m)
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
    return lines or [
        "Nothing hangs, nothing is pinned, and the side to move has no check or capture."
    ]
