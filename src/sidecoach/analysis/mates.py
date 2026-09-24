"""Named mating patterns: which textbook mate a checkmate is.

A coach that says "It's an Anastasia mate" is teaching a shape the player will meet again, so
the name has to be right every time: a name put on a mate that lacks the shape teaches the
wrong shape, which is worse than saying "a mating attack". Tag names are the Lichess puzzle
theme names, camelCase, spelled exactly as Lichess spells them, and the definitions are the
textbook ones as Lichess describes its themes. Two rules sit on top of them:

1. Every piece named as evidence is necessary. The knight of a Vukovic mate, the second
   bishop of a Boden, the queen of a box: each is named only if taking it off the board
   leaves no mate. A mate that still works without the piece is not made by it, however the
   pieces happen to stand (the 2026-09-23 audit found Boden, Vukovic and box mates named on
   positions that mate just as well with the "supporting" piece removed).
2. A name that describes a picture needs the picture. A corner mate has the king in the
   corner, a dovetail has the two squares the queen cannot reach blocked by the king's own
   men, an epaulette has them on both shoulders. Lichess tags some of these more loosely (a
   "corner mate" of a king on f3); the coach does not follow it there. The mates named after
   people (Morphy, Pillsbury, Opera, Balestra) say which pieces do what, not where, and are
   read that way.

Pure functions over a position that is already checkmate: no engine, no I/O, and the board
handed in is never mutated.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import chess

from sidecoach.events import MotifTag

__all__ = ["MATE_PATTERNS", "mating_pattern_tags"]


# The order the patterns are reported in, and the order they belong in `motifs.TAG_ORDER`:
# the Lichess patterns motifs already knew first, then the nine it lacked.
MATE_PATTERNS: tuple[str, ...] = (
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

_CORNERS = (chess.A1, chess.A8, chess.H1, chess.H8)
_HEAVY = (chess.ROOK, chess.QUEEN)

Evidence = list[chess.Square] | None


# --- geometry ----------------------------------------------------------------------------


def _step(a: chess.Square, b: chess.Square) -> tuple[int, int]:
    """File and rank difference from `a` to `b`."""
    return chess.square_file(b) - chess.square_file(a), chess.square_rank(b) - chess.square_rank(a)


def _at(square: chess.Square, df: int, dr: int) -> chess.Square | None:
    file_, rank_ = chess.square_file(square) + df, chess.square_rank(square) + dr
    if 0 <= file_ <= 7 and 0 <= rank_ <= 7:
        return chess.square(file_, rank_)
    return None


def _beside(a: chess.Square, b: chess.Square) -> bool:
    """Next to each other along a rank or a file."""
    df, dr = _step(a, b)
    return abs(df) + abs(dr) == 1


def _touching_corners(a: chess.Square, b: chess.Square) -> bool:
    """Next to each other on a diagonal."""
    df, dr = _step(a, b)
    return abs(df) == 1 and abs(dr) == 1


def _on_edge(square: chess.Square) -> bool:
    return chess.square_file(square) in (0, 7) or chess.square_rank(square) in (0, 7)


def _diagonal_kind(a: chess.Square, b: chess.Square) -> str | None:
    """Which way the diagonal through `a` and `b` runs, or None when they share none."""
    df, dr = _step(a, b)
    if df == 0 or abs(df) != abs(dr):
        return None
    return "rising" if df == dr else "falling"


def _tag(name: str, squares: Iterable[chess.Square]) -> MotifTag:
    names: list[str] = []
    for square in squares:
        label = chess.square_name(square)
        if label not in names:
            names.append(label)
    return MotifTag(name=name, squares=names, move=None)


# --- the mated position, read once -------------------------------------------------------


class _Mate:
    """The mated king, the piece that mates, and who guards each square around the king.

    `mater` is the piece that gives the mate: the one that moved when it gives check, else
    the piece whose line it opened. The king's squares split into `own` (held by its own
    men) and `free` (empty, or holding an enemy piece it might take). A square is guarded
    by the pieces that attack it with the king lifted off the board, so a rook checking
    along a file also guards the square behind the king on that file, which is the square
    the king cannot step back to.
    """

    def __init__(self, board: chess.Board, move: chess.Move, king: chess.Square) -> None:
        self.board = board
        self.loser = board.turn
        self.winner = not board.turn
        self.king = king
        self.checkers = list(board.checkers())
        self.mater = move.to_square if move.to_square in self.checkers else self.checkers[0]
        self.kind = board.piece_type_at(self.mater)
        self.ring = list(chess.SquareSet(chess.BB_KING_ATTACKS[king]))
        self.own = [s for s in self.ring if board.color_at(s) == self.loser]
        self.free = [s for s in self.ring if board.color_at(s) != self.loser]
        lifted = board.copy(stack=False)
        lifted.remove_piece_at(king)
        self._guards = {s: lifted.attackers(self.winner, s) for s in self.ring}
        self._needed: dict[frozenset[chess.Square], bool] = {}

    def piece(self, square: chess.Square) -> chess.PieceType | None:
        return self.board.piece_type_at(square)

    def mine(self, piece_type: chess.PieceType) -> list[chess.Square]:
        """The mating side's pieces of one type."""
        return list(self.board.pieces(piece_type, self.winner))

    def guards(self, square: chess.Square) -> chess.SquareSet:
        """Who guards a square next to the king; for the mater's square, who defends it."""
        return self._guards.get(square, chess.SquareSet())

    def covers(self, piece: chess.Square) -> list[chess.Square]:
        """The king's free squares this piece guards."""
        return [s for s in self.free if piece in self._guards[s]]

    def sole(self, piece: chess.Square) -> list[chess.Square]:
        """The king's free squares no piece but this one guards."""
        return [s for s in self.free if list(self._guards[s]) == [piece]]

    def needed(self, *pieces: chess.Square) -> bool:
        """Whether the mate falls apart once these pieces are taken off the board."""
        key = frozenset(pieces)
        if key not in self._needed:
            without = self.board.copy(stack=False)
            for square in pieces:
                without.remove_piece_at(square)
            self._needed[key] = not without.is_checkmate()
        return self._needed[key]

    def made_by(self, *pieces: chess.Square) -> bool:
        """Whether the mater and these pieces between them guard every free square but the
        mater's own, the shape being theirs rather than a third piece's."""
        named = {self.mater, *pieces}
        return all(any(p in named for p in self._guards[s]) for s in self.free if s != self.mater)

    def mater_beside(self) -> bool:
        return _beside(self.king, self.mater)


# --- the patterns --------------------------------------------------------------------------


def _back_rank(m: _Mate) -> Evidence:
    """A rook or queen checks along the back rank, the king walled in by its own men in front.

    Every square in front of the king holds one of its own pieces: a king held in by an
    enemy piece covering its luft is mated on the back rank, but not by its own pawns.
    """
    home = 0 if m.loser == chess.WHITE else 7
    if chess.square_rank(m.king) != home:
        return None
    heavy = [c for c in m.checkers if m.piece(c) in _HEAVY and chess.square_rank(c) == home]
    if not heavy:
        return None
    ahead = 1 if m.loser == chess.WHITE else -1
    front = [s for s in (_at(m.king, df, ahead) for df in (-1, 0, 1)) if s is not None]
    if any(m.board.color_at(s) != m.loser for s in front):
        return None
    return [m.king, *heavy, *front]


def _smothered(m: _Mate) -> Evidence:
    """A knight checks a king whose every neighbouring square holds one of its own men."""
    knights = [c for c in m.checkers if m.piece(c) == chess.KNIGHT]
    if not knights or m.free:
        return None
    return [m.king, knights[0], *m.own]


def _arabian(m: _Mate) -> Evidence:
    """A rook mates beside a cornered king, held there by a knight that is needed.

    Rook and knight take every square between them: a second knight or a bishop covering
    the last square makes it some other mate.
    """
    if m.kind != chess.ROOK or m.king not in _CORNERS or not m.mater_beside():
        return None
    for knight in m.guards(m.mater):
        if m.piece(knight) == chess.KNIGHT and m.needed(knight) and m.made_by(knight):
            return [m.king, m.mater, knight]
    return None


def _anastasia(m: _Mate) -> Evidence:
    """A rook or queen mates down the edge file, the king boxed in by its own man and a knight.

    The king stands on the a- or h-file, off the corner; its own piece stands beside it,
    one file in; a knight takes both squares diagonally inward, above and below. That knight
    must be needed: with a queen giving the check it can be doing nothing at all.
    """
    king_file = chess.square_file(m.king)
    if king_file not in (0, 7) or chess.square_rank(m.king) in (0, 7):
        return None
    if m.kind not in _HEAVY or chess.square_file(m.mater) != king_file:
        return None
    inward = 1 if king_file == 0 else -1
    beside = _at(m.king, inward, 0)
    if beside is None or m.board.color_at(beside) != m.loser:
        return None
    flights = [s for s in (_at(m.king, inward, 1), _at(m.king, inward, -1)) if s is not None]
    for knight in m.mine(chess.KNIGHT):
        reach = m.board.attacks(knight)
        if all(f in reach for f in flights) and m.needed(knight):
            return [m.king, m.mater, knight, beside]
    return None


def _hook(m: _Mate) -> Evidence:
    """A rook beside the king, held by a knight, the knight held by a pawn; the king's own
    pawn takes one of its squares.

    That is the textbook hook: rook, knight and pawn against one enemy pawn. The knight must
    be needed, and so must the pawn behind it (it is when the knight stands next to the king
    and would otherwise be taken). The enemy pawn must stand on one of the king's squares that
    nothing of ours guards, so it is the pawn, not our pieces, that shuts that square.
    """
    if m.kind != chess.ROOK or not m.mater_beside():
        return None
    blockers = [s for s in m.own if m.piece(s) == chess.PAWN and not m.guards(s)]
    if not blockers:
        return None
    for knight in m.guards(m.mater):
        if m.piece(knight) != chess.KNIGHT or not m.needed(knight):
            continue
        for pawn in m.board.attackers(m.winner, knight):
            if m.piece(pawn) == chess.PAWN and m.needed(pawn):
                return [m.king, m.mater, knight, pawn, blockers[0]]
    return None


def _kill_box(m: _Mate) -> Evidence:
    """Rook and queen on opposite corners of a three-by-three box with the king inside it.

    They guard each other along the box's diagonal, one of them gives the check, and the
    other must be needed and must be the only one of the two to guard some empty square of
    the king's: a queen that only backs up the rook's check is not making a box.
    """
    king_file, king_rank = chess.square_file(m.king), chess.square_rank(m.king)
    for rook in m.mine(chess.ROOK):
        for queen in m.mine(chess.QUEEN):
            if m.mater not in (rook, queen):
                continue
            df, dr = _step(rook, queen)
            if abs(df) != 2 or abs(dr) != 2:
                continue
            middle = _at(rook, df // 2, dr // 2)
            if middle is None or (middle != m.king and m.board.piece_at(middle) is not None):
                continue
            files = sorted((chess.square_file(rook), chess.square_file(queen)))
            ranks = sorted((chess.square_rank(rook), chess.square_rank(queen)))
            if not (files[0] <= king_file <= files[1] and ranks[0] <= king_rank <= ranks[1]):
                continue
            other = queen if m.mater == rook else rook
            alone = [
                s
                for s in m.covers(other)
                if m.board.piece_at(s) is None and m.mater not in m.guards(s)
            ]
            if alone and m.needed(other):
                return [m.king, m.mater, other]
    return None


def _vukovic(m: _Mate) -> Evidence:
    """A rook mates beside a king on the edge, backed by a third piece, a knight taking squares.

    The knight must be needed, and the rook's backing, the pieces other than that knight
    defending it, must be needed too: a rook the king could not take anyway is not
    "supported", it is just there.
    """
    if m.kind != chess.ROOK or not m.mater_beside() or not _on_edge(m.king):
        return None
    for knight in m.mine(chess.KNIGHT):
        if not [s for s in m.covers(knight) if s != m.mater] or not m.needed(knight):
            continue
        backing = [d for d in m.guards(m.mater) if d != knight]
        if backing and m.needed(*backing):
            return [m.king, m.mater, knight, *backing]
    return None


def _two_bishops(m: _Mate) -> tuple[str, list[chess.Square]] | None:
    """A bishop mates and a second, needed bishop takes squares: Boden's or the double bishop.

    Boden's diagonals cross, the second bishop working the other way from the check; in the
    double-bishop mate they run side by side. The direction is read off the squares only the
    second bishop guards, the reason it is needed. Both need the king hemmed in by at least
    one of its own men, and the two bishops take every square between them.
    """
    if m.kind != chess.BISHOP or not m.own:
        return None
    along = _diagonal_kind(m.mater, m.king)
    for other in m.mine(chess.BISHOP):
        if other == m.mater:
            continue
        squares = m.sole(other) or m.covers(other)
        if not squares or not m.needed(other) or not m.made_by(other):
            continue
        kinds = {_diagonal_kind(other, s) for s in squares}
        evidence = [m.king, m.mater, other, *squares]
        if kinds == {along}:
            return "doubleBishopMate", evidence
        if along not in kinds:
            return "bodenMate", evidence
    return None


def _boden(m: _Mate) -> Evidence:
    found = _two_bishops(m)
    return found[1] if found is not None and found[0] == "bodenMate" else None


def _double_bishop(m: _Mate) -> Evidence:
    found = _two_bishops(m)
    return found[1] if found is not None and found[0] == "doubleBishopMate" else None


def _queen_alone(m: _Mate) -> bool:
    """The mating queen guards every free square of the king's but its own."""
    return all(m.mater in m.guards(s) for s in m.free if s != m.mater)


def _dovetail(m: _Mate) -> Evidence:
    """A queen mates from a square diagonally next to the king; the two squares it cannot
    reach, beside and behind the king, hold the king's own men, and the queen covers the rest.
    """
    if m.kind != chess.QUEEN or not _touching_corners(m.king, m.mater):
        return None
    df, dr = _step(m.king, m.mater)
    tail = [_at(m.king, -df, 0), _at(m.king, 0, -dr)]
    blocked = [s for s in tail if s is not None and m.board.color_at(s) == m.loser]
    if len(blocked) != 2 or not _queen_alone(m):
        return None
    return [m.king, m.mater, *blocked]


def _swallows_tail(m: _Mate) -> Evidence:
    """A queen mates from the square straight next to the king; the two squares diagonally
    behind the king, which it cannot reach, hold the king's own men, and it covers the rest.
    """
    if m.kind != chess.QUEEN or not m.mater_beside():
        return None
    df, dr = _step(m.king, m.mater)
    across_f, across_r = dr, df
    tail = [
        _at(m.king, -df + across_f, -dr + across_r),
        _at(m.king, -df - across_f, -dr - across_r),
    ]
    blocked = [s for s in tail if s is not None and m.board.color_at(s) == m.loser]
    if len(blocked) != 2 or not _queen_alone(m):
        return None
    return [m.king, m.mater, *blocked]


def _epaulette(m: _Mate) -> Evidence:
    """A rook or queen checks along a rank or file; the king's own men stand on both of its
    shoulders, the two squares beside it across the line of the check.
    """
    if m.kind not in _HEAVY:
        return None
    df, dr = _step(m.king, m.mater)
    if df != 0 and dr != 0:
        return None
    across_f, across_r = (1, 0) if df == 0 else (0, 1)
    shoulders = [_at(m.king, across_f, across_r), _at(m.king, -across_f, -across_r)]
    blocked = [s for s in shoulders if s is not None and m.board.color_at(s) == m.loser]
    if len(blocked) != 2:
        return None
    return [m.king, m.mater, *blocked]


def _triangle(m: _Mate) -> Evidence:
    """A queen mates from diagonally next to the king with a rook on the other near corner of
    the same rank or file: king, queen and rook make the triangle. The rook must be needed.
    """
    if m.kind != chess.QUEEN or not _touching_corners(m.king, m.mater):
        return None
    for rook in m.mine(chess.ROOK):
        if not _touching_corners(m.king, rook):
            continue
        df, dr = _step(m.mater, rook)
        if (df == 0 or dr == 0) and m.needed(rook):
            return [m.king, m.mater, rook]
    return None


def _blind_swine(m: _Mate) -> Evidence:
    """Two rooks mate side by side next to the king, the three of them in one 2x2 square.

    One rook checks from straight beside the king, the other stands next to it on the
    king's diagonal, backing it up and taking squares; that second rook must be needed.
    """
    if m.kind != chess.ROOK or not m.mater_beside():
        return None
    for rook in m.mine(chess.ROOK):
        if rook == m.mater:
            continue
        if _touching_corners(m.king, rook) and _beside(m.mater, rook) and m.needed(rook):
            return [m.king, m.mater, rook]
    return None


def _corner(m: _Mate) -> Evidence:
    """A knight mates a king in the corner that a rook or queen has shut in.

    The rook or queen must guard a free square, be needed, and with the knight take every
    square. A corner king hemmed in by its own men alone is a smothered mate, not this.
    """
    if m.kind != chess.KNIGHT or m.king not in _CORNERS or not m.free:
        return None
    for heavy in (*m.mine(chess.ROOK), *m.mine(chess.QUEEN)):
        if m.covers(heavy) and m.needed(heavy) and m.made_by(heavy):
            return [m.king, m.mater, heavy]
    return None


def _opera(m: _Mate) -> Evidence:
    """A rook mates from straight next to the king, defended by a bishop that is needed."""
    if m.kind != chess.ROOK or not m.mater_beside():
        return None
    for bishop in m.guards(m.mater):
        if m.piece(bishop) == chess.BISHOP and m.needed(bishop):
            return [m.king, m.mater, bishop]
    return None


def _pillsbury(m: _Mate) -> Evidence:
    """A rook mates and a bishop, not the rook's defender, takes the king's squares.

    The bishop must be needed. A bishop that only defends a rook next to the king makes the
    Opera mate instead.
    """
    if m.kind != chess.ROOK:
        return None
    defenders = m.guards(m.mater) if m.mater_beside() else chess.SquareSet()
    for bishop in m.mine(chess.BISHOP):
        if bishop in defenders or bishop in m.checkers:
            continue
        if m.covers(bishop) and m.needed(bishop):
            return [m.king, m.mater, bishop]
    return None


def _morphy(m: _Mate) -> Evidence:
    """A bishop mates and a rook takes the king's squares; the rook must be needed.

    A rook that only defends a bishop next to the king is backing, not confining, so it has
    to guard a square other than the bishop's.
    """
    if m.kind != chess.BISHOP:
        return None
    for rook in m.mine(chess.ROOK):
        if rook in m.checkers:
            continue
        if [s for s in m.covers(rook) if s != m.mater] and m.needed(rook):
            return [m.king, m.mater, rook]
    return None


def _balestra(m: _Mate) -> Evidence:
    """A bishop mates from a distance and a queen a knight's move from the king takes every
    square the bishop leaves. The queen gives no check itself and must be needed.
    """
    if m.kind != chess.BISHOP or m.mater_beside() or _touching_corners(m.king, m.mater):
        return None
    for queen in m.mine(chess.QUEEN):
        if queen in m.checkers or chess.square_knight_distance(queen, m.king) != 1:
            continue
        rest = [s for s in m.free if m.mater not in m.guards(s)]
        if all(queen in m.guards(s) for s in rest) and m.needed(queen):
            return [m.king, m.mater, queen]
    return None


_DETECTORS: dict[str, Callable[[_Mate], Evidence]] = {
    "backRankMate": _back_rank,
    "smotheredMate": _smothered,
    "arabianMate": _arabian,
    "anastasiaMate": _anastasia,
    "hookMate": _hook,
    "killBoxMate": _kill_box,
    "vukovicMate": _vukovic,
    "bodenMate": _boden,
    "doubleBishopMate": _double_bishop,
    "dovetailMate": _dovetail,
    "swallowstailMate": _swallows_tail,
    "epauletteMate": _epaulette,
    "triangleMate": _triangle,
    "blindSwineMate": _blind_swine,
    "cornerMate": _corner,
    "operaMate": _opera,
    "pillsburysMate": _pillsbury,
    "morphysMate": _morphy,
    "balestraMate": _balestra,
}


# --- public API --------------------------------------------------------------------------


def mating_pattern_tags(board_after_mate: chess.Board, move: chess.Move) -> list[MotifTag]:
    """The named patterns of a checkmate, in `MATE_PATTERNS` order.

    `board_after_mate` is the position with the mated side to move and `move` is the move
    that mated. A position that is not checkmate names no pattern. Evidence squares start
    with the king and the mating piece, then the pieces that help, then the squares that
    make the shape.
    """
    if not board_after_mate.is_checkmate():
        return []
    king = board_after_mate.king(board_after_mate.turn)
    if king is None:
        return []
    mate = _Mate(board_after_mate, move, king)
    tags: list[MotifTag] = []
    for name in MATE_PATTERNS:
        evidence = _DETECTORS[name](mate)
        if evidence is None:
            continue
        # In a double check the second checker is part of the mate whenever it is needed (it
        # stops the capture of the first), so a pattern that leaves it out is not the mate.
        if any(c not in evidence and mate.needed(c) for c in mate.checkers if c != mate.mater):
            continue
        tags.append(_tag(name, evidence))
    return tags
