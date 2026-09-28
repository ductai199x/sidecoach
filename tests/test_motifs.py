"""Motif tagger tests.

Every position here is one a coach could talk through, and every expectation is what a coach
would say about it, worked out from the board. The negative tests matter most: a tagger that
invents a pin makes the coach say something untrue, which is worse than saying nothing.

The mating patterns themselves are test_mates.py's; here, the moves that carry them.
"""

from collections.abc import Sequence

import chess
import pytest

from sidecoach.analysis.mates import MATE_PATTERNS
from sidecoach.analysis.motifs import TAG_ORDER, tags_for_move, tags_for_position

# Everything that claims a tactic. Group A (what the move physically is) is not in here.
QUIET_GROUP = {
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
}
TACTICAL = frozenset(TAG_ORDER) - QUIET_GROUP

# Ordinary moves in ordinary positions. None of them does anything tactical.
QUIET_MOVES = [
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "d2d4"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "g1f3"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "g1f3"),
    ("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "b8c6"),
    ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f1b5"),
    ("r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8", "d2d3"),
    ("r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8", "a1b1"),
    ("r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8", "f1e1"),
    ("r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8", "h2h3"),
    ("r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8", "e2d3"),
]

# The positions several cases share.
# 1.e4 e5: white to move.
OPEN_GAME = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
# The Italian after 3...Nc6 4.Nf3, white to move and able to castle.
ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
# A quiet castled middlegame: nothing tactical anywhere.
STEADY = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
# Black has just played ...f7-f5 past the white pawn on e5, so exf6 e.p. is available.
EN_PASSANT = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
PROMOTION = "8/P7/8/8/8/8/8/K5k1 w - - 0 1"
# The queen on a8 is boxed in by her own rook and pawn; Bc1-b2 defends the rook on a1, which
# she was winning outright, so that Qxa1 stops working and every square on the a-file is
# covered.
RESCUE = "qr4k1/1p6/8/8/8/8/8/R1B3K1 w - - 0 1"
# The king on g8 is walled in by its own f7, g7 and h7 pawns: Ra8 is mate.
BACK_RANK = "6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1"
# Philidor's legacy: Qg8+ forces Rxg8 and Nf7 is mate.
SMOTHERED = "5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1"
# Ne4-d6+ checks from d6 and uncovers the rook on e1 at the same time.
DISCOVERED_CHECK = "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1"
# The rook on e1 already pins the knight on e5 to the king.
PINNED_BY_E1 = "4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1"
# The bishop on h4 already pins the knight on f6 to the queen on d8 through g5.
PINNED_FROM_H4 = "3q1rk1/ppp2ppp/5n2/8/7B/8/PPP2PPP/3QK2R w K - 0 1"
# The Greek gift: Bxh7+ Kxh7 Ng5+ Kg8 Nxe6 fxe6, the queen on e6 won for the bishop.
GREEK_GIFT = "5rk1/5ppp/4q3/8/8/3B1N2/8/3Q2K1 w - - 0 1"


def tag_names(fen: str, uci: str, pv: Sequence[str] = (), mate_in: int | None = None) -> list[str]:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    # Guard every negative assertion below: an illegal move tags as nothing, which would make
    # "this tag is absent" true for the wrong reason.
    assert move in board.legal_moves, f"{uci} is not legal in {fen}"
    tags = tags_for_move(board, move, pv=[chess.Move.from_uci(x) for x in pv], mate_in=mate_in)
    assert board.fen() == fen, "tags_for_move mutated the board it was given"
    return [tag.name for tag in tags]


def squares_of(
    fen: str, uci: str, name: str, pv: Sequence[str] = (), mate_in: int | None = None
) -> list[str]:
    board = chess.Board(fen)
    moves = [chess.Move.from_uci(x) for x in pv]
    tags = tags_for_move(board, chess.Move.from_uci(uci), pv=moves, mate_in=mate_in)
    return next(tag.squares for tag in tags if tag.name == name)


def position_names(fen: str, pv: Sequence[str] = (), mate_in: int | None = None) -> list[str]:
    board = chess.Board(fen)
    moves = [chess.Move.from_uci(x) for x in pv]
    tags = tags_for_position(board, pv=moves, mate_in=mate_in)
    assert board.fen() == fen, "tags_for_position mutated the board it was given"
    return [tag.name for tag in tags]


def _played(case: str, fen: str, line: Sequence[str]) -> chess.Board:
    """The board after the move and its line, every move of it legal."""
    board = chess.Board(fen)
    for uci in line:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, f"{case}: {uci} is not legal in {board.fen()}"
        board.push(move)
    return board


# --- what a move makes -------------------------------------------------------------------

# (case, FEN, the move, the engine's line after it, the mate distance given, and each tag
# the move must carry with its evidence squares: a list exactly, a set in any order, a
# string the first square only (the piece it is about), None unchecked).
Squares = list[str] | set[str] | str | None
MADE: list[tuple[str, str, str, list[str], int | None, dict[str, Squares]]] = [
    # --- group A: what the move is
    # The captured pawn is on f5, not on the square the capturing pawn lands on.
    (
        "en passant",
        EN_PASSANT,
        "e5f6",
        [],
        None,
        {"capture": None, "enPassant": ["e5", "f6", "f5"]},
    ),
    (
        "castling",
        ITALIAN,
        "e1g1",
        [],
        None,
        {"castling": ["e1", "g1", "h1", "f1"], "quietMove": None},
    ),
    ("queening", PROMOTION, "a7a8q", [], None, {"promotion": None, "advancedPawn": None}),
    ("knighting", PROMOTION, "a7a8n", [], None, {"underPromotion": None}),
    # b5-b6 reaches the sixth rank.
    ("to the sixth", "6k1/8/8/1P6/8/8/8/6K1 w - - 0 1", "b5b6", [], None, {"advancedPawn": None}),
    # From black's side the sixth rank is rank 3.
    (
        "to black's sixth",
        "6k1/8/8/8/8/8/1p6/6K1 b - - 0 1",
        "b2b1q",
        [],
        None,
        {"advancedPawn": None},
    ),
    # The rook lands on a8; the king on g8 is shut in by f7/g7/h7. Checkmate on the board is
    # enough on its own for the mate distance.
    (
        "back rank, the board alone",
        BACK_RANK,
        "a1a8",
        [],
        None,
        {"check": ["a8", "g8"], "mateIn1": None},
    ),
    # Ra8 is mate on the spot, so the caller's 1 and the board agree.
    (
        "back rank, mate in one given",
        BACK_RANK,
        "a1a8",
        [],
        1,
        {"mate": None, "mateIn1": None, "backRankMate": {"g8", "a8", "f7", "g7", "h7"}},
    ),
    # Bc1-b2 defends the rook on a1, which the queen on a8 was winning outright.
    ("rescue", RESCUE, "c1b2", [], None, {"defensiveMove": ["b2", "a1"]}),
    # --- group B: what the move creates
    # Nd5-f6 checks the king on g8 and hits the queen on e8 at the same time.
    (
        "knight fork",
        "4q1k1/8/8/3N4/8/8/8/7K w - - 0 1",
        "d5f6",
        [],
        None,
        {"check": None, "fork": ["f6", "e8", "g8"]},
    ),
    # Nh6-f7+ forks the king on h8, the queen on d8 and the rook on d6.
    (
        "royal fork",
        "3q3k/8/3r3N/8/8/8/8/K7 w - - 0 1",
        "h6f7",
        [],
        None,
        {"fork": ["f7", "d6", "d8", "h8"]},
    ),
    # e4-e5 attacks both knights; a pawn taking a knight is a win either way.
    (
        "pawn fork",
        "4k3/8/3n1n2/8/4P3/8/8/4K3 w - - 0 1",
        "e4e5",
        [],
        None,
        {"fork": ["e5", "d6", "f6"]},
    ),
    # Qd1-a4+ checks the king on e8 and hits the loose bishop on b4. The bishop is worth less
    # than the queen, but the check forces a king move and it falls: a real fork the plain
    # value test would miss.
    (
        "queen check fork",
        "4k3/8/8/8/1b6/8/8/3Q3K w - - 0 1",
        "d1a4",
        [],
        None,
        {"check": None, "fork": ["a4", "b4", "e8"]},
    ),
    # Rd1-e1 lines up on the knight on e5 with the black king behind it on e8.
    (
        "absolute pin",
        "4k3/8/8/4n3/8/8/8/3R2K1 w - - 0 1",
        "d1e1",
        [],
        None,
        {"pin": None, "absolutePin": ["e1", "e5", "e8"]},
    ),
    # Bc1-g5 pins the knight on f6 to the queen on d8. From c1 the bishop was not on that
    # diagonal, so the move is what creates the pin.
    (
        "relative pin",
        "3q1rk1/ppp2ppp/5n2/8/8/8/PPP2PPP/2BQK2R w K - 0 1",
        "c1g5",
        [],
        None,
        {"pin": ["g5", "f6", "d8"]},
    ),
    # Rh1-a1+ checks the king on a5; behind it on the file sits the queen on a8. Every king
    # step off the file leaves her to Rxa8, and none reaches a square that guards her.
    ("skewer", "q7/8/8/k7/8/8/7K/7R w - - 0 1", "h1a1", [], None, {"skewer": ["a1", "a5", "a8"]}),
    # Nd4-f5 steps off the long diagonal and the bishop on b2 hits the queen on h8.
    (
        "discovered attack",
        "7q/8/8/8/3N4/8/1B6/K5k1 w - - 0 1",
        "d4f5",
        [],
        None,
        {"discoveredAttack": ["d4", "b2", "h8"]},
    ),
    (
        "discovered check",
        DISCOVERED_CHECK,
        "e4d6",
        [],
        None,
        {"check": ["e1", "d6", "e8"], "doubleCheck": None, "discoveredAttack": None},
    ),
    # Bc1-f4 rakes the long diagonal onto the rook on b8, which nothing defends.
    (
        "hanging rook",
        "1r4k1/5ppp/8/8/8/8/5PPP/2B3K1 w - - 0 1",
        "c1f4",
        [],
        None,
        {"hangingPiece": ["b8"]},
    ),
    # Qxd8+ Rxd8 Rxd8#: the rook on d1 mates on d8, bearing down the file through the black
    # rook on d5 the whole time; the line clears it and the x-ray lands (Lichess xRayAttack).
    (
        "x-ray",
        "3r2k1/1q3ppp/p3p3/Qp1r4/7P/P4P2/1PP3P1/1K1R3R w - - 0 22",
        "a5d8",
        ["d5d8", "d1d8"],
        None,
        {"xRayAttack": ["d8", "d5", "d1"]},
    ),
    # Bg3-d6 sits down between the rook on d8 and the knight on d5, which the e4 pawn wins.
    # The c5 pawn holds the bishop there: Rxd6 cxd6 costs the rook.
    (
        "interference",
        "3r2k1/8/8/2Pn4/4P3/6B1/8/6K1 w - - 0 1",
        "g3d6",
        [],
        None,
        {"interference": ["d6", "d8", "d5"], "hangingPiece": None},
    ),
    # Rd7+ Kf8 Rxc8#: the check drives the king to f8, where it blocks the rook on h8 off
    # c8, so c8 is no longer defended: the enemy forced to interfere with itself (Lichess).
    (
        "self-interference",
        "2r4r/p3kpp1/1N2p1n1/4P3/7p/q7/P4PPP/2RR2K1 w - - 0 32",
        "d1d7",
        ["e7f8", "c1c8"],
        None,
        {"interference": ["c8", "f8", "h8"]},
    ),
    # Bb2 shuts the queen in: she is the trapped piece (the hunters vary).
    ("trapped queen", RESCUE, "c1b2", [], None, {"trappedPiece": "a8"}),
    # Qh5 joins the bishop on c4 against f7, which only the king holds: Qxf7 wins it.
    (
        "f7",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
        "d1h5",
        [],
        None,
        {"attackingF2F7": ["h5", "f7"]},
    ),
    # And the mirror image for black: Qh4 joins the bishop on c5 against f2.
    (
        "f2",
        "rnbqk1nr/pppp1ppp/8/2b1p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 0 3",
        "d8h4",
        [],
        None,
        {"attackingF2F7": None},
    ),
    # --- group C: what the line does
    # Qg8+ forces Rxg8 and Nf7 is mate, the knight smothered by g7, h7 and the rook dragged
    # to g8. The rook on f8 was the only guard of f7 and the queen check pulled it away.
    (
        "smothered by a queen sacrifice",
        SMOTHERED,
        "b3g8",
        ["f8g8", "h6f7"],
        2,
        {
            "sacrifice": None,
            "mate": None,
            "mateIn2": None,
            "smotheredMate": None,
            "deflection": ["f8", "g8", "f7"],
        },
    ),
    # Bxc6 removes the knight that held e5; after dxc6 the knight takes on e5.
    (
        "capturing the defender",
        "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
        "b5c6",
        ["d7c6", "f3e5"],
        None,
        {"capturingDefender": ["c6", "e5"]},
    ),
    # Bxh7+ Kxh7 and Ng5+ forks the king on the square it was pulled to and the queen on e6,
    # which falls: the bishop comes back with a queen's interest.
    (
        "attraction",
        GREEK_GIFT,
        "d3h7",
        ["g8h7", "f3g5", "h7g8", "g5e6", "f7e6"],
        None,
        {"attraction": ["g8", "h7", "g5"]},
    ),
    # Nxe6 takes a pawn and steps off d4, and the rook swings down the d-file it opened.
    (
        "clearance",
        "4k3/p7/4p3/8/3N4/8/8/3R2K1 w - - 0 1",
        "d4e6",
        ["a7a6", "d1d8"],
        None,
        {"clearance": ["d4", "d1", "d8"]},
    ),
    # The Greek gift: a bishop for a pawn on move one, paid for by the attack that follows.
    (
        "a piece handed to the reply",
        "rnbq1rk1/ppp2ppp/3b1n2/8/3P4/3B1N2/PPP2PPP/RNBQ1RK1 w - - 0 1",
        "d3h7",
        ["g8h7", "f3g5", "h7g6"],
        4,
        {"sacrifice": None, "capture": None, "check": None},
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "uci", "pv", "mate_in", "made"), MADE, ids=[row[0] for row in MADE]
)
def test_a_move_carries_what_it_makes(
    case: str, fen: str, uci: str, pv: list[str], mate_in: int | None, made: dict[str, Squares]
) -> None:
    end = _played(case, fen, [uci, *pv])
    if set(made) & set(MATE_PATTERNS):
        assert end.is_checkmate(), f"{case}: a mating pattern on a line that does not mate"
    names = tag_names(fen, uci, pv, mate_in)
    for name, squares in made.items():
        assert name in names, f"{case}: no {name} in {names}"
        if squares is None:
            continue
        got = squares_of(fen, uci, name, pv, mate_in)
        if isinstance(squares, set):
            assert set(got) == squares, f"{case}: {name} {got}"
        elif isinstance(squares, str):
            assert got[0] == squares, f"{case}: {name} {got}"
        else:
            assert got == squares, f"{case}: {name} {got}"


# --- what a move does not make -----------------------------------------------------------

# (case, FEN, the move, the engine's line after it, the tag -- or tags -- it must not carry).
# The "audit" rows were tagged by the geometry alone (the audit of 2026-09-23, its item
# number): the coach would have taught a tactic that wins nothing.
NOT_MADE: list[tuple[str, str, str, list[str], str | frozenset[str]]] = [
    # --- group A
    ("en passant is no tactic", EN_PASSANT, "e5f6", [], TACTICAL),
    ("queening is no underpromotion", PROMOTION, "a7a8q", [], "underPromotion"),
    # A bare passer on b5 has not reached the sixth rank.
    ("to the fifth", "6k1/8/8/8/1P6/8/8/6K1 w - - 0 1", "b4b5", [], "advancedPawn"),
    # d2-d3 props up e4, but e4 was already defended by the knight on c3 and only attacked
    # once. A second guard on a pawn that was holding is not a rescue.
    ("a second guard", STEADY, "d2d3", [], "defensiveMove"),
    # --- group B
    # The Qa4+ fork, but a rook on a8 can capture the checking queen: the bishop is not won.
    ("the check is answered", "r3k3/8/8/8/1b6/8/8/3Q3K w - - 0 1", "d1a4", [], "fork"),
    # e4-e5 attacks the d6 and f6 pawns, both held by the pawn behind: nothing is won.
    ("two defended pawns", "4k3/2p1pp2/3p1p2/8/4P3/8/8/4K3 w - - 0 1", "e4e5", [], "fork"),
    # The relative pin is not absolute.
    (
        "a pin on the queen",
        "3q1rk1/ppp2ppp/5n2/8/8/8/PPP2PPP/2BQK2R w K - 0 1",
        "c1g5",
        [],
        "absolutePin",
    ),
    # Stepping to g5 creates nothing: the pin belongs to the position, not to the move.
    ("a pin already there", PINNED_FROM_H4, "h4g5", [], "pin"),
    # Shuffling the king on the other wing did not create the pin on e5.
    ("a pin not inherited", PINNED_BY_E1, "g1h1", [], frozenset({"pin", "absolutePin"})),
    # Rd1-e1 hits the queen on e5 with the loose rook on e8 behind, but Qb8 steps off the
    # file and guards e8: Rxe8+ Qxe8 is only a trade.
    ("skewer, the queen guards", "4r1k1/8/8/4q3/8/8/8/3R1K2 w - - 0 1", "d1e1", [], "skewer"),
    # The same with Rb1 ...Qe8 guarding b8.
    ("skewer, Qe8 guards", "1r6/6pk/7p/1q6/8/8/K5PP/R7 w - - 0 1", "a1b1", ["b5e8"], "skewer"),
    # With the white king on g1 the rook on e1 is loose, so Qxe1 answers it.
    ("skewer, the rook is taken", "4r1k1/8/8/4q3/8/8/8/3R2K1 w - - 0 1", "d1e1", [], "skewer"),
    # Nf3 attacks the undefended e5 pawn after 1.e4 e5: calling that a hanging piece would
    # have the coach warning about a pawn on move two of every open game.
    ("a loose pawn", OPEN_GAME, "g1f3", [], "hangingPiece"),
    # Rd1 looks through the d5 rook at d8, but the line never clears d5.
    (
        "x-ray never realised",
        "3r2k1/1q3ppp/p3p3/1p1r4/7P/P4P2/1PP3P1/1K1R3R w - - 0 22",
        "h4h5",
        ["b5b4", "h5h6"],
        "xRayAttack",
    ),
    # Without the c5 pawn the rook just takes the bishop and guards d5 again.
    ("interference, taken", "3r2k1/8/8/3n4/4P3/6B1/8/6K1 w - - 0 1", "g3d6", [], "interference"),
    # Bc4 on move two only looks at f7: the king holds it and Bxf7+ Kxf7 loses the bishop.
    ("f7 held", OPEN_GAME, "f1c4", [], "attackingF2F7"),
    # --- group C
    # Rxe8+ Rxe8 is a straight swap of rooks: the material dips and comes straight back.
    ("an even trade", "r3r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "e1e8", ["a8e8"], "sacrifice"),
    # 1.e4 e5 2.f4 exf4 gives a pawn. One point is a gambit, not a sacrifice.
    ("a gambit", OPEN_GAME, "f2f4", ["e5f4", "g1f3"], "sacrifice"),
    # The opponent captures first and the material comes back: measured over the whole line,
    # 2.Nf3 came out as a sacrifice in the game of 2026-09-14 20:06.
    (
        "a trade deep in the line",
        OPEN_GAME,
        "g1f3",
        ["b8c6", "f1b5", "g8f6", "b5c6", "d7c6", "f3e5"],
        "sacrifice",
    ),
    # The opponent taking a knight and giving it straight back is still not a sacrifice.
    ("a knight given back", OPEN_GAME, "g1f3", ["b8c6", "f3e5", "c6e5", "d2d4"], "sacrifice"),
    # The same Greek gift, cut before it wins anything: a sacrifice, not yet a lure.
    ("no lure yet", GREEK_GIFT, "d3h7", ["g8h7", "f3g5"], "attraction"),
    # --- mates
    # The rook check, but the h-pawn has moved and Kh7 is free: no mate, no pattern.
    (
        "the king has air",
        "6k1/5pp1/7p/8/8/8/8/R3K3 w Q - 0 1",
        "a1a8",
        [],
        frozenset({"backRankMate", "mate"}),
    ),
    # A check that is not mate names no mating pattern at all.
    ("a check that is not mate", DISCOVERED_CHECK, "e4d6", [], frozenset(MATE_PATTERNS)),
    # --- the audit of 2026-09-23
    # 1: the checking forker is simply taken (gxf6, Kxf7).
    ("audit 1a", "6k1/3q1ppp/8/8/4N3/8/5PPP/6K1 w - - 0 1", "e4f6", ["g7f6"], "fork"),
    ("audit 1b", "4r1k1/6pp/6B1/8/8/8/6PP/6K1 w - - 0 1", "g6f7", ["g8f7"], "fork"),
    # 2: the bishop is pinned to its king by the f6 bishop and may take neither rook.
    ("audit 2", "8/r5k1/5b2/4B3/8/5N2/1K6/6r1 w - - 0 1", "e5d4", [], "fork"),
    # 3: ...Re5, held by d6, blocks the check and pins the queen: the knight is safe.
    ("audit 3", "4k3/8/3p4/7r/n6Q/8/8/4K3 w - - 0 1", "h4e4", [], "fork"),
    # 4: the pinner is simply won (...Rxd1+, ...Rxe1+, ...axb5).
    ("audit 4a", "3q2k1/5ppp/8/3r4/8/8/5PPP/R5K1 w - - 0 1", "a1d1", [], "pin"),
    ("audit 4b", "4k3/8/8/4r3/8/8/6PP/R5K1 w - - 0 1", "a1e1", [], "absolutePin"),
    ("audit 4c", "4r1k1/5ppp/p1n5/8/8/8/5PPP/5BK1 w - - 0 1", "f1b5", [], "pin"),
    # 6: the e5 pawn's only move is down the file: the "pin" holds back nothing.
    ("audit 6", "4q1k1/5ppp/8/4p3/8/8/5PPP/R5K1 w - - 0 1", "a1e1", [], "pin"),
    # 7: ...axb5 wins the skewering bishop.
    ("audit 7", "4r1k1/5ppp/p1q5/8/P7/8/4BPPP/6K1 w - - 0 1", "e2b5", [], "skewer"),
    # 10a: ...fxe4 takes the hunter, so the rook on a8 is not trapped.
    ("audit 10a", "rn4k1/p1p3pp/8/5p2/8/3B2P1/5P1P/6K1 w - - 0 1", "d3e4", [], "trappedPiece"),
    # 11: the knight guards the rook, but ...Nxe4 Nxe4 still wins the exchange.
    ("audit 11", "6k1/5ppp/3n4/8/4R3/8/5PPP/5NK1 w - - 0 1", "f1g3", [], "defensiveMove"),
    # 12: 2.d4 opens the c1 bishop onto h6, which the g7 pawn holds.
    (
        "audit 12",
        "rnbqkbnr/ppppppp1/7p/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "d2d4",
        [],
        "discoveredAttack",
    ),
    # 13: ...Nfd7 is a free retreat, and d5 was taken before it, not after.
    (
        "audit 13",
        "r1bqkb1r/ppp2ppp/5n2/4p3/4P3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 1",
        "c3d5",
        ["f6d7"],
        "interference",
    ),
    # 14: ...Rxc5 takes the interposer; the rook on e4 is pinned to its king and takes nothing.
    ("audit 14a", "6k1/5ppp/8/r3p3/8/3N1N2/5PPP/6K1 w - - 0 1", "d3c5", [], "interference"),
    ("audit 14b", "4r3/5k2/8/8/q6r/3P4/4R3/4K3 w - - 0 1", "e2e4", [], "interference"),
    # 15: the rook on a1 is blocked by its own knight on a4 when the bishop takes on a7.
    (
        "audit 15",
        "6k1/p4ppp/8/8/n7/2N1B3/5PPP/R5K1 w - - 0 1",
        "c3a4",
        ["g8h8", "e3a7"],
        "xRayAttack",
    ),
    # 16: g7 still holds f6; and the rook given for the knight costs more than the pawn.
    (
        "audit 16a",
        "6k1/5ppp/4pn2/3n2B1/8/2N5/5PPP/6K1 w - - 0 1",
        "c3d5",
        ["e6d5", "g5f6", "g7f6"],
        "capturingDefender",
    ),
    (
        "audit 16b",
        "6k1/5ppp/5n2/8/4p3/2N5/6PP/5RK1 w - - 0 1",
        "f1f6",
        ["g7f6", "c3e4"],
        "capturingDefender",
    ),
    # 17: ...Qh4 is a free choice; and Re8+ Kg7 wins nothing.
    (
        "audit 17a",
        "3q2k1/p2b1ppp/8/8/8/4B3/5PPP/3R2K1 w - - 0 1",
        "e3a7",
        ["d8h4", "d1d7"],
        "deflection",
    ),
    (
        "audit 17b",
        "6k1/4qp1p/5np1/6B1/8/8/5PPP/4R1K1 w - - 0 1",
        "g5f6",
        ["e7f6", "e1e8", "g8g7"],
        "deflection",
    ),
    # 18: a knight taking back a knight, then a bishop for a knight: a trade, not a lure.
    (
        "audit 18",
        "6k1/5ppp/4pn2/3n4/8/1BN5/5PPP/6K1 w - - 0 1",
        "c3d5",
        ["f6d5", "b3d5", "e6d5"],
        "attraction",
    ),
    # 19: the queen grabs b7 and comes back to d5: nothing else used the square.
    ("audit 19", "5rk1/1p3ppp/8/3Q4/8/8/5PPP/6K1 w - - 0 1", "d5b7", ["h7h6", "b7d5"], "clearance"),
    # 20: the king on f7 is checked; there is no weak pawn there.
    ("audit 20", "8/5k2/8/8/8/8/6PP/R5K1 w - - 0 1", "a1a7", [], "attackingF2F7"),
]


@pytest.mark.parametrize(
    ("case", "fen", "uci", "pv", "absent"), NOT_MADE, ids=[row[0] for row in NOT_MADE]
)
def test_a_move_carries_nothing_it_does_not_make(
    case: str, fen: str, uci: str, pv: list[str], absent: str | frozenset[str]
) -> None:
    _played(case, fen, [uci, *pv])
    names = set(tag_names(fen, uci, pv=pv))
    wrong = ({absent} if isinstance(absent, str) else set(absent)) & names
    assert not wrong, f"{case}: {sorted(wrong)}"


# --- what a position holds ---------------------------------------------------------------

# (case, FEN, the tag it holds -- or, as a list, exactly the tags it holds).
HELD: list[tuple[str, str, str | list[str]]] = [
    # The bishop on h4 pins the knight on f6 to the queen on d8, whatever it does next.
    ("a standing pin", PINNED_FROM_H4, "pin"),
    ("a standing pin to the king", PINNED_BY_E1, "absolutePin"),
    # The bishop on b5 pins the knight on c6 to the queen on d7, which the king defends:
    # bishop for queen still wins material, so the knight cannot move.
    ("a pin on a defended queen", "4k3/3q4/2n5/1B6/8/8/8/4K3 w - - 0 1", "pin"),
    # The bishop on a4 against the pawn on c6 and a loose rook on e8 behind it.
    ("a pin on a loose rook", "4r2k/8/2p5/8/B7/8/8/4K3 w - - 0 1", "pin"),
    # And the king behind always pins.
    ("a pin on the king", "4k3/8/2n5/1B6/8/8/8/4K3 w - - 0 1", "absolutePin"),
    # The bishop on b5, held by a4, is pinned to b1 by the b8 rook, but the knight on c6 may
    # not leave the line to the king on e8: that is the rule, not a threat, so it is a pin.
    ("a pin by a pinned piece", "1r2k3/5ppp/2n5/1B6/P7/8/5PPP/1K6 w - - 0 1", "absolutePin"),
    # Black's king on g8 has no pawn in front of it and white has queen, rook and bishop
    # pointing at the ring.
    ("an exposed king", "r5k1/ppp5/8/7Q/2B5/8/8/5RK1 w - - 0 1", ["exposedKing"]),
]


@pytest.mark.parametrize(("case", "fen", "held"), HELD, ids=[row[0] for row in HELD])
def test_a_position_holds_its_feature(case: str, fen: str, held: str | list[str]) -> None:
    names = position_names(fen)
    if isinstance(held, list):
        assert names == held, case
    else:
        assert held in names, f"{case}: {names}"


# (case, FEN, the tag the position must not hold).
NOT_HELD = [
    # The Italian bishop on c4 looks through the f7 pawn to the knight on g8, which the h8
    # rook holds: taking the knight would cost the bishop, so nothing is pinned in a sense the
    # coach may teach, though Lichess would tag it (Tai, 2026-09-23: "if that piece is
    # defended then there is no winning that piece hence it's not a valid pin").
    ("a pin on a defended knight", ITALIAN, "pin"),
    # A queen against a knight with a defended rook behind it: queen for rook loses.
    ("a queen against a defended rook", "8/3k4/3r4/8/3n4/8/8/3QK3 w - - 0 1", "pin"),
    # The same bare king with only a rook and knight around is an endgame, not an attack.
    ("a bare king in an endgame", "r5k1/ppp5/8/8/8/5N2/8/5RK1 w - - 0 1", "exposedKing"),
    # Before the bishop came to b2 the queen was not trapped: Qxa1 simply won a rook.
    ("not trapped yet", RESCUE, "trappedPiece"),
    # Audit 5: the b5 bishop (held by a4) is pinned to b1 by the b8 rook: it can never take e8.
    ("audit 5", "1r2r1k1/5ppp/2n5/1B6/P7/8/5PPP/1K6 w - - 0 1", "pin"),
    # Audit 9: the knight on e2 is pinned to its king; white is in check and cannot take on e5.
    ("audit 9a", "4r1k1/8/8/8/5n2/8/4N3/4K3 w - - 0 1", "hangingPiece"),
    ("audit 9b", "k7/8/8/4n3/8/8/1B4PP/3r2K1 w - - 0 1", "hangingPiece"),
    # Audit 10: the knight on h6 has f5 and g4, both held; the hunting bishop is pinned to e1.
    ("audit 10b", "6k1/5pp1/4p2n/6Pp/8/5Q2/5PPP/6K1 w - - 0 1", "trappedPiece"),
    ("audit 10c", "rn2r1k1/p5pp/8/8/4B3/8/6PP/4K3 w - - 0 1", "trappedPiece"),
    # Audit 21: a rook and knight ending with the king walking up the board: no queen, no
    # attack.
    ("audit 21", "r1b5/8/4k2N/pp6/8/8/6PP/3R2K1 w - - 0 1", "exposedKing"),
]


@pytest.mark.parametrize(("case", "fen", "name"), NOT_HELD, ids=[row[0] for row in NOT_HELD])
def test_a_feature_that_wins_nothing_is_not_named(case: str, fen: str, name: str) -> None:
    assert name not in position_names(fen), case


# --- the mate distance -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mate_in", "tagged"),
    [
        # The mover is the one getting mated: nothing about this move is a mate for them.
        (-3, set()),
        (5, {"mate", "mateIn5"}),
        # Beyond five only the mate tag.
        (7, {"mate"}),
    ],
)
def test_the_mate_distance_is_the_movers_and_named_up_to_five(mate_in: int, tagged: set) -> None:
    names = tag_names(STEADY, "h2h3", mate_in=mate_in)
    assert {name for name in names if name.startswith("mate")} == tagged


# --- the two entry points ---------------------------------------------------------------


def test_position_with_a_tactic_reports_it() -> None:
    # White to move, Nd5-f6 forks the king and queen. The best move's tags come through.
    names = position_names("4q1k1/8/8/3N4/8/8/8/7K w - - 0 1", pv=["d5f6"])
    assert "fork" in names
    assert "check" in names


def test_quiet_position_reports_no_tactic() -> None:
    assert not TACTICAL.intersection(position_names(STEADY))
    assert not TACTICAL.intersection(position_names(STEADY, pv=["d2d3", "e7e5"]))


@pytest.mark.parametrize(("fen", "uci"), QUIET_MOVES)
def test_quiet_moves_produce_no_tactical_tags(fen: str, uci: str) -> None:
    names = tag_names(fen, uci)
    assert "quietMove" in names
    assert not TACTICAL.intersection(names), f"{uci} in {fen} invented {names}"


def test_illegal_move_returns_no_tags() -> None:
    # Documented behaviour: an illegal move is tagged as nothing rather than raising, because
    # the caller feeds this from a vision pipeline that can misread a board.
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert tags_for_move(chess.Board(start), chess.Move.from_uci("e2e5")) == []
    assert tags_for_move(chess.Board(start), chess.Move.from_uci("e7e5")) == []
    assert tags_for_move(chess.Board(start), chess.Move.null()) == []


def test_position_ignores_an_illegal_principal_variation() -> None:
    start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert tags_for_position(chess.Board(start), pv=[chess.Move.from_uci("e2e5")]) == []


def test_tags_are_deduplicated_and_merge_their_squares() -> None:
    # Two loose knights, both hanging after the pawn push: one tag, both squares.
    fen = "4k3/8/3n1n2/8/4P3/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    tags = tags_for_move(board, chess.Move.from_uci("e4e5"))
    names = [tag.name for tag in tags]
    assert len(names) == len(set(names))
    hanging = next(tag for tag in tags if tag.name == "hangingPiece")
    assert hanging.squares == ["d6", "f6"]
    for tag in tags:
        assert len(tag.squares) == len(set(tag.squares))


def test_every_square_is_a_real_square_name() -> None:
    cases = [
        (SMOTHERED, "b3g8", ["f8g8", "h6f7"], 2),
        (DISCOVERED_CHECK, "e4d6", [], None),
        (EN_PASSANT, "e5f6", [], None),
        (RESCUE, "c1b2", [], None),
    ]
    for fen, uci, pv, mate_in in cases:
        board = chess.Board(fen)
        moves = [chess.Move.from_uci(x) for x in pv]
        tags = tags_for_move(board, chess.Move.from_uci(uci), pv=moves, mate_in=mate_in)
        assert tags
        for tag in tags:
            assert tag.squares
            for square in tag.squares:
                assert square in chess.SQUARE_NAMES
            if tag.move is not None:
                assert chess.Move.from_uci(tag.move) in chess.Board(fen).legal_moves


def test_tags_come_out_in_the_documented_order() -> None:
    board = chess.Board(SMOTHERED)
    pv = [chess.Move.from_uci("f8g8"), chess.Move.from_uci("h6f7")]
    tags = tags_for_move(board, chess.Move.from_uci("b3g8"), pv=pv, mate_in=2)
    indexes = [TAG_ORDER.index(tag.name) for tag in tags]
    assert indexes == sorted(indexes)
    assert len(tags) > 4


def test_every_emitted_name_is_a_known_tag() -> None:
    fens = [fen for fen, _ in QUIET_MOVES] + [
        SMOTHERED,
        RESCUE,
        "3r2k1/8/8/3r4/8/8/8/2KR4 w - - 0 1",
    ]
    for fen in dict.fromkeys(fens):  # each position once
        board = chess.Board(fen)
        for move in board.legal_moves:
            for tag in tags_for_move(board, move):
                assert tag.name in TAG_ORDER
    # None of these positions mates on the move: the patterns are known by their list.
    assert set(MATE_PATTERNS) <= set(TAG_ORDER)
