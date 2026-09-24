"""Mating-pattern tests.

Each pattern gets a position that has its shape and one that nearly does. The positives are
final positions of Lichess puzzles carrying that theme (the id is in the comment); the
negatives are the audit's false names of 2026-09-23 where there is one, and otherwise the
nearest shape of another pattern. A wrong name teaches the player the wrong picture, so the
negatives matter most.
"""

import chess
import pytest

from sidecoach.analysis.mates import MATE_PATTERNS, mating_pattern_tags


def mate_tags(fen: str, san: str) -> dict[str, list[str]]:
    """Pattern name -> evidence squares, for a move that really is checkmate."""
    board = chess.Board(fen)
    move = board.parse_san(san)
    board.push(move)
    assert board.is_checkmate(), f"{san} does not mate in {fen}"
    after = board.fen()
    tags = mating_pattern_tags(board, move)
    assert board.fen() == after, "mating_pattern_tags mutated the board it was given"
    return {tag.name: tag.squares for tag in tags}


# (pattern, FEN before the mating move, the mating move, the evidence squares)
POSITIVES = [
    # GwU59: the queen takes on the back rank, the king shut in by its own g7 and h7.
    ("backRankMate", "Q1q4k/6pp/8/4p3/1p6/6PK/7P/5n2 w - - 1 37", "Qxc8#", None),
    # ytp8R: the knight mates a king whose every square holds its own men.
    ("smotheredMate", "knR5/pn6/1p6/1N2p1p1/4q3/P7/5PPP/1R4K1 w - - 1 32", "Nc7#", None),
    # VeMvr: the rook mates beside the cornered king, held by the knight on f6.
    ("arabianMate", "5Rrk/8/1p3N1p/3p3P/4b3/3n4/6p1/6K1 w - - 8 54", "Rxg8#", ["h8", "g8", "f6"]),
    # OCiND: down the h-file, g7 blocked by the king's own pawn, the knight on e7 on g8/g6.
    ("anastasiaMate", "4r3/R3Npk1/6p1/1P6/8/8/PP2nPPK/5R2 b - - 0 29", "Rh8#", None),
    # The queen version the old tagger missed: Qh4 down the file, the knight still needed.
    ("anastasiaMate", "8/4N1pk/8/8/Q7/8/8/1K6 w - - 0 1", "Qh4#", ["h7", "h4", "e7", "g7"]),
    # qJLw6: rook beside the king, knight holding it, pawn holding the knight, e6 pawn.
    ("hookMate", "5rk1/7p/4p1p1/1p1nP1P1/1P1pKQ2/3P3R/3R4/8 b - - 0 37", "Rxf4#", None),
    # OSqye: rook h8 and queen f6 on opposite corners of the box f6-h8.
    ("killBoxMate", "4rk2/ppp2r2/1n3Q1R/2p5/4q3/4p3/PP5P/6K1 w - - 4 40", "Rh8#", None),
    # oJdB1: the rook beside the king on the edge, backed by a pawn, the knight on f4.
    ("vukovicMate", "6r1/4R3/k2p4/p2P3p/NPP2n1K/8/P4P1P/1R6 b - - 2 38", "Rg4#", None),
    # eN0bT: Bxa6 and the bishop on g3 cross at the king.
    ("bodenMate", "2kr4/3n4/p1p1p3/8/4P3/1Pb3B1/1PP1B1rP/1K5R w - - 0 26", "Bxa6#", None),
    # qy80M: two bishops on neighbouring diagonals, the king's own bishop on h2.
    ("doubleBishopMate", "8/b7/P1bP4/3B2k1/8/2P5/7B/7K b - - 2 59", "Bxd5#", None),
    # GgkHn: Qe5 diagonal to the king on d6; c6 and d7, out of its reach, hold its own men.
    ("dovetailMate", "7k/3Q2p1/2PK1p1p/1q6/6P1/4PP2/8/8 b - - 15 62", "Qe5#", None),
    # jj9Rd: the queen straight in front, the two squares behind the king its own men.
    ("swallowstailMate", "8/1b3k2/1p5p/p3P1p1/2q5/4K1P1/1P1Q1P2/4R3 b - - 0 37", "Qe4#", None),
    # un2Lg: check down the b-file, the king's own pawns on a3 and c3 on its shoulders.
    ("epauletteMate", "R1Q5/p7/1p6/1k6/6p1/PKP5/3r3P/3r4 b - - 4 55", "Rb1#", None),
    # zjsBG: queen f7 and rook h7 on the two near corners of the king's rank below.
    ("triangleMate", "6k1/3Q3R/2p3r1/pp4p1/4p3/P1q1P3/6PP/7K w - - 1 32", "Qf7#", None),
    # hvV5X: two rooks and the king in one 2x2 square.
    ("blindSwineMate", "8/8/2r3pk/pKPN1pb1/P6p/1P1R1R1P/2r2P2/8 b - - 4 39", "R2xc5#", None),
    # HzBD1: the knight mates the cornered king that the rook on g5 shuts in.
    ("cornerMate", "1R6/p1p3p1/k1p4p/2P3r1/4R3/P2n4/P4P1P/7K b - - 0 32", "Nxf2#", None),
    # 6CeKP: the rook beside the king, defended by the bishop on e4.
    ("operaMate", "1RQ5/5pk1/6p1/P6p/4bb2/6PP/7K/4r3 b - - 0 40", "Rh1#", None),
    # g8oJt: the rook mates from h5, the bishop on f2 takes g1.
    ("pillsburysMate", "4R3/pk5p/6p1/1p1r4/8/2P3P1/PP3bP1/R6K b - - 0 36", "Rh5#", None),
    # Nbxh2: the bishop mates, the rook on g2 takes the g-file.
    ("morphysMate", "8/p7/1b5k/3R4/7K/5N1P/6r1/8 b - - 3 40", "Bf2#", None),
    # V2tGu: the bishop mates from f4, the queen a knight's move away takes the rest.
    ("balestraMate", "8/p7/2Q1p2k/1p3pb1/3P1P2/PqP5/1P6/1BK5 b - - 0 35", "Bxf4#", None),
]


@pytest.mark.parametrize(("pattern", "fen", "san", "squares"), POSITIVES)
def test_pattern_is_named(pattern: str, fen: str, san: str, squares: list[str] | None) -> None:
    tags = mate_tags(fen, san)
    assert pattern in tags
    if squares is not None:
        assert tags[pattern] == squares


# (pattern that must NOT be named, FEN before the mating move, the mating move)
NEGATIVES = [
    # The king has luft on h7; a bishop covers it, not its own pawn.
    ("backRankMate", "6k1/5pp1/7p/8/8/3B4/8/R5K1 w - - 0 1", "Ra8#"),
    # A knight mates in the corner, but a rook, not the king's own men, takes a square.
    ("smotheredMate", "1R6/p1p3p1/k1p4p/2P3r1/4R3/P2n4/P4P1P/7K b - - 0 32", "Nxf2#"),
    # F2TzY: the rook's knight holds h7 but a second knight takes g8.
    ("arabianMate", "r4r1k/2R5/1p2p2N/p3n1Np/7P/8/PP4P1/5bK1 w - - 2 32", "Rh7#"),
    # Down the h-file with the knight on e7, but g7 is empty and a bishop covers it.
    ("anastasiaMate", "8/4N2k/8/8/R7/8/1B6/1K6 w - - 0 1", "Rh4#"),
    # Audit 25: an Arabian mate with no pawn of the king's to hook onto.
    ("hookMate", "7k/R7/5N2/4P3/8/8/8/6K1 w - - 0 1", "Rh7#"),
    # Audit 26: a back-rank mate; the queen on e7 covers nothing the rook does not.
    ("killBoxMate", "R4rk1/4Qppp/8/8/8/8/5PPP/6K1 w - - 0 1", "Rxf8#"),
    # Audit 24: the knight on e6 does nothing; the king on f7 holds g7 and g8.
    ("vukovicMate", "7k/5K1p/4N3/8/8/8/8/6R1 w - - 0 1", "Rg8#"),
    # Audit 23: the bishop on e5 only eyes the king's own pawn; the knight and rook mate.
    ("bodenMate", "2kr4/2p5/2N5/4B3/2B5/8/5PPP/3R2K1 w - - 0 1", "Ba6#"),
    ("doubleBishopMate", "2kr4/2p5/2N5/4B3/2B5/8/5PPP/3R2K1 w - - 0 1", "Ba6#"),
    # Audit 22: d5 and e6 are empty; the rook and the knights take them, not the king's men.
    ("dovetailMate", "8/2N4N/8/4kp2/4p3/6P1/7P/3R1QK1 w - - 0 1", "Qf4#"),
    # A dovetail: the queen is diagonal to the king, not straight in front of it.
    ("swallowstailMate", "7k/3Q2p1/2PK1p1p/1q6/6P1/4PP2/8/8 b - - 15 62", "Qe5#"),
    # A back-rank mate along the rank: the king in the corner has one shoulder, not two.
    ("epauletteMate", "Q1q4k/6pp/8/4p3/1p6/6PK/7P/5n2 w - - 1 37", "Qxc8#"),
    # A dovetail: the queen mates diagonally with no rook beside it.
    ("triangleMate", "7k/3Q2p1/2PK1p1p/1q6/6P1/4PP2/8/8 b - - 15 62", "Qe5#"),
    # An Arabian mate: a rook beside the king, but backed by a knight, not a second rook.
    ("blindSwineMate", "5Rrk/8/1p3N1p/3p3P/4b3/3n4/6p1/6K1 w - - 8 54", "Rxg8#"),
    # A smothered mate in the corner: no rook or queen is needed.
    ("cornerMate", "knR5/pn6/1p6/1N2p1p1/4q3/P7/5PPP/1R4K1 w - - 1 32", "Nc7#"),
    # A Pillsbury: the bishop takes squares but does not defend the rook.
    ("operaMate", "4R3/pk5p/6p1/1p1r4/8/2P3P1/PP3bP1/R6K b - - 0 36", "Rh5#"),
    # An Opera: the only bishop that matters defends the rook.
    ("pillsburysMate", "1RQ5/5pk1/6p1/P6p/4bb2/6PP/7K/4r3 b - - 0 40", "Rh1#"),
    # A Balestra: the bishop mates with a queen, not a rook, taking the squares.
    ("morphysMate", "8/p7/2Q1p2k/1p3pb1/3P1P2/PqP5/1P6/1BK5 b - - 0 35", "Bxf4#"),
    # A Morphy: the bishop mates with a rook, not a queen, taking the squares.
    ("balestraMate", "8/p7/1b5k/3R4/7K/5N1P/6r1/8 b - - 3 40", "Bf2#"),
]


@pytest.mark.parametrize(("pattern", "fen", "san"), NEGATIVES)
def test_pattern_is_not_named(pattern: str, fen: str, san: str) -> None:
    assert pattern not in mate_tags(fen, san)


def test_every_pattern_has_a_positive_and_a_negative() -> None:
    assert {p for p, *_ in POSITIVES} == set(MATE_PATTERNS)
    assert {p for p, *_ in NEGATIVES} == set(MATE_PATTERNS)


def test_audit_positions_keep_their_true_names() -> None:
    # The audit's hook and box positions are still an Arabian and a back-rank mate.
    assert "arabianMate" in mate_tags("7k/R7/5N2/4P3/8/8/8/6K1 w - - 0 1", "Rh7#")
    assert "backRankMate" in mate_tags("R4rk1/4Qppp/8/8/8/8/5PPP/6K1 w - - 0 1", "Rxf8#")


def test_a_check_that_is_not_mate_names_nothing() -> None:
    board = chess.Board("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1")
    move = chess.Move.from_uci("e4d6")
    board.push(move)
    assert mating_pattern_tags(board, move) == []


def test_a_needed_second_checker_must_be_named() -> None:
    # Rook g8 and knight h6 make an Arabian shape, but the bishop's check on the long
    # diagonal is what stops Rxg8: the mate is a double check, not an Arabian mate.
    # Lichess J0cYX, final position.
    after = chess.Board("r4rRk/pp5p/4n2N/1b6/3B4/8/P4PPP/6K1 b - - 0 1")
    assert after.is_checkmate()
    tags = {t.name for t in mating_pattern_tags(after, chess.Move.from_uci("g1g8"))}
    assert "arabianMate" not in tags
