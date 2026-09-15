"""Motif tagger tests.

Every position here is one a coach could talk through, and every expectation is what a coach
would say about it, worked out from the board. The negative tests matter most: a tagger that
invents a pin makes the coach say something untrue, which is worse than saying nothing.
"""

from collections.abc import Sequence

import chess
import pytest

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
TACTICAL = set(TAG_ORDER) - QUIET_GROUP

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


def tag_names(fen: str, uci: str, pv: Sequence[str] = (), mate_in: int | None = None) -> list[str]:
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    # Guard every negative assertion below: an illegal move tags as nothing, which would make
    # "this tag is absent" true for the wrong reason.
    assert move in board.legal_moves, f"{uci} is not legal in {fen}"
    tags = tags_for_move(board, move, pv=[chess.Move.from_uci(x) for x in pv], mate_in=mate_in)
    assert board.fen() == fen, "tags_for_move mutated the board it was given"
    return [tag.name for tag in tags]


def mate_names(fen: str, uci: str, pv: Sequence[str] = ()) -> list[str]:
    """Tags for a move that really is checkmate on the board, not just claimed to be."""
    board = chess.Board(fen)
    for played in [chess.Move.from_uci(uci), *[chess.Move.from_uci(x) for x in pv]]:
        assert played in board.legal_moves, f"{played} is not legal in {board.fen()}"
        board.push(played)
    assert board.is_checkmate(), f"{fen} then {uci} is not checkmate"
    return tag_names(fen, uci, pv=pv, mate_in=(len(pv) // 2) + 1)


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


# --- group A: what the move is ---------------------------------------------------------


def test_capture_and_en_passant_point_at_the_pawn_that_disappears() -> None:
    # Black has just played ...f7-f5 past the white pawn on e5, so exf6 e.p. is available.
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    names = tag_names(fen, "e5f6")
    assert "capture" in names
    assert "enPassant" in names
    # The captured pawn is on f5, not on the square the capturing pawn lands on.
    assert squares_of(fen, "e5f6", "enPassant") == ["e5", "f6", "f5"]
    assert not TACTICAL.intersection(names)


def test_castling_names_both_rook_squares() -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    assert squares_of(fen, "e1g1", "castling") == ["e1", "g1", "h1", "f1"]
    assert "quietMove" in tag_names(fen, "e1g1")


def test_promotion_and_underpromotion() -> None:
    fen = "8/P7/8/8/8/8/8/K5k1 w - - 0 1"
    queening = tag_names(fen, "a7a8q")
    assert "promotion" in queening
    assert "underPromotion" not in queening
    assert "advancedPawn" in queening
    knighting = tag_names(fen, "a7a8n")
    assert "underPromotion" in knighting


def test_advanced_pawn_starts_on_the_sixth_rank() -> None:
    # b5-b6 reaches the sixth; a bare passer on b5 has not.
    assert "advancedPawn" in tag_names("6k1/8/8/1P6/8/8/8/6K1 w - - 0 1", "b5b6")
    assert "advancedPawn" not in tag_names("6k1/8/8/8/1P6/8/8/6K1 w - - 0 1", "b4b5")
    # From black's side the sixth rank is rank 3.
    assert "advancedPawn" in tag_names("6k1/8/8/8/8/8/1p6/6K1 b - - 0 1", "b2b1q")


def test_check_evidence_is_the_checker_and_the_king() -> None:
    # White rook lands on a8; the black king on g8 is shut in by f7/g7/h7.
    fen = "6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1"
    assert squares_of(fen, "a1a8", "check") == ["a8", "g8"]


def test_defensive_move_only_when_the_piece_was_really_in_danger() -> None:
    # Bc1-b2 defends the rook on a1, which the black queen on a8 was winning outright.
    rescue = "qr4k1/1p6/8/8/8/8/8/R1B3K1 w - - 0 1"
    assert "defensiveMove" in tag_names(rescue, "c1b2")
    assert squares_of(rescue, "c1b2", "defensiveMove") == ["b2", "a1"]
    # d2-d3 props up e4, but e4 was already defended by the knight on c3 and only attacked
    # once. A second guard on a pawn that was holding is not a rescue.
    steady = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
    assert "defensiveMove" not in tag_names(steady, "d2d3")


# --- group B: what the move creates -----------------------------------------------------


def test_knight_fork_of_king_and_queen() -> None:
    # Nd5-f6 checks the king on g8 and hits the queen on e8 at the same time.
    fen = "4q1k1/8/8/3N4/8/8/8/7K w - - 0 1"
    names = tag_names(fen, "d5f6")
    assert "check" in names
    assert "fork" in names
    assert squares_of(fen, "d5f6", "fork") == ["f6", "e8", "g8"]


def test_royal_fork_hits_king_queen_and_rook() -> None:
    # Nh6-f7+ forks the king on h8, the queen on d8 and the rook on d6.
    fen = "3q3k/8/3r3N/8/8/8/8/K7 w - - 0 1"
    assert "fork" in tag_names(fen, "h6f7")
    assert squares_of(fen, "h6f7", "fork") == ["f7", "d6", "d8", "h8"]


def test_pawn_fork_of_two_knights() -> None:
    # e4-e5 attacks both knights; a pawn taking a knight is a win either way.
    fen = "4k3/8/3n1n2/8/4P3/8/8/4K3 w - - 0 1"
    assert "fork" in tag_names(fen, "e4e5")
    assert squares_of(fen, "e4e5", "fork") == ["e5", "d6", "f6"]


def test_a_pawn_push_hitting_two_defended_pawns_is_not_a_fork() -> None:
    # e4-e5 attacks the d6 and f6 pawns, but both are held by the pawn behind them, so
    # nothing is won and there is nothing to point at.
    fen = "4k3/2p1pp2/3p1p2/8/4P3/8/8/4K3 w - - 0 1"
    assert "fork" not in tag_names(fen, "e4e5")


def test_absolute_pin_against_the_king() -> None:
    # Rd1-e1 lines up on the knight on e5 with the black king behind it on e8.
    fen = "4k3/8/8/4n3/8/8/8/3R2K1 w - - 0 1"
    names = tag_names(fen, "d1e1")
    assert "pin" in names
    assert "absolutePin" in names
    assert squares_of(fen, "d1e1", "absolutePin") == ["e1", "e5", "e8"]


def test_relative_pin_against_the_queen() -> None:
    # Bc1-g5 pins the knight on f6 to the queen on d8. Real, but not absolute. From c1 the
    # bishop was not on that diagonal, so the move is what creates the pin.
    fen = "3q1rk1/ppp2ppp/5n2/8/8/8/PPP2PPP/2BQK2R w K - 0 1"
    names = tag_names(fen, "c1g5")
    assert "pin" in names
    assert "absolutePin" not in names
    assert squares_of(fen, "c1g5", "pin") == ["g5", "f6", "d8"]
    # The same bishop already on h4 is already pinning f6 to d8 through g5, so stepping to g5
    # creates nothing: the pin belongs to the position, not to the move.
    standing = "3q1rk1/ppp2ppp/5n2/8/7B/8/PPP2PPP/3QK2R w K - 0 1"
    assert "pin" not in tag_names(standing, "h4g5")
    assert "pin" in position_names(standing)


def test_a_pawn_shielding_a_defended_piece_is_not_a_pin() -> None:
    # The Italian bishop on c4 looks through f7 to the knight on g8, but g8 is held by the
    # rook on h8, so taking there is a bishop for a knight. Castling has nothing to do with
    # the bishop either way.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    assert "pin" not in tag_names(fen, "e1g1")
    assert "pin" not in position_names(fen)


def test_a_move_does_not_inherit_a_pin_that_was_already_there() -> None:
    # The rook on e1 already pins the knight on e5 to the king. Shuffling the king on the
    # other wing did not create that, so it is not part of what the move did; the position
    # still reports it.
    fen = "4k3/8/8/4n3/8/8/8/4R1K1 w - - 0 1"
    assert "pin" not in tag_names(fen, "g1h1")
    assert "absolutePin" not in tag_names(fen, "g1h1")
    assert "absolutePin" in position_names(fen)


def test_skewer_drives_the_queen_off_a_loose_rook() -> None:
    # Rd1-e1 attacks the queen on e5; behind her sits the rook on e8, which only she guards.
    # The white rook is held by the king on f1, so she cannot just take it.
    fen = "4r1k1/8/8/4q3/8/8/8/3R1K2 w - - 0 1"
    assert "skewer" in tag_names(fen, "d1e1")
    assert squares_of(fen, "d1e1", "skewer") == ["e1", "e5", "e8"]


def test_no_skewer_when_the_front_piece_can_simply_take_the_rook() -> None:
    # Same picture with the white king on g1: the rook on e1 is loose, so Qxe1 answers it.
    fen = "4r1k1/8/8/4q3/8/8/8/3R2K1 w - - 0 1"
    assert "skewer" not in tag_names(fen, "d1e1")


def test_discovered_attack_opens_a_diagonal() -> None:
    # Nd4-f5 steps off the long diagonal and the bishop on b2 hits the queen on h8.
    fen = "7q/8/8/8/3N4/8/1B6/K5k1 w - - 0 1"
    names = tag_names(fen, "d4f5")
    assert "discoveredAttack" in names
    assert squares_of(fen, "d4f5", "discoveredAttack") == ["d4", "b2", "h8"]


def test_discovered_check_is_a_double_check_here() -> None:
    # Ne4-d6+ checks from d6 and uncovers the rook on e1 at the same time.
    fen = "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1"
    names = tag_names(fen, "e4d6")
    assert "check" in names
    assert "doubleCheck" in names
    assert "discoveredAttack" in names
    assert squares_of(fen, "e4d6", "check") == ["e1", "d6", "e8"]


def test_hanging_piece_is_the_rook_left_en_prise() -> None:
    # Bc1-f4 rakes the long diagonal onto the rook on b8, which nothing defends.
    fen = "1r4k1/5ppp/8/8/8/8/5PPP/2B3K1 w - - 0 1"
    names = tag_names(fen, "c1f4")
    assert "hangingPiece" in names
    assert squares_of(fen, "c1f4", "hangingPiece") == ["b8"]


def test_loose_pawns_are_not_hanging_pieces() -> None:
    # Nf3 attacks the undefended e5 pawn after 1.e4 e5. Calling that a hanging piece would
    # have the coach warning about a pawn on move two of every open game.
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    assert "hangingPiece" not in tag_names(fen, "g1f3")


def test_x_ray_through_a_rook_onto_a_loose_rook() -> None:
    # Rd1 bears down the d-file through the rook on d5 onto the one on d8. Equal values, so
    # it is neither a pin nor a skewer; the white rook is held by the king on c1.
    assert position_names("3r2k1/8/8/3r4/8/8/8/2KR4 w - - 0 1") == ["xRayAttack"]


def test_interference_cuts_a_defender_off_its_charge() -> None:
    # Bg3-d6 sits down between the rook on d8 and the knight on d5, which the e4 pawn wins.
    fen = "3r2k1/8/8/3n4/4P3/6B1/8/6K1 w - - 0 1"
    names = tag_names(fen, "g3d6")
    assert "interference" in names
    assert squares_of(fen, "g3d6", "interference") == ["d6", "d8", "d5"]
    assert "hangingPiece" in names


def test_trapped_piece_has_nowhere_safe_to_go() -> None:
    # The queen on a8 is boxed in by her own rook and pawn; Bc1-b2 defends the rook on a1 so
    # that Qxa1 stops working, and then every square on the a-file is covered.
    fen = "qr4k1/1p6/8/8/8/8/8/R1B3K1 w - - 0 1"
    names = tag_names(fen, "c1b2")
    assert "trappedPiece" in names
    assert squares_of(fen, "c1b2", "trappedPiece")[0] == "a8"
    # Before the bishop came to b2 she was not trapped: Qxa1 simply won a rook.
    assert "trappedPiece" not in position_names(fen)


def test_attacking_f2_f7() -> None:
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    assert "attackingF2F7" in tag_names(fen, "f1c4")
    assert squares_of(fen, "f1c4", "attackingF2F7") == ["c4", "f7"]
    # And the mirror image for black: Bf8-c5 takes aim at f2.
    black = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    assert "attackingF2F7" in tag_names(black, "f8c5")


def test_exposed_king_needs_a_stripped_shelter_and_real_force() -> None:
    # Black's king on g8 has no pawn in front of it and white has queen, rook and bishop
    # pointing at the ring.
    assert position_names("r5k1/ppp5/8/7Q/2B5/8/8/5RK1 w - - 0 1") == ["exposedKing"]
    # The same bare king with only a rook and knight around is an endgame, not an attack.
    assert "exposedKing" not in position_names("r5k1/ppp5/8/8/8/5N2/8/5RK1 w - - 0 1")
    # And a castled king behind three pawns is never exposed.
    steady = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
    assert "exposedKing" not in position_names(steady)


# --- group C: what the line does --------------------------------------------------------


def test_queen_sacrifice_leading_to_smothered_mate() -> None:
    # Philidor's legacy: Qg8+ forces Rxg8 and Nf7 is mate, the knight smothered by g7, h7
    # and the rook that was dragged to g8.
    fen = "5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1"
    names = mate_names(fen, "b3g8", pv=["f8g8", "h6f7"])
    assert "sacrifice" in names
    assert "mate" in names
    assert "mateIn2" in names
    assert "smotheredMate" in names
    # The rook on f8 was the only guard of f7 and the queen check pulled it away.
    assert "deflection" in names
    assert squares_of(fen, "b3g8", "deflection", pv=["f8g8", "h6f7"], mate_in=2) == [
        "f8",
        "g8",
        "f7",
    ]


def test_an_even_trade_is_not_a_sacrifice() -> None:
    # Rxe8 Rxe8 is a straight swap of rooks: the material dips and comes straight back.
    fen = "4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1"
    assert "sacrifice" not in tag_names(fen, "e1e8", pv=["g8f8"])


def test_a_pawn_gambit_is_not_a_sacrifice() -> None:
    # 1.e4 e5 2.f4 exf4 gives a pawn. One point is a gambit, not a sacrifice.
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    assert "sacrifice" not in tag_names(fen, "f2f4", pv=["e5f4", "g1f3"])


def test_capturing_defender_takes_the_guard_of_a_square_the_line_uses() -> None:
    # Bxc6 removes the knight that held e5; after dxc6 the knight takes on e5.
    fen = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
    names = tag_names(fen, "b5c6", pv=["d7c6", "f3e5"])
    assert "capturingDefender" in names
    assert squares_of(fen, "b5c6", "capturingDefender", pv=["d7c6", "f3e5"]) == ["c6", "e5"]


def test_attraction_drags_the_king_onto_a_square_the_knight_hits() -> None:
    # The Greek gift: Bxh7+ Kxh7 and Ng5+ forks the king on the square it was pulled to.
    fen = "5rk1/5ppp/8/8/8/3B1N2/8/3Q2K1 w - - 0 1"
    names = tag_names(fen, "d3h7", pv=["g8h7", "f3g5"])
    assert "attraction" in names
    assert squares_of(fen, "d3h7", "attraction", pv=["g8h7", "f3g5"]) == ["g8", "h7", "g5"]


def test_clearance_vacates_a_file_with_tempo() -> None:
    # Nxe6 takes a pawn and steps off d4, and the rook then swings down the d-file it opened.
    fen = "4k3/p7/4p3/8/3N4/8/8/3R2K1 w - - 0 1"
    names = tag_names(fen, "d4e6", pv=["a7a6", "d1d8"])
    assert "clearance" in names
    assert squares_of(fen, "d4e6", "clearance", pv=["a7a6", "d1d8"]) == ["d4", "d1", "d8"]


def test_mate_in_counts_the_movers_own_moves() -> None:
    fen = "6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1"
    # Ra8 is mate on the spot, so the caller's 1 and the board agree.
    assert "mateIn1" in tag_names(fen, "a1a8", mate_in=1)
    assert "mate" in tag_names(fen, "a1a8", mate_in=1)
    # Checkmate on the board is enough on its own.
    assert "mateIn1" in tag_names(fen, "a1a8")


def test_a_negative_mate_distance_never_tags_the_mover() -> None:
    # The mover is the one getting mated. Nothing about this move is a mate for them.
    quiet = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
    names = tag_names(quiet, "h2h3", mate_in=-3)
    assert not any(name.startswith("mate") for name in names)


def test_mate_distances_beyond_five_only_get_the_mate_tag() -> None:
    quiet = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
    names = tag_names(quiet, "h2h3", mate_in=7)
    assert "mate" in names
    assert not any(name.startswith("mateIn") for name in names)
    assert "mateIn5" in tag_names(quiet, "h2h3", mate_in=5)


# --- group D: mating patterns -----------------------------------------------------------


def test_back_rank_mate() -> None:
    # The king on g8 is walled in by its own f7, g7 and h7 pawns.
    fen = "6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1"
    names = mate_names(fen, "a1a8")
    assert "backRankMate" in names
    assert set(squares_of(fen, "a1a8", "backRankMate", mate_in=1)) == {
        "g8",
        "a8",
        "f7",
        "g7",
        "h7",
    }


def test_back_rank_mate_is_not_claimed_when_the_king_has_air() -> None:
    # Same rook check, but the h-pawn has moved and Kh7 is available, so it is not mate and
    # there is no pattern to name.
    fen = "6k1/5pp1/7p/8/8/8/8/R3K3 w Q - 0 1"
    names = tag_names(fen, "a1a8")
    assert "backRankMate" not in names
    assert "mate" not in names


def test_smothered_mate_needs_every_neighbour_blocked() -> None:
    # Nf7 mates the king on h8, hemmed in by its own g7, h7 and the rook on g8.
    assert "smotheredMate" in mate_names("6rk/6pp/8/6N1/8/8/8/4K3 w - - 0 1", "g5f7")


def test_arabian_mate() -> None:
    # Rh7 mates the king in the corner, held by the knight on f6 which also covers g8.
    names = mate_names("7k/R7/5N2/8/8/8/8/6K1 w - - 0 1", "a7h7")
    assert "arabianMate" in names
    assert "hookMate" not in names
    assert "vukovicMate" not in names


def test_hook_mate() -> None:
    # Rg7 checks beside the king on g8, the knight on f5 holds the rook and the e4 pawn holds
    # the knight; the king's own rooks on f8 and h8 take the last two squares away.
    names = mate_names("5rkr/R7/8/5N2/4P3/8/8/1K6 w - - 0 1", "a7g7")
    assert "hookMate" in names
    assert "arabianMate" not in names


def test_kill_box_mate() -> None:
    # Rh7 checks next to the king on h8; the queen on g6 is a knight's move from the king,
    # holds the rook and covers g7 and g8. The 3x3 box has no way out.
    names = mate_names("7k/R7/6Q1/8/8/8/8/1K6 w - - 0 1", "a7h7")
    assert "killBoxMate" in names
    assert "arabianMate" not in names


def test_anastasia_mate() -> None:
    # The king on h7 is pinned to the edge by its own g7 pawn; the knight on e7 covers g6 and
    # g8, and Rh4 comes down the h-file.
    names = mate_names("8/4N1pk/8/8/R7/8/8/1K6 w - - 0 1", "a4h4")
    assert "anastasiaMate" in names


def test_vukovic_mate() -> None:
    # Rh7 mates the king on h6 held by the g6 pawn rather than by the knight, while the
    # knight on g3 takes h5 away. That support is what separates it from the Arabian mate.
    names = mate_names("3B4/R7/6Pk/5P2/8/6N1/8/1K6 w - - 0 1", "a7h7")
    assert "vukovicMate" in names
    assert "arabianMate" not in names
    assert "hookMate" not in names


def test_bodens_mate_has_crossing_diagonals() -> None:
    # Ba6 checks the king on c8 along a6-c8 while the bishop on f4 covers b8 along f4-b8.
    # The diagonals cross, and the king is walled in by its own d7 pawn and d8 rook.
    names = mate_names("2kr4/p2p4/8/8/2B2B2/8/8/6K1 w - - 0 1", "c4a6")
    assert "bodenMate" in names
    assert "doubleBishopMate" not in names


def test_double_bishop_mate_has_parallel_diagonals() -> None:
    # Bb2 checks the king on h8 along a1-h8 and the bishop on d3 covers h7 along b1-h7. The
    # two diagonals run the same way, which is what makes this the double bishop mate. Black's
    # own bishop on g8 takes the last square and is on the wrong colour to interpose.
    names = mate_names("6bk/8/8/8/8/B2B4/8/7K w - - 0 1", "a3b2")
    assert "doubleBishopMate" in names
    assert "bodenMate" not in names


def test_dovetail_mate() -> None:
    # Qe6 mates the king on d7 from a diagonal, held by the f5 pawn, with the king's own c7
    # pawn and d8 rook blocking the two squares beside it: the dove's tail.
    names = tag_names("3r4/2pk4/B7/5P2/8/4Q3/8/7K w - - 0 1", "e3e6", mate_in=1)
    assert "dovetailMate" in names


def test_no_mating_pattern_without_mate() -> None:
    # A check that is not mate must produce no pattern at all.
    names = tag_names("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "e4d6")
    assert not any(name.endswith("Mate") for name in names)


# --- the two entry points ---------------------------------------------------------------


def test_position_with_a_tactic_reports_it() -> None:
    # White to move, Nd5-f6 forks the king and queen. The best move's tags come through.
    names = position_names("4q1k1/8/8/3N4/8/8/8/7K w - - 0 1", pv=["d5f6"])
    assert "fork" in names
    assert "check" in names


def test_quiet_position_reports_no_tactic() -> None:
    fen = "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 8"
    assert not TACTICAL.intersection(position_names(fen))
    assert not TACTICAL.intersection(position_names(fen, pv=["d2d3", "e7e5"]))


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
        ("5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1", "b3g8", ["f8g8", "h6f7"], 2),
        ("4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1", "e4d6", [], None),
        ("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "e5f6", [], None),
        ("qr4k1/1p6/8/8/8/8/8/R1B3K1 w - - 0 1", "c1b2", [], None),
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


def test_neither_function_mutates_the_board() -> None:
    fen = "5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    pv = [chess.Move.from_uci("f8g8"), chess.Move.from_uci("h6f7")]
    tags_for_move(board, chess.Move.from_uci("b3g8"), pv=pv, mate_in=2)
    assert board.fen() == fen
    assert board.move_stack == []
    tags_for_position(board, pv=[chess.Move.from_uci("b3g8"), *pv], mate_in=2)
    assert board.fen() == fen
    assert board.move_stack == []


def test_tags_come_out_in_the_documented_order() -> None:
    fen = "5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    pv = [chess.Move.from_uci("f8g8"), chess.Move.from_uci("h6f7")]
    tags = tags_for_move(board, chess.Move.from_uci("b3g8"), pv=pv, mate_in=2)
    indexes = [TAG_ORDER.index(tag.name) for tag in tags]
    assert indexes == sorted(indexes)
    assert len(tags) > 4


def test_every_emitted_name_is_a_known_tag() -> None:
    fens = [fen for fen, _ in QUIET_MOVES] + [
        "5r1k/6pp/7N/8/8/1Q6/8/6K1 w - - 0 1",
        "qr4k1/1p6/8/8/8/8/8/R1B3K1 w - - 0 1",
        "3r2k1/8/8/3r4/8/8/8/2KR4 w - - 0 1",
    ]
    for fen in fens:
        board = chess.Board(fen)
        for move in board.legal_moves:
            for tag in tags_for_move(board, move):
                assert tag.name in TAG_ORDER


def test_a_trade_deep_in_the_line_is_not_this_move_s_sacrifice() -> None:
    """The opponent captures first and the material comes back: an ordinary line, not a gift.

    Measured over the whole variation instead of over the move, 2. Nf3 came out as a
    sacrifice in the game of 2026-09-14 20:06, which is what this pins down.
    """
    fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    deep_trade = ["b8c6", "f1b5", "g8f6", "b5c6", "d7c6", "f3e5"]
    assert "sacrifice" not in tag_names(fen, "g1f3", pv=deep_trade)
    # The opponent taking a knight and giving it straight back is still not a sacrifice.
    assert "sacrifice" not in tag_names(fen, "g1f3", pv=["b8c6", "f3e5", "c6e5", "d2d4"])


def test_a_piece_handed_to_the_reply_is_a_sacrifice() -> None:
    """The Greek gift: a bishop for a pawn on move one, paid for by the attack that follows."""
    fen = "rnbq1rk1/ppp2ppp/3b1n2/8/3P4/3B1N2/PPP2PPP/RNBQ1RK1 w - - 0 1"
    names = tag_names(fen, "d3h7", pv=["g8h7", "f3g5", "h7g6"], mate_in=4)
    assert "sacrifice" in names and "capture" in names and "check" in names
