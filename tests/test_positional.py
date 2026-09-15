"""Tests for the positional layer: static strategic features and the best-move idea.

Every fact these produce is deterministic geometry, so each test is a position where the
answer is unambiguous. The point is that the coach's strategic facts are as grounded and
un-fabricatable as the tactical ones.
"""

from __future__ import annotations

import chess

from sidecoach.analysis import positional as P


def kinds(facts: list[P.PositionalFact]) -> list[str]:
    return [fact.kind for fact in facts]


def test_a_pawn_with_no_enemy_pawn_ahead_is_passed() -> None:
    # White a-pawn on a5; no black pawn on the a or b files ahead of it.
    board = chess.Board("4k3/8/8/P7/8/8/8/4K3 w - - 0 1")
    assert P.is_passed(board, chess.A5)
    assert kinds(P.passed_pawns(board, chess.WHITE)) == ["passedPawn"]
    # A black pawn on b6 stops it: it can capture on a-file's promotion path.
    board = chess.Board("4k3/8/1p6/P7/8/8/8/4K3 w - - 0 1")
    assert not P.is_passed(board, chess.A5)
    assert P.passed_pawns(board, chess.WHITE) == []


def test_a_file_with_no_pawns_is_open_and_with_only_enemy_pawns_half_open() -> None:
    board = chess.Board("3rk3/8/8/8/8/8/8/3RK3 w - - 0 1")
    assert kinds(P.open_files(board, chess.WHITE)) == ["openFile"]
    board = chess.Board("3rk3/3p4/8/8/8/8/8/3RK3 w - - 0 1")
    assert kinds(P.open_files(board, chess.WHITE)) == ["halfOpenFile"]
    # A friendly pawn on the file means it is neither open nor half-open for you.
    board = chess.Board("3rk3/8/8/8/8/8/3P4/3RK3 w - - 0 1")
    assert P.open_files(board, chess.WHITE) == []


def test_a_pawn_guarded_square_no_enemy_pawn_can_reach_is_an_outpost() -> None:
    # White knight on d5, guarded by the e4 pawn; black has no c- or e-pawn to challenge it.
    board = chess.Board("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert P._is_outpost(board, chess.D5, chess.WHITE)
    assert kinds(P.outposts(board, chess.WHITE)) == ["outpost"]
    # A black c-pawn that can advance to c6 and hit d5 spoils it.
    board = chess.Board("4k3/2p5/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert not P._is_outpost(board, chess.D5, chess.WHITE)


def test_development_counts_minor_pieces_at_home() -> None:
    board = chess.Board()
    board.push_san("e4")
    facts = P.development(board, chess.BLACK)
    assert kinds(facts) == ["undeveloped"]
    assert set(facts[0].squares) == {"b8", "c8", "f8", "g8"}


def test_an_uncastled_king_with_rights_is_named() -> None:
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    assert kinds(P.uncastled_king(board, chess.WHITE)) == ["uncastled"]
    board.push_san("O-O")
    # After castling, the black king still in the centre is the one named for black.
    assert kinds(P.uncastled_king(board, chess.WHITE)) == []


def test_up_a_piece_is_a_convert_plan_and_down_a_piece_a_complicate_plan() -> None:
    up = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
    assert kinds(P.material_plan(up, chess.WHITE)) == ["convert"]
    assert kinds(P.material_plan(up, chess.BLACK)) == ["complicate"]


def test_best_move_idea_reads_the_strategic_point_of_the_move() -> None:
    italian = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(italian)
    assert (
        P.best_move_idea(board, board.parse_san("O-O"), chess.WHITE, phase="opening").kind
        == "castle"
    )
    assert (
        P.best_move_idea(board, board.parse_san("Nc3"), chess.WHITE, phase="opening").kind
        == "develop"
    )
    # A knight to a pawn-guarded square in enemy ground is an outpost move.
    board = chess.Board("4k3/8/8/8/4P3/2N5/8/4K3 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("Nd5"), chess.WHITE).kind == "outpostMove"
    # A rook onto a file with no friendly pawn takes the file.
    board = chess.Board("3rk3/8/8/8/8/8/4P3/R3K3 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("Rd1"), chess.WHITE).kind == "takeFile"
    # Pushing a passed pawn.
    board = chess.Board("4k3/8/8/P7/8/8/8/4K3 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("a6"), chess.WHITE).kind == "pushPasser"


def test_best_move_idea_rescues_a_piece_in_danger() -> None:
    # The bishop on b5 is attacked by the a6 pawn; moving it away is a rescue, not development.
    board = chess.Board("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
    idea = P.best_move_idea(board, board.parse_san("Ba4"), chess.WHITE, phase="opening")
    assert idea is not None and idea.kind == "rescue"


def test_top_features_are_ranked_deduped_and_capped() -> None:
    # A position rich in features: up material, a passed a-pawn, and open files.
    board = chess.Board("3r2k1/6pp/8/P7/8/8/6PP/3R2K1 w - - 0 1")
    top = P.top_features(board, chess.WHITE, phase="middlegame", limit=2)
    assert len(top) <= 2
    # Passed pawn outranks the open file, which outranks nothing here; no kind repeats.
    assert "passedPawn" in kinds(top)
    assert len(set(kinds(top))) == len(kinds(top))


def test_pawn_weaknesses_name_isolated_doubled_and_backward_pawns() -> None:
    # White d4 pawn, no white c- or e-pawn: isolated.
    board = chess.Board("4k3/8/8/8/3P4/8/8/4K3 w - - 0 1")
    assert kinds(P.pawn_weaknesses(board, chess.WHITE)) == ["weakPawn"]
    assert P.is_isolated(board, chess.D4)
    # An enemy isolated pawn is a target, listed first.
    board = chess.Board("4k3/8/8/3p4/8/8/8/4K3 w - - 0 1")
    facts = P.pawn_weaknesses(board, chess.WHITE)
    assert facts[0].kind == "targetPawn" and "target" in facts[0].text
    # Backward: white d3 behind c4 and e4, its advance to d4 held by the black c5 pawn.
    board = chess.Board("4k3/8/8/2p5/2P1P3/3P4/8/4K3 w - - 0 1")
    assert P.is_backward(board, chess.D3)


def test_bishop_pair_and_bad_bishop() -> None:
    # White has both bishops, black has one: the pair.
    board = chess.Board("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    assert kinds(P.bishop_pair(board, chess.WHITE)) == ["bishopPair"]
    # A light-squared bishop on f1 walled in by four of its own pawns on light squares.
    board = chess.Board("4k3/8/8/8/2P1P3/1P6/6P1/4KB2 w - - 0 1")
    assert kinds(P.bad_bishop(board, chess.WHITE)) == ["badBishop"]


def test_endgame_king_activity_and_pawn_break_and_rook_behind_passer() -> None:
    # Endgame: a passive King on the rim is named, and a King step toward the centre is the idea.
    board = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
    assert kinds(P.king_activity(board, chess.WHITE, phase="endgame")) == ["passiveKing"]
    board = chess.Board("8/8/8/8/8/8/4K3/6k1 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("Kd3"), chess.WHITE, phase="endgame").kind == (
        "activateKing"
    )
    # A pawn push that hits an enemy pawn is a break.
    board = chess.Board("4k3/8/3p4/8/2P5/8/8/4K3 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("c5"), chess.WHITE).kind == "break"
    # A rook dropping behind a passed pawn is the classic rook-behind-passer idea.
    board = chess.Board("4k3/8/8/P7/8/8/8/1R2K3 w - - 0 1")
    assert P.best_move_idea(board, board.parse_san("Ra1"), chess.WHITE).kind == "rookBehindPasser"


def test_pawn_storm_is_a_pawn_push_at_the_castled_king() -> None:
    # Black has castled kingside (g8); g4 storms that wing, a4 is only queenside space.
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQ - 0 7")
    assert P.best_move_idea(board, board.parse_san("g4"), chess.WHITE).kind == "pawnStorm"
    assert P.best_move_idea(board, board.parse_san("a4"), chess.WHITE).kind == "space"


def test_prophylaxis_is_a_quiet_move_that_stops_the_opponent_s_threat() -> None:
    # Black threatens ...Nxe4: the e4 pawn hangs to the f6 knight. Nc3 defends it -- prophylaxis,
    # and it outranks the fact that Nc3 also develops.
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    threat = chess.Move.from_uci("f6e4")
    idea = P.best_move_idea(
        board, board.parse_san("Nc3"), chess.WHITE, phase="opening", threat=threat
    )
    assert idea is not None and idea.kind == "prophylaxis"
    # A move that ignores the threat is not prophylaxis.
    assert (
        P.best_move_idea(board, board.parse_san("a3"), chess.WHITE, phase="opening", threat=threat)
        is None
    )
    # With no threat supplied, Nc3 is read as ordinary development, not prophylaxis.
    assert P.best_move_idea(board, board.parse_san("Nc3"), chess.WHITE, phase="opening").kind == (
        "develop"
    )


def test_central_infiltration_is_not_called_centralising() -> None:
    # A knight dropping to c2 (rank 2) is infiltration, not a central strong square (c3-f6).
    board = chess.Board("4k3/8/8/8/8/1n6/8/4K3 b - - 0 1")
    assert not P._is_central_strong(board, chess.C2, chess.BLACK)
    # But a knight to d4 is a central strong square.
    assert P._is_central_strong(board, chess.D4, chess.BLACK)
