"""The plan the best line carries out, and the threat the best move makes (`coach/plans.py`)."""

from __future__ import annotations

import chess

from sidecoach.analysis import positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach import plans


def _line(fen: str, ucis: str) -> tuple[chess.Board, list[chess.Move]]:
    return chess.Board(fen), [chess.Move.from_uci(u) for u in ucis.split()]


def test_rooks_brought_to_an_open_file_are_doubled() -> None:
    board, pv = _line("6k1/p4ppp/8/8/8/8/5PPP/R4RK1 w - - 0 1", "a1d1 a7a6 d1d2 a6a5 f1d1 a5a4")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "double"
    assert plan.where == "d" and plan.detail == "open"
    assert plans.pieces(plan, board) == "your rook on f1 and your rook on a1"
    assert plans.why(plan, board) == "Rad1 and Rfd1 double your rooks on the open d-file"


def test_a_knight_taken_in_two_moves_to_an_outpost() -> None:
    board, pv = _line("4k3/8/3p4/8/4P3/8/8/1N2K3 w - - 0 1", "b1c3 e8f7 c3d5 f7e6")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "outpost"
    assert plan.square == chess.D5
    assert plans.target(plan) == "the d5 square, an outpost for a knight"


def test_a_knight_that_goes_out_and_back_is_no_route() -> None:
    board, pv = _line("4k3/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1", "d5c3 e8f7 c3d5 f7e6")
    assert plans.read(board, pv) is None


def test_a_passed_pawn_pushed_twice() -> None:
    board, pv = _line("4k3/7p/8/8/P7/8/8/4K3 w - - 0 1", "a4a5 h7h6 a5a6 h6h5")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "passer" and plan.square == chess.A4


def test_pawns_rolled_at_a_king_on_the_other_wing_are_a_storm() -> None:
    board, pv = _line("6k1/p4ppp/8/8/8/8/6PP/2K5 w - - 0 1", "g2g4 a7a6 h2h4 a6a5 g4g5 a5a4")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "storm" and plan.where == "kingside"


def test_pawns_in_front_of_your_own_king_are_no_storm() -> None:
    # Both kings behind the kingside pawns: h3 and h4 make room or keep a piece out.
    board = chess.Board("6k1/p4ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    assert positional._pawn_storm_side(board, chess.Move.from_uci("h2h4"), chess.WHITE) is None
    board, pv = _line(board.fen(), "g2g4 a7a6 h2h4 a6a5")
    assert plans.read(board, pv) is None


def test_queenside_pawns_away_from_both_kings_gain_space() -> None:
    board, pv = _line("6k1/p4ppp/8/8/8/8/PP3PPP/6K1 w - - 0 1", "b2b4 g8f8 a2a4 f8e7")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "wing" and plan.where == "queenside"


def test_a_break_prepared_by_the_best_move() -> None:
    board, pv = _line("4k3/8/8/4p3/8/8/3P4/4K1N1 w - - 0 1", "g1f3 e8e7 d2d4 e7e6")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "break" and plan.square == chess.D4
    assert (
        plans.why(plan, board) == "Nf3 brings a piece to bear on d4, then d4 breaks open the centre"
    )


def test_queen_and_rook_lifted_to_their_king_are_an_attack() -> None:
    board, pv = _line("6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4")
    plan = plans.read(board, pv)
    assert plan is not None and plan.kind == "attack" and plan.square == chess.G8
    assert set(plan.pieces) == {chess.D1, chess.E1}
    # Not for a side the engine calls worse: there it is a hope, not a plan.
    assert plans.read(board, pv, score=0.3) is None


def test_captures_make_no_plan() -> None:
    # Pawns traded on the queenside are an exchange, not an expansion.
    board, pv = _line("6k1/p4ppp/1p6/P7/8/8/1P3PPP/6K1 w - - 0 1", "a5b6 a7b6 b2b4 g8f8")
    assert plans.read(board, pv) is None


def test_a_move_that_attacks_a_loose_piece_is_a_threat() -> None:
    board = chess.Board("4k3/8/8/1b6/8/8/8/1N2K3 w - - 0 1")
    move = chess.Move.from_uci("b1c3")
    t = plans.threat(board, move)
    assert t is not None and t.kind == "win" and t.square == chess.B5
    assert plans.threat_target(t, board) == "their bishop on b5"
    assert plans.threat_why(t, board, move) == (
        "Nc3 goes after their bishop on b5: if they let it be, Nxb5 wins it"
    )


def test_a_move_that_gives_its_piece_away_is_no_threat() -> None:
    # The knight lands where a pawn takes it: an offer, not a threat.
    board = chess.Board("4k3/8/8/1b6/1p6/8/8/1N2K3 w - - 0 1")
    assert plans.threat(board, chess.Move.from_uci("b1c3")) is None


def test_a_quiet_move_that_sets_up_mate_threatens_it() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/1R3PPP/6K1 w - - 0 1")
    move = chess.Move.from_uci("b2e2")
    t = plans.threat(board, move)
    assert t is not None and t.kind == "mate" and t.square == chess.G8
    assert plans.threat_why(t, board, move) == "Re2 threatens Re8 mate"


def test_like_pieces_face_to_face_are_a_trade_when_the_line_makes_it() -> None:
    board = chess.Board("7k/8/8/8/8/8/RK6/7r w - - 0 1")
    move = chess.Move.from_uci("a2a1")
    traded = [move, chess.Move.from_uci("h1a1"), chess.Move.from_uci("b2a1")]
    t = plans.threat(board, move, traded)
    assert t is not None and t.kind == "trade" and t.square == chess.H1
    assert plans.threat_why(t, board, move) == (
        "Ra1 puts your rook face to face with their rook on h1, to trade it off"
    )
    # Not when the line does not take it up.
    elsewhere = [move, chess.Move.from_uci("h8g8"), chess.Move.from_uci("b2c2")]
    assert plans.threat(board, move, elsewhere) is None


def test_a_level_line_that_repeats_is_a_draw_only_when_it_saves_the_game() -> None:
    board, pv = _line("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "g1h1 g8h8 h1g1 h8g8")
    boards = [board.copy(stack=False)]
    for move in pv:
        nxt = boards[-1].copy(stack=False)
        nxt.push(move)
        boards.append(nxt)
    assert idea_of._drawn(boards, pv, 0.5) == "repetition"
    assert idea_of._drawn(boards, pv, 0.5, second=0.2) == "repetition"
    # Every other move holds as well: the engine marking time, not a rescue.
    assert idea_of._drawn(boards, pv, 0.5, second=0.48) is None
