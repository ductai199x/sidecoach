"""A position read from pixels learns what the site's last-move highlight shows: a pawn that
has just advanced two squares can be taken in passing (Tai's chess.com puzzle, 2026-09-24:
after c2-c4 the coach did not see bxc3 e.p. and recommended Bd7)."""

from __future__ import annotations

import chess

from sidecoach.state.reconcile import LastMoveSeen, Reconciler, last_move_seen

# Tai's puzzle, Black to move after c2-c4, with the black pawn on b4.
PLACEMENT = "2r1k2r/2q1bppp/p2p4/1b1Pp3/1pP2P2/1P1Q4/P2BB1PP/R4RK1"


def _placement() -> chess.Board:
    board = chess.Board(None)
    board.set_board_fen(PLACEMENT)
    return board


def test_the_highlighted_double_step_gives_the_en_passant_square() -> None:
    board = _placement()
    assert last_move_seen(board, frozenset({chess.C2, chess.C4})) == LastMoveSeen(
        chess.WHITE, chess.C3
    )
    adopted = Reconciler._adopt_placement(
        board, prefer=chess.BLACK, highlighted=frozenset({chess.C2, chess.C4})
    )
    assert adopted is not None
    position, turn_unknown = adopted
    assert position.turn == chess.BLACK and not turn_unknown
    assert position.ep_square == chess.C3
    assert chess.Move.from_uci("b4c3") in position.legal_moves
    assert " c3 " in position.fen()


def test_no_highlight_no_en_passant() -> None:
    adopted = Reconciler._adopt_placement(_placement(), prefer=chess.BLACK)
    assert adopted is not None
    assert adopted[0].ep_square is None


def test_a_one_square_advance_says_whose_move_it_is_and_no_en_passant() -> None:
    board = _placement()
    assert last_move_seen(board, frozenset({chess.C3, chess.C4})) == LastMoveSeen(chess.WHITE)
    # The start square occupied: no move from it.
    board.set_piece_at(chess.C2, chess.Piece(chess.PAWN, chess.WHITE))
    assert last_move_seen(board, frozenset({chess.C2, chess.C4})) is None


def test_a_piece_move_highlighted_settles_the_side_to_move() -> None:
    # Lichess 000rO after the setup move Ka5 (a6-a5): both sides could be to move on the
    # placement alone; the highlight says White moved, so Black is to move.
    board = chess.Board(None)
    board.set_board_fen("3R4/8/8/KB2b3/1p6/1P2k3/3p4/8")
    adopted = Reconciler._adopt_placement(
        board, prefer=chess.WHITE, highlighted=frozenset({chess.A6, chess.A5})
    )
    assert adopted is not None
    assert adopted[0].turn == chess.BLACK and adopted[1] is False
