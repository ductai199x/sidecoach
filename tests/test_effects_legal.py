"""What a move does, read off legal moves: a pinned piece attacks and defends nothing off its
line, a king takes nothing the other side still guards, a defence counts only when it saves
the piece, a fork only when it wins something, a pin only when it holds. Most positions are
ones the audit of 2026-09-23 reproduced a false sentence on; the rest are their sound
counterparts, and a check being answered."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach.effects import describe_effects

# In check, the only legal captures are of the checker; read that way, the knight on c2
# "attacked" the bishop on b4 only once the checking queen was gone.
QUEEN_ON_F2 = "rn2k1nr/2pp1p2/P3p3/7p/1b1P2P1/4P3/PPNN1q1P/R1BQKR2 w Qkq - 0 15"
# What a move does, all of it (`describe_effects`, up to nine phrases). (case, FEN, the
# move, its effects.)
EFFECTS: list[tuple[str, str, str, list[str]]] = [
    # --- a pinned piece, or a king that cannot take, attacks nothing
    # Nd2 blocks the check and is pinned there: it does not attack the knight on e4, and it
    # stands where the bishop and the knight both take it, the king unable to take back.
    (
        "a pinned blocker",
        "4k3/8/8/8/1b2n3/8/8/1N2K3 w - - 0 1",
        "Nd2",
        [
            "puts the knight where it is undefended and attacked by the bishop on b4 and the "
            "knight on e4"
        ],
    ),
    # The d5 pawn guards the knight: the king beside it cannot take it.
    ("the king cannot take", "4k3/8/8/3p4/4n3/8/6K1/8 w - - 0 1", "Kf3", []),
    # Blocking the check on c5 pins the bishop, so the d6 pawn is not attacked; the rook's
    # line to f7, which the bishop left, is a real discovery.
    (
        "a pinned blocker uncovers",
        "4r3/2p2ppp/B1Pp3n/K5q1/4k3/3p4/PPP2BP1/RN1Q1R2 w - - 6 26",
        "Bc5",
        [
            "uncovers an attack on the pawn on f7 by the rook on f1",
            "puts the bishop where it is undefended and attacked by the queen on g5 and the pawn "
            "on d6",
        ],
    ),
    # --- attacked is a legal capture, defended a legal recapture
    # The knight on c6 is pinned to its king: the rook on d4 is not attacked.
    ("attacked by a pinned knight", "4k3/8/2n5/1B6/8/8/8/3RK3 w - - 0 1", "Rd4", []),
    # A discovered check: gxf5 does not answer it, so the knight is not attacked.
    (
        "a discovered check",
        "3k4/8/6p1/8/3N4/8/8/3RK3 w - - 0 1",
        "Nf5+",
        ["gives check", "uncovers an attack on the king on d8 by the rook on d1"],
    ),
    # The c7 pawn, attacked by the bishop, steps to c5 beside the king on b5: the knight on
    # a6 guards c5, so the king cannot take and the pawn is out of attack.
    (
        "beside a king that cannot take",
        "b4k2/2p3pQ/n3p1p1/BK6/4PP2/8/PP6/R2N4 b - - 0 28",
        "c5",
        ["gets the pawn out of attack from the bishop on a5"],
    ),
    # --- a defence counts only when it saves the piece
    # The knight blocking on d2 is pinned: it cannot take back on e4.
    ("a pinned defender", "4k3/8/5n2/8/1b2B3/8/8/1N2K3 w - - 0 1", "Nd2", []),
    # The b4 pawn wins the knight on c3 defended or not.
    ("a pawn wins it anyway", "4k3/8/8/8/1p6/2N5/8/2B1K3 w - - 0 1", "Bd2", []),
    # Against a knight the bishop's guard holds: Nxc3 Bxc3 is only a trade.
    (
        "a guard that holds",
        "4k3/8/8/1n6/8/2N5/8/2B1K3 w - - 0 1",
        "Bd2",
        ["defends the knight on c3, which is attacked by the knight on b5"],
    ),
    # --- a fork needs two targets it wins, from a square it keeps
    # Two knights the pawns guard: the queen attacks one and forks nothing.
    (
        "two guarded knights",
        "k7/1p5p/2n3n1/8/8/8/8/4Q1K1 w - - 0 1",
        "Qe4",
        ["attacks the knight on c6"],
    ),
    # Rooks are worth more than the knight; the king is always a target.
    (
        "two rooks",
        "r3r1k1/5ppp/8/1N6/8/8/8/6K1 w - - 0 1",
        "Nc7",
        ["forks the rook on a8 and the rook on e8"],
    ),
    (
        "the king and a rook",
        "r3k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
        "Nc7+",
        ["gives check", "forks the king on e8 and the rook on a8"],
    ),
    # The same knight fork with a bishop on b8 to take the knight: no fork.
    (
        "a fork taken",
        "rb2k3/8/8/1N6/8/8/8/4K3 w - - 0 1",
        "Nc7+",
        [
            "gives check",
            "attacks the king on e8",
            "puts the knight where it is undefended and attacked by the bishop on b8",
        ],
    ),
    # --- a pin the pinned piece breaks at a profit is not named
    # Rxe1 wins the unguarded rook: the "pin" holds nothing.
    (
        "a pin broken at a profit",
        "4k3/4r3/8/8/8/8/7K/7R w - - 0 1",
        "Re1",
        [
            "attacks the rook on e7",
            "puts the rook where it is undefended and attacked by the rook on e7",
            "takes the open e-file",
        ],
    ),
    # With the king on f1 guarding e1, Rxe1+ Kxe1 is only a trade: the pin stands.
    (
        "a pin that stands",
        "4k3/4r3/8/8/8/8/8/R4K2 w - - 0 1",
        "Re1",
        ["attacks the rook on e7", "pins the rook on e7 to the king", "takes the open e-file"],
    ),
    # --- answering a check does not make every attack new
    ("the king takes the checker", QUEEN_ON_F2, "Kxf2", ["takes the queen on f2"]),
    (
        "the rook takes the checker",
        QUEEN_ON_F2,
        "Rxf2",
        [
            "takes the queen on f2",
            "attacks the pawn on f7",
            "gets the rook out of attack from the queen on f2",
        ],
    ),
]


@pytest.mark.parametrize(("case", "fen", "san", "effects"), EFFECTS, ids=[r[0] for r in EFFECTS])
def test_what_a_move_does_is_read_off_legal_moves(
    case: str, fen: str, san: str, effects: list[str]
) -> None:
    board = chess.Board(fen)
    assert describe_effects(board, board.parse_san(san), limit=9) == effects, case


def test_a_line_move_by_move_says_what_each_move_does_and_what_a_check_forces() -> None:
    """The notes a line is explained by: a reply to check says whether only a king move got
    out of it, or it was one of several ways out (the bench's gold labels: "the check forced
    the King away" where a block was legal)."""
    from sidecoach.coach.effects import line_notes

    board = chess.Board("8/p1q1k2K/2p1p3/3p4/3P1R1P/6r1/1r6/8 w - - 0 49")
    notes = line_notes(board, ["Rf1", "Qd6", "Rf7+", "Kxf7"])
    assert notes.startswith(
        "Rf1 (gets the rook out of attack from the queen on c7), Qd6, Rf7+ (gives check"
    )
    assert notes.endswith("Kxf7 (the king must move: Kd8, Ke8 or Kxf7; takes the rook on f7)")
    pinned = chess.Board("6rk/5prq/1p1p1n2/p1pP1B1p/P1P2P1n/2P1R1P1/2Q4K/2B2NR1 b - - 0 35")
    notes = line_notes(pinned, ["f6g4", "f5g4", "h7c2"])  # UCI, as the engine gives it
    assert "Bxg4 (one of 3 ways out of check; takes the knight on g4" in notes
    assert notes.split(", ")[-1].startswith("Qxc2+ (takes the queen on c2")
    assert (
        line_notes(board, ["Rf1", "Ke9"])
        == "Rf1 (gets the rook out of attack from the queen on c7)"
    )


def test_a_line_said_aloud_gives_each_move_what_it_does() -> None:
    """The template's answer to "show me the line": exact, each move with what it does."""
    from sidecoach.coach.effects import line_spoken

    board = chess.Board("1r3rk1/p4pp1/3p1q1p/1p1BnP2/2p1P3/1P2Q2P/P5P1/3R1RK1 w - - 0 22")
    assert line_spoken(board, ["Qxa7", "c3", "Rf2", "b4"]) == (
        "Qxa7, taking the pawn on a7 and attacking the rook on b8; c3, getting the pawn out "
        "of attack from the pawn on b3 and the bishop on d5; Rf2; b4"
    )
    mate = chess.Board("6k1/5pp1/p2p4/1p3Pb1/2p1r1q1/P1P1p3/1P5K/4R3 b - - 1 48")
    assert line_spoken(mate, ["Qh4+", "Kg2", "Qf2+", "Kh3", "Rh4#", "Kg3"]).endswith(
        "Rh4#, checkmate"
    )


def test_the_words_hold_for_the_board_the_move_leaves() -> None:
    """The facts audit, 2026-10-01: the file as the capture leaves it, "defends" only of the
    moved piece, and no fork that one answer to the check undoes."""
    board = chess.Board("8/6bk/6p1/3QP1Bp/3N4/Pp6/1q4PP/2r3RK b - - 0 40")
    said = describe_effects(board, board.parse_san("Qxa3"), limit=9)
    assert "takes the open a-file" in said
    # Kc1 steps off the d-file: the knight on d2, no longer pinned, guards the queen on b3.
    board = chess.Board("2kr1b1r/p4ppp/1P6/3q4/8/1Q6/1P1N4/R2K4 w - - 7 28")
    said = describe_effects(board, board.parse_san("Kc1"), limit=9)
    assert not any(s.startswith("defends") for s in said)
    # Qa4+ "forks" the king and the knight on b4, but Nc6 blocks and saves the knight.
    board = chess.Board("r1bqk2r/ppp2ppp/3p4/8/1nBPP1n1/5N2/PP3PPP/R1BQR1K1 w kq - 0 12")
    said = describe_effects(board, board.parse_san("Qa4+"), limit=9)
    assert not any(s.startswith("forks") for s in said)
