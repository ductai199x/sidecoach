"""The plan the best line carries out, and the threat the best move makes (`coach/plans.py`)."""

from __future__ import annotations

import chess
import pytest

from sidecoach.analysis import positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach import plans


def _line(fen: str, ucis: str) -> tuple[chess.Board, list[chess.Move]]:
    return chess.Board(fen), [chess.Move.from_uci(u) for u in ucis.split()]


# The plan `plans.read` finds in a best line, white's. (case, FEN, the line in UCI, the
# score given (None: the default), the plan's kind (None: no plan), its fields (a set is
# compared as one), and its words: `pieces`, `target` and `why`.)
READS: list[tuple[str, str, str, float | None, str | None, dict[str, object], dict[str, str]]] = [
    (
        "rooks doubled on an open file",
        "6k1/p4ppp/8/8/8/8/5PPP/R4RK1 w - - 0 1",
        "a1d1 a7a6 d1d2 a6a5 f1d1 a5a4",
        None,
        "double",
        {"where": "d", "detail": "open"},
        {
            "pieces": "your rook on f1 and your rook on a1",
            "why": "Rad1 and Rfd1 double your rooks on the open d-file",
        },
    ),
    (
        "a knight in two moves to an outpost",
        "4k3/8/3p4/8/4P3/8/8/1N2K3 w - - 0 1",
        "b1c3 e8f7 c3d5 f7e6",
        None,
        "outpost",
        {"square": chess.D5},
        {"target": "the d5 square, an outpost for a knight"},
    ),
    # A knight that goes out and back is no route.
    (
        "out and back",
        "4k3/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1",
        "d5c3 e8f7 c3d5 f7e6",
        None,
        None,
        {},
        {},
    ),
    (
        "a passed pawn pushed twice",
        "4k3/7p/8/8/P7/8/8/4K3 w - - 0 1",
        "a4a5 h7h6 a5a6 h6h5",
        None,
        "passer",
        {"square": chess.A4},
        {},
    ),
    (
        "pawns rolled at a king on the other wing",
        "6k1/p4ppp/8/8/8/8/6PP/2K5 w - - 0 1",
        "g2g4 a7a6 h2h4 a6a5 g4g5 a5a4",
        None,
        "storm",
        {"where": "kingside"},
        {},
    ),
    # Both kings behind the kingside pawns: h3 and h4 make room or keep a piece out.
    (
        "pawns in front of your own king",
        "6k1/p4ppp/8/8/8/8/5PPP/6K1 w - - 0 1",
        "g2g4 a7a6 h2h4 a6a5",
        None,
        None,
        {},
        {},
    ),
    (
        "queenside pawns away from both kings",
        "6k1/p4ppp/8/8/8/8/PP3PPP/6K1 w - - 0 1",
        "b2b4 g8f8 a2a4 f8e7",
        None,
        "wing",
        {"where": "queenside"},
        {},
    ),
    (
        "a break prepared by the best move",
        "4k3/8/8/4p3/8/8/3P4/4K1N1 w - - 0 1",
        "g1f3 e8e7 d2d4 e7e6",
        None,
        "break",
        {"square": chess.D4},
        {"why": "Nf3 brings a piece to bear on d4, then d4 breaks open the centre"},
    ),
    # The lift and the swing both belong to it.
    (
        "queen and rook lifted to their king",
        "6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",
        "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4",
        None,
        "attack",
        {"square": chess.G8, "pieces": {chess.D1, chess.E1}},
        {"why": "Qd3, Re3 and Rh3 bring your queen and rook against their king"},
    ),
    # Not for a side the engine calls worse: there it is a hope, not a plan.
    (
        "an attack for the worse side",
        "6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1",
        "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4",
        0.3,
        None,
        {},
        {},
    ),
    # Pawns traded on the queenside are an exchange, not an expansion.
    (
        "captures",
        "6k1/p4ppp/1p6/P7/8/8/1P3PPP/6K1 w - - 0 1",
        "a5b6 a7b6 b2b4 g8f8",
        None,
        None,
        {},
        {},
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "ucis", "score", "kind", "fields", "words"), READS, ids=[r[0] for r in READS]
)
def test_the_plan_a_best_line_carries_out(
    case: str,
    fen: str,
    ucis: str,
    score: float | None,
    kind: str | None,
    fields: dict[str, object],
    words: dict[str, str],
) -> None:
    board, pv = _line(fen, ucis)
    plan = plans.read(board, pv) if score is None else plans.read(board, pv, score=score)
    if kind is None:
        assert plan is None, case
        return
    assert plan is not None and plan.kind == kind, case
    for name, value in fields.items():
        got = getattr(plan, name)
        assert (set(got) if isinstance(value, set) else got) == value, f"{case}: {name}"
    said = {
        "pieces": lambda: plans.pieces(plan, board),
        "target": lambda: plans.target(plan),
        "why": lambda: plans.why(plan, board),
    }
    for name, text in words.items():
        assert said[name]() == text, f"{case}: {name}"


def test_a_push_in_front_of_your_own_king_is_no_storm() -> None:
    # Both kings behind the kingside pawns: h4 makes room or keeps a piece out.
    board = chess.Board("6k1/p4ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    assert positional._pawn_storm_side(board, chess.Move.from_uci("h2h4"), chess.WHITE) is None


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
