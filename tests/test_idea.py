"""The idea of the best line (`coach/idea.py`) and its words (`coach/idea_words.py`).

Each case is a Lichess puzzle the coach misread before the reader was rebuilt (2026-09-23),
with the engine's line written out so no engine is needed: the goal, the blow that delivers
it, the sacrifices on the way, and the words at each rung.
"""

from __future__ import annotations

import chess

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.coach import focus_template, idea_template, move_template
from sidecoach.coach.disclosure import leak_check
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from sidecoach.events import Analysis, DisclosureLevel, Line


def analysis(fen: str, line: str, *, score: float = 0.95, mate: int | None = None) -> Analysis:
    board = chess.Board(fen)
    moves = []
    for san in line.split():
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    return Analysis(
        game_id="g",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=20, expected_score=score, mate=mate)],
        decisive=True,
        only_move=False,
    )


def read(fen: str, line: str, **kw: object) -> idea_of.Idea:
    return idea_of.read(chess.Board(fen), analysis(fen, line, **kw))  # type: ignore[arg-type]


# ---- the goal ---------------------------------------------------------------------------


def test_a_line_that_stops_on_a_capture_counts_what_can_take_back() -> None:
    """dxe4+ Ke3 Rxd4, the line cut there: the rook on d4 is taken back, so a pawn is won,
    not "a rook and a pawn" (Lichess 00VDN)."""
    idea = read("3r4/3r2k1/p7/3p2p1/3RP1Pp/1P3K2/P2R4/8 b - - 0 38", "dxe4+ Ke3 Rxd4")
    assert idea.goal == "win" and idea.gain == 1 and idea.won == "a pawn"
    assert idea.blow is not None and idea.blow.name == "discoveredAttack"


def test_queening_before_the_material_comes_is_a_promotion() -> None:
    """A pawn race won with Kxb8: the queen made first is the point, not "a bishop and a
    pawn" (Lichess 02YZr)."""
    idea = read(
        "1bK5/pP3p2/B7/1P4p1/6k1/8/7p/8 w - - 0 44",
        "Kxb8 h1=Q Kxa7 Kh3 b8=Q Qa1 Qc8+ g4 b6",
    )
    assert idea.goal == "promote"


def test_a_mate_is_named_by_its_pattern_never_by_a_pin_on_the_way() -> None:
    """Qh7+ Kf6 Qxh6#: "It's a pin" was said of it (Lichess 01xbu)."""
    idea = read(
        "2r2r2/2b1qpk1/p1n1p2p/1pPp2pn/1P1P4/P2Q3P/2BB1PP1/R3R1K1 w - - 0 24",
        "Qh7+ Kf6 Qxh6#",
        mate=2,
    )
    assert idea.goal == "mate" and idea.mate_in == 2
    assert idea.blow is None or idea.blow.name in idea_of.MATE_BLOWS


def test_a_rook_sacrifice_that_opens_the_king_is_said() -> None:
    """Rxh6+ gxh6 Qxh6+ ... was read as "meet their threat first", the sacrifice unsaid
    (Lichess 0199s)."""
    fen = "r3qr1k/1p2b1p1/4p1Qp/2ppP3/p4P2/2P1P1P1/PP4P1/1K1R3R w - - 1 21"
    idea = read(fen, "Rxh6+ gxh6 Qxh6+ Kg8 Rh1 Bh4 Rxh4 Qd7 Rg4+ Kf7 Rg7+ Ke8")
    assert idea.goal in ("win", "attack")
    assert [kind for _, kind in idea.sacrifices] == ["rook"]


# ---- the blow ---------------------------------------------------------------------------


def test_a_trapped_queen_is_trapped_not_loose() -> None:
    """Na4 traps the queen on b2 (Lichess 048VU): it was called "a loose piece"."""
    idea = read(
        "rbb2rk1/1p3ppp/p1n2n2/3p4/3Pp3/P1N1B2P/1qP1BPPN/R2QK2R w KQ - 0 14", "Na4 Qxa1 Qxa1"
    )
    assert idea.blow is not None and idea.blow.name == "trappedPiece"
    board = chess.Board("rbb2rk1/1p3ppp/p1n2n2/3p4/3Pp3/P1N1B2P/1qP1BPPN/R2QK2R w KQ - 0 14")
    assert idea_words.target(idea, board) == "their queen on b2: it has no safe square"


def test_a_pin_is_gone_once_its_pinner_is_taken() -> None:
    """Rxg7 pins c7 to b7 as it takes the queen, and Kxg7 takes the rook at once: the pin
    explained nothing (Lichess 07BKU)."""
    fen = "rn1Q2rk/1bp1n2p/4p3/pp6/2N1P3/2q5/P1P2P1P/1K1R2R1 w - - 2 27"
    idea = read(fen, "Rxg8+ Nxg8 Rg1 Qg7 Rxg7 Kxg7 Qxc7+ Nd7 Qxd7+")
    assert idea.goal == "win"
    assert idea.blow is None or idea.blow.name not in ("pin", "absolutePin", "attraction")


def test_a_fork_collected_by_another_piece_is_still_the_fork() -> None:
    """Bxe6+ forks king and queen; the queen takes the bishop and ours takes hers behind
    it (Lichess 0BD4O)."""
    idea = read("r5k1/1p4bp/p1n1p1p1/3q4/3P3r/1Q2P2B/PP3P1P/3RK2R w K - 0 22", "Bxe6+ Qxe6 Qxe6+")
    assert idea.blow is not None and idea.blow.name == "fork"


def test_a_fork_of_the_king_and_a_pawn_is_not_the_point() -> None:
    """Qb1+ forking the king and the pawn on a2, five moves after the queen was won, was
    named as the idea; the deflection that won the queen is (Lichess 02JZv)."""
    idea = read(
        "8/6kp/2Qq1pp1/p7/3r4/1P5P/P4PP1/2R3K1 b - - 0 32",
        "Rd1+ Rxd1 Qxc6 Re1 Qc2 Re7+ Kh6 g4 Qb1+ Kg2 Qxa2",
    )
    assert idea.blow is not None and idea.blow.name != "fork"
    assert idea.blow_at == 1


# ---- sacrifices -------------------------------------------------------------------------


def test_a_rook_given_in_two_steps_is_one_sacrifice() -> None:
    """Rxf7+ Rxf7 Rxf7+ Kxf7: the first rook comes back, the second does not (00ZPK)."""
    fen = "5rk1/5p2/P7/8/8/8/5R2/5RK1 w - - 0 1"
    board = chess.Board(fen)
    line = []
    for san in ["Rxf7", "Rxf7", "Rxf7", "Kxf7", "a7"]:
        move = board.parse_san(san)
        line.append(move)
        board.push(move)
    boards = [chess.Board(fen)]
    for move in line:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    assert idea_of._sacrifices(boards, line, 8) == ((2, "rook"),)


def test_a_rook_given_for_a_queen_next_move_is_no_sacrifice() -> None:
    """Na4 Qxa1 Qxa1: the rook goes, the trapped queen follows at once (048VU)."""
    fen = "rbb2rk1/1p3ppp/p1n2n2/3p4/3Pp3/P1N1B2P/1qP1BPPN/R2QK2R w KQ - 0 14"
    assert read(fen, "Na4 Qxa1 Qxa1").sacrifices == ()


def test_a_queen_sacrifice_for_mate_is_said_at_each_rung() -> None:
    fen = "r3nr1k/4N1pp/p2p1p2/4nR2/1p2P3/2q4Q/P3B1PP/4R2K w - - 4 24"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, analysis(fen, "Qxh7+ Kxh7 Rh5#", mate=2))
    idea = idea_template(facts)
    assert "mate in two" in idea.split(".")[0]
    assert "sacrifice" in idea and not leak_check(idea, facts, DisclosureLevel.IDEA)
    assert "the piece you give up" in focus_template(facts)
    assert move_template(facts) == "Qxh7+, a queen sacrifice, and after Kxh7, Rh5 is mate."


# ---- the words --------------------------------------------------------------------------


def test_every_piece_is_named_where_it_stands_now() -> None:
    """The skewer's king stood on a3; "their king on b3" was where it would go (05TRT)."""
    fen = "6R1/8/4r3/5P1p/7P/k5PK/8/1q6 w - - 0 54"
    board = chess.Board(fen)
    idea = read(fen, "Ra8+ Kb4 Rb8+ Kc5 Rxb1")
    assert idea.blow is not None and idea.blow.name == "skewer"
    target = idea_words.target(idea, board)
    assert target is not None and "king on a3" in target and "queen on b1" in target


def test_a_pawn_move_is_never_capitalised() -> None:
    fen = "r2r1bk1/1b5p/p4Pp1/1p2N1P1/4P2Q/1PPpq2P/6K1/R4R2 w - - 1 31"
    facts = facts_from_analysis(chess.Board(fen), analysis(fen, "f7+ Kh8 Nxg6+ Kg7 Qh6#", mate=3))
    assert move_template(facts).startswith("f7+")


def test_a_threat_is_named_in_words_below_the_move_rung() -> None:
    """ "Look at their threat is Qxc4" named a move at the focus rung (Lichess 00Ocp)."""
    fen = "r3k3/p1pb4/2p2p2/2q1p1p1/2B3p1/2P4r/P1P2PQP/3RR1K1 w q - 4 21"
    board = chess.Board(fen)
    a = analysis(fen, "Qe4", score=0.55)
    a = a.model_copy(update={"threats": ["c5c4"], "decisive": False})
    facts = facts_from_analysis(board, a)
    assert facts.goal and "threatens" in facts.goal
    assert facts.target == "their queen on c5, which threatens to take your bishop on c4"
    assert not leak_check(focus_template(facts), facts, DisclosureLevel.FOCUS)


def test_the_tactic_named_is_true_on_its_board() -> None:
    fen = "1k2r3/1p3ppp/3p4/2n5/1rPN4/1P3P2/P5PP/2KRR3 b - - 2 23"
    board = chess.Board(fen)
    idea = read(fen, "Rxe1 Rxe1 Nd3+ Kd2 Nxe1")
    why = idea_words.why(idea, board, describe_tactic)
    assert why == (
        "Rxe1, and after Rxe1, Nd3+, and the knight on d3 forks the king on c1 and the rook "
        "on e1; you win the exchange"
    )
