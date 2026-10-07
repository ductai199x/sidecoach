"""The idea of the best line (`coach/idea.py`) and its words (`coach/idea_words.py`).

Each case is a Lichess puzzle the coach misread before the reader was rebuilt (2026-09-23),
with the engine's line written out so no engine is needed: the goal, the blow that delivers
it, the sacrifices on the way, and the words at each rung.
"""

from __future__ import annotations

import re

import chess

from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.coach import focus_template, idea_template, move_template
from sidecoach.coach.disclosure import leak_check
from sidecoach.coach.facts import describe_tactic, facts_from_analysis
from sidecoach.events import Analysis, DisclosureLevel, Line
from tests.lines import analysed, boards_along, san_moves


def analysis(fen: str, line: str, *, score: float = 0.95, mate: int | None = None) -> Analysis:
    """A puzzle's analysis: a decisive position, `line` its best line, a win by default."""
    return analysed(fen, line, score=score, mate=mate, decisive=True)


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


def test_material_given_for_their_attack_is_not_won() -> None:
    # 29.exf4 in a lost game: the knight Black gives with ...Ng4+ is not "won back".
    board = chess.Board("6rk/3b1prq/1p1p1n2/p1pP1n1p/P1P1pp2/2P1P1PP/R6K/2BQ1NRB w - - 0 29")
    moves = ["e3f4", "f6g4", "h3g4", "h5h4", "h1e4", "h4g3", "h2g2", "f5h4", "g2h3", "d7g4"]
    analysis = Analysis(
        game_id="",
        fen=board.fen(),
        ply=0,
        lines=[Line(moves=moves, depth=18, expected_score=0.055)],
        decisive=False,
        only_move=False,
    )
    assert idea_of.read(board, analysis).goal != "win"


def test_a_level_line_that_repeats_is_a_draw_only_when_it_saves_the_game() -> None:
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1")
    pv = [chess.Move.from_uci(uci) for uci in ("g1h1", "g8h8", "h1g1", "h8g8")]
    boards = boards_along(board, pv)
    assert idea_of._drawn(boards, pv, 0.5) == "repetition"
    assert idea_of._drawn(boards, pv, 0.5, second=0.2) == "repetition"
    # Every other move holds as well: the engine marking time, not a rescue.
    assert idea_of._drawn(boards, pv, 0.5, second=0.48) is None


# ---- the blow ---------------------------------------------------------------------------


def test_a_trapped_queen_is_trapped_not_loose() -> None:
    """Na4 traps the queen on b2 (Lichess 048VU): it was called "a loose piece". And the rook
    that goes, Qxa1 Qxa1, is no sacrifice: the trapped queen follows at once."""
    fen = "rbb2rk1/1p3ppp/p1n2n2/3p4/3Pp3/P1N1B2P/1qP1BPPN/R2QK2R w KQ - 0 14"
    idea = read(fen, "Na4 Qxa1 Qxa1")
    assert idea.blow is not None and idea.blow.name == "trappedPiece"
    assert (
        idea_words.target(idea, chess.Board(fen))
        == "their queen on b2: after your first move, it has no safe square"
    )
    assert idea.sacrifices == ()


def test_a_piece_given_for_nothing_is_no_sacrifice() -> None:
    """In check and lost, Qxf6 Qxf6 gives the queen for a knight and nothing comes back:
    "your queen on e5 -- the piece you give up" (Fischer-Benko)."""
    fen = "r1b2rk1/p4p1p/4pN1Q/4qp2/1p3p2/1n6/PPP3PP/3R1R1K b - - 0 22"
    line = "Qxf6 Qxf6 Nc5 Rxf4 Ne4 Rxe4 fxe4 h3 Bb7 Rf1 Ba6 Qg5+ Kh8 Qe5+ Kg8 Rf4"
    idea = read(fen, line, score=0.07)
    assert idea.sacrifices == ()
    assert "give up" not in (idea_words.piece(idea, chess.Board(fen)) or "")


def test_a_sound_sacrifice_in_a_worse_position_is_one() -> None:
    """Bxh7+ Kxh7 Ng5+ Kg8 Qh5 a little worse (0.47), nothing back in the line: giving the
    bishop costs less than an inaccuracy against the even material the player had, a Good
    move's loss, so the bishop is sacrificed (Tai, 2026-10-02: no "below 0.5 it needs two
    points back" gate; a sacrifice keeps at least the Good label)."""
    fen = "r1bq1rk1/pppn1ppp/4p3/3pP3/1b1P4/2NB1N2/PPP2PPP/R2QK2R w KQ - 0 8"
    idea = read(fen, "Bxh7+ Kxh7 Ng5+ Kg8 Qh5", score=0.47)
    assert idea.sacrifices == ((0, "bishop"),)


def test_a_bishop_given_in_a_lost_position_for_nothing_is_no_sacrifice() -> None:
    """Six points down, ...Bg4 gives the bishop and the line is worth what nine points down is
    (-8.99): the gift costs more than an inaccuracy against the six-down count, below a Good
    move -- given up, not sacrificed ("your bishop on c8 -- the piece you give up", a bot game
    of 2026-09-25; the strict comparison of d32239a passed it by a centipawn)."""
    fen = "r1b3k1/ppp3pp/5r2/n7/8/5B2/PPP1NbPP/R2Q1R1K b - - 1 15"
    idea = read(fen, "Bg4 Bxg4 Kh8 Bh3 Nc6 a3 Bb6 Rxf6 gxf6 Qf1 f5 Qxf5", score=0.0352)
    assert idea.sacrifices == ()


def test_a_piece_another_move_could_defend_is_sacrificed_not_lost_anyway() -> None:
    """The knight on a4 is attacked and every square it can go to is covered, but Ra1 or Rd4
    would defend it: h3 Rxa4 gives it, a sacrifice, not a piece lost whatever it did."""
    fen = "r6k/2p3pp/3p4/4b3/N7/8/5PPP/3R2K1 w - - 0 30"
    idea = read(fen, "h3 Rxa4", score=0.6)
    assert idea.sacrifices == ((0, "knight"),)


def test_the_target_says_when_the_pattern_comes_about() -> None:
    """The loose piece is loose once the blow is on the board, not now: fxe2 takes the knight
    that guards the rook on g1 (Lichess 01M89, "nothing defends it" of a guarded rook)."""
    fen = "8/2p5/1p2B3/p7/5r2/P4p2/1PP1N2k/1K4R1 b - - 2 59"
    idea = read(fen, "fxe2 Re1 Rf1 Kc1 Rxe1+ Kd2")
    assert (
        idea_words.target(idea, chess.Board(fen))
        == "their rook on g1: after your first move, nothing defends it"
    )
    # A fork on the third move comes after the second, not "after your first move" (00SIE).
    fen = "2k3nr/ppp2ppp/2n5/8/8/1Q2P3/q4rPP/1R2KB1R w K - 0 16"
    line = "Qxb7+ Kd7 Rd1+ Ke7 Qxc7+ Kf6 Qxc6+ Kg5 h4+ Kh5 g4+ Kxg4 Qe4+ Kh5 Qe5+ f5"
    idea = read(fen, line)
    assert idea.blow_at == 3
    said = idea_words.target(idea, chess.Board(fen))
    assert said is not None and said.endswith("one move can hit both after your second move")


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
    board = chess.Board("5rk1/5p2/P7/8/8/8/5R2/5RK1 w - - 0 1")
    line = san_moves(board, "Rxf7 Rxf7 Rxf7 Kxf7 a7")
    assert idea_of._sacrifices(boards_along(board, line), line, 8) == ((2, "rook"),)


def test_a_queen_sacrifice_for_mate_is_said_at_each_rung() -> None:
    fen = "r3nr1k/4N1pp/p2p1p2/4nR2/1p2P3/2q4Q/P3B1PP/4R2K w - - 4 24"
    board = chess.Board(fen)
    facts = facts_from_analysis(board, analysis(fen, "Qxh7+ Kxh7 Rh5#", mate=2))
    idea = idea_template(facts)
    assert "mate in two" in idea.split(".")[0]
    assert "sacrifice" in idea and not leak_check(idea, facts, DisclosureLevel.IDEA)
    assert "the piece you give up" in focus_template(facts)
    assert move_template(facts) == "Qxh7+, a queen sacrifice, and after Kxh7, Rh5 is mate."


def test_a_mate_that_starts_quietly_is_never_hinted_as_checks() -> None:
    """Rh5 Kf8 Rh8#: a mate in two whose first move gives no check. Its hints said "Go through
    your checks one by one" and "Which check comes first?" beside "It starts with a quiet
    move" (the audit, 2026-10-07). They point at the quiet move's threat now, without naming
    the move, its square or its piece; a mate that starts with a check keeps its checks."""
    fen = "6k1/8/5K2/3R4/8/8/8/8 w - - 0 1"
    board = chess.Board(fen)
    idea = read(fen, "Rh5 Kf8 Rh8#", mate=2)
    assert idea.goal == "mate" and not idea.checks_first
    facts = facts_from_analysis(board, analysis(fen, "Rh5 Kf8 Rh8#", mate=2))
    said = {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "question": idea_words.question(idea, None, board),
        "nudge": idea_words.nudge(idea, board) or "",
        "target": idea_words.target(idea, board) or "",
    }
    for rung, text in said.items():
        unsaid = re.sub(r"no check", "", text, flags=re.IGNORECASE)
        assert not re.search(r"\bchecks?\b", unsaid, re.IGNORECASE), (rung, text)
    assert "quiet move" in said["question"] and "mate" in said["nudge"]
    assert not leak_check(said["idea"], facts, DisclosureLevel.IDEA)
    assert not leak_check(said["focus"], facts, DisclosureLevel.FOCUS)

    checking = "r3nr1k/4N1pp/p2p1p2/4nR2/1p2P3/2q4Q/P3B1PP/4R2K w - - 4 24"
    first = read(checking, "Qxh7+ Kxh7 Rh5#", mate=2)
    assert first.checks_first
    assert idea_words.nudge(first, chess.Board(checking)) == "Which check comes first?"


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


def test_a_queen_won_by_taking_it_is_not_won_by_the_fork_the_capture_makes() -> None:
    """35...Nxf5 in Johner-Nimzowitsch: Ng4+ pulls the pinned bishop off the diagonal and
    Qxc2+ takes the queen, forking a bishop, a pawn and the king as it does. "That's a fork"
    beside a queen won on c2 whose targets were c1, h2 and a4 was false: the pin is it."""
    fen = "6rk/5prq/1p1p1n2/p1pP1B1p/P1P2P1n/2P1R1P1/2Q4K/2B2NR1 b - - 0 35"
    idea = read(fen, "Ng4+ Bxg4 Qxc2+ Re2 Qxc1 Ne3", score=0.96)
    assert idea.goal == "win" and idea.won == "a queen for a knight"
    assert idea.blow is not None and idea.blow.name == "pin"


def test_a_queen_taken_back_leaves_the_pin_the_point() -> None:
    """The capture is the point only of what the line wins: a queen taken back after they
    took one is not, and the pin that nets the rook after it stays the blow."""
    from sidecoach.coach.idea import _capture_is_the_point
    from sidecoach.events import MotifTag

    board = chess.Board("6rk/5prq/1p1p1n2/p1pP1B1p/P1P2P1n/2P1R1P1/2Q4K/2B2NR1 b - - 0 35")
    for san in ("Ng4+", "Bxg4"):
        board.push_san(san)
    move = board.parse_san("Qxc2+")
    fork = MotifTag(name="fork", squares=["c2", "c1", "h2", "a4"])
    assert _capture_is_the_point(board, move, fork, frozenset({chess.QUEEN}))
    assert not _capture_is_the_point(board, move, fork, frozenset({chess.ROOK}))
    rook = MotifTag(name="fork", squares=["c2", "e3", "h2"])  # a rook aimed at: worth less
    assert _capture_is_the_point(board, move, rook, frozenset({chess.QUEEN}))
