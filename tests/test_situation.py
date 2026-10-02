"""The first hint's parts (`coach/situation.py`, `analysis/traps.py`), held to the moments of
the master games it was read against on 2026-09-24 (Capablanca-Marshall, Johner-Nimzowitsch)
and to the Lichess puzzles that caught its first slips."""

from __future__ import annotations

import chess
import pytest

from sidecoach.analysis import positional
from sidecoach.analysis.traps import _worse
from sidecoach.coach import situation
from sidecoach.coach.coach import _said
from sidecoach.events import Line


def _line(moves: list[str], score: float, mate: int | None = None) -> Line:
    return Line(moves=moves, depth=18, expected_score=score, mate=mate)


def _moves(*uci: str) -> list[chess.Move]:
    return [chess.Move.from_uci(u) for u in uci]


# The capture that looks free and is not (`situation.trap_words`). (case, FEN, the trap's
# line, its score and mate, the square the line wins later (or None), the best line's score,
# and what is said: the words, their start (a `...` at the end), or None: no trap.)
TRAPS: list[tuple[str, str, list[str], float, int | None, int | None, float, str | None]] = [
    # Capablanca-Marshall, 14.Qf3: "14 hxg4? Qh4" and Black's attack comes.
    (
        "the bait knight",
        "r1bq1rk1/2p2ppp/p2b4/1p6/6n1/1BP4P/PP1P1PP1/RNBQR1K1 w - - 1 14",
        ["h3g4", "d8h4", "d1f3", "h4h2", "g1f1", "h2h1", "f1e2", "f8e8"],
        0.45,
        None,
        None,
        0.60,
        "careful, though: hxg4 looks like it wins a knight, but after hxg4 Qh4...",
    ),
    # 16.Re2: the rook on a8 falls later; 16.Qxa8? Nxh3+ mates.
    (
        "the rook the line wins later",
        "r1b2rk1/2p2ppp/p2b4/1p6/3P3q/1BP2Q1P/PP3nP1/RNB1R1K1 w - - 0 16",
        ["f3a8", "f2h3", "g2h3", "h4e1", "g1g2", "e1e2", "g2g1", "e2h2"],
        0.0,
        -8,
        chess.A8,
        0.68,
        "their rook on a8 can be won, but not at once: Qxa8 right away runs into Nxh3+, and they "
        "have mate",
    ),
    # 19...Bxf2 Qxf2 wins the exchange: only worse than 19...Qh1+, in a game lost either way.
    (
        "a capture that comes out ahead",
        "r4rk1/2p2ppp/p7/1p6/3P2Pq/1BP2Qb1/PP3RP1/RNB2K2 b - - 0 19",
        ["g3f2", "f3f2", "h4h1", "f2g1", "h1h4", "b1d2", "a8e8", "d2f3"],
        0.052,
        None,
        None,
        0.25,
        None,
    ),
    # Johner-Nimzowitsch, 34.Bxe4: "34 Kxh3 allows mate after 34...Qf5+", 0.06 below the best.
    (
        "a capture into mate in a lost game",
        "6rk/5prq/1p1p1n2/p1pP3p/P1P1pP1n/2P1R1Pb/2Q4K/2B2NRB w - - 0 34",
        ["h2h3", "h7f5", "g3g4", "g7g4", "g1g4", "f5g4", "h3h2", "g4g1"],
        0.0,
        -9,
        None,
        0.06,
        "careful, though: Kxh3 looks like it wins a bishop, but after Kxh3 Qf5+, they have mate",
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "moves", "score", "mate", "target", "best", "said"),
    TRAPS,
    ids=[r[0] for r in TRAPS],
)
def test_a_trap_is_the_capture_that_looks_free_and_is_not(
    case: str,
    fen: str,
    moves: list[str],
    score: float,
    mate: int | None,
    target: int | None,
    best: float,
    said: str | None,
) -> None:
    words = situation.trap_words(chess.Board(fen), _line(moves, score, mate), target, best)
    if said is None or not said.endswith("..."):
        assert words == said, case
    else:
        assert words is not None and words.startswith(said[:-3]), f"{case}: {words}"


def test_the_wrong_piece_to_take_back_with_is_the_trap() -> None:
    """Tai's game g03, 13.Raxd1 for Rexd1: the queen taken back with the wrong rook, and after
    ...Bb4 the balance is gone. With the best move known, the trap is said as the recapture
    it is; without it, as a capture that looks free (the parity audit, 2026-09-29)."""
    board = chess.Board("r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13")
    trap = _line(["a1d1", "e7b4"], 0.3)
    assert situation.trap_words(board, trap, None, 0.54, chess.Move.from_uci("e1d1")) == (
        f"{situation.RIGHT_PIECE}, though: Raxd1 looks natural, but after Raxd1 Bb4, you are "
        "left clearly worse"
    )
    assert situation.trap_words(board, trap, None, 0.54) == (
        "careful, though: Raxd1 looks like it wins a queen, but after Raxd1 Bb4, you are left "
        "clearly worse"
    )


def test_the_trap_on_the_target_comes_first_then_mate_then_the_worst() -> None:
    """Capablanca-Marshall, 16.Re2: of the captures that lose, the one that takes the rook
    the line wins later ("not yet") is said first, then the one that walks into mate, then
    the rest by score; with no target, mate leads (the parity audit, 2026-09-29)."""
    board = chess.Board("r1b2rk1/2p2ppp/p2b4/1p6/3P3q/1BP2Q1P/PP3nP1/RNB1R1K1 w - - 0 16")
    grab = _line(["f3f2", "h4f2", "g1f2", "c8e6"], 0.35)
    mated = _line(["f3f2", "d6h2", "g1f1", "h2g1"], 0.0, mate=-4)
    early = _line(["f3a8", "f2h3", "g2h3", "h4e1", "g1g2", "e1e2", "g2g1", "e2h2"], 0.0, mate=-8)
    assert situation.ranked_traps([grab, mated, early], chess.A8) == [early, mated, grab]
    assert situation.ranked_traps([grab, mated, early], None) == [mated, early, grab]
    best = chess.Move.from_uci("e1e2")
    assert [situation.trap_words(board, t, chess.A8, 0.68, best) for t in (early, mated)] == [
        "their rook on a8 can be won, but not at once: Qxa8 right away runs into Nxh3+, and they "
        "have mate",
        "careful, though: Qxf2 looks like it wins a knight, but after Qxf2 Bh2+, they have mate",
    ]


def test_their_threat_to_the_square_the_best_move_takes_is_said_by_their_piece() -> None:
    """Lichess vgdbX: were White to pass, ...e6 would come, and then exd5 wins the queen --
    and e6 is White's own answer. "They are preparing e6" named it; the key names their pawn
    instead, never the move (the parity audit, 2026-09-29)."""
    from sidecoach.coach.facts import facts_from_analysis
    from sidecoach.events import Analysis

    fen = "1rbqkb1r/3npppp/2B5/p2QP3/8/5N2/PPP2nPP/R1B1K2R w KQk - 0 12"
    board = chess.Board(fen)
    line = [
        "e5e6", "f7e6", "d5e6", "f2g4", "f3g5", "g4h6", "c1e3", "d8c7",
        "a1d1", "g7g6", "e1g1", "b8b4", "a2a3", "b4b2", "h2h3", "b2c2",
    ]  # fmt: skip

    def key(threats: list[str]) -> str | None:
        analysis = Analysis(
            game_id="g",
            fen=fen,
            ply=0,
            lines=[_line(line, 0.905)],
            decisive=True,
            only_move=True,
            threats=threats,
        )
        return facts_from_analysis(board, analysis, last_move="e4").key

    assert key(["e7e6"]) == "Their pawn on e7 is one move from winning material"
    assert key([]) != key(["e7e6"]), "the key is their threat's"


def test_a_trap_is_worse_than_the_best_by_a_margin() -> None:
    # Johner-Nimzowitsch, 34.Kxh3: mate, 0.06 below the best, is worse; 0.02 is not.
    trap = _line(["h2h3", "h7f5", "g3g4", "g7g4", "g1g4", "f5g4", "h3h2", "g4g1"], 0.0, mate=-9)
    assert _worse(trap, _line(["h1e4"], 0.06))
    assert not _worse(_line(["h1e4"], 0.04), _line(["h1e4"], 0.06))


def test_one_good_move_is_said_in_the_singular() -> None:
    # "Only 1 moves keep you in the game" (30.Qb3 in Johner-Nimzowitsch).
    stakes = situation.Stakes("few", 1, 0.5, _line(["a2a3"], 0.40), 20)
    said = situation.stakes_words(chess.Board(), stakes, None)
    assert said is not None and said.startswith("Only one move keeps the balance here")


def test_the_stakes_in_check_say_what_the_lines_bear_out() -> None:
    # In check and lost: every way out is "within a hair" because all of them lose.
    board = chess.Board("4k3/8/8/8/8/8/3q4/4K3 w - - 0 1")
    stakes = situation.Stakes("check", 3, 0.02, None, 2)
    said = situation.stakes_words(board, stakes, None)
    assert said is not None and "fine" not in said and "losing" in said
    # One way out best, the next only 0.09 behind: not "only one way out holds".
    stakes = situation.Stakes("check", 1, 0.93, _line(["e1f1"], 0.84), 2)
    said = situation.stakes_words(board, stakes, None)
    assert said is not None and "only one" not in said
    stakes = situation.Stakes("check", 1, 0.6, _line(["e1f1"], 0.52), 2)
    said = situation.stakes_words(board, stakes, None)
    assert said is not None and "only one" not in said
    # Winning, the next still winning: "only one move keeps it that way" is false.
    stakes = situation.Stakes("won", 1, 0.99, _line(["e1f1"], 0.82), 20)
    assert situation.stakes_words(board, stakes, None) is None


def test_the_plan_words_hold_on_the_board_now() -> None:
    from sidecoach.coach.plans import LinePlan

    def key(fen: str, uci: str, kind: str) -> str:
        board = chess.Board(fen)
        said = situation.positional_key(board, chess.Move.from_uci(uci), kind, None, "endgame")
        assert said is not None
        return said

    # Rec2 with a rook already on c7: "none of your rooks is there yet" was false.
    said = key("r4k2/2Rn1p2/p2r2p1/3p4/1P3P1p/3B4/P3RKPP/8 w - - 0 28", "e2c2", "takeFile")
    assert "none of your rooks" not in said
    # Behind their passed pawn on g7, the rook holds it back; it pushes nothing.
    said = key("6k1/R5P1/1p6/3pK3/P2P2r1/8/8/8 b - - 0 46", "g4g1", "rookBehindPasser")
    assert "their passed pawn on g7" in said and "pushes" not in said
    # A king on e1 has no wing: not "the other wing from yours".
    fen = "r1bq1rk1/ppp1npbp/3pn1p1/4p3/4P2P/2PPNN2/PPB2PP1/R1BQK2R w KQ - 1 12"
    said = key(fen, "h4h5", "pawnStorm")
    assert "other wing" not in said
    storm = situation.plan_key(chess.Board(fen), LinePlan("storm", square=chess.G8))
    assert storm is not None and "opposite wings" not in storm
    # b6 hits c5 now: c5 is no square "no pawn of theirs can ever attack" yet.
    board = chess.Board("4k3/8/1p6/8/3P4/1N6/8/4K3 w - - 0 1")
    said = situation.plan_key(board, LinePlan("outpost", square=chess.C5))
    assert said is not None and "can ever attack" not in said


def test_their_idea_is_said_only_when_it_changes_something() -> None:
    board = chess.Board()
    assert situation.their_idea_key(board, chess.Move.from_uci("g8f6")) is None
    assert situation.their_idea_key(board, chess.Move.from_uci("g8f6"), sharp=True) == (
        "If you let them, their best is Nf6"
    )
    assert situation.their_idea_key(board, chess.Move.from_uci("e7e5")) == (
        "If you let them, their best is e5"
    )


def test_a_feature_is_the_point_only_of_the_move_that_acts_on_it() -> None:
    board = chess.Board()
    home = positional.PositionalFact(
        "undeveloped", "your bishop on c1 is still at home, not yet in the game", ("c1",)
    )
    # 15.Bd2 in Johner-Nimzowitsch: a4 was best, Bd2 only later in its line.
    assert situation.relevant_feature(board, [home], _moves("a2a4", "e7e5", "c1d2")) is None
    assert situation.relevant_feature(board, [home], _moves("c1d2")) == home.text
    # 13...Ng4 in Capablanca-Marshall: the queen's file is not the point of Qd7.
    asset = positional.PositionalFact(
        "halfOpenFile", "your queen on d8 has the half-open d-file", ("d8",)
    )
    assert situation.relevant_feature(board, [asset], _moves("d8d7")) is None


def test_a_knight_back_on_f1_is_not_at_home() -> None:
    # 39.Ne3 in Johner-Nimzowitsch: "2 pieces on their home squares (c1, f1)".
    board = chess.Board("6k1/8/8/8/8/8/8/2B2NK1 w - - 0 40")
    assert positional.at_home(board, chess.WHITE) == [chess.C1]


def test_the_kings_face_to_face_in_a_pawn_ending() -> None:
    board = chess.Board("8/8/5k2/8/4K3/5P2/8/8 b - - 5 51")  # Lichess R8gGn, ...Ke6
    assert situation.kings_key(board, chess.Move.from_uci("f6e6")) == (
        "The kings stand close, yours on f6 and theirs on e4: whichever has to give way loses "
        "ground"
    )


def test_their_passer_near_queening_is_the_point_of_taking_it() -> None:
    board = chess.Board("8/P3R3/6p1/5p2/5P2/1P2p1k1/r7/3K4 w - - 0 41")  # Lichess UkpFL, Rxe3+
    assert situation.passer_key(board, chess.Move.from_uci("e7e3")) == (
        "Their passed pawn on e3 is two squares from queening"
    )


def test_a_quiet_move_says_what_it_is_for() -> None:
    board = chess.Board()
    assert situation.centre_move(board, chess.Move.from_uci("f2f4")) == (
        "Which pawn move covers e5?",
        "f4: the pawn now covers e5",
    )
    assert situation.reach_why(board, chess.Move.from_uci("b1c3")) == (
        "Nc3: your knight goes from two squares to five"
    )


def test_the_pattern_to_keep_is_not_said_twice() -> None:
    # 11.f4 in Johner-Nimzowitsch: the reason said it already.
    text = (
        "e4 was better -- their pawn on e5 wants to come forward: it is worth stopping before it "
        "comes."
    )
    assert _said("their idea is worth stopping before it starts", text)
    assert not _said("more space gives your pieces room and cramps theirs", text)


def test_the_first_hint_is_said_the_way_a_coach_speaks() -> None:
    # Tai, 2026-09-25: "make it sound a bit better and more natural". Contractions, except
    # where the verb ends its clause.
    assert situation.spoken(
        "It is Boden's mate. Their pawn cannot take back, it is not defended, and you are "
        "winning -- leave the knight where it is."
    ) == (
        "It's Boden's mate. Their pawn can't take back, it isn't defended, and you're "
        "winning -- leave the knight where it is."
    )
    # A trap said first has nothing to turn from; a target named already leaves the order.
    assert situation.first_said("careful, though: Nxe5 looks like it wins a pawn") == (
        "careful: Nxe5 looks like it wins a pawn"
    )
    assert (
        situation.in_order(
            "their rook on b8 can be won, but not at once: Qxb8 right away runs into Qd2+"
        )
        == "careful with the order, though: Qxb8 right away runs into Qd2+"
    )


def test_a_long_first_hint_drops_the_details_first() -> None:
    """Ten sentences when every detail stacked up (Anderssen-Kieseritzky, move 20): past the
    budget the second pattern, the quiet start, the in-between move and the timing go, the
    least telling first; the prize, the key fact, the pattern, a sacrifice and the question
    stay."""
    text = (
        "You're in check but winning. Their queen on a1 gives check. This position has a "
        "tactic that wins a bishop. The prize is their bishop on c8. Their rook on a8 and "
        "their king on e8: one move can hit both after your first move. That is a fork: one "
        "piece attacks two of theirs at once. It comes on your second move, once a line has "
        "been cleared. There is a discovered attack in it too. It starts with a quiet move: "
        "no check and no capture, but a threat they cannot meet. Which two of their pieces "
        "could one of yours hit at once?"
    )
    short = situation.within_budget(text, budget=8)
    assert "in it too" not in short and "quiet move" not in short
    assert "It comes on your second move" in short, "dropped only as far as the budget"
    assert "The prize is their bishop on c8." in short and short.endswith("hit at once?")
    shortest = situation.within_budget(text, budget=5)
    assert "It comes on your" not in shortest and "The prize is" in shortest
    sacrifice = "That is a fork: two at once. It comes on your third move, after a sacrifice."
    assert situation.within_budget(sacrifice, budget=1) == sacrifice, "a sacrifice is said"
    assert situation.within_budget("One. Two.", budget=5) == "One. Two."
