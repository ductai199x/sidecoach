"""The first hint's parts (`coach/situation.py`, `analysis/traps.py`), held to the moments of
the master games it was read against on 2026-09-24 (Capablanca-Marshall, Johner-Nimzowitsch)
and to the Lichess puzzles that caught its first slips."""

from __future__ import annotations

import chess

from sidecoach.analysis import positional
from sidecoach.analysis.traps import _worse
from sidecoach.coach import idea as idea_of
from sidecoach.coach import situation
from sidecoach.coach.coach import _said
from sidecoach.events import Analysis, Line


def _line(moves: list[str], score: float, mate: int | None = None) -> Line:
    return Line(moves=moves, depth=18, expected_score=score, mate=mate)


def _moves(*uci: str) -> list[chess.Move]:
    return [chess.Move.from_uci(u) for u in uci]


def test_the_bait_knight_is_a_trap() -> None:
    # Capablanca-Marshall, 14.Qf3: "14 hxg4? Qh4" and Black's attack comes.
    board = chess.Board("r1bq1rk1/2p2ppp/p2b4/1p6/6n1/1BP4P/PP1P1PP1/RNBQR1K1 w - - 1 14")
    trap = _line(["h3g4", "d8h4", "d1f3", "h4h2", "g1f1", "h2h1", "f1e2", "f8e8"], 0.45)
    said = situation.trap_words(board, trap, None, 0.60)
    assert said is not None
    assert said.startswith("careful, though: hxg4 looks like it wins a knight, but after hxg4 Qh4")


def test_the_rook_the_line_wins_later_is_not_yet() -> None:
    # 16.Re2: the rook on a8 falls later; 16.Qxa8? Nxh3+ mates.
    board = chess.Board("r1b2rk1/2p2ppp/p2b4/1p6/3P3q/1BP2Q1P/PP3nP1/RNB1R1K1 w - - 0 16")
    trap = _line(["f3a8", "f2h3", "g2h3", "h4e1", "g1g2", "e1e2", "g2g1", "e2h2"], 0.0, mate=-8)
    assert situation.trap_words(board, trap, chess.A8, 0.68) == (
        "their rook on a8 can be won, but not at once: Qxa8 right away runs into Nxh3+, and they "
        "have mate"
    )


def test_a_capture_that_comes_out_ahead_is_no_trap() -> None:
    # 19...Bxf2 Qxf2 wins the exchange: only worse than 19...Qh1+, in a game lost either way.
    board = chess.Board("r4rk1/2p2ppp/p7/1p6/3P2Pq/1BP2Qb1/PP3RP1/RNB2K2 b - - 0 19")
    trap = _line(["g3f2", "f3f2", "h4h1", "f2g1", "h1h4", "b1d2", "a8e8", "d2f3"], 0.052)
    assert situation.trap_words(board, trap, None, 0.25) is None


def test_a_capture_into_mate_is_a_trap_in_a_lost_game() -> None:
    # Johner-Nimzowitsch, 34.Bxe4: "34 Kxh3 allows mate after 34...Qf5+", 0.06 below the best.
    board = chess.Board("6rk/5prq/1p1p1n2/p1pP3p/P1P1pP1n/2P1R1Pb/2Q4K/2B2NRB w - - 0 34")
    trap = _line(["h2h3", "h7f5", "g3g4", "g7g4", "g1g4", "f5g4", "h3h2", "g4g1"], 0.0, mate=-9)
    assert _worse(trap, _line(["h1e4"], 0.06))
    assert not _worse(_line(["h1e4"], 0.04), _line(["h1e4"], 0.06))
    assert situation.trap_words(board, trap, None, 0.06) == (
        "careful, though: Kxh3 looks like it wins a bishop, but after Kxh3 Qf5+, they have mate"
    )


def test_material_given_for_their_attack_is_not_won() -> None:
    # 29.exf4 in a lost game: the knight Black gives with ...Ng4+ is not "won back".
    board = chess.Board("6rk/3b1prq/1p1p1n2/p1pP1n1p/P1P1pp2/2P1P1PP/R6K/2BQ1NRB w - - 0 29")
    moves = ["e3f4", "f6g4", "h3g4", "h5h4", "h1e4", "h4g3", "h2g2", "f5h4", "g2h3", "d7g4"]
    analysis = Analysis(
        game_id="",
        fen=board.fen(),
        ply=0,
        lines=[_line(moves, 0.055)],
        decisive=False,
        only_move=False,
    )
    assert idea_of.read(board, analysis).goal != "win"


def test_one_good_move_is_said_in_the_singular() -> None:
    # "Only 1 moves keep you in the game" (30.Qb3 in Johner-Nimzowitsch).
    stakes = situation.Stakes("few", 1, 0.5, _line(["a2a3"], 0.40), 20)
    said = situation.stakes_words(chess.Board(), stakes, None)
    assert said is not None and said.startswith("Only one move keeps the balance here")


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
        "Nc3: your knight goes from 2 squares to 5"
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
