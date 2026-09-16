"""From analysis to the fact bundle: the parts test_coach_policy.py does not already pin down.

test_coach_policy.py covers the theme/region/candidate naming, tactic demotion, phase and
material. This file covers the rest of the glue in `facts.py`: the prophylaxis wiring (a
threat read off the analysis becomes the strategic plan), and the two move/tactic describers
the coach leans on, which had no direct tests.
"""

from __future__ import annotations

import chess

from sidecoach.coach.facts import describe_move, describe_tactic, facts_from_analysis
from sidecoach.events import Analysis, Line

# 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, white to move: Qxf7 is mate.
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_a_threat_read_off_the_analysis_becomes_the_prophylaxis_plan() -> None:
    # Black threatens ...Nxe4 (the e4 pawn hangs to the f6 knight). The engine's best move,
    # Nc3, defends it. facts_from_analysis must read the null-move threat into a named threat
    # and turn the quiet defence into the prophylaxis plan -- the live path test_positional
    # only exercises with an explicit threat argument.
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    board = chess.Board(fen)
    analysis = Analysis(
        game_id="g",
        fen=fen,
        ply=6,
        lines=[Line(moves=["b1c3"], depth=12, expected_score=0.3)],
        decisive=False,
        only_move=False,
        threats=["f6e4"],
    )
    facts = facts_from_analysis(board, analysis)
    assert facts.threats == ["Nxe4"], "the null-move threat, named as the opponent would play it"
    assert facts.plan == "stopping your opponent's plan before your own"
    assert facts.theme is None, "no tactic leads, so the plan takes the headline"


def test_describe_move_reads_a_move_in_words() -> None:
    start = chess.Board()
    assert describe_move(start, chess.Move.from_uci("g1f3")) == "the knight from g1 to f3"

    castle = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    assert describe_move(castle, chess.Move.from_uci("e1g1")) == "castles kingside"

    capture = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    assert (
        describe_move(capture, chess.Move.from_uci("b5c6")) == "the bishop takes the knight on c6"
    )

    promo = chess.Board("8/P7/8/4k3/8/8/8/7K w - - 0 1")
    assert describe_move(promo, chess.Move.from_uci("a7a8q")) == (
        "the pawn from a7 to a8, promoting to a queen"
    )

    en_passant = chess.Board("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3")
    assert describe_move(en_passant, chess.Move.from_uci("e5f6")) == (
        "the pawn takes the pawn en passant on f6"
    )

    assert describe_move(chess.Board(SCHOLAR), chess.Move.from_uci("h5f7")) == (
        "the queen takes the pawn on f7, checkmate"
    )


def test_describe_tactic_names_the_pieces_or_says_nothing() -> None:
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3")
    # A pin, given as [attacker, pinned, target].
    assert describe_tactic(board, "pin", ["b5", "c6", "e8"]) == (
        "the knight on c6 is pinned to the king on e8 by the bishop on b5"
    )
    # A loose piece.
    assert describe_tactic(board, "hangingPiece", ["e5"]) == "the pawn on e5 is loose"
    # A fork, given as [forker, then the forked].
    fork = chess.Board("3r1r2/8/4N3/8/8/8/8/4K1k1 w - - 0 1")
    assert describe_tactic(fork, "fork", ["e6", "d8", "f8"]) == (
        "the knight on e6 forks the rook on d8 and the rook on f8"
    )
    # When the squares do not hold what the tag claims, nothing is said.
    assert describe_tactic(board, "pin", ["a1", "a2", "a3"]) is None


def test_describe_tactic_reads_a_fork_the_best_move_creates() -> None:
    # The tactic is not on the board yet: the knight forks only after it lands. describe_tactic
    # is handed the best move and tries the position after it, so the fork still reads true.
    board = chess.Board("5rkr/8/8/4N3/8/8/8/4K3 w - - 0 1")
    best = chess.Move.from_uci("e5g6")  # Ng6 forks the rooks on f8 and h8
    text = describe_tactic(board, "fork", ["g6", "f8", "h8"], best_move=best)
    assert text == "the knight on g6 forks the rook on f8 and the rook on h8"


def test_the_nature_of_the_best_move_is_forcing_or_quiet() -> None:
    """From the question level the sheet says whether the answer is a check, a capture or a
    promotion, or a quiet improving move: it names nothing, and it stops the stuck hint
    from being the same pawn-advance homily in every position (the first live game's
    endgame had a promotion with mate to find and got "support your central pawns", four
    times over)."""
    from sidecoach.coach.coach import fact_sheet
    from sidecoach.events import DisclosureLevel

    def analysed(fen: str, best: str) -> Analysis:
        return Analysis(
            game_id="g",
            fen=fen,
            ply=6,
            lines=[Line(moves=[best], depth=12, expected_score=0.8)],
            decisive=False,
            only_move=False,
        )

    mate = facts_from_analysis(chess.Board(SCHOLAR), analysed(SCHOLAR, "h5f7"))
    assert mate.nature is not None and mate.nature.startswith("forcing")
    sheet = fact_sheet(mate, DisclosureLevel.QUESTION)
    assert "nature: forcing: the answer is a check, a capture or a promotion" in sheet
    assert "f7" not in sheet.split("Fact sheet:")[1] and "queen" not in sheet
    # The trial game's endgame: a rook up, a pawn on d7, dxe8=Q+ leads to mate.
    fen = "1q2r2k/3P4/p6p/1pRQ2p1/8/5P2/PP4PP/5R1K w - - 1 34"
    promote = facts_from_analysis(chess.Board(fen), analysed(fen, "d7e8q"))
    assert promote.nature is not None and promote.nature.startswith("forcing")
    quiet_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    quiet = facts_from_analysis(chess.Board(quiet_fen), analysed(quiet_fen, "b1c3"))
    assert quiet.nature is not None and quiet.nature.startswith("quiet")
    assert "nature: quiet" in fact_sheet(quiet, DisclosureLevel.QUESTION)
