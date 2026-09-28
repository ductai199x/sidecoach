"""From analysis to the fact sheet (`coach/facts.py`): the theme, region, piece and
candidates it names, a tactic demoted where the plan leads, the phase and the material in the
player's words, the prophylaxis wiring (a threat read off the analysis becomes the strategic
plan), and the move and tactic describers the coach leans on."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach.facts import (
    describe_move,
    describe_tactic,
    facts_from_analysis,
    material_text,
    phase_of,
    region_of,
    theme_of,
)
from sidecoach.events import Analysis, Line, MotifTag
from tests.coaching import SCHOLAR


def _analysis(fen: str, **kw: object) -> Analysis:
    base: dict[str, object] = dict(
        game_id="g",
        fen=fen,
        ply=6,
        lines=[Line(moves=["h5f7"], depth=12, expected_score=1.0, mate=1)],
        decisive=True,
        only_move=False,
    )
    base.update(kw)
    return Analysis(**base)  # type: ignore[arg-type]


def test_facts_name_the_theme_region_piece_and_candidates() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(
        SCHOLAR,
        tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")],
        threats=["f6h5"],
    )
    facts = facts_from_analysis(board, analysis, last_move="Nf6")
    assert facts.side_to_move == "white" and facts.phase == "opening"
    assert facts.theme == "a mating attack"
    assert facts.region == "the kingside, around e8", "around the king being mated"
    assert facts.piece == "your queen on h5", "the piece where it stands, never its destination"
    assert facts.candidates == ["Qxf7#"]
    assert facts.threats == ["Nxh5"], "what the opponent would play if it were their turn"
    assert facts.line == ["Qxf7#"]
    assert facts.decisive and facts.last_move == "Nf6"


def test_facts_survive_a_line_the_board_cannot_play() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(SCHOLAR, lines=[Line(moves=["a7a5", "b2b3"], depth=1, expected_score=0.5)])
    facts = facts_from_analysis(board, analysis)
    assert facts.candidates == [] and facts.piece is None and facts.line == []
    assert facts.theme is None and facts.region is None


def test_near_equal_candidates_are_all_named() -> None:
    board = chess.Board()
    analysis = _analysis(
        board.fen(),
        lines=[
            Line(moves=["e2e4"], depth=12, expected_score=0.53),
            Line(moves=["d2d4"], depth=12, expected_score=0.52),
            Line(moves=["g1f3"], depth=12, expected_score=0.50),
        ],
        decisive=False,
    )
    assert facts_from_analysis(board, analysis).candidates == ["e4", "d4"]


def test_an_incidental_tactic_is_demoted_and_the_strategic_plan_leads() -> None:
    # A pin sits on the e-file but the best move (Bf1-e2, developing) has nothing to do with
    # it and the position is not decisive, so the tactic is withheld and the plan takes over.
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 9"
    board = chess.Board(fen)
    analysis = _analysis(
        fen,
        lines=[Line(moves=["f1e2"], depth=12, expected_score=0.40)],
        decisive=False,
        only_move=False,
        tags=[MotifTag(name="pin", squares=["e5", "e4", "e1"], move=None)],
    )
    facts = facts_from_analysis(board, analysis)
    assert facts.theme is None and facts.focus == [] and facts.tactic is None
    assert facts.plan == "finishing your development"
    assert facts.ideas, "the position's static features are given as ideas"


def test_a_decisive_tactic_still_leads_and_no_plan_competes() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(
        SCHOLAR, tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")]
    )
    facts = facts_from_analysis(board, analysis, last_move="Nf6")
    assert facts.theme == "a mating attack"  # decisive: the tactic is the point
    assert facts.plan is None, "no second headline when a tactic leads"


def test_theme_ranks_the_most_specific_tag_first() -> None:
    tags = [
        MotifTag(name="capture", squares=["e5"]),
        MotifTag(name="hangingPiece", squares=["e5"]),
        MotifTag(name="fork", squares=["c7", "e8", "a8"]),
    ]
    chosen = theme_of(tags)
    assert chosen is not None and chosen.name == "fork"
    assert theme_of([MotifTag(name="quietMove", squares=["e4"])]) is None
    # Bookkeeping tags never headline: they hold on most quiet moves.
    assert theme_of([MotifTag(name="interference", squares=["e4"])]) is None
    assert theme_of([MotifTag(name="attackingF2F7", squares=["f7"])]) is None


def test_regions_and_anchors() -> None:
    assert region_of(["f7", "g8", "h7"]) == "the kingside, around f7"
    assert region_of(["b2", "a1"]) == "the queenside, around b2"
    assert region_of(["d4", "e5"]) == "the centre, around d4"
    assert region_of([]) is None
    # The anchor lies in the region the squares chose.
    assert region_of(["g4", "d7", "c8"]) == "the queenside, around d7"


def test_phase_and_material_in_the_users_words() -> None:
    assert phase_of(chess.Board()) == "opening"
    endgame = chess.Board("8/5k2/8/8/8/8/P4K2/8 w - - 0 50")
    assert phase_of(endgame) == "endgame"
    middlegame = chess.Board(
        "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 12"
    )
    assert phase_of(middlegame) == "middlegame"
    assert material_text(chess.Board()) == "level"
    assert material_text(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    ) == ("you are down a queen")
    assert material_text(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1")
    ) == ("you are up a queen")
    assert material_text(chess.Board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")) == "you are up a knight"
    # The pieces, not a sum: a bishop against a pawn is not "down two pawns".
    assert material_text(chess.Board("4k3/4p3/8/8/8/8/8/2B1K3 b - - 0 1")) == (
        "you are down a bishop for a pawn"
    )
    assert material_text(chess.Board("4k3/8/8/8/8/8/8/2B1K2R b - - 0 1")) == (
        "you are down a rook and a bishop"
    )
    assert material_text(chess.Board("2b1k3/8/8/8/8/8/8/4K2R w - - 0 1")) == (
        "you are up the exchange"
    )
    assert material_text(chess.Board("2b1k3/8/8/8/8/8/8/1N2K3 w - - 0 1")) == (
        "level in value: your knight against their bishop"
    )


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


# A tactic's squares in words (`describe_tactic`), or None when the board does not bear
# them. (case, FEN, the tag, its squares, the best move (UCI) whose board is tried after the
# one given, the sentence or None.)
RUY = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3"
DESCRIBED: list[tuple[str, str, str, list[str], str | None, str | None]] = [
    # A pin, given as [attacker, pinned, target].
    (
        "a pin",
        RUY,
        "pin",
        ["b5", "c6", "e8"],
        None,
        "the knight on c6 is pinned to the king on e8 by the bishop on b5",
    ),
    ("a loose piece", RUY, "hangingPiece", ["e5"], None, "the pawn on e5 is loose"),
    # A fork, given as [forker, then the forked].
    (
        "a fork",
        "3r1r2/8/4N3/8/8/8/8/4K1k1 w - - 0 1",
        "fork",
        ["e6", "d8", "f8"],
        None,
        "the knight on e6 forks the rook on d8 and the rook on f8",
    ),
    # When the squares do not hold what the tag claims, nothing is said.
    ("squares that hold nothing", RUY, "pin", ["a1", "a2", "a3"], None, None),
    # The fork is not on the board yet: the knight forks only after it lands, and the board
    # after the best move is tried, so it still reads true.
    (
        "a fork the best move creates",
        "5rkr/8/8/4N3/8/8/8/4K3 w - - 0 1",
        "fork",
        ["g6", "f8", "h8"],
        "e5g6",
        "the knight on g6 forks the rook on f8 and the rook on h8",
    ),
    # A master-game pass offline (2026-09-21) said "the rook on e5 bears on the pawn on d6"
    # of the player's own pawn: the capturing-defender tag's second square is where a later
    # move lands, and the board had moved on. A described target is the enemy's, and the
    # piece named must bear on it, or the square alone is named.
    # Discovered attack, given as [vacated, slider, target], read after the move only: the
    # knight steps off the rank and the rook does not see e8.
    (
        "a discovery onto nothing",
        "4k3/8/8/8/8/8/8/R2N1K2 w - - 0 1",
        "discoveredAttack",
        ["d1", "a1", "e8"],
        "d1e3",
        None,
    ),
    # A slider whose line opens onto an enemy piece.
    (
        "a discovery onto a rook",
        "r3k3/8/8/8/8/8/8/R2N3K w - - 0 1",
        "discoveredAttack",
        ["d1", "a1", "a8"],
        "d1e3",
        "the rook on a1 now bears on the rook on a8",
    ),
    # Capturing the defender: nothing is said unless an enemy piece stands on the square.
    (
        "the defender of our own pawn",
        "4k3/8/3P4/4R3/8/8/8/4K3 w - - 0 1",
        "capturingDefender",
        ["e5", "d6"],
        None,
        None,
    ),
    (
        "the defender of their pawn",
        "4k3/8/3p4/4R3/8/8/8/4K3 w - - 0 1",
        "capturingDefender",
        ["e5", "d6"],
        None,
        "the rook on e5 removes a defender of the pawn on d6",
    ),
    # A queen check that picks up pawns: the king first, the rest by value, three at most.
    (
        "a queen forking pawns",
        "8/5pQp/5p1k/8/8/1p6/8/4K3 b - - 0 1",
        "fork",
        ["g7", "f6", "h6", "f7", "h7"],
        None,
        "the queen on g7 forks the king on h6, the pawn on f6 and the pawn on f7",
    ),
    # A fork whose targets are not both the enemy's is not described.
    (
        "a fork of our own rook",
        "3r1R2/8/4N3/8/8/8/8/4K1k1 w - - 0 1",
        "fork",
        ["e6", "d8", "f8"],
        None,
        None,
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "name", "squares", "best", "said"), DESCRIBED, ids=[r[0] for r in DESCRIBED]
)
def test_describe_tactic_names_the_pieces_or_says_nothing(
    case: str, fen: str, name: str, squares: list[str], best: str | None, said: str | None
) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(best) if best else None
    assert describe_tactic(board, name, squares, best_move=move) == said, case


def test_in_check_the_question_is_the_check() -> None:
    """ "A quiet improvement -- which of your pieces has no job right now?" was the hint at
    the check positions of seven of ten AI sessions, and the students did not understand it
    (the simulated students' round 4)."""
    from sidecoach.coach.facts import IN_CHECK_NATURE, IN_CHECK_QUESTION

    # 1.e4 e5 2.Nf3 d6 3.Bb5+: Black answers the check with the quiet ...c6.
    fen = "rnbqkbnr/ppp2ppp/3p4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"
    analysis = Analysis(
        game_id="g",
        fen=fen,
        ply=5,
        lines=[Line(moves=["c7c6"], depth=12, expected_score=0.45)],
        decisive=False,
        only_move=False,
    )
    facts = facts_from_analysis(chess.Board(fen), analysis)
    assert facts.question == IN_CHECK_QUESTION
    assert facts.nature == IN_CHECK_NATURE


def test_the_nature_of_the_best_move_is_forcing_or_quiet() -> None:
    """From the question level the sheet says whether the answer is a check, a capture or a
    promotion, or a quiet improving move: it names nothing, and it stops the stuck hint
    from being the same pawn-advance homily in every position (the first live game's
    endgame had a promotion with mate to find and got "support your central pawns", four
    times over)."""
    from sidecoach.coach.disclosure import visible_facts
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
    shown = visible_facts(mate, DisclosureLevel.QUESTION)
    assert shown["nature"] == "forcing -- the answer is a check, a capture or a promotion"
    # The first hint names the board -- the target may be named -- but never the move or
    # the piece that makes it (Tai, 2026-09-24).
    assert not any("Qxf7" in str(v) or "queen" in str(v) for v in shown.values())
    # The trial game's endgame: a rook up, a pawn on d7, dxe8=Q+ leads to mate.
    fen = "1q2r2k/3P4/p6p/1pRQ2p1/8/5P2/PP4PP/5R1K w - - 1 34"
    promote = facts_from_analysis(chess.Board(fen), analysed(fen, "d7e8q"))
    assert promote.nature is not None and promote.nature.startswith("forcing")
    quiet_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
    quiet = facts_from_analysis(chess.Board(quiet_fen), analysed(quiet_fen, "b1c3"))
    assert quiet.nature is not None and quiet.nature.startswith("quiet")
    assert str(visible_facts(quiet, DisclosureLevel.QUESTION)["nature"]).startswith("quiet")


def test_a_quiet_move_whose_line_wins_material_is_a_tactic_not_quiet() -> None:
    """ "Quiet: there is no tactic to find here" was said of a won position whose best move,
    a quiet knight move, uncovered a rook on the queen: the nature looks down the line."""
    # White to move: Nh6, a quiet move, uncovers the rook on g6; the line wins the knight.
    fen = "5k2/3n1p2/b5p1/3P4/5QN1/qr1p2RP/5PPK/8 w - - 0 34"
    line = ["g4h6", "d7f6", "f4f6", "a3a2"]
    analysis = Analysis(
        game_id="g",
        fen=fen,
        ply=66,
        lines=[Line(moves=line, depth=20, expected_score=0.97)],
        decisive=True,
        only_move=True,
    )
    facts = facts_from_analysis(chess.Board(fen), analysis)
    assert facts.nature is not None and facts.nature.startswith("a tactic")
    # The same first move with no line given is judged by the move alone.
    alone = Analysis(
        game_id="g",
        fen=fen,
        ply=66,
        lines=[Line(moves=["g4h6"], depth=20, expected_score=0.97)],
        decisive=True,
        only_move=True,
    )
    assert str(facts_from_analysis(chess.Board(fen), alone).nature).startswith("quiet")
