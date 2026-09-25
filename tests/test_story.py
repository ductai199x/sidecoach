"""The hints tell one story (`coach/story.py`), held on Lichess puzzles read by hand on
2026-09-24, with the engine's lines as the app's settings saw them; and the story checks
(`tools/puzzles/story_checks.py`) pass on each."""

from __future__ import annotations

import sys
from pathlib import Path

import chess

from sidecoach.coach import idea as idea_of
from sidecoach.coach.coach import focus_template, idea_template, move_template
from sidecoach.coach.facts import facts_from_analysis
from sidecoach.events import Analysis, Line

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "puzzles"))
from story_checks import story_checks  # noqa: E402


def _hints(
    fen: str, lines: list[tuple[list[str], float]], threats: tuple[str, ...] = ()
) -> tuple[dict[str, str], bool]:
    board = chess.Board(fen)
    analysis = Analysis(
        game_id="",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=18, expected_score=score) for moves, score in lines],
        decisive=False,
        only_move=False,
        threats=list(threats),
    )
    facts = facts_from_analysis(board, analysis, user=board.turn)
    texts = {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "move": move_template(facts),
    }
    pv = [chess.Move.from_uci(u) for u in lines[0][0]]
    checks, _ = story_checks(board, pv, idea_of.read(board, analysis), texts)
    return texts, all(checks.values())


def test_a_queen_that_simply_hangs_is_said_plainly() -> None:
    # gNzJg: a pawn takes the queen -- no discovered attack about it.
    texts, told = _hints(
        "5r1k/p5pp/1p6/3p1r2/3P2q1/2P1QP1n/PP5P/R4R1K w - - 0 29",
        [(["f3g4", "h3f2", "h1g1", "f5f3", "e3f2", "f3f2", "f1f2", "f8f2", "g1f2"], 0.949)],
    )
    assert "Their queen on g4 can simply be taken: nothing defends it" in texts["idea"]
    assert "discovered" not in texts["idea"]
    assert texts["move"].startswith("fxg4 takes their queen")
    assert told


def test_a_free_piece_first_then_what_the_line_wins_after_it() -> None:
    # DWvS7: Qxe8+ takes a free rook, and the line wins more after it.
    texts, told = _hints(
        "4r1k1/6p1/3qp2p/1Q1pr3/p1p1nP2/Pp6/1P4PP/4RRK1 w - - 0 29",
        [(["b5e8", "g8h7", "f4e5", "d6c5", "g1h1", "e4f2", "f1f2", "c5f2", "e1g1"], 0.762)],
    )
    assert "Their rook on e8 can simply be taken" in texts["idea"]
    assert told


def test_in_check_the_checker_comes_first() -> None:
    # EcCX4: their queen gives check, and taking it is the point.
    texts, told = _hints(
        "4r3/5ppp/5n2/2Q2k2/P1qP1B2/7P/1P2p1P1/4R1K1 b - - 4 30",
        [
            (["c4c5", "d4c5", "f5f4", "g1f2", "f6e4", "f2e2", "e4c5", "e2d2", "e8e1"], 0.911),
            (["c4d5", "f4d2", "d5c5", "d4c5", "e8b8", "g2g4", "f5e4", "e1e2", "e4d3"], 0.278),
        ],
    )
    assert "Their queen on c5 gives check" in texts["idea"]
    assert "the first move takes their queen on c5" in texts["idea"]
    assert told


def test_the_pawn_the_win_rests_on_is_named() -> None:
    # uSAIW: f7, and the rook must give itself up for the pawn.
    texts, told = _hints(
        "5k2/p7/5PRp/1p6/7p/8/r2r2P1/5RK1 w - - 0 36",
        [
            (["f6f7", "d2g2", "g6g2", "a2g2", "g1g2", "b5b4", "g2h3", "b4b3", "h3h4"], 0.88),
            (["g6g7", "a7a5", "g1h2", "h4h3", "h2h3", "d2f2", "f1c1", "a2c2", "c1a1"], 0.454),
        ],
    )
    assert "Your pawn on f6 is two squares from queening" in texts["idea"]
    assert told


def test_a_win_with_no_pattern_says_how() -> None:
    # jlCZP, a zugzwang: never "their pawn on h5 is the target" and "you win two pawns" alone.
    texts, told = _hints(
        "8/8/p3k3/1p3p1p/1P6/P3K1PP/8/8 w - - 1 41",
        [
            (["e3f4", "e6f6", "h3h4", "f6e6", "f4g5", "e6d5", "g5h5", "a6a5", "b4a5"], 0.895),
            (["e3d4", "e6d6", "h3h4", "d6e6", "d4e3", "e6e7", "e3f4", "e7f6", "f4f3"], 0.5),
        ],
        threats=("e6e5",),
    )
    assert "Their pawn on h5 is the target, and the first move stops Ke5." in texts["idea"]
    assert texts["move"].startswith("Kf4") and "and after" in texts["move"]
    assert told


def test_where_only_one_move_holds_the_move_says_what_it_does() -> None:
    # TzhLp: "The move is gxh5." said nothing.
    texts, told = _hints(
        "8/4k3/3p1pp1/7p/4PKPP/5P2/8/8 w - - 1 44",
        [
            (["g4h5", "g6h5", "f4f5", "e7f7", "f3f4", "f7e7", "f5g6", "e7e6", "f4f5"], 0.862),
            (["f4e3", "e7f7", "e3f4", "f7e6", "g4g5", "e6f7", "g5f6", "f7f6", "f4e3"], 0.512),
            (["g4g5", "e7f7", "g5f6", "f7f6", "f4e3", "g6g5", "h4g5", "f6g5", "f3f4"], 0.512),
        ],
    )
    assert "takes their pawn on h5" in texts["move"]
    assert "?" in texts["focus"]
    assert told


def test_a_piece_with_no_safe_square_is_taught_as_trapped() -> None:
    # Tai's chess.com puzzle: Bg4 against a queen on h5 with nowhere to go; the hint said only
    # "the first move attacks their queen on h5".
    texts, told = _hints(
        "r1bq1k1r/1p2bp2/p2p3p/2p1n2Q/P3P3/1BN5/1PP2PPP/R1B2RK1 b - - 1 15",
        [
            (["c8g4", "b3f7", "g4h5", "f7h5", "b7b5", "c3d5", "b5b4", "f2f4"], 0.797),
            (["c5c4", "h2h3", "c4b3", "c2b3", "e7e6", "a1a3", "a8c8", "c1e3"], 0.63),
        ],
    )
    assert "The first move attacks it, and it has no safe square to go to" in texts["idea"]
    assert "That's a trapped piece" in texts["idea"]
    assert "no safe square" in texts["focus"]
    assert told


def _hints_with(
    fen: str,
    lines: list[tuple[list[str], float]],
    defences: list[Line],
    traps: list[Line] = (),
    threatens: list[Line] = (),
) -> tuple[dict[str, str], bool]:
    board = chess.Board(fen)
    analysis = Analysis(
        game_id="",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=18, expected_score=score) for moves, score in lines],
        decisive=True,
        only_move=True,
        defences=defences,
        traps=list(traps),
        threatens=list(threatens),
    )
    facts = facts_from_analysis(board, analysis, user=board.turn)
    texts = {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "move": move_template(facts),
    }
    pv = [chess.Move.from_uci(u) for u in lines[0][0]]
    checks, _ = story_checks(board, pv, idea_of.read(board, analysis), texts)
    return texts, all(checks.values())


def test_why_the_obvious_recapture_fails_is_the_lesson() -> None:
    # Tai's chess.com puzzle, 2026-09-24: 1...Qxg3 and White does not take back -- 2.fxg3
    # Bc5+ loses more. "You can win their queen" alone read as a lie beside fxg3.
    texts, told = _hints_with(
        "2kr1b1r/ppq2p2/2p1pp2/8/2B5/3P2Q1/PPPB1PP1/R4RK1 b - - 1 1",
        [
            (["c7g3", "f1e1", "g3h4", "g1f1", "h4g4", "f1g1"], 0.975),
            (["f8d6", "f2f4", "f7f5", "f1f3", "c7b6", "g3f2"], 0.267),
        ],
        [
            Line(
                moves=["c7g3", "f2g3", "f8c5", "d3d4", "c5d4", "f1f2", "d4f2"],
                depth=14,
                expected_score=0.931,
            )
        ],
    )
    # Said as a coach speaks (Tai, 2026-09-25): a tactic to find, the target and the defender
    # that cannot take back as one sentence, the pattern said of it.
    assert "tactic that wins their queen." in texts["idea"]
    assert (
        "Their queen on g3 is the target, and their pawn on f2 can't take back there: it also "
        "shields their king. That's an overloaded defender: one piece with two jobs can't do "
        "both." in texts["idea"]
    )
    assert "Which of their pieces has two jobs?" in texts["focus"]
    assert "if fxg3, Bc5+" in texts["move"]
    assert told


def test_a_defender_with_two_jobs_is_not_a_loose_piece() -> None:
    # Tai's chess.com puzzle, 2026-09-24: 2.exf7+ Kh8 3.Qxb8 wins the rook because the rook
    # on f8 must stop f8=Q; "a loose piece: nothing defends it" was the wrong lesson, and
    # 2.Qxb8? at once (Qd2+) was the natural mistake, unwarned.
    texts, told = _hints_with(
        "1r3rk1/pp3ppp/2pQP3/1P1pP1q1/3Nn3/6P1/P5KP/2R2R2 w - - 1 2",
        [
            (["e6f7", "g8h8", "d6b8", "g5d2", "g2h1", "d2b4"], 0.975),
            (["d6c7", "g5d2", "g2h1", "f7e6", "d4e6"], 0.261),
        ],
        [
            Line(
                moves=["e6f7", "f8f7", "d6b8", "g5d8", "b8d8", "f7f8", "d8f8"],
                depth=14,
                expected_score=1.0,
                mate=3,
            ),
            Line(
                moves=["e6f7", "g8h8", "d6b8", "f8b8", "f7f8q", "b8f8", "f1f8"],
                depth=14,
                expected_score=1.0,
                mate=2,
            ),
        ],
        [Line(moves=["d6b8", "g5d2", "g2h1", "f8b8", "d4f3"], depth=14, expected_score=0.09)],
    )
    assert (
        "Your pawn on e6 is two squares from queening, and their rook on f8 has to stop it, so it "
        "can't also take back on b8." in texts["idea"]
    )
    assert "loose" not in texts["idea"] and "no defender" not in texts["idea"]
    assert (
        "Their rook on b8 can be won, but not at once: Qxb8 right away runs into Qd2+"
        in texts["idea"]
    )
    assert "if Rxb8, f8=Q+" in texts["move"]
    assert told


def test_a_quiet_move_is_told_by_what_it_threatens() -> None:
    # The 28 wins of the puzzle sample whose first hint was "their X is the target" and no
    # more (2026-09-25): a quiet first move's point is its threat, asked of the engine
    # (`threats.aim_lines`). Lichess 06VoP: 32.Qg4, threatening Qxg7#.
    texts, told = _hints_with(
        "2rr2k1/5pp1/p1qp3p/4pP1N/1p6/1P4PK/PP5P/R2Q4 w - - 5 32",
        [
            (["d1g4", "g8f8", "g4g7", "f8e8", "g7g8", "e8e7", "f5f6", "e7e6"], 0.924),
            (["d1f1"], 0.241),
        ],
        [],
        threatens=[Line(moves=["g4g7"], depth=14, expected_score=1.0, mate=1)],
    )
    assert "The first move is quiet, but it threatens mate." in texts["idea"]
    assert "Which quiet move sets up a mate?" in texts["focus"]
    assert told


def test_a_quiet_move_that_threatens_a_piece_names_it_where_it_stands() -> None:
    # Lichess 05cb6: in check, 35.Kd2 steps out and threatens Rxb1, their queen.
    texts, told = _hints_with(
        "r4rk1/p3pp1p/1p4p1/3P4/2P4b/1Q3R2/PP2N3/1q1K2R1 w - - 3 35",
        [
            (["d1d2", "b1e4", "f3f4", "e4e5", "f4h4", "e7e6", "b3c3", "e5c3"], 0.797),
            (["e2c1", "a8c8", "g1g4", "h4f6", "f3f6", "e7f6", "b3c2", "b1c2"], 0.093),
        ],
        [],
        threatens=[
            Line(
                moves=["g1b1", "e7e6", "b1h1", "a7a5", "h1h4", "a5a4"],
                depth=14,
                expected_score=0.959,
            )
        ],
    )
    assert "The first move is quiet, but it threatens to take their queen on b1." in texts["idea"]
    assert told


def test_a_quiet_pawn_move_threatens_a_break() -> None:
    # Lichess 00pWF, a king and pawn ending: 53...a5, threatening a4 and a passed pawn.
    texts, told = _hints_with(
        "8/8/p3k3/1p1pP3/5P2/1P3K2/8/8 b - - 1 53",
        [
            (
                [
                    "a6a5",
                    "f3g4",
                    "e6f7",
                    "f4f5",
                    "a5a4",
                    "b3a4",
                    "b5a4",
                    "f5f6",
                    "f7g6",
                    "f6f7",
                    "g6f7",
                    "g4f5",
                    "f7e7",
                    "e5e6",
                    "a4a3",
                    "f5e5",
                ],
                0.874,
            ),
            (["b5b4"], 0.456),
            (
                [
                    "e6f5",
                    "b3b4",
                    "a6a5",
                    "b4a5",
                    "b5b4",
                    "f3e3",
                    "b4b3",
                    "e3d3",
                    "b3b2",
                    "d3c2",
                    "f5e4",
                ],
                0.089,
            ),
        ],
        [],
        threatens=[
            Line(
                moves=["a5a4", "b3a4", "b5a4", "f3e3", "a4a3", "f4f5"],
                depth=14,
                expected_score=0.872,
            )
        ],
    )
    assert "The first move is quiet, but it threatens a pawn break." in texts["idea"]
    assert told
