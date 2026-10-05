"""The harness says only what is true of the board: each cause of a false claim the v2 audit
traced to the harness (dev and test reruns, 2026-09-26), on the position it was found in."""

from __future__ import annotations

import chess
import pytest

from sidecoach.analysis.computed import Computed, ComputedFirst
from sidecoach.coach import situation
from sidecoach.coach.coach import _great_meaning, in_check_threat
from sidecoach.coach.conversation import _also_stopped
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.coach.effects import describe_effects, line_notes
from sidecoach.coach.engine_tools import EngineTools
from sidecoach.coach.facts import _threatens
from sidecoach.coach.idea import Idea
from sidecoach.coach.idea_words import (
    PROMOTE_HELP_NUDGE,
    PROMOTE_HELP_QUESTION,
    clears_path,
    nudge,
    question,
)
from sidecoach.coach.named import Probe, examine, examine_piece, parse_probe, spoken
from sidecoach.events import DisclosureLevel, Line


def _facts(fen: str, **kw: object) -> PositionFacts:
    return PositionFacts(fen=fen, side_to_move="white", phase="middlegame", **kw)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fen", "san", "why"),
    [
        # 27...Kh7: every legal king move lands where nothing attacks it, so "a square nothing
        # of theirs can reach" read as the reason for Kh7 over Kf7, Kh5 and Kf5.
        (
            "2r5/3r2p1/b1N1R1kp/3n4/5B2/5P2/PP3KPP/2R5 b - - 4 27",
            "Kh7",
            "Kh7 steps out of check",
        ),
        # Kd2 walks into Nb3+; Kc2 leaves them no check at all.
        (
            "8/8/6P1/8/8/8/7k/2nK3r w - - 0 1",
            "Kc2",
            "Kc2 steps out of check to the one square where no check follows",
        ),
        ("8/8/6P1/8/8/8/7k/2nK3r w - - 0 1", "Kd2", "Kd2 steps out of check"),
    ],
    ids=["every king move is safe", "the one square with no check", "into a check"],
)
def test_a_king_step_claims_only_what_sets_it_apart(fen: str, san: str, why: str):
    board = chess.Board(fen)
    assert situation.evasion_why(board, board.parse_san(san)) == why


def test_a_move_after_a_pass_is_a_threat_only_when_it_threatens():
    # 39...: White's best free move, Rc3, guards the third rank against the mate; it attacks
    # nothing, and was said as "They threaten Rc3".
    board = chess.Board("5rk1/B5pp/8/2R5/8/6KP/4q3/8 b - - 7 39")
    passed = board.copy()
    passed.push(chess.Move.null())
    assert not _threatens(passed, passed.parse_san("Rc3"))
    assert _threatens(passed, passed.parse_san("Rc8"))  # attacks the rook on f8
    # A quiet move whose threat is a mate on an empty square still counts.
    quiet = chess.Board("7k/5Qpp/8/8/8/8/8/6K1 b - - 0 1")
    quiet.push(chess.Move.null())
    assert _threatens(quiet, quiet.parse_san("Qe7"))


def test_in_check_the_threat_is_the_check_and_what_else_it_wins():
    # 12.Qxe6+: "Nothing is threatened" with the queen also hitting the loose pawn on c6;
    # the defended bishop on c8 and pawn on d5 are no threat.
    board = chess.Board("r1bqk2r/p1p4p/1bp1Qpp1/3p4/1P1P1B1P/2P1P3/P4PP1/RN2KB1R b KQkq - 0 12")
    assert in_check_threat(board) == (
        "Their queen on e6 gives check, and it also attacks your pawn on c6."
    )


def test_a_great_move_says_which_kind_of_great():
    fen = chess.STARTING_FEN
    assert _great_meaning(_facts(fen, only_move=True)).startswith("the only good move")
    # Not the only move: the best reply to their error (29...Qg5 after 29.Rc1, a miss), never
    # "the one move that keeps the advantage".
    assert _great_meaning(_facts(fen)).startswith("the chance taken")


def test_the_lesson_says_when_the_move_played_stopped_the_threat_too():
    # 19.Qxg2 took the pawn that threatened gxh1=Q+; Qc4+ "stops gxh1=Q+" was taught as
    # "the priority was to stop the promotion threat".
    faced = chess.Board("1r1q3r/pPp2kbp/6p1/8/1P2QP2/P3P3/6pP/1NB1K2R w K - 2 19")
    facts = _facts(
        faced.fen(),
        threats=["gxh1=Q+"],
        why="Qc4+ gives check and stops gxh1=Q+: it is the only move that keeps the balance",
    )
    played = faced.copy()
    played.push_san("Qxg2")
    assert _also_stopped(facts, played) == "gxh1=Q+"
    # A move that left the threat on says nothing of the kind.
    other = faced.copy()
    other.push_san("Qe5")
    assert _also_stopped(facts, other) is None


class _NoSearch:
    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        raise AssertionError("a move the position's own lines play needs no second search")

    def close(self) -> None:
        return


def test_a_named_move_the_lines_play_is_read_from_them():
    # 64.Bd2: engine line 2 had Rxh3, a second search after Bd2 had Kh6 -- and the model,
    # shown both, said Bd2 avoided Rxh3.
    board = chess.Board("8/7k/6p1/8/4KPP1/4B2P/8/7r w - - 5 64")
    known = [
        Line(moves=["f4f5", "h1h3", "e3g5"], depth=20, expected_score=0.50),
        Line(moves=["e3d2", "h1h3", "f4f5", "h3g3"], depth=20, expected_score=0.50),
    ]
    # The look-ahead's own search after Bd2, which answers it with Kh6: the sheet's line wins.
    after = board.copy()
    after.push_san("Bd2")
    computed = Computed()
    computed.add(after, [Line(moves=["h7h6", "d2e3"], depth=18, expected_score=0.48)])
    text = examine(
        parse_probe("your Bd2", ["Bd2"], board),
        board,
        ComputedFirst(_NoSearch(), computed),
        known[0],
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
        known=known,
    )
    assert "the opponent's reply would be 64... Rxh3" in text
    assert "0.50 from the player's side" in text


RXF1_CHECK = "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4"


def test_a_square_says_whether_a_piece_of_theirs_can_get_there():
    # Tai's puzzle, 3...Rxf1+: "Why can't I block on g1?" -- told only who controls g1, Luna
    # answered that g1 was empty. No piece of White's can go there; after Kg2 two of Black's can.
    board = chess.Board(RXF1_CHECK)
    fact = examine_piece(
        Probe(label="g1", moves=[], square=chess.G1), board, user_colour=chess.WHITE
    )
    assert fact.endswith("; none of the player's pieces can move there.")
    assert spoken(fact).endswith("; none of your pieces can move there.")
    tools = EngineTools(board, None, _NoSearch(), chess.WHITE)
    assert tools.run("square", {"square": "g1"}).endswith(
        "None of the player's pieces can move there."
    )
    assert tools.run("square", {"square": "g1", "moves": ["Kg2"]}).endswith(
        "The opponent can move the rook on f1 or the bishop on a7 there."
    )


def test_a_piece_got_out_of_attack_names_what_attacked_it():
    # The engine's line 4. Kg2 Rg1+ 5. Kf3 g2 6. Nb2: told "gets the knight out of attack",
    # Luna said 5...g2 attacked the knight -- the rook on g1 did.
    board = chess.Board(RXF1_CHECK)
    notes = line_notes(board, ["h1g2", "f1g1", "g2f3", "g3g2", "d1b2"], player=chess.WHITE)
    assert notes.endswith("the player's Nb2 (gets the knight out of attack from the rook on g1)")


def test_a_passed_pawn_is_told_to_clear_its_way_only_when_the_move_does():
    # 57...Re5: the rook stands behind the a-pawn, White's rook on a3 in front of it; "Which
    # move clears its way?" was taught as "that rook is blocking your passed pawn's route".
    board = chess.Board("8/8/1K4pp/r7/p5Pk/R6P/8/8 b - - 10 57")
    line = tuple(
        chess.Move.from_uci(u)
        for u in ("a5e5", "b6c7", "e5b5", "a3c3", "a4a3", "c3c1", "a3a2", "c7d6", "a2a1q")
    )
    behind = Idea(goal="promote", first=line[0], line=line)
    assert not clears_path(behind, board)
    assert question(behind, None, board) == PROMOTE_HELP_QUESTION
    assert nudge(behind, board) == PROMOTE_HELP_NUDGE
    # A knight that steps off the pawn's square in front of it does clear the way.
    ahead = chess.Board("4k3/1N6/1P6/8/8/8/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("b7d6")
    lane = (move, chess.Move.from_uci("e8e7"), chess.Move.from_uci("b6b7"))
    lane += (chess.Move.from_uci("e7d7"), chess.Move.from_uci("b7b8q"))
    assert clears_path(Idea(goal="promote", first=move, line=lane), ahead)


def test_a_knight_is_freed_never_uncovered():
    # 25...Ke7 stepped off the d-file, and the knight on d6, pinned there by the rook on d1,
    # could take on e4 again: "uncovers an attack ... by the knight on d6" was said back.
    board = chess.Board("7r/1p1k2p1/1Ppnbp2/r3p3/NRP1P2p/pK3P2/P3B1PP/3R4 b - - 2 25")
    # (A piece that stood in the line does uncover it: test_effects_legal.py's discoveries.)
    effects = describe_effects(board, board.parse_san("Ke7"))
    assert "frees the knight on d6 to attack the pawn on e4" in effects
