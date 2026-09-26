"""The harness says only what is true of the board: each cause of a false claim the v2 audit
traced to the harness (dev and test reruns, 2026-09-26), on the position it was found in."""

from __future__ import annotations

import chess

from sidecoach.coach import situation
from sidecoach.coach.coach import _great_meaning, in_check_threat
from sidecoach.coach.conversation import _also_stopped
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.coach.effects import describe_effects
from sidecoach.coach.facts import _threatens
from sidecoach.coach.idea import Idea
from sidecoach.coach.idea_words import (
    PROMOTE_HELP_NUDGE,
    PROMOTE_HELP_QUESTION,
    clears_path,
    nudge,
    question,
)
from sidecoach.coach.named import examine, parse_probe
from sidecoach.events import DisclosureLevel, Line


def _facts(fen: str, **kw: object) -> PositionFacts:
    return PositionFacts(fen=fen, side_to_move="white", phase="middlegame", **kw)  # type: ignore[arg-type]


def test_a_king_step_claims_nothing_every_king_move_has():
    # 27...Kh7: every legal king move lands where nothing attacks it, so "a square nothing of
    # theirs can reach" read as the reason for Kh7 over Kf7, Kh5 and Kf5.
    board = chess.Board("2r5/3r2p1/b1N1R1kp/3n4/5B2/5P2/PP3KPP/2R5 b - - 4 27")
    assert situation.evasion_why(board, board.parse_san("Kh7")) == "Kh7 steps out of check"


def test_a_king_step_to_the_one_square_without_a_check_says_so():
    # Kd2 walks into Nb3+; Kc2 leaves them no check at all.
    board = chess.Board("8/8/6P1/8/8/8/7k/2nK3r w - - 0 1")
    assert situation.evasion_why(board, board.parse_san("Kc2")) == (
        "Kc2 steps out of check to the one square where no check follows"
    )
    assert situation.evasion_why(board, board.parse_san("Kd2")) == "Kd2 steps out of check"


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
    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        raise AssertionError("a move the position's own lines play needs no second search")


def test_a_named_move_the_lines_play_is_read_from_them():
    # 64.Bd2: engine line 2 had Rxh3, a second search after Bd2 had Kh6 -- and the model,
    # shown both, said Bd2 avoided Rxh3.
    board = chess.Board("8/7k/6p1/8/4KPP1/4B2P/8/7r w - - 5 64")
    known = [
        Line(moves=["f4f5", "h1h3", "e3g5"], depth=20, expected_score=0.50),
        Line(moves=["e3d2", "h1h3", "f4f5", "h3g3"], depth=20, expected_score=0.50),
    ]
    text = examine(
        parse_probe("your Bd2", ["Bd2"], board),
        board,
        _NoSearch(),
        known[0],
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
        known=known,
    )
    assert "the opponent's reply would be 64... Rxh3" in text
    assert "0.50 from the player's side" in text


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
    effects = describe_effects(board, board.parse_san("Ke7"))
    assert "frees the knight on d6 to attack the pawn on e4" in effects
    # A piece that stood in the line does uncover it.
    line = chess.Board("7r/8/8/8/3N4/8/1B6/K6k w - - 0 1")
    effects = describe_effects(line, line.parse_san("Nb5"))
    assert "uncovers an attack on the rook on h8 by the bishop on b2" in effects
