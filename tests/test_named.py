"""The engine on what the player asks about: probes, their examination, the spoken form,
and the frames along a disclosed line."""

from __future__ import annotations

import chess

from sidecoach.coach.claims import check_draft
from sidecoach.coach.named import (
    disclosed_frames,
    examine,
    parse_probe,
    probes_by_words,
    spoken,
)
from sidecoach.events import DisclosureLevel, Line

ITALIAN_START = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


class _Engine:
    """Scripted lines keyed by the position's FEN board part."""

    def __init__(self, lines: dict[str, list[Line]]) -> None:
        self.lines = lines

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        return self.lines.get(board.board_fen(), [])


def test_words_name_a_line_as_far_as_it_is_legal() -> None:
    board = chess.Board(ITALIAN_START)
    probes = probes_by_words("what about Bc4 Bc5 then d3?", board)
    assert len(probes) == 1
    assert probes[0].moves == ["Bc4", "Bc5", "d3"] and probes[0].illegal is None
    assert probes[0].label == "3. Bc4 Bc5 4. d3"


def test_an_illegal_move_ends_the_probe_and_is_remembered() -> None:
    board = chess.Board(ITALIAN_START)
    probe = parse_probe("your line", ["Bc4", "Qh5"], board)
    assert [m.uci() for m in probe.parsed] == ["f1c4"] and probe.illegal == "Qh5"


def test_the_opponents_move_first_is_a_pass() -> None:
    board = chess.Board(ITALIAN_START)
    probe = parse_probe("if they play Bc5", ["pass", "Bc5"], board)
    assert probe.parsed[0] == chess.Move.null() and probe.parsed[1].uci() == "f8c5"


def test_examine_reports_the_score_the_gap_the_reply_and_the_effects() -> None:
    board = chess.Board(ITALIAN_START)
    after = board.copy()
    after.push_san("Bc4")
    engine = _Engine(
        {
            after.board_fen(): [
                Line(rank=1, depth=14, moves=["f8c5", "d2d3"], expected_score=0.48, score_cp=-10)
            ]
        }
    )
    best = Line(rank=1, depth=14, moves=["f1b5"], expected_score=0.55, score_cp=20)
    fact = examine(
        parse_probe("your Bc4", ["Bc4"], board),
        board,
        engine,
        best,
        user_colour=chess.WHITE,
        level=DisclosureLevel.QUESTION,
    )
    assert fact == (
        "Engine on your Bc4, which the player asked about: after 3. Bc4 the expected score is 0.52 "
        "from the player's side, 0.03 behind the engine's best; the opponent's reply would be "
        "3... Bc5 4. d3; what Bc4 does: attacks the pawn on f7."
    )
    # Below LINE the engine's own best move is not named; at LINE it is.
    assert "Bb5" not in fact
    at_line = examine(
        parse_probe("your Bc4", ["Bc4"], board),
        board,
        engine,
        best,
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
    )
    assert "0.03 behind the engine's best (Bb5)" in at_line


def test_spoken_puts_the_same_facts_in_plain_words() -> None:
    fact = (
        "Engine on your Bc4, which the player asked about: after 3. Bc4 the expected score is 0.52 "
        "from the player's side, 0.03 behind the engine's best; the opponent's reply would be "
        "3... Bc5 4. d3; what Bc4 does: attacks the pawn on f7."
    )
    assert spoken(fact) == (
        "your Bc4: after 3. Bc4 it is about equal, a little behind the best; they would "
        "answer 3... Bc5 4. d3; Bc4 attacks the pawn on f7."
    )
    worse = (
        "Engine on Nb1, which the player asked about: after 10. Nb1 the expected score is 0.30 "
        "from the player's side, 0.20 behind the engine's best (Nd5); the engine then plays "
        "10... Nxe4."
    )
    assert spoken(worse) == (
        "Nb1: after 10. Nb1 you are worse, a serious step down from the best (the best is "
        "Nd5); then the engine plays 10... Nxe4."
    )
    assert spoken("not an examination") == "not an examination"


def test_disclosed_frames_follow_the_probe_and_the_reply() -> None:
    board = chess.Board(ITALIAN_START)
    fact = (
        "Engine on your Bc4, which the player asked about: after 3. Bc4 the expected score is 0.52 "
        "from the player's side, 0.03 behind the engine's best; the opponent's reply would be "
        "3... Bc5 4. d3; what Bc4 does: attacks the pawn on f7."
    )
    frames = disclosed_frames(board, [fact])
    assert [f.peek().uci() for f in frames] == ["f1c4", "f8c5", "d2d3"]
    # A pass for the opponent's move first is a null move.
    theirs = (
        "Engine on if they play Bc5, which the player asked about: after pass Bc5 the expected "
        "score is 0.50 from the player's side, as good as the engine's best; the engine then plays "
        "3. Bc4 Nf6."
    )
    frames = disclosed_frames(board, [theirs])
    assert [f.peek().uci() for f in frames] == ["0000", "f8c5", "f1c4", "g8f6"]


def test_a_move_recited_from_the_examined_line_is_not_faulted() -> None:
    board = chess.Board("r1bq1rk1/1pp1bppp/p1np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10")
    fact = (
        "Engine on 10. h3, which the player asked about: after 10. h3 the expected score is 0.45 "
        "from the player's side, 0.08 behind the engine's best; the opponent's reply would be "
        "10... Na5 11. Re1."
    )
    draft = "After h3 Na5 Re1 they can play Nxc4."
    assert [p.text for p in check_draft(draft, board)] == ["Nxc4"]
    assert check_draft(draft, board, after=disclosed_frames(board, [fact])) == []


def test_undefended_means_no_defender_and_hanging_means_attacked_too() -> None:
    lone = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
    assert check_draft("Your knight on f3 is undefended.", lone) == []
    problems = check_draft("Your knight on f3 is hanging.", lone)
    assert [p.reason for p in problems] == ["nothing attacks the knight on f3"]
    held = chess.Board("4k3/8/8/8/8/5N2/4P3/4K3 w - - 0 1")
    problems = check_draft("Your knight on f3 is undefended.", held)
    assert [p.reason for p in problems] == ["the knight on f3 is defended"]
