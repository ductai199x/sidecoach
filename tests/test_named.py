"""The engine on what the player asks about: probes, their examination, the spoken form,
and the frames along a disclosed line."""

from __future__ import annotations

import chess

from sidecoach.coach.claims import check_draft
from sidecoach.coach.named import (
    Probe,
    disclosed_frames,
    examine,
    examine_piece,
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
                Line(depth=14, moves=["f8c5", "d2d3"], expected_score=0.48, score_cp=-10)
            ]
        }
    )
    best = Line(depth=14, moves=["f1b5"], expected_score=0.55, score_cp=20)
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


def _piece(fen: str, square: str, user_colour: chess.Color) -> str:
    probe = Probe(label=square, moves=[], square=chess.parse_square(square))
    return examine_piece(probe, chess.Board(fen), user_colour=user_colour)


def test_a_piece_attacks_and_is_defended_by_legal_captures_only() -> None:
    """A knight pinned to its king attacks and defends nothing off its line; a king takes
    back only where the other side does not still guard the square (the audit of
    2026-09-23)."""
    head = "Engine on {}, which the player asked about: "
    pinned = "4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1"
    assert _piece(pinned, "c3", chess.WHITE) == head.format("c3") + (
        "the player's knight on c3 stands there; it has no legal move; it is attacked by the "
        "bishop on b4 and the knight on e4; nothing defends it; it is pinned to the king on e1 "
        "by the bishop on b4."
    )
    assert _piece(pinned, "e4", chess.WHITE) == head.format("e4") + (
        "the opponent's knight on e4 stands there; it can go to c3, c5, d2, d6, f2, f6, g3, g5; "
        "it attacks the knight on c3; nothing attacks it; nothing defends it."
    )
    assert _piece("4k3/8/5n2/8/1b2P3/2N5/8/4K3 b - - 0 1", "e4", chess.WHITE) == head.format(
        "e4"
    ) + (
        "the player's pawn on e4 stands there; it can go to e5; it is attacked by the knight on "
        "f6; nothing defends it."
    )
    # Scholar's mate threatened: the queen guards f7, so the king cannot take back there.
    scholar = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4"
    assert _piece(scholar, "f7", chess.BLACK) == head.format("f7") + (
        "the player's pawn on f7 stands there; it has no legal move; it is attacked by the "
        "bishop on c4 and the queen on h5; nothing defends it; it is pinned to the king on e8 "
        "by the queen on h5."
    )
    # With a single attacker the king does take back: Bxd2+ Kxd2.
    alone = "4k3/8/5n2/8/1b2B3/8/3N4/4K3 b - - 0 1"
    assert _piece(alone, "d2", chess.WHITE) == head.format("d2") + (
        "the player's knight on d2 stands there; it has no legal move; it is attacked by the "
        "bishop on b4; it is defended by the king on e1; it is pinned to the king on e1 by the "
        "bishop on b4."
    )


def test_a_king_is_attacked_by_check_and_never_a_square_to_go_to() -> None:
    """A king is attacked by what gives it check. The checking queen, examined with its own
    side handed the move, was offered the king's square to go to."""
    board = "4k3/8/8/8/8/8/8/4QK2 b - - 0 1"
    assert _piece(board, "e1", chess.WHITE) == (
        "Engine on e1, which the player asked about: the player's queen on e1 stands there; it "
        "can go to a1, a5, b1, b4, c1, c3, d1, d2, e2, e3, e4, e5, e6, e7, f2, g3, h4; it "
        "attacks the king on e8; nothing attacks it; it is defended by the king on f1."
    )
    assert _piece(board, "e8", chess.WHITE) == (
        "Engine on e8, which the player asked about: the opponent's king on e8 stands there; it "
        "can go to d7, d8, f7, f8; it is attacked by the queen on e1."
    )


def test_a_pinned_piece_controls_a_square_only_against_the_king() -> None:
    assert _piece("4k3/8/8/8/1b6/2N5/8/4K3 w - - 0 1", "d5", chess.BLACK) == (
        "Engine on d5, which the player asked about: d5 is empty; the opponent controls it "
        "with the knight on c3 (pinned)."
    )


def test_undefended_means_no_defender_and_hanging_means_attacked_too() -> None:
    lone = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
    assert check_draft("Your knight on f3 is undefended.", lone) == []
    problems = check_draft("Your knight on f3 is hanging.", lone)
    assert [p.reason for p in problems] == ["nothing attacks the knight on f3"]
    held = chess.Board("4k3/8/8/8/8/5N2/4P3/4K3 w - - 0 1")
    problems = check_draft("Your knight on f3 is undefended.", held)
    assert [p.reason for p in problems] == ["the knight on f3 is defended"]


def test_a_piece_examined_is_said_to_the_player() -> None:
    """A simulated student's 9.Nf3? -- the square the student named went out as written for
    the model: 'f3: the player's knight on f3 stands there; it can go to ...' (2026-09-26)."""
    fen = "r3kbnr/pbpp1ppp/1pn1p3/8/8/4PN1P/PPPPQPP1/RNB1KB1R w KQkq - 0 1"
    said = spoken(_piece(fen, "f3", chess.WHITE))
    assert said.startswith("Your knight on f3 can go to ")
    assert "the player" not in said and "stands there" not in said
    empty = spoken(_piece(fen, "d4", chess.WHITE))
    assert empty == (
        "d4 is empty; you control it with the pawn on e3 and the knight on f3; they control it"
        " with the knight on c6."
    )
