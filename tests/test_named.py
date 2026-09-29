"""The engine on what the player asks about: probes, their examination, the spoken form,
and the frames along a disclosed line."""

from __future__ import annotations

import chess
import pytest

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
# The Italian at move ten, white to move.
MOVE_TEN = "r1bq1rk1/1pp1bppp/p1np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10"


class _Engine:
    """Scripted lines keyed by the position's FEN board part."""

    def __init__(self, lines: dict[str, list[Line]]) -> None:
        self.lines = lines

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        return self.lines.get(board.board_fen(), [])


def _examined(
    fen: str,
    label: str,
    moves: list[str],
    reply: list[str],
    score: float,
    best: Line,
    *,
    user_colour: chess.Color = chess.WHITE,
    level: DisclosureLevel = DisclosureLevel.IDEA,
) -> str:
    """`examine`'s own words on a probe, the engine answering the position after it with
    `reply` at `score` (for the side to move there): what the spoken form and the frames are
    read from, so they follow the examination as it is written."""
    board = chess.Board(fen)
    after = board.copy()
    for san in moves:
        after.push(chess.Move.null() if san == "pass" else after.parse_san(san))
    engine = _Engine({after.board_fen(): [Line(depth=14, moves=reply, expected_score=score)]})
    probe = parse_probe(label, moves, board)
    return examine(probe, board, engine, best, user_colour=user_colour, level=level)


def _your_bc4(level: DisclosureLevel = DisclosureLevel.IDEA, reply: tuple[str, ...] = ()) -> str:
    """3.Bc4 asked about in the Italian, the engine preferring 3.Bb5."""
    best = Line(depth=14, moves=["f1b5"], expected_score=0.55, score_cp=20)
    line = list(reply) or ["f8c5", "d2d3"]
    return _examined(ITALIAN_START, "your Bc4", ["Bc4"], line, 0.48, best, level=level)


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


def test_examine_reports_the_score_the_gap_the_reply_and_the_effects() -> None:
    fact = _your_bc4()
    assert fact == (
        "Engine on your Bc4, which the player asked about: after 3. Bc4 the expected score is 0.52 "
        "from the player's side, 0.03 behind the engine's best; the opponent's reply would be "
        "3... Bc5 4. d3; what Bc4 does: attacks the pawn on f7."
    )
    # Below LINE the engine's own best move is not named; at LINE it is.
    assert "Bb5" not in fact
    at_line = _your_bc4(DisclosureLevel.LINE)
    assert "0.03 behind the engine's best (Bb5)" in at_line


def test_the_engines_own_best_move_is_never_behind_the_best() -> None:
    """Searched again after it is played, the best move came back a hair off its own score:
    "12. Nc3: ... a hair behind the best (the best is Nc3)" (the simulated students)."""
    board = chess.Board(ITALIAN_START)
    after = board.copy()
    after.push_san("Bb5")
    engine = _Engine({after.board_fen(): [Line(depth=14, moves=["a7a6"], expected_score=0.46)]})
    best = Line(depth=14, moves=["f1b5"], expected_score=0.55)
    fact = examine(
        parse_probe("your Bb5", ["Bb5"], board),
        board,
        engine,
        best,
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
    )
    assert "as good as the engine's best" in fact and "behind" not in fact


def test_their_move_first_is_not_measured_against_the_best_and_is_said_as_their_move() -> None:
    """ "if they take on b4" is the player passing, no move of theirs: it came out "after
    pass axb4 you are losing, a serious step down from the best (the best is Bxc3)"."""
    board = chess.Board(ITALIAN_START)
    after = board.copy()
    after.push(chess.Move.null())
    after.push_san("Nd4")
    engine = _Engine({after.board_fen(): [Line(depth=14, moves=["d2d3"], expected_score=0.40)]})
    best = Line(depth=14, moves=["f1b5"], expected_score=0.55)
    fact = examine(
        parse_probe("if they play Nd4", ["pass", "Nd4"], board),
        board,
        engine,
        best,
        user_colour=chess.WHITE,
        level=DisclosureLevel.LINE,
    )
    assert "best" not in fact
    said = spoken(fact)
    assert said.startswith("if they play Nd4: if they got to play Nd4 now, ")
    assert "pass" not in said


def test_a_line_said_aloud_is_its_first_reply() -> None:
    """ "they would answer 9... Bxg2 10. Kxg2 Qd7 11. d5 Na6 12. Nc3 O-O" came before every
    answer the AI then gave in words: aloud, the reply's first two moves."""
    # At LINE the fact holds the whole reply.
    fact = _your_bc4(DisclosureLevel.LINE, reply=("f8c5", "d2d3", "g8f6", "b1c3", "d7d6"))
    assert "the opponent's reply would be 3... Bc5 4. d3 Nf6 5. Nc3 d6;" in fact
    said = spoken(fact)
    assert "they would answer 3... Bc5 4. d3;" in said and "Nf6" not in said


def test_spoken_puts_the_same_facts_in_plain_words() -> None:
    assert spoken(_your_bc4()) == (
        "your Bc4: after 3. Bc4 it is about equal, a little behind the best; they would "
        "answer 3... Bc5 4. d3; Bc4 attacks the pawn on f7."
    )
    # White's move asked about by Black: the engine's reply is its own line.
    best = Line(depth=14, moves=["c3d5"], expected_score=0.50)
    worse = _examined(
        MOVE_TEN,
        "Nb1",
        ["Nb1"],
        ["f6e4"],
        0.30,
        best,
        user_colour=chess.BLACK,
        level=DisclosureLevel.LINE,
    )
    assert spoken(worse) == (
        "Nb1: after 10. Nb1 you are worse, a serious step down from the best (the best is "
        "Nd5); then the engine plays 10... Nxe4."
    )
    assert spoken("not an examination") == "not an examination"


def test_disclosed_frames_follow_the_probe_and_the_reply() -> None:
    board = chess.Board(ITALIAN_START)
    frames = disclosed_frames(board, [_your_bc4()])
    assert [f.peek().uci() for f in frames] == ["f1c4", "f8c5", "d2d3"]
    # A pass for the opponent's move first is a null move.
    best = Line(depth=14, moves=["f1b5"], expected_score=0.55)
    theirs = _examined(
        ITALIAN_START, "if they play Bc5", ["pass", "Bc5"], ["f1c4", "g8f6"], 0.50, best
    )
    frames = disclosed_frames(board, [theirs])
    assert [f.peek().uci() for f in frames] == ["0000", "f8c5", "f1c4", "g8f6"]


def test_a_move_recited_from_the_examined_line_is_not_faulted() -> None:
    board = chess.Board(MOVE_TEN)
    best = Line(depth=14, moves=["c3d5"], expected_score=0.53)
    fact = _examined(MOVE_TEN, "10. h3", ["h3"], ["c6a5", "f1e1"], 0.55, best)
    assert "the opponent's reply would be 10... Na5 11. Re1." in fact
    draft = "After h3 Na5 Re1 they can play Nxc4."
    assert [p.text for p in check_draft(draft, board)] == ["Nxc4"]
    assert check_draft(draft, board, after=disclosed_frames(board, [fact])) == []


def _piece(fen: str, square: str, user_colour: chess.Color) -> str:
    probe = Probe(label=square, moves=[], square=chess.parse_square(square))
    return examine_piece(probe, chess.Board(fen), user_colour=user_colour)


# A piece or a square examined (`examine_piece`): what attacks and defends it, read off legal
# captures. (case, FEN, the square, the player's colour, the words after "Engine on <square>,
# which the player asked about: ".)
PINNED_C3 = "4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1"
QUEEN_CHECK = "4k3/8/8/8/8/8/8/4QK2 b - - 0 1"
PIECES: list[tuple[str, str, str, chess.Color, str]] = [
    # A knight pinned to its king attacks and defends nothing off its line; a king takes back
    # only where the other side does not still guard the square (the audit of 2026-09-23).
    (
        "a pinned knight",
        PINNED_C3,
        "c3",
        chess.WHITE,
        "the player's knight on c3 stands there; it has no legal move; it is attacked by the "
        "bishop on b4 and the knight on e4; nothing defends it; it is pinned to the king on e1 "
        "by the bishop on b4.",
    ),
    (
        "the knight the pinned one faces",
        PINNED_C3,
        "e4",
        chess.WHITE,
        "the opponent's knight on e4 stands there; it can go to c3, c5, d2, d6, f2, f6, g3, g5; "
        "it attacks the knight on c3; nothing attacks it; nothing defends it.",
    ),
    (
        "a pawn its pinned knight cannot defend",
        "4k3/8/5n2/8/1b2P3/2N5/8/4K3 b - - 0 1",
        "e4",
        chess.WHITE,
        "the player's pawn on e4 stands there; it can go to e5; it is attacked by the knight on "
        "f6; nothing defends it.",
    ),
    # Scholar's mate threatened: the queen guards f7, so the king cannot take back there.
    (
        "the king cannot take back",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4",
        "f7",
        chess.BLACK,
        "the player's pawn on f7 stands there; it has no legal move; it is attacked by the "
        "bishop on c4 and the queen on h5; nothing defends it; it is pinned to the king on e8 "
        "by the queen on h5.",
    ),
    # With a single attacker the king does take back: Bxd2+ Kxd2.
    (
        "the king takes back",
        "4k3/8/5n2/8/1b2B3/8/3N4/4K3 b - - 0 1",
        "d2",
        chess.WHITE,
        "the player's knight on d2 stands there; it has no legal move; it is attacked by the "
        "bishop on b4; it is defended by the king on e1; it is pinned to the king on e1 by the "
        "bishop on b4.",
    ),
    # A king is attacked by what gives it check. The checking queen, examined with its own
    # side handed the move, was offered the king's square to go to.
    (
        "the checking queen",
        QUEEN_CHECK,
        "e1",
        chess.WHITE,
        "the player's queen on e1 stands there; it can go to a1, a5, b1, b4, c1, c3, d1, d2, e2, "
        "e3, e4, e5, e6, e7, f2, g3, h4; it attacks the king on e8; nothing attacks it; it is "
        "defended by the king on f1.",
    ),
    (
        "the king in check",
        QUEEN_CHECK,
        "e8",
        chess.WHITE,
        "the opponent's king on e8 stands there; it can go to d7, d8, f7, f8; it is attacked by "
        "the queen on e1.",
    ),
    # A pinned piece controls a square only against the king.
    (
        "a square a pinned knight controls",
        "4k3/8/8/8/1b6/2N5/8/4K3 w - - 0 1",
        "d5",
        chess.BLACK,
        "d5 is empty; the opponent controls it with the knight on c3 (pinned).",
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "square", "user_colour", "said"), PIECES, ids=[r[0] for r in PIECES]
)
def test_a_piece_attacks_and_is_defended_by_legal_captures_only(
    case: str, fen: str, square: str, user_colour: chess.Color, said: str
) -> None:
    head = f"Engine on {square}, which the player asked about: "
    assert _piece(fen, square, user_colour) == head + said, case


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
