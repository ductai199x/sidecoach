"""A reviewed game summed up (profile/game_summary.py): each side's accuracy and counts, the
grade of each phase, the moves the review stops at, the graph."""

from __future__ import annotations

import chess

from sidecoach.events import (
    Analysis,
    Label,
    Line,
    MoveAssessment,
    ReviewedGame,
    ReviewedMove,
    Severity,
    Side,
)
from sidecoach.profile.game_summary import phase_grade, shape_of, summarize


def test_a_phase_is_graded_as_the_kind_of_move_it_amounts_to() -> None:
    assert phase_grade(95.0, [Label.BEST]) is Label.BEST
    assert phase_grade(95.0, [Label.BEST, Label.GREAT]) is Label.GREAT
    assert phase_grade(92.0, [Label.GREAT, Label.BRILLIANT]) is Label.BRILLIANT
    # A brilliant move in a phase played badly does not make it brilliant.
    assert phase_grade(60.0, [Label.BRILLIANT, Label.BLUNDER]) is Label.INACCURACY
    assert [phase_grade(a, []) for a in (85, 75, 60, 45, 20)] == [
        Label.EXCELLENT,
        Label.GOOD,
        Label.INACCURACY,
        Label.MISTAKE,
        Label.BLUNDER,
    ]


def _game(labels: list[tuple[str, Label, float, float]]) -> ReviewedGame:
    """A game of these moves (san, label, accuracy, the mover's expected score after)."""
    board = chess.Board()
    start = Analysis(
        game_id="g",
        fen=board.fen(),
        ply=0,
        lines=[Line(moves=["e2e4"], depth=18, expected_score=0.52)],
        decisive=False,
        only_move=False,
    )
    moves = []
    for ply, (san, label, accuracy, mover_after) in enumerate(labels, start=1):
        move = board.parse_san(san)
        side = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
        before = board.fen()
        board.push(move)
        after = Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            decisive=False,
            only_move=False,
            # The side to move after is the opponent: its score is one minus the mover's.
            lines=[
                Line(
                    moves=[next(iter(board.legal_moves)).uci()],
                    depth=18,
                    expected_score=1 - mover_after,
                )
            ],
            in_book=label is Label.BOOK,
            opening="C50 Italian Game" if ply == 5 else None,
        )
        moves.append(
            ReviewedMove(
                ply=ply,
                san=san,
                side=side,
                fen_before=before,
                assessment=MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=move.uci(),
                    expected_loss=0.0,
                    severity=Severity.NONE,
                    best=[],
                    label=label,
                    accuracy=accuracy,
                ),
                after=after,
            )
        )
    return ReviewedGame(
        headers={"White": "Anakin_737", "Black": "nosegazm", "WhiteElo": "880", "BlackElo": "1018"},
        start=start,
        moves=moves,
        strength="standard",
        depth=18,
    )


def test_the_summary_counts_grades_and_finds_where_to_stop() -> None:
    game = _game(
        [
            ("e4", Label.BOOK, 100, 0.52),
            ("e5", Label.BOOK, 100, 0.48),
            ("Nf3", Label.BOOK, 100, 0.52),
            ("Nc6", Label.BOOK, 100, 0.48),
            ("Bc4", Label.BOOK, 100, 0.52),
            ("Nf6", Label.BEST, 100, 0.49),
            ("d3", Label.INACCURACY, 30, 0.44),
            ("Bc5", Label.BEST, 100, 0.57),
            ("Nc3", Label.GOOD, 70, 0.41),
            ("d6", Label.GREAT, 100, 0.62),
        ]
    )
    summary = summarize(game)
    white, black = summary.white, summary.black
    assert (white.name, white.rating, black.name, black.rating) == (
        "Anakin_737",
        880,
        "nosegazm",
        1018,
    )
    assert white.counts == {"book": 3, "inaccuracy": 1, "good": 1}
    assert black.counts == {"book": 2, "best": 2, "great": 1}
    assert white.accuracy == 80.0 and black.accuracy == 100.0
    # All ten moves are the opening: White's 80 is excellent, Black's 100 with a great move great.
    assert [(p.phase, p.grade) for p in white.phases] == [("opening", Label.EXCELLENT)]
    assert [(p.phase, p.grade) for p in black.phases] == [("opening", Label.GREAT)]
    # The last book move, then every move worth a word of its own, in order.
    assert summary.last_book_ply == 5
    assert [(k.ply, k.kind) for k in summary.key_moments] == [
        (5, "book"),
        (7, "inaccuracy"),
        (10, "great"),
    ]
    assert summary.opening == "C50 Italian Game"
    # White's expected score: the start, then after each move from White's side.
    assert summary.graph[:3] == [0.52, 0.52, 0.52]
    assert summary.graph[7] == 0.44 and summary.graph[8] == 0.43
    assert len(summary.graph) == 11


def test_each_side_is_given_the_rating_it_played_at() -> None:
    sans = "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O a4 a6 Re1 h6 h3 Re8 Nbd2 Be6 Bb3 Ba7 Nf1 d5"
    game = _game([(san, Label.BEST, 100, 0.5) for san in sans.split()])
    # White plays the best move every time; Black throws a quarter of the game away four times.
    moves = [
        m.model_copy(
            update={
                "assessment": m.assessment.model_copy(
                    update={"expected_loss": 0.3 if m.side is Side.BLACK and m.ply % 3 == 0 else 0}
                )
            }
        )
        for m in game.moves
    ]
    summary = summarize(game.model_copy(update={"moves": moves}))
    white, black = summary.white.performance, summary.black.performance
    assert white is not None and black is not None
    assert white.rating > black.rating + 400
    assert white.low <= white.rating <= white.high
    # Too few moves say nothing: the ten-move game above has no rating.
    short = _game([("e4", Label.BOOK, 100, 0.52), ("e5", Label.BOOK, 100, 0.48)])
    assert summarize(short).white.performance is None


def test_what_the_mistakes_passed_up_is_what_to_practise() -> None:
    from sidecoach.events import MotifTag
    from sidecoach.profile.game_summary import missed_motifs

    game = _game(
        [
            ("e4", Label.BOOK, 100, 0.52),
            ("e5", Label.BOOK, 100, 0.48),
            ("Nf3", Label.BEST, 100, 0.52),
            ("Nc6", Label.MISTAKE, 40, 0.40),
            ("Bc4", Label.BLUNDER, 10, 0.20),
            ("Nf6", Label.GOOD, 70, 0.75),
        ]
    )
    fork = MotifTag(name="fork", squares=["d4"])
    pin = MotifTag(name="pin", squares=["b5"])
    # The positions Black's Nc6 and White's Bc4 were played from held a fork and a pin.
    moves = list(game.moves)
    moves[2] = moves[2].model_copy(
        update={"after": moves[2].after.model_copy(update={"tags": [fork]})}
    )
    moves[3] = moves[3].model_copy(
        update={"after": moves[3].after.model_copy(update={"tags": [pin]})}
    )
    game = game.model_copy(update={"moves": moves})
    assert missed_motifs(game, {Side.WHITE, Side.BLACK}) == [("fork", 1), ("pin", 1)]
    assert missed_motifs(game, {Side.BLACK}) == [("fork", 1)]
    assert missed_motifs(game, {Side.WHITE}) == [("pin", 1)]


def test_the_shape_of_a_game_from_its_graph() -> None:
    even = [0.5] * 20
    # Close all game and drawn: balanced.
    assert shape_of(even, "1/2-1/2") == "balanced"
    # Close for three quarters, then one side won: sudden.
    assert shape_of(even[:16] + [0.8, 0.9, 0.95, 1.0], "1-0") == "sudden"
    # White took the lead early and kept it: smooth.
    assert (
        shape_of([0.5, 0.55, 0.6, 0.72, 0.75, 0.8, 0.82, 0.85, 0.9, 0.93, 0.95, 1.0], "1-0")
        == "smooth"
    )
    # White was clearly winning, then lost: giveaway.
    assert (
        shape_of([0.5, 0.6, 0.8, 0.9, 0.9, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0], "0-1") == "giveaway"
    )
    # Winning changed hands three times: wild; twice: intense.
    wild = [0.5, 0.75, 0.8, 0.25, 0.2, 0.75, 0.8, 0.2, 0.1, 0.05, 0.0, 0.0]
    assert shape_of(wild, "0-1") == "wild"
    intense = [0.5, 0.75, 0.8, 0.25, 0.2, 0.5, 0.75, 0.8, 0.9, 0.95, 1.0, 1.0]
    assert shape_of(intense, "1-0") == "intense"
    # Too short to have a shape.
    assert shape_of([0.5] * 6, "1-0") is None
