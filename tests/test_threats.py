"""The opponent's threat: what they would play if it were their turn.

A null move hands them the move, so the engine's answer from the passed position is the
threat the position carries. test_analysis.py checks this against a real engine; here a
recording fake pins down the plumbing deterministically -- the guards that must not call the
engine at all, and the null-move position it is handed.
"""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.threats import threat_lines, threat_moves
from sidecoach.events import Line


class _RecordingEngine(Engine):
    """Answers from a script and records every position it was asked about; no process."""

    def __init__(self, lines: list[Line]) -> None:
        super().__init__(path=Path("fake"))
        self._lines = lines
        self.seen: list[tuple[str, object, object]] = []

    def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:  # type: ignore[override]
        self.seen.append((board.fen(), kwargs.get("depth"), kwargs.get("multipv")))
        return self._lines

    def open(self) -> None:  # never starts a process
        return

    def close(self) -> None:
        return


CANNED = [
    Line(moves=["d8d1"], depth=8, expected_score=0.9),
    Line(moves=["a7a5"], depth=8, expected_score=0.4),
]


def test_the_engine_sees_the_position_with_the_move_handed_over() -> None:
    engine = _RecordingEngine(CANNED)
    board = chess.Board()  # White to move
    lines = threat_lines(engine, board, depth=8, multipv=2)
    assert lines == CANNED
    # Exactly one call, on a position that is Black to move (the move was passed) and carries
    # no null move in its history, so python-chess never warns.
    assert len(engine.seen) == 1
    seen_fen, seen_depth, seen_multipv = engine.seen[0]
    assert seen_fen.split()[1] == "b", "the opponent is now to move"
    assert (seen_depth, seen_multipv) == (8, 2), "depth and multipv are passed through"
    # The board the caller passed is untouched.
    assert board.turn is chess.WHITE and board.fen() == chess.STARTING_FEN


def test_threat_moves_are_the_first_move_of_each_line() -> None:
    engine = _RecordingEngine(CANNED)
    assert threat_moves(engine, chess.Board(), depth=8, multipv=2) == ["d8d1", "a7a5"]


def test_no_threat_and_no_engine_call_when_in_check() -> None:
    engine = _RecordingEngine(CANNED)
    # White is in check from the queen on e2's... use a simple check: black queen checks e1.
    in_check = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    assert in_check.is_check()
    assert threat_lines(engine, in_check, depth=8) == [], "you cannot pass out of check"
    assert engine.seen == [], "the engine is never asked"


def test_no_threat_and_no_engine_call_when_the_game_is_over() -> None:
    engine = _RecordingEngine(CANNED)
    # Fool's mate: 1. f3 e5 2. g4 Qh4#. White is to move and mated.
    mated = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mated.is_checkmate()
    assert threat_lines(engine, mated, depth=8) == []
    assert engine.seen == [], "a finished game carries no threat to ask about"
