"""What the engine has computed this game, for whoever reads it.

The analyser is the producer: it measures every position of the game and the candidates in
each, the one the player faces first. Each answer it has goes in here -- a best line, the
hint's lines at the early depth, the position whole -- and a reader takes the newest word on
a position at any time (Tai, 2026-09-28: "stockfish is a reactive producer ... everything else just
consumes it"). The coach's engine is such a reader (`ComputedFirst`): it searches only a
position that is not here, and puts what it found here too (Tai, 2026-09-29: "our harness
must first look at what was already computed, then if that's not computed only then invoke
a call to stockfish to ask for analysis over that specific line").

A move on a line the model was shown is read from that line, before any position is looked
up (`rest_of_line`, the coach's own step): a separate search of the position after it --
deeper or not -- can answer differently, and a second search after 64. Bd2
answered it with Kh6 where the lines had Rxh3, and the model, shown both, said Bd2 avoided
Rxh3 (the dev audit, 2026-09-26). Only positions measured are kept here: the lines passing
through a position went stale as the search deepened, a move refuted at depth 18 still read
off its line at depth 12 (the code review, 2026-09-29).

The analyser's thread and the coach's both write; a lock keeps each write whole.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Protocol

import chess

from sidecoach.events import Line


class Searcher(Protocol):
    """What `ComputedFirst` searches with; `analysis.engine.Engine` satisfies it."""

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]: ...

    def close(self) -> None: ...


def position_key(board: chess.Board) -> str:
    """The position as a key: the pieces, the side to move, the castling rights and the en
    passant square -- the FEN without its clocks, which a board read off the screen guesses."""
    return " ".join(board.fen().split()[:4])


def line_after(line: Line, plies: int) -> Line:
    """The rest of `line` once its first `plies` are played, as a search from there would
    give it: from the side to move there, a mate counted in the moves still to come, and the
    depth what the search had left."""
    turned = plies % 2 == 1
    mate = None
    if line.mate:
        # Whose mate it is, and how many of the mating side's moves the plies played; a mate
        # past its count (a line carried on beyond it) is no mate this line knows.
        mover_mates = line.mate > 0
        played = (plies + 1) // 2 if mover_mates else plies // 2
        left = abs(line.mate) - played
        if left > 0:
            mate = left if mover_mates != turned else -left
    return Line(
        moves=line.moves[plies:],
        depth=max(1, line.depth - plies),
        expected_score=1.0 - line.expected_score if turned else line.expected_score,
        score_cp=-line.score_cp if turned and line.score_cp is not None else line.score_cp,
        mate=mate,
        wdl=(line.wdl[2], line.wdl[1], line.wdl[0]) if turned and line.wdl else line.wdl,
    )


def rest_of_line(lines: Sequence[Line], moves: Sequence[str]) -> Line | None:
    """The rest of the first of `lines` that begins with `moves` (UCI) and goes on past them,
    from the side to move after them; None when none does."""
    n = len(moves)
    if not n:
        return None
    for line in lines:
        if len(line.moves) > n and line.moves[:n] == list(moves):
            return line_after(line, n)
    return None


class Computed:
    """The engine's lines of each position measured."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Measured whole, the newest depth: the lines a coach's sheet shows.
        self._whole: dict[str, list[Line]] = {}
        # Measured for a verdict only: the best line alone, at the full depth.
        self._thin: dict[str, list[Line]] = {}

    def clear(self) -> None:
        """A new game: none of this is about it."""
        with self._lock:
            self._whole.clear()
            self._thin.clear()

    def add(self, board: chess.Board, lines: Sequence[Line], *, thin: bool = False) -> None:
        """The lines of `board`, as a search just gave them. A shallower search than the one
        here already is left out: the deeper word stands."""
        if not lines:
            return
        key = position_key(board)
        with self._lock:
            store = self._thin if thin else self._whole
            here = store.get(key)
            if here is None or here[0].depth <= lines[0].depth:
                store[key] = list(lines)

    def lines(self, board: chess.Board) -> list[Line]:
        """What is known of `board`: its lines measured whole, else its verdict's line; none
        when it was not measured. Measured whole first, as the sheet shows it."""
        key = position_key(board)
        with self._lock:
            return list(self._whole.get(key) or self._thin.get(key) or [])

    def whole(self, board: chess.Board) -> list[Line]:
        """`board`'s lines measured whole; none while it is not."""
        with self._lock:
            return list(self._whole.get(position_key(board)) or [])


class ComputedFirst:
    """The coach's engine: a position measured whole read (`Computed`), a search for the rest
    -- through the analyser's pool (`analyser.PoolSearcher`), or a Stockfish of its own started
    at its first search, so a coach whose questions the analyser has answered never starts one.
    Closed, it searches no more and answers with what is known, the best line included: a
    question still being answered as the coach closes would start a process nobody closes. A
    searcher that keeps what it finds itself (`fills_computed`) is not kept for twice -- nor
    is what it found once its wait ran out filed as whole."""

    def __init__(self, engine: Searcher, computed: Computed | None = None) -> None:
        self.engine = engine
        self.computed = computed if computed is not None else Computed()
        self.searches = 0  # positions searched, not found computed
        self._closed = False

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        known = self.computed.whole(board)
        if known:
            return known
        if self._closed:
            return self.computed.lines(board)
        self.searches += 1
        lines = self.engine.analyse(board, depth=depth)
        if not getattr(self.engine, "fills_computed", False):
            self.computed.add(board, lines)
        return lines

    def close(self) -> None:
        self._closed = True
        self.engine.close()
