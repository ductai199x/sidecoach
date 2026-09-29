"""What the engine has computed this game, for whoever reads it.

The analyser is the producer: it measures every position of the game, the one the player
faces early and then deeper, and the positions ahead of the game while it would otherwise
idle. Each depth it finishes goes in here, and a reader takes the newest word on a position
at any time (Tai, 2026-09-28: "stockfish is a reactive producer ... everything else just
consumes it"). The coach's engine is such a reader (`ComputedFirst`): it searches only what
is not here, and puts what it found here too (Tai, 2026-09-29: "our harness must first look
at what was already computed, then if that's not computed only then invoke a call to
stockfish to ask for analysis over that specific line").

A position is here when it was measured, or when a line measured from another position
passes through it: the rest of that line is the engine's word on it. Read so, a move the
lines already play is answered as the lines play it -- a second, shallower search after 64.
Bd2 answered it with Kh6 where the lines had Rxh3, and the model, shown both, said Bd2
avoided Rxh3 (the dev audit, 2026-09-26).

One thread writes (the analyser's, or the coach's own search) while others read; each
dictionary operation is atomic under the GIL, and what goes in is never changed after.
"""

from __future__ import annotations

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
        # Whose mate it is, and how many of the mating side's moves the plies played.
        mover_mates = line.mate > 0
        played = (plies + 1) // 2 if mover_mates else plies // 2
        left = abs(line.mate) - played
        mate = left if mover_mates != turned else -left
    return Line(
        moves=line.moves[plies:],
        depth=max(1, line.depth - plies),
        expected_score=1.0 - line.expected_score if turned else line.expected_score,
        score_cp=-line.score_cp if turned and line.score_cp is not None else line.score_cp,
        mate=mate,
        wdl=(line.wdl[2], line.wdl[1], line.wdl[0]) if turned and line.wdl else line.wdl,
    )


class Computed:
    """The engine's lines of each position measured, and where each line leads."""

    def __init__(self) -> None:
        # Measured whole, the newest depth: the lines a coach's sheet shows.
        self._whole: dict[str, list[Line]] = {}
        # Measured for a verdict only: the best line alone, at the full depth.
        self._thin: dict[str, list[Line]] = {}
        # The positions the lines pass through: the line, and how many of its plies lead there.
        self._through: dict[str, tuple[Line, int]] = {}

    def clear(self) -> None:
        """A new game: none of this is about it."""
        self._whole.clear()
        self._thin.clear()
        self._through.clear()

    def add(self, board: chess.Board, lines: Sequence[Line], *, thin: bool = False) -> None:
        """The lines of `board`, as a search just gave them. A shallower search than the one
        here already is left out: the deeper word stands."""
        if not lines:
            return
        store = self._thin if thin else self._whole
        key = position_key(board)
        here = store.get(key)
        if here is not None and here[0].depth > lines[0].depth:
            return
        store[key] = list(lines)
        for line in lines:
            self._index(board, line)

    def lines(self, board: chess.Board) -> list[Line]:
        """What is known of `board`: its own lines -- measured whole, else for a verdict --
        or else the rest of the deepest line through it; none when nothing reaches it."""
        key = position_key(board)
        own = self._whole.get(key) or self._thin.get(key)
        if own:
            return list(own)
        through = self._through.get(key)
        if through is None:
            return []
        return [line_after(*through)]

    def _index(self, board: chess.Board, line: Line) -> None:
        """The positions `line` passes through with moves still to come, each kept with the
        deepest line through it (the newest of equals)."""
        replay = board.copy(stack=False)
        for plies, uci in enumerate(line.moves[:-1], start=1):
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                return
            if move not in replay.legal_moves:
                return
            replay.push(move)
            key = position_key(replay)
            here = self._through.get(key)
            if here is None or line.depth - plies >= here[0].depth - here[1]:
                self._through[key] = (line, plies)


class ComputedFirst:
    """The coach's engine: what is computed first (`Computed`), a search for the rest. The
    engine starts at its first search (`Engine.analyse` opens it), so a coach whose questions
    the analyser has answered never starts a second Stockfish."""

    def __init__(self, engine: Searcher, computed: Computed | None = None) -> None:
        self.engine = engine
        self.computed = computed if computed is not None else Computed()
        self.searches = 0  # positions searched, not found computed

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        known = self.computed.lines(board)
        if known:
            return known
        self.searches += 1
        lines = self.engine.analyse(board, depth=depth)
        self.computed.add(board, lines)
        return lines

    def close(self) -> None:
        self.engine.close()
