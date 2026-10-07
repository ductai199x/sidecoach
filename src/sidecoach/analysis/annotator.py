"""From moves to assessments: what a move cost, what the position holds, what is threatened.

`GameAnnotator` does the work for one game and is synchronous, which is what a post-game
review wants. `Analyser` (analyser.py) runs the same searches on a pool of workers, which is
what a live session needs: the capture loop must not wait.

Every search is one line to the full depth (the engine audit, 2026-10-04/05). Asking Stockfish
for the exact value of several moves at once -- a search of several lines -- explodes on
decided and forcing positions: two lines of one real position took 53 s, one line 0.18 s; 84 of
8,886 positions of real sessions took over 5 s with six lines at depth 18, the worst 47 s. So a
position is measured whole as:

- its best line: one line to the full depth (the verdict, the best move);
- its candidates: the best moves of a six-line search to EARLY_DEPTH (never over 1.9 s), each
  then searched as its own position, one line to the full depth -- the position's lines at the
  full depth, built from those, and exactly the positions the game may go to next;
- its side searches (threats, traps, declined recaptures, what a quiet best move threatens).

Every position is measured once, to the same depth, and cached, so the position after one move
is the position before the next, and a candidate searched is the next position measured. The
matching depth is what makes the loss of a move meaningful: every value a verdict compares is
one position's own one-line search to the full depth.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Protocol

import chess

from sidecoach.analysis import motifs
from sidecoach.analysis.computed import Computed, line_after
from sidecoach.analysis.defences import DEFENCE_DEPTH, defence_lines
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import (
    Previous,
    Thresholds,
    eval_loss,
    is_decisive,
    is_only_move,
    move_accuracy,
    position_score,
    severity,
    terminal_expected_score,
    verdict,
)
from sidecoach.analysis.threats import AIM_DEPTH, aim_lines, threat_moves
from sidecoach.analysis.traps import TRAP_DEPTH, trap_lines
from sidecoach.events import Analysis, Line, MotifTag, MoveAssessment

# The depth of a position's first answer: its six candidates at once, the full depth to follow.
# Its best move is the full depth's in every one of 77 puzzle positions and 81 % of 232 game
# positions (Tai, 2026-09-28: "first usable answer is depth 12 ... the computation continues
# until max depth of 18"), and six lines to it never took more than 1.9 s on any of 8,886
# positions of real sessions (p99 0.29 s; the engine audit, 2026-10-05).
EARLY_DEPTH = 12


def at_full_depth(lines: Sequence[Line]) -> bool:
    """Lines a reader may set against a full-depth search: none of them the first answer's
    hint, at EARLY_DEPTH (the audits of 2026-10-05: the hint's 12 set against a reply searched
    to 18 read a move as good as the best as worse, or better, by what the depths differ by)."""
    return bool(lines) and min(line.depth for line in lines) > EARLY_DEPTH


class Tagger(Protocol):
    """What `sidecoach.analysis.motifs` provides; a module satisfies this."""

    def tags_for_move(
        self,
        before: chess.Board,
        move: chess.Move,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]: ...

    def tags_for_position(
        self,
        board: chess.Board,
        *,
        pv: Sequence[chess.Move] = (),
        mate_in: int | None = None,
    ) -> list[MotifTag]: ...


class Book(Protocol):
    """What `sidecoach.analysis.openings.OpeningBook` provides."""

    def identify(self, moves: Sequence[chess.Move]) -> object | None: ...

    def in_book(self, moves: Sequence[chess.Move]) -> bool: ...


@dataclass
class Measured:
    """One position, measured once: its lines and what they mean."""

    fen: str
    lines: list[Line]
    score: float
    decisive: bool
    only_move: bool
    threats: list[str] = field(default_factory=list)
    traps: list[Line] = field(default_factory=list)
    defences: list[Line] = field(default_factory=list)
    threatens: list[Line] = field(default_factory=list)
    # What it is cached by (`cache_key`): the FEN, and the moves since the last capture
    # or pawn move when there are any. Empty in a record kept before it had one.
    key: str = ""

    @property
    def depth(self) -> int:
        """How deep the search got: the depth asked for, or less for a search stopped."""
        return self.lines[0].depth if self.lines else 0


def cache_key(board: chess.Board) -> str:
    """What a position is cached by: its FEN, and the moves since the last capture or pawn
    move, the ones a repetition can reach back over. The engine is told the moves that led to
    a position and scores a repetition as the draw it is, so the same placement after
    different quiet moves is a different search; after a capture or a pawn move nothing
    before can repeat and the FEN is the whole key."""
    fen = board.fen()
    count = min(board.halfmove_clock, len(board.move_stack))
    if count == 0:
        return fen
    return fen + " " + " ".join(move.uci() for move in board.move_stack[-count:])


@dataclass
class GameAnnotator:
    """Assessments for the moves of one game. `Analyser`'s workers search through one at once:
    its caches are written under a lock, with the era checked there."""

    engine: Engine
    depth: int = 18
    multipv: int = 6
    thresholds: Thresholds = field(default_factory=Thresholds)
    tagger: Tagger | None = None
    book: Book | None = None
    with_threats: bool = True
    _cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)
    # Each position's best line alone (`best_line`): what a verdict reads of the position a
    # move led to, and a candidate's value for the position before it. Kept apart from
    # `_cache`, so a later whole measurement of the same position is not answered with it.
    _verdict_cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)
    # Each move assessed, by its ply: the game, the position it led to, its loss, where it
    # landed and whether it captured, so the move after it knows whether the opponent just
    # erred and whether it is merely taking back. By ply, not the last one assessed: after a
    # takeback the move before the replacement is the one the game kept, and the last one
    # assessed was the withdrawn move, a ply too far -- "the opponent erred" and "a
    # recapture" were lost on every replacement (the state review, 2026-10-01).
    _assessed: dict[int, tuple[str, str, float, chess.Square, bool]] = field(
        default_factory=dict, init=False, repr=False
    )
    # Every search's lines as it finishes each usable depth, for any reader at any time (the
    # coach's engine reads it before it searches).
    computed: Computed = field(default_factory=Computed, init=False, repr=False)
    # How many times `forget_measured` has run: a search that began before it (another
    # thread changed the settings mid-search) was made the old way, and is not kept.
    _era: int = field(default=0, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def reset(self) -> None:
        """Forget the cache; a new game shares no positions with the last one."""
        with self._lock:
            self._cache.clear()
            self._verdict_cache.clear()
            self.computed.clear()
            self._assessed.clear()

    def forget_assessed(self, ply: int) -> None:
        """The move at `ply` taken back: the move before the next one played there is not it."""
        self._assessed.pop(ply, None)

    def _keep(self, era: int, board: chess.Board, measured: Measured, *, whole: bool) -> None:
        """`measured` kept for readers and for the next asking -- unless the settings changed
        since its search began (`era`): checked and written under the lock, so a change in
        between cannot let an old search's answer in."""
        with self._lock:
            if era != self._era:
                return
            self.computed.add(board, measured.lines, thin=not whole)
            store = self._cache if whole else self._verdict_cache
            store[measured.key or cache_key(board)] = measured

    def forget_measured(self) -> None:
        """Forget the positions measured, keeping the game's moves: the depth, the lines, the
        curve or the thresholds changed, and a loss compares two positions measured the same
        way, so none measured before is set against one measured after. Called from another
        thread (the settings' save): a search under way now finishes, and is handed to its
        caller, but is not kept."""
        with self._lock:
            self._era += 1
            self._cache.clear()
            self._verdict_cache.clear()
            self.computed.clear()

    @property
    def era(self) -> int:
        """How many times what was measured has been forgotten (`forget_measured`)."""
        return self._era

    # ---- the searches ---------------------------------------------------------------------

    def best_line(
        self,
        board: chess.Board,
        *,
        engine: Engine | None = None,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> Measured:
        """The position's best line: one line to the full depth, carried on to its point (the
        coach reads the line for it). What a verdict reads of a position, and a candidate's
        value for the position before it. A finished position has no line; a position with one
        legal move has that move's line, from its own position's search -- the move needs no
        search to be chosen (the slowest one-line search of 8,886 was such a position: 10.6 s).
        A position measured whole answers with its whole, whose first line is this one. Cached;
        `on_depth` is handed each depth the search finishes on its way."""
        whole = self._cached_quietly(board)
        if whole is not None:
            return whole
        key = cache_key(board)
        known = self._verdict_cache.get(key)
        if known is not None:
            return known
        era = self._era
        legal = list(board.legal_moves)
        if terminal_expected_score(board) is not None or not legal:
            measured = self._one_line(board, [])
        elif len(legal) == 1:
            child = board.copy()
            child.push(legal[0])
            found = self.best_line(child, engine=engine, on_depth=on_depth)
            measured = self.from_only_move(board, legal[0], found)
        else:
            lines = (engine or self.engine).analyse(
                board, depth=self.depth, multipv=1, carry_best=True, on_depth=on_depth
            )
            measured = self._one_line(board, lines)
        self._keep(era, board, measured, whole=False)
        return measured

    def from_only_move(self, board: chess.Board, move: chess.Move, child: Measured) -> Measured:
        """The best line of a position whose one legal move is `move`: that move's, from the
        position it leads to (`child`, its best line)."""
        return self._one_line(board, [self.candidate_line(board, move, child)])

    def _one_line(self, board: chess.Board, lines: list[Line]) -> Measured:
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=False,
            only_move=False,
            key=cache_key(board),
        )

    def measure_for_verdict(self, board: chess.Board) -> Measured:
        """What a verdict reads of a position: whole if it already is, else its best line."""
        return self.best_line(board)

    def hint_lines(self, board: chess.Board, *, engine: Engine | None = None) -> list[Line]:
        """The position's candidates at EARLY_DEPTH: `multipv` lines, the first answer to a
        hint and the moves whose positions the full depth then searches (`candidates`). None
        for a position with one legal move or none: its best line says all there is. Not kept
        for readers (`computed`): a reader takes the position whole, or its best line."""
        if terminal_expected_score(board) is not None or board.legal_moves.count() < 2:
            return []
        return (engine or self.engine).analyse(
            board, depth=min(EARLY_DEPTH, self.depth), multipv=self.multipv
        )

    def candidates(
        self, board: chess.Board, best: Measured, hint: Sequence[Line]
    ) -> list[chess.Move]:
        """The moves the position is measured whole by: its best line's move first, then the
        hint's in its order, `multipv` at most, each legal and once."""
        out: list[chess.Move] = []
        for line in [*best.lines[:1], *hint]:
            move = _first_move(line)
            if move is not None and move in board.legal_moves and move not in out:
                out.append(move)
            if len(out) == self.multipv:
                break
        return out

    def candidate_line(self, board: chess.Board, move: chess.Move, child: Measured) -> Line:
        """`move`'s line for the side to move in `board`, from its own position's search
        (`child`, measured for the side to move there): the move, then the child's best line;
        every score turned to this side -- the expected score and the win-draw-loss mirrored,
        the centipawns negated, a mate the child's side gives a mate this side takes, and the
        reverse a move longer. A finished child is scored as it stands."""
        after = board.copy()
        after.push(move)
        finished = terminal_expected_score(after)
        line = child.lines[0] if child.lines else None
        if finished is not None or line is None:
            score = 1.0 - (finished if finished is not None else child.score)
            mate = 1 if after.is_checkmate() else None
            return Line(moves=[move.uci()], depth=self.depth, expected_score=score, mate=mate)
        mate = None
        if line.mate is not None:
            mate = -line.mate if line.mate > 0 else abs(line.mate) + 1
        return Line(
            moves=[move.uci(), *line.moves],
            depth=line.depth,
            expected_score=1.0 - line.expected_score,
            score_cp=-line.score_cp if line.score_cp is not None else None,
            mate=mate,
            wdl=(line.wdl[2], line.wdl[1], line.wdl[0]) if line.wdl is not None else None,
        )

    def whole(
        self,
        board: chess.Board,
        best: Measured,
        children: Sequence[tuple[chess.Move, Measured]],
    ) -> Measured:
        """The position whole: its best line first -- its own search, which picks the best move
        more often than its candidates' searches do (231 against 222 of 300 positions against
        a depth-24 truth, the engine audit 2026-10-05), so the engine's own move is never
        charged a loss -- then its other candidates from their own positions' searches, best
        first, a shorter mate ahead of a longer one; the decisive-moment and only-move calls
        read from them. No side searches (`with_side`)."""
        first = best.lines[:1]
        shown = {line.moves[0] for line in first if line.moves}
        rest = [
            self.candidate_line(board, move, child)
            for move, child in children
            if move.uci() not in shown
        ]
        rest.sort(key=_rank)  # stable: the hint's order on ties
        lines = [*first, *rest]
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            key=cache_key(board),
        )

    def measure(self, board: chess.Board) -> Measured:
        """The position whole, at the full depth: its best line, its candidates' own searches
        and its side searches (the module's docstring). Cached, as every search under it."""
        cached = self._cached_quietly(board)
        if cached is not None:
            return cached
        era = self._era
        best = self.best_line(board)
        hint = self.hint_lines(board)
        children = []
        for move in self.candidates(board, best, hint):
            child = board.copy()
            child.push(move)
            children.append((move, self.best_line(child)))
        measured = self.with_side(board, self.whole(board, best, children))
        self._keep(era, board, measured, whole=True)
        return measured

    def with_side(
        self,
        board: chess.Board,
        measured: Measured,
        *,
        engine: Engine | None = None,
    ) -> Measured:
        """`measured` with its side searches: what the coach's words on the position are made
        of (the threats, the traps, the declined recaptures, what a quiet best move
        threatens). The position's own lines are what they search from; none of them is a
        search of the position again. Nothing when the annotator searches no threats."""
        if not self.with_threats or terminal_expected_score(board) is not None:
            return measured
        engine = engine or self.engine
        depth = self.depth
        lines = measured.lines
        return replace(
            measured,
            threats=threat_moves(engine, board, depth=depth),
            # The captures that look free and are not, beside the threats: what the first
            # hint warns of ("the knight on g4 is bait").
            traps=trap_lines(engine, board, lines, depth=min(TRAP_DEPTH, depth)),
            # And the recaptures the best line declines, with what refutes them: why the
            # obvious defence fails is the lesson ("the pawn on f2 cannot take back").
            defences=defence_lines(engine, board, lines, depth=min(DEFENCE_DEPTH, depth)),
            # And what a quiet best move threatens: how it wins ("it threatens mate").
            threatens=aim_lines(engine, board, lines, depth=min(AIM_DEPTH, depth)),
        )

    # ---- what the readers get ---------------------------------------------------------------

    def analysis_event(
        self, game_id: str, board: chess.Board, ply: int, *, thin: bool = False
    ) -> Analysis:
        """What the position holds, for the side to move. `thin`: only its best line (no
        candidates, no threats, no decisive call) -- what a verdict measured."""
        measured = self.measure_for_verdict(board) if thin else self.measure(board)
        return self.analysis_from(game_id, board, ply, measured)

    def analysis_from(
        self,
        game_id: str,
        board: chess.Board,
        ply: int,
        measured: Measured,
        *,
        provisional: bool = False,
    ) -> Analysis:
        """What `measured` says of `board`, for the side to move: the event the readers get.
        `provisional`: the first answer (the best line, the candidates at EARLY_DEPTH), the
        whole to follow and replace it."""
        return Analysis(
            game_id=game_id,
            fen=measured.fen,
            ply=ply,
            lines=measured.lines,
            decisive=measured.decisive,
            only_move=measured.only_move,
            threats=measured.threats,
            traps=measured.traps,
            defences=measured.defences,
            threatens=measured.threatens,
            tags=self._position_tags(board, measured),
            opening=self._opening(board),
            in_book=self._in_book(board),
            provisional=provisional,
        )

    def first_answer(
        self, board: chess.Board, best: Measured | None, hint: Sequence[Line]
    ) -> Measured:
        """The position's first answer, before its candidates are searched: the hint's lines
        at EARLY_DEPTH when they are in, else the best line -- one search's lines, never two
        searches' set against each other (the calls, a probe's distance from the best, the
        lines best first all compare them); the calls read from them."""
        lines = list(hint) if hint else (list(best.lines) if best is not None else [])
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            key=cache_key(board),
        )

    def measured_lines(self, board: chess.Board, lines: list[Line]) -> Measured:
        """`lines` of `board` as a measurement: the calls they make."""
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            key=cache_key(board),
        )

    def measured(self, board: chess.Board) -> bool:
        """Whether the position is measured whole already."""
        return self._cached_quietly(board) is not None

    def adopt(self, measured: Iterable[Measured]) -> None:
        """Positions measured whole by another annotator -- another engine, on another thread,
        for a review measuring a game's positions side by side (`game_review`) -- kept as if
        measured here: the walk through the game's moves then searches nothing."""
        for m in measured:
            self._cache[m.key or m.fen] = m

    def _cached_quietly(self, board: chess.Board) -> Measured | None:
        """The position measured whole, by its key or (adopted without one) its FEN."""
        return self._cache.get(cache_key(board)) or self._cache.get(board.fen())

    def assess(
        self, game_id: str, before: chess.Board, move: chess.Move, ply: int
    ) -> tuple[MoveAssessment, Analysis]:
        """What the move cost its mover, and what the position it led to holds."""
        assessment = self.assess_move(game_id, before, move, ply)
        after = before.copy()
        after.push(move)
        return assessment, self.analysis_event(game_id, after, ply)

    def assess_move(
        self, game_id: str, before: chess.Board, move: chess.Move, ply: int
    ) -> MoveAssessment:
        """What the move cost its mover (`assess_from`), the positions measured here when
        they are not measured yet: the position before, and the one after only when the move
        is not among the lines of the one before."""
        if move not in before.legal_moves:
            raise ValueError(f"{move.uci()} is not legal in {before.fen()}")
        measured_before = self.measure_for_verdict(before)
        measured_after = None
        if own_line(measured_before.lines, move) is None:
            after = before.copy()
            after.push(move)
            measured_after = self.measure_for_verdict(after)
        return self.assess_from(game_id, before, move, ply, measured_before, measured_after)

    def judgeable(self, measured_before: Measured, move: chess.Move) -> bool:
        """Whether the position before alone judges `move`: the move is among its lines."""
        return own_line(measured_before.lines, move) is not None

    def assess_from(
        self,
        game_id: str,
        before: chess.Board,
        move: chess.Move,
        ply: int,
        measured_before: Measured,
        measured_after: Measured | None = None,
    ) -> MoveAssessment:
        """What the move cost its mover, from searches already made. A move among the lines
        of the position before is judged from its own line there: the same search and the
        same depth as the best move it is set against, and the position after read as the
        rest of that line -- no search of its own (85% of a session's moves, the same label
        in 319 of 320: the engine audit, 2026-10-04). Any other move is judged from the
        position after's own measure (`measured_after`, needed then), as before."""
        # The history comes along: the opening a game is in is a fact about its moves, not
        # about the position, and the position after this move is the next move's history.
        after = before.copy()
        after.push(move)
        own = own_line(measured_before.lines, move)
        if own is not None:
            best_line = measured_before.lines[0]
            loss = (
                0.0 if own is best_line else max(0.0, best_line.expected_score - own.expected_score)
            )
            measured_after = Measured(
                fen=after.fen(),
                lines=verdict_lines(own, []),
                score=position_score(after, [line_after(own, 1)]),
                decisive=False,
                only_move=False,
                key=cache_key(after),
            )
        else:
            if measured_after is None:
                raise ValueError(
                    f"{move.uci()} is not among the lines: the position after is needed"
                )
            loss = eval_loss(measured_before.lines, move.uci(), measured_after.score)
        tags = self._move_tags(before, move, measured_after)
        best = measured_before.lines[0].moves if measured_before.lines else []
        previous = self._assessed.get(ply - 1)
        if previous is not None and (previous[0] != game_id or previous[1] != before.epd()):
            # Another game, or a move that did not lead here: the move before is unknown.
            previous = None
        label = verdict(
            before,
            move,
            measured_before.lines,
            measured_after.lines,
            loss=loss,
            before_score=measured_before.score,
            decisive=measured_before.decisive,
            in_book=self._in_book(after),
            sacrifice=any(tag.name == "sacrifice" for tag in tags),
            previous=(
                Previous(previous[2], previous[3], previous[4]) if previous is not None else None
            ),
            thresholds=self.thresholds,
        )
        self._assessed[ply] = (
            game_id,
            after.epd(),
            loss,
            move.to_square,
            before.is_capture(move),
        )
        assessment = MoveAssessment(
            game_id=game_id,
            ply=ply,
            uci=move.uci(),
            expected_loss=loss,
            severity=severity(loss, self.thresholds),
            best=best,
            tags=tags,
            depth=(
                measured_before.depth
                if own is not None
                else min(measured_before.depth, measured_after.depth or measured_before.depth)
            ),
            label=label,
            accuracy=move_accuracy(loss),
        )
        return assessment

    # Tagging and openings are optional collaborators, so a bare engine still works.

    def _position_tags(self, board: chess.Board, measured: Measured) -> list[MotifTag]:
        return position_tags(self.tagger, board, measured.lines)

    def _move_tags(self, before: chess.Board, move: chess.Move, after: Measured) -> list[MotifTag]:
        return move_tags(self.tagger, before, move, after.lines)

    def _in_book(self, board: chess.Board) -> bool:
        """Whether the game is still in theory after this move; needs the moves that got here."""
        if self.book is None or not board.move_stack:
            return False
        if board.root().fen() != chess.STARTING_FEN:
            return False
        return self.book.in_book(list(board.move_stack))

    def _opening(self, board: chess.Board) -> str | None:
        """The opening this game is in, when the board carries the moves that got here.

        A position read off the screen, or one adopted by a resync, has no history to look
        up, and a fragment of a history would name the wrong opening.
        """
        if self.book is None or not board.move_stack:
            return None
        if board.root().fen() != chess.STARTING_FEN:
            return None
        return opening_name(self.book, board.move_stack)


class WholeSearcher:
    """`lines` best lines of a position on one engine of its own, measured whole the one-line
    way (`GameAnnotator.measure`, no side searches) -- never several lines to the full depth at
    once, which explode on decided and forcing positions. For what searches with no analyser
    beside it: a review's line view, a coach made on demand. Each depth asked for keeps its own
    measurements."""

    def __init__(self, engine: Engine, *, lines: int, depth: int) -> None:
        self.engine = engine
        self.lines = lines
        self.depth = depth
        self._annotators: dict[int, GameAnnotator] = {}

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        """`board`'s lines whole; with `best_only`, its best line alone (or its whole, if
        measured)."""
        depth = depth or self.depth
        annotator = self._annotators.get(depth)
        if annotator is None:
            annotator = self._annotators[depth] = GameAnnotator(
                engine=self.engine, depth=depth, multipv=self.lines, with_threats=False
            )
        if best_only:
            return annotator.best_line(board).lines
        return annotator.measure(board).lines

    def close(self) -> None:
        self.engine.close()


def opening_name(book: Book, moves: Sequence[chess.Move]) -> str | None:
    """The opening `moves` from the start position are in, as the analysis carries it ("C50
    Italian Game"); None when the book names none."""
    found = book.identify(list(moves))
    if found is None:
        return None
    eco = getattr(found, "eco", None)
    name = getattr(found, "name", None)
    if name is None:
        return None
    return f"{eco} {name}" if eco else str(name)


def own_line(lines: Sequence[Line], move: chess.Move) -> Line | None:
    """The line among `lines` that begins with `move`, or None."""
    uci = move.uci()
    return next((line for line in lines if line.moves and line.moves[0] == uci), None)


def _rank(line: Line) -> tuple[float, int]:
    """A line's place among a position's lines: the better first; of equal score, a mate
    given sooner, then no mate, then a mate taken later (every mate scores alike)."""
    if line.mate is None:
        mate = 10**6
    elif line.mate > 0:
        mate = line.mate
    else:
        mate = 2 * 10**6 + line.mate
    return (-line.expected_score, mate)


def _first_move(line: Line) -> chess.Move | None:
    """A line's first move, or None for a line with none or one that does not parse."""
    moves = _moves(line)
    return moves[0] if moves else None


def _moves(line: Line) -> list[chess.Move]:
    out: list[chess.Move] = []
    for uci in line.moves:
        try:
            out.append(chess.Move.from_uci(uci))
        except ValueError:
            break
    return out


def verdict_lines(own: Line | None, after_lines: Sequence[Line]) -> list[Line]:
    """What a verdict reads of the position a move led to: for a move among the lines of the
    position before (`own`, its line there), the rest of that line -- the same search and depth
    as the best move it is set against; one that stops at the move leaves none. Any other
    move's, the position after's own lines (`after_lines`). The annotator judging a move and a
    saved review labelled again (`game_review.relabel`) both read it here, so they cannot
    drift."""
    if own is None:
        return list(after_lines)
    rest = line_after(own, 1)
    return [rest] if rest.moves else []


def position_tags(
    tagger: Tagger | None, board: chess.Board, lines: Sequence[Line]
) -> list[MotifTag]:
    """What the position holds for the side to move, from its best line; their mate distance
    counts their own next move, which is the same convention a move's tags use."""
    if tagger is None or not lines:
        return []
    return tagger.tags_for_position(board, pv=_moves(lines[0]), mate_in=lines[0].mate)


def move_tags(
    tagger: Tagger | None, before: chess.Board, move: chess.Move, after_lines: Sequence[Line]
) -> list[MotifTag]:
    """What the move did (a sacrifice is what makes a move brilliant), read with the line after
    it (`after_lines`, `verdict_lines`)."""
    if tagger is None:
        return []
    pv = _moves(after_lines[0]) if after_lines else []
    tags = tagger.tags_for_move(before, move, pv=pv, mate_in=_mate_in(before, move, after_lines))
    # A tactic the move makes is only its point when the line after it cashes it in:
    # "the queen on e7 is loose" was said of a queen that simply steps away, 98 times
    # in the master games at moves whose note names no tactic (2026-09-24).
    line = [move, *pv]
    boards = [before.copy(stack=False)]
    for m in line:
        if m not in boards[-1].legal_moves:
            break
        nxt = boards[-1].copy(stack=False)
        nxt.push(m)
        boards.append(nxt)
    line = line[: len(boards) - 1]
    return [t for t in tags if t.name not in motifs.TARGETED or motifs.pays_off(boards, line, 0, t)]


def _mate_in(before: chess.Board, move: chess.Move, after_lines: Sequence[Line]) -> int | None:
    """Moves the mover needs to mate, counting this one; negative when the mover gets mated.

    The engine reports the mate distance in the position after the move, where the opponent
    is to move and counts their own moves: "mated in two" there means two more moves by the
    mover, three counting the one just played.
    """
    after = before.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return 1
    if not after_lines or after_lines[0].mate is None:
        return None
    mate = after_lines[0].mate
    return -mate if mate > 0 else abs(mate) + 1
