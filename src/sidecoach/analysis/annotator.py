"""From moves to assessments: what a move cost, what the position holds, what is threatened.

`GameAnnotator` does the work for one game and is synchronous, which is what a post-game
review wants. `Analyser` wraps it in a worker thread, which is what a live session needs:
the engine takes about a third of a second per position and the capture loop must not wait
for it.

Every position of a game is measured once, to the same depth, and cached, so the position
after one move is the position before the next and a game costs one analysis per ply. The
matching depth is what makes the loss of a move meaningful: two searches of different
lengths would differ by more than most mistakes do.
"""

from __future__ import annotations

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
    verdict,
)
from sidecoach.analysis.threats import AIM_DEPTH, aim_lines, threat_moves
from sidecoach.analysis.traps import TRAP_DEPTH, trap_lines
from sidecoach.events import Analysis, Line, MotifTag, MoveAssessment

# The first depth of a position's search told to its readers, each deeper one following as
# the search finishes it: its best move is the full depth's in every one of 77 puzzle
# positions and 81 % of 232 game positions, found in a twentieth of the time, so a hint, the
# move or the line asked for at once is answered at once (Tai, 2026-09-28: "first usable
# answer is depth 12 ... the computation continues until max depth of 18"). Depth 12 of the
# position's own search: a search of its own to depth 12 first cost 27% more than the full
# depth alone (the engine audit, 2026-10-04).
EARLY_DEPTH = 12


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
    """Assessments for the moves of one game. Not thread-safe; `Analyser` owns one."""

    engine: Engine
    depth: int = 18
    multipv: int = 6
    thresholds: Thresholds = field(default_factory=Thresholds)
    tagger: Tagger | None = None
    book: Book | None = None
    with_threats: bool = True
    _cache: dict[str, Measured] = field(default_factory=dict, init=False, repr=False)
    # Positions measured for a verdict only: the best line at the full depth, no
    # candidates, no threats. Kept apart from `_cache`, so a later whole measurement of
    # the same position is not answered with the thin one.
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
    # Positions measured ahead of the game (`look_ahead`) and not yet reached, and how many
    # the game has reached: what looking ahead saved.
    _ahead: set[str] = field(default_factory=set, init=False, repr=False)
    ahead_hits: int = field(default=0, init=False)
    # Every search's lines as it finishes each usable depth, for any reader at any time (the
    # coach's engine reads it before it searches).
    computed: Computed = field(default_factory=Computed, init=False, repr=False)
    # How many times `forget_measured` has run: a search that began before it (another
    # thread changed the settings mid-search) was made the old way, and is not kept.
    _era: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        """Forget the cache; a new game shares no positions with the last one."""
        self._cache.clear()
        self._verdict_cache.clear()
        self._ahead.clear()
        self.computed.clear()
        self._assessed.clear()

    def forget_measured(self) -> None:
        """Forget the positions measured, keeping the game's moves: the depth, the lines, the
        time cap, the curve or the thresholds changed, and a loss compares two positions
        measured the same way, so none measured before is set against one measured after.
        Called from another thread (the settings' save): a search under way now finishes,
        and is handed to its caller, but is not kept."""
        self._era += 1
        self._cache.clear()
        self._verdict_cache.clear()
        self._ahead.clear()
        self.computed.clear()

    def look_ahead(self, board: chess.Board) -> Measured:
        """A position the game may reach, measured whole before it does, as the game's own
        positions are, and cached with them. Inside `Engine.interruptible` a search stopped
        raises `Interrupted`, and nothing is cached."""
        cached = self._cache.get(cache_key(board))
        if cached is not None:
            return cached
        measured = self.measure(board)
        if measured.key in self._cache:
            self._ahead.add(measured.key)
        return measured

    def _cached(self, board: chess.Board) -> Measured | None:
        """The position measured whole, counting a position measured ahead reached. A
        position adopted without a key (`adopt`) is found by its FEN."""
        key = cache_key(board)
        measured = self._cache.get(key)
        if measured is None:
            measured = self._cache.get(board.fen())
        if measured is not None and key in self._ahead:
            self._ahead.discard(key)
            self.ahead_hits += 1
        return measured

    def measure_for_verdict(self, board: chess.Board) -> Measured:
        """The position measured for a verdict: whole if it already is, else its best
        line alone at the full depth -- what `eval_loss` needs, at a fraction of the
        cost. When the engine is behind a fast game, this is all a stale ply gets
        (2026-09-22: a queen ending played at two seconds a move ran the analyser forty
        plies behind, and every word on a move came after the board had moved on)."""
        key = cache_key(board)
        whole = self._cached(board)
        if whole is not None:
            return whole
        thin = self._verdict_cache.get(key)
        if thin is not None:
            return thin
        era = self._era
        lines = self.engine.analyse(board, depth=self.depth, multipv=1)
        thin = Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=False,
            only_move=False,
            key=key,
        )
        if era == self._era:  # not searched the old way (`forget_measured`)
            self.computed.add(board, lines, thin=True)
            self._verdict_cache[key] = thin
        return thin

    def measure(
        self,
        board: chess.Board,
        *,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> Measured:
        """The position whole, at the full depth: the candidates, the threats, the
        decisive-moment and only-move calls. `on_depth`: handed the candidates of each
        depth the search finishes on its way. A position the player is not to wait for is
        measured early first (`measure_early`), never in place of this."""
        cached = self._cached(board)
        if cached is not None:
            return cached
        era = self._era
        measured = self._measured(board, self.depth, on_depth=on_depth)
        if era == self._era:  # not searched the old way (`forget_measured`)
            self._cache[measured.key] = measured
        return measured

    def measure_early(self, board: chess.Board) -> Measured:
        """The position whole at EARLY_DEPTH, its side searches no deeper: what the player
        can be told while the full depth is searched. Not cached: the full depth follows."""
        return self._measured(board, min(EARLY_DEPTH, self.depth), early=True)

    def _measured(
        self,
        board: chess.Board,
        depth: int,
        *,
        early: bool = False,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> Measured:
        measured = self.search(board, depth=depth, early=early, on_depth=on_depth)
        return self.with_side(board, measured, depth=depth)

    def search(
        self,
        board: chess.Board,
        *,
        engine: Engine | None = None,
        depth: int | None = None,
        early: bool = False,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> Measured:
        """The position's own search, nothing beside it: its lines at `depth` (the full depth
        unless told), the decisive-moment and only-move calls read from them. `engine`: the
        one to search with (a worker's own; the annotator's unless told). Each depth from the
        first usable one is kept for any reader as the search finishes it (`computed`), and
        handed to `on_depth`."""
        engine = engine or self.engine
        depth = depth or self.depth
        usable = min(EARLY_DEPTH, depth)
        era = self._era

        def finished(found: list[Line]) -> None:
            """A depth done: kept from the first usable one, and handed on."""
            if found and found[0].depth >= usable and era == self._era:
                self.computed.add(board, found)
            if on_depth is not None:
                on_depth(found)

        lines = engine.analyse(
            board,
            depth=depth,
            multipv=self.multipv,
            # The best line carried on to its point costs seconds; the full depth does it.
            carry_best=not early,
            on_depth=finished,
        )
        # Before the side searches: the lines are the news, and a reader has them now.
        if era == self._era:
            self.computed.add(board, lines)
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            key=cache_key(board),
        )

    def with_side(
        self,
        board: chess.Board,
        measured: Measured,
        *,
        engine: Engine | None = None,
        depth: int | None = None,
    ) -> Measured:
        """`measured` with its side searches: what the coach's words on the position are made
        of (the threats, the traps, the declined recaptures, what a quiet best move
        threatens). The position's own lines are what they search from; none of them is a
        search of the position again. Nothing when the annotator searches no threats."""
        if not self.with_threats:
            return measured
        engine = engine or self.engine
        depth = depth or self.depth
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

    def analysis_event(
        self,
        game_id: str,
        board: chess.Board,
        ply: int,
        *,
        thin: bool = False,
        early: bool = False,
        on_depth: Callable[[list[Line]], None] | None = None,
    ) -> Analysis:
        """What the position holds, for the side to move. `thin`: only what the verdict
        already measured (the best line; no threats, no decisive call) -- the analysis
        of a position the game has left, so the coach still has the facts of the
        position each move was played from and the review has its score. `early`: at
        EARLY_DEPTH (`measure_early`), a provisional analysis the full depth follows."""
        if early:
            measured = self.measure_early(board)
        elif thin:
            measured = self.measure_for_verdict(board)
        else:
            measured = self.measure(board, on_depth=on_depth)
        return self.analysis_from(game_id, board, ply, measured, provisional=early)

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
        `provisional`: a depth on the way to the full one, or the full one before its side
        searches -- what follows replaces it."""
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

    def measured_lines(self, board: chess.Board, lines: list[Line]) -> Measured:
        """`lines` of `board` as a measurement: the calls they make (a depth on the way)."""
        return Measured(
            fen=board.fen(),
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
            key=cache_key(board),
        )

    def deepened(self, early: Analysis, board: chess.Board, lines: list[Line]) -> Analysis:
        """The early analysis of `board` with the lines of a deeper depth: its candidates,
        the decisive-moment and only-move calls and the pattern read from them again; its
        threats, traps and defences the early depth's, until the full depth brings its own."""
        measured = Measured(
            fen=early.fen,
            lines=lines,
            score=position_score(board, lines),
            decisive=is_decisive(board, lines, self.thresholds),
            only_move=is_only_move(board, lines, self.thresholds),
        )
        return early.model_copy(
            update={
                "lines": lines,
                "decisive": measured.decisive,
                "only_move": measured.only_move,
                "tags": self._position_tags(board, measured),
            }
        )

    def measured(self, board: chess.Board) -> bool:
        """Whether the position is measured whole already (the game's own, or ahead of it)."""
        return self._cached_quietly(board) is not None

    def adopt(self, measured: Iterable[Measured]) -> None:
        """Positions measured whole by another annotator -- another engine, on another thread,
        for a review measuring a game's positions side by side (`game_review`) -- kept as if
        measured here: the walk through the game's moves then searches nothing."""
        for m in measured:
            self._cache[m.key or m.fen] = m

    def _cached_quietly(self, board: chess.Board) -> Measured | None:
        """`_cached` without counting a position measured ahead as reached."""
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
            rest = line_after(own, 1)
            # The rest of the line is the position after's line; one that stops at the move
            # still carries the position's score.
            after_lines = [rest] if rest.moves else []
            measured_after = Measured(
                fen=after.fen(),
                lines=after_lines,
                score=position_score(after, [rest]),
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
        """What the position holds for the side to move; their mate distance counts their
        own next move, which is the same convention a move's tags use."""
        if self.tagger is None or not measured.lines:
            return []
        return self.tagger.tags_for_position(
            board, pv=_moves(measured.lines[0]), mate_in=measured.lines[0].mate
        )

    def _move_tags(self, before: chess.Board, move: chess.Move, after: Measured) -> list[MotifTag]:
        if self.tagger is None:
            return []
        pv = _moves(after.lines[0]) if after.lines else []
        tags = self.tagger.tags_for_move(
            before, move, pv=pv, mate_in=_mate_in(before, move, after.lines)
        )
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
        return [
            t for t in tags if t.name not in motifs.TARGETED or motifs.pays_off(boards, line, 0, t)
        ]

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


def _moves(line: Line) -> list[chess.Move]:
    out: list[chess.Move] = []
    for uci in line.moves:
        try:
            out.append(chess.Move.from_uci(uci))
        except ValueError:
            break
    return out


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
