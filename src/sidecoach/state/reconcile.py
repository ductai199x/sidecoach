"""Reconcile per-frame cell readings into a legal python-chess game.

The core idea from DESIGN.md section 5.3: detect moves, not positions. Given the current
legal position, a frame is explained by the legal move whose touched squares match what the
cells now show, or, when moves were missed, by a short path of moves. Everything else is
held: a piece lifted mid-drag empties one square and matches no move, a promotion dialog
hides cells, an arrow makes a cell unreadable.

Cell readings can be exact (a symbol), occupied (a piece whose type is hidden by a badge),
empty, or unknown; unknown cells carry no evidence either way. A move is committed only after
it has explained `settle_frames` consecutive frames, and, once the game has shown that it
highlights the last move, only when both its squares are highlighted. That is what stops a
dragged piece drawn over a neighbouring square from being taken as a move.

Browsing history is normal: the user steps back through the move list with the arrow keys
or by clicking a move, and the board shows an earlier position. A frame that matches an
earlier position of this game puts the reconciler in a viewing state; nothing is undone. A
takeback is recorded only when a different move is then played from that earlier position,
which is the only way the two can be told apart on the board.

Cells that keep disagreeing without any move explaining them are treated as stale after a
few frames and ignored, so one unreadable cell cannot stall the game. Resync fires only when
several cells stay unexplained for a long time, which a long drag never triggers.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

import chess

from sidecoach.events import (
    Event,
    GameEnded,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    Side,
    Takeback,
)
from sidecoach.vision.pieces import CellReading
from sidecoach.vision.squares import square_at

Readings = list[list[CellReading]]
Observation = dict[chess.Square, CellReading]
Explanation = tuple[chess.Move, ...]
PendingKey = tuple[object, ...]


def touched_squares(board: chess.Board, move: chess.Move) -> set[chess.Square]:
    touched = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            touched |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            touched |= {chess.square(0, rank), chess.square(3, rank)}
    if board.is_en_passant(move):
        touched.add(
            chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        )
    return touched


def cell_agrees(reading: CellReading, piece: chess.Piece | None) -> bool:
    """Whether a reading is consistent with a square's content.

    A piece reading without a symbol only asserts that the square is occupied. Its colour is
    not trusted: a capture ring drawn around an enemy piece during a drag can flip the
    measured luminance, and a wrong colour would let the move search fabricate captures.
    """
    if reading.kind == "unknown":
        return True
    if reading.kind == "empty":
        return piece is None
    if piece is None:
        return False
    if reading.symbol is not None:
        return piece.symbol() == reading.symbol
    return True


def exact(reading: CellReading) -> bool:
    return reading.kind == "empty" or (reading.kind == "piece" and reading.symbol is not None)


def matches(board: chess.Board, obs: Observation, squares: set[chess.Square]) -> bool:
    return all(cell_agrees(obs[sq], board.piece_at(sq)) for sq in squares if sq in obs)


def single_explanations(
    board: chess.Board, obs: Observation, diff: set[chess.Square]
) -> list[Explanation]:
    """Legal moves from `board` whose touched squares cover `diff` and agree with the readings."""
    found: list[Explanation] = []
    for move in board.legal_moves:
        touched = touched_squares(board, move)
        if not diff <= touched:
            continue
        after = board.copy(stack=False)
        after.push(move)
        if matches(after, obs, touched):
            found.append((move,))
    return found


@dataclass
class Reconciler:
    bottom: Side
    user_side: Side = Side.WHITE
    settle_frames: int = 2
    max_plies: int = 4
    search_budget: int = 4000
    history_depth: int = 40
    stale_after: int = 5
    resync_cells: int = 3
    resync_after: int = 150
    board: chess.Board = field(default_factory=chess.Board)
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    highlights: bool | None = None
    viewing: int | None = None
    _unhighlighted_commits: int = 0
    _hover_seen: bool = False
    _pending: tuple[PendingKey, int] | None = None
    _stale: dict[chess.Square, tuple[tuple[object, ...], int]] = field(default_factory=dict)
    _unexplained_frames: int = 0
    _started: bool = False
    _finished: bool = False
    # Position resync: adopt a fully readable, stable board that no move story explains.
    resync_hold_frames: int = 10
    max_stale_subset_cells: int = 4
    resync_max_soft: int = 2
    _base_fen: str = chess.STARTING_FEN
    _turn_unknown: bool = False
    _resync_candidate: tuple[str, int] | None = None
    # Last stable reading of every cell and the frame it was seen in, plus the key of the
    # previous frame's reading, which a reading must repeat to count as stable.
    memory_frames: int = 30
    _last_seen: dict[chess.Square, tuple[CellReading, int]] = field(default_factory=dict)
    _previous: dict[chess.Square, tuple[object, ...]] = field(default_factory=dict)
    _frame_index: int = 0

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def base_fen(self) -> str:
        """The position the move stack starts from: the start, or the last resync."""
        return self._base_fen

    @property
    def turn_unknown(self) -> bool:
        return self._turn_unknown

    # Observation helpers

    def observed(self, readings: Readings) -> Observation:
        out: Observation = {}
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                if reading.kind != "unknown":
                    out[square_at(r, c, self.bottom)] = reading
        return out

    def _observe(self, readings: Readings) -> Observation:
        """Readings as evidence, with covered cells remembered for a short while.

        The game-over dialog fades in about a second after the mating move and hides the
        squares it came from. Nothing changes under a dialog, so a cell a confirmed overlay
        has covered keeps its last stable reading for `memory_frames`; the explanation
        found while the board was still uncovered then keeps settling under the dialog
        instead of turning ambiguous. Only an exact, settled reading repeated in two
        consecutive frames is remembered: the fading dialog and a piece in flight both
        produce readings that change from frame to frame, and neither may be kept.
        """
        self._frame_index += 1
        out: Observation = {}
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                sq = square_at(r, c, self.bottom)
                if reading.kind == "unknown":
                    self._previous.pop(sq, None)
                    if reading.covered:
                        remembered = self._last_seen.get(sq)
                        if remembered and self._frame_index - remembered[1] <= self.memory_frames:
                            out[sq] = remembered[0]
                    continue
                out[sq] = reading
                if not (exact(reading) and reading.settled and not reading.hover):
                    self._previous.pop(sq, None)
                    continue
                key = (reading.kind, reading.symbol, reading.colour)
                if self._previous.get(sq) == key:
                    self._last_seen[sq] = (reading, self._frame_index)
                self._previous[sq] = key
        return out

    def disagreements(self, obs: Observation) -> set[chess.Square]:
        return {
            sq for sq, reading in obs.items() if not cell_agrees(reading, self.board.piece_at(sq))
        }

    # Lifecycle

    def start(self) -> list[Event]:
        self.board = chess.Board()
        self.game_id = uuid.uuid4().hex[:8]
        self._pending = None
        self._stale = {}
        self._unexplained_frames = 0
        self._started = True
        self._finished = False
        self.highlights = None
        self.viewing = None
        self._unhighlighted_commits = 0
        self._base_fen = chess.STARTING_FEN
        self._turn_unknown = False
        self._resync_candidate = None
        self._last_seen = {}
        self._previous = {}
        return [
            GameStarted(game_id=self.game_id, user_side=self.user_side, start_fen=self.board.fen()),
            PositionChanged(game_id=self.game_id, fen=self.board.fen(), ply=0),
        ]

    # Explanations

    def _path_explanations(self, obs: Observation, diff: set[chess.Square]) -> list[Explanation]:
        """Sequences of up to `max_plies` legal moves explaining `diff`.

        This is the catch-up for missed moves: the window was minimised, a dialog covered the
        board, or a badge hid a piece until the next move. Every move must touch a still
        unexplained square, every touched square must be read exactly, and the final position
        must agree with every readable cell. The shortest paths are kept, as a person would
        assume the simplest story; transpositions among them lead to one position and are
        accepted, while different final positions are a genuine ambiguity and nothing is
        committed.
        """
        # A move touches at most four squares (castling), so more disagreements than that
        # times the depth cannot be a missed-move story.
        if len(diff) < 3 or len(diff) > 4 * self.max_plies:
            return []

        # Iterative deepening: the shortest story is searched exhaustively before any
        # longer one is attempted, so a budget spent on deep branches can never hide a
        # simple two-ply explanation. Each depth also prunes branches whose remaining
        # plies cannot touch the cells still unexplained.
        for depth in range(2, self.max_plies + 1):
            solutions: list[Explanation] = []
            budget = [self.search_budget]

            def search(
                board: chess.Board,
                remaining: set[chess.Square],
                path: list[chess.Move],
                depth: int = depth,
                solutions: list[Explanation] = solutions,
                budget: list[int] = budget,
            ) -> None:
                plies_left = depth - len(path)
                if plies_left <= 0 or len(solutions) >= 64 or budget[0] <= 0:
                    return
                if len(remaining) > 4 * plies_left:
                    return
                for move in board.legal_moves:
                    budget[0] -= 1
                    if budget[0] <= 0:
                        return
                    touched = touched_squares(board, move)
                    if not (touched & remaining):
                        continue
                    if not all(sq in obs and exact(obs[sq]) for sq in touched):
                        continue
                    after = board.copy(stack=False)
                    after.push(move)
                    path.append(move)
                    still = remaining - touched
                    if not still:
                        if matches(after, obs, set(obs)):
                            solutions.append(tuple(path))
                    else:
                        search(after, still, path)
                    path.pop()

            search(self.board, set(diff), [])
            if not solutions:
                continue
            finals: set[str] = set()
            for path in solutions:
                board = self.board.copy(stack=False)
                for move in path:
                    board.push(move)
                finals.add(board.board_fen() + " " + ("w" if board.turn else "b"))
            if len(finals) > 1:
                return solutions
            return [solutions[0]]
        return []

    def _history_index(self, obs: Observation, diff: set[chess.Square]) -> int | None:
        """Index of the earlier position of this game the frame shows, if any."""
        earlier = self.board.copy()
        undone: set[chess.Square] = set()
        for back in range(1, min(len(self.board.move_stack), self.history_depth) + 1):
            move = earlier.pop()
            undone |= touched_squares(earlier, move)
            if diff <= undone and matches(earlier, obs, set(obs)):
                return len(self.board.move_stack) - back
        return None

    def _board_at(self, index: int) -> chess.Board:
        earlier = self.board.copy()
        while len(earlier.move_stack) > index:
            earlier.pop()
        return earlier

    def _highlight_ok(
        self, obs: Observation, moves: Explanation, base: chess.Board | None = None
    ) -> bool:
        """Geometry evidence that `moves` are complete, independent of any colour.

        `base` is the position the moves start from; the latest position by default, or
        an earlier one when the user plays on from a browsed position.

        A settled piece is drawn exactly where its exemplar from the start position sits in
        the square; a dragged piece follows the cursor and an animating piece is between
        squares, so neither is centred. During a drag the site also outlines the hovered
        square. So every square the moves touch must hold a settled piece or be empty, and no
        square on the board may carry a hover outline. Highlights are not consulted: their
        colours vary by theme and by what the user clicks, and the reconciler stays agnostic
        to them by design (DESIGN.md section 5.3).
        """
        if self._hover_seen or any(reading.hover for reading in obs.values()):
            return False
        touched: set[chess.Square] = set()
        board = (base if base is not None else self.board).copy(stack=False)
        for move in moves:
            if move not in board.legal_moves:
                return False
            touched |= touched_squares(board, move)
            board.push(move)
        for sq in touched:
            reading = obs.get(sq)
            if reading is not None and reading.kind == "piece" and not reading.settled:
                return False
        return True

    # Main entry

    def update(self, readings: Readings, ts: float, think_ms: int | None = None) -> list[Event]:
        events = self.start() if not self._started else []
        # A hover outline anywhere means a drag is in progress, even on a cell whose content
        # could not be read; it must be seen before unknown cells are dropped.
        self._hover_seen = any(reading.hover for row in readings for reading in row)
        obs = self._observe(readings)
        diff = self.disagreements(obs)
        self._advance_stale(diff, obs)

        if not diff:
            # The frame shows the latest position: live again if the user was browsing.
            self._pending = None
            self.viewing = None
            self._unexplained_frames = 0
            return events

        # Explain the whole disagreement first. Only if nothing fits, retry ignoring cells
        # that have disagreed with an unchanged reading for a while: a badge or a ring on
        # one cell must not stall the game, but a stale cell may also hold a real move whose
        # commit was blocked for other reasons, so it is never dropped from the first try.
        explained = self._explain(obs, diff, obs, think_ms, allow_paths=True)
        if explained is not None:
            return events + explained
        stale = (
            {sq for sq in diff if self._stale[sq][1] > self.stale_after}
            if self.viewing is None
            else set()
        )
        if stale:
            # A badge or a ring on one or two cells must not stall the game, but a move whose
            # commit it blocked has cells just as stale as the badge itself. So the stale
            # cells are ignored one at a time, junk-looking readings first, then in pairs;
            # the rest of the disagreement must still be explained by legal moves, which is
            # what keeps a board that changed wholesale from being explained piecemeal.
            for ignored in self._stale_subsets(stale, obs):
                fresh = diff - ignored
                if not fresh:
                    continue
                live_obs = {sq: r for sq, r in obs.items() if sq not in ignored}
                explained = self._explain(
                    live_obs, fresh, obs, think_ms, allow_paths=True, cautious=True
                )
                if explained is not None:
                    return events + explained
            if stale == diff and len(stale) < self.resync_cells:
                # Only stale garbage differs from the latest position.
                self._pending = None
                self.viewing = None
                return events + self._maybe_resync(diff)

        self._pending = None

        # New game: the start position is back on the board.
        if self.board.move_stack and len(obs) >= 48 and matches(chess.Board(), obs, set(obs)):
            if not self._finished:
                events.append(self._ended("*", "new game started"))
            events.extend(self.start())
            return events

        adopted = self._maybe_position_resync(obs, diff)
        if adopted:
            return events + adopted
        return events + self._maybe_resync(diff)

    # Position resync

    def _maybe_position_resync(self, obs: Observation, diff: set[chess.Square]) -> list[Event]:
        """Adopt the observed position when it is complete, still, and unexplained.

        This is what recovers from a gap no move search can bridge: the window minimised
        for a minute, a dialog over the board, a restart mid-game. Castling rights are kept
        only for a king and rook still on their home squares, and whose turn it is, when the
        position does not settle it, is resolved by the first move seen afterwards.
        """
        if len(diff) < self.resync_cells or len(obs) < 64:
            self._resync_candidate = None
            return []
        if any(r.hover for r in obs.values()):
            self._resync_candidate = None
            return []
        soft = 0
        for reading in obs.values():
            if reading.kind != "piece":
                continue
            if reading.symbol is None:
                # A badge or ring hides the type; the best guess stands in, for a few cells.
                if reading.guess is None:
                    self._resync_candidate = None
                    return []
                soft += 1
            elif not reading.settled:
                self._resync_candidate = None
                return []
        if soft > self.resync_max_soft:
            self._resync_candidate = None
            return []
        placement = self._placement(obs).board_fen()
        if self._resync_candidate and self._resync_candidate[0] == placement:
            self._resync_candidate = (placement, self._resync_candidate[1] + 1)
        else:
            self._resync_candidate = (placement, 1)
        if self._resync_candidate[1] < self.resync_hold_frames:
            return []

        board = self._placement(obs)
        board.castling_rights = chess.BB_EMPTY
        for colour, rank in ((chess.WHITE, 0), (chess.BLACK, 7)):
            king = chess.Piece(chess.KING, colour)
            rook = chess.Piece(chess.ROOK, colour)
            if board.piece_at(chess.square(4, rank)) == king:
                if board.piece_at(chess.square(7, rank)) == rook:
                    board.castling_rights |= chess.BB_SQUARES[chess.square(7, rank)]
                if board.piece_at(chess.square(0, rank)) == rook:
                    board.castling_rights |= chess.BB_SQUARES[chess.square(0, rank)]
        valid_turns = []
        for turn in (chess.WHITE, chess.BLACK):
            board.turn = turn
            if board.is_valid():
                valid_turns.append(turn)
        if not valid_turns:
            self._resync_candidate = None
            return []
        board.turn = valid_turns[0]
        self._turn_unknown = len(valid_turns) > 1
        self.board = board
        self._base_fen = board.fen()
        self._pending = None
        self._stale = {}
        self.viewing = None
        self._unexplained_frames = 0
        self._resync_candidate = None
        reason = f"position adopted after {len(diff)} unexplained cells"
        if self._turn_unknown:
            reason += "; side to move pending"
        return [
            Resync(game_id=self.game_id, fen=board.fen(), reason=reason),
            PositionChanged(game_id=self.game_id, fen=board.fen(), ply=0),
        ]

    def _placement(self, obs: Observation) -> chess.Board:
        board = chess.Board(None)
        for sq, reading in obs.items():
            symbol = reading.symbol or reading.guess
            if reading.kind == "piece" and symbol is not None:
                board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
        return board

    def _resolve_turn(self, obs: Observation, diff: set[chess.Square]) -> bool:
        """After a resync with unknown side to move, let the first move decide it."""
        found: list[tuple[bool, Explanation]] = []
        for turn in (chess.WHITE, chess.BLACK):
            trial = self.board.copy(stack=False)
            trial.turn = turn
            found += [(turn, expl) for expl in single_explanations(trial, obs, diff)]
        if len(found) != 1:
            return False
        self.board.turn = found[0][0]
        self._base_fen = self.board.fen()
        self._turn_unknown = False
        return True

    def _explain(
        self,
        obs: Observation,
        diff: set[chess.Square],
        full_obs: Observation,
        think_ms: int | None,
        *,
        allow_paths: bool,
        cautious: bool = False,
    ) -> list[Event] | None:
        """Events for one explanation of `diff`, or None when nothing fits.

        Order: one move from the latest position; an earlier position of this game (the user
        is browsing); a different move from the viewed position (a takeback); a short path of
        missed moves. `full_obs` carries the highlight evidence. `cautious` marks the retry
        that ignores stale cells; its paths need an extra frame to settle.
        """
        if self._turn_unknown and not self._resolve_turn(obs, diff):
            return None
        singles = single_explanations(self.board, obs, diff)
        if len(singles) == 1 and self._highlight_ok(full_obs, singles[0]):
            return self._settle(
                ("moves", singles[0]), full_obs, self._commit_moves(singles[0], think_ms)
            )

        index = self._history_index(obs, diff)
        if index is not None:
            if index == 0 and self._finished and self._base_fen == chess.STARTING_FEN:
                # The start position after a finished game is the next game, not browsing.
                return self.start()
            self.viewing = index
            self._pending = None
            self._stale = {}
            self._unexplained_frames = 0
            return []

        if self.viewing is not None:
            earlier = self._board_at(self.viewing)
            diff_earlier = {sq for sq, r in obs.items() if not cell_agrees(r, earlier.piece_at(sq))}
            branches = single_explanations(earlier, obs, diff_earlier)
            if len(branches) == 1 and self._highlight_ok(full_obs, branches[0], earlier):
                if self.viewing == 0 and self._base_fen == chess.STARTING_FEN:
                    # A move from the start position while the game had moves: a new game
                    # began (resigned, aborted, or rematch), not a takeback of everything.
                    # After a position resync, position 0 is the adopted position, and a
                    # move from it is an ordinary takeback.
                    return self._settle(
                        ("newgame", branches[0]),
                        full_obs,
                        self._commit_new_game(branches[0], think_ms),
                    )
                plies = len(self.board.move_stack) - self.viewing
                return self._settle(
                    ("takeback", self.viewing, branches[0]),
                    full_obs,
                    self._commit_takeback(plies, branches[0], think_ms),
                )

        if allow_paths:
            paths = self._path_explanations(obs, diff)
            if len(paths) == 1 and self._highlight_ok(full_obs, paths[0]):
                key = ("moves-cautious" if cautious else "moves", paths[0])
                return self._settle(key, full_obs, self._commit_moves(paths[0], think_ms))
        return None

    # Settling and committing

    def _settle(
        self, key: PendingKey, obs: Observation, commit: Callable[[], list[Event]]
    ) -> list[Event]:
        """Count frames an explanation has held; commit it once it has settled."""
        if self._pending and self._pending[0] == key:
            self._pending = (key, self._pending[1] + 1)
        else:
            self._pending = (key, 1)
        moves = key[len(key) - 1]
        assert isinstance(moves, tuple)
        needed = (
            self.settle_frames
            + (1 if len(moves) > 1 else 0)
            + (1 if key[0] == "moves-cautious" else 0)
        )
        if self._pending[1] < needed:
            return []
        last = moves[len(moves) - 1]
        both = all(sq in obs and obs[sq].highlighted for sq in (last.from_square, last.to_square))
        if self.highlights is None:
            if both:
                self.highlights = True
            else:
                self._unhighlighted_commits += 1
                if self._unhighlighted_commits >= 3:
                    self.highlights = False
        events = commit()
        self._pending = None
        self._stale = {}
        self._unexplained_frames = 0
        self.viewing = None
        return events

    def _commit_moves(self, moves: Explanation, think_ms: int | None) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            events: list[Event] = []
            for move in moves:
                events.extend(self._commit(move, think_ms))
            return events

        return commit

    def _commit_takeback(
        self, plies: int, moves: Explanation, think_ms: int | None
    ) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            for _ in range(plies):
                self.board.pop()
            events: list[Event] = [
                Takeback(game_id=self.game_id, plies=plies, fen=self.board.fen()),
                PositionChanged(
                    game_id=self.game_id, fen=self.board.fen(), ply=len(self.board.move_stack)
                ),
            ]
            for move in moves:
                events.extend(self._commit(move, think_ms))
            return events

        return commit

    def _commit_new_game(
        self, moves: Explanation, think_ms: int | None
    ) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            events: list[Event] = []
            if not self._finished:
                events.append(self._ended("*", "new game started"))
            events.extend(self.start())
            for move in moves:
                events.extend(self._commit(move, think_ms))
            return events

        return commit

    def _stale_subsets(
        self, stale: set[chess.Square], obs: Observation
    ) -> list[frozenset[chess.Square]]:
        """Stale cells to ignore, singles before pairs, junk-looking readings first.

        At most `resync_cells - 1` cells are ever ignored together: more than that is no
        longer a badge or two, and only a position resync may act. Junk readings (an
        unsettled piece, a piece of no known kind) are what badges produce, so they are
        always tried, however many cells of a blocked move have turned stale alongside
        them; clean stale cells join only when the stale set is small enough to enumerate.
        """

        def is_junk(sq: chess.Square) -> bool:
            reading = obs[sq]
            unsettled = reading.kind == "piece" and not reading.settled
            nameless = reading.kind == "piece" and reading.symbol is None
            return unsettled or nameless

        junk = sorted(sq for sq in stale if is_junk(sq))
        clean = sorted(sq for sq in stale if not is_junk(sq))
        pool = junk if len(stale) > self.max_stale_subset_cells else junk + clean
        subsets: list[frozenset[chess.Square]] = [frozenset({sq}) for sq in pool]
        if self.resync_cells > 2:
            subsets += [frozenset({a, b}) for i, a in enumerate(pool) for b in pool[i + 1 :]]
        return subsets

    def _maybe_resync(self, diff: set[chess.Square]) -> list[Event]:
        """Resync only when several cells have stayed unexplained for a long time."""
        if not diff:
            self._unexplained_frames = 0
            return []
        self._unexplained_frames += 1
        persistent = [sq for sq in diff if self._stale[sq][1] > self.stale_after]
        if len(persistent) < self.resync_cells or self._unexplained_frames < self.resync_after:
            return []
        self._unexplained_frames = 0
        names = ", ".join(chess.square_name(sq) for sq in sorted(persistent))
        return [Resync(game_id=self.game_id, fen=self.board.fen(), reason=f"unexplained: {names}")]

    def _advance_stale(self, diff: set[chess.Square], obs: Observation) -> None:
        """Count consecutive frames each cell has disagreed with an unchanged reading.

        A changed reading resets the count, so a cell that was stale during a long drag
        becomes fresh again the moment the piece lands and the highlight appears.
        """
        for sq in list(self._stale):
            if sq not in diff:
                del self._stale[sq]
        for sq in diff:
            reading = obs[sq]
            key = (reading.kind, reading.symbol, reading.colour, reading.highlighted)
            previous = self._stale.get(sq)
            if previous is not None and previous[0] == key:
                self._stale[sq] = (key, previous[1] + 1)
            else:
                self._stale[sq] = (key, 1)

    def _commit(self, move: chess.Move, think_ms: int | None) -> list[Event]:
        if move not in self.board.legal_moves:
            # Defensive: an explanation computed against another position must never
            # corrupt the game. Report it and let the next frames explain the board again.
            self._pending = None
            return [
                Resync(
                    game_id=self.game_id,
                    fen=self.board.fen(),
                    reason=f"refused illegal commit {move.uci()}",
                )
            ]
        fen_before = self.board.fen()
        san = self.board.san(move)
        by_user = (self.board.turn == chess.WHITE) == (self.user_side == Side.WHITE)
        self.board.push(move)
        ply = len(self.board.move_stack)
        events: list[Event] = [
            MovePlayed(
                game_id=self.game_id,
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=self.board.fen(),
                by_user=by_user,
                think_ms=think_ms,
            ),
            PositionChanged(game_id=self.game_id, fen=self.board.fen(), ply=ply),
        ]
        if self.board.is_game_over():
            self._finished = True
            events.append(self._ended(self.board.result(), "game over on board"))
        return events

    def _ended(self, result: str, reason: str) -> GameEnded:
        return GameEnded(
            game_id=self.game_id,
            result=result,
            reason=reason,
            pgn=self.pgn(),
            final_fen=self.board.fen(),
        )

    def pgn(self) -> str:
        """Moves since the base position; after a resync the numbering restarts at 1."""
        replay = chess.Board(self._base_fen)
        parts: list[str] = []
        for i, move in enumerate(self.board.move_stack):
            if move not in replay.legal_moves:
                parts.append(f"[{move.uci()}]")
                continue
            if replay.turn == chess.WHITE:
                parts.append(f"{replay.fullmove_number}.")
            elif i == 0:
                parts.append(f"{replay.fullmove_number}...")
            parts.append(replay.san(move))
            replay.push(move)
        return " ".join(parts)
