"""Reconcile per-frame cell readings into a legal python-chess game.

The core idea from DESIGN.md section 5.3: detect moves, not positions. Given the current
legal position, a frame is explained by the legal move whose touched squares match what the
cells now show, or, when moves were missed, by a short path of moves. Everything else is
held: a piece lifted mid-drag empties one square and matches no move, a promotion dialog
hides cells, an arrow makes a cell unreadable.

Cell readings can be exact (a symbol), occupied (a piece whose type is hidden by a badge),
empty, or unknown; unknown cells carry no evidence either way. A move is committed only after
it has explained `settle_frames` consecutive frames in which every square it touches holds a
settled piece or is empty and no square carries the hover outline of a drag. Highlight
colours are never consulted. That is what stops a dragged piece drawn over a neighbouring
square from being taken as a move.

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
from dataclasses import dataclass, field, replace

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


def _turn_valid(board: chess.Board, turn: chess.Color) -> bool:
    board.turn = turn
    return board.is_valid()


Readings = list[list[CellReading]]
Observation = dict[chess.Square, CellReading]
Explanation = tuple[chess.Move, ...]
PendingKey = tuple[object, ...]
# Pending keys whose last element is a story of moves from the latest position.
STORY_KEYS = ("moves", "moves-cautious")


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


def is_junk(reading: CellReading) -> bool:
    """A reading a badge or a ring leaves: an unsettled piece, or a piece of no known kind."""
    return reading.kind == "piece" and (not reading.settled or reading.symbol is None)


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
    viewing: int | None = None
    _hover_seen: bool = False
    _pending: tuple[PendingKey, int] | None = None
    _stale: dict[chess.Square, tuple[tuple[object, ...], int]] = field(default_factory=dict)
    _unexplained_frames: int = 0
    _started: bool = False
    _finished: bool = False
    # The result the game ended with; "*" when the end was seen but not who won.
    _result: str | None = None
    # Position resync: adopt a fully readable, stable board that no move story explains.
    resync_hold_frames: int = 10
    max_stale_subset_cells: int = 4
    # A pending story survives frames a drag blocks, up to this many, and may be extended
    # by the moves that follow it, up to this many plies in all.
    pending_max_age: int = 40
    max_chain_plies: int = 8
    max_queue_steps: int = 6
    _pending_age: int = 0
    resync_max_soft: int = 2
    _base_fen: str = chess.STARTING_FEN
    _turn_unknown: bool = False
    _resync_candidate: tuple[str, int] | None = None
    # Last stable reading of every cell and the frame it was seen in, plus the key of the
    # previous frame's reading, which a reading must repeat to count as stable.
    memory_frames: int = 30
    # The board unseen for this many ticks (lost, covered, minimised: the pipeline reports
    # each such tick) means anything remembered about covered cells may have changed.
    memory_gap_ticks: int = 5
    _unseen: int = 0
    # Fewer readable cells than this is no evidence of anything: a theme the recogniser
    # was not built for, or a page-wide overlay.
    min_readable_cells: int = 32
    _last_seen: dict[chess.Square, tuple[CellReading, int]] = field(default_factory=dict)
    _previous: dict[chess.Square, tuple[object, ...]] = field(default_factory=dict)
    _frame_index: int = 0
    _blocked: bool = False
    _ts: float = 0.0
    _last_commit_ts: float | None = None

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

    @property
    def in_progress(self) -> bool:
        """Whether a game is under way: moves were committed, or a position was adopted."""
        return bool(self.board.move_stack) or self._base_fen != chess.STARTING_FEN

    # Observation helpers

    def _observe(self, readings: Readings) -> Observation:
        """Readings as evidence, with covered cells remembered for a short while.

        The game-over dialog fades in about a second after the mating move and hides the
        squares it came from. Nothing changes under a dialog, so a cell a confirmed overlay
        has covered keeps its last stable reading for `memory_frames`; the explanation
        found while the board was still uncovered then keeps settling under the dialog
        instead of turning ambiguous. Only an exact, settled reading repeated in two
        consecutive frames is remembered: the fading dialog and a piece in flight both
        produce readings that change from frame to frame, and neither may be kept. A
        cell seen clearly with something else on it forgets its memory at once, so that
        a cover lifting for one frame cannot bring an old reading back under it.
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
                remembered = self._last_seen.get(sq)
                if (
                    remembered
                    and (remembered[0].kind, remembered[0].symbol, remembered[0].colour) != key
                ):
                    del self._last_seen[sq]
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
        self._result = None
        self.viewing = None
        self._base_fen = chess.STARTING_FEN
        self._turn_unknown = False
        self._resync_candidate = None
        self._last_seen = {}
        self._previous = {}
        self._pending_age = 0
        self._last_commit_ts = self._ts
        return [
            GameStarted(game_id=self.game_id, user_side=self.user_side, start_fen=self.board.fen()),
            PositionChanged(game_id=self.game_id, fen=self.board.fen(), ply=0),
        ]

    def reorient(self, bottom: Side) -> list[Event]:
        """The board was flipped or re-themed before the first move: the same game, seen anew.

        Everything remembered per square is dropped, since the squares were mapped with the
        old orientation; a game already announced is announced again with the user's side.
        """
        self.bottom = bottom
        self.user_side = bottom
        self._pending = None
        self._pending_age = 0
        self._stale = {}
        self._last_seen = {}
        self._previous = {}
        self.viewing = None
        self._unexplained_frames = 0
        if not self._started:
            return []
        return [
            GameStarted(game_id=self.game_id, user_side=self.user_side, start_fen=self.board.fen())
        ]

    # Explanations

    def _path_explanations(
        self,
        obs: Observation,
        diff: set[chess.Square],
        base: chess.Board | None = None,
        wild: frozenset[chess.Square] = frozenset(),
    ) -> list[Explanation]:
        """Sequences of up to `max_plies` legal moves explaining `diff` from `base`.

        `wild` cells are stale junk a badge left, ignored by the caller: a move may touch
        them instead of an unexplained cell, and they need no exact reading, so that the
        mating move still settles when the checkmate badge has made the king's square
        unreadable.

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
        if len(diff) + len(wild) < 3 or len(diff) > 4 * self.max_plies:
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
                    if not (touched & (remaining | wild)):
                        continue
                    if not all(sq in wild or (sq in obs and exact(obs[sq])) for sq in touched):
                        continue
                    after = board.copy(stack=False)
                    after.push(move)
                    path.append(move)
                    still = remaining - touched
                    if not still:
                        if matches(after, obs, set(obs) - wild):
                            solutions.append(tuple(path))
                    else:
                        search(after, still, path)
                    path.pop()

            start = base if base is not None else self.board
            search(start, set(diff), [])
            if not solutions:
                continue
            finals: set[str] = set()
            for path in solutions:
                board = start.copy(stack=False)
                for move in path:
                    board.push(move)
                finals.add(board.board_fen() + " " + ("w" if board.turn else "b"))
            if len(finals) > 1:
                return solutions
            return [solutions[0]]
        return []

    def _history_index(self, obs: Observation, diff: set[chess.Square]) -> int | None:
        """Index of the earlier position of this game the frame shows, if any.

        The last `history_depth` positions are checked one by one. The base position is
        checked as well however long the game is: a click on the first move of the list
        shows it, and it must never pass for a new game while this one is running.
        """
        earlier = self.board.copy()
        undone: set[chess.Square] = set()
        plies = len(self.board.move_stack)
        for back in range(1, min(plies, self.history_depth) + 1):
            move = earlier.pop()
            undone |= touched_squares(earlier, move)
            if diff <= undone and matches(earlier, obs, set(obs)):
                return plies - back
        if plies > self.history_depth:
            base = chess.Board(self._base_fen)
            if matches(base, obs, set(obs)):
                return 0
        return None

    def _board_at(self, index: int) -> chess.Board:
        earlier = self.board.copy()
        while len(earlier.move_stack) > index:
            earlier.pop()
        return earlier

    def _highlight_ok(
        self,
        obs: Observation,
        moves: Explanation,
        base: chess.Board | None = None,
        wild: frozenset[chess.Square] = frozenset(),
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
            if sq in wild:
                continue
            if reading is not None and reading.kind == "piece" and not reading.settled:
                return False
        return True

    # Main entry

    def board_unseen(self) -> None:
        """A tick on which the board could not be read at all; see `memory_gap_ticks`."""
        self._unseen += 1

    def update(self, readings: Readings, ts: float) -> list[Event]:
        self._ts = ts
        events = self.start() if not self._started else []
        if self._unseen >= self.memory_gap_ticks:
            self._last_seen = {}
            self._previous = {}
        self._unseen = 0
        # A hover outline anywhere means a drag is in progress, even on a cell whose content
        # could not be read; it must be seen before unknown cells are dropped.
        self._hover_seen = any(reading.hover for row in readings for reading in row)
        # A drag in progress blocks every explanation without contradicting any. A story
        # seen settled just before the drag stays pending through it, so that the frames
        # after the drop can confirm or extend it; a quiet frame that fits nothing drops it.
        self._blocked = self._hover_seen or any(
            reading.kind == "piece" and not reading.settled for row in readings for reading in row
        )
        obs = self._unqueue(self._observe(readings))
        if len(obs) < self.min_readable_cells:
            self._hold_pending(True)
            return events
        diff = self.disagreements(obs)
        self._advance_stale(diff, obs)
        ghosts = self._ghosts(diff, obs)
        if ghosts:
            # An arrow drawn over an empty square reads as a piece of no known kind, off
            # centre, for as long as it is drawn. Once such a reading has gone stale it is
            # taken as unknown, so a few arrows neither stall the game nor multiply the
            # stale retries below.
            obs = {sq: r for sq, r in obs.items() if sq not in ghosts}
            diff = diff - ghosts

        if not diff:
            # The frame shows the latest position: live again if the user was browsing.
            self._pending = None
            self.viewing = None
            self._unexplained_frames = 0
            return events

        blocked = self._blocked
        # Shortest story first: one move, then the pending story confirmed or extended,
        # then a premove queue, then the opponent's move on its own, and only then a path
        # of missed moves, which could otherwise explain a queue display or a drag with a
        # longer story than what happened.
        explained = self._explain(obs, diff, obs, allow_paths=False)
        if explained is not None:
            return events + explained
        extended = self._extend_pending(obs, diff)
        if extended is not None:
            return events + extended
        queued = self._explain_premove_queue(obs, diff)
        if queued is not None:
            return events + queued
        replied = self._explain_opponent_move(obs)
        if replied is not None:
            return events + replied
        if self._hover_seen:
            # A hover outline refuses every story below; the searches would only burn time.
            self._hold_pending(blocked)
            return events
        explained = self._explain_paths(obs, diff, obs)
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
            # Cells a badge has spoiled without contradicting the position (an occupied
            # square of no known kind, or an unsettled piece) cannot be ignored as stale,
            # yet they can be neither read exactly nor trusted as settled; in this retry
            # they are wildcards alongside the ignored cells.
            soft = frozenset(
                sq
                for sq, reading in obs.items()
                if reading.kind == "piece" and (reading.symbol is None or not reading.settled)
            )
            subsets = self._stale_subsets(stale, obs)
            # Soft wildcards only go with junk cells: ignoring a clean reading as well
            # would let the search invent a story through the cells nobody can read.
            attempts = [(sub, sub | soft) for sub in subsets if all(is_junk(obs[sq]) for sq in sub)]
            attempts += [(sub, frozenset(sub)) for sub in subsets]
            tried: set[tuple[frozenset[chess.Square], frozenset[chess.Square]]] = set()
            for ignored, wild in attempts:
                if (ignored, wild) in tried:
                    continue
                tried.add((ignored, wild))
                fresh = diff - ignored
                if not fresh:
                    continue
                live_obs = {sq: r for sq, r in obs.items() if sq not in ignored}
                explained = self._explain(
                    live_obs, fresh, obs, allow_paths=True, cautious=True, wild=wild
                )
                if explained is not None:
                    return events + explained
            if stale == diff and len(stale) < self.resync_cells:
                # Only stale garbage differs from the latest position, unless a drag is
                # what kept a real move from committing while its cells turned stale.
                self._hold_pending(blocked)
                self.viewing = None
                self._unexplained_frames += 1
                return events

        self._hold_pending(blocked)

        # New game: the start position is back on the board.
        if self.in_progress and len(obs) >= 48 and matches(chess.Board(), obs, set(obs)):
            if not self._finished:
                events.append(self._ended("*", "new game started"))
            events.extend(self.start())
            return events

        adopted = self._maybe_position_resync(obs, diff)
        if adopted:
            return events + adopted
        return events + self._maybe_resync(diff)

    def _unqueue(self, obs: Observation) -> Observation:
        """The picture with queued premoves undone: each of the user's pieces drawn at the
        end of a red queue is put back on the red, empty square it will leave from.

        Chess.com draws a premoved piece at its destination and tints the queue's squares
        red, so the screen is not a position. The one-piece case is explained as a queue
        below; but two pieces can be queued at once (the third live game ended with both
        queens premoved), and a queue can form right after a real move whose settled frame
        the capture never caught. Undoing the queue gives the true position, which the
        ordinary explanations then read -- the real move that slipped by, the opponent's
        reply, the queued step the moment it executes and its squares turn from red to the
        site's last-move colour. A piece on a red square that the position already has
        there is not a ghost (the user's own king in check, on chess.com's red square).
        The queue's end goes unread rather than empty: a queued capture is drawn over the
        piece it will take. Nothing is undone when the ghosts and origins do not pair off
        one to one.
        """
        red_empty = [sq for sq, r in obs.items() if r.tint == "red" and r.kind == "empty"]
        if not red_empty:
            return obs
        ghosts: list[tuple[chess.Square, CellReading]] = []
        for sq, reading in obs.items():
            if reading.tint != "red" or reading.kind != "piece" or reading.symbol is None:
                continue
            if reading.colour != self.user_side:
                continue
            here = self.board.piece_at(sq)
            if here is not None and here.symbol() == reading.symbol:
                continue  # really there; the square is red for another reason
            ghosts.append((sq, reading))
        if not ghosts:
            return obs
        origins = frozenset(red_empty)
        pairs: list[tuple[chess.Square, chess.Square, CellReading]] = []
        taken: set[chess.Square] = set()
        for target, reading in ghosts:
            symbol = reading.symbol
            assert symbol is not None
            candidates = [
                o for o in origins - taken if self._queue_reaches(symbol, o, target, origins, obs)
            ]
            if len(candidates) != 1:
                # Prefer the origin the position has this very piece on.
                candidates = [
                    o
                    for o in candidates
                    if (piece := self.board.piece_at(o)) is not None and piece.symbol() == symbol
                ]
                if len(candidates) != 1:
                    return obs
            origin = candidates[0]
            taken.add(origin)
            pairs.append((origin, target, reading))
        out = dict(obs)
        for origin, target, reading in pairs:
            out[origin] = replace(reading, highlighted=False, tint=None, settled=True)
            out.pop(target, None)
        for sq in origins - taken:
            out.pop(sq, None)  # an intermediate square of a longer queue: unread
        return out

    def _queue_reaches(
        self,
        symbol: str,
        origin: chess.Square,
        target: chess.Square,
        via: frozenset[chess.Square],
        obs: Observation,
    ) -> bool:
        """Whether a piece `symbol` could be queued from `origin` to `target`, hopping only
        over the red squares `via`, past the pieces the picture shows, within the queue
        length chess.com allows."""
        piece = chess.Piece.from_symbol(symbol)
        scratch = chess.Board(None)
        for sq, reading in obs.items():
            if reading.kind == "piece" and reading.symbol is not None and reading.tint != "red":
                scratch.set_piece_at(sq, chess.Piece.from_symbol(reading.symbol))
        scratch.turn = piece.color
        frontier = {origin}
        seen = {origin}
        for _ in range(self.max_queue_steps):
            next_frontier: set[chess.Square] = set()
            for sq in frontier:
                scratch.set_piece_at(sq, piece)
                for move in scratch.generate_pseudo_legal_moves(chess.BB_SQUARES[sq]):
                    to = move.to_square
                    if to == target:
                        return True
                    if to in via and to not in seen:
                        seen.add(to)
                        next_frontier.add(to)
                scratch.remove_piece_at(sq)
            frontier = next_frontier
            if not frontier:
                break
        return False

    def pending_terminal(self) -> tuple[Explanation, chess.Board] | None:
        """The story seen once and still pending, when it ends the game: the mate the
        site's dialog then covered before a second clean frame could confirm it."""
        story = self._pending_story()
        if story is None or not story[1].is_game_over():
            return None
        return story

    def commit_pending_terminal(self) -> list[Event]:
        """Commit the pending story that ends the game (see `pending_terminal`)."""
        story = self.pending_terminal()
        if story is None:
            return []
        moves, _ = story
        events = self._commit_moves(moves)()
        self._pending = None
        self._stale = {}
        self._unexplained_frames = 0
        return events

    def _ghosts(self, diff: set[chess.Square], obs: Observation) -> set[chess.Square]:
        """Stale readings of something unnameable on squares the position has empty."""
        return {
            sq
            for sq in diff
            if self._stale[sq][1] > self.stale_after
            and obs[sq].kind == "piece"
            and obs[sq].symbol is None
            and self.board.piece_at(sq) is None
        }

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

        adopted = self._adopt_placement(self._placement(obs))
        if adopted is None:
            self._resync_candidate = None
            return []
        board, self._turn_unknown = adopted
        events: list[Event] = []
        if self._finished:
            # The finished game is over for good; a position that appears after it is
            # another game, seen from the middle.
            self.game_id = uuid.uuid4().hex[:8]
            events.append(
                GameStarted(game_id=self.game_id, user_side=self.user_side, start_fen=board.fen())
            )
        self.board = board
        self._base_fen = board.fen()
        self._finished = False
        self._result = None
        self._last_commit_ts = self._ts
        self._pending = None
        self._stale = {}
        self.viewing = None
        self._unexplained_frames = 0
        self._resync_candidate = None
        reason = f"position adopted after {len(diff)} unexplained cells"
        if self._turn_unknown:
            reason += "; side to move pending"
        events += [
            Resync(game_id=self.game_id, fen=board.fen(), reason=reason),
            PositionChanged(game_id=self.game_id, fen=board.fen(), ply=0),
        ]
        if not self._turn_unknown and board.is_game_over():
            self._finished = True
            events.append(self._ended(board.result(), "game over on board"))
        return events

    @staticmethod
    def _adopt_placement(placement: chess.Board) -> tuple[chess.Board, bool] | None:
        """A bare placement made into a position: castling rights assumed conservatively from
        the king and rook home squares, and the turn from legality. Returns the board and
        whether the turn is still unknown (both colours legal to move), or None if neither is.
        Shared by the position resync and the cold start from the theme cache.
        """
        board = placement.copy(stack=False)
        board.castling_rights = chess.BB_EMPTY
        for colour, rank in ((chess.WHITE, 0), (chess.BLACK, 7)):
            king = chess.Piece(chess.KING, colour)
            rook = chess.Piece(chess.ROOK, colour)
            if board.piece_at(chess.square(4, rank)) == king:
                if board.piece_at(chess.square(7, rank)) == rook:
                    board.castling_rights |= chess.BB_SQUARES[chess.square(7, rank)]
                if board.piece_at(chess.square(0, rank)) == rook:
                    board.castling_rights |= chess.BB_SQUARES[chess.square(0, rank)]
        valid_turns = [turn for turn in (chess.WHITE, chess.BLACK) if _turn_valid(board, turn)]
        if not valid_turns:
            return None
        board.turn = valid_turns[0]
        return board, len(valid_turns) > 1

    def start_from(self, placement: chess.Board) -> list[Event]:
        """Begin a game from an arbitrary observed position -- a puzzle or a mid-game board read
        cold from the theme cache -- rather than from the start. Rights and turn are assumed the
        same way a position resync assumes them; the game continues normally from here."""
        adopted = self._adopt_placement(placement)
        if adopted is None:
            return []
        board, self._turn_unknown = adopted
        self.board = board
        self._base_fen = board.fen()
        self._started = True
        self._finished = False
        self._result = None
        self._last_commit_ts = self._ts
        self.game_id = uuid.uuid4().hex[:8]
        return [GameStarted(game_id=self.game_id, user_side=self.user_side, start_fen=board.fen())]

    def _placement(self, obs: Observation) -> chess.Board:
        board = chess.Board(None)
        for sq, reading in obs.items():
            symbol = reading.symbol or reading.guess
            if reading.kind == "piece" and symbol is not None:
                board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
        return board

    def _explain_turn_unknown(
        self,
        obs: Observation,
        diff: set[chess.Square],
        full_obs: Observation,
        *,
        paths: bool,
        cautious: bool = False,
        wild: frozenset[chess.Square] = frozenset(),
    ) -> list[Event] | None:
        """After a resync that left the side to move open, the first story decides it.

        Stories are searched for both sides; a unique one settles like any other and
        fixes the turn only when it commits, so a drag or a hypothesis seen once can never
        flip the turn on its own.
        """
        found: list[tuple[chess.Color, Explanation]] = []
        for turn in (chess.WHITE, chess.BLACK):
            trial = self.board.copy(stack=False)
            trial.turn = turn
            stories = (
                self._path_explanations(obs, diff, trial, wild)
                if paths
                else single_explanations(trial, obs, diff)
            )
            found += [(turn, story) for story in stories]
        if len(found) != 1:
            return None
        turn, moves = found[0]
        trial = self.board.copy(stack=False)
        trial.turn = turn
        if not self._highlight_ok(full_obs, moves, trial, wild=wild):
            return None
        key = ("turn-cautious" if cautious else "turn", turn, moves)
        return self._settle(key, full_obs, self._commit_with_turn(turn, moves))

    def _explain(
        self,
        obs: Observation,
        diff: set[chess.Square],
        full_obs: Observation,
        *,
        allow_paths: bool,
        cautious: bool = False,
        wild: frozenset[chess.Square] = frozenset(),
    ) -> list[Event] | None:
        """Events for one explanation of `diff`, or None when nothing fits.

        Order: one move from the latest position; an earlier position of this game (the user
        is browsing); a different move, or a short path, from the viewed position (a
        takeback); a short path of missed moves. `full_obs` is the whole frame, for the
        settled and hover checks, while `obs` may leave ignored cells out. `cautious` marks
        the retry that ignores stale cells; its paths need an extra frame to settle.

        After the game has ended the board is a review board: browsing is followed, and
        moves played on it are analysis, not the game. Only the start position begins the
        next game, or a position far enough away to be adopted by a resync.
        """
        if self._turn_unknown:
            explained = self._explain_turn_unknown(obs, diff, full_obs, paths=False)
            if explained is None and allow_paths:
                explained = self._explain_turn_unknown(
                    obs, diff, full_obs, paths=True, cautious=cautious, wild=wild
                )
            return explained

        if not self._finished:
            singles = single_explanations(self.board, obs, diff)
            if len(singles) == 1 and self._highlight_ok(full_obs, singles[0], wild=wild):
                return self._settle(("moves", singles[0]), full_obs, self._commit_moves(singles[0]))

        index = self._history_index(obs, diff)
        if index is not None:
            if index == 0 and self._finished and self._base_fen == chess.STARTING_FEN:
                # The start position after a finished game is the next game, not browsing.
                return self.start()
            if self._blocked:
                # A piece dragged back over its own square shows an earlier position for a
                # moment; browsing is only ever entered from a quiet frame.
                return None
            self.viewing = index
            self._pending = None
            self._stale = {}
            self._unexplained_frames = 0
            return []

        if self._finished:
            return None

        if self.viewing is not None:
            earlier = self._board_at(self.viewing)
            diff_earlier = {sq for sq, r in obs.items() if not cell_agrees(r, earlier.piece_at(sq))}
            branches = single_explanations(earlier, obs, diff_earlier)
            if len(branches) != 1:
                # The user played on from the viewed position and the opponent replied
                # before the board was quiet.
                branches = self._path_explanations(obs, diff_earlier, earlier)
            if len(branches) == 1 and self._highlight_ok(full_obs, branches[0], earlier):
                story = branches[0]
                if allow_paths:
                    # The same frame told from the latest position: with fewer plies it
                    # is the live game moving on while the user browsed; with more it
                    # retells the takeback as retreats, which nobody played.
                    latest = self._path_explanations(obs, diff, wild=wild)
                    if (
                        len(latest) == 1
                        and len(latest[0]) < len(story)
                        and self._highlight_ok(full_obs, latest[0], wild=wild)
                    ):
                        return self._explain_paths(
                            obs, diff, full_obs, cautious=cautious, wild=wild
                        )
                if self.viewing == 0 and self._base_fen == chess.STARTING_FEN:
                    # A move from the start position while the game had moves: a new game
                    # began (resigned, aborted, or rematch), not a takeback of everything.
                    # After a position resync, position 0 is the adopted position, and a
                    # move from it is an ordinary takeback.
                    return self._settle(("newgame", story), full_obs, self._commit_new_game(story))
                plies = len(self.board.move_stack) - self.viewing
                return self._settle(
                    ("takeback", self.viewing, story),
                    full_obs,
                    self._commit_takeback(plies, story),
                )

        if allow_paths:
            return self._explain_paths(obs, diff, full_obs, cautious=cautious, wild=wild)
        return None

    def _explain_paths(
        self,
        obs: Observation,
        diff: set[chess.Square],
        full_obs: Observation,
        *,
        cautious: bool = False,
        wild: frozenset[chess.Square] = frozenset(),
    ) -> list[Event] | None:
        """A short path of missed moves explaining `diff`; see `_path_explanations`."""
        if self._finished:
            return None
        if self._turn_unknown:
            return self._explain_turn_unknown(
                obs, diff, full_obs, paths=True, cautious=cautious, wild=wild
            )
        paths = self._path_explanations(obs, diff, wild=wild)
        if len(paths) == 1 and self._highlight_ok(full_obs, paths[0], wild=wild):
            key = ("moves-cautious" if cautious else "moves", paths[0])
            return self._settle(key, full_obs, self._commit_moves(paths[0]))
        return None

    @property
    def user_colour(self) -> chess.Color:
        return chess.WHITE if self.user_side == Side.WHITE else chess.BLACK

    def _pending_story(self) -> tuple[Explanation, chess.Board] | None:
        """The pending story from the latest position and the position it leads to."""
        if self._pending is None or self._pending[0][0] not in STORY_KEYS:
            return None
        moves = self._pending[0][-1]
        assert isinstance(moves, tuple)
        base = self.board.copy(stack=False)
        for move in moves:
            if move not in base.legal_moves:
                return None
            base.push(move)
        return moves, base

    @staticmethod
    def _cells_clean(obs: Observation, squares: set[chess.Square]) -> bool:
        """Every square is read, holds a settled piece or nothing, and is not under the cursor."""
        for sq in squares:
            reading = obs.get(sq)
            if reading is None or reading.hover:
                return False
            if reading.kind == "piece" and not reading.settled:
                return False
        return True

    def _hold_pending(self, blocked: bool) -> None:
        """Keep the pending story through a blocked frame, drop it on a quiet one or when old."""
        self._pending_age += 1
        if not blocked or self._pending_age > self.pending_max_age:
            self._pending = None

    def _extend_pending(self, obs: Observation, diff: set[chess.Square]) -> list[Event] | None:
        """Explain the frame from the position the pending story leads to.

        With premoves, two plies land in one frame and the next drag starts a fraction of
        a second later, before the story has settled; by the time the board is quiet again
        it is several plies further on, and a piece may have returned to a square it left,
        which the missed-move search cannot express from the committed position. The
        story seen just before the drag is a hypothesis; if the moves that follow it
        explain the frame, the whole chain is what settles. An intermediate frame also
        disambiguates transpositions the search alone would have to guess.
        """
        if self.viewing is not None:
            return None
        story = self._pending_story()
        if story is None or self._pending is None:
            return None
        moves, base = story
        diff_base = {sq for sq, r in obs.items() if not cell_agrees(r, base.piece_at(sq))}
        if not diff_base:
            # The frame shows the position the story leads to: one more frame of evidence.
            if not self._highlight_ok(obs, moves):
                return None
            return self._settle(self._pending[0], obs, self._commit_moves(moves))
        singles = single_explanations(base, obs, diff_base)
        if len(singles) == 1:
            story = singles[0]
        else:
            paths = self._path_explanations(obs, diff_base, base)
            if len(paths) != 1:
                return None
            story = paths[0]
        combined = tuple(moves) + tuple(story)
        if len(combined) > self.max_chain_plies or not self._highlight_ok(obs, combined):
            return None
        return self._settle(("moves", combined), obs, self._commit_moves(combined))

    def _explain_opponent_move(self, obs: Observation) -> list[Event] | None:
        """The opponent's move, committed whether or not the user is dragging.

        A drag blocks the user's own moves, since a dragged piece is not a move, but it
        cannot move the opponent's pieces. So the opponent's move commits from the cells
        it touches alone, as long as they are settled and not under the cursor. It is read
        from the position the pending story leads to when there is one: the story seen
        just before the drag is then extended, not contradicted. Without this, a bot that
        replies while the user is already dragging the next premove is seen only as a net
        displacement once the board is quiet, and a queue of premoves advances one step
        for every two or three the site executed.
        """
        if self.viewing is not None or self._turn_unknown:
            return None
        moves: Explanation = ()
        base = self.board.copy(stack=False)
        story = self._pending_story()
        if story is not None:
            moves, base = story
        elif self._pending is not None and self._pending[0][0] in STORY_KEYS:
            return None  # a pending story that is no longer legal explains nothing
        user_colour = self.user_colour
        bot_diff: set[chess.Square] = set()
        for sq, reading in obs.items():
            if cell_agrees(reading, base.piece_at(sq)):
                continue
            piece = base.piece_at(sq)
            if (piece is not None and piece.color != user_colour) or (
                reading.kind == "piece" and reading.colour not in (None, self.user_side)
            ):
                bot_diff.add(sq)
        if base.turn == user_colour:
            # The pending story ends with the opponent's move. The user's part of it was
            # seen settled before the drag; the opponent's part is confirmed by its own
            # cells, which the drag cannot touch.
            if bot_diff or not moves or self._pending is None:
                return None
            before = self.board.copy(stack=False)
            for move in moves[:-1]:
                before.push(move)
            if not self._cells_clean(obs, touched_squares(before, moves[-1])):
                return None
            return self._settle(self._pending[0], obs, self._commit_moves(moves))
        if not bot_diff:
            return None
        replies = single_explanations(base, obs, bot_diff)
        if len(replies) != 1:
            return None
        reply = replies[0][0]
        if not self._cells_clean(obs, touched_squares(base, reply)):
            return None
        combined = moves + (reply,)
        if len(combined) > self.max_chain_plies:
            return None
        return self._settle(("moves", combined), obs, self._commit_moves(combined))

    def _explain_premove_queue(
        self, obs: Observation, diff: set[chess.Square]
    ) -> list[Event] | None:
        """Frames during a premove queue, which chess.com does not draw as a position.

        While the opponent is to move, the user may queue several moves of one piece. The
        site then draws that piece at the end of the queue and leaves its real square
        empty, tints the queued squares, and executes the first queued move the moment the
        opponent moves, without changing the picture; a queue the opponent's move makes
        illegal is cleared and the piece snaps back. So during a queue the frame is the
        true position with exactly one of the user's pieces displaced from its square to
        the queue's end, and the queue itself is the unique shortest way for that piece to
        get there. Each visible opponent move then commits together with the next queued
        step, while the picture stays the same.
        """
        if self.viewing is not None or self._turn_unknown or self._hover_seen:
            return None
        user_colour = self.user_colour
        shown: list[chess.Square] = []
        gone: list[chess.Square] = []
        for sq in diff:
            reading = obs[sq]
            piece = self.board.piece_at(sq)
            users = piece is not None and piece.color == user_colour
            if (
                reading.kind == "piece"
                and reading.symbol is not None
                and reading.settled
                and reading.colour == self.user_side
                and not users
            ):
                shown.append(sq)
            elif reading.kind == "empty" and users:
                gone.append(sq)
        if len(shown) != 1 or len(gone) != 1:
            return None
        target, origin = shown[0], gone[0]
        symbol = obs[target].symbol
        assert symbol is not None
        # Chess.com tints every square of the queue red; the route the user actually
        # queued runs through exactly those squares, which tells a4 from a3 then a4.
        tinted = frozenset(
            sq
            for sq, reading in obs.items()
            if reading.tint == "red" and (reading.kind == "empty" or sq in (origin, target))
        )
        path = self._queue_path(origin, target, symbol, tinted)
        if path is None:
            return None
        rest = diff - {origin, target}
        rest_obs = {sq: r for sq, r in obs.items() if sq not in (origin, target)}
        if self.board.turn == user_colour:
            # The opponent's move is already committed, so the queue's first step has
            # executed; only the displaced piece may differ from the position.
            if rest or path[0] not in self.board.legal_moves:
                return None
            if not self._highlight_ok(obs, (path[0],)):
                return None
            return self._settle(("moves", (path[0],)), obs, self._commit_moves((path[0],)))
        if not rest:
            # The queue formed or changed; nothing has happened on the board yet.
            self._unexplained_frames = 0
            return []
        replies = single_explanations(self.board, rest_obs, rest)
        if len(replies) != 1:
            return None
        reply = replies[0][0]
        after = self.board.copy(stack=False)
        after.push(reply)
        if path[0] not in after.legal_moves:
            return None
        story = (reply, path[0])
        if not self._highlight_ok(obs, story):
            return None
        return self._settle(("moves", story), obs, self._commit_moves(story))

    def _queue_path(
        self,
        origin: chess.Square,
        target: chess.Square,
        symbol: str,
        tinted: frozenset[chess.Square] = frozenset(),
    ) -> tuple[chess.Move, ...] | None:
        """The sequence of moves taking the user's piece from origin to target.

        When the site tints the queued squares (`tinted` holds them, the target included;
        the origin only until its first step has executed), the route through exactly
        those squares is the queue, however long.
        Otherwise the unique shortest way is assumed. Squares on the way must be empty; only
        the last step may capture. A pawn reaching the last rank promotes to the piece
        drawn there, and a pawn still drawn there means the choice is not made yet, so
        nothing is known. Several equally short ways, as a king has across an open board,
        are an ambiguity and nothing is assumed.
        """
        piece = self.board.piece_at(origin)
        if piece is None or chess.Piece.from_symbol(symbol).color != piece.color:
            return None
        promotion: int | None = None
        if piece.piece_type != chess.PAWN and symbol.upper() != piece.symbol().upper():
            return None
        if piece.piece_type == chess.PAWN and symbol.upper() != "P":
            if chess.square_rank(target) not in (0, 7):
                return None
            promotion = chess.Piece.from_symbol(symbol).piece_type
        elif piece.piece_type == chess.PAWN and chess.square_rank(target) in (0, 7):
            return None
        scratch = self.board.copy(stack=False)
        scratch.turn = piece.color
        scratch.ep_square = None  # a queue never passes through the en-passant square
        scratch.remove_piece_at(origin)
        if target in tinted:
            route = self._route_through(scratch, piece, origin, target, tinted, promotion)
            if route is not None:
                return route
        # Breadth-first over the piece's own moves, counting the shortest ways to each square.
        frontier = [origin]
        ways: dict[chess.Square, int] = {origin: 1}
        parent: dict[chess.Square, chess.Move] = {}
        depth = 0
        while frontier and target not in ways and depth < self.max_queue_steps:
            depth += 1
            next_frontier: list[chess.Square] = []
            for sq in frontier:
                scratch.set_piece_at(sq, piece)
                for move in scratch.generate_pseudo_legal_moves(chess.BB_SQUARES[sq]):
                    to = move.to_square
                    if to != target and scratch.piece_at(to) is not None:
                        continue
                    if to == target and move.promotion not in (None, promotion):
                        continue
                    if to in ways and to not in next_frontier:
                        continue
                    if to not in ways:
                        ways[to] = 0
                        next_frontier.append(to)
                        parent[to] = move
                    ways[to] += ways[sq]
                scratch.remove_piece_at(sq)
            frontier = next_frontier
        if ways.get(target, 0) != 1:
            return None
        path: list[chess.Move] = []
        sq = target
        while sq != origin:
            move = parent[sq]
            path.append(move)
            sq = move.from_square
        path.reverse()
        return tuple(path)

    def _route_through(
        self,
        scratch: chess.Board,
        piece: chess.Piece,
        origin: chess.Square,
        target: chess.Square,
        tinted: frozenset[chess.Square],
        promotion: int | None,
    ) -> tuple[chess.Move, ...] | None:
        """The unique route from origin to target visiting every tinted square once."""
        stops = tinted - {origin}
        routes: list[tuple[chess.Move, ...]] = []

        def walk(
            sq: chess.Square, visited: frozenset[chess.Square], path: list[chess.Move]
        ) -> None:
            if len(routes) > 1 or len(path) > self.max_queue_steps:
                return
            if sq == target:
                if visited == stops:
                    routes.append(tuple(path))
                return
            scratch.set_piece_at(sq, piece)
            for move in scratch.generate_pseudo_legal_moves(chess.BB_SQUARES[sq]):
                to = move.to_square
                if to not in stops or to in visited:
                    continue
                if to != target and (scratch.piece_at(to) is not None or move.promotion):
                    continue
                if to == target and move.promotion not in (None, promotion):
                    continue
                path.append(move)
                walk(to, visited | {to}, path)
                path.pop()
            scratch.remove_piece_at(sq)

        walk(origin, frozenset(), [])
        return routes[0] if len(routes) == 1 else None

    # Settling and committing

    def _settle(
        self, key: PendingKey, obs: Observation, commit: Callable[[], list[Event]]
    ) -> list[Event]:
        """Count frames an explanation has held; commit it once it has settled."""
        if self._pending and self._pending[0] == key:
            self._pending = (key, self._pending[1] + 1)
        else:
            self._pending = (key, 1)
        self._pending_age = 0
        moves = key[-1]
        assert isinstance(moves, tuple)
        needed = (
            self.settle_frames
            + (1 if len(moves) > 1 else 0)
            + (1 if key[0] == "moves-cautious" else 0)
        )
        if self._pending[1] < needed:
            return []
        events = commit()
        self._pending = None
        self._stale = {}
        self._unexplained_frames = 0
        self.viewing = None
        return events

    def _commit_moves(self, moves: Explanation) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            events: list[Event] = []
            for move in moves:
                events.extend(self._commit(move))
            return events

        return commit

    def _commit_with_turn(self, turn: chess.Color, moves: Explanation) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            self.board.turn = turn
            self._base_fen = self.board.fen()
            self._turn_unknown = False
            return self._commit_moves(moves)()

        return commit

    def _commit_takeback(self, plies: int, moves: Explanation) -> Callable[[], list[Event]]:
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
                events.extend(self._commit(move))
            return events

        return commit

    def _commit_new_game(self, moves: Explanation) -> Callable[[], list[Event]]:
        def commit() -> list[Event]:
            events: list[Event] = []
            if not self._finished:
                events.append(self._ended("*", "new game started"))
            events.extend(self.start())
            for move in moves:
                events.extend(self._commit(move))
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

        junk = sorted(sq for sq in stale if is_junk(obs[sq]))
        clean = sorted(sq for sq in stale if not is_junk(obs[sq]))
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

    def _commit(self, move: chess.Move) -> list[Event]:
        if self._finished:
            # Defensive: nothing is played on a finished game. A new game starts it over.
            self._pending = None
            return []
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
        # Time since the previous commit, as seen on screen: a chain committed at once
        # gives it all to its first move and leaves the others open.
        think_ms = None
        if self._last_commit_ts is not None and self._ts > self._last_commit_ts:
            think_ms = int(round((self._ts - self._last_commit_ts) * 1000))
        self._last_commit_ts = self._ts
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
            self._result = self.board.result()
            events.append(self._ended(self._result, "game over on board"))
        return events

    def end(self, result: str, reason: str) -> list[Event]:
        """End the game: on the user's word (a resignation, a draw, an abort) or on the
        site's dialog. A game that ended with the result unknown ("*") can be ended again
        with the result the user supplies."""
        if not self._started or (self._finished and self._result != "*"):
            return []
        self._finished = True
        self._pending = None
        self._result = result
        return [self._ended(result, reason)]

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
