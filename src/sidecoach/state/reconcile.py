"""Reconcile per-frame cell readings into a legal python-chess game.

The core idea from DESIGN.md section 5.3: detect moves, not positions. Given the current
legal position, a frame is explained by the legal move whose touched squares match what the
cells now show, or, when moves were missed, by a short path of moves. Everything else is
held: a piece lifted mid-drag empties one square and matches no move, a promotion dialog
hides cells, an arrow makes a cell unreadable.

Cell readings can be exact (a symbol), occupied (a piece whose type is hidden by a badge),
empty, or unknown; unknown cells carry no evidence either way. A move is committed only after
it has explained `settle_frames` consecutive frames in which every square it touches holds a
settled piece or is empty, none of them is under something drawn (an arrow), and no square
carries the hover outline of a drag. Highlight colours are never consulted. That is what
stops a dragged piece drawn over a neighbouring square from being taken as a move, and a
move from committing through a square an arrow hides while the arrow, which is no drag,
holds nothing else back.

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

import itertools
import uuid
from collections.abc import Callable, Sequence
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
    Unexplained,
)
from sidecoach.vision.pieces import CellReading
from sidecoach.vision.squares import square_at


def result_on_board(board: chess.Board) -> tuple[str, str] | None:
    """The result a game's final position proves and how, in the words the review reads; None
    when it proves none (the game was resigned, flagged, agreed or left). A draw by repetition or
    by fifty moves counts only when it was played -- the position on the board for the third
    time, or the hundredth half-move without a capture or a pawn move -- as the sites draw it,
    never one the next move could reach (the library's `stat_facts.ended_by`, which reads an
    import's ending the same way)."""
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
    if board.is_stalemate():
        return "1/2-1/2", "stalemate"
    if board.is_insufficient_material():
        return "1/2-1/2", "insufficient material"
    if board.is_repetition(3):
        return "1/2-1/2", "repetition"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "fifty-move rule"
    return None


def _turn_valid(board: chess.Board, turn: chess.Color) -> bool:
    board.turn = turn
    return board.is_valid()


Readings = list[list[CellReading]]
Observation = dict[chess.Square, CellReading]


def highlighted_squares(obs: Observation) -> frozenset[chess.Square]:
    """The squares the site marks as its last move (or a selection); a red tint is never a
    highlight (`pieces._classify`)."""
    return frozenset(sq for sq, r in obs.items() if r.highlighted)


def by_square(readings: Readings, bottom: Side) -> Observation:
    """A frame's readings keyed by the squares they show, `bottom` at the bottom."""
    return {square_at(r, c, bottom): readings[r][c] for r in range(8) for c in range(8)}


def loose_placement(obs: Observation) -> chess.Board:
    """The pieces a read names, a held or a ringed one included (its kind, or its best
    guess), an unreadable cell taken as empty: what the board shows, not a position."""
    board = chess.Board(None)
    for sq, reading in obs.items():
        symbol = reading.symbol or reading.guess
        if reading.kind == "piece" and symbol is not None:
            board.set_piece_at(sq, chess.Piece.from_symbol(symbol))
    return board


def _came_from(board: chess.Board, start: chess.Square, to: chess.Square) -> bool:
    """Whether the piece on `to` could have come from the empty `start` in one move: its own
    geometry on the board as it stands, the pawn's pushes and captures, castling's king."""
    piece = board.piece_at(to)
    if piece is None or board.piece_at(start) is not None:
        return False
    if piece.piece_type == chess.PAWN:
        step = 8 if piece.color == chess.WHITE else -8
        if start == to - step:
            return True
        if start == to - 2 * step and board.piece_at(to - step) is None:
            return chess.square_rank(to) == (3 if piece.color == chess.WHITE else 4)
        return start in (to - step - 1, to - step + 1) and (chess.square_distance(start, to) == 1)
    if (
        piece.piece_type == chess.KING
        and chess.square_rank(start) == chess.square_rank(to)
        and abs(chess.square_file(start) - chess.square_file(to)) == 2
    ):
        return True  # castling
    back = board.copy(stack=False)
    back.remove_piece_at(to)
    back.set_piece_at(start, piece)
    return to in back.attacks(start)


@dataclass(frozen=True)
class LastMoveSeen:
    """What the site's last-move highlight tells a position read from pixels: who moved, and
    the en-passant square when the move was a pawn's two-square advance."""

    mover: chess.Color
    ep_square: chess.Square | None = None


def last_move_seen(board: chess.Board, highlighted: frozenset[chess.Square]) -> LastMoveSeen | None:
    """The last move as the site highlights it: an empty marked square and a marked square
    whose piece could have come from it. Every such pair must name the same mover, or nothing
    is known. A position read from pixels has no history; the highlight is the history:

    - the side to move is the other side -- a puzzle adopted with the side to move open could
      not tell the solver's move and the instant reply from the same two moves in the other
      order, and the record stayed empty ("sometimes there is no move list whatsoever");
    - a pawn's two-square advance can be taken in passing: after c2-c4 the coach did not see
      bxc3 e.p. and recommended Bd7 (Tai's chess.com puzzles, 2026-09-24)."""
    movers: set[chess.Color] = set()
    ep: chess.Square | None = None
    for to in highlighted:
        piece = board.piece_at(to)
        if piece is None:
            continue
        for start in highlighted:
            if start == to or not _came_from(board, start, to):
                continue
            movers.add(piece.color)
            step = 8 if piece.color == chess.WHITE else -8
            if piece.piece_type == chess.PAWN and start == to - 2 * step:
                ep = to - step
    if len(movers) != 1:
        return None
    return LastMoveSeen(mover=movers.pop(), ep_square=ep)


@dataclass(frozen=True)
class _QueueWay:
    """One way a premove ghost could have been queued: by the piece `symbol` from `origin`,
    landing on the red squares of `route` in turn (the target not among them); `passes_over`
    when a slider's step crossed a red square. The symbol is the piece standing on the
    origin: the ghost's own, or a pawn's when the ghost is the piece a queued pawn promotes
    to."""

    origin: chess.Square
    route: tuple[chess.Square, ...]
    symbol: str
    passes_over: bool = False

    @property
    def visited(self) -> frozenset[chess.Square]:
        return frozenset(self.route)


Explanation = tuple[chess.Move, ...]
PendingKey = tuple[object, ...]
# Pending keys whose last element is a story of moves from the latest position.
STORY_KEYS = ("moves", "moves-cautious")


# How long a premove queue, once its red tint is seen, is taken to be about: a queue
# executes one step for each of the opponent's replies, within a second or two.
QUEUE_MEMORY_S = 3.0
# How long a premove queue drawn on the board holds a position resync off (`_red_since`). A
# player may queue a whole opening before the opponent's first move (a stream of Hikaru's:
# some twenty premoves, over ten seconds); a red mark left on the board past this is not a
# queue, and must not stall the game for good.
QUEUE_RESYNC_S = 30.0


def in_flight(moves: Sequence[chess.Move]) -> bool:
    """Whether a story has a piece stop partway along a straight line and then go on the same
    way: a to x, and later x to y by that piece, x strictly between a and y. Seen on one frame,
    that is one move in flight, never two. Every site slides a moved piece in a straight line
    from its square to the next, and a frame that catches it near the centre of a square on
    its way reads as a piece standing there; the next frame, with the piece further on (or
    landed), then explains as the same piece moving on. Chess.com's phone app slides every
    move and cannot be told not to, speeding up as it goes, with a trail (Tai, 2026-09-25): "I
    played Rxc1+ but it misunderstood as Rc7". Geometry alone, for every piece, both sides,
    any speed and any capture rate; the highlights are not read (they can be turned off, and
    section 5.3 of DESIGN.md keeps the reconciler free of them). A piece that returns to a
    square it left, or another piece moving next, is not in flight. The caller decides when
    the shape means flight (`Reconciler._flight`): a pawn pushed twice is two moves when each
    stood still to be seen."""
    landed: dict[chess.Square, chess.Move] = {}
    for move in moves:
        came = landed.pop(move.from_square, None)
        if came is not None and move.from_square in chess.SquareSet(
            chess.between(came.from_square, move.to_square)
        ):
            return True
        landed[move.to_square] = move
    return False


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
    # Positions the missed-move search may expand per depth. Only moves that touch an
    # unexplained square are expanded, so a four-ply gap in a middlegame costs a few
    # hundred; counting every legal move looked at, as before, spent the budget on the
    # thirty rejected for each one expanded and left four-ply gaps unbridged.
    search_budget: int = 4000
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
    # Position resync: adopt a fully readable, stable board that no move story explains --
    # held this many frames; or, when at least `resync_jump_cells` cells disagree (a new
    # position, not a move), at least `resync_min_frames` spanning `resync_hold_s`: on a
    # phone reading seven frames a second, ten frames were a second and a half, and more
    # when the frames were slow ones (Tai's puzzles on the Pixel, 2026-09-23). A move is a
    # handful of cells, and one whose squares wear the site's badges waits for the stale
    # retry to explain it: adopted early, the mating move under the result badges went
    # unrecorded (stress seed 839).
    resync_hold_frames: int = 10
    resync_min_frames: int = 3
    resync_hold_s: float = 0.4
    resync_jump_cells: int = 12
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
    # What this game is (GameStarted.kind): a game from the start, or a position adopted.
    kind: str = "game"
    # The placement the resync would adopt, how many frames have shown it, and since when.
    _resync_candidate: tuple[str, int, float] | None = None
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
    # Squares something is drawn over in the current frame: unreadable, and no move
    # commits through them, though the drawing blocks nothing else.
    _drawn: set[chess.Square] = field(default_factory=set)
    # In the retry that ignores stale cells: the exact, settled readings among the ignored
    # cells. A story need not explain them, but may not contradict one it changes.
    _denied: Observation = field(default_factory=dict)
    _frame_index: int = 0
    _blocked: bool = False
    _last_obs: Observation = field(default_factory=dict)
    _ts: float = 0.0
    _last_commit_ts: float | None = None
    # When a premove queue's red tint was last seen (`_flight`), and since when it has been
    # on every frame (`_maybe_position_resync`).
    _red_ts: float | None = None
    _red_since: float | None = None
    # In the current frame: the squares a premove ghost or its queue covers (`_unqueue`),
    # unread and wild for every story -- a real move may leave or reach one unseen. The queen
    # that left b5 as the pawn's queue was drawn onto it was never told, and the game after
    # it was lost (Tai's game, 2026-09-28).
    _hidden: frozenset[chess.Square] = frozenset()
    # Each premove queue as last undone: its origin, the squares it lands on in turn, and its
    # ghost's square. A picture shows a queue's squares and not their order; the frames it
    # was built on and executed on show the order (`_route_seen`).
    _routes: tuple[tuple[chess.Square, ...], ...] = ()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def result(self) -> str | None:
        """The finished game's result ("1-0", "0-1", "1/2-1/2", "*"), else None."""
        return self._result if self._finished else None

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
        self._drawn = set()
        kings = [k for k in (self.board.king(chess.WHITE), self.board.king(chess.BLACK)) if k]
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                sq = square_at(r, c, self.bottom)
                if reading.drawn:
                    self._drawn.add(sq)
                if (
                    reading.kind == "unknown"
                    and reading.drawn
                    and not reading.covered
                    and any(chess.square_distance(sq, k) <= 1 for k in kings)
                ):
                    # The check glow is saturated colour the set never draws with, so
                    # on some themes it reads as a drawing over the king's square, or
                    # the square the king just left for. A drawing on or beside a king
                    # is the king under its glow (or an arrow from it, which is as
                    # occupied): a piece of no known kind, which the decorated-king
                    # rules take as the king, and which blocks nothing.
                    reading = replace(
                        reading, kind="piece", symbol=None, colour=None, settled=False
                    )
                if reading.kind == "unknown":
                    self._previous.pop(sq, None)
                    if reading.covered:
                        remembered = self._last_seen.get(sq)
                        if remembered and self._frame_index - remembered[1] <= self.memory_frames:
                            out[sq] = remembered[0]
                    continue
                out[sq] = reading
                if not (reading.exact and reading.settled and not reading.hover):
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

    def disagreements(
        self, obs: Observation, board: chess.Board | None = None
    ) -> set[chess.Square]:
        """The squares whose reading disagrees with `board` (the latest position)."""
        board = self.board if board is None else board
        return {sq for sq, reading in obs.items() if not cell_agrees(reading, board.piece_at(sq))}

    # Lifecycle

    def start(self) -> list[Event]:
        self._clear_story()
        self.viewing = None
        self._turn_unknown = False
        self._resync_candidate = None
        self._last_seen = {}
        self._previous = {}
        self._pending_age = 0
        started = self._begin(chess.Board(), "game")
        return [started, PositionChanged(game_id=self.game_id, fen=self.board.fen(), ply=0)]

    def _begin(self, board: chess.Board, kind: str) -> GameStarted:
        """A game of `kind` begun on `board`, under a new id."""
        self.game_id = uuid.uuid4().hex[:8]
        self.kind = kind
        self._started = True
        self._set_position(board)
        return self._game_started()

    def _set_position(self, board: chess.Board) -> None:
        """`board` adopted as the game's position, the base its moves are counted from; the
        first move's think time counts from now."""
        self.board = board
        self._base_fen = board.fen()
        self._finished = False
        self._result = None
        self._last_commit_ts = self._ts

    def _game_started(self) -> GameStarted:
        return GameStarted(
            game_id=self.game_id,
            user_side=self.user_side,
            start_fen=self.board.fen(),
            kind=self.kind,
        )

    def _clear_story(self) -> None:
        """No story pending, no cell remembered as stale or unexplained: the position they
        were read against is gone."""
        self._pending = None
        self._stale = {}
        self._unexplained_frames = 0

    def _restart(self) -> list[Event]:
        """The game under way ended, unless it had, and a new one begun from the start."""
        ended = [] if self._finished else [self.replaced_event("new game started")]
        return ended + self.start()

    def reorient(self, bottom: Side) -> list[Event]:
        """The board was flipped or re-themed before the first move: the same game, seen anew.

        Everything remembered per square is dropped, since the squares were mapped with the
        old orientation; a game already announced is announced again with the user's side.
        """
        self.bottom = bottom
        self.user_side = bottom
        self._clear_story()
        self._pending_age = 0
        self._last_seen = {}
        self._previous = {}
        self.viewing = None
        return [self._game_started()] if self._started else []

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
        unreadable. One move of a story may live on wild cells alone; a second is a
        move nobody saw invented to fill a turn, and a piece that goes and comes straight
        back is the same invention in one piece. An ignored cell that reads exactly
        (`_denied`) may stay as wrong as it was, but a story that changes it must agree
        with it: a stale empty square is no place to land a queen. All three once told a
        queen's shuffle, or its tour to a square that read empty, through the cells a
        badge and a stale reading had hidden, while the queen's real move went untold.

        This is the catch-up for missed moves: the window was minimised, a dialog covered the
        board, or a badge hid a piece until the next move. Every move must touch a still
        unexplained square, every touched square must be read exactly, and the final position
        must agree with every readable cell. The shortest paths are kept, as a person would
        assume the simplest story; transpositions among them lead to one position and are
        accepted, while different final positions are a genuine ambiguity and nothing is
        committed.
        """
        wild = wild | self._hidden
        # A move touches at most four squares (castling), so more disagreements than that
        # times the depth cannot be a missed-move story.
        if len(diff) + len(wild) < 3 or len(diff) > 4 * self.max_plies:
            return []

        start = base if base is not None else self.board
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
                unseen: int,
                decorated: set[chess.Square] = frozenset(),  # type: ignore[assignment]
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
                    if (
                        len(path) >= 2
                        and move.from_square == path[-2].to_square
                        and move.to_square == path[-2].from_square
                    ):
                        continue  # there and straight back: no trace, no story
                    touched = touched_squares(board, move)
                    seen = bool(touched & remaining)
                    if not seen and (unseen >= 1 or not touched & wild):
                        continue  # a second move told through wild cells alone
                    lenient = {
                        sq
                        for sq in touched
                        if sq not in wild
                        and not (sq in obs and obs[sq].exact)
                        and self._king_lands(board, move, sq, obs)
                    }
                    if not all(
                        sq in wild or (sq in obs and obs[sq].exact) or sq in lenient
                        for sq in touched
                    ):
                        continue
                    budget[0] -= 1
                    if budget[0] < 0:
                        return
                    after = board.copy(stack=False)
                    after.push(move)
                    path.append(move)
                    still = remaining - touched
                    used = set(decorated) | lenient
                    if not still:
                        if (
                            matches(after, obs, set(obs) - wild)
                            and self._not_denied(after, start)
                            and self._checked_kings(after, used)
                        ):
                            solutions.append(tuple(path))
                    else:
                        search(after, still, path, unseen + (0 if seen else 1), used)
                    path.pop()

            search(start, set(diff), [], 0)
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

    def _decorated_king(self, square: chess.Square, reading: CellReading) -> bool:
        """Whether an unsettled reading of a piece of no known kind is a king under the
        site's decoration: on the king's own square, or one step from it (a king that
        just moved, its square decorated in the same frame). The colour, when read,
        must be the king's."""
        if reading.symbol is not None:
            return False
        for colour in chess.COLORS:
            if reading.colour is not None and reading.colour != Side.of(colour):
                continue
            king = self.board.king(colour)
            if king is not None and chess.square_distance(king, square) <= 1:
                return True
        return False

    @staticmethod
    def _king_lands(
        board: chess.Board, move: chess.Move, square: chess.Square, obs: Observation
    ) -> bool:
        """Whether `move` is a king's move onto `square`, where the reading is a piece of
        the king's colour whose kind could not be read: the king under its decoration.
        A site decorates a king's square when the king is in check, so a story that uses
        this must end with that king in check (`_checked_kings`): a king moves into no
        check of its own, and the decoration on the square it landed on is the reply's."""
        if move.to_square != square or square not in obs:
            return False
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.KING:
            return False
        reading = obs[square]
        if reading.kind != "piece" or reading.symbol is not None:
            return False
        return reading.colour is None or reading.colour == Side.of(piece.color)

    @staticmethod
    def _checked_kings(after: chess.Board, decorated: set[chess.Square]) -> bool:
        """Whether every decorated landing square of a story holds a king in check at
        its end -- the mate, or the check, that the decoration marks."""
        if not decorated:
            return True
        return after.is_check() and after.king(after.turn) in decorated

    def _not_denied(self, after: chess.Board, start: chess.Board) -> bool:
        """Whether `after` agrees with every ignored exact reading on a cell it changed."""
        return all(
            after.piece_at(sq) == start.piece_at(sq) or cell_agrees(reading, after.piece_at(sq))
            for sq, reading in self._denied.items()
        )

    def _history_index(self, obs: Observation, diff: set[chess.Square]) -> int | None:
        """Index of the earlier position of this game the frame shows, if any.

        Every position back to the base is checked, one by one, however long the game: a
        click anywhere in the move list shows one. Only the last forty were, and the base;
        a click further back read as a position no move explains and was adopted by a
        resync -- the game numbered again from it, or, after the end, a new game begun --
        and stepping forward through the list then recorded the game's own moves a second
        time (the state review, 2026-10-01).
        """
        earlier = self.board.copy()
        undone: set[chess.Square] = set()
        plies = len(self.board.move_stack)
        for back in range(1, plies + 1):
            move = earlier.pop()
            undone |= touched_squares(earlier, move)
            if diff <= undone and matches(earlier, obs, set(obs)):
                return plies - back
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
        square. So every square the moves touch must hold a settled piece or be empty, the
        square a move lands on must be read at all, and no square on the board may carry a
        hover outline. A square with an arrow over it is no witness either: an arrowhead
        over a piece can look like another piece, so nothing commits through it, while the
        arrow, which is no drag, holds nothing else back.
        Highlights are not consulted: their colours vary by theme and by what the user
        clicks, and the reconciler stays agnostic to them by design (DESIGN.md section 5.3).
        One tint is: a piece of the user's on a premove square is a queue's ghost, drawn
        where the queue ends, and never a move played -- a queued piece that arrives for real
        turns its square the last-move colour. Taken for a move (Qe4+ where the queen had
        only got to h4 of its queue h4-e4), it left the game a piece off once the queue
        cleared, two cells too few for a resync to put right (the premove sweep, seed 90).
        A king is exempt: a king in check glows red.
        """
        if self._hover_seen or any(reading.hover for reading in obs.values()):
            return False
        wild = wild | self._hidden
        touched: set[chess.Square] = set()
        # The squares a king of the story lands on: its reading may be unsettled and of
        # no known kind, the king under what the site draws on it (see `_king_lands`).
        decorated: set[chess.Square] = set()
        board = (base if base is not None else self.board).copy(stack=False)
        user = self.user_side.color
        for move in moves:
            if move not in board.legal_moves:
                return False
            if move.to_square not in wild and move.to_square not in obs:
                # The piece must be seen where it landed. A bishop animated across a
                # square read as unknown while it passed over it, and the square it
                # had left explained the move to that square on its own.
                return False
            landed = obs.get(move.to_square)
            if (
                board.turn == user
                and board.piece_type_at(move.from_square) != chess.KING
                and landed is not None
                and landed.kind == "piece"
                and landed.tint == "red"
            ):
                return False
            if self._king_lands(board, move, move.to_square, obs):
                decorated.add(move.to_square)
            touched |= touched_squares(board, move)
            board.push(move)
        if not self._checked_kings(board, decorated):
            return False
        for sq in touched:
            if sq in wild or sq in decorated:
                continue
            if sq in self._drawn:
                return False
            reading = obs.get(sq)
            if reading is not None and reading.unsettled:
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
        # A piece of no known kind, off its centre, on a king's square or the one next to
        # it is not a drag but the king under what the site draws on it -- the check glow,
        # the badge, the label at the end -- and blocks nothing.
        self._blocked = self._hover_seen or any(
            reading.kind == "piece"
            and not reading.settled
            and not self._decorated_king(square_at(r, c, self.bottom), reading)
            for r, row in enumerate(readings)
            for c, reading in enumerate(row)
        )
        seen = self._observe(readings)
        # A queue leaves its piece's real square red and empty; a checked king's red glow
        # stands on a piece, and says nothing of a queue.
        if any(r.tint == "red" and r.kind == "empty" for r in seen.values()):
            self._red_ts = ts
            if self._red_since is None:
                self._red_since = ts
        else:
            self._red_since = None
        obs = self._unqueue(seen)  # premove ghosts put back
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
        self._last_obs = obs

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
            soft = frozenset(sq for sq, reading in obs.items() if reading.junk)
            subsets = self._stale_subsets(stale, obs)
            # Soft wildcards only go with junk cells: ignoring a clean reading as well
            # would let the search invent a story through the cells nobody can read.
            attempts = [(sub, sub | soft) for sub in subsets if all(obs[sq].junk for sq in sub)]
            attempts += [(sub, frozenset(sub)) for sub in subsets]
            tried: set[tuple[frozenset[chess.Square], frozenset[chess.Square]]] = set()
            for ignored, wild in attempts:
                if (ignored, wild) in tried:
                    continue
                tried.add((ignored, wild))
                fresh = diff - ignored
                if not fresh - wild:
                    # A story must rest on a disagreement read cleanly -- a square read
                    # empty or a settled piece of a known kind. When every cell left to
                    # explain is a wildcard, nothing on the board says a move was played:
                    # a board locked 0.28 of a cell off (it had slid on after the lock was
                    # taken) read every piece unsettled, the top of the king on d2 read as
                    # a piece on d3, and the king's move there was told, d2 and d3 both
                    # wildcards (the geometry sweep, 2026-09-27, seed 97).
                    continue
                live_obs = {sq: r for sq, r in obs.items() if sq not in ignored}
                self._denied = {sq: obs[sq] for sq in ignored if obs[sq].exact and not obs[sq].junk}
                try:
                    explained = self._explain(
                        live_obs, fresh, obs, allow_paths=True, cautious=True, wild=wild
                    )
                finally:
                    self._denied = {}
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
            return events + self._restart()

        adopted = self._maybe_position_resync(obs, diff)
        if adopted:
            return events + adopted
        return events + self._maybe_resync(diff)

    def _unqueue(self, obs: Observation) -> Observation:
        """The picture with queued premoves undone: each of the user's pieces drawn at the
        end of a red queue is put back on the empty square it will leave from.

        Chess.com draws a premoved piece at its destination and tints the queue's squares
        red, so the screen is not a position. The one-piece case is explained as a queue
        below; but two pieces can be queued at once (the third live game ended with both
        queens premoved), and a queue can form right after a real move whose settled frame
        the capture never caught. Undoing the queue gives the true position, which the
        ordinary explanations then read -- the real move that slipped by, the opponent's
        reply, the queued step the moment it executes and its squares turn from red to the
        site's last-move colour.

        Where a ghost's piece really stands is read against the position -- the one the
        pending story leads to when there is one, since a bot answering every frame or two
        keeps the committed position a picture behind: on the square that position has
        the piece, when that square is red and empty (the queue formed there); else one
        move of the piece away from it, on a tinted, empty square (the first queued step
        executed and is not yet committed). From its square the piece must reach the
        ghost in queued steps, each landing on a red square; the chains of all the ghosts
        together must account for every red, empty square, each once (the ladder
        d3-d4-d5-d6 starts at d3, or d3 goes unaccounted for; two chains never share a
        square); chains built square by square, a slider stopping at each red square on
        its ray, are preferred to ones that pass a red square over (another piece's queue
        crossing the ray), and red origins to ones in the last-move colour; and there must
        then be exactly one reading. A piece on a red square that the position already
        has there is not a ghost (the user's own king in check, on chess.com's red
        square). The queue's end goes unread rather than empty: a queued capture is drawn
        over the piece it will take. Nothing is undone when the picture cannot be read for
        certain.
        """
        self._hidden = frozenset()
        red_empty = frozenset(sq for sq, r in obs.items() if r.tint == "red" and r.kind == "empty")
        if not red_empty:
            if self._red_ts is None or self._ts - self._red_ts > QUEUE_MEMORY_S:
                self._routes = ()  # no queue for a while: what was seen of one is history
            return obs
        pending = self._pending_story()
        anchor = pending[1] if pending is not None else self.board
        ghosts: list[tuple[chess.Square, CellReading]] = []
        for sq, reading in obs.items():
            if reading.tint != "red" or reading.kind != "piece" or reading.symbol is None:
                continue
            if reading.colour != self.user_side:
                continue
            here = anchor.piece_at(sq)
            if here is not None and here.symbol() == reading.symbol:
                continue  # really there; the square is red for another reason
            ghosts.append((sq, reading))
        if not ghosts:
            return obs
        # A piece of the opponent's on a red square is a queue's victim, taken on the way and
        # gone on from (g4xf3 queued after g7-g6-g5, then f3-f2: the premove sweep,
        # 2026-09-28): a square a route lands on and must account for, under a real piece
        # that stays. Never a king, which no queue takes and whose check glows red.
        victims = frozenset(
            sq
            for sq, r in obs.items()
            if r.tint == "red"
            and r.kind == "piece"
            and r.colour not in (None, self.user_side)
            and (r.symbol or "k").upper() != "K"
        )
        red = red_empty | victims
        options: list[list[_QueueWay]] = []
        for target, reading in ghosts:
            symbol = reading.symbol
            assert symbol is not None
            options.append(self._queue_origins(symbol, target, red, obs, anchor))
            if not options[-1]:
                return obs
        # Every red square is one queue's origin or one queue's landing, never two. A way
        # that goes on from a queue as it was last seen ranks first (`_route_seen`), then one
        # built square by square, then one whose origin is red.
        targets = [target for target, _ in ghosts]
        # By rank, the readings of every ghost's way, grouped by their origins.
        ranked: dict[
            tuple[int, int, int], dict[tuple[chess.Square, ...], list[tuple[_QueueWay, ...]]]
        ] = {}
        for choice in itertools.product(*options):
            origins = tuple(way.origin for way in choice)
            if len(set(origins)) != len(origins):
                continue
            covered = [o for o in origins if o in red_empty]
            for way in choice:
                covered.extend(way.visited)
            if len(covered) != len(set(covered)) or set(covered) != set(red):
                continue
            unseen = [not self._route_seen(w, t) for w, t in zip(choice, targets, strict=True)]
            rank = (
                sum(unseen),
                sum(1 for way in choice if way.passes_over),
                sum(1 for o in origins if o not in red_empty),
            )
            ranked.setdefault(rank, {}).setdefault(origins, []).append(choice)
        if not ranked:
            return obs
        best = ranked[min(ranked)]
        if len(best) != 1:
            return obs
        chosen, choices = next(iter(best.items()))
        ways = choices[0]
        routes = {
            tuple((way.origin, *way.route, t) for way, t in zip(c, targets, strict=True))
            for c in choices
        }
        if len(routes) == 1:
            # Remembered only when the picture leaves one order: a whole queue shown at once
            # reads as well rook c2-c5 with queen b3-c3-c4-d5-d6-d8 as the true c2-c3-c4-c5
            # with b3-d5-d6-d8, and the wrong one remembered misread every step after.
            self._routes = next(iter(routes))
        out = dict(obs)
        for (target, reading), way in zip(ghosts, ways, strict=True):
            # The piece on the origin: a pawn under a promoted ghost.
            out[way.origin] = replace(
                reading, symbol=way.symbol, highlighted=False, tint=None, settled=True
            )
            out.pop(target, None)
        for sq in red_empty:
            if sq not in chosen:
                out.pop(sq, None)  # an intermediate square of a queue: unread
        # What the ghosts and the queues cover is unknown, not contradicted: a real move may
        # start or end there without a reading (`_hidden`).
        self._hidden = frozenset(
            sq for sq in (*(t for t, _ in ghosts), *red_empty) if sq not in chosen
        )
        return out

    def _route_seen(self, way: _QueueWay, target: chess.Square) -> bool:
        """Whether a way to the ghost on `target` goes on from a queue as last seen: that
        route with its executed steps gone from the front, or new ones added at the end, or
        both. With no queue seen, every way does. A king queued g2-f3-f2-e3-f4 whose first
        step had executed read as well from f2 by f3 and e3 as from f3 by f2 and e3; the
        frames it was built on said which (the premove sweep, 2026-09-28)."""
        if not self._routes:
            return True
        route = (way.origin, *way.route, target)
        for seen in self._routes:
            for i in range(len(seen)):
                rest = seen[i:]
                if route[: len(rest)] == rest or rest[: len(route)] == route:
                    return True
        return False

    def _queue_origins(
        self,
        symbol: str,
        target: chess.Square,
        red: frozenset[chess.Square],
        obs: Observation,
        anchor: chess.Board,
    ) -> list[_QueueWay]:
        """Every way a ghost of `symbol` at `target` could have been queued: from a square
        `anchor` has such a piece on, or one move of it away (see `_unqueue`). A queen, rook,
        bishop or knight may also be a pawn queued to promotion: chess.com draws the piece it
        will promote to, a queen, from the moment the queue reaches the last rank (Tai's game,
        2026-09-28: b6 to b1 premoved, drawn as a queen on b1), and a queue may go on with
        that queen."""
        piece = chess.Piece.from_symbol(symbol)
        walkers = [piece]
        if piece.piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            walkers.append(chess.Piece(chess.PAWN, piece.color))
        out: list[_QueueWay] = []
        scratch = anchor.copy(stack=False)
        scratch.turn = piece.color
        for walker in walkers:
            for stand, here in anchor.piece_map().items():
                if here != walker:
                    continue
                reading = obs.get(stand)
                if reading is None or reading.kind != "empty":
                    continue  # still shown there, or unreadable: not this piece
                if stand in red:
                    out.extend(self._queue_ways(walker, stand, target, red, obs, piece))
                    continue
                # The first queued step has executed: the piece is one move on, on a tinted,
                # empty square the queue continues from -- promoted, if that step promoted.
                for move in scratch.generate_pseudo_legal_moves(chess.BB_SQUARES[stand]):
                    to = move.to_square
                    if to == target or move.promotion not in (None, piece.piece_type):
                        continue
                    landed = obs.get(to)
                    if landed is None or landed.kind != "empty" or landed.tint is None:
                        continue
                    moved = piece if move.promotion else walker
                    out.extend(self._queue_ways(moved, to, target, red, obs, piece))
        return out

    def _queue_ways(
        self,
        walker: chess.Piece,
        origin: chess.Square,
        target: chess.Square,
        red: frozenset[chess.Square],
        obs: Observation,
        ghost: chess.Piece,
    ) -> list[_QueueWay]:
        """The ways the piece `walker` on `origin` could be queued to `target`, where the
        picture draws `ghost`: every step lands on a red square, past the pieces the picture
        shows, within the queue length chess.com allows; a pawn reaching its last rank goes
        on as the ghost, the piece it promotes to, and must arrive as the ghost. A slider's
        step that passes a red square over (another piece's queue crossing its ray) marks its
        way `passes_over`, which ranks below one built square by square. Every route counts,
        not only the shortest: a king queued e1-d2-d1 lands on d2, and the direct e1-d1 left
        d2 unaccounted for, so nothing was undone (the premove sweep, 2026-09-28)."""
        # The board the queue was made on, as far as the picture tells it: the user's own
        # pieces standing still (not the ghosts, on red), and the opponent's pieces under red,
        # a queue's victims. The opponent's other pieces may have come since, into the way:
        # chess.com draws a queue until its step comes and fails (h2-h4-h5 queued, then
        # h7-h5 played in front of it: the premove sweep, 2026-09-28).
        scratch = chess.Board(None)
        for sq, reading in obs.items():
            if reading.kind != "piece" or reading.symbol is None:
                continue
            piece = chess.Piece.from_symbol(reading.symbol)
            if (piece.color == walker.color) == (reading.tint != "red"):
                scratch.set_piece_at(sq, piece)
        scratch.turn = walker.color
        landings = red | {target}
        last_rank = 7 if walker.color == chess.WHITE else 0
        ways: list[_QueueWay] = []

        def steps(sq: chess.Square, piece: chess.Piece) -> list[chess.Square]:
            scratch.set_piece_at(sq, piece)
            out = {m.to_square for m in scratch.generate_pseudo_legal_moves(chess.BB_SQUARES[sq])}
            scratch.remove_piece_at(sq)
            if piece.piece_type == chess.PAWN:
                if chess.BB_PAWN_ATTACKS[piece.color][sq] & chess.BB_SQUARES[target]:
                    # A queued capture is drawn over its victim, which the picture then does
                    # not show; a pawn takes only onto a piece (d5xc6 queued after d4-d5 onto
                    # a knight on c6, the premove sweep, 2026-09-28).
                    out.add(target)
                ahead = sq + (8 if piece.color == chess.WHITE else -8)
                if 0 <= ahead < 64 and ahead in red:
                    out.add(ahead)  # a push queued before a piece stepped in front of it
            # In square order: a set's order decided which of two equal routes was kept,
            # and it differs between Python and the phone (the parity audit).
            return sorted(out & landings)

        def walk(
            sq: chess.Square, piece: chess.Piece, route: tuple[chess.Square, ...], over: bool
        ) -> None:
            for to in steps(sq, piece):
                if to == origin or to in route:
                    continue
                crossed = over or any(b in red for b in chess.SquareSet(chess.between(sq, to)))
                promotes = piece.piece_type == chess.PAWN and chess.square_rank(to) == last_rank
                now = ghost if promotes else piece
                if to == target:
                    if now == ghost:
                        ways.append(_QueueWay(origin, route, walker.symbol(), crossed))
                elif len(route) + 2 <= self.max_queue_steps:
                    walk(to, now, (*route, to), crossed)

        walk(origin, walker, (), False)
        return ways

    @property
    def pending_story(self) -> bool:
        """Whether a move story seen once is waiting for its confirming frame."""
        return self._pending_story() is not None

    @property
    def blocked(self) -> bool:
        """Whether the last frame held a drag in progress -- a hover outline, or a piece
        not settled on its square -- which blocks every story without contradicting any."""
        return self._blocked

    @property
    def unexplained(self) -> bool:
        """Whether the last frame disagreed with the position and no story explained it."""
        return self._unexplained_frames > 0

    @property
    def unexplained_frames(self) -> int:
        """For how many frames in a row the board has disagreed and no story explained it."""
        return self._unexplained_frames

    @property
    def agrees(self) -> bool:
        """Whether the last readable frame shows the latest position, ghosts aside: the
        record is up to date with the screen. False while a story is pending or a frame
        is unexplained, and in the frame a story's first move commits with its second
        still to be told (a premove landing on the opponent's reply)."""
        return not self.disagreements(self._last_obs)

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
        self._clear_story()
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
        if self._red_since is not None and self._ts - self._red_since < QUEUE_RESYNC_S:
            # A premove queue drawn: the picture is the position with the queued pieces at
            # the ends of their queues, not a position. Adopted, it was the queue's end
            # taken for the game (Tai's game, 2026-09-28: b6-b1=Q premoved; the moves till
            # then lost). Once the queue has resolved it is a position again.
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
        # Held as a view: the pieces and the squares the site highlights, which a move
        # starting changes first (`pipeline._view`).
        placement = (
            loose_placement(obs).board_fen()
            + "|"
            + ",".join(chess.square_name(sq) for sq in sorted(highlighted_squares(obs)))
        )
        if self._resync_candidate and self._resync_candidate[0] == placement:
            self._resync_candidate = (
                placement,
                self._resync_candidate[1] + 1,
                self._resync_candidate[2],
            )
        else:
            self._resync_candidate = (placement, 1, self._ts)
        _, frames, since = self._resync_candidate
        held = frames >= self.resync_hold_frames or (
            len(diff) >= self.resync_jump_cells
            and frames >= self.resync_min_frames
            and self._ts - since >= self.resync_hold_s
        )
        if not held:
            return []

        adopted = self._adopt_placement(
            loose_placement(obs),
            prefer=self.bottom.color,
            highlighted=highlighted_squares(obs),
        )
        if adopted is None:
            self._resync_candidate = None
            return []
        board, self._turn_unknown = adopted
        self._clear_story()
        self.viewing = None
        self._resync_candidate = None
        if self._finished:
            # The finished game is over for good; a position that appears after it is
            # another game, seen from the middle -- or a puzzle, or an analysis board. Its
            # start is all there is to say: a Resync after it as well had the engine measure
            # the position twice and the coach note a re-read board that was never read
            # before (the state review, 2026-10-01).
            return [self._begin(board, "position"), *self._positioned()]
        self._set_position(board)
        reason = f"position adopted after {len(diff)} unexplained cells"
        if self._turn_unknown:
            reason += "; side to move pending"
        return self._resynced(reason)

    def _resynced(self, reason: str) -> list[Event]:
        """A new position for the game: Resync and PositionChanged, and the end when the
        position is over and the side to move known."""
        return [
            Resync(game_id=self.game_id, fen=self.board.fen(), reason=reason),
            *self._positioned(),
        ]

    def _positioned(self) -> list[Event]:
        """The position the game now stands on, numbered from it: PositionChanged, and the
        end when the position is over and the side to move known."""
        events: list[Event] = [PositionChanged(game_id=self.game_id, fen=self.board.fen(), ply=0)]
        if not self._turn_unknown and self.board.is_game_over():
            self._finished = True
            events.append(self.ended_event(self.board.result(), "game over on board"))
        return events

    @staticmethod
    def _adopt_placement(
        placement: chess.Board,
        prefer: chess.Color = chess.WHITE,
        highlighted: frozenset[chess.Square] = frozenset(),
    ) -> tuple[chess.Board, bool] | None:
        """A bare placement made into a position: castling rights assumed conservatively from
        the king and rook home squares, and the turn from legality. Returns the board and
        whether the turn is still unknown (both colours legal to move), or None if neither is.
        Shared by the position resync and the cold start from the theme cache. When both
        could move, `prefer` is the turn the board shows meanwhile: the side at the bottom,
        which on a puzzle is the solver and to move (the coach then looks at the right side
        while the clock, or the first move seen, settles it); white was the old default,
        and a puzzle for Black was coached for White.
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
        # The site's last-move highlight: the other side is to move, and may take a pawn
        # that has just advanced two squares in passing (`last_move_seen`).
        seen = last_move_seen(board, highlighted)
        if seen is not None and (not seen.mover) in valid_turns:
            board.turn = not seen.mover
            board.ep_square = seen.ep_square
            return board, False
        board.turn = prefer if prefer in valid_turns else valid_turns[0]
        return board, len(valid_turns) > 1

    def turn_from_clock(self, colour: chess.Color) -> list[Event]:
        """The side to move settled by the clock that is running.

        A resync or a cold start leaves the turn open when both sides could legally
        move, and the first story then decides it -- which a takeback answers wrongly:
        two pieces going back to where they stood read as two moves in either order,
        and the order chosen fixed the turn on the wrong side (2026-09-19: the coach
        then named the opponent's best move for 173 seconds of the player's own clock).
        The running clock is evidence of the turn that no story can contradict. A colour
        that cannot be to move -- the other king is in check -- is ignored; a turn that
        changes re-announces the position, so the coach and the engine look at the right
        side."""
        if not self._turn_unknown or not self._started or self._finished:
            return []
        trial = self.board.copy(stack=False)
        if not _turn_valid(trial, colour):
            return []
        self._turn_unknown = False
        if self.board.turn == colour:
            return []
        # The turn stays open only until the first commit, so nothing has been played
        # from the position yet: the base is the position with the turn corrected.
        assert not self.board.move_stack
        self.board.turn = colour
        self._base_fen = self.board.fen()
        return self._resynced(f"side to move settled by the clock: {Side.of(colour).value}")

    def start_from(
        self,
        placement: chess.Board,
        ts: float | None = None,
        highlighted: frozenset[chess.Square] = frozenset(),
    ) -> list[Event]:
        """Begin a game from an arbitrary observed position -- a puzzle or a mid-game board read
        cold from the theme cache -- rather than from the start. Rights and turn are assumed the
        same way a position resync assumes them; the game continues normally from here. `ts`
        is when the position was read, so the first move's think time counts from it (it
        counted from zero: a first move that "took 412630 seconds")."""
        adopted = self._adopt_placement(
            placement, prefer=self.bottom.color, highlighted=highlighted
        )
        if adopted is None:
            return []
        if ts is not None:
            self._ts = ts
        board, self._turn_unknown = adopted
        return [self._begin(board, "position")]

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
            self._clear_story()
            return []

        if self._finished:
            return None

        if self.viewing is not None:
            earlier = self._board_at(self.viewing)
            diff_earlier = self.disagreements(obs, earlier)
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

    def _flight(self, moves: Explanation) -> bool:
        """Whether `moves`, the pending story or a story built on it, catch a piece in flight:
        the pending story seen on one frame only -- a piece that stood still on a square was
        seen there twice, which a slide never is -- no premove queue about, whose steps land
        one for each reply faster than any frame can witness them, and a piece stopping
        partway along its line and going on the same way (`in_flight`)."""
        seen_once = self._pending is None or self._pending[1] < 2
        queue = self._red_ts is not None and self._ts - self._red_ts <= QUEUE_MEMORY_S
        return seen_once and not queue and in_flight(moves)

    def _pending_story(self) -> tuple[Explanation, chess.Board] | None:
        """The pending story from the latest position and the position it leads to. A story
        with a piece in flight (`in_flight`) is nothing to build on: it settles, if it ever
        does, on frames of its own."""
        if self._pending is None or self._pending[0][0] not in STORY_KEYS:
            return None
        moves = self._pending[0][-1]
        assert isinstance(moves, tuple)
        if self._flight(moves):
            return None
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
            if reading.unsettled:
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
        diff_base = self.disagreements(obs, base)
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
        if self._flight(combined):
            # The frame shows a piece of the pending story further along its own line: the
            # story caught it in flight, and this frame is no second witness of it.
            return None
        if not self._highlight_ok(obs, combined):
            return None
        # A bot that answers within a frame or two and a queue that executes on every
        # answer make a chain that grows by two plies a picture and never shows the same
        # picture long enough to settle. This frame is explained from the position the
        # pending plies lead to: the second witness those plies were waiting for. So they
        # are committed now, and only the plies this frame added stay pending.
        events = self._commit_moves(tuple(moves))()
        self._pending = (("moves", tuple(story)), 1)
        self._pending_age = 0
        self._stale = {}
        self._unexplained_frames = 0
        return events

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
        user_colour = self.user_side.color
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
        if len(combined) > self.max_chain_plies or self._flight(combined):
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
        user_colour = self.user_side.color
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
        # The queue's way is the way on the board the reply leaves: a knight that just
        # left the queen's diagonal made her one step of two (a premoved mate once
        # committed as the queen's stop on the way, Qf4 for Qf8#).
        path = self._queue_path(origin, target, symbol, tinted, after)
        if path is None or path[0] not in after.legal_moves:
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
        board: chess.Board | None = None,
    ) -> tuple[chess.Move, ...] | None:
        """The sequence of moves taking the user's piece from origin to target, on
        `board` (the latest position by default).

        When the site tints the queued squares (`tinted` holds them, the target included;
        the origin only until its first step has executed), the route through exactly
        those squares is the queue, however long.
        Otherwise the unique shortest way is assumed. Squares on the way must be empty; only
        the last step may capture. A pawn reaching the last rank promotes to the piece
        drawn there, and a pawn still drawn there means the choice is not made yet, so
        nothing is known. Several equally short ways, as a king has across an open board,
        are an ambiguity and nothing is assumed.
        """
        if board is None:
            board = self.board
        piece = board.piece_at(origin)
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
        scratch = board.copy(stack=False)
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
        self._clear_story()
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
            events = self._restart()
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

        junk = sorted(sq for sq in stale if obs[sq].junk)
        clean = sorted(sq for sq in stale if not obs[sq].junk)
        pool = junk if len(stale) > self.max_stale_subset_cells else junk + clean
        subsets: list[frozenset[chess.Square]] = [frozenset({sq}) for sq in pool]
        if self.resync_cells > 2:
            subsets += [frozenset({a, b}) for i, a in enumerate(pool) for b in pool[i + 1 :]]
        return subsets

    def _maybe_resync(self, diff: set[chess.Square]) -> list[Event]:
        """Say so when several cells have stayed unexplained for a long time. The game is
        kept as it stands -- a position resync is what adopts a new one -- so this is a
        note (`Unexplained`), not a Resync, which everything downstream reads as the
        numbering starting again."""
        if not diff:
            self._unexplained_frames = 0
            return []
        self._unexplained_frames += 1
        persistent = [sq for sq in diff if self._stale[sq][1] > self.stale_after]
        if len(persistent) < self.resync_cells or self._unexplained_frames < self.resync_after:
            return []
        self._unexplained_frames = 0
        names = ", ".join(chess.square_name(sq) for sq in sorted(persistent))
        return [
            Unexplained(game_id=self.game_id, fen=self.board.fen(), reason=f"unexplained: {names}")
        ]

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
                Unexplained(
                    game_id=self.game_id,
                    fen=self.board.fen(),
                    reason=f"refused illegal commit {move.uci()}",
                )
            ]
        fen_before = self.board.fen()
        san = self.board.san(move)
        by_user = self.board.turn == self.user_side.color
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
            events.append(self.ended_event(self._result, "game over on board"))
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
        return [self.ended_event(result, reason)]

    def replaced_event(self, reason: str) -> GameEnded:
        """The game under way ended because the next one replaced it, before its result was
        seen: its result is what its final position proves, as the library reads an import's
        (`stat_facts.ended_by`) -- a mate, a stalemate, dead material, a repetition or fifty
        moves played -- and "*" only when the position proves nothing (Tai, 2026-10-02)."""
        proved = result_on_board(self.board)
        if proved is None:
            return self.ended_event("*", reason)
        return self.ended_event(*proved)

    def end_unseen(self) -> list[Event]:
        """The session ends with this game under way and its end never seen: it ends with the
        result its final position proves, by the rule a replaced game's is set with
        (`replaced_event`). Nothing when the position proves none: the record keeps the game
        unfinished ("*"), as it did (Tai, 2026-10-02)."""
        if not self.in_progress or self._finished:
            return []
        proved = result_on_board(self.board)
        return [] if proved is None else self.end(*proved)

    def ended_event(self, result: str, reason: str) -> GameEnded:
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
