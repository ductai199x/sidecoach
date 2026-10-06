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
    TurnPending,
    Unexplained,
)
from sidecoach.state.stories import Search, Story, premove_picture
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


Explanation = tuple[chess.Move, ...]


# How long after the last frame a story with a premove queue fitted a position resync is
# held off (`_queue_ts`): a picture with a queue drawn is not a position. While the queue
# fits, nothing is re-read anyway (a player may queue a whole opening, some twenty premoves
# over ten seconds); once nothing has fitted for this long, the game is lost, and the board
# is read whole rather than left stuck (the premove sweep, 2026-10-05, seed 70).
QUEUE_RESYNC_S = 1.0
# A frame later than this after the last is a gap: moves may have come and gone unseen.
GAP_S = 0.5
# Stories kept at most, cheapest first (`stories.Story`).
MAX_STORIES = 12


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


def _en_passant_squares(board: chess.Board) -> list[chess.Square]:
    """The squares en passant could be open on, with `board.turn` to move: behind each pawn of
    the other side on its fourth rank whose two squares behind are empty (it may have just come
    from there)."""
    other = not board.turn
    rank = 3 if other == chess.WHITE else 4  # its fourth rank
    step = -8 if other == chess.WHITE else 8  # towards where it came from
    out = []
    for sq in board.pieces(chess.PAWN, other):
        if chess.square_rank(sq) != rank:
            continue
        behind, start = sq + step, sq + 2 * step
        if board.piece_at(behind) is None and board.piece_at(start) is None:
            out.append(behind)
    return out


def _lights(board: chess.Board, move: chess.Move, lit: frozenset[chess.Square]) -> bool:
    """Whether `move` is the one the two `lit` squares show: its own two squares, or, for a
    castling, the king's or the rook's."""
    if {move.from_square, move.to_square} == lit:
        return True
    if board.is_castling(move):
        file = 7 if chess.square_file(move.to_square) == 6 else 0
        rook_from = chess.square(file, chess.square_rank(move.from_square))
        return move.from_square in lit and (move.to_square in lit or rook_from in lit)
    return False


def _story_order(story: Story) -> tuple[object, ...]:
    """Stories cheapest first; of equal cost, a takeback or a new game before moves that
    retell it (a takeback and the bot's quick reply, never four retreats), then fewer moves.
    Python's sort is stable: what is left in the order the search found them, the order the
    moves are generated in, so the story followed does not change from frame to frame."""
    head = 0 if any(op[0] in ("back", "newgame") for op in story.head) else 1
    return (story.cost, head, len(story.ops), len(story.queue))


def _retold(story: Story, played: tuple[object, ...]) -> Story | None:
    """`story` told from a record that has played `played`, which it does not begin with:
    the moves the record played past their common start taken back, then the story's own.
    None when the record's part past the common start is no run of moves to take back, or
    the story's own head lies past it."""
    common = len(_common_start([story.ops, played]))
    if common < len(story.head):
        return None
    tail = played[common:]
    if not tail or not all(isinstance(op, chess.Move) for op in tail):
        return None
    moves = story.ops[common:]
    assert all(isinstance(op, chess.Move) for op in moves)
    return replace(story, head=(("back", len(tail)),), moves=moves)  # type: ignore[arg-type]


def _common_start(lists: list[tuple[object, ...]]) -> tuple[object, ...]:
    """The ops every list begins with."""
    out: list[object] = []
    for ops in zip(*lists, strict=False):
        if any(op != ops[0] for op in ops[1:]):
            break
        out.append(ops[0])
    return tuple(out)


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
    # Frames a board stays unexplained before the log is told what each square reads as.
    unexplained_note_frames: int = 20
    _unexplained_noted: bool = False
    board: chess.Board = field(default_factory=chess.Board)
    game_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    viewing: int | None = None
    _hover_seen: bool = False
    # The stories that fit the frames so far (`stories.Story`, docs/reconciler-model.md),
    # and what the record does next: each op every story agrees on, with the frames it has
    # held and the frames it needs before it is played (`_decide`).
    _stories: list[Story] = field(default_factory=lambda: [Story()])
    _agreed: list[tuple[object, int, int]] = field(default_factory=list)
    # Whether the last frame showed the record as it stands (a queue drawn on it aside).
    _agrees: bool = True
    # Whether the side to move was last told pending (`TurnPending`).
    _turn_pending_told: bool = False
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
    # Premove steps a frame may add to a queue (`stories.Search`).
    max_queue_steps: int = 3
    resync_max_soft: int = 2
    _base_fen: str = chess.STARTING_FEN
    _turn_unknown: bool = False
    # What this game is (GameStarted.kind): a game from the start, or a position adopted.
    kind: str = "game"
    # The placement the resync would adopt, how many frames have shown it, and since when.
    _resync_candidate: tuple[str, int, float] | None = None
    # The last `brief_kept` distinct placements a resync held for `brief_frames` clean frames: the
    # placement, its highlighted squares, its view key, and when it was last seen. A new
    # puzzle's first board stands for well under a second before the site plays the opening
    # move, too briefly to be adopted; the board after the move was, and the move was lost
    # (Tai's puzzles, 2026-10-03: "the coach misses the first move ... 75-80% of the time").
    # Adopted, the board after is played from the brief one when one legal move joins them.
    _resync_seen: list[tuple[chess.Board, frozenset[chess.Square], str, float]] = field(
        default_factory=list
    )
    # One clean read: the phone, its voice and engine busy, read the next puzzle's first board,
    # up a third of a second, once (Tai's dump 20261003-171356). What keeps a stray read from
    # standing in is that exactly one legal move must join it to the board adopted.
    brief_frames: int = 1
    # How many distinct boards held that long are kept: the first board, and frames past it.
    brief_kept: int = 4
    # How long before the adopted board the brief one may have been seen and still be its start.
    brief_recent_s: float = 2.0
    # The squares of a position read from the screen whose pieces have not moved since: their
    # kinds rest on that one read. A knight on a highlighted square read as a bishop made its
    # first move illegal, and the game's next three moves went unread until a resync (Tai's
    # puzzle on the Pixel, 2026-10-03); `_explain_misread` corrects such a piece by its move.
    _read_squares: set[chess.Square] = field(default_factory=set)
    misread_max_cells: int = 8
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
    _frame_index: int = 0
    _blocked: bool = False
    # Whether a piece is in the user's hand this frame (`update`).
    _dragging: bool = False
    # When the last frame came, and how many unseen pictures a story may need this frame.
    _frame_ts: float | None = None
    _max_unseen: int = 1
    _last_obs: Observation = field(default_factory=dict)
    _ts: float = 0.0
    _last_commit_ts: float | None = None
    # When a frame last fitted a story with a premove queue drawn (`_maybe_position_resync`).
    _queue_ts: float | None = None

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
        self._clear_story()
        # The start position is known, not read; any other is the screen's word for it.
        read = board.board_fen() != chess.STARTING_BOARD_FEN
        self._read_squares = set(board.piece_map()) if read else set()
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
        self._stories = [Story()]
        self._agreed = []
        self._stale = {}
        self._unexplained_frames = 0
        self._unexplained_noted = False

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
        self._last_seen = {}
        self._previous = {}
        self.viewing = None
        return [self._game_started()] if self._started else []

    # Explanations

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
        the king's colour whose kind could not be read, or one read off its square (a badge
        over the king reads as some piece): the king under its decoration.
        A site decorates a king's square when the king is in check, so a story that uses
        this must end with that king in check (`_checked_kings`): a king moves into no
        check of its own, and the decoration on the square it landed on is the reply's."""
        if move.to_square != square or square not in obs:
            return False
        piece = board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.KING:
            return False
        reading = obs[square]
        if reading.kind != "piece" or (reading.symbol is not None and reading.settled):
            return False
        return reading.colour is None or reading.colour == Side.of(piece.color)

    @staticmethod
    def _king_leaves(
        board: chess.Board, move: chess.Move, square: chess.Square, obs: Observation
    ) -> bool:
        """Whether `move` is a king's move from `square` onto a decorated square
        (`_king_lands`), and `square` reads as the decoration's spill: a shape off its
        square, of the king's colour or none. The badge of a mate spills over the square the
        king came from."""
        if move.from_square != square or square not in obs:
            return False
        if not Reconciler._king_lands(board, move, move.to_square, obs):
            return False
        reading = obs[square]
        piece = board.piece_at(square)
        assert piece is not None
        return (
            reading.kind == "piece"
            and not reading.settled
            and (reading.colour is None or reading.colour == Side.of(piece.color))
        )

    @staticmethod
    def _checked_kings(after: chess.Board, decorated: set[chess.Square]) -> bool:
        """Whether every decorated landing square of a story holds a king in check at
        its end -- the mate, or the check, that the decoration marks."""
        if not decorated:
            return True
        return after.is_check() and after.king(after.turn) in decorated

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
        # Pictures a story may need that no frame showed: one between frames that follow
        # each other (a bot answering inside a frame), as many as a catch-up may span after a
        # gap -- the board unseen, or a frame long in coming. Three events in a tenth of a
        # second is a story about cells nobody could read, not a game.
        gap = self._unseen > 0 or (self._frame_ts is not None and ts - self._frame_ts > GAP_S)
        self._max_unseen = self.max_plies if gap else 1
        self._frame_ts = ts
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
        obs = seen
        if len(obs) < self.min_readable_cells:
            return events
        diff = self.disagreements(obs)
        # What stays wrong is counted against the picture the likeliest story draws, not the
        # record: a premove queue drawn for ten seconds is the story, not stale junk.
        lead = self._lead_picture()
        off = self.disagreements(obs, lead)
        self._advance_stale(off, obs)
        ghosts = self._ghosts(off, obs, lead)
        if ghosts:
            # An arrow drawn over an empty square reads as a piece of no known kind, off
            # centre, for as long as it is drawn. Once such a reading has gone stale it is
            # taken as unknown, so a few arrows neither stall the game nor multiply the
            # stale retries below.
            obs = {sq: r for sq, r in obs.items() if sq not in ghosts}
            diff = diff - ghosts
            off = off - ghosts
            # Nor is an arrow a piece in someone's hand.
            self._blocked = self._hover_seen or any(
                r.kind == "piece" and not r.settled and not self._decorated_king(sq, r)
                for sq, r in obs.items()
            )
        self._last_obs = obs
        # A piece in the user's hand: the hover outline a site draws under it, or one of the
        # user's own pieces, known and off its square. A shape of no known kind off a square
        # is a decoration -- a badge, an arrow, a glow -- and no hand.
        self._dragging = self._hover_seen or any(
            r.kind == "piece"
            and not r.settled
            and r.symbol is not None
            and r.colour == self.user_side
            for r in obs.values()
        )

        if self._finished:
            return events + self._after_the_end(obs, diff)

        # A cell that has read the same wrong thing for a while (a badge, a ring) is left
        # out of the matching: it no longer stalls every story, as it witnesses no move.
        # A few only: more than a badge or two is a board that changed, for a resync.
        quiet = frozenset(sq for sq in off if self._stale[sq][1] > self.stale_after)
        if len(quiet) >= self.resync_cells:
            quiet = frozenset()
        stories = self._fitting(obs, diff, quiet=quiet)
        if stories:
            return events + self._decide(stories)
        # Nothing fits the frame: the record is not what the screen shows. A drag hides what is
        # under the hand; the stories wait for the drop, as they are, but the frame is one no
        # story explains. Left uncounted, a new piece set read as pieces off their squares --
        # every frame a "drag" -- kept the record agreeing with a screen it could not read,
        # and the look change was never told (the look-change fixture, 2026-10-05).
        self._agrees = False
        if self._blocked:
            return events + self._maybe_resync(diff)
        stale = (
            {sq for sq in off if self._stale[sq][1] > self.stale_after}
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
                if not (off - ignored) - wild:
                    # A story must rest on a disagreement read cleanly -- a square read
                    # empty or a settled piece of a known kind. When every cell left to
                    # explain is a wildcard, nothing on the board says a move was played:
                    # a board locked 0.28 of a cell off (it had slid on after the lock was
                    # taken) read every piece unsettled, the top of the king on d2 read as
                    # a piece on d3, and the king's move there was told, d2 and d3 both
                    # wildcards (the geometry sweep, 2026-09-27, seed 97).
                    continue
                denied = {sq: obs[sq] for sq in ignored if obs[sq].exact and not obs[sq].junk}
                stories = self._fitting(obs, diff, wild=wild, denied=denied)
                if stories:
                    return events + self._decide(stories, cautious=True)
            if stale == off and len(stale) < self.resync_cells:
                # Only stale garbage differs from the latest position, unless a drag is
                # what kept a real move from committing while its cells turned stale.
                self.viewing = None
                self._unexplained_frames += 1
                return events
        # A piece of a position read from the screen was of another kind; or, last of all,
        # the side to move was told wrong and the other side has moved.
        stories = self._fitting(obs, diff, roots=self._misread_roots(obs, diff))
        if not stories:
            stories = self._fitting(obs, diff, roots=[Story(head=(("flip",),))])
        if stories:
            return events + self._decide(stories, cautious=True)

        # New game: the start position is back on the board.
        if self.in_progress and len(obs) >= 48 and matches(chess.Board(), obs, set(obs)):
            return events + self._restart()

        adopted = self._maybe_position_resync(obs, diff)
        if adopted:
            return events + adopted
        return events + self._maybe_resync(diff)

    def _witness(
        self,
        obs: Observation,
        wild: frozenset[chess.Square],
        quiet: frozenset[chess.Square] = frozenset(),
    ) -> Callable[[chess.Board, chess.Move], bool]:
        """Whether a move is seen played: every square it changes is read -- the square it
        lands on above all -- nothing on them is in motion or under a drawing, and no
        hover outline is anywhere. A dragged piece follows the cursor and an animating one is
        between squares, so neither is centred; a bishop animated across a square read as
        unknown while it passed, and the square it had left explained the move there on its
        own. A king may land on a square read as a piece of no known kind: the king under
        what the site draws on it (`_king_lands`). Wild cells (a stale badge) witness
        anything. While the user drags a piece, only the opponent moves: a hand holding a
        piece plays nothing else, and the opponent's move is seen from its own cells, away
        from the cursor -- a bot answering while the user drags the next premove."""

        def witness(board: chess.Board, move: chess.Move) -> bool:
            if self._dragging and board.turn == self.user_side.color:
                return False
            for sq in touched_squares(board, move):
                if sq in wild:
                    continue
                reading = obs.get(sq)
                if reading is None:
                    if sq == move.to_square:
                        return False  # seen where it landed, or not seen at all
                    continue  # a square left, unread (an arrow over it): no contradiction
                if sq in quiet or reading.hover:
                    return False
                if self._king_lands(board, move, sq, obs) or self._king_leaves(
                    board, move, sq, obs
                ):
                    continue  # the king under what the site draws on it: glow, badge, label
                if sq in self._drawn or reading.unsettled:
                    return False
            return True

        return witness

    def _fitting(
        self,
        obs: Observation,
        diff: set[chess.Square],
        *,
        wild: frozenset[chess.Square] = frozenset(),
        quiet: frozenset[chess.Square] = frozenset(),
        denied: Observation | None = None,
        roots: list[Story] | None = None,
    ) -> list[Story]:
        """The stories that draw this frame (docs/reconciler-model.md, section 5): each story
        kept, and the record itself, extended by the cheapest events that fit; the record
        shown at an earlier ply (browsing); a move played from the ply shown (a takeback).
        `quiet` cells are left out of the matching but witness no move; `wild` cells carry no
        evidence at all, and a move may pass through them (stale badges); `denied` readings
        among them must not be contradicted by a square a story changes; `roots`, when given,
        are the only stories extended."""

        def ignore(sq: chess.Square, reading: CellReading) -> bool:
            return (
                sq in wild
                or sq in quiet
                or sq in self._drawn
                or reading.hover
                or (reading.kind == "piece" and not reading.settled)
            )

        search = Search(
            record=self.board,
            user=self.user_side.color,
            obs=obs,
            ignore=ignore,
            witness=self._witness(obs, wild, quiet),
            lifted=self._dragging,
            liftable=self._liftable(),
            max_plies=self.max_plies,
            max_steps=self.max_queue_steps,
            budget=self.search_budget,
            max_unseen=self._max_unseen,
        )
        if roots is None:
            # The stories kept are what every frame since the record allows: each carries
            # their evidence, and only they are extended. A side to move left open is a story
            # of its own while the record still stands as it was adopted.
            roots = list(self._stories)
            if self._turn_unknown and any(not s.ops and s.view is None for s in roots):
                flipped = self.board.copy(stack=False)
                if _turn_valid(flipped, not self.board.turn):
                    roots.append(Story(head=(("flip",),)))
            found = self._found(roots, search, obs)
            if not found and not all(not s.ops and not s.queue for s in self._stories):
                # No story kept fits: the frames before were misread. The game is told again
                # from the record, with a picture the frames since it do not show.
                found = self._found([Story(cost=1.0)], search, obs)
        else:
            found = self._found(roots, search, obs)
        browsed: list[Story] = []
        if diff and not self._blocked:
            # The move list browsed: an earlier ply of the record, as it stands.
            index = self._history_index(obs, diff) if self.board.move_stack else None
            if index is not None:
                # Browsing changes nothing of the game: every story kept goes on underneath,
                # the ply shown over it.
                for story in self._stories:
                    viewed = replace(story, view=index)
                    if viewed not in browsed:
                        browsed.append(viewed)
            # A move played from the ply shown: a takeback, or, from the start position of a
            # game begun there, a new game. An earlier position of the game is browsing: a
            # different move from the one shown is the only thing a takeback shows.
            for story in self._stories:
                if story.view is None or index is not None:
                    continue
                if story.view == 0 and self._base_fen == chess.STARTING_FEN:
                    head: tuple[tuple[object, ...], ...] = (("newgame",),)
                else:
                    head = (("back", len(self.board.move_stack) - story.view),)
                browsed += [
                    s
                    for s in search.extend(Story(head=head))
                    if s.moves and self._decorations_hold(s, obs)
                ]
        return self._kept(found, browsed, denied)

    def _found(self, roots: list[Story], search: Search, obs: Observation) -> list[Story]:
        """Every way the `roots` go on to draw the frame, the cheapest first."""
        found: list[Story] = []
        seen_roots: set[tuple[object, ...]] = set()
        # The cheapest stories first; each found sets a ceiling on what the rest may cost, so
        # a story that could only end up dearer than the ones kept is not searched through.
        ceiling = float("inf")
        for root in sorted(roots, key=lambda s: s.cost):
            # Browsing goes on as the story underneath it: the live board shown again, or a
            # move played on it (an opponent's while the user browsed).
            story = root if root.view is None else replace(root, view=None)
            if story.cost > ceiling:
                continue
            ident = (story.ops, story.queue)
            if ident in seen_roots:
                continue
            seen_roots.add(ident)
            more = search.extend(story, ceiling)
            found += more
            if more:
                cheapest = min(s.cost for s in found)
                ceiling = min(ceiling, cheapest + (1.0 if cheapest > 0 else 0.0))
        return [s for s in found if self._decorations_hold(s, obs)]

    def _kept(
        self, found: list[Story], browsed: list[Story], denied: Observation | None
    ) -> list[Story]:
        """Of the stories that draw the frame, the ones kept: every one the pictures allow,
        one to each state."""
        found = found + browsed
        if denied:
            found = [s for s in found if self._spares(s, denied)]
        # A piece corrected is one the story moves: its move is what says what it was.
        found = [
            s
            for s in found
            if all(any(m.from_square == op[1] for m in s.moves) for op in s.head if op[0] == "fix")
        ]
        # A side to move read the other way is told by a move played, never by a queue alone:
        # a rook sliding h1-h4 to take the checking queen, caught on h2 and h3, was read as the
        # turn wrong and the rook premoved, and the side to move was flipped (the stress sweep,
        # 2026-10-05, seed 116).
        found = [s for s in found if s.moves or ("flip",) not in s.head]
        if not found:
            return []
        # When the cheapest story needs no picture the frames did not show, the frames showed
        # everything, and a story that needs one did not happen: after e2-e4, "e2-e3, then e4
        # premoved" draws the same pieces from a picture nobody saw. When even the cheapest
        # needs one, frames were missed, and a story needing one more is as possible.
        least = min(s.cost for s in found)
        allowed = [s for s in found if s.cost == least or (least > 0 and s.cost <= least + 1.0)]
        # The move list browsed, or a takeback, against a premove queue drawing the same
        # pieces: the browsing or the takeback is read (Tai, 2026-10-05: "takeback and browse
        # back has to work correctly always; premove can be a bit wrong"). That costs neither
        # the side to move nor the position for long: a queue read as browsing executes when
        # the opponent moves, and the record reads it then as the user's instant reply.
        if any(s in browsed for s in allowed):
            allowed = [s for s in allowed if s in browsed or not s.queue]
        # Likelihood never drops a story (Tai, 2026-10-05: the side to move must never go
        # wrong, and the position must be right once premoves resolve): every story the
        # pictures allow is kept, whatever it assumes, and the record takes only what all of
        # them agree on. Stories that end in one state -- position, side to move, queue, ply
        # shown -- have one future, and are kept as one: the simplest of them, which is the
        # one told when the record follows it.
        allowed.sort(key=_story_order)
        by_state: dict[tuple[object, ...], Story] = {}
        for story in allowed:
            by_state.setdefault(story.key(self.board), story)
        kept = list(by_state.values())
        if len(kept) > MAX_STORIES:
            # More states than are searched each frame: the cap may never cost a side to
            # move -- the first story of each comes before the rest.
            firsts: dict[bool, Story] = {}
            for story in kept:
                firsts.setdefault(story.board(self.board).turn, story)
            heads = list(firsts.values())
            kept = heads + [s for s in kept if s not in heads][: MAX_STORIES - len(heads)]
        return kept

    def _decorations_hold(self, story: Story, obs: Observation) -> bool:
        """Whether every king the story lands on a square read as a piece of no known kind
        (`_king_lands`) is in check when the story ends: a site decorates a king in check,
        and a king moves into no check of its own, so the decoration is the reply's. A story
        that leans on it otherwise is a king walked through cells nobody could read."""
        board = story.board(self.board)
        decorated: set[chess.Square] = set()
        replay = board.copy()
        for move in reversed(story.moves):
            if not replay.move_stack:
                break
            replay.pop()
            if self._king_lands(replay, move, move.to_square, obs):
                decorated.add(move.to_square)
        return self._checked_kings(board, decorated)

    def _spares(self, story: Story, denied: Observation) -> bool:
        """Whether a story leaves alone, or agrees with, every exact reading among the cells
        ignored as stale: a stale empty square is no place to land a queen."""
        after = premove_picture(story.board(self.board), story.queue)
        return all(
            after.piece_at(sq) == self.board.piece_at(sq)
            or cell_agrees(reading, after.piece_at(sq))
            for sq, reading in denied.items()
        )

    def _shown_alike(self, stories: list[Story]) -> Story | None:
        """The story whose position the screen shows, when the stories differ only in what no
        picture can tell apart: each leaves the same side to move and draws the same board,
        and the first with no premove queue (and no ply browsed) has the board shown as its
        position. None otherwise."""
        if any(s.view is not None for s in stories):
            return None
        if len({s.board(self.board).turn for s in stories}) != 1:
            return None
        drawn = {premove_picture(s.board(self.board), s.queue).board_fen() for s in stories}
        if len(drawn) != 1:
            return None
        return next((s for s in stories if not s.queue), None)

    def _decide(self, stories: list[Story], cautious: bool = False) -> list[Event]:
        """What the record does with the stories that fit (docs/reconciler-model.md, section
        5, steps 3 and 4). When every story leads to one key -- one position, one side to
        move, one queue, one ply shown -- they differ only in how they got there, and the
        cheapest is the record's to follow. Otherwise the record follows only the ops every
        story begins with. An op is played once it has held `settle_frames` frames, one
        more when it came with others in one frame or from the stale retry. A wrong side to
        move is never told: while stories disagree on it, nothing past their common start is
        played."""
        if len(stories) == 1 and stories[0].cost:
            stories = [replace(stories[0], cost=0.0)]  # alone, its past is the past
        self._stories = stories
        told = self._tell_turn()
        self._unexplained_frames = 0
        self._unexplained_noted = False
        self._agrees = any(not s.ops and s.view is None for s in stories)
        self._queue_ts = self._ts if any(s.queue for s in stories) else None
        if all(s.view is not None for s in stories):
            # The user browses: the ply shown, and nothing played until the live board is
            # shown again -- the stories underneath wait as they are.
            self._agreed = []
            if not self._blocked:
                self.viewing = stories[0].view
            return told
        keys = {s.key(self.board) for s in stories}
        # One key is one future: the position, the side to move, the queue and the ply shown
        # are certain, and the record follows the simplest story to them (Tai, 2026-10-05:
        # the side to move first, then the position current, then the moves themselves).
        if len(keys) == 1:
            chosen = stories[0]
            ops = chosen.ops
            if not ops:
                self._agreed = []
                if chosen.view is None or not self._blocked:
                    self.viewing = chosen.view
                return told
        else:
            ops = _common_start([s.ops for s in stories])
            lead = self._shown_alike(stories)
            if lead is not None:
                # Every story has the side to move of the others and draws the same board,
                # and one of them has no premove queue: the board shown is its position. The
                # record follows it at once (Tai, 2026-10-05: the position current, the side
                # to move exact); the others stay, retold from it (`_play`), and the moves are
                # what may be put right later.
                ops = lead.ops
            if not ops:
                # Nothing all stories agree on yet. What is shown meanwhile is the simplest
                # story's ply: browsing the move list before a premove that draws the same
                # picture (Tai, 2026-10-05: a takeback and browsing are always read right). It
                # is shown only, never played: the record waits for agreement.
                self._agreed = []
                if stories[0].view is None or not self._blocked:
                    self.viewing = stories[0].view
                return told
        self.viewing = None
        # A frame confirms a move only when the squares it changed are seen settled in it: a
        # frame with a piece in motion keeps the stories, and confirms nothing it cannot see
        # (the bishop caught on c5 sliding d6-b4 was told as Bc5 on the slide's next frame:
        # the stress sweep, 2026-10-05, seed 1).
        seen = self._seen_ops(ops)
        agreed: list[tuple[object, int, int]] = []
        for i, op in enumerate(ops):
            if i >= len(self._agreed) or self._agreed[i][0] != op:
                break
            _, count, need = self._agreed[i]
            agreed.append((op, count + seen[i], need))
        fresh = ops[len(agreed) :]
        plies = sum(isinstance(op, chess.Move) for op in fresh)
        need = self.settle_frames + (1 if plies > 1 else 0) + (1 if cautious else 0)
        agreed += [(op, seen[len(agreed) + i], need) for i, op in enumerate(fresh)]
        self._agreed = agreed
        ready = 0
        while ready < len(agreed) and agreed[ready][1] >= agreed[ready][2]:
            ready += 1
        if not ready:
            return told
        events = self._play(ops[:ready])
        return told + events + self._tell_turn()

    def _seen_ops(self, ops: tuple[object, ...]) -> list[int]:
        """For each op, 1 when this frame shows it settled: a move whose squares are all read,
        none in motion, under the cursor or under a drawing (a king under its decoration
        aside); what is done to the record before the moves always. Nor does a frame that shows
        the moved piece off its square further along the move's line (`_sliding_on`)."""
        obs = self._last_obs
        head = tuple(op for op in ops if not isinstance(op, chess.Move))
        board = Story(head=head).board(self.board).copy(stack=False)  # type: ignore[arg-type]
        out: list[int] = []
        for op in ops:
            if not isinstance(op, chess.Move):
                out.append(1)
                continue
            clean = all(
                self._settled_on(board, op, sq, obs) for sq in touched_squares(board, op)
            ) and not self._sliding_on(board, op, obs)
            out.append(1 if clean else 0)
            if op in board.legal_moves:
                board.push(op)
        return out

    @staticmethod
    def _sliding_on(board: chess.Board, move: chess.Move, obs: Observation) -> bool:
        """Whether the frame reads the moved piece off its square further along the move's
        line: still sliding on, the square it seems to have stopped on a frame of its way.
        A pawn sliding a7-a5 was caught on a6, told as a6 on the next frame with the pawn
        still on its way, and a5 then read as a premove nobody played (the stress sweep,
        2026-10-05, seed 69)."""
        piece = board.piece_at(move.from_square)
        if piece is None:
            return False
        files = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        ranks = chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        if files and ranks and abs(files) != abs(ranks):
            return False  # a knight's jump: no line to go on along
        step_file = (files > 0) - (files < 0)
        step_rank = (ranks > 0) - (ranks < 0)
        file = chess.square_file(move.to_square) + step_file
        rank = chess.square_rank(move.to_square) + step_rank
        while 0 <= file < 8 and 0 <= rank < 8:
            reading = obs.get(chess.square(file, rank))
            if (
                reading is not None
                and reading.kind == "piece"
                and not reading.settled
                and reading.symbol == piece.symbol()
            ):
                return True
            file += step_file
            rank += step_rank
        return False

    def _settled_on(
        self, board: chess.Board, move: chess.Move, sq: chess.Square, obs: Observation
    ) -> bool:
        """Whether `sq`, a square `move` changes, is read and nothing on it is moving, under
        the cursor or under a drawing -- a king under its decoration aside."""
        reading = obs.get(sq)
        if reading is None:
            return sq != move.to_square  # a square left may go unread; the landing may not
        if reading.hover:
            return False
        if self._king_lands(board, move, sq, obs) or self._king_leaves(board, move, sq, obs):
            return True
        return sq not in self._drawn and not reading.unsettled

    def _tell_turn(self) -> list[Event]:
        """`TurnPending` when whether the stories agree on the side to move has changed."""
        pending = self.turn_pending
        if pending == self._turn_pending_told:
            return []
        self._turn_pending_told = pending
        return [TurnPending(game_id=self.game_id, pending=pending)]

    def _play(self, ops: tuple[object, ...]) -> list[Event]:
        """The record does `ops`, and every story is told from it."""
        stories, agreed = self._stories, self._agreed
        events: list[Event] = []
        for op in ops:
            if isinstance(op, chess.Move):
                # The first move played settles a side to move left open: who played it.
                self._turn_unknown = False
                events += self._commit(op)
                continue
            assert isinstance(op, tuple)
            if op[0] == "newgame":
                events += self._restart()
            elif op[0] == "back":
                plies = op[1]
                assert isinstance(plies, int)
                for _ in range(plies):
                    self.board.pop()
                events += [
                    Takeback(game_id=self.game_id, plies=plies, fen=self.board.fen()),
                    PositionChanged(
                        game_id=self.game_id, fen=self.board.fen(), ply=len(self.board.move_stack)
                    ),
                ]
            elif op[0] == "flip":
                events += self._flip()
            elif op[0] == "fix":
                square, kind = op[1], op[2]
                assert isinstance(square, int) and isinstance(kind, int)
                events += self._fix(square, kind)
        done = len(ops)
        kept = []
        for story in stories:
            if story.ops[:done] == ops:
                kept.append(story.rebased(done))
                continue
            retold = _retold(story, ops)
            if retold is not None:
                kept.append(retold)
        self._stories = kept or [Story()]
        self._agreed = agreed[done:]
        self._stale = {}
        self.viewing = None
        return events

    def _flip(self) -> list[Event]:
        """The record's side to move read the other way: the first move of a position adopted
        with the turn open settles it, silently; a turn told wrong is told again."""
        board = self.board.copy(stack=False)
        board.turn = not board.turn
        board.ep_square = None
        if self._turn_unknown:
            self.board.turn = board.turn
            self.board.ep_square = None
            self._base_fen = self.board.fen()
            self._turn_unknown = False
            return []
        read, since = self._read_squares, self._last_commit_ts
        self._set_position(board)
        self._read_squares, self._last_commit_ts = read, since
        return self._resynced(f"side to move corrected: {Side.of(board.turn).value} to move")

    def _fix(self, square: chess.Square, kind: chess.PieceType) -> list[Event]:
        """A piece of a position read from the screen, never moved since, was of another kind:
        the position is adopted again with it corrected. Still the same game: the pieces that
        had moved stay known, and the next move's think time counts from the last move."""
        was = self.board.piece_at(square)
        assert was is not None
        board = self.board.copy(stack=False)
        board.set_piece_at(square, chess.Piece(kind, was.color))
        read, since = self._read_squares - {square}, self._last_commit_ts
        self._set_position(board)
        self._read_squares, self._last_commit_ts = read, since
        return self._resynced(
            f"the {chess.piece_name(was.piece_type)} read on {chess.square_name(square)} "
            f"was a {chess.piece_name(kind)}"
        )

    def _after_the_end(self, obs: Observation, diff: set[chess.Square]) -> list[Event]:
        """After the game has ended the board is a review board: browsing is followed, and
        moves played on it are analysis, not the game. Only the start position begins the
        next game, or a position far enough away to be adopted by a resync."""
        if not diff:
            self.viewing = None
            return []
        index = self._history_index(obs, diff) if self.board.move_stack else None
        if index is not None:
            if index == 0 and self._base_fen == chess.STARTING_FEN:
                # The start position after a finished game is the next game, not browsing.
                return self.start()
            if not self._blocked:
                self.viewing = index
            return []
        if self.in_progress and len(obs) >= 48 and matches(chess.Board(), obs, set(obs)):
            return self._restart()
        return self._maybe_position_resync(obs, diff)

    @property
    def pending_story(self) -> bool:
        """Whether the screen is ahead of the record: a story with moves still to be told."""
        return any(s.ops for s in self._stories)

    @property
    def agrees(self) -> bool:
        """Whether the last readable frame shows the record as it stands, a premove queue
        drawn on it aside: the record is up to date with the screen."""
        return self._agrees

    @property
    def turn_pending(self) -> bool:
        """Whether the stories disagree on the side to move: the record is behind the screen
        and cannot yet say whose move it is."""
        turns = {s.board(self.board).turn for s in self._stories}
        return len(turns) > 1

    def pending_terminal(self, result: str | None = None) -> tuple[Explanation, chess.Board] | None:
        """The story that ends the game, when the end is all the stories leave: the mate the
        site's dialog or badges then covered before a clean frame could confirm it. The end
        shown is evidence as good as a frame: a story that does not end the game -- with
        `result`, when the site shows one -- is ruled out by it (a premove queue still drawn
        is not a game over; the site clears it). Every story left must end in one state."""
        ended = []
        for story in self._stories:
            board = story.board(self.board)
            if story.head or not story.moves or story.queue or not board.is_game_over():
                continue
            if result is not None and board.result() != result:
                continue
            ended.append((story, board))
        if not ended or len({s.key(self.board) for s, _ in ended}) != 1:
            return None
        story, board = ended[0]
        return story.moves, board

    def commit_pending_terminal(self, result: str | None = None) -> list[Event]:
        """Commit the story that ends the game (see `pending_terminal`)."""
        story = self.pending_terminal(result)
        if story is None:
            return []
        self._stories = [Story(moves=story[0])]
        self._agreed = []
        return self._play(story[0])

    def _misread_roots(self, obs: Observation, diff: set[chess.Square]) -> list[Story]:
        """Stories in which one piece of a position read from the screen, never moved since,
        is of another kind: the piece has left its square, so the frame cannot say what it
        was, and the move it made says. Without this the game stood on a piece that could not
        make the move it made, and every move after it went unread until the board was
        adopted afresh (Tai's puzzle, 2026-10-03: a knight on h4 read as a bishop; Nxg6, hxg6
        and Bf3 lost)."""
        if self.viewing is not None or self._turn_unknown or len(diff) > self.misread_max_cells:
            return []
        roots: list[Story] = []
        for square in sorted(diff & self._read_squares):
            piece = self.board.piece_at(square)
            reading = obs.get(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
            if reading is None or reading.kind != "empty":
                continue
            for kind in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                if kind == piece.piece_type:
                    continue
                if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
                    continue
                trial = self.board.copy(stack=False)
                trial.set_piece_at(square, chess.Piece(kind, piece.color))
                if _turn_valid(trial, trial.turn):
                    roots.append(Story(head=(("fix", square, kind),)))
        return roots

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

    def _ghosts(
        self, off: set[chess.Square], obs: Observation, picture: chess.Board
    ) -> set[chess.Square]:
        """Stale readings of something unnameable on squares the picture has empty."""
        return {
            sq
            for sq in off
            if self._stale[sq][1] > self.stale_after
            and obs[sq].kind == "piece"
            and obs[sq].symbol is None
            and picture.piece_at(sq) is None
        }

    def _liftable(self) -> frozenset[chess.Square]:
        """Where the user's pieces stood before this frame, as the likeliest story draws them:
        the squares a piece in the hand can have come from."""
        lead = self._lead_picture()
        return frozenset(lead.pieces(chess.PAWN, self.user_side.color)).union(
            *(lead.pieces(kind, self.user_side.color) for kind in range(2, 7))
        )

    def _lead_picture(self) -> chess.Board:
        """What the likeliest story draws: its position with its queue played out, or the
        ply of the record it shows."""
        story = self._stories[0] if self._stories else Story()
        if story.view is not None:
            return self._board_at(story.view)
        return premove_picture(story.board(self.board), story.queue)

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
        if self._queue_ts is not None and self._ts - self._queue_ts < QUEUE_RESYNC_S:
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
        if frames >= self.brief_frames:
            # Remembered as a board shown: the start a move may have been played from.
            seen = (loose_placement(obs), highlighted_squares(obs), placement, self._ts)
            if self._resync_seen and self._resync_seen[-1][2] == placement:
                self._resync_seen[-1] = seen
            else:
                self._resync_seen = [*self._resync_seen[-(self.brief_kept - 1) :], seen]
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
        first = self._opening_move(board, placement, since, highlighted_squares(obs))
        self._resync_seen = []
        self._clear_story()
        self.viewing = None
        self._resync_candidate = None
        if first is not None:
            # The brief board before it is the start, and the move between them is played.
            start, move = first
            self._turn_unknown = False
            if self._finished:
                events = [self._begin(start, "position"), *self._positioned()]
            else:
                self._set_position(start)
                events = self._resynced(
                    f"position adopted after {len(diff)} unexplained cells; its first move seen"
                )
            return events + self._commit(move)
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

    def _opening_move(
        self, after: chess.Board, placement: str, since: float, lit: frozenset[chess.Square]
    ) -> tuple[chess.Board, chess.Move] | None:
        """The board shown just before the one being adopted, and the one legal move from it
        to this one -- a new puzzle's first board and the site's opening move on it -- or
        None. The earlier board must have held `brief_frames` clean frames, ended at most
        `brief_recent_s` before this one began, and be one move away: anything else is a
        jump no single move explains, and this board stands alone. The newest of the boards
        kept that one move joins is taken: a frame caught mid-slide, repeated by the phone's
        still ticks, was held long enough to be kept, and as the only earlier board it stood in
        for the first one (Tai's dump 20261003-171356)."""
        for prior in reversed(self._resync_seen):
            if prior[2] == placement or since - prior[3] > self.brief_recent_s:
                continue
            found = self._one_move(prior[0], prior[1], after.board_fen(), lit)
            if found is not None:
                return found
        return None

    def _one_move(
        self,
        placed: chess.Board,
        highlighted: frozenset[chess.Square],
        target: str,
        lit: frozenset[chess.Square] = frozenset(),
    ) -> tuple[chess.Board, chess.Move] | None:
        """The position `placed` shows and the one legal move from it to the placement
        `target`, when exactly one move by either side gets there."""
        return Reconciler.one_move(placed, highlighted, target, prefer=self.bottom.color, lit=lit)

    @staticmethod
    def one_move(
        placed: chess.Board,
        highlighted: frozenset[chess.Square],
        target: str,
        prefer: chess.Color = chess.WHITE,
        lit: frozenset[chess.Square] = frozenset(),
    ) -> tuple[chess.Board, chess.Move] | None:
        """`_one_move` for any caller: the pipeline asks it which of the boards it read before
        is the one a move joins to the board it adopts. `lit`: the squares the site lights on
        the board after the move, its last move's -- when two are lit, the move must be theirs.
        Halfway through a pawn's double step the pawn stands centred on the square between, a
        board of its own, and e6-e5 joined it to the board after e7-e5 (the puzzle sweep,
        2026-10-03): e7 and e5 lit say it is not the move."""
        adopted = Reconciler._adopt_placement(placed, prefer=prefer, highlighted=highlighted)
        if adopted is None:
            return None
        found: dict[tuple[chess.Color, chess.Move], chess.Board] = {}
        for turn in (chess.WHITE, chess.BLACK):
            base = adopted[0].copy(stack=False)
            base.turn = turn
            if not _turn_valid(base, turn):
                continue
            # A board read from the screen cannot say en passant is open; each square it could
            # be open on is tried as well (d5xe6 e.p. as a puzzle's opening move was no move at
            # all: the puzzle sweep, 2026-10-03).
            for ep in (None, *_en_passant_squares(base)):
                start = base.copy(stack=False)
                start.ep_square = ep
                for move in start.legal_moves:
                    if len(lit) == 2 and not _lights(start, move, lit):
                        continue
                    start.push(move)
                    hit = start.board_fen() == target
                    start.pop()
                    if hit:
                        found.setdefault((turn, move), start.copy(stack=False))
        # Exactly one move, or the jump is not read as one.
        if len(found) != 1:
            return None
        (_, move), start = next(iter(found.items()))
        return start, move

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
        before: tuple[chess.Board, frozenset[chess.Square]] | None = None,
    ) -> list[Event]:
        """Begin a game from an arbitrary observed position -- a puzzle or a mid-game board read
        cold from the theme cache -- rather than from the start. Rights and turn are assumed the
        same way a position resync assumes them; the game continues normally from here. `ts`
        is when the position was read, so the first move's think time counts from it (it
        counted from zero: a first move that "took 412630 seconds").

        `before`: a placement read moments earlier, and its highlighted squares. When exactly
        one legal move takes it to this one -- a new puzzle's first board and the site's
        opening move, played before the read could confirm the first -- the game begins from
        it and the move is played (`_one_move`)."""
        adopted = self._adopt_placement(
            placement, prefer=self.bottom.color, highlighted=highlighted
        )
        if adopted is None:
            return []
        if ts is not None:
            self._ts = ts
        if before is not None:
            first = self._one_move(before[0], before[1], placement.board_fen(), highlighted)
            if first is not None:
                start, move = first
                self._turn_unknown = False
                return [self._begin(start, "position"), *self._commit(move)]
        board, self._turn_unknown = adopted
        return [self._begin(board, "position")]

    # Settling and committing

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
            self._unexplained_noted = False
            return []
        self._unexplained_frames += 1
        if self._unexplained_frames >= self.unexplained_note_frames and not self._unexplained_noted:
            self._unexplained_noted = True
            # Early, for the log: what the reader sees on each square the game cannot explain,
            # while it is still fresh -- a dump's log said only the squares' names (Tai's
            # puzzle on the Pixel, 2026-10-03: 34 squares, and why was nowhere).
            return [
                Unexplained(
                    game_id=self.game_id,
                    fen=self.board.fen(),
                    reason=f"unexplained for now: {self._describe(sorted(diff))}",
                )
            ]
        persistent = [
            sq for sq in diff if sq in self._stale and self._stale[sq][1] > self.stale_after
        ]
        if len(persistent) < self.resync_cells or self._unexplained_frames < self.resync_after:
            return []
        self._unexplained_frames = 0
        return [
            Unexplained(
                game_id=self.game_id,
                fen=self.board.fen(),
                reason=f"unexplained: {self._describe(sorted(persistent))}",
                told=True,
            )
        ]

    def _describe(self, squares: list[chess.Square]) -> str:
        """Each square as read against the game: "b1 N off its square, game N", "e4 empty,
        game P", "d5 piece of no kind, game -"."""
        parts = []
        for sq in squares:
            reading = self._last_obs.get(sq)
            piece = self.board.piece_at(sq)
            if reading is None:
                read = "unread"
            elif reading.kind == "empty":
                read = "empty"
            elif reading.kind != "piece":
                read = "unknown"
            else:
                read = reading.symbol or "piece of no kind"
                if not reading.settled:
                    read += " off its square"
            game = piece.symbol() if piece is not None else "-"
            parts.append(f"{chess.square_name(sq)} {read}, game {game}")
        return "; ".join(parts)

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
            return []
        if move not in self.board.legal_moves:
            # Defensive: an explanation computed against another position must never
            # corrupt the game. Report it and let the next frames explain the board again.
            self._clear_story()
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
        self._read_squares -= {move.from_square, move.to_square}
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
        self._clear_story()
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
