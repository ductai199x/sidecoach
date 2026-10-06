"""Stories: the ways the game may have gone on past the record, kept side by side.

docs/reconciler-model.md is the design. The screen is drawn from a hidden state: the game,
the user's premove queue and the ply shown. A story is one such state past the record --
the moves played since, the queue as it stands, the ply shown -- and what it costs to
believe. Every frame each story is extended by the cheapest events that make it draw what
the frame shows; the record takes only what every story agrees on. Nothing here reads a
colour: a story is checked against the pieces on the squares, and nothing else.

How a state is drawn (section 3): a site that draws a premove queue (chess.com) plays it out
over the position, each step lifting the piece and putting it down over whatever stood
there (`premove_picture`); a site that does not (lichess) shows the position, and its queue
is then no different from the user's moves played the instant the opponent's lands.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import chess

from sidecoach.vision.pieces import CellReading

Observation = dict[chess.Square, CellReading]
# Positions the premove search may try for one way the queue stood (`Search._edits`).
EDIT_BUDGET = 300


def premove_picture(board: chess.Board, queue: Sequence[chess.Move]) -> chess.Board:
    """The pieces a site that draws premoves shows while `queue` waits: the position with
    every step applied in queue order, each lifting the piece off its square and putting it
    down on the step's square over whatever stood there, of either side (chess.com's board
    bundle, `v`, read 2026-10-05). A step whose square no longer holds a piece of the queue's
    side is skipped (the opponent took it); a promotion puts down the promoted piece;
    castling puts the king on g/c and the rook on f/d."""
    if not queue:
        return board
    shown = board.copy(stack=False)
    side: chess.Color | None = None
    for move in queue:
        piece = shown.piece_at(move.from_square)
        if side is None:
            first = board.piece_at(move.from_square)
            side = first.color if first is not None else None
        if piece is None or piece.color != side:
            continue
        shown.remove_piece_at(move.from_square)
        rank = chess.square_rank(move.from_square)
        files = chess.square_file(move.to_square) - chess.square_file(move.from_square)
        if piece.piece_type == chess.KING and abs(files) == 2:
            rook_from, rook_to = (7, 5) if files > 0 else (0, 3)
            rook = shown.remove_piece_at(chess.square(rook_from, rank))
            if rook is not None:
                shown.set_piece_at(chess.square(rook_to, rank), rook)
        if move.promotion:
            piece = chess.Piece(move.promotion, piece.color)
        shown.set_piece_at(move.to_square, piece)
    return shown


def lone_moves(board: chess.Board, square: chess.Square) -> tuple[chess.Move, ...]:
    """The premoves a site allows the piece on `square`: its moves alone on an empty board --
    through anything, onto anything -- a pawn's forward diagonals always (it may premove a
    capture of a piece not there yet), and a king's castling the position still allows
    (chess.com's generator `_`; lichess's chessground premoves the same way)."""
    piece = board.piece_at(square)
    if piece is None:
        return ()
    rights = board.castling_rights if piece.piece_type == chess.KING else 0
    return _lone_moves(piece, square, rights)


@functools.lru_cache(maxsize=8192)
def _lone_moves(piece: chess.Piece, square: chess.Square, rights: int) -> tuple[chess.Move, ...]:
    """`lone_moves`, once for each piece, square and castling rights: they depend on nothing
    else, and the premove search asks for them a few hundred times a frame."""
    lone = chess.Board(None)
    lone.turn = piece.color
    lone.set_piece_at(square, piece)
    if piece.piece_type == chess.PAWN:
        rank = chess.square_rank(square) + (1 if piece.color == chess.WHITE else -1)
        for file in (chess.square_file(square) - 1, chess.square_file(square) + 1):
            if 0 <= file < 8 and 0 <= rank < 8:
                lone.set_piece_at(
                    chess.square(file, rank), chess.Piece(chess.KNIGHT, not piece.color)
                )
    if piece.piece_type == chess.KING:
        home = 0 if piece.color == chess.WHITE else 7
        for file in (7, 0):
            rook = chess.square(file, home)
            if square == chess.square(4, home) and rights & chess.BB_SQUARES[rook]:
                lone.set_piece_at(rook, chess.Piece(chess.ROOK, piece.color))
                lone.castling_rights |= chess.BB_SQUARES[rook]
    # A premoved promotion is to a queen, the sites' own choice.
    return tuple(
        move
        for move in lone.generate_pseudo_legal_moves(chess.BB_SQUARES[square])
        if move.promotion in (None, chess.QUEEN)
    )


@dataclass(frozen=True)
class Story:
    """One way the game may have gone on past the record.

    `head` is what was done to the record before anything was played on it, in order:
    `("newgame",)` (a new game from the start position), `("fix", square, kind)` (a piece
    of a position read from the screen was of another kind), `("back", plies)` (a takeback,
    seen as a move from an earlier position), `("flip",)` (the record's side to move read
    the other way: a position adopted with the turn open, or a turn told wrong). Then
    `moves` were played. `queue` is the user's premove queue as it stands, which exists only
    while the opponent is to move; `view` is the ply of the record the board shows while the
    user browses the move list; `cost` is how many pictures the story needs that no frame
    showed. Nothing else is weighed: of the stories the pictures allow, none is likelier."""

    moves: tuple[chess.Move, ...] = ()
    queue: tuple[chess.Move, ...] = ()
    head: tuple[tuple[object, ...], ...] = ()
    view: int | None = None
    cost: float = 0.0
    # Whether the story holds an opponent's move no picture showed (`Search._unseen_replies`).
    hidden: bool = False

    @property
    def ops(self) -> tuple[object, ...]:
        """What the record must do to become this story's game, in order."""
        return self.head + self.moves

    def board(self, record: chess.Board) -> chess.Board:
        """The position this story's game stands on."""
        board = record
        for op in self.head:
            if op[0] == "newgame":
                board = chess.Board()
            elif op[0] == "fix":
                board = board.copy()
                square, kind = op[1], op[2]
                piece = board.piece_at(square)  # type: ignore[arg-type]
                assert piece is not None
                board.set_piece_at(square, chess.Piece(kind, piece.color))  # type: ignore[arg-type]
            elif op[0] == "back":
                board = board.copy()
                for _ in range(op[1]):  # type: ignore[arg-type]
                    board.pop()
            elif op[0] == "flip":
                board = board.copy(stack=False)
                board.turn = not board.turn
                board.ep_square = None
        if self.moves:
            board = board.copy()
            for move in self.moves:
                board.push(move)
        return board

    def key(self, record: chess.Board) -> tuple[object, ...]:
        """What the screen will go on from: the position, its side to move, the queue and the
        ply shown. Stories with one key differ only in how they got there."""
        board = self.board(record)
        return (board.board_fen(), board.turn, board.castling_rights, self.queue, self.view)

    def rebased(self, done: int) -> Story:
        """This story told from a record that has done its first `done` ops."""
        head = self.head[done:]
        moves = self.moves[max(0, done - len(self.head)) :]
        return replace(self, head=head, moves=moves)


def disagreeing(
    picture: chess.Board, obs: Observation, ignore: Callable[[chess.Square, CellReading], bool]
) -> set[chess.Square]:
    """The squares whose reading contradicts `picture`, skipping those `ignore` says carry no
    evidence (a piece in motion, a stale badge, an arrow)."""
    out = set()
    for sq, reading in obs.items():
        if reading.kind == "unknown" or ignore(sq, reading):
            continue
        piece = picture.piece_at(sq)
        if reading.kind == "empty":
            if piece is not None:
                out.add(sq)
        elif piece is None or (reading.symbol is not None and piece.symbol() != reading.symbol):
            out.add(sq)
    return out


@dataclass(frozen=True)
class Evidence:
    """What a frame reads, as bitboards, for comparing pictures fast (a story's search compares
    hundreds a frame): the squares read empty, the squares read occupied, and for each piece
    read by its kind the squares it is read on. Cells that carry no evidence are in none."""

    empty: int
    occupied: int
    kinds: tuple[tuple[chess.PieceType, chess.Color, int], ...]

    @staticmethod
    def of(obs: Observation, ignore: Callable[[chess.Square, CellReading], bool]) -> Evidence:
        empty = occupied = 0
        kinds: dict[tuple[chess.PieceType, chess.Color], int] = {}
        for sq, reading in obs.items():
            if reading.kind == "unknown" or ignore(sq, reading):
                continue
            mask = chess.BB_SQUARES[sq]
            if reading.kind == "empty":
                empty |= mask
                continue
            occupied |= mask
            if reading.symbol is not None:
                piece = chess.Piece.from_symbol(reading.symbol)
                kinds[(piece.piece_type, piece.color)] = (
                    kinds.get((piece.piece_type, piece.color), 0) | mask
                )
        return Evidence(empty, occupied, tuple((t, c, m) for (t, c), m in kinds.items()))

    def wrong(self, picture: chess.Board) -> int:
        """The squares the frame reads otherwise than `picture`, as a bitboard (`disagreeing`)."""
        out = (picture.occupied & self.empty) | (self.occupied & ~picture.occupied)
        for kind, colour, mask in self.kinds:
            out |= mask & ~picture.pieces_mask(kind, colour)
        return out


@dataclass
class Search:
    """The cheapest stories that make one story draw the frame (section 5, step 1).

    From the story's position: up to `max_plies` moves, the sides in turn, each touching a
    square the picture gets wrong or one a queue hides, and each witnessed (`witness`: every
    square it changes is read, and nothing on them is in motion); every opponent's move
    plays the queue's first step at once if it is legal and clears the queue if not
    (chess.com's `consume`). While the opponent is to move, the user may also have cancelled
    the queue, or queued up to `max_steps` more steps from where their pieces are drawn, or
    both. What a way costs is the pictures it needs that the frames never showed: each move
    (an opponent's with the queued step it sets off, drawn in one picture) and each queue
    edit makes a picture, and only the last of a frame's is seen (`unseen`). Ways more than
    `margin` dearer than the cheapest are dropped. `budget` bounds the positions expanded.
    `lifted`: a drag is under way, and one of the user's pieces may be off its square in the
    user's hand."""

    record: chess.Board
    user: chess.Color
    obs: Observation
    ignore: Callable[[chess.Square, CellReading], bool]
    witness: Callable[[chess.Board, chess.Move], bool]
    lifted: bool = False
    # The squares a piece in the user's hand may have been lifted from: where the user's
    # pieces stood before this frame. A story never excuses a piece it moved itself.
    liftable: frozenset[chess.Square] = frozenset()
    max_plies: int = 4
    max_steps: int = 3
    budget: int = 4000
    margin: float = 1.0
    # Pictures this frame's events may need that neither frame showed.
    max_unseen: int = 1

    def _unread(self, square: chess.Square) -> bool:
        """Whether the frame says nothing of `square`."""
        reading = self.obs.get(square)
        return reading is None or reading.kind == "unknown" or self.ignore(square, reading)

    def __post_init__(self) -> None:
        self._evidence = Evidence.of(self.obs, self.ignore)

    def wrong(self, picture: chess.Board) -> set[chess.Square]:
        """The squares the frame reads otherwise than `picture`; during a drag, less one of
        the user's pieces read gone from its square (the one in the user's hand)."""
        mask = self._evidence.wrong(picture)
        if not mask:
            return set()
        out = set(chess.SquareSet(mask))
        if self.lifted:
            for sq in sorted(out):
                piece = picture.piece_at(sq)
                if (
                    sq in self.liftable
                    and piece is not None
                    and piece.color == self.user
                    and self.obs[sq].kind == "empty"
                ):
                    out.discard(sq)
                    break
        return out

    def extend(self, story: Story, ceiling: float = float("inf")) -> list[Story]:
        """The ways `story` goes on to draw the frame, none dearer than `ceiling`: what the
        stories already extended this frame have left worth looking for."""
        found: list[Story] = []
        best = [float("inf")]
        budget = [self.budget]

        def allowed() -> float:
            # A story dearer than the cheapest is kept only once even the cheapest needs a
            # picture the frames did not show (`Reconciler._fitting`).
            return min(ceiling, best[0] + (self.margin if best[0] > story.cost else 0.0))

        def add(new: Story) -> None:
            if new.cost <= allowed() or new.cost < best[0]:
                found.append(new)
                best[0] = min(best[0], new.cost)

        def node(
            board: chess.Board,
            queue: tuple[chess.Move, ...],
            moves: tuple[chess.Move, ...],
            events: int,
            plies: int,
            limit: int,
            edits: bool,
            blind: bool = False,
            paired: bool = False,
        ) -> None:
            cost = story.cost + unseen(events)
            if cost > allowed() or unseen(events) > self.max_unseen or budget[0] <= 0:
                return
            budget[0] -= 1
            wrong = self.wrong(premove_picture(board, queue))
            if not wrong:
                add(replace(story, moves=moves, queue=queue, view=None, cost=cost))
                return
            if edits and board.turn != self.user and plies == limit:
                # Once for each position, at the deepest level of the pass that reaches it.
                self._edits(story, board, queue, moves, events, add, budget)
            # A move changes four squares at most (castling), and a queue hides its own.
            hidden = {sq for step in queue for sq in (step.from_square, step.to_square)}
            if plies >= limit or len(wrong - hidden) > 4 * (limit - plies):
                return
            reach = wrong | hidden
            children: list[tuple[int, int, tuple[object, ...]]] = []
            for move in board.legal_moves:
                touched = _touches(board, move)
                # One move of a frame may live on cells nobody can read (a king under the
                # mate's badge, its two squares spoiled); a second is invented to fill a turn.
                unread = (
                    not touched & reach and not blind and all(self._unread(sq) for sq in touched)
                )
                if not (touched & reach or unread) or not self.witness(board, move):
                    continue
                if (
                    plies >= 2
                    and move.from_square == moves[-2].to_square
                    and move.to_square == moves[-2].from_square
                ):
                    continue  # there and straight back: no trace, no story
                after = board.copy(stack=False)
                after.push(move)
                told = moves + (move,)
                rest = queue
                if board.turn != self.user and queue:
                    step = queue[0]
                    if step in after.legal_moves:
                        after.push(step)
                        told += (step,)
                        rest = queue[1:]
                    else:
                        rest = ()
                # A move the instant the other side's lands is drawn in one picture with it:
                # the user's is a premove played, on a site that draws the queue or one that
                # does not; the opponent's is a puzzle's reply, a bot's, a premove of theirs.
                # Two to a picture: a reply to a reply is a picture of its own.
                instant = plies > 0 and not paired
                left = len(self.wrong(premove_picture(after, rest)))
                args = (
                    after,
                    rest,
                    told,
                    events + (0 if instant else 1),
                    plies + 1,
                    limit,
                    edits,
                    blind or unread,
                    instant,
                )
                children.append((left, len(children), args))
            # The moves that leave the least wrong first: a budget spent on hopeless moves
            # (a queen tried on every square before the one it went to, each with premoves to
            # explain the rest) once ran out before the move that was played.
            for _, _, args in sorted(children, key=lambda child: child[:2]):
                node(*args)  # type: ignore[arg-type]

        # Shortest first: every story of each length is searched before any longer one, so
        # a budget spent deep can never hide a simple one; and every length is searched, up
        # to the plies a frame can hold, since a longer story that needs no more pictures
        # nobody saw is as possible as a shorter one. At each length the moves alone are
        # tried, then with the queue edited.
        start = story.board(self.record)
        for limit in range(self.max_plies + 1):
            node(start, story.queue, story.moves, 0, 0, limit, False)
            node(start, story.queue, story.moves, 0, 0, limit, True)
        if (
            not story.hidden
            and story.queue
            and start.turn != self.user
            and not self.wrong(premove_picture(start, story.queue))
        ):
            found += self._unseen_replies(story, start)
        return [s for s in found if s.cost <= allowed()]

    def _unseen_replies(self, story: Story, start: chess.Board) -> list[Story]:
        """The opponent's moves the picture cannot show: a queue drawn over the squares a
        move leaves and lands on hides it whole, and the queue's step it sets off as well. A
        bot's pawn went c4-c5 under a king queued d3-c4-c5, nothing changed on the screen,
        and the game stayed a step behind the site for good (the premove sweep, 2026-10-05,
        seed 205). Each such move is kept as a story beside the one without it, as likely,
        one to a line of stories: the moves after tell them apart (`Reconciler._fitting`)."""
        hidden = {sq for step in story.queue for sq in (step.from_square, step.to_square)}
        out = []
        for move in start.legal_moves:
            if not _touches(start, move) <= hidden:
                continue
            after = start.copy(stack=False)
            after.push(move)
            told = story.moves + (move,)
            rest = story.queue[1:]
            if story.queue[0] in after.legal_moves:
                after.push(story.queue[0])
                told += (story.queue[0],)
            else:
                rest = ()
            if self.wrong(premove_picture(after, rest)):
                continue
            out.append(replace(story, moves=told, queue=rest, view=None, hidden=True))
        return out

    def _edits(
        self,
        story: Story,
        board: chess.Board,
        queue: tuple[chess.Move, ...],
        moves: tuple[chess.Move, ...],
        events: int,
        add: Callable[[Story], None],
        budget: list[int],
    ) -> None:
        """The queue cancelled, or grown by steps from where the user's pieces are drawn, or
        cancelled and begun again, so that the picture fits. A step lands on a square the
        picture gets wrong, or on one the frame reads empty on its way to another. A queue
        moves only the user's pieces, over whatever is in their way: a square where the frame
        shows an opponent's piece the picture has not is beyond any edit."""
        if queue and not self.wrong(board) and unseen(events + 1) <= self.max_unseen:
            cost = story.cost + unseen(events + 1)
            add(replace(story, moves=moves, queue=(), view=None, cost=cost))
        free = {sq for sq, r in self.obs.items() if r.kind == "empty"}
        bases = [(queue, events)] + ([((), events + 1)] if queue else [])
        for base, paid in bases:
            start = premove_picture(board, base)
            off = self.wrong(start)
            if not all(self._editable(start, sq) for sq in off):
                continue
            movable = [
                sq
                for sq in sorted(off)
                if (p := start.piece_at(sq)) is not None and p.color == self.user
            ]
            if base and base[-1].to_square not in movable:
                last = start.piece_at(base[-1].to_square)
                if last is not None and last.color == self.user:
                    movable.append(base[-1].to_square)

            # Where a piece moved by the queue now stands, the frame shows it: every way ends
            # on such a square, and a step before the last ends one step from one.
            targets = {sq for sq in off if self.obs[sq].kind == "piece"}
            if not targets:
                continue

            # One base's premoves get a share of the budget, not all of it.
            share = [min(budget[0], EDIT_BUDGET)]

            def grow(
                picture: chess.Board,
                steps: tuple[chess.Move, ...],
                base: tuple[chess.Move, ...] = base,
                paid: int = paid,
                off: set[chess.Square] = off,
                targets: set[chess.Square] = targets,
                movable: list[chess.Square] = movable,
                share: list[int] = share,
            ) -> None:
                if budget[0] <= 0 or share[0] <= 0:
                    return
                budget[0] -= 1
                share[0] -= 1
                if unseen(paid + len(steps)) > self.max_unseen:
                    return
                if steps and steps[-1].to_square in targets and not self.wrong(picture):
                    cost = story.cost + unseen(paid + len(steps))
                    add(replace(story, moves=moves, queue=base + steps, view=None, cost=cost))
                    return
                left = self.max_steps - len(steps)
                if not left:
                    return
                for sq in [steps[-1].to_square] if steps else movable:
                    walked = _walked(base + steps, sq, moves)
                    for step in lone_moves(picture, sq):
                        to = step.to_square
                        if to in walked:
                            continue  # back over its own way: the shorter queue draws the same
                        if to not in targets:
                            if left == 1 or (to not in off and to not in free):
                                continue
                            if left == 2 and not any(
                                t in _lone_reach(picture, sq, to) for t in targets
                            ):
                                continue
                        after = picture.copy(stack=False)
                        _apply_step(after, step)
                        grow(after, steps + (step,))

            grow(start, ())

    def _editable(self, picture: chess.Board, square: chess.Square) -> bool:
        """Whether a queue edit could put `square` right: the picture has one of the user's
        pieces there (it moved on), or the frame shows a piece that could be the user's (a
        piece moved there), or the frame shows it empty where the picture has the opponent's
        (a step went over it and on)."""
        piece = picture.piece_at(square)
        if piece is not None and piece.color == self.user:
            return True
        reading = self.obs[square]
        if reading.kind == "empty":
            return piece is not None
        if reading.symbol is None:
            return True
        return chess.Piece.from_symbol(reading.symbol).color == self.user


def _walked(
    queue: Sequence[chess.Move], square: chess.Square, moves: Sequence[chess.Move] = ()
) -> set[chess.Square]:
    """The squares the piece queued to `square` has stood on in `queue`, and the square the
    story's last move of it came from: its own way, not another piece's -- a pawn may be
    premoved onto the square a queen's step leaves. A piece premoved back where it just came
    from draws the picture before the move: a bishop that took on h3 and was premoved back to
    e6 explained a knight caught in flight as taken (the cold start, 2026-10-05)."""
    walked = {square}
    for step in reversed(queue):
        if step.to_square in walked:
            walked.add(step.from_square)
    origin = _origin(queue, square)
    for move in reversed(moves):
        if move.to_square == origin:
            walked.add(move.from_square)
            break
    return walked


def _origin(queue: Sequence[chess.Move], square: chess.Square) -> chess.Square:
    """Where the piece queued to `square` stood before its queue."""
    origin = square
    for step in reversed(queue):
        if step.to_square == origin:
            origin = step.from_square
    return origin


def _apply_step(board: chess.Board, move: chess.Move) -> None:
    """One queued step drawn on `board`: the piece lifted and put down over whatever is
    there (`premove_picture`)."""
    piece = board.remove_piece_at(move.from_square)
    if piece is None:
        return
    rank = chess.square_rank(move.from_square)
    files = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    if piece.piece_type == chess.KING and abs(files) == 2:
        rook_from, rook_to = (7, 5) if files > 0 else (0, 3)
        rook = board.remove_piece_at(chess.square(rook_from, rank))
        if rook is not None:
            board.set_piece_at(chess.square(rook_to, rank), rook)
    if move.promotion:
        piece = chess.Piece(move.promotion, piece.color)
    board.set_piece_at(move.to_square, piece)


def _lone_reach(picture: chess.Board, square: chess.Square, to: chess.Square) -> set[chess.Square]:
    """Where the piece on `square` could step next from `to`, alone on the board."""
    after = picture.copy(stack=False)
    _apply_step(after, chess.Move(square, to))
    return {m.to_square for m in lone_moves(after, to)}


def unseen(events: int) -> int:
    """The pictures `events` made between two frames that neither frame showed: all but the
    last."""
    return max(0, events - 1)


def _touches(board: chess.Board, move: chess.Move) -> set[chess.Square]:
    """The squares a move changes: its two, a castling rook's, an en-passant victim's."""
    touched = {move.from_square, move.to_square}
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        if chess.square_file(move.to_square) > chess.square_file(move.from_square):
            touched |= {chess.square(7, rank), chess.square(5, rank)}
        else:
            touched |= {chess.square(0, rank), chess.square(3, rank)}
    elif board.is_en_passant(move):
        touched.add(
            chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        )
    return touched
