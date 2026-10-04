"""The engine service for a game followed live: every position searched once, on a pool of
single-thread Stockfish workers, and nothing searched that will not be read.

The engine audit of 2026-10-04 (a replay of a 38-minute session) found the one-engine analyser
searching 258 of 624 positions two to six times, throwing away 211 s of searching by
interrupting it, and searching again from depth 1 the very positions its look-ahead had been
searching when the game reached them; and four threads costing 7.7 times the processor time of
one for the same depth-18 answer (Tai, 2026-10-04: "a lot of work is duplicated, interrupted,
and some are just discarded ... the code is not very smart about all of it").

So, here:

- A position is a node, by its cache key (the FEN and the moves a repetition could reach back
  over). Its search is the canonical one -- the full depth, every line, one thread -- made
  once; every depth it finishes is a reader's at once (the first usable one is depth 12 of that
  same search, not a search of its own).
- The game's positions are its path. While the player thinks, the workers search ahead: the
  best moves from the position on the board, the best reply to each, and so on to
  `look_ahead` plies (Tai, 2026-09-27; 2026-10-04: "we absolutely need look-ahead").
- A position searched ahead that the game reaches is promoted: its search, done or under way,
  becomes the game's, never begun again. A search is stopped only when its position is on no
  path any more: off the game and out of the look-ahead, nothing will read it.
- One worker is kept for the game's own positions, so a position the game reaches never waits
  behind the look-ahead and never stops it.
- A move is judged from the position before's lines when it is among them (85% of moves), from
  the position after's search otherwise -- which is the game's next position, searched anyway.
- The side searches (threats, traps, declined recaptures, what a quiet move threatens) are
  made for the game's positions, once their own search is done; a position searched ahead gets
  them when the game reaches it.

Events go out as they did: an `Analysis` for each depth of a game position (provisional until
the full depth and its side searches are in), and a `MoveAssessment` for each move, in the
order of the moves.
"""

from __future__ import annotations

import heapq
import logging
import math
import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import chess

from sidecoach.analysis.annotator import (
    EARLY_DEPTH,
    GameAnnotator,
    Measured,
    cache_key,
)
from sidecoach.analysis.engine import Engine, Interrupted
from sidecoach.events import Event, Line

log = logging.getLogger(__name__)

# The look-ahead searches the positions the game is likely to reach, likeliest first, while it
# waits (Tai, 2026-10-04: "an adaptive, dynamic system that is able to maximize both comp eff and
# the chance of catching the opponent's move"). A position's chance is the product of the chances
# of the moves leading there. A move's chance comes from the position's own lines: a line that
# loses `loss` of expected score against the best is played with a chance in proportion to
# exp(-loss / T) -- a forced position puts nearly all of it on the one move, a quiet one spreads
# it -- and T, how closely a side's moves follow the engine, is learned from this game's own moves,
# each side's, as they are played. Nothing is trained and nothing is kept between games (Tai,
# 2026-10-04: "move accuracy needs to be dependent on the game ... a static model ... is going to
# drift"). A fixed width searched what was unlikely and missed what was likely: the player played
# the engine's best 93% of the time in his puzzles and 46% in his games, and more than half the
# look-ahead's time went to positions never reached (tools/engine/move_odds.py, simulate.py).
#
# Where T starts in a game, and how many moves' worth that start counts for: a little, so the
# game's own moves decide it within a few.
T_START = 0.05
T_START_WEIGHT = 2.0
T_GRID = tuple(0.005 * 1.35**i for i in range(20))  # 0.005 .. ~1.5, the values T is fitted over
# The share of moves played outside the lines searched, at the start of a game (learned too).
OUTSIDE_START = 0.15
# A position is searched ahead while its chance of being reached is at least REACH, at most
# LOOK_AHEAD plies from the board (tuned by simulation, tools/engine/simulate.py).
REACH = 0.15
LOOK_AHEAD = 4
# How many workers search side by side, one thread each (Tai, 2026-10-04: three).
WORKERS = 3
# The positions kept, searched, for the game and for any game after it (a puzzle tried again
# is answered at once); the oldest go past this.
KEEP_NODES = 20000

MAIN = "main"
SIDE = "side"


@dataclass
class _Node:
    """One position: its board (with the moves that led there, which the engine reads for
    repetitions), and what has been found of it."""

    key: str
    board: chess.Board
    measured: Measured | None = None  # its own search, at the full depth
    whole: Measured | None = None  # with its side searches
    partial: list[Line] | None = None  # the deepest depth so far, while its search runs
    failed: bool = False


@dataclass
class _Move:
    ply: int
    before: str
    after: str
    move: chess.Move
    by_user: bool


@dataclass
class _Job:
    kind: str  # MAIN or SIDE
    key: str
    era: int


class Analyser:
    """A game's positions measured on a pool of workers: submit positions and moves, poll
    events, never block. `engines`: one per worker (open or not), the first kept for the game's
    own positions when there are several; the annotator's own engine alone when not given.
    `look_ahead`: plies searched ahead of the game; none unless asked (a live session asks for
    LOOK_AHEAD, `open_analyser`)."""

    def __init__(
        self,
        annotator: GameAnnotator,
        *,
        engines: Sequence[Engine] | None = None,
        look_ahead: int = 0,
    ) -> None:
        self.annotator = annotator
        self.engines: list[Engine] = list(engines) if engines else [annotator.engine]
        self.look_ahead = look_ahead
        self._cond = threading.Condition()
        self._nodes: dict[str, _Node] = {}
        self._game_id: str | None = None
        self._path: dict[int, str] = {}  # ply -> the game's position at it
        self._ply_of: dict[str, int] = {}
        self._moves: dict[int, _Move] = {}
        self._current = 0
        self._next_verdict: int | None = None
        # What each game ply has been told: its position, the deepest depth, whether whole.
        self._told: dict[int, tuple[str, int, bool]] = {}
        self._ahead: dict[str, tuple[float, int]] = {}  # key -> (-chance, plies on)
        self._user: chess.Color | None = None  # the player's colour, once the game says it
        # This game's moves, each side's (True: the player's), as what the look-ahead's chances
        # learn from: the losses of the lines each was played among, and which it was (None: a
        # move outside them); each side's T fitted to them (`_fit`).
        self._played: dict[bool, list[tuple[tuple[float, ...], int | None]]] = {True: [], False: []}
        self._t: dict[bool, float] = {True: T_START, False: T_START}
        self.reach = REACH
        self._running: dict[int, _Job] = {}  # by worker: the job a stop can reach
        self._claimed: dict[int, _Job] = {}  # by worker: the job it took, running or about to
        self._era = 0
        self._results: queue.SimpleQueue[Event] = queue.SimpleQueue()
        self._threads: list[threading.Thread] = []
        self._closed = False
        # What it did, for the logs and the replay's measure.
        self.searches = 0  # positions searched to the full depth
        self.side_searches = 0
        self.looked = 0  # positions searched ahead of the game
        self.ahead_hits = 0  # of them, reached by the game
        self.stopped = 0  # searches let go because nothing would read them
        self.withdrawn = 0
        self.failures = 0
        self.last_error: str | None = None

    # ---- the caller's side ----------------------------------------------------------------

    def start(self) -> None:
        if self._threads:
            return
        for index, engine in enumerate(self.engines):
            thread = threading.Thread(
                target=self._work,
                args=(index, engine),
                name=f"sidecoach-analyser-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def __enter__(self) -> Analyser:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            self._closed = True
            for index in list(self._running):
                self.engines[index].interrupt()
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout)
        self._threads = []
        for engine in self._unique_engines():
            engine.close()

    @property
    def pending(self) -> int:
        """What is still owed to the game: its positions not measured whole, and its moves not
        judged. The look-ahead owes nothing."""
        with self._cond:
            owed = sum(1 for key in self._path.values() if not self._whole_told(key))
            if self._next_verdict is not None:
                owed += sum(1 for ply in self._moves if ply >= self._next_verdict)
            return owed

    def poll(self) -> list[Event]:
        out: list[Event] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def submit_position(
        self, game_id: str, board: chess.Board, ply: int, *, user: chess.Color | None = None
    ) -> None:
        """A position the game stands on with no move to it: its start, or a resync. `user`:
        the player's colour, when the game says it (the look-ahead's widths follow it)."""
        with self._cond:
            if self._game_id != game_id:
                self._new_game(game_id)
            if user is not None:
                self._user = user
            key = self._node(board).key
            self._set_path(ply, key)
            self._current = ply
            # Told again: a position submitted is one the player faces anew (a takeback to the
            # start, a re-read), whatever was told of it before.
            self._told.pop(ply, None)
            self._reached(key, ply)
            self._reframe()
            self._cond.notify_all()

    def submit_move(
        self,
        game_id: str,
        before: chess.Board,
        move: chess.Move,
        ply: int,
        *,
        by_user: bool = True,
    ) -> None:
        with self._cond:
            if self._game_id != game_id:
                self._new_game(game_id)
            if move not in before.legal_moves:
                self.failures += 1
                self.last_error = f"{move.uci()} is not legal in {before.fen()}"
                return
            # Whose move it was says the player's colour, every move.
            self._user = before.turn if by_user else not before.turn
            source = self._node(before.copy())
            if self._path.get(ply - 1) != source.key:
                self._set_path(ply - 1, source.key)
                self._reached(source.key, ply - 1)
            after = before.copy()
            after.push(move)
            target = self._node(after)
            if target.key in self._ahead:
                self.ahead_hits += 1
                self.annotator.ahead_hits += 1
            self._set_path(ply, target.key)
            self._moves[ply] = _Move(ply, source.key, target.key, move, by_user)
            if self._next_verdict is None or ply < self._next_verdict:
                self._next_verdict = ply
            self._current = ply
            # The verdict first: the word on the move is the news, the position after it next.
            self._judge()
            self._reached(target.key, ply)
            self._reframe()
            self._cond.notify_all()

    def withdraw(self, game_id: str, ply: int) -> None:
        """The game's moves after `ply` taken back: nothing more is said of them, and a search
        made only for them stops -- unless the look-ahead from the position taken back to
        wants it, which it often does (the move taken back was a likely one)."""
        with self._cond:
            if game_id != self._game_id:
                return
            gone = [p for p in self._path if p > ply]
            self.withdrawn += sum(1 for p in gone if p in self._moves)
            for p in gone:
                key = self._path.pop(p)
                if self._ply_of.get(key) == p:
                    del self._ply_of[key]
                self._moves.pop(p, None)
                self._told.pop(p, None)
                self.annotator._assessed.pop(p, None)
            if self._next_verdict is not None and self._next_verdict > ply + 1:
                self._next_verdict = ply + 1
            if not self._moves:
                self._next_verdict = None
            self._current = ply
            self._reframe()
            self._cond.notify_all()

    def reset(self) -> None:
        """A new game, or the board re-read: the game's path and look-ahead let go, and what
        is under way stops; what was measured is kept, for a position seen again."""
        with self._cond:
            self._game_id = None
            self._path.clear()
            self._ply_of.clear()
            self._moves.clear()
            self._told.clear()
            self._ahead.clear()
            self._next_verdict = None
            self._current = 0
            self._stop_unwanted()
            self.annotator.reset()
            self._cond.notify_all()

    def forget_measured(self) -> None:
        """The settings a search depends on changed: nothing measured before counts, and a
        search under way is not kept (`GameAnnotator.forget_measured`)."""
        with self._cond:
            self._era += 1
            self._nodes = {
                k: _Node(k, n.board) for k, n in self._nodes.items() if k in self._ply_of
            }
            self._told = {p: (k, 0, False) for p, (k, _, _) in self._told.items()}
            for index in list(self._running):
                self.engines[index].interrupt()
            self.annotator.forget_measured()
            self._cond.notify_all()

    # ---- the graph ----------------------------------------------------------------------

    def _new_game(self, game_id: str) -> None:
        """Another game: what the look-ahead learned of how its sides play starts over."""
        self._game_id = game_id
        self._played = {True: [], False: []}
        self._t = {True: T_START, False: T_START}

    def _set_path(self, ply: int, key: str) -> None:
        """The game's position at `ply` is `key`: the one it held before, if another, is no
        longer the game's -- nothing more is told of it at this ply, and its search stops unless
        the look-ahead wants it."""
        old = self._path.get(ply)
        if old is not None and old != key:
            if self._ply_of.get(old) == ply:
                del self._ply_of[old]
            self._told.pop(ply, None)
        self._path[ply] = key
        self._ply_of[key] = ply

    def _node(self, board: chess.Board) -> _Node:
        key = cache_key(board)
        node = self._nodes.get(key)
        if node is None:
            if len(self._nodes) >= KEEP_NODES:
                self._forget_oldest()
            node = self._nodes[key] = _Node(key, board.copy())
        return node

    def _forget_oldest(self) -> None:
        keep = set(self._path.values()) | set(self._ahead) | {j.key for j in self._claimed.values()}
        for key in list(self._nodes)[: KEEP_NODES // 10]:
            if key not in keep:
                del self._nodes[key]

    def _alive(self, key: str) -> bool:
        return key in self._ply_of or key in self._ahead

    def _reframe(self) -> None:
        """The look-ahead from the position the game is on: the positions whose chance of being
        reached is at least `reach`, at most `look_ahead` plies on, each with its chance --
        grown from searched positions only; a search for a position on no path stops."""
        ahead: dict[str, tuple[float, int]] = {}
        root = self._path.get(self._current)
        frontier: list[tuple[float, int, str]] = [(-1.0, 0, root)] if root else []
        while frontier:
            negative, distance, key = heapq.heappop(frontier)
            node = self._nodes.get(key)
            if node is None or node.measured is None or distance >= self.look_ahead:
                continue
            mover = self._is_player(node.board, distance)
            odds = self._odds(mover, node.measured.lines)
            for line, odd in zip(node.measured.lines, odds, strict=True):
                chance = -negative * odd
                if chance < self.reach:
                    continue
                try:
                    move = chess.Move.from_uci(line.moves[0])
                except (ValueError, IndexError):
                    continue
                if move not in node.board.legal_moves:
                    continue
                child = node.board.copy()
                child.push(move)
                if child.is_game_over():
                    continue
                child_node = self._node(child)
                if child_node.key in ahead or child_node.key in self._ply_of:
                    continue
                ahead[child_node.key] = (-chance, distance + 1)
                heapq.heappush(frontier, (-chance, distance + 1, child_node.key))
        self._ahead = ahead
        self._stop_unwanted()

    def _is_player(self, board: chess.Board, distance: int) -> bool:
        """Whether the player is to move in `board`; while their colour is unknown, the side to
        move on the board is taken for theirs."""
        if self._user is None:
            return distance % 2 == 0
        return board.turn == self._user

    def _odds(self, player: bool, lines: Sequence[Line]) -> list[float]:
        """The chance that each of `lines` is the move played, by the player or the opponent:
        in proportion to exp(-loss / T) of the line's loss against the best, T this game's for
        that side; the share of moves this game played outside the lines left out."""
        if not lines:
            return []
        best = lines[0].expected_score
        weights = [math.exp(-max(0.0, best - ln.expected_score) / self._t[player]) for ln in lines]
        total = sum(weights)
        played = self._played[player]
        outside = (OUTSIDE_START * T_START_WEIGHT + sum(1 for _, i in played if i is None)) / (
            T_START_WEIGHT + len(played)
        )
        return [(1.0 - outside) * w / total for w in weights]

    def _learn(self, move: _Move, lines: list[Line]) -> None:
        """A move played: which of the lines of the position it came from it was, kept for the
        side that played it, and that side's T fitted again."""
        if not lines:
            return
        uci = move.move.uci()
        best = lines[0].expected_score
        losses = tuple(max(0.0, best - ln.expected_score) for ln in lines)
        index = next((i for i, ln in enumerate(lines) if ln.moves and ln.moves[0] == uci), None)
        self._played[move.by_user].append((losses, index))
        self._t[move.by_user] = self._fit(self._played[move.by_user])

    @staticmethod
    def _fit(played: list[tuple[tuple[float, ...], int | None]]) -> float:
        """The T that makes this game's moves likeliest (each the line it was among its lines,
        chance in proportion to exp(-loss / T)), the start counting as T_START_WEIGHT moves
        played at the start's T; over T_GRID. Moves outside the lines say nothing of it."""
        chosen = [(losses, i) for losses, i in played if i is not None]

        def likelihood(t: float) -> float:
            total = 0.0
            for losses, i in chosen:
                logs = [-loss / t for loss in losses]
                top = max(logs)
                total += logs[i] - top - math.log(sum(math.exp(x - top) for x in logs))
            # The start: as if T_START_WEIGHT moves had shown it, a penalty for straying from it.
            return total - T_START_WEIGHT * (math.log(t / T_START)) ** 2

        return max(T_GRID, key=likelihood)

    def _stop_unwanted(self) -> None:
        for index, job in list(self._running.items()):
            wanted = self._alive(job.key) if job.kind == MAIN else job.key in self._ply_of
            if not wanted:
                self.stopped += 1
                self.engines[index].interrupt()

    # ---- the workers --------------------------------------------------------------------

    def _pick(self, index: int) -> _Job | None:
        """The next job for worker `index`: the game's newest position not searched, then the
        side searches of the game's positions, then -- unless this is the game's own worker --
        the look-ahead, likeliest first."""
        busy = {(j.kind, j.key) for j in self._claimed.values()}
        for ply in sorted(self._path, reverse=True):
            node = self._nodes[self._path[ply]]
            if node.measured is None and not node.failed and (MAIN, node.key) not in busy:
                return _Job(MAIN, node.key, self._era)
        if self.annotator.with_threats:
            for ply in sorted(self._path, reverse=True):
                node = self._nodes[self._path[ply]]
                if (
                    node.measured is not None
                    and node.whole is None
                    and not node.failed
                    and (SIDE, node.key) not in busy
                ):
                    return _Job(SIDE, node.key, self._era)
        if index == 0 and len(self.engines) > 1:
            return None  # the game's own worker
        for key, _ in sorted(self._ahead.items(), key=lambda kv: kv[1]):
            node = self._nodes.get(key)
            wanted = node is not None and node.measured is None and not node.failed
            if wanted and (MAIN, key) not in busy:
                return _Job(MAIN, key, self._era)
        return None

    def _work(self, index: int, engine: Engine) -> None:
        while True:
            with self._cond:
                job = None
                while not self._closed:
                    job = self._pick(index)
                    if job is not None:
                        break
                    self._cond.wait()
                if self._closed or job is None:
                    return
                # Taken the moment it is picked: no other worker picks it meanwhile.
                self._claimed[index] = job
                board = self._nodes[job.key].board.copy()
                measured = self._nodes[job.key].measured
            try:
                found = self._run(index, engine, job, board, measured)
            except Interrupted:
                found = None
            except Exception as exc:  # noqa: BLE001 - a failed search must never end a session
                with self._cond:
                    self.failures += 1
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    node = self._nodes.get(job.key)
                    if node is not None:
                        node.failed = True
                found = None
            with self._cond:
                self._running.pop(index, None)
                self._claimed.pop(index, None)
                if found is not None and job.era == self._era and job.key in self._nodes:
                    if job.kind == MAIN:
                        self._searched(job.key, found)
                    else:
                        self._sided(job.key, found)
                self._cond.notify_all()

    def _run(
        self,
        index: int,
        engine: Engine,
        job: _Job,
        board: chess.Board,
        measured: Measured | None,
    ) -> Measured | None:
        """One job's search, stoppable from the moment it is running (`_stop_unwanted`);
        None when it was no longer wanted by then."""
        with engine.interruptible():
            with self._cond:
                if self._closed or job.era != self._era or not self._wanted(job):
                    return None
                self._running[index] = job  # inside the stoppable block: a stop reaches it
            if job.kind == MAIN:
                return self.annotator.search(
                    board,
                    engine=engine,
                    on_depth=lambda lines, key=job.key, era=job.era: self._depth(key, era, lines),
                )
            assert measured is not None
            return self.annotator.with_side(board, measured, engine=engine)

    def _wanted(self, job: _Job) -> bool:
        node = self._nodes.get(job.key)
        if node is None:
            return False
        if job.kind == MAIN:
            return node.measured is None and self._alive(job.key)
        return node.whole is None and job.key in self._ply_of

    # ---- what comes back ----------------------------------------------------------------

    def _depth(self, key: str, era: int, lines: list[Line]) -> None:
        """A depth of a search finished, the search going on."""
        with self._cond:
            node = self._nodes.get(key)
            if node is None or era != self._era or not lines:
                return
            node.partial = lines
            ply = self._ply_of.get(key)
            if ply is not None:
                self._tell_partial(ply, node)

    def _searched(self, key: str, measured: Measured) -> None:
        node = self._nodes[key]
        node.measured = measured
        node.partial = None
        self.searches += 1
        if key in self._ahead and key not in self._ply_of:
            self.looked += 1
        if not self.annotator.with_threats:
            node.whole = measured
        self._judge()  # the verdict first: the word on the move, then the position after it
        ply = self._ply_of.get(key)
        if ply is not None:
            self._tell(ply, node)
        self._reframe()

    def _sided(self, key: str, whole: Measured) -> None:
        node = self._nodes[key]
        node.whole = whole
        self.side_searches += 1
        ply = self._ply_of.get(key)
        if ply is not None:
            self._tell(ply, node)

    def _reached(self, key: str, ply: int) -> None:
        """The game is on this position at `ply`: what is known of it is told at once."""
        node = self._nodes[key]
        told = self._told.get(ply)
        if told is not None and told[0] != key:
            del self._told[ply]
        if node.measured is not None:
            self._tell(ply, node)
        elif node.partial is not None:
            self._tell_partial(ply, node)

    def _whole_told(self, key: str) -> bool:
        ply = self._ply_of.get(key)
        told = self._told.get(ply) if ply is not None else None
        return told is not None and told[0] == key and told[2]

    def _tell_partial(self, ply: int, node: _Node) -> None:
        lines = node.partial or []
        depth = lines[0].depth if lines else 0
        if not EARLY_DEPTH <= depth < self.annotator.depth:
            return
        told = self._told.get(ply)
        if told is not None and told[0] == node.key and told[1] >= depth:
            return
        self._told[ply] = (node.key, depth, False)
        measured = self.annotator.measured_lines(node.board, lines)
        self._put(
            self.annotator.analysis_from(
                self._game_id or "", node.board, ply, measured, provisional=True
            )
        )

    def _tell(self, ply: int, node: _Node) -> None:
        """The full depth told: whole once its side searches are in, provisional before."""
        found = node.whole or node.measured
        if found is None or self._game_id is None:
            return
        whole = node.whole is not None
        told = self._told.get(ply)
        if (
            told is not None
            and told[0] == node.key
            and (told[2] or (not whole and told[1] >= found.depth))
        ):
            return
        self._told[ply] = (node.key, found.depth, whole)
        self._put(
            self.annotator.analysis_from(
                self._game_id, node.board, ply, found, provisional=not whole
            )
        )

    def _judge(self) -> None:
        """Each move judged as soon as it can be, in the order of the moves: the move before
        is what a verdict reads ("the opponent just erred", "a recapture")."""
        while self._next_verdict is not None and self._next_verdict in self._moves:
            move = self._moves[self._next_verdict]
            before = self._nodes[move.before]
            after = self._nodes[move.after]
            own = before.measured is not None and self.annotator.judgeable(
                before.measured, move.move
            )
            if before.failed or (not own and after.failed):
                # A search that failed judges nothing: this move goes without a verdict, and
                # the moves after it are judged all the same -- it held them all back.
                self._next_verdict += 1
                continue
            if before.measured is None:
                return
            if not own and after.measured is None:
                return  # not among the lines: the position after is needed
            try:
                verdict = self.annotator.assess_from(
                    self._game_id or "",
                    before.board,
                    move.move,
                    move.ply,
                    before.measured,
                    after.measured,
                )
            except Exception as exc:  # noqa: BLE001 - a move that cannot be judged is skipped
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            else:
                self._put(verdict)
                self._learn(move, before.measured.lines)
            self._next_verdict += 1

    def _put(self, event: Event) -> None:
        self._results.put(event)

    def _unique_engines(self) -> list[Engine]:
        seen: list[Engine] = []
        for engine in self.engines:
            if not any(engine is e for e in seen):
                seen.append(engine)
        return seen

    def reconfigure(self, **options: Any) -> None:
        """Hash, the lines' length or the scoring saved mid-session: on every worker's engine."""
        for engine in self._unique_engines():
            for name, value in options.items():
                if name == "hash_mb":
                    engine.reconfigure(hash_mb=value)
                else:
                    setattr(engine, name, value)


def open_analyser(
    engine_path: object,
    *,
    depth: int,
    multipv: int,
    workers: int = WORKERS,
    hash_mb: int,
    pv_plies: int,
    thresholds: Any,
    score_model: str = "human",
    look_ahead: int = LOOK_AHEAD,
    tagger: Any = None,
    book: Any = None,
    make_engine: Callable[..., Engine] = Engine,
) -> Analyser:
    """An analyser ready to take moves, its workers' engines started, one thread each."""
    from pathlib import Path

    engines = [
        make_engine(
            path=Path(str(engine_path)),
            threads=1,
            hash_mb=hash_mb,
            multipv=multipv,
            pv_plies=pv_plies,
            score_model=score_model,
        )
        for _ in range(max(1, workers))
    ]
    for engine in engines:
        engine.open()
    annotator = GameAnnotator(
        engine=engines[0],
        depth=depth,
        multipv=multipv,
        thresholds=thresholds,
        tagger=tagger,
        book=book,
    )
    analyser = Analyser(annotator, engines=engines, look_ahead=look_ahead)
    analyser.start()
    return analyser


__all__ = ["LOOK_AHEAD", "WORKERS", "Analyser", "open_analyser"]
