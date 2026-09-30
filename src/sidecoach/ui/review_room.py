"""A game reviewed in a window of its own (`/review/<key>`; docs/game-review-plan.md), apart
from the screen the panel watches, as a puzzle is (ui/puzzle_room.py).

A room opens a game of the library: the review saved for it at this strength when there is one
of the current format, else the game reviewed whole (`analysis.game_review`) on engines of the
room's own, side by side, its progress shown as it goes, then summed up
(`profile.game_summary`) and saved beside the profile. The page polls the room as the panel
polls the engine: the state -- the review's status, the summary, every move as the page draws
it, the AI coach's words on the key moves -- and the room's log lines. Those words are written
only when the player asks (`write_words`; coach/review_words.py) and kept with the review.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.game_review import STRENGTHS, THREADS, Strength, review_game, workers_for
from sidecoach.analysis.score import Thresholds, position_score
from sidecoach.coach import review_links, review_words
from sidecoach.config import Settings, find_stockfish
from sidecoach.events import GameSummary, Line, ReviewedGame, Side
from sidecoach.llm.usage import prices_for
from sidecoach.profile import library
from sidecoach.profile.game_summary import missed_motifs, summarize
from sidecoach.profile.stat_facts import keep_stats
from sidecoach.profile.store import LibraryRow, Store
from sidecoach.ui.log import DebugLog
from sidecoach.ui.review_retry import Retries

# The review's format: a review saved in an older one is done again.
REVIEW_VERSION = 1
# The loop's pace, and how long an untouched room lives.
TICK_S = 0.25
IDLE_CLOSE_S = 20 * 60.0
# The lines a move's card plays on the board (the best, and what follows the move), in plies.
LINE_PLIES = 6
# The last card's training: the motifs the game missed most, and puzzles of each.
PRACTICE_MOTIFS = 2
PRACTICE_PUZZLES = 3
# A model call on one key move may take this long (a model that reasons first is slow).
AI_TIMEOUT_S = 300.0
# The review's words reason at this effort (Tai 2026-09-30: "Reasoning none is usually not
# very accurate"), though the live coach may be set to none for speed.
REVIEW_EFFORT = "medium"
NO_AI = "No AI coach is set up: choose one in Settings."

# The model the review's words are asked of, or why there is none.
Model = Callable[[], tuple[Any | None, str | None]]


class ReviewRoom:
    def __init__(
        self,
        settings: Settings,
        row: LibraryRow,
        *,
        store_path: Path | None,
        strength: str = "standard",
        make_engine: Callable[..., Engine] | None = None,
        mirror: Callable[..., None] | None = None,
        model: Model | None = None,
    ) -> None:
        self.settings = settings
        self.row = row
        self.key = row.key
        self.strength = STRENGTHS.get(strength, STRENGTHS["standard"])
        self.store = Store(store_path)
        self._make_engine = make_engine
        self._mirror = mirror
        self._log = DebugLog()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.status = "starting"  # starting | analysing | ready | failed
        self.done = 0
        self.total = 0
        self.error: str | None = None
        self.reviewed: ReviewedGame | None = None
        self.summary: GameSummary | None = None
        self._moves: list[dict[str, Any]] = []
        self.practice: list[dict[str, Any]] = []
        self.retries: Retries | None = None
        self._model = model
        # The AI coach's words on the key moves, as saved: none yet | writing | ready | failed.
        self.words: dict[str, Any] | None = None
        self.words_status = "none"
        self.words_error: str | None = None
        self._estimated: tuple[str | None, float | None] | None = None  # by model name
        self.touched = time.monotonic()
        self.closed = False
        self._thread = threading.Thread(target=self._run, name=f"review-{self.key}", daemon=True)
        self._thread.start()

    # ---- the page's side --------------------------------------------------------------------

    def poll(self, since: int = 0) -> dict[str, Any]:
        return {"state": self.state(), "lines": self._log.since(since), "cursor": self._log.cursor}

    def state(self) -> dict[str, Any]:
        self.touched = time.monotonic()
        model, unavailable = self._model() if self._model is not None else (None, NO_AI)
        estimate = self._estimate(model)
        with self._lock:
            return {
                "key": self.key,
                "status": self.status,
                "progress": {"done": self.done, "total": self.total},
                "error": self.error,
                "strength": self.strength.name,
                "game": {
                    "source": self.row.source,
                    "white": self.row.white,
                    "black": self.row.black,
                    "white_elo": self.row.white_elo,
                    "black_elo": self.row.black_elo,
                    "result": self.row.result,
                    "time_class": self.row.time_class,
                    "played_at": self.row.played_at,
                    "user_side": self.row.user_side,
                },
                "start_fen": self.reviewed.start.fen if self.reviewed else None,
                "summary": self.summary.model_dump(mode="json") if self.summary else None,
                "moves": self._moves,
                "practice": self.practice,
                "ai": {
                    "status": self.words_status,
                    "error": self.words_error,
                    "words": self.words,
                    "unavailable": unavailable,
                    "moves": len(review_words.key_plies(self.reviewed)) if self.reviewed else 0,
                    "estimate": estimate,
                },
                "closed": self.closed,
            }

    def _estimate(self, model: Any | None) -> float | None:
        """What the AI coach's words would cost with `model` (dollars), made once a model; None
        when its prices are not known (a model on this computer's)."""
        name = getattr(model, "model", None)
        if model is None or self.reviewed is None:
            return None
        if self._estimated is None or self._estimated[0] != name:
            opening = _opening(self.reviewed)
            prices = prices_for(self.settings, name)
            cost = review_words.estimate(self.reviewed, self.row.user_side, opening, prices)
            self._estimated = (name, cost)
        return self._estimated[1]

    def write_words(self) -> str | None:
        """Set the AI coach writing on the key moves, in the background; why it cannot, else
        None (already writing is no reason)."""
        model, unavailable = self._model() if self._model is not None else (None, NO_AI)
        with self._lock:
            if self.status != "ready" or self.reviewed is None:
                return "the review is not ready"
            if self.words_status == "writing":
                return None
            if model is None:
                return unavailable
            self.words_status, self.words_error = "writing", None
            reviewed = self.reviewed
        threading.Thread(
            target=self._write_words, args=(reviewed, model), name=f"words-{self.key}", daemon=True
        ).start()
        return None

    def _write_words(self, reviewed: ReviewedGame, model: Any) -> None:
        # A model that reasons by its effort (luna) reasons at REVIEW_EFFORT here, whatever the
        # live coach is set to; a local model does not reason: it reasons worse about chess,
        # not better (Tai, 2026-09-30).
        reasons = getattr(model, "thinking_control", None) == "effort"
        if reasons:
            model = model.with_effort(REVIEW_EFFORT)
        try:
            words = review_words.write(
                reviewed,
                model,
                user=self.row.user_side,
                opening=_opening(reviewed),
                prices=prices_for(self.settings, getattr(model, "model", None)),
                thinking=reasons,
            )
            saved = {
                "model": words.model,
                "cost": words.cost,
                "texts": {str(ply): parts for ply, parts in words.texts.items()},
                "reveals": sorted(words.reveals),
            }
            shown = self._shown(saved, reviewed)
        except Exception as exc:  # noqa: BLE001 - a failure is said, never left "writing"
            with self._lock:
                self.words_status, self.words_error = "failed", str(exc)
            self._emit("sys", "error", f"the AI coach's words failed: {exc!r}")
            return
        with contextlib.suppress(Exception):  # a room closed meanwhile keeps them unsaved
            self.store.set_review_words(self.key, self.strength.name, json.dumps(saved))
        with self._lock:
            self.words, self.words_status = shown, "ready"
        cost = f", ${words.cost:.4f}" if words.cost is not None else ""
        self._emit(
            "sys",
            "info",
            f"AI words on {len(words.texts)} of {len(review_words.key_plies(reviewed))} key "
            f"moves ({words.model}{cost})",
        )
        for ply, why in sorted(words.dropped.items()):
            self._emit("sys", "warn", f"AI words on ply {ply} dropped: {why}")

    def _shown(self, saved: dict[str, Any], reviewed: ReviewedGame) -> dict[str, Any]:
        """The saved words as the page draws them: each paragraph also as its links to the
        board (coach/review_links.py), made on showing so they follow the linking's rules."""
        links = review_links.linked(saved.get("texts", {}), reviewed, self.row.user_side)
        return {**saved, "links": links}

    def close(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)

    # ---- the review ---------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            try:
                if not self._load():
                    self._review()
            except Exception as exc:  # noqa: BLE001 - a failed review is said, not raised
                # The room stays: the page reads the failure at its next poll.
                with self._lock:
                    self.status = "failed"
                    self.error = str(exc)
                self._emit("sys", "error", f"review failed: {exc!r}")
            while not self._stop.is_set():
                if time.monotonic() - self.touched > IDLE_CLOSE_S:
                    self._emit("sys", "info", "room idle; closing")
                    break
                self._stop.wait(TICK_S)
        finally:
            self.closed = True
            self.store.close()
            if self.retries is not None:
                self.retries.close()

    def _load(self) -> bool:
        """The review saved at this strength, when it is of the current format."""
        saved = self.store.review(self.key, self.strength.name)
        if saved is None or saved.version != REVIEW_VERSION:
            return False
        reviewed = ReviewedGame.model_validate_json(saved.report)
        # Its words ready with it: the page stops asking once the review is.
        words = self._shown(json.loads(saved.words), reviewed) if saved.words else None
        self._ready(reviewed, GameSummary.model_validate_json(saved.summary or "{}"), words)
        self._emit("sys", "info", f"review of {self.key} loaded ({self.strength.name})")
        return True

    def _review(self) -> None:
        workers = workers_for()
        with self._lock:
            self.status = "analysing"
        self._emit(
            "sys",
            "info",
            f"reviewing {self.key}: depth {self.strength.depth}, {workers} engines",
        )
        done = review_row(
            self.settings,
            self.row,
            self.store,
            self.strength,
            workers=workers,
            make_engine=self._make_engine,
            on_progress=self._progress,
            stop=self._stop,
        )
        if done is None:
            return
        reviewed, summary = done
        self._ready(reviewed, summary)
        self._emit("sys", "info", f"reviewed in {reviewed.seconds:.1f}s")

    def _progress(self, done: int, total: int) -> None:
        with self._lock:
            self.done, self.total = done, total

    def _practice(self, reviewed: ReviewedGame) -> list[dict[str, Any]]:
        """Puzzles for what the game passed up (the last card's training): the motifs the
        player's mistakes missed -- both sides' when the library does not know which was the
        player's -- each with puzzles of it at the player's puzzle level."""
        from sidecoach.profile.puzzles import open_store, puzzle_level

        user = self.row.user_side
        sides = {Side(user)} if user else {Side.WHITE, Side.BLACK}
        motifs = missed_motifs(reviewed, sides)[:PRACTICE_MOTIFS]
        puzzles = open_store(self.settings.data_dir) if motifs else None
        if puzzles is None:
            return []
        try:
            level = puzzle_level(self.store)
            return [
                {
                    "motif": motif,
                    "missed": missed,
                    "puzzles": [
                        {"id": p.id, "rating": p.rating}
                        for p in puzzles.pick(motif, level, count=PRACTICE_PUZZLES)
                    ],
                }
                for motif, missed in motifs
            ]
        finally:
            puzzles.close()

    def _ready(
        self, reviewed: ReviewedGame, summary: GameSummary, words: dict[str, Any] | None = None
    ) -> None:
        moves = move_views(reviewed)
        practice = self._practice(reviewed)
        retries = Retries(
            self.settings,
            reviewed,
            self.strength,
            game_id=self.key,
            make_engine=self._make_engine,
        )
        with self._lock:
            self.reviewed, self.summary, self._moves = reviewed, summary, moves
            self.practice = practice
            self.retries = retries
            if words is not None:
                self.words, self.words_status = words, "ready"
            self.status = "ready"
            self.done = self.total = len(reviewed.moves) + 1

    def _emit(self, chan: str, level: str, text: str, **data: Any) -> None:
        self._log.emit(chan, level, text, **data)
        if self._mirror is not None:
            with contextlib.suppress(Exception):
                self._mirror("review", level, f"{self.key} {chan}: {text}", **data)


def review_row(
    settings: Settings,
    row: LibraryRow,
    store: Store,
    strength: Strength,
    *,
    workers: int,
    make_engine: Callable[..., Engine] | None = None,
    low_priority: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    stop: threading.Event | None = None,
) -> tuple[ReviewedGame, GameSummary] | None:
    """A game of the library reviewed whole on `workers` engines of its own, summed up, and
    saved with its facts for Stats; None when stopped part way. The review room's, and the
    background reviewer's (ui/backlog.py) with its engines at a low priority."""
    games = library.games_in(row.pgn)
    if not games:
        raise ValueError("the game's PGN has no moves")
    path = find_stockfish(settings)
    if path is None:
        raise RuntimeError("no Stockfish to review with")
    engine = settings.engine
    with contextlib.ExitStack() as stack:
        engines = [
            stack.enter_context(
                (make_engine or Engine)(
                    path=path,
                    threads=THREADS,
                    hash_mb=min(engine.hash_mb, 128),
                    multipv=strength.multipv,
                    pv_plies=engine.pv_plies,
                    score_model=engine.score_model,
                    max_seconds=engine.max_seconds,
                    low_priority=low_priority,
                )
            )
            for _ in range(workers)
        ]
        from sidecoach.cli import _load_book, _load_tagger

        reviewed = review_game(
            games[0],
            engines,
            depth=strength.depth,
            multipv=strength.multipv,
            thresholds=Thresholds.from_settings(settings.coach),
            tagger=_load_tagger(),
            book=_load_book(settings),
            strength=strength.name,
            game_id=row.key,
            on_progress=on_progress,
            stop=stop,
        )
    if stop is not None and stop.is_set():
        return None
    summary = summarize(reviewed)
    store.save_review(
        row.key,
        strength.name,
        REVIEW_VERSION,
        time.time(),
        reviewed.model_dump_json(),
        summary.model_dump_json(),
    )
    keep_stats(store, strength.name, reviewed, row)
    return reviewed, summary


def _opening(reviewed: ReviewedGame) -> str | None:
    """The opening the game reached, as the book last named it."""
    return next((m.after.opening for m in reversed(reviewed.moves) if m.after.opening), None)


def move_views(reviewed: ReviewedGame) -> list[dict[str, Any]]:
    """Every move as the review page draws it: the move and its squares, its label and what it
    cost, the best move and its line, what follows the move (the best line of the position it
    led to), and the position after -- its FEN, the evaluation from White's side (centipawns
    or mate) and White's expected score. A line comes as steps the board plays: each move in
    SAN and UCI and the position it leaves."""
    out = []
    for move in reviewed.moves:
        before = chess.Board(move.fen_before)
        played = chess.Move.from_uci(move.assessment.uci)
        after = before.copy(stack=False)
        after.push(played)
        best = move.assessment.best
        line = _steps(before, best[:LINE_PLIES])
        follow = move.after.lines[0].moves[:LINE_PLIES] if move.after.lines else []
        out.append(
            {
                "ply": move.ply,
                "san": move.san,
                "side": move.side.value,
                "uci": move.assessment.uci,
                "from": chess.square_name(played.from_square),
                "to": chess.square_name(played.to_square),
                "fen_before": move.fen_before,
                "fen_after": after.fen(),
                "label": move.assessment.label.value if move.assessment.label else None,
                "accuracy": move.assessment.accuracy,
                "loss": round(move.assessment.expected_loss, 4),
                "best_uci": best[0] if best else None,
                "best_san": line[0]["san"] if line else None,
                "best_line": line,
                "follow_line": _steps(after, follow),
                # What the move threatens: the mover's best next move were it their turn again.
                "idea": move.after.threats[0] if move.after.threats else None,
                "eval": _white_eval(after, move.after.lines),
                "white_score": round(_white_score(after, move.after.lines), 4),
                "tags": sorted({t.name for t in move.assessment.tags}),
                "opening": move.after.opening,
                "clock_ms": move.clock_ms,
                "spent_ms": move.spent_ms,
            }
        )
    return out


def _steps(board: chess.Board, ucis: list[str]) -> list[dict[str, str]]:
    """A line of moves from `board` as far as it stays legal, each in SAN and UCI with the
    position it leaves."""
    b = board.copy(stack=False)
    out = []
    for uci in ucis:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in b.legal_moves:
            break
        san = b.san(move)
        b.push(move)
        out.append({"san": san, "uci": uci, "fen": b.fen()})
    return out


def _white_score(board: chess.Board, lines: list[Line]) -> float:
    score = position_score(board, lines)
    return score if board.turn == chess.WHITE else 1.0 - score


def _white_eval(board: chess.Board, lines: list[Line]) -> dict[str, int | str | None]:
    """The evaluation from White's side, as the eval bar and pill read it: a mate in moves
    (+ White mates, - Black mates) or centipawns; a finished game, its result instead."""
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return {"cp": None, "mate": None, "result": outcome.result()}
    if not lines:
        return {"cp": 0, "mate": None, "result": None}
    sign = 1 if board.turn == chess.WHITE else -1
    line = lines[0]
    if line.mate is not None:
        return {"cp": None, "mate": sign * line.mate, "result": None}
    return {"cp": sign * (line.score_cp or 0), "mate": None, "result": None}


class ReviewRooms:
    """The open review rooms by library key, and the model their words are asked of."""

    def __init__(
        self,
        settings: Settings,
        *,
        store_path: Path | None,
        make_engine: Callable[..., Engine] | None = None,
        mirror: Callable[..., None] | None = None,
    ) -> None:
        self.settings = settings
        self.store_path = store_path
        self._make_engine = make_engine
        self._mirror = mirror
        self._rooms: dict[str, ReviewRoom] = {}
        self._lock = threading.Lock()
        # The model as the settings name it, made once for them.
        self._model: Any | None = None
        self._model_for: tuple[Any, ...] | None = None

    def model(self) -> tuple[Any | None, str | None]:
        """The model the reviews' words are asked of, and why there is none."""
        llm = self.settings.llm
        if llm.provider == "none":
            return None, NO_AI
        named = (llm.provider, llm.model, llm.base_url, llm.reasoning_effort)
        with self._lock:
            if named != self._model_for:
                from sidecoach.llm.factory import make_provider

                try:
                    self._model = make_provider(llm, timeout_s=AI_TIMEOUT_S)
                except (RuntimeError, ValueError) as exc:  # a key not given, a name not known
                    return None, str(exc)
                self._model_for = named
            return self._model, None

    def usages(self) -> list[Any]:
        """The model calls of the reviews' words (llm.usage.Usage), for the panel's strip."""
        with self._lock:
            usage = getattr(self._model, "usage", None)
        return [usage] if usage is not None else []

    def row(self, key: str) -> LibraryRow | None:
        """A game of the library by its key: imported, or one Sidecoach watched."""
        store = Store(self.store_path)
        try:
            return library.find(store, key)
        finally:
            store.close()

    def busy(self) -> bool:
        """Whether a room is reviewing a game now (the background reviewer waits for it)."""
        with self._lock:
            return any(r.status == "analysing" and not r.closed for r in self._rooms.values())

    def open(self, key: str, strength: str = "standard") -> ReviewRoom | None:
        with self._lock:
            room = self._rooms.get(key)
            if room is not None and not room.closed and room.strength.name == strength:
                return room
            row = self.row(key)
            if row is None:
                return None
            if room is not None:
                room.close()
            room = ReviewRoom(
                self.settings,
                row,
                store_path=self.store_path,
                strength=strength,
                make_engine=self._make_engine,
                mirror=self._mirror,
                model=self.model,
            )
            self._rooms[key] = room
            return room

    def get(self, key: str) -> ReviewRoom | None:
        with self._lock:
            room = self._rooms.get(key)
        return room if room is not None and not room.closed else None
