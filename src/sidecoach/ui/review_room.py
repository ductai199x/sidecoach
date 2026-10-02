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
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from sidecoach.analysis.annotator import Measured
from sidecoach.analysis.engine import Engine
from sidecoach.analysis.game_review import (
    STRENGTHS,
    THREADS,
    Strength,
    clocks,
    measured_from_json,
    measured_json,
    positions,
    review_game,
    workers_for,
)
from sidecoach.analysis.score import Thresholds, position_score
from sidecoach.coach import review_links, review_words
from sidecoach.config import Settings, find_stockfish
from sidecoach.events import GameSummary, Line, ReviewedGame, Side
from sidecoach.llm.usage import Usage, prices_for
from sidecoach.profile import library
from sidecoach.profile.game_summary import missed_motifs, summarize
from sidecoach.profile.stat_facts import keep_stats
from sidecoach.profile.store import STRENGTH_RANK, WATCHED, LibraryRow, ReviewRow, Store
from sidecoach.ui import review_analysis
from sidecoach.ui.log import DebugLog
from sidecoach.ui.review_retry import Retries, measured

log = logging.getLogger(__name__)

# Why a room has nothing more to do: its game was removed (or the profile reset) meanwhile.
GONE = "This game is no longer in your record."
# The review's format: a review saved in an older one is done again.
REVIEW_VERSION = 1
# What a review's summary and its Stats facts are made of from its report (`rederive`): raised
# when that changes, the reviews kept. 2: the phases are Lichess's Divider's (efdd8a3).
DERIVED = 2
DERIVED_KEY = "reviews.derived"
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


class NoStockfish(RuntimeError):
    """No Stockfish to review with yet -- a first run, Setup still fetching it: no game's fault."""


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
        # The answers paid for by ply, and what they cost, while words are being written or
        # after the app was closed in the middle of them: kept, and never asked again.
        self._answered: dict[int, str] = {}
        self._spent = 0.0
        # The words paid for on the watched game this one stands for, by this review's plies
        # (`carried_words`): shown with this game's own, never asked again.
        self._carried: dict[int, dict[str, str]] = {}
        self._carried_model = ""
        # The words' cost as the button says it, by model name and the moves answered already.
        self._estimated: tuple[tuple[str | None, frozenset[int]], float | None] | None = None
        self._playback = playback(row.pgn)
        # Analysis: the engine made on the first position asked of it, the explorer on the first
        # explore, the opening book on the first opening named (False: none to read).
        self._lines = review_analysis.Lines(settings, self.strength, make_engine)
        self._explorer: review_analysis.Explorer | None = None
        self._book: Any = None
        self.touched = time.monotonic()
        self.closed = False
        # Set once the review is ready or failed, and while no words are being written: what a
        # caller waits on rather than polling the state against a clock.
        self.settled = threading.Event()
        self.words_done = threading.Event()
        self.words_done.set()
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
                # While the game is analysed, its positions: the page's board plays the game out
                # as the progress goes, as chess.com's review does (Tai, 2026-09-30).
                "playback": self._playback if self.status in ("starting", "analysing") else None,
                "summary": self.summary.model_dump(mode="json") if self.summary else None,
                "moves": self._moves,
                "practice": self.practice,
                "ai": {
                    "status": self.words_status,
                    "error": self.words_error,
                    "words": self.words,
                    "unavailable": unavailable,
                    "moves": len(review_words.key_plies(self.reviewed)) if self.reviewed else 0,
                    "answered": len(self._answered) if self.words_status != "ready" else 0,
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
        answered = frozenset(self._answered) | frozenset(self._carried)
        if self._estimated is None or self._estimated[0] != (name, answered):
            opening = _opening(self.reviewed)
            prices = prices_for(self.settings, name)
            cost = review_words.estimate(
                self.reviewed, self.row.user_side, opening, prices, answered
            )
            self._estimated = ((name, answered), cost)
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
            if not self.store.in_record(self.key):
                # Removed meanwhile: the words would be paid for and saved on nothing (the
                # review of 2026-10-01).
                return GONE
            self.words_status, self.words_error = "writing", None
            self.words_done.clear()
            reviewed = self.reviewed
        threading.Thread(
            target=self._write_words, args=(reviewed, model), name=f"words-{self.key}", daemon=True
        ).start()
        return None

    def _write_words(self, reviewed: ReviewedGame, model: Any) -> None:
        try:
            self._words(reviewed, model)
        finally:
            self.words_done.set()

    def _words(self, reviewed: ReviewedGame, model: Any) -> None:
        # A model that reasons by its effort (luna) reasons at REVIEW_EFFORT here, whatever the
        # live coach is set to; a local model does not reason: it reasons worse about chess,
        # not better (Tai, 2026-09-30).
        reasons = getattr(model, "thinking_control", None) == "effort"
        if reasons:
            model = model.with_effort(REVIEW_EFFORT)
        prices = prices_for(self.settings, getattr(model, "model", None))
        with self._lock:
            answered, spent_before = dict(self._answered), self._spent
            carried = frozenset(self._carried)

        def keep(ply: int, text: str, stats: dict[str, Any]) -> None:
            # Each answer kept the moment it is paid for: the app closed now loses none of them.
            one = Usage()
            one.record(stats)
            with self._lock:
                self._answered[ply] = text
                self._spent += one.cost(prices) or 0.0
                partial = {
                    "partial": True,
                    "model": getattr(model, "model", ""),
                    "cost": self._spent,
                    "answers": {str(p): t for p, t in self._answered.items()},
                }
            self.store.set_review_words(self.key, self.strength.name, json.dumps(partial))

        try:
            words = review_words.write(
                reviewed,
                model,
                user=self.row.user_side,
                opening=_opening(reviewed),
                prices=prices,
                thinking=reasons,
                answered=answered,
                on_answer=keep,
                stop=lambda: not self.store.in_record(self.key),
                carried=carried,
            )
            if not self.store.in_record(self.key):
                raise RuntimeError(GONE)
            if words.failed:
                # Some calls failed: the answers paid for are kept (`keep`), and Try again asks
                # the rest. They were saved as done, the failed moves without words for good and
                # nothing offered to finish them (the review of 2026-10-01).
                total = len(review_words.key_plies(reviewed))
                first = next(iter(words.dropped[p] for p in sorted(words.failed)))
                with self._lock:
                    self.words_status = "failed"
                    self.words_error = (
                        f"Written on {total - len(words.failed)} of {total} key moves; the rest"
                        f" failed ({first})."
                    )
                self._emit("sys", "error", f"the AI coach's words: {self.words_error}")
                return
            cost = None if words.cost is None else words.cost + spent_before
            saved = {
                "model": words.model,
                "cost": cost,
                "texts": {str(ply): parts for ply, parts in words.texts.items()},
            }
            shown = self._shown(saved, reviewed)
        except Exception as exc:  # noqa: BLE001 - a failure is said, never left "writing"
            # Whether the game went is asked of the store, which may be what failed: asked
            # unguarded, its error ended the thread with the words "writing" for good, and the
            # room, waiting on them, never closed (the audit of 2026-10-02).
            try:
                gone = not self.store.in_record(self.key)
            except Exception:  # noqa: BLE001 - the failure said is the words'
                gone = False
            with self._lock:
                self.words_status, self.words_error = "failed", GONE if gone else str(exc)
            self._emit("sys", "error", f"the AI coach's words failed: {exc!r}")
            return
        # The room stays open while they are written (`_run`), so its store is open to keep them.
        try:
            self.store.set_review_words(self.key, self.strength.name, json.dumps(saved))
        except Exception as exc:  # noqa: BLE001 - shown even so; said, never left "writing"
            self._emit("sys", "error", f"the AI coach's words could not be saved: {exc!r}")
        with self._lock:
            self.words, self.words_status = shown, "ready"
            self._answered, self._spent = {}, 0.0
        cost = f", ${saved['cost']:.4f}" if saved["cost"] is not None else ""
        self._emit(
            "sys",
            "info",
            f"AI words on {len(words.texts)} of {len(review_words.key_plies(reviewed))} key "
            f"moves ({words.model}{cost})",
        )
        for ply, why in sorted(words.dropped.items()):
            self._emit("sys", "warn", f"AI words on ply {ply} dropped: {why}")

    def _shown(self, saved: dict[str, Any], reviewed: ReviewedGame) -> dict[str, Any]:
        """The saved words as the page draws them, with the words carried from the watched game
        this one stands for on the moves that have none of their own: each paragraph also as its
        links to the board (coach/review_links.py), made on showing so they follow the linking's
        rules."""
        own = saved.get("texts", {})
        carried = {str(p): parts for p, parts in self._carried.items() if str(p) not in own}
        texts = (
            dict(sorted({**carried, **own}.items(), key=lambda t: int(t[0]))) if carried else own
        )
        links = review_links.linked(texts, reviewed, self.row.user_side)
        return {**saved, "texts": texts, "links": links}

    def _carry(self, reviewed: ReviewedGame) -> dict[str, Any] | None:
        """The words paid for on the watched game this one stands for, by this review's plies,
        kept for showing and for the calls not to ask again; shown as the words when this game
        has none of its own yet (None when there are none)."""
        try:
            carried, model = carried_words(self.store, self.key, reviewed, self.strength.name)
        except Exception as exc:  # noqa: BLE001 - the review stands without them; said
            self._emit("sys", "error", f"the watched game's AI words could not be read: {exc!r}")
            carried, model = {}, ""
        with self._lock:
            self._carried, self._carried_model = carried, model
        if not carried:
            return None
        self._emit("sys", "info", f"AI words on {len(carried)} moves carried from the watched game")
        return self._shown({"model": model, "cost": None, "texts": {}}, reviewed)

    # ---- analysis (ui/review_analysis.py): the engine's lines for the board, the explorer, the
    # game as a PGN ---------------------------------------------------------------------------

    def lines(self, fen: str) -> dict[str, Any]:
        """The engine's best lines for a position of the analysis board."""
        self.touched = time.monotonic()
        return self._lines.lines(fen)

    def explore(self, fen: str, moves: list[str]) -> dict[str, Any]:
        """What the player's games went on with from `fen`, and the opening `moves` (UCI, from
        the game's start) are in."""
        self.touched = time.monotonic()
        with self._lock:
            if self._explorer is None:
                twins = self.store.twins()
                watched = [r for r in library.from_watched(self.store) if r.key not in twins]
                self._explorer = review_analysis.Explorer([*self.store.library(), *watched])
            explorer = self._explorer
        opening = self._opening_of(moves)
        return {**explorer.explore(fen), "opening": opening, "book": bool(self._book)}

    def pgn(self, tree: list[dict[str, Any]], *, annotated: bool, save: bool) -> dict[str, Any]:
        """The game as a PGN with the analysis board's lines (`tree`), annotated from the review
        when asked; kept in the Downloads folder when `save`."""
        self.touched = time.monotonic()
        with self._lock:
            moves = self._moves if self.status == "ready" else None
            words = self.words
        text = review_analysis.pgn(self.row, moves, tree, annotated=annotated, words=words)
        out: dict[str, Any] = {"pgn": text, "name": review_analysis.pgn_name(self.row)}
        if save:
            out["saved"] = str(review_analysis.save(text, out["name"]))
        return out

    def _opening_of(self, moves: list[str]) -> str | None:
        """The opening the moves are in, by the book, from the game's start position."""
        games = library.games_in(self.row.pgn)
        if not games or games[0].board().fen() != chess.STARTING_FEN:
            return None
        if self._book is None:
            from sidecoach.cli import _load_book

            self._book = _load_book(self.settings) or False
        if not self._book:
            return None
        board, played = chess.Board(), []
        for uci in moves:
            try:
                m = chess.Move.from_uci(uci)
            except ValueError:
                break
            if m not in board.legal_moves:
                break
            board.push(m)
            played.append(m)
        found = self._book.identify(played) if played else None
        if found is None:
            return None
        return f"{found.eco} {found.name}" if found.eco else str(found.name)

    def close(self) -> None:
        """The page left. The room closes at its next tick -- once the AI coach's words being
        written are saved, if they are (`_run`)."""
        self._stop.set()

    def reopen(self) -> bool:
        """Opened again: whether this room goes on. One closed but still open -- its words being
        written, or its next tick not come -- stays open; one closed, or whose analysis is
        stopping, does not (a new room goes on from what it kept)."""
        with self._lock:
            if self.closed:
                return False
            if not self._stop.is_set():
                return True
            if self.status != "ready":
                return False
            self._stop.clear()
            return True

    def closing(self, now: float) -> bool:
        """Whether the room ends at `now`: closed, or idle, and no words being written. Closed or
        idle while the AI coach's words are written, it stays open until they are saved: its
        store closed under them lost every answer that came after, each one paid for (the review
        of 2026-10-01)."""
        writing = self.words_status == "writing"
        idle = now - self.touched > IDLE_CLOSE_S
        return not writing and (self._stop.is_set() or idle)

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
            self.settled.set()
            while True:
                with self._lock:
                    stopped = self._stop.is_set()
                    if self.closing(time.monotonic()):
                        self.closed = True
                        break
                if stopped:
                    time.sleep(TICK_S)
                else:
                    self._stop.wait(TICK_S)
            if not stopped:
                self._emit("sys", "info", "room idle; closing")
        finally:
            self.closed = True
            self.store.close()
            if self.retries is not None:
                self.retries.close()
            self._lines.close()

    def _load(self) -> bool:
        """The review saved at this strength, when it is of the current format."""
        saved = self.store.review(self.key, self.strength.name)
        if saved is None or saved.version != REVIEW_VERSION:
            return False
        reviewed = ReviewedGame.model_validate_json(saved.report)
        with contextlib.suppress(Exception):  # a kill between the save and the clearing
            self.store.clear_progress(self.key, self.strength.name)
        # Its words ready with it: the page stops asking once the review is. Words the app was
        # closed in the middle of writing are the answers paid for so far, kept to go on from.
        kept = json.loads(saved.words) if saved.words else None
        words = None
        if kept and kept.get("partial"):
            self._answered = {int(p): str(t) for p, t in kept.get("answers", {}).items()}
            self._spent = float(kept.get("cost") or 0.0)
        elif kept:
            words = kept
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
            if not self._stop.is_set():  # not stopped: the game went from the record meanwhile
                with self._lock:
                    self.status = "failed"
                    self.error = GONE
                self._emit("sys", "warn", f"{self.key} left the record while it was reviewed")
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
        self, reviewed: ReviewedGame, summary: GameSummary, saved: dict[str, Any] | None = None
    ) -> None:
        """The review ready, with its saved words (`saved`) as the page draws them, and the words
        carried from the watched game it stands for: the words are done once every key move has
        some, else the rest are there to ask for."""
        moves = move_views(reviewed)
        practice = self._practice(reviewed)
        carried = self._carry(reviewed)
        words = self._shown(saved, reviewed) if saved is not None else carried
        every = set(review_words.key_plies(reviewed)) <= set(self._carried)
        done = saved is not None or (carried is not None and every)
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
                self.words = words
            if done:
                self.words_status = "ready"
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
    background reviewer's (ui/backlog.py) with its engines at a low priority. Every position is
    kept as it is measured (Store.add_progress), so a review stopped part way -- the room
    closed, the app quit or killed -- goes on from there next time instead of starting over
    (Tai, 2026-09-30); what was kept goes once the review is saved."""
    games = library.games_in(row.pgn)
    if not games:
        raise ValueError("the game's PGN has no moves")
    path = find_stockfish(settings)
    if path is None:
        raise NoStockfish("no Stockfish to review with")
    engine = settings.engine
    known: dict[str, Measured] = {}
    for text in store.progress(row.key, strength.name, strength.depth, strength.multipv):
        with contextlib.suppress(ValueError):  # a record that no longer reads is measured again
            kept = measured_from_json(text)
            known[kept.fen] = kept
    # The watched game this import stands for, reviewed at this strength: what its review
    # measured is not searched again (its record differs from the import's, or carry_twins
    # would have taken the review whole).
    for watched, imported in store.twins().items():
        twin = store.review(watched, strength.name) if imported == row.key else None
        if twin is None or twin.version != REVIEW_VERSION:
            continue
        with contextlib.suppress(ValueError):  # a review that no longer reads gives nothing
            was = ReviewedGame.model_validate_json(twin.report)
            if was.depth == strength.depth:
                for analysis in [was.start, *(m.after for m in was.moves)]:
                    known.setdefault(analysis.fen, measured(analysis))

    def keep(position: Measured) -> None:
        store.add_progress(
            row.key,
            strength.name,
            strength.depth,
            strength.multipv,
            position.fen,
            measured_json(position),
        )

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
            known=known,
            on_measured=keep,
        )
    if stop is not None and stop.is_set():
        return None
    if not store.in_record(row.key):
        # Taken out of the player's record while it was reviewed: removed, or the profile reset
        # (the review of 2026-10-01: a review saved after a reset left Stats a game that nothing
        # listed and nothing could remove).
        store.clear_progress(row.key)
        return None
    summary = summarize(reviewed)
    saved = store.save_review(
        row.key,
        strength.name,
        REVIEW_VERSION,
        time.time(),
        reviewed.model_dump_json(),
        summary.model_dump_json(),
    )
    if not saved:
        # Another review of the game at this strength finished first -- the background review
        # and a review window at once -- and stays, with any words the player paid for on it
        # (the review of 2026-10-01): this one is handed that one. None if the game was removed.
        kept = store.review(row.key, strength.name)
        if kept is None:
            store.clear_progress(row.key)
            return None
        store.clear_progress(row.key, strength.name)
        return (
            ReviewedGame.model_validate_json(kept.report),
            GameSummary.model_validate_json(kept.summary or "{}"),
        )
    # What this strength's review kept part way goes; another strength's stays, to go on from.
    store.clear_progress(row.key, strength.name)
    keep_stats(store, strength.name, reviewed, row, held=True)
    return reviewed, summary


def rederive(store: Store) -> int:
    """Every saved review's summary, and its facts for Stats, made again from its report when
    what they are made of has changed since (DERIVED): no engine, a few hundredths of a second
    a game. The phases became Lichess's (efdd8a3) with the reviews saved as they were: 98 of 106
    sides' phase grades and 745 of 3,522 moves' phases in Stats were the old rule's (the review
    of 2026-10-01). Returns the reviews made again."""
    if store.setting(DERIVED_KEY) == str(DERIVED):
        return 0
    counted = store.stat_strengths()
    done = 0
    for key, strength in store.saved_reviews():
        saved = store.review(key, strength)
        if saved is None or saved.version != REVIEW_VERSION:
            continue
        try:
            reviewed = ReviewedGame.model_validate_json(saved.report)
            store.set_review_summary(key, strength, summarize(reviewed).model_dump_json())
            row = library.find(store, key)
            if counted.get(key) == strength and row is not None:
                keep_stats(store, strength, reviewed, row, held=True)
        except Exception as exc:  # noqa: BLE001 - a review that cannot be read is left as it is
            log.warning(
                "the saved review of %s (%s) could not be summed up again: %r", key, strength, exc
            )
            continue
        done += 1
    store.set_setting(DERIVED_KEY, str(DERIVED))
    return done


def carried_words(
    store: Store, key: str, reviewed: ReviewedGame, strength: str
) -> tuple[dict[int, dict[str, str]], str]:
    """The AI coach's words paid for on the games this one stands for (a watched game its import
    stands for, whose record differs from it, so its review was not carried whole), by this
    review's plies, and the model that wrote them. A word goes to the move played from the same
    position (the FEN's board, side, castling and en passant: a game joined late counts its
    moves from another number) with the same move; the n-th time a position and move come in
    one game, to their n-th time in the other. Words on moves the two games do not share stay
    with the watched game's own review, where they were saved: nothing is moved or deleted (Tai,
    2026-10-02). The review at this strength first, then the strongest; a move's first words
    found stand."""
    sources = [watched for watched, imported in store.twins().items() if imported == key]
    if not sources:
        return {}, ""
    plies: dict[tuple[str, str], list[int]] = {}
    for move in reviewed.moves:
        plies.setdefault((_position(move.fen_before), move.assessment.uci), []).append(move.ply)
    out: dict[int, dict[str, str]] = {}
    model = ""
    for source in sources:
        reviews = store.reviews_of(source)
        reviews.sort(key=lambda r: r.strength != strength)  # this strength first, stable
        for old in reviews:
            try:
                words = json.loads(old.words) if old.words else None
                if not words or words.get("partial") or not words.get("texts"):
                    continue
                was = ReviewedGame.model_validate_json(old.report)
            except ValueError:
                continue  # a review or words that no longer read give nothing
            texts = words["texts"]
            seen: dict[tuple[str, str], int] = {}
            for move in was.moves:
                at = (_position(move.fen_before), move.assessment.uci)
                nth = seen.get(at, 0)
                seen[at] = nth + 1
                said = texts.get(str(move.ply))
                mine = plies.get(at, [])
                if said and nth < len(mine) and mine[nth] not in out:
                    out[mine[nth]] = dict(said)
                    model = model or str(words.get("model") or "")
    return out, model


def _position(fen: str) -> str:
    """A FEN's position: the board, the side to move, castling and en passant."""
    return " ".join(fen.split()[:4])


def carry_twins(store: Store, twins: dict[str, str]) -> list[str]:
    """Each watched game's review carried to its import when the import has none and the two
    are the same game position for position (Sidecoach read every move): the watched review's
    measurements with the import's own headers and clocks, summed up and saved with its facts
    for Stats at once, no engine -- pairing left the game with no accuracy until it was reviewed
    again (Tai, 2026-09-30). A record that differs is reviewed as the import, from the positions
    the watched review measured (`review_row`). Returns the imports given a review.

    The same for a copy of a game that stands aside for another (`library.pair_copies`: a game
    pasted and imported), which is the same game move for move, with one more rule: nothing
    the player has of either is lost -- a review the survivor lacks is carried when it is
    stronger or has the AI coach's words (paid for), and words one has that the other has not
    go to the survivor (Tai, 2026-10-02)."""
    carried = []
    for watched, imported in twins.items():
        copy = not watched.startswith(WATCHED)
        best = store.best_review(imported)
        if best is not None and not copy:
            continue
        row = library.find(store, imported)
        if row is None:
            continue
        mine = store.reviews_of(watched)
        for old in mine if copy else mine[:1]:
            if old.version != REVIEW_VERSION:
                continue
            kept = store.review(imported, old.strength) if copy else None
            if kept is not None:
                if old.words and not kept.words and kept.version == old.version:
                    store.set_review_words(imported, old.strength, old.words)
                continue
            if (
                best is not None
                and not old.words
                and STRENGTH_RANK[old.strength] <= STRENGTH_RANK[best.strength]
            ):
                continue
            if _carry_one(store, old, imported, row):
                carried.append(imported)
                best = store.best_review(imported)
    return carried


def _carry_one(store: Store, old: ReviewRow, imported: str, row: LibraryRow) -> bool:
    """One review of a game carried to the game that stands for it (`carry_twins`); whether it
    was."""
    games = library.games_in(row.pgn)
    try:
        was = ReviewedGame.model_validate_json(old.report)
    except ValueError:
        return False
    if not games or not was.moves:
        return False
    game = games[0]
    seen = [m.fen_before for m in was.moves] + [was.moves[-1].after.fen]
    if seen != [b.fen() for b in positions(game)]:
        return False
    moves = [
        m.model_copy(
            update={
                "assessment": m.assessment.model_copy(update={"game_id": imported}),
                "after": m.after.model_copy(update={"game_id": imported}),
                "clock_ms": clock_ms,
                "spent_ms": spent_ms,
            }
        )
        for m, (clock_ms, spent_ms) in zip(was.moves, clocks(game), strict=True)
    ]
    reviewed = was.model_copy(
        update={
            "headers": {k: str(v) for k, v in game.headers.items()},
            "start": was.start.model_copy(update={"game_id": imported}),
            "moves": moves,
        }
    )
    if not store.save_review(
        imported,
        old.strength,
        REVIEW_VERSION,
        time.time(),
        reviewed.model_dump_json(),
        summarize(reviewed).model_dump_json(),
    ):
        return False  # reviewed meanwhile (that review stays, with its words), or removed
    if old.words:
        store.set_review_words(imported, old.strength, old.words)
    store.clear_progress(imported, old.strength)
    keep_stats(store, old.strength, reviewed, row, held=True)
    return True


def playback(pgn: str) -> list[dict[str, str | None]]:
    """The game's main line as the board plays it while the game is analysed: the position
    before any move, then each move (UCI) and the position it leaves; none for a PGN with no
    game in it."""
    games = library.games_in(pgn)
    if not games:
        return []
    board = games[0].board()
    out: list[dict[str, str | None]] = [{"fen": board.fen(), "uci": None}]
    for move in games[0].mainline_moves():
        board.push(move)
        out.append({"fen": board.fen(), "uci": move.uci()})
    return out


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
                "severity": move.assessment.severity.value,
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
        self._spent_ledgers: list[Any] = []  # the ledgers of the models replaced

    def model(self) -> tuple[Any | None, str | None]:
        """The model the reviews' words are asked of, and why there is none."""
        from sidecoach.llm.factory import api_key_of, make_provider

        llm = self.settings.llm
        if llm.provider == "none":
            return None, NO_AI
        # The key too: a key fixed or forgotten in Settings left the model made with the old
        # one until a restart (the settings review, 2026-10-01).
        named = (llm.provider, llm.model, llm.base_url, llm.reasoning_effort, api_key_of(llm))
        with self._lock:
            if named != self._model_for:
                old = getattr(self._model, "usage", None)
                try:
                    self._model = make_provider(llm, timeout_s=AI_TIMEOUT_S)
                    if old is not None:
                        self._spent_ledgers.append(old)  # its calls still count
                except (RuntimeError, ValueError) as exc:  # a key not given, a name not known
                    return None, str(exc)
                self._model_for = named
            return self._model, None

    def usages(self) -> list[Any]:
        """The model calls of the reviews' words (llm.usage.Usage), for the panel's strip."""
        with self._lock:
            usage = getattr(self._model, "usage", None)
            return [*self._spent_ledgers, *([usage] if usage is not None else [])]

    def row(self, key: str) -> LibraryRow | None:
        """A game of the library by its key: imported, or one Sidecoach watched."""
        store = Store(self.store_path)
        try:
            return library.find(store, key)
        finally:
            store.close()

    def open(self, key: str, strength: str | None = None) -> ReviewRoom | None:
        """The game's room at `strength`; with none, at the strongest review saved (else
        Standard). It opened at Standard always: a game reviewed Deep was reviewed again at
        Standard, its Stats overwritten (the review of 2026-10-01)."""
        if strength is None:
            store = Store(self.store_path)
            try:
                best = store.best_review(key)
            finally:
                store.close()
            strength = best.strength if best is not None and best.strength in STRENGTHS else None
            strength = strength or "standard"
        with self._lock:
            room = self._rooms.get(key)
            if room is not None and room.strength.name == strength and room.reopen():
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
