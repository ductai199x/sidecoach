"""A puzzle in its own tab (`/puzzle/<id>`), apart from the screen the panel watches.

Tai, 2026-09-20: puzzle mode "should open in a new tab with a new UI and not interfere
with the current window where the game capture is going on". So a puzzle room has its
own coach (quiet unless asked), its own engine, its own log and its own little loop
thread; the main engine goes on following the screen. The room drives a PuzzleSession
(ui/puzzle.py) -- the board, the line, the outcome -- and the page polls it the way the
panel polls the engine. The outcome goes to the profile store beside the puzzle's motif
(profile/puzzles.record_attempt). The coach speaks through the voice the panel was
given, when the page has it on.
"""

from __future__ import annotations

import contextlib
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sidecoach.config import Settings
from sidecoach.events import MovePlayed, Trigger
from sidecoach.profile.puzzles import Puzzle, PuzzleStore, puzzle_level, record_attempt
from sidecoach.ui.log import DebugLog, elide
from sidecoach.ui.puzzle import PuzzleSession

_MARKUP = re.compile(r"\[/?[a-z ]+\]")


def _plain(text: str) -> str:
    return _MARKUP.sub("", text)


# The loop's pace, and how long an untouched room lives.
TICK_S = 0.25
IDLE_CLOSE_S = 20 * 60.0


class PuzzleRoom:
    def __init__(
        self,
        settings: Settings,
        puzzle: Puzzle,
        motif: str | None,
        *,
        store_path: Path | None,
        speak: Callable[[str], None] | None = None,
        stop_speaking: Callable[[], None] | None = None,
        analysis: bool = True,
        coach: bool = True,
        mirror: Callable[..., None] | None = None,
    ) -> None:
        from sidecoach.cli import _Assessing, _Coaching
        from sidecoach.profile.store import Store

        self.settings = settings
        self.session = PuzzleSession(puzzle, motif)
        self.store = Store(store_path)
        self._log = DebugLog()
        # The parent panel's log, when there is one: every line of this room goes there
        # too, under a "puzzle" channel, so the puzzle window itself shows no terminal
        # (Tai, 2026-09-21: the debugging stays in the parent window).
        self._mirror = mirror
        self._id = puzzle.id
        self._lock = threading.Lock()
        self._commands: list[tuple[str, Any]] = []
        self._stop = threading.Event()
        self._speak = speak
        self._stop_speaking = stop_speaking
        self.voice = speak is not None
        self.touched = time.monotonic()
        self.closed = False
        self.closing = False  # told to close; the loop thread finishes it within a tick
        # The engine and the coach of this room alone; the coach is told the position as
        # a game of kind "puzzle" and kept quiet: it speaks when asked.
        self._assessing = _Assessing(settings, analysis)
        self._coaching = _Coaching(settings, coach and self._assessing.on)
        # The engine's lines are written for the console (rich markup); plain here.
        self._assessing.sink = lambda line: self._emit("engine", "debug", _plain(line))
        self._coaching.sink = self._on_message
        if self._coaching.coach is not None:
            self._coaching.coach.quiet = True
        self._emit("sys", "info", f"engine: {self._assessing.detail}")
        self._emit("sys", "info", f"coach: {self._coaching.detail}")
        self._emit(
            "sys",
            "info",
            f"puzzle {puzzle.id}: {motif or 'practice'}, rated {puzzle.rating}, "
            f"{self.session.side.value} to move; level {puzzle_level(self.store)}",
        )
        started = self.session.started()
        self._assessing.submit(started)
        self._coaching.events([started], time.monotonic())
        self._thread = threading.Thread(target=self._run, name=f"puzzle-{puzzle.id}", daemon=True)
        self._thread.start()

    # ---- the page's side --------------------------------------------------------------------

    def poll(self, since: int = 0) -> dict[str, Any]:
        return {"state": self.state(), "lines": self._log.since(since), "cursor": self._log.cursor}

    def state(self) -> dict[str, Any]:
        self.touched = time.monotonic()
        with self._lock:
            state = self.session.state()
        state["voice"] = self.voice
        state["voice_ready"] = self._speak is not None
        state["closed"] = self.closed
        usage = getattr(getattr(self._coaching.coach, "speaker", None), "provider", None)
        usage = getattr(usage, "usage", None)
        if usage is not None:
            from sidecoach.llm.usage import prices_for

            model = getattr(self._coaching.coach.speaker.provider, "model", None)  # type: ignore[union-attr]
            state["llm"] = usage.summary(prices_for(self.settings, model))
        return state

    # A move, a take-back and a give-up are played on the page's request and answered with
    # the state they leave: they waited for the loop's next tick, and the page for its next
    # poll after that, half a second the board stood still after a move (Tai, 2026-09-27).
    # What the engine and the coach are told goes through the loop as before.

    def move(self, uci: str) -> dict[str, Any]:
        with self._lock:
            events = self.session.move(uci)
            self._commands.append(("played", (uci, events)))
        return self.state()

    def retract(self) -> dict[str, Any]:
        with self._lock:
            self.session.retract()
        return self.state()

    def give_up(self) -> dict[str, Any]:
        with self._lock:
            events = self.session.give_up()
            self._commands.append(("gave_up", events))
        return self.state()

    def command(self, name: str, payload: Any = None) -> None:
        """A command for the loop thread: ask <text> | voice <bool> | close."""
        self.touched = time.monotonic()
        if name == "close":
            # Said at once, so a page reloading (its old tab's close, then its open)
            # gets a fresh room rather than this one on its way out.
            self.closing = True
        with self._lock:
            self._commands.append((name, payload))

    # ---- the loop ----------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ts = time.monotonic()
                self._drain_commands(ts)
                engine_events = self._assessing.drain()
                self._coaching.events(engine_events, ts)
                self._coaching.tick(ts)
                self._coaching.drain()
                if time.monotonic() - self.touched > IDLE_CLOSE_S:
                    self._emit("sys", "info", "room idle; closing")
                    break
                self._stop.wait(TICK_S)
        finally:
            self._finish()

    def _drain_commands(self, ts: float) -> None:
        with self._lock:
            commands, self._commands = self._commands, []
        for name, payload in commands:
            if name == "played":
                uci, events = payload
                self._dispatch(events, ts)
                if not events:
                    if self.session.wrong == uci:
                        self._emit("game", "info", f"{uci}: not the move")
                elif self.session.done and self.session.status == "solved":
                    self._record(solved=not self.session.failed)
            elif name == "gave_up":
                self._dispatch(payload, ts)
                self._record(solved=False)
            elif name == "ask":
                text = str(payload)
                self._emit("user", "info", f"you: {text}")
                self._coaching.ask(text, ts)
            elif name == "voice":
                self.voice = bool(payload) and self._speak is not None
                if not self.voice and self._stop_speaking is not None:
                    self._stop_speaking()
            elif name == "close":
                if not self.session.done:
                    self._record(solved=False)
                self._stop.set()

    def _dispatch(self, events: list[MovePlayed], ts: float) -> None:
        for event in events:
            who = "you" if event.by_user else "the line"
            self._emit("game", "info", f"{event.san}  ({who})", uci=event.uci, ply=event.ply)
            self._assessing.submit(event)
        self._coaching.events(list(events), ts)

    def _record(self, *, solved: bool) -> None:
        session = self.session
        if session.recorded:
            return
        session.recorded = True
        try:
            outcome = record_attempt(
                self.store,
                session.puzzle.id,
                session.motif,
                solved,
                list(session.sans),
                time.time(),
            )
        except Exception as exc:  # noqa: BLE001 - the record must not break the puzzle
            self._emit("sys", "error", f"profile error: {exc!r}")
            return
        self._emit(
            "game",
            "info",
            f"puzzle {outcome.puzzle_id} {'solved' if outcome.solved else 'failed'}"
            + (f" ({outcome.motif})" if outcome.motif else "")
            + f"; puzzle level {outcome.level}",
            solved=outcome.solved,
            puzzle_level=outcome.level,
        )

    def _on_message(self, message: Any) -> None:
        """A coach message: logged for the page, spoken when the voice is on. Only what
        the player asked for comes (the coach is quiet), so nothing is stale."""
        text = getattr(message, "text", "")
        trigger = getattr(getattr(message, "trigger", None), "value", "coach")
        self._emit(
            "coach",
            "info",
            f"{trigger} — {elide(text, 80)}",
            full=text,
            trigger=trigger,
            annotations=[m.model_dump() for m in getattr(message, "annotations", [])],
            fen=getattr(message, "fen", None),
        )
        if self.voice and self._speak is not None and trigger == Trigger.USER_ASK.value:
            try:
                self._speak(text)
            except Exception as exc:  # noqa: BLE001
                self._emit("voice", "warn", f"voice failed: {exc}")

    def _emit(self, chan: str, level: str, text: str, **data: Any) -> None:
        self._log.emit(chan, level, text, **data)
        if self._mirror is not None:
            with contextlib.suppress(Exception):
                self._mirror("puzzle", level, f"{self._id} {chan}: {text}", **data)

    def usage(self) -> Any | None:
        """The room's model calls (llm.usage.Usage), for the parent panel's strip."""
        provider = getattr(getattr(self._coaching.coach, "speaker", None), "provider", None)
        return getattr(provider, "usage", None)

    def _finish(self) -> None:
        self.closed = True
        with contextlib.suppress(Exception):
            self._assessing.close()
        with contextlib.suppress(Exception):
            self._coaching.close()
        self.store.close()

    def close(self) -> None:
        self.command("close")

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout)


class PuzzleRooms:
    """The open rooms by puzzle id, made on demand from the puzzle store."""

    def __init__(
        self,
        settings: Settings,
        *,
        store_path: Path | None,
        puzzles: PuzzleStore | None,
        data_dir: Path | None = None,
        speak: Callable[[str], None] | None = None,
        stop_speaking: Callable[[], None] | None = None,
        analysis: bool = True,
        coach: bool = True,
        mirror: Callable[..., None] | None = None,
    ) -> None:
        self.settings = settings
        self.store_path = store_path
        self.puzzles = puzzles
        # Where the puzzle store is opened when there was none: puzzles Setup fetches after
        # the server started are then found at the next open, as the session finds them at
        # its next Start.
        self.data_dir = data_dir
        self._speak = speak
        self._stop_speaking = stop_speaking
        self.analysis = analysis
        self.coach = coach
        self._mirror = mirror
        self._rooms: dict[str, PuzzleRoom] = {}
        self._lock = threading.Lock()
        # The motif each puzzle was recommended for, told by the engine's profiler.
        self.motifs: dict[str, str] = {}

    def open(self, puzzle_id: str) -> PuzzleRoom | None:
        with self._lock:
            room = self._rooms.get(puzzle_id)
            if room is not None and not room.closed and not room.closing:
                return room
            if self.puzzles is None and self.data_dir is not None:
                from sidecoach.profile.puzzles import open_store

                with contextlib.suppress(Exception):
                    self.puzzles = open_store(self.data_dir)
            if self.puzzles is None:
                return None
            puzzle = self.puzzles.get(puzzle_id)
            if puzzle is None:
                return None
            room = PuzzleRoom(
                self.settings,
                puzzle,
                self.motifs.get(puzzle_id),
                store_path=self.store_path,
                speak=self._speak,
                stop_speaking=self._stop_speaking,
                analysis=self.analysis,
                coach=self.coach,
                mirror=self._mirror,
            )
            self._rooms[puzzle_id] = room
            return room

    def usages(self) -> list[Any]:
        """The model-call ledgers of the rooms, open or closed, for the panel's strip."""
        with self._lock:
            return [u for u in (room.usage() for room in self._rooms.values()) if u is not None]

    def get(self, puzzle_id: str) -> PuzzleRoom | None:
        with self._lock:
            room = self._rooms.get(puzzle_id)
        return room if room is not None and not room.closed and not room.closing else None
