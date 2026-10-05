"""Stand-ins for what the app talks to outside itself -- the engine, the model, the
microphone, the clock, the Lichess puzzle file -- so a test needs no Stockfish, no model
server, no sound card, no download and no wait on the real time."""

from __future__ import annotations

import queue
import threading
import time as _time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import game_over
from sidecoach.events import Line
from sidecoach.llm.base import ChatMessage, Completion

# What a quiet position gets from the fake engine when the test gave it nothing to say.
QUIET = [Line(moves=["a2a3"], depth=1, expected_score=0.5)]


class FakeEngine(Engine):
    """An engine that answers from a script: the lines for a position's FEN, else `default`,
    and nothing for a finished game. It remembers every position it was asked about, with
    the depth and the number of lines asked for, and never starts a process."""

    def __init__(
        self, answers: dict[str, list[Line]] | None = None, *, default: list[Line] = QUIET
    ) -> None:
        super().__init__(path=Path("fake"))
        self.answers = answers or {}
        self.default = default
        self.calls: list[str] = []
        self.depths: list[int | None] = []
        self.multipvs: list[int] = []
        # The moves each position came with: what the engine is told of its history.
        self.histories: list[list[str]] = []

    def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
        self.calls.append(board.fen())
        self.histories.append([move.uci() for move in board.move_stack])
        self.depths.append(kwargs.get("depth"))
        self.multipvs.append(int(kwargs.get("multipv") or 1))
        if game_over(board):
            return []
        return self.answers.get(board.fen(), self.default)

    def open(self) -> None:  # never starts a process
        return

    def close(self) -> None:
        return


class GatedEngine(FakeEngine):
    """A fake engine whose searches of the positions in `gated` (FENs) wait until the test
    lets them go (`release`) or the analyser stops them (`interrupt`, raising `Interrupted`, as
    a real one stops); every other search, and every depth it finishes (`on_depth`, from
    EARLY_DEPTH to the depth asked for), is answered at once. `entered[fen]` is set when a gated
    search begins. So a test acts mid-search by the engine's state, never by a clock."""

    def __init__(
        self,
        answers: dict[str, list[Line]] | None = None,
        *,
        gated: Sequence[str] = (),
        default: list[Line] = QUIET,
    ) -> None:
        super().__init__(answers, default=default)
        self.gated = set(gated)
        self.entered = {fen: threading.Event() for fen in gated}
        self.released: set[str] = set()
        self.stopped: list[str] = []  # gated searches the analyser stopped
        # Told when a gated search begins or is stopped: a test's watch woken for it.
        self.poke: Callable[[], None] | None = None
        self._wake = threading.Event()

    def release(self, fen: str) -> None:
        self.released.add(fen)
        self._wake.set()

    def interrupt(self) -> None:
        super().interrupt()
        self._wake.set()

    def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
        from sidecoach.analysis.annotator import EARLY_DEPTH
        from sidecoach.analysis.engine import Interrupted

        fen = board.fen()
        depth = int(kwargs.get("depth") or 0)
        if fen in self.gated:
            self.entered[fen].set()
            if self.poke is not None:
                self.poke()
            while fen not in self.released:
                if self._interruptible and self._stopped.is_set():
                    self.stopped.append(fen)
                    if self.poke is not None:
                        self.poke()
                    raise Interrupted
                assert self._wake.wait(HANG_S), f"the search of {fen} was never let go"
                self._wake.clear()
        lines = super().analyse(board, **kwargs)
        lines = [line.model_copy(update={"depth": depth}) for line in lines] if depth else lines
        on_depth = kwargs.get("on_depth")
        if on_depth is not None and lines:
            for d in range(EARLY_DEPTH, depth):
                on_depth([line.model_copy(update={"depth": d}) for line in lines])
        return lines


class _Poking:
    """The analyser's results queue, poking a watcher at every result put."""

    def __init__(self, inner: Any, poke: Callable[[], None]) -> None:
        self._inner, self._poke = inner, poke

    def put(self, item: object) -> None:
        self._inner.put(item)
        self._poke()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class AnalyserWatch:
    """An analyser's workers, watched without a clock: every result they put and every change
    of state they signal (`Analyser._cond.notify_all`) wakes `until`, which returns once its
    condition holds. HANG_S only bounds a hang; it is never the expected path. Made before the
    workers start."""

    def __init__(self, analyser: Any) -> None:
        self.analyser = analyser
        self._cond = threading.Condition()
        signal = analyser._cond.notify_all

        def notify_all() -> None:
            signal()
            self.poke()

        analyser._cond.notify_all = notify_all
        analyser._results = _Poking(analyser._results, self.poke)
        self._pokes = 0
        # A gated fake's search beginning or stopped wakes the watch too: a test waits on it.
        for engine in getattr(analyser, "engines", []):
            if isinstance(engine, GatedEngine):
                engine.poke = self.poke

    def poke(self) -> None:
        with self._cond:
            self._pokes += 1
            self._cond.notify_all()

    def until(self, condition: Callable[[], bool], what: str = "the condition") -> None:
        """Until `condition` holds, woken by the workers' own steps. The condition is read
        outside the watch's lock: it takes the analyser's, under which the workers poke the
        watch -- read inside, the two locks were taken in both orders."""
        deadline = _time.monotonic() + HANG_S
        while True:
            with self._cond:
                seen = self._pokes
            if condition():
                return
            with self._cond:
                while self._pokes == seen:
                    left = deadline - _time.monotonic()
                    assert left > 0, f"never came: {what}"
                    self._cond.wait(left)

    def idle(self) -> None:
        """Everything owed to the game answered. A result is put before the state that says
        it was told, so none can come after `pending` is 0; what was said stays queued for
        whoever drains it (the engine glue's `drain`)."""
        self.until(lambda: self.analyser.pending == 0, "the analyser settled")

    def settled(self) -> list[object]:
        """Everything owed answered (`idle`), and what the workers said, polled."""
        self.idle()
        return list(self.analyser.poll())


class ScriptedProvider:
    """A model that answers from a script and then with `then`, or with `fail` raises as a
    server that is down does. It remembers every call: `calls` everything it was asked,
    `prompts` the system message and the messages of each."""

    name = "scripted"

    def __init__(
        self, replies: Sequence[str] = (), *, then: str = "Keep looking.", fail: bool = False
    ) -> None:
        self.replies = list(replies)
        self.then = then
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    @property
    def prompts(self) -> list[tuple[str, list[ChatMessage]]]:
        """Read from `calls`: to forget what was asked, clear `calls`."""
        return [(call["system"], call["messages"]) for call in self.calls]

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "temperature": temperature,
                "thinking": thinking,
            }
        )
        if self.fail:
            raise RuntimeError("server down")
        return Completion(text=self.replies.pop(0) if self.replies else self.then)


class FedMic:
    """A microphone whose frames the test puts in; quiet when it puts none."""

    def __init__(self) -> None:
        self.frames: queue.Queue = queue.Queue()
        self.detail = "fed by the test"
        self.dropped = 0
        self.opened = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, self.detail

    def open(self) -> bool:
        self.opened = True
        return True

    def close(self) -> None:
        self.opened = False


# How long a fake waits on the test (or a test on a thread it started) before it calls the
# step lost: a bound on a hang, never what a test asserts.
HANG_S = 30.0


class SteppedFrames(queue.Queue):
    """A microphone's frames, put in by the test, read by the listener; when there are none,
    the listener's wait for the next ends only when the test says (`idle`), never on a clock.
    So the listener's quiet rounds (its level decaying, a dead stream reopened) run exactly
    when and as often as the test says, and `idle` returning means every frame put in was
    read and handled. `free` lets the listener go (before it is switched off)."""

    def __init__(self) -> None:
        super().__init__()
        self._go = threading.Semaphore(0)
        self._waiting = threading.Semaphore(0)
        self._parked = False
        self._free = False

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        try:
            return super().get(block=False)
        except queue.Empty:
            if self._free:
                raise
        self._waiting.release()  # every frame read and handled: the listener waits here
        self._go.acquire(timeout=HANG_S)
        raise queue.Empty

    def _park(self) -> None:
        assert self._waiting.acquire(timeout=HANG_S), "the listener never came for a frame"

    def idle(self, rounds: int = 1) -> None:
        """Every frame put in read, then `rounds` quiet rounds of the listener, each run to its
        end: on return the listener waits for the next frame."""
        if not self._parked:
            self._park()
            self._parked = True
        for _ in range(rounds):
            self._go.release()
            self._park()

    def free(self) -> None:
        """No more stepping: the listener's wait for a frame ends at once (it can stop)."""
        self._free = True
        self._go.release()


class SteppedMic(FedMic):
    """A FedMic whose listener the test steps: `steps` is its `frames` (`SteppedFrames`)."""

    def __init__(self) -> None:
        super().__init__()
        self.steps = SteppedFrames()
        self.frames = self.steps


class _Facade:
    """A module as another module sees it: its own names, but for the ones given."""

    def __init__(self, real: Any, **given: Any) -> None:
        self._real = real
        self.__dict__.update(given)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


class ManualTimer:
    """A `threading.Timer` on a ManualClock: it fires when the clock is moved past it."""

    def __init__(
        self,
        clock: ManualClock,
        interval: float,
        function: Callable[..., Any],
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.clock = clock
        self.interval = float(interval)
        self.function = function
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.daemon = True
        self.due: float | None = None
        self.cancelled = False
        self.fired = False

    def start(self) -> None:
        self.clock._add(self)

    def cancel(self) -> None:
        self.cancelled = True

    def is_alive(self) -> bool:
        return self.due is not None and not self.cancelled and not self.fired


class ManualClock:
    """A clock the test moves by hand, for code that reads the time or waits on a timer.

    Its `monotonic`, `perf_counter` and `time` read it; its `sleep` moves it on by what was
    asked (written down in `sleeps`); a `Timer` made on it fires when `advance` or `advance_to`
    passes its time -- on the thread that moved the clock, in the order they fall due. Given to
    a module in place of its own `time` (`time_module`) or `threading` (`threading_module`),
    every other name of those modules stays the real one, and no other module sees it."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)
        self.sleeps: list[float] = []
        self._timers: list[ManualTimer] = []
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(max(0.0, seconds))

    def Timer(  # noqa: N802 - the name threading gives it
        self,
        interval: float,
        function: Callable[..., Any],
        args: Sequence[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> ManualTimer:
        return ManualTimer(self, interval, function, args, kwargs)

    def _add(self, timer: ManualTimer) -> None:
        with self._lock:
            timer.due = self.now + timer.interval
            self._timers.append(timer)

    def pending(self) -> list[ManualTimer]:
        """The timers started and neither cancelled nor fired."""
        with self._lock:
            return [t for t in self._timers if t.is_alive()]

    def advance(self, seconds: float) -> None:
        self.advance_to(self.now + seconds)

    def advance_to(self, when: float) -> None:
        """Move the clock to `when`, firing every timer due by then, each at its own time."""
        while True:
            with self._lock:
                due = [t for t in self._timers if t.is_alive() and t.due <= when]  # type: ignore[operator]
                if not due:
                    self.now = max(self.now, when)
                    return
                timer = min(due, key=lambda t: t.due)  # type: ignore[arg-type,return-value]
                self.now = max(self.now, timer.due)  # type: ignore[arg-type]
                timer.fired = True
            timer.function(*timer.args, **timer.kwargs)

    def time_module(self) -> Any:
        """The `time` module with this clock in it, for `monkeypatch.setattr(mod, "time", ...)`."""
        return _Facade(
            _time,
            monotonic=self.monotonic,
            perf_counter=self.perf_counter,
            time=self.time,
            sleep=self.sleep,
        )

    def threading_module(self) -> Any:
        """The `threading` module with this clock's timers in it."""
        return _Facade(threading, Timer=self.Timer)


# The Lichess puzzle database's columns (database.lichess.org, lichess_db_puzzle.csv).
PUZZLE_HEADER = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
)


def puzzle_file(path: Path, rows: Sequence[str]) -> Path:
    """A puzzle file in the database's format at `path`: the header, then `rows`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([PUZZLE_HEADER, *rows, ""]), encoding="utf-8")
    return path
