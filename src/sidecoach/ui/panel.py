"""The panel's engine: the live coaching loop behind the web UI, with a debug terminal.

The capture -> pipeline -> engine -> coach loop runs on its own thread as `PanelEngine`,
exactly the loop `sidecoach watch` runs, but with its lines routed to the page's terminal
and its coach messages both shown and spoken. `PanelApi` is the thin control surface the
server (`sidecoach.ui.server`, FastAPI) exposes over HTTP. Control actions from the page
(start, ask, flip, pick a voice, analyse a position) are marshalled onto the loop thread
through a command queue, so nothing touches the pipeline or the coach from two threads at
once.

The engine takes an injectable frame source, so it can be driven from a recording -- in a
test, or with `sidecoach panel --replay` while developing -- without a live screen.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from sidecoach.config import Settings
from sidecoach.voice import TTSProvider, make_tts

# The English Kokoro voices offered in the picker (American / British, female / male).
PICKER_VOICES: list[str] = [
    "af_heart",
    "af_sarah",
    "af_bella",
    "af_nicole",
    "af_nova",
    "af_sky",
    "am_adam",
    "am_michael",
    "am_echo",
    "am_liam",
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bm_george",
    "bm_lewis",
    "bm_daniel",
]

Frame = tuple[int, float, Any]  # (frame_id, ts, frame)
FrameSource = Iterable[Frame]


def _plain(text: str) -> str:
    """Rich console markup ("[dim]...[/dim]") stripped, for a line shown on the page."""
    from rich.text import Text

    try:
        return Text.from_markup(text).plain
    except Exception:  # noqa: BLE001 - a stray bracket in a line is not worth losing the line
        return text


@dataclass
class _LogLine:
    seq: int
    level: str  # "info" | "move" | "coach" | "warn" | "error" | "engine"
    text: str
    ts: float


@dataclass
class PanelEngine:
    """The live coaching loop behind the panel. Thread-safe to read; controlled by commands."""

    settings: Settings
    monitor: int | None = None
    fps: int = 10
    record: bool = False
    analysis: bool = True
    coach: bool = True
    voice: str = "af_sarah"
    muted: bool = False
    # A frame source for tests; when None the engine opens the live screen capture on start.
    frame_source: FrameSource | None = None

    _thread: threading.Thread | None = field(default=None, init=False)
    _commands: queue.Queue[tuple[str, Any]] = field(default_factory=queue.Queue, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _log: deque[_LogLine] = field(default_factory=lambda: deque(maxlen=1000), init=False)
    _seq: int = field(default=0, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False)
    _tts: TTSProvider = field(default=None, init=False)  # type: ignore[assignment]
    _forced_bottom: Any = field(default=None, init=False)  # a Side chosen by the flip button
    _view_flip: bool = field(default=False, init=False)  # the panel's board shown turned around

    def __post_init__(self) -> None:
        self._tts = self._build_tts(self.voice)
        self._state = {
            "running": False,
            "fen": None,
            "moves": [],
            "last_move": None,
            "view_flip": False,
            "game_id": None,
            "user_side": None,
            "bottom": None,
            "locked": False,
            "following": False,
            "ply": 0,
            "fps": 0.0,
            "status": "idle",
            "voice": self.voice,
            "voices": PICKER_VOICES,
            "muted": self.muted,
            "coach_on": self.coach,
            "analysis_on": self.analysis,
            "voice_detail": self._tts.healthcheck()[1],
        }

    # ---- control surface (called from the API / GUI thread) -------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="panel-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def command(self, name: str, payload: Any = None) -> None:
        """Queue a control action for the loop thread (ask, hint, flip, analyse, voice)."""
        self._commands.put((name, payload))

    def set_voice(self, voice: str) -> None:
        with self._lock:
            self._tts = self._build_tts(voice)
            self.voice = voice
            self._state["voice"] = voice
            self._state["voice_detail"] = self._tts.healthcheck()[1]
        self._emit("info", f"voice set to {voice}")

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self.muted = muted
            self._state["muted"] = muted
        if muted:
            self._tts.stop()
        self._emit("info", "muted" if muted else "unmuted")

    def poll(self, since: int = 0) -> dict[str, Any]:
        """A snapshot for the page: the current state and any log lines after `since`."""
        with self._lock:
            state = dict(self._state)
            lines = [
                {"seq": ln.seq, "level": ln.level, "text": ln.text}
                for ln in self._log
                if ln.seq > since
            ]
        return {"state": state, "lines": lines, "cursor": self._seq}

    # ---- internals ------------------------------------------------------------------------

    def _build_tts(self, voice: str) -> TTSProvider:
        cfg = self.settings.model_copy(deep=True)
        cfg.voice = cfg.voice.model_copy(
            update={"enabled": True, "provider": "kokoro", "voice": voice}
        )
        return make_tts(cfg)

    def _emit(self, level: str, text: str) -> None:
        with self._lock:
            self._seq += 1
            self._log.append(_LogLine(self._seq, level, text, time.monotonic()))

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    def _say(self, text: str) -> None:
        if not self.muted:
            self._tts.say(text)

    def _iter_frames(self) -> Iterator[Frame]:
        if self.frame_source is not None:
            yield from self.frame_source
            return
        from sidecoach.capture.loop import capture_loop
        from sidecoach.capture.mss_backend import MssCapture

        cap = MssCapture()
        from sidecoach.cli import _monitor_region

        monitor, region = _monitor_region(cap, self.monitor)
        self._region = region
        self._emit("info", f"watching monitor {monitor} {region.w}x{region.h} at {self.fps} fps")
        try:
            yield from capture_loop(cap, region, self.fps)
        finally:
            cap.close()

    def _run(self) -> None:
        import chess

        from sidecoach.cli import _Assessing, _Coaching
        from sidecoach.events import (
            Event,
            Side,
        )
        from sidecoach.state.pipeline import GamePipeline
        from sidecoach.vision.clock import ClockReader
        from sidecoach.vision.squares import BoardGeometry

        self._side = Side
        cache_dir = self.settings.data_dir / "themes"
        try:
            from sidecoach.vision.theme_cache import ThemeCache

            cache = ThemeCache(cache_dir)
        except Exception:  # noqa: BLE001 - a cache is a nicety; the panel runs without it
            cache = None
        pipe = GamePipeline(clock=ClockReader(), cache=cache)
        assessing = _Assessing(self.settings, self.analysis)
        coaching = _Coaching(self.settings, self.coach and assessing.on)
        coaching.sink = self._on_coach_message
        # The assessment lines are written for the rich console; the page wants plain text.
        assessing.sink = lambda line: self._emit("engine", _plain(line))
        self._set(
            running=True,
            status="waiting for a board",
            coach_on=coaching.on,
            analysis_on=assessing.on,
        )
        self._emit("info", "engine started; open a game on the board to bootstrap")
        if coaching.on:
            self._emit("info", "coach on")

        latest: Any = None
        latest_geom: Any = None
        locked = False
        bootstraps = 0
        ticks = 0
        last_status = time.monotonic()
        try:
            for frame_id, ts, frame in self._iter_frames():
                if self._stop.is_set():
                    break
                latest = frame
                events: list[Event] = []
                try:
                    events = pipe.process(frame, ts, changed=True, frame_id=frame_id)
                except Exception as exc:  # noqa: BLE001 - a pipeline bug must not end the panel
                    self._emit("error", f"pipeline error at frame {frame_id}: {exc!r}")
                if pipe.board is not None:
                    latest_geom = BoardGeometry(pipe.board.region, pipe.board.cell_px)
                ticks += 1

                events += self._drain_commands(pipe, assessing, coaching, latest, latest_geom, ts)

                if (pipe.board is not None) != locked:
                    locked = pipe.board is not None
                    self._set(locked=locked)
                    self._emit("info", "board locked" if locked else "board lost")
                if pipe.bootstraps != bootstraps:
                    bootstraps = pipe.bootstraps
                    bottom = pipe.bottom.value if pipe.bottom else "?"
                    self._emit("info", f"start position seen, {bottom} at the bottom")

                for event in events:
                    self._dispatch(event, assessing, pipe, chess)

                coaching.events(events, ts)
                coaching.events(assessing.drain(), ts)
                coaching.tick(ts)
                coaching.drain()
                self._refresh_state(pipe, ticks, last_status)

                now = time.monotonic()
                if now - last_status >= 2.0:
                    with self._lock:
                        self._state["fps"] = round(ticks / (now - last_status), 1)
                    ticks = 0
                    last_status = now
        except Exception as exc:  # noqa: BLE001 - report and shut down cleanly
            self._emit("error", f"engine stopped: {exc!r}")
        finally:
            coaching.events(assessing.wait(), time.monotonic())
            assessing.close()
            coaching.wait()
            coaching.close()
            self._set(running=False, status="stopped")
            self._emit("info", "engine stopped")

    def _dispatch(self, event: Any, assessing: Any, pipe: Any, chess: Any) -> None:
        from sidecoach.events import GameEnded, GameStarted, MovePlayed, Resync, Takeback

        if isinstance(event, GameStarted):
            self._emit("info", f"new game {event.game_id}, you play {event.user_side.value}")
            self._set(game_id=event.game_id, user_side=event.user_side.value)
        elif isinstance(event, MovePlayed):
            before = chess.Board(event.fen_before)
            number = before.fullmove_number
            dots = "." if before.turn == chess.WHITE else "..."
            who = "you" if event.by_user else "opponent"
            assessing.submit(event)
            self._emit("move", f"{number}{dots} {event.san}  ({who})")
        elif isinstance(event, Takeback):
            self._emit("warn", f"takeback {event.plies} ply")
        elif isinstance(event, Resync):
            self._emit("warn", f"resync: {event.reason}")
            assessing.reset()
        elif isinstance(event, GameEnded):
            self._emit("info", f"game over {event.result} ({event.reason})")

    def _drain_commands(
        self, pipe: Any, assessing: Any, coaching: Any, frame: Any, geom: Any, ts: float
    ) -> list[Any]:
        events: list[Any] = []
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                break
            if name == "ask" and coaching.on:
                coaching.ask(str(payload), ts)
                self._emit("info", f"you: {payload}")
            elif name == "flip":
                events += self._flip(pipe, ts)
            elif name == "analyze":
                events += self._analyze(pipe, frame, geom)
        return events

    def _flip(self, pipe: Any, ts: float) -> list[Any]:
        """Flip means two things, and never disturbs a game that is being followed.

        While a game is running, the pipeline's orientation is a fact it read from the
        screen, so flipping only turns the panel's board around (a view preference).
        Reorienting the reconciler here would re-read an unchanged screen the wrong way up
        and resync to a rotated, bogus position. With no game running, flipping chooses
        the side the next analyse / cold start assumes at the bottom -- the way out when
        the coordinates are off and the labels cannot be read.
        """
        Side = self._side
        if pipe.reconciler is not None and pipe.reconciler.started:
            self._view_flip = not self._view_flip
            self._set(view_flip=self._view_flip)
            self._emit("info", "view flipped (the game is still read from the screen as is)")
            return []
        current = self._forced_bottom or pipe.bottom or Side.WHITE
        target = Side.BLACK if current == Side.WHITE else Side.WHITE
        self._forced_bottom = target
        self._set(bottom=target.value)
        self._emit("info", f"analysis orientation: {target.value} at the bottom")
        return []

    def _analyze(self, pipe: Any, frame: Any, geom: Any) -> list[Any]:
        if frame is None or geom is None:
            self._emit("warn", "no board on screen to analyse yet")
            return []
        try:
            events = pipe.cold_start(frame, geom, bottom=self._forced_bottom)
        except Exception as exc:  # noqa: BLE001
            self._emit("error", f"analyse failed: {exc!r}")
            return []
        if not events:
            self._emit(
                "warn",
                "could not read the position cold: no cached style match, coordinates off "
                "(use Flip), or the board is mid-move",
            )
        else:
            self._emit("info", "analysing the position on the board")
        return events

    def _on_coach_message(self, message: Any) -> None:
        text = getattr(message, "text", str(message))
        self._emit("coach", text)
        self._say(text)

    def _refresh_state(self, pipe: Any, ticks: int, last_status: float) -> None:
        rec = pipe.reconciler
        fen = rec.board.fen() if rec is not None else None
        moves = [m.uci() for m in rec.board.move_stack] if rec is not None else []
        sans = self._sans(rec) if rec is not None else []
        self._set(
            fen=fen,
            moves=sans,
            last_move=moves[-1] if moves else None,  # uci, for the from/to highlight
            ply=len(moves),
            following=pipe.model is not None,
            bottom=pipe.bottom.value if pipe.bottom else None,
            status="following a game" if pipe.model is not None else "waiting for a start",
        )

    @staticmethod
    def _sans(rec: Any) -> list[str]:
        import chess

        board = chess.Board(rec._base_fen) if getattr(rec, "_base_fen", None) else chess.Board()
        out: list[str] = []
        for move in rec.board.move_stack:
            try:
                out.append(board.san(move))
            except Exception:  # noqa: BLE001 - a resync base can make SAN ambiguous; show uci
                out.append(move.uci())
            board.push(move)
        return out


class PanelApi:
    """The bridge the web page calls through `window.pywebview.api`. Thin over the engine."""

    def __init__(self, engine: PanelEngine) -> None:
        self._engine = engine

    def start(self) -> bool:
        self._engine.start()
        return True

    def stop(self) -> bool:
        self._engine.stop()
        return True

    def poll(self, since: int = 0) -> dict[str, Any]:
        return self._engine.poll(since)

    def ask(self, text: str) -> bool:
        self._engine.command("ask", text)
        return True

    def hint(self) -> bool:
        self._engine.command("ask", "hint")
        return True

    def move(self) -> bool:
        self._engine.command("ask", "what should I play")
        return True

    def line(self) -> bool:
        self._engine.command("ask", "show me the line")
        return True

    def flip(self) -> bool:
        self._engine.command("flip")
        return True

    def analyze(self) -> bool:
        self._engine.command("analyze")
        return True

    def set_voice(self, voice: str) -> bool:
        self._engine.set_voice(voice)
        return True

    def set_muted(self, muted: bool) -> bool:
        self._engine.set_muted(bool(muted))
        return True
