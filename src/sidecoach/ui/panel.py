"""The panel's engine: the live coaching loop behind the web UI, with a debug log.

The capture -> pipeline -> engine -> coach loop runs on its own thread as `PanelEngine`,
exactly the loop `sidecoach watch` runs, but everything it learns goes to a structured debug
log (see `sidecoach.ui.log`) the page shows as a terminal, and its coach messages are both
shown and spoken. `PanelApi` is the thin control surface the server (`sidecoach.ui.server`,
FastAPI) exposes over HTTP. Control actions from the page (start, ask, flip, pick a voice,
analyse a position) are marshalled onto the loop thread through a command queue, so nothing
touches the pipeline or the coach from two threads at once.

What the log carries, by channel:
  sys     the session: capture, recording, whether the engine, coach and voice are on and why
  vision  the board: locked / lost / in doubt, the start position, degraded readings
  game    the game: moves with think time, takebacks, resyncs, game over
  engine  each move's assessment, and the position signals that drive the coach
          (decisive moment, only move, threats, the opening)
  coach   each message with why it spoke, at what disclosure level, how many attempts, and
          whether the checks rejected a draft; at debug, why it is quiet
  voice   what was spoken or muted, and any synthesis failure
  user    the user's asks
Continuously-changing health (fps, lock fit, pipeline timing, lag, clock, recording size) is
not logged at all; it lives in the state as a status strip.

The engine takes an injectable frame source, so it can be driven from a recording -- in a
test, or with `sidecoach panel --replay` while developing -- without a live screen.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from sidecoach.config import Settings
from sidecoach.ui.log import DebugLog, elide
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

# Unreadable cells (unknown or covered) beyond which a frame's reading is called degraded,
# and how many frames in a row it must stay that way before it is worth a line.
DEGRADED_CELLS = 6
DEGRADED_FRAMES = 5
# A coach message about a position this many plies behind the board is shown, not spoken.
STALE_PLIES = 2
# A "verdict" this short (post-move "exf5: good.") is skipped when the voice is busy.
SHORT_VERDICT = 40
HEALTH_EVERY_S = 1.0


class _VoiceLogHandler(logging.Handler):
    """Forwards the voice provider's warnings (a failed load, a failed line) to the log."""

    def __init__(self, engine: PanelEngine) -> None:
        super().__init__(level=logging.WARNING)
        self._engine = engine

    def emit(self, record: logging.LogRecord) -> None:
        level = "error" if record.levelno >= logging.ERROR else "warn"
        self._engine._emit("voice", level, record.getMessage())


@dataclass
class PanelEngine:
    """The live coaching loop behind the panel. Thread-safe to read; controlled by commands."""

    settings: Settings
    monitor: int | None = None
    fps: int = 10
    # Record a live session (frames + events.jsonl + debug.jsonl) like `watch` does. A replay
    # is never recorded; it still gets a session folder for its debug log.
    record: bool = True
    analysis: bool = True
    coach: bool = True
    voice: str = "af_sarah"
    muted: bool = False
    # A frame source for tests and replays; when None the engine opens the live screen capture.
    frame_source: FrameSource | None = None

    _thread: threading.Thread | None = field(default=None, init=False)
    _commands: queue.Queue[tuple[str, Any]] = field(default_factory=queue.Queue, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _log: DebugLog = field(default_factory=DebugLog, init=False)
    _state: dict[str, Any] = field(default_factory=dict, init=False)
    _tts: TTSProvider = field(default=None, init=False)  # type: ignore[assignment]
    _forced_bottom: Any = field(default=None, init=False)  # a Side chosen by the flip button
    _view_flip: bool = field(default=False, init=False)  # the panel's board shown turned around
    _fen_before: dict[int, str] = field(default_factory=dict, init=False)  # ply -> fen before
    _bad_run: int = field(default=0, init=False)  # consecutive frames read as degraded

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
            "status": "idle",
            "voice": self.voice,
            "voices": PICKER_VOICES,
            "muted": self.muted,
            "coach_on": self.coach,
            "analysis_on": self.analysis,
            "voice_detail": self._tts.healthcheck()[1],
            "clock": None,
            "health": {},
            "session_dir": None,
        }
        logging.getLogger("sidecoach.voice").addHandler(_VoiceLogHandler(self))

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
        """Queue a control action for the loop thread (ask, flip, analyse)."""
        self._commands.put((name, payload))

    def set_voice(self, voice: str) -> None:
        with self._lock:
            self._tts = self._build_tts(voice)
            self.voice = voice
            self._state["voice"] = voice
            self._state["voice_detail"] = self._tts.healthcheck()[1]
        self._emit("voice", "info", f"voice set to {voice}", detail=self._state["voice_detail"])

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self.muted = muted
            self._state["muted"] = muted
        if muted:
            self._tts.stop()
        self._emit("voice", "info", "muted" if muted else "unmuted")

    def poll(self, since: int = 0) -> dict[str, Any]:
        """A snapshot for the page: the current state and every log record after `since`."""
        with self._lock:
            state = dict(self._state)
        return {"state": state, "lines": self._log.since(since), "cursor": self._log.cursor}

    # ---- internals ------------------------------------------------------------------------

    def _build_tts(self, voice: str) -> TTSProvider:
        cfg = self.settings.model_copy(deep=True)
        cfg.voice = cfg.voice.model_copy(
            update={"enabled": True, "provider": "kokoro", "voice": voice}
        )
        return make_tts(cfg)

    def _emit(self, chan: str, level: str, text: str, **data: Any) -> None:
        self._log.emit(chan, level, text, **data)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    @staticmethod
    def voice_action(trigger: str, text: str, ply: int | None, ply_now: int, busy: bool) -> str:
        """What the voice should do with a coach message, so it never talks about a
        position the board has left behind.

        A fast game (or a replay run faster than it was played) can move on while a line is
        still being synthesised or spoken. So: a message whose position is already
        STALE_PLIES behind is shown but not spoken ("stale"); a short post-move verdict is
        dropped if the voice is busy rather than queued behind it ("skip"); anything
        substantive cuts off whatever is still playing about an older moment ("supersede").
        """
        if ply is not None and ply_now - ply >= STALE_PLIES:
            return "stale"
        if trigger == "post_move" and len(text) < SHORT_VERDICT:
            return "skip" if busy else "speak"
        return "supersede" if busy else "speak"

    def _speak(self, trigger: str, text: str, ply: int | None) -> str:
        """Apply `voice_action` and speak accordingly; returns what was done, for the log."""
        if self.muted:
            return "muted"
        with self._lock:
            ply_now = int(self._state.get("ply") or 0)
        action = self.voice_action(trigger, text, ply, ply_now, self._tts.busy)
        if action == "supersede":
            self._tts.stop()
        if action in ("speak", "supersede"):
            self._tts.say(text)
        return action

    def _say(self, text: str) -> None:
        if self.muted:
            self._emit("voice", "debug", f"muted, not spoken: {elide(text, 60)}")
            return
        self._tts.say(text)
        self._emit("voice", "debug", f"speaking ({self.voice}): {elide(text, 60)}")

    def _session_dir(self, live: bool) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kind = "recordings" if live and self.record else "sessions"
        return self.settings.data_dir / kind / stamp

    def _run(self) -> None:
        import chess

        from sidecoach.cli import _Assessing, _Coaching
        from sidecoach.events import Event, Side
        from sidecoach.state.pipeline import GamePipeline
        from sidecoach.vision.clock import ClockReader
        from sidecoach.vision.squares import BoardGeometry

        self._side = Side
        live = self.frame_source is None
        session = self._session_dir(live)
        self._log.attach(session / "debug.jsonl")
        self._set(running=True, status="starting", session_dir=str(session))
        self._emit("sys", "info", "engine started")

        # Frames: the live screen (gated, and recorded when asked) or the injected source.
        cap: Any = None
        recorder: Any = None
        gate: Any = None
        region: Any = None
        events_log: IO[str] | None = None
        if live:
            from sidecoach.capture.loop import capture_loop
            from sidecoach.capture.mss_backend import MssCapture
            from sidecoach.cli import _monitor_region
            from sidecoach.replay.recording import ChangeGate, Recorder

            cap = MssCapture()
            monitor, region = _monitor_region(cap, self.monitor)
            self._emit(
                "sys",
                "info",
                f"capturing monitor {monitor}, {region.w}x{region.h} at {self.fps} fps",
            )
            gate = ChangeGate(keyframe_interval_s=1.0)
            if self.record:
                recorder = Recorder(
                    session, monitor=monitor, region=region, fps=self.fps, keyframe_interval_s=1.0
                )
                events_log = (session / "events.jsonl").open("w", encoding="utf-8")
                self._emit("sys", "info", f"recording to {session}")
            else:
                self._emit("sys", "info", "not recording")
            frames: Iterable[Frame] = capture_loop(cap, region, self.fps)
        else:
            frames = self.frame_source or ()
            self._emit("sys", "info", "replaying a recording (not recorded again)")

        cache = None
        try:
            from sidecoach.vision.theme_cache import ThemeCache

            cache = ThemeCache(self.settings.data_dir / "themes")
        except Exception as exc:  # noqa: BLE001 - a cache is a nicety; the panel runs without it
            self._emit("sys", "warn", f"theme cache unavailable: {exc}")
        pipe = GamePipeline(clock=ClockReader(), cache=cache)
        assessing = _Assessing(self.settings, self.analysis)
        coaching = _Coaching(self.settings, self.coach and assessing.on)
        # Their console output is replaced by the log: assessments are formatted here from
        # the events, and coach messages are logged with their metadata below.
        assessing.sink = lambda _line: None
        coaching.sink = lambda _message: None
        self._emit("sys", "info" if assessing.on else "warn", f"engine: {assessing.detail}")
        self._emit("sys", "info" if coaching.on else "warn", f"coach: {coaching.detail}")
        ready, detail = self._tts.healthcheck()
        self._emit("sys", "info" if ready else "warn", f"voice: {detail}")
        self._set(status="waiting for a board", coach_on=coaching.on, analysis_on=assessing.on)

        latest: Any = None
        latest_geom: Any = None
        locked = False
        verified = False
        bootstraps = 0
        degraded = False
        opening: str | None = None
        spoken_seen = 0
        failures_seen = 0
        silence: str | None = None
        ticks = 0
        pipeline_ms: list[float] = []
        last_health = time.monotonic()
        try:
            for frame_id, ts, frame in frames:
                if self._stop.is_set():
                    break
                latest = frame
                changed = True
                if live:
                    from sidecoach.replay.recording import board_gate

                    gate.region = board_gate(pipe.board, region.w, region.h)
                    if recorder is not None:
                        recorder.change_region = gate.region
                        try:
                            changed = recorder.write(frame, ts).stored is not None
                        except (ValueError, RuntimeError) as exc:
                            # A display change or a full disk must not end the session.
                            self._emit("sys", "error", f"recording stopped: {exc}")
                            recorder.close()
                            recorder = None
                            changed = gate.changed(frame, ts)
                    else:
                        changed = gate.changed(frame, ts)

                begun = time.perf_counter()
                events: list[Event] = []
                try:
                    events = pipe.process(frame, ts, changed=changed, frame_id=frame_id)
                except Exception as exc:  # noqa: BLE001 - a pipeline bug must not end the panel
                    self._emit("sys", "error", f"pipeline error at frame {frame_id}: {exc!r}")
                pipeline_ms.append((time.perf_counter() - begun) * 1000.0)
                ticks += 1
                if pipe.board is not None:
                    latest_geom = BoardGeometry(pipe.board.region, pipe.board.cell_px)

                events += self._drain_commands(pipe, coaching, latest, latest_geom, ts)

                # Vision transitions: only the changes are logged, never every frame.
                if (pipe.board is not None) != locked:
                    locked = pipe.board is not None
                    self._set(locked=locked)
                    if locked and pipe.board is not None:
                        b = pipe.board
                        self._emit(
                            "vision",
                            "info",
                            f"board locked at ({b.region.x},{b.region.y}) "
                            f"{b.region.w}x{b.region.h}, cell {b.cell_px:.1f}px, fit {b.fit:.2f}",
                            region=list(b.region),
                            cell_px=round(b.cell_px, 2),
                            fit=round(b.fit, 3),
                        )
                    else:
                        self._emit("vision", "warn", "board lost")
                if locked and pipe.tracker.verified != verified:
                    verified = pipe.tracker.verified
                    self._emit("vision", "debug", "lock verified" if verified else "lock in doubt")
                if pipe.bootstraps != bootstraps:
                    bootstraps = pipe.bootstraps
                    bottom = pipe.bottom.value if pipe.bottom else "?"
                    labels = len(pipe.model.rank_labels) if pipe.model is not None else 0
                    self._emit(
                        "vision",
                        "info",
                        f"start position seen, {bottom} at the bottom; theme learned"
                        f"{' and cached' if cache is not None else ''}, {labels}/8 rank labels",
                    )
                degraded = self._check_degraded(pipe, degraded)

                if events_log is not None:
                    for event in events:
                        events_log.write(event.model_dump_json() + "\n")
                for event in events:
                    self._dispatch_game(event, assessing, chess)

                engine_events = assessing.drain()
                if events_log is not None:
                    for event in engine_events:
                        events_log.write(event.model_dump_json() + "\n")
                opening = self._dispatch_engine(engine_events, opening, chess)

                coaching.events(events, ts)
                coaching.events(engine_events, ts)
                coaching.tick(ts)
                coaching.drain()
                spoken_seen, failures_seen = self._coach_meta(coaching, spoken_seen, failures_seen)
                silence = self._coach_silence(coaching, silence)
                self._refresh_state(pipe)

                now = time.monotonic()
                if now - last_health >= HEALTH_EVERY_S:
                    self._health(
                        pipe, coaching, recorder, ticks, pipeline_ms, now - last_health, ts, live
                    )
                    ticks = 0
                    pipeline_ms = []
                    last_health = now
        except Exception as exc:  # noqa: BLE001 - report and shut down cleanly
            self._emit("sys", "error", f"engine stopped: {exc!r}")
        finally:
            coaching.events(assessing.wait(), time.monotonic())
            assessing.close()
            coaching.wait()
            self._coach_meta(coaching, spoken_seen, failures_seen)
            coaching.close()
            if recorder is not None:
                recorder.close()
            if events_log is not None:
                events_log.close()
            if cap is not None:
                cap.close()
            self._set(running=False, status="stopped")
            self._emit("sys", "info", "engine stopped")
            self._log.close()

    # ---- log sources ------------------------------------------------------------------------

    def _check_degraded(self, pipe: Any, degraded: bool) -> bool:
        """Warn once when many cells cannot be read for a while (an overlay, a dialog), and
        say once when the reading is clean again. A single frame's blip -- a piece being
        dragged across the board -- is not worth a line, hence the run of frames required."""
        readings = pipe.readings
        if readings is None:
            return degraded
        bad = sum(
            1 for row in readings for reading in row if reading.kind == "unknown" or reading.covered
        )
        self._bad_run = self._bad_run + 1 if bad >= DEGRADED_CELLS else 0
        if self._bad_run >= DEGRADED_FRAMES and not degraded:
            self._emit(
                "vision",
                "warn",
                f"reading degraded: {bad} cells unreadable (an overlay or a dialog?)",
                unreadable=bad,
            )
            return True
        if self._bad_run == 0 and degraded:
            self._emit("vision", "info", "reading clean again")
            return False
        return degraded

    def _dispatch_game(self, event: Any, assessing: Any, chess: Any) -> None:
        from sidecoach.events import (
            ClockObservation,
            GameEnded,
            GameStarted,
            MovePlayed,
            Resync,
            ScreenStateChanged,
            Takeback,
        )

        if isinstance(event, GameStarted):
            self._fen_before.clear()
            self._emit(
                "game",
                "info",
                f"new game {event.game_id}, you play {event.user_side.value}",
                game_id=event.game_id,
                start_fen=event.start_fen,
            )
            self._set(game_id=event.game_id, user_side=event.user_side.value)
        elif isinstance(event, MovePlayed):
            before = chess.Board(event.fen_before)
            number = before.fullmove_number
            dots = "." if before.turn == chess.WHITE else "..."
            who = "you" if event.by_user else "opponent"
            think = f", {event.think_ms / 1000:.1f}s" if event.think_ms is not None else ""
            self._fen_before[event.ply] = event.fen_before
            assessing.submit(event)
            self._emit(
                "game",
                "info",
                f"{number}{dots} {event.san}  ({who}{think})",
                ply=event.ply,
                uci=event.uci,
                by_user=event.by_user,
                think_ms=event.think_ms,
                fen_after=event.fen_after,
            )
        elif isinstance(event, Takeback):
            self._emit("game", "warn", f"takeback {event.plies} ply", fen=event.fen)
        elif isinstance(event, Resync):
            self._emit("game", "warn", f"resync: {event.reason}", fen=event.fen)
            assessing.reset()
        elif isinstance(event, GameEnded):
            self._emit(
                "game",
                "info",
                f"game over {event.result} ({event.reason})",
                result=event.result,
                pgn=event.pgn,
            )
        elif isinstance(event, ClockObservation):
            self._set(
                clock={
                    "white_ms": event.white_ms,
                    "black_ms": event.black_ms,
                    "running": event.running.value if event.running else None,
                }
            )
        elif isinstance(event, ScreenStateChanged):
            self._emit("vision", "debug", f"screen: {event.state.value}")

    def _dispatch_engine(self, events: list[Any], opening: str | None, chess: Any) -> str | None:
        from sidecoach.analysis.engine import san_line
        from sidecoach.events import Analysis, MoveAssessment

        for event in events:
            if isinstance(event, MoveAssessment):
                fen = self._fen_before.pop(event.ply, None)
                board = chess.Board(fen) if fen else None
                san = board.san(chess.Move.from_uci(event.uci)) if board else event.uci
                parts = [f"ply {event.ply} {san}: loss {event.expected_loss:.2f}"]
                if event.severity.value != "none":
                    parts.append(event.severity.value)
                    better = san_line(board, event.best, limit=4) if board else ""
                    if better:
                        parts.append(f"better {better}")
                tags = list(dict.fromkeys(tag.name for tag in event.tags))
                # "quietMove" is the tagger's "nothing special"; it stays in the data only.
                shown = [t for t in tags if t != "quietMove"]
                if shown:
                    parts.append(", ".join(shown[:3]) + (" …" if len(shown) > 3 else ""))
                self._emit(
                    "engine",
                    "info",
                    " · ".join(parts),
                    ply=event.ply,
                    loss=round(event.expected_loss, 3),
                    severity=event.severity.value,
                    label=event.label.value if event.label else None,
                    best=self._sans_of(fen, event.best, chess) if fen else event.best,
                    tags=tags,
                    depth=event.depth,
                )
            elif isinstance(event, Analysis):
                top = event.lines[0] if event.lines else None
                if event.opening and event.opening != opening:
                    opening = event.opening
                    self._emit("engine", "info", f"opening: {event.opening}", ply=event.ply)
                score = f"{top.expected_score:.2f}" if top else "?"
                depth = top.depth if top else 0
                threats = self._sans_of(event.fen, event.threats, chess, by_opponent=True)
                pv = self._sans_of(event.fen, top.moves, chess) if top else []
                data = {
                    "ply": event.ply,
                    "decisive": event.decisive,
                    "only_move": event.only_move,
                    "threats": threats,
                    "pv": pv,
                    "score": top.expected_score if top else None,
                    "depth": depth,
                }
                # Only the moments that drive the coach are worth a line at info; almost
                # every position carries some threat, so plain threats stay at debug.
                signals = []
                if event.decisive:
                    signals.append("decisive moment")
                if event.only_move:
                    signals.append("only move")
                tail = f"  (score {score}, d{depth})"
                threat_text = ("; threats " + ", ".join(threats[:3])) if threats else ""
                if signals:
                    self._emit(
                        "engine",
                        "info",
                        f"ply {event.ply}: " + ", ".join(signals) + threat_text + tail,
                        **data,
                    )
                else:
                    self._emit(
                        "engine", "debug", f"analysed ply {event.ply}" + threat_text + tail, **data
                    )
        return opening

    @staticmethod
    def _sans_of(fen: str, ucis: list[str], chess: Any, *, by_opponent: bool = False) -> list[str]:
        """UCI moves from `fen` as SAN, stopping at the first that does not parse.

        A threat is what the side NOT to move would play if it could, so it is read after a
        null move; naming it on the wrong side's turn makes python-chess call a quiet pawn
        push a capture ("dxd4").
        """
        try:
            board = chess.Board(fen)
        except ValueError:
            return list(ucis)
        if by_opponent:
            board.push(chess.Move.null())
        out: list[str] = []
        for uci in ucis:
            try:
                move = chess.Move.from_uci(uci)
                out.append(board.san(move))
                board.push(move)
            except (ValueError, AssertionError):
                break
        return out

    def _coach_meta(self, coaching: Any, seen: int, failures: int) -> tuple[int, int]:
        """Log each new coach message with why it spoke and what it was checked against, speak
        it, and log any new LLM failure."""
        coach = coaching.coach
        if coach is None:
            return seen, failures
        for spoken in coach.spoken[seen:]:
            message = spoken.message
            meta = [
                message.trigger.value,
                f"L{int(spoken.level)}",
                f"ply {message.ply}" if message.ply is not None else "no ply",
            ]
            if spoken.attempts:
                meta.append(f"{spoken.attempts} attempt{'s' if spoken.attempts != 1 else ''}")
            if spoken.problems:
                meta.append("draft rejected → template")
            action = self._speak(message.trigger.value, message.text, message.ply)
            if action == "stale":
                meta.append("not spoken: board moved on")
            elif action == "skip":
                meta.append("not spoken: voice busy")
            elif action == "supersede":
                meta.append("cut in")
            self._emit(
                "coach",
                "warn" if spoken.problems else "info",
                " · ".join(meta) + " — " + elide(message.text, 80),
                full=message.text,
                trigger=message.trigger.value,
                disclosure=int(spoken.level),
                ply=message.ply,
                attempts=spoken.attempts,
                problems=spoken.problems,
                fallback=bool(spoken.problems),
                voice=action,
                fen=message.fen,
                annotations=[mark.model_dump() for mark in message.annotations],
            )
        seen = len(coach.spoken)
        speaker = coach.speaker
        if speaker.failures > failures:
            self._emit("coach", "error", f"LLM call failed: {speaker.last_error}")
            failures = speaker.failures
        return seen, failures

    def _coach_silence(self, coaching: Any, last: str | None) -> str | None:
        """At debug, why the coach is quiet -- only when the reason changes."""
        coach = coaching.coach
        if coach is None:
            return last
        reason = coach.policy.last_reason
        if reason is not None and reason != last:
            self._emit("coach", "debug", f"quiet: {reason}")
        return reason

    def _health(
        self,
        pipe: Any,
        coaching: Any,
        recorder: Any,
        ticks: int,
        pipeline_ms: list[float],
        elapsed: float,
        ts: float,
        live: bool,
    ) -> None:
        """The status strip: what `watch` prints every few seconds, kept current instead."""
        timings = sorted(pipeline_ms) or [0.0]
        p50 = timings[len(timings) // 2]
        p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
        if pipe.board is None:
            lock = "no board"
        elif pipe.tracker.verified:
            lock = f"locked, fit {pipe.board.fit:.2f}"
        else:
            lock = "lock in doubt"
        pending = coaching.coach.speaker.pending if coaching.coach is not None else 0
        self._set(
            health={
                "fps": round(ticks / elapsed, 1) if elapsed else 0.0,
                "lock": lock,
                "stage": "following" if pipe.model is not None else "waiting for a start",
                "pipeline_ms": [round(p50), round(p95)],
                # A recording's timestamps are on its own clock, so lag only means
                # something for a live capture.
                "lag_ms": round((time.monotonic() - ts) * 1000.0) if live else None,
                "recording": (
                    f"{recorder.stored} frames, {recorder.size_bytes() / 1e6:.0f} MB"
                    if recorder is not None
                    else "off"
                ),
                "coach_pending": pending,
            }
        )

    # ---- controls (loop thread) ---------------------------------------------------------------

    def _drain_commands(
        self, pipe: Any, coaching: Any, frame: Any, geom: Any, ts: float
    ) -> list[Any]:
        events: list[Any] = []
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                break
            if name == "ask":
                if coaching.on:
                    coaching.ask(str(payload), ts)
                    self._emit("user", "info", f"you: {payload}")
                else:
                    self._emit("user", "warn", f"no coach to ask: {payload}")
            elif name == "flip":
                events += self._flip(pipe)
            elif name == "analyze":
                events += self._analyze(pipe, frame, geom)
        return events

    def _flip(self, pipe: Any) -> list[Any]:
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
            self._emit(
                "user", "info", "view flipped (the game is still read from the screen as is)"
            )
            return []
        current = self._forced_bottom or pipe.bottom or Side.WHITE
        target = Side.BLACK if current == Side.WHITE else Side.WHITE
        self._forced_bottom = target
        self._set(bottom=target.value)
        self._emit("user", "info", f"analysis orientation: {target.value} at the bottom")
        return []

    def _analyze(self, pipe: Any, frame: Any, geom: Any) -> list[Any]:
        if frame is None or geom is None:
            self._emit("vision", "warn", "no board on screen to analyse yet")
            return []
        try:
            events = pipe.cold_start(frame, geom, bottom=self._forced_bottom)
        except Exception as exc:  # noqa: BLE001
            self._emit("vision", "error", f"analyse failed: {exc!r}")
            return []
        if not events:
            self._emit(
                "vision",
                "warn",
                "could not read the position cold: no cached style match, coordinates off "
                "(use Flip), or the board is mid-move",
            )
        else:
            self._emit("vision", "info", "analysing the position on the board (cold start)")
        return events

    def _refresh_state(self, pipe: Any) -> None:
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
    """The control surface the server exposes over HTTP. Thin over the engine."""

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
