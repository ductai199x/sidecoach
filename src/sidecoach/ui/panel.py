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
  voice   what was spoken or muted, what was heard (and how it was corrected), the
          microphone, barge-ins, and any synthesis or recognition failure
  user    the user's asks, typed or spoken
Continuously-changing health (fps, lock fit, pipeline timing, lag, clock, recording size) is
not logged at all; it lives in the state as a status strip.

The engine takes an injectable frame source, so it can be driven from a recording -- in a
test, or with `sidecoach panel --replay` while developing -- without a live screen.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from sidecoach.config import Settings
from sidecoach.ui.log import DebugLog, elide
from sidecoach.voice import STTProvider, TTSProvider, make_recogniser, make_tts
from sidecoach.voice.correct import correct
from sidecoach.voice.hotkey import GlobalPushToTalk
from sidecoach.voice.mic import MODES, Listener, Microphone
from sidecoach.voice.vad import SileroGate
from sidecoach.voice.voices import english_voices, voice_label

# The coach settings the annotator's thresholds are built from (analysis.score.Thresholds).
THRESHOLD_FIELDS = frozenset(
    f"coach.{name}"
    for name in (
        "inaccuracy",
        "mistake",
        "blunder",
        "decisive_gap",
        "only_move_gap",
        "excellent",
        "decided",
    )
)
# The voice settings that rebuild the voice or speech recognition when saved (a model reloads).
VOICE_RELOAD = frozenset(f"voice.{name}" for name in ("lang", "model_dir", "tts_device"))
# The captions' settings (`caption_settings`); a change to any but `keep` shows a sample line.
CAPTION_SETTINGS = (
    "captions",
    "captions_keep",
    "captions_text_color",
    "captions_background",
    "captions_opacity",
    "captions_text_size",
)
CAPTION_PREVIEWED = frozenset(f"voice.{n}" for n in CAPTION_SETTINGS if n != "captions_keep")
# The words the captions' window can still catch up on, at a poll every quarter second.
CAPTIONS_KEPT = 20


def caption_settings(settings: Settings, *, preview: int) -> dict[str, Any]:
    """The captions' settings as their window reads them, and `preview`: when their look last
    changed (ms since the epoch; 0, never), when the window shows a sample line (the phone's
    `preview`: Tai, 2026-09-23, "applies instantly on save"). A time, not a count: the window
    is built when captions are turned on, after the change that turned them on."""
    voice = settings.voice
    return {
        "on": voice.captions,
        "keep": voice.captions_keep,
        "text": voice.captions_text_color,
        "background": voice.captions_background,
        "opacity": voice.captions_opacity,
        "size": voice.captions_text_size,
        "preview": preview,
    }


STT_RELOAD = frozenset(f"voice.{name}" for name in ("stt_device", "mic_device"))
# The warm-up's stages, in the order they finish, as the splash screen shows them: one
# square of a rank each, the pieces landing left to right. A stage is pending, loading,
# done, or off (missing or failed, with the reason).
WARM_STAGES: tuple[tuple[str, str], ...] = (
    ("pieces", "Piece recognition"),
    ("voice", "Voice"),
    ("stt", "Speech recognition"),
    ("mic", "Microphone"),
    ("vision", "Board reading"),
    ("engine", "Stockfish"),
    ("coach", "Coach"),
    ("ready", "Ready"),
)
# The warm-up stage each voice model reports as (the log calls them by these names).
STAGE_OF = {"voice": "voice", "speech recognition": "stt"}
# Words said go into the text box as they are recognised and are sent as the ask this
# long after the utterance ends -- unless the person presses again and adds to them, or
# types, which makes the text theirs to edit and send (Tai, 2026-09-22).
DICTATION_SEND_DELAY_S = 1.0
# The settings page's microphone test turns itself off this long after the page's
# last look at it.
HEAR_TEST_TIMEOUT_S = 20.0
# What the settings page's Listen button says, in the voice being tried.
VOICE_SAMPLE = (
    "That's the move -- rooks belong on open files. Now, what is your opponent's most "
    "dangerous idea?"
)

Frame = tuple[int, float, Any]  # (frame_id, ts, frame)
FrameSource = Iterable[Frame]

# How a game ends when the board cannot say so (a resignation, a flag, an agreed draw):
# outcome -> (result from the user's side, reason). "forget" is not an outcome but a reset.
OUTCOMES = ("resigned", "won", "draw", "forget")

# Unreadable cells (unknown or covered) beyond which a frame's reading is called degraded,
# and how many frames in a row it must stay that way before it is worth a line.
DEGRADED_CELLS = 6
DEGRADED_FRAMES = 5
# How far the board may have moved on (in plies) before a message is shown but not spoken.
# A post-move verdict is about the position before the user's move: after the move (+1) and
# the bot's instant reply (+2) it is still the news; once the user has moved again (+3) it
# is stale. A hint about the live position is moot the moment the user moves (+1).
STALE_AFTER = {"post_move": 2, "post_mistake": 2, "decisive_question": 0, "stuck": 0}
STALE_PLIES = 2  # the fallback for any other trigger
# A "verdict" this short (post-move praise alone, "Good.") is skipped when the voice is busy;
# one with a lesson in it is longer, and waits its turn.
SHORT_VERDICT = 40
# A fast game (coach.fast_pace): the coach's words are shown, not spoken -- a sentence
# would be talked over the next two moves. The last unspoken word is said after all
# once the player has been thinking in that position this long: they slowed down, and
# a lesson or a question is worth hearing then (Tai, 2026-09-22).
PACE_RESUME_S = 6.0
HEALTH_EVERY_S = 1.0
# A tick of the engine loop that takes longer than this is logged with where the time
# went: the capture, the recorder, the vision, or the rest (engine, profile, coach). A
# live game froze for 323 s once (2026-09-19) with nothing in the log to say which.
STALL_WARN_S = 2.0


class FrameRing:
    """The last frames the pipeline saw, in memory only, as JPEGs at half scale: about
    200 KB each, so `maxlen` 150 frames (a minute at a changing board) stay under 40 MB.
    Encoding costs a few milliseconds per changed frame and nothing on a still one."""

    def __init__(self, maxlen: int = 150, quality: int = 85) -> None:
        self._frames: deque[tuple[float, bytes]] = deque(maxlen=maxlen)
        self._quality = quality
        self._lock = threading.Lock()

    def push(self, ts: float, frame: Any) -> None:
        try:
            import cv2

            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
        except Exception:  # noqa: BLE001 - a frame the ring cannot keep is not an error
            return
        if ok:
            with self._lock:
                self._frames.append((ts, buf.tobytes()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def write(self, folder: Path) -> int:
        """The frames as files under `folder`, named by their timestamp; returns the count."""
        with self._lock:
            frames = list(self._frames)
        for ts, data in frames:
            (folder / f"{ts:014.3f}.jpg").write_bytes(data)
        return len(frames)


class _VoiceLogHandler(logging.Handler):
    """Forwards the voice provider's warnings (a failed load, a failed line) to the log."""

    def __init__(self, engine: PanelEngine) -> None:
        super().__init__(level=logging.WARNING)
        self._engine = engine

    def emit(self, record: logging.LogRecord) -> None:
        level = "error" if record.levelno >= logging.ERROR else "warn"
        self._engine._emit("voice", level, record.getMessage())


# What the person is told when a board stays unread (`BoardUnread`), by why.
BOARD_UNREAD = {
    "unlearned": (
        "I can't read this board yet: I haven't learned these pieces on it. Show me the "
        "starting position with this board and these pieces once -- a new game, or an "
        "analysis board at the start -- and I'll know them from then on, puzzles included."
    ),
    "orientation": (
        "I can see the pieces, but not which side is at the bottom: the board's "
        "coordinates seem to be off. Turn them on in the site's board settings, or press "
        "Flip to tell me."
    ),
}


@dataclass
class PanelEngine:
    """The live coaching loop behind the panel. Thread-safe to read; controlled by commands."""

    settings: Settings
    monitor: int | None = None
    fps: int = 10
    # Record a live session (frames + events.jsonl + debug.jsonl) like `watch` does. A replay
    # is never recorded; it still gets a session folder for its debug log.
    record: bool = True
    # The app's mode (docs/app-plan.md, Tai 2026-09-21): with persist off nothing of a
    # session reaches the disk on its own -- no session folder, no debug.jsonl, no
    # recording -- only the profile, which is the player's own model. The recent frames
    # live in `frames`, a ring in memory, and `dump()` writes them out when asked.
    persist: bool = True
    # Whether the voice models are built at all. Off, the panel has a silent voice and
    # speech recognition off, with a reason; SIDECOACH_VOICE_MODELS=0 is the tests' default
    # (no worker loads Kokoro and Parakeet on the GPU; 2026-09-20 that, not the vision, was
    # what lagged the machine) and a headless run's.
    voice_models: bool = field(
        default_factory=lambda: os.environ.get("SIDECOACH_VOICE_MODELS", "1") != "0"
    )
    analysis: bool = True
    coach: bool = True
    # The speaking voice: the settings' unless given (a test); "" means the settings'.
    voice: str = ""
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
    _stt: STTProvider = field(default=None, init=False)  # type: ignore[assignment]
    _gate: SileroGate = field(default_factory=SileroGate, init=False)
    _mic: Microphone = field(default=None, init=False)  # type: ignore[assignment]
    _listener: Listener = field(default=None, init=False)  # type: ignore[assignment]
    _hotkey: GlobalPushToTalk = field(default=None, init=False)  # type: ignore[assignment]
    _forced_bottom: Any = field(default=None, init=False)  # a Side chosen by the flip button
    _view_flip: bool = field(default=False, init=False)  # the panel's board shown turned around
    _fen_before: dict[int, str] = field(default_factory=dict, init=False)  # ply -> fen before
    # ply -> the engine's label for the move played to reach it (best, blunder, ...), for
    # the move list's badges.
    _labels: dict[int, str] = field(default_factory=dict, init=False)
    # The game's last ply as the moves came, so a takeback knows which plies it withdrew.
    _last_ply: int = field(default=0, init=False)
    _bad_run: int = field(default=0, init=False)  # consecutive frames read as degraded
    # What the voice was last given: the trigger and the ply of that message, so the
    # next one knows whether what is playing is about a position the board has left.
    _voicing: tuple[str, int | None] | None = field(default=None, init=False)
    # Counts the times the voice was cut off by the player (a barge-in, a mute): a reply
    # being written sentence by sentence says nothing more once it moves.
    _hushed: int = field(default=0, init=False)
    # Counts the presses and utterances begun (see _dictation_settle).
    _dictation_holds: int = field(default=0, init=False)
    # The puzzles recommended lately, in order, and the motif each was for: the puzzle
    # page (ui/puzzle_room.py) opens them in a tab of their own, with Next in that order.
    practice_ids: list[str] = field(default_factory=list, init=False)
    puzzle_motifs: dict[str, str] = field(default_factory=dict, init=False)
    # Latency, amortised over the session (see _health): when each move was committed
    # (ply -> monotonic), the reply's first sentence waiting for its audio (the line ->
    # the move's time), and the running (count, total, last) of move-to-first-sound and
    # of the model's first token.
    _move_at: dict[int, float] = field(default_factory=dict, init=False)
    _awaiting_audio: dict[str, float] = field(default_factory=dict, init=False)
    _lat_speech: list[float] = field(default_factory=lambda: [0, 0.0, 0.0], init=False)
    _lat_token: list[float] = field(default_factory=lambda: [0, 0.0, 0.0], init=False)
    # Screen rectangles (x, y, w, h in physical pixels) painted flat before the pipeline
    # sees a live frame -- the panel's own board, so the vision never locks onto it.
    _exclude: list[tuple[int, int, int, int]] = field(default_factory=list, init=False)
    # The engine and the coach glue (cli._Assessing, cli._Coaching). They belong to the
    # panel's lifetime, not a session's: `warm()` builds them when the app opens, Start
    # and Stop only start and stop reading the screen (Tai, 2026-09-21), and `close()`
    # ends them at exit. A settings save reaches into them whenever they exist.
    _assessing: Any = field(default=None, init=False)
    _coaching: Any = field(default=None, init=False)
    # Recording is the saved opt-in `capture.record` (Settings, or Debug > Record sessions,
    # which writes the same setting), or the launch flags in development
    # (--record with --persist). A running session starts or stops its recording within a
    # tick of the setting changing (Tai, 2026-09-21: off by default, but there must be a
    # way to record a bug; 2026-09-22: a setting, and on the spot). A recording that
    # failed (a display change, a full disk) stays off until the next Start.
    _record_failed: bool = field(default=False, init=False)
    # Dictation: the words on the screen so far and the timer that sends them.
    _dictation_text: str = field(default="", init=False)
    _dictation_timer: threading.Timer | None = field(default=None, init=False)
    _partial_busy: threading.Lock = field(default_factory=threading.Lock, init=False)
    _utterance_gen: int = field(default=0, init=False)  # bumped when an utterance ends
    _warm_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _provisioning: threading.Lock = field(default_factory=threading.Lock, init=False)
    _warm_done: threading.Event = field(default_factory=threading.Event, init=False)
    _warm_stages: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)
    _warm_begun: float = field(default=0.0, init=False)
    # A settings save that changed what the engine was built from (the Stockfish binary,
    # the tools folder): the next Start rebuilds it.
    _engine_stale: bool = field(default=False, init=False)
    # The coach's last word held back for pace: (trigger, text, ply, when), spoken once
    # the player slows down in that position.
    _held_word: tuple[str, str, int | None, float] | None = field(default=None, init=False)
    _promotion_logged: bool = field(default=False, init=False)
    _loop_ts: float = field(default=0.0, init=False)  # the last frame's time (the loop's clock)
    # The settings page's live test of the microphone and speech recognition: while on, it
    # listens (voice activity) and what it hears goes to `_heard_test`, not to the coach;
    # the mode it interrupted is restored after. It switches itself off when the page
    # stops asking (HEAR_TEST_TIMEOUT_S).
    _hear_test_on: bool = field(default=False, init=False)
    _hear_test_mode: str = field(default="off", init=False)
    _hear_test_seen: float = field(default=0.0, init=False)
    _heard_test: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=6), init=False)
    frames: FrameRing = field(default_factory=lambda: FrameRing(), init=False)
    # Other model-call ledgers to count in the strip (the puzzle rooms'): callables
    # returning llm.usage.Usage objects.
    extra_usage: list[Callable[[], list[Any]]] = field(default_factory=list, init=False)
    # Who keeps the settings besides the engine -- the review and puzzle rooms, the setup, the
    # background reviewer (ui/server.py) -- each handed every new copy, so a save is seen there
    # at once, as on the phone, which reads its store live (the parity audit, 2026-10-01).
    settings_followers: list[Callable[[Settings], None]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # The voice the settings name, unless the caller named one (Tai, 2026-09-21: the
        # saved voice was ignored at start, the field's old default always won).
        self.voice = self.voice or self.settings.voice.voice
        # Muted as the last session was left (Tai, 2026-10-01).
        self.muted = self.muted or self.settings.voice.muted
        # Each voice model is built exactly once here and kept for the engine's life; the
        # graphs load on the device configured (see `_warm_voice`) and are never reloaded.
        self._tts = self._build_tts(self.voice)
        self._hook_voice()
        self._stt = make_recogniser(self.settings, off=not self.voice_models)
        self._mic = Microphone(self.settings.voice.mic_device)
        self._listener = Listener(
            self._mic,
            self._gate,
            on_utterance=self._on_utterance,
            on_speech_start=self._on_speech_start,
            coach_speaking=lambda: self._tts.busy,
            on_event=lambda level, text: self._emit("voice", level, text),
            on_partial=self._on_partial,
            on_dropped=self._on_dropped,
        )
        # Held from any window -- the game has the focus mid-game, not the panel.
        self._hotkey = GlobalPushToTalk(
            self.settings.voice.ptt_key,
            on_press=lambda: self.ptt(True),
            on_release=lambda: self.ptt(False),
        )
        self._state = {
            "running": False,
            "fen": None,
            "moves": [],
            "fens": [],
            "ucis": [],
            "labels": {},
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
            "voices": english_voices(self.settings.voice_model_dir() / "voices-v1.0.bin"),
            "muted": self.muted,
            "recording_dir": None,
            "record_wanted": self._recording_wanted(),
            "coach_on": self.coach,
            "analysis_on": self.analysis,
            "voice_detail": self._tts.healthcheck()[1],
            "stt_detail": self._stt.healthcheck()[1],
            # Whether the page's listening button has anything behind it. True while the
            # models are still being built, so it does not appear partway through a run;
            # the warm-up replaces it with the provider's own answer.
            "stt_ready": self.voice_models,
            "listen": "off",
            "ptt": False,
            "ptt_key": self.settings.voice.ptt_key,
            "heard": None,
            # What is being said, as recognised so far: text, whether the microphone is
            # still capturing it, and when it sends (a wall-clock ms timestamp, or null).
            "dictation": {"text": "", "live": False, "sends_at": None},
            "clock": None,
            "health": {},
            "session_dir": None,
            # The review of the last game to end, and the player model as it stands.
            "review": None,
            "profile": None,
            # The captions' window (desktop/src/routes/captions): its settings, the words the
            # voice was given, numbered, the last CAPTIONS_KEPT of them, and where the board
            # the vision follows is on the screen, in pixels (none while it follows none).
            "captions": caption_settings(self.settings, preview=0),
            "caption_lines": [],
            # Raised when a line captioned earlier is said now (a word held back in a fast
            # game): the box, faded since, comes back for it.
            "caption_again": 0,
            # Raised by every change the server makes to the settings or a piece set: the main
            # window reads it with each poll and tells the other windows (lib/panel.svelte.ts).
            "settings_version": 0,
            "board_screen": None,
        }
        self._captioned = 0  # the words captioned so far, the last one's number
        self._profiler: Any = None
        # How much of each of the coach's growing lists has been logged, by the list itself: a
        # Start makes a new coach with new lists, and a count kept from the last one skipped
        # the new one's first entries.
        self._logged: dict[str, tuple[list[Any], int]] = {}
        self._failures_seen: tuple[Any, int] = (None, 0)  # the coach's speaker, its failures logged
        logging.getLogger("sidecoach.voice").addHandler(_VoiceLogHandler(self))
        if self.settings.unreadable:
            self._emit("sys", "warn", self.settings.unreadable)  # where its copy is kept

    # ---- control surface (called from the API / GUI thread) -------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="panel-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def warm(self, background: bool = True) -> None:
        """Load everything a session needs before any Start: the embedding net, the voice
        and speech recognition, Stockfish, the coach and its model. Off the caller's thread unless
        told otherwise; a Start that comes first waits for it."""
        if background:
            threading.Thread(target=self._warm, name="warm", daemon=True).start()
        else:
            self._warm()

    def _warm(self) -> None:
        with self._warm_lock:
            self._warm_done.clear()
            self._warm_begun = time.monotonic()
            with self._lock:
                self._warm_stages = {
                    name: {"name": name, "label": label, "state": "pending"}
                    for name, label in WARM_STAGES
                }
            try:
                from sidecoach.vision.pieces import embedder

                self._stage("pieces", "loading")
                net = embedder()
                if net is not None:
                    primed = net.prime()
                    self._emit(
                        "sys",
                        "info",
                        f"pieces: embedding net on {net.device.detail}, primed in {primed:.1f}s",
                    )
                    self._stage("pieces", "done", f"on {net.device.detail}")
                else:
                    self._emit(
                        "sys", "warn", "pieces: no embedding net (a broken install): no set learned"
                    )
                    self._stage("pieces", "off", "no embedding net: no piece set can be learned")
                self._warm_voice()
                self._stage("vision", "loading")
                self._warm_pipeline()
                self._stage("vision", "done")
                self._stage("engine", "loading")
                self._stage("coach", "loading")
                assessing, coaching = self._prepare_stages()
                if not assessing.on:
                    self._stage("engine", "off", assessing.detail)
                if not coaching.on:
                    self._stage("coach", "off", coaching.detail)
                self._warm_engines()
                if assessing.on:
                    self._stage("engine", "done", assessing.detail)
                # In the background: a local model's first load took up to a minute, and
                # the splash waited on it (setup finding 18). The coach is ready to be
                # asked; its first answer waits for the model, as without a warm-up.
                threading.Thread(target=self._warm_model, name="warm-model", daemon=True).start()
                if coaching.on:
                    self._stage("coach", "done", coaching.detail)
                self._stage("ready", "done")
            except Exception as exc:  # noqa: BLE001 - the session builds what is missing
                self._emit("sys", "warn", f"warm-up: {exc!r}")
                with self._lock:
                    for stage in self._warm_stages.values():
                        if stage["state"] in ("pending", "loading"):
                            stage["state"] = "off"
                            stage["detail"] = f"{exc!r}"
            finally:
                self._warm_done.set()

    def _stage(self, name: str, state: str, detail: str = "") -> None:
        """One warm-up stage's state, with the seconds since the warm-up began."""
        with self._lock:
            stage = self._warm_stages.get(name)
            if stage is None:
                return
            stage["state"] = state
            stage["detail"] = detail
            stage["seconds"] = round(time.monotonic() - self._warm_begun, 1)

    def warm_status(self) -> dict[str, Any]:
        """What the splash screen polls: every stage and whether the warm-up is over."""
        with self._lock:
            stages = [dict(s) for s in self._warm_stages.values()]
        seconds = round(time.monotonic() - self._warm_begun, 1) if self._warm_begun else 0.0
        return {"done": self._warm_done.is_set(), "seconds": seconds, "stages": stages}

    def _warm_pipeline(self) -> None:
        """The session's modules imported (~0.2 s) and the vision's first pass paid (cv2
        and numpy set themselves up on the first board search, ~0.2 s), so the first
        frame is read at speed. Nothing here touches the screen."""
        import numpy as np
        import psutil  # noqa: F401

        from sidecoach.capture import loop, wgc_backend  # noqa: F401
        from sidecoach.profile import ask, profiler, puzzles, store  # noqa: F401
        from sidecoach.replay import recording  # noqa: F401
        from sidecoach.state import pipeline  # noqa: F401
        from sidecoach.vision import clock, squares, theme_cache  # noqa: F401
        from sidecoach.vision.ownboard import is_own_board
        from sidecoach.vision.tracker import BoardTracker

        begun = time.perf_counter()
        BoardTracker(reject=is_own_board).update(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self._emit(
            "sys", "info", f"vision: first pass primed in {time.perf_counter() - begun:.1f}s"
        )

    def _warm_engines(self) -> None:
        """One search of the start position on the analyser's Stockfish, so the first real
        one does not pay the hash allocation and the thread start (~0.3 s). Not the coach's:
        it starts only when a question needs a position nothing has computed (Tai,
        2026-09-29), and warming it started it at every launch, as the phone never did."""
        import chess

        analyser = getattr(self._assessing, "analyser", None)
        if analyser is None:
            return
        begun = time.perf_counter()
        with contextlib.suppress(Exception):
            analyser.annotator.engine.analyse(chess.Board(), depth=analyser.annotator.depth)
        self._emit("sys", "info", f"engine: primed in {time.perf_counter() - begun:.1f}s")

    def _warm_model(self) -> None:
        """A model server on this machine loads its model on the first request (LM Studio:
        seconds to a minute); one short request now, so the coach's first line is not
        the one that waits. A paid provider is never called for this."""
        from urllib.parse import urlparse

        coach = getattr(self._coaching, "coach", None)
        provider = getattr(getattr(coach, "speaker", None), "provider", None)
        if provider is None or getattr(provider, "offline", False):
            return
        host = urlparse(str(getattr(provider, "base_url", "") or "")).hostname or ""
        if host not in ("localhost", "127.0.0.1", "::1"):
            return
        begun = time.perf_counter()
        try:
            provider.chat(
                "Reply with the one word: ready.",
                [{"role": "user", "content": "Ready?"}],
                max_tokens=8,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001 - the coach reports the model when asked
            self._emit("sys", "warn", f"coach: the local model did not answer a warm-up: {exc}")
            return
        self._emit(
            "sys", "info", f"coach: local model answered in {time.perf_counter() - begun:.1f}s"
        )

    def _prepare_stages(self) -> tuple[Any, Any]:
        """Stockfish and the coach, built once and kept; rebuilt only when a save changed
        what they were built from. Returns them."""
        from sidecoach.cli import _Assessing, _Coaching
        from sidecoach.config import find_stockfish

        # Built before there was an engine, and there is one now (Setup fetched it after
        # the warm-up): built again, and not only at the next launch.
        if (
            self._assessing is not None
            and not self._assessing.on
            and self.analysis
            and find_stockfish(self.settings) is not None
        ):
            self._engine_stale = True
        if self._engine_stale:
            # The coach too: its second engine runs the same binary, and it was built
            # knowing whether there was an engine at all.
            for stage in (self._assessing, self._coaching):
                if stage is not None:
                    with contextlib.suppress(Exception):
                        stage.close()
            self._assessing = self._coaching = None
            self._engine_stale = False
        # The opening tables fetched after the engine was built (Setup, on a first run):
        # the kept annotator takes the book now rather than at the next launch.
        analyser = getattr(self._assessing, "analyser", None)
        if analyser is not None and analyser.annotator.book is None:
            from sidecoach.cli import _load_book

            analyser.annotator.book = _load_book(self.settings)
        if self._assessing is None:
            assessing = _Assessing(self.settings, self.analysis)
            assessing.sink = lambda _line: None
            self._assessing = assessing
            self._emit("sys", "info" if assessing.on else "warn", f"engine: {assessing.detail}")
        if self._coaching is None:
            coaching = _Coaching(
                self.settings, self.coach and self._assessing.on, self._assessing.computed
            )
            coaching.sink = lambda _message: None
            if coaching.coach is not None:
                coaching.coach.stream_sink = self._stream_sink
            self._coaching = coaching
            self._emit("sys", "info" if coaching.on else "warn", f"coach: {coaching.detail}")
        return self._assessing, self._coaching

    def close(self) -> None:
        """The panel is going away: stop the session, close Stockfish and the coach, let go
        of the microphone, the push-to-talk key and the voice."""
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        for stage in (self._assessing, self._coaching):
            if stage is not None:
                with contextlib.suppress(Exception):
                    stage.close()
        self._assessing = self._coaching = None
        with contextlib.suppress(Exception):
            self._hotkey.stop()
        with contextlib.suppress(Exception):
            self._listener.set_mode("off")
        close_voice = getattr(self._tts, "close", None)
        if close_voice is not None:
            with contextlib.suppress(Exception):
                close_voice()

    def command(self, name: str, payload: Any = None) -> None:
        """Queue a control action for the loop thread (ask, flip, analyse). With no
        session running there is no loop to take it: an ask is answered on the spot
        rather than kept for a Start minutes later, about another game -- in the captions too,
        where a question asked from their buttons looks for its answer."""
        with self._lock:
            running = bool(self._state.get("running"))
        if not running and name in ("ask", "ask_spoken"):
            self._emit("user", "info", f"you: {payload}", spoken=name == "ask_spoken")
            text = "Press Start first: I can only answer about a game I am watching."
            self._emit(
                "coach",
                "info",
                f"user_ask · not watching — {text}",
                full=text,
                trigger="user_ask",
                ply=None,
                voice="text",
            )
            self.caption(text)
            return
        self._commands.put((name, payload))

    def set_voice(self, voice: str) -> None:
        """Change the speaking voice. A style vector on the loaded model: nothing reloads."""
        set_voice = getattr(self._tts, "set_voice", None)
        if set_voice is not None:
            set_voice(voice)
        with self._lock:
            self.voice = voice
            self._state["voice"] = voice
            self._state["voice_detail"] = self._tts.healthcheck()[1]
        self._emit("voice", "info", f"voice set to {voice}", detail=self._state["voice_detail"])

    def adopt_settings(self, new: Settings) -> None:
        """The settings saved are the engine's now, and every follower's."""
        with self._lock:
            self.settings = new
        for follow in list(self.settings_followers):
            follow(new)
        self.settings_touched()

    def settings_touched(self) -> None:
        """The settings or a piece set changed: every window draws from them again."""
        with self._lock:
            self._state["settings_version"] = int(self._state.get("settings_version", 0)) + 1

    def apply_settings(self, new: Settings, changed: Sequence[str]) -> list[str]:
        """The settings saved: what can change on the running session does, and is
        returned as the list of dotted names applied. Everything the models load at
        start (a device, a provider of voice or speech recognition, the data directory) waits for a
        restart; the screen's backend and the Stockfish binary for the next Start. Each
        change is applied on its own: one that fails is named in the log and left out of
        the list, and the others still go (one raise used to skip the rest silently)."""
        self.adopt_settings(new)
        applied: list[str] = []
        changed = list(changed)

        def attempt(names: Sequence[str], apply: Callable[[], object]) -> None:
            """Apply one change (or one group) if any of `names` changed."""
            hit = [n for n in names if n in changed]
            if not hit:
                return
            try:
                apply()
            except Exception as exc:  # noqa: BLE001 - the page must hear of it
                self._emit("sys", "warn", f"{', '.join(hit)}: not applied ({exc})")
                return
            applied.extend(hit)

        def set_record() -> None:
            # The session loop reads the wish every tick; the page's switch follows now.
            self._record_failed = False
            self._set(record_wanted=self._recording_wanted())

        attempt(["capture.record"], set_record)
        attempt(["voice.voice"], lambda: self.set_voice(new.voice.voice))
        attempt(["voice.listen"], lambda: self.set_listen(new.voice.listen))
        attempt(["voice.muted"], lambda: self.set_muted(new.voice.muted))
        if hasattr(self._tts, "speed"):
            attempt(["voice.speed"], lambda: setattr(self._tts, "speed", new.voice.speed))
        if hasattr(self._tts, "set_output_device"):
            attempt(
                ["voice.output_device"],
                lambda: self._tts.set_output_device(new.voice.output_device),  # type: ignore[attr-defined]
            )

        def rebind_hotkey() -> None:
            self._hotkey.stop()
            self._hotkey = GlobalPushToTalk(
                new.voice.ptt_key,
                on_press=lambda: self.ptt(True),
                on_release=lambda: self.ptt(False),
            )
            if self._listener.mode == "ptt":
                self.set_listen("ptt")
            self._set(ptt_key=new.voice.ptt_key)

        attempt(["voice.ptt_key"], rebind_hotkey)

        def restyle_captions() -> None:
            # Their window reads them at its next poll; a new look shows a sample line.
            with self._lock:
                preview = int(self._state["captions"]["preview"])
            if any(c in CAPTION_PREVIEWED for c in changed):
                preview = int(time.time() * 1000)
            self._set(captions=caption_settings(new, preview=preview))

        attempt([f"voice.{name}" for name in CAPTION_SETTINGS], restyle_captions)
        if any(c in VOICE_RELOAD for c in changed):
            self._reload_voice()
            applied.extend(c for c in changed if c in VOICE_RELOAD)
        if any(c in STT_RELOAD for c in changed):
            self._reload_recogniser()
            applied.extend(c for c in changed if c in STT_RELOAD)
        attempt(["capture.fps"], lambda: None)  # the capture loop reads the rate every frame
        if any(c in ("engine.stockfish_path", "tools_dir") for c in changed):
            self._engine_stale = True
        for name in changed:
            if name.startswith("llm.price_"):
                applied.append(name)  # the session's cost is priced at each poll
        coach = self._coaching.coach if self._coaching is not None else None
        if coach is not None:
            attempt(
                ["coach.stuck_seconds"],
                lambda: setattr(coach.policy.settings, "stuck_seconds", new.coach.stuck_seconds),
            )
            attempt(
                ["coach.quiet"], lambda: setattr(coach.policy.settings, "quiet", new.coach.quiet)
            )

            def reword() -> None:
                coach.reword(new.coach.persona, (new.profile.rating_low, new.profile.rating_high))

            attempt(["coach.persona", "profile.rating_low", "profile.rating_high"], reword)
            for name in ("local_verdicts", "verbalized_sampling", "summarise_every"):
                attempt(
                    [f"coach.{name}"],
                    lambda name=name: setattr(coach, name, getattr(new.coach, name)),
                )
            attempt(
                ["coach.compact_record"],
                lambda: setattr(coach.conversation, "compact_record", new.coach.compact_record),
            )
        # The model's fields are the settings API's to rebuild, after this (`reprovision`).
        # The engine's settings apply whether a coach runs or not: nested under the coach,
        # they were reported "applied now" and left undone with the coach off (the parity
        # audit, 2026-09-26).
        from sidecoach.analysis.score import Thresholds

        analyser = self._assessing.analyser if self._assessing is not None else None
        if analyser is not None:
            attempt(
                [c for c in changed if c in THRESHOLD_FIELDS],
                lambda: setattr(
                    analyser.annotator, "thresholds", Thresholds.from_settings(new.coach)
                ),
            )
            engine = analyser.annotator.engine
            attempt(
                ["engine.depth"], lambda: setattr(analyser.annotator, "depth", new.engine.depth)
            )

            def multipv() -> None:
                analyser.annotator.multipv = new.engine.multipv
                engine.multipv = new.engine.multipv

            attempt(["engine.multipv"], multipv)
            for name in ("pv_plies", "score_model", "max_seconds"):
                attempt(
                    [f"engine.{name}"],
                    lambda name=name: setattr(engine, name, getattr(new.engine, name)),
                )
            attempt(
                ["engine.threads", "engine.hash_mb"],
                lambda: engine.reconfigure(threads=new.engine.threads, hash_mb=new.engine.hash_mb),
            )
            # What was measured before is not set against what is measured now: a loss
            # compares two searches made the same way (the engine review, 2026-10-01).
            remeasure = [
                *(c for c in changed if c in THRESHOLD_FIELDS),
                "engine.depth",
                "engine.multipv",
                "engine.score_model",
                "engine.max_seconds",
            ]
            if any(c in changed for c in remeasure):
                analyser.annotator.forget_measured()
        return sorted(set(applied))

    def reprovision(self, names: Sequence[str]) -> bool:
        """The model's settings or its key changed (`names`): the running coach's providers are
        rebuilt from the settings as they are when this runs, one rebuild at a time, so of two
        saves close together the later one's model is the one kept. False when it failed, which
        the log says; True with no coach to rebuild, which the next build reads anyway."""
        with self._provisioning:
            coaching = self._coaching
            if coaching is None or coaching.coach is None:
                return True
            try:
                coaching.reprovision(self.settings)
            except Exception as exc:  # noqa: BLE001 - the page must hear of it
                self._emit("sys", "warn", f"{', '.join(names)}: not applied ({exc})")
                return False
            self._emit("sys", "info", f"coach: {coaching.detail}")
            return True

    def _reload_voice(self) -> None:
        """A new voice model, provider, language, folder or device: the old voice is cut
        off, the new one built and loaded off the caller's thread."""
        old = self._tts
        with contextlib.suppress(Exception):
            old.stop()
        # Its worker, model session and output stream are released, not left running.
        close = getattr(old, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()
        self._tts = self._build_tts(self.settings.voice.voice)
        self._hook_voice()
        with self._lock:
            self.voice = self.settings.voice.voice
            self._state["voice"] = self.voice
            self._state["voice_detail"] = self._tts.healthcheck()[1]
        self._emit("voice", "info", f"voice reloading: {self._state['voice_detail']}")
        threading.Thread(target=self._warm_voice, name="voice-reload", daemon=True).start()

    def _reload_recogniser(self) -> None:
        """A new recogniser or microphone: listening pauses, both are rebuilt, listening resumes
        in the mode it was in -- the one the microphone test gives back, while it runs."""
        mode = self._hear_test_mode if self._hear_test_on else self._listener.mode
        with contextlib.suppress(Exception):
            self._listener.set_mode("off")
        self._stt = make_recogniser(self.settings, off=not self.voice_models)
        self._mic = Microphone(self.settings.voice.mic_device)
        self._listener.mic = self._mic
        with self._lock:
            ready, detail = self._stt.healthcheck()
            self._state["stt_detail"] = detail
            self._state["stt_ready"] = ready
        self._emit("voice", "info", f"speech recognition reloading: {self._state['stt_detail']}")
        threading.Thread(target=self._warm_voice, name="stt-reload", daemon=True).start()
        if mode != "off" or self._hear_test_on:  # the test listens again either way
            with contextlib.suppress(Exception):
                self.set_listen(mode)

    def set_hear_test(self, on: bool) -> dict[str, Any]:
        """Start or stop the live test of the microphone and speech recognition."""
        if on and not self._hear_test_on:
            ok, mic = self._mic.healthcheck()
            ready, stt = self._stt.healthcheck()
            if not ok or not ready:
                what = "microphone" if not ok else "speech recognition"
                return {"ok": False, "error": f"{what}: {mic if not ok else stt}"}
            self._hear_test_mode = self._listener.mode
            self._heard_test.clear()
            self._hear_test_on = True
            self._hear_test_seen = time.monotonic()
            with contextlib.suppress(Exception):
                self._listener.set_mode("vad")
            # The recogniser loads on its first utterance otherwise, and the first result would
            # carry the load in its time.
            load = getattr(self._stt, "load", None)
            if load is not None and getattr(self._stt, "device", None) is None:
                threading.Thread(target=load, name="stt-warm", daemon=True).start()
            self._emit("voice", "info", "microphone test on: say something")
        elif not on and self._hear_test_on:
            self._hear_test_on = False
            with contextlib.suppress(Exception):
                self.set_listen(self._hear_test_mode)
            self._emit("voice", "info", "microphone test off")
        return self.hear_status()

    def _note_heard_test(self, raw: str, seconds: float, ms: float) -> None:
        import chess

        board = None
        with self._lock:
            fen = self._state.get("fen")
        if fen:
            with contextlib.suppress(ValueError):
                board = chess.Board(fen)
        heard = correct(raw, board).text if raw else ""
        self._heard_test.appendleft(
            {
                "raw": raw,
                "heard": heard,
                "seconds": round(seconds, 1),
                "stt_ms": round(ms),
                "t": time.time(),
            }
        )
        self._emit(
            "voice",
            "info",
            f"microphone test heard: {raw!r}" + (f" -> {heard!r}" if heard != raw else ""),
        )

    def hear_status(self) -> dict[str, Any]:
        """What the page shows while the test is on: the level and what was heard."""
        self._hear_test_seen = time.monotonic()
        return {
            "ok": True,
            "on": self._hear_test_on,
            "level": round(min(1.0, self._listener.level), 3),
            "heard": list(self._heard_test),
            "microphone": self._mic.healthcheck()[1],
            "speech_recognition": self._stt.healthcheck()[1],
        }

    def say_sample(self, voice: str | None = None, text: str = VOICE_SAMPLE) -> bool:
        """Speak one line in `voice` (the coach's own when None), for the settings page's
        Listen button; the coach's voice is unchanged. False when the voice cannot speak."""
        ready, detail = self._tts.healthcheck()
        if not ready:
            self._emit("voice", "warn", f"preview: the voice cannot speak: {detail}")
            return False
        self._tts.stop()
        self._tts.say(text, voice=voice or None)
        self._emit("voice", "info", f"preview in {voice_label(voice or self.voice)}: {text}")
        return True

    def set_listen(self, mode: str) -> str:
        """How the coach listens: off, push-to-talk, or voice activity."""
        if mode not in MODES:
            raise ValueError(f"listen mode must be one of {MODES}")
        if self._hear_test_on:
            # The microphone test listens until it ends, and then gives back this mode. It gave
            # back the one it found, undoing a Listening saved while it ran, and the save
            # switched its own listening, so it went deaf (the settings review, 2026-10-01).
            self._hear_test_mode = mode
            if self._listener.mode != "vad":  # a reloaded recogniser left it off
                with contextlib.suppress(Exception):
                    self._listener.set_mode("vad")
            self._set(listen=mode, ptt=False)
            self._emit("voice", "info", f"listening: {mode} once the microphone test ends")
            return mode
        detail = self._listener.set_mode(mode)
        self._set(listen=self._listener.mode, ptt=False)
        self._emit("voice", "info", f"listening: {detail}")
        if self._listener.mode == "ptt":
            if self._hotkey.start():
                self._emit("voice", "info", f"push-to-talk: {self._hotkey.detail}")
            else:
                self._emit("voice", "warn", f"push-to-talk key: {self._hotkey.detail}")
        else:
            self._hotkey.stop()
        return self._listener.mode

    def ptt(self, down: bool) -> None:
        """Push-to-talk pressed or released. Pressed, the coach goes quiet at once (the
        listener's speech start cuts what is playing, `_on_speech_start`) and stays quiet
        until the key is up (`_ptt_held`)."""
        if down:
            held = self._listener.mode == "ptt"
            self._set(ptt=held)  # first: nothing is said from here on
            if held:
                self._dictation_hold()
            self._listener.press()
        else:
            self._listener.release()
            self._set(ptt=False)

    def _ptt_held(self) -> bool:
        """The push-to-talk key is down in push-to-talk mode: the voice stays silent."""
        with self._lock:
            return bool(self._state.get("ptt"))

    def reset_profile(self) -> str:
        """Empty the player model (maintenance.reset_profile). The running profiler's
        store stays open on the emptied file; its game in progress is dropped and the
        panel shows a fresh, empty profile."""
        from sidecoach.maintenance import reset_profile

        path = reset_profile(self.settings)
        profiler = self._profiler
        if profiler is not None:
            profiler._game = None
            profiler._reviews = []
            profiler._recommended_for = {}
            self._set(profile=self._profile_state(profiler.snapshot()), review=None, practice=None)
        self.practice_ids = []
        self.puzzle_motifs.clear()
        self._emit("user", "warn", f"profile reset: {path} is empty")
        return str(path)

    def clear_caches(self) -> list[str]:
        """Remove the caches (maintenance.clear_caches): the theme files, and the puzzle
        store rebuilt from its CSVs when this engine holds it open."""
        from sidecoach.maintenance import clear_caches

        puzzles = getattr(self._profiler, "puzzles", None)
        done = clear_caches(self.settings, puzzles)
        for line in done:
            self._emit("user", "warn", f"cache: {line}")
        if not done:
            self._emit("user", "info", "cache: nothing to clear")
        return done

    def end_game(self, outcome: str) -> None:
        """End the game on the user's word: they resigned or flagged, the opponent did, a
        draw was agreed -- or forget the game altogether and wait for the next start."""
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}")
        self.command("end", outcome)

    def correct(self, game_id: str, segment: int, ply: int, cause: str | None) -> bool:
        """The user's word on why a mistake happened, on the review's card: the profile
        moves the evidence and the review is redrawn."""
        from sidecoach.profile.causes import CAUSES

        if cause is not None and cause not in CAUSES:
            raise ValueError(f"cause must be one of {CAUSES}")
        profiler = self._profiler
        if profiler is None:
            return False
        ok = profiler.correct(game_id, int(segment), int(ply), cause)
        if ok:
            with self._lock:
                review = self._state.get("review")
            if review and review.get("game_id") == game_id:
                for tp in review.get("turning_points", []):
                    if tp.get("segment", 0) == int(segment) and tp.get("ply") == int(ply):
                        tp["cause"] = cause
                self._set(review=review, profile=self._profile_state(profiler.snapshot()))
            self._emit(
                "user", "info", f"you corrected the cause of the mistake at ply {ply}: {cause}"
            )
        return ok

    @staticmethod
    def _profile_state(snapshot: Any) -> dict[str, Any]:
        return {
            "games": snapshot.games,
            "insights": list(snapshot.insights),
            "last_session": snapshot.last_session,
        }

    @staticmethod
    def outcome_result(outcome: str, user_side: str | None) -> tuple[str, str]:
        """The PGN result and the reason for an outcome, from the user's side."""
        user_white = user_side != "black"
        if outcome == "resigned":
            return ("0-1" if user_white else "1-0"), "user resigned or lost on time"
        if outcome == "won":
            return ("1-0" if user_white else "0-1"), "opponent resigned or lost on time"
        if outcome == "draw":
            return "1/2-1/2", "draw agreed"
        return "*", "abandoned by the user"

    def set_exclusion(self, x: int, y: int, w: int, h: int) -> None:
        """Tell the vision where the panel's own board is on screen, so it is masked out.

        The page reports its board's screen rectangle; if the game shares that monitor, the
        detector would otherwise find the bigger, cleaner board -- the panel's -- and never
        see a start position. Painted flat grey before recording and processing.
        """
        rect = (int(x), int(y), int(w), int(h))
        with self._lock:
            changed = rect not in self._exclude
            self._exclude = [rect] if w > 0 and h > 0 else []
        if changed and w > 0 and h > 0:
            self._emit(
                "vision", "info", f"masking the panel's own board at ({x},{y}) {w}x{h}", rect=rect
            )

    @staticmethod
    def mask(frame: Any, region: Any, rects: list[tuple[int, int, int, int]]) -> None:
        """Paint `rects` (screen pixels) flat over `frame`, which covers `region`. In place."""
        margin = 24  # window chrome and DPI rounding: err on the side of covering more
        for x, y, w, h in rects:
            x0 = max(0, x - margin - region.x)
            y0 = max(0, y - margin - region.y)
            x1 = min(region.w, x + w + margin - region.x)
            y1 = min(region.h, y + h + margin - region.y)
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = 128

    def hear(self, audio: Any, seconds: float | None = None) -> None:
        """Treat a float32 16 kHz clip as something the user said (the test and dev path)."""
        import numpy as np

        clip = np.asarray(audio, dtype=np.float32)
        self._on_utterance(clip, seconds if seconds is not None else len(clip) / 16000)

    def _recording_wanted(self) -> bool:
        """Record: the saved opt-in, or the development launch flags."""
        return bool(self.settings.capture.record or (self.record and self.persist))

    def set_record(self, on: bool) -> dict[str, Any]:
        """The Debug menu's switch: the same setting as Settings > Screen > Record sessions,
        saved to the settings file, applied within a tick of a running session. Saved as every
        other settings write is (`settings_api.remember`): written on its own, a Settings save
        at the same moment put it back, and the change counter the other windows follow stayed
        still (the settings review, 2026-10-01)."""
        from sidecoach.ui import settings_api

        with settings_api.saving:
            try:
                settings_api.remember(self, {"capture": {"record": on}})
            except OSError as exc:
                self._emit("sys", "warn", f"the setting could not be saved: {exc}")
                new = self.settings.model_copy(deep=True)
                new.capture.record = on
                self.adopt_settings(new)
        with self._lock:
            self._record_failed = False
            self._state["record_wanted"] = self._recording_wanted()
            running = self._state.get("running", False)
            folder = self._state.get("recording_dir")
        self._emit(
            "sys",
            "info",
            f"recording {'on' if on else 'off'}" + ("" if running else " (from the next Start)"),
        )
        return {"ok": True, "on": on, "running": running, "folder": folder}

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self.muted = muted
            self._state["muted"] = muted
        if muted:
            self._hushed += 1
            self._tts.stop()
        self._emit("voice", "info", "muted" if muted else "unmuted")

    def poll(self, since: int = 0) -> dict[str, Any]:
        """A snapshot for the page: the current state and every log record after `since`;
        and whether the voice is speaking now, for the captions to fade once it stops."""
        if self._hear_test_on and time.monotonic() - self._hear_test_seen > HEAR_TEST_TIMEOUT_S:
            # The Settings window closed during the microphone test without saying so: the
            # test ended only with the next utterance, listening on meanwhile in voice mode
            # whatever the player had chosen (voice finding 10, 2026-10-01).
            self.set_hear_test(False)
        with self._lock:
            state = dict(self._state)
        state["speaking"] = bool(self._tts.busy)
        return {"state": state, "lines": self._log.since(since), "cursor": self._log.cursor}

    def caption(self, text: str, *, continues: bool = False) -> None:
        """Words given to the voice, for the captions' window, numbered: said or not -- muted,
        or held back for pace -- they are captioned whole, the moment they come, as the
        phone's `Captions.show` is (the voice follows when it is made). `continues`: a later
        sentence of a reply written sentence by sentence, which the box runs on from the one
        before; any other line starts a line of its own there."""
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._captioned += 1
            entry = [self._captioned, text, continues]
            lines = [*self._state["caption_lines"], entry][-CAPTIONS_KEPT:]
            self._state["caption_lines"] = lines

    # ---- internals ------------------------------------------------------------------------

    def _build_tts(self, voice: str) -> TTSProvider:
        cfg = self.settings.model_copy(deep=True)
        cfg.voice = cfg.voice.model_copy(update={"voice": voice})
        return make_tts(cfg, off=not self.voice_models)

    def _warm_voice(self) -> None:
        """Load both voice models once, off the pipeline thread, and say where they landed."""
        for label, provider in (("voice", self._tts), ("speech recognition", self._stt)):
            load = getattr(provider, "load", None)
            if load is None:
                self._stage(STAGE_OF[label], "off", provider.healthcheck()[1])
                continue
            self._stage(STAGE_OF[label], "loading")
            ok = load()
            device = getattr(provider, "device", None)
            seconds = getattr(provider, "load_seconds", None)
            detail = provider.healthcheck()[1]
            if ok and device is not None:
                # One inference now, so the first real one is not the slow one.
                prime = getattr(provider, "prime", None)
                primed = prime() if prime is not None else None
                wanted = getattr(provider, "device_wanted", "cuda")
                fell_back = wanted != "cpu" and device.name == "cpu"
                self._emit(
                    "sys",
                    "warn" if fell_back else "info",
                    f"{label}: {detail}; loaded once in {seconds or 0:.1f}s"
                    + (f", primed in {primed:.1f}s" if primed is not None else ""),
                    device=device.name,
                )
                self._stage(STAGE_OF[label], "done", detail)
            else:
                self._emit("sys", "warn", f"{label}: {detail}")
                self._stage(STAGE_OF[label], "off", detail)
        with self._lock:
            self._state["voice_detail"] = self._tts.healthcheck()[1]
            stt_ready, stt_detail = self._stt.healthcheck()
            self._state["stt_detail"] = stt_detail
            self._state["stt_ready"] = stt_ready
        ok, detail = self._mic.healthcheck()
        self._emit("sys", "info" if ok else "warn", f"mic: {detail}")
        self._stage("mic", "loading")
        wanted = self.settings.voice.listen
        if wanted != "off":
            # Speech recognition that is off has no use for the microphone or the push-to-talk
            # key: neither opens (the tests run with it off and must not take the real
            # microphone or hook the keyboard).
            if self._stt.healthcheck()[0]:
                self.set_listen(wanted)
            else:
                self._emit("voice", "warn", "listening: off (speech recognition is not available)")
        self._stage("mic", "done" if ok else "off", detail)

    def _board_on_screen(self) -> Any:
        """The board the screen shows, for correcting what was heard, or None."""
        import chess

        with self._lock:
            fen = self._state.get("fen")
        if not fen:
            return None
        try:
            return chess.Board(fen)
        except ValueError:
            return None

    def _on_partial(self, audio: Any) -> None:
        """The utterance so far, on the listener thread: recognised off it, one at a time
        (a partial still running means this one is skipped, the next will catch up)."""
        if self._hear_test_on or not self._partial_busy.acquire(blocking=False):
            return
        gen = self._utterance_gen

        def work() -> None:
            try:
                raw = self._stt.transcribe(audio)
                text = correct(raw, self._board_on_screen()).text if raw else ""
                # Dropped when its utterance ended meanwhile: the final result has it.
                if text and gen == self._utterance_gen:
                    with self._lock:
                        joined = f"{self._dictation_text} {text}".strip()
                        self._state["dictation"] = {"text": joined, "live": True, "sends_at": None}
            except Exception as exc:  # a partial is best effort; the final result still comes
                self._emit("voice", "debug", f"partial recognition failed: {exc}")
            finally:
                self._partial_busy.release()

        threading.Thread(target=work, name="stt-partial", daemon=True).start()

    def _dictation_arm(self) -> None:
        """(Re)start the second of quiet after which the dictation is sent."""
        with self._lock:
            if self._dictation_timer is not None:
                self._dictation_timer.cancel()
            self._dictation_timer = threading.Timer(DICTATION_SEND_DELAY_S, self._dictation_send)
            self._dictation_timer.daemon = True
            self._dictation_timer.start()
            self._state["dictation"] = {
                "text": self._dictation_text,
                "live": False,
                "sends_at": int((time.time() + DICTATION_SEND_DELAY_S) * 1000),
            }

    def _dictation_settle(self, holds: int | None = None) -> None:
        """An utterance ended (with words or with none): the words (if any) go on to send
        after the second of quiet, else the box stops saying it is listening. `holds`: the
        count of presses and utterances when this one ended its recording; one more came while
        it was being recognised, and the box waits for that one, which settles it in turn: the
        second of quiet ran from the first, and the question went in two halves (voice
        finding 5, 2026-10-01)."""
        if holds is not None and holds != self._dictation_holds:
            with self._lock:
                self._state["dictation"] = {
                    "text": self._dictation_text,
                    "live": True,
                    "sends_at": None,
                }
            return
        if self._dictation_text:
            self._dictation_arm()
        else:
            with self._lock:
                self._state["dictation"] = {"text": "", "live": False, "sends_at": None}

    def _dictation_hold(self) -> None:
        """A new press or a new utterance: whatever was about to send waits for the rest."""
        with self._lock:
            self._dictation_holds += 1
            if self._dictation_timer is not None:
                self._dictation_timer.cancel()
                self._dictation_timer = None
            self._state["dictation"] = {
                "text": self._dictation_text,
                "live": True,
                "sends_at": None,
            }

    def _dictation_send(self) -> None:
        with self._lock:
            text, self._dictation_text = self._dictation_text, ""
            self._dictation_timer = None
            self._state["dictation"] = {"text": "", "live": False, "sends_at": None}
        if text:
            self.command("ask_spoken", text)

    def cancel_dictation(self) -> dict[str, Any]:
        """The person typed: the words stay theirs in the box, nothing sends on its own."""
        with self._lock:
            if self._dictation_timer is not None:
                self._dictation_timer.cancel()
                self._dictation_timer = None
            text, self._dictation_text = self._dictation_text, ""
            self._state["dictation"] = {"text": "", "live": False, "sends_at": None}
        return {"ok": True, "text": text}

    def _on_speech_start(self) -> None:
        """The user started talking: the coach stops mid-sentence (barge-in), and a
        dictation about to send waits for what is coming."""
        if not self._hear_test_on:
            self._dictation_hold()
        # A reply still being written says nothing more, even between two sentences, when
        # the voice is briefly idle: the voice never talks over the player.
        self._hushed += 1
        if self._tts.busy:
            self._tts.stop()
            self._emit("voice", "debug", "barge-in: the coach was cut off")

    def _position_moved_on(self, why: str) -> None:
        """The position the coach is speaking about is left -- a new game, a takeback, the
        board re-read, the player's own next move -- so what the voice is saying about it stops
        (Tai, 2026-10-02: "interrupt the tts so it stops its long rant"). The same mechanism as
        the push-to-talk silencing and the barge-in: the voice stops, and the rest of a reply
        still being written goes to the captions and the record, silent (`_stream_sink`). The
        text is never cut. The opponent's move alone does not stop it."""
        self._hushed += 1
        if self._tts.busy:
            self._tts.stop()
            self._emit("voice", "debug", f"cut off: {why}")

    def _on_dropped(self) -> None:
        """An utterance began and came to nothing (a click, a tap too short to hold a word):
        a dictation it held goes on to send, or the box stops saying it is listening."""
        if not self._hear_test_on:
            self._dictation_settle()

    def _on_utterance(self, audio: Any, seconds: float) -> None:
        """Something was said: transcribe it, correct it against the board, ask the coach.

        Runs on the listener thread; the ask is queued for the pipeline thread.
        """
        self._utterance_gen += 1
        holds = self._dictation_holds  # a press or an utterance after this one began
        begun = time.perf_counter()
        raw = self._stt.transcribe(audio)
        ms = (time.perf_counter() - begun) * 1000
        if self._hear_test_on and time.monotonic() - self._hear_test_seen > HEAR_TEST_TIMEOUT_S:
            self.set_hear_test(False)  # the page went away without saying so
        if self._hear_test_on:
            self._note_heard_test(raw, seconds, ms)
            return
        if not raw:
            self._emit(
                "voice", "debug", f"heard nothing in {seconds:.1f}s of audio", stt_ms=round(ms)
            )
            self._dictation_settle(holds)
            return
        fixed = correct(raw, self._board_on_screen())
        text = fixed.text
        if not text:
            self._dictation_settle(holds)
            return
        # Into the text box, after whatever was said before in this dictation; sent after
        # a second of quiet unless more comes. The terminal keeps the detail.
        with self._lock:
            self._dictation_text = f"{self._dictation_text} {text}".strip()
        self._set(heard=text)
        self._emit(
            "voice",
            "debug",
            f"heard: {text}",
            raw=raw,
            moves=fixed.moves,
            seconds=round(seconds, 2),
            stt_ms=round(ms),
        )
        self._dictation_settle(holds)

    def _emit(self, chan: str, level: str, text: str, **data: Any) -> None:
        self._log.emit(chan, level, text, **data)

    def _set(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    @staticmethod
    def voice_action(
        trigger: str,
        text: str,
        ply: int | None,
        ply_now: int,
        busy: bool,
        playing: tuple[str, int | None] | None = None,
        fast: bool = False,
    ) -> str:
        """What the voice should do with a coach message, so it never talks about a
        position the board has left behind.

        A fast game (or a replay run faster than it was played) can move on while a line is
        still being synthesised or spoken. So: a message whose position is further behind
        the board than its trigger allows (STALE_AFTER: a post-move verdict survives the
        user's move and the bot's instant reply; a hint about the live position does not
        survive the user moving) is shown but not spoken ("stale"); a short post-move
        verdict is dropped if the voice is busy rather than queued behind it ("skip");
        anything substantive cuts off what is still playing when that is about a moment
        the board has left ("supersede"), and waits its turn behind it otherwise
        ("speak": the word on the last move, the review and the practice all come within
        seconds of a game's end, and each cut the one before, so only the last was heard
        -- Tai, 2026-09-20). The answer to the user's own question is never stale and
        never waits: they asked.
        """
        if trigger == "user_ask":
            # The user asked; the answer is owed aloud even if the board has moved on.
            return "supersede" if busy else "speak"
        if fast and trigger != "review":
            return "text"  # a fast game: on the page now, aloud if the player slows down
        allowed = STALE_AFTER.get(trigger, STALE_PLIES - 1)
        # The review sums up a game that is over: the board has nowhere to leave it for. Its
        # message carries the latest position with facts, which after a mate is a move or two
        # behind the board, and must not count as behind.
        if ply is not None and trigger != "review" and ply_now - ply > allowed:
            return "stale"
        if trigger == "post_move" and len(text) < SHORT_VERDICT:
            return "skip" if busy else "speak"
        if not busy:
            return "speak"
        if playing is None:
            return "supersede"
        old_trigger, old_ply = playing
        old_allowed = STALE_AFTER.get(old_trigger, STALE_PLIES - 1)
        old_stale = old_ply is not None and ply_now - old_ply > old_allowed
        return "supersede" if old_stale else "speak"

    def _stream_sink(
        self, trigger: Any, ply: int | None, moved_on: Callable[[], bool] | None = None
    ) -> Callable[[str], bool] | None:
        """The voice's sink for one coach reply written sentence by sentence (see
        coach.Speaker): the first sentence decides, by `voice_action`, whether the reply
        is spoken at all -- stale, skipped, superseding -- and each sentence is queued as
        it comes, so the first is heard while the model writes the rest. A reply that
        has begun is heard to its end unless the position it is about is left: a new game,
        a takeback, a re-read or the player's next move cuts it (`_position_moved_on`); the
        opponent's move alone must not stop the voice mid-sentence (Tai). Called on the
        coach's thread; the voice's queue and the state lock are safe from there.

        No reply is dropped (Tai, 2026-10-02: "no need to drop any replies, the text message
        should come out in the same correct order ... interrupt the tts so it stops its long
        rant"): a reply that is not to be spoken -- stale, skipped, or asked for a moment that
        is over (`moved_on`) -- is taken sentence by sentence all the same, captioned and kept
        as the reply's text, and only silent. Declining it, as before, made the coach say a
        template in its place."""
        if self.muted:
            return None
        name = getattr(trigger, "value", str(trigger))
        started: list[str] = []
        hushed_at = [self._hushed]

        def sink(sentence: str) -> bool:
            continues = bool(started)
            if started and (
                self.muted or self._hushed != hushed_at[0] or started[0] in ("ptt", "silent")
            ):
                # Muted, or cut off (barge-in, hush) mid-reply: the rest of the reply is
                # still the reply -- in the chat, the record and the captions -- only
                # silent. Declining here cut the reply's text at what had been said.
                self.caption(sentence, continues=True)
                return True
            if self.muted:
                return False
            if not started and self._ptt_held():
                # A reply that begins while the player holds the key to talk is never
                # spoken, however long the key is held: captioned, in the chat, silent.
                started.append("ptt")
                self.caption(sentence)
                return True
            with self._lock:
                ply_now = int(self._state.get("ply") or 0)
            if not started:
                hushed_at[0] = self._hushed
                action = (
                    "stale"
                    if moved_on is not None and moved_on()
                    else self.voice_action(
                        name,
                        sentence,
                        ply,
                        ply_now,
                        self._tts.busy,
                        self._voicing,
                        self._coach_fast(),
                    )
                )
                if action == "supersede":
                    self._tts.stop()
                if action == "text":
                    return False  # a fast game: held for the pace (`_speak`), not lost
                if action not in ("speak", "supersede"):
                    # Shown, in its place, not spoken: the rest of the reply follows the same.
                    started.append("silent")
                    self.caption(sentence)
                    self._emit("voice", "debug", f"not spoken ({action}): {elide(sentence, 60)}")
                    return True
                started.append(action)
                self._voicing = (name, ply)
                self._await_audio(name, ply, sentence)
            # Each sentence as the model writes it, the reply's later ones run on from the
            # first; its voice follows.
            self.caption(sentence, continues=continues)
            self._tts.say(sentence)
            self._emit("voice", "debug", f"speaking ({self.voice}): {elide(sentence, 60)}")
            return True

        return sink

    def _speak(self, trigger: str, text: str, ply: int | None, moved_on: bool = False) -> str:
        """Apply `voice_action` and speak accordingly; returns what was done, for the log. The
        captions show every line whatever the voice does with it -- muted, skipped as short,
        stale, held back for pace (Tai, 2026-10-01: "captions show lines regardless of voice
        skips"). While the push-to-talk key is held the voice says nothing: the player is
        talking ("if push to talk is on then the coach must be muted"). A reply asked for a
        moment that is over (`moved_on`: a new game, a takeback, a re-read, the player's next
        move) is shown in its place and not spoken (Tai, 2026-10-02)."""
        if self.muted or self._ptt_held():
            self.caption(text)
            return "muted"
        if moved_on:
            self.caption(text)
            return "stale"
        with self._lock:
            ply_now = int(self._state.get("ply") or 0)
        action = self.voice_action(
            trigger, text, ply, ply_now, self._tts.busy, self._voicing, self._coach_fast()
        )
        if action == "supersede":
            self._tts.stop()
        if action in ("speak", "supersede"):
            self._voicing = (trigger, ply)
            self._await_audio(trigger, ply, text)
            self.caption(text)  # at once; the voice follows when it is made
            self._tts.say(text)
            self._emit("voice", "debug", f"speaking ({self.voice}): {elide(text, 60)}")
        elif action == "text":
            self._held_word = (trigger, text, ply, time.monotonic())
            self.caption(text)
        else:  # "skip", "stale": not said, and shown all the same
            self.caption(text)
        return action

    def _coach_fast(self) -> bool:
        """The game is fast and the player has not slowed down in this position."""
        coach = self._coaching.coach if self._coaching is not None else None
        return bool(coach is not None and coach.fast_now(self._loop_ts))

    def _say_held_word(self, coaching: Any, ts: float) -> None:
        """A word held back for pace is spoken once the player has been thinking in that
        position for PACE_RESUME_S: the game slowed down. A word about a position the
        board has left is dropped. `ts` is the loop's clock (the frame's time)."""
        held = self._held_word
        if held is None or self.muted or self._ptt_held():
            return
        trigger, text, ply, since = held
        coach = coaching.coach
        with self._lock:
            ply_now = int(self._state.get("ply") or 0)
        if ply is not None and ply_now - ply > STALE_AFTER.get(trigger, STALE_PLIES - 1):
            self._held_word = None
            return
        if coach is None or coach.finished:
            self._held_word = None  # the game is over: the review speaks, not a late verdict
            return
        if not coach.policy.user_to_move:
            return
        # Thinking in this position for PACE_RESUME_S -- counted from when the position
        # appeared, not from when the word arrived.
        began = coach.policy.position_since  # on the loop's clock, as `ts` is
        thinking = ts - began if began is not None else time.monotonic() - since
        if thinking < PACE_RESUME_S or self._tts.busy:
            return
        self._held_word = None
        self._voicing = (trigger, ply)
        self._await_audio(trigger, ply, text)
        with self._lock:
            # Captioned when it was held back, long faded: the box comes back while it is
            # said (voice finding 20, 2026-10-01).
            self._state["caption_again"] = int(self._state.get("caption_again") or 0) + 1
        self._tts.say(text)
        self._emit("voice", "debug", f"speaking ({self.voice}), the game slowed: {elide(text, 60)}")

    # ---- latency, amortised: the move to the first sound, the model's first token ----------

    def _hook_voice(self) -> None:
        if hasattr(self._tts, "on_play"):
            self._tts.on_play = self._on_audio_starts  # type: ignore[attr-defined]

    def _note_move(self, ply: int) -> None:
        """A move was committed now (the panel's clock, not the frame's)."""
        with self._lock:
            self._move_at[ply] = time.monotonic()
            # Only the last few matter; a move nobody spoke about is forgotten.
            for old in [p for p in self._move_at if p < ply - 4]:
                del self._move_at[old]

    def _await_audio(self, trigger: str, ply: int | None, line: str) -> None:
        """The first line of a word on a move is about to be queued: remember which move,
        so its first sound can be timed. The message's ply is the position the move was
        played from; the move itself is the next ply."""
        if trigger not in ("post_move", "post_mistake") or ply is None:
            return
        with self._lock:
            moved = self._move_at.pop(ply + 1, None)
            if moved is not None:
                self._awaiting_audio[line] = moved

    def _on_audio_starts(self, line: str) -> None:
        """The voice's thread: a line's audio is playing now."""
        now = time.monotonic()
        with self._lock:
            moved = self._awaiting_audio.pop(line, None)
            if moved is None:
                return
            seconds = now - moved
            self._lat_speech[0] += 1
            self._lat_speech[1] += seconds
            self._lat_speech[2] = seconds

    def _note_first_token(self, seconds: float) -> None:
        with self._lock:
            self._lat_token[0] += 1
            self._lat_token[1] += seconds
            self._lat_token[2] = seconds

    def _latency(self) -> dict[str, Any]:
        with self._lock:
            speech_n, speech_total, speech_last = self._lat_speech
            token_n, token_total, token_last = self._lat_token
        return {
            "move_to_speech_n": int(speech_n),
            "move_to_speech_s": round(speech_total / speech_n, 2) if speech_n else None,
            "move_to_speech_last_s": round(speech_last, 2) if speech_n else None,
            "first_token_n": int(token_n),
            "first_token_s": round(token_total / token_n, 2) if token_n else None,
            "first_token_last_s": round(token_last, 2) if token_n else None,
        }

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._state.get("running"))

    def dump(self) -> str:
        """Write the frames held in memory and the log to data/dumps/<stamp>/: the only way
        a session reaches the disk when nothing is persisted. Returns the folder."""
        folder = self.settings.data_dir / "dumps" / time.strftime("%Y%m%d-%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        n = self.frames.write(folder)
        with (folder / "debug.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self._log.since(0):
                fh.write(json.dumps(rec) + "\n")
        self._emit("sys", "info", f"dumped {n} frames and the log to {folder}")
        return str(folder)

    def _session_dir(self, live: bool) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kind = "recordings" if live and self._recording_wanted() else "sessions"
        return self.settings.data_dir / kind / stamp

    def _run(self) -> None:
        import chess

        from sidecoach.events import Event, Side
        from sidecoach.state.pipeline import GamePipeline
        from sidecoach.vision.clock import ClockReader
        from sidecoach.vision.squares import BoardGeometry

        self._side = Side
        live = self.frame_source is None
        session = self._session_dir(live)
        if self.persist:
            self._log.attach(session / "debug.jsonl")
        self._set(
            running=True, status="starting", session_dir=str(session) if self.persist else None
        )
        self._emit("sys", "info", "engine started" + ("" if self.persist else " (nothing saved)"))

        # Frames: the live screen (gated, and recorded when asked) or the injected source.
        cap: Any = None
        recorder: Any = None
        gate: Any = None
        region: Any = None
        events_log: IO[str] | None = None

        def ensure_recording() -> None:  # a live capture replaces this below
            return None

        # Where the captured monitor sits on the screen, for the board's place there (the
        # captions keep off it); a replay has no screen.
        screen_origin: tuple[int, int] | None = None
        if live:
            from sidecoach.capture.loop import capture_loop
            from sidecoach.capture.wgc_backend import make_capture
            from sidecoach.cli import _monitor_region
            from sidecoach.replay.recording import ChangeGate, Recorder

            cap = make_capture(self.settings.capture.backend)
            monitor, region = _monitor_region(cap, self.monitor)
            screen_origin = (region.x, region.y)
            self._emit(
                "sys",
                "info",
                f"capturing monitor {monitor}, {region.w}x{region.h} at {self.fps} fps "
                f"({getattr(cap, 'name', 'capture')})",
            )
            gate = ChangeGate(keyframe_interval_s=1.0)
            first_folder: Path | None = session  # the launch flags' recording goes to the session

            def _ensure_recording() -> None:
                """Start or stop recording as the setting says, every tick."""
                nonlocal recorder, events_log, first_folder
                want = self._recording_wanted() and not self._record_failed
                if want and recorder is None:
                    folder = first_folder or (
                        self.settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
                    )
                    first_folder = None
                    folder.mkdir(parents=True, exist_ok=True)
                    recorder = Recorder(
                        folder,
                        monitor=monitor,
                        region=region,
                        fps=self.fps,
                        keyframe_interval_s=1.0,
                    )
                    events_log = (folder / "events.jsonl").open("w", encoding="utf-8")
                    self._log.attach(folder / "debug.jsonl")  # with everything logged so far
                    self._set(recording_dir=str(folder))
                    self._emit("sys", "info", f"recording to {folder}")
                elif not want and recorder is not None:
                    stored, mb = recorder.stored, recorder.size_bytes() / 1e6
                    recorder.close()
                    recorder = None
                    if events_log is not None:
                        events_log.close()
                        events_log = None
                    if not self.persist:
                        self._log.detach()
                    with self._lock:
                        folder_name = self._state.get("recording_dir")
                    self._set(recording_dir=None)
                    self._emit(
                        "sys",
                        "info",
                        f"recording stopped: {stored} frames, {mb:.0f} MB in {folder_name}",
                    )

            ensure_recording = _ensure_recording
            ensure_recording()
            if recorder is None:
                self._emit("sys", "info", "not recording")
            # The rate is read every frame, so a saved capture.fps applies at once.
            frames: Iterable[Frame] = capture_loop(
                cap, region, lambda: float(self.settings.capture.fps)
            )
        else:
            frames = self.frame_source or ()
            self._emit("sys", "info", "replaying a recording (not recorded again)")

        cache = None
        try:
            from sidecoach.vision.theme_cache import ThemeCache

            cache = ThemeCache(self.settings.data_dir / "themes")
        except Exception as exc:  # noqa: BLE001 - a cache is a nicety; the panel runs without it
            self._emit("sys", "warn", f"theme cache unavailable: {exc}")
        from sidecoach.vision.ownboard import is_own_board
        from sidecoach.vision.tracker import BoardTracker

        # The vision must never lock onto the panel's own board: it carries a signature
        # (vision.ownboard) the tracker vetoes, on top of the screen-rectangle mask.
        pipe = GamePipeline(
            clock=ClockReader(), cache=cache, tracker=BoardTracker(reject=is_own_board)
        )
        # A side chosen with Flip was for the last session's board, as the new pipeline's is
        # none: kept, it overrode the labels of the next board's Re-read (the parity audit,
        # 2026-09-29; the phone's is per session too).
        self._forced_bottom = None
        # The engine, the coach and the models were built when the app opened (`warm`);
        # a Start that comes before that is done waits for it. Without a warm-up (the
        # CLI's replay, the tests) they are built here, once, and kept.
        if not self._warm_done.is_set() and self._warm_lock.locked():
            self._set(status="loading the models")
            self._warm_done.wait(timeout=120.0)
        with self._warm_lock:
            if self._assessing is None or self._coaching is None or self._engine_stale:
                from sidecoach.vision.pieces import embedder

                net = embedder()
                if net is not None and self._assessing is None:
                    self._emit("sys", "info", f"pieces: embedding net on {net.device.detail}")
            assessing, coaching = self._prepare_stages()
        self.fps = self.settings.capture.fps
        # The player model: a live session teaches the profile on disk, a replay a copy
        # in memory that is forgotten with it.
        from sidecoach.profile.profiler import Profiler
        from sidecoach.profile.store import Store

        store = Store(self.settings.data_dir / "profile.sqlite" if live else None)
        puzzles = None
        try:
            from sidecoach.profile.puzzles import open_store

            puzzles = open_store(self.settings.data_dir)
        except Exception as exc:  # noqa: BLE001 - practice is a nicety; the panel runs without
            self._emit("sys", "warn", f"puzzles unavailable: {exc}")
        profiler = Profiler(store, ts=time.time(), puzzles=puzzles)
        self._profiler = profiler
        self._emit("sys", "info", f"profile: {store.path or 'in memory (replay)'}")
        if puzzles is not None:
            self._emit(
                "sys",
                "info",
                f"puzzles: {puzzles.count()} from {puzzles.path}, level {profiler.puzzle_level()}",
            )
        else:
            self._emit(
                "sys",
                "warn",
                "puzzles: none (get them in Set up, for practice after a game)",
            )
        first_snapshot = profiler.snapshot()
        self._set(profile=self._profile_state(first_snapshot))
        coaching.events([first_snapshot], time.monotonic())
        if coaching.coach is not None:
            # A question about the player over their games: the coach's model writes the
            # queries itself (profile/ask.py), read-only over this store.
            from sidecoach.profile import ask as profile_ask

            # The provider as it is when asked: a model changed on Save answers at once.
            coaching.coach.profile_ask = lambda q: profile_ask.answer(
                q, store, coaching.coach.speaker.provider
            )
        if not self._warm_done.is_set():  # no warm-up loaded the voice (replay, tests)
            threading.Thread(target=self._warm_voice, name="voice-load", daemon=True).start()
        self._set(status="waiting for a board", coach_on=coaching.on, analysis_on=assessing.on)

        latest: Any = None
        latest_geom: Any = None
        ts = time.monotonic()  # the last frame's time; the first, before any frame
        locked = False
        board_screen: list[int] | None = None
        verified = False
        bootstraps = 0
        degraded = False
        opening: str | None = None
        silence: str | None = None
        ticks = 0
        pipeline_ms: list[float] = []
        last_health = time.monotonic()
        try:
            tick_end = time.perf_counter()
            for frame_id, ts, frame in frames:
                if self._stop.is_set():
                    break
                t_grab = time.perf_counter()
                self._loop_ts = ts
                latest = frame
                changed = True
                if live:
                    from sidecoach.replay.recording import board_gate

                    with self._lock:
                        excluded = list(self._exclude)
                    if excluded:
                        self.mask(frame, region, excluded)

                    gate.region = board_gate(pipe.board, region.w, region.h)
                    ensure_recording()
                    if recorder is not None:
                        recorder.change_region = gate.region
                        try:
                            changed = recorder.write(frame, ts).stored is not None
                        except (ValueError, RuntimeError) as exc:
                            # A display change or a full disk must not end the session, and
                            # must not be retried every tick.
                            self._emit("sys", "error", f"recording stopped: {exc}")
                            self._record_failed = True
                            ensure_recording()
                            changed = gate.changed(frame, ts)
                    else:
                        changed = gate.changed(frame, ts)

                if changed:
                    self.frames.push(ts, frame)
                begun = time.perf_counter()
                events: list[Event] = []
                try:
                    events = pipe.process(frame, ts, changed=changed, frame_id=frame_id)
                except Exception as exc:  # noqa: BLE001 - a pipeline bug must not end the panel
                    self._emit("sys", "error", f"pipeline error at frame {frame_id}: {exc!r}")
                t_vision = time.perf_counter()
                pipeline_ms.append((t_vision - begun) * 1000.0)
                ticks += 1
                if pipe.board is not None:
                    latest_geom = BoardGeometry(pipe.board.region, pipe.board.cell_px)
                on_screen = (
                    [
                        screen_origin[0] + pipe.board.region.x,
                        screen_origin[1] + pipe.board.region.y,
                        pipe.board.region.w,
                        pipe.board.region.h,
                    ]
                    if screen_origin is not None and pipe.board is not None
                    else None
                )
                if on_screen != board_screen:
                    board_screen = on_screen
                    self._set(board_screen=board_screen)

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
                followed = pipe.reconciler.game_id if pipe.reconciler is not None else None
                for event in events:
                    self._dispatch_game(event, assessing, chess, followed=followed)

                # Idle read before the drain: then everything the engine will say about
                # this game is in what the drain returns, and a finished game can be
                # reviewed now -- a failed search never sends its verdict, and the review
                # would otherwise wait out the grace period for it.
                engine_idle = assessing.idle
                engine_events = assessing.drain()
                if events_log is not None:
                    for event in engine_events:
                        events_log.write(event.model_dump_json() + "\n")
                opening = self._dispatch_engine(engine_events, opening, chess)
                profile_events = self._profile(
                    profiler, [*events, *engine_events], tick=True, flush=engine_idle
                )

                coaching.events(events, ts)
                coaching.events(engine_events, ts)
                coaching.events(profile_events, ts)
                coaching.tick(ts)
                coaching.drain()
                self._coach_meta(coaching)
                self._say_held_word(coaching, ts)
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
                t_done = time.perf_counter()
                if t_done - tick_end > STALL_WARN_S:
                    self._emit(
                        "sys",
                        "warn",
                        f"tick {frame_id} took {t_done - tick_end:.1f}s: capture "
                        f"{t_grab - tick_end:.1f}s, recorder {begun - t_grab:.1f}s, vision "
                        f"{t_vision - begun:.1f}s, engine/coach {t_done - t_vision:.1f}s",
                    )
                tick_end = t_done
        except Exception as exc:  # noqa: BLE001 - report and shut down cleanly
            self._emit("sys", "error", f"engine stopped: {exc!r}")
        finally:
            # Listening is left as it is: the warm-up opened it and `close()` lets it go, as
            # Start and Stop only start and stop reading the screen.
            # A replay that ran out on its own waits for the engine's backlog, as the
            # terminal replay does, so the review has every verdict; a Stop is prompt.
            patience = 5.0 if self._stop.is_set() else (30.0 if live else 1800.0)
            late = assessing.wait(timeout=patience)
            coaching.events(late, time.monotonic())
            coaching.events(self._profile(profiler, late, flush=True), time.monotonic())
            # The session's end is written before the wait for the coach's replies below (up
            # to 30 s): at a quit the shell's close gives the loop 10 s (`close`), and the
            # session's end, last, was never written when a reply was on its way.
            try:
                summary = profiler.end_session(time.time())
                if summary:
                    self._emit("game", "info", f"session: {summary}")
            except Exception as exc:  # noqa: BLE001 - the profile must not block the shutdown
                self._emit("sys", "error", f"profile error at close: {exc!r}")
            if coaching.coach is not None:
                # Nothing unasked until the next Start reads the board again: the coach
                # outlives the session, and the first ticks after Start gave stuck hints about
                # this one's last position (the game-state review, 2026-10-01).
                coaching.coach.pause(time.monotonic())
            coaching.wait()
            self._coach_meta(coaching)
            # The strip once more, with the replies that came after the last tick.
            with contextlib.suppress(Exception):
                self._health(pipe, coaching, recorder, ticks, pipeline_ms, 1.0, ts, live)
            # Stockfish and the coach stay for the next Start (only the screen stops);
            # the coach begins afresh with the next game's GameStarted.
            assessing.reset()
            self._profiler = None
            store.close()
            if recorder is not None:
                recorder.close()
            if events_log is not None:
                events_log.close()
            if cap is not None:
                cap.close()
            self._record_failed = False  # a fresh Start may record again
            self._set(running=False, status="stopped", recording_dir=None)
            self._emit("sys", "info", "engine stopped")
            self._log.close()

    # ---- log sources ------------------------------------------------------------------------

    @staticmethod
    def _reviewable(profiler: Any, game_id: str) -> str | None:
        """The key Game Review opens a game Sidecoach watched by (the end-of-game card's button),
        when the library holds it as a game: finished, and its moves play out (library)."""
        from sidecoach.profile import library

        key = f"watched:{game_id}"
        try:
            return key if library.find(profiler.store, key) is not None else None
        except Exception:  # noqa: BLE001 - no button is better than a broken card
            return None

    def _profile(
        self, profiler: Any, events: list[Any], *, tick: bool = False, flush: bool = False
    ) -> list[Any]:
        """Feed the profile stage; its own events go on to the coach, and the review and
        the snapshot to the page (the review as a card in the coach's stream, with the
        insights of the moment). `tick` lets a finished game's review time out waiting for
        the engine; `flush` reviews it now."""
        from sidecoach.events import (
            GameReview,
            ProfileEvidence,
            ProfileSnapshot,
            PuzzleOutcome,
            Recommendation,
        )

        out: list[Any] = []
        now = time.time()
        try:
            for event in events:
                out += profiler.on_event(event, now)
            if flush and profiler.game_over_unreviewed:
                out += profiler.flush(now)
            elif tick:
                out += profiler.tick(now)
        except Exception as exc:  # noqa: BLE001 - a profile bug must not end the session
            self._emit("sys", "error", f"profile error: {exc!r}")
        snapshot = next((e for e in out if isinstance(e, ProfileSnapshot)), None)
        for event in out:
            if isinstance(event, GameReview):
                review = event.model_dump(exclude={"ts", "source_frame_id"})
                self._set(review=review)
                insights = list(snapshot.insights) if snapshot is not None else []
                self._emit(
                    "coach",
                    "info",
                    "review: " + elide(" ".join(event.lines), 80),
                    review=review,
                    insights=insights,
                    review_key=self._reviewable(profiler, event.game_id),
                )
            elif isinstance(event, Recommendation):
                practice = event.model_dump(exclude={"ts", "source_frame_id"})
                self.practice_ids = [p.id for p in event.puzzles]
                for p in event.puzzles:
                    self.puzzle_motifs[p.id] = event.motif
                self._set(practice=practice)
                self._emit(
                    "coach",
                    "info",
                    f"practice: {event.motif_words} -- "
                    + ", ".join(p.id for p in event.puzzles)
                    + f" (level {event.level}); {event.reason}",
                    practice=practice,
                )
            elif isinstance(event, PuzzleOutcome):
                self._log_outcome(event)
            elif isinstance(event, ProfileSnapshot):
                self._set(profile=self._profile_state(event))
            elif isinstance(event, ProfileEvidence):
                motif = f" ({event.motif})" if event.motif else ""
                self._emit(
                    "game",
                    "debug",
                    f"evidence at ply {event.ply}: {event.cause}{motif}, {event.expected_loss:.2f}"
                    f" in the {event.phase}",
                )
        return out

    def _check_degraded(self, pipe: Any, degraded: bool) -> bool:
        """Warn once when many cells cannot be read for a while (an overlay, a dialog), and
        say once when the reading is clean again. A single frame's blip -- a piece being
        dragged across the board -- is not worth a line, hence the run of frames required.

        A picked-up rook or queen puts a legal-move dot on a dozen squares, each too small
        to be a piece and read as unknown; that is the site showing where it may go, not a
        degraded reading, so those cells are not counted."""
        from sidecoach.vision.pieces import PIECE_MIN_FRACTION

        readings = pipe.readings
        if readings is None:
            return degraded
        bad = sum(
            1
            for row in readings
            for reading in row
            if reading.covered
            or (
                reading.kind == "unknown"
                and (reading.hover or reading.fg_fraction >= PIECE_MIN_FRACTION)
            )
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

    def _dispatch_game(
        self, event: Any, assessing: Any, chess: Any, *, followed: str | None = None
    ) -> None:
        """One of the game's events to the page and the engine. `followed`: the id of the
        game the reconciler holds once the tick's events are out -- the one End game ends."""
        from sidecoach.events import (
            BoardUnread,
            ClockObservation,
            GameEnded,
            GameStarted,
            MovePlayed,
            Resync,
            ScreenStateChanged,
            Takeback,
            Unexplained,
        )

        if isinstance(event, GameStarted):
            self._position_moved_on("a new game")
            self._fen_before.clear()
            self._labels.clear()
            self._last_ply = 0
            self._promotion_logged = False
            what = (
                "new game"
                if event.kind == "game"
                else "puzzle on the panel's board"
                if event.kind == "puzzle"
                else "position joined (a puzzle, an analysis board, or a game joined late)"
            )
            self._emit(
                "game",
                "info",
                f"{what} {event.game_id}, you play {event.user_side.value}",
                game_id=event.game_id,
                start_fen=event.start_fen,
                kind=event.kind,
            )
            self._set(game_id=event.game_id, user_side=event.user_side.value, kind=event.kind)
            assessing.submit(event)
        elif isinstance(event, MovePlayed):
            before = chess.Board(event.fen_before)
            number = before.fullmove_number
            dots = "." if before.turn == chess.WHITE else "..."
            who = "you" if event.by_user else "opponent"
            think = f", {event.think_ms / 1000:.1f}s" if event.think_ms is not None else ""
            self._fen_before[event.ply] = event.fen_before
            self._last_ply = event.ply
            if event.by_user:
                self._position_moved_on("the player moved")
            self._note_move(event.ply)
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
            self._position_moved_on("a takeback")
            # The engine lets go of the withdrawn moves at once, and their badges leave the
            # move list: a replacement played at the same ply showed the withdrawn move's.
            assessing.submit(event)
            self._last_ply = max(0, self._last_ply - event.plies)
            for gone in [p for p in self._labels if p > self._last_ply]:
                del self._labels[gone]
            for gone in [p for p in self._fen_before if p > self._last_ply]:
                del self._fen_before[gone]
            self._set(labels={str(k): v for k, v in self._labels.items()})
        elif isinstance(event, Resync):
            self._emit("game", "warn", f"resync: {event.reason}", fen=event.fen)
            self._position_moved_on("the board was re-read")
            assessing.submit(event)
            # The numbering starts again from the adopted position: the old plies' badges
            # would sit on the new moves.
            self._labels.clear()
            self._fen_before.clear()
            self._last_ply = 0
            self._set(labels={})
        elif isinstance(event, Unexplained):
            # A note: cells the game has not explained for a while. The game is kept as it
            # stands; nothing downstream starts again.
            self._emit("vision", "warn", f"board unexplained ({event.reason}); the game is kept")
        elif isinstance(event, BoardUnread):
            # A board up and nothing read from it: said in the coach's pane, where the
            # person looks, as "Press Start first" is.
            text = BOARD_UNREAD[event.reason]
            self._emit(
                "coach",
                "info",
                f"board · {event.reason} — {text}",
                full=text,
                trigger="board",
                ply=None,
                voice="text",
            )
            self.caption(text)
        elif isinstance(event, GameEnded):
            text = f"game over {event.result} ({event.reason})"
            if event.result == "*":
                text = f"game over ({event.reason}); the result was not read"
                if event.game_id == followed:
                    # End game ends the game the reconciler holds. A game ended because the
                    # next one began is not that game: told it could set the result, the
                    # player ended the new game, its moves were refused from then on, and
                    # the record kept a one-move win (the state review, 2026-10-01).
                    text += " -- End game… can set it"
            self._emit("game", "info", text, result=event.result, pgn=event.pgn)
        elif isinstance(event, ClockObservation):
            self._set(
                clock={
                    "white_ms": event.white_ms,
                    "black_ms": event.black_ms,
                    "running": event.running.value if event.running else None,
                }
            )
        elif isinstance(event, ScreenStateChanged):
            if event.state.value == "covered":
                self._emit(
                    "game",
                    "info",
                    "the board is covered by a dialog: game over? The game stays open until "
                    "a new one starts or End game… says how it ended; the coach waits",
                )
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
                if event.label is not None:
                    self._labels[event.ply] = event.label.value
                    self._set(labels={str(k): v for k, v in self._labels.items()})
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
                if event.provisional:
                    # Measured early, the full depth to follow: its line is the news.
                    self._emit(
                        "engine", "debug", f"early ply {event.ply}" + threat_text + tail, **data
                    )
                elif signals:
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

    def _new_items(self, name: str, items: list[Any]) -> list[Any]:
        """The entries of the coach's growing list `name` not logged yet: all of them when the
        list is a new coach's."""
        held, done = self._logged.get(name, (None, 0))
        if held is not items:
            done = 0
        self._logged[name] = (items, len(items))
        return items[done:]

    def _coach_meta(self, coaching: Any) -> None:
        """Log each new coach message with why it spoke and what it was checked against, speak
        it, and log any new LLM failure. What is new is kept with the coach, not the session:
        the coach outlives a Stop, and a count started again at each Start sent every earlier
        session's messages through again -- logged, recorded into the game's dialogue, and
        spoken when their ply matched (2026-10-01)."""
        coach = coaching.coach
        if coach is None:
            return
        # Every message of the game's conversation, as the model will see it, at debug: the
        # panel's Conversation tab is built from these.
        for entry in coach.conversation.drain_new():
            if entry.kind == "coach":
                continue  # the coach's turns are logged below, with their calls
            text = entry.text
            if entry.kind == "ply" and entry.record is not None:
                text = coach.conversation.render_ply(entry.record)
            self._emit(
                "coach",
                "debug",
                f"turn ({entry.kind}): {elide(text, 80)}",
                role=entry.role,
                kind=entry.kind,
                body=text[:6000],
            )
        # A prompt that failed its check before going to the model (`harness_problem`).
        for problem in self._new_items("harness", coach.harness_problems):
            self._emit("coach", "error", f"prompt check failed: {problem}")
        for kind, answer in self._new_items("profile", coach.profile_answers):
            # What the coach looked up in the record for a question about the player.
            if kind == "error":
                self._emit("coach", "error", f"record lookup failed: {answer}")
                continue
            for i, exchange in enumerate(answer.exchanges, start=1):
                head = exchange.error or f"{len(exchange.rows)} rows"
                self._emit(
                    "coach",
                    "warn" if exchange.error else "info",
                    f"record query {i}: {elide(' '.join(exchange.sql.split()), 90)} — {head}",
                    sql=exchange.sql,
                    columns=exchange.columns,
                    rows=[[str(v) for v in row] for row in exchange.rows[:20]],
                    error=exchange.error,
                )
            if answer.failed:
                self._emit("coach", "warn", "record: the model wrote no answer; the rows stand in")
        for spoken in self._new_items("spoken", coach.spoken):
            message = spoken.message
            if message.text:
                self._record_dialogue(message)
            meta = [
                message.trigger.value,
                f"L{int(spoken.level)}",
                f"ply {message.ply}" if message.ply is not None else "no ply",
            ]
            if spoken.attempts:
                meta.append(f"{spoken.attempts} attempt{'s' if spoken.attempts != 1 else ''}")
            if spoken.leaks:
                # For debugging: what the reply gave away of the answer, spoken anyway.
                meta.append("leak noted: " + "; ".join(spoken.leaks))
            if spoken.notes:
                # The same for its form: what it did against the form asked of it.
                meta.append("form noted: " + "; ".join(spoken.notes))
            if not message.text:
                # The model's reply failed the checks and the quick answer already stands.
                self._emit(
                    "coach",
                    "warn",
                    " · ".join(meta) + " — model reply rejected; the quick answer stands",
                    trigger=message.trigger.value,
                    disclosure=int(spoken.level),
                    ply=message.ply,
                    attempts=spoken.attempts,
                    problems=spoken.problems,
                )
            else:
                if spoken.problems and not spoken.streamed:
                    meta.append("draft rejected → template")
                elif spoken.problems:
                    meta.append("cut short: " + "; ".join(spoken.problems))
                if spoken.streamed:
                    action = "streamed"
                else:
                    action = self._speak(
                        message.trigger.value, message.text, message.ply, moved_on=spoken.late
                    )
                if action == "streamed":
                    meta.append("streamed")
                elif action == "stale":
                    meta.append("not spoken: board moved on")
                if spoken.late:
                    meta.append("late: asked for a moment that is over")  # shown in its place
                elif action == "skip":
                    meta.append("not spoken: voice busy")
                elif action == "text":
                    meta.append("text: fast game")
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
                    leaks=spoken.leaks,
                    notes=spoken.notes,
                    fallback=bool(spoken.problems),
                    voice=action,
                    fen=message.fen,
                    annotations=[mark.model_dump() for mark in message.annotations],
                )
            for index, call in enumerate(spoken.calls, start=1):
                # What the model actually wrote, so a stump or a leak can be traced to its
                # source; a reply the token budget cut short is worth a warning.
                finish = call.stats.get("finish_reason") or "?"
                tokens = call.stats.get("completion_tokens")
                seconds = call.stats.get("elapsed_s")
                first = call.stats.get("first_token_s")
                if first is not None:
                    self._note_first_token(float(first))
                stream_problems = call.stats.get("stream_problems")
                summary = (
                    f"model reply {index}: {call.candidates} candidates, {call.passed} passed"
                    + (f", {tokens} tokens" if tokens is not None else "")
                    + (f" in {seconds:.1f}s" if isinstance(seconds, (int, float)) else "")
                    + (f" (first token {first:.1f}s)" if isinstance(first, (int, float)) else "")
                    + (", cut off by the token budget" if finish == "length" else "")
                    + (f"; stream stopped: {'; '.join(stream_problems)}" if stream_problems else "")
                )
                self._emit(
                    "coach",
                    "warn" if finish == "length" else "debug",
                    summary,
                    raw=elide(call.raw, 2000),
                    prompt=call.prompt[-6000:],
                    finish_reason=finish,
                    completion_tokens=tokens,
                    elapsed_s=seconds,
                    candidates=call.candidates,
                    passed=call.passed,
                )
        speaker = coach.speaker
        held, failures = self._failures_seen
        if held is not speaker:
            failures = 0
        if speaker.failures > failures:
            self._emit("coach", "error", f"LLM call failed: {speaker.last_error}")
        self._failures_seen = (speaker, speaker.failures)

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
        rss_mb = None
        try:
            import psutil

            rss_mb = round(psutil.Process().memory_info().rss / 1e6)
        except Exception:  # noqa: BLE001 - psutil is a panel extra; no number is fine
            rss_mb = None
        tts_dev = getattr(getattr(self._tts, "device", None), "name", None)
        stt_dev = getattr(getattr(self._stt, "device", None), "name", None)
        # The session's model calls: tokens, and dollars when the prices are known.
        # Every ledger: the models replaced mid-session and a verifier on its own endpoint
        # included, each priced as its own model.
        llm = None
        speaker = getattr(coaching.coach, "speaker", None)
        provider = getattr(speaker, "provider", None)
        ledgers: list[Any] = list(getattr(speaker, "usages", list)())
        for source in self.extra_usage:
            with contextlib.suppress(Exception):
                ledgers.extend(source())
        if ledgers:
            from sidecoach.llm.usage import total

            llm = total(ledgers, self.settings, getattr(provider, "model", None))
        self._set(
            health={
                "llm": llm,
                "latency": self._latency(),
                "rss_mb": rss_mb,
                "tts_device": tts_dev,
                "stt_device": stt_dev,
                "mic_dropped": self._mic.dropped,
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

    def _record_dialogue(self, event: Any) -> None:
        profiler = self._profiler
        if profiler is None:
            return
        try:
            profiler.on_event(event, time.time())
        except Exception as exc:  # noqa: BLE001
            self._emit("sys", "error", f"profile error: {exc!r}")

    def _drain_commands(
        self, pipe: Any, coaching: Any, frame: Any, geom: Any, ts: float
    ) -> list[Any]:
        from sidecoach.events import UserUtterance

        events: list[Any] = []
        while True:
            try:
                name, payload = self._commands.get_nowait()
            except queue.Empty:
                break
            if name in ("ask", "ask_spoken"):
                if coaching.on:
                    # Spoken or typed, the ask is a turn of the conversation on the page.
                    self._emit("user", "info", f"you: {payload}", spoken=name == "ask_spoken")
                    coaching.ask(str(payload), ts)
                    self._record_dialogue(
                        UserUtterance(
                            text=str(payload), source="voice" if name == "ask_spoken" else "text"
                        )
                    )
                else:
                    self._emit("user", "warn", f"no coach to ask: {payload}")
            elif name == "end":
                events += self._end(pipe, str(payload), ts)
            elif name == "flip":
                events += self._flip(pipe, frame, geom, ts)
            elif name == "analyze":
                events += self._analyze(pipe, frame, geom, ts)
        return events

    def _log_outcome(self, event: Any) -> None:
        self._emit(
            "game",
            "info",
            f"puzzle {event.puzzle_id} {'solved' if event.solved else 'failed'}"
            + (f" ({event.motif})" if event.motif else "")
            + f"; puzzle level {event.level}",
            puzzle_id=event.puzzle_id,
            solved=event.solved,
            puzzle_level=event.level,
        )

    def _end(self, pipe: Any, outcome: str, ts: float) -> list[Any]:
        if pipe.reconciler is None or not pipe.reconciler.started:
            self._emit("user", "warn", "no game to end")
            return []
        with self._lock:
            user_side = self._state.get("user_side")
        result, reason = self.outcome_result(outcome, user_side)
        events: list[Any] = []
        events = pipe.end_game(result, reason, ts)
        if events:
            self._emit("user", "info", f"you ended the game: {reason} ({result})")
        if outcome == "forget":
            pipe.reset()
            self._fen_before.clear()
            self._set(game_id=None, user_side=None, following=False)
            self._emit("game", "info", "forgot the game; waiting for a start position")
        return events

    def _flip(self, pipe: Any, frame: Any, geom: Any, ts: float) -> list[Any]:
        """Flip means three things.

        A game seen from its start position has its orientation from the pieces
        themselves, so flipping only turns the panel's board around (a view preference);
        re-reading an unchanged screen the wrong way up would resync to a rotated, bogus
        position. A game joined from the middle -- a puzzle, an analysis board, a resync
        -- had its orientation from the labels or from the game before, which can be
        wrong (a puzzle for Black read as White's, 2026-09-20): flipping re-reads the
        screen with the other side at the bottom and begins again from that. With no game
        running, the board on the screen is read that way; with none to read, flipping
        chooses the side the next cold read assumes when the labels say nothing.
        """
        import chess

        Side = self._side
        rec = pipe.reconciler
        if rec is not None and rec.started and rec.base_fen == chess.STARTING_FEN:
            self._view_flip = not self._view_flip
            self._set(view_flip=self._view_flip)
            self._emit(
                "user", "info", "view flipped (the game is still read from the screen as is)"
            )
            return []
        # The side the board is read with, when a game is: the labels may have turned it
        # since the last Flip.
        running = pipe.bottom if rec is not None and rec.started else None
        current = running or self._forced_bottom or pipe.bottom or Side.WHITE
        target = current.other
        self._forced_bottom = target
        pipe.forced_bottom = target
        self._set(bottom=target.value)
        # The board on the screen is read that way now, whether a game was running or it
        # was the one left unread for its side -- and its labels learn the side from it.
        events = (
            pipe.reorient(frame, geom, target, ts, told=True)
            if frame is not None and geom is not None
            else []
        )
        if events:
            self._emit("user", "info", f"board re-read with {target.value} at the bottom")
            return events
        if rec is not None and rec.started:
            # Mid-move, or unreadable now: forget the game; the next quiet frame is read
            # cold with this side at the bottom.
            events = self._end(pipe, "forget", ts)
            self._emit("user", "info", f"re-reading the board with {target.value} at the bottom")
            return events
        self._emit("user", "info", f"analysis orientation: {target.value} at the bottom")
        return []

    def _analyze(self, pipe: Any, frame: Any, geom: Any, ts: float) -> list[Any]:
        if frame is None or geom is None:
            self._emit("vision", "warn", "no board on screen to analyse yet")
            return []
        try:
            events = pipe.cold_start(frame, geom, bottom=self._forced_bottom, ts=ts)
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
        sans, fens = self._sans(rec) if rec is not None else ([], [])
        # A joined position that has run long enough is a game of the record (the
        # profile's call, profile.profiler.PROVISIONAL_PLIES): the page says so once.
        profiler = self._profiler
        a_game = rec is not None and (
            rec.kind == "game" or (profiler is not None and profiler.following_a_game)
        )
        if a_game and rec is not None and rec.kind != "game" and not self._promotion_logged:
            self._promotion_logged = True
            self._emit(
                "game",
                "info",
                f"{len(moves)} plies on the joined position: a game joined late, recorded "
                "from the join and reviewed at the end",
            )
        self._set(
            fen=fen,
            moves=sans,
            fens=fens,  # the position after each ply, for browsing the move list
            ucis=moves,  # each ply's move, for the from/to highlight while browsing
            last_move=moves[-1] if moves else None,  # uci, for the from/to highlight
            ply=len(moves),
            following=pipe.model is not None,
            bottom=pipe.bottom.value if pipe.bottom else None,
            status=(
                "waiting for a start"
                if pipe.model is None
                else f"game over {rec.result or ''} · waiting for the next game".replace("  ", " ")
                if rec is not None and rec.finished
                else "following a game"
                if a_game
                else "following a position"
            ),
        )

    @staticmethod
    def _sans(rec: Any) -> tuple[list[str], list[str]]:
        """The game's moves as SAN, and the FEN after each, from the reconciler's base."""
        import chess

        board = chess.Board(rec._base_fen) if getattr(rec, "_base_fen", None) else chess.Board()
        out: list[str] = []
        fens: list[str] = []
        for move in rec.board.move_stack:
            try:
                out.append(board.san(move))
            except Exception:  # noqa: BLE001 - a resync base can make SAN ambiguous; show uci
                out.append(move.uci())
            board.push(move)
            fens.append(board.fen())
        return out, fens


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

    def end(self, outcome: str) -> bool:
        self._engine.end_game(outcome)
        return True

    def correct(self, game_id: str, segment: int, ply: int, cause: str | None) -> bool:
        return self._engine.correct(game_id, segment, ply, cause)

    def set_muted(self, muted: bool) -> bool:
        self._engine.set_muted(bool(muted))
        return True

    def reset_profile(self) -> dict[str, Any]:
        return {"ok": True, "path": self._engine.reset_profile()}

    def clear_caches(self) -> dict[str, Any]:
        return {"ok": True, "done": self._engine.clear_caches()}
