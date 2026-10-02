"""The microphone and the listener: turning what the user says into utterances to transcribe.

`Microphone` reads 16 kHz mono frames from the default (or a chosen) input device through
sounddevice into a bounded queue -- if nobody drains it, the oldest frames are dropped, so it
can never grow. `Listener` consumes those frames on its own thread in one of two modes:

  push-to-talk  frames are kept between `press()` and `release()`, then handed over as one
                utterance -- no VAD involved, no false triggers;
  voice         Silero VAD decides: an utterance starts after two confident frames (and the
                coach is cut off -- barge-in), keeps a short pre-roll so the first syllable
                is not clipped, and ends after 400 ms of silence. Utterances are capped at
                MAX_UTTERANCE_S so a noisy room cannot buffer forever.

Everything degrades: no sounddevice, no input device, or no VAD model leaves the coach deaf
with a stated reason, never crashed. `on_utterance` runs on the listener thread; the caller
does its STT there and hands the text to its own loop.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any

import numpy as np

from sidecoach.voice.devices import follows_system_default, system_default
from sidecoach.voice.vad import HOP, SAMPLE_RATE, Endpointer, SileroGate

log = logging.getLogger(__name__)

MODES = ("off", "ptt", "vad")
PRE_ROLL_FRAMES = 10  # 320 ms kept before a detected start, so "knight" is not "ight"
MAX_UTTERANCE_S = 20.0
QUEUE_FRAMES = SAMPLE_RATE // HOP * 10  # ten seconds of backlog before the oldest is dropped


class Microphone:
    """16 kHz mono frames from an input device, into a bounded queue."""

    def __init__(self, device: int | str | None = None) -> None:
        self.device = device
        self._capture: int | None = None  # resolved once, when unconfigured
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_FRAMES)
        self.dropped = 0
        self._stream: Any = None
        self.detail = "not open"

    def healthcheck(self) -> tuple[bool, str]:
        if find_spec("sounddevice") is None:
            return False, "sounddevice not installed; run `uv sync --group voice`"
        try:
            import sounddevice as sd  # type: ignore[reportMissingImports]

            device = self.capture_device()
            index = device if device is not None else sd.default.device[0]
            if index is None or (isinstance(index, int) and index < 0):
                return False, "no default input device"
            info = sd.query_devices(index)
            if info.get("max_input_channels", 0) < 1:
                return False, f"device {index} has no input channels"
            if follows_system_default(device, "input"):
                now = sd.query_devices(sd.default.device[0]).get("name", "?")
                return True, f"the system default input ({now} now; follows a change when reopened)"
            return True, f"input device {index}: {info.get('name', '?')}"
        except Exception as exc:  # noqa: BLE001
            return False, f"no input device: {exc}"

    def capture_device(self) -> int | str | None:
        """The device frames come from: the configured one, else the one that follows the
        system default (Windows' Sound Mapper), else PortAudio's own default."""
        if self.device is not None:
            return self.device
        if self._capture is None:
            self._capture = system_default("input")
        return self._capture

    def open(self) -> bool:
        if self._stream is not None:
            return True
        ok, detail = self.healthcheck()
        self.detail = detail
        if not ok:
            return False
        try:
            import sounddevice as sd  # type: ignore[reportMissingImports]

            def callback(indata: np.ndarray, frames: int, _time: Any, status: Any) -> None:
                frame = np.array(indata[:, 0], dtype=np.float32, copy=True)
                try:
                    self.frames.put_nowait(frame)
                except queue.Full:
                    # Nobody is listening fast enough: drop the oldest, never grow.
                    self.dropped += 1
                    with contextlib.suppress(queue.Empty):
                        self.frames.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self.frames.put_nowait(frame)

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=HOP,
                device=self.capture_device(),
                callback=callback,
            )
            self._stream.start()
            return True
        except Exception as exc:  # noqa: BLE001
            self.detail = f"could not open input: {exc}"
            self._stream = None
            return False

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break


class Listener:
    """Frames in, utterances out, in push-to-talk or voice mode."""

    def __init__(
        self,
        mic: Microphone,
        gate: SileroGate,
        *,
        on_utterance: Callable[[np.ndarray, float], None],
        on_speech_start: Callable[[], None] | None = None,
        coach_speaking: Callable[[], bool] | None = None,
        on_event: Callable[[str, str], None] | None = None,
        on_partial: Callable[[np.ndarray], None] | None = None,
        partial_every_s: float = 0.4,
        on_dropped: Callable[[], None] | None = None,
    ) -> None:
        self.mic = mic
        self.gate = gate
        self.endpointer = Endpointer()
        self._on_utterance = on_utterance
        # An utterance that began (`on_speech_start` was called, or the key went down) and
        # is not handed over -- a click the endpointer dropped, a tap too short to hold a
        # word, a mode switch mid-sentence -- is said too, so whoever was told it began is
        # told it came to nothing (2026-09-23: a cough held a dictation's send for good).
        self._on_dropped = on_dropped or (lambda: None)
        # While an utterance is being captured, the audio so far every `partial_every_s`,
        # for words on the screen as they are said (Tai, 2026-09-22: streamed, not at the
        # end). Called on the listener thread; the handler must not block it.
        self._on_partial = on_partial
        self._partial_every_s = partial_every_s
        self._last_partial = 0.0
        self._on_speech_start = on_speech_start or (lambda: None)
        self._coach_speaking = coach_speaking or (lambda: False)
        self._on_event = on_event or (lambda level, text: None)
        self.mode = "off"
        self._ptt = False
        # The microphone's level, for a meter: the latest frame's peak, decaying.
        self.level = 0.0
        self._buffer: list[np.ndarray] = []
        self._pre: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Utterances are handed over on a thread of their own: `release` runs on the
        # keyboard hook's thread, and a recognition there (hundreds of milliseconds)
        # held the listener's lock and, on Windows, overran the hook's timeout -- after
        # which the system removes the hook and push-to-talk silently dies (2026-09-22).
        self._deliveries: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._deliverer: threading.Thread | None = None

    # ---- control ------------------------------------------------------------------------

    def set_mode(self, mode: str) -> str:
        """Switch modes; opens the mic on demand and closes it when off. Returns a detail."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self._finish(discard=True)
            self.mode = mode
            # A key held across the switch: the next press starts afresh (it was ignored).
            self._ptt = False
        if mode == "off":
            self._stop.set()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
            self._thread = None
            self.mic.close()
            return "off"
        if not self.mic.open():
            self.mode = "off"
            return f"off ({self.mic.detail})"
        if mode == "vad" and not self.gate.load():
            self.mode = "ptt"
            why = self.gate.healthcheck()[1]
            self._on_event("warn", f"voice mode unavailable ({why}); push-to-talk instead")
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="listener", daemon=True)
            self._thread.start()
        return f"{self.mode} ({self.mic.detail})"

    def press(self) -> None:
        """Push-to-talk down: start keeping frames."""
        with self._lock:
            if self.mode != "ptt" or self._ptt:
                return
            self._ptt = True
            self._buffer = list(self._pre)
            self._started_at = time.monotonic()
        self._on_speech_start()

    def release(self) -> None:
        """Push-to-talk up: what was kept is one utterance."""
        with self._lock:
            if not self._ptt:
                return
            self._ptt = False
            self._finish(discard=False)

    # ---- the loop -----------------------------------------------------------------------

    def _run(self) -> None:
        self.gate.reset()
        self.endpointer.reset()
        while not self._stop.is_set():
            try:
                frame = self.mic.frames.get(timeout=0.2)
            except queue.Empty:
                self.level *= 0.5
                continue
            self.level = max(float(np.abs(frame).max()), self.level * 0.8)
            partial: np.ndarray | None = None
            with self._lock:
                mode = self.mode
                if mode == "ptt":
                    if self._ptt:
                        self._buffer.append(frame)
                        if self._cap():
                            # A key held past the longest utterance: what was said goes to
                            # the recogniser, and holding on keeps nothing more until the
                            # next press. The buffer grew without end, and each partial
                            # recognised it all (voice finding 7, 2026-10-01).
                            self._ptt = False
                            self._finish(discard=False)
                            continue
                        partial = self._partial_due()
                    else:
                        self._pre.append(frame)
            if mode != "vad":
                if partial is not None:
                    self._on_partial(partial)  # type: ignore[misc]
                continue
            # Voice mode: the VAD runs outside the lock; it is the slow part.
            prob = self.gate.probability(frame)
            with self._lock:
                event = self.endpointer.feed(prob, coach_speaking=self._coach_speaking())
                if self.endpointer.speaking:
                    if event == "start":
                        self._buffer = list(self._pre)
                        self._started_at = time.monotonic()
                    self._buffer.append(frame)
                    if self._cap():
                        self.endpointer.reset()
                        self._finish(discard=False)
                        continue
                    partial = self._partial_due()
                else:
                    self._pre.append(frame)
                if event == "start":
                    self._on_event("debug", "speech started")
                    self._on_speech_start()
                elif event == "end":
                    self._finish(discard=False)
                elif event == "drop":
                    self._finish(discard=True)
                    self._on_event("debug", "a sound too short to be an ask, ignored")
            if partial is not None:
                self._on_partial(partial)  # type: ignore[misc]

    def _cap(self) -> bool:
        """True (and stop growing) once an utterance has reached its maximum length."""
        return len(self._buffer) * HOP >= MAX_UTTERANCE_S * SAMPLE_RATE

    def _partial_due(self) -> np.ndarray | None:
        """The audio so far, when a partial is wanted and it is time for one (under the lock)."""
        if self._on_partial is None or len(self._buffer) * HOP < 0.6 * SAMPLE_RATE:
            return None
        now = time.monotonic()
        if now - self._last_partial < self._partial_every_s:
            return None
        self._last_partial = now
        return np.concatenate(self._buffer)

    def _finish(self, *, discard: bool) -> None:
        """Hand the buffered frames over as one utterance (or drop them) and reset; an
        utterance that had begun and is dropped is said (`on_dropped`)."""
        buffer, self._buffer = self._buffer, []
        started = self._started_at
        self._started_at = None
        self._pre.clear()
        self.gate.reset()
        audio = np.concatenate(buffer) if buffer and not discard else None
        seconds = len(audio) / SAMPLE_RATE if audio is not None else 0.0
        if audio is None or seconds < 0.25:
            if started is not None:
                self._on_dropped()
            return
        self._deliveries.put((audio, time.monotonic() - started if started else seconds))
        if self._deliverer is None or not self._deliverer.is_alive():
            self._deliverer = threading.Thread(target=self._deliver, name="utterances", daemon=True)
            self._deliverer.start()

    def _deliver(self) -> None:
        """Utterances to the handler, one after the other, off the hook's thread."""
        while True:
            item = self._deliveries.get()
            if item is None:
                return
            audio, seconds = item
            try:
                self._on_utterance(audio, seconds)
            except Exception as exc:  # noqa: BLE001 - a bad utterance must not stop listening
                log.warning("utterance handler failed: %s", exc)
