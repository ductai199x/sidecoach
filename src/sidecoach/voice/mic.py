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
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=QUEUE_FRAMES)
        self.dropped = 0
        self._stream: Any = None
        self.detail = "not open"

    def healthcheck(self) -> tuple[bool, str]:
        if find_spec("sounddevice") is None:
            return False, "sounddevice not installed; run `uv sync --group voice`"
        try:
            import sounddevice as sd  # type: ignore[reportMissingImports]

            index = self.device if self.device is not None else sd.default.device[0]
            if index is None or (isinstance(index, int) and index < 0):
                return False, "no default input device"
            info = sd.query_devices(index)
            if info.get("max_input_channels", 0) < 1:
                return False, f"device {index} has no input channels"
            return True, f"input device {index}: {info.get('name', '?')}"
        except Exception as exc:  # noqa: BLE001
            return False, f"no input device: {exc}"

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
                device=self.device,
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
    ) -> None:
        self.mic = mic
        self.gate = gate
        self.endpointer = Endpointer()
        self._on_utterance = on_utterance
        self._on_speech_start = on_speech_start or (lambda: None)
        self._coach_speaking = coach_speaking or (lambda: False)
        self._on_event = on_event or (lambda level, text: None)
        self.mode = "off"
        self._ptt = False
        self._buffer: list[np.ndarray] = []
        self._pre: deque[np.ndarray] = deque(maxlen=PRE_ROLL_FRAMES)
        self._started_at: float | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ---- control ------------------------------------------------------------------------

    def set_mode(self, mode: str) -> str:
        """Switch modes; opens the mic on demand and closes it when off. Returns a detail."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self._finish(discard=True)
            self.mode = mode
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
                continue
            with self._lock:
                mode = self.mode
                if mode == "ptt":
                    if self._ptt:
                        self._buffer.append(frame)
                        self._cap()
                    else:
                        self._pre.append(frame)
                    continue
                if mode != "vad":
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

    def _cap(self) -> bool:
        """True (and stop growing) once an utterance has reached its maximum length."""
        return len(self._buffer) * HOP >= MAX_UTTERANCE_S * SAMPLE_RATE

    def _finish(self, *, discard: bool) -> None:
        """Hand the buffered frames over as one utterance (or drop them) and reset."""
        buffer, self._buffer = self._buffer, []
        started = self._started_at
        self._started_at = None
        self._pre.clear()
        self.gate.reset()
        if discard or not buffer:
            return
        audio = np.concatenate(buffer)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < 0.25:
            return
        try:
            self._on_utterance(audio, time.monotonic() - started if started else seconds)
        except Exception as exc:  # noqa: BLE001 - a bad utterance must not stop listening
            log.warning("utterance handler failed: %s", exc)
