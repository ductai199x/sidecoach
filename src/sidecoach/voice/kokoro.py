"""Kokoro-82M text-to-speech, via kokoro-onnx on the shared ONNX runtime, CPU or GPU.

Kokoro is small, natural, Apache-2.0, real-time on the CPU and ~24x real-time on the GPU. The
graph is loaded ONCE (`load`, or lazily on the first line) into a session that lives as long
as the provider; a voice is only a style vector chosen at synthesis time, so changing the
voice never reloads anything. The device comes from `sidecoach.voice.runtime`.

Everything heavy -- the model, the audio device -- is imported lazily and loaded off the
calling thread, so importing this module is cheap and a machine without the voice extras or
without an audio device degrades to silence rather than failing. `say` queues a line and
returns at once; a single worker thread synthesises and plays in order.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
import time
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

from sidecoach.voice.devices import follows_system_default, system_default
from sidecoach.voice.notation import speak_notation
from sidecoach.voice.runtime import Device
from sidecoach.voice.runtime import session as make_session

log = logging.getLogger(__name__)

ONNX_FILE = "kokoro-v1.0.onnx"
# Silence appended to every line: a breath between one sentence and the next, and, on
# the fallback path (`play` then `wait`, the stream stopped the moment the last sample
# is handed over), what a Bluetooth headset a few hundred milliseconds behind clips
# instead of the words (Tai, 2026-09-19).
TAIL_SILENCE_S = 0.35
# The output stream stays open between lines, silence flowing: opened per line, a
# Bluetooth headset swallowed the first syllables while its link woke and the last while
# the stream closed under it (Tai, 2026-09-20: "beginnings or ends of sentences cut off").
# The stream is reopened when the system's default output changes, checked this often.
DEVICE_CHECK_S = 2.0
VOICES_FILE = "voices-v1.0.bin"


class _Stream:
    """One open output stream, kept running. Lines are queued as sample blocks and
    played by the device's callback; between lines the callback writes silence, so the
    device never sleeps and never closes on a tail."""

    def __init__(self, sd: object, samplerate: int, device: int | str | None) -> None:
        import numpy as np

        self._np = np
        self.samplerate = samplerate
        self.device = device
        self._chunks: list = []
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._stream = sd.OutputStream(  # type: ignore[attr-defined]
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata: object, frames: int, time_info: object, status: object) -> None:
        out = outdata  # (frames, 1) float32
        filled = 0
        with self._lock:
            while filled < frames and self._chunks:
                chunk = self._chunks[0]
                take = min(frames - filled, len(chunk))
                out[filled : filled + take, 0] = chunk[:take]  # type: ignore[index]
                if take == len(chunk):
                    self._chunks.pop(0)
                else:
                    self._chunks[0] = chunk[take:]
                filled += take
            if not self._chunks:
                self._idle.set()
        if filled < frames:
            out[filled:, 0] = 0  # type: ignore[index]

    def play(self, samples: object, timeout: float) -> None:
        """Queue `samples` and return once they have been handed to the device."""
        block = self._np.asarray(samples, dtype=self._np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(block)
            self._idle.clear()
        self._idle.wait(timeout)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._idle.set()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()


class KokoroTTS:
    """Speak coach lines with Kokoro. Non-blocking; never raises out of `say`/`stop`."""

    name = "kokoro"

    def __init__(
        self,
        model_dir: Path,
        *,
        voice: str = "af_sarah",
        speed: float = 1.0,
        lang: str = "en-us",
        device: str = "auto",
        output_device: int | str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.device_wanted = device
        self.output_device = output_device  # a sounddevice output index or name
        self._playback: int | None = None  # resolved once, when unconfigured
        self.device: Device | None = None  # where the graph actually loaded
        self.load_seconds: float | None = None
        self._queue: queue.Queue[tuple[str, str, str | None] | None] = queue.Queue()
        # Called with the line as given to `say` the moment its audio starts to play,
        # for whoever measures the way from a move to the first sound (the panel).
        self.on_play: Callable[[str], None] | None = None
        self._worker: threading.Thread | None = None
        self._kokoro: object | None = None
        self._sd: object | None = None
        self._failed = False
        self._load_lock = threading.Lock()
        # Bumped on every stop so a synthesis in flight knows its line was cancelled and
        # does not play over what comes next.
        self._generation = 0
        self._lock = threading.Lock()
        self._playing = False
        # The open output stream (see _Stream), the device name it was opened on and when
        # that was last checked against the system default; None when the stream could
        # not be opened, and lines then play one by one.
        self._stream: _Stream | None = None
        self._stream_device_name: str | None = None
        self._stream_checked = 0.0

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / ONNX_FILE

    @property
    def voices_path(self) -> Path:
        return self.model_dir / VOICES_FILE

    @property
    def loaded(self) -> bool:
        return self._kokoro is not None

    def healthcheck(self) -> tuple[bool, str]:
        """Whether the voice can speak, checked without loading the model.

        The extras must be importable and the two model files present; the model itself is
        big and is loaded once, by `load` or on the first line.
        """
        if find_spec("kokoro_onnx") is None:
            return False, "kokoro-onnx not installed; run `uv sync --group voice`"
        if find_spec("sounddevice") is None:
            return False, "sounddevice not installed; run `uv sync --group voice`"
        missing = [p.name for p in (self.onnx_path, self.voices_path) if not p.exists()]
        if missing:
            return False, f"model files missing from {self.model_dir}: {', '.join(missing)}"
        where = self.device.detail if self.device else f"{self.device_wanted} (not loaded yet)"
        return True, f"kokoro-onnx, voice {self.voice}, {where}, speaker: {self.speaker_name()}"

    def playback_device(self) -> int | str | None:
        """The device lines play through: the configured one, else the one that follows the
        system default (Windows' Sound Mapper), else PortAudio's own default."""
        if self.output_device is not None:
            return self.output_device
        if self._playback is None:
            self._playback = system_default("output")
        return self._playback

    def speaker_name(self) -> str:
        """The output device the voice plays through, by name."""
        try:
            import sounddevice as sd  # type: ignore[reportMissingImports]

            device = self.playback_device()
            if follows_system_default(device, "output"):
                now = sd.query_devices(sd.default.device[1]).get("name", "?")
                return f"the system default output ({now} now; follows a change)"
            index = device if device is not None else sd.default.device[1]
            return str(sd.query_devices(index).get("name", index))
        except Exception:  # noqa: BLE001
            return str(self.output_device or "default output")

    def set_voice(self, voice: str) -> None:
        """Change the voice. A style vector, not a model: nothing is reloaded."""
        self.voice = voice

    def set_output_device(self, device: int | str | None) -> None:
        """Play through another device from the next line: the open stream is dropped and
        reopened on it."""
        self.output_device = device
        self._playback = None
        stream, self._stream = self._stream, None
        self._stream_device_name = None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()

    def load(self) -> bool:
        """Load the graph and the audio module once. Returns whether the voice is usable.

        Safe to call from any thread and any number of times; only the first call does
        work. The engine calls it at start so the first line is not the slow one.
        """
        with self._load_lock:
            if self._kokoro is not None:
                return True
            if self._failed:
                return False
            try:
                # Lazy, heavy, and optional (the `voice` extra); pyright need not resolve them.
                import sounddevice  # type: ignore[reportMissingImports]
                from kokoro_onnx import Kokoro  # type: ignore[reportMissingImports]

                begun = time.perf_counter()
                sess, where = make_session(self.onnx_path, self.device_wanted)
                self._kokoro = Kokoro.from_session(sess, str(self.voices_path))
                self._sd = sounddevice
                self.device = where
                self.load_seconds = time.perf_counter() - begun
                return True
            except Exception as exc:  # noqa: BLE001 - any load failure degrades to silence
                log.warning("Kokoro voice unavailable, going silent: %s", exc)
                self._failed = True
                return False

    def prime(self) -> float | None:
        """One short line synthesised and thrown away, so the first spoken one does not
        pay the runtime's warm-up. Returns the seconds it took, None if the voice did not
        load."""
        if not self.load():
            return None
        begun = time.perf_counter()
        with contextlib.suppress(Exception):
            self._kokoro.create("Ready.", voice=self.voice, speed=self.speed, lang=self.lang)  # type: ignore[attr-defined]
        return time.perf_counter() - begun

    def say(self, text: str, *, voice: str | None = None) -> None:
        """Queue a line to be spoken. Notation in it is read aloud in words. `voice` speaks
        this line in another voice (a preview) without changing the coach's."""
        if self._failed:
            return
        spoken = speak_notation(text).strip()
        if not spoken:
            return
        self._ensure_worker()
        self._queue.put((spoken, text, voice))

    @property
    def busy(self) -> bool:
        """Whether a line is playing or waiting to play."""
        return self._playing or not self._queue.empty()

    def stop(self) -> None:
        """Drop everything queued and silence anything now playing."""
        with self._lock:
            self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._stream is not None:
            self._stream.clear()
        if self._sd is not None:
            # Stopping audio must never raise out of stop().
            with contextlib.suppress(Exception):
                self._sd.stop()  # type: ignore[attr-defined]

    def close(self) -> None:
        """Release the output stream."""
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="kokoro-tts", daemon=True)
        self._worker.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            # Busy from the moment a line is taken up, through synthesis, until it has
            # finished playing -- so a caller asking `busy` mid-synthesis gets the truth.
            self._playing = True
            try:
                self._speak_one(*item)
            finally:
                self._playing = False

    def _speak_one(self, text: str, given: str = "", voice: str | None = None) -> None:
        if not self.load():
            return  # degraded to silence
        with self._lock:
            generation = self._generation
        try:
            samples, sample_rate = self._kokoro.create(  # type: ignore[attr-defined]
                text, voice=voice or self.voice, speed=self.speed, lang=self.lang
            )
        except Exception as exc:  # noqa: BLE001 - a bad line must not kill the voice
            log.warning("Kokoro synthesis failed for %r: %s", text, exc)
            return
        # A stop between synthesis and playback means this line was cancelled.
        with self._lock:
            if generation != self._generation:
                return
        if self.on_play is not None:
            with contextlib.suppress(Exception):
                self.on_play(given or text)
        try:
            import numpy as np

            samples = np.asarray(samples)
            pad = np.zeros((int(sample_rate * TAIL_SILENCE_S), *samples.shape[1:]), samples.dtype)
            samples = np.concatenate([samples, pad])
            stream = self._open_stream(int(sample_rate))
            if stream is not None:
                stream.play(samples, timeout=len(samples) / sample_rate + 5.0)
                return
            self._sd.play(samples, sample_rate, device=self.playback_device())  # type: ignore[attr-defined]
            self._sd.wait()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - audio errors are not fatal
            log.warning("Kokoro playback failed: %s", exc)

    def _open_stream(self, sample_rate: int) -> _Stream | None:
        """The open output stream for this rate and the current output device, opened or
        reopened as needed; None when the device will not take a stream, in which case
        lines play one by one as before."""
        if self._sd is None:
            return None
        now = time.monotonic()
        stream = self._stream
        if stream is not None and now - self._stream_checked >= DEVICE_CHECK_S:
            self._stream_checked = now
            if stream.samplerate != sample_rate or self._device_name() != self._stream_device_name:
                stream.close()
                stream = self._stream = None
        if stream is not None:
            return stream
        try:
            stream = _Stream(self._sd, sample_rate, self.playback_device())
        except Exception as exc:  # noqa: BLE001 - no stream: the per-line path stands in
            log.warning("Kokoro output stream unavailable, playing line by line: %s", exc)
            return None
        self._stream = stream
        self._stream_device_name = self._device_name()
        self._stream_checked = now
        return stream

    def _device_name(self) -> str | None:
        """The name of the output the voice plays through now: the configured device,
        or whatever the system default is at the moment."""
        try:
            device = self.playback_device()
            if follows_system_default(device, "output"):
                return str(self._sd.query_devices(self._sd.default.device[1]).get("name"))  # type: ignore[attr-defined]
            index = device if device is not None else self._sd.default.device[1]  # type: ignore[attr-defined]
            return str(self._sd.query_devices(index).get("name", index))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            return None
