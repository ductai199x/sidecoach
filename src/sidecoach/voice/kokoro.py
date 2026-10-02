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

from sidecoach.voice.devices import follows_system_default, output_changes, system_default
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
# The stream is reopened on the new device when the system's default output changes
# (`devices.output_changes`), before a line or while one plays.
VOICES_FILE = "voices-v1.0.bin"
# A stream whose device has not asked for samples for this long is taken for dead and
# reopened at the next line.
STREAM_DEAD_S = 1.5


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
        # When the device last asked for samples: a stream whose device went away (an HDMI
        # monitor asleep, a headset gone) stops calling back, and is dropped (see `play`).
        self.last_callback = time.monotonic()
        self.dead = False
        self._stream = sd.OutputStream(  # type: ignore[attr-defined]
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            device=device,
            callback=self._callback,
        )
        self._stream.start()
        self.last_callback = time.monotonic()

    def _callback(self, outdata: object, frames: int, time_info: object, status: object) -> None:
        out = outdata  # (frames, 1) float32
        self.last_callback = time.monotonic()
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

    def queue(self, samples: object) -> None:
        """Queue `samples` for the device; `wait` for them to be handed over."""
        block = self._np.asarray(samples, dtype=self._np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(block)
            self._idle.clear()

    def wait(self, timeout: float, between: Callable[[], _Stream | None] | None = None) -> bool:
        """Wait until what is queued has been handed to the device; False, and the stream
        marked dead, when the device stopped asking for samples (it went away) or the
        timeout passed. A dead stream held each line for its length and five seconds,
        silent, and kept the voice `busy` (voice finding 6, 2026-10-01). `between` is asked
        at every look meanwhile: a stream it returns has taken the rest of the line over
        (the system's output changed), and the wait goes on there."""
        end = time.monotonic() + timeout
        stream = self
        while not stream._idle.wait(0.1):
            if between is not None:
                moved = between()
                if moved is not None:
                    stream = moved
                    continue
            now = time.monotonic()
            if now >= end or now - stream.last_callback > STREAM_DEAD_S:
                stream.dead = True
                stream.clear()
                return False
        return True

    def take(self) -> list:
        """What is still queued, taken off this stream (for another to play)."""
        with self._lock:
            chunks, self._chunks = self._chunks, []
            self._idle.set()
        return chunks

    def play(self, samples: object, timeout: float) -> bool:
        """Queue `samples` and return once they have been handed to the device."""
        self.queue(samples)
        return self.wait(timeout)

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
        self._failed_files: tuple[tuple[int, int] | None, ...] = ()
        self._load_lock = threading.Lock()
        # Bumped on every stop so a synthesis in flight knows its line was cancelled and
        # does not play over what comes next.
        self._generation = 0
        self._lock = threading.Lock()
        self._playing = False
        # The open output stream (see _Stream) and the output it was opened for (`_output`):
        # None when the stream could not be opened, and lines then play one by one.
        self._stream: _Stream | None = None
        self._stream_output: tuple | None = None
        # The output a stream would not open on: not tried again until the output changes.
        self._failed_output: tuple | None = None
        self._follows: tuple[int | str | None, bool] | None = None

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
        self._follows = None
        stream, self._stream = self._stream, None
        self._stream_output = None
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
            if self._failed and not self._files_arrived():
                return False
            self._failed = False
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
                output_changes()  # the watch on the system's output set up now, not at a line
                return True
            except Exception as exc:  # noqa: BLE001 - any load failure degrades to silence
                log.warning("Kokoro voice unavailable, going silent: %s", exc)
                self._failed = True
                self._failed_files = self._model_files()
                return False

    def _model_files(self) -> tuple[tuple[int, int] | None, ...]:
        """The model files as they are now (size and time), None for a missing one."""
        found: list[tuple[int, int] | None] = []
        for path in (self.onnx_path, self.voices_path):
            try:
                st = path.stat()
                found.append((st.st_size, st.st_mtime_ns))
            except OSError:
                found.append(None)
        return tuple(found)

    def _files_arrived(self) -> bool:
        """A load that failed is tried again once the model files have changed since: the
        first run's warm-up ran before Setup fetched them, and the voice stayed dead until
        a restart (voice finding 4, 2026-10-01)."""
        now = self._model_files()
        return None not in now and now != self._failed_files

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
        if self._failed and not self._files_arrived():
            return
        spoken = speak_notation(text, voice=True).strip()
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
            # Under the lock with the voice's following of the output: a line moving to a new
            # stream is cleared there too, never carried past the stop.
            if self._stream is not None:
                self._stream.clear()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._sd is not None:
            # Stopping audio must never raise out of stop().
            with contextlib.suppress(Exception):
                self._sd.stop()  # type: ignore[attr-defined]

    def close(self) -> None:
        """Release everything: the worker, the model's session and the output stream. A
        voice replaced (the panel's `_reload_voice`) kept all three alive (voice finding
        11, 2026-10-01). A line said after this loads the voice again."""
        self.stop()
        worker, self._worker = self._worker, None
        if worker is not None and worker.is_alive():
            self._queue.put(None)
        stream, self._stream = self._stream, None
        self._stream_output = None
        if stream is not None:
            stream.close()
        with self._load_lock:
            self._kokoro = None

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
        try:
            import numpy as np

            samples = np.asarray(samples)
            pad = np.zeros((int(sample_rate * TAIL_SILENCE_S), *samples.shape[1:]), samples.dtype)
            samples = np.concatenate([samples, pad])
            stream = self._open_stream(int(sample_rate))
            # The cancel check and the queueing under one lock with `stop`'s count: a stop
            # landing while the stream opened was ignored, and the line played over a mute
            # or a barge-in (voice finding 13, 2026-10-01).
            with self._lock:
                if generation != self._generation:
                    return
                if stream is not None:
                    stream.queue(samples)
                else:
                    self._sd.play(samples, sample_rate, device=self.playback_device())  # type: ignore[attr-defined]
            if self.on_play is not None:
                with contextlib.suppress(Exception):
                    self.on_play(given or text)
            if stream is not None:
                follow = lambda: self._follow_output(int(sample_rate))  # noqa: E731
                if not stream.wait(len(samples) / sample_rate + 5.0, follow):
                    current = self._stream
                    if current is not None and current.dead:
                        log.warning("Kokoro output stream stopped; reopening at the next line")
                        self._drop_stream(current)
                return
            self._sd.wait()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - audio errors are not fatal
            log.warning("Kokoro playback failed: %s", exc)

    def _drop_stream(self, stream: _Stream) -> None:
        if self._stream is stream:
            self._stream = None
            self._stream_output = None
        with contextlib.suppress(Exception):
            stream.close()

    def _open_stream(self, sample_rate: int) -> _Stream | None:
        """The open output stream for this rate and the current output, opened or reopened as
        needed; None when the device will not take a stream, in which case lines play one by
        one as before."""
        if self._sd is None:
            return None
        output = self._output()
        stream = self._stream
        if stream is not None and (
            stream.samplerate != sample_rate or output != self._stream_output
        ):
            self._drop_stream(stream)
            stream = None
        if stream is not None:
            return stream
        stream = self._new_stream(sample_rate, output)
        if stream is not None:
            self._stream = stream
            self._stream_output = output
        return stream

    def _new_stream(self, sample_rate: int, output: tuple) -> _Stream | None:
        try:
            stream = _Stream(self._sd, sample_rate, self.playback_device())
        except Exception as exc:  # noqa: BLE001 - no stream: the per-line path stands in
            log.warning("Kokoro output stream unavailable, playing line by line: %s", exc)
            self._failed_output = output
            return None
        self._failed_output = None
        return stream

    def _follow_output(self, sample_rate: int) -> _Stream | None:
        """While a line plays: when the system's default output has changed, the rest of the
        line goes on a stream opened on the new one, and that stream is returned (Tai,
        2026-10-02: the voice follows headphones plugged in, mid-line if it must). None while
        nothing changed, or when the new output takes no stream (the old one plays on)."""
        output = self._output()
        old = self._stream
        if old is None or output == self._stream_output or output == self._failed_output:
            return None
        with self._lock:
            generation = self._generation
        new = self._new_stream(sample_rate, output)
        if new is None:
            return None
        with self._lock:
            if self._stream is not old or generation != self._generation:
                new.close()  # stopped or replaced meanwhile: nothing to carry over
                return None
            for chunk in old.take():
                new.queue(chunk)
            self._stream = new
            self._stream_output = output
        log.info("the system's output changed; the voice follows it")
        with contextlib.suppress(Exception):
            old.close()
        return new

    def _output(self) -> tuple:
        """The output the voice plays through, as a key that changes when it does: the
        configured device as given, or, when it follows the system default, the count of the
        system's changes of its default output (`devices.output_changes`, Windows' own word,
        free to read)."""
        device = self.playback_device()
        if self._follows is None or self._follows[0] != device:
            self._follows = (device, follows_system_default(device, "output"))
        if self._follows[1]:
            return ("default", output_changes())
        return ("device", device)
