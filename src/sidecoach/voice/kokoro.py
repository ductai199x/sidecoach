"""Kokoro-82M text-to-speech, via kokoro-onnx on the CPU.

Kokoro is small, natural, Apache-2.0, and real-time on the CPU (about seven times faster than
it speaks), which leaves the GPU free for the local LLM. A future port to the GPU would swap
kokoro-onnx for the PyTorch `kokoro` package on a CUDA build of torch; the onnxruntime wheel
here exposes no CUDA provider, so ONNX stays on the CPU for now.

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
from importlib.util import find_spec
from pathlib import Path

from sidecoach.voice.notation import speak_notation

log = logging.getLogger(__name__)

ONNX_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


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
    ) -> None:
        self.model_dir = Path(model_dir)
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._kokoro: object | None = None
        self._sd: object | None = None
        self._failed = False
        # Bumped on every stop so a synthesis in flight knows its line was cancelled and
        # does not play over what comes next.
        self._generation = 0
        self._lock = threading.Lock()

    @property
    def onnx_path(self) -> Path:
        return self.model_dir / ONNX_FILE

    @property
    def voices_path(self) -> Path:
        return self.model_dir / VOICES_FILE

    def healthcheck(self) -> tuple[bool, str]:
        """Whether the voice can speak, checked without loading the model.

        The extras must be importable and the two model files present; the model itself is
        big and is loaded lazily on the first line.
        """
        if find_spec("kokoro_onnx") is None:
            return False, "kokoro-onnx not installed; run `uv sync --group voice`"
        if find_spec("sounddevice") is None:
            return False, "sounddevice not installed; run `uv sync --group voice`"
        missing = [p.name for p in (self.onnx_path, self.voices_path) if not p.exists()]
        if missing:
            return False, f"model files missing from {self.model_dir}: {', '.join(missing)}"
        return True, f"kokoro-onnx, voice {self.voice}, CPU"

    def say(self, text: str) -> None:
        """Queue a line to be spoken. Notation in it is read aloud in words."""
        if self._failed:
            return
        spoken = speak_notation(text).strip()
        if not spoken:
            return
        self._ensure_worker()
        self._queue.put(spoken)

    def stop(self) -> None:
        """Drop everything queued and silence anything now playing."""
        with self._lock:
            self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._sd is not None:
            # Stopping audio must never raise out of stop().
            with contextlib.suppress(Exception):
                self._sd.stop()  # type: ignore[attr-defined]

    def _ensure_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, name="kokoro-tts", daemon=True)
        self._worker.start()

    def _load(self) -> bool:
        """Load the model and the audio module once. Returns whether the voice is usable."""
        if self._kokoro is not None:
            return True
        try:
            # Lazy, heavy, and optional (the `voice` extra); pyright need not resolve them.
            import sounddevice  # type: ignore[reportMissingImports]
            from kokoro_onnx import Kokoro  # type: ignore[reportMissingImports]

            self._kokoro = Kokoro(str(self.onnx_path), str(self.voices_path))
            self._sd = sounddevice
            return True
        except Exception as exc:  # noqa: BLE001 - any load failure degrades to silence
            log.warning("Kokoro voice unavailable, going silent: %s", exc)
            self._failed = True
            return False

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            if text is None:
                return
            if not self._load():
                return  # degraded to silence; drain nothing further
            with self._lock:
                generation = self._generation
            try:
                samples, sample_rate = self._kokoro.create(  # type: ignore[attr-defined]
                    text, voice=self.voice, speed=self.speed, lang=self.lang
                )
            except Exception as exc:  # noqa: BLE001 - a bad line must not kill the voice
                log.warning("Kokoro synthesis failed for %r: %s", text, exc)
                continue
            # A stop between synthesis and playback means this line was cancelled.
            with self._lock:
                if generation != self._generation:
                    continue
            try:
                self._sd.play(samples, sample_rate)  # type: ignore[attr-defined]
                self._sd.wait()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - audio errors are not fatal
                log.warning("Kokoro playback failed: %s", exc)
