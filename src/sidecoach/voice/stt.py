"""Speech-to-text providers: the coach's hearing, on the CPU or the GPU as configured.

Mirrors the TTS side: a small protocol, a null provider that is always safe to call, and
real backends that load their model ONCE and keep it. The default, Parakeet-TDT-0.6B, runs on
the shared ONNX runtime (`sidecoach.voice.runtime`) like Kokoro does -- one runtime, two
graphs -- and needs no GPU to answer a 3 s ask in ~100 ms. It is the one recogniser: Whisper
through faster-whisper was tried (a second runtime, CTranslate2, and ~1.5 GB of VRAM for
better raw accuracy the chess correction did not need) and is not shipped.

`transcribe` takes float32 mono audio at 16 kHz and returns raw text; the chess-aware
correction lives in `sidecoach.voice.correct` and is the caller's step.
"""

from __future__ import annotations

import logging
import threading
import time
from importlib.util import find_spec
from typing import Any, Protocol, runtime_checkable

import numpy as np

from sidecoach.voice.runtime import Device, providers_for

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@runtime_checkable
class STTProvider(Protocol):
    name: str

    def healthcheck(self) -> tuple[bool, str]:
        """(ready, detail): whether speech recognition works, and why not when it does not."""
        ...

    def load(self) -> bool:
        """Load the model once (idempotent, thread-safe). Returns whether it is usable."""
        ...

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        """Raw text for a float32 mono clip. Empty string when nothing was heard."""
        ...


class NullSTT:
    """The deaf provider: healthcheck says why, `transcribe` hears nothing."""

    name = "null"

    def __init__(self, reason: str = "speech recognition disabled") -> None:
        self._reason = reason

    def healthcheck(self) -> tuple[bool, str]:
        return False, self._reason

    def load(self) -> bool:
        return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        return ""


PARAKEET_MODEL = "nemo-parakeet-tdt-0.6b-v3"


class ParakeetSTT:
    """NVIDIA Parakeet-TDT-0.6B via onnx-asr on the shared ONNX runtime."""

    name = "parakeet"

    def __init__(
        self,
        model: str = "nemo-parakeet-tdt-0.6b-v3",
        *,
        device: str = "auto",
        quantization: str | None = None,
    ) -> None:
        self.model = model
        self.device_wanted = device
        self.quantization = quantization
        self.device: Device | None = None
        self.load_seconds: float | None = None
        self._asr: Any = None
        self._failed: str | None = None
        self._lock = threading.Lock()

    def healthcheck(self) -> tuple[bool, str]:
        if find_spec("onnx_asr") is None:
            return False, "onnx-asr not installed; run `uv sync --group voice`"
        if self._failed:
            return False, f"{self.model} failed to load: {self._failed}"
        where = self.device.detail if self.device else f"{self.device_wanted} (not loaded yet)"
        return True, f"{self.model}, {where}"

    def load(self) -> bool:
        with self._lock:
            if self._asr is not None:
                return True
            if self._failed:
                return False
            try:
                import onnx_asr  # type: ignore[reportMissingImports]
                import onnxruntime as ort

                providers, where = providers_for(self.device_wanted)
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                begun = time.perf_counter()
                self._asr = onnx_asr.load_model(
                    self.model,
                    quantization=self.quantization,
                    providers=providers,
                    sess_options=opts,
                )
                self.load_seconds = time.perf_counter() - begun
                self.device = where
                return True
            except Exception as exc:  # noqa: BLE001 - degrade to off, never crash the session
                if "LocalEntryNotFound" in type(exc).__name__ or "outgoing traffic" in str(exc):
                    # The app keeps the hub offline: the model is fetched by the setup only.
                    self._failed = "not downloaded yet: open Setup from the menu"
                else:
                    self._failed = f"{type(exc).__name__}: {exc}"
                log.warning("Parakeet unavailable, speech recognition is off: %s", exc)
                return False

    def prime(self) -> float | None:
        """Half a second of silence through the model, so the first real clip does not pay
        the runtime's warm-up (~0.7 s on CUDA). Returns the seconds it took, None if the
        recogniser did not load."""
        if not self.load():
            return None
        begun = time.perf_counter()
        self.transcribe(np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
        return time.perf_counter() - begun

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if not self.load():
            return ""
        try:
            text = self._asr.recognize(np.asarray(audio, dtype=np.float32), sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001 - a bad clip must not kill the recogniser
            log.warning("Parakeet failed on a clip: %s", exc)
            return ""
        return (text or "").strip()


def make_stt(device: str | None, *, off: bool = False) -> STTProvider:
    """Speech recognition on `device` (cpu or cuda), or off when `off`. Never raises."""
    if off or not device:
        return NullSTT("speech recognition off")
    return ParakeetSTT(PARAKEET_MODEL, device=device)
