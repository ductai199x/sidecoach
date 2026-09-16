"""Speech-to-text providers: the coach's hearing, on the CPU or the GPU as configured.

Mirrors the TTS side: a small protocol, a null provider that is always safe to call, and
real backends that load their model ONCE and keep it. The default, Parakeet-TDT-0.6B, runs on
the shared ONNX runtime (`sidecoach.voice.runtime`) like Kokoro does -- one runtime, two
graphs -- and needs no GPU to answer a 3 s ask in ~100 ms. Whisper large-v3-turbo through
faster-whisper is the alternative when the GPU is free: the best raw accuracy, but a second
runtime (CTranslate2) and ~1.5 GB of VRAM.

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
        """(ready, detail): whether the ear can hear, and why not when it cannot."""
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
            except Exception as exc:  # noqa: BLE001 - degrade to deaf, never crash the session
                self._failed = f"{type(exc).__name__}: {exc}"
                log.warning("Parakeet unavailable, coach is deaf: %s", exc)
                return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if not self.load():
            return ""
        try:
            text = self._asr.recognize(np.asarray(audio, dtype=np.float32), sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001 - a bad clip must not kill the ear
            log.warning("Parakeet failed on a clip: %s", exc)
            return ""
        return (text or "").strip()


class WhisperSTT:
    """OpenAI Whisper via faster-whisper (CTranslate2). Best raw accuracy; GPU strongly advised."""

    name = "whisper"

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "auto",
        compute_type: str | None = None,
        prompt: str = "Chess coaching: knight, bishop, rook, queen, king, castle, takes, check.",
    ) -> None:
        self.model = model
        self.device_wanted = device
        self.compute_type = compute_type
        self.prompt = prompt
        self.device: Device | None = None
        self.load_seconds: float | None = None
        self._whisper: Any = None
        self._failed: str | None = None
        self._lock = threading.Lock()

    def healthcheck(self) -> tuple[bool, str]:
        if find_spec("faster_whisper") is None:
            return False, "faster-whisper not installed; run `uv sync --group whisper`"
        if self._failed:
            return False, f"whisper {self.model} failed to load: {self._failed}"
        where = self.device.detail if self.device else f"{self.device_wanted} (not loaded yet)"
        return True, f"whisper {self.model}, {where}"

    def load(self) -> bool:
        with self._lock:
            if self._whisper is not None:
                return True
            if self._failed:
                return False
            try:
                from faster_whisper import WhisperModel  # type: ignore[reportMissingImports]

                device = self.device_wanted
                if device == "auto":
                    try:
                        import ctranslate2  # type: ignore[reportMissingImports]

                        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                    except Exception:  # noqa: BLE001
                        device = "cpu"
                compute = self.compute_type or ("float16" if device == "cuda" else "int8")
                begun = time.perf_counter()
                self._whisper = WhisperModel(self.model, device=device, compute_type=compute)
                self.load_seconds = time.perf_counter() - begun
                self.device = Device(device, f"{device}, {compute}")
                return True
            except Exception as exc:  # noqa: BLE001
                self._failed = f"{type(exc).__name__}: {exc}"
                log.warning("Whisper unavailable, coach is deaf: %s", exc)
                return False

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        if not self.load():
            return ""
        try:
            segments, _ = self._whisper.transcribe(
                np.asarray(audio, dtype=np.float32),
                language="en",
                beam_size=1,
                initial_prompt=self.prompt,
                vad_filter=False,
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("Whisper failed on a clip: %s", exc)
            return ""


def make_stt(provider: str, model: str | None, device: str) -> STTProvider:
    """The configured ear, or a deaf one with a reason. Never raises."""
    if provider in ("null", "off", ""):
        return NullSTT("speech recognition disabled")
    if provider == "parakeet":
        return ParakeetSTT(model or "nemo-parakeet-tdt-0.6b-v3", device=device)
    if provider == "whisper":
        return WhisperSTT(model or "large-v3-turbo", device=device)
    return NullSTT(f"unknown voice.stt_provider {provider!r}")
