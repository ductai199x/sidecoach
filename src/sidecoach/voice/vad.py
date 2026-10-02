"""Voice activity: when the user starts talking (barge-in) and when they have finished.

Silero VAD, on the shared ONNX runtime, always on the CPU -- it is a 2 MB model that costs
under a millisecond per 32 ms frame, and the GPU is for the big graphs. `SileroGate` turns
16 kHz audio into a speech probability per frame, carrying the model's state across frames.
`Endpointer` is the state machine on top: speech has STARTED after a couple of confident
frames (cut the coach off), and an utterance has ENDED after enough trailing silence. Both
are plain and testable with synthetic probabilities; neither knows about microphones.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
HOP = 512  # samples per frame: 32 ms at 16 kHz, the Silero v5 window
CONTEXT = 64  # samples of the previous frame the model wants in front of each window


class SileroGate:
    """Speech probability per 32 ms frame, with the model's recurrent state carried along."""

    def __init__(self) -> None:
        self._session: Any = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._failed: str | None = None
        self._lock = threading.Lock()

    def healthcheck(self) -> tuple[bool, str]:
        if self._failed:
            return False, f"silero vad failed to load: {self._failed}"
        return True, "silero vad, cpu"

    def load(self) -> bool:
        with self._lock:
            if self._session is not None:
                return True
            if self._failed:
                return False
            try:
                import onnx_asr  # type: ignore[reportMissingImports]
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                vad = onnx_asr.load_vad(
                    "silero", providers=["CPUExecutionProvider"], sess_options=opts
                )
                # onnx-asr keeps the InferenceSession it made on `_model`; drive it per frame.
                self._session = getattr(vad, "_model")  # noqa: B009
                return True
            except Exception as exc:  # noqa: BLE001 - no VAD means push-to-talk only
                self._failed = f"{type(exc).__name__}: {exc}"
                log.warning("Silero VAD unavailable: %s", exc)
                return False

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one HOP-sample frame (float32, 16 kHz)."""
        if not self.load():
            return 0.0
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape[0] > HOP:
            frame = frame[:HOP]
        elif frame.shape[0] < HOP:
            frame = np.pad(frame, (0, HOP - frame.shape[0]))
        window = np.concatenate([self._context, frame])[None, :]
        feeds = {
            "input": window,
            "state": self._state,
            "sr": np.array([SAMPLE_RATE], dtype=np.int64),
        }
        out, self._state = self._session.run(["output", "stateN"], feeds)
        self._context = frame[-CONTEXT:]
        return float(out[0, 0])


@dataclass
class Endpointer:
    """Start and end of an utterance from a stream of per-frame speech probabilities.

    Speech starts when `start_frames` consecutive frames exceed `start` (two frames = 64 ms:
    quick enough for a barge-in, long enough to ignore a click). It ends after `end_ms` of
    frames below `end`. While the coach is speaking, the mic hears the speakers, so the start
    threshold is raised to `start_while_speaking` -- a headset makes this moot. (The listener
    no longer feeds it the coach's voice: since 2026-10-02 it ignores the microphone while the
    voice plays, `mic.Listener._run`.)
    """

    start: float = 0.5
    end: float = 0.35
    start_frames: int = 2
    end_ms: int = 400
    start_while_speaking: float = 0.85
    min_utterance_ms: int = 250
    frame_ms: int = 32

    speaking: bool = field(default=False, init=False)  # inside an utterance
    _run: int = field(default=0, init=False)  # consecutive frames over the start threshold
    _silence_ms: int = field(default=0, init=False)
    _length_ms: int = field(default=0, init=False)

    def feed(self, prob: float, *, coach_speaking: bool = False) -> str | None:
        """Advance one frame. Returns "start", "end", "drop" (too short to be an ask), or None."""
        if not self.speaking:
            threshold = self.start_while_speaking if coach_speaking else self.start
            self._run = self._run + 1 if prob >= threshold else 0
            if self._run >= self.start_frames:
                self.speaking = True
                self._silence_ms = 0
                self._length_ms = self._run * self.frame_ms
                self._run = 0
                return "start"
            return None
        self._length_ms += self.frame_ms
        if prob < self.end:
            self._silence_ms += self.frame_ms
            if self._silence_ms >= self.end_ms:
                self.speaking = False
                spoken = self._length_ms - self._silence_ms
                self._length_ms = 0
                self._silence_ms = 0
                return "end" if spoken >= self.min_utterance_ms else "drop"
        else:
            self._silence_ms = 0
        return None

    def reset(self) -> None:
        self.speaking = False
        self._run = 0
        self._silence_ms = 0
        self._length_ms = 0
