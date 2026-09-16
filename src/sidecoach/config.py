"""Settings with defaults, optionally overridden by a TOML file.

Lookup order for the file: the explicit path, then `SIDECOACH_CONFIG`, then `sidecoach.toml`
in the working directory. A missing file means defaults.
"""

from __future__ import annotations

import os
import shutil
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field


class CaptureSettings(BaseModel):
    # "auto" (Windows Graphics Capture when available, else mss), "wgc", or "mss". GDI
    # capture (mss) can show a hardware-accelerated browser window frozen on Windows until
    # it is moved; WGC returns the composed screen and does not have that problem.
    backend: str = "auto"
    fps: int = 10
    record_frames: bool = True
    record_crops: bool = True


class EngineSettings(BaseModel):
    stockfish_path: str | None = None
    multipv: int = 4
    movetime_ms: int = 300
    threads: int = 4
    hash_mb: int = 256
    # Assessment runs to a fixed depth so that the position before a move and the position
    # after it are measured the same way; a time limit would compare unequal searches.
    depth: int = 18
    pv_plies: int = 12
    # "human" turns centipawns into expected score on a curve fitted to games between
    # people; "engine" uses Stockfish's own win-draw-loss, which is 1.0 from a knight up.
    score_model: str = "human"
    # A single position never takes longer than this, whatever depth was asked for.
    max_seconds: float = 5.0


class LLMSettings(BaseModel):
    # "local" (LM Studio) or "dashscope". Unset fields fall back to the provider defaults
    # in sidecoach.llm.factory. API keys come from the environment, never from this file.
    provider: str = "local"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    # How reasoning is switched off: "prefill", "param", or "none". Provider default if unset.
    thinking_control: str | None = None
    thinking: bool = False
    # Total output budget. Reasoning tokens count against it when thinking is on.
    max_output_tokens: int = 1500
    temperature: float = 0.7
    timeout_s: float = 60.0
    extra_body: dict[str, object] = Field(default_factory=dict)


class CoachSettings(BaseModel):
    stuck_seconds: int = 45
    quiet: bool = False
    persona: str = "warm_direct"
    # Fitted to chess.com's review tallies over six games; see analysis.score.Thresholds.
    inaccuracy: float = 0.04
    mistake: float = 0.10
    blunder: float = 0.28
    decisive_gap: float = 0.10
    only_move_gap: float = 0.30
    excellent: float = 0.02
    decided: float = 0.90


class ProfileSettings(BaseModel):
    rating_low: int = 1000
    rating_high: int = 1400


class VoiceSettings(BaseModel):
    # Off by default: the coach is silent until voice is turned on (a CLI flag, the panel).
    enabled: bool = False
    # "kokoro" (Kokoro-82M via kokoro-onnx, on the CPU) or "null" (silent). Unknown names
    # degrade to silent rather than failing -- voice never takes the session down.
    provider: str = "kokoro"
    # Any Kokoro voice id (am_*/af_*/bm_*/bf_*); af_sarah is an American female voice.
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"
    # Where the Kokoro model files live; defaults to <data_dir>/models/kokoro when unset.
    model_dir: str | None = None
    # Where each voice model runs: "cuda" (the default: the GPU when onnxruntime's CUDA
    # provider initialises, otherwise the CPU, with a warning in the log) or "cpu" (never the
    # GPU). "auto" is accepted as a synonym for "cuda". A model is loaded once, on the device
    # chosen at start, and kept for the session; changing the device means restarting.
    tts_device: str = "cuda"
    stt_device: str = "cuda"
    # Hearing: "parakeet" (NVIDIA Parakeet-TDT-0.6B via onnx-asr, the default; ~100 ms on the
    # CPU), "whisper" (faster-whisper, best raw accuracy, wants the GPU), or "null".
    stt_provider: str = "parakeet"
    # The model name for the provider (onnx-asr's for parakeet, faster-whisper's for whisper);
    # the provider's default when unset.
    stt_model: str | None = None
    # How the coach listens when the panel starts: "off", "ptt" (hold to talk), or "vad"
    # (voice activity: speak, and the coach is cut off when you start).
    listen: str = "ptt"
    # An input device index or name for sounddevice; the default input when unset.
    mic_device: str | None = None
    # An output device index or name for the coach's voice; the default output when unset.
    # (`sidecoach doctor` lists the devices.) The default is often a monitor's HDMI audio.
    output_device: str | None = None
    # The global push-to-talk key, held from any window (the game has the focus, not the
    # panel): a pynput key name such as "ctrl_r", "alt_r", "f9", a single character, or a
    # mouse side button "mouse_x1" / "mouse_x2". Empty disables the global key.
    ptt_key: str = "ctrl_r"


class Settings(BaseModel):
    data_dir: Path = Path("data")
    tools_dir: Path = Path("tools")
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    coach: CoachSettings = Field(default_factory=CoachSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)

    def voice_model_dir(self) -> Path:
        """Where the Kokoro model files live: the configured path, else under data_dir."""
        if self.voice.model_dir:
            return Path(self.voice.model_dir)
        return self.data_dir / "models" / "kokoro"


def load_dotenv(path: str | os.PathLike[str] = ".env") -> dict[str, str]:
    """Minimal .env loader: KEY=VALUE lines, optional quotes, no expansion.

    Variables already present in the environment win. Returns what was newly set.
    """
    dotenv = Path(path)
    loaded: dict[str, str] = {}
    if not dotenv.exists():
        return loaded
    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    load_dotenv()
    candidate = Path(path or os.environ.get("SIDECOACH_CONFIG", "sidecoach.toml"))
    if candidate.exists():
        with candidate.open("rb") as fh:
            return Settings.model_validate(tomllib.load(fh))
    return Settings()


def find_stockfish(settings: Settings) -> Path | None:
    """Explicit path, then anything under tools/, then PATH."""
    if settings.engine.stockfish_path:
        explicit = Path(settings.engine.stockfish_path)
        return explicit if explicit.exists() else None
    if settings.tools_dir.exists():
        found = sorted(settings.tools_dir.rglob("stockfish*.exe"))
        if found:
            return found[0]
    on_path = shutil.which("stockfish")
    return Path(on_path) if on_path else None
