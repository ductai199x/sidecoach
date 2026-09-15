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
    fps: int = 10
    record_frames: bool = True
    record_crops: bool = True


class EngineSettings(BaseModel):
    stockfish_path: str | None = None
    multipv: int = 4
    movetime_ms: int = 300
    threads: int = 4
    hash_mb: int = 256


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
    inaccuracy: float = 0.05
    mistake: float = 0.10
    blunder: float = 0.20
    decisive_gap: float = 0.10


class ProfileSettings(BaseModel):
    rating_low: int = 1000
    rating_high: int = 1400


class Settings(BaseModel):
    data_dir: Path = Path("data")
    tools_dir: Path = Path("tools")
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    engine: EngineSettings = Field(default_factory=EngineSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    coach: CoachSettings = Field(default_factory=CoachSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)


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
