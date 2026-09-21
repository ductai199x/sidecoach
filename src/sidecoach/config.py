"""Settings with defaults, optionally overridden by a TOML file.

Lookup order for the file: the explicit path, then `SIDECOACH_CONFIG`, then `sidecoach.toml`
in the working directory. A missing file means defaults. `SIDECOACH_DATA_DIR` overrides
`data_dir` (the desktop shell sets it to the per-user data folder). Every field carries a
description: the settings page of the app is generated from this schema
(`Settings.model_json_schema()`), so what is written here is what the user reads.

Secrets (API keys) never live in the file: they come from the environment, `.env`, or the
OS credential store (`secrets_get`, `secrets_set`, through `keyring` when it is installed).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr


def _extra(**kw: Any) -> dict[str, Any]:
    """The schema extras the settings page reads: applies (now | next | restart), choices,
    kind (path, input_device, output_device, voice, persona), group (the AI Coach page's
    sections: provider, local, service, general), advanced, dev."""
    return dict(kw)


class CaptureSettings(BaseModel):
    backend: str = Field(
        default="auto",
        description="How the screen is read: auto (Windows Graphics Capture when available, "
        "else mss), wgc, or mss. GDI capture (mss) can show a hardware-accelerated browser "
        "window frozen on Windows until it is moved; WGC returns the composed screen.",
        json_schema_extra=_extra(choices=["auto", "wgc", "mss"], applies="next"),
    )
    fps: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Frames read per second.",
        json_schema_extra=_extra(applies="now"),
    )


class EngineSettings(BaseModel):
    stockfish_path: str | None = Field(
        default=None,
        description="The Stockfish binary. Unset: the one under tools/, else PATH.",
        json_schema_extra=_extra(applies="next", kind="path"),
    )
    multipv: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Candidate lines the engine keeps.",
        json_schema_extra=_extra(applies="now"),
    )
    movetime_ms: int = Field(
        default=300,
        ge=50,
        le=5000,
        description="Milliseconds per live search.",
        json_schema_extra=_extra(applies="now"),
    )
    threads: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Engine threads.",
        json_schema_extra=_extra(applies="now"),
    )
    hash_mb: int = Field(
        default=256,
        ge=16,
        le=8192,
        description="Engine hash table, in MB.",
        json_schema_extra=_extra(applies="now"),
    )
    depth: int = Field(
        default=18,
        ge=6,
        le=40,
        description="Assessment runs to a fixed depth so the position before a move and the "
        "position after it are measured the same way.",
        json_schema_extra=_extra(applies="now"),
    )
    pv_plies: int = Field(
        default=12,
        ge=2,
        le=40,
        description="Plies of the principal variation kept.",
        json_schema_extra=_extra(applies="now"),
    )
    score_model: str = Field(
        default="human",
        description="human: centipawns into expected score on a curve fitted to games between "
        "people; engine: Stockfish's own win-draw-loss, which is 1.0 from a knight up.",
        json_schema_extra=_extra(choices=["human", "engine"], applies="now"),
    )
    max_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="A single position never takes longer than this.",
        json_schema_extra=_extra(applies="now"),
    )


class LLMSettings(BaseModel):
    provider: str = Field(
        default="openai",
        description="A preset -- openai (gpt-5.6-luna at medium effort, what the coach was "
        "evaluated on), local (LM Studio on this machine), dashscope -- or any other name for "
        "a custom OpenAI-compatible endpoint, which then needs base_url and model; or none: "
        "no model, the coach speaks from the board and its templates, free and offline.",
        json_schema_extra=_extra(
            group="provider",
            choices=["openai", "local", "dashscope", "custom", "none"],
            applies="now",
        ),
    )
    model: str | None = Field(
        default=None,
        description="The model name; the preset's default when unset.",
        json_schema_extra=_extra(group="general", applies="now"),
    )
    base_url: str | None = Field(
        default=None,
        description="The endpoint of a model served on this machine or the network, for the "
        "local preset (LM Studio's is http://localhost:1234/v1, the default) or a custom "
        "provider.",
        json_schema_extra=_extra(group="local", applies="now"),
    )
    api_key_env: str | None = Field(
        default=None,
        description="The environment variable holding a custom provider's key. The presets "
        "have theirs (OPENAI_API_KEY, DASHSCOPE_API_KEY); keys are kept in the credential "
        "store, never in this file.",
        json_schema_extra=_extra(group="service", applies="now"),
    )
    thinking_control: str | None = Field(
        default=None,
        description="How reasoning is controlled: prefill (an empty think block, LM Studio), "
        "param (enable_thinking, DashScope), effort (reasoning_effort, OpenAI's reasoning "
        "models) or none. The preset's default when unset.",
        json_schema_extra=_extra(
            group="general", choices=["", "prefill", "param", "effort", "none"], applies="now"
        ),
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="For effort providers: none, low, medium, high; the preset's when unset.",
        json_schema_extra=_extra(
            group="service", choices=["", "none", "low", "medium", "high"], applies="now"
        ),
    )
    routine_effort: str | None = Field(
        default="none",
        description="The effort for a mundane coach reply (a word on a fine move): none gets "
        "the first sound out in under a second; low reasons about as long as medium. Unset "
        "keeps reasoning_effort.",
        json_schema_extra=_extra(
            group="service", choices=["", "none", "low", "medium", "high"], applies="now"
        ),
    )
    price_prompt_per_m: float | None = Field(
        default=None,
        ge=0,
        description="USD per million fresh prompt tokens, for the session's cost.",
        json_schema_extra=_extra(group="service", applies="now"),
    )
    price_cached_per_m: float | None = Field(
        default=None,
        ge=0,
        description="USD per million cached prompt tokens.",
        json_schema_extra=_extra(group="service", applies="now"),
    )
    price_completion_per_m: float | None = Field(
        default=None,
        ge=0,
        description="USD per million completion tokens, reasoning included.",
        json_schema_extra=_extra(group="service", applies="now"),
    )
    thinking: bool = Field(
        default=False,
        description="Let the model reason before answering (reasoning tokens count against "
        "the output budget).",
        json_schema_extra=_extra(group="general", applies="now"),
    )
    max_output_tokens: int = Field(
        default=1500,
        ge=64,
        le=32000,
        description="Total output budget per call.",
        json_schema_extra=_extra(group="general", applies="now"),
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
        json_schema_extra=_extra(group="general", applies="now"),
    )
    timeout_s: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="Seconds before a call is given up.",
        json_schema_extra=_extra(group="general", applies="now"),
    )
    extra_body: dict[str, object] = Field(
        default_factory=dict,
        description="Extra fields sent with every request (advanced).",
        json_schema_extra=_extra(group="general", applies="now", advanced=True),
    )


class CoachSettings(BaseModel):
    stuck_seconds: int = Field(
        default=45,
        ge=5,
        le=600,
        description="Seconds without a move before the coach offers a nudge.",
        json_schema_extra=_extra(applies="now"),
    )
    quiet: bool = Field(
        default=False,
        description="The coach speaks only when asked.",
        json_schema_extra=_extra(applies="now"),
    )
    persona: str = Field(
        default="warm_direct",
        description="The coach's manner: how it phrases things. The lessons and the rules "
        "are the same in every one.",
        json_schema_extra=_extra(
            applies="now",
            kind="persona",
            # sidecoach.coach.prompt.PERSONAS, listed here so the settings need no import
            # of the coach; a test keeps the two together.
            choices=["warm_direct", "calm_quiet", "sharp_demanding", "playful"],
        ),
    )
    verify: str = Field(
        default="",
        description="Test-time verification: a second model reads every candidate reply and the "
        "clean ones are spoken. Empty is off; same uses the coach's own model; else a provider "
        "name (dashscope with verify_model qwen3.8-flash is what the evaluation measured).",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    verify_model: str | None = Field(
        default=None,
        description="The verifier's model.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    summarise_every: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Every this many plies, what was said earlier is folded into a summary. "
        "0 is off.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    compact_record: bool = Field(
        default=False,
        description="Run sound moves nobody spoke after together as a move list.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    verbalized_sampling: bool = Field(
        default=False,
        description="The model offers several replies and one is spoken, for variety of phrasing.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    local_verdicts: bool = Field(
        default=True,
        description="The word on a best, excellent or book move is written from the board, free "
        "and immediate, instead of asked of the model; good, great and brilliant keep the model.",
        json_schema_extra=_extra(applies="now"),
    )
    inaccuracy: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="Expected-score loss that counts as an inaccuracy.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    mistake: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Expected-score loss that counts as a mistake.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    blunder: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        description="Expected-score loss that counts as a blunder.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    decisive_gap: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="The gap between the best move and the rest that makes a moment decisive.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    only_move_gap: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="The gap that makes the best move the only move.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    excellent: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Loss under which a move is excellent.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )
    decided: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Expected score beyond which the game is decided.",
        json_schema_extra=_extra(applies="now", advanced=True),
    )


class ProfileSettings(BaseModel):
    rating_low: int = Field(
        default=1000,
        ge=100,
        le=3000,
        description="The bottom of your rating band.",
        json_schema_extra=_extra(applies="now"),
    )
    rating_high: int = Field(
        default=1400,
        ge=100,
        le=3200,
        description="The top of your rating band.",
        json_schema_extra=_extra(applies="now"),
    )


class VoiceSettings(BaseModel):
    voice: str = Field(
        default="af_sarah",
        description="The coach's voice: Kokoro's English ones, American or British, female "
        "or male.",
        json_schema_extra=_extra(applies="now", kind="voice"),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speaking speed.",
        json_schema_extra=_extra(applies="now"),
    )
    lang: str = Field(
        default="en-us",
        description="The voice's language code.",
        json_schema_extra=_extra(applies="now"),
    )
    model_dir: str | None = Field(
        default=None,
        description="Where the Kokoro model files live; <data>/models/kokoro when unset.",
        json_schema_extra=_extra(applies="now", kind="path"),
    )
    tts_device: str = Field(
        default="cuda",
        description="Where the voice runs: cuda (the GPU when it initialises, else the CPU) "
        "or cpu.",
        json_schema_extra=_extra(choices=["cuda", "cpu"], applies="now"),
    )
    stt_device: str = Field(
        default="cuda",
        description="Where the ear runs: cuda or cpu.",
        json_schema_extra=_extra(choices=["cuda", "cpu"], applies="now"),
    )
    listen: str = Field(
        default="ptt",
        description="How the coach listens when the panel starts: off, ptt (hold to talk), "
        "or vad (voice activity).",
        json_schema_extra=_extra(choices=["off", "ptt", "vad"], applies="now"),
    )
    mic_device: str | None = Field(
        default=None,
        description="An input device index or name; the default input when unset.",
        json_schema_extra=_extra(applies="now", kind="input_device"),
    )
    output_device: str | None = Field(
        default=None,
        description="An output device index or name for the coach's voice; the default "
        "output when unset.",
        json_schema_extra=_extra(applies="now", kind="output_device"),
    )
    ptt_key: str = Field(
        default="ctrl_r",
        description="The global push-to-talk key, held from any window: a pynput key name "
        "(ctrl_r, alt_r, f9), a character, or mouse_x1 / mouse_x2. Empty disables it.",
        json_schema_extra=_extra(applies="now"),
    )


class GpuSettings(BaseModel):
    mode: str = Field(
        default="auto",
        description="auto: the CUDA pack is fetched when an eligible NVIDIA GPU is found and "
        "used when it initialises; cpu: never; cuda: always, and a failure is an error.",
        json_schema_extra=_extra(choices=["auto", "cpu", "cuda"], applies="restart"),
    )


class Settings(BaseModel):
    # Where these settings were read from, so a save goes back there: the panel started
    # with `--config x.toml` once saved to `sidecoach.toml` in the working directory.
    _source: Path | None = PrivateAttr(default=None)

    @property
    def source(self) -> Path:
        """The file these settings came from, or would be written to."""
        return self._source or settings_path()

    data_dir: Path = Field(
        default=Path("data"),
        description="Where the app keeps its data: the profile, the caches, the models.",
        json_schema_extra=_extra(applies="restart", kind="path"),
    )
    tools_dir: Path = Field(
        default=Path("tools"),
        description="Where Stockfish is looked for when no path is set.",
        json_schema_extra=_extra(applies="next", kind="path"),
    )
    capture: CaptureSettings = Field(default_factory=CaptureSettings, description="The screen")
    engine: EngineSettings = Field(default_factory=EngineSettings, description="Stockfish")
    llm: LLMSettings = Field(default_factory=LLMSettings, description="The model")
    coach: CoachSettings = Field(default_factory=CoachSettings, description="The coach")
    profile: ProfileSettings = Field(default_factory=ProfileSettings, description="You")
    voice: VoiceSettings = Field(default_factory=VoiceSettings, description="Voice and ear")
    gpu: GpuSettings = Field(default_factory=GpuSettings, description="GPU acceleration")

    def voice_model_dir(self) -> Path:
        """Where the Kokoro model files live: the configured path, else under data_dir."""
        if self.voice.model_dir:
            return Path(self.voice.model_dir)
        return self.data_dir / "models" / "kokoro"


# The environment variables the presets read their keys from (sidecoach.llm.factory).
KEY_ENVS = {"openai": "OPENAI_API_KEY", "dashscope": "DASHSCOPE_API_KEY"}
KEYRING_SERVICE = "sidecoach"


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


def secrets_get(name: str) -> str | None:
    """A secret from the OS credential store, or None (also when keyring is missing)."""
    try:
        import keyring  # type: ignore[import-not-found]

        return keyring.get_password(KEYRING_SERVICE, name)
    except Exception:  # noqa: BLE001 - no store, no backend, no entry: all mean None
        return None


def secrets_set(name: str, value: str | None) -> bool:
    """Store (or with None, delete) a secret; False when there is no credential store."""
    try:
        import keyring  # type: ignore[import-not-found]

        if value:
            keyring.set_password(KEYRING_SERVICE, name, value)
        else:
            with contextlib.suppress(Exception):  # nothing to delete
                keyring.delete_password(KEYRING_SERVICE, name)
        return True
    except Exception:  # noqa: BLE001
        return False


def load_secrets(settings: Settings) -> None:
    """Put the keys the settings need into the environment from the credential store,
    unless the environment (or .env) already has them."""
    names = set(KEY_ENVS.values())
    if settings.llm.api_key_env:
        names.add(settings.llm.api_key_env)
    for name in names:
        if name not in os.environ:
            value = secrets_get(name)
            if value:
                os.environ[name] = value


def settings_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Where the settings file is (or would be): the explicit path, `SIDECOACH_CONFIG`,
    else `sidecoach.toml` in the working directory."""
    return Path(path or os.environ.get("SIDECOACH_CONFIG", "sidecoach.toml"))


def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    load_dotenv()
    candidate = settings_path(path)
    if candidate.exists():
        with candidate.open("rb") as fh:
            settings = Settings.model_validate(tomllib.load(fh))
    else:
        settings = Settings()
    settings._source = candidate
    data_dir = os.environ.get("SIDECOACH_DATA_DIR")
    if data_dir:
        settings.data_dir = Path(data_dir)
        # In the app, Stockfish lives under the data directory too, unless the file says
        # where else; the development default `tools/` is relative to a checkout.
        if settings.tools_dir == Path("tools"):
            settings.tools_dir = settings.data_dir / "tools"
    load_secrets(settings)
    return settings


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, dict):
        inner = ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def to_toml(settings: Settings) -> str:
    """The settings as TOML: the top-level scalars, then one table per section. Fields at
    their defaults are written too, so the file is a complete, readable record; None is
    left out (TOML has no null), which reads back as the default."""
    lines: list[str] = ["# Sidecoach settings. Written by the app; edit by hand if you like.", ""]
    data = settings.model_dump(mode="json")
    for key in ("data_dir", "tools_dir"):
        lines.append(f"{key} = {_toml_value(data[key])}")
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, dict) and not value:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def save_settings(settings: Settings, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the settings file: `path`, else where the settings came from (see `source`);
    returns where."""
    target = Path(path) if path else settings.source
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_toml(settings), encoding="utf-8")
    return target


def require_stockfish(settings: Settings) -> Path:
    """The Stockfish binary, or a clear error for a tool that cannot run without it."""
    path = find_stockfish(settings)
    if path is None:
        raise RuntimeError(
            "Stockfish not found: set engine.stockfish_path in sidecoach.toml, put the binary "
            "under tools/, or add it to PATH"
        )
    return path


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
