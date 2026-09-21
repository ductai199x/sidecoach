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
    """The schema extras the settings page reads: applies (now | next | restart), choices
    and choice_labels, kind (path, input_device, output_device, voice, persona), label (as
    a person reads it), section (the heading a field sits under on its tab), only_for
    (the AI Coach page: the providers a field is shown for), advanced (shown when asked),
    dev (never shown in the app)."""
    return dict(kw)


class CaptureSettings(BaseModel):
    backend: str = Field(
        default="auto",
        description="Automatic picks Windows Graphics Capture when it is available, else GDI. "
        "GDI can show a hardware-accelerated browser window frozen until it is "
        "moved.",
        json_schema_extra=_extra(
            choices=["auto", "wgc", "mss"],
            applies="next",
            label="How the screen is read",
            choice_labels={
                "auto": "Automatic",
                "wgc": "Windows Graphics Capture",
                "mss": "GDI (mss)",
            },
            advanced=True,
            section="Reading the screen",
        ),
    )
    fps: int = Field(
        default=10,
        ge=1,
        le=30,
        description="How often the screen is read. Ten is plenty for a game; more costs CPU.",
        json_schema_extra=_extra(
            applies="now", label="Frames per second", section="Reading the screen"
        ),
    )


class EngineSettings(BaseModel):
    stockfish_path: str | None = Field(
        default=None,
        description="The Stockfish executable. Leave empty: the one the app fetched (under the "
        "tools folder), else the one on PATH.",
        json_schema_extra=_extra(
            applies="next",
            kind="path",
            label="Stockfish program",
            advanced=True,
            section="The program",
        ),
    )
    multipv: int = Field(
        default=6,
        ge=1,
        le=10,
        description="How many of the best moves Stockfish reports for each position, and so how "
        "many the coach can compare. More costs time.",
        json_schema_extra=_extra(
            applies="now", label="Candidate lines", advanced=True, section="What it looks at"
        ),
    )
    movetime_ms: int = Field(
        default=300,
        ge=50,
        le=5000,
        description="Time for the quick check of a position during play, in milliseconds, "
        "before the full analysis follows.",
        json_schema_extra=_extra(
            applies="now", label="Live search time (ms)", advanced=True, section="What it looks at"
        ),
    )
    threads: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Processor threads Stockfish may use. More is faster to a point; leave a "
        "few for the game and the browser.",
        json_schema_extra=_extra(
            applies="now", label="CPU threads", section="How hard Stockfish works"
        ),
    )
    hash_mb: int = Field(
        default=256,
        ge=16,
        le=8192,
        description="Memory Stockfish keeps for positions it has already seen, in MB. More "
        "helps a deep search; 256 is plenty.",
        json_schema_extra=_extra(
            applies="now", label="Memory (MB)", section="How hard Stockfish works"
        ),
    )
    depth: int = Field(
        default=18,
        ge=6,
        le=40,
        description="How deep Stockfish looks at every position. Deeper is more accurate and "
        "slower; 18 is a good balance on a modern machine.",
        json_schema_extra=_extra(applies="now", label="Depth", section="How hard Stockfish works"),
    )
    pv_plies: int = Field(
        default=12,
        ge=2,
        le=40,
        description="How far ahead each line is shown, in half-moves, when the coach gives a line.",
        json_schema_extra=_extra(
            applies="now", label="Line length (plies)", advanced=True, section="What it looks at"
        ),
    )
    score_model: str = Field(
        default="human",
        description="How Stockfish's evaluation becomes your winning chance. 'As people play' "
        "is fitted to games between people, so a pawn up means what it means for "
        "you; Stockfish's own numbers assume perfect play, where a knight up is "
        "already won.",
        json_schema_extra=_extra(
            choices=["human", "engine"],
            applies="now",
            label="Scoring",
            choice_labels={
                "human": "As people play (recommended)",
                "engine": "Stockfish's own win-draw-loss",
            },
            advanced=True,
            section="What it looks at",
        ),
    )
    max_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
        description="The most time one position may take. A think that runs long is cut off "
        "here so the coach keeps up with the game.",
        json_schema_extra=_extra(
            applies="now",
            label="Time limit per position (s)",
            advanced=True,
            section="How hard Stockfish works",
        ),
    )


class LLMSettings(BaseModel):
    provider: str = Field(
        default="openai",
        description="Who the coach thinks with. OpenAI is what the coach was evaluated on; a "
        "model on this computer costs nothing per game; with no model the coach "
        "still speaks, from the board and its templates.",
        json_schema_extra=_extra(
            group="provider",
            choices=["openai", "local", "dashscope", "custom", "none"],
            applies="now",
            label="Provider",
            choice_labels={
                "openai": "OpenAI (gpt-5.6-luna) -- recommended",
                "dashscope": "DashScope (Alibaba Cloud Model Studio)",
                "local": "A model on this computer (LM Studio, Ollama, vLLM)",
                "custom": "Another OpenAI-compatible service",
                "none": "No model -- free and offline",
            },
            section="Which model",
        ),
    )
    model: str | None = Field(
        default=None,
        description="The model's name. Empty uses the provider's default (gpt-5.6-luna; "
        "qwen3.8-max; for a local server, whichever model it has loaded).",
        json_schema_extra=_extra(
            group="general",
            applies="now",
            label="Model",
            only_for=["openai", "dashscope", "local", "custom"],
            section="Which model",
        ),
    )
    base_url: str | None = Field(
        default=None,
        description="Where the model is served. LM Studio's is http://localhost:1234/v1, the "
        "default for a model on this computer.",
        json_schema_extra=_extra(
            group="local",
            applies="now",
            label="Server address",
            only_for=["local", "custom"],
            section="Which model",
        ),
    )
    api_key_env: str | None = Field(
        default=None,
        description="Development: the environment variable a custom endpoint's key is read "
        "from. The app stores the key itself (CUSTOM_API_KEY).",
        json_schema_extra=_extra(
            group="service",
            applies="now",
            label="API key variable",
            dev=True,
            section="Development",
        ),
    )
    thinking_control: str | None = Field(
        default=None,
        description="Development: how a preset switches reasoning on the wire -- prefill, a "
        "parameter, or an effort. Each preset knows its own.",
        json_schema_extra=_extra(
            group="general",
            choices=["", "prefill", "param", "effort", "none"],
            applies="now",
            label="Reasoning switch",
            dev=True,
            section="Development",
        ),
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="How long the model may think before it answers. More is slower and costs "
        "more; medium is what the coach was evaluated at.",
        json_schema_extra=_extra(
            group="service",
            choices=["", "none", "low", "medium", "high"],
            applies="now",
            label="Thinking effort",
            only_for=["openai"],
            choice_labels={
                "": "Default (medium)",
                "none": "None -- fastest",
                "low": "Low",
                "medium": "Medium",
                "high": "High -- slowest",
            },
            section="Which model",
        ),
    )
    routine_effort: str | None = Field(
        default="none",
        description="A word on a fine move needs no deep thought: with none it is spoken in "
        "under a second, at the least cost. Lessons and answers keep the effort "
        "above.",
        json_schema_extra=_extra(
            group="service",
            choices=["", "none", "low", "medium", "high"],
            applies="now",
            label="Thinking effort for quick remarks",
            only_for=["openai"],
            choice_labels={
                "": "Same as above",
                "none": "None -- fastest",
                "low": "Low",
                "medium": "Medium",
                "high": "High",
            },
            advanced=True,
            section="Speed and cost",
        ),
    )
    price_prompt_per_m: float | None = Field(
        default=None,
        ge=0,
        description="What the provider charges per million tokens sent to it, from its price "
        "list. Only used for the running cost shown in the status strip; the app "
        "charges nothing.",
        json_schema_extra=_extra(
            group="service",
            applies="now",
            label="Price per million prompt tokens (USD)",
            only_for=["openai", "dashscope", "custom"],
            advanced=True,
            section="Speed and cost",
        ),
    )
    price_cached_per_m: float | None = Field(
        default=None,
        ge=0,
        description="The provider's price per million tokens it had already seen (cached), "
        "usually much lower. For the cost shown in the status strip.",
        json_schema_extra=_extra(
            group="service",
            applies="now",
            label="Price per million cached prompt tokens (USD)",
            only_for=["openai", "dashscope", "custom"],
            advanced=True,
            section="Speed and cost",
        ),
    )
    price_completion_per_m: float | None = Field(
        default=None,
        ge=0,
        description="The provider's price per million tokens it writes back, thinking included. "
        "For the cost shown in the status strip.",
        json_schema_extra=_extra(
            group="service",
            applies="now",
            label="Price per million answer tokens (USD)",
            only_for=["openai", "dashscope", "custom"],
            advanced=True,
            section="Speed and cost",
        ),
    )
    timeout_s: float = Field(
        default=60.0,
        ge=5.0,
        le=600.0,
        description="How long the coach waits for the model. Past this it gives up on that "
        "reply and speaks from the board instead, so a slow service never stalls "
        "the game.",
        json_schema_extra=_extra(
            group="general",
            applies="now",
            label="Timeout (seconds)",
            only_for=["openai", "dashscope", "local", "custom"],
            advanced=True,
            section="Speed and cost",
        ),
    )
    extra_body: dict[str, object] = Field(
        default_factory=dict,
        description="Extra fields sent with every request to the model, as JSON. For a service "
        "that needs something unusual; empty otherwise.",
        json_schema_extra=_extra(
            group="general",
            applies="now",
            advanced=True,
            label="Extra request fields",
            dev=True,
            section="Development",
        ),
    )


class CoachSettings(BaseModel):
    stuck_seconds: int = Field(
        default=45,
        ge=5,
        le=600,
        description="How long you may think on a move before the coach offers a nudge. With a "
        "clock it comes sooner when you are short of time.",
        json_schema_extra=_extra(
            applies="now", label="Nudge after (seconds)", section="When the coach speaks"
        ),
    )
    quiet: bool = Field(
        default=False,
        description="No verdicts, questions or nudges on its own; the coach answers when you ask.",
        json_schema_extra=_extra(
            applies="now", label="Only speak when asked", section="When the coach speaks"
        ),
    )
    persona: str = Field(
        default="warm_direct",
        description="The coach's manner: how it phrases things. The lessons and the rules "
        "are the same in every one.",
        json_schema_extra=_extra(
            applies="now",
            kind="persona",
            choices=["warm_direct", "calm_quiet", "sharp_demanding", "playful"],
            label="Manner",
            section="How the coach speaks",
        ),
    )
    verify: str = Field(
        default="",
        description="A second model reads every reply the coach drafts and only the ones that "
        "stick to the facts are spoken. Empty is off; 'same' uses the coach's own "
        "model; a provider name uses that one (slower, safer).",
        json_schema_extra=_extra(
            applies="now", label="Verifier", advanced=True, section="A second opinion"
        ),
    )
    verify_model: str | None = Field(
        default=None,
        description="The model the second opinion uses, when a provider is named above.",
        json_schema_extra=_extra(
            applies="now", label="Verifier's model", advanced=True, section="A second opinion"
        ),
    )
    summarise_every: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Every this many half-moves, what was said earlier is folded into a short "
        "summary, so the model reads less as the game goes on. 0 keeps everything.",
        json_schema_extra=_extra(
            applies="now",
            label="Summarise every (plies)",
            advanced=True,
            section="How the coach speaks",
        ),
    )
    compact_record: bool = Field(
        default=False,
        description="Sound moves nobody spoke about are shown to the model as a plain list "
        "instead of one line each, to keep what it reads short.",
        json_schema_extra=_extra(
            applies="now",
            label="Compact move record",
            advanced=True,
            section="How the coach speaks",
        ),
    )
    verbalized_sampling: bool = Field(
        default=False,
        description="The model drafts a few phrasings and one is spoken, so the coach does not "
        "say the same thing the same way twice. Off is a little quicker.",
        json_schema_extra=_extra(
            applies="now", label="Varied phrasing", advanced=True, section="How the coach speaks"
        ),
    )
    local_verdicts: bool = Field(
        default=True,
        description="A best, excellent or book move gets its short word written from the board "
        "itself: instant and free. Only good, great and brilliant moves go to the "
        "model. Off sends every move to the model, which is slower and costs more.",
        json_schema_extra=_extra(
            applies="now",
            label="Quick verdicts from the board",
            advanced=True,
            section="How the coach speaks",
        ),
    )
    inaccuracy: float = Field(
        default=0.04,
        ge=0.0,
        le=1.0,
        description="A move that costs at least this much of your winning chances is called an "
        "inaccuracy (0.04 is 4%).",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Inaccuracy threshold",
            section="What counts as a mistake",
        ),
    )
    mistake: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="A move that costs at least this much of your winning chances is called a "
        "mistake, and the coach explains it.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Mistake threshold",
            section="What counts as a mistake",
        ),
    )
    blunder: float = Field(
        default=0.28,
        ge=0.0,
        le=1.0,
        description="A move that costs at least this much of your winning chances is called a "
        "blunder.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Blunder threshold",
            section="What counts as a mistake",
        ),
    )
    decisive_gap: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="When the best move is better than the next by this much, the moment is "
        "decisive and the coach asks you to slow down before you move.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Decisive-moment gap",
            section="What counts as a mistake",
        ),
    )
    only_move_gap: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="When every other move loses at least this much, the best move is the only "
        "move.",
        json_schema_extra=_extra(
            applies="now", advanced=True, label="Only-move gap", section="What counts as a mistake"
        ),
    )
    excellent: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="A move that costs less than this is called excellent, as good as the best.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Excellent threshold",
            section="What counts as a mistake",
        ),
    )
    decided: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Beyond this winning chance (0.9 is 90%) the game is decided, and small "
        "losses no longer earn a lesson.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Game decided beyond",
            section="What counts as a mistake",
        ),
    )


class ProfileSettings(BaseModel):
    rating_low: int = Field(
        default=1000,
        ge=100,
        le=3000,
        description="The bottom of your rating where you play (chess.com, Lichess, over the "
        "board); a rough band is fine. The coach pitches what it says at it.",
        json_schema_extra=_extra(applies="now", label="Your rating, from", section="Your rating"),
    )
    rating_high: int = Field(
        default=1400,
        ge=100,
        le=3200,
        description="The top of that band.",
        json_schema_extra=_extra(applies="now", label="Your rating, to", section="Your rating"),
    )


class VoiceSettings(BaseModel):
    voice: str = Field(
        default="af_sarah",
        description="The coach's voice: Kokoro's English ones, American or British, female "
        "or male.",
        json_schema_extra=_extra(
            applies="now", kind="voice", label="Voice", section="The coach's voice"
        ),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speaking speed; 1 is normal.",
        json_schema_extra=_extra(applies="now", label="Speed", section="The coach's voice"),
    )
    lang: str = Field(
        default="en-us",
        description="The voice's accent code: en-us for the American voices, en-gb for the "
        "British.",
        json_schema_extra=_extra(
            applies="now", label="Language code", advanced=True, section="Under the hood"
        ),
    )
    model_dir: str | None = Field(
        default=None,
        description="Where the voice model files are. Empty means the app's own folder, where "
        "Set up put them.",
        json_schema_extra=_extra(
            applies="now",
            kind="path",
            label="Voice model folder",
            advanced=True,
            section="Under the hood",
        ),
    )
    tts_device: str = Field(
        default="cuda",
        description="Whether the voice model runs on the graphics card or the processor. The "
        "card is faster; if it cannot be used, the processor takes over on its own.",
        json_schema_extra=_extra(
            choices=["cuda", "cpu"],
            applies="now",
            label="Run the voice on",
            choice_labels={"cuda": "GPU (the CPU if it fails)", "cpu": "CPU"},
            advanced=True,
            section="Under the hood",
        ),
    )
    stt_device: str = Field(
        default="cuda",
        description="Whether speech recognition runs on the graphics card or the "
        "processor. On the processor it still answers in a fraction of a second.",
        json_schema_extra=_extra(
            choices=["cuda", "cpu"],
            applies="now",
            label="Run speech recognition on",
            choice_labels={"cuda": "GPU (the CPU if it fails)", "cpu": "CPU"},
            advanced=True,
            section="Under the hood",
        ),
    )
    listen: str = Field(
        default="ptt",
        description="How the coach hears you. The push-to-talk key works from any window.",
        json_schema_extra=_extra(
            choices=["off", "ptt", "vad"],
            applies="now",
            label="Listening",
            choice_labels={
                "off": "Off",
                "ptt": "While I hold the push-to-talk key",
                "vad": "Whenever I speak",
            },
            section="Hearing you",
        ),
    )
    mic_device: str | None = Field(
        default=None,
        description="The microphone the coach listens to. Empty follows the system's default "
        "input.",
        json_schema_extra=_extra(
            applies="now", kind="input_device", label="Microphone", section="Hearing you"
        ),
    )
    output_device: str | None = Field(
        default=None,
        description="Where the voice plays. Empty follows the system's default output.",
        json_schema_extra=_extra(
            applies="now", kind="output_device", label="Speaker", section="The coach's voice"
        ),
    )
    ptt_key: str = Field(
        default="ctrl_r",
        description="Held from any window: ctrl_r (the right Ctrl), alt_r, f9, a letter, or "
        "mouse_x1 / mouse_x2 (the side buttons). Empty disables it.",
        json_schema_extra=_extra(applies="now", label="Push-to-talk key", section="Hearing you"),
    )


class GpuSettings(BaseModel):
    mode: str = Field(
        default="auto",
        description="Whether the voice, speech recognition and the board reader run on an "
        "NVIDIA card. Automatic fetches what that needs once and falls back to the CPU on "
        "any trouble.",
        json_schema_extra=_extra(
            choices=["auto", "cpu", "cuda"],
            applies="restart",
            label="Graphics card",
            choice_labels={
                "auto": "Automatic -- use it when it works (recommended)",
                "cpu": "Never -- CPU only",
                "cuda": "Always",
            },
            section="Graphics card",
        ),
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
        json_schema_extra=_extra(applies="restart", kind="path", label="Data folder"),
    )
    tools_dir: Path = Field(
        default=Path("tools"),
        description="Where Stockfish is looked for when no path is set.",
        json_schema_extra=_extra(applies="next", kind="path", label="Tools folder", advanced=True),
    )
    capture: CaptureSettings = Field(default_factory=CaptureSettings, description="The screen")
    engine: EngineSettings = Field(default_factory=EngineSettings, description="Stockfish")
    llm: LLMSettings = Field(default_factory=LLMSettings, description="The model")
    coach: CoachSettings = Field(default_factory=CoachSettings, description="The coach")
    profile: ProfileSettings = Field(default_factory=ProfileSettings, description="You")
    voice: VoiceSettings = Field(
        default_factory=VoiceSettings, description="Voice and speech recognition"
    )
    gpu: GpuSettings = Field(default_factory=GpuSettings, description="GPU acceleration")

    def voice_model_dir(self) -> Path:
        """Where the Kokoro model files live: the configured path, else under data_dir."""
        if self.voice.model_dir:
            return Path(self.voice.model_dir)
        return self.data_dir / "models" / "kokoro"


# The environment variables the presets read their keys from (sidecoach.llm.factory).
KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "dashscope": "DASHSCOPE_API_KEY",
    "custom": "CUSTOM_API_KEY",
}
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
