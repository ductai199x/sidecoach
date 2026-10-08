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
import logging
import os
import re
import shutil
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator

log = logging.getLogger(__name__)


def _extra(**kw: Any) -> dict[str, Any]:
    """The schema extras the settings page reads: applies (now | next | restart), choices
    and choice_labels, kind (path, input_device, output_device, voice, persona, color, pieces,
    board_theme, hotkey), label (as a person reads it), section (the heading a field sits under
    on its tab), only_for (the AI Coach page: the providers a field is shown for), advanced
    (shown when asked), dev (never shown in the app), kept (what the app remembers for the
    player, never a field on the page: llm.models)."""
    return dict(kw)


class _Settings(BaseModel):
    """What every section of the settings is: a number that is not finite ("inf", "nan") is
    refused, as the phone, whose settings are JSON, must refuse it. Taken, a price of "inf"
    made every cost infinite (the audit of 2026-10-02)."""

    model_config = ConfigDict(allow_inf_nan=False)


class CaptureSettings(_Settings):
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
    # Off, nothing from the screen is ever saved (Tai, 2026-09-21). An opt-in that stays
    # until turned off, and applies on the spot: a running session starts (or stops)
    # recording within a second of Save; otherwise it holds from the next Start.
    record: bool = Field(
        default=False,
        description="Save every session (the frames of the screen, the events and the log) under "
        "the data folder's recordings/, to send along with a bug report. Off, "
        "nothing from your screen is saved. Takes effect at once, and stays on until you "
        "turn it off; a recording is a few hundred MB per game.",
        json_schema_extra=_extra(
            applies="now", label="Record sessions (to show a bug)", section="Reading the screen"
        ),
    )


class EngineSettings(_Settings):
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
    # Engines of one thread each, not one engine of several threads: at a fixed depth, more
    # threads widen the search instead of speeding it -- four took 7.7 times the processor
    # time of one, and twice as long, for an answer no better (the engine audit, 2026-10-04).
    # One thread also gives the same answer every time. Three (Tai, 2026-10-04): verdicts,
    # hints and the coach's answers are as quick as with four or six, and only a position's
    # whole analysis comes 0.1 s later than with four (tools/engine/workers_compare.py).
    workers: int = Field(
        default=3,
        ge=1,
        le=16,
        description="How many positions Stockfish works on at once, one processor core each: "
        "the position you face, the best moves in it, and the coach's questions. More answers "
        "sooner and uses more of the processor.",
        json_schema_extra=_extra(
            applies="next", label="Positions at once", section="How hard Stockfish works"
        ),
    )
    hash_mb: int = Field(
        default=256,
        ge=16,
        le=8192,
        description="Memory each of those keeps for positions it has already seen, in MB. More "
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


class LLMSettings(_Settings):
    provider: str = Field(
        default="openai",
        description="The AI model the coach thinks with. OpenAI is the one the coach was "
        "tested on; a model on this computer costs nothing per game; with no model the "
        "coach still speaks, from the board and its templates.",
        json_schema_extra=_extra(
            group="provider",
            choices=["openai", "local", "dashscope", "custom", "none"],
            applies="now",
            label="Provider",
            choice_labels={
                "openai": "OpenAI (gpt-5.6-luna) — recommended",
                "dashscope": "DashScope (Alibaba Cloud Model Studio)",
                "local": "A model on this computer (LM Studio, Ollama, vLLM)",
                "custom": "Another OpenAI-compatible service",
                "none": "No model — free and offline",
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
    models: dict[str, str] = Field(
        default_factory=dict,
        description="The model chosen for each service, by service: switching to another "
        "service and back brings that service's model back.",
        json_schema_extra=_extra(group="general", applies="now", label="Models", kept=True),
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
    # None by default: the wait is what the player feels, and thinking is most of it (Tai,
    # 2026-10-04: "no reasoning should be the default for AI coach since latency is a big issue").
    reasoning_effort: str | None = Field(
        default="none",
        description="How long the model may think before it answers. None answers fastest; more "
        "is slower and costs more (medium is what the coach was evaluated at).",
        json_schema_extra=_extra(
            group="service",
            choices=["", "none", "low", "medium", "high"],
            applies="now",
            label="Thinking effort",
            only_for=["openai"],
            choice_labels={
                "": "Default (medium)",
                "none": "None — fastest",
                "low": "Low",
                "medium": "Medium",
                "high": "High — slowest",
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
                "none": "None — fastest",
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
        description="How long the coach waits for the model each time it asks. Past this it "
        "gives up on that reply and speaks from the board instead, so a slow service never "
        "stalls the game.",
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


class CoachSettings(_Settings):
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
        default=0.05,
        ge=0.0,
        le=1.0,
        description="A move that costs at least this much of your winning chances is called an "
        "inaccuracy (0.05 is 5%, as chess.com has it).",
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
        default=0.20,
        ge=0.0,
        le=1.0,
        description="A move that costs at least this much of your winning chances is called a "
        "blunder (0.20, as chess.com has it).",
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
        description="When the best move is better than the next best by this much, the moment "
        "counts as decisive: finding the move can earn Great, Stats can count it as a tactic "
        "chance, and the coach is told the move matters.",
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
        description="A side whose winning chance is at or below 1 minus this (0.9: 10% or less) "
        "counts as lost: none of its moves earns Great, and walking into a mate is judged only "
        "by what it costs.",
        json_schema_extra=_extra(
            applies="now",
            advanced=True,
            label="Game decided beyond",
            section="What counts as a mistake",
        ),
    )


class ProfileSettings(_Settings):
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
    chesscom_user: str = Field(
        default="",
        description="Your username on chess.com. Game Review brings in your recent games with "
        "it and knows which side you played. Only your public games are read, through "
        "chess.com's public data; nothing is sent to your account.",
        json_schema_extra=_extra(
            applies="now", label="Your chess.com username", section="Your accounts"
        ),
    )
    lichess_user: str = Field(
        default="",
        description="Your username on Lichess, for the same: your recent public games, and "
        "which side was yours.",
        json_schema_extra=_extra(
            applies="now", label="Your Lichess username", section="Your accounts"
        ),
    )

    @field_validator("chesscom_user", "lichess_user")
    @classmethod
    def _trimmed(cls, v: str) -> str:
        # A space after a pasted name went into the address the games are fetched from, and
        # the site knew no such player (the settings review, 2026-10-01).
        return v.strip()


class VoiceSettings(_Settings):
    voice: str = Field(
        default="af_sarah",
        description="The coach's voice: Kokoro's English ones, American or British, female "
        "or male.",
        json_schema_extra=_extra(
            applies="now", kind="voice", label="Voice", section="The coach's voice"
        ),
    )
    muted: bool = Field(
        default=False,
        description="The coach stays silent; its words still show, on the panel and in the "
        "captions. The speaker button at the top of the panel does the same, and so does M.",
        json_schema_extra=_extra(
            applies="now", label="Mute the coach's voice", section="The coach's voice"
        ),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speaking speed; 1 is normal.",
        json_schema_extra=_extra(applies="now", label="Speed", section="The coach's voice"),
    )
    pace: str = Field(
        default="follow",
        description="How the voice keeps up with your moves. Follow my pace: every word when "
        "you play slowly; quiet through a line you play fast, then a word on the move you "
        "found; a lesson on a mistake even if you have moved again; your question always "
        "answered in full. Simple: your move stops the voice, and a word that comes after "
        "your next move is only shown.",
        json_schema_extra=_extra(
            choices=["follow", "simple"],
            applies="now",
            label="Keeping up with your moves",
            choice_labels={"follow": "Follow my pace", "simple": "Simple"},
            section="The coach's voice",
        ),
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
        description="The microphone the coach listens to. System default follows the one "
        "your computer is set to use.",
        json_schema_extra=_extra(
            applies="now", kind="input_device", label="Microphone", section="Hearing you"
        ),
    )
    output_device: str | None = Field(
        default=None,
        description="Where the voice plays. System default follows the speaker your computer "
        "is set to use.",
        json_schema_extra=_extra(
            applies="now", kind="output_device", label="Speaker", section="The coach's voice"
        ),
    )
    ptt_key: str = Field(
        default="ctrl_r",
        description="Hold it from any window, the game's too, and the coach listens. A key "
        "that types nothing, like Right Ctrl, or a mouse side button is best.",
        json_schema_extra=_extra(
            applies="now", kind="hotkey", label="Push-to-talk key", section="Hearing you"
        ),
    )
    # The coach's words in a box over the other windows, the way Live Caption shows speech:
    # the phone's since 2026-09-23, the desktop's since Tai asked for it in fast games
    # (2026-09-29: "I think it's really useful in fast games").
    captions: bool = Field(
        default=False,
        description="Shows what the coach says in a small box over your other windows, like "
        "Live Caption, so you can read it while you play. Drag the box to move it; "
        "double-click it to see more; click it for its buttons; drag it onto the cross at the "
        "bottom of the screen to turn captions off. CC at the top of the panel turns them on "
        "and off too.",
        json_schema_extra=_extra(
            applies="now", label="Captions over other windows", section="Captions"
        ),
    )
    captions_keep: bool = Field(
        default=False,
        description="Keeps the caption box on screen all the time, over every window, "
        "Sidecoach's too: its buttons, and the coach's last words once it has spoken. Off, the "
        "box comes up when the coach speaks, fades a few seconds after, and steps aside while "
        "Sidecoach's own window is in front. The pin beside CC at the top of the panel does "
        "the same.",
        json_schema_extra=_extra(
            applies="now", label="Keep captions on screen", section="Captions"
        ),
    )
    captions_text_color: str = Field(
        default="#FFFFFF",
        description="The colour of the captions' words.",
        json_schema_extra=_extra(
            applies="now", kind="color", label="Text colour", section="Captions"
        ),
    )
    captions_background: str = Field(
        default="#1F1F1F",
        description="The colour of the box behind the words.",
        json_schema_extra=_extra(
            applies="now", kind="color", label="Background colour", section="Captions"
        ),
    )
    captions_opacity: int = Field(
        default=92,
        ge=20,
        le=100,
        description="How solid the box is, in percent: 100 hides what is behind it, lower lets "
        "it show through.",
        json_schema_extra=_extra(applies="now", label="Background opacity (%)", section="Captions"),
    )
    captions_text_size: int = Field(
        default=17,
        ge=12,
        le=32,
        description="How big the captions' words are; 17 is the usual size.",
        json_schema_extra=_extra(applies="now", label="Text size", section="Captions"),
    )


class BoardSettings(_Settings):
    """How the boards look (looks.py): the pieces and the squares' colours."""

    pieces: str = Field(
        default="sidecoach",
        description="The pieces on every board: Sidecoach's own, a freely licensed set, or your "
        "own images.",
        json_schema_extra=_extra(applies="now", kind="pieces", label="Pieces", section="The board"),
    )
    theme: str = Field(
        default="green",
        description="The colours of the board's squares.",
        json_schema_extra=_extra(
            applies="now", kind="board_theme", label="Board colours", section="The board"
        ),
    )

    @field_validator("pieces")
    @classmethod
    def _known_pieces(cls, v: str) -> str:
        from sidecoach.looks import PIECE_SETS, own_id

        # A shipped set, or one of the player's own ("own:<id>"; "custom", the one there was).
        if v not in PIECE_SETS and own_id(v) is None:
            raise ValueError(f"no piece set named {v!r}")
        return v

    @field_validator("theme")
    @classmethod
    def _known_theme(cls, v: str) -> str:
        from sidecoach.looks import BOARD_THEMES

        if v not in BOARD_THEMES:
            raise ValueError(f"no board theme named {v!r}")
        return v


class GpuSettings(_Settings):
    mode: str = Field(
        default="auto",
        description="Whether the voice, speech recognition and the board reader run on the "
        "NVIDIA graphics card (CUDA). Automatic fetches what that needs once and falls back "
        "to the CPU on any trouble.",
        json_schema_extra=_extra(
            choices=["auto", "cpu", "cuda"],
            applies="restart",
            label="Hardware acceleration",
            choice_labels={
                "auto": "Automatic — use it when it works (recommended)",
                "cpu": "Never — CPU only",
                "cuda": "Always",
            },
            section="Hardware acceleration",
        ),
    )


class Settings(_Settings):
    # Where these settings were read from, so a save goes back there: the panel started
    # with `--config x.toml` once saved to `sidecoach.toml` in the working directory.
    _source: Path | None = PrivateAttr(default=None)
    # What could not be read of that file at the start, and where its copy is kept, for the
    # panel's log (`load_settings`); None when it read whole.
    _unreadable: str | None = PrivateAttr(default=None)

    @property
    def source(self) -> Path:
        """The file these settings came from, or would be written to."""
        return self._source or settings_path()

    @property
    def unreadable(self) -> str | None:
        """What of the settings file the start could not read, and where its copy is."""
        return self._unreadable

    data_dir: Path = Field(
        default_factory=lambda: Path("data"),  # a factory: a Path is no JSON default
        description="Where the app keeps its data: the profile, the caches, the models. In the "
        "app this is the app's own folder and not a setting.",
        json_schema_extra=_extra(applies="restart", kind="path", label="Data folder", dev=True),
    )
    tools_dir: Path = Field(
        default_factory=lambda: Path("tools"),
        description="Where Stockfish is looked for when no path is set.",
        json_schema_extra=_extra(applies="next", kind="path", label="Tools folder", dev=True),
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
    board: BoardSettings = Field(default_factory=BoardSettings, description="The board")

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
    settings = _read_settings(candidate) if candidate.exists() else Settings()
    settings._source = candidate
    data_dir = os.environ.get("SIDECOACH_DATA_DIR")
    if data_dir:
        # In the app the shell owns the data folder (one folder named for the app), and
        # Stockfish lives under it: whatever an older file says, these two are the shell's
        # (a file that pinned a folder since moved sent Stockfish missing, 2026-09-21).
        settings.data_dir = Path(data_dir)
        settings.tools_dir = settings.data_dir / "tools"
    load_secrets(settings)
    return settings


# Defaults since replaced, by setting: the old value and the new. A file holds every field
# once the app has saved it, so a value equal to a retired default is the default it was
# saved as, never one the user chose, and it moves on with the default.
RETIRED_DEFAULTS: dict[tuple[str, str], tuple[float, float]] = {
    # chess.com's published cut-offs (Tai, 2026-10-06: "use their definition").
    ("coach", "inaccuracy"): (0.04, 0.05),
    ("coach", "blunder"): (0.28, 0.20),
}


def _retired_defaults(data: dict[str, Any]) -> None:
    """Every retired default in the file's `data` replaced by its successor
    (`RETIRED_DEFAULTS`)."""
    for (section, name), (old, new) in RETIRED_DEFAULTS.items():
        values = data.get(section)
        if isinstance(values, dict) and values.get(name) == old:
            values[name] = new


def _read_settings(path: Path) -> Settings:
    """The settings in `path`. A file the start cannot read whole is never run over in silence:
    a copy of it is kept beside it, the log says so, and what can be read of it is kept -- a
    field the settings refuse takes its default and the others stay; a file that is no TOML
    leaves every default. It stopped the app at every start, and a file run over by defaults
    is lost at the next save (Tai lost his settings; the settings review, 2026-10-01). A file
    that cannot be opened at all still raises: there is nothing to read or copy."""
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        settings = Settings()
        settings._unreadable = _keep_unreadable(path, raw, f"is not readable ({exc})")
        return settings
    _retired_defaults(data)
    refused: list[str] = []
    for _ in range(len(data) + 1):  # each round takes at least one field out
        try:
            settings = Settings.model_validate(data)
        except ValidationError as exc:
            dropped = drop_refused(data, exc)
            if not dropped:
                break
            refused.extend(dropped)
            continue
        if refused:
            why = f"has values the settings refuse, left at their defaults: {', '.join(refused)}"
            settings._unreadable = _keep_unreadable(path, raw, why)
        return settings
    settings = Settings()
    settings._unreadable = _keep_unreadable(path, raw, "is not settings Sidecoach can read")
    return settings


def drop_refused(data: dict[str, Any], exc: ValidationError) -> list[str]:
    """Take out of `data` each field (`section.field`, or a top-level one) an error names, so it
    takes its default; returns the dotted names taken."""
    dropped: list[str] = []
    for error in exc.errors(include_url=False):
        loc = [str(part) for part in error["loc"][:2]]
        if not loc:
            continue
        holder: Any = data
        for part in loc[:-1]:
            holder = holder.get(part) if isinstance(holder, dict) else None
        if isinstance(holder, dict) and loc[-1] in holder:
            del holder[loc[-1]]
            dropped.append(".".join(loc))
    return dropped


def _keep_unreadable(path: Path, raw: bytes, why: str) -> str:
    """Keep `raw`, the file at `path` the start could not read whole, beside it as
    `<name>.unreadable-<when>.toml` -- once: a start that finds the same file again keeps no
    second copy. Logged; returns the line said, for the panel's log."""
    copy: Path | None = None
    for old in sorted(path.parent.glob(f"{path.stem}.unreadable-*{path.suffix}")):
        with contextlib.suppress(OSError):
            if old.read_bytes() == raw:
                copy = old
                break
    if copy is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        copy = path.with_name(f"{path.stem}.unreadable-{stamp}{path.suffix}")
        n = 1
        while copy.exists():
            n += 1
            copy = path.with_name(f"{path.stem}.unreadable-{stamp}-{n}{path.suffix}")
        try:
            copy.write_bytes(raw)
        except OSError as exc:
            line = f"the settings file {path} {why}; no copy could be kept ({exc})"
            log.warning(line)
            return line
    line = f"the settings file {path} {why}; it is kept as {copy.name}"
    log.warning(line)
    return line


# The escapes of a TOML basic string; any other control character is written \uXXXX.
_TOML_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")


def _toml_string(text: str) -> str:
    """`text` as a TOML basic string that reads back as it was. Only the backslash and the
    quote were escaped: a newline or a control character went in raw, and the next start could
    not read the file (the settings review, 2026-10-01). A lone surrogate is no character a
    file can hold: refused."""
    out = ['"']
    for ch in text:
        code = ord(ch)
        if ch in _TOML_ESCAPES:
            out.append(_TOML_ESCAPES[ch])
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{code:04X}")
        elif 0xD800 <= code <= 0xDFFF:
            raise ValueError("a character that cannot be saved (a lone surrogate)")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _toml_key(key: str) -> str:
    """A key as TOML reads it: bare when it can be, quoted otherwise -- written bare, a space
    made the file unreadable and a dot nested the key in a table of its own."""
    return key if _BARE_KEY.fullmatch(key) else _toml_string(key)


def _toml_value(value: Any) -> str:
    if value is None:
        raise ValueError("a null cannot be saved in the settings file; leave the key out")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, dict):
        inner = ", ".join(f"{_toml_key(str(k))} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + inner + " }"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(f"cannot write {type(value).__name__} to TOML")


def unwritable(settings: Settings) -> list[dict[str, Any]]:
    """The fields the settings file cannot hold, as pydantic's errors are shaped (less the
    input, which may not even encode), so the page shows each by its field and stops sending
    it: a null inside llm.extra_body, a lone surrogate. One raised in the writer was a 500, and
    the page sent it again with every later change, so nothing else saved (the settings
    review, 2026-10-01)."""
    errors: list[dict[str, Any]] = []
    for name, value in settings.model_dump(mode="json").items():
        fields = value.items() if isinstance(value, dict) else [(None, value)]
        for field, one in fields:
            if one is None:
                continue  # a field's own null: `to_toml` leaves it out or writes ""
            try:
                _toml_value(one)
            except (TypeError, ValueError) as exc:
                loc = [name] if field is None else [name, field]
                errors.append({"type": "value_error", "loc": loc, "msg": f"Value error, {exc}"})
    return errors


def to_toml(settings: Settings) -> str:
    """The settings as TOML: the top-level scalars, then one table per section. Fields at
    their defaults are written too, so the file is a complete, readable record. TOML has no
    null: a None that is the field's default is left out, and reads back as it; one that is
    not is written "", which every reader takes as unset (llm.routine_effort's "Same as
    above" read back as "none" after a restart, the parity audit 2026-10-01)."""
    lines: list[str] = ["# Sidecoach settings. Written by the app; edit by hand if you like.", ""]
    data = settings.model_dump(mode="json")
    if not os.environ.get("SIDECOACH_DATA_DIR"):
        # The folders are the shell's in the app and are never written there; a checkout's
        # file keeps them.
        for key in ("data_dir", "tools_dir"):
            lines.append(f"{_toml_key(key)} = {_toml_value(data[key])}")
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        lines.append("")
        lines.append(f"[{_toml_key(section)}]")
        model = type(getattr(settings, section))
        for key, value in values.items():
            if value is None:
                field = model.model_fields.get(key)
                if field is None or field.default is None:
                    continue
                value = ""
            if isinstance(value, dict) and not value:
                continue
            lines.append(f"{_toml_key(key)} = {_toml_value(value)}")
    return "\n".join(lines) + "\n"


def save_settings(settings: Settings, path: str | os.PathLike[str] | None = None) -> Path:
    """Write the settings file: `path`, else where the settings came from (see `source`);
    returns where. Whole or not at all: the text is checked to read back, written to a file of
    its own beside the old one, flushed to the disk, and put in the old one's place in one
    step. Written over in place, a save that failed or was cut short (the app quitting, two
    saves at once) left the file empty or torn, and the next start read every default, or
    could not start (Tai lost his settings; the settings review, 2026-10-01)."""
    target = Path(path) if path else settings.source
    text = to_toml(settings)
    tomllib.loads(text)  # never a file the next start cannot read
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        for attempt in range(5):
            try:
                os.replace(scratch, target)
                break
            except PermissionError:
                # Windows: a reader holding the old file open (a scan, an editor) for a moment.
                if attempt == 4:
                    raise
                time.sleep(0.05)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(scratch)
        raise
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
        # Windows: the .exe from the zip. Elsewhere: the executable from the tarball
        # (stockfish-ubuntu-x86-64-avx2 and the like), never the archive or its notes.
        if os.name == "nt":
            found = sorted(settings.tools_dir.rglob("stockfish*.exe"))
        else:
            found = sorted(
                f
                for f in settings.tools_dir.rglob("stockfish-*")
                if f.is_file()
                and f.suffix not in (".tar", ".zip", ".exe")
                and os.access(f, os.X_OK)
            )
        if found:
            return found[0]
    on_path = shutil.which("stockfish")
    return Path(on_path) if on_path else None
