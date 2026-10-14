"""Build the configured LLM provider. Any OpenAI-compatible endpoint works: a preset name
fills in the endpoint, the model, the key's environment variable and how reasoning is
controlled; any other name is a custom provider and needs `base_url` and `model` in the
settings. Secrets live in the environment, never in the settings file.

  [llm]
  provider = "openai"                 # local | dashscope | openai | <anything else>
  model = "gpt-5.6-luna"              # overrides the preset's
  reasoning_effort = "medium"         # "effort" providers: none | low | medium | high | ...
  # a custom endpoint:
  provider = "groq"
  base_url = "https://api.groq.com/openai/v1"
  model = "llama-3.3-70b-versatile"
  api_key_env = "GROQ_API_KEY"
  thinking_control = "none"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from sidecoach.config import LLMSettings
from sidecoach.llm.openai_compat import (
    THINKING_CONTROLS,
    OpenAICompatProvider,
    ThinkingControl,
    Wire,
)


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str
    api_key_env: str | None
    thinking_control: ThinkingControl
    reasoning_effort: str | None = None
    # The API spoken (openai_compat.Wire): chat completions, or OpenAI's responses.
    wire: Wire = "chat"
    # An env var that overrides base_url: a hosted provider's endpoint can be per account.
    base_url_env: str | None = None


PROVIDER_DEFAULTS: dict[str, ProviderDefaults] = {
    # LM Studio on this machine. No key. Reasoning suppressed by prefill.
    "local": ProviderDefaults(
        base_url="http://localhost:1234/v1",
        model="huihui-qwen3.8-27b-abliterated@q8_k_l",
        api_key_env=None,
        thinking_control="prefill",
    ),
    # Alibaba Cloud Model Studio, US region. qwen3.8-flash is the faster, cheaper option.
    "dashscope": ProviderDefaults(
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        model="qwen3.8-max",
        api_key_env="DASHSCOPE_API_KEY",
        thinking_control="param",
        base_url_env="DASHSCOPE_BASE_URL",
    ),
    # OpenAI's reasoning models: the effort is the model's setting, not the call's. This is
    # what the coach was evaluated on (docs/eval-results.md).
    "openai": ProviderDefaults(
        base_url="https://api.openai.com/v1",
        model="gpt-5.6-luna",
        api_key_env="OPENAI_API_KEY",
        thinking_control="effort",
        reasoning_effort="medium",
        # Its reasoning models take function tools with an effort only on /v1/responses.
        wire="responses",
        base_url_env="OPENAI_BASE_URL",
    ),
}


# The presets served by a company, at an address of their own (and OPENAI_BASE_URL or
# DASHSCOPE_BASE_URL): llm.base_url is not theirs to take.
HOSTED = frozenset({"openai", "dashscope"})


def _key_env(settings: LLMSettings, defaults: ProviderDefaults | None) -> str | None:
    """The variable the key is read from: the settings', else the preset's; a custom service
    keeps its key under CUSTOM_API_KEY unless the settings name another ("": none)."""
    if defaults is None:
        return settings.api_key_env if settings.api_key_env is not None else "CUSTOM_API_KEY"
    # A preset's key is its own: a custom service's variable left in the settings sent no
    # key, or the wrong one, to OpenAI (the API review, 2026-10-01).
    return defaults.api_key_env


def key_name(settings: LLMSettings) -> str | None:
    """The variable a provider made from `settings` reads its key from; None for none."""
    if settings.provider == "none":
        return None
    return _key_env(settings, PROVIDER_DEFAULTS.get(settings.provider))


def api_key_of(settings: LLMSettings) -> str | None:
    """The key a provider made from `settings` now would carry, or None: for a model kept and
    made again when the key changes (Game Review's kept the old one until a restart, the
    settings review 2026-10-01)."""
    if settings.provider == "none":
        return None
    name = _key_env(settings, PROVIDER_DEFAULTS.get(settings.provider))
    return os.environ.get(name) if name else None


def make_provider(settings: LLMSettings, *, timeout_s: float | None = None) -> Any:
    """The coach's model: a preset, a custom endpoint, or none at all (llm/offline.py),
    where the coach speaks from the board and its templates."""
    if settings.provider == "none":
        from sidecoach.llm.offline import OfflineProvider

        return OfflineProvider()
    defaults = PROVIDER_DEFAULTS.get(settings.provider)
    if defaults is None:
        # A custom endpoint: everything comes from the settings.
        if not settings.base_url or not settings.model:
            known = ", ".join(sorted(PROVIDER_DEFAULTS))
            raise ValueError(
                f"unknown llm.provider {settings.provider!r}: use one of {known}, or set "
                "llm.base_url and llm.model for a custom OpenAI-compatible endpoint"
            )
        defaults = ProviderDefaults(
            base_url=settings.base_url,
            model=settings.model,
            # The app stores a custom service's key under CUSTOM_API_KEY; a service that
            # takes no key at all is the empty string.
            api_key_env=settings.api_key_env
            if settings.api_key_env is not None
            else "CUSTOM_API_KEY",
            thinking_control="none",
        )

    key_env = _key_env(settings, defaults)
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(f"{key_env} is not set: paste the key in Settings.")

    control = settings.thinking_control or defaults.thinking_control
    if control not in THINKING_CONTROLS:
        raise ValueError(
            f"llm.thinking_control must be one of {THINKING_CONTROLS}, got {control!r}"
        )
    effort = settings.reasoning_effort or defaults.reasoning_effort

    env_url = os.environ.get(defaults.base_url_env) if defaults.base_url_env else None
    # The server address is the local server's or a custom service's (config: only_for):
    # a hosted preset's is its own, or the key went over plain HTTP to an old LAN address.
    own_url = settings.base_url if settings.provider not in HOSTED else None
    return OpenAICompatProvider(
        settings.provider,
        own_url or env_url or defaults.base_url,
        settings.model or defaults.model,
        api_key=api_key,
        thinking_control=control,  # type: ignore[arg-type]
        reasoning_effort=effort,
        wire=defaults.wire,
        timeout_s=timeout_s if timeout_s is not None else settings.timeout_s,
        extra_body=dict(settings.extra_body),
    )


def provider_from_spec(spec: str, *, timeout_s: float | None = None) -> OpenAICompatProvider:
    """A provider from a short spec, for command lines: `local`, `dashscope`, `flash`
    (dashscope's qwen3.8-flash), `openai`, `openai:<model>`, `openai:<model>@<effort>`,
    or `<provider>:<model>` for any preset."""
    if spec == "flash":
        return make_provider(
            LLMSettings(provider="dashscope", model="qwen3.8-flash"), timeout_s=timeout_s
        )
    provider, _, rest = spec.partition(":")
    model, _, effort = rest.partition("@")
    return make_provider(
        LLMSettings(provider=provider, model=model or None, reasoning_effort=effort or None),
        timeout_s=timeout_s,
    )
