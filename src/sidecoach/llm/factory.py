"""Build the configured LLM provider. Per-provider defaults live here, secrets in the env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from sidecoach.config import LLMSettings
from sidecoach.llm.openai_compat import THINKING_CONTROLS, OpenAICompatProvider, ThinkingControl


@dataclass(frozen=True)
class ProviderDefaults:
    base_url: str
    model: str
    api_key_env: str | None
    thinking_control: ThinkingControl
    # An env var that overrides base_url: a hosted provider's endpoint is per account.
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
}


def make_provider(settings: LLMSettings, *, timeout_s: float | None = None) -> OpenAICompatProvider:
    defaults = PROVIDER_DEFAULTS.get(settings.provider)
    if defaults is None:
        known = ", ".join(sorted(PROVIDER_DEFAULTS))
        raise ValueError(f"unknown llm.provider {settings.provider!r}; known: {known}")

    key_env = settings.api_key_env or defaults.api_key_env
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise RuntimeError(f"{key_env} is not set; put it in .env or the environment")

    control = settings.thinking_control or defaults.thinking_control
    if control not in THINKING_CONTROLS:
        raise ValueError(
            f"llm.thinking_control must be one of {THINKING_CONTROLS}, got {control!r}"
        )

    env_url = os.environ.get(defaults.base_url_env) if defaults.base_url_env else None
    return OpenAICompatProvider(
        settings.provider,
        settings.base_url or env_url or defaults.base_url,
        settings.model or defaults.model,
        api_key=api_key,
        thinking_control=control,  # type: ignore[arg-type]
        timeout_s=timeout_s if timeout_s is not None else settings.timeout_s,
        extra_body=dict(settings.extra_body),
    )
