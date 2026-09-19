import pytest

from sidecoach.config import LLMSettings
from sidecoach.llm.factory import make_provider


def test_the_default_is_luna_and_local_is_a_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    provider = make_provider(LLMSettings())
    assert provider.name == "openai" and provider.model == "gpt-5.6-luna"
    assert provider.thinking_control == "effort" and provider.reasoning_effort == "medium"
    local = make_provider(LLMSettings(provider="local"))
    assert local.name == "local"
    assert local.base_url == "http://localhost:1234/v1"
    assert local.thinking_control == "prefill"


def test_dashscope_defaults_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    provider = make_provider(LLMSettings(provider="dashscope"))
    assert provider.base_url == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert provider.model == "qwen3.8-max"
    assert provider.thinking_control == "param"


def test_dashscope_endpoint_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The account's endpoint is set beside the key; a settings base_url still wins."""
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://account.example/compatible-mode/v1")
    provider = make_provider(LLMSettings(provider="dashscope"))
    assert provider.base_url == "https://account.example/compatible-mode/v1"
    provider = make_provider(LLMSettings(provider="dashscope", base_url="https://x.example/v1"))
    assert provider.base_url == "https://x.example/v1"


def test_dashscope_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        make_provider(LLMSettings(provider="dashscope"))


def test_overrides_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    settings = LLMSettings(provider="dashscope", model="qwen3.8-flash", thinking_control="none")
    provider = make_provider(settings)
    assert provider.model == "qwen3.8-flash"
    assert provider.thinking_control == "none"


def test_unknown_provider_and_control() -> None:
    with pytest.raises(ValueError, match="unknown llm.provider"):
        make_provider(LLMSettings(provider="nope"))
    with pytest.raises(ValueError, match="thinking_control"):
        make_provider(LLMSettings(provider="local", thinking_control="maybe"))


def test_openai_preset_is_an_effort_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    provider = make_provider(LLMSettings(provider="openai"))
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider.thinking_control == "effort"
    assert provider.reasoning_effort == "medium"
    quiet = provider.with_effort("none")
    assert quiet is not provider and quiet.reasoning_effort == "none"
    assert quiet.model == provider.model and quiet.base_url == provider.base_url
    assert provider.with_effort("medium") is provider


def test_custom_endpoint_needs_only_base_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k")
    provider = make_provider(
        LLMSettings(
            provider="groq",
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.3-70b-versatile",
            api_key_env="GROQ_API_KEY",
        )
    )
    assert provider.name == "groq"
    assert provider.thinking_control == "none"
    with pytest.raises(ValueError, match="custom OpenAI-compatible endpoint"):
        make_provider(LLMSettings(provider="groq"))


def test_provider_from_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    from sidecoach.llm.factory import provider_from_spec

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    provider = provider_from_spec("openai:gpt-5.6-luna@none")
    assert (provider.model, provider.reasoning_effort) == ("gpt-5.6-luna", "none")
    assert provider_from_spec("flash").model == "qwen3.8-flash"
    assert provider_from_spec("local").name == "local"
