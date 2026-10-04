import pytest

from sidecoach.config import LLMSettings
from sidecoach.llm.factory import make_provider
from sidecoach.llm.offline import OfflineProvider


def test_the_default_is_luna_an_effort_provider_and_local_is_a_preset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    provider = make_provider(LLMSettings())
    assert provider.name == "openai" and provider.model == "gpt-5.6-luna"
    assert provider.base_url == "https://api.openai.com/v1"
    # No thinking by default: the wait is what the player feels (Tai, 2026-10-04).
    assert provider.thinking_control == "effort" and provider.reasoning_effort == "none"
    named = make_provider(LLMSettings(provider="openai"))

    def preset(p):  # noqa: ANN001, ANN202
        return (p.name, p.model, p.base_url, p.thinking_control, p.reasoning_effort, p.wire)

    assert preset(named) == preset(provider), "the default is the openai preset"
    # The effort is the provider's: another effort is another provider of the same model.
    deeper = provider.with_effort("medium")
    assert deeper is not provider and deeper.reasoning_effort == "medium"
    assert deeper.model == provider.model and deeper.base_url == provider.base_url
    # OpenAI's reasoning models take tools with an effort only on /v1/responses; every other
    # preset speaks chat completions.
    assert provider.wire == "responses" and deeper.wire == "responses"
    assert provider.with_effort("none") is provider
    local = make_provider(LLMSettings(provider="local"))
    assert local.name == "local"
    assert local.base_url == "http://localhost:1234/v1"
    assert local.thinking_control == "prefill" and local.wire == "chat"


def test_dashscope_defaults_and_what_the_environment_and_settings_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)
    provider = make_provider(LLMSettings(provider="dashscope"))
    assert provider.base_url == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert provider.model == "qwen3.8-max"
    assert provider.thinking_control == "param"
    # The model and the reasoning control from the settings.
    settings = LLMSettings(provider="dashscope", model="qwen3.8-flash", thinking_control="none")
    provider = make_provider(settings)
    assert provider.model == "qwen3.8-flash" and provider.thinking_control == "none"
    # The account's endpoint is set beside the key. A settings base_url is the local server's
    # or a custom service's, never a hosted preset's: one left from before is not used.
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://account.example/compatible-mode/v1")
    provider = make_provider(LLMSettings(provider="dashscope"))
    assert provider.base_url == "https://account.example/compatible-mode/v1"
    provider = make_provider(LLMSettings(provider="dashscope", base_url="http://192.168.1.20/v1"))
    assert provider.base_url == "https://account.example/compatible-mode/v1"


def test_a_switch_of_service_leaves_nothing_of_the_old_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    # What a custom service left in the settings: its address, its key variable.
    left = LLMSettings(
        provider="openai", base_url="http://192.168.1.20:1234/v1", api_key_env="CUSTOM_API_KEY"
    )
    provider = make_provider(left)
    assert provider.base_url == "https://api.openai.com/v1"
    assert provider._api_key == "sk-openai"
    # The local server keeps its own address.
    local = make_provider(LLMSettings(provider="local", base_url="http://192.168.1.20:1234/v1"))
    assert local.base_url == "http://192.168.1.20:1234/v1"


def test_the_factory_gives_the_offline_provider_and_it_reports_itself() -> None:
    """No model at all (llm.provider = none, llm/offline.py): the coach's templates speak,
    free and offline (Tai, 2026-09-21)."""
    provider = make_provider(LLMSettings(provider="none"))
    assert isinstance(provider, OfflineProvider) and provider.offline
    ok, detail = provider.healthcheck()
    assert ok and "offline" in detail
    assert provider.with_effort("low") is provider
    assert provider.usage.calls == 0


def test_a_provider_that_cannot_be_built_says_what_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A name that is no preset is a custom endpoint only once it has one.
    with pytest.raises(ValueError, match="unknown llm.provider 'groq'.*llm.base_url and llm.model"):
        make_provider(LLMSettings(provider="groq"))
    with pytest.raises(ValueError, match="thinking_control"):
        make_provider(LLMSettings(provider="local", thinking_control="maybe"))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        make_provider(LLMSettings(provider="dashscope"))


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


def test_provider_from_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    from sidecoach.llm.factory import provider_from_spec

    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "k")
    provider = provider_from_spec("openai:gpt-5.6-luna@none")
    assert (provider.model, provider.reasoning_effort) == ("gpt-5.6-luna", "none")
    assert provider_from_spec("flash").model == "qwen3.8-flash"
    assert provider_from_spec("local").name == "local"
