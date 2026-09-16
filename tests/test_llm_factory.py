import pytest

from sidecoach.config import LLMSettings
from sidecoach.llm.factory import make_provider


def test_local_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = make_provider(LLMSettings())
    assert provider.name == "local"
    assert provider.base_url == "http://localhost:1234/v1"
    assert provider.thinking_control == "prefill"


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
        make_provider(LLMSettings(thinking_control="maybe"))
