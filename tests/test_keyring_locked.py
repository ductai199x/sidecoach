"""Linux keeps the API key where no dialog guards it, as Windows' Credential Manager does: the
keyring is asked once at most, at the first launch, never again.

keyring's own lookup opens the default collection first: on a desktop logged in automatically
that was a "Choose password for new keyring" or an "Authentication required" dialog the
sidecar's start waited on, two minutes, and then the start failed (Ubuntu 24.04, 2026-10-13);
and the keyring locks again at every login, so it would ask at every start. The app's keys live
in its own file (config.keys_file, this user only); a key found in the keyring is moved there,
asked for once if the keyring is locked, after the start. A fake `secretstorage` stands in for
the desktop's Secret Service."""

from __future__ import annotations

import os
import stat
import sys
import types
from pathlib import Path

import pytest

from sidecoach import config


class _Item:
    def __init__(self, secret: str, locked: bool, opens: bool = True) -> None:
        self.secret, self.locked, self.opens = secret, locked, opens
        self.unlocks = 0
        self.reads = 0
        self.deleted = False

    def is_locked(self) -> bool:
        return self.locked

    def unlock(self) -> bool:
        self.unlocks += 1
        if self.opens:
            self.locked = False
        return not self.opens

    def get_secret(self) -> bytes:
        assert not self.locked, "a locked item's secret was read"
        self.reads += 1
        return self.secret.encode()

    def delete(self) -> None:
        self.deleted = True


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[_Item]]:
    store: dict[str, list[_Item]] = {}

    class _Conn:
        def close(self) -> None:
            pass

    fake = types.ModuleType("secretstorage")
    fake.dbus_init = lambda: _Conn()  # type: ignore[attr-defined]
    fake.search_items = lambda conn, attrs: iter(  # type: ignore[attr-defined]
        store.get(attrs["username"], []) if attrs.get("service") == config.KEYRING_SERVICE else []
    )
    monkeypatch.setitem(sys.modules, "secretstorage", fake)
    monkeypatch.setattr(config, "_secret_service", lambda: True)
    monkeypatch.setenv("SIDECOACH_CONFIG", str(tmp_path / "sidecoach.toml"))
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "CUSTOM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return store


def test_the_start_reads_an_open_keyring_and_asks_nothing_of_a_locked_one(
    service: dict[str, list[_Item]],
) -> None:
    open_item = _Item("sk-open", locked=False)
    service["DASHSCOPE_API_KEY"] = [open_item]
    locked = _Item("sk-locked", locked=True)
    service["OPENAI_API_KEY"] = [locked]
    assert config.secrets_get("DASHSCOPE_API_KEY") == "sk-open"
    assert config.secrets_get("OPENAI_API_KEY") is None
    assert locked.unlocks == 0
    # Stored all the same: Settings says a key is there.
    assert config.secrets_locked("OPENAI_API_KEY") and config.secrets_stored("OPENAI_API_KEY")
    assert not config.secrets_locked("DASHSCOPE_API_KEY")
    assert not config.secrets_stored("CUSTOM_API_KEY")


def test_a_locked_key_is_asked_for_once_and_never_again(service: dict[str, list[_Item]]) -> None:
    locked = _Item("sk-locked", locked=True)
    service["OPENAI_API_KEY"] = [locked]
    assert config.unlock_secret("OPENAI_API_KEY") is True
    assert os.environ["OPENAI_API_KEY"] == "sk-locked" and locked.unlocks == 1
    # The next login locks the keyring again; the next start asks nothing, reads nothing of it.
    del os.environ["OPENAI_API_KEY"]
    locked.locked = True
    reads = locked.reads
    assert config.secrets_get("OPENAI_API_KEY") == "sk-locked"
    assert not config.secrets_locked("OPENAI_API_KEY")
    assert config.unlock_secret("OPENAI_API_KEY") is False
    assert locked.unlocks == 1 and locked.reads == reads


def test_an_open_keyrings_key_is_moved_without_a_word(service: dict[str, list[_Item]]) -> None:
    item = _Item("sk-open", locked=False)
    service["OPENAI_API_KEY"] = [item]
    assert config.secrets_get("OPENAI_API_KEY") == "sk-open"
    item.locked = True  # the next login
    assert config.secrets_get("OPENAI_API_KEY") == "sk-open" and item.unlocks == 0


def test_a_dismissed_unlock_leaves_no_key(service: dict[str, list[_Item]]) -> None:
    service["OPENAI_API_KEY"] = [_Item("sk-locked", locked=True, opens=False)]
    assert config.unlock_secret("OPENAI_API_KEY") is False
    assert "OPENAI_API_KEY" not in os.environ
    assert not config.keys_file().exists()


def test_nothing_stored_nothing_asked(service: dict[str, list[_Item]]) -> None:
    """No key anywhere (a first run): no dialog at all, the start's or later."""
    assert config.unlock_secret("OPENAI_API_KEY") is False


def test_a_key_saved_in_settings_needs_no_keyring(service: dict[str, list[_Item]]) -> None:
    """Saving a key asks for no keyring password: it goes to the keys file, this user's only."""
    assert config.secrets_set("OPENAI_API_KEY", "sk-new") is True
    assert config.secrets_get("OPENAI_API_KEY") == "sk-new"
    path = config.keys_file()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # Deleted, it is gone from both, the keyring's open copy too.
    old = _Item("sk-old", locked=False)
    service["OPENAI_API_KEY"] = [old]
    assert config.secrets_set("OPENAI_API_KEY", None) is True
    assert old.deleted and "OPENAI_API_KEY" not in config._keys_read()


def test_no_secret_service_is_no_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = types.ModuleType("secretstorage")

    def no_bus() -> None:
        raise RuntimeError("no session bus")

    fake.dbus_init = no_bus  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "secretstorage", fake)
    monkeypatch.setattr(config, "_secret_service", lambda: True)
    monkeypatch.setenv("SIDECOACH_CONFIG", str(tmp_path / "sidecoach.toml"))
    assert config.secrets_get("OPENAI_API_KEY") is None
    assert not config.secrets_locked("OPENAI_API_KEY")
