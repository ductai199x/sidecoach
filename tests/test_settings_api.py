"""The settings over HTTP (ui/settings_api.py): the schema with its marks, a save that
writes the TOML and says what applies when, validation errors by field, and the secrets
round trip (skipped where no credential store answers)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sidecoach.config import Settings, VoiceSettings, secrets_get, secrets_set, to_toml
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, PanelEngine]:
    monkeypatch.setenv("SIDECOACH_CONFIG", str(tmp_path / "sidecoach.toml"))
    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    engine = PanelEngine(
        Settings(data_dir=tmp_path, voice=VoiceSettings(provider="null", stt_provider="null")),
        analysis=False,
        coach=False,
        frame_source=iter(()),
    )
    return TestClient(build_app(engine)), engine


def test_the_schema_carries_the_marks_the_page_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _ = _client(tmp_path, monkeypatch)
    got = client.get("/api/settings").json()
    assert set(got) >= {"settings", "schema", "path", "secrets", "keyring", "devices"}
    listen = got["schema"]["$defs"]["VoiceSettings"]["properties"]["listen"]
    assert listen["choices"] == ["off", "ptt", "vad"] and listen["applies"] == "now"
    assert listen["description"].startswith("How the coach listens")
    assert got["settings"]["coach"]["stuck_seconds"] == 45
    assert "OPENAI_API_KEY" in got["secrets"]


def test_a_save_writes_the_file_and_says_what_applies_when(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, engine = _client(tmp_path, monkeypatch)
    values = client.get("/api/settings").json()["settings"]
    values["coach"]["stuck_seconds"] = 60
    values["voice"]["voice"] = "am_adam"
    values["capture"]["backend"] = "mss"
    res = client.put("/api/settings", json={"settings": values}).json()
    assert res["ok"] and res["now"] == ["voice.voice"]
    assert res["next"] == ["coach.stuck_seconds"] and res["restart"] == ["capture.backend"]
    written = tomllib.loads((tmp_path / "sidecoach.toml").read_text(encoding="utf-8"))
    assert written["coach"]["stuck_seconds"] == 60 and written["capture"]["backend"] == "mss"
    assert engine.settings.coach.stuck_seconds == 60 and engine.voice == "am_adam"
    # A bad value is refused by field, and nothing is written.
    before = (tmp_path / "sidecoach.toml").read_text(encoding="utf-8")
    values["engine"]["depth"] = 999
    res = client.put("/api/settings", json={"settings": values}).json()
    assert res["ok"] is False and res["errors"][0]["loc"] == ["engine", "depth"]
    assert (tmp_path / "sidecoach.toml").read_text(encoding="utf-8") == before


def test_the_toml_round_trips_every_field() -> None:
    settings = Settings()
    settings.llm.extra_body = {"top_k": 20, "flag": True}
    settings.engine.stockfish_path = r"C:\tools\stockfish.exe"
    back = Settings.model_validate(tomllib.loads(to_toml(settings)))
    assert back == settings


def test_secrets_round_trip_through_the_credential_store() -> None:
    if not secrets_set("SIDECOACH_TEST_SECRET", "s3cret"):
        pytest.skip("no credential store here")
    try:
        assert secrets_get("SIDECOACH_TEST_SECRET") == "s3cret"
    finally:
        secrets_set("SIDECOACH_TEST_SECRET", None)
    assert secrets_get("SIDECOACH_TEST_SECRET") is None
    assert "SIDECOACH_TEST_SECRET" not in os.environ
