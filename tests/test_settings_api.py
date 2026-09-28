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
        Settings(data_dir=tmp_path, voice=VoiceSettings()),
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
    assert listen["description"].startswith("How the coach hears you")
    # As a person reads it: a label, human choice labels, the heading it sits under.
    assert listen["label"] == "Listening" and listen["section"] == "Hearing you"
    assert listen["choice_labels"]["ptt"].startswith("While I hold")
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
    # The voice applies at once, and so does the coach's stuck time: no coach runs before
    # Start, and the one that starts is built from these settings ("at the next Start"
    # for that read as a restart, Tai 2026-09-21). The screen's backend is built at
    # Start and says so. Nothing here needs a restart.
    assert res["ok"] and res["now"] == ["coach.stuck_seconds", "voice.voice"]
    assert res["next"] == ["capture.backend"] and res["restart"] == []
    written = tomllib.loads((tmp_path / "sidecoach.toml").read_text(encoding="utf-8"))
    assert written["coach"]["stuck_seconds"] == 60 and written["capture"]["backend"] == "mss"
    assert engine.settings.coach.stuck_seconds == 60 and engine.voice == "am_adam"
    # A bad value is refused by field, and nothing is written.
    before = (tmp_path / "sidecoach.toml").read_text(encoding="utf-8")
    values["engine"]["depth"] = 999
    res = client.put("/api/settings", json={"settings": values}).json()
    assert res["ok"] is False and res["errors"][0]["loc"] == ["engine", "depth"]
    assert (tmp_path / "sidecoach.toml").read_text(encoding="utf-8") == before


def test_a_save_goes_back_to_the_file_the_settings_came_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The panel started with `--config x.toml` saved to `sidecoach.toml` in the working
    directory (Tai's development config, 2026-09-21): the settings remember their file."""
    from sidecoach.config import load_settings

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SIDECOACH_CONFIG", raising=False)
    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    given = tmp_path / "given.toml"
    given.write_text("[coach]\nstuck_seconds = 50\n", encoding="utf-8")
    settings = load_settings(given)
    settings.data_dir = tmp_path
    assert settings.source == given and settings.coach.stuck_seconds == 50, "the file was read"
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    client = TestClient(build_app(engine))
    assert client.get("/api/settings").json()["path"] == str(given)
    values = client.get("/api/settings").json()["settings"]
    values["coach"]["stuck_seconds"] = 61
    assert client.put("/api/settings", json={"settings": values}).json()["ok"]
    assert tomllib.loads(given.read_text(encoding="utf-8"))["coach"]["stuck_seconds"] == 61
    assert not (tmp_path / "sidecoach.toml").exists(), "nothing in the working directory"
    assert engine.settings.source == given, "and the next save goes there too"


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


def test_a_save_reaches_the_running_coach_and_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-21: as few settings as possible wait for a restart. With a coach and
    an analyser running, a save changes them in place: the persona and the rating band in
    the system message, the nudge time, the thresholds, the engine's depth, the model."""
    from types import SimpleNamespace

    from sidecoach.analysis.score import Thresholds
    from sidecoach.coach.coach import Coach
    from sidecoach.llm.offline import OfflineProvider

    client, engine = _client(tmp_path, monkeypatch)
    coach = Coach(OfflineProvider())
    engine._coaching = SimpleNamespace(coach=coach, detail="offline", reprovision=lambda s: None)
    annotator = SimpleNamespace(
        depth=18, multipv=6, thresholds=Thresholds(), engine=SimpleNamespace(multipv=6)
    )
    engine._assessing = SimpleNamespace(analyser=SimpleNamespace(annotator=annotator))
    values = client.get("/api/settings").json()["settings"]
    values["coach"]["persona"] = "playful"
    values["coach"]["stuck_seconds"] = 30
    values["coach"]["blunder"] = 0.4
    values["engine"]["depth"] = 12
    values["engine"]["multipv"] = 3
    values["profile"]["rating_low"] = 1500
    values["profile"]["rating_high"] = 1700
    res = client.put("/api/settings", json={"settings": values}).json()
    assert res["ok"] and res["next"] == [] and res["restart"] == []
    assert set(res["now"]) == {
        "coach.persona",
        "coach.stuck_seconds",
        "coach.blunder",
        "engine.depth",
        "engine.multipv",
        "profile.rating_low",
        "profile.rating_high",
    }
    assert "playful" in coach.system and "1500 to 1700" in coach.system
    assert coach.policy.settings.stuck_seconds == 30
    assert annotator.thresholds.blunder == 0.4
    assert annotator.depth == 12 and annotator.multipv == 3 and annotator.engine.multipv == 3
