"""The settings over HTTP (ui/settings_api.py): the schema with its marks, a save that
writes the TOML and says what applies when, validation errors by field, and the secrets
round trip (skipped where no credential store answers)."""

from __future__ import annotations

import os
import threading
import tomllib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sidecoach.config import Settings, secrets_get, secrets_set, to_toml
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, PanelEngine]:
    monkeypatch.setenv("SIDECOACH_CONFIG", str(tmp_path / "sidecoach.toml"))
    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    engine = PanelEngine(
        Settings(data_dir=tmp_path), analysis=False, coach=False, frame_source=iter(())
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


def test_a_save_changes_only_what_it_sends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A page sends what it changed; the rest stays as it is now: a Settings window opened
    before another saved put the chess.com username back to empty (Tai 2026-09-30)."""
    client, engine = _client(tmp_path, monkeypatch)
    assert client.put(
        "/api/settings", json={"settings": {"profile": {"chesscom_user": "nosegazm"}}}
    ).json()["ok"]
    res = client.put("/api/settings", json={"settings": {"coach": {"stuck_seconds": 60}}}).json()
    assert res["ok"] and res["now"] == ["coach.stuck_seconds"]
    written = tomllib.loads((tmp_path / "sidecoach.toml").read_text(encoding="utf-8"))
    assert written["profile"]["chesscom_user"] == "nosegazm"
    assert engine.settings.profile.chesscom_user == "nosegazm"


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


def test_same_as_above_survives_a_restart(tmp_path: Path) -> None:
    """llm.routine_effort's "Same as above" is None, and TOML has no null: dropped from the
    file, it read back as the default "none" after a restart (the parity audit, 2026-10-01).
    Written "", it reads back as unset."""
    from sidecoach.config import Settings, load_settings, save_settings

    settings = Settings(data_dir=tmp_path)
    settings.llm.routine_effort = None
    path = save_settings(settings, tmp_path / "sidecoach.toml")
    again = load_settings(path).llm.routine_effort
    assert not again and again != "none"
    plain = Settings(data_dir=tmp_path)
    plain.llm.base_url = None  # a None that is the default stays out of the file
    written = save_settings(plain, tmp_path / "plain.toml").read_text(encoding="utf-8")
    assert "base_url" not in written


def test_a_key_taken_out_of_an_object_field_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """llm.extra_body is one field: saved, it is what was sent, whole. Merged key by key, a
    key taken out stayed for good (the parity audit, 2026-10-01)."""
    client, engine = _client(tmp_path, monkeypatch)
    two = {"settings": {"llm": {"extra_body": {"top_k": 20, "seed": 7}}}, "secrets": {}}
    assert client.put("/api/settings", json=two).json()["ok"]
    one = {"settings": {"llm": {"extra_body": {"seed": 7}}}, "secrets": {}}
    assert client.put("/api/settings", json=one).json()["ok"]
    assert engine.settings.llm.extra_body == {"seed": 7}
    assert engine.settings.llm.provider == "openai", "the rest of the section is kept"


@pytest.mark.parametrize("other", ["muted", "record", "captions"])
def test_two_writes_at_once_keep_both_changes(
    other: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Settings save and a mute, the Debug menu's Record or the captions' own turn-off, at
    the same moment: each read the settings, laid its change over them and wrote, and the
    later write put back what the earlier one changed -- 81 usernames lost in 150 rounds, and
    the file left unreadable in 15 of 400 (the settings review, 2026-10-01). One write at a
    time, the whole read-merge-write: both changes stay, in the file and in the panel."""
    from sidecoach.config import load_settings
    from sidecoach.ui import settings_api

    client, engine = _client(tmp_path, monkeypatch)
    real = settings_api.save_settings
    saving, other_done = threading.Event(), threading.Event()
    calls: list[int] = []

    def slow(settings: Settings, path: Path | None = None) -> Path:
        """The Settings save holds still between its read and its write."""
        if not calls:
            calls.append(1)
            saving.set()
            other_done.wait(0.15)  # long enough for an unlocked write to land in between
        return real(settings, path)

    monkeypatch.setattr(settings_api, "save_settings", slow)
    body = {"settings": {"profile": {"chesscom_user": "nosegazm"}}, "secrets": {}}
    first = threading.Thread(target=lambda: client.put("/api/settings", json=body))
    first.start()
    assert saving.wait(5)
    if other == "muted":
        assert client.post("/api/muted", json={"muted": True}).json()["ok"]
    elif other == "record":
        assert client.post("/api/record", json={"on": True}).json()["ok"]
    else:
        on = {"settings": {"voice": {"captions": True}}, "secrets": {}}
        assert client.put("/api/settings", json=on).json()["ok"]
    other_done.set()
    first.join(5)
    section, field = {
        "muted": ("voice", "muted"),
        "record": ("capture", "record"),
        "captions": ("voice", "captions"),
    }[other]
    for settings in (load_settings(tmp_path / "sidecoach.toml"), engine.settings):
        assert settings.profile.chesscom_user == "nosegazm"
        assert getattr(getattr(settings, section), field) is True


def test_a_save_that_fails_partway_leaves_the_file_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file was emptied, then written: a save that failed (a character the encoder could
    not take) or was cut short (the app quitting) left an empty file, which reads as every
    default, silently (the settings review, 2026-10-01). Written beside it and put in its
    place in one step, the old file stays whole until the new one is."""
    from sidecoach import config
    from sidecoach.config import load_settings, save_settings

    path = tmp_path / "sidecoach.toml"
    good = Settings(data_dir=tmp_path)
    good.profile.chesscom_user = "nosegazm"
    save_settings(good, path)
    before = path.read_bytes()
    bad = good.model_copy(deep=True)
    bad.profile.lichess_user = "x\ud800"  # a lone surrogate: no file can hold it
    with pytest.raises(ValueError):
        save_settings(bad, path)
    assert path.read_bytes() == before
    # The disk failing under the write: the same.
    changed = good.model_copy(deep=True)
    changed.coach.stuck_seconds = 61

    def disk_full(fd: int) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(config.os, "fsync", disk_full)
    with pytest.raises(OSError):
        save_settings(changed, path)
    assert path.read_bytes() == before
    assert load_settings(path).profile.chesscom_user == "nosegazm"
    assert [p.name for p in tmp_path.iterdir()] == ["sidecoach.toml"], "no scratch file left"


def test_what_the_writer_writes_reads_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the backslash and the quote were escaped, and llm.extra_body's keys went in bare: a
    newline in a name, a control character, or a key with a space or a dot wrote a file the
    next start could not read, or read as something else (the settings review, 2026-10-01).
    A null in llm.extra_body, which TOML cannot hold, raised: a 500, and the page sent it
    again with every later change, so nothing else saved. It is refused by field now."""
    settings = Settings()
    settings.profile.lichess_user = 'tai\n\t"q" \\ \x1b \x7f é ♞'
    settings.llm.model = "gpt\x00x"
    settings.llm.extra_body = {
        "top p": 0.9,
        "response.format": {"type": "json", "a b": [1, "two", {"c.d": True}]},
        "": "empty key",
        'quo"te': "x",
        "ключ": 1e-05,
    }
    back = Settings.model_validate(tomllib.loads(to_toml(settings)))
    assert back == settings

    client, engine = _client(tmp_path, monkeypatch)
    null = {"settings": {"llm": {"extra_body": {"seed": None}}}, "secrets": {}}
    res = client.put("/api/settings", json=null)
    assert res.status_code == 200 and res.json()["ok"] is False
    assert res.json()["errors"][0]["loc"] == ["llm", "extra_body"]
    assert not (tmp_path / "sidecoach.toml").exists(), "nothing written"
    # The page's next change goes without the refused field, and saves.
    user = {"settings": {"profile": {"chesscom_user": "nosegazm"}}, "secrets": {}}
    assert client.put("/api/settings", json=user).json()["ok"]
    assert engine.settings.profile.chesscom_user == "nosegazm"


def test_a_settings_file_that_cannot_be_read_is_kept_not_run_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_panel: Any
) -> None:
    """A torn file stopped the app at every start; and a file the start could not read, run
    over by defaults, is lost at the next save (Tai lost his settings so). A copy of it is
    kept beside it, once, and the log says so; a field the settings refuse costs that field,
    not the file (the settings review, 2026-10-01)."""
    from sidecoach.config import EngineSettings, load_settings

    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    path = tmp_path / "sidecoach.toml"
    path.write_text('[profile]\nchesscom_user = "nosegazm', encoding="utf-8")  # cut short
    settings = load_settings(path)
    assert settings.profile.chesscom_user == ""
    copies = sorted(tmp_path.glob("sidecoach.unreadable-*.toml"))
    assert len(copies) == 1 and copies[0].read_bytes() == path.read_bytes()
    engine = make_panel(settings)
    assert any(copies[0].name in ln["text"] for ln in engine.poll(0)["lines"])
    load_settings(path)
    assert len(list(tmp_path.glob("sidecoach.unreadable-*.toml"))) == 1, "one copy a file"
    # A value out of bounds, written by hand: that field takes its default, the rest stays.
    path.write_text(
        '[engine]\ndepth = 999\n[profile]\nchesscom_user = "nosegazm"\n', encoding="utf-8"
    )
    settings = load_settings(path)
    assert settings.profile.chesscom_user == "nosegazm"
    assert settings.engine.depth == EngineSettings().depth
    assert len(list(tmp_path.glob("sidecoach.unreadable-*.toml"))) == 2


def test_record_from_the_debug_menu_is_a_settings_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Debug > Record wrote capture.record without raising the change counter the other
    windows follow (the settings review, 2026-10-01)."""
    client, engine = _client(tmp_path, monkeypatch)
    before = client.get("/api/poll").json()["state"]["settings_version"]
    assert client.post("/api/record", json={"on": True}).json()["ok"]
    assert client.get("/api/poll").json()["state"]["settings_version"] > before
    assert engine.settings.capture.record is True
