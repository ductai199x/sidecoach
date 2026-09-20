"""The panel's FastAPI backend, exercised over real HTTP with the test client.

The controls map to the engine and the poll returns its state and terminal lines; feeding the
engine the frames of a game, the API reports the game it followed -- the whole front-to-back
path bar the browser, which chrome-devtools drives separately.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app
from tests.synth import game_frames


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, voice=VoiceSettings(provider="null", stt_provider="null"))


def test_controls_and_static_page(tmp_path: Path) -> None:
    engine = PanelEngine(_settings(tmp_path), analysis=False, coach=False, frame_source=iter(()))
    client = TestClient(build_app(engine))

    poll = client.get("/api/poll").json()
    assert set(poll) == {"state", "lines", "cursor"}
    assert poll["state"]["voice"] == "af_sarah"

    assert client.post("/api/voice", json={"voice": "am_adam"}).json() == {"ok": True}
    assert engine.voice == "am_adam"
    assert client.post("/api/muted", json={"muted": True}).json()["ok"] is True
    assert engine.muted is True
    assert client.post("/api/ask", json={"text": "why not d5"}).json()["ok"] is True
    for path in ("/api/flip", "/api/analyze", "/api/hint", "/api/start", "/api/stop"):
        assert client.post(path).json()["ok"] is True

    index = client.get("/")
    assert index.status_code == 200
    assert "sidecoach" in index.text
    assert client.get("/app.js").status_code == 200


def test_poll_reports_a_game(tmp_path: Path) -> None:
    engine = PanelEngine(
        _settings(tmp_path), analysis=False, coach=False, frame_source=game_frames()
    )
    client = TestClient(build_app(engine))
    assert client.post("/api/start").json()["ok"] is True
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive()

    data = client.get("/api/poll").json()
    assert data["state"]["fen"], "no position reported over HTTP"
    assert data["state"]["ply"] >= 8
    assert any(
        ln["chan"] == "game" and ("(you" in ln["text"] or "(opponent" in ln["text"])
        for ln in data["lines"]
    ), "no moves surfaced over HTTP"
    assert any("start position seen" in ln["text"] for ln in data["lines"])
