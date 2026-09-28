"""The panel's FastAPI backend, exercised over real HTTP with the test client.

The controls map to the engine and the poll returns its state and terminal lines as JSON;
the static page and its routes are served. The game a session follows is the engine's
(test_panel_engine.py); the browser is driven separately with chrome-devtools.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, voice=VoiceSettings())


def test_controls_and_static_page(tmp_path: Path) -> None:
    engine = PanelEngine(_settings(tmp_path), analysis=False, coach=False, frame_source=iter(()))
    client = TestClient(build_app(engine))

    poll = client.get("/api/poll").json()
    assert set(poll) == {"state", "lines", "cursor"}
    assert poll["state"]["voice"] == "af_sarah" and "af_sarah" in poll["state"]["voices"]

    assert client.post("/api/voice", json={"voice": "am_adam"}).json() == {"ok": True}
    assert engine.voice == "am_adam"
    assert client.post("/api/muted", json={"muted": True}).json()["ok"] is True
    assert engine.muted is True
    assert client.post("/api/ask", json={"text": "why not d5"}).json()["ok"] is True
    for path in ("/api/flip", "/api/analyze", "/api/hint", "/api/start", "/api/stop"):
        assert client.post(path).json()["ok"] is True

    index = client.get("/")
    assert index.status_code == 200
    assert "<title>Sidecoach</title>" in index.text
    # The single-page frontend: its routes all serve the page, its assets come with it.
    assert client.get("/puzzle/abc").status_code == 200
    assert client.get("/settings").status_code == 200
    assert "_app/" in index.text
    assert client.get("/pieces/wK.svg").status_code == 200
