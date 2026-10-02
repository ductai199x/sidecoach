"""The panel's FastAPI backend, exercised over real HTTP with the test client.

The controls map to the engine and the poll returns its state and terminal lines as JSON;
the static page and its routes are served. The game a session follows is the engine's
(test_panel_engine.py); the browser is driven separately with chrome-devtools.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_controls_and_static_page(make_panel: Callable[..., PanelEngine]) -> None:
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))

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


def test_mute_and_the_listening_mode_are_kept_for_the_next_session(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """Tai, 2026-10-01: "mute setting isn't remembered between sessions". Muted, or a way of
    listening picked, on the panel is saved with the settings, and the next panel starts so;
    nothing else in the file moves."""
    from sidecoach.config import Settings, load_settings

    path = tmp_path / "sidecoach.toml"
    first = Settings(data_dir=tmp_path)
    first.profile.chesscom_user = "nosegazm"
    first._source = path
    engine = make_panel(first)
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.get("/api/poll").json()["state"]["muted"] is False
    assert client.post("/api/muted", json={"muted": True}).json()["ok"] is True
    assert client.post("/api/listen", json={"mode": "off"}).json() == {"ok": True, "mode": "off"}
    saved = load_settings(path)
    assert saved.voice.muted is True and saved.voice.listen == "off"
    assert saved.profile.chesscom_user == "nosegazm", "the rest of the file is kept"
    # The next session starts muted, listening as it was left.
    again = make_panel(saved)
    state = (
        TestClient(base_url="http://127.0.0.1", app=build_app(again))
        .get("/api/poll")
        .json()["state"]
    )
    assert again.muted is True and state["muted"] is True and state["listen"] == "off"
    # Settings > Voice's switch is the same setting, applied at once.
    put = TestClient(base_url="http://127.0.0.1", app=build_app(again)).put(
        "/api/settings", json={"settings": {"voice": {"muted": False}}}
    )
    assert put.json()["ok"] and again.muted is False
    assert load_settings(path).voice.muted is False


def test_the_captions_switch_from_the_panel_follows_the_box(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """The header's CC button and its menu save the captions' two settings; the box turned
    off from itself (dragged onto its cross) is the same setting, so the button follows it
    (Tai, 2026-10-01: "if I dismiss the box then the button state should reflect that")."""
    from sidecoach.config import Settings

    settings = Settings(data_dir=tmp_path)
    settings._source = tmp_path / "sidecoach.toml"
    engine = make_panel(settings)
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))

    def captions() -> dict[str, object]:
        return client.get("/api/poll").json()["state"]["captions"]

    assert captions()["on"] is False
    on = {"settings": {"voice": {"captions": True, "captions_keep": True}}, "secrets": {}}
    assert client.put("/api/settings", json=on).json()["ok"]
    first = captions()
    assert first["on"] is True and first["keep"] is True and first["preview"], "a sample line"
    # Dismissed from the box: the captions' window sends the one change.
    off = {"settings": {"voice": {"captions": False}}, "secrets": {}}
    assert client.put("/api/settings", json=off).json()["ok"]
    assert captions()["on"] is False and captions()["keep"] is True
    # Pressed again: back, kept on screen as before.
    back = {"settings": {"voice": {"captions": True}}, "secrets": {}}
    assert client.put("/api/settings", json=back).json()["ok"]
    assert captions()["on"] is True and captions()["keep"] is True


def test_every_change_to_the_settings_raises_the_counter_the_main_window_reads(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """The main window reads `settings_version` with each poll and, when it moves, draws the
    settings again and tells the other windows: a change made by any way reaches them, with no
    window asking on its own (2026-10-01: the Neo set loaded through the API left every window
    drawing the old pieces)."""
    from sidecoach.config import Settings

    settings = Settings(data_dir=tmp_path)
    settings._source = tmp_path / "sidecoach.toml"
    client = TestClient(base_url="http://127.0.0.1", app=build_app(make_panel(settings)))

    def version() -> int:
        return client.get("/api/poll").json()["state"]["settings_version"]

    start = version()
    pick = {"settings": {"board": {"pieces": "cburnett"}}, "secrets": {}}
    assert client.put("/api/settings", json=pick).json()["ok"]
    after_save = version()
    assert after_save > start, "a save"
    client.post("/api/muted", json={"muted": True})
    after_mute = version()
    assert after_mute > after_save, "a switch kept for the next session"
    png = "iVBORw0KGgoAAAAAAAAAAAAAAAAAAAAA"
    one = {"name": "Neo", "files": [{"name": "wK.png", "data": png}]}
    made = client.post("/api/pieces/own", json=one)
    assert made.json()["ok"] and version() > after_mute, "a piece set made"
    before = version()
    assert not client.put("/api/pieces/own/nope", json={"name": "x"}).json()["ok"]
    assert version() == before, "nothing changed, nothing raised"
