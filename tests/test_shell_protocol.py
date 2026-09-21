"""The sidecar side of the desktop shell (docs/app-plan.md, section 2): the health route,
the bearer token, the page's perf log, the in-memory frames with their on-demand dump, a
session that persists nothing, and the lock file a second launch attaches through."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui import server
from sidecoach.ui.panel import FrameRing, PanelEngine
from sidecoach.ui.server import build_app
from tests.synth import game_frames


def _settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, voice=VoiceSettings(provider="null", stt_provider="null"))


def test_health_perf_and_the_token(tmp_path: Path) -> None:
    engine = PanelEngine(_settings(tmp_path), analysis=False, coach=False, frame_source=iter(()))
    client = TestClient(build_app(engine, token="s3cret"))
    health = client.get("/api/health").json()
    assert health["ok"] is True and health["running"] is False and health["pid"] > 0
    # Everything else wants the bearer.
    assert client.get("/api/poll").status_code == 401
    assert client.post("/api/start").status_code == 401
    auth = {"Authorization": "Bearer s3cret"}
    assert set(client.get("/api/poll", headers=auth).json()) == {"state", "lines", "cursor"}
    # The page's meter lands in data/perf/<label>.jsonl, the label kept to a safe name.
    sample = {"host": "tauri", "fps": 143.9, "long_frames": 2, "poll_p95": 6}
    res = client.post("/api/perf", json={"label": "tauri idle/1", "sample": sample}, headers=auth)
    assert res.json() == {"ok": True}
    line = json.loads((tmp_path / "perf" / "tauriidle1.jsonl").read_text(encoding="utf-8"))
    assert line["fps"] == 143.9 and line["host"] == "tauri" and line["t"] > 0
    # The static page is not behind the token; the shell's origins are allowed by CORS.
    assert client.get("/").status_code == 200
    pre = client.options(
        "/api/poll",
        headers={"Origin": "http://tauri.localhost", "Access-Control-Request-Method": "GET"},
    )
    assert pre.headers.get("access-control-allow-origin") == "http://tauri.localhost"


def test_a_session_that_persists_nothing_keeps_its_frames_in_memory(tmp_path: Path) -> None:
    engine = PanelEngine(
        _settings(tmp_path),
        analysis=False,
        coach=False,
        persist=False,
        frame_source=game_frames(),
    )
    client = TestClient(build_app(engine))
    assert client.post("/api/start").json()["ok"] is True
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive()
    state = client.get("/api/poll").json()["state"]
    assert state["session_dir"] is None and state["ply"] >= 8
    # No session folder, no debug.jsonl, no recording anywhere under the data directory.
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert not any(p.suffix in {".jsonl", ".mkv", ".avi", ".mp4"} for p in written), written
    assert not (tmp_path / "sessions").exists() and not (tmp_path / "recordings").exists()
    # The frames are in memory, half scale as JPEGs, and reach the disk only on request.
    assert len(engine.frames) > 0
    res = client.post("/api/dump").json()
    folder = Path(res["path"])
    assert res["ok"] and folder.parent == tmp_path / "dumps"
    jpgs = sorted(folder.glob("*.jpg"))
    assert len(jpgs) == len(engine.frames) and (folder / "debug.jsonl").exists()
    lines = (folder / "debug.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("engine started (nothing saved)" in ln for ln in lines)


def test_the_frame_ring_bounds_itself() -> None:
    ring = FrameRing(maxlen=3)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    for ts in range(5):
        ring.push(float(ts), frame)
    assert len(ring) == 3


def test_a_second_launch_attaches_through_the_lock_file(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert server._attached_port(settings, "127.0.0.1") is None
    # A stale lock (a dead pid) is removed.
    lock = tmp_path / server.LOCK_FILE
    lock.write_text(json.dumps({"port": 1, "pid": 2**22 + 12345}), encoding="utf-8")
    assert server._attached_port(settings, "127.0.0.1") is None and not lock.exists()
    # A live lock whose port answers /api/health with a 200 is attached to.
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve() -> None:
        conn, _ = listener.accept()
        conn.recv(256)
        conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    import os

    lock.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
    assert server._attached_port(settings, "127.0.0.1") == port
    listener.close()
