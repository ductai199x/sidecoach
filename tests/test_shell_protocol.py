"""The sidecar side of the desktop shell (docs/app-plan.md, section 2): the health route,
the bearer token, the page's perf log, the in-memory frames with their on-demand dump, a
session that persists nothing, and the lock file a second launch attaches through."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sidecoach.config import Settings
from sidecoach.ui import server
from sidecoach.ui.panel import FrameRing, PanelEngine
from sidecoach.ui.server import build_app
from tests.synth import game_frames


def test_health_perf_and_the_token(make_panel: Callable[..., PanelEngine], tmp_path: Path) -> None:
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine, token="s3cret"))
    health = client.get("/api/health").json()
    assert health["ok"] is True and health["running"] is False and health["pid"] > 0
    # Everything else wants the bearer.
    assert client.get("/api/poll").status_code == 401
    assert client.post("/api/start").status_code == 401
    auth = {"Authorization": "Bearer s3cret"}
    first = client.get("/api/poll", headers=auth).json()
    assert set(first) == {"state", "lines", "cursor"}
    # The page's meter lands in data/perf/<label>.jsonl, the label kept to a safe name.
    sample = {"host": "tauri", "fps": 143.9, "long_frames": 2, "poll_p95": 6}
    res = client.post("/api/perf", json={"label": "tauri idle/1", "sample": sample}, headers=auth)
    assert res.json() == {"ok": True}
    line = json.loads((tmp_path / "perf" / "tauriidle1.jsonl").read_text(encoding="utf-8"))
    assert line["fps"] == 143.9 and line["host"] == "tauri" and line["t"] > 0
    # A page's own error lands in the terminal, on the sys channel, with the page named. The
    # poll is incremental: past its cursor, only what came after, and once.
    res = client.post(
        "/api/client_log",
        json={"text": "TypeError: x is undefined @ app.js:12", "page": "/settings"},
        headers=auth,
    )
    assert res.json() == {"ok": True}
    got = client.get("/api/poll", params={"since": first["cursor"]}, headers=auth).json()
    [said] = got["lines"]
    assert said["chan"] == "sys" and said["text"].startswith("page [/settings]")
    assert said["level"] == "error" and "TypeError" in said["text"]
    assert (
        client.get("/api/poll", params={"since": got["cursor"]}, headers=auth).json()["lines"] == []
    )
    # The captions' lean poll: the state, never a log line, even from the start.
    lean = client.get("/api/poll", params={"lines": 0}, headers=auth).json()
    assert lean["lines"] == [] and "state" in lean and lean["cursor"] == got["cursor"]
    # The static page is not behind the token; the shell's origins are allowed by CORS.
    assert client.get("/").status_code == 200
    pre = client.options(
        "/api/poll",
        headers={"Origin": "http://tauri.localhost", "Access-Control-Request-Method": "GET"},
    )
    assert pre.headers.get("access-control-allow-origin") == "http://tauri.localhost"


def test_the_shell_hands_the_token_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Not on the command line, which any local process can read.
    from typer.testing import CliRunner

    from sidecoach.cli import app

    seen: dict[str, str] = {}
    monkeypatch.setattr(server, "run_server", lambda *a, **k: seen.update(token=k["token"]))
    monkeypatch.setenv("SIDECOACH_DATA_DIR", str(tmp_path))
    env = {"SIDECOACH_PANEL_TOKEN": "from-env"}
    res = CliRunner().invoke(app, ["panel", "--no-browser"], env=env)
    assert res.exit_code == 0, res.output
    assert seen["token"] == "from-env"


def test_cross_site_pages_and_foreign_hosts_are_refused(
    make_panel: Callable[..., PanelEngine],
) -> None:
    # No token (the CLI panel): a page on another site must still change nothing.
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1:8765", app=build_app(engine))
    evil = {"Origin": "https://evil.example", "Content-Type": "text/plain"}
    for path in ("/api/reset_profile", "/api/clear_caches", "/api/dump", "/api/hint"):
        assert client.post(path, content=b"", headers=evil).status_code == 403, path
    # The key's way out: a settings PUT pointing base_url elsewhere.
    put = client.put(
        "/api/settings",
        json={"llm": {"base_url": "http://evil.example"}},
        headers={"Origin": "https://evil.example"},
    )
    assert put.status_code == 403
    assert "evil" not in (engine.settings.llm.base_url or "")
    # Sec-Fetch-Site alone (no Origin) is enough to know.
    navigated = {"Sec-Fetch-Site": "cross-site"}
    assert client.post("/api/stop", headers=navigated).status_code == 403
    # DNS rebinding: a name that resolves here is not this server.
    assert client.get("/api/settings", headers={"Host": "evil.example:8765"}).status_code == 403
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403
    # A same-origin page posting without JSON is refused too (a form, a beacon of text).
    same = {"Origin": "http://127.0.0.1:8765", "Content-Type": "text/plain"}
    assert client.post("/api/stop", content=b"{}", headers=same).status_code == 403
    # What the panel itself sends goes through: same origin with JSON, the shell, localhost,
    # and tools with no browser headers at all.
    ok = {"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"}
    assert client.post("/api/stop", json={}, headers=ok).status_code == 200
    assert client.get("/api/settings", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    shell = {"Origin": "http://tauri.localhost", "Sec-Fetch-Site": "cross-site"}
    assert client.post("/api/stop", json={}, headers=shell).status_code == 200
    assert client.get("/api/health", headers={"Host": "localhost:8765"}).status_code == 200
    assert client.get("/api/health", headers={"Host": "[::1]:8765"}).status_code == 200
    assert client.post("/api/stop").status_code == 200


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_a_session_that_persists_nothing_keeps_its_frames_in_memory(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    # A few frames are enough to keep.
    engine = make_panel(persist=False, frame_source=game_frames(("e4", "e5")))
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post("/api/start").json()["ok"] is True
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive()
    state = client.get("/api/poll").json()["state"]
    assert state["session_dir"] is None and state["ply"] == 2
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


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_shutdown_ends_the_session_before_the_process_goes(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    """The shell's quit: /api/shutdown stops the running session and the server, and
    run_server returns only after the session's own close ran (the review and profile)."""
    engines: list[PanelEngine] = []

    def make(settings: Settings, **_: Any) -> PanelEngine:
        engine = make_panel(settings, frame_source=game_frames())
        engine.warm = lambda background=True: None  # type: ignore[method-assign]
        engines.append(engine)
        return engine

    monkeypatch.setattr(server, "make_engine", make)
    settings = Settings(data_dir=tmp_path)
    done = threading.Event()

    def serve() -> None:
        server.run_server(settings, port=0, open_browser=False, token="t", announce=True)
        done.set()

    threading.Thread(target=serve, daemon=True).start()
    port = 0
    deadline = time.monotonic() + 30
    while not port and time.monotonic() < deadline:
        for line in capsys.readouterr().out.splitlines():
            if line.startswith("{"):
                port = json.loads(line)["port"]
        time.sleep(0.05)
    assert port
    base = f"http://127.0.0.1:{port}"
    auth = {"Authorization": "Bearer t"}
    assert httpx.post(f"{base}/api/start", headers=auth).json()["ok"] is True
    engine = engines[0]
    deadline = time.monotonic() + 30
    while engine.poll(0)["state"]["ply"] < 4 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert engine.running
    # Without the token the route is refused; with it the server ends.
    assert httpx.post(f"{base}/api/shutdown").status_code == 401
    assert httpx.post(f"{base}/api/shutdown", headers=auth).json() == {"ok": True}
    assert done.wait(30), "run_server did not return"
    assert not engine.running
    texts = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert "engine stopped" in texts  # the loop's own close ran to its end
    assert not (tmp_path / server.LOCK_FILE).exists()


def test_a_second_launch_attaches_through_the_lock_file(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
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


def test_the_lock_check_is_answered_by_a_server_that_refuses_foreign_hosts(
    tmp_path: Path,
) -> None:
    """The lock check asked /api/health with `Host: x`. Since the server refuses a Host that
    is not this machine (DNS rebinding), a live sidecar answered 403: its lock was taken for
    stale and deleted, and a second sidecar started on the same data directory. The fake
    answers as the server's own rule (`_refusal`) decides."""
    import os

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    asked: list[str] = []

    def serve() -> None:
        conn, _ = listener.accept()
        head = conn.recv(1024).decode("latin-1").split("\r\n")
        headers = {
            k.strip().lower(): v.strip()
            for k, _, v in (line.partition(":") for line in head[1:] if ":" in line)
        }
        asked.append(headers.get("host", ""))
        refused = server._refusal("GET", "/api/health", headers)
        status = b"403 Forbidden" if refused else b"200 OK"
        conn.sendall(b"HTTP/1.0 " + status + b"\r\nContent-Length: 2\r\n\r\n{}")
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    lock = tmp_path / server.LOCK_FILE
    lock.write_text(json.dumps({"port": port, "pid": os.getpid(), "token": ""}), encoding="utf-8")
    try:
        assert server._attached_port(Settings(data_dir=tmp_path), "127.0.0.1") == port, asked
        assert lock.exists(), "a live sidecar's lock is kept"
        assert asked == [f"127.0.0.1:{port}"]
    finally:
        listener.close()


def test_a_relaunch_never_attaches_to_a_sidecar_of_another_token(tmp_path: Path) -> None:
    """Setup finding 7: a shell relaunched soon after its predecessor died attached to the
    dying sidecar, which wants the old token: every call was 401. Only a sidecar that takes
    this launch's token is attached to; the lock keeps a digest of it, never the token."""
    import os

    settings = Settings(data_dir=tmp_path)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(4)
    port = listener.getsockname()[1]

    def serve() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.recv(256)
            conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: 2\r\n\r\n{}")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    lock = tmp_path / server.LOCK_FILE
    lock.write_text(
        json.dumps({"port": port, "pid": os.getpid(), "token": server._token_digest("old")}),
        encoding="utf-8",
    )
    try:
        assert server._attached_port(settings, "127.0.0.1", "new") is None
        assert server._attached_port(settings, "127.0.0.1", "old") == port
        assert "old" not in lock.read_text(encoding="utf-8").replace(
            server._token_digest("old"), ""
        )
        # A lock of before the digest (no token) is not attached to with one.
        lock.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
        assert server._attached_port(settings, "127.0.0.1", "new") is None
    finally:
        listener.close()
    # The wait for a dying sidecar ends when it is gone, and gives up on a live one.
    assert server._wait_for_exit(2**22 + 12345, 1.0)
    assert not server._wait_for_exit(os.getpid(), 0.3)
