"""The panel backend: a local FastAPI server over the live coaching engine.

The engine (`PanelEngine`) is UI-agnostic -- it runs the same capture -> pipeline -> coach
loop as `watch`, exposes a thread-safe `poll()` and takes control commands. This wraps it in
plain HTTP: the browser polls `GET /api/poll` for state and new terminal lines, and the
controls are `POST /api/...`. Because the transport is ordinary HTTP, the page opens in any
browser (debuggable with real devtools) and the API is testable end to end with httpx.

`build_app` makes the app around an engine (a recording-fed engine in a test, a live one in
the field); `run_server` opens the browser and serves it.
"""

from __future__ import annotations

import contextlib
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from sidecoach.config import Settings
from sidecoach.ui.panel import PanelApi, PanelEngine

if TYPE_CHECKING:
    from fastapi import FastAPI

WEB = Path(__file__).parent / "web"


class _Ask(BaseModel):
    text: str


class _Voice(BaseModel):
    voice: str


class _Muted(BaseModel):
    muted: bool


def build_app(engine: PanelEngine) -> FastAPI:
    """A FastAPI app serving the panel UI and the control API over `engine`."""
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles

    api = PanelApi(engine)
    app = FastAPI(title="sidecoach panel")

    @app.get("/api/poll")
    def poll(since: int = 0) -> dict[str, Any]:
        return api.poll(since)

    @app.post("/api/start")
    def start() -> dict[str, bool]:
        return {"ok": api.start()}

    @app.post("/api/stop")
    def stop() -> dict[str, bool]:
        return {"ok": api.stop()}

    @app.post("/api/hint")
    def hint() -> dict[str, bool]:
        return {"ok": api.hint()}

    @app.post("/api/move")
    def move() -> dict[str, bool]:
        return {"ok": api.move()}

    @app.post("/api/line")
    def line() -> dict[str, bool]:
        return {"ok": api.line()}

    @app.post("/api/flip")
    def flip() -> dict[str, bool]:
        return {"ok": api.flip()}

    @app.post("/api/analyze")
    def analyze() -> dict[str, bool]:
        return {"ok": api.analyze()}

    @app.post("/api/ask")
    def ask(body: _Ask) -> dict[str, bool]:
        return {"ok": api.ask(body.text)}

    @app.post("/api/voice")
    def voice(body: _Voice) -> dict[str, bool]:
        return {"ok": api.set_voice(body.voice)}

    @app.post("/api/muted")
    def muted(body: _Muted) -> dict[str, bool]:
        return {"ok": api.set_muted(body.muted)}

    # Everything else is the static web UI; html=True serves index.html at /.
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    return app


def make_engine(
    settings: Settings,
    *,
    monitor: int | None = None,
    fps: int = 10,
    replay: Path | None = None,
) -> PanelEngine:
    """A live engine, or one fed from a recording when `replay` is given (deterministic dev)."""
    frame_source = None
    if replay is not None:
        from sidecoach.replay.recording import Recording

        def frames() -> Any:
            for i, (tick, frame) in enumerate(Recording(replay).frames()):
                yield (i, tick.ts, frame)

        frame_source = frames()
    return PanelEngine(settings, monitor=monitor, fps=fps, frame_source=frame_source)


def run_server(
    settings: Settings,
    *,
    monitor: int | None = None,
    fps: int = 10,
    host: str = "127.0.0.1",
    port: int = 8765,
    replay: Path | None = None,
    open_browser: bool = True,
) -> None:
    """Serve the panel and open it in the browser. Blocks until the server is stopped."""
    import uvicorn

    engine = make_engine(settings, monitor=monitor, fps=fps, replay=replay)
    app = build_app(engine)
    if replay is not None:
        engine.start()  # a recording plays on its own; live capture waits for the Start button
    url = f"http://{host}:{port}/"
    if open_browser:
        # A headless box has no browser; the URL still serves either way.
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
