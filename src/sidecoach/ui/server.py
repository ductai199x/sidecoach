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

# This module needs the panel extras (`uv sync --group panel`); the CLI imports it behind an
# ImportError guard. FastAPI resolves handler annotations from the module globals, so
# `Request` is imported here, not inside `build_app`.
from fastapi import Request
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


class _Listen(BaseModel):
    mode: str


class _Ptt(BaseModel):
    down: bool


class _Correct(BaseModel):
    game_id: str
    segment: int = 0
    ply: int
    cause: str | None = None


class _End(BaseModel):
    outcome: str


class _Move(BaseModel):
    uci: str


class _Square(BaseModel):
    square: str


class _OnOff(BaseModel):
    on: bool


class _Rect(BaseModel):
    x: int
    y: int
    w: int
    h: int


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

    @app.post("/api/exclude")
    def exclude(body: _Rect) -> dict[str, bool]:
        engine.set_exclusion(body.x, body.y, body.w, body.h)
        return {"ok": True}

    @app.post("/api/end")
    def end(body: _End) -> dict[str, bool]:
        return {"ok": api.end(body.outcome)}

    # ---- a puzzle in a tab of its own (ui/puzzle_room.py) -------------------------------
    from fastapi.responses import FileResponse

    rooms = _rooms_for(engine)

    @app.get("/puzzle/{puzzle_id}")
    def puzzle_page(puzzle_id: str) -> Any:
        return FileResponse(WEB / "puzzle.html")

    @app.post("/api/puzzle/{puzzle_id}/open")
    def puzzle_open(puzzle_id: str) -> dict[str, Any]:
        room = rooms.open(puzzle_id)
        if room is None:
            return {"ok": False, "error": f"no puzzle {puzzle_id} in the store"}
        following = [pid for pid in engine.practice_ids if pid != puzzle_id]
        return {"ok": True, "next": following, "state": room.poll(0)["state"]}

    @app.get("/api/puzzle/{puzzle_id}/poll")
    def puzzle_poll(puzzle_id: str, since: int = 0) -> dict[str, Any]:
        room = rooms.get(puzzle_id)
        if room is None:
            return {"ok": False, "closed": True}
        return {"ok": True, **room.poll(since)}

    @app.post("/api/puzzle/{puzzle_id}/targets")
    def puzzle_targets(puzzle_id: str, body: _Square) -> dict[str, Any]:
        room = rooms.get(puzzle_id)
        return {"ok": room is not None, "targets": room.targets(body.square) if room else []}

    @app.post("/api/puzzle/{puzzle_id}/move")
    def puzzle_move(puzzle_id: str, body: _Move) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "move", body.uci)}

    @app.post("/api/puzzle/{puzzle_id}/retract")
    def puzzle_retract(puzzle_id: str) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "retract")}

    @app.post("/api/puzzle/{puzzle_id}/giveup")
    def puzzle_giveup(puzzle_id: str) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "giveup")}

    @app.post("/api/puzzle/{puzzle_id}/ask")
    def puzzle_ask(puzzle_id: str, body: _Ask) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "ask", body.text)}

    @app.post("/api/puzzle/{puzzle_id}/voice")
    def puzzle_voice(puzzle_id: str, body: _OnOff) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "voice", body.on)}

    @app.post("/api/puzzle/{puzzle_id}/close")
    def puzzle_close(puzzle_id: str) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "close")}

    @app.post("/api/correct")
    def correct(body: _Correct) -> dict[str, bool]:
        return {"ok": api.correct(body.game_id, body.segment, body.ply, body.cause)}

    @app.post("/api/listen")
    def listen(body: _Listen) -> dict[str, Any]:
        return {"ok": True, "mode": engine.set_listen(body.mode)}

    @app.post("/api/ptt")
    def ptt(body: _Ptt) -> dict[str, bool]:
        engine.ptt(body.down)
        return {"ok": True}

    @app.post("/api/hear")
    async def hear(request: Request) -> dict[str, Any]:
        """A 16 kHz mono WAV in the body is treated as something the user said -- the way to
        exercise the whole hear -> correct -> ask path without a microphone."""
        import io
        import wave

        import numpy as np

        body = await request.body()
        with wave.open(io.BytesIO(body), "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
                return {"ok": False, "error": "need 16 kHz mono 16-bit PCM WAV"}
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        audio = pcm.astype(np.float32) / 32768.0
        engine.hear(audio)
        return {"ok": True, "seconds": round(len(audio) / 16000, 2)}

    # Everything else is the static web UI; html=True serves index.html at /.
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    return app


def _rooms_for(engine: PanelEngine) -> Any:
    """The puzzle rooms of this server: the puzzle store under the data directory, the
    profile store beside it for the outcomes, the panel's voice for the coach's answers,
    and the motifs the engine's recommendations name."""
    from sidecoach.profile.puzzles import open_store
    from sidecoach.ui.puzzle_room import PuzzleRooms

    puzzles = None
    with contextlib.suppress(Exception):
        puzzles = open_store(engine.settings.data_dir)
    tts = getattr(engine, "_tts", None)
    speak = getattr(tts, "say", None)
    stop = getattr(tts, "stop", None)
    rooms = PuzzleRooms(
        engine.settings,
        store_path=engine.settings.data_dir / "profile.sqlite",
        puzzles=puzzles,
        speak=speak,
        stop_speaking=stop,
        analysis=engine.analysis,
        coach=engine.coach,
    )
    rooms.motifs = engine.puzzle_motifs
    return rooms


def _room_command(rooms: Any, puzzle_id: str, name: str, payload: Any = None) -> bool:
    room = rooms.get(puzzle_id)
    if room is None:
        return False
    room.command(name, payload)
    return True


def make_engine(
    settings: Settings,
    *,
    monitor: int | None = None,
    fps: int = 10,
    replay: Path | None = None,
    record: bool = True,
    speed: float = 1.0,
    tts_device: str | None = None,
    stt_device: str | None = None,
    listen: str | None = None,
) -> PanelEngine:
    """A live engine, or one fed from a recording when `replay` is given (deterministic dev).

    A replay is paced to its own timestamps at `speed` times real time (1.0 = as it was
    played, so the coach and the voice have the time they had; 0 = as fast as the pipeline
    goes, for tests).
    """
    # CLI overrides for where the voice models run and how the coach listens.
    if tts_device or stt_device or listen:
        settings = settings.model_copy(deep=True)
        if tts_device:
            settings.voice.tts_device = tts_device
        if stt_device:
            settings.voice.stt_device = stt_device
        if listen:
            settings.voice.listen = listen
    frame_source = None
    if replay is not None:
        import time

        from sidecoach.replay.recording import Recording

        def frames() -> Any:
            first_ts: float | None = None
            wall0 = time.monotonic()
            for i, (tick, frame) in enumerate(Recording(replay).frames()):
                if first_ts is None:
                    first_ts = tick.ts
                if speed > 0:
                    due = wall0 + (tick.ts - first_ts) / speed
                    delay = due - time.monotonic()
                    if delay > 0:
                        time.sleep(min(delay, 1.0))
                yield (i, tick.ts, frame)

        frame_source = frames()
    return PanelEngine(settings, monitor=monitor, fps=fps, record=record, frame_source=frame_source)


def run_server(
    settings: Settings,
    *,
    monitor: int | None = None,
    fps: int = 10,
    host: str = "127.0.0.1",
    port: int = 8765,
    replay: Path | None = None,
    record: bool = True,
    speed: float = 1.0,
    open_browser: bool = True,
    tts_device: str | None = None,
    stt_device: str | None = None,
    listen: str | None = None,
) -> None:
    """Serve the panel and open it in the browser. Blocks until the server is stopped."""
    import uvicorn

    engine = make_engine(
        settings,
        monitor=monitor,
        fps=fps,
        replay=replay,
        record=record,
        speed=speed,
        tts_device=tts_device,
        stt_device=stt_device,
        listen=listen,
    )
    app = build_app(engine)
    if replay is not None:
        engine.start()  # a recording plays on its own; live capture waits for the Start button
    url = f"http://{host}:{port}/"
    if open_browser:
        # A headless box has no browser; the URL still serves either way.
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    uvicorn.run(app, host=host, port=port, log_level="warning")
