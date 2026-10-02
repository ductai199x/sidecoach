"""The panel backend: a local FastAPI server over the live coaching engine.

The engine (`PanelEngine`) is UI-agnostic -- it runs the same capture -> pipeline -> coach
loop as `watch`, exposes a thread-safe `poll()` and takes control commands. This wraps it in
plain HTTP: the browser polls `GET /api/poll` for state and new terminal lines, and the
controls are `POST /api/...`. Because the transport is ordinary HTTP, the page opens in any
browser (debuggable with real devtools) and the API is testable end to end with httpx.

`build_app` makes the app around an engine (a recording-fed engine in a test, a live one in
the field); `run_server` opens the browser and serves it.

The desktop shell (desktop/, docs/app-plan.md) runs this server as a sidecar: `run_server`
with `port=0` binds an ephemeral port and prints one JSON line `{"port": N}` on stdout for
the shell to read; `parent_pid` makes the server exit when that process is gone, so a
crashed shell never leaves an orphan; a lock file under the data directory lets a second
launch attach to the first instead of starting another engine; `token` guards every /api
route with a bearer the shell hands to the page; and the shell's own origins are allowed
by CORS, because the bundled frontend calls the sidecar cross-origin.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

# This module needs the panel extras (`uv sync --group panel`); the CLI imports it behind an
# ImportError guard. FastAPI resolves handler annotations from the module globals, so
# `Request` is imported here, not inside `build_app`.
from fastapi import Request
from pydantic import BaseModel

from sidecoach import looks
from sidecoach.config import Settings
from sidecoach.ui import settings_api
from sidecoach.ui.panel import VOICE_SAMPLE, PanelApi, PanelEngine
from sidecoach.ui.settings_api import SettingsUpdate  # resolved from the module globals

if TYPE_CHECKING:
    from fastapi import FastAPI


WEB = Path(__file__).parent / "web"

# The origins the desktop shell's bundled frontend is served from (Tauri on Windows, Tauri
# on Linux, the Vite dev server), allowed to call the API cross-origin. The browser flow
# is same-origin and needs none of this.
SHELL_ORIGINS = ["http://tauri.localhost", "tauri://localhost", "http://localhost:1420"]
LOCK_FILE = "panel.lock"
# The request bodies a cross-site page can send without a CORS preflight.
_SIMPLE_TYPES = ("application/x-www-form-urlencoded", "multipart/form-data", "text/plain")


def _host_ok(host: str) -> bool:
    """Whether a request's Host names this machine by an address or as localhost. Any other
    name is a page whose domain was made to resolve here (DNS rebinding)."""
    import ipaddress

    name = host.strip().lower()
    if name.startswith("["):  # [::1]:8765
        name = name[1 : name.find("]")] if "]" in name else ""
    elif name.count(":") == 1:
        name = name.split(":")[0]
    if name == "localhost":
        return True
    try:
        ipaddress.ip_address(name)
    except ValueError:
        return False
    return True


def _refusal(method: str, path: str, headers: Any) -> str:
    """Why a request is refused ('' when it is not): a foreign Host, or a browser request
    from a page that is neither this server's nor the desktop shell's -- a state change
    must come with a JSON body, which no cross-site page can send without a preflight."""
    host = headers.get("host", "")
    if not _host_ok(host):
        return "host"
    origin = headers.get("origin", "")
    site = headers.get("sec-fetch-site", "")
    if origin in SHELL_ORIGINS:
        return ""
    if origin and origin.lower() not in (f"http://{host}".lower(), f"https://{host}".lower()):
        return "origin"
    if site in ("cross-site", "same-site"):
        return "origin"
    browser = bool(origin or site)
    if browser and path.startswith("/api/") and method in ("POST", "PUT", "PATCH"):
        kind = headers.get("content-type", "").split(";")[0].strip().lower()
        if not kind or kind in _SIMPLE_TYPES:
            return "content-type"
    return ""


# How often the server checks that the parent it was told to follow is still alive.
PARENT_CHECK_S = 2.0


class _Ask(BaseModel):
    text: str


class _HearTest(BaseModel):
    on: bool = True


class _Say(BaseModel):
    voice: str | None = None
    text: str | None = None


class _ClientLog(BaseModel):
    """A line from the page itself -- an uncaught error, a failed asset -- so a window that
    never booted still says why, in the terminal of the panel (the settings window of the
    installed app opened white, and nothing anywhere said what had failed)."""

    text: str
    page: str = ""
    level: str = "error"


class _Perf(BaseModel):
    """A sample from the page's own meter (`?perf=<label>`): frame rate, long frames and
    the poll round trip, appended to data/perf/<label>.jsonl for the shell's evaluation."""

    label: str = "page"
    sample: dict[str, Any]


class _Voice(BaseModel):
    voice: str


class _Paused(BaseModel):
    paused: bool


class _PieceFile(BaseModel):
    name: str
    data: str  # base64


class _Pieces(BaseModel):
    """Images a player chose for one of their own sets (looks.create_own, looks.add_images)."""

    files: list[_PieceFile]
    name: str = ""


class _SetName(BaseModel):
    name: str


class _Muted(BaseModel):
    muted: bool


class _Record(BaseModel):
    on: bool


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


class _OnOff(BaseModel):
    on: bool


class _Import(BaseModel):
    source: str  # pgn | chesscom | lichess
    text: str = ""  # the PGN, for "pgn"
    months: int = 3  # how far back on chess.com


class _Review(BaseModel):
    strength: str = "standard"


class _Card(BaseModel):
    text: str


class _Try(BaseModel):
    ply: int
    uci: str


class _Hint(BaseModel):
    ply: int
    level: int


class _Fen(BaseModel):
    fen: str


class _Play(BaseModel):
    fen: str
    uci: str


class _Explore(BaseModel):
    fen: str
    moves: list[str] = []  # UCI from the game's start: the opening's name


class _Pgn(BaseModel):
    tree: list[dict[str, Any]] = []  # the analysis board's moves: {"uci", "children"}
    annotated: bool = True
    save: bool = False


class _Keys(BaseModel):
    keys: list[str] = []  # library keys: the games selected in Your games


class _Rect(BaseModel):
    x: int
    y: int
    w: int
    h: int


class _Ids(BaseModel):
    ids: list[str] | None = None


def build_app(engine: PanelEngine, token: str = "", *, background: bool = False) -> FastAPI:
    """A FastAPI app serving the panel UI and the control API over `engine`. With `token`,
    every /api route but /api/health wants `Authorization: Bearer <token>`. With
    `background`, the library's games are reviewed in the background for Stats (the panel
    that owns the data directory: `serve`)."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles

    from sidecoach import __version__

    api = PanelApi(engine)
    app = FastAPI(title="sidecoach panel")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=SHELL_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if token:

        @app.middleware("http")
        async def _bearer(request: Request, call_next: Any) -> Any:
            path = request.url.path
            if path.startswith("/api/") and path != "/api/health" and request.method != "OPTIONS":
                header = request.headers.get("authorization", "")
                if header != f"Bearer {token}":
                    return JSONResponse({"ok": False, "error": "unauthorised"}, status_code=401)
            return await call_next(request)

    @app.middleware("http")
    async def _same_site(request: Request, call_next: Any) -> Any:
        # Outermost: before the bearer and CORS, a foreign Host or a cross-site page is refused.
        why = _refusal(request.method, request.url.path, request.headers)
        if why:
            return JSONResponse({"ok": False, "error": f"refused: {why}"}, status_code=403)
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """What the shell polls after spawning the sidecar, and what a lock-file check asks."""
        return {
            "ok": True,
            "version": __version__,
            "pid": os.getpid(),
            "running": engine.running,
            "warm": engine.warm_status(),
        }

    perf_lock = threading.Lock()  # two pages posting at once must not tear a line

    @app.post("/api/perf")
    def perf(body: _Perf) -> dict[str, bool]:
        folder = engine.settings.data_dir / "perf"
        folder.mkdir(parents=True, exist_ok=True)
        label = "".join(ch for ch in body.label if ch.isalnum() or ch in "-_") or "page"
        line = json.dumps({"t": time.time(), **body.sample}) + "\n"
        with perf_lock, (folder / f"{label}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(line)
        return {"ok": True}

    @app.post("/api/client_log")
    def client_log(body: _ClientLog) -> dict[str, bool]:
        level = body.level if body.level in ("info", "warn", "error") else "error"
        where = f" [{body.page}]" if body.page else ""
        engine._emit("sys", level, f"page{where}: {body.text[:2000]}")
        return {"ok": True}

    @app.post("/api/dump")
    def dump() -> dict[str, Any]:
        """Write the frames the engine keeps in memory, and its log, to the data directory:
        the one way anything about a live session reaches the disk in the app."""
        return {"ok": True, "path": engine.dump()}

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

    @app.post("/api/hear_test")
    def hear_test(body: _HearTest) -> dict[str, Any]:
        """The settings page's live test of the microphone and speech recognition: on or off."""
        return engine.set_hear_test(body.on)

    @app.get("/api/hear_test")
    def hear_test_status() -> dict[str, Any]:
        return engine.hear_status()

    @app.post("/api/say")
    def say(body: _Say) -> dict[str, bool]:
        """One line in a voice, for the settings page's Listen button."""
        return {"ok": engine.say_sample(body.voice, body.text or VOICE_SAMPLE)}

    @app.post("/api/voice")
    def voice(body: _Voice) -> dict[str, bool]:
        return {"ok": api.set_voice(body.voice)}

    # One of the panel's own switches at a time, set and kept in one step: two quick presses
    # saved out of order left the switch one way and the setting the other. The settings' own
    # lock, so a Settings save at the same moment cannot put the switch back either (the
    # settings review, 2026-10-01).
    keeping = settings_api.saving

    @app.post("/api/muted")
    def muted(body: _Muted) -> dict[str, bool]:
        with keeping:
            ok = api.set_muted(body.muted)
            settings_api.remember(engine, {"voice": {"muted": body.muted}})  # the next session too
        return {"ok": ok}

    @app.post("/api/dictation/cancel")
    def dictation_cancel() -> dict[str, Any]:
        """The person typed over the dictation: it stays in the box, nothing sends itself."""
        return engine.cancel_dictation()

    @app.post("/api/record")
    def record(body: _Record) -> dict[str, Any]:
        """Record this session to the data folder (Debug > Record this session), or stop."""
        return engine.set_record(body.on)

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

    # The frontend is a single page (desktop/, SvelteKit with adapter-static): every
    # route it owns is served index.html and routed on the client.
    @app.get("/puzzle/{puzzle_id}")
    def puzzle_page(puzzle_id: str) -> Any:
        return FileResponse(WEB / "index.html")

    @app.get("/settings")
    @app.get("/setup")
    # The shell loads the splash from its own bundle; served here too, it runs outside the
    # shell as it does on the phone (and `/splash?demo` shows the design in a browser).
    @app.get("/splash")
    def spa_page() -> Any:
        return FileResponse(WEB / "index.html")

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

    # A move, a take-back, a give-up: played at once, answered with the state they leave.
    @app.post("/api/puzzle/{puzzle_id}/move")
    def puzzle_move(puzzle_id: str, body: _Move) -> dict[str, Any]:
        room = rooms.get(puzzle_id)
        return {"ok": True, "state": room.move(body.uci)} if room else {"ok": False}

    @app.post("/api/puzzle/{puzzle_id}/retract")
    def puzzle_retract(puzzle_id: str) -> dict[str, Any]:
        room = rooms.get(puzzle_id)
        return {"ok": True, "state": room.retract()} if room else {"ok": False}

    @app.post("/api/puzzle/{puzzle_id}/giveup")
    def puzzle_giveup(puzzle_id: str) -> dict[str, Any]:
        room = rooms.get(puzzle_id)
        return {"ok": True, "state": room.give_up()} if room else {"ok": False}

    @app.post("/api/puzzle/{puzzle_id}/ask")
    def puzzle_ask(puzzle_id: str, body: _Ask) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "ask", body.text)}

    @app.post("/api/puzzle/{puzzle_id}/voice")
    def puzzle_voice(puzzle_id: str, body: _OnOff) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "voice", body.on)}

    @app.post("/api/puzzle/{puzzle_id}/close")
    def puzzle_close(puzzle_id: str) -> dict[str, bool]:
        return {"ok": _room_command(rooms, puzzle_id, "close")}

    # ---- Game Review: the library of games, and a review in a window of its own ----------
    # (profile/library.py, ui/review_room.py; docs/game-review-plan.md)
    from sidecoach.ui.review_room import ReviewRooms

    store_path = engine.settings.data_dir / "profile.sqlite"
    reviews = ReviewRooms(engine.settings, store_path=store_path, mirror=engine._emit)
    engine.settings_followers.append(lambda s: setattr(reviews, "settings", s))
    # The AI coach's words on reviews count in the panel's strip with the game's.
    engine.extra_usage.append(reviews.usages)
    backlog = None
    if background:
        from sidecoach.ui.backlog import Backlog

        backlog = Backlog(engine.settings, store_path)
        backlog.start()
        engine.settings_followers.append(lambda s: setattr(backlog, "settings", s))
    app.state.backlog = backlog  # closed with the panel (`serve`)

    @app.get("/games")
    @app.get("/stats")
    @app.get("/review/{key}")
    def review_pages(key: str = "") -> Any:
        return FileResponse(WEB / "index.html")

    @app.get("/api/games")
    def games() -> dict[str, Any]:
        # And how far the background reviewer has got (none beside another panel).
        return {**_library(engine, store_path), "backlog": backlog and backlog.state()}

    @app.post("/api/games/import")
    def games_import(body: _Import) -> dict[str, Any]:
        done = _import(engine, store_path, body)
        if done.get("added") and backlog is not None:
            backlog.nudge()  # reviewed from now, the progress shown at once
        return done

    # Games out of the player's record, the ones selected in Your games, each with the game it is
    # paired with; Undo puts them back for a while (Store.remove_game, restore_game).
    @app.post("/api/games/remove")
    def games_remove(body: _Keys) -> dict[str, Any]:
        from sidecoach.profile.store import Store

        store = Store(store_path)
        try:
            removed = [k for key in body.keys for k in store.remove_game(key)]
        finally:
            store.close()
        return {"ok": True, "removed": removed}

    @app.post("/api/games/restore")
    def games_restore(body: _Keys) -> dict[str, Any]:
        from sidecoach.profile.store import Store

        store = Store(store_path)
        try:
            restored = [k for key in body.keys for k in store.restore_game(key)]
        finally:
            store.close()
        if not restored:
            return {"ok": False, "error": "Those games were removed too long ago to put back."}
        return {"ok": True, "restored": restored}

    # The card on the move shown, said in the panel's voice and captioned: a new card cuts off
    # the one before, as the player steps on; closing the review silences it.
    review_speak, review_hush = _room_voice(engine)

    @app.post("/api/review/{key}/open")
    def review_open(key: str, body: _Review) -> dict[str, Any]:
        room = reviews.open(key, body.strength)
        if room is None:
            return {"ok": False, "error": f"no game {key} in the library"}
        return {"ok": True, "voice": review_speak is not None, **room.poll()}

    @app.get("/api/review/{key}/poll")
    def review_poll(key: str, since: int = 0) -> dict[str, Any]:
        room = reviews.get(key)
        if room is None:
            return {"ok": False, "closed": True}
        return {"ok": True, **room.poll(since)}

    # The AI coach's words on the key moves: only when the player asks, a call a move; the page
    # reads them at its next polls.
    @app.post("/api/review/{key}/ai")
    def review_ai(key: str) -> dict[str, Any]:
        room = reviews.get(key)
        if room is None:
            return {"ok": False, "error": "the review is not open"}
        refused = room.write_words()
        return {"ok": refused is None, "error": refused}

    @app.post("/api/review/{key}/say")
    def review_say(key: str, body: _Card) -> dict[str, bool]:
        if review_speak is None or review_hush is None:
            return {"ok": False}
        review_hush()
        review_speak(body.text)
        return {"ok": True}

    # Retry (ui/review_retry.py): the position a move was played from, a new move graded as the
    # review graded the game's, and the coach's hints there.
    def _retries(key: str) -> Any:
        room = reviews.get(key)
        return room.retries if room is not None else None

    @app.get("/api/review/{key}/retry")
    def review_retry(key: str, ply: int) -> dict[str, Any]:
        retries = _retries(key)
        if retries is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **retries.position(ply)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/try")
    def review_try(key: str, body: _Try) -> dict[str, Any]:
        retries = _retries(key)
        if retries is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **retries.attempt(body.ply, body.uci)}
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/hint")
    def review_hint(key: str, body: _Hint) -> dict[str, Any]:
        retries = _retries(key)
        if retries is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **retries.hint(body.ply, body.level)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/hush")
    def review_quiet(key: str) -> dict[str, bool]:
        if review_hush is not None:
            review_hush()
        return {"ok": review_hush is not None}

    @app.post("/api/review/{key}/close")
    def review_close(key: str) -> dict[str, bool]:
        room = reviews.get(key)
        if room is not None:
            room.close()
        if review_hush is not None:
            review_hush()
        return {"ok": room is not None}

    # Analysis (ui/review_analysis.py): the player's own moves from any position, the engine's
    # lines for the board, the explorer of the player's games, the game as a PGN.
    from sidecoach.ui import review_analysis

    @app.post("/api/review/{key}/position")
    def review_position(key: str, body: _Fen) -> dict[str, Any]:
        try:
            return {"ok": True, **review_analysis.position(body.fen)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/move")
    def review_move(key: str, body: _Play) -> dict[str, Any]:
        try:
            return {"ok": True, **review_analysis.move(body.fen, body.uci)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/lines")
    def review_lines(key: str, body: _Fen) -> dict[str, Any]:
        room = reviews.get(key)
        if room is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **room.lines(body.fen)}
        except (ValueError, RuntimeError) as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/explore")
    def review_explore(key: str, body: _Explore) -> dict[str, Any]:
        room = reviews.get(key)
        if room is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **room.explore(body.fen, body.moves)}
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}

    @app.post("/api/review/{key}/pgn")
    def review_pgn(key: str, body: _Pgn) -> dict[str, Any]:
        room = reviews.get(key)
        if room is None:
            return {"ok": False, "error": "the review is not open"}
        try:
            return {"ok": True, **room.pgn(body.tree, annotated=body.annotated, save=body.save)}
        except (ValueError, OSError) as exc:
            return {"ok": False, "error": str(exc)}

    # ---- Stats: the player's games added up (profile/stats.py; docs/stats-plan.md) ---------
    from sidecoach.profile import stats as stat_sums
    from sidecoach.profile.store import Store

    def _filter(time_class: str, days: int, opponents: str) -> stat_sums.Filter:
        return stat_sums.Filter(
            time_class=time_class or None, days=days or None, opponents=opponents
        )

    @app.get("/api/stats")
    def stats_now(time_class: str = "", days: int = 0, opponents: str = "all") -> dict[str, Any]:
        store = Store(store_path)
        try:
            numbers = stat_sums.stats(store, _filter(time_class, days, opponents))
            _drills(numbers["work"], engine.settings.data_dir, store)
        finally:
            store.close()
        return {"ok": True, "stats": numbers, "backlog": backlog and backlog.state()}

    @app.get("/api/stats/positions")
    def stats_positions(
        what: str, arg: str = "", time_class: str = "", days: int = 0, opponents: str = "all"
    ) -> dict[str, Any]:
        store = Store(store_path)
        try:
            found = stat_sums.positions(store, _filter(time_class, days, opponents), what, arg)
        finally:
            store.close()
        return {"ok": True, "positions": found}

    # ---- how the boards look (looks.py): every window reads it; the pieces a player chose ----
    @app.get("/api/look")
    def get_look() -> dict[str, Any]:
        board = engine.settings.board
        return looks.look(board.pieces, board.theme, engine.settings.data_dir)

    # A player's own sets: a new one from the images chosen, more images for one, a new name,
    # deleted (put aside for its Undo) and back. Each that took raises the settings' counter,
    # so every window draws the sets again.
    def touched(result: dict[str, Any]) -> dict[str, Any]:
        if result.get("ok"):
            engine.settings_touched()
        return result

    @app.post("/api/pieces/own")
    def post_own_set(body: _Pieces) -> dict[str, Any]:
        files = [f.model_dump() for f in body.files]
        return touched(looks.create_own(engine.settings.data_dir, body.name, files))

    @app.post("/api/pieces/own/{set_id}/images")
    def post_own_images(set_id: str, body: _Pieces) -> dict[str, Any]:
        files = [f.model_dump() for f in body.files]
        return touched(looks.add_images(engine.settings.data_dir, set_id, files))

    @app.put("/api/pieces/own/{set_id}")
    def put_own_set(set_id: str, body: _SetName) -> dict[str, Any]:
        return touched(looks.rename_own(engine.settings.data_dir, set_id, body.name))

    @app.delete("/api/pieces/own/{set_id}")
    def delete_own_set(set_id: str) -> dict[str, Any]:
        return touched(looks.delete_own(engine.settings.data_dir, set_id))

    @app.post("/api/pieces/own/{set_id}/restore")
    def restore_own_set(set_id: str) -> dict[str, Any]:
        return touched(looks.restore_own(engine.settings.data_dir, set_id))

    # ---- the settings, for the app's settings window (ui/settings_api.py) ---------------
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return settings_api.describe(engine.settings)

    @app.put("/api/settings")
    def put_settings(body: SettingsUpdate) -> dict[str, Any]:
        return settings_api.update(engine, body)

    # ---- first-run setup: the models, Stockfish, the openings, the pack (setup/) ------
    from sidecoach.setup import Setup, free_space
    from sidecoach.setup import gpu as gpu_pack

    setup = Setup(engine.settings)
    engine.settings_followers.append(lambda s: setattr(setup, "settings", s))

    @app.get("/api/setup/status")
    def setup_status() -> dict[str, Any]:
        out = setup.status()
        out["gpu"] = gpu_pack.status(engine.settings)
        out["free_bytes"] = free_space(engine.settings.data_dir)
        return out

    @app.post("/api/setup/start")
    def setup_start(body: _Ids | None = None) -> dict[str, Any]:
        return {"ok": setup.start(body.ids if body and body.ids is not None else None)}

    @app.post("/api/setup/cancel")
    def setup_cancel() -> dict[str, bool]:
        setup.cancel()
        return {"ok": True}

    @app.post("/api/setup/remove_pack")
    def setup_remove_pack() -> dict[str, bool]:
        return {"ok": gpu_pack.remove(engine.settings)}

    # ---- the long jobs the panel's bell shows (lib/Jobs.svelte): the background review of the
    # library, and a fetch of the setup under way. Asked every couple of seconds: cheap.
    @app.get("/api/jobs")
    def jobs() -> dict[str, Any]:
        return {
            "reviews": backlog.state() if backlog is not None else None,
            "setup": setup.run_state(),
        }

    @app.post("/api/jobs/reviews")
    def jobs_reviews(body: _Paused) -> dict[str, Any]:
        if backlog is None:
            return {"ok": False, "error": "no background review here"}
        return {"ok": True, "reviews": backlog.pause(body.paused)}

    @app.post("/api/reset_profile")
    def reset_profile() -> dict[str, Any]:
        return api.reset_profile()

    @app.post("/api/clear_caches")
    def clear_caches() -> dict[str, Any]:
        # The rooms let go of the puzzle store first (a file held open cannot go), and
        # open it again after -- rebuilt from the CSVs by the engine, or at the open.
        from sidecoach.profile.puzzles import open_store

        if rooms.puzzles is not None:
            with contextlib.suppress(Exception):
                rooms.puzzles.close()
            rooms.puzzles = None
        done = api.clear_caches()
        with contextlib.suppress(Exception):
            rooms.puzzles = open_store(engine.settings.data_dir)
        return done

    @app.post("/api/correct")
    def correct(body: _Correct) -> dict[str, bool]:
        return {"ok": api.correct(body.game_id, body.segment, body.ply, body.cause)}

    @app.post("/api/listen")
    def listen(body: _Listen) -> dict[str, Any]:
        with keeping:
            mode = engine.set_listen(body.mode)  # refuses a mode there is not
            settings_api.remember(engine, {"voice": {"listen": body.mode}})  # the next session too
        return {"ok": True, "mode": mode}

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


def _library(engine: PanelEngine, store_path: Path) -> dict[str, Any]:
    """The games Game Review can open, the latest played first -- imported and watched, a
    watched game its import stands for once, as the import -- each with its review's
    accuracies when it has one, and the player's usernames."""
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui.review_room import carry_twins

    store = Store(store_path)
    try:
        twins = library.reconcile(store)
        carry_twins(store, twins)  # a pair's review goes with the import that stands for it
        watched = [r for r in library.from_watched(store) if r.key not in twins]
        rows = [*store.library(), *watched]
        rows.sort(key=lambda r: r.played_at or r.added_at, reverse=True)
        stood_for: dict[str, list[str]] = {}  # an import -> the watched games it stands for
        for was, imported in twins.items():
            stood_for.setdefault(imported, []).append(was)
        out = []
        for row in rows:
            review = store.latest_review(row.key)
            # An import not reviewed yet whose watched game was: that review's accuracies until
            # the import has its own (the review of 2026-10-01: pairing a record that differs
            # from its import took the game's accuracy away; it is not reviewed for that).
            shown = review or _latest(store, stood_for.get(row.key, []))
            summary = json.loads(shown.summary) if shown and shown.summary else None
            out.append(
                {
                    **row.__dict__,
                    "pgn": None,  # the page lists games; a review reads its own PGN
                    "reviewed": review is not None,
                    "accuracy": (
                        {s: summary[s].get("accuracy") for s in ("white", "black")}
                        if summary
                        else None
                    ),
                    "performance": (
                        {
                            s: (summary[s].get("performance") or {}).get("rating")
                            for s in ("white", "black")
                        }
                        if summary
                        else None
                    ),
                }
            )
    finally:
        store.close()
    profile = engine.settings.profile
    return {
        "games": out,
        "users": {"chesscom": profile.chesscom_user, "lichess": profile.lichess_user},
    }


def _latest(store: Any, keys: list[str]) -> Any | None:
    """The latest review of any of these games, or None."""
    reviews = [r for r in (store.latest_review(k) for k in keys) if r is not None]
    return max(reviews, key=lambda r: r.reviewed_at, default=None)


def _import(engine: PanelEngine, store_path: Path, body: Any) -> dict[str, Any]:
    """Games into the library: a pasted PGN, or the player's recent games on chess.com or
    Lichess (their usernames from the settings). What came in says how many are new, how many
    were there already and how many the player removed before (a site's games leave those out;
    a game pasted by hand comes back); a failure says what to do, and names the setting when
    the username is at fault (`field`: the page links to it)."""
    import httpx

    from sidecoach.profile import library
    from sidecoach.profile.store import Store

    profile = engine.settings.profile
    names = [n for n in (profile.chesscom_user, profile.lichess_user) if n]
    site = {"chesscom": "chess.com", "lichess": "Lichess"}.get(body.source, "")
    field = f"profile.{body.source}_user" if site else None
    try:
        if body.source == "pgn":
            rows = library.from_pgn(body.text, user_names=names)
        elif body.source == "chesscom":
            if not profile.chesscom_user:
                return {"ok": False, "error": "Add your chess.com username first.", "field": field}
            rows = library.from_chesscom(profile.chesscom_user, months=max(1, body.months))
        elif body.source == "lichess":
            if not profile.lichess_user:
                return {"ok": False, "error": "Add your Lichess username first.", "field": field}
            rows = library.from_lichess(profile.lichess_user)
        else:
            return {"ok": False, "error": f"no source {body.source}"}
    except library.NoPlayer as exc:
        return {"ok": False, "error": f"{exc} Check the spelling.", "field": field}
    except httpx.HTTPError:
        return {
            "ok": False,
            "error": f"Could not reach {site}. Check the connection and try again.",
        }
    except Exception as exc:  # noqa: BLE001 - anything else is said, never raised
        return {"ok": False, "error": f"Could not bring the games in: {exc}"}
    store = Store(store_path)
    try:
        if body.source == "pgn":
            store.unremove([row.key for row in rows])
            out: set[str] = set()
        else:
            out = library.left_out(store, rows)
        rows = [row for row in rows if row.key not in out]
        new = [row for row in rows if store.library_game(row.key) is None]
        for row in rows:
            store.add_library_game(row)
        library.reconcile(store)  # a game Sidecoach watched, now imported: one game
    finally:
        store.close()
    return {
        "ok": True,
        "added": len(new),
        "already": len(rows) - len(new),
        "left_out": len(out),
        "keys": [r.key for r in rows],
    }


def _drills(work: list[dict[str, Any]], data_dir: Path, store: Any) -> None:
    """Each area Stats says to work on gets puzzles of its motif at the player's puzzle level, in
    place (none where the area has no motif, or there are no puzzles)."""
    from sidecoach.profile.puzzles import open_store, puzzle_level

    puzzles = open_store(data_dir) if any(w["motif"] for w in work) else None
    level = puzzle_level(store)
    try:
        for w in work:
            picked = puzzles.pick(w["motif"], level) if puzzles and w["motif"] else []
            w["puzzles"] = [{"id": p.id, "rating": p.rating} for p in picked]
    finally:
        if puzzles is not None:
            puzzles.close()


def _rooms_for(engine: PanelEngine) -> Any:
    """The puzzle rooms of this server: the puzzle store under the data directory, the
    profile store beside it for the outcomes, the panel's voice for the coach's answers,
    and the motifs the engine's recommendations name."""
    from sidecoach.profile.puzzles import open_store
    from sidecoach.ui.puzzle_room import PuzzleRooms

    puzzles = None
    with contextlib.suppress(Exception):
        puzzles = open_store(engine.settings.data_dir)
    speak, stop = _room_voice(engine)
    rooms = PuzzleRooms(
        engine.settings,
        store_path=engine.settings.data_dir / "profile.sqlite",
        puzzles=puzzles,
        data_dir=engine.settings.data_dir,
        speak=speak,
        stop_speaking=stop,
        analysis=engine.analysis,
        coach=engine.coach,
        mirror=engine._emit,
    )
    rooms.motifs = engine.puzzle_motifs
    engine.settings_followers.append(lambda s: setattr(rooms, "settings", s))
    # The rooms' model calls count in the panel's strip with the game's.
    engine.extra_usage.append(rooms.usages)
    return rooms


def _room_voice(
    engine: PanelEngine,
) -> tuple[Callable[[str], None] | None, Callable[[], None] | None]:
    """The panel's voice for a room of its own (a puzzle, a review): speak -- the words
    captioned as the panel's are, then said -- and stop; (None, None) when the panel has no
    voice. The voice is looked up when it speaks, not when the room was made: a reload swaps
    it (`_reload_voice`), and the rooms spoke on through the old one, over the new (the parity
    audit, 2026-09-29; the phone looks it up at each call)."""
    if getattr(engine, "_tts", None) is None:
        return None, None

    def speak(text: str) -> None:
        # The phone's rooms: `Captions.show`.
        caption = getattr(engine, "caption", None)
        if caption is not None:
            caption(text)
        engine._tts.say(text)

    def stop() -> None:
        engine._tts.stop()

    return speak, stop


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
    persist: bool = True,
    speed: float = 1.0,
    loop: bool = False,
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
            # Looped, the recording plays again from its start on a fresh clock, its
            # timestamps offset so time never runs backwards for the pipeline (a soak).
            i = 0
            offset = 0.0
            while True:
                first_ts: float | None = None
                last_ts = 0.0
                wall0 = time.monotonic()
                for tick, frame in Recording(replay).frames():
                    if first_ts is None:
                        first_ts = tick.ts
                    if speed > 0:
                        due = wall0 + (tick.ts - first_ts) / speed
                        delay = due - time.monotonic()
                        if delay > 0:
                            time.sleep(min(delay, 1.0))
                    last_ts = tick.ts - first_ts
                    yield (i, offset + last_ts, frame)
                    i += 1
                if not loop:
                    return
                offset += last_ts + 5.0

        frame_source = frames()
    return PanelEngine(
        settings,
        monitor=monitor,
        fps=fps,
        record=record,
        persist=persist,
        frame_source=frame_source,
    )


def _attached_port(settings: Settings, host: str) -> int | None:
    """The port of a sidecar already serving this data directory, if its lock file names
    a live process that answers /api/health; else None (and a stale lock is removed)."""
    lock = settings.data_dir / LOCK_FILE
    if not lock.exists():
        return None
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
        port, pid = int(info["port"]), int(info["pid"])
        import psutil

        if psutil.pid_exists(pid):
            with socket.create_connection((host, port), timeout=0.5) as conn:
                conn.sendall(b"GET /api/health HTTP/1.0\r\nHost: x\r\n\r\n")
                head = conn.recv(64)
            if head.startswith(b"HTTP/1.") and b" 200" in head:
                return port
    except Exception:  # noqa: BLE001 - any doubt means no attach
        pass
    with contextlib.suppress(OSError):
        lock.unlink()
    return None


def _watch_parent(pid: int, stop: Any, lock: Path | None = None) -> None:
    """Exit the process when `pid` is gone: the shell that spawned this sidecar has died or
    was killed, and an engine nobody can reach must not keep the screen capture going.
    The lock file goes too, so nothing names this process once it is gone."""
    import psutil

    def loop() -> None:
        while True:
            time.sleep(PARENT_CHECK_S)
            if not psutil.pid_exists(pid):
                with contextlib.suppress(Exception):
                    stop()
                if lock is not None:
                    with contextlib.suppress(OSError):
                        lock.unlink()
                os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def run_server(
    settings: Settings,
    *,
    monitor: int | None = None,
    fps: int = 10,
    host: str = "127.0.0.1",
    port: int = 8765,
    replay: Path | None = None,
    record: bool = True,
    persist: bool = True,
    speed: float = 1.0,
    loop: bool = False,
    open_browser: bool = True,
    tts_device: str | None = None,
    stt_device: str | None = None,
    listen: str | None = None,
    parent_pid: int | None = None,
    token: str = "",
    announce: bool = False,
) -> None:
    """Serve the panel and open it in the browser. Blocks until the server is stopped.

    `port=0` picks a free port; `announce` prints `{"port": N}` as one JSON line on stdout
    (the shell reads it); `parent_pid` ends the server when that process dies; a lock file
    under the data directory makes a second launch print the first one's port and return.
    """
    import uvicorn

    if announce:
        attached = _attached_port(settings, host)
        if attached is not None:
            print(json.dumps({"port": attached, "attached": True}), flush=True)
            return
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        # The app asks for port 0 and gets a free one; only the CLI names a port.
        raise SystemExit(
            f"port {port} is in use ({exc.strerror}): another panel? Stop it, pick another "
            "with --port, or --port 0 for a free one."
        ) from exc
    sock.listen(128)
    port = sock.getsockname()[1]

    engine = make_engine(
        settings,
        monitor=monitor,
        fps=fps,
        replay=replay,
        record=record,
        persist=persist,
        speed=speed,
        loop=loop,
        tts_device=tts_device,
        stt_device=stt_device,
        listen=listen,
    )
    lock = settings.data_dir / LOCK_FILE
    other = _attached_port(settings, host)
    # One background reviewer per data directory: none beside the app's, none for a replay.
    app = build_app(engine, token=token, background=other is None and replay is None)
    if other is None:
        with contextlib.suppress(OSError):
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
    else:
        # A development panel beside the app, on the same data directory: the app's
        # lock names the app's sidecar and stays so, or the shell attaches to nothing.
        print(f"another panel serves this data directory on port {other}; its lock file stays")
    # Everything a session needs loads now, so Start only starts the screen.
    engine.warm()

    def close() -> None:
        backlog = app.state.backlog
        if backlog is not None:
            backlog.close()
            backlog.join(10.0)
        engine.close()  # the session, then Stockfish and the coach

    if parent_pid is not None:
        _watch_parent(parent_pid, close, lock)
    if replay is not None:
        engine.start()  # a recording plays on its own; live capture waits for the Start button
    url = f"http://{host}:{port}/"
    if announce:
        print(json.dumps({"port": port, "pid": os.getpid()}), flush=True)
    if open_browser:
        # A headless box has no browser; the URL still serves either way.
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        with contextlib.suppress(Exception):
            close()
        with contextlib.suppress(OSError):
            if (
                lock.exists()
                and json.loads(lock.read_text(encoding="utf-8")).get("pid") == os.getpid()
            ):
                lock.unlink()
        sys.stdout.flush()
