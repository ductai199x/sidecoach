"""Suite-wide options: `--seeds N` (or `--seeds A-B`) widens the random-game stress sweep
(tests/test_stress.py); the guard that keeps the suite off a live session; and the tests
that load the embedding net kept to a few workers while the rest use every core."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import socket
import tempfile
import zlib

import pytest

# The tests that run the embedding net (vision/embedding.py: the synthetic vision suite,
# the panel over rendered games) are dealt into this many xdist groups, so this many
# workers hold a CUDA session (half a GB each); everything else spreads over every worker
# (`--dist loadgroup` in pyproject; the group is the mark below). The voice models, loaded
# in every worker by the panel tests, once had the GPU at 18 GB and the machine lagging
# (Tai: "the rest should run on all 24"); they no longer load (PanelEngine._voice_settings).
# Measured 2026-09-28 on 24 workers, once the net really was on the GPU (a broken ml_dtypes
# had kept it on the CPU: voice/runtime._probe_cuda) and OpenCV kept to a thread a worker
# (pytest_configure): 8 groups 119 s, 16 groups 93 s and 78 s with one OpenCV thread, 24
# no faster -- the vision suite is our own Python on the CPU, the net 10 % of it.
EMBEDDER_WORKERS = int(os.environ.get("SIDECOACH_TEST_EMBEDDER_WORKERS", "16"))
# Test modules that load the net without the `vision` mark.
EMBEDDER_MODULES = {
    "test_clock_pipeline",
    "test_embedding",
    "test_panel_engine",
    "test_panel_server",
    "test_panel_voice_policy",
    "test_theme_cache",
}

# The panel's port when it serves a live game (sidecoach panel). The suite runs a dozen
# workers that each load the embedding net on the GPU: started during a live game it froze
# the pipeline for 323 s (2026-09-19, the vision's CUDA calls starved). So it refuses to
# start while that port is served, unless SIDECOACH_TEST_ALONGSIDE=1 says the port is not a
# live game (a replay, a test server of one's own).
LIVE_PANEL_PORT = 8765


def _app_lock_ports() -> list[int]:
    """The ports of the desktop app's sidecars, from the lock files in the app's data
    folders: the shell picks a free port, never 8765, so the port check alone missed a
    live game in the installed app (2026-09-22)."""
    homes = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            homes.append(pathlib.Path(local) / "Sidecoach" / "data")
    else:
        homes.append(pathlib.Path.home() / ".local" / "share" / "sidecoach")
    ports: list[int] = []
    for home in homes:
        lock = home / "panel.lock"
        try:
            ports.append(int(json.loads(lock.read_text(encoding="utf-8"))["port"]))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return ports


def _serving_a_session(port: int) -> bool:
    """A panel on `port` with a session running (Start pressed): its health says so
    without a token. A panel merely up is not in anyone's way."""
    try:
        import urllib.request

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("running"))
    except Exception:  # noqa: BLE001 - not a panel, or not answering
        return False


def _live_panel_up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", LIVE_PANEL_PORT), timeout=0.5).close()
        return True
    except OSError:
        pass
    return any(_serving_a_session(port) for port in _app_lock_ports())


def _below_normal_priority() -> None:
    """The test processes yield to the desktop: a worker per core at normal priority
    lags the machine. Windows has priority classes; elsewhere, nice."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        below_normal = 0x4000
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal)
    else:
        with contextlib.suppress(OSError):
            os.nice(5)


def pytest_configure(config: pytest.Config) -> None:
    _below_normal_priority()  # the controller and, since this runs in each, every worker
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # One OpenCV thread a worker: every worker's own pool, one thread a core each, had
        # two dozen workers' pools fighting over the cores.
        import cv2

        cv2.setNumThreads(1)
    # No panel built by a test loads Kokoro or Parakeet; the one that wants the real recogniser
    # says so (PanelEngine(voice_models=True)).
    os.environ.setdefault("SIDECOACH_VOICE_MODELS", "0")
    # A Settings made by a test has no file of its own; a save from it (set_record, the
    # settings API) would land on `sidecoach.toml` in the working directory -- the
    # developer's -- as one did on 2026-09-22. It lands in a scratch file instead.
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sidecoach-tests-")) / "sidecoach.toml"
    os.environ["SIDECOACH_CONFIG"] = str(scratch)
    if os.environ.get("SIDECOACH_TEST_ALONGSIDE") == "1":
        return
    if _live_panel_up():
        pytest.exit(
            f"a sidecoach panel is serving on :{LIVE_PANEL_PORT}, or the desktop app has a "
            "session running (a live game?): the suite would starve it. Wait for the session "
            "to end, or set SIDECOACH_TEST_ALONGSIDE=1 if it is not a live game.",
            returncode=3,
        )


@pytest.hookimpl(tryfirst=True)  # before xdist's own, which reads the marks on the worker
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        module = item.path.stem
        if item.get_closest_marker("vision") is None and module not in EMBEDDER_MODULES:
            continue
        # The same test lands in the same group on every run; a module's tests spread
        # over the groups rather than queueing behind one another.
        group = zlib.crc32(item.nodeid.encode()) % EMBEDDER_WORKERS
        item.add_marker(pytest.mark.xdist_group(f"embedder-{group}"))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seeds",
        action="store",
        default="",
        help="how many seeded random games the stress tests play, or a range A-B of seeds "
        "(default: a fixed handful)",
    )
