"""Suite-wide options: `--seeds N` (or `--seeds A-B`) widens the random-game stress sweep
(tests/test_stress.py); the guard that keeps the suite off a live session; and the tests
that load the embedding net kept to a few workers while the rest use every core."""

from __future__ import annotations

import contextlib
import os
import socket
import zlib

import pytest

# The tests that run the embedding net (vision/embedding.py: the synthetic vision suite,
# the panel over rendered games) are dealt into this many xdist groups, so this many
# workers hold a CUDA session (half a GB each) and the GPU stays free for the desktop;
# everything else spreads over every worker (`--dist loadgroup` in pyproject; the group
# is the mark below). Measured 2026-09-20 on the 3090 Ti with twelve workers: 4 groups
# 124 s, 8 groups 80 s, all twelve 70 s -- and it was the voice models, loaded in every
# worker by the panel tests, that had the GPU at 18 GB and the machine lagging (Tai:
# "the rest should run on all 24"); they no longer load (see PanelEngine._voice_settings).
EMBEDDER_WORKERS = int(os.environ.get("SIDECOACH_TEST_EMBEDDER_WORKERS", "8"))
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


def _live_panel_up() -> bool:
    try:
        socket.create_connection(("127.0.0.1", LIVE_PANEL_PORT), timeout=0.5).close()
    except OSError:
        return False
    return True


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
    # No panel built by a test loads Kokoro or Parakeet; the one that wants the real recogniser
    # says so (PanelEngine(voice_models=True)).
    os.environ.setdefault("SIDECOACH_VOICE_MODELS", "0")
    if os.environ.get("SIDECOACH_TEST_ALONGSIDE") == "1":
        return
    if _live_panel_up():
        pytest.exit(
            f"a sidecoach panel is serving on :{LIVE_PANEL_PORT} (a live game?): the suite "
            "would starve it. Wait for the session to end, or set "
            "SIDECOACH_TEST_ALONGSIDE=1 if that port is not a live game.",
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
