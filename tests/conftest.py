"""Suite-wide options: `--seeds N` (or `--seeds A-B`) widens the random-game sweeps
(tests/test_stress.py, tests/test_board_geometry.py) and `--sweep` adds the cases marked
`sweep` (every lichess set, every board size): the default run keeps a representative few of
each and is fast (Tai, 2026-10-01); the guard that keeps the suite off a live session; the
workers started together; the vision tests kept together a module at a time; and
`make_panel`, the panel most panel tests start from."""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import socket
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from sidecoach.config import Settings
    from sidecoach.ui.panel import PanelEngine

# The tests that run the embedding net (vision/embedding.py: the synthetic vision suite, the
# panel over rendered games) go to the workers a module at a time, in chunks of this many
# consecutive tests: a module's tests read the same themes and sets, and dealt out one by one
# every worker learned them again (test_cold_start: 6.8 s alone, 23 s summed over the
# workers; measured 2026-10-01). Everything else spreads over every worker (`--dist
# loadgroup` in pyproject; the group is the mark below).
VISION_CHUNK = 6
# Test modules that load the net without the `vision` mark. A test elsewhere that loads it
# -- a panel session started, which builds the stages -- is marked `embedder` instead, so
# the rest of its module runs on every core.
EMBEDDER_MODULES = {
    "test_theme_cache",
}


def _start_workers_together() -> None:
    """xdist starts its workers one after another (NodeManager.setup_nodes), a Python start
    each, 0.9 s on this machine: twelve took eight seconds before the first test, and every
    worker waited for the last. Started from threads they come up together (2026-10-01: one
    small file on 12 workers 9.6 s -> 4.4 s). xdist 3's own code, threaded; any other version
    is left alone."""
    try:
        import xdist
        from xdist import workermanage
    except ImportError:
        return
    if not xdist.__version__.startswith("3."):
        return

    def setup_nodes(self: Any, putevent: Callable[..., None]) -> list[Any]:
        self.config.hook.pytest_xdist_setupnodes(config=self.config, specs=self.specs)
        self.trace("setting up nodes")
        with ThreadPoolExecutor(max_workers=max(1, len(self.specs))) as pool:
            return list(pool.map(lambda spec: self.setup_node(spec, putevent), self.specs))

    workermanage.NodeManager.setup_nodes = setup_nodes  # type: ignore[method-assign]


if not os.environ.get("PYTEST_XDIST_WORKER"):
    _start_workers_together()

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

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.2) as r:
            return bool(json.loads(r.read().decode("utf-8")).get("running"))
    except Exception:  # noqa: BLE001 - not a panel, or not answering
        return False


def _live_panel_up() -> bool:
    # A port on this machine answers at once; on Windows a closed one sat out the whole
    # timeout, 0.5 s in the controller and in every worker (2026-10-01).
    try:
        socket.create_connection(("127.0.0.1", LIVE_PANEL_PORT), timeout=0.05).close()
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
    config.addinivalue_line(
        "markers", "embedder: loads the embedding net outside the vision suite (tests/conftest.py)"
    )
    config.addinivalue_line(
        "markers", "sweep: one more case of a sweep, run with --sweep (tests/conftest.py)"
    )
    _below_normal_priority()  # the controller and, since this runs in each, every worker
    if os.environ.get("PYTEST_XDIST_WORKER"):
        # One OpenCV thread a worker: every worker's own pool, one thread a core each, had
        # two dozen workers' pools fighting over the cores.
        import cv2

        cv2.setNumThreads(1)
    # No panel built by a test loads Kokoro or Parakeet; the one that wants the real recogniser
    # says so (PanelEngine(voice_models=True)).
    os.environ.setdefault("SIDECOACH_VOICE_MODELS", "0")
    # The embedding net on the CPU: a dozen workers each starting a CUDA session took 4 s for
    # a worker's first vision test, against 1.4 s on the CPU, and a test alone is no slower
    # there (2026-10-01). The GPU path keeps one test of its own (-m slow, test_pieces).
    os.environ.setdefault("SIDECOACH_EMBED_DEVICE", "cpu")
    # A Settings made by a test has no file of its own; a save from it (set_record, the
    # settings API) would land on `sidecoach.toml` in the working directory -- the
    # developer's -- as one did on 2026-09-22. It lands in a scratch file instead.
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="sidecoach-tests-")) / "sidecoach.toml"
    os.environ["SIDECOACH_CONFIG"] = str(scratch)
    if os.environ.get("SIDECOACH_TEST_ALONGSIDE") == "1" or os.environ.get("PYTEST_XDIST_WORKER"):
        return  # the controller has looked, once
    if _live_panel_up():
        pytest.exit(
            f"a sidecoach panel is serving on :{LIVE_PANEL_PORT}, or the desktop app has a "
            "session running (a live game?): the suite would starve it. Wait for the session "
            "to end, or set SIDECOACH_TEST_ALONGSIDE=1 if it is not a live game.",
            returncode=3,
        )


@pytest.hookimpl(tryfirst=True)  # before xdist's own, which reads the marks on the worker
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not config.getoption("--sweep"):
        # A sweep's other cases only when asked: the default run keeps its representative few.
        swept = [item for item in items if item.get_closest_marker("sweep")]
        if swept:
            config.hook.pytest_deselected(items=swept)
            items[:] = [item for item in items if not item.get_closest_marker("sweep")]
    counted: dict[str, int] = {}
    for item in items:
        module = item.path.stem
        marked = item.get_closest_marker("vision") or item.get_closest_marker("embedder")
        if marked is None and module not in EMBEDDER_MODULES:
            continue
        # A module's consecutive tests together, VISION_CHUNK at a time: they share what a
        # worker learns (the themes, the sets), and a long module still spreads.
        index = counted.get(module, 0)
        counted[module] = index + 1
        item.add_marker(pytest.mark.xdist_group(f"{module}-{index // VISION_CHUNK}"))


# A run of this many test files or fewer goes on one process: the 24 workers' start (each
# importing the app) cost six seconds before the first test, and a few files run in one or two.
FEW_FILES = 3


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """`-n auto` (pyproject): every core for the suite, none for a targeted run."""
    named = [a for a in config.args if a.split("::")[0].endswith(".py")]
    if (
        named
        and len(named) == len(config.args)
        and len({a.split("::")[0] for a in named}) <= FEW_FILES
    ):
        return 0
    return None  # xdist's own count, every core


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--sweep",
        action="store_true",
        default=False,
        help="run every case of the sweeps (the cases marked sweep), not the representative few",
    )
    parser.addoption(
        "--seeds",
        action="store",
        default="",
        help="how many seeded random games the stress tests play, or a range A-B of seeds "
        "(default: a fixed handful)",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """A test taking `seed` plays its module's few seeds (the module's SEEDS), or the ones
    `--seeds` asks for."""
    if "seed" not in metafunc.fixturenames:
        return
    spec = str(metafunc.config.getoption("--seeds") or "")
    if not spec:
        assert metafunc.module is not None
        seeds = list(metafunc.module.SEEDS)
    elif "-" in spec:  # "251-1000": a range of seeds
        lo, hi = (int(x) for x in spec.split("-", 1))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = list(range(1, int(spec) + 1))
    metafunc.parametrize("seed", seeds)


@pytest.fixture
def make_panel(tmp_path: pathlib.Path) -> Callable[..., PanelEngine]:
    """Makes a panel with its data under `tmp_path`: the settings' defaults unless a test
    gives its own (no voice models: SIDECOACH_VOICE_MODELS=0 above), and no engine, no coach
    and no frames unless it asks for them."""
    from sidecoach.config import Settings
    from sidecoach.ui.panel import PanelEngine

    def make(settings: Settings | None = None, **kw: Any) -> PanelEngine:
        kw = {"analysis": False, "coach": False, "frame_source": iter(()), **kw}
        return PanelEngine(settings or Settings(data_dir=tmp_path), **kw)

    return make
