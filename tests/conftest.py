"""How the suite runs (Tai, 2026-10-02: the whole suite in under two minutes, on at most 70% of
the machine, nothing cut to get there):

- **A CPU budget.** `-n auto` starts CPU_SHARE of the logical processors as workers, less
  RESERVED for the controller and the threads a test starts of its own, and every process
  computes on one thread: the BLAS/OpenMP pools, OpenCV, onnxruntime's sessions and Stockfish
  (`_one_thread_each`). A worker per core, each with pools a core wide, had peaked at the
  whole machine (2026-10-02: 12 workers, 100% for seconds at a time).
- **Longest first.** The tests go to the workers longest first by `tests/durations.json`
  (tests/scheduling.py); `--store-durations` measures them again.
- **A run of a few files on one process**, the workers started together, and the guard that
  keeps the suite off a live session.
- `--seeds N` (or `--seeds A-B`) widens the seeded sweeps (tests/test_stress.py,
  tests/test_board_geometry.py, tests/test_premove_sweep.py), and `make_panel` is the panel
  most panel tests start from.
"""

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

# One thread for every native pool, set before anything imports numpy or OpenCV; the workers
# inherit it from the controller.
for _pool in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_pool] = "1"

import pytest  # noqa: E402

if TYPE_CHECKING:
    from sidecoach.config import Settings
    from sidecoach.ui.panel import PanelEngine

# The share of the machine's logical processors the suite may keep busy (Tai, 2026-10-02).
CPU_SHARE = 0.70
# Processors of that share left to the controller and to the threads a test starts besides
# its worker's (a panel's pipeline and analyser, a server, a Stockfish search).
RESERVED = 2
# A run of this many test files or fewer goes on one process: starting the workers (each
# importing the app) costs seconds before the first test, and a few files run in one or two.
FEW_FILES = 3


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


def _one_thread_each() -> None:
    """Every computing pool of a test process one thread wide (the BLAS/OpenMP ones are set
    above, from the environment):

    - OpenCV's own pool, a thread a core by default;
    - onnxruntime's sessions (the embedding net asks for two intra-op threads, which spin
      while they wait), made through `sidecoach.voice.runtime.session`;
    - Stockfish's search threads (`Threads`, 4 in the default settings, 2 in the engine
      tests): a search on one thread is also the same search every run.

    Only the threads change: every session and engine is made and asked as the app makes
    and asks it. The embedding net's session also keeps its outputs across runs
    (tests/netcache.py)."""
    import chess.engine
    import cv2

    import tests.netcache as netcache
    from sidecoach.voice import runtime

    cv2.setNumThreads(1)

    make_session = runtime.session

    def session(path: Any, device: str, **options: Any) -> Any:
        options.update(intra_op_num_threads=1, inter_op_num_threads=1)
        made = make_session(path, device, **options)
        return netcache.wrap(made, path, made[1].name)

    runtime.session = session  # type: ignore[assignment]

    configure = chess.engine.SimpleEngine.configure

    def one_thread(self: chess.engine.SimpleEngine, options: Any) -> None:
        if "Threads" in options:
            options = {**options, "Threads": 1}
        configure(self, options)

    chess.engine.SimpleEngine.configure = one_thread  # type: ignore[method-assign]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "embedder: loads the embedding net outside the vision suite (tests/conftest.py)"
    )
    _below_normal_priority()  # the controller and, since this runs in each, every worker
    if os.environ.get("PYTEST_XDIST_WORKER") or config.getvalue("dist") == "no":
        _one_thread_each()  # a process that runs tests (the controller of workers runs none)
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


def pytest_xdist_auto_num_workers(config: pytest.Config) -> int | None:
    """`-n auto` (pyproject): the CPU budget's workers for the suite, none for a run of a few
    files."""
    named = [a for a in config.args if a.split("::")[0].endswith(".py")]
    if (
        named
        and len(named) == len(config.args)
        and len({a.split("::")[0] for a in named}) <= FEW_FILES
    ):
        return 0
    return max(1, int((os.cpu_count() or 1) * CPU_SHARE) - RESERVED)


def pytest_xdist_make_scheduler(config: pytest.Config, log: Any) -> Any:
    """`--dist loadgroup` (pyproject), dealt longest first (tests/scheduling.py)."""
    if config.getvalue("dist") != "loadgroup":
        return None
    from tests.scheduling import LongestFirst

    return LongestFirst(config, log)


_measured: dict[str, float] = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Each test's setup, call and teardown summed, for --store-durations (in the
    controller, from every worker's reports)."""
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return
    _measured[report.nodeid] = _measured.get(report.nodeid, 0.0) + report.duration


def pytest_sessionfinish(session: pytest.Session) -> None:
    if os.environ.get("PYTEST_XDIST_WORKER") or not session.config.getoption("--store-durations"):
        return
    from tests.scheduling import store_durations

    store_durations(_measured)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seeds",
        action="store",
        default="",
        help="how many seeded random games the stress tests play, or a range A-B of seeds "
        "(default: a fixed handful)",
    )
    parser.addoption(
        "--store-durations",
        action="store_true",
        default=False,
        help="write this run's test durations to tests/durations.json, the order the "
        "workers are dealt the tests in (tests/scheduling.py)",
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
