"""The suite's own caches never change a result: the frames the renderer keeps
(tests/synth.py `Kept`) and the embedding net's outputs kept across runs (tests/netcache.py).
A kept result is exactly the fresh one; anything that changes the result is a new key;
the bypass and rebuild switches do what they say; a broken store costs speed, not answers."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

import tests.netcache as netcache
import tests.synth as synth
from tests.fakes import HANG_S

# -- the renderer's frames ------------------------------------------------------------


@pytest.fixture(autouse=True)
def _own_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test here its own empty frame caches: what the process's other tests drew is
    not this test's hit (a cold run failed on a frame another test had kept, 2026-10-02)."""
    monkeypatch.setattr(synth, "_BOARDS", synth.Kept(192 * synth.MB))
    monkeypatch.setattr(synth, "_BACKGROUNDS", synth.Kept(64 * synth.MB))
    # The cache's own tests read and write in its default mode, whatever mode the run was
    # started in: with SIDECOACH_TEST_CACHE=rebuild set for the whole run, four of them
    # counted every row as run again and failed (2026-10-02). A test that wants a mode sets it.
    monkeypatch.delenv("SIDECOACH_TEST_CACHE", raising=False)
    monkeypatch.setattr(synth.Kept, "enabled", True)  # read from the variable at import


def _fresh(**kw: Any) -> np.ndarray:
    enabled = synth.Kept.enabled
    synth.Kept.enabled = False
    try:
        return synth.synthetic_board(**kw)
    finally:
        synth.Kept.enabled = enabled


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"theme": "wood", "style": "cburnett", "cell": 48, "coordinates": True, "seed": 3},
        {"highlight": chess.Move.from_uci("e2e4"), "arrows": (chess.Move.from_uci("g1f3"),)},
    ],
    ids=["default", "textured-real-set", "decorated"],
)
def test_a_kept_frame_is_the_frame_drawn_afresh(kw: dict[str, Any]) -> None:
    first = synth.synthetic_board(**kw)
    hits = synth._BOARDS.hits
    again = synth.synthetic_board(**kw)
    assert synth._BOARDS.hits == hits + 1, "asked again: read back, not drawn"
    assert np.array_equal(first, again) and np.array_equal(again, _fresh(**kw))


def test_each_caller_draws_on_its_own_copy() -> None:
    first = synth.synthetic_board(seed=11)
    first[:] = 0
    again = synth.synthetic_board(seed=11)
    assert again.flags.writeable and again.any(), "the kept frame was not drawn on"
    background = synth.busy_background(300, 200, 5)
    background[:] = 0
    assert synth.busy_background(300, 200, 5).any()


def test_whatever_changes_the_pixels_is_a_new_frame() -> None:
    base = chess.Board()
    moved = base.copy()
    moved.push_san("e4")
    to_move = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 0 1")
    other_side = chess.Board("4k3/8/8/8/8/8/4q3/4K3 b - - 0 1")
    changes = [
        ({"board": base}, {"board": moved}),
        ({"theme": "green"}, {"theme": "brown"}),
        ({"seed": 1}, {"seed": 2}),
        ({"style": "classic"}, {"style": "bold"}),
        ({"cell": 64}, {"cell": 56}),
        # The same pieces, the other side to move: the glow is under the king in check only.
        ({"board": to_move, "check": True}, {"board": other_side, "check": True}),
    ]
    for one, other in changes:
        a = synth.synthetic_board(**one)
        misses = synth._BOARDS.misses
        b = synth.synthetic_board(**other)  # never drawn before in this test: drawn now
        assert synth._BOARDS.misses == misses + 1, (one, other)
        assert not np.array_equal(a, b), (one, other)
        assert np.array_equal(b, _fresh(**other)), other


def _geometry_frames() -> list[dict[str, Any]]:
    """Frames as the geometry sweep draws them: 1920x1200, boards of several sizes at odd
    places, partly off the screen, a dragged piece reaching past the board, labels."""
    moved = chess.Board()
    moved.push_san("e4")
    return [
        {"width": 1920, "height": 1200, "x0": 37, "y0": 411, "cell": 71, "seed": 3},
        {"width": 1920, "height": 1200, "x0": 1500, "y0": 900, "cell": 90, "board": moved},
        {"width": 1920, "height": 1200, "x0": -60, "y0": -30, "cell": 48, "coordinates": True},
        {
            "width": 1920,
            "height": 1200,
            "x0": 600,
            "y0": 300,
            "cell": 64,
            "dragging": (chess.E2, chess.E4, 0.4, -0.9),
            "theme": "wood",
            "style": "cburnett",
        },
    ]


def test_a_frame_kept_as_its_board_is_the_frame_drawn_afresh_and_a_fraction_of_it() -> None:
    """A board frame is kept as the rectangle where it differs from its background: put back
    together it is the frame, pixel for pixel (its SHA-256 the fresh one's), and it costs a
    fraction of the frame's memory."""
    import hashlib

    for kw in _geometry_frames():
        first = synth.synthetic_board(**kw)
        again = synth.synthetic_board(**kw)
        fresh = _fresh(**kw)
        digest = hashlib.sha256(fresh.tobytes()).hexdigest()
        assert hashlib.sha256(first.tobytes()).hexdigest() == digest, kw
        assert hashlib.sha256(again.tobytes()).hexdigest() == digest, kw
        assert again.flags.writeable and again.shape == (1200, 1920, 3)
    assert synth._BOARDS.hits == len(_geometry_frames())
    whole = 1920 * 1200 * 3
    assert synth._BOARDS.bytes < 0.25 * whole * len(_geometry_frames()), synth._BOARDS.bytes


def test_the_frames_kept_stay_within_their_budget() -> None:
    """The oldest frames go first so that what is kept stays under the budget, however many
    frames are drawn; a frame asked for after it went is drawn again, the same pixels."""
    synth._BOARDS = synth.Kept(3 * synth.MB)  # the fixture puts the suite's back
    kws: list[dict[str, Any]] = [
        {"width": 1920, "height": 1200, "x0": 100 + 37 * i, "y0": 200, "cell": 80, "seed": 1}
        for i in range(12)
    ]
    first = [synth.synthetic_board(**kw) for kw in kws]
    assert 0 < synth._BOARDS.bytes <= 3 * synth.MB
    assert len(synth._BOARDS.frames) < len(kws), "some went"
    misses = synth._BOARDS.misses
    assert np.array_equal(synth.synthetic_board(**kws[0]), first[0])
    assert synth._BOARDS.misses == misses + 1, "the oldest went, and is drawn again"
    assert np.array_equal(synth.synthetic_board(**kws[0]), first[0])
    assert synth._BOARDS.hits >= 1


def test_with_the_cache_off_every_frame_is_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(synth.Kept, "enabled", False)
    hits = synth._BOARDS.hits
    synth.synthetic_board(seed=21)
    synth.synthetic_board(seed=21)
    assert synth._BOARDS.hits == hits


# -- the embedding net's outputs ------------------------------------------------------


class _Net:
    """A stand-in session: each row's output is a function of the row, counted."""

    def __init__(self, width: int = 4) -> None:
        self.width = width
        self.rows_run = 0

    def run(self, names: Any, feed: dict[str, np.ndarray], options: Any = None) -> list:
        cells = feed[netcache.INPUT]
        self.rows_run += len(cells)
        flat = cells.reshape(len(cells), -1).astype(np.float32)
        return [np.stack([flat.sum(1) * (k + 1) for k in range(self.width)], axis=1)]

    def get_inputs(self) -> list:
        return [type("I", (), {"name": netcache.INPUT})()]

    def get_outputs(self) -> list:
        return [object()]


def _cells(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, 3, 4, 4), dtype=np.float32)


def _session(tmp_path: Path, net: _Net, prefix: bytes = b"p" * 16) -> netcache.CachedSession:
    return netcache.CachedSession(net, prefix, netcache.Store(tmp_path / "net.sqlite"))


def _run(session: netcache.CachedSession, cells: np.ndarray) -> np.ndarray:
    return session.run(None, {netcache.INPUT: cells})[0]


def test_a_row_run_once_is_read_back_the_same(tmp_path: Path) -> None:
    net = _Net()
    cells = _cells(5)
    first = _run(_session(tmp_path, net), cells)
    assert net.rows_run == 5
    again = _run(_session(tmp_path, net), cells)  # another process: a new session, one store
    assert net.rows_run == 5, "every row kept"
    assert np.array_equal(first, again) and np.array_equal(
        first, _Net().run(None, {"cells": cells})[0]
    )


def test_only_the_rows_not_kept_are_run(tmp_path: Path) -> None:
    net = _Net()
    session = _session(tmp_path, net)
    cells = _cells(4)
    _run(session, cells)
    changed = cells.copy()
    changed[2, 0, 0, 0] += 1.0  # one value of one row
    out = _run(session, changed)
    assert net.rows_run == 4 + 1
    assert np.array_equal(out, _Net().run(None, {"cells": changed})[0])


def test_another_model_runtime_or_device_is_another_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnxruntime

    model = tmp_path / "m.onnx"
    model.write_bytes(b"model one")
    base = netcache.prefix_for(model, "cpu")
    assert netcache.prefix_for(model, "cpu") == base
    assert netcache.prefix_for(model, "cuda") != base
    monkeypatch.setattr(onnxruntime, "__version__", "0.0.0-other")
    assert netcache.prefix_for(model, "cpu") != base
    monkeypatch.undo()
    model.write_bytes(b"model two")
    retrained = netcache.prefix_for(model, "cpu")
    assert retrained != base
    # Its store is a file of its own, and the old one goes when the new one is first made.
    folder = tmp_path / "cache"
    old = netcache.store_for(base, folder)
    old.put({b"k": b"v"})
    old.close()
    assert (folder / f"net-{base.hex()}.sqlite").exists()
    netcache.store_for(retrained, folder)
    assert not (folder / f"net-{base.hex()}.sqlite").exists()
    net = _Net()
    cells = _cells(3)
    _run(netcache.CachedSession(net, base, netcache.Store(tmp_path / "s.sqlite")), cells)
    _run(netcache.CachedSession(net, retrained, netcache.Store(tmp_path / "s.sqlite")), cells)
    assert net.rows_run == 6, "a kept output of another model is never read"


def test_the_switches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cells = _cells(3)
    net = _Net()
    monkeypatch.setenv("SIDECOACH_TEST_CACHE", "0")
    _run(_session(tmp_path, net), cells)
    _run(_session(tmp_path, net), cells)
    assert net.rows_run == 6, "off: every row run, nothing kept"
    monkeypatch.setenv("SIDECOACH_TEST_CACHE", "1")
    _run(_session(tmp_path, net), cells)
    assert net.rows_run == 9, "nothing was kept while off"
    monkeypatch.setenv("SIDECOACH_TEST_CACHE", "rebuild")
    _run(_session(tmp_path, net), cells)
    assert net.rows_run == 12, "rebuild runs every row again"
    monkeypatch.setenv("SIDECOACH_TEST_CACHE", "verify")
    _run(_session(tmp_path, net), cells)
    assert net.rows_run == 15
    # A kept output that is not the net's own fails a verifying run.
    session = _session(tmp_path, net)
    key = session._key(cells[0])
    session.store.put({key: np.zeros(4, dtype=np.float32).tobytes()})
    with pytest.raises(AssertionError, match="differs"):
        _run(session, cells)
    monkeypatch.setenv("SIDECOACH_TEST_CACHE", "sometimes")
    with pytest.raises(ValueError):
        _run(session, cells)


def test_a_store_that_cannot_be_read_costs_speed_not_answers(tmp_path: Path) -> None:
    broken = tmp_path / "net.sqlite"
    broken.write_bytes(b"this is not a database" * 100)
    net = _Net()
    cells = _cells(3)
    out = _run(netcache.CachedSession(net, b"p" * 16, netcache.Store(broken)), cells)
    assert np.array_equal(out, _Net().run(None, {"cells": cells})[0]) and net.rows_run == 3


def test_workers_share_one_store(tmp_path: Path) -> None:
    """Several processes' connections to one new file, opening it at the same moment (as the
    workers do on a cold cache), then writing and reading at once: every answer right, no
    connection given up on, and every row in the store afterwards. (When each connection
    set the file up itself, a setup found it locked by another's and that worker cached
    nothing: 32 of 48 rows run again, 2026-10-02.)"""
    cells = [_cells(8, seed) for seed in range(6)]
    errors: list[BaseException] = []
    stores: list[netcache.Store] = []
    together = threading.Barrier(6)

    def worker(i: int) -> None:
        try:
            store = netcache.Store(tmp_path / "n.sqlite")
            stores.append(store)
            session = netcache.CachedSession(_Net(), b"p" * 16, store)
            together.wait(HANG_S)
            for batch in (cells[i], cells[(i + 1) % 6], cells[i]):
                assert np.array_equal(_run(session, batch), _Net().run(None, {"cells": batch})[0])
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert not any(store.broken for store in stores), "a worker gave up on the store"
    net = _Net()
    late = netcache.CachedSession(net, b"p" * 16, netcache.Store(tmp_path / "n.sqlite"))
    for batch in cells:
        _run(late, batch)
    assert net.rows_run == 0, "every row was kept by the workers"


def test_the_real_net_reads_back_its_own_bits(tmp_path: Path) -> None:
    """The embedding net on the CPU, one thread, as the suite runs it: a batch kept and read
    back is bit for bit the net's; and only that net, on the CPU, is wrapped."""
    from sidecoach.vision.embedding import MODEL_PATH
    from sidecoach.voice.runtime import session as make

    plain, where = make(MODEL_PATH, "cpu", intra_op_num_threads=1)
    plain = getattr(plain, "_session", plain)  # the suite's own wrapper, unwrapped
    rows = np.random.default_rng(3).random((3, 3, 224, 224), dtype=np.float32) * 255
    want = plain.run(None, {netcache.INPUT: rows})[0]
    prefix = netcache.prefix_for(MODEL_PATH, "cpu")
    kept = netcache.CachedSession(plain, prefix, netcache.Store(tmp_path / "n.sqlite"))
    assert np.array_equal(_run(kept, rows), want)
    again = netcache.CachedSession(plain, prefix, netcache.Store(tmp_path / "n.sqlite"))
    assert np.array_equal(_run(again, rows[::-1]), want[::-1]) and again.hits == 3
    assert isinstance(netcache.wrap((plain, where), MODEL_PATH, "cpu")[0], netcache.CachedSession)
    assert netcache.wrap((plain, where), MODEL_PATH, "cuda")[0] is plain
    other = _Net()
    other.get_inputs = lambda: [type("I", (), {"name": "audio"})()]  # type: ignore[method-assign]
    assert netcache.wrap((other, where), MODEL_PATH, "cpu")[0] is other
