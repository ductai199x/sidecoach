"""The embedding net's outputs kept across test runs, by the exact input they came from.

A fifth of the suite's work was the net (2026-10-02: 12,499 cells through it, 84 s of CPU
in a run of 420), and half of those cells were the same pixels another test, or another
run, had already embedded. On the CPU the net is a pure function of its input: a cell's
embedding is the same bits whatever batch it runs in and on one thread or two (measured:
every row of a batch of 17 equal to the row run alone, and to two threads). So each cell's
output is kept under a digest of

- the model file's bytes, onnxruntime's version and the device (`CachedSession.prefix`),
- the input row itself: the cell after `CellEmbedder.prepare` and the batch layout, as
  float32 bytes -- whatever the app's code did to the crop is in it,

and a cell asked for again is read back instead of run. Everything the app does around the
net -- cropping, preparing, its own cache by crop, batching, the fallback off a failed
device -- still runs; only onnxruntime's arithmetic is not done twice. A changed model, a
new onnxruntime or a changed `prepare` is a new key, never a stale hit.

The store is SQLite under `tests/.cache/` (git-ignored): one file per prefix, safe for the
parallel workers (WAL; a write is one transaction), the files of other prefixes removed when
a new one is made. `SIDECOACH_TEST_CACHE`: `1` (default) reads and writes; `0` neither;
`rebuild` runs every cell again and writes it over the kept one; `verify` runs every cell
and fails on any kept output that differs. A store that cannot be opened or read is left
alone and the net runs: the cache can make a run faster, never different.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIR = Path(__file__).with_name(".cache")
INPUT = "cells"  # the embedding net's one input (vision/embedding.py)
MODES = ("1", "0", "rebuild", "verify")
# How a key is made from a row: a change of it is a new store, not a store of dead keys.
FORMAT = 2


def mode() -> str:
    value = os.environ.get("SIDECOACH_TEST_CACHE", "1")
    if value not in MODES:
        raise ValueError(f"SIDECOACH_TEST_CACHE must be one of {MODES}, got {value!r}")
    return value


def prefix_for(model: Path, device: str) -> bytes:
    """What besides the input decides the output: the model's bytes, the runtime, the device.
    (Read each time a session is made: once a process, and a model rewritten between two
    sessions is a new prefix.)"""
    import onnxruntime

    digest = hashlib.blake2b(digest_size=16)
    digest.update(model.read_bytes())
    digest.update(f"|onnxruntime {onnxruntime.__version__}|{device}|format {FORMAT}|".encode())
    return digest.digest()


class Store:
    """Rows of float32 by key, in one SQLite file a process opens once."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db: sqlite3.Connection | None = None
        self.broken = False

    def _create(self) -> None:
        """The file made whole before anyone opens it: written aside (WAL, which the file
        keeps, and its table), then linked into place, which fails when another process got
        there first. The workers starting together each opened a new file and set it up at
        once, and one setup could find the file locked by another (2026-10-02, 32 rows of 48
        not kept): now no connection ever sets a file up."""
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        aside = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.new")
        db = sqlite3.connect(aside)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                "CREATE TABLE IF NOT EXISTS rows (key BLOB PRIMARY KEY, value BLOB NOT NULL)"
                " WITHOUT ROWID"
            )
            db.commit()
        finally:
            db.close()
        try:
            os.link(aside, self.path)
        except FileExistsError:
            pass  # another process's file is the one: the same, made the same way
        finally:
            with contextlib.suppress(OSError):
                aside.unlink()

    def _open(self) -> sqlite3.Connection | None:
        if self._db is None and not self.broken:
            try:
                self._create()
                db = sqlite3.connect(self.path, timeout=30.0, check_same_thread=False)
                db.execute("PRAGMA synchronous=NORMAL")
                self._db = db
            except (OSError, sqlite3.Error):
                self.broken = True
        return self._db

    def _failed(self, exc: sqlite3.Error) -> None:
        """A busy file (another process writing past the 30 s wait) costs this read or write
        only; a file that is not a database ends the cache for this process."""
        if not isinstance(exc, sqlite3.OperationalError) or "locked" not in str(exc):
            self.broken = True

    def get(self, keys: list[bytes]) -> dict[bytes, bytes]:
        with self._lock:
            db = self._open()
            if db is None or not keys:
                return {}
            try:
                marks = ",".join("?" * len(keys))
                return dict(db.execute(f"SELECT key, value FROM rows WHERE key IN ({marks})", keys))
            except sqlite3.Error as exc:
                self._failed(exc)
                return {}

    def put(self, rows: dict[bytes, bytes]) -> None:
        with self._lock:
            db = self._open()
            if db is None or not rows:
                return
            try:
                with db:
                    db.executemany("INSERT OR REPLACE INTO rows VALUES (?, ?)", rows.items())
            except sqlite3.Error as exc:
                self._failed(exc)

    def close(self) -> None:
        with self._lock:
            if self._db is not None:
                self._db.close()
                self._db = None


_stores: dict[Path, Store] = {}


def store_for(prefix: bytes, folder: Path = CACHE_DIR) -> Store:
    """The process's store for `prefix`; a first open removes the files of other prefixes
    (an older model or runtime: never read again)."""
    path = folder / f"net-{prefix.hex()}.sqlite"
    if path not in _stores:
        if folder.is_dir():
            for old in folder.glob("net-*.sqlite*"):
                if not old.name.startswith(path.name):
                    # In use by a run of the old one: left for the next.
                    with contextlib.suppress(OSError):
                        old.unlink()
        _stores[path] = Store(path)
    return _stores[path]


class CachedSession:
    """An onnxruntime session whose `run` over the embedding net's input reads each row's
    output from the store when it was kept, and runs the rest in one batch."""

    def __init__(self, session: Any, prefix: bytes, store: Store) -> None:
        self._session = session
        self.prefix = prefix
        self.store = store
        self.hits = self.runs = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def _key(self, row: np.ndarray) -> bytes:
        # SHA-256: the processor's own instructions hash a 600 kB cell in a quarter of a
        # millisecond, twice as fast as BLAKE2 here (2026-10-02), against 7 ms to run it.
        digest = hashlib.sha256(self.prefix)
        digest.update(str((row.dtype.str, row.shape)).encode())
        digest.update(np.ascontiguousarray(row).data)
        return digest.digest()[:16]

    def run(self, output_names: Any, feed: dict[str, Any], run_options: Any = None) -> list:
        how = mode()
        cells = feed.get(INPUT) if isinstance(feed, dict) else None
        if how == "0" or cells is None or len(feed) != 1 or output_names is not None:
            return self._session.run(output_names, feed, run_options)
        keys = [self._key(row) for row in cells]
        kept = {} if how == "rebuild" else self.store.get(keys)
        todo = (
            list(range(len(keys)))
            if how == "verify"
            else [i for i, k in enumerate(keys) if k not in kept]
        )
        fresh: np.ndarray | None = None
        if todo:
            batch = np.ascontiguousarray(cells[todo])
            fresh = np.asarray(self._session.run(None, {INPUT: batch}, run_options)[0])
            self.runs += len(todo)
        width = fresh.shape[1:] if fresh is not None else None
        rows: list[np.ndarray | None] = [None] * len(keys)
        if fresh is not None:
            for j, i in enumerate(todo):
                rows[i] = fresh[j]
        if how == "verify":
            for i, k in enumerate(keys):
                row = rows[i]
                assert row is not None
                if k in kept:
                    old = np.frombuffer(kept[k], dtype=row.dtype).reshape(row.shape)
                    assert np.array_equal(old, row), "a kept output differs from the net's"
        for i, k in enumerate(keys):
            if rows[i] is None:
                rows[i] = np.frombuffer(kept[k], dtype=np.float32).reshape(width or (-1,))
                self.hits += 1
        if fresh is not None:
            self.store.put(
                {keys[i]: np.ascontiguousarray(fresh[j]).tobytes() for j, i in enumerate(todo)}
            )
        dtype = fresh.dtype if fresh is not None else np.float32
        return [np.stack([np.asarray(r, dtype=dtype) for r in rows])]


def wrap(made: tuple[Any, Any], path: Path, device_name: str) -> tuple[Any, Any]:
    """The session `sidecoach.voice.runtime.session` made, cached when it is the embedding
    net on the CPU (a GPU's output is the GPU's own; the voice models are not kept)."""
    session, where = made
    if mode() == "0" or device_name != "cpu":
        return made
    try:
        names = [i.name for i in session.get_inputs()]
        outputs = len(session.get_outputs())
    except Exception:  # noqa: BLE001 - not a session we know
        return made
    if names != [INPUT] or outputs != 1:
        return made
    prefix = prefix_for(Path(path), device_name)
    return CachedSession(session, prefix, store_for(prefix)), where
