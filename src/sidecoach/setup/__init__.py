"""First-run setup: what the app needs beyond its code, fetched into the data directory.

The installer carries the code and nothing else (docs/app-plan.md, section 6, step 4).
At first run the sidecar fetches, with one progress bar for the lot: the Kokoro voice
(two files), Parakeet and the Silero gate (through onnx-asr's own hub downloads,
into a Hugging Face cache under the data directory), Stockfish (the latest release), the
openings tables, and a sample of the Lichess puzzles. Each item knows whether it is
present, how big it is, and how to fetch itself; `Setup` runs the missing ones on a
thread and reports through `/api/setup/status`. The CUDA pack is an item too, offered
only when an eligible GPU is found (section 7; setup/gpu.py).
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import os
import platform
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sidecoach.config import Settings

USER_AGENT = "sidecoach-setup"
KOKORO_FILES = {
    "kokoro-v1.0.onnx": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        325_532_387,
    ),
    "voices-v1.0.bin": (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        28_214_398,
    ),
}
# The release the app takes, pinned as the phone (tools/fetch_stockfish.py) and the tests
# are: "latest" moved under them, a renamed asset broke new installs, and the API that
# named it is rate-limited for anonymous callers. The download URL needs no API call.
STOCKFISH_TAG = "sf_19"
STOCKFISH_RELEASES = "https://github.com/official-stockfish/Stockfish/releases/download"


def stockfish_url(name: str, tag: str = STOCKFISH_TAG) -> str:
    return f"{STOCKFISH_RELEASES}/{tag}/{name}"


def stockfish_assets(system: str | None = None, machine: str | None = None) -> tuple[str, ...]:
    """The release assets to take for this platform, best first. Stockfish 19 ships one
    `universal` build per platform that picks the CPU code path at run time
    (stockfish-windows-x86-64-universal.zip, stockfish-linux-x86-64-universal.tar.gz,
    ...-arm64-...); the older per-ISA names are kept as fallbacks for older releases."""
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86-64"
    if system == "windows":
        return (f"stockfish-windows-{arch}-universal.zip", f"stockfish-windows-{arch}-avx2.zip")
    if system == "darwin":
        return ("stockfish-macos-universal.tar.gz", f"stockfish-macos-{arch}.tar")
    return (
        f"stockfish-linux-{arch}-universal.tar.gz",
        f"stockfish-ubuntu-{arch}-avx2.tar",
        f"stockfish-ubuntu-{arch}.tar",
    )


def unpack_stockfish(archive: Path, dest: Path) -> list[Path]:
    """Extract a release archive into `dest` and return the executables, made executable."""
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
        return sorted(dest.rglob("stockfish*.exe"))
    import tarfile

    with tarfile.open(archive) as t:
        t.extractall(dest, filter="data")
    found = sorted(
        f
        for f in dest.rglob("stockfish-*")
        if f.is_file() and f.suffix not in (".tar", ".gz", ".zip", ".exe")
    )
    for exe in found:
        exe.chmod(0o755)
    return found


def install_stockfish(
    dest: Path, staging: Path, progress: Progress | None = None, tag: str = STOCKFISH_TAG
) -> list[Path]:
    """The pinned release's build for this platform, into `dest`. Downloaded and unpacked
    in `staging` (outside `dest`, where the finder looks) and moved in only once an
    executable came out: an extraction cut short left a half-written exe the finder took."""
    archive: Path | None = None
    for name in stockfish_assets():
        try:
            archive = download(stockfish_url(name, tag), staging / name, progress)
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
    if archive is None:
        raise RuntimeError(f"no Stockfish build for {platform.system()} in {tag}")
    unpacked = staging / "unpacked"
    if unpacked.exists():
        shutil.rmtree(unpacked)
    found = unpack_stockfish(archive, unpacked)
    if not found:
        raise RuntimeError(f"{archive.name} holds no Stockfish executable")
    dest.mkdir(parents=True, exist_ok=True)
    for child in list(unpacked.iterdir()):
        where = dest / child.name
        if where.is_dir():
            shutil.rmtree(where)
        elif where.exists():
            where.unlink()
        shutil.move(str(child), str(where))
    archive.unlink()
    unpacked.rmdir()
    return [dest / f.relative_to(unpacked) for f in found]


OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
OPENINGS_VOLUMES = ("a", "b", "c", "d", "e")
PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
PUZZLE_THEMES = [
    "fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "trappedPiece",
    "deflection", "attraction", "interference", "clearance", "exposedKing", "sacrifice",
    "advancedPawn", "promotion", "xRayAttack", "capturingDefender", "doubleCheck",
    "backRankMate", "smotheredMate", "anastasiaMate", "arabianMate", "bodenMate",
    "hookMate", "dovetailMate", "mateIn1", "mateIn2", "mateIn3",
]  # fmt: skip
PUZZLES_PER_THEME = 200
PUZZLES_MAX_SCAN = 600_000
# The sample's size for the progress bar: about what the scan downloads.
PUZZLES_BYTES = 30_000_000

Progress = Callable[[int, int | None], None]  # (bytes done, bytes total or None)


def hf_home(settings: Settings) -> Path:
    """The Hugging Face cache the speech-recognition models go to: under the data directory
    in the app, so the app owns what it downloaded; the user's own cache otherwise."""
    if os.environ.get("SIDECOACH_DATA_DIR"):
        return settings.data_dir / "models" / "hf"
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def apply_hf_home(settings: Settings) -> None:
    """Point onnx-asr's cache at the app's folder, and keep the hub offline: nothing is
    downloaded behind the person's back -- only the setup fetches, with its meter (the
    warm-up of a fresh install pulled 2.4 GB on its own, 2026-09-21). A no-op outside
    the app."""
    if os.environ.get("SIDECOACH_DATA_DIR"):
        # The app's own cache, whatever the person's environment says: an HF_HOME or
        # HF_HUB_CACHE of their own sent the download elsewhere while `stt_present` looked
        # here, and speech recognition stayed "missing" for good.
        home = hf_home(settings)
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(home)
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            os.environ.pop(name, None)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


@contextlib.contextmanager
def hub_online() -> Any:
    """The hub allowed to download, for the setup's fetch only."""
    try:
        from huggingface_hub import constants  # type: ignore[import-not-found]
    except ImportError:
        yield
        return
    before = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = before


class Cancelled(Exception):
    """The setup's Cancel, raised from the progress callback of the item under way."""


class IncompleteDownload(RuntimeError):
    """The connection ended before the body did; the `.part` is kept for a resume."""


def _fetch_into(url: str, part: Path, done: int, progress: Progress | None) -> int:
    """One request for what `part` lacks (all of it when `done` is 0); the bytes it then
    holds. Raises `IncompleteDownload` when fewer bytes came than the server announced: a
    read returns b'' on an early close, which looked like the end of the file."""
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed hosts
        if done and response.status != 206:
            done = 0  # the server did not resume; start over
        length = response.headers.get("Content-Length")
        expected = int(length) if length and length.isdigit() else None
        total = (expected + done) if expected is not None else None
        mode = "ab" if done else "wb"
        received = 0
        with part.open(mode) as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                received += len(chunk)
                if progress:
                    progress(done, total)
        if expected is not None and received < expected:
            raise IncompleteDownload(f"{part.name}: {received} of {expected} bytes came")
    return done


def download(
    url: str,
    target: Path,
    progress: Progress | None = None,
    expect_sha256: str | None = None,
    expect_size: int | None = None,
) -> Path:
    """Fetch `url` to `target` through a temporary file, resuming a partial one, with
    progress callbacks; verified against `expect_sha256` and `expect_size` when given. A
    body cut short raises `IncompleteDownload` and never becomes `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    if expect_size is not None and done > expect_size:
        part.unlink()
        done = 0
    # A .part already whole (killed between the last write and the rename) is not fetched
    # again: a Range past its end is answered 416.
    if expect_size is None or done != expect_size:
        try:
            done = _fetch_into(url, part, done, progress)
        except urllib.error.HTTPError as exc:
            if exc.code != 416 or not done:
                raise
            # Range not satisfiable: the .part is whole when the server says it is exactly
            # this long ("bytes */<size>"); anything else starts over.
            whole = (exc.headers.get("Content-Range") or "").strip() == f"bytes */{done}"
            if not whole:
                part.unlink()
                done = _fetch_into(url, part, 0, progress)
    if expect_size is not None and done != expect_size:
        if done > expect_size:
            part.unlink()
        raise IncompleteDownload(f"{target.name}: {done} bytes, {expect_size} expected")
    if expect_sha256:
        digest = hashlib.sha256()
        with part.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != expect_sha256:
            part.unlink()
            raise RuntimeError(f"{target.name}: checksum mismatch")
    part.replace(target)
    return target


@dataclass
class Item:
    """One thing the app needs: how to tell it is there, how big it is, how to get it."""

    id: str
    title: str
    size: int  # bytes, an estimate for the progress bar
    present: Callable[[], bool]
    fetch: Callable[[Progress], None]
    optional: bool = False
    detail: str = ""


def _size_of(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths if p.exists())


def items(settings: Settings) -> list[Item]:
    """What this machine needs, in the order they are fetched."""
    out: list[Item] = []
    kokoro_dir = settings.voice_model_dir()

    def kokoro_whole(name: str, size: int) -> bool:
        path = kokoro_dir / name
        return path.is_file() and path.stat().st_size == size

    def kokoro_present() -> bool:
        return all(kokoro_whole(name, size) for name, (_, size) in KOKORO_FILES.items())

    def kokoro_fetch(progress: Progress) -> None:
        base = 0
        for name, (url, size) in KOKORO_FILES.items():
            if kokoro_whole(name, size):
                base += size
                continue
            download(
                url,
                kokoro_dir / name,
                lambda d, t, b=base: progress(b + d, None),
                expect_size=size,
            )
            base += size

    out.append(
        Item(
            "kokoro",
            "The coach's voice",
            sum(s for _, s in KOKORO_FILES.values()),
            kokoro_present,
            kokoro_fetch,
        )
    )

    def stockfish_present() -> bool:
        from sidecoach.config import find_stockfish

        return find_stockfish(settings) is not None

    def stockfish_fetch(progress: Progress) -> None:
        install_stockfish(settings.tools_dir, settings.data_dir / "tmp" / "stockfish", progress)

    out.append(
        Item(
            "stockfish",
            "The chess engine (Stockfish)",
            60_000_000,
            stockfish_present,
            stockfish_fetch,
        )
    )

    openings_dir = settings.data_dir / "openings"

    def openings_present() -> bool:
        return all((openings_dir / f"{v}.tsv").exists() for v in OPENINGS_VOLUMES)

    def openings_fetch(progress: Progress) -> None:
        for i, v in enumerate(OPENINGS_VOLUMES):
            target = openings_dir / f"{v}.tsv"
            if not target.exists():
                download(f"{OPENINGS_BASE}/{v}.tsv", target)
            progress((i + 1) * 100_000, None)

    out.append(Item("openings", "The opening names", 500_000, openings_present, openings_fetch))

    def stt_present() -> bool:
        home = hf_home(settings) / "hub"
        return any(
            home.glob(
                "models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/*/encoder-model*.onnx"
            )
        ) and any(home.glob("models--istupakov--silero-vad-onnx/snapshots/*/*.onnx"))

    def stt_fetch(progress: Progress) -> None:
        # onnx-asr downloads through the hub itself; the progress is the cache growing.
        apply_hf_home(settings)
        home = hf_home(settings)
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                # A Cancel raised here ends only the meter; the hub's fetch runs its course.
                with contextlib.suppress(Exception):
                    progress(_size_of([p for p in home.rglob("*") if p.is_file()]), None)
                time.sleep(1.0)

        threading.Thread(target=watch, daemon=True).start()
        try:
            from onnx_asr.loader import (  # type: ignore[import-not-found]
                create_asr_resolver,
                create_vad_resolver,
            )

            # The files only: loading the 0.6B model to get them took ~2.5 GB of memory.
            with hub_online():
                create_asr_resolver("nemo-parakeet-tdt-0.6b-v3").resolve_model()
                create_vad_resolver("silero").resolve_model()
        finally:
            stop.set()

    out.append(
        Item(
            "stt",
            "Speech recognition, to talk to the coach",
            2_450_000_000,
            stt_present,
            stt_fetch,
        )
    )

    puzzles_csv = settings.data_dir / "puzzles" / "sample.csv"

    def puzzles_present() -> bool:
        return puzzles_csv.exists()

    def puzzles_fetch(progress: Progress) -> None:
        try:
            import zstandard  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("the puzzle sample needs the zstandard package") from exc
        kept: dict[str, list[list[str]]] = {t: [] for t in PUZZLE_THEMES}
        seen: set[str] = set()
        scanned = 0
        request = urllib.request.Request(PUZZLES_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            reader = zstandard.ZstdDecompressor().stream_reader(response)
            rows = csv.reader(io.TextIOWrapper(reader, encoding="utf-8", newline=""))
            header = next(rows)
            for row in rows:
                scanned += 1
                if scanned % 20_000 == 0:
                    # In the item's own units (bytes), as the run's bar adds them up.
                    progress(scanned * PUZZLES_BYTES // PUZZLES_MAX_SCAN, PUZZLES_BYTES)
                themes = set(row[7].split())
                wanted = [
                    t for t in PUZZLE_THEMES if t in themes and len(kept[t]) < PUZZLES_PER_THEME
                ]
                if wanted and row[0] not in seen:
                    seen.add(row[0])
                    for t in wanted:
                        kept[t].append(row)
                if (
                    all(len(kept[t]) >= PUZZLES_PER_THEME for t in PUZZLE_THEMES)
                    or scanned >= PUZZLES_MAX_SCAN
                ):
                    break
        unique = {row[0]: row for rows_ in kept.values() for row in rows_}
        puzzles_csv.parent.mkdir(parents=True, exist_ok=True)
        with puzzles_csv.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerows(unique.values())

    out.append(
        Item(
            "puzzles",
            "A sample of the Lichess puzzles",
            PUZZLES_BYTES,
            puzzles_present,
            puzzles_fetch,
            optional=True,
        )
    )

    with contextlib.suppress(Exception):
        from sidecoach.setup.gpu import pack_item

        pack = pack_item(settings)
        if pack is not None:
            out.append(pack)
    return out


@dataclass
class Setup:
    """The setup's state and the thread that runs it, one per sidecar."""

    settings: Settings
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _state: dict[str, Any] = field(default_factory=dict)
    _cancel: threading.Event = field(default_factory=threading.Event)

    def status(self) -> dict[str, Any]:
        """Every item with whether it is present, and the run in progress if any."""
        rows = []
        for item in items(self.settings):
            with contextlib.suppress(Exception):
                rows.append(
                    {
                        "id": item.id,
                        "title": item.title,
                        "size": item.size,
                        "present": item.present(),
                        "optional": item.optional,
                        "detail": item.detail,
                    }
                )
        with self._lock:
            run = dict(self._state)
        needed = [r for r in rows if not r["present"] and not r["optional"]]
        return {
            "items": rows,
            "ready": not needed,
            "run": run,
            "data_dir": str(self.settings.data_dir),
        }

    def start(self, ids: list[str] | None = None) -> bool:
        """Fetch the items named (default: every missing one that is not optional) on a
        thread; False when a run is already going."""
        with self._lock:
            # "running" is claimed under the lock: the thread is made only after the items
            # are looked at, and a second click in that time started a second run into the
            # same .part files.
            if self._state.get("running") or (self._thread is not None and self._thread.is_alive()):
                return False
            self._cancel.clear()
            self._state = {
                "running": True,
                "item": None,
                "done": 0,
                "total": 0,
                "errors": {},
                "finished": [],
            }
        try:
            todo = [
                i
                for i in items(self.settings)
                if (ids is None and not i.optional and not i.present())
                or (ids and i.id in ids and not i.present())
            ]
        except BaseException:
            with self._lock:
                self._state["running"] = False
            raise
        total = sum(i.size for i in todo)
        with self._lock:
            self._state["total"] = total
            self._state["queue"] = [i.id for i in todo]
            self._state["titles"] = {i.id: i.title for i in todo}

        def run() -> None:
            base = 0
            for item in todo:
                if self._cancel.is_set():
                    break
                with self._lock:
                    self._state["item"] = item.id

                def progress(
                    done: int, _total: int | None, base: int = base, size: int = item.size
                ) -> None:
                    # Cancel stops the item under way at its next chunk; its .part stays
                    # for a resume.
                    if self._cancel.is_set():
                        raise Cancelled
                    with self._lock:
                        self._state["done"] = base + min(done, size)

                free = free_space(self.settings.data_dir)
                try:
                    if free is not None and free < item.size:
                        raise OSError(
                            f"not enough free space: {item.size // 1_000_000} MB needed, "
                            f"{free // 1_000_000} MB free"
                        )
                    item.fetch(progress)
                    with self._lock:
                        self._state["finished"].append(item.id)
                except Cancelled:
                    break
                except Exception as exc:  # noqa: BLE001 - reported, the next item goes on
                    with self._lock:
                        self._state["errors"][item.id] = f"{type(exc).__name__}: {exc}"
                base += item.size
                with self._lock:
                    self._state["done"] = base
            with self._lock:
                self._state["running"] = False
                self._state["item"] = None

        self._thread = threading.Thread(target=run, name="setup", daemon=True)
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def run_state(self) -> dict[str, Any]:
        """The fetch under way or last run, alone (the panel's Jobs button asks often; `status`
        also looks at every item on disk): empty when none has run this session."""
        with self._lock:
            return {**self._state, "titles": dict(self._state.get("titles", {}))}


def free_space(path: Path) -> int | None:
    with contextlib.suppress(Exception):
        return shutil.disk_usage(path if path.exists() else path.parent).free
    return None
