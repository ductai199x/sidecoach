"""First-run setup: what the app needs beyond its code, fetched into the data directory.

The installer carries the code and nothing else (docs/app-plan.md, section 6, step 4).
At first run the sidecar fetches, with one progress bar for the lot: the Kokoro voice
(two files), the Parakeet ear and the Silero gate (through onnx-asr's own hub downloads,
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
import json
import os
import platform
import shutil
import threading
import time
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
STOCKFISH_API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
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

Progress = Callable[[int, int | None], None]  # (bytes done, bytes total or None)


def hf_home(settings: Settings) -> Path:
    """The Hugging Face cache the ear's models go to: under the data directory in the app,
    so the app owns what it downloaded; the user's own cache otherwise."""
    if os.environ.get("SIDECOACH_DATA_DIR"):
        return settings.data_dir / "models" / "hf"
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def apply_hf_home(settings: Settings) -> None:
    """Point onnx-asr's downloads at the app's cache (a no-op outside the app)."""
    if os.environ.get("SIDECOACH_DATA_DIR") and "HF_HOME" not in os.environ:
        home = hf_home(settings)
        home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(home)


def download(
    url: str, target: Path, progress: Progress | None = None, expect_sha256: str | None = None
) -> Path:
    """Fetch `url` to `target` through a temporary file, resuming a partial one, with
    progress callbacks; verified against `expect_sha256` when given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT}
    if done:
        headers["Range"] = f"bytes={done}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed hosts
        if done and response.status != 206:
            done = 0  # the server did not resume; start over
        length = response.headers.get("Content-Length")
        total = (int(length) + done) if length else None
        mode = "ab" if done else "wb"
        with part.open(mode) as fh:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
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

    def kokoro_present() -> bool:
        return all((kokoro_dir / name).exists() for name in KOKORO_FILES)

    def kokoro_fetch(progress: Progress) -> None:
        base = 0
        for name, (url, size) in KOKORO_FILES.items():
            if (kokoro_dir / name).exists():
                base += size
                continue
            download(url, kokoro_dir / name, lambda d, t, b=base: progress(b + d, None))
            base += size

    out.append(
        Item(
            "kokoro",
            "The voice (Kokoro)",
            sum(s for _, s in KOKORO_FILES.values()),
            kokoro_present,
            kokoro_fetch,
        )
    )

    def stockfish_present() -> bool:
        from sidecoach.config import find_stockfish

        return find_stockfish(settings) is not None

    def stockfish_fetch(progress: Progress) -> None:
        request = urllib.request.Request(
            STOCKFISH_API,
            headers={"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            release = json.load(response)
        system = platform.system().lower()
        if system == "windows":
            wanted = ("stockfish-windows-x86-64-universal.zip", "stockfish-windows-x86-64-avx2.zip")
        else:
            wanted = ("stockfish-ubuntu-x86-64-avx2.tar", "stockfish-ubuntu-x86-64.tar")
        asset = next((a for a in release["assets"] if a["name"] in wanted), None)
        if asset is None:
            raise RuntimeError(f"no Stockfish build for {system} in {release['tag_name']}")
        dest = settings.tools_dir
        archive = download(asset["browser_download_url"], dest / asset["name"], progress)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        else:
            import tarfile

            with tarfile.open(archive) as t:
                t.extractall(dest, filter="data")
            for exe in dest.rglob("stockfish-ubuntu*"):
                if exe.is_file():
                    exe.chmod(0o755)
        archive.unlink()

    out.append(Item("stockfish", "Stockfish", 60_000_000, stockfish_present, stockfish_fetch))

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

    def ear_present() -> bool:
        home = hf_home(settings) / "hub"
        return any(
            home.glob(
                "models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/*/encoder-model*.onnx"
            )
        ) and any(home.glob("models--istupakov--silero-vad-onnx/snapshots/*/*.onnx"))

    def ear_fetch(progress: Progress) -> None:
        # onnx-asr downloads through the hub itself; the progress is the cache growing.
        apply_hf_home(settings)
        home = hf_home(settings)
        stop = threading.Event()

        def watch() -> None:
            while not stop.is_set():
                progress(_size_of([p for p in home.rglob("*") if p.is_file()]), None)
                time.sleep(1.0)

        threading.Thread(target=watch, daemon=True).start()
        try:
            import onnx_asr  # type: ignore[import-not-found]

            onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=["CPUExecutionProvider"])
            onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
        finally:
            stop.set()

    out.append(
        Item(
            "ear",
            "The ear (Parakeet) and the voice gate (Silero)",
            2_450_000_000,
            ear_present,
            ear_fetch,
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
                    progress(scanned, PUZZLES_MAX_SCAN)
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
            30_000_000,
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
            if self._thread is not None and self._thread.is_alive():
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
        todo = [
            i
            for i in items(self.settings)
            if (ids is None and not i.optional and not i.present())
            or (ids and i.id in ids and not i.present())
        ]
        total = sum(i.size for i in todo)
        with self._lock:
            self._state["total"] = total
            self._state["queue"] = [i.id for i in todo]

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
                    with self._lock:
                        self._state["done"] = base + min(done, size)

                try:
                    item.fetch(progress)
                    with self._lock:
                        self._state["finished"].append(item.id)
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


def free_space(path: Path) -> int | None:
    with contextlib.suppress(Exception):
        return shutil.disk_usage(path if path.exists() else path.parent).free
    return None
