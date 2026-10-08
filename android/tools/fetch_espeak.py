"""Fetch the espeak-ng source the app builds against, and the pronunciation data it reads.

The desktop's voice turns text into phonemes with espeak-ng (through `phonemizer`, the way
kokoro-onnx asks it), from the library and data `espeakng_loader` ships: espeak-ng 1.52.0,
built from that tag. The phone builds the same tag from source with the NDK -- the library
part only, listed in `app/src/main/cpp/CMakeLists.txt` as Stockfish's files are -- and carries
the English part of the same compiled data in its assets, copied here from the desktop's
`espeakng_loader`. The data is the same bytes on every platform, and the phonemes are the
same only if both the library and the data are the desktop's.

    uv run --no-sync python android/tools/fetch_espeak.py   (from the desktop repo)

The source goes to `third_party/` (not committed); the data subset is written into
`app/src/main/assets/espeak-ng-data` (committed: it is what the app reads).
"""

from __future__ import annotations

import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
ASSETS = ROOT / "app" / "src" / "main" / "assets" / "espeak-ng-data"

# Pinned: the tag `espeakng_loader` 0.2.4 builds (its build.yml clones `-b 1.52.0`).
TAG = "1.52.0"
SOURCE_URL = f"https://github.com/espeak-ng/espeak-ng/archive/refs/tags/{TAG}.tar.gz"

# What English needs from the data: the phoneme tables, the intonation, the English
# dictionary, and the voice and language files that `espeak_ListVoices` reads to find
# "en-us". Everything else in the directory is other languages.
DATA_FILES = ["phondata", "phondata-manifest", "phonindex", "phontab", "intonations", "en_dict"]
DATA_DIRS = ["lang/gmw", "voices/!v"]


def fetch_source() -> Path:
    dest = THIRD_PARTY / "espeak-ng"
    if (dest / "src" / "libespeak-ng" / "speech.c").exists():
        print(f"already there: {dest}")
        return dest
    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:
        data = response.read()
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        archive.extractall(THIRD_PARTY, filter="data")
    (THIRD_PARTY / f"espeak-ng-{TAG}").rename(dest)
    print(f"source in {dest}")
    return dest


def copy_data() -> None:
    import espeakng_loader  # the desktop's own, from the desktop repo's environment

    source = Path(espeakng_loader.get_data_path())
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True)
    for name in DATA_FILES:
        shutil.copy2(source / name, ASSETS / name)
    for sub in DATA_DIRS:
        shutil.copytree(source / sub, ASSETS / sub)
        # The package ships some voice files with CRLF; the repo holds no Windows line endings
        # (Tai, 2026-10-08). Text files only: a file with a NUL byte is binary and kept whole.
        for path in (ASSETS / sub).rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                if b"\0" not in data and b"\r\n" in data:
                    path.write_bytes(data.replace(b"\r\n", b"\n"))
    size = sum(p.stat().st_size for p in ASSETS.rglob("*") if p.is_file())
    count = sum(1 for p in ASSETS.rglob("*") if p.is_file())
    print(f"data: {count} files, {size / 1e6:.1f} MB -> {ASSETS}")


if __name__ == "__main__":
    fetch_source()
    copy_data()
