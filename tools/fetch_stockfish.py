"""Fetch the Stockfish source the app builds against, and its network.

The desktop downloads the official Stockfish binary for its platform. The phone cannot:
the official Android build carries its 98 MB network inside the executable, there is no
official build for the x86-64 emulator, and a binary downloaded at run time may not be
executed from an app's own data directory on Android 10 and later.

So the engine is built from source with the NDK, without the embedded network, and shipped
in the APK as a native library -- a couple of megabytes, one per ABI. The network is a data
file, needs no permission to execute, and is fetched at first run the way the desktop
fetches the engine.

    python tools/fetch_stockfish.py            # the source, into third_party/
    python tools/fetch_stockfish.py --net      # ... and the network, for a dev push

Neither is committed: `third_party/` and the network are in .gitignore.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"

# Pinned: the engine's version decides its numbers, and the coach's thresholds were tuned
# on the desktop's. Both must be the same Stockfish.
TAG = "sf_19"
SOURCE_URL = f"https://github.com/official-stockfish/Stockfish/archive/refs/tags/{TAG}.tar.gz"

# The network `src/evaluate.h` names for this version.
NET = "nn-1a298aa575a0.nnue"
NET_URL = f"https://tests.stockfishchess.org/api/nn/{NET}"


def fetch_source() -> Path:
    dest = THIRD_PARTY / "stockfish"
    if (dest / "src" / "uci.cpp").exists():
        print(f"already there: {dest}")
        return dest
    print(f"downloading {SOURCE_URL}")
    with urllib.request.urlopen(SOURCE_URL, timeout=300) as response:
        data = response.read()
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        top = archive.getnames()[0].split("/")[0]
        archive.extractall(THIRD_PARTY, filter="data")
    (THIRD_PARTY / top).rename(dest)
    print(f"ready: {dest}")
    return dest


def fetch_net() -> Path:
    dest = THIRD_PARTY / NET
    if dest.exists():
        print(f"already there: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"downloading {NET_URL} (98 MB)")
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    # The net server turns away a request with no User-Agent.
    request = urllib.request.Request(NET_URL, headers={"User-Agent": "sidecoach"})
    with urllib.request.urlopen(request, timeout=900) as response:
        dest.write_bytes(response.read())
    print(f"ready: {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--net", action="store_true", help="also fetch the network")
    args = parser.parse_args()
    fetch_source()
    if args.net:
        fetch_net()


if __name__ == "__main__":
    main()
