"""Download the latest Stockfish release binary for this platform into tools/.

    uv run python scripts/fetch_stockfish.py [--build universal|avx2] [--dest tools]

Windows gets the zip (since Stockfish 19 one `universal` build that picks the best CPU
code path at run time; older releases had per-ISA builds such as avx2); Linux gets the
Ubuntu tarball (avx2, or the plain x86-64 build with --build x86-64). The same choice the
app's first-run setup makes (sidecoach/setup).
"""

from __future__ import annotations

import argparse
import io
import json
import platform
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


def wanted_asset(build: str | None) -> tuple[str, ...]:
    """The release assets to look for, best first, for this platform."""
    if platform.system().lower() == "windows":
        return (f"stockfish-windows-x86-64-{build or 'universal'}.zip",)
    if build and build != "universal":
        return (f"stockfish-ubuntu-x86-64-{build}.tar",)
    return ("stockfish-ubuntu-x86-64-avx2.tar", "stockfish-ubuntu-x86-64.tar")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default=None)
    parser.add_argument("--dest", default="tools")
    args = parser.parse_args()

    request = urllib.request.Request(
        API, headers={"Accept": "application/vnd.github+json", "User-Agent": "sidecoach"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    wanted = wanted_asset(args.build)
    asset = next((a for a in release["assets"] if a["name"] in wanted), None)
    if asset is None:
        names = ", ".join(a["name"] for a in release["assets"])
        print(f"None of {wanted} in {release['tag_name']}. Assets: {names}")
        return 1

    print(f"Downloading {release['tag_name']} {asset['name']} ({asset['size'] // 1_000_000} MB)")
    with urllib.request.urlopen(asset["browser_download_url"], timeout=600) as response:
        payload = response.read()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if asset["name"].endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(dest)
        executables = sorted(dest.rglob("stockfish*.exe"))
    else:
        with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
            archive.extractall(dest, filter="data")
        executables = sorted(
            f for f in dest.rglob("stockfish-*") if f.is_file() and f.suffix not in (".tar", ".zip")
        )
        for exe in executables:
            exe.chmod(0o755)

    if not executables:
        print("Archive extracted but no stockfish executable found.")
        return 1
    for exe in executables:
        print(f"ready: {exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
