"""Download the latest Stockfish release binary into tools/.

    uv run python scripts/fetch_stockfish.py [--build universal]

Since Stockfish 19 the Windows release ships a single `universal` build that selects the best
CPU code path at runtime. Older releases had per-ISA builds such as avx2 or vnni512.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", default="universal")
    parser.add_argument("--dest", default="tools")
    args = parser.parse_args()

    request = urllib.request.Request(
        API, headers={"Accept": "application/vnd.github+json", "User-Agent": "sidecoach"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    wanted = f"stockfish-windows-x86-64-{args.build}.zip"
    asset = next((a for a in release["assets"] if a["name"] == wanted), None)
    if asset is None:
        names = ", ".join(a["name"] for a in release["assets"] if "windows" in a["name"])
        print(f"No asset {wanted} in {release['tag_name']}. Windows assets: {names}")
        return 1

    print(f"Downloading {release['tag_name']} {wanted} ({asset['size'] // 1_000_000} MB)")
    with urllib.request.urlopen(asset["browser_download_url"], timeout=600) as response:
        payload = response.read()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(dest)

    executables = sorted(dest.rglob("stockfish*.exe"))
    if not executables:
        print("Archive extracted but no stockfish executable found.")
        return 1
    for exe in executables:
        print(f"ready: {exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
