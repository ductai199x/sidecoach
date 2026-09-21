"""Download the latest Stockfish release binary for this platform into tools/.

    uv run python scripts/fetch_stockfish.py [--dest tools]

The same choice the app's first-run setup makes (sidecoach.setup.stockfish_assets): the
`universal` build for this platform and CPU, which picks its code path at run time.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidecoach.setup import STOCKFISH_API, stockfish_assets, unpack_stockfish  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="tools")
    args = parser.parse_args()

    request = urllib.request.Request(
        STOCKFISH_API, headers={"Accept": "application/vnd.github+json", "User-Agent": "sidecoach"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        release = json.load(response)

    wanted = stockfish_assets()
    by_name = {a["name"]: a for a in release["assets"]}
    asset = next((by_name[n] for n in wanted if n in by_name), None)
    if asset is None:
        print(f"None of {wanted} in {release['tag_name']}. Assets: {', '.join(sorted(by_name))}")
        return 1

    print(f"Downloading {release['tag_name']} {asset['name']} ({asset['size'] // 1_000_000} MB)")
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / asset["name"]
    with urllib.request.urlopen(asset["browser_download_url"], timeout=600) as response:
        archive.write_bytes(response.read())
    executables = unpack_stockfish(archive, dest)
    archive.unlink()
    if not executables:
        print("Archive extracted but no stockfish executable found.")
        return 1
    for exe in executables:
        print(f"ready: {exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
