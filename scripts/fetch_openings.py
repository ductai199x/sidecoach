"""Download the Lichess chess-openings tables into data/openings/.

    uv run python scripts/fetch_openings.py [--force]

Five TSV files, a.tsv through e.tsv, one per ECO volume, with columns eco, name, pgn.
The data is CC0. `data/` is gitignored, so every checkout fetches its own copy and
`sidecoach.analysis.openings` works without it.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
VOLUMES = ("a", "b", "c", "d", "e")


def fetch(url: str, target: Path, force: bool) -> int:
    """Download `url` to `target`. Returns the number of lines written, or -1 on failure."""
    if target.exists() and not force:
        lines = sum(1 for _ in target.open("rb"))
        print(f"skip: {target} already there ({lines} lines), --force to replace")
        return lines

    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload: bytes = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"failed: {url} ({exc})")
        return -1

    target.write_bytes(payload)
    lines = payload.count(b"\n")
    print(f"ready: {target} ({lines} lines, {len(payload) // 1024} KB)")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="data/openings")
    parser.add_argument("--force", action="store_true", help="re-download files already present")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for volume in VOLUMES:
        written = fetch(f"{BASE}/{volume}.tsv", dest / f"{volume}.tsv", args.force)
        if written < 0:
            return 1
        total += written

    print(f"{total} lines in {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
