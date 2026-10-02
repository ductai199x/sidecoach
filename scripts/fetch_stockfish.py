"""Download the pinned Stockfish release's binary for this platform into tools/.

    uv run python scripts/fetch_stockfish.py [--dest tools]

The same choice the app's first-run setup makes (sidecoach.setup.install_stockfish): the
release `STOCKFISH_TAG` (as the phone and the tests pin it), its `universal` build for this
platform and CPU, which picks its code path at run time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidecoach.setup import STOCKFISH_TAG, install_stockfish  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", default="tools")
    args = parser.parse_args()

    dest = Path(args.dest)
    print(f"Downloading Stockfish {STOCKFISH_TAG}")
    executables = install_stockfish(dest, dest.parent / f"{dest.name}.staging")
    for exe in executables:
        print(f"ready: {exe}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
