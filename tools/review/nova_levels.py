"""Nova games at every level, for fitting the performance rating (docs/game-review-plan.md,
phase 0): equal-rated games from 800 to 2300 in steps of 100 (Nova's Lichess-rapid scale), N a
level, each at the temperature at which Nova's mistakes match real players' at that rating
(`tools/bots/nova.temperature_for`). Equal ratings, so a game's move quality is one level's.

    uv run --no-sync python tools/review/nova_levels.py --games 25 --workers 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEVELS = range(800, 2301, 100)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=25)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    for level in LEVELS:
        print(f"level {level}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "bots" / "play.py"),
                "--games",
                str(args.games),
                "--white",
                str(level),
                "--black",
                str(level),
                "--workers",
                str(args.workers),
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
