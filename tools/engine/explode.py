"""Where a slow search spends itself: raw UCI on the slowest position of the workloads (in
check, three moves), each depth's nodes, time and speed, for 1, 2, 3 and 6 lines and two hash
sizes. A healthy search grows each depth by a steady factor; an explosion shows as one depth
costing many times the one before.

    uv run --no-sync python tools/engine/explode.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.engine.measure import STOCKFISH  # noqa: E402

MOVES = (
    "e2e4 c7c5 g1f3 b8c6 f1b5 e7e5 c2c3 g8f6 b5c6 b7c6 d2d3 c8a6 f3e5 f8d6 e5c4 a6c4 d3c4 f6e4 "
    "e1g1 e8g8 f1e1 f8e8 f2f3 d6h2"
)


def run(lines: int, hash_mb: int, position: str) -> None:
    script = "\n".join(
        [
            "uci",
            "setoption name Threads value 1",
            f"setoption name Hash value {hash_mb}",
            f"setoption name MultiPV value {lines}",
            "isready",
            position,
            "go depth 18",
        ]
    )
    proc = subprocess.Popen(
        [str(STOCKFISH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    assert proc.stdin and proc.stdout
    proc.stdin.write(script + "\n")
    proc.stdin.flush()
    last: dict[int, tuple[int, int, int]] = {}
    for text in proc.stdout:
        if text.startswith("bestmove"):
            break
        m = re.search(r"depth (\d+) seldepth (\d+) multipv 1 .*?nodes (\d+) nps (\d+).*? time (\d+)", text)
        if not m:
            m2 = re.search(r"depth (\d+) seldepth (\d+) multipv 1 .*?nodes (\d+) nps (\d+)", text)
            t = re.search(r" time (\d+)", text)
            if m2 and t:
                d, sd, n, nps = map(int, m2.groups())
                last[d] = (n, nps, int(t.group(1)))
            continue
        d, sd, n, nps, ms = map(int, m.groups())
        last[d] = (n, nps, ms)
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()
    row = []
    prev = None
    for d in sorted(last):
        if d < 12:
            continue
        n, nps, ms = last[d]
        grow = f"x{n / prev:.1f}" if prev else ""
        row.append(f"d{d} {n / 1e6:.1f}M {ms / 1000:.1f}s {grow}")
        prev = n
    speed = last[max(last)][1] / 1e6 if last else 0
    print(f"{lines} lines, hash {hash_mb}: {speed:.2f}M nodes/s | " + " | ".join(row), flush=True)


def main() -> None:
    position = f"position startpos moves {MOVES}"
    for lines in (1, 2, 3, 6):
        run(lines, 256, position)
    run(2, 16, position)
    run(2, 1024, position)


if __name__ == "__main__":
    main()
