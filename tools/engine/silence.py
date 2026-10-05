"""How long Stockfish goes without a word during one search: the position that a 30-second
silence watchdog killed while it was still searching (a look-ahead position of a real blitz
game; depth 18 takes 53 s on two lines). Every line the engine prints, timed; the longest gap.

    uv run --no-sync python tools/engine/silence.py
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.engine.measure import STOCKFISH  # noqa: E402

MOVES = (
    "e2e4 c7c5 g1f3 b8c6 f1b5 e7e5 c2c3 g8f6 b5c6 b7c6 d2d3 c8a6 f3e5 f8d6 e5c4 a6c4 d3c4 f6e4 "
    "e1g1 e8g8 f1e1 f8e8 f2f3 d6h2"
)


def main() -> None:
    for lines in (2, 6):
        proc = subprocess.Popen(
            [str(STOCKFISH)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        assert proc.stdin and proc.stdout
        times: list[tuple[float, str]] = []

        def read(out=proc.stdout, times=times) -> None:  # noqa: ANN001
            for text in out:
                times.append((time.perf_counter(), text.strip()))
                if text.startswith("bestmove"):
                    return

        for cmd in (
            "uci",
            "setoption name Threads value 1",
            "setoption name Hash value 256",
            f"setoption name MultiPV value {lines}",
            "isready",
        ):
            proc.stdin.write(cmd + "\n")
        proc.stdin.write(f"position startpos moves {MOVES}\n")
        reader = threading.Thread(target=read)
        reader.start()
        t0 = time.perf_counter()
        proc.stdin.write("go depth 18\n")
        reader.join()
        proc.stdin.write("quit\n")
        proc.wait()
        searched = [(t, s) for t, s in times if t >= t0]
        gaps = [
            (b[0] - a[0], a[1][:70], b[1][:70])
            for a, b in zip(searched, searched[1:], strict=False)
        ]
        worst = max(gaps)
        currmove = sum(1 for _, s in searched if " currmove " in s)
        print(
            f"{lines} lines: {searched[-1][0] - t0:.1f}s, {len(searched)} lines printed "
            f"({currmove} currmove); longest silence {worst[0]:.1f}s"
        )
        print(f"   after: {worst[1]}\n   then:  {worst[2]}")


if __name__ == "__main__":
    main()
