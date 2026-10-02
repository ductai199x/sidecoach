"""What each device test class costs, for the lanes (androidTest com/sidecoach/Lanes.kt reads
app/src/androidTest/assets/lanes/seconds.tsv): its seconds, to deal the pools longest first, and
the most Java heap it held, to keep the heap-hungry classes from running too many at once.

    python tools/lane_seconds.py [app/build/test-timings.tsv]

Reads the timings tools/test.ps1 pulls after a run -- one run with `-OneAtATime`, so each class's
numbers are its own and not its neighbours' -- and writes, longest first, `class<TAB>seconds<TAB>
peak heap MB`. A stale table changes only the order the pools take their classes in and which
wait for a heap slot, never what runs.
"""

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "app/build/test-timings.tsv"
    seconds: dict[str, float] = defaultdict(float)
    heap: dict[str, int] = defaultdict(int)
    with source.open(encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        at = {name: i for i, name in enumerate(header)}
        if "peak_heap_mb" not in at:
            raise SystemExit(f"{source} has no peak_heap_mb column: run tools\\test.ps1 -OneAtATime first")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            start = float(parts[at["start_s"]].replace(",", "."))
            end = float(parts[at["end_s"]].replace(",", "."))
            cls = parts[at["class"]]
            seconds[cls] += end - start
            heap[cls] = max(heap[cls], int(parts[at["peak_heap_mb"]]))
    out = ROOT / "app/src/androidTest/assets/lanes/seconds.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seconds.items(), key=lambda kv: -kv[1])
    out.write_text("".join(f"{cls}\t{secs:.2f}\t{heap[cls]}\n" for cls, secs in rows), encoding="utf-8", newline="\n")
    print(f"{len(rows)} classes, {sum(seconds.values()):.1f} s, from {source} -> {out}")


if __name__ == "__main__":
    main()
