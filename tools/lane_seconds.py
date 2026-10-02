"""The seconds each device test class took, for the lanes to deal their pools longest first
(androidTest com/sidecoach/Lanes.kt reads app/src/androidTest/assets/lanes/seconds.tsv).

    python tools/lane_seconds.py [app/build/test-timings.tsv]

Reads the timings tools/test.ps1 pulls after a run -- best one run with `-OneAtATime`, so each
class's time is its own and not its neighbours' -- and writes each class's seconds, longest
first. A stale table changes only the order the pools take their classes in, never what runs.
"""

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "app/build/test-timings.tsv"
    seconds: dict[str, float] = defaultdict(float)
    with source.open(encoding="utf-8") as f:
        header = next(f).rstrip("\n").split("\t")
        at = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            start = float(parts[at["start_s"]].replace(",", "."))
            end = float(parts[at["end_s"]].replace(",", "."))
            seconds[parts[at["class"]]] += end - start
    out = ROOT / "app/src/androidTest/assets/lanes/seconds.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(seconds.items(), key=lambda kv: -kv[1])
    out.write_text("".join(f"{cls}\t{secs:.2f}\n" for cls, secs in rows), encoding="utf-8", newline="\n")
    print(f"{len(rows)} classes, {sum(seconds.values()):.1f} s, from {source} -> {out}")


if __name__ == "__main__":
    main()
