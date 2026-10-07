"""Read a Spike A capture log off the phone and say what it measured.

    adb pull /sdcard/... no: the log lives in the app's own files directory, so
    `adb exec-out run-as com.sidecoach cat files/spike/<name>.jsonl > local.jsonl`
    (debuggable builds only), then `python tools/spike.py local.jsonl`.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python tools/spike.py <capture.jsonl>")
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("the log is empty: did the capture run?")

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if r.get(name) is not None]

    minutes = rows[-1]["t"] / 60
    frames = sum(int(r["frames"]) for r in rows)
    dropped = sum(int(r["dropped"]) for r in rows)
    fps = col("fps")
    battery = [int(r["battery_percent"]) for r in rows if int(r["battery_percent"]) >= 0]
    thermal = {}
    for r in rows:
        thermal[r["thermal"]] = thermal.get(r["thermal"], 0) + 1

    print(f"{minutes:.1f} minutes, {frames} frames, {dropped} skipped")
    print(f"fps      median {statistics.median(fps):.2f}  min {min(fps):.2f}")
    print(f"interval p50 {statistics.median(col('interval_p50')):.0f} ms  "
          f"p95 {max(col('interval_p95')):.0f} ms  worst {max(col('interval_max')):.0f} ms")
    print(f"copy     p50 {statistics.median(col('copy_p50')):.0f} ms  p95 {max(col('copy_p95')):.0f} ms  "
          f"frame {statistics.median(col('frame_kb')):.0f} KB")
    print(f"cpu      median {statistics.median(col('cpu_percent')):.1f} %  peak {max(col('cpu_percent')):.1f} % of a core")
    if battery:
        print(f"battery  {battery[0]} % -> {battery[-1]} % ({battery[0] - battery[-1]} points in {minutes:.0f} min), "
              f"median {statistics.median(col('battery_ma')):.0f} mA")
    print("thermal  " + ", ".join(f"{k} x{v}" for k, v in sorted(thermal.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
