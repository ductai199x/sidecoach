"""Sample a process tree's memory and CPU for the desktop shell's evaluation.

docs/app.md: the shell plus its webview processes (WebView2's are children
of the host process, Chrome's of the chrome.exe that was launched) sampled at 1 Hz for a
run, with the page's own meter (`?perf=<label>` in the panel) writing beside it. Prints
p50/p95/max of the tree's RSS and CPU and the growth from the first to the last minute.

    uv run --no-sync python scripts/perf_sample.py --pid 1234 --seconds 600 --label idle
    uv run --no-sync python scripts/perf_sample.py --name sidecoach.exe --label loaded
    uv run --no-sync python scripts/perf_sample.py --summary data/perf/idle.csv data/perf/idle.jsonl

The CSV goes to data/perf/<label>.csv; `--summary` reads a CSV (and a page JSONL when
given) back and prints the figures for section 5.8.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import psutil

# The sidecar's processes, when the shell spawned it inside its own tree: counted apart, so
# the UI's figures (the shell and its webview) stay comparable with a browser's.
SIDECAR_NAMES = {"python.exe", "python", "uv.exe", "uv", "sidecoach.exe", "conhost.exe"}
SIDECAR_PREFIXES = ("stockfish",)


def is_sidecar(name: str) -> bool:
    low = name.lower()
    return low in SIDECAR_NAMES or low.startswith(SIDECAR_PREFIXES)


def tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except psutil.Error:
        return []


def find(name: str) -> int | None:
    for proc in psutil.process_iter(["name", "ppid"]):
        if (proc.info["name"] or "").lower() == name.lower():
            # The topmost process of that name: its parent has another name.
            try:
                parent = proc.parent()
                if parent is None or (parent.name() or "").lower() != name.lower():
                    return proc.pid
            except psutil.Error:
                return proc.pid
    return None


def sample(pid: int, seconds: float, out: Path) -> None:
    """One row a second: the tree's process count, RSS and CPU, plus the top processes by
    RSS at the end. Process objects are kept between samples, which `cpu_percent` needs."""
    known: dict[int, psutil.Process] = {}

    def refresh() -> list[psutil.Process]:
        procs = tree(pid)
        for proc in procs:
            if proc.pid not in known:
                known[proc.pid] = proc
                with _quiet():
                    proc.cpu_percent(None)
        for gone in set(known) - {p.pid for p in procs}:
            del known[gone]
        return [known[p.pid] for p in procs]

    if not refresh():
        sys.exit(f"no process {pid}")
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    by_rss: dict[str, float] = {}
    missing = 0
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["t", "procs", "rss_mb", "cpu_pct", "sidecar_rss_mb", "sidecar_cpu_pct"])
        while time.monotonic() - t0 < seconds:
            time.sleep(1.0)
            procs = refresh()
            if not procs:
                missing += 1
                if missing > 5:
                    print("process tree gone", file=sys.stderr)
                    break
                continue
            missing = 0
            rss = 0
            cpu = 0.0
            side_rss = 0
            side_cpu = 0.0
            by_rss = {}
            for proc in procs:
                with _quiet():
                    r = proc.memory_info().rss
                    c = proc.cpu_percent(None)
                    name = proc.name()
                    if is_sidecar(name):
                        side_rss += r
                        side_cpu += c
                        continue
                    rss += r
                    cpu += c
                    with _quiet():
                        args = proc.cmdline()
                        kind = next(
                            (a.split("=", 1)[1] for a in args if a.startswith("--type=")), ""
                        )
                        if kind:
                            name = f"{name} ({kind})"
                    by_rss[f"{name} [{proc.pid}]"] = r / 1e6
            t = round(time.monotonic() - t0, 1)
            writer.writerow(
                [
                    t,
                    len(procs),
                    round(rss / 1e6, 1),
                    round(cpu, 1),
                    round(side_rss / 1e6, 1),
                    round(side_cpu, 1),
                ]
            )
            fh.flush()
            if sys.stdout.isatty():  # a progress line for a terminal, silence for a log
                print(
                    f"\r{t:7.1f}s  {len(procs)} procs  ui {rss / 1e6:7.1f} MB {cpu:5.1f}%"
                    f"  sidecar {side_rss / 1e6:7.1f} MB {side_cpu:5.1f}%",
                    end="",
                )
    print()
    print("  processes at the end, by RSS:")
    for name, mb in sorted(by_rss.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {mb:7.1f} MB  {name}")
    summary(out, None)


class _quiet:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return exc[0] is not None and issubclass(exc[0], psutil.Error)  # type: ignore[arg-type]


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * p))]


def summary(csv_path: Path, page: Path | None) -> None:
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        print("no samples")
        return
    rss = [float(r["rss_mb"]) for r in rows]
    cpu = [float(r["cpu_pct"]) for r in rows]
    n = len(rows)
    first = rss[: min(60, n)]
    last = rss[-min(60, n) :]
    growth = (statistics.mean(last) - statistics.mean(first)) / max(statistics.mean(first), 1e-9)
    print(f"{csv_path.name}: {n} samples over {float(rows[-1]['t']):.0f} s")
    print(
        f"  ui rss MB   p50 {_pct(rss, 0.5):7.1f}  p95 {_pct(rss, 0.95):7.1f}  max {max(rss):7.1f}"
        f"  growth first->last minute {growth * 100:+.1f}%"
    )
    print(
        f"  ui cpu %    p50 {_pct(cpu, 0.5):7.1f}  p95 {_pct(cpu, 0.95):7.1f}  max {max(cpu):7.1f}"
        "  (percent of one core, summed over the shell and its webview)"
    )
    if "sidecar_rss_mb" in rows[0]:
        srss = [float(r["sidecar_rss_mb"]) for r in rows]
        scpu = [float(r["sidecar_cpu_pct"]) for r in rows]
        if max(srss) > 0:
            print(
                f"  sidecar     rss p50 {_pct(srss, 0.5):7.1f} MB  max {max(srss):7.1f}"
                f"  cpu p50 {_pct(scpu, 0.5):6.1f}%  p95 {_pct(scpu, 0.95):6.1f}%"
            )
    if page is not None and page.exists():
        samples = []
        for line in page.read_text(encoding="utf-8").splitlines():
            try:
                samples.append(json.loads(line))
            except ValueError:
                continue  # a torn line from two pages posting at once
        if samples:
            fps_min = min(s.get("fps_min", float("inf")) for s in samples)
            fps_mean = statistics.mean(s.get("fps", 0.0) for s in samples)
            long_frames = samples[-1].get("long_frames", 0)
            frames = samples[-1].get("frames", 0)
            p50 = [s["poll_p50"] for s in samples if s.get("poll_p50") is not None]
            p95 = [s["poll_p95"] for s in samples if s.get("poll_p95") is not None]
            print(
                f"  page     host {samples[-1].get('host')}  fps mean {fps_mean:5.1f}"
                f"  min {fps_min:5.1f}  long frames {long_frames} of {frames}"
                f" ({100.0 * long_frames / max(frames, 1):.2f}%)"
                f"  poll p50 {statistics.median(p50) if p50 else float('nan'):.0f} ms"
                f"  p95 {max(p95) if p95 else float('nan'):.0f} ms"
            )
            # What the page held, first and last, when the meter reports it: the JS heap,
            # the DOM's size and the store's lists -- where growth in the renderer comes from.
            held = [s for s in samples if s.get("dom") is not None]
            if held:
                a, b = held[0], held[-1]
                parts = [
                    f"{key} {a.get(key)} -> {b.get(key)}"
                    for key in ("heap_mb", "dom", "rows", "coach", "convo")
                ]
                print("  page held  " + ", ".join(parts))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pid", type=int)
    ap.add_argument("--name", help="the tree's root by process name (sidecoach.exe)")
    ap.add_argument("--seconds", type=float, default=600.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out-dir", type=Path, default=Path("data/perf"))
    ap.add_argument("--summary", nargs="+", type=Path, help="a CSV (and a page JSONL)")
    args = ap.parse_args()
    if args.summary:
        summary(args.summary[0], args.summary[1] if len(args.summary) > 1 else None)
        return
    pid = args.pid or (find(args.name) if args.name else None)
    if pid is None:
        sys.exit("give --pid or --name")
    sample(pid, args.seconds, args.out_dir / f"{args.label}.csv")


if __name__ == "__main__":
    main()
