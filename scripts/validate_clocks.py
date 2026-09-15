"""Check the clock reader against chess.com's per-move times, or learn its templates.

    uv run python scripts/validate_clocks.py data/recordings/g04-antonio-takebacks
    uv run python scripts/validate_clocks.py --learn src/sidecoach/vision/clock_templates.npz \
        data/recordings/g0*

Validation replays a recording through the board tracker, the game pipeline and the clock
reader, works out the think time of every ply from the clock readings, and compares it with
`time_ds` in the recording's `chesscom.json`. It prints one row per ply and a summary: the
fraction of plies within half a second, the mean absolute error, every ply off by more than
a second with the readings around it, the time control the game started with, and the
reader's cost per frame.

The think time of a ply, from the readings. A clock shows whole seconds, so the mover's clock
at the tick their move landed against its value when the previous move landed is only good
to a second: the two unseen fractions add up to an error of up to a second either way. The
moments the display ticks down recover the fractions. With `n` the number of seconds the
mover's clock lost over the ply (its idle value before the ply, less its idle value after
the ply, plus the increment), `S` and `E` the ticks the previous move and this one landed,
and `T1` and `Tn` the first and last ticks the display fell:

    think = (n - 1) + (T1 - S) + (E - Tn)        or  E - S  when the display never fell

which holds whether the display floors or ceils the remaining time. A misread digit shows up
as a wrong `n`, so the comparison tests the digits, and the timestamps only carry the
fractions.

Learning collects the glyphs cut from every clock box in the given recordings, clusters them
by normalised correlation, and labels the clusters from the countdown itself: the colon is
the middle glyph of every reading, the 1 is the narrowest digit, and following the
transitions of the seconds digit from one stored frame to the next walks 1, 0, 9, 8, ... 2.
Variants of a digit that clustered apart (dark text on a light box is thinner than light
text on a dark box) are labelled by similarity to the walked clusters and checked against
the tens-of-seconds cycle. The tenths dot, never seen above a minute, is the lower dot of
the colon.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sidecoach.events import GameStarted, MovePlayed, Side  # noqa: E402
from sidecoach.replay.recording import Recording  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from sidecoach.vision.clock import (  # noqa: E402
    GLYPH_H,
    GLYPH_W,
    SEARCH_CELLS,
    ClockReader,
    ClockReading,
    GlyphTemplates,
    find_box,
    glyph_canvas,
    ink_map,
    segment,
)
from sidecoach.vision.tracker import BoardTracker  # noqa: E402

INCREMENTS = (0, 1, 2, 3, 5, 10, 15, 20, 30, 45, 60)
# Two stored frames of one box this close cannot straddle two ticks of the display.
MAX_STEP_TICKS = 12


def fmt(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    s, tenths = divmod(ms // 100, 10)
    m, s = divmod(s, 60)
    return f"{m}:{s:02d}" + (f".{tenths}" if m == 0 else "")


# Learning


@dataclass
class Sample:
    rec: str
    tick: int
    side: str
    pos: int
    count: int
    light: bool
    canvas: np.ndarray
    aspect: float


def collect(rec_dir: Path) -> list[Sample]:
    import cv2

    rec = Recording(rec_dir)
    tracker = BoardTracker()
    samples: list[Sample] = []
    for tick, frame in rec.frames():
        if tick.stored is None:
            continue
        board = tracker.update(frame)
        if board is None or not tracker.verified:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = board.region
        cell = board.cell_px
        reach = int(round(SEARCH_CELLS * cell))
        for side, rows in (("top", (y - reach, y - 1)), ("bottom", (y + h + 1, y + h + reach))):
            box = find_box(gray, x + w, rows, cell)
            if box is None:
                continue
            ink = ink_map(gray, box, cell)
            seg = segment(ink)
            if seg is None:
                continue
            t0, t1 = seg.line
            for pos, span in enumerate(seg.glyphs):
                samples.append(
                    Sample(
                        rec_dir.name,
                        tick.frame_id,
                        side,
                        pos,
                        len(seg.glyphs),
                        box.light,
                        glyph_canvas(ink, seg.line, span),
                        (span[1] - span[0]) / (t1 - t0),
                    )
                )
    return samples


def _unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-6)


def cluster(samples: list[Sample], threshold: float = 0.92) -> tuple[list[np.ndarray], list[int]]:
    """Greedy clustering by normalised correlation; deterministic in the sample order."""
    centroids: list[np.ndarray] = []
    sums: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[int] = []
    for sample in samples:
        flat = sample.canvas.ravel()
        if centroids:
            sims = np.array([_unit(c) @ _unit(flat) for c in centroids])
            j = int(np.argmax(sims))
            if sims[j] > threshold:
                sums[j] += flat
                counts[j] += 1
                centroids[j] = sums[j] / counts[j]
                labels.append(j)
                continue
        centroids.append(flat.copy())
        sums.append(flat.copy())
        counts.append(1)
        labels.append(len(centroids) - 1)
    return centroids, labels


def label_clusters(
    samples: list[Sample], centroids: list[np.ndarray], labels: list[int]
) -> dict[int, str]:
    """Name every big cluster from the countdown, with no hand-labelled digit.

    One digit renders as several clusters: the glyph lands on different sub-pixel offsets,
    and light text on a dark box is bolder than dark text on a light one. So the labelling
    works on the graph of clusters rather than on one exemplar per digit. Between two
    stored frames of the same box less than `MAX_STEP_TICKS` apart, a seconds digit that
    changed changed by exactly one, counting down. Those transitions are edges "one less
    than"; a breadth-first walk over them labels every cluster modulo ten, and the rotation
    is pinned by the tens-of-seconds position, which only ever shows 0 to 5. The colon is
    the middle glyph of every five-glyph reading.
    """
    counts = Counter(labels)
    big = {k for k, n in counts.items() if n >= 0.002 * len(samples)}
    readings: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(dict)
    for sample, k in zip(samples, labels, strict=True):
        if sample.count == 5:
            readings[(sample.rec, sample.side)].setdefault(sample.tick, [-1] * 5)[sample.pos] = k
    colon = Counter(seq[2] for d in readings.values() for seq in d.values()).most_common(1)[0][0]
    steps: Counter[tuple[int, int]] = Counter()
    tens_seen: Counter[int] = Counter()
    for d in readings.values():
        ticks = sorted(d)
        for before, after in zip(ticks, ticks[1:], strict=False):
            a, b = d[before], d[after]
            if -1 in a or -1 in b or after - before > MAX_STEP_TICKS:
                continue
            if a[:4] == b[:4] and a[4] != b[4]:
                steps[(a[4], b[4])] += 1  # b is one less than a
        for seq in d.values():
            tens_seen[seq[3]] += 1
    # Keep an edge only when it is a cluster's dominant successor and was seen often.
    best_step: dict[int, tuple[int, int]] = {}
    for (a, b), n in steps.items():
        if a not in best_step or n > best_step[a][1]:
            best_step[a] = (b, n)
    edges = [(a, b) for a, (b, n) in best_step.items() if n >= 5 and a in big and b in big]
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append((b, -1))
        adjacency[b].append((a, +1))
    order: dict[int, int] = {}
    root = max((k for k in big if k != colon), key=lambda k: counts[k])
    order[root] = 0
    queue = [root]
    while queue:
        node = queue.pop(0)
        for other, delta in adjacency[node]:
            value = (order[node] + delta) % 10
            if other in order:
                if order[other] != value:
                    raise SystemExit(f"countdown graph contradicts itself at cluster {other}")
            else:
                order[other] = value
                queue.append(other)
    # Pin the rotation: the tens-of-seconds digit is 0 to 5 and never 6 to 9.
    rotations = [
        r
        for r in range(10)
        if all((order[k] + r) % 10 <= 5 for k in tens_seen if k in order and tens_seen[k] >= 20)
    ]
    if len(rotations) != 1:
        raise SystemExit(f"the tens-of-seconds digits pin {len(rotations)} rotations, not one")
    names = {k: str((v + rotations[0]) % 10) for k, v in order.items()}
    names[colon] = ":"
    for k in sorted(big - set(names), key=lambda k: -counts[k]):
        # A variant too rare to have its own transitions: nearest labelled centroid.
        sims = sorted(
            ((float(_unit(centroids[k]) @ _unit(centroids[j])), name) for j, name in names.items()),
            reverse=True,
        )
        other = next((score for score, name in sims if name != sims[0][1]), -1.0)
        if sims[0][0] >= 0.8 and sims[0][0] - other >= 0.05:
            names[k] = sims[0][1]
        else:
            print(f"cluster {k} ({counts[k]} glyphs) unlabelled: best {sims[0]}, next {other:.2f}")
    # Independent check: the tens-of-seconds digit counts 5, 4, 3, 2, 1, 0 and wraps to 5.
    tens_steps: Counter[tuple[int, int]] = Counter()
    for d in readings.values():
        ticks = sorted(d)
        for before, after in zip(ticks, ticks[1:], strict=False):
            a, b = d[before], d[after]
            if -1 in a or -1 in b or after - before > MAX_STEP_TICKS:
                continue
            if a[:3] == b[:3] and a[3] != b[3]:
                tens_steps[(a[3], b[3])] += 1
    bad = sum(
        n
        for (a, b), n in tens_steps.items()
        if a in names and b in names and (int(names[a]) - 1) % 6 != int(names[b])
    )
    print(f"tens-of-seconds steps against the labels: {bad} of {sum(tens_steps.values())}")
    return names


def learn(rec_dirs: list[Path], out: Path) -> None:
    samples: list[Sample] = []
    for rec_dir in rec_dirs:
        got = collect(rec_dir)
        print(f"{rec_dir.name}: {len(got)} glyphs")
        samples += got
    centroids, labels = cluster(samples)
    names = label_clusters(samples, centroids, labels)
    print("clusters:", {k: (names.get(k, "?"), n) for k, n in Counter(labels).most_common()})
    by_key: dict[tuple[str, bool], list[Sample]] = defaultdict(list)
    for sample, k in zip(samples, labels, strict=True):
        if k in names:
            by_key[(names[k], sample.light)].append(sample)
    template_labels: list[str] = []
    canvases: list[np.ndarray] = []
    advances: list[float] = []
    for (name, light), group in sorted(by_key.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        template_labels.append(name)
        canvases.append(np.mean([s.canvas for s in group], axis=0))
        advances.append(float(statistics.median(s.aspect for s in group)))
        print(f"  {name!r} {'light' if light else 'dark'} box: {len(group)} glyphs, advance {advances[-1]:.2f}")
    # The tenths dot: the lower dot of the colon, at the baseline.
    for i, name in enumerate(list(template_labels)):
        if name == ":":
            dot = canvases[i].copy()
            dot[: GLYPH_H // 2] = 0.0
            template_labels.append(".")
            canvases.append(dot)
            advances.append(advances[i])
    templates = GlyphTemplates(
        tuple(template_labels), np.stack(canvases).astype(np.float32), np.array(advances, np.float32)
    )
    templates.save(out)
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(template_labels)} templates of {GLYPH_H}x{GLYPH_W})")


# Validation


@dataclass
class Landing:
    ply: int
    san: str
    tick: int
    by_user: bool


def _mode(values: list[int]) -> int | None:
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def _idle_value(series: dict[int, int | None], ticks: range) -> int | None:
    values = [series[t] for t in ticks if series.get(t) is not None]
    return _mode(values)


def _after_value(series: dict[int, int | None], ticks: range) -> int | None:
    """The post-increment value: the largest value seen at least twice, else the largest."""
    values = [series[t] for t in ticks if series.get(t) is not None]
    if not values:
        return None
    twice = [v for v, n in Counter(values).items() if n >= 2]
    return max(twice) if twice else max(values)


def _falls(series: dict[int, int | None], ticks: range) -> list[int]:
    """Ticks in `ticks` at which the value read is one less than the last value read."""
    falls: list[int] = []
    prev: int | None = None
    for t in ticks:
        v = series.get(t)
        if v is None:
            continue
        if prev is not None and v == prev - 1000:
            falls.append(t)
        prev = v
    return falls


def validate(rec_dir: Path, verbose: bool = True) -> dict[str, object]:
    rec = Recording(rec_dir)
    truth = json.loads((rec_dir / "chesscom.json").read_text(encoding="utf-8"))["moves"]
    pipe = GamePipeline()
    reader = ClockReader()
    ts_of: dict[int, float] = {}
    readings: dict[int, ClockReading | None] = {}
    board_at: dict[int, bool] = {}
    landings: list[Landing] = []
    started_at: int | None = None
    costs: list[float] = []
    for tick, frame in rec.frames():
        ts_of[tick.frame_id] = tick.ts
        events = pipe.process(frame, tick.ts, changed=tick.stored is not None, frame_id=tick.frame_id)
        board = pipe.board if pipe.tracker.verified else None
        board_at[tick.frame_id] = board is not None
        if board is not None:
            t = time.perf_counter()
            reading = reader.read(frame, board, tick.ts)
            costs.append(time.perf_counter() - t)
        else:
            reading = None
        readings[tick.frame_id] = reading
        for event in events:
            if isinstance(event, GameStarted) and started_at is None:
                started_at = tick.frame_id
            if isinstance(event, MovePlayed):
                landings.append(Landing(event.ply, event.san, tick.frame_id, event.by_user))
    bottom = pipe.bottom or Side.WHITE
    # Per-clock series in ms, by tick, white and black.
    white: dict[int, int | None] = {}
    black: dict[int, int | None] = {}
    light_votes: Counter[str] = Counter()
    running_votes: Counter[str] = Counter()
    for tick_id, reading in readings.items():
        if reading is None:
            white[tick_id] = black[tick_id] = None
            continue
        top_is_white = bottom == Side.BLACK
        white[tick_id] = reading.top_ms if top_is_white else reading.bottom_ms
        black[tick_id] = reading.bottom_ms if top_is_white else reading.top_ms
        if reading.light is not None:
            light_votes[reading.light] += 1
        running_votes[str(reading.running)] += 1
    # The last landing of every truth ply, matched by SAN: a taken-back ply is replayed.
    by_ply: dict[int, Landing] = {}
    for landing in landings:
        expected = truth[landing.ply - 1]["san"] if landing.ply - 1 < len(truth) else None
        if expected == landing.san:
            by_ply[landing.ply] = landing
    if started_at is not None:
        by_ply[0] = Landing(0, "start", started_at, False)
    # Increment: the smallest standard increment covering the largest rise of a clock over
    # a ply (a fast move keeps nearly all of it).
    rises: list[int] = []
    for ply in range(1, len(truth) + 1):
        if ply not in by_ply or ply - 1 not in by_ply:
            continue
        clock = white if ply % 2 == 1 else black
        before = _idle_value(clock, range(by_ply[ply - 1].tick - 2, by_ply[ply - 1].tick + 1))
        after = _after_value(clock, range(by_ply[ply].tick, by_ply[ply].tick + 6))
        if before is not None and after is not None:
            rises.append((after - before) // 1000)
    inc = next((i for i in INCREMENTS if rises and i >= max(rises)), 0)
    rows: list[tuple[int, str, float | None, float, str]] = []
    for ply in range(1, len(truth) + 1):
        target = truth[ply - 1]["time_ds"] / 10.0
        if ply not in by_ply or ply - 1 not in by_ply:
            rows.append((ply, truth[ply - 1]["san"], None, target, "no landing"))
            continue
        clock = white if ply % 2 == 1 else black
        s_tick, e_tick = by_ply[ply - 1].tick, by_ply[ply].tick
        before = _idle_value(clock, range(s_tick - 2, s_tick + 1))
        after = _after_value(clock, range(e_tick, e_tick + 6))
        if before is None or after is None:
            rows.append((ply, truth[ply - 1]["san"], None, target, "clock unread"))
            continue
        n = (before - after) // 1000 + inc
        falls = [t for t in _falls(clock, range(s_tick - 2, e_tick + 6)) if t > s_tick and t <= e_tick + 1]
        S, E = ts_of[s_tick], ts_of[e_tick]
        if n <= 0 or not falls:
            estimate = E - S
            note = "" if n == 0 else f"n={n} without a fall"
        else:
            estimate = (n - 1) + (ts_of[falls[0]] - S) + (E - ts_of[falls[-1]])
            note = "" if len(falls) == n else f"{len(falls)} falls for n={n}"
        rows.append((ply, truth[ply - 1]["san"], estimate, target, note))
    errors = [abs(e - t) for _, _, e, t, _ in rows if e is not None]
    within = sum(1 for err in errors if err <= 0.5)
    tc = None
    if started_at is not None:
        tc = (fmt(_idle_value(white, range(started_at, started_at + 10))), fmt(_idle_value(black, range(started_at, started_at + 10))))
    board_ticks = sum(board_at.values())
    read_ticks = sum(1 for t, r in readings.items() if r is not None)
    if verbose:
        print(f"\n== {rec_dir.name}: {len(truth)} plies, {bottom.value} at the bottom, increment {inc} s")
        print(f"   time control read at the start: white {tc[0] if tc else '?'} black {tc[1] if tc else '?'}")
        print(f"   {read_ticks} of {board_ticks} board ticks read a clock; light box votes {dict(light_votes)}; running votes {dict(running_votes)}")
        print("   ply  san      clock   chess.com   error")
        for ply, san, estimate, target, note in rows:
            est = f"{estimate:6.1f}" if estimate is not None else "   n/a"
            err = f"{estimate - target:+5.1f}" if estimate is not None else "     "
            flag = " <-- off" if estimate is not None and abs(estimate - target) > 1.0 else ""
            print(f"   {ply:3d}  {san:8s} {est}  {target:6.1f}  {err}  {note}{flag}")
        for ply, san, estimate, target, _ in rows:
            if estimate is None or abs(estimate - target) <= 1.0:
                continue
            s_tick, e_tick = by_ply[ply - 1].tick, by_ply[ply].tick
            print(f"   ply {ply} {san}: clock {estimate:.1f} s vs chess.com {target:.1f} s; readings (tick white black running):")
            for t in list(range(s_tick - 3, s_tick + 3)) + list(range(e_tick - 3, e_tick + 6)):
                r = readings.get(t)
                print(f"      {t:5d} {fmt(white.get(t))} {fmt(black.get(t))} {r.running if r else '-'}")
    summary = {
        "recording": rec_dir.name,
        "plies": len(truth),
        "measured": len(errors),
        "within_0_5": within / len(errors) if errors else 0.0,
        "mae": sum(errors) / len(errors) if errors else float("nan"),
        "worst": sorted(((abs(e - t), p) for p, _, e, t, _ in rows if e is not None), reverse=True)[:3],
        "time_control": tc,
        "increment": inc,
        "cost_ms": 1000 * statistics.mean(costs) if costs else float("nan"),
        "cost_p95_ms": 1000 * float(np.percentile(costs, 95)) if costs else float("nan"),
        "read_ticks": read_ticks,
        "board_ticks": board_ticks,
        "unread_before_board": sum(1 for t, r in readings.items() if r is not None and not board_at[t]),
    }
    if verbose:
        print(
            f"   summary: {within}/{len(errors)} plies within 0.5 s ({summary['within_0_5']:.1%}), "
            f"MAE {summary['mae']:.2f} s, worst {summary['worst']}, "
            f"cost {summary['cost_ms']:.2f} ms/frame (p95 {summary['cost_p95_ms']:.2f})"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--learn", type=Path, metavar="OUT", help="learn glyph templates into OUT instead of validating")
    args = parser.parse_args()
    if args.learn:
        learn(args.recordings, args.learn)
        return
    summaries = [validate(rec_dir) for rec_dir in args.recordings]
    if len(summaries) > 1:
        measured = sum(int(s["measured"]) for s in summaries)  # type: ignore[call-overload]
        within = sum(float(s["within_0_5"]) * int(s["measured"]) for s in summaries)  # type: ignore[call-overload]
        mae = sum(float(s["mae"]) * int(s["measured"]) for s in summaries) / max(1, measured)  # type: ignore[call-overload]
        print(f"\n== all: {within / max(1, measured):.1%} of {measured} plies within 0.5 s, MAE {mae:.2f} s")


if __name__ == "__main__":
    main()
