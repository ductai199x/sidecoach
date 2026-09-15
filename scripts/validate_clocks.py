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

from sidecoach.events import MovePlayed, Side  # noqa: E402
from sidecoach.replay.recording import Recording  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from sidecoach.vision.clock import (  # noqa: E402
    GLYPH_H,
    GLYPH_W,
    SEARCH_CELLS,
    ClockReader,
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
# The reconciler confirms a move about a third of a second after the clocks hand over,
# once the pieces have settled; a handover further than LAG_SPAN away is another ply's.
LAG = 0.3
LAG_SPAN = 1.5
# How far a ply's span may differ from the whole seconds its mover's clock lost.
SLACK = 1.5


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

    One digit renders as several clusters: a glyph lands on different sub-pixel offsets, and
    light text on a dark box is bolder than dark text on a light one. So the labelling works
    on the graph of clusters, not on one exemplar per digit, in three steps.

    1. Between two stored frames of one box less than `MAX_STEP_TICKS` apart, a seconds
       digit that changed fell by one. Those transitions are edges "one less than", and a
       walk over them puts every cluster they touch on the countdown cycle, up to a
       rotation shared by all of them.
    2. A cluster the transitions never caught (a digit that only ever appears in the
       minutes, say) is placed by how much it looks like the clusters already placed: a
       whole component at once, at the rotation its glyphs agree best with them.
    3. The rotation is pinned twice over: the seconds units digit rolls from 0 to 9 exactly
       when the tens-of-seconds digit falls, and the tens-of-seconds digit is never above
       5. The two must agree.
    """
    counts = Counter(labels)
    big = {k for k, n in counts.items() if n >= 0.002 * len(samples)}
    readings: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(dict)
    for sample, k in zip(samples, labels, strict=True):
        if sample.count == 5:
            readings[(sample.rec, sample.side)].setdefault(sample.tick, [-1] * 5)[sample.pos] = k
    colon = Counter(seq[2] for d in readings.values() for seq in d.values()).most_common(1)[0][0]

    def transitions() -> list[tuple[list[int], list[int]]]:
        out = []
        for d in readings.values():
            ticks = sorted(d)
            for before, after in zip(ticks, ticks[1:], strict=False):
                a, b = d[before], d[after]
                if -1 not in a and -1 not in b and after - before <= MAX_STEP_TICKS:
                    out.append((a, b))
        return out

    steps: Counter[tuple[int, int]] = Counter()
    for a, b in transitions():
        if a[:4] == b[:4] and a[4] != b[4]:
            steps[(a[4], b[4])] += 1  # b is one less than a
    # Keep an edge only where it is a cluster's dominant successor and was seen often: two
    # stored frames can straddle two ticks of the display, which would step by two.
    best_step: dict[int, tuple[int, int]] = {}
    for (a, b), n in steps.items():
        if a not in best_step or n > best_step[a][1]:
            best_step[a] = (b, n)
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, (b, n) in best_step.items():
        if n >= 5 and a in big and b in big:
            adjacency[a].append((b, -1))
            adjacency[b].append((a, +1))

    def component(root: int) -> dict[int, int]:
        """Every cluster reachable from `root`, with its digit relative to root's."""
        found = {root: 0}
        queue = [root]
        while queue:
            node = queue.pop(0)
            for other, delta in adjacency[node]:
                value = (found[node] + delta) % 10
                if other in found:
                    if found[other] != value:
                        raise SystemExit(f"countdown graph contradicts itself at cluster {other}")
                else:
                    found[other] = value
                    queue.append(other)
        return found

    order = component(max(adjacency, key=lambda k: counts[k]))
    if len(set(order.values())) != 10:
        raise SystemExit(f"the countdown walk found {len(set(order.values()))} digits, not ten")

    def similarity(k: int, digit: int) -> float:
        return max(
            (
                float(_unit(centroids[k]) @ _unit(centroids[j]))
                for j, v in order.items()
                if v == digit
            ),
            default=-1.0,
        )

    for k in sorted(big - {colon}, key=lambda k: -counts[k]):
        if k in order:
            continue
        group = component(k) if k in adjacency else {k: 0}
        scored = sorted(
            (
                (sum(counts[j] * similarity(j, (v + r) % 10) for j, v in group.items()), r)
                for r in range(10)
            ),
            reverse=True,
        )
        weight = sum(counts[j] for j in group)
        (best, rot), (second, _) = scored[0], scored[1]
        if best / weight >= 0.7 and (best - second) / weight >= 0.05:
            order.update({j: (v + rot) % 10 for j, v in group.items()})
        else:
            print(
                f"clusters {sorted(group)} ({weight} glyphs) unnamed: "
                f"best {best / weight:.2f}, next {second / weight:.2f}"
            )

    # The rotation. The tens-of-seconds digit is never above 5, which leaves one rotation
    # once all six of its digits have been seen. The rollovers check it: where the units
    # digit fell by one and the tens digit changed with it, the tens either fell by one
    # (the units rolled 0 to 9) or rose by one (the mover's clock took its increment).
    tens: Counter[int] = Counter()
    for d in readings.values():
        for seq in d.values():
            if seq[3] in order:
                tens[order[seq[3]]] += 1
    allowed = [r for r in range(10) if all((v + r) % 10 <= 5 for v, n in tens.items() if n >= 20)]
    if len(allowed) != 1:
        raise SystemExit(f"the tens-of-seconds digits allow {allowed}, not one rotation")
    rotation = allowed[0]
    names = {k: str((v + rotation) % 10) for k, v in order.items()}
    names[colon] = ":"
    steps_seen: Counter[int] = Counter()
    for a, b in transitions():
        if a[:3] != b[:3] or a[3] == b[3] or a[3] not in names or b[3] not in names:
            continue
        if a[4] in names and b[4] in names and (int(names[a[4]]) - 1) % 10 == int(names[b[4]]):
            steps_seen[(int(names[b[3]]) - int(names[a[3]])) % 6] += 1
    good = steps_seen[5] + steps_seen[1]
    print(
        f"digits pinned at rotation {rotation}; of {sum(steps_seen.values())} one-second steps "
        f"that changed the tens digit, {steps_seen[5]} rolled over and {steps_seen[1]} took an "
        f"increment, {sum(steps_seen.values()) - good} neither"
    )
    return names


def build_templates(
    samples: list[Sample], labels: list[int], names: dict[int, str], out: Path
) -> None:
    """Average the glyphs of each named cluster into templates and write them to `out`.

    A label keeps one template per box polarity: light text on a dark box is drawn bolder
    than dark text on a light one, and averaging the two blurs both.
    """
    groups: dict[tuple[str, bool], list[Sample]] = defaultdict(list)
    for sample, k in zip(samples, labels, strict=True):
        if k in names:
            groups[(names[k], sample.light)].append(sample)
    template_labels: list[str] = []
    canvases: list[np.ndarray] = []
    advances: list[float] = []
    for (name, light), group in sorted(groups.items(), key=lambda kv: (kv[0][0], not kv[0][1])):
        template_labels.append(name)
        canvases.append(np.mean([s.canvas for s in group], axis=0))
        advances.append(float(statistics.median(s.aspect for s in group)))
        print(
            f"  {name!r} on a {'light' if light else 'dark'} box: {len(group)} glyphs, "
            f"advance {advances[-1]:.2f}"
        )
    # The tenths dot. These recordings are fifteen-minute games and no clock ever fell
    # under a minute, so the dot was never drawn; it is the colon's lower dot, which is
    # how chess.com draws it, and the reader also accepts a colon in the tenths place.
    for i, name in enumerate(list(template_labels)):
        if name == ":":
            dot = canvases[i].copy()
            dot[: GLYPH_H // 2] = 0.0
            template_labels.append(".")
            canvases.append(dot)
            advances.append(advances[i])
    templates = GlyphTemplates(
        tuple(template_labels),
        np.stack(canvases).astype(np.float32),
        np.array(advances, np.float32),
    )
    templates.save(out)
    print(
        f"wrote {out} ({out.stat().st_size} bytes, {len(template_labels)} templates "
        f"of {GLYPH_H}x{GLYPH_W})"
    )


def learn(rec_dirs: list[Path], out: Path) -> None:
    samples: list[Sample] = []
    for rec_dir in rec_dirs:
        got = collect(rec_dir)
        print(f"{rec_dir.name}: {len(got)} glyphs")
        samples += got
    centroids, labels = cluster(samples)
    names = label_clusters(samples, centroids, labels)
    print("clusters:", {k: (names.get(k, "?"), n) for k, n in Counter(labels).most_common()})
    build_templates(samples, labels, names, out)


# Validation


@dataclass
class Landing:
    """A move the pipeline reported, and the tick it reported it on."""

    ply: int
    san: str
    tick: int


@dataclass
class Ply:
    """One ply compared: what the clocks say it took, and what chess.com says."""

    ply: int
    san: str
    think: float | None
    target: float
    note: str


@dataclass
class Replay:
    """Everything one replay of a recording produced, tick by tick."""

    ts: dict[int, float]
    white: dict[int, int | None]
    black: dict[int, int | None]
    running: dict[int, str | None]
    board_at: dict[int, bool]
    landings: list[Landing]
    bottom: Side
    costs: list[float]
    read_ticks: int


def fmt(ms: int | None) -> str:
    if ms is None:
        return "--:--"
    seconds, tenths = divmod(ms // 100, 10)
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}:{seconds:02d}" + (f".{tenths}" if minutes == 0 else "")


def align(landings: list[Landing], truth: list[str]) -> dict[int, int]:
    """Put every landing on the truth line: which ply of chess.com's game it is.

    The pipeline's own ply numbers restart when it resyncs, and a takeback replays moves it
    has already reported, so a landing is matched by its move instead: the occurrence of
    that move, of the right colour, nearest to where the last landing left off.
    """
    on_ply: dict[int, int] = {}
    pos = 0
    for landing in landings:
        hits = [
            k
            for k, san in enumerate(truth)
            if san == landing.san and k % 2 == (landing.ply - 1) % 2
        ]
        if not hits:
            continue
        hit = min(hits, key=lambda k: (abs(k - pos), k))
        on_ply[hit + 1] = landing.tick
        pos = hit + 1
    return on_ply


def changes(series: dict[int, int | None], ts: dict[int, float]) -> list[tuple[float, float]]:
    """(when, by how many seconds) for every change of a clock's display.

    The moment is the midpoint between the last frame showing the old value and the first
    showing the new one, which is the best a display sampled at ten frames a second allows.
    """
    out: list[tuple[float, float]] = []
    previous: int | None = None
    previous_tick = 0
    for tick in sorted(series):
        value = series[tick]
        if value is None:
            continue
        if previous is not None and value != previous:
            out.append((0.5 * (ts[previous_tick] + ts[tick]), (value - previous) / 1000))
        previous, previous_tick = value, tick
    return out


def replay(rec_dir: Path) -> Replay:
    rec = Recording(rec_dir)
    pipe = GamePipeline()
    reader = ClockReader()
    run = Replay({}, {}, {}, {}, {}, [], Side.WHITE, [], 0)
    for tick, frame in rec.frames():
        run.ts[tick.frame_id] = tick.ts
        events = pipe.process(
            frame, tick.ts, changed=tick.stored is not None, frame_id=tick.frame_id
        )
        board = pipe.board if pipe.tracker.verified else None
        run.board_at[tick.frame_id] = board is not None
        reading = None
        if board is not None:
            started = time.perf_counter()
            reading = reader.read(frame, board, tick.ts)
            run.costs.append(time.perf_counter() - started)
        # The reader never learns which side is which; the pipeline does, and maps here.
        top_is_white = pipe.bottom == Side.BLACK
        run.white[tick.frame_id] = run.black[tick.frame_id] = None
        run.running[tick.frame_id] = None
        if reading is not None:
            run.read_ticks += 1
            run.white[tick.frame_id] = reading.top_ms if top_is_white else reading.bottom_ms
            run.black[tick.frame_id] = reading.bottom_ms if top_is_white else reading.top_ms
            if reading.running is not None:
                run.running[tick.frame_id] = (
                    "white" if (reading.running == "top") == top_is_white else "black"
                )
        for event in events:
            if isinstance(event, MovePlayed):
                run.landings.append(Landing(event.ply, event.san, tick.frame_id))
    run.bottom = pipe.bottom or Side.WHITE
    return run


def increment(run: Replay) -> int:
    """The time control's increment, from the rise of a clock at the move that earns it."""
    rises = Counter(
        round(by) for _, by in changes(run.white, run.ts) + changes(run.black, run.ts) if by > 0.5
    )
    # The idle clock shows floor(remaining) and the running clock ceil, so a ten-second
    # increment reads as a nine-second rise at the handover.
    raw = rises.most_common(1)[0][0] if rises else 0
    return min(INCREMENTS, key=lambda i: abs(i - (raw + 1)))


def handovers(run: Replay, on_ply: dict[int, int], plies: int, inc: int) -> dict[int, float]:
    """When each ply's move reached the clocks, read from the clocks themselves.

    The mover's clock rises by the increment at the moment their move lands, which is the
    moment the opponent's clock starts running: the boundary between two plies of thinking.
    The clocks show it within a frame, where the move reaches the reconciler only once the
    pieces have settled, a third of a second later and sometimes much more. Ply 1 starts
    with the game, a second before white's clock first falls, and only when the recording
    caught the clock still at its full value.
    """
    rises = {
        1: [when for when, by in changes(run.white, run.ts) if inc - 1.5 <= by <= inc + 0.5],
        0: [when for when, by in changes(run.black, run.ts) if inc - 1.5 <= by <= inc + 0.5],
    }
    hand: dict[int, float] = {}
    read = [t for t in sorted(run.ts) if run.white.get(t) is not None]
    falls = [when for when, by in changes(run.white, run.ts) if by < -0.5]
    if read and falls:
        full = max(v for v in run.white.values() if v is not None)
        if run.white[read[0]] == full and falls[0] - run.ts[read[0]] <= 1.5:
            hand[0] = falls[0] - 1.0
    for ply in range(1, plies + 1):
        if ply not in on_ply:
            continue
        want = run.ts[on_ply[ply]] - LAG
        floor = max((hand[q] for q in range(ply - 1, -1, -1) if q in hand), default=-1e18)
        near = [when for when in rises[ply % 2] if abs(when - want) <= LAG_SPAN and when > floor]
        if near:
            hand[ply] = min(near, key=lambda when: abs(when - want))
    return hand


def steady(clock: dict[int, int | None], ticks: range) -> int | None:
    """The value a clock shows most often over a span of ticks."""
    seen = [clock[t] for t in ticks if clock.get(t) is not None]
    return Counter(seen).most_common(1)[0][0] if seen else None


def validate(rec_dir: Path, verbose: bool = True) -> dict[str, object]:
    run = replay(rec_dir)
    truth = json.loads((rec_dir / "chesscom.json").read_text(encoding="utf-8"))["moves"]
    sans = [move["san"] for move in truth]
    on_ply = align(run.landings, sans)
    inc = increment(run)
    hand = handovers(run, on_ply, len(truth), inc)
    clocks = {1: run.white, 0: run.black}
    ordered = sorted(run.ts)

    def idle(clock: dict[int, int | None], when: float, after: bool) -> int | None:
        """The clock's steady display just before, or just after, a moment."""
        picks = [
            clock[t]
            for t in ordered
            if clock.get(t) is not None
            and (run.ts[t] > when if after else run.ts[t] < when)
            and abs(run.ts[t] - when) < 1.2
        ]
        if not picks:
            return None
        return picks[0] if after else picks[-1]

    rows: list[Ply] = []
    for ply in range(1, len(truth) + 1):
        target = truth[ply - 1]["time_ds"] / 10.0
        san = sans[ply - 1]
        if ply not in on_ply:
            rows.append(Ply(ply, san, None, target, "the pipeline reported no such move"))
            continue
        if ply not in hand or ply - 1 not in hand:
            rows.append(Ply(ply, san, None, target, "no clock handover near the move"))
            continue
        start, end = hand[ply - 1], hand[ply]
        before = idle(clocks[ply % 2], start, after=False)
        after = idle(clocks[ply % 2], end, after=True)
        lost = None if before is None or after is None else (before - after) // 1000 + inc
        think = end - start
        if lost is None:
            rows.append(Ply(ply, san, None, target, "a clock was unreadable at a handover"))
        elif abs(think - lost) > SLACK:
            # The digits say the mover's clock did not lose what this span implies, so the
            # handover belongs to another ply: the move landed well before the reconciler
            # saw it, or a takeback put time back on the clock.
            rows.append(Ply(ply, san, None, target, f"lost {lost} s over a {think:.1f} s span"))
        else:
            rows.append(Ply(ply, san, think, target, ""))

    measured = [row for row in rows if row.think is not None]
    errors = [abs(row.think - row.target) for row in measured if row.think is not None]
    within = sum(1 for error in errors if error <= 0.5)
    # The time control, as both clocks' first steady reading of this game: the ticks just
    # before the first move landed. The reading when the recording opens is whatever page
    # was up, which for a recording that joins a game in progress is not the time control.
    control = None
    first = min((tick for ply, tick in on_ply.items() if ply == 1), default=None)
    if first is None:
        first = next((t for t in ordered if run.board_at[t]), None)
    if first is not None:
        window = range(max(0, first - 25), first + 1)
        control = (fmt(steady(run.white, window)), fmt(steady(run.black, window)), inc)
    right = Counter()
    for ply in range(1, len(truth) + 1):
        if ply not in hand or ply - 1 not in hand:
            continue
        mover = "white" if ply % 2 == 1 else "black"
        for tick in ordered:
            if hand[ply - 1] + 0.5 < run.ts[tick] < hand[ply] - 0.5 and run.running[tick]:
                right[run.running[tick] == mover] += 1
    if verbose:
        report(rec_dir, run, rows, measured, errors, within, control, hand, right)
    return {
        "recording": rec_dir.name,
        "plies": len(truth),
        "measured": len(measured),
        "within_0_5": within / len(errors) if errors else 0.0,
        "mae": sum(errors) / len(errors) if errors else float("nan"),
        "time_control": control,
        "cost_ms": 1000 * statistics.mean(run.costs) if run.costs else float("nan"),
        "cost_p95_ms": 1000 * float(np.percentile(run.costs, 95)) if run.costs else float("nan"),
        "running_right": right[True] / sum(right.values()) if right else float("nan"),
        "worst": sorted(
            (
                (round(abs(row.think - row.target), 1), row.ply, row.san)
                for row in measured
                if row.think is not None
            ),
            reverse=True,
        )[:3],
    }


def report(
    rec_dir: Path,
    run: Replay,
    rows: list[Ply],
    measured: list[Ply],
    errors: list[float],
    within: int,
    control: tuple[str, str, int] | None,
    hand: dict[int, float],
    right: Counter,
) -> None:
    board_ticks = sum(run.board_at.values())
    blind = sum(1 for on in run.board_at.values() if not on)
    print(f"\n== {rec_dir.name}: {len(rows)} plies, {run.bottom.value} at the bottom")
    if control is not None:
        print(f"   time control read at the start: {control[0]} and {control[1]}, +{control[2]} s")
    print(
        f"   {run.read_ticks} of {board_ticks} ticks with a board read a clock; "
        f"{blind} ticks had no board and the reader was not called"
    )
    if right:
        print(
            f"   the clock reported running was the mover's on {right[True]} of "
            f"{sum(right.values())} ticks between handovers"
        )
    print("   ply  move       clock  chess.com   error")
    for row in rows:
        think = f"{row.think:6.1f}" if row.think is not None else "   n/a"
        error = f"{row.think - row.target:+5.1f}" if row.think is not None else "     "
        flag = " <--" if row.think is not None and abs(row.think - row.target) > 1.0 else ""
        print(
            f"   {row.ply:3d}  {row.san:9s} {think}  {row.target:6.1f}  {error}  {row.note}{flag}"
        )
    for row in measured:
        if row.think is None or abs(row.think - row.target) <= 1.0:
            continue
        print(f"   ply {row.ply} {row.san}: clock {row.think:.1f} s, chess.com {row.target:.1f} s")
        for tick in sorted(run.ts):
            if hand[row.ply - 1] - 0.6 < run.ts[tick] < hand[row.ply] + 0.6:
                print(
                    f"      {tick:5d} {run.ts[tick] - hand[row.ply - 1]:+6.2f} "
                    f"white {fmt(run.white[tick])} black {fmt(run.black[tick])} "
                    f"running {run.running[tick]}"
                )
    skipped = Counter(row.note.split(" over")[0] for row in rows if row.think is None)
    share = within / len(errors) if errors else 0.0
    mae = sum(errors) / len(errors) if errors else float("nan")
    print(
        f"   summary: {within}/{len(errors)} measured plies within 0.5 s ({share:.1%}), "
        f"MAE {mae:.2f} s, {len(rows) - len(measured)} plies not measured {dict(skipped)}"
    )
    print(
        f"   cost {1000 * statistics.mean(run.costs):.2f} ms a frame "
        f"(p95 {1000 * float(np.percentile(run.costs, 95)):.2f} ms) over {len(run.costs)} frames"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument(
        "--learn",
        type=Path,
        metavar="OUT",
        help="learn glyph templates into OUT instead of validating",
    )
    args = parser.parse_args()
    if args.learn:
        learn(args.recordings, args.learn)
        return
    summaries = [validate(rec_dir) for rec_dir in args.recordings]
    if len(summaries) > 1:
        counts = [int(summary["measured"]) for summary in summaries]  # type: ignore[arg-type]
        plies = sum(int(summary["plies"]) for summary in summaries)  # type: ignore[arg-type]
        measured = sum(counts)
        within = sum(
            float(summary["within_0_5"]) * n  # type: ignore[arg-type]
            for summary, n in zip(summaries, counts, strict=True)
        )
        mae = sum(
            float(summary["mae"]) * n  # type: ignore[arg-type]
            for summary, n in zip(summaries, counts, strict=True)
        ) / max(1, measured)
        share = within / max(1, measured)
        print(
            f"\n== all six: {share:.1%} of {measured} measured plies within 0.5 s, "
            f"MAE {mae:.2f} s; {measured} of {plies} plies measured"
        )


if __name__ == "__main__":
    main()
