"""The lowest capture rate that still follows every game: the stress sweep's random games
(tests/test_stress.py) played in real time and read at a given rate.

The sweep shows each state a fixed number of frames (three after a move, every step of a
slide), so no rate can make it miss anything. Here each state lasts what it lasts on a site --
a slide a tenth to three tenths of a second, a drag a flick or a slow carry, an instant bot
reply or a premove a tenth of a second, a think one to six -- and the capture reads the screen
every 1/fps seconds, taking the confirming frame 1/30 s after one that showed a move
(`GamePipeline.awaiting_confirmation`, as the desktop's loop and the phone's CaptureService
do). A game passes as the sweep's does: begun once with the right side at the bottom, every
move committed and nothing else, the board the game's at the end. The phone's Pipeline is a
1:1 port held to the desktop's by its parity fixtures, so the rate holds for both.

    uv run --no-sync python tools/bench_capture_rate.py --fps 10 5 3 2 1 --seeds 100
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecoach.events import GameStarted, MovePlayed  # noqa: E402
from sidecoach.state.pipeline import CONFIRM_INTERVAL_S, GamePipeline  # noqa: E402
from tests.games import Scenario, drag_path, scenario  # noqa: E402
from tests.test_stress import _render  # noqa: E402

# How long the player and the opponent leave a position on screen, in seconds, by weight: a
# premove or an instant bot reply, a quick move, a think.
REPLY_HOLDS = ((0.1, 2), (0.3, 2), (1.0, 3), (2.5, 3), (6.0, 1))
SLIDE_S = (0.1, 0.2, 0.3)  # a site's move animation, start to end
DRAG_S = (0.15, 0.3, 0.6)  # a drag, a flick to a slow carry
SELECT_S = 0.4  # a piece clicked, its hints lit
PREMOVE_S = 0.8  # a premove queued while the opponent thinks


@dataclass
class Segment:
    start: float  # when it is first on screen
    seconds: float
    kw: dict[str, object]
    board: chess.Board
    move_index: int | None = None  # the move whose result it is the first look at


@dataclass
class Result:
    seed: int
    fps: float
    ok: bool
    why: str = ""
    lags: list[float] = field(default_factory=list)  # move on screen -> committed, seconds


def timeline(sc: Scenario, rng: random.Random) -> list[Segment]:
    """The game as it is on screen: each state and how long it stays."""
    out: list[Segment] = []
    board = chess.Board()
    t = 0.0

    def hold(seconds: float, move_index: int | None = None, **kw: object) -> None:
        nonlocal t
        out.append(Segment(t, seconds, kw, board.copy(), move_index))
        t += seconds

    user = chess.WHITE if sc.bottom.value == "white" else chess.BLACK
    hold(1.5)
    for i, (san, decoration) in enumerate(zip(sc.sans, sc.decorations, strict=True)):
        move = board.parse_san(san)
        mover = board.turn
        if decoration == "drag" and mover == user:
            path = drag_path(move, sc.bottom, rng)
            step = rng.choice(DRAG_S) / len(path)
            for where in path:
                hold(step, dragging=where)
        if decoration == "animate":
            slide = rng.choice(SLIDE_S)
            frames = 6
            for k in range(1, frames + 1):
                hold(
                    slide / frames,
                    highlight=move,
                    animating=(move.from_square, move.to_square, k / (frames + 1)),
                )
        if decoration == "select" and mover == user:
            hold(SELECT_S, selected=move.from_square)
        if decoration == "premove" and mover != user:
            after = board.copy()
            after.push(move)
            replies = [
                m
                for m in after.legal_moves
                if after.piece_at(m.from_square) and not after.is_capture(m)
            ]
            if replies:
                hold(PREMOVE_S, premoves=((rng.choice(replies),),))
        board.push(move)
        extra: dict[str, object] = {"highlight": move, "check": True}
        if decoration == "arrow":
            targets = [m for m in board.legal_moves if not board.is_capture(m)]
            if targets:
                extra["arrows"] = (rng.choice(targets),)
        if decoration == "badge":
            extra["badge"] = (move.to_square, rng.choice(["best", "good", "blunder", "brilliant"]))
        seconds = rng.choices([s for s, _ in REPLY_HOLDS], [w for _, w in REPLY_HOLDS])[0]
        hold(seconds, i, **extra)
    hold(3.0)  # quiet at the end, so a pending story can settle
    return out


def play(seed: int, fps: float) -> Result:
    sc = scenario(seed)
    # The same game and the same on-screen timings at every rate: only the reading differs.
    segments = timeline(sc, random.Random(seed * 7919))
    end = segments[-1].start + segments[-1].seconds
    pipe = GamePipeline()
    frames: dict[int, object] = {}
    got: list[str] = []
    lags: list[float] = []
    shown_at = {s.move_index: s.start for s in segments if s.move_index is not None}
    starts = 0
    t = random.Random(seed).uniform(0, 1 / fps)  # the capture's phase against the game
    k = 0
    while t < end:
        while k + 1 < len(segments) and segments[k + 1].start <= t:
            k += 1
        seg = segments[k]
        frame = frames.get(k)
        if frame is None:
            frame = frames[k] = _render(sc, seg.board, **seg.kw)
        for event in pipe.process(frame, t):  # type: ignore[arg-type]
            if isinstance(event, MovePlayed):
                index = len(got)
                got.append(event.san)
                if index in shown_at:
                    lags.append(t - shown_at[index])
            elif isinstance(event, GameStarted):
                starts += 1
        t += CONFIRM_INTERVAL_S if pipe.awaiting_confirmation else 1 / fps
    final = chess.Board()
    for san in sc.sans:
        final.push_san(san)
    if starts != 1 or pipe.bottom != sc.bottom:
        return Result(seed, fps, False, f"{starts} games begun, {pipe.bottom} at the bottom")
    same = pipe.reconciler is not None and pipe.reconciler.board.board_fen() == final.board_fen()
    if not same or sorted(got) != sorted(sc.sans):
        return Result(seed, fps, False, f"committed {got}, played {list(sc.sans)}")
    return Result(seed, fps, True, lags=lags)


def _run(job: tuple[int, float]) -> Result:
    try:
        return play(*job)
    except Exception as exc:  # a crash is a failure of that game, reported, not the sweep's
        return Result(job[0], job[1], False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fps", type=float, nargs="+", default=[10, 7, 5, 3, 2, 1])
    parser.add_argument("--seeds", type=int, default=100, help="games per rate (seeds 1..N)")
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) * 7 // 10), help="processes"
    )
    args = parser.parse_args()
    jobs = [(seed, fps) for fps in args.fps for seed in range(1, args.seeds + 1)]
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_run, jobs, chunksize=4))
    print(f"{'fps':>5} {'games':>6} {'passed':>7}  {'lag median':>10} {'lag p95':>8}  failures")
    for fps in args.fps:
        mine = [r for r in results if r.fps == fps]
        passed = [r for r in mine if r.ok]
        lags = sorted(x for r in passed for x in r.lags)
        p95 = lags[int(0.95 * (len(lags) - 1))] if lags else float("nan")
        med = statistics.median(lags) if lags else float("nan")
        failed = [r for r in mine if not r.ok]
        print(
            f"{fps:5g} {len(mine):6d} {len(passed):7d}  {med * 1e3:8.0f}ms {p95 * 1e3:6.0f}ms"
            f"  {', '.join(str(r.seed) for r in failed[:12])}"
        )
        for r in failed[:3]:
            print(f"      seed {r.seed}: {r.why[:220]}")


if __name__ == "__main__":
    main()
