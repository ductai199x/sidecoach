"""How long the vision pipeline takes to see a move: per-frame cost and move-to-commit delay.

A game is rendered frame by frame with `tests/synth.py` (so every move's first frame is known
exactly) and fed at the capture's frame rate through what the panel's loop runs on each frame:
the change gate on the board (`board_gate`, `ChangeGate.changed`), then `GamePipeline.process`.
For each move it reports the frame it first shows on and the frame `MovePlayed` comes out on;
for every frame, the wall time of that tick. Nothing else runs: no engine, no coach, no voice.

Run it with the installed app's interpreter to measure the shipped runtime as the app uses it
(the piece net on the GPU when there is one):

    C:/Users/Tai/AppData/Local/Sidecoach/python/python.exe tools/bench_vision.py
    ... tools/bench_vision.py --width 3840 --height 2160 --cell 120 --fps 10

The installed package is imported first; the repository's root is added after it only so that
`tests.synth` (which the app does not ship) can render the frames.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))

from sidecoach.events import Event, MovePlayed  # noqa: E402
from sidecoach.replay.recording import ChangeGate, board_gate  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from tests.synth import synthetic_board  # noqa: E402

# A Ruy Lopez, 32 plies: quiet moves, captures, castling on both sides.
GAME = (
    *("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"),
    *("Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7"),
    *("c4", "c6", "cxb5", "axb5", "Nc3", "Bb7", "Bg5", "b4", "Nb1", "h6"),
    *("Bh4", "c5"),
)
# Frames given to each position before a move counts as missed, and to the start for its lock.
PATIENCE = 30


@dataclass(frozen=True)
class Screen:
    width: int
    height: int
    cell: int

    @property
    def origin(self) -> tuple[int, int]:
        """The board's top-left corner, centred on the screen."""
        side = self.cell * 8
        return (self.width - side) // 2, (self.height - side) // 2

    def frame(self, board: chess.Board, move: chess.Move | None) -> np.ndarray:
        x0, y0 = self.origin
        return synthetic_board(
            board=board,
            highlight=move,
            width=self.width,
            height=self.height,
            cell=self.cell,
            x0=x0,
            y0=y0,
        )


@dataclass
class Result:
    lock_frames: int | None
    lock_seconds: float
    move_ticks: list[float]  # the first frame showing each move
    still_ticks: list[float]  # every other frame
    commit_frames: list[int | None]  # per move: frames until MovePlayed, None if never


def run(screen: Screen, fps: float, sans: Sequence[str]) -> Result:
    pipe = GamePipeline()
    gate = ChangeGate(keyframe_interval_s=1.0)  # the panel's
    clock = 0.0
    step = 1.0 / fps

    def tick(frame: np.ndarray) -> tuple[list[Event], float]:
        """One tick of the panel's loop (`PanelEngine` in ui/panel.py), timed."""
        nonlocal clock
        start = time.perf_counter()
        gate.region = board_gate(pipe.board, screen.width, screen.height)
        changed = gate.changed(frame, clock)
        events = pipe.process(frame, clock, changed=changed)
        clock += step
        return events, time.perf_counter() - start

    board = chess.Board()
    start_frame = screen.frame(board, None)
    lock_frames: int | None = None
    lock_seconds = 0.0
    still: list[float] = []
    for n in range(1, PATIENCE + 1):
        _, cost = tick(start_frame)
        lock_seconds += cost
        if pipe.reconciler is not None:
            lock_frames = n
            break
    if lock_frames is None:
        return Result(None, lock_seconds, [], still, [None] * len(sans))

    moves: list[float] = []
    commits: list[int | None] = []
    for san in sans:
        move = board.push_san(san)
        frame = screen.frame(board, move)
        commit: int | None = None
        for n in range(1, PATIENCE + 1):
            events, cost = tick(frame)
            (moves if n == 1 else still).append(cost)
            if any(isinstance(event, MovePlayed) for event in events):
                commit = n
                break
        commits.append(commit)
    return Result(lock_frames, lock_seconds, moves, still, commits)


def summary(seconds: list[float]) -> str:
    ordered = sorted(seconds)
    p90 = ordered[max(0, round(0.9 * len(ordered)) - 1)]
    return (
        f"median {statistics.median(ordered) * 1e3:5.1f} ms   p90 {p90 * 1e3:5.1f} ms   "
        f"max {ordered[-1] * 1e3:5.1f} ms   (n={len(ordered)})"
    )


def report(result: Result, screen: Screen, fps: float) -> None:
    print(f"interpreter   {sys.executable}")
    print(
        f"screen        {screen.width}x{screen.height}, board {screen.cell * 8} px, "
        f"{fps:g} fps, {len(result.commit_frames)} moves"
    )
    if result.lock_frames is None:
        print(f"first lock    none in {PATIENCE} frames ({result.lock_seconds * 1e3:.0f} ms)")
        return
    print(
        f"first lock    frame {result.lock_frames}, {result.lock_seconds * 1e3:.0f} ms of compute"
    )
    print(f"tick, move    {summary(result.move_ticks)}")
    print(f"tick, still   {summary(result.still_ticks)}")
    committed = [n for n in result.commit_frames if n is not None]
    missed = len(result.commit_frames) - len(committed)
    if committed:
        frames = statistics.median(committed)
        compute = statistics.median(result.move_ticks) + statistics.median(result.still_ticks) * (
            frames - 1
        )
        wait = (frames - 1) * 1e3 / fps
        print(
            f"commit        on frame {min(committed)}-{max(committed)} showing the move "
            f"(median {frames:g}), missed {missed}"
        )
        print(
            f"move->commit  capture wait 0-{1e3 / fps:.0f} ms + {wait:.0f} ms of later frames "
            f"+ ~{compute * 1e3:.0f} ms compute"
        )


def write_frames(screen: Screen, fps: float, sans: Sequence[str], folder: Path) -> None:
    """The game's positions as PNG files and a `game.json` naming them in order, for the
    phone's `VisionBenchTest`, which feeds them through the phone's own tick."""
    folder.mkdir(parents=True, exist_ok=True)
    board = chess.Board()
    positions = [{"file": "000.png", "san": None}]
    cv2.imwrite(str(folder / "000.png"), screen.frame(board, None))
    for n, san in enumerate(sans, start=1):
        move = board.push_san(san)
        name = f"{n:03d}.png"
        cv2.imwrite(str(folder / name), screen.frame(board, move))
        positions.append({"file": name, "san": san})
    game = {"fps": fps, "patience": PATIENCE, "positions": positions}
    (folder / "game.json").write_text(json.dumps(game, indent=1), encoding="utf-8")
    print(f"{len(positions)} frames of {screen.width}x{screen.height} in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="The vision pipeline's per-frame cost and move-to-commit delay."
    )
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--cell", type=int, default=100, help="square size in pixels")
    parser.add_argument("--fps", type=float, default=10.0, help="the capture's frame rate")
    parser.add_argument(
        "--frames",
        type=Path,
        help="write the game's frames here for the phone's bench instead of measuring",
    )
    args = parser.parse_args()
    screen = Screen(args.width, args.height, args.cell)
    if args.frames is not None:
        write_frames(screen, args.fps, GAME, args.frames)
        return
    report(run(screen, args.fps, GAME), screen, args.fps)


if __name__ == "__main__":
    main()
