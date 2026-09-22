"""Render a game as a folder of frames, to play on a device's screen.

The capture service has nothing to watch on an emulator, and on a phone a real game means
waiting for one. This renders a game with the desktop's own renderer and writes one PNG per
state; `debug.BoardActivity` shows them one after another, so the whole pipeline -- capture, tracker, theme, recogniser, reconciler
-- runs against a game it has never seen, on the device, the way it will in a real one.

They go into the debug source set, so they are in the debug APK and never in a release
one: a test harness, not part of what is installed. They are generated, so they are not
committed either.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_game.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402

from tests.synth import synthetic_board  # noqa: E402

OUT = Path("D:/sidecoach-android/app/src/debug/assets/game")

PHONE_W, PHONE_H = 672, 1496
CELL = 83
X0, Y0 = 4, 300

# Thirty plies of a real opening into a middlegame: long enough to see what a game
# costs rather than what a handful of moves do.
GAME = (
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4",
    "cxd4", "Bb4+", "Nc3", "Nxe4", "O-O", "Bxc3", "d5", "Bf6", "Re1", "Ne7",
    "Rxe4", "d6", "Bg5", "Bxg5", "Nxg5", "O-O", "Nxh7", "Kxh7", "Qh5+", "Kg8",
)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    board = chess.Board()
    frames = [
        synthetic_board(
            width=PHONE_W, height=PHONE_H, x0=X0, y0=Y0, cell=CELL,
            board=board.copy(), theme="green", style="cburnett", coordinates=True,
        )
    ]
    for san in GAME:
        move = board.push_san(san)
        frames.append(
            synthetic_board(
                width=PHONE_W, height=PHONE_H, x0=X0, y0=Y0, cell=CELL,
                board=board.copy(), theme="green", style="cburnett", coordinates=True,
                highlight=move,
            )
        )
    for i, frame in enumerate(frames):
        cv2.imwrite(str(OUT / f"{i:02d}.png"), frame)
    print(f"{len(frames)} frames -> {OUT}")
    print("expected: " + " ".join(GAME))


if __name__ == "__main__":
    main()
