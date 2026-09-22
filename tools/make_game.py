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

from tests.synth import draw_clock_box, synthetic_board  # noqa: E402
from sidecoach.vision.clock import GlyphTemplates  # noqa: E402

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


# A 5+2 game, so the clocks fall and take their increment the way a real one does. The
# player is White at the bottom, so White's box is the lower one and is the light one.
START_MS = 5 * 60 * 1000
INCREMENT_MS = 2000
THINK_MS = 6000


def clock_text(ms: int) -> str:
    """The clock as chess.com draws it: M:SS, and tenths under a minute."""
    ms = max(0, ms)
    if ms < 60_000:
        return f"{ms // 60000}:{ms % 60000 // 1000:02d}.{ms % 1000 // 100}"
    return f"{ms // 60000}:{ms % 60000 // 1000:02d}"


def with_clocks(frame, white_ms: int, black_ms: int, running: str, templates) -> None:
    """Both boxes, right-aligned with the board, White's below and Black's above."""
    side = 8 * CELL
    offset = int(round(0.37 * CELL))
    for which, ms, light in (("black", black_ms, False), ("white", white_ms, True)):
        y = Y0 - offset if which == "black" else Y0 + side + offset
        draw_clock_box(
            frame,
            X0 + side,
            y,
            CELL,
            clock_text(ms),
            light=light,
            # The running clock's hand sweeps; the idle one's is a dash.
            icon_angle=(ms % 6000) / 6000.0 * 6.28 if which == running else None,
            templates=templates,
        )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    templates = GlyphTemplates.load()
    board = chess.Board()
    white_ms, black_ms = START_MS, START_MS
    first = synthetic_board(
        width=PHONE_W, height=PHONE_H, x0=X0, y0=Y0, cell=CELL,
        board=board.copy(), theme="green", style="cburnett", coordinates=True,
    )
    with_clocks(first, white_ms, black_ms, "white", templates)
    frames = [first]
    for san in GAME:
        mover = "white" if board.turn == chess.WHITE else "black"
        if mover == "white":
            white_ms = white_ms - THINK_MS + INCREMENT_MS
        else:
            black_ms = black_ms - THINK_MS + INCREMENT_MS
        move = board.push_san(san)
        frame = synthetic_board(
            width=PHONE_W, height=PHONE_H, x0=X0, y0=Y0, cell=CELL,
            board=board.copy(), theme="green", style="cburnett", coordinates=True,
            highlight=move,
        )
        with_clocks(
            frame, white_ms, black_ms, "black" if mover == "white" else "white", templates
        )
        frames.append(frame)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(OUT / f"{i:02d}.png"), frame)
    print(f"{len(frames)} frames -> {OUT}")
    print("expected: " + " ".join(GAME))


if __name__ == "__main__":
    main()
