"""A recording of a game on "the site you play on" (tools/media/site.html), for the README's
live pictures: the real pipeline reads it through `sidecoach panel --replay`, exactly as it
reads a screen. Never shipped; development only.

A script is a list of moves in SAN, each with the seconds the side to move thinks before it
(the clock runs down by as much); the recording holds the position for that long, then slides
the piece over 0.25 s, filmed at the recording's rate. Looks change the board's colours and
piece set (Lichess's open-licensed ones, desktop/static/pieces/) from a ply on.

    uv run --with playwright --no-sync python tools/media/film_site.py legal data/media/rec-legal

Scripts: SCRIPTS below. It drives the installed Microsoft Edge (Playwright's msedge channel:
no browser download).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import cv2
import numpy as np
from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.events import Region  # noqa: E402
from sidecoach.replay.recording import Recorder  # noqa: E402

WIDTH, HEIGHT = 1280, 720
FPS = 10
SLIDE_S = 0.25


@dataclass(frozen=True)
class Look:
    set: str
    light: str
    dark: str
    size: int = 512


BROWN = Look("cburnett", "#f0d9b5", "#b58863")
BLUE = Look("merida", "#dee3e6", "#8ca2ad", 448)
GREY = Look("chessnut", "#e0e0e0", "#a0a0a0", 528)
OLIVE = Look("papercut", "#eeeed2", "#9a9a6a", 400)
PURPLE = Look("fantasy", "#e8e0f0", "#9f90b0", 480)


@dataclass
class Script:
    white: tuple[str, int]
    black: tuple[str, int]
    bottom: chess.Color
    clock_s: float
    # (SAN, seconds thought before it)
    moves: list[tuple[str, float]]
    # seconds the start position is held, and the end
    lead_s: float = 3.0
    tail_s: float = 6.0
    # ply index -> look from that ply on (0: the start)
    looks: dict[int, Look] = field(default_factory=lambda: {0: BROWN})
    size: int = 512


def _quick(sans: str, think: float = 1.2) -> list[tuple[str, float]]:
    return [(san, think) for san in sans.split()]


SCRIPTS: dict[str, list[Script]] = {
    # Legal's mate: White, stuck at move 5, finds the queen sacrifice and mates in three.
    "legal": [
        Script(
            white=("you", 1350),
            black=("rook_and_roll", 1310),
            bottom=chess.WHITE,
            clock_s=600,
            moves=[
                ("e4", 2),
                ("e5", 3),
                ("Nf3", 3),
                ("d6", 4),
                ("Bc4", 4),
                ("Bg4", 5),
                ("Nc3", 5),
                ("g6", 4),
                ("Nxe5", 31),
                ("Bxd1", 4),
                ("Bxf7+", 4),
                ("Ke7", 3),
                ("Nd5#", 4),
            ],
            # The panel starts reading at once; its page opens a few seconds later.
            lead_s=10,
        )
    ],
    # Any board: games in four looks, each learned from its start position, one played as
    # Black, and boards resized (and so moved) mid-game.
    "looks": [
        Script(
            white=("you", 1350),
            black=("knightowl", 1380),
            bottom=chess.WHITE,
            clock_s=300,
            moves=_quick("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6"),
            lead_s=10,
            tail_s=2,
            looks={0: BLUE, 5: Look("merida", "#dee3e6", "#8ca2ad", 360)},
        ),
        Script(
            white=("pawnstar", 1290),
            black=("you", 1350),
            bottom=chess.BLACK,
            clock_s=300,
            moves=_quick("d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O"),
            lead_s=2.5,
            tail_s=2,
            looks={0: GREY},
        ),
        Script(
            white=("you", 1350),
            black=("quietmove", 1335),
            bottom=chess.WHITE,
            clock_s=180,
            moves=_quick("e4 e6 d4 d5 Nc3 Bb4 e5 c5"),
            lead_s=2.5,
            tail_s=2,
            looks={0: OLIVE, 5: Look("papercut", "#eeeed2", "#9a9a6a", 520)},
        ),
        Script(
            white=("you", 1350),
            black=("tempo_tom", 1360),
            bottom=chess.WHITE,
            clock_s=600,
            moves=_quick("c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5"),
            lead_s=2.5,
            tail_s=4,
            looks={0: PURPLE},
        ),
    ],
}


def _state(board: chess.Board, script: Script, clocks: list[float], last: list[str]) -> dict:
    pieces = {
        chess.square_name(sq): ("w" if p.color else "b") + p.symbol().upper()
        for sq, p in board.piece_map().items()
    }
    replay = chess.Board()
    san = []
    for move in board.move_stack:
        san.append(replay.san(move))
        replay.push(move)
    return {
        "pieces": pieces,
        "last": last,
        "bottom": "w" if script.bottom == chess.WHITE else "b",
        "turn": "w" if board.turn == chess.WHITE else "b",
        "clocks": clocks,
        "san": san,
        "white": {"name": script.white[0], "rating": script.white[1]},
        "black": {"name": script.black[0], "rating": script.black[1]},
    }


def _shot(page: Page) -> np.ndarray:
    png = page.screenshot(type="png")
    return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def film(scripts: list[Script], out: Path) -> None:
    """Each script's game, one after another on the same page."""
    page_url = (Path(__file__).resolve().parent / "site.html").as_uri()
    recorder = Recorder(
        out, monitor=0, region=Region(0, 0, WIDTH, HEIGHT), fps=FPS, keyframe_interval_s=1.0
    )
    ts = 0.0

    def hold(frame: np.ndarray, seconds: float) -> None:
        nonlocal ts
        for _ in range(max(1, round(seconds * FPS))):
            recorder.write(frame, ts)
            ts += 1 / FPS

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        page.goto(page_url)
        page.wait_for_function("window.site !== undefined")
        # When each ply of each game landed on the page.
        landed: list[list[float]] = []
        for script in scripts:
            board = chess.Board()
            clocks = [script.clock_s, script.clock_s]
            last: list[str] = []
            landed.append([])

            def show() -> None:
                page.evaluate("s => window.site.show(s)", _state(board, script, clocks, last))  # noqa: B023
                page.evaluate("document.fonts.ready")

            def apply_look(ply: int) -> None:
                look = script.looks.get(ply)  # noqa: B023
                if look is not None:
                    page.evaluate(
                        "l => window.site.setLook(l)",
                        {
                            "set": look.set,
                            "light": look.light,
                            "dark": look.dark,
                            "size": look.size,
                        },
                    )

            apply_look(0)
            show()
            hold(_shot(page), script.lead_s)
            for ply, (san, think) in enumerate(script.moves):
                mover = 0 if board.turn == chess.WHITE else 1
                # The clock ticks while the side thinks: one frame a second of the hold.
                for _ in range(int(think)):
                    clocks[mover] -= 1
                    show()
                    hold(_shot(page), 1.0)
                clocks[mover] -= think - int(think)
                move = board.parse_san(san)
                frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
                for k in range(1, round(SLIDE_S * FPS)):
                    page.evaluate(f"window.site.slide('{frm}', '{to}', {k / (SLIDE_S * FPS)})")
                    hold(_shot(page), 1 / FPS)
                board.push(move)
                landed[-1].append(round(ts, 2))
                last = [frm, to]
                apply_look(ply + 1)
                show()
                hold(_shot(page), 0.2)
            hold(_shot(page), script.tail_s)
        browser.close()
    recorder.close()
    # For setting the page beside the panel (compose.py).
    (out / "plies.json").write_text(json.dumps(landed), encoding="utf-8")
    print(f"{out}: {ts:.1f} s, {recorder.ticks} ticks, {recorder.stored} stored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("script", choices=sorted(SCRIPTS))
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    film(SCRIPTS[args.script], args.out)


if __name__ == "__main__":
    main()
