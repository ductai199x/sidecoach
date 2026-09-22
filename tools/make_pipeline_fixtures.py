"""Record what the desktop's whole pipeline makes of a game, frame by frame.

`make_reconcile_fixtures.py` holds the reconciler to the Python from readings the Python
took. This holds everything before it as well: the change gate on the board's own region,
the tracker's lock, the start-position bootstrap, the recogniser, the site's result badges
and the dialog over the board. The frames are written out as PNGs, so the phone reads the
same pixels, and beside them what the desktop's `GamePipeline` said about each one.

The frames are committed, so they are kept small: the board's cell is 50 px, above the
floor the recogniser needs; everything more than a cell and a half outside the board is
flattened, which costs the detector nothing it is not already tested on (the board fixtures
keep their busy backgrounds) and costs the repository ten megabytes; and a state held for
several frames is stored once and named several times.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_pipeline_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402

from sidecoach.events import (  # noqa: E402
    GameEnded,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    ScreenStateChanged,
    Takeback,
)
from sidecoach.replay.recording import ChangeGate, board_gate  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from tests import synth  # noqa: E402

OUT = Path("D:/sidecoach-android/app/src/androidTest/assets/pipeline")

WIDTH, HEIGHT = 420, 860
CELL = 50
X0, Y0 = 8, 180
FPS = 10.0


def event_json(event) -> str:
    """One event as a line to compare. The game id is random, so it is left out."""
    if isinstance(event, GameStarted):
        return f"GameStarted {event.kind} {event.user_side.value} {event.start_fen}"
    if isinstance(event, PositionChanged):
        return f"PositionChanged {event.ply} {event.fen}"
    if isinstance(event, MovePlayed):
        return (
            f"MovePlayed {event.ply} {event.san} {event.uci} "
            f"{'user' if event.by_user else 'them'}"
        )
    if isinstance(event, Takeback):
        return f"Takeback {event.plies} {event.fen}"
    if isinstance(event, Resync):
        return f"Resync {event.reason} | {event.fen}"
    if isinstance(event, GameEnded):
        return f"GameEnded {event.result} {event.reason} | {event.pgn}"
    if isinstance(event, ScreenStateChanged):
        return f"ScreenState {event.state.value}"
    return type(event).__name__


def flatten_around(frame, pad_cells: float = 1.5):
    """Everything well outside the board set to one colour: the board's own pixels are
    untouched, and what is left compresses to a fraction of a busy background."""
    pad = int(round(pad_cells * CELL))
    x0 = max(0, X0 - pad)
    y0 = max(0, Y0 - pad)
    x1 = min(frame.shape[1], X0 + 8 * CELL + pad)
    y1 = min(frame.shape[0], Y0 + 8 * CELL + pad)
    keep = frame[y0:y1, x0:x1].copy()
    frame[:, :] = (32, 32, 32)
    frame[y0:y1, x0:x1] = keep
    return frame


def board_kwargs(**extra) -> dict:
    return dict(width=WIDTH, height=HEIGHT, x0=X0, y0=Y0, cell=CELL, coordinates=True, **extra)


def scholars_mate(theme: str, style: str) -> list[dict]:
    """The start position, the seven moves of the scholar's mate, the site's badges on the
    two kings, and its dialog fading in over the board."""
    board = chess.Board()
    states = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"):
        move = board.push_san(san)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=2,
            )
        )
    # The badges alone, then the dialog over them, each held for seconds.
    states.append(
        dict(
            board_kwargs(board=board.copy(), theme=theme, style=style, result="1-0"),
            _repeat=20,
        )
    )
    for alpha in (0.4, 1.0):
        states.append(
            dict(
                board_kwargs(
                    board=board.copy(), theme=theme, style=style, result="1-0", dialog=alpha
                ),
                _repeat=20,
            )
        )
    return states


def a_game_and_a_takeback(theme: str, style: str) -> list[dict]:
    """Six plies, then two taken back and a different move played from there."""
    board = chess.Board()
    states = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in ("d4", "d5", "c4", "e6", "Nc3", "Nf6"):
        move = board.push_san(san)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=2,
            )
        )
    board.pop()
    board.pop()
    states.append(dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=4))
    move = board.push_san("Nf3")
    states.append(
        dict(
            board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=4
        )
    )
    return states


def a_resignation(theme: str, style: str) -> list[dict]:
    """Four plies and then the site's badges on the two kings: a game the board cannot end,
    which only the badges can (`pipeline._end_on_badges`)."""
    board = chess.Board()
    states = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in ("e4", "c5", "Nf3", "d6"):
        move = board.push_san(san)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=2,
            )
        )
    states.append(
        dict(
            board_kwargs(board=board.copy(), theme=theme, style=style, result="1-0"), _repeat=40
        )
    )
    return states


SCENARIOS = {
    "mate-green": lambda: scholars_mate("green", "cburnett"),
    "resign-green": lambda: a_resignation("green", "cburnett"),
    "takeback-brown": lambda: a_game_and_a_takeback("brown", "cburnett"),
}


def run(name: str, states: list[dict]) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    pipeline = GamePipeline()
    gate = ChangeGate()
    frames: list[dict] = []
    frame_id = 0
    for index, state in enumerate(states):
        repeat = state.pop("_repeat", 1)
        image = flatten_around(synth.synthetic_board(**state))
        # A state held for several frames is one file, named several times: what the
        # pipeline sees is the same pixels again, which is what a still board is.
        path = folder / f"{index:03d}.png"
        cv2.imwrite(str(path), image)
        stored = cv2.imread(str(path), cv2.IMREAD_COLOR)
        for _ in range(repeat):
            ts = frame_id / FPS
            gate.region = board_gate(pipeline.board, WIDTH, HEIGHT)
            changed = gate.changed(stored, ts)
            events = pipeline.process(stored, ts, changed=changed, frame_id=frame_id)
            frames.append(
                {
                    "file": f"{name}/{index:03d}.png",
                    "ts": ts,
                    "changed": changed,
                    "events": [event_json(event) for event in events],
                }
            )
            frame_id += 1
    rec = pipeline.reconciler
    return {
        "name": name,
        "frames": frames,
        "pgn": rec.pgn() if rec is not None else "",
        "final_fen": rec.board.fen() if rec is not None else "",
        "bootstraps": pipeline.bootstraps,
    }


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    runs = []
    for name, build in SCENARIOS.items():
        data = run(name, build())
        moves = sum(
            1 for f in data["frames"] for line in f["events"] if line.startswith("MovePlayed")
        )
        ends = [line for f in data["frames"] for line in f["events"] if line.startswith("GameEnded")]
        print(f"{name}: {len(data['frames'])} frames, {moves} moves — {data['pgn']}")
        for end in ends:
            print(f"    {end}")
        runs.append(data)
    (OUT / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*.png"))
    print(f"-> {OUT} ({size / 1e6:.1f} MB of frames)")


if __name__ == "__main__":
    main()
