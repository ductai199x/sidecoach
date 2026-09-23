"""Record what the desktop reconciler makes of a game, frame by frame.

The reconciler is the hardest thing in the pipeline to get right and the easiest to get
subtly wrong: it decides when a move has settled, when a drag is only a drag, when a badge
may be ignored, when the board is showing history rather than the game. So the port is not
judged by whether it follows a game -- it is judged against the Python's own output, frame
for frame.

This renders games with the desktop's own renderer (`tests/synth.py`, the real chess.com
themes and piece sets), reads every frame with the desktop's own recogniser, writes down
the readings it produced, and runs the desktop's `Reconciler` over them, writing down every
event it emitted on every frame. `ChessReconcileTest` on the phone feeds the same readings
to the ported reconciler and must produce the same events in the same order.

Run from the desktop repo, whose environment has python-chess and the test helpers:

    cd D:/chess-coaching-companion
    uv run python D:/sidecoach-android/tools/make_reconcile_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import chess

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

from sidecoach.events import (  # noqa: E402
    GameEnded,
    Region,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    Unexplained,
    Side,
    Takeback,
)
from sidecoach.state.reconcile import Reconciler  # noqa: E402
from sidecoach.vision.pieces import ThemeModel, orientation_from_start  # noqa: E402
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from tests import synth  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"

# The phone's half-resolution capture: the size `make_fixtures.py` renders at.
WIDTH, HEIGHT = 672, 1496
CELL = 83
X0, Y0 = 4, 300
FPS = 10.0


def reading_json(reading) -> dict:
    """The fields the reconciler reads. Nothing else crosses over."""
    return {
        "kind": reading.kind,
        "symbol": reading.symbol,
        "colour": reading.colour.value if reading.colour is not None else None,
        "settled": reading.settled,
        "hover": reading.hover,
        "covered": reading.covered,
        "tint": reading.tint,
        "drawn": reading.drawn,
        "highlighted": reading.highlighted,
        "guess": reading.guess,
    }


def event_json(event) -> str:
    """One event as a line to compare. The game id is random, so it is left out."""
    if isinstance(event, GameStarted):
        return f"GameStarted {event.kind} {event.user_side.value} {event.start_fen}"
    if isinstance(event, PositionChanged):
        return f"PositionChanged {event.ply} {event.fen}"
    if isinstance(event, MovePlayed):
        return (
            f"MovePlayed {event.ply} {event.san} {event.uci} "
            f"{'user' if event.by_user else 'them'} {event.think_ms}"
        )
    if isinstance(event, Takeback):
        return f"Takeback {event.plies} {event.fen}"
    if isinstance(event, Resync):
        return f"Resync {event.reason} | {event.fen}"
    if isinstance(event, Unexplained):
        return f"Unexplained {event.reason} | {event.fen}"
    if isinstance(event, GameEnded):
        return f"GameEnded {event.result} {event.reason} | {event.pgn} | {event.final_fen}"
    return type(event).__name__


def board_kwargs(**extra) -> dict:
    return dict(
        width=WIDTH, height=HEIGHT, x0=X0, y0=Y0, cell=CELL, coordinates=True, **extra
    )


def plain_game(sans: tuple[str, ...], theme: str, style: str, per_state: int = 3) -> list[dict]:
    """The start position, then every move, each held for a few frames."""
    board = chess.Board()
    states: list[dict] = [board_kwargs(board=board.copy(), theme=theme, style=style)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move)
        )
    return [dict(state, _repeat=per_state) for state in states]


def dragged_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """Every move played by hand: the piece lifted, carried, then landed."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.parse_san(san)
        for progress in (0.25, 0.7):
            states.append(
                dict(
                    board_kwargs(
                        board=board.copy(),
                        theme=theme,
                        style=style,
                        dragging=(move.from_square, move.to_square, progress, progress),
                    ),
                    _repeat=1,
                )
            )
        board.push(move)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=3,
            )
        )
    return states


def animated_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """Every move animated across the board, as the opponent's moves arrive."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.parse_san(san)
        states.append(
            dict(
                board_kwargs(
                    board=board.copy(),
                    theme=theme,
                    style=style,
                    animating=(move.from_square, move.to_square, 0.5),
                ),
                _repeat=1,
            )
        )
        board.push(move)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=3,
            )
        )
    return states


def decorated_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """A game with the user's arrows and right-click marks over it."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for i, san in enumerate(sans):
        move = board.push_san(san)
        extra: dict = {"highlight": move}
        if i % 3 == 1:
            extra["arrows"] = (chess.Move(chess.D2, chess.D4),)
        if i % 3 == 2:
            extra["red_marks"] = (chess.E5, chess.F6)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, **extra), _repeat=3)
        )
    return states


def mating_game(theme: str, style: str) -> list[dict]:
    """The scholar's mate, then the site's badges and the dialog over the board."""
    sans = ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7")
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=3)
        )
    for alpha in (0.3, 0.6, 1.0, 1.0, 1.0):
        states.append(
            dict(
                board_kwargs(
                    board=board.copy(), theme=theme, style=style, result="1-0", dialog=alpha
                ),
                _repeat=2,
            )
        )
    return states


def browsing_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """The game, then the user steps back through the move list and plays on."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=3)
        )
    # Back two plies, held, then a different move from there: a takeback.
    board.pop()
    board.pop()
    states.append(dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=4))
    move = board.push_san("d4")
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=4)
    )
    return states


SCENARIOS = {
    "italian-green-classic": lambda: plain_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"), "green", "classic"
    ),
    "sicilian-brown-wooden": lambda: plain_game(
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"), "brown", "wooden"
    ),
    "castling-green-classic": lambda: plain_game(
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "O-O"),
        "green",
        "classic",
    ),
    "dragged-green-classic": lambda: dragged_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"), "green", "classic"
    ),
    "animated-brown-classic": lambda: animated_game(
        ("d4", "d5", "c4", "e6", "Nc3", "Nf6"), "brown", "classic"
    ),
    "decorated-green-classic": lambda: decorated_game(
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"), "green", "classic"
    ),
    "mate-green-classic": lambda: mating_game("green", "classic"),
    "browsing-green-classic": lambda: browsing_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"), "green", "classic"
    ),
}


def run(name: str, states: list[dict]) -> dict:
    geometry = BoardGeometry(region=Region(X0, Y0, CELL * 8, CELL * 8), cell_px=float(CELL))

    frames: list[dict] = []
    model: ThemeModel | None = None
    reconciler: Reconciler | None = None
    bottom = Side.WHITE
    frame_id = 0
    for state in states:
        repeat = state.pop("_repeat", 1)
        image = synth.synthetic_board(**state)
        for _ in range(repeat):
            ts = frame_id / FPS
            if model is None:
                bottom = orientation_from_start(image, geometry)
                model = ThemeModel.from_start_position(image, geometry, bottom)
                if model is None:
                    raise SystemExit(f"{name}: the theme would not bootstrap")
                reconciler = Reconciler(bottom=bottom, user_side=bottom)
            readings = model.classify_frame(image, geometry)
            events = reconciler.update(readings, ts)
            frames.append(
                {
                    "ts": ts,
                    "readings": [reading_json(readings[r][c]) for r in range(8) for c in range(8)],
                    "events": [event_json(event) for event in events],
                }
            )
            frame_id += 1
    return {
        "name": name,
        "bottom": bottom.value,
        "frames": frames,
        "pgn": reconciler.pgn() if reconciler is not None else "",
        "final_fen": reconciler.board.fen() if reconciler is not None else "",
    }


def main() -> None:
    runs = []
    for name, build in SCENARIOS.items():
        run_data = run(name, build())
        moves = sum(
            1
            for frame in run_data["frames"]
            for line in frame["events"]
            if line.startswith("MovePlayed")
        )
        print(f"{name}: {len(run_data['frames'])} frames, {moves} moves — {run_data['pgn']}")
        runs.append(run_data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reconcile.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    print(f"-> {OUT / 'reconcile.json'}")


if __name__ == "__main__":
    main()
