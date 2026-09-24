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

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import cv2  # noqa: E402

from sidecoach.events import (  # noqa: E402
    BoardUnread,
    GameEnded,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    ScreenStateChanged,
    Takeback,
    Unexplained,
)
from sidecoach.replay.recording import ChangeGate, board_gate  # noqa: E402
from sidecoach.state.pipeline import GamePipeline  # noqa: E402
from sidecoach.vision.theme_cache import ThemeCache  # noqa: E402
from tests import synth  # noqa: E402

OUT = Path("D:/sidecoach-android/app/src/androidTest/assets/pipeline")

# Every frame written, by its pixels: a board shown again -- in the same scenario or another --
# is the file already written, named again, so the committed frames stay few.
WRITTEN: dict[str, tuple[str, object]] = {}

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
    if isinstance(event, Unexplained):
        return f"Unexplained {event.reason} | {event.fen}"
    if isinstance(event, BoardUnread):
        return f"BoardUnread {event.reason}"
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


GAME = [
    "e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4",
    "cxd4", "Bb4+", "Bd2", "Bxd2+", "Nbxd2", "d5", "exd5", "Nxd5", "Qb3", "Nce7",
    "Qb5+", "c6",
]  # fmt: skip


def _played(theme: str, style: str, sans: list[str]) -> tuple[chess.Board, list[dict]]:
    board = chess.Board()
    states = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=2)
        )
    return board, states


def _replace(board: chess.Board, states: list[dict], theme: str, style: str, sans: list[str]) -> None:
    """The player looks at the position a moment, then plays on from it."""
    states.append(dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3))
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=3)
        )


def quick_takebacks(theme: str, style: str, per_position: int, slide: bool) -> list[dict]:
    """Takeback pressed again and again, fast (Tai's game on the Pixel, 2026-09-23): back two
    plies a press from ply 22 to ply 10, each earlier position up for a frame or two -- the
    pieces caught sliding back in between -- or all twelve at once; then a different move."""
    board, states = _played(theme, style, GAME)
    for _press in range(6):
        for _ply in range(2):
            move = board.peek()
            if slide:
                states.append(
                    dict(
                        board_kwargs(
                            board=board.copy(), theme=theme, style=style,
                            animating=(move.to_square, move.from_square, 0.5),
                        ),
                        _repeat=1,
                    )
                )
            board.pop()
        if per_position:
            states.append(
                dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=per_position)
            )
    _replace(board, states, theme, style, ["e5", "d5"])
    return states


def an_unexplained_stretch(theme: str, style: str) -> list[dict]:
    """A board that keeps changing and never settles, with cells the game cannot account for,
    for longer than the reconciler waits (`resync_after`): a note, `Unexplained`, and the game
    kept -- then back eleven plies with a different move, counted from the same game."""
    board, states = _played(theme, style, GAME[:12])
    odd = []
    for knight in (chess.H3, chess.H4):
        shown = board.copy()
        for square in (chess.A2, chess.B2, chess.G2):
            shown.remove_piece_at(square)
        shown.set_piece_at(knight, chess.Piece(chess.KNIGHT, chess.BLACK))
        odd.append(shown)
    for i in range(170):
        states.append(
            dict(board_kwargs(board=odd[i % 2].copy(), theme=theme, style=style), _repeat=1)
        )
    for _ in range(11):
        board.pop()
    _replace(board, states, theme, style, ["Nf6", "e5"])
    return states


# The puzzle Tai had up on the Pixel (2026-09-23), just after Black's d5xc4.
PUZZLE = "6r1/pp3k2/4b2R/2p1b1p1/2p3P1/P5P1/1P2BP2/2K2N2 w - - 0 1"
PUZZLE_LAST = chess.Move.from_uci("d5c4")


def piece_sets(theme: str) -> list[dict]:
    """Sessions one after another on one theme cache (`_session`: a fresh pipeline, as a
    Stop and Start is), two piece sets on one board's colours. A set learned from its start
    position; a puzzle in a set not learned, which is not adopted and is explained once after
    a few seconds; that set learned beside the first; its puzzle held with a piece selected
    (not explained), then let go and adopted; the first set's puzzle adopted with its own;
    and a puzzle without coordinates, explained as its orientation (Tai's Pixel, 2026-09-23:
    a test board's set under chess.com's green, his puzzle never registered)."""

    def start(style: str) -> dict:
        return dict(board_kwargs(board=chess.Board(), theme=theme, style=style), _repeat=5)

    def puzzle(style: str, seconds: float, **extra) -> dict:
        kwargs = board_kwargs(
            board=chess.Board(PUZZLE), theme=theme, style=style, highlight=PUZZLE_LAST
        )
        kwargs.update(extra)
        return dict(kwargs, _repeat=int(round(seconds * FPS)))

    return [
        dict(start("cburnett"), _session=True),
        dict(puzzle("staunty", 8.0), _session=True),
        dict(start("staunty"), _session=True),
        dict(puzzle("staunty", 8.0, selected=chess.E2), _session=True),
        puzzle("staunty", 3.0),
        dict(puzzle("cburnett", 3.0), _session=True),
        dict(puzzle("staunty", 8.0, coordinates=False), _session=True),
    ]


def puzzle_flip(theme: str) -> list[dict]:
    """A set learned from both sides, a puzzle for White followed, then the next one shown
    from Black's side: turned round from its labels at once, from the readings already
    taken (Tai's Pixel, 2026-09-23: five to ten seconds a puzzle, and stuck)."""

    def start(bottom) -> dict:
        return dict(
            board_kwargs(board=chess.Board(), theme=theme, style="cburnett", bottom=bottom),
            _repeat=5,
        )

    second = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1")
    return [
        dict(start(synth.Side.WHITE), _session=True),
        dict(start(synth.Side.BLACK), _session=True),
        dict(
            board_kwargs(
                board=chess.Board(PUZZLE), theme=theme, style="cburnett", highlight=PUZZLE_LAST
            ),
            _repeat=30,
            _session=True,
        ),
        dict(
            board_kwargs(board=second, theme=theme, style="cburnett", bottom=synth.Side.BLACK),
            _repeat=20,
        ),
    ]


def fast_reply(theme: str) -> list[dict]:
    """Lichess 000rO as chess.com plays it: the setup move a6-a5 highlighted, then the
    solver's Bc7 and the instant reply Kb4 seen together. Adopted with the side to move open,
    the two moves fit either order and nothing was recorded; the highlight says who moved
    (desktop 6c571dd, `reconcile.last_move_seen`)."""
    board = chess.Board("3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58")
    black = synth.Side.BLACK
    states = [
        dict(board_kwargs(board=chess.Board(), theme=theme, style="cburnett"), _repeat=5, _session=True),
        dict(board_kwargs(board=board.copy(), theme=theme, style="cburnett", bottom=black), _repeat=2),
    ]
    setup = board.push_uci("a6a5")
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style="cburnett", bottom=black, highlight=setup), _repeat=6)
    )
    board.push_uci("e5c7")
    reply = board.push_uci("a5b4")
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style="cburnett", bottom=black, highlight=reply), _repeat=6)
    )
    return states


def flip_slide(theme: str) -> list[dict]:
    """The next puzzle shown from Black's side while its setup move Bb2 still slides in, the
    frame held as a capture at two a second holds it: the sliding bishop read as two empty
    squares, and the board adopted without it lost every move after. A turned board is now
    adopted only when it reads the same twice (desktop 6c571dd)."""
    board = chess.Board("4rrk1/ppp2pp1/7p/3n4/3P3q/1P2p2P/P5P1/R1BQRBK1 w - - 2 23")
    black = synth.Side.BLACK
    setup = chess.Move.from_uci("c1b2")
    states = [
        dict(board_kwargs(board=chess.Board(), theme=theme, style="cburnett"), _repeat=5, _session=True),
        dict(
            board_kwargs(
                board=board.copy(),
                theme=theme,
                style="cburnett",
                bottom=black,
                animating=(setup.from_square, setup.to_square, 0.5),
            ),
            _repeat=4,
        ),
    ]
    board.push(setup)
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style="cburnett", bottom=black, highlight=setup), _repeat=10)
    )
    move = board.push_uci("h4f2")
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style="cburnett", bottom=black, highlight=move), _repeat=6)
    )
    return states


SCENARIOS = {
    "mate-green": lambda: scholars_mate("green", "cburnett"),
    "resign-green": lambda: a_resignation("green", "cburnett"),
    "takeback-brown": lambda: a_game_and_a_takeback("brown", "cburnett"),
    "quick-takebacks-green": lambda: quick_takebacks("green", "cburnett", 2, True),
    "takeback-jump-green": lambda: quick_takebacks("green", "cburnett", 0, False),
    "unexplained-green": lambda: an_unexplained_stretch("green", "cburnett"),
    "piece-sets-green": lambda: piece_sets("green"),
    "puzzle-flip-green": lambda: puzzle_flip("green"),
    "fast-reply-green": lambda: fast_reply("green"),
    "flip-slide-green": lambda: flip_slide("green"),
}


def run(name: str, states: list[dict]) -> dict:
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    # A scenario in sessions shares one theme cache between them, as the app's does.
    themes = Path(tempfile.mkdtemp()) if any(s.get("_session") for s in states) else None
    cache = ThemeCache(themes) if themes is not None else None
    pipeline = GamePipeline(cache=cache)
    gate = ChangeGate()
    frames: list[dict] = []
    frame_id = 0
    for index, state in enumerate(states):
        repeat = state.pop("_repeat", 1)
        session = state.pop("_session", False)
        if session:
            pipeline = GamePipeline(cache=cache)
            gate = ChangeGate()
        # A state held for several frames is one file, named several times: what the
        # pipeline sees is the same pixels again, which is what a still board is; and a
        # board seen before is the file it was written to then.
        image = flatten_around(synth.synthetic_board(**state))
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        if digest in WRITTEN:
            name_of, stored = WRITTEN[digest]
        else:
            path = folder / f"{index:03d}.png"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            stored = cv2.imread(str(path), cv2.IMREAD_COLOR)
            name_of = f"{name}/{index:03d}.png"
            WRITTEN[digest] = (name_of, stored)
        for i in range(repeat):
            ts = frame_id / FPS
            gate.region = board_gate(pipeline.board, WIDTH, HEIGHT)
            changed = gate.changed(stored, ts)
            events = pipeline.process(stored, ts, changed=changed, frame_id=frame_id)
            frame = {
                "file": name_of,
                "ts": ts,
                "changed": changed,
                "events": [event_json(event) for event in events],
            }
            if session and i == 0:
                frame["session"] = True
            frames.append(frame)
            frame_id += 1
    rec = pipeline.reconciler
    data = {
        "name": name,
        "frames": frames,
        "pgn": rec.pgn() if rec is not None else "",
        "final_fen": rec.board.fen() if rec is not None else "",
        "bootstraps": pipeline.bootstraps,
    }
    if themes is not None:
        data["cache"] = True
        data["themes"] = len(list(themes.glob("theme-*.pkl")))
        shutil.rmtree(themes)
    return data


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
        for f in data["frames"]:
            for line in f["events"]:
                if line.startswith(("BoardUnread", "GameStarted")) and "themes" in data:
                    print(f"    {f['ts']:.1f}s {line}")
        if "themes" in data:
            print(f"    {data['themes']} learned sets in the cache")
        runs.append(data)
    (OUT / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*.png"))
    print(f"-> {OUT} ({size / 1e6:.1f} MB of frames)")


if __name__ == "__main__":
    main()
