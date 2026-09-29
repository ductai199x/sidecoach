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

The frames land beside this file's checkout (`app/src/androidTest/assets/pipeline`), so a
worktree's copy writes into that worktree.

A scenario may also do what the desktop's pipeline tests do to a pipeline between frames
(`_call`: a reset, the panel's Flip re-reading the board, the End game menu, a side forced, a
clock as a reader would have left it, a reconciler's tunable), and may say whether a frame
counts as changed (`_changed`) where the desktop's test passes it by hand; the phone does the
same at the same point (the parity audit's st-T04..st-T08). The parity audit of 2026-09-29
added the desktop's pipeline tests of a game's end (the dialog's and the badges' commit of a
pending mate), of decisions the phone had replayed only as reconciler ops, and of a cold
start's notice and reads, drawn as those tests draw them (`_desk`): a state may name its first
frame (`_mark`) for its EXPECT, and a scenario may count its last session's reads of the theme
cache (`COUNT_READS`, `reads`), which the phone counts too.
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
import numpy as np  # noqa: E402

from sidecoach.events import (  # noqa: E402
    BoardUnread,
    GameEnded,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    ScreenStateChanged,
    Side,
    Takeback,
    Unexplained,
)
from sidecoach.replay.recording import ChangeGate, board_gate  # noqa: E402
from sidecoach.state.pipeline import COLD_NOTICE_S, DIALOG_SECONDS, GamePipeline  # noqa: E402
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from sidecoach.vision.theme_cache import ThemeCache  # noqa: E402
from tests import synth  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "pipeline"

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


def flatten_around(frame, pad_cells: float = 1.5, where: tuple[int, int, int] = (X0, Y0, CELL)):
    """Everything well outside the board set to one colour: the board's own pixels are
    untouched, and what is left compresses to a fraction of a busy background. `where` is the
    board's left, top and cell."""
    bx, by, cell = where
    pad = int(round(pad_cells * cell))
    x0 = max(0, bx - pad)
    y0 = max(0, by - pad)
    x1 = min(frame.shape[1], bx + 8 * cell + pad)
    y1 = min(frame.shape[0], by + 8 * cell + pad)
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


# The next puzzle of Tai's desktop, 2026-09-24 (dump 20260924-165745), before its setup move.
SETUP_LIT = "2k5/7N/3p3q/1n2p2P/4P3/3P4/2P1KPP1/3Q1B1R w - - 0 1"


def setup_lit(theme: str) -> list[dict]:
    """The next puzzle, turned round, read twice with White's queen still on d1 -- the second
    time with d1 and b1 already lit for the setup move Qd1-b1 -- and adopted then: the
    highlight said White had just moved, the queen's arrival on b1 could not be explained,
    and the record invented 1...Qxh7 2.Qb1 2...Qh6 (Tai's desktop, 2026-09-24). A board is
    adopted only when the reads confirming it show the same view: the same pieces and the
    same lit squares (desktop 815af2f, `test_the_next_puzzle_is_not_adopted_as_its_setup_move_lights_up`)."""
    board = chess.Board(SETUP_LIT)
    black = synth.Side.BLACK
    setup = chess.Move.from_uci("d1b1")

    def shown(**extra) -> dict:
        return board_kwargs(
            board=board.copy(), theme=theme, style="cburnett", bottom=black, **extra
        )

    states = [
        dict(board_kwargs(board=chess.Board(), theme=theme, style="cburnett"), _repeat=3),
        dict(shown(), _repeat=4),  # the puzzle, nothing lit yet
        dict(shown(highlight=setup), _repeat=3),  # lit, the queen not yet off d1
    ]
    for progress in (0.35, 0.8):
        states.append(
            shown(highlight=setup, animating=(setup.from_square, setup.to_square, progress))
        )
    board.push(setup)
    states.append(dict(shown(highlight=setup), _repeat=6))
    # The fork the puzzle is about, recorded as the solver's move and nothing else.
    fork = board.push_uci("b5c3")
    states.append(dict(shown(highlight=fork), _repeat=6))
    return states


# After 10 moves of a Ruy Lopez, both sides castled: a quiet middlegame board.
MIDGAME = "r1bq1rk1/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 w - - 4 8"


def reread_selected(theme: str) -> list[dict]:
    """A knight left selected puts chess.com's legal-move dots on the empty squares it can go
    to, the dots read as unknown, and every "Re-read the screen" was refused (Tai's desktop,
    2026-09-24). Nobody asked, the board is not taken while the dots are up; asked -- the
    panel's Re-read on the state's last frame (`_asked`, `GamePipeline.cold_start`) -- a dot
    is the empty square it is (desktop 815af2f,
    `test_asked_to_reread_a_board_with_a_piece_left_selected`)."""
    selected = board_kwargs(board=chess.Board(MIDGAME), theme=theme, selected=chess.F3)
    return [
        dict(board_kwargs(board=chess.Board(), theme=theme), _repeat=5, _session=True),
        dict(selected, _repeat=30, _session=True, _asked=True),
        dict(selected, _repeat=5),
    ]


# The phone's frame period (~9.6 frames a second), as `tests/test_slides.py` samples it, and
# the test's frame: the board at (300, 120), 64 px cells, on 1280 by 800 -- at this tool's 50 px
# the queen's slide never reads as a piece standing on a square, and the scenario would hold
# nothing the rule is for.
SLIDE_PERIOD = 0.104
SLIDE_BOARD = dict(x0=300, y0=120, cell=64, width=1280, height=800, coordinates=True)


def slides(
    start: str, setup: str, moves: list[str], slide: float, power: float, phase: float
) -> list[dict]:
    """A puzzle played at the phone's ~9.6 frames a second, every move sliding for `slide`
    seconds with progress `u ** power` and a faint three-layer trail, the site replying the
    moment the solver's piece lands: desktop 8fadeda's `tests/test_slides.py`, frame for
    frame, drawn on this tool's board. A frame that caught the rook near the centre of c7 read
    as a rook standing there, and the next frame, the rook further on and the reply already
    sliding, explained as the rook moving on (Tai, 2026-09-25: "I played Rxc1+ but it
    misunderstood as Rc7"). Each state is one frame, its time given (`_ts`), drawn on the
    test's own board (`SLIDE_BOARD`, `_where`)."""
    board = chess.Board(start)
    solver = not board.turn
    bottom = synth.Side.WHITE if solver == chess.WHITE else synth.Side.BLACK
    events = [(0.0, chess.Move.from_uci(setup))]
    t = 3.0
    for i, uci in enumerate(moves):
        events.append((t, chess.Move.from_uci(uci)))
        t += slide if i % 2 == 0 else 2.0
    end = t + 1.5

    def frame(b: chess.Board, side, **kw):  # noqa: ANN001, ANN202
        return synth.synthetic_board(board=b, bottom=side, **SLIDE_BOARD, **kw)

    def scene(time: float):  # noqa: ANN202
        b = board.copy()
        last = None
        for t0, move in events:
            if time < t0:
                break
            u = (time - t0) / slide
            if u < 1.0:
                p = u**power
                sq = (move.from_square, move.to_square)
                img = frame(b, bottom, highlight=move, animating=(*sq, p))
                if p > 0.05:
                    g1 = frame(b, bottom, highlight=move, animating=(*sq, max(0.0, p - 0.08)))
                    g2 = frame(b, bottom, highlight=move, animating=(*sq, max(0.0, p - 0.16)))
                    img = (
                        0.70 * img.astype(np.float32)
                        + 0.20 * g1.astype(np.float32)
                        + 0.10 * g2.astype(np.float32)
                    ).astype(np.uint8)
                return img
            b.push(move)
            last = move
        return frame(b, bottom, highlight=last)

    where = (SLIDE_BOARD["x0"], SLIDE_BOARD["y0"], SLIDE_BOARD["cell"])
    states: list[dict] = []
    ts = 0.0
    for _ in range(3):
        ts += SLIDE_PERIOD
        states.append({"_image": frame(chess.Board(), synth.Side.WHITE), "_ts": ts, "_where": where})
    k = 0
    while (time := phase + k * SLIDE_PERIOD) <= end:
        ts += SLIDE_PERIOD
        states.append({"_image": scene(time), "_ts": ts, "_where": where})
        k += 1
    return states


# ---- The desktop's pipeline tests, as scenarios (the parity audit's st-T04..st-T08) -------


def _shown(board: chess.Board, repeat: int, **extra) -> dict:
    """`board` held for `repeat` frames; `extra` overrides the drawing (and may carry the
    state's own keys, `_where`, `_changed`, ...)."""
    kwargs = board_kwargs(board=board.copy(), theme="green", style="cburnett")
    kwargs.update(extra)
    return dict(kwargs, _repeat=repeat)


def _moves(board: chess.Board, sans: tuple[str, ...], repeat: int = 2, **extra) -> list[dict]:
    """Each move played on `board`, its frames with the last-move highlight."""
    states = []
    for san in sans:
        move = board.push_san(san)
        states.append(_shown(board, repeat, highlight=move, **extra))
    return states


def new_theme_new_game() -> list[dict]:
    """`test_pipeline_bootstraps_and_rebootstraps_on_new_theme`: a game on one theme, then the
    next game's start position on another with Black at the bottom -- the old game ended, the
    recogniser learned again and the orientation with it -- and that game followed."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5"))]
    board = chess.Board()
    brown = dict(theme="brown", bottom=Side.BLACK)
    states.append(_shown(board, 3, **brown))
    states += _moves(board, ("d4", "Nf6"), **brown)
    return states


def flipped_before_the_first_move() -> list[dict]:
    """`test_board_flipped_before_the_first_move_keeps_the_game`: the pre-game board watched as
    White, then paired as Black: the board flips before a move, and the same game goes on,
    announced again from Black's side."""
    board = chess.Board()
    states = [_shown(board, 2), _shown(board, 4, bottom=Side.BLACK)]
    states += _moves(board, ("e4", "e5"), bottom=Side.BLACK)
    return states


def reset_then_one_changed_tick() -> list[dict]:
    """`test_reset_forgets_the_game_and_bootstraps_again`, then
    `test_bootstrap_counts_ticks_so_a_fast_first_move_does_not_starve_it`: the game forgotten,
    and the next start position stored on one changed tick and left unchanged for two more
    still bootstraps, in time for a first move a moment later."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5")), {"_call": {"reset": True}}]
    board = chess.Board()
    states.append(dict(_shown(board, 3), _changed=[True, False, False]))
    states += _moves(board, ("d4",))
    return states


def the_lock_moves() -> list[dict]:
    """`test_readings_are_retaken_after_the_lock_moves` and `test_window_moved_mid_game`: a move
    and a window move in one changed tick, then only unchanged ticks; the readings are taken
    again on the new lock, the move is told, and the game goes on there."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4",))]
    # The desktop's test moves the board 30 px at 64 px cells; as far at 50 px, and down, as
    # the frame is no wider than the board.
    moved = dict(y0=Y0 + 23, _where=(X0, Y0 + 23, CELL))
    move = board.push_san("e5")
    states.append(dict(_shown(board, 13, highlight=move, **moved), _changed=[True] + [False] * 12))
    states += _moves(board, ("Nf3",), **moved)
    return states


def _blank() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def covered_then_gone() -> list[dict]:
    """`test_a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing`,
    `test_screen_state_falls_back_to_unknown_when_the_board_is_lost` and
    `test_board_minimised_then_restored`: a dialog that flashes past counts for nothing; one
    that stays marks the screen covered, the game open; lifted, live again. Then the board
    gone (unknown) while a move is played, and back: the move is told. Then the End game menu
    says how it ended, once."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5"))]
    states += [_shown(board, 5, dialog=1.0), _shown(board, 1), _shown(board, 30, dialog=1.0)]
    states.append(_shown(board, 1))
    states.append({"_image": _blank(), "_repeat": 8})
    move = board.push_san("Nf3")
    states.append(_shown(board, 10, highlight=move))
    states.append({"_call": {"end_game": ["1-0", "opponent resigned or lost on time"]}})
    states.append({"_call": {"end_game": ["0-1", "changed my mind"]}})
    states.append(_shown(board, 1, highlight=move))
    return states


def _partly_covered(board: chess.Board, repeat: int, **extra) -> dict:
    """The board with another window over its right three files and beyond."""
    image = synth.synthetic_board(**board_kwargs(board=board.copy(), theme="green", style="cburnett", **extra))
    image[Y0 - 40 : Y0 + 8 * CELL + 40, X0 + 5 * CELL :] = (240, 240, 240)
    return {"_image": image, "_repeat": repeat}


def partly_covered() -> list[dict]:
    """`test_board_partly_covered_by_another_window`: a window over the right third of the
    board; a move on the left commits, one that lands under the window waits for it to go."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5"))]
    move = board.push_san("d4")
    states.append(_partly_covered(board, 4, highlight=move))
    move = board.push_san("Nf6")
    states.append(_partly_covered(board, 4, highlight=move))
    states.append(_shown(board, 4, highlight=move))
    return states


def scrolled_off_and_back() -> list[dict]:
    """`test_board_scrolled_partly_off_screen_and_back`: the page scrolled, the top rows off the
    screen: nothing invented; scrolled back, the game goes on."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5"))]
    states.append(dict(_shown(board, 8, y0=Y0 - 330), _where=(X0, Y0 - 330, CELL)))
    states.append(_shown(board, 6))
    states += _moves(board, ("Nf3",))
    return states


def a_draw_badge_then_a_flag() -> list[dict]:
    """`test_a_draw_badge_is_not_read_until_one_is_on_record` and `test_a_flag_is_a_loss_on_time`:
    the rendered draw badges end nothing; the clock read at zero for White, then the badges of
    Black's win: lost on time."""
    board = chess.Board()
    states = [_shown(board, 2), *_moves(board, ("e4", "e5", "Nf3", "Nc6"))]
    last = board.peek()
    # Every frame read, as the desktop's test reads them (`process` changed by default): the
    # badges are counted on the frames that are read.
    states.append(_shown(board, 4, highlight=last, result="1/2-1/2", _changed=True))
    states.append({"_call": {"last_clock": [0, 240000, None], "last_clock_ts": None}})
    states.append(_shown(board, 2, highlight=last, result="0-1", _changed=True))
    return states


def joined_and_the_clock() -> list[dict]:
    """`test_a_game_joined_from_the_middle_is_adopted_without_being_asked`,
    `test_a_stale_clock_does_not_settle_the_side_to_move` and
    `test_the_side_to_move_of_an_adopted_position_comes_from_the_running_clock`: a learned
    set's board joined in the middle, read three times half a second apart and adopted with
    the side to move open; a clock read long ago says nothing; the running clock read a moment
    ago settles it, once."""
    board = chess.Board(MIDGAME)
    return [
        dict(board_kwargs(board=chess.Board(), theme="green", style="cburnett"), _repeat=3, _session=True),
        dict(_shown(board, 3), _session=True, _ts=0.0, _dt=0.6, _changed=True),
        {"_call": {"last_clock": [300000, 290000, "black"], "last_clock_ts": 1.2 - 10.0}},
        dict(_shown(board, 1), _ts=1.3, _changed=True),
        {"_call": {"last_clock": [300000, 290000, "black"], "last_clock_ts": 1.3}},
        dict(_shown(board, 2), _ts=1.4, _dt=0.1, _changed=True),
    ]


# Ten plies on from the start: only a position resync can follow.
LATER = "r1bqk2r/1pppbppp/p1n2n2/4p3/B3P3/5N2/PPPP1PPP/RNBQ1RK1 w kq - 4 6"


def a_jump_on_a_slow_stream() -> list[dict]:
    """`test_a_new_position_on_a_slow_stream_is_taken_after_a_moment_not_ten_frames`: frames a
    third of a second apart, the pixels new only on the first; the next position is adopted
    once it has stood still three frames and 0.4 s -- a Resync, the side to move pending."""
    return [
        _shown(chess.Board(), 3),
        dict(_shown(chess.Board(LATER), 5), _ts=10.0, _dt=0.33, _changed=[True] + [False] * 4),
    ]


# A position that is legal read either way up, shown after a game with Black at the bottom.
TURNED = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"


def a_resync_the_other_way_up() -> list[dict]:
    """`_resync_reoriented`: the next position, shown the other way up, is adopted by a resync,
    on its head, before the labels read twice a quarter of a second apart turn it round
    (`_flip_if_turned`) -- and the labels then say which side is at the bottom: the board is
    read again that way and begun as a new game, in place of the resync. The side chosen with
    Flip is for a board the labels say nothing about; these say which (desktop 3d0a201)."""
    board = chess.Board()
    states = [_shown(board, 3), *_moves(board, ("e4", "e5"))]
    states.append({"_call": {"forced_bottom": "white"}})
    states.append(_shown(chess.Board(TURNED), 8, bottom=Side.BLACK))
    return states


def labels_read_past_by_the_8() -> list[dict]:
    """`test_labels_cached_wrong_are_read_past_by_the_8_and_learned_again` (desktop 495dee2): a
    start seen through a lock off its board cached the labels as solid blocks
    (`solid_labels`), so every puzzle asked for Flip (Tai's Pixel, 2026-09-27). The 8 reads the
    side with nothing learned: the puzzle from Black's side is taken up by itself, unasked, and
    its labels learned from it; the next puzzle, from White's side, turns round; a fresh
    pipeline on the cache takes the puzzle from Black's side up again. The desktop's test reads
    the cached labels back."""
    turned = dict(board=chess.Board(TURNED), theme="green", style="staunty", bottom=Side.BLACK)
    puzzle = dict(board=chess.Board(PUZZLE), theme="green", style="staunty", highlight=PUZZLE_LAST)

    def watch(kwargs: dict, seconds: float, start: float, **extra) -> dict:
        """`_watch`: held for `seconds`, a tick every 0.3 s, its pixels new only on the first."""
        ticks = len(np.arange(start, start + seconds, 0.3))
        return dict(
            board_kwargs(**kwargs), _repeat=ticks, _ts=start, _dt=0.3,
            _changed=[True] + [False] * (ticks - 1), **extra,
        )  # fmt: skip

    return [
        dict(board_kwargs(board=chess.Board(), theme="green", style="staunty"), _repeat=3, _session=True),
        {"_call": {"solid_labels": True}},
        watch(turned, 8.0, 0.0, _session=True),
        watch(puzzle, 3.0, 20.0),
        watch(turned, 3.0, 30.0, _session=True),
    ]


def flip_a_joined_position() -> list[dict]:
    """`test_flip_re_reads_a_joined_position_the_other_way_up`: no labels to read, Flip says
    White is at the bottom and the board is joined upside down; Flip again says Black, and the
    screen is read the other way and begun again, the bogus game ended."""
    black = dict(bottom=Side.BLACK, coordinates=False)
    board = chess.Board(MIDGAME)
    return [
        dict(board_kwargs(board=chess.Board(), theme="green", style="cburnett", bottom=Side.BLACK), _repeat=3, _session=True),
        dict(_shown(board, 1, **black), _session=True, _ts=0.0, _changed=True),
        {"_call": {"forced_bottom": "white"}},
        dict(_shown(board, 4, **black), _ts=0.6, _dt=0.6, _changed=True),
        {"_call": {"reorient": "black"}, "_ts": 20.0},
    ]


# `tests/test_scenarios.py`'s `Session`: the board at (440, 200) on 1280 by 800, drawn as it
# draws it, every frame read (`process` changed by default).
SESSION_FRAME = dict(width=1280, height=800)
SESSION_X0, SESSION_Y0 = 440, 200


def moved_any_fraction(cell: int, dx: float, dy: float) -> list[dict]:
    """`test_a_board_moved_any_fraction_of_a_cell_is_found_again` (desktop 17c45a6): 1. e4 e5,
    the board moved by (dx, dy) cells, eight frames to settle, then 2. Nf3 Nc6 on it. Moves
    that kept the old lock: two cells and a fraction (the cells it left behind still fit), 0.8
    of a cell (the palette re-read inverted), two whole cells (a window part board and part
    background ranked first), the diagonals."""
    board = chess.Board()
    x0, y0 = SESSION_X0, SESSION_Y0

    def shown(repeat: int, highlight=None) -> dict:  # noqa: ANN001
        image = synth.synthetic_board(
            board=board, x0=x0, y0=y0, cell=cell, highlight=highlight, **SESSION_FRAME
        )
        return {"_image": image, "_repeat": repeat, "_where": (x0, y0, cell), "_changed": True}

    states = [shown(2)]
    for san in ("e4", "e5"):
        states.append(shown(2, board.push_san(san)))
    x0, y0 = int(SESSION_X0 + dx * cell), int(SESSION_Y0 + dy * cell)
    # The lock after the settle, which the test holds to within three pixels of the board.
    states.append(dict(shown(8), _lock=True))
    for san in ("Nf3", "Nc6"):
        states.append(shown(2, board.push_san(san)))
    states.append(shown(3))
    return states


MOVED = {
    "moved-50-0+2.4": (50, 0.0, 2.4),
    "moved-50-2.4+0": (50, -2.4, 0.0),
    "moved-50+2+0": (50, 2.0, 0.0),
    "moved-50-0-0.8": (50, 0.0, -0.8),
    "moved-48+3+3": (48, 3.0, 3.0),
    "moved-64-1.4-1.4": (64, -1.4, -1.4),
    # Its Nc6 did not land on the highlighted square until the landing compared the reference
    # and the piece at the detail both have (desktop cfaa416; the case back in the desktop's
    # test_scenarios.py since 9a568bc).
    "moved-48-1.4-1.4": (48, -1.4, -1.4),
}


# ---- The desktop's pipeline tests drawn as they draw them (the parity audit, 2026-09-29) ------
#
# `tests/games.py`'s `Game(pipeline=True)` and `tests/test_pipeline.py` draw with
# `synthetic_board`'s defaults -- 64 px cells at (300, 120) on 1280 by 800, no coordinates, the
# classic set on green -- and read every frame (`process` changed by default), a tenth of a
# second apart from 0.0: the frame ids here, at FPS. These scenarios are those tests' frames.
DESK_WHERE = (300, 120, 64)
# `tests/test_pipeline.py`'s other board colours.
BROWN_LIGHT = (181, 217, 240)
BROWN_DARK = (99, 136, 181)


def _desk(board: chess.Board, repeat: int, **extra) -> dict:
    """`board` as the desktop's pipeline tests draw it, held for `repeat` frames, each read."""
    return dict(board=board.copy(), **extra, _repeat=repeat, _where=DESK_WHERE, _changed=True)


def _desk_image(image: np.ndarray, repeat: int, **extra) -> dict:
    """A frame the desktop's test composed itself (a dialog over the board), each read."""
    return dict(_image=image, _repeat=repeat, _where=DESK_WHERE, _changed=True, **extra)


def _desk_played(sans: tuple[str, ...], start: int = 2) -> tuple[chess.Board, list[dict]]:
    """`Game(pipeline=True).play(*sans)`: the start for `start` frames, then each move for
    two, highlighted."""
    board = chess.Board()
    states = [_desk(board, start)]
    for san in sans:
        move = board.push_san(san)
        states.append(_desk(board, 2, highlight=move))
    return board, states


def mate_read_once_then_the_dialog() -> list[dict]:
    """`test_the_dialog_confirms_a_mate_read_once`: the opponent's move and the user's
    premoved mate land in one frame, read once, and the dialog comes up before a second: what
    it leaves in sight reads as the premove still queued, and the story waits; the dialog that
    stays is the confirmation (`GamePipeline._dialog_over_the_board`, `commit_pending_terminal`
    after DIALOG_SECONDS). The parity audit (2026-09-29): no phone fixture ran that commit."""
    board, states = _desk_played(("e4", "e5", "Qh5", "Nc6", "Bc4"))
    board.push_san("Nd4")  # c6 and d4 lie under the dialog
    mate = board.push_san("Qxf7#")  # h5 and f7 do not
    states.append(_desk(board, 1, highlight=mate))  # one clean frame
    covered = synth.with_dialog(synth.synthetic_board(board=board, highlight=mate), dark=True)
    states.append(_desk_image(covered, 30, _mark="dialog"))
    return states


def premoved_mate_with_the_badges() -> list[dict]:
    """`test_the_badges_commit_a_premoved_mate_still_settling`: the mate queued as a premove
    lands in the frame of the opponent's move before it, the site's badges stamped at once: a
    story of two plies the board alone settles on its third frame. The badges, read on
    BADGE_FRAMES frames, commit the pending mate they agree with (`_end_on_badges`,
    `commit_pending_terminal`): the game ends on their second frame, both moves in its record,
    and once. The parity audit (2026-09-29): no phone fixture ran that commit."""
    board, states = _desk_played(("e4", "e5", "Bc4", "Nc6", "Qh5"))
    # White queues Qxf7 while Black thinks.
    states.append(_desk(board, 2, premoves=((chess.Move.from_uci("h5f7"),),)))
    board.push_san("Nf6")
    mate = board.push_san("Qxf7#")
    states.append(_desk(board, 2, highlight=mate, check=True, result="1-0", _mark="badges"))
    # The badges stay; the game does not end twice.
    states.append(_desk(board, 2, highlight=mate, check=True, result="1-0"))
    return states


def mate_and_badges_together() -> list[dict]:
    """`test_a_mate_on_the_board_ends_the_game_once_whatever_the_badges_say`: the mating move
    and the badges arrive together and stay; the game ends once, the mate in its record."""
    board, states = _desk_played(("f3", "e5", "g4"))
    move = board.push_san("Qh4#")
    states.append(_desk(board, 4, highlight=move, result="0-1"))
    return states


def the_mated_king_steps_then_the_mate() -> list[dict]:
    """The mate lands with the mated king's own move: 1.e4 e5 2.Qh5, White queues Qxe5, Black
    plays Ke7 and the premove mates in the same frame, the badges and the check glow stamped
    on the king where it now stands, beside the square the record has it on. The badges are
    looked for on the kings' neighbours while the story is pending (`_badged_kings`, `wide`),
    and a badge there names the king's square. No desktop test plays it; it is the stress
    sweep's `together` ending (`test_a_random_game_ends_on_screen`) with the king's step for
    the premoved move. The parity audit (2026-09-29): no phone fixture found a badge on a
    neighbour. Here the story settles on the board's own frames with the search or without it
    (tried on five sets): the scenario runs the phone's neighbour search and holds its events
    to the desktop's, but a search that found nothing would end this game the same."""
    board, states = _desk_played(("e4", "e5", "Qh5"))
    states.append(_desk(board, 2, premoves=((chess.Move.from_uci("h5e5"),),)))
    board.push_san("Ke7")
    mate = board.push_san("Qxe5#")
    states.append(_desk(board, 2, highlight=mate, check=True, result="1-0", _mark="badges"))
    states.append(_desk(board, 2, highlight=mate, check=True, result="1-0"))
    return states


def the_next_game_on_the_same_board() -> list[dict]:
    """`test_next_game_on_the_same_board_is_begun_by_the_reconciler`: a game mated on the
    board, then the next game's start on the same theme, the same way up: no bootstrap runs;
    the reconciler, its game finished, begins the next one, and the pipeline learns the
    recogniser again from that frame (`process`, `_rebuild_model` after the reconciler's own
    GameStarted) -- a second bootstrap counted. The mate's game is not ended again, and the
    next game's moves follow in a record of their own."""
    _, states = _desk_played(("f3", "e5", "g4", "Qh4#"))
    states.append(_desk(chess.Board(), 2))
    second = chess.Board()
    move = second.push_san("d4")
    states.append(_desk(second, 2, highlight=move))
    return states


def browsed_to_the_start() -> list[dict]:
    """`test_browsing_to_the_start_position_keeps_the_game`: the move list's first entry shows
    the start position on the same theme for a while -- no new game, no bootstrap -- and the
    game goes on from where it was."""
    board, states = _desk_played(("e4", "e5"))
    states.append(_desk(chess.Board(), 5))
    move = board.push_san("Nf3")
    states.append(_desk(board, 2, highlight=move))
    return states


def a_new_game_right_after_a_resync() -> list[dict]:
    """`test_new_game_right_after_a_position_resync_is_a_new_game`: a position ten plies on,
    adopted by a resync (`resync_hold_frames` 3, as the test sets it), then the next game's
    start on another theme the other way up: a new game on a new recogniser, Black at the
    bottom, even straight after the resync."""
    _, states = _desk_played(("e4", "e5"))
    states.append({"_call": {"reconciler": {"resync_hold_frames": 3}}})
    states.append(_desk(chess.Board(LATER), 6))
    states.append(
        _desk(chess.Board(), 4, bottom=Side.BLACK, light=BROWN_LIGHT, dark=BROWN_DARK)
    )
    return states


def rethemed_before_the_first_move() -> list[dict]:
    """`test_board_flipped_or_rethemed_before_the_first_move_keeps_the_game`, rethemed: the
    start position seen on other colours before any move -- a new recogniser, the same game
    announced again, and its move follows. (Flipped is `flipped-before-a-move`.)"""
    brown = dict(light=BROWN_LIGHT, dark=BROWN_DARK)
    states = [_desk(chess.Board(), 2), _desk(chess.Board(), 4, **brown)]
    board = chess.Board()
    move = board.push_san("d4")
    states.append(_desk(board, 2, highlight=move, **brown))
    return states


# `tests/test_cold_start.py`: a board left unread is explained after COLD_NOTICE_S.
EXPLAINED_BY = COLD_NOTICE_S + 2.0


def _watched(state: dict, seconds: float, start: float, **extra) -> dict:
    """`_watch`: `state` held on screen for `seconds`, a tick every 0.3 s from `start`, its
    pixels new only on the first: a puzzle sitting there while the person thinks."""
    ticks = len(np.arange(start, start + seconds, 0.3))
    return dict(state, _repeat=ticks, _ts=start, _dt=0.3, _changed=[True] + [False] * (ticks - 1), **extra)


def _puzzle_in(style: str) -> dict:
    """`_puzzle`: Tai's Pixel puzzle in `style` on chess.com's green, with coordinates."""
    return board_kwargs(board=chess.Board(PUZZLE), theme="green", style=style, highlight=PUZZLE_LAST)


def unlearned_away_and_back() -> list[dict]:
    """`test_a_set_not_learned_is_explained_once_then_learned_from_a_start_position`, its
    middle: a puzzle in a set not learned is explained once ("unlearned"); the board gone a
    moment and back, it is explained again (the parity audit, 2026-09-29: the phone's
    `piece-sets-green` explains it once and never takes the board away)."""
    return [
        dict(board_kwargs(board=chess.Board(), theme="green", style="cburnett"), _repeat=3, _session=True),
        _watched(_puzzle_in("staunty"), EXPLAINED_BY, 0.0, _session=True),
        _watched({"_image": synth.busy_background(WIDTH, HEIGHT, 0)}, 1.0, EXPLAINED_BY + 10.0),
        _watched(_puzzle_in("staunty"), EXPLAINED_BY, EXPLAINED_BY + 20.0),
    ]


def a_puzzle_left_still() -> list[dict]:
    """`test_a_puzzle_left_still_is_read_once`: a still puzzle is adopted after the three
    reads a cold start waits for, within two seconds, and read from the cache once, not every
    tick: pixels that did not change read the same (`reads`: the theme cache's `candidates`
    asked in the last session, counted on both sides; the parity audit, 2026-09-29)."""
    return [
        dict(board_kwargs(board=chess.Board(), theme="green", style="staunty"), _repeat=3, _session=True),
        _watched(_puzzle_in("staunty"), 3.0, 0.0, _session=True),
    ]


RESIZED_FIRST = ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
RESIZED_AFTER = ("c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+", "Bd2", "Bxd2+")


def resized_half_again(style: str, small: int, large: int) -> list[dict]:
    """Desktop 8a4b4d7: a board enlarged half again after its set was learned read some
    pieces as others -- the embeddings of a crop drawn larger drift -- until the embedder saw
    every cell at one resolution (`pieces.canonical`). The fantasy set, learned at 32 px cells
    and read at 48 (`Session`'s board), read the game after the resize to 3. c3 and no further
    at desktop 17c45a6, and to its end at 8a4b4d7."""
    board = chess.Board()
    cell = small

    def shown(repeat: int, highlight=None) -> dict:  # noqa: ANN001
        image = synth.synthetic_board(
            board=board, x0=SESSION_X0, y0=SESSION_Y0, cell=cell, highlight=highlight,
            style=style, **SESSION_FRAME,
        )  # fmt: skip
        return {"_image": image, "_repeat": repeat, "_where": (SESSION_X0, SESSION_Y0, cell), "_changed": True}

    states = [shown(2)]
    for san in RESIZED_FIRST:
        states.append(shown(2, board.push_san(san)))
    cell = large
    states.append(shown(8))
    for san in RESIZED_AFTER:
        states.append(shown(2, board.push_san(san)))
    states.append(shown(3))
    return states


SCENARIOS = {
    "mate-green": lambda: scholars_mate("green", "cburnett"),
    "resign-green": lambda: a_resignation("green", "cburnett"),
    "takeback-brown": lambda: a_game_and_a_takeback("brown", "cburnett"),
    "quick-takebacks-green": lambda: quick_takebacks("green", "cburnett", 2, True),
    "takeback-jump-green": lambda: quick_takebacks("green", "cburnett", 0, False),
    # `test_ten_plies_taken_back_quickly[a-frame-each]`: each earlier position up a frame, no
    # slide (the parity audit, 2026-09-29: the desktop's third case, missing here).
    "takebacks-a-frame-each-green": lambda: quick_takebacks("green", "cburnett", 1, False),
    "unexplained-green": lambda: an_unexplained_stretch("green", "cburnett"),
    "piece-sets-green": lambda: piece_sets("green"),
    "puzzle-flip-green": lambda: puzzle_flip("green"),
    "fast-reply-green": lambda: fast_reply("green"),
    "flip-slide-green": lambda: flip_slide("green"),
    "setup-lit-green": lambda: setup_lit("green"),
    "reread-selected-green": lambda: reread_selected("green"),
    "reread-selected-brown": lambda: reread_selected("brown"),
    # Tai's phone puzzle: Rc1+ Rxc1 Rxc1+ Rxc1 Qxd5; Rxc1+ was recorded as Rc7.
    "slide-rook-green": lambda: slides(
        "2r3k1/pQ3p2/6p1/q6p/2r5/P4P2/1P4PP/1K1R3R w - - 0 1",
        "b7d5",
        ["c4c1", "d1c1", "c8c1", "h1c1", "a5d5"],
        0.4,
        2.0,
        0.02,
    ),
    # The desktop dump's: Qd1+ Rxd1 Rxd1+ Qe1; Rxd1 was recorded as Rd3 and Rxd1.
    "slide-queen-green": lambda: slides(
        "2kr2nr/2p5/2p1p1pp/3QPp2/3R4/P3P3/1Pq2PPP/R1B3K1 w - - 0 1",
        "d5a5",
        ["c2d1", "d4d1", "d8d1", "a5e1"],
        0.3,
        2.0,
        0.04,
    ),
    "new-theme-new-game": new_theme_new_game,
    "flipped-before-a-move": flipped_before_the_first_move,
    "reset-one-tick": reset_then_one_changed_tick,
    "lock-moves": the_lock_moves,
    "covered-then-gone": covered_then_gone,
    "partly-covered": partly_covered,
    "scrolled": scrolled_off_and_back,
    "draw-badge-flag": a_draw_badge_then_a_flag,
    "joined-clock": joined_and_the_clock,
    "slow-jump": a_jump_on_a_slow_stream,
    "resync-turned": a_resync_the_other_way_up,
    "flip-joined": flip_a_joined_position,
    "labels-read-past-by-8": labels_read_past_by_the_8,
    **{name: (lambda args=args: moved_any_fraction(*args)) for name, args in MOVED.items()},
    "resized-fantasy-32-48": lambda: resized_half_again("fantasy", 32, 48),
    # The parity audit (2026-09-29): the desktop's pipeline tests of the game's end, of its
    # decisions replayed only as reconciler ops, and of a cold start's notice and reads.
    "dialog-confirms-mate": mate_read_once_then_the_dialog,
    "badges-commit-premoved-mate": premoved_mate_with_the_badges,
    "mate-and-badges": mate_and_badges_together,
    "king-steps-into-mate-badges": the_mated_king_steps_then_the_mate,
    "next-game-same-board": the_next_game_on_the_same_board,
    "browsed-to-start": browsed_to_the_start,
    "new-game-after-resync": a_new_game_right_after_a_resync,
    "rethemed-before-a-move": rethemed_before_the_first_move,
    "unlearned-away-and-back": unlearned_away_and_back,
    "puzzle-read-once": a_puzzle_left_still,
}

# Scenarios whose last session counts the theme cache's reads (`candidates`), as the
# desktop's test counts them: `reads` in the fixture, held to the phone's count.
COUNT_READS = {"puzzle-read-once"}

# Scenarios the phone replays once, not again as a still screen read from its last pixels
# (`PipelineTest.aStillScreenReadAgain...`): each pins a behaviour of the pipeline's, which
# the shortcut the other scenarios hold it to adds nothing to, and a replay costs a minute.
READ_ONCE = {
    "new-theme-new-game",
    "flipped-before-a-move",
    "reset-one-tick",
    "lock-moves",
    "covered-then-gone",
    "partly-covered",
    "scrolled",
    "draw-badge-flag",
    "joined-clock",
    "slow-jump",
    "resync-turned",
    "flip-joined",
    "labels-read-past-by-8",
    *MOVED,
    "resized-fantasy-32-48",
    "dialog-confirms-mate",
    "badges-commit-premoved-mate",
    "mate-and-badges",
    "king-steps-into-mate-badges",
    "next-game-same-board",
    "browsed-to-start",
    "new-game-after-resync",
    "rethemed-before-a-move",
    "unlearned-away-and-back",
    "puzzle-read-once",
}

# What each of those scenarios must show on the desktop, as its test asserts: a fixture that
# does not is no fixture of the behaviour.
EXPECT = {
    "new-theme-new-game": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "d4", "Nf6"]
        and any(line.startswith("GameEnded * new game started") for line in lines)
        and data["bootstraps"] == 2
    ),
    "flipped-before-a-move": lambda lines, data: (
        _sans(lines) == ["e4", "e5"]
        and not any(line.startswith("GameEnded") for line in lines)
        and any(line.startswith("GameStarted game black") for line in lines)
        and data["bootstraps"] == 2
    ),
    "reset-one-tick": lambda lines, data: _sans(lines) == ["e4", "e5", "d4"] and data["bootstraps"] == 2,
    "lock-moves": lambda lines, data: _sans(lines) == ["e4", "e5", "Nf3"],
    "covered-then-gone": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "Nf3"]
        and [line for line in lines if line.startswith("ScreenState")]
        == [
            "ScreenState live_game",
            "ScreenState covered",
            "ScreenState live_game",
            "ScreenState unknown",
            "ScreenState live_game",
            "ScreenState game_over",
        ]
        and [line.split(" |")[0] for line in lines if line.startswith("GameEnded")]
        == ["GameEnded 1-0 opponent resigned or lost on time"]
    ),
    "partly-covered": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "d4", "Nf6"] and not any(line.startswith("Resync") for line in lines)
    ),
    "scrolled": lambda lines, data: _sans(lines) == ["e4", "e5", "Nf3"],
    "draw-badge-flag": lambda lines, data: (
        [line.split(" |")[0] for line in lines if line.startswith("GameEnded")]
        == ["GameEnded 0-1 lost on time"]
    ),
    "joined-clock": lambda lines, data: (
        sum(line.startswith("GameStarted position") for line in lines) == 1
        and [line.split(" |")[0] for line in lines if line.startswith("Resync")]
        == ["Resync side to move settled by the clock: black"]
    ),
    "slow-jump": lambda lines, data: any(
        line.startswith("Resync position adopted") for line in lines
    ),
    "resync-turned": lambda lines, data: (
        not any(line.startswith("Resync") for line in lines)
        and any(line.startswith("GameEnded * board reoriented") for line in lines)
        and any(line.startswith("GameStarted position black") for line in lines)
    ),
    "flip-joined": lambda lines, data: (
        [line.split(" |")[0] for line in lines if line.startswith("GameEnded")]
        == ["GameEnded * board reoriented"]
        and sum(line.startswith("GameStarted position") for line in lines) == 2
    ),
    "labels-read-past-by-8": lambda lines, data: (
        [" ".join(line.split()[:3]) for line in lines if line.startswith(("BoardUnread", "GameStarted"))]
        == [
            "GameStarted game white",
            "GameStarted position black",
            "GameStarted position white",
            "GameStarted position black",
        ]
        and [line.split()[3] for line in lines if line.startswith("GameStarted position")]
        == [chess.Board(fen).board_fen() for fen in (TURNED, PUZZLE, TURNED)]
    ),
    **{
        name: (
            lambda lines, data, args=args: _sans(lines) == ["e4", "e5", "Nf3", "Nc6"]
            and _lock_near(data, *args)
        )
        for name, args in MOVED.items()
    },
    "resized-fantasy-32-48": lambda lines, data: _sans(lines) == [*RESIZED_FIRST, *RESIZED_AFTER],
    "dialog-confirms-mate": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nd4", "Qxf7#"]
        and [line.split()[1] for line in _ends(lines)] == ["1-0"]
        # Settled under the dialog, not before it: DIALOG_SECONDS after it came up.
        and _ts_of(data, "MovePlayed 7 ") - _mark_ts(data, "dialog") >= DIALOG_SECONDS
    ),
    "badges-commit-premoved-mate": lambda lines, data: (
        _ends(lines) == ["GameEnded 1-0 game over on board"]
        and _sans(lines)[-2:] == ["Nf6", "Qxf7#"]
        and all(
            _mark_at(data, "badges") <= _at(data, line) <= _mark_at(data, "badges") + 1
            for line in ("MovePlayed 6 ", "MovePlayed 7 ", "GameEnded ")
        )
        and any("3. Qh5 Nf6 4. Qxf7#" in line for line in lines if line.startswith("GameEnded"))
    ),
    "mate-and-badges": lambda lines, data: (
        [line.split()[1] for line in _ends(lines)] == ["0-1"] and "Qh4#" in _sans(lines)
    ),
    "king-steps-into-mate-badges": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "Qh5", "Ke7", "Qxe5#"]
        and [line.split()[1] for line in _ends(lines)] == ["1-0"]
    ),
    "next-game-same-board": lambda lines, data: (
        _sans(lines) == ["f3", "e5", "g4", "Qh4#", "d4"]
        and sum(line.startswith("GameStarted") for line in lines) == 2
        and [line.split()[1] for line in _ends(lines)] == ["0-1"]
        and data["bootstraps"] == 2
        and data["pgn"] == "1. d4"
    ),
    "browsed-to-start": lambda lines, data: (
        _sans(lines) == ["e4", "e5", "Nf3"]
        and sum(line.startswith("GameStarted") for line in lines) == 1
        and not _ends(lines)
        and data["bootstraps"] == 1
        and data["pgn"] == "1. e4 e5 2. Nf3"
    ),
    "new-game-after-resync": lambda lines, data: (
        [line.split(" | ")[1].split()[0] for line in lines if line.startswith("Resync position adopted")]
        == [chess.Board(LATER).board_fen()]
        and [" ".join(line.split()[:3]) for line in lines if line.startswith(("Resync", "GameStarted"))][-2:]
        == ["Resync position adopted", "GameStarted game black"]
        and data["bootstraps"] == 2
    ),
    "rethemed-before-a-move": lambda lines, data: (
        _sans(lines) == ["d4"]
        and [" ".join(line.split()[:3]) for line in lines if line.startswith("GameStarted")]
        == ["GameStarted game white", "GameStarted game white"]
        and not _ends(lines)
        and data["bootstraps"] == 2
    ),
    "unlearned-away-and-back": lambda lines, data: (
        [line for line in lines if line.startswith("BoardUnread")] == ["BoardUnread unlearned"] * 2
        and not any(line.startswith("GameStarted position") for line in lines)
    ),
    "puzzle-read-once": lambda lines, data: (
        sum(line.startswith("GameStarted position") for line in lines) == 1
        and 1.0 <= _ts_of(data, "GameStarted position") <= 2.0
        and data["reads"] == 1
    ),
}


def _ends(lines: list[str]) -> list[str]:
    """The GameEnded lines, their record left out."""
    return [line.split(" |")[0] for line in lines if line.startswith("GameEnded")]


def _at(data: dict, prefix: str) -> int:
    """The index of the first frame with an event starting with `prefix`, or -1."""
    for index, frame in enumerate(data["frames"]):
        if any(line.startswith(prefix) for line in frame["events"]):
            return index
    return -1


def _ts_of(data: dict, prefix: str) -> float:
    at = _at(data, prefix)
    return data["frames"][at]["ts"] if at >= 0 else float("nan")


def _mark_at(data: dict, mark: str) -> int:
    """The index of the first frame of the state marked `mark` (`_mark`)."""
    return next(i for i, frame in enumerate(data["frames"]) if frame.get("mark") == mark)


def _mark_ts(data: dict, mark: str) -> float:
    return data["frames"][_mark_at(data, mark)]["ts"]


def _lock_near(data: dict, cell: int, dx: float, dy: float) -> bool:
    """The lock after the settle within three pixels of where the board was moved to."""
    locks = [f["lock"] for f in data["frames"] if "lock" in f]
    x0, y0 = int(SESSION_X0 + dx * cell), int(SESSION_Y0 + dy * cell)
    return (
        len(locks) == 1
        and locks[0] is not None
        and abs(locks[0][0] - x0) <= 3
        and abs(locks[0][1] - y0) <= 3
    )


def _sans(lines: list[str]) -> list[str]:
    return [line.split()[2] for line in lines if line.startswith("MovePlayed")]


def _call(pipeline: GamePipeline, call: dict, image, ts: float) -> list:
    """What a desktop test does to a pipeline between frames, done here the same way."""
    if "reset" in call:
        pipeline.reset()
        return []
    if "reorient" in call:
        assert pipeline.board is not None, "no board locked to re-read"
        geometry = BoardGeometry(pipeline.board.region, pipeline.board.cell_px)
        told = call.get("told", False)
        return pipeline.reorient(image, geometry, Side(call["reorient"]), ts, told=told)
    if "solid_labels" in call:
        # The set's rank labels cached as solid blocks, as a start seen through a lock off
        # its board cached them (`_cached_wrong` in the desktop's tests).
        model = pipeline.model
        assert pipeline.cache is not None and model is not None, "no set learned to spoil"
        [(key, cached)] = pipeline.cache.candidates(model.light_bg, model.dark_bg)
        cached.rank_labels = {rank: np.ones((24, 24), dtype=bool) for rank in range(1, 9)}
        pipeline.cache.save(cached, key=key)
        return []
    if "end_game" in call:
        result, reason = call["end_game"]
        return pipeline.end_game(result, reason, ts)
    if "forced_bottom" in call:
        pipeline.forced_bottom = Side(call["forced_bottom"])
        return []
    if "last_clock" in call:
        white, black, running = call["last_clock"]
        pipeline._last_clock = (white, black, None if running is None else Side(running))
        pipeline._last_clock_ts = call["last_clock_ts"]
        return []
    if "reconciler" in call:
        # A reconciler's tunable, as a desktop test sets it (`resync_hold_frames`).
        assert pipeline.reconciler is not None, "no reconciler to tune"
        for name, value in call["reconciler"].items():
            if name not in RECONCILER_TUNABLES:
                raise SystemExit(f"a tunable the phone's test does not set: {name}")
            setattr(pipeline.reconciler, name, value)
        return []
    raise SystemExit(f"a call the fixtures do not know: {call}")


# The reconciler's tunables a scenario may set (`_call`), as `PipelineTest.call` sets them.
RECONCILER_TUNABLES = {"resync_hold_frames"}


class _Reads:
    """The theme cache's `candidates`, counted since the last session began: how often a
    board was read from the cache (`COUNT_READS`)."""

    count = 0


_CANDIDATES = ThemeCache.candidates


def _counted_candidates(self, *args):  # noqa: ANN001, ANN002, ANN202
    _Reads.count += 1
    return _CANDIDATES(self, *args)


ThemeCache.candidates = _counted_candidates  # type: ignore[method-assign]


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
    last_image = None
    for index, state in enumerate(states):
        call = state.pop("_call", None)
        if call is not None:
            # What a desktop test does to the pipeline between frames, at the last frame's
            # pixels (a Flip's re-read) and time, or the time the state gives.
            ts = state.pop("_ts", (frame_id - 1) / FPS if frames else 0.0)
            done = _call(pipeline, call, last_image, ts)
            frames.append({"call": call, "ts": ts, "events": [event_json(e) for e in done]})
            continue
        repeat = state.pop("_repeat", 1)
        session = state.pop("_session", False)
        asked = state.pop("_asked", False)
        # A frame drawn by the scenario itself, at the time it gives (`slides`), or from a
        # time a step apart (`_dt`); and whether it counts as changed, when the desktop's test
        # says so by hand (`_changed`: one flag, or one per frame).
        given = state.pop("_image", None)
        at = state.pop("_ts", None)
        step = state.pop("_dt", 0.0)
        forced = state.pop("_changed", None)
        where = state.pop("_where", (X0, Y0, CELL))
        lock = state.pop("_lock", False)
        # A name for the state's first frame, for the scenario's EXPECT to find it by.
        mark = state.pop("_mark", None)
        if session:
            pipeline = GamePipeline(cache=cache)
            gate = ChangeGate()
            _Reads.count = 0
        # A state held for several frames is one file, named several times: what the
        # pipeline sees is the same pixels again, which is what a still board is; and a
        # board seen before is the file it was written to then.
        image = flatten_around(
            given if given is not None else synth.synthetic_board(**state), where=where
        )
        digest = hashlib.sha1(image.tobytes()).hexdigest()
        if digest in WRITTEN:
            name_of, stored = WRITTEN[digest]
        else:
            path = folder / f"{index:03d}.png"
            cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            stored = cv2.imread(str(path), cv2.IMREAD_COLOR)
            name_of = f"{name}/{index:03d}.png"
            WRITTEN[digest] = (name_of, stored)
        last_image = stored
        for i in range(repeat):
            ts = at + i * step if at is not None else frame_id / FPS
            if forced is not None:
                # Passed by hand, as the desktop's test passes it: the gate is not asked.
                changed = forced if isinstance(forced, bool) else forced[i]
            else:
                # The frame's own size, as the phone's test takes it: a scenario may draw its own.
                gate.region = board_gate(pipeline.board, stored.shape[1], stored.shape[0])
                changed = gate.changed(stored, ts)
            events = pipeline.process(stored, ts, changed=changed, frame_id=frame_id)
            for event in events:
                assert event.ts == ts and event.source_frame_id == frame_id, event
            frame = {
                "file": name_of,
                "id": frame_id,
                "ts": ts,
                "changed": changed,
                "events": [event_json(event) for event in events],
            }
            if forced is not None:
                frame["forced"] = True
            if session and i == 0:
                frame["session"] = True
            if mark is not None and i == 0:
                frame["mark"] = mark
            if lock and i == repeat - 1:
                # Where the tracker's lock stands after the state's last frame.
                held = pipeline.board
                frame["lock"] = (
                    None
                    if held is None
                    else [held.region.x, held.region.y, held.region.w, held.region.h]
                )
            if asked and i == repeat - 1:
                # The panel's "Re-read the screen" on this frame: a cold read the player
                # asked for, from the tracker's lock (`panel._analyze`).
                assert pipeline.board is not None, "no board locked to re-read"
                geometry = BoardGeometry(pipeline.board.region, pipeline.board.cell_px)
                frame["asked"] = [
                    event_json(event) for event in pipeline.cold_start(stored, geometry, ts=ts)
                ]
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
    if name in READ_ONCE:
        data["read_once"] = True
    if name in COUNT_READS:
        data["reads"] = _Reads.count
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
        lines = [line for f in data["frames"] for line in f["events"]]
        if name in EXPECT and not EXPECT[name](lines, data):
            for line in lines:
                print(f"    {line}")
            raise SystemExit(f"{name}: the desktop does not show what its test asserts")
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
            if "asked" in f:
                print(f"    {f['ts']:.1f}s asked to re-read: {f['asked']}")
        if "themes" in data:
            print(f"    {data['themes']} learned sets in the cache")
        runs.append(data)
    (OUT / "index.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    size = sum(p.stat().st_size for p in OUT.rglob("*.png"))
    print(f"-> {OUT} ({size / 1e6:.1f} MB of frames)")


if __name__ == "__main__":
    main()
