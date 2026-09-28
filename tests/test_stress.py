"""Random games through the whole vision pipeline, rendered every way a site draws them.

The renderer is used against the pipeline: a seeded generator picks a theme (bright, dark,
textured), a piece style (the real cburnett set in three colourings, polygons of several
outline widths and sizes), a board size, the highlight and arrow colours and the arrow's
width, whether coordinates are drawn, an orientation and a game, then shows every move
with what a session throws at the recogniser -- the last-move highlight, a drag on the way
to the move at any speed, the site's own animation of a move at any speed, an arrow the
player drew, the classification badge, a premove queue, a selected piece with its hints
-- and the pipeline must commit exactly the moves played, nothing else, in order. At the
end the learned theme goes through the cache and a second pipeline must recognise the
final position cold. A handful of seeds run in the suite; `--seeds` runs a sweep to hunt
for the next weakness:

    uv run pytest tests/test_stress.py -q -n auto --seeds 250

Found so far and fixed: a theme whose light squares hide the black pieces' outline; a
badge's ink swinging a piece's colour; a badged piece never settling on a small cell; a
black set's grey outline lost on a dark board (the tone is now what lies nearer the core
than the square); a bold set's outline wider than the erosion the core was read with (the
core is now the deep inside of the silhouette); an arrow standing across four plies holding
every one of them back (a drawing is no longer a hover) and a four-ply gap the missed-move
search's budget could not reach (it counts expansions now). With the real set
(tests/pieceset.py) in half the games: a fill colour read off the deepest band of a
queen's body, which is where the set draws its lines (the fill is the depth-weighted
mode now); a rank of piece bases pulling the board lock at 144-pixel cells (piece pixels
are masked out of the boundary fit); a cream knight on a cream square that was only its
outline (the outline is closed and what it encloses filled). With moves animated and drags
at any speed, and the cache round-trip: a bishop sliding across a square read as unknown
committed as a move to that square (a move's destination must be read); a lone label in
a corner kept an empty square from reading empty (a part of a silhouette that does not
reach the middle is dropped, however alone); a 44-pixel digit on a dark board was all
blends (the label ink threshold is 30); and a queen's colour, read off the deepest band
of its body, then off the whole silhouette's weighted mode, both wrong somewhere, is now
the tone that forms the widest region of its own, classes by the set's learned fills.
With thirty-five of lichess's sets (tests/fixtures/pieces): a set's detail in colours
its two sides' fills and outlines do not name (a pink emblem on a navy piece) read as a
drawing (the set's palette is learned by clustering its ink); a mid-tone set's piece
bases pulled the boundary fit (each cell is masked against its own colour); a queen's
fill flipping colour by square (the fill is the tone forming the widest region of its
own; a piece's side is read from its brightness against each side's exemplars, its shape
among that side); the digit reader took a rook's colour for the square's, let a rook's
corner into the digit's window, and gave up on the first digit it could not read. A set
whose bishop is drawn apart from its base lost the base (a part of a silhouette is
dropped only when it lies wholly in a corner); a piece with little contrast against its
square is read again at a tolerance to match; the boundary fit runs again from where it
landed, keeps a boundary's blends on the grid lines only, and weighs a re-fit against
the geometry it was handed. Open at 1000 seeds: about one game in seventy, mostly dark
shaded sets on the dark board (riohacha, pirouetti: a piece's shaded half is the
square's colour) and 44-pixel cells on which a set's piece does not land where its
exemplar did -- the two tolerances explained at the top of sidecoach/vision/pieces.py,
left where they are by decision (2026-09-19).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import pytest

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Region, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry, square_at
from sidecoach.vision.theme_cache import ThemeCache
from tests.games import Scenario, drag_path, scenario
from tests.synth import (
    LOW_CONTRAST_GREY,
    synthetic_board,
)

pytestmark = pytest.mark.vision


@dataclass(frozen=True)
class Played:
    got: list[str]  # the moves the pipeline committed
    played: list[str]
    same_board: bool  # the pipeline's board is the game's at the end
    cold: str | None  # the placement a second pipeline read cold from the cache, or None
    starts: int  # the games the pipeline began
    bottom: Side | None  # the side it read at the bottom from the start position


def play(sc: Scenario, cache_dir: Path | None = None) -> Played:
    rng = random.Random(sc.seed)
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    got: list[str] = []
    ended = 0
    starts = 0

    def frame_for(**kw: object) -> np.ndarray:
        return synthetic_board(
            width=8 * sc.cell + 600,
            height=8 * sc.cell + 240,
            board=board,
            theme=sc.theme,
            cell=sc.cell,
            bottom=sc.bottom,
            coordinates=sc.coordinates,
            style=sc.style,
            highlight_colour=sc.highlight_colour,
            arrow_colour=sc.arrow_colour,
            arrow_width=sc.arrow_width,
            **kw,  # type: ignore[arg-type]
        )

    def show(frames: int, **kw: object) -> None:
        nonlocal ts, ended, starts
        for _ in range(frames):
            for event in pipe.process(frame_for(**kw), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
                elif isinstance(event, GameEnded):
                    ended += 1
                elif isinstance(event, GameStarted):
                    starts += 1
            ts += 0.1

    show(3)
    assert pipe.reconciler is not None and pipe.reconciler.started
    bottom = pipe.bottom
    user_colour = chess.WHITE if sc.bottom == Side.WHITE else chess.BLACK
    for san, decoration in zip(sc.sans, sc.decorations, strict=True):
        move = board.parse_san(san)
        mover = board.turn
        if decoration == "drag" and mover == user_colour:
            # The piece lifted and carried to its square under the outline, fast or slow.
            for step in drag_path(move, sc.bottom, rng):
                show(1, dragging=step)
        if decoration == "animate":
            # The site slides the piece to its square, a fast bot reply or a slow setting,
            # with the move's highlight already on both squares.
            frames = rng.choice([1, 2, 3, 5])
            for i in range(1, frames + 1):
                show(
                    1,
                    highlight=move,
                    animating=(move.from_square, move.to_square, i / (frames + 1)),
                )
        if decoration == "select" and mover == user_colour:
            # The piece clicked: its square lit and its legal moves hinted, for a moment.
            show(2, selected=move.from_square)
        if decoration == "premove" and mover != user_colour:
            # While the opponent thinks, the user queues a legal reply of theirs.
            after = board.copy()
            after.push(move)
            replies = [
                m
                for m in after.legal_moves
                if after.piece_at(m.from_square) and not after.is_capture(m)
            ]
            if replies:
                queued = rng.choice(replies)
                show(2, premoves=((queued,),))
        board.push(move)
        extra: dict[str, object] = {"highlight": move, "check": True}
        if decoration == "arrow":
            targets = [m for m in board.legal_moves if not board.is_capture(m)]
            if targets:
                extra["arrows"] = (rng.choice(targets),)
        if decoration == "badge":
            extra["badge"] = (move.to_square, rng.choice(["best", "good", "blunder", "brilliant"]))
        show(3, **extra)
    show(5)  # quiet frames at the end, so a pending story can settle
    assert pipe.reconciler is not None
    same_board = pipe.reconciler.board.board_fen() == board.board_fen()
    cold: str | None = None
    if cache_dir is not None and pipe.model is not None and pipe.board is not None:
        # The theme learned in this game through the cache: a second pipeline, told
        # nothing, must read the final position cold from the same frame.
        cache = ThemeCache(cache_dir)
        assert cache.save(pipe.model) is not None
        other = GamePipeline(cache=cache)
        geometry = BoardGeometry(pipe.board.region, pipe.board.cell_px)
        events = other.cold_start(
            frame_for(), geometry, bottom=None if sc.coordinates else sc.bottom
        )
        if any(isinstance(e, GameStarted) for e in events) and other.reconciler is not None:
            cold = other.reconciler.board.board_fen()
            if other.bottom != sc.bottom:
                cold = f"{cold} (read {other.bottom} at the bottom)"
    return Played(got, list(sc.sans), same_board, cold, starts, bottom)


# The seeds a default run plays (tests/conftest.py; `--seeds` sweeps more).
SEEDS = range(1, 9)


def test_a_random_game_is_followed_to_its_position(seed: int, tmp_path: Path) -> None:
    """The game begun once, with the side at the bottom read from its start position
    (test_a_random_game_starts_once); every move committed, nothing invented, the board
    the game's at the end, and the final position read cold from the cached theme. Two
    moves that commute (b4 and g4 across an opponent's reply) may commit in either order
    when a drag or an arrow held the frames in between: the pixels do not say which came
    first, and highlight colours are, by design, never consulted."""
    sc = scenario(seed)
    result = play(sc, tmp_path)
    assert result.starts == 1 and result.bottom == sc.bottom, (
        f"{sc.where}: {result.starts} games begun, {result.bottom} read at the bottom"
    )
    assert result.same_board and sorted(result.got) == sorted(result.played), (
        f"{sc.where}: committed {result.got}, played {result.played}; decorations {sc.decorations}"
    )
    final = chess.Board()
    for san in sc.sans:
        final.push_san(san)
    assert result.cold == final.board_fen(), (
        f"{sc.where}: read cold from the cache as {result.cold}, the position is "
        f"{final.board_fen()}"
    )


def test_a_theme_that_hides_the_black_outline_is_still_read() -> None:
    """Light squares within tolerance of the black pieces' outline (synth.LOW_CONTRAST_GREY)
    hide that outline on light squares, so a black piece had two silhouettes and the
    exemplars matched one, and a game on that theme lost its moves. Matching on the piece's
    own tone lifted that: this was the sweep's first find, and the case is kept as its
    regression -- every square of a start position and a middlegame read, either way up,
    where a whole game once was played through."""
    light, dark = LOW_CONTRAST_GREY
    geometry = BoardGeometry(Region(300, 120, 512, 512), 64.0)
    later = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "d5", "exd5", "Nxd5"):
        later.push_san(san)
    for bottom in (Side.WHITE, Side.BLACK):
        start = synthetic_board(light=light, dark=dark, bottom=bottom)
        model = ThemeModel.from_start_position(start, geometry, bottom)
        assert model is not None, f"no recogniser, {bottom.value} at the bottom"
        for board in (chess.Board(), later):
            frame = synthetic_board(board=board, light=light, dark=dark, bottom=bottom)
            readings = model.classify_frame(frame, geometry)
            wrong = []
            for r in range(8):
                for c in range(8):
                    piece = board.piece_at(square_at(r, c, bottom))
                    if readings[r][c].symbol != (piece.symbol() if piece else None):
                        wrong.append(chess.square_name(square_at(r, c, bottom)))
            assert not wrong, f"{bottom.value} at the bottom: {wrong} misread"


# ---- how games end on screen ------------------------------------------------------------

# A random game that ends in mate within this many plies; the seed's draws are retried
# until one does, so every seed has an ending.
ENDING_PLIES = 30


@dataclass(frozen=True)
class Ending:
    """The way a site shows the end: whether the mate lands in the same frame as the
    opponent's premoved move before it (a queued king step, then the mate); whether the
    site stamps result badges on the kings; whether it shows labels first, fading in;
    whether a dialog then covers the board, fading in."""

    together: bool
    badges: bool
    labels: bool
    dialog: bool


def ending_scenario(seed: int) -> tuple[Scenario, Ending]:
    for attempt in range(60):
        rng = random.Random(seed * 1000 + attempt)
        board = chess.Board()
        sans: list[str] = []
        mated = False
        for ply in range(ENDING_PLIES):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            if ply >= 6:
                mate = next((m for m in moves if _mates(board, m)), None)
                if mate is not None:
                    sans.append(board.san(mate))
                    board.push(mate)
                    mated = True
                    break
            captures = [m for m in moves if board.is_capture(m)]
            move = rng.choice(captures) if captures and rng.random() < 0.3 else rng.choice(moves)
            sans.append(board.san(move))
            board.push(move)
        if mated:
            break
    else:  # pragma: no cover - sixty draws without a mate
        raise AssertionError(f"seed {seed}: no random game mates within {ENDING_PLIES} plies")
    base = scenario(seed)
    sc = Scenario(
        seed,
        base.theme,
        base.cell,
        base.bottom,
        tuple(sans),
        tuple("plain" for _ in sans),
        base.style,
        base.highlight_colour,
        base.arrow_colour,
        base.arrow_width,
        base.coordinates,
    )
    ending = Ending(
        together=rng.random() < 0.35,
        badges=rng.random() < 0.7,
        labels=rng.random() < 0.6,
        dialog=rng.random() < 0.8,
    )
    return sc, ending


def _mates(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


@dataclass(frozen=True)
class Ended:
    got: list[str]
    result: str | None  # the GameEnded's result, or None
    # The phase the game ended in: "labels" (the mate on screen with the site's labels
    # over it), "clean" (the labels gone, badges at most), "dialog" (the dialog fading
    # in or up), "after" (the dialog closed, the board clean again), or "never".
    when: str


def play_ending(sc: Scenario, ending: Ending) -> Ended:
    """The game to its last move, then the end as the site shows it."""
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    got: list[str] = []
    results: list[str] = []
    phase = "clean"
    when: str | None = None

    def frame_for(**kw: object) -> np.ndarray:
        return synthetic_board(
            width=8 * sc.cell + 600,
            height=8 * sc.cell + 240,
            board=board,
            theme=sc.theme,
            cell=sc.cell,
            bottom=sc.bottom,
            coordinates=sc.coordinates,
            style=sc.style,
            highlight_colour=sc.highlight_colour,
            arrow_colour=sc.arrow_colour,
            arrow_width=sc.arrow_width,
            **kw,  # type: ignore[arg-type]
        )

    def show(frames: int, **kw: object) -> None:
        nonlocal ts, when
        for _ in range(frames):
            for event in pipe.process(frame_for(**kw), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
                elif isinstance(event, GameEnded):
                    results.append(event.result)
                    if when is None:
                        when = phase
            ts += 0.1

    show(3)
    assert pipe.reconciler is not None and pipe.reconciler.started
    sans = list(sc.sans)
    # The game up to the position before the ending: two frames a move.
    last = 2 if ending.together and len(sans) >= 2 else 1
    for san in sans[:-last]:
        move = board.push_san(san)
        show(2, highlight=move)
    show(3)
    if ending.together:
        # The user queues the mate while the opponent thinks, then both land at once.
        mate = board.copy()
        mate.push_san(sans[-2])
        queued = mate.parse_san(sans[-1])
        show(2, premoves=((queued,),))
        board.push_san(sans[-2])
    final = board.push_san(sans[-1])
    result = board.result()
    decorated: dict[str, object] = {
        "highlight": final,
        "check": True,
        "result": result,
        "badges": ending.badges,
    }
    # The end as a site shows it: the labels over the kings for about a second (they
    # fade in over the first frame), then gone; the dialog fading in and staying a
    # while; the dialog closed and the board clean again, the badges at most.
    if ending.labels:
        phase = "labels"
        show(1, **{**decorated, "labels": 0.4})
        show(7, **{**decorated, "labels": 1.0})
    phase = "clean"
    show(6, **decorated)
    if ending.dialog:
        phase = "dialog"
        for alpha in (0.35, 0.7, 1.0, 1.0, 1.0, 1.0):
            show(1, **{**decorated, "dialog": alpha})
        phase = "after"
        show(6, **decorated)
    return Ended(got, results[0] if results else None, when or "never")


def test_a_random_game_ends_on_screen(seed: int) -> None:
    """The mate is committed and the game ends, from a clean view of the board, however
    the site decorates the end: the mated king's glow, badges, labels over the kings for
    a second, the mate landing with the premoved move before it, the dialog fading in
    over the board and closing again. The labels and the dialog are passing; the clean
    frames before and after them are where the mate is read."""
    sc, ending = ending_scenario(seed)
    ended = play_ending(sc, ending)
    board = chess.Board()
    for san in sc.sans:
        board.push_san(san)
    where = f"{sc.where}, ending {ending}"
    assert ended.result == board.result(), f"{where}: ended {ended.result}, committed {ended.got}"
    assert sc.sans[-1] in ended.got, f"{where}: the mating move was not committed ({ended.got})"
    assert ended.when in ("labels", "clean", "dialog", "after"), f"{where}: ended {ended.when}"
