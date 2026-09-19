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
from dataclasses import dataclass, replace
from pathlib import Path

import chess
import numpy as np
import pytest

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.squares import BoardGeometry, row_col, square_at
from sidecoach.vision.theme_cache import ThemeCache
from tests.synth import ARROWS, HIGHLIGHTS, PIECE_STYLES, THEMES, synthetic_board

pytestmark = pytest.mark.vision

SIZES = (44, 56, 64, 80, 96, 112, 128, 144)
ARROW_WIDTHS = (0.12, 0.2, 0.3)  # thin lichess-like, chess.com's, a fat one
PLIES = 16


@dataclass(frozen=True)
class Scenario:
    seed: int
    theme: str
    cell: int
    bottom: Side
    sans: tuple[str, ...]
    # Per ply: "plain" | "drag" | "arrow" | "badge" | "premove" | "select" (the user clicks
    # a piece before moving it: its square highlighted, its legal moves hinted) | "animate"
    # (the site slides the piece to its square over a few frames).
    decorations: tuple[str, ...]
    style: str = "classic"
    highlight_colour: str = "yellow"
    arrow_colour: str = "orange"
    arrow_width: float = 0.2
    coordinates: bool = True

    @property
    def where(self) -> str:
        return (
            f"seed {self.seed} ({self.theme}, {self.style} pieces, {self.cell}px, "
            f"{self.bottom.value} at the bottom, {self.highlight_colour} highlight, "
            f"{self.arrow_colour} arrow x{self.arrow_width}, "
            f"coordinates {'on' if self.coordinates else 'off'})"
        )


def scenario(seed: int) -> Scenario:
    rng = random.Random(seed)
    board = chess.Board()
    sans: list[str] = []
    for _ in range(PLIES):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        # Prefer captures and developing moves a little, so the games look like games.
        captures = [m for m in moves if board.is_capture(m)]
        move = rng.choice(captures) if captures and rng.random() < 0.3 else rng.choice(moves)
        sans.append(board.san(move))
        board.push(move)
    decorations = tuple(
        rng.choices(
            ["plain", "drag", "arrow", "badge", "premove", "select", "animate"],
            [4, 2, 1, 2, 1, 1, 2],
        )[0]
        for _ in sans
    )
    return Scenario(
        seed,
        rng.choice(list(THEMES)),
        rng.choice(SIZES),
        rng.choice([Side.WHITE, Side.BLACK]),
        tuple(sans),
        decorations,
        # Half the games on the real set, half on the polygons and their variants.
        rng.choice([s for s in PIECE_STYLES if PIECE_STYLES[s].real])
        if rng.random() < 0.5
        else rng.choice([s for s in PIECE_STYLES if not PIECE_STYLES[s].real]),
        rng.choice(list(HIGHLIGHTS)),
        rng.choice(list(ARROWS)),
        rng.choice(ARROW_WIDTHS),
        rng.random() < 0.8,
    )


def _drag_path(
    move: chess.Move, bottom: Side, rng: random.Random
) -> list[tuple[chess.Square, chess.Square, float, float]]:
    """The frames of a drag from the move's origin to its destination, as (origin, square
    under the cursor, dx, dy) for the renderer: one to five frames along the straight
    line, a touch of wobble, the piece off the cursor's square centre by up to a third
    of a cell -- a flick or a slow deliberate drag."""
    r0, c0 = row_col(move.from_square, bottom)
    r1, c1 = row_col(move.to_square, bottom)
    frames = rng.choice([1, 2, 3, 5])
    out = []
    for i in range(1, frames + 1):
        t = i / (frames + 1)
        row = r0 + (r1 - r0) * t + rng.uniform(-0.2, 0.2)
        col = c0 + (c1 - c0) * t + rng.uniform(-0.2, 0.2)
        over_r, over_c = min(7, max(0, round(row))), min(7, max(0, round(col)))
        over = square_at(over_r, over_c, bottom)
        out.append((move.from_square, over, col - over_c, row - over_r))
    return out


@dataclass(frozen=True)
class Played:
    got: list[str]  # the moves the pipeline committed
    played: list[str]
    same_board: bool  # the pipeline's board is the game's at the end
    cold: str | None  # the placement a second pipeline read cold from the cache, or None


def play(sc: Scenario, cache_dir: Path | None = None) -> Played:
    rng = random.Random(sc.seed)
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    got: list[str] = []
    ended = 0

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
        nonlocal ts, ended
        for _ in range(frames):
            for event in pipe.process(frame_for(**kw), ts):
                if isinstance(event, MovePlayed):
                    got.append(event.san)
                elif isinstance(event, GameEnded):
                    ended += 1
            ts += 0.1

    show(3)
    assert pipe.reconciler is not None and pipe.reconciler.started
    user_colour = chess.WHITE if sc.bottom == Side.WHITE else chess.BLACK
    for san, decoration in zip(sc.sans, sc.decorations, strict=True):
        move = board.parse_san(san)
        mover = board.turn
        if decoration == "drag" and mover == user_colour:
            # The piece lifted and carried to its square under the outline, fast or slow.
            for step in _drag_path(move, sc.bottom, rng):
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
    return Played(got, list(sc.sans), same_board, cold)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "seed" in metafunc.fixturenames:
        spec = str(metafunc.config.getoption("--seeds") or "")
        if not spec:
            seeds = [1, 2, 3, 4, 5, 6, 7, 8]
        elif "-" in spec:  # "251-1000": a range of seeds
            lo, hi = (int(x) for x in spec.split("-", 1))
            seeds = list(range(lo, hi + 1))
        else:
            seeds = list(range(1, int(spec) + 1))
        metafunc.parametrize("seed", seeds)


def test_a_random_game_is_followed_to_its_position(seed: int, tmp_path: Path) -> None:
    """Every move committed, nothing invented, the board the game's at the end, and the
    final position read cold from the cached theme. Two moves that commute (b4 and g4
    across an opponent's reply) may commit in either order when a drag or an arrow held
    the frames in between: the pixels do not say which came first, and highlight colours
    are, by design, never consulted."""
    sc = scenario(seed)
    result = play(sc, tmp_path)
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


def test_a_random_game_starts_once(seed: int) -> None:
    sc = scenario(seed)
    pipe = GamePipeline()
    starts = 0
    for i in range(4):
        frame = synthetic_board(
            width=8 * sc.cell + 600,
            height=8 * sc.cell + 240,
            theme=sc.theme,
            cell=sc.cell,
            bottom=sc.bottom,
            coordinates=sc.coordinates,
            style=sc.style,
        )
        starts += sum(1 for e in pipe.process(frame, 0.1 * i) if isinstance(e, GameStarted))
    assert starts == 1 and pipe.bottom == sc.bottom, sc.where


def test_a_theme_that_hides_the_black_outline_is_still_followed() -> None:
    """Light squares within tolerance of the black pieces' outline (synth.LOW_CONTRAST_GREY)
    hide that outline on light squares, so a black piece had two silhouettes and the
    exemplars matched one. Matching on the piece's own tone lifted that: this was the
    sweep's first find, and the case is kept as its regression."""
    from tests import synth

    kept = synth.THEMES["grey"]
    synth.THEMES["grey"] = synth.LOW_CONTRAST_GREY
    try:
        sc = replace(scenario(5), theme="grey", style="classic", cell=64)
        result = play(sc)
    finally:
        synth.THEMES["grey"] = kept
    assert result.same_board and sorted(result.got) == sorted(result.played), (
        f"committed {result.got}, played {result.played}"
    )
