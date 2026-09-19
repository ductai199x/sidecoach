"""Random games through the whole vision pipeline, rendered every way a site draws them.

The renderer is used against the pipeline: a seeded generator picks a theme (bright, dark,
textured), a piece style (outline width, size, a coloured set), a board size, the highlight
and arrow colours and the arrow's width, whether coordinates are drawn, an orientation and
a game, then shows every move with what a session throws at the recogniser -- the last-move
highlight, a drag on the way to the move, an arrow the player drew, the classification
badge, a premove queue -- and the pipeline must commit exactly the moves played, nothing
else, in order. A handful of seeds run in the suite; `--seeds` runs a sweep to hunt for
the next weakness:

    uv run pytest tests/test_stress.py -q -n auto --seeds 250

Found so far and fixed: a theme whose light squares hide the black pieces' outline; a
badge's ink swinging a piece's colour; a badged piece never settling on a small cell; a
black set's grey outline lost on a dark board (the tone is now what lies nearer the core
than the square); a bold set's outline wider than the erosion the core was read with (the
core is now the deep inside of the silhouette); an arrow standing across four plies holding
every one of them back (a drawing is no longer a hover) and a four-ply gap the missed-move
search's budget could not reach (it counts expansions now). 250 seeds pass.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import chess
import pytest

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from tests.synth import ARROWS, HIGHLIGHTS, PIECE_STYLES, THEMES, synthetic_board

pytestmark = pytest.mark.vision

SIZES = (44, 56, 64, 80, 96, 112, 128)
ARROW_WIDTHS = (0.12, 0.2, 0.3)  # thin lichess-like, chess.com's, a fat one
PLIES = 16


@dataclass(frozen=True)
class Scenario:
    seed: int
    theme: str
    cell: int
    bottom: Side
    sans: tuple[str, ...]
    decorations: tuple[str, ...]  # per ply: "plain" | "drag" | "arrow" | "badge" | "premove"
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
        rng.choices(["plain", "drag", "arrow", "badge", "premove"], [4, 2, 1, 2, 1])[0]
        for _ in sans
    )
    return Scenario(
        seed,
        rng.choice(list(THEMES)),
        rng.choice(SIZES),
        rng.choice([Side.WHITE, Side.BLACK]),
        tuple(sans),
        decorations,
        rng.choice(list(PIECE_STYLES)),
        rng.choice(list(HIGHLIGHTS)),
        rng.choice(list(ARROWS)),
        rng.choice(ARROW_WIDTHS),
        rng.random() < 0.8,
    )


def play(sc: Scenario) -> tuple[list[str], list[str], bool]:
    """The moves the pipeline committed, the ones played, and whether the pipeline's board
    is the game's at the end."""
    rng = random.Random(sc.seed)
    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    got: list[str] = []
    ended = 0

    def show(frames: int, **kw: object) -> None:
        nonlocal ts, ended
        for _ in range(frames):
            frame = synthetic_board(
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
            for event in pipe.process(frame, ts):
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
            # The piece lifted and carried over two squares on its way, under the outline.
            for over in (move.from_square, move.to_square):
                show(1, dragging=(move.from_square, over, 0.15, 0.1))
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
        extra: dict[str, object] = {"highlight": move}
        if decoration == "arrow":
            targets = [m for m in board.legal_moves if not board.is_capture(m)]
            if targets:
                extra["arrows"] = (rng.choice(targets),)
        if decoration == "badge":
            extra["badge"] = (move.to_square, rng.choice(["best", "good", "blunder", "brilliant"]))
        show(3, **extra)
    show(5)  # quiet frames at the end, so a pending story can settle
    assert pipe.reconciler is not None
    return got, list(sc.sans), pipe.reconciler.board.board_fen() == board.board_fen()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "seed" in metafunc.fixturenames:
        n = int(metafunc.config.getoption("--seeds") or 0)
        metafunc.parametrize("seed", list(range(1, n + 1)) if n else [1, 2, 3, 4, 5, 6, 7, 8])


def test_a_random_game_is_followed_to_its_position(seed: int) -> None:
    """Every move committed, nothing invented, the board the game's at the end. Two moves
    that commute (b4 and g4 across an opponent's reply) may commit in either order when a
    drag or an arrow held the frames in between: the pixels do not say which came first,
    and highlight colours are, by design, never consulted."""
    sc = scenario(seed)
    got, played, same_board = play(sc)
    assert same_board and sorted(got) == sorted(played), (
        f"{sc.where}: committed {got}, played {played}; decorations {sc.decorations}"
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
        got, played, same = play(sc)
    finally:
        synth.THEMES["grey"] = kept
    assert same and sorted(got) == sorted(played), f"committed {got}, played {played}"
