"""Games on rendered boards: the recogniser learned from the start position, a reconciler set
at a position, a game fed frame by frame to a reconciler or to the whole pipeline, and the
seeded random games the sweeps play.

The phone's fixture maker (tools/make_reconcile_fixtures.py in the Android repo) calls the
desktop's reconciler tests directly and swaps `Reconciler` here for one that records what is
done to it, so these are plain functions and classes, not fixtures.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Iterable
from dataclasses import dataclass

import chess
import numpy as np

from sidecoach.events import Event, GameEnded, MovePlayed, Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.state.reconcile import Reconciler
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import row_col, square_at
from tests.synth import ARROWS, GEOM, HIGHLIGHTS, PIECE_STYLES, THEMES, synthetic_board

Queue = tuple[chess.Move, ...]


@functools.cache
def recogniser(bottom: Side) -> ThemeModel:
    """The recogniser learned from the start position with `bottom` at the bottom, built once
    for each side and shared: it reads pixels and nothing else (its cell cache compares crops
    pixel for pixel), so a shared one reads every frame as a fresh one would."""
    model = ThemeModel.from_start_position(synthetic_board(bottom=bottom), GEOM, bottom)
    assert model is not None
    return model


def sans_of(events: Iterable[Event]) -> list[str]:
    """The moves among `events`, in SAN."""
    return [e.san for e in events if isinstance(e, MovePlayed)]


def reconciler_at(
    fen: str, bottom: Side = Side.WHITE
) -> tuple[ThemeModel, Reconciler, chess.Board]:
    """A reconciler started on the start position and then set at `fen`, the way a cold start
    adopts a position, with two frames of it read; the recogniser and the board with it."""
    model = recogniser(bottom)
    rec = Reconciler(bottom=bottom, user_side=bottom, settle_frames=2)
    rec.update(model.classify_frame(synthetic_board(bottom=bottom), GEOM), 0.0)
    board = chess.Board(fen)
    rec.board = board.copy()
    rec._base_fen = fen
    for i in range(2):
        frame = synthetic_board(board=board, bottom=bottom)
        rec.update(model.classify_frame(frame, GEOM), 0.1 + 0.1 * i)
    return model, rec, board


class Game:
    """Rendered frames of a game fed to a reconciler through the shared recogniser, or with
    `pipeline` to the whole pipeline (the result badges and the game-over dialog are the
    pipeline's), a tenth of a second apart. `board` is the truth; `events` what came out,
    `sans` the moves committed and `ended` the ends."""

    def __init__(
        self, bottom: Side = Side.WHITE, fen: str | None = None, pipeline: bool = False
    ) -> None:
        self.bottom = bottom
        self.pipe = GamePipeline() if pipeline else None
        self._rec = (
            None if pipeline else Reconciler(bottom=bottom, user_side=bottom, settle_frames=2)
        )
        self.board = chess.Board()
        self.ts = 0.0
        self.events: list[Event] = []
        self.show(frames=2)
        assert self.rec.started
        if fen is not None:
            # A mid-game position: play the pieces there through a resync-free path is
            # not always possible, so the reconciler adopts it like a cold start would.
            self.board = chess.Board(fen)
            rec = self.rec
            rec.board = self.board.copy()
            rec._base_fen = fen
            self.show(frames=2)

    @property
    def rec(self) -> Reconciler:
        rec = self.pipe.reconciler if self.pipe is not None else self._rec
        assert rec is not None
        return rec

    @property
    def sans(self) -> list[str]:
        return sans_of(self.events)

    @property
    def ended(self) -> list[GameEnded]:
        return [e for e in self.events if isinstance(e, GameEnded)]

    def frame(self, **kw: object) -> np.ndarray:
        return synthetic_board(board=self.board, bottom=self.bottom, **kw)  # type: ignore[arg-type]

    def process(self, frame: np.ndarray) -> list[Event]:
        """One frame read; what it told."""
        events: list[Event]
        if self.pipe is not None:
            events = self.pipe.process(frame, self.ts)
        else:
            assert self._rec is not None
            readings = recogniser(self.bottom).classify_frame(frame, GEOM)
            events = self._rec.update(readings, self.ts)
        self.events += events
        self.ts += 0.1
        return events

    def show(self, frames: int = 2, **kw: object) -> list[Event]:
        """The board as it stands for `frames` frames; what they told."""
        out: list[Event] = []
        for _ in range(frames):
            out += self.process(self.frame(**kw))
        return out

    def play(self, *sans: str, frames: int = 2, **kw: object) -> chess.Move:
        """Moves made on the board, each followed by its frames with the last-move
        highlight; the last move is returned."""
        move: chess.Move | None = None
        for san in sans:
            move = self.board.push_san(san)
            self.show(frames=frames, highlight=move, **kw)
        assert move is not None, "no move to play"
        return move

    def queue(self, *queues: Iterable[str]) -> tuple[Queue, ...]:
        """Premove queues from uci strings, checked to be the user's pieces."""
        out: list[Queue] = []
        for spec in queues:
            moves = tuple(chess.Move.from_uci(uci) for uci in spec)
            piece = self.board.piece_at(moves[0].from_square)
            assert piece is not None and piece.color == (self.bottom == Side.WHITE)
            out.append(moves)
        return tuple(out)


# ---- the seeded random games (test_stress, test_board_geometry) ----------------------------

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
    """The random game of `seed`: its moves, how each is shown, and how the board looks."""
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


def drag_path(
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
