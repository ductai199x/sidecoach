"""Typed events that connect the pipeline stages.

Every stage consumes and emits these models and nothing else, so any stage can be replayed
from a recording. Pixels never travel inside an event: a Frame carries the frame id and
geometry, and the pixel buffer is looked up from the capture ring by id.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from time import monotonic
from typing import NamedTuple

from pydantic import BaseModel, Field


class Region(NamedTuple):
    """Axis-aligned rectangle in screen pixels."""

    x: int
    y: int
    w: int
    h: int


class Event(BaseModel):
    """Base event. `source_frame_id` is the frame an event ultimately derives from."""

    ts: float = Field(default_factory=monotonic)
    source_frame_id: int | None = None


# Capture and vision


class Frame(Event):
    frame_id: int
    monitor: int
    region: Region
    changed: bool = True


class Side(StrEnum):
    WHITE = "white"
    BLACK = "black"


class BoardLocated(Event):
    monitor: int
    region: Region
    cell_px: float
    fit_score: float


class BoardLost(Event):
    reason: str


class SquareObservation(Event):
    """Per-frame board reading. `placement` is the piece-placement field of a FEN."""

    placement: str
    bottom: Side
    min_confidence: float


class ClockObservation(Event):
    white_ms: int | None = None
    black_ms: int | None = None
    running: Side | None = None


class ScreenState(StrEnum):
    LIVE_GAME = "live_game"
    ANALYSIS = "analysis"
    PUZZLES = "puzzles"
    LESSON = "lesson"
    LOBBY = "lobby"
    GAME_OVER = "game_over"
    # A dialog or a large overlay over the board of a live game: the game may be over (a
    # resignation, a flag, the site's game-over dialog) or merely interrupted; the board
    # cannot say which, so the game stays open and the coach stays quiet.
    COVERED = "covered"
    UNKNOWN = "unknown"


class ScreenStateChanged(Event):
    state: ScreenState


# State


class GameStarted(Event):
    game_id: str
    user_side: Side
    start_fen: str


class PositionChanged(Event):
    game_id: str
    fen: str
    ply: int


class MovePlayed(Event):
    game_id: str
    ply: int
    san: str
    uci: str
    fen_before: str
    fen_after: str
    by_user: bool
    think_ms: int | None = None


class Takeback(Event):
    game_id: str
    plies: int
    fen: str


class Resync(Event):
    game_id: str
    fen: str
    reason: str


class GameEnded(Event):
    game_id: str
    result: str
    reason: str | None = None
    pgn: str | None = None
    final_fen: str | None = None


# Analysis


class Line(BaseModel):
    """One engine line. `expected_score` is (W + 0.5 D) / 1000 from the mover's view."""

    moves: list[str]
    depth: int
    expected_score: float
    score_cp: int | None = None
    mate: int | None = None
    wdl: tuple[int, int, int] | None = None


class MotifTag(BaseModel):
    name: str
    squares: list[str] = Field(default_factory=list)
    move: str | None = None


class Analysis(Event):
    game_id: str
    fen: str
    ply: int
    lines: list[Line]
    decisive: bool
    only_move: bool
    threats: list[str] = Field(default_factory=list)
    tags: list[MotifTag] = Field(default_factory=list)
    opening: str | None = None


class Severity(StrEnum):
    NONE = "none"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    BLUNDER = "blunder"


class Label(StrEnum):
    """What a played move was, in the vocabulary the user already reads on chess.com.

    Severity says how much a move cost; the label says what kind of move it was, which
    needs the move before it and the opening book as well: a miss is a chance the opponent
    handed over and the mover did not take, a great move is one that took it or was the
    only move, a brilliant move is a sound sacrifice, a book move is still in theory.
    """

    BOOK = "book"
    BRILLIANT = "brilliant"
    GREAT = "great"
    BEST = "best"
    EXCELLENT = "excellent"
    GOOD = "good"
    INACCURACY = "inaccuracy"
    MISTAKE = "mistake"
    MISS = "miss"
    BLUNDER = "blunder"


class MoveAssessment(Event):
    game_id: str
    ply: int
    uci: str
    expected_loss: float
    severity: Severity
    best: list[str]
    tags: list[MotifTag] = Field(default_factory=list)
    # The shallower of the two searches the loss was measured from. A position that hit the
    # engine's time cap before reaching the depth asked for makes the comparison less
    # exact, and a reader deciding whether to act on a small loss should know.
    depth: int = 0
    label: Label | None = None
    # Per-move accuracy on Lichess's curve, 0 to 100, so a game can be summed up.
    accuracy: float | None = None


# Coach and UI


class DisclosureLevel(IntEnum):
    SILENT = 0
    QUESTION = 1
    THEME = 2
    REGION = 3
    PIECE = 4
    CANDIDATE = 5
    LINE = 6


class Trigger(StrEnum):
    USER_ASK = "user_ask"
    POST_MISTAKE = "post_mistake"
    POST_MOVE = "post_move"
    DECISIVE_QUESTION = "decisive_question"
    STUCK = "stuck"
    REVIEW = "review"


class UserUtterance(Event):
    text: str
    source: str = "text"


class Annotation(BaseModel):
    """A mark on the board that goes with a coach message: what the words are about.

    `kind` is "square" (a highlight), "ring" (around a piece) or "arrow" (from, to);
    `squares` holds one name, or two for an arrow. `color` is a role, not a hue --
    good / warn / bad / threat / focus / mention / region / line -- so the page keeps the
    palette. Marks are built from the same disclosure-filtered facts the message was written
    from, so a mark never shows more than the words do.
    """

    kind: str
    squares: list[str]
    color: str = "focus"
    opacity: float = 1.0
    label: str | None = None


class CoachMessage(Event):
    text: str
    level: DisclosureLevel
    trigger: Trigger
    game_id: str | None = None
    ply: int | None = None
    # The position the message is about (a played move's origin, or the live position), so
    # a panel can show the marks on the board they belong to rather than the one now on screen.
    fen: str | None = None
    annotations: list[Annotation] = Field(default_factory=list)


class ProfileEvidence(Event):
    game_id: str
    ply: int
    phase: str
    expected_loss: float
    motif: str | None = None
    cause: str | None = None
    time_pressure: bool = False
