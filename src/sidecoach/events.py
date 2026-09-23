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
    # "game": begun from the start position, seen. "position": begun from a board read
    # cold -- a puzzle, an analysis board, a game joined from the middle -- which the
    # coach coaches like a game but the profile does not record and nobody reviews.
    kind: str = "game"


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


class Unexplained(Event):
    """Cells that have disagreed with the game for a long while, or a move refused as
    illegal: a note for the log, the game kept as it stands. Not a resync -- the position
    did not jump and the numbering goes on. Sent as a Resync, it made the coach and the
    engine restart their numbering while the reconciler kept its own, and every later ply
    and takeback disagreed (Tai's game on the Pixel, 2026-09-23: a "takeback of 10 plies"
    the coach read as the last move)."""

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


# Profile (DESIGN section 5.6)


class ProfileEvidence(Event):
    """One mistake of the player's, placed on the two axes: why (the cause) and what
    (the motif), in which phase, and what it cost."""

    game_id: str
    ply: int
    phase: str
    expected_loss: float
    motif: str | None = None
    cause: str | None = None
    time_pressure: bool = False


class TurningPoint(BaseModel):
    """A move of the player's that cost real ground, with what was better and why it
    happened, for the review."""

    ply: int
    san: str
    label: str
    expected_loss: float
    segment: int = 0
    best_san: str | None = None
    cause: str | None = None
    motif: str | None = None
    phase: str = "middlegame"
    fen_before: str | None = None


class PhaseSummary(BaseModel):
    phase: str
    moves: int
    expected_loss: float
    mistakes: int


class GameReview(Event):
    """The game in numbers once it is over: the tally of the player's moves by label,
    their accuracy, where the game turned, and how the phases went. Computed from the
    engine's verdicts, so the coach can sum the game up without inventing it."""

    game_id: str
    user_side: Side
    result: str
    reason: str | None = None
    opening: str | None = None
    moves: int = 0
    accuracy: float | None = None
    tallies: dict[str, int] = Field(default_factory=dict)
    turning_points: list[TurningPoint] = Field(default_factory=list)
    phases: list[PhaseSummary] = Field(default_factory=list)
    # Expected score lost by cause over the game, largest first.
    causes: dict[str, float] = Field(default_factory=dict)
    # The review in sentences, for the coach's summary and the panel's card.
    lines: list[str] = Field(default_factory=list)


class ProfileCell(BaseModel):
    """One cell of the error taxonomy: a Beta-distributed error rate as counts. `errors`
    mistakes out of `chances` occasions; the mean rate is (errors + 1) / (chances + 2)."""

    cause: str | None = None
    motif: str | None = None
    phase: str | None = None
    errors: int = 0
    chances: int = 0
    expected_loss: float = 0.0

    @property
    def rate(self) -> float:
        return (self.errors + 1) / (self.chances + 2)


class PuzzleRef(BaseModel):
    """A Lichess puzzle recommended: where to open it, how hard, what it trains."""

    id: str
    url: str
    rating: int
    themes: list[str] = Field(default_factory=list)


class Recommendation(Event):
    """Practice for the player's weak spot after a game (DESIGN section 5.7): the motif
    the profile says costs them most, and a few puzzles of it near their level."""

    game_id: str
    motif: str
    motif_words: str
    reason: str
    level: int
    puzzles: list[PuzzleRef] = Field(default_factory=list)


class PuzzleOutcome(Event):
    """A recommended (or any known) puzzle worked through on the screen: solved when
    every move of the solution came, failed at the first wrong one."""

    game_id: str
    puzzle_id: str
    motif: str | None = None
    solved: bool
    level: int  # the player's puzzle level after this outcome


class ProfileSnapshot(Event):
    """The player model as it stands: the cells with evidence, what they add up to in
    words, and the last session's summary, for the coach's continuity."""

    games: int = 0
    cells: list[ProfileCell] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)
    last_session: str | None = None
