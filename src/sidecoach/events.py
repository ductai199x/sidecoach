"""Typed events that connect the pipeline stages.

Every stage consumes and emits these models and nothing else, so any stage can be replayed
from a recording. Pixels never travel inside an event: a Frame carries the frame id and
geometry, and the pixel buffer is looked up from the capture ring by id.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from time import monotonic
from typing import Literal, NamedTuple

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

    @property
    def color(self) -> bool:
        """The side as python-chess's colour (`chess.WHITE` is True)."""
        return self is Side.WHITE

    @property
    def other(self) -> Side:
        return Side.BLACK if self is Side.WHITE else Side.WHITE

    @staticmethod
    def of(color: bool) -> Side:
        """The side of a python-chess colour."""
        return Side.WHITE if color else Side.BLACK


class ClockObservation(Event):
    white_ms: int | None = None
    black_ms: int | None = None
    running: Side | None = None


class ScreenState(StrEnum):
    LIVE_GAME = "live_game"
    PUZZLES = "puzzles"
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


class BoardUnread(Event):
    """A board on screen, no game followed on it, and nothing read from it for a while (Tai
    on the Pixel, 2026-09-23: a puzzle up, "the program is not registering anything"). A
    note for the person, said once while the board stays: "unlearned" when no learned
    board-and-piece set reads it -- the colours never seen, or pieces the sets on those
    colours misread -- so a start position shown once teaches it; "orientation" when a
    learned set reads it but the coordinates cannot be, so Flip must say which side is at
    the bottom."""

    reason: Literal["unlearned", "orientation"]


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
    # The captures that look free and are not (`analysis/traps.py`): each the capture, then
    # the engine's best play after it, scored from the side to move's view.
    traps: list[Line] = Field(default_factory=list)
    # The recaptures the best line declines (`analysis/defences.py`): each from the root, the
    # line up to the player's capture, the recapture, then the engine's best play after it,
    # scored from the side to move's view.
    defences: list[Line] = Field(default_factory=list)
    # What the best move threatens when it is quiet (`analysis/threats.aim_lines`): the side
    # to move's best line after it and a free move for the opponent, from that side's view.
    threatens: list[Line] = Field(default_factory=list)
    tags: list[MotifTag] = Field(default_factory=list)
    opening: str | None = None
    # The game is still in the opening book at this position (`opening` is the last named
    # position it reached in its first 40 plies, as Lichess names it, and stays after the game
    # leaves the book).
    in_book: bool = False
    # The position measured early (`annotator.EARLY_DEPTH`), the full depth to follow: what
    # the player asks is answered from it at once; the coach's own words, a verdict and the
    # record wait for the full depth.
    provisional: bool = False


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
    # Per-move accuracy on a steepened Lichess curve (analysis/score.py), 0 to 100, so a
    # game can be summed up.
    accuracy: float | None = None


class ReviewedMove(BaseModel):
    """One move of a game reviewed whole (`analysis/game_review.py`): who played it from
    where, what it cost, and what the position it led to holds -- the facts the review's card
    on the move is written from, as the live coach's are."""

    ply: int
    san: str
    side: Side
    fen_before: str
    assessment: MoveAssessment
    after: Analysis
    # The mover's clock after the move, and how long the move took, in milliseconds, from
    # the PGN's clock comments; None without them (or in a correspondence game).
    clock_ms: int | None = None
    spent_ms: int | None = None


class ReviewedGame(BaseModel):
    """A game reviewed whole: its headers, what its first position holds, and every move."""

    headers: dict[str, str] = Field(default_factory=dict)
    start: Analysis
    moves: list[ReviewedMove] = Field(default_factory=list)
    strength: str
    depth: int
    seconds: float = 0.0


class PhaseGrade(BaseModel):
    """How one side played one phase: its accuracy there and the grade drawn from it, in the
    vocabulary of the moves themselves (`profile/game_summary.phase_grade`)."""

    phase: str
    moves: int
    accuracy: float
    grade: Label


class PerformanceRating(BaseModel):
    """The rating a side played a game at, from its moves alone (`profile/performance.py`),
    with the middle 80% of where it may be -- one game says little -- on the scale of the site
    the game was played on."""

    rating: int
    low: int
    high: int


class SideSummary(BaseModel):
    """One side of a reviewed game in numbers: the summary screen's column."""

    side: Side
    name: str | None = None
    rating: int | None = None
    accuracy: float | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    phases: list[PhaseGrade] = Field(default_factory=list)
    performance: PerformanceRating | None = None


class KeyMoment(BaseModel):
    """A move the review stops at: the last book move, or a move with a label worth a word
    of its own (brilliant, great, inaccuracy, mistake, miss, blunder)."""

    ply: int
    side: Side
    kind: str


class GameSummary(BaseModel):
    """A reviewed game summed up: each side's numbers, the moves the review stops at, and
    White's expected score after every ply (the graph; the start position first)."""

    white: SideSummary
    black: SideSummary
    opening: str | None = None
    last_book_ply: int | None = None
    key_moments: list[KeyMoment] = Field(default_factory=list)
    graph: list[float] = Field(default_factory=list)
    shape: str | None = None


# Coach and UI


class DisclosureLevel(IntEnum):
    """Two hints, then the answer (Tai, 2026-09-23: "2 is probably the right number with the
    third giving the answer with a good and correct explanation"). IDEA: the goal and the
    pattern, and why it works. FOCUS: the target on the board and the piece that does it.
    MOVE: the move, explained. LINE: the whole line."""

    SILENT = 0
    IDEA = 1
    FOCUS = 2
    MOVE = 3
    LINE = 4


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
