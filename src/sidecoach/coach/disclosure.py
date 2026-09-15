"""Disclosure state machine and fact-bundle filtering.

Hiding is by construction: the LLM receives only the facts allowed at the current level, so
it cannot repeat what it never saw. The position itself (FEN) is always visible because the
coach needs it to ask sensible questions; `leak_check` is the second line of defence over the
coach's draft text for anything it might work out on its own.

Levels, from DESIGN.md section 5.5:

    0 SILENT     nothing yet
    1 QUESTION   an orienting question
    2 THEME      the theme category
    3 REGION     a board region; autonomous ceiling
    4 PIECE      the piece carrying the idea; only on a hint request
    5 CANDIDATE  the move or near-equal set; user named it, played it, or asked
    6 LINE       the full line; user asked for it, or after the move
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from sidecoach.events import DisclosureLevel

AUTONOMOUS_CEILING = DisclosureLevel.REGION
HINT_CEILING = DisclosureLevel.PIECE


class Unlock(StrEnum):
    STUCK = "stuck"
    DECISIVE = "decisive"
    HINT = "hint"
    ASK_MOVE = "ask_move"
    ASK_LINE = "ask_line"
    USER_NAMED_MOVE = "user_named_move"
    MOVE_PLAYED = "move_played"


class PositionFacts(BaseModel):
    """Everything analysis knows about one position, before filtering."""

    fen: str
    side_to_move: str
    phase: str
    material: str | None = None
    last_move: str | None = None
    decisive: bool = False
    only_move: bool = False
    theme: str | None = None
    region: str | None = None
    piece: str | None = None
    # The squares that carry the theme: the pinner, the pinned piece and what it shields;
    # the forking piece and its targets. Given only at the piece level and above, since
    # before that they would give the answer away.
    focus: list[str] = Field(default_factory=list)
    # The theme as a finished sentence, grounded on the board: "the knight on d3 is pinned
    # to the queen on d7 by the rook on d1". Given at the piece level and above, like focus.
    tactic: str | None = None
    # The strategic idea behind the best move, as a square-free theme ("finishing your
    # development", "converting your extra material"). Given from the theme level, since it
    # orients without naming the move. This is how the coach teaches ideas, not just tactics.
    plan: str | None = None
    # The most instructive static features of the position ("your rook holds the open e-file",
    # "you are up material, so trade and convert"), in the coach's words. From the region
    # level: they describe what is on the board, not the move to find.
    ideas: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    line: list[str] = Field(default_factory=list)
    user_claims: list[dict[str, Any]] = Field(default_factory=list)


_ALWAYS: tuple[str, ...] = (
    "fen",
    "side_to_move",
    "phase",
    "material",
    "last_move",
    "decisive",
    "only_move",
    "user_claims",
)
_BY_LEVEL: dict[DisclosureLevel, tuple[str, ...]] = {
    DisclosureLevel.SILENT: (),
    DisclosureLevel.QUESTION: (),
    DisclosureLevel.THEME: ("theme", "plan"),
    DisclosureLevel.REGION: ("theme", "plan", "ideas", "region"),
    DisclosureLevel.PIECE: ("theme", "plan", "ideas", "region", "piece", "focus", "tactic"),
    DisclosureLevel.CANDIDATE: (
        "theme",
        "plan",
        "ideas",
        "region",
        "piece",
        "focus",
        "tactic",
        "candidates",
        "threats",
    ),
    DisclosureLevel.LINE: (
        "theme",
        "plan",
        "ideas",
        "region",
        "piece",
        "focus",
        "tactic",
        "candidates",
        "threats",
        "line",
    ),
}


def visible_facts(facts: PositionFacts, level: DisclosureLevel) -> dict[str, Any]:
    """The fact bundle the LLM may see at `level`."""
    allowed = set(_ALWAYS) | set(_BY_LEVEL[level])
    return {key: value for key, value in facts.model_dump().items() if key in allowed}


@dataclass
class Disclosure:
    """Per-position disclosure state. Create a fresh one for every new position."""

    level: DisclosureLevel = DisclosureLevel.SILENT
    history: list[tuple[Unlock, DisclosureLevel]] = field(default_factory=list)

    def apply(self, unlock: Unlock) -> DisclosureLevel | None:
        """Apply an unlock. Returns the new level, or None when nothing changed."""
        before = self.level
        match unlock:
            case Unlock.STUCK | Unlock.DECISIVE:
                if self.level < AUTONOMOUS_CEILING:
                    self.level = DisclosureLevel(self.level + 1)
            case Unlock.HINT:
                if self.level < HINT_CEILING:
                    self.level = DisclosureLevel(self.level + 1)
            case Unlock.ASK_MOVE | Unlock.USER_NAMED_MOVE:
                self.level = max(self.level, DisclosureLevel.CANDIDATE)
            case Unlock.ASK_LINE | Unlock.MOVE_PLAYED:
                self.level = DisclosureLevel.LINE
        if self.level == before:
            return None
        self.history.append((unlock, self.level))
        return self.level


# A token is a bare square ("f7") or a SAN move ending in its destination square
# ("Ng5", "Nxf7+", "exd5", "R1e1", "e8=Q"). Group 1 is the destination square.
_SQUARE_TOKEN = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9+#=!?]+")
_TRAILING_MARKS = re.compile(r"[+#!?]+$")


def squares_in(text: str) -> set[str]:
    """Destination squares of every bare square or SAN token in `text`."""
    found: set[str] = set()
    for token in _TOKEN_SPLIT.split(text):
        match = _SQUARE_TOKEN.fullmatch(token)
        if match:
            found.add(match.group(1))
    return found


def leak_check(text: str, facts: PositionFacts, level: DisclosureLevel) -> list[str]:
    """Withheld items that `text` mentions. An empty list means the draft is clean.

    Below CANDIDATE the coach may not name any move at all, and may mention only squares
    that occur in visible facts other than the FEN, such as a region anchor. The model will
    volunteer moves on its own even when told not to, so this is not limited to the withheld
    facts. From CANDIDATE up only the withheld line is checked. Below PIECE the piece word
    from the facts is checked as well.
    """
    visible = visible_facts(facts, level)
    allowed_squares: set[str] = set()
    for key, value in visible.items():
        if key == "fen" or not value:
            continue
        parts = value if isinstance(value, list) else [value]
        allowed_squares |= squares_in(" ".join(str(part) for part in parts))

    leaks: list[str] = []

    def add(item: str) -> None:
        if item not in leaks:
            leaks.append(item)

    if level < DisclosureLevel.CANDIDATE:
        for token in _TOKEN_SPLIT.split(text):
            match = _SQUARE_TOKEN.fullmatch(token)
            if not match:
                continue
            square = match.group(1)
            core = _TRAILING_MARKS.sub("", token)
            if core != square:
                add(f"move {core}")
            elif square not in allowed_squares:
                add(f"square {square}")
    elif level < DisclosureLevel.LINE:
        shown = set(facts.candidates) | set(facts.threats)
        withheld_moves = [move for move in facts.line if move not in shown]
        withheld_squares = squares_in(" ".join(withheld_moves)) - allowed_squares
        for square in sorted(squares_in(text) & withheld_squares):
            add(f"square {square}")
        for move in withheld_moves:
            if re.search(rf"(?<![A-Za-z]){re.escape(move)}(?![A-Za-z0-9])", text):
                add(f"move {move}")

    if level < DisclosureLevel.PIECE and facts.piece:
        piece_word = facts.piece.lower().split()[-1]
        if re.search(rf"\b{re.escape(piece_word)}\b", text.lower()):
            add(f"piece {piece_word}")
    return leaks
