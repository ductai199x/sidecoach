"""Disclosure state machine and fact-bundle filtering.

Hiding is by construction: the LLM receives only the facts allowed at the current level, so
it cannot repeat what it never saw. The position itself (FEN) is always visible because the
coach needs it to ask sensible questions; `leak_check` is the second line of defence over the
coach's draft text for anything it might work out on its own.

Levels, from DESIGN.md section 5.5 -- two hints, then the answer (Tai, 2026-09-23: four
rungs before the move left each saying a sliver, and the slivers came out generic):

    0 SILENT  nothing yet
    1 IDEA    the situation, as a short lesson on the position: what is at stake (only one
              move holds, a few keep the edge, nothing is urgent), the key fact named on the
              board (their threat, the target, the plan and its pieces), the principle behind
              it, the trap -- pieces, squares and their moves, never the player's move
              (Tai, 2026-09-24: a square-free first hint could only say "fight for the
              centre", and in a middlegame that teaches nothing)
    2 FOCUS   the piece that does it and the question that leads to the move; the
              autonomous ceiling
    3 MOVE    the move, with a correct explanation; the third hint, or asked, named or played
    4 LINE    the full line; asked for, or after the move
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from sidecoach.events import DisclosureLevel

# Unprompted, the coach gives the idea and, if the player is still stuck, the focus; never
# the move (Tai, 2026-09-23). Asked for hints, the third is the move.
AUTONOMOUS_CEILING = DisclosureLevel.FOCUS
HINT_CEILING = DisclosureLevel.MOVE


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
    # The side the coach speaks to, when the position is not that side's to move (the
    # opponent is thinking); None means the side to move.
    user_side: str | None = None
    phase: str
    material: str | None = None
    last_move: str | None = None
    decisive: bool = False
    only_move: bool = False
    # Whether the best idea is forcing -- a check, a capture, a promotion -- or quiet. Given
    # from the question level: "have you looked at every check and capture?" is the first
    # thing a coach asks a stuck player, and it points at no square, piece or move; "there
    # is no tactic here" stops a fruitless hunt. Without it the model, seeing only the
    # board, fell back on the same pawn-advance homily for every stuck hint.
    nature: str | None = None
    # What the best line is for, from the side to move: "you can force mate in two", "you
    # can win their queen", "your opponent has a threat to meet first". The first thing a
    # puzzle book says and the one the coach never did: without it the first word on a mate
    # in one was "what is your opponent's most dangerous idea?" (Tai, 2026-09-23). Given at
    # the idea level; it names no square, piece or move.
    goal: str | None = None
    theme: str | None = None
    # The pattern at work and why it works here, square-free: "a sacrifice first drags their
    # king out, then a rook and a knight mate it on the edge". The idea level.
    mechanism: str | None = None
    region: str | None = None
    # The target on the board, grounded: "their king on h8, with no free square", "their
    # king on c1 and their rook on e1: one check can hit both". The focus level.
    target: str | None = None
    piece: str | None = None
    # Why the move works, played out: "after Kxh7, Rh5 is mate". The move level.
    why: str | None = None
    # The question that sets the player looking the right way for this goal and pattern:
    # "Which two of their pieces could one of yours hit at once, and from where?". The idea
    # rung's question, for the templates and as the model's starting point.
    question: str | None = None
    # The focus rung's question, from the target to the move: "Which square hits both?".
    nudge: str | None = None
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
    # The first hint's parts (`coach/situation.py`), named on the board: what is at stake
    # ("only one move holds here: even the next best move loses a knight"), the key fact
    # ("they threaten Nxe5, winning your pawn on e5"), the trap ("hxg4 looks like it wins a
    # knight, but it is a trap: ..."), the good moves when several are fine or the position
    # is opening theory, and the whole of it as the first hint says it.
    stakes: str | None = None
    key: str | None = None
    trap: str | None = None
    choices: list[str] = Field(default_factory=list)
    situation: str | None = None


_ALWAYS: tuple[str, ...] = (
    "fen",
    "side_to_move",
    "user_side",
    "phase",
    "material",
    "last_move",
    "decisive",
    "only_move",
    "user_claims",
)
_IDEA = (
    "nature",
    "goal",
    "theme",
    "mechanism",
    "question",
    "plan",
    "stakes",
    "key",
    "trap",
    "choices",
    "situation",
)
_FOCUS = (*_IDEA, "target", "piece", "nudge", "region", "ideas")
_MOVE = (*_FOCUS, "why", "tactic", "focus", "candidates", "threats")
_BY_LEVEL: dict[DisclosureLevel, tuple[str, ...]] = {
    DisclosureLevel.SILENT: (),
    DisclosureLevel.IDEA: _IDEA,
    DisclosureLevel.FOCUS: _FOCUS,
    DisclosureLevel.MOVE: _MOVE,
    DisclosureLevel.LINE: (*_MOVE, "line"),
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
            case Unlock.STUCK:
                # A player stuck for a while gets the idea at once, the focus next.
                if self.level < AUTONOMOUS_CEILING:
                    self.level = DisclosureLevel(self.level + 1)
            case Unlock.DECISIVE:
                # The critical moment's first word is the idea.
                if self.level < DisclosureLevel.IDEA:
                    self.level = DisclosureLevel.IDEA
            case Unlock.HINT:
                # Two hints, then the move.
                if self.level < HINT_CEILING:
                    self.level = DisclosureLevel(self.level + 1)
            case Unlock.ASK_MOVE | Unlock.USER_NAMED_MOVE:
                self.level = max(self.level, DisclosureLevel.MOVE)
            case Unlock.ASK_LINE | Unlock.MOVE_PLAYED:
                self.level = DisclosureLevel.LINE
        if self.level == before:
            return None
        self.history.append((unlock, self.level))
        return self.level


# Where a piece word is never a hint at the piece to move: an opening's name, the material
# balance, a check or capture said of the move just played.
_PIECE = r"(?:king|queen|rook|bishop|knight|pawn)"
_PIECE_WORD_CONTEXTS = re.compile(
    r"(?:king's|queen's|bishop's|knight's|rook's) "
    r"(?:indian|gambit|pawn|opening|attack|defen[cs]e)"
    r"|(?:up|down|ahead|behind|level|even|for|against|exchange)(?: a| an| two| three| the)? "
    rf"(?:{_PIECE}|piece)s?(?: (?:and|for) (?:a |an |two |three |the )?(?:{_PIECE}|piece)s?)*"
    rf"|(?:their|opponent's|the opponent's|enemy) {_PIECE}s?"
    rf"|(?:wins?|winning|won|trap|traps|trapped|lose|loses|losing) (?:a |an |the |their |two )?"
    rf"{_PIECE}s?"
    r"|(?:checks?|attacks?|takes|took|captures?d?|removes?d?|hits?) (?:your|their|the|my) "
    rf"{_PIECE} on [a-h][1-8]"
    rf"|(?:your|their|my|the) {_PIECE} on [a-h][1-8] (?:is|was|remains|stays) (?:in )?"
    r"(?:check|attacked|under attack|hanging|loose|defended|undefended|pinned)",
    re.IGNORECASE,
)

# A token is a bare square ("f7") or a SAN move ending in its destination square
# ("Ng5", "Nxf7+", "exd5", "R1e1", "e8=Q"). Group 1 is the destination square.
_SQUARE_TOKEN = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9+#=!?]+")
_TRAILING_MARKS = re.compile(r"[+#!?]+$")
_WORD_RE = re.compile(r"[a-z0-9=+#-]+")


def squares_in(text: str) -> set[str]:
    """Destination squares of every bare square or SAN token in `text`."""
    found: set[str] = set()
    for token in _TOKEN_SPLIT.split(text):
        match = _SQUARE_TOKEN.fullmatch(token)
        if match:
            found.add(match.group(1))
    return found


def leak_check(
    text: str, facts: PositionFacts, level: DisclosureLevel, *, said: str | None = None
) -> list[str]:
    """Withheld items that `text` mentions. An empty list means the draft is clean.

    Below MOVE the coach may not name any move at all, and may mention only squares that
    occur in visible facts other than the FEN, such as the target's. The model will volunteer
    moves on its own even when told not to, so this is not limited to the withheld facts.
    From MOVE up only the withheld line is checked. Below FOCUS the word for the player's own
    piece is checked as well -- their pieces, and what a line wins, are not it.

    `said` is what the user asked this turn: a square, piece word or move they named is theirs
    already, so confirming it back is not a leak and never faulted.
    """
    visible = visible_facts(facts, level)
    allowed_squares: set[str] = set()
    for key, value in visible.items():
        if key == "fen" or not value:
            continue
        parts = value if isinstance(value, list) else [value]
        allowed_squares |= squares_in(" ".join(str(part) for part in parts))
    said_squares = squares_in(said) if said else set()
    # A move said with its check or mate mark ("Qh4+") is the same move said without it.
    said_words = (
        {w for word in _WORD_RE.findall(said.lower()) for w in (word, word.rstrip("+#"))}
        if said
        else set()
    )
    allowed_squares |= said_squares
    # The moves the visible facts name -- their threat, the trap, the moves that are all as
    # good -- are said in the open; never the one the player is to find.
    allowed_moves: set[str] = set()
    for key, value in visible.items():
        if key == "fen" or not value:
            continue
        parts = value if isinstance(value, list) else [value]
        for token in _TOKEN_SPLIT.split(" ".join(str(part) for part in parts)):
            if _SQUARE_TOKEN.fullmatch(token):
                allowed_moves.add(_TRAILING_MARKS.sub("", token))

    leaks: list[str] = []

    def add(item: str) -> None:
        if item not in leaks:
            leaks.append(item)

    if level < DisclosureLevel.MOVE:
        for token in _TOKEN_SPLIT.split(text):
            match = _SQUARE_TOKEN.fullmatch(token)
            if not match:
                continue
            square = match.group(1)
            core = _TRAILING_MARKS.sub("", token)
            if core != square:
                if core.lower() not in said_words and core not in allowed_moves:
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

    if level < DisclosureLevel.FOCUS and facts.piece:
        # The words for the player's own pieces ("your queen on h5", "your rook first, then
        # your knight") are a hint only where they could point at the piece to move: not in
        # an opening's name ("Queen's Gambit"), not in the material balance ("down a pawn"),
        # not in what a line wins or a fact the level shows ("you can win a queen and a pawn
        # for a bishop"), not said of a move already played -- the caller's `said` carries
        # those.
        scrubbed = text.lower()
        # The whole first hint first: once a part of it is taken out, the whole no longer
        # matches, and its words were read as the coach's own.
        for key in (
            "situation",
            "goal",
            "theme",
            "mechanism",
            "question",
            "plan",
            "nature",
            "stakes",
            "key",
            "trap",
        ):
            value = visible.get(key)
            if isinstance(value, str) and value:
                scrubbed = scrubbed.replace(value.lower(), " ")
        scrubbed = _PIECE_WORD_CONTEXTS.sub(" ", scrubbed)
        for piece_word in dict.fromkeys(re.findall(_PIECE, facts.piece.lower())):
            # "The king behind it", "their king runs out of squares": the word is in every
            # explanation of a pin or a mate, and says nothing about which piece to move.
            if piece_word in said_words or piece_word == "king":
                continue
            if re.search(rf"\b{re.escape(piece_word)}\b", scrubbed):
                add(f"piece {piece_word}")
    return leaks
