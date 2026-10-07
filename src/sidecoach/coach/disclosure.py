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
from collections.abc import Collection
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
    HINT = "hint"
    ASK_MOVE = "ask_move"
    ASK_LINE = "ask_line"
    USER_NAMED_MOVE = "user_named_move"
    MOVE_PLAYED = "move_played"
    # What the AI coach's reply to a question actually said (`said_level`): the ladder follows
    # the words, not a guess at what the question meant.
    REPLY = "reply"


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
    # The moves about as good as the best (graded Best or Excellent), and those graded Good, a
    # little behind them (`facts.graded_moves`): a good move the player names is said to be
    # good, never refused as bad.
    candidates: list[str] = Field(default_factory=list)
    good: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)
    line: list[str] = Field(default_factory=list)
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
    # The principle's sentence as `situation` says it ("A move that attacks something they
    # must then defend keeps you a step ahead: ..."), so the coach can leave it out while it
    # cools down (coach.RULE_COOLDOWN): it came 218 times in the master games' hints, 59 of
    # them again in the same game. Not on the model's sheet: `situation` carries it.
    principle: str | None = None


_ALWAYS: tuple[str, ...] = (
    "fen",
    "side_to_move",
    "user_side",
    "phase",
    "material",
    "last_move",
    "decisive",
    "only_move",
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
    # Their threat is the first hint's key fact as often as not ("they threaten Nxe5,
    # winning your pawn"): withheld to the move level, "what is their threat?" got "nothing"
    # (the runtime audit, 2026-09-25).
    "threats",
)
_FOCUS = (*_IDEA, "target", "piece", "nudge", "region", "ideas")
_MOVE = (*_FOCUS, "why", "tactic", "focus", "candidates")
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
    # What the combination had said before the player came here (`carry`).
    carried: DisclosureLevel = DisclosureLevel.SILENT

    def apply(self, unlock: Unlock) -> DisclosureLevel | None:
        """Apply an unlock. Returns the new level, or None when nothing changed."""
        before = self.level
        match unlock:
            case Unlock.STUCK:
                # A player stuck for a while gets the idea at once, the focus next.
                if self.level < AUTONOMOUS_CEILING:
                    self.level = DisclosureLevel(self.level + 1)
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

    def said(self, level: DisclosureLevel) -> DisclosureLevel | None:
        """A reply to the player's question said up to `level` (`said_level`): the ladder
        climbs to it and never falls. Returns the new level, or None when nothing climbed."""
        if level <= self.level:
            return None
        self.level = level
        self.history.append((Unlock.REPLY, level))
        return level

    @property
    def hinted(self) -> DisclosureLevel:
        """The highest level said here before the player moved: every unlock but the move
        played, which opens a past position to its line."""
        return max(
            (level for unlock, level in self.history if unlock is not Unlock.MOVE_PLAYED),
            default=DisclosureLevel.SILENT,
        )

    def carry(self, level: DisclosureLevel) -> None:
        """Start from the hints already given for the combination this position continues
        (`Coach._ladder`): each level is said once, and none below the highest (Tai,
        2026-09-29: the stuck hint's idea at a puzzle's start came again as the question
        after the move was found and the reply left the engine's line)."""
        self.carried = max(self.carried, level)
        if level > self.level:
            self.level = level

    def withdraw_move(self) -> None:
        """The move played from here is taken back: the position is the player's to move
        again, at the level of the hints it had had. The line the move opened (`MOVE_PLAYED`)
        belonged to the move withdrawn; what was said before it stays said, and nothing lower
        is said after it (Tai, 2026-09-29)."""
        self.history = [(u, lvl) for u, lvl in self.history if u is not Unlock.MOVE_PLAYED]
        self.level = max(self.hinted, self.carried)


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
    r"|(?:check(?:s|ing|ed)?|attack(?:s|ing|ed)?|takes|taking|took|captur(?:es?|ing|ed)"
    r"|remov(?:es?|ing|ed)|hit(?:s|ting)?|threaten(?:s|ing|ed)?|win(?:s|ning)?|won"
    r"|pin(?:s|ning|ned)?|fork(?:s|ing|ed)?|skewer(?:s|ing|ed)?|go(?:es|ing)? after) "
    rf"(?:your|their|the|my) {_PIECE} on [a-h][1-8]"
    # What their threat goes after, said without its square: "g5 attacks your queen" is the
    # threat the sheet gives, not the move to find -- never "take your queen" or "attack
    # with your queen" (the smoke run, 2026-09-25).
    r"|(?:attack(?:s|ing|ed)?|threaten(?:s|ing|ed)?|hit(?:s|ting)?|pin(?:s|ning|ned)?"
    r"|fork(?:s|ing|ed)?|skewer(?:s|ing|ed)?|go(?:es|ing)? after) (?:your|the|my) "
    rf"{_PIECE}s?\b(?! to\b)"
    rf"|(?:your|their|my|the) {_PIECE} on [a-h][1-8] (?:is|was|remains|stays) (?:in )?"
    r"(?:check|attacked|under attack|hanging|loose|defended|undefended|pinned)",
    re.IGNORECASE,
)

# A move with its number in front: the game's record when the game played it.
_NUMBERED_MOVE = re.compile(
    r"\b\d+\.(?:\.\.)?\s*(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?"
)


def numbered_key(text: str) -> str:
    """A numbered move in one spelling: "17. Rh1+" and "17.Rh1" are "17.Rh1"."""
    return re.sub(r"\s+", "", text).rstrip("+#")


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
    text: str,
    facts: PositionFacts,
    level: DisclosureLevel,
    *,
    said: str | None = None,
    record: Collection[str] = (),
    shown: str = "",
) -> list[str]:
    """Withheld items that `text` mentions. An empty list means the draft is clean.

    Below MOVE only the answer is withheld (see below); from MOVE up only the withheld
    line. `record` is the game's moves as numbered in the record ("17. Rh1"). Below FOCUS
    the word for the player's own piece is checked as well -- their pieces, and what a line
    wins, are not it.

    `said` is what the user asked this turn: a square, piece word or move they named is theirs
    already, so confirming it back is not a leak and never faulted. `shown` is what the sheet
    shows beyond the facts' own words -- what the threats do -- whose pieces on their squares
    are facts about the board, like the facts'.

    Below MOVE only the answer is withheld (Tai, 2026-09-25: "leaking is not a criminal
    offense ... it's useful to point out in precise manner the plan (yours or opponent's) or
    the positional advantage or disadvantage ... clamp down egregious offense but be smart
    about it"): the moves the player is to find -- the good moves, the player's moves in the
    best line -- and the squares they go to, unless the visible facts name them. Any other
    move or square -- a plan, a weakness, their threat, a line the facts give -- may be said.
    The moves of the game are no exemption: the answer is withheld even when it was played
    before (by mid-game that exemption covered most of the board: the runtime audit).
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
    # And the squares the sheet shows beyond the facts' words: a threat "attacks the knight on
    # e7" may be said though e7 is where the answer goes (the benchmark's draw, 2026-09-25).
    allowed_squares |= squares_in(shown)
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

    # A move of the game said with its number ("17. Rh1", "4... Nf6") is the record, not a
    # suggestion: the review names the moves where ground was lost, and one may be the
    # answer again. Only a numbered move the game actually played -- "5. cxd5 wins it" of
    # the move to find was let through by any number at all (the second audit).
    played = {numbered_key(m) for m in record}
    moves_text = _NUMBERED_MOVE.sub(
        lambda m: " " if numbered_key(m.group(0)) in played else m.group(0), text
    )
    if level < DisclosureLevel.MOVE:
        answer = {_TRAILING_MARKS.sub("", m) for m in [*facts.candidates, *facts.line[0::2]] if m}
        answer_squares = squares_in(" ".join(answer)) - allowed_squares
        for token in _TOKEN_SPLIT.split(moves_text):
            match = _SQUARE_TOKEN.fullmatch(token)
            if not match:
                continue
            square = match.group(1)
            core = _TRAILING_MARKS.sub("", token)
            if core != square:
                known = core.lower() in said_words or core in allowed_moves
                if core in answer and not known:
                    add(f"move {core}")
            elif square in answer_squares and square not in said_squares:
                add(f"square {square}")
    elif level < DisclosureLevel.LINE:
        visible_moves = {_TRAILING_MARKS.sub("", m) for m in [*facts.candidates, *facts.threats]}
        # The move's why names its continuation ("after Qf6, Bg2 ..."): what the level shows
        # may be said -- compared without check marks, as `allowed_moves` holds them ("Rc6+"
        # was withheld while the sheet gave it: the pre-spend audit, 2026-09-25).
        withheld_moves = [
            m
            for m in facts.line
            if _TRAILING_MARKS.sub("", m) not in visible_moves
            and _TRAILING_MARKS.sub("", m) not in allowed_moves
        ]
        withheld_squares = squares_in(" ".join(withheld_moves)) - allowed_squares
        for square in sorted(squares_in(moves_text) & withheld_squares):
            add(f"square {square}")
        for move in withheld_moves:
            if re.search(rf"(?<![A-Za-z]){re.escape(move)}(?![A-Za-z0-9])", moves_text):
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
            "material",
        ):
            value = visible.get(key)
            if isinstance(value, str) and value:
                scrubbed = scrubbed.replace(value.lower(), " ")
        # A piece the visible facts name on its square, said in other words ("their pawn on
        # b5 is about to take your bishop on a4" for "winning your bishop on a4"), is the
        # fact, not a hint at the piece to move.
        on_sheet = " ".join([*(str(v) for v in visible.values() if isinstance(v, str)), shown])
        on_sheet = on_sheet.lower()
        for piece_word, square in set(re.findall(rf"({_PIECE})s? on ([a-h][1-8])", on_sheet)):
            scrubbed = re.sub(rf"\b{piece_word}s? on {square}\b", " ", scrubbed)
        scrubbed = _PIECE_WORD_CONTEXTS.sub(" ", scrubbed)
        # The piece to move is named when it is said on its own square, or as "your <piece>"
        # with no square; another piece of that kind on another square ("attacks the pawn
        # on d2" when the pawn to move is on e4) is a fact about the board, not the hint.
        # Without a square on the sheet, any mention counts, as before.
        piece_squares = squares_in(facts.piece)
        # The file of a pawn to move: "the c-pawn" names the pawn on c4.
        piece_files = {sq[0] for sq in piece_squares}
        # The piece that does it, said as the one that takes or goes ("take it with the queen",
        # "capture with a pawn", "the c-pawn should take it": the second audit) -- not every
        # mention of its kind: "the pawn on c3", "your pawn on h2" are the board (round 2
        # refused four good replies for them).
        for piece_word in dict.fromkeys(re.findall(_PIECE, facts.piece.lower())):
            # "The king behind it", "their king runs out of squares": the word is in every
            # explanation of a pin or a mate, and says nothing about which piece to move.
            if piece_word in said_words or piece_word == "king":
                continue
            # The sheet itself names the player's piece of this kind at this level ("the first
            # move takes your bishop out of danger"): said in other words, it is no leak -- a
            # paraphrase of the first hint was refused for it, and the bare template spoke
            # (the benchmark's smoke run, 2026-09-25; Tai: "a vague coach is worse than no
            # coach at all").
            if re.search(rf"\byour {piece_word}s?\b", on_sheet):
                continue
            mention = re.compile(
                rf"\b(your |my |with (?:the |a |your )?|(?:([a-h])-))?{re.escape(piece_word)}s?\b"
                r"(?: (?:on|from) ([a-h][1-8]))?"
            )
            for found in mention.finditer(scrubbed):
                prefix, file_, square = found.group(1), found.group(2), found.group(3)
                if square is not None:
                    named = square in piece_squares
                elif file_ is not None:
                    named = file_ in piece_files or not piece_files
                else:
                    named = prefix is not None or not piece_squares
                if named:
                    add(f"piece {piece_word}")
                    break
    return leaks


def names_move(text: str, move: str) -> bool:
    """Whether `text` names `move` as a move, its own token: e5 is not named by "Be5" or
    "dxe5". The coach's "move already given" reads it too: a substring test took "They
    threaten Be5" for the move e5 given, and the model was told to say it again (the audit,
    2026-10-07)."""
    core = _TRAILING_MARKS.sub("", move)
    if not core:
        return False
    if core.startswith("O-O"):
        castle = rf"(?<![-A-Za-z0-9]){re.escape(core)}(?!-O)(?![A-Za-z0-9])"
        return re.search(castle, text) is not None
    return re.search(rf"(?<![A-Za-z]){re.escape(core)}[+#]?(?![A-Za-z0-9])", text) is not None


# A piece named by whose it is: "their king", "your queen", "the opponent's rook".
_OWNED_PIECE = re.compile(
    rf"\b(your|their|the opponent's|opponent's)\s+({_PIECE})s?\b", re.IGNORECASE
)
_MATE_NAMED = re.compile(r"\b(?:check)?mat(?:e|es|ed|ing)\b", re.IGNORECASE)
_PROMOTION_NAMED = re.compile(r"\bpromot(?:e|es|ed|ing|ion)\b", re.IGNORECASE)


def _things_named(text: str) -> set[str]:
    """What `text` names besides squares and moves: the pieces it names by whose they are
    ("their king"), a mate, a promotion."""
    found = {
        f"{'your' if owner.lower() == 'your' else 'their'} {piece.lower()}"
        for owner, piece in _OWNED_PIECE.findall(text)
    }
    if _MATE_NAMED.search(text):
        found.add("mate")
    if _PROMOTION_NAMED.search(text):
        found.add("promotion")
    return found


def _moves_in(text: str) -> list[str]:
    """The moves `text` names in SAN (a piece move, a capture, a promotion, castling), not
    the bare squares."""
    out = [m.group(0) for m in re.finditer(r"(?<![-A-Za-z0-9])O-O(?:-O)?(?![-A-Za-z0-9])", text)]
    for token in _TOKEN_SPLIT.split(text):
        match = _SQUARE_TOKEN.fullmatch(token)
        if match and _TRAILING_MARKS.sub("", token) != match.group(1):
            out.append(token)
    return out


def said_level(text: str, facts: PositionFacts) -> DisclosureLevel:
    """The rung a reply of the AI coach carried, from what it names: the first two moves of the
    engine's line (LINE); a move the sheet lists as a candidate (MOVE); a square of the target,
    the piece or the squares that carry the theme, or the piece that does it (FOCUS); a square,
    a move, a piece, a mate or a promotion that the idea's facts name -- the goal, the theme,
    the key fact, the stakes, the trap, the threats, the target's piece (IDEA); else SILENT --
    nothing of the ladder was said. A reply that only answers "am I winning?" names none of it,
    and the ladder does not move (Tai, 2026-10-02: no word-overlap guess, no floor).

    The ladder of hints (each rung said once, none below one already said) follows this, not a
    guess from the question's words: with the AI on, the model reads the question and decides
    what it answers (Tai, 2026-10-02)."""
    moves_text = _NUMBERED_MOVE.sub(" ", text)
    line = [m for m in facts.line if m]
    if len(line) >= 2 and names_move(moves_text, line[0]) and names_move(moves_text, line[1]):
        return DisclosureLevel.LINE
    answers = [*facts.candidates, *line[:1]]
    if any(names_move(moves_text, m) for m in answers):
        return DisclosureLevel.MOVE
    squares = squares_in(text)
    named = _things_named(text)
    focus = " ".join(x for x in (facts.target, facts.piece, *facts.focus) if x)
    if squares & squares_in(focus) or (facts.piece and named & _things_named(facts.piece)):
        return DisclosureLevel.FOCUS
    idea = " ".join(
        x
        for x in (
            facts.goal,
            facts.theme,
            facts.mechanism,
            facts.key,
            facts.stakes,
            facts.situation,
            facts.trap,
            facts.target,
        )
        if x
    )
    if (
        squares & squares_in(idea)
        or named & _things_named(idea)
        or any(names_move(moves_text, m) for m in [*facts.threats, *_moves_in(idea)])
    ):
        return DisclosureLevel.IDEA
    return DisclosureLevel.SILENT
