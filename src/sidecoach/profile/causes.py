"""Why a mistake happened: the cause axis of the error taxonomy (DESIGN section 5.6).

The cause is read from engine evidence, in the order a coach would look: the clock first
(under ten seconds, it was time pressure whatever else it was), then what the move
allowed (a tactic for the opponent, a mate, a piece for free: an opponent's threat
missed), then what the move was (a tactic or a capture that did not work: a line
miscalculated), then what was there instead (a tactic the player had and did not play:
a candidate never considered), then the phase (opening theory, endgame technique), and
positional misjudgment for the rest. The motif axis comes from the tagger: the tactic
allowed, missed or attempted.

Self-reports refine this -- "I saw it but thought it failed" turns a candidate never
considered into a line miscalculated -- and every entry is the user's to correct.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import chess

from sidecoach.analysis import exchange
from sidecoach.analysis.motifs import TAG_ORDER
from sidecoach.events import MotifTag

MISSED_THREAT = "missed opponent threat"
MISCALCULATED = "miscalculated line"
NOT_CONSIDERED = "never considered the candidate"
POSITIONAL = "positional misjudgment"
OPENING_GAP = "opening knowledge gap"
ENDGAME = "endgame technique"
TIME_PRESSURE = "time pressure"

CAUSES: tuple[str, ...] = (
    MISSED_THREAT,
    MISCALCULATED,
    NOT_CONSIDERED,
    POSITIONAL,
    OPENING_GAP,
    ENDGAME,
    TIME_PRESSURE,
)

# Each cause as the player hears it, and the habit that answers it: what the review says
# when one cause keeps coming back. Asked "What do I keep missing?", the reviews named the
# move that cost most and never the pattern -- the simulated students' threat-blind player
# heard "you lost the most ground at 12. Re1" and wrote five unrelated rules (2026-09-26).
HABIT: dict[str, tuple[str, str]] = {
    MISSED_THREAT: (
        "what their moves threatened",
        "before each move, look at what their last move attacks, and at the captures and "
        "checks your move would leave them",
    ),
    MISCALCULATED: (
        "how the exchanges came out",
        "before a capture or a check, follow it to where the exchanges stop and count what is left",
    ),
    NOT_CONSIDERED: (
        "the winning moves you had",
        "before each move, look at every check, capture and attack you have first -- one of "
        "them may win material",
    ),
    POSITIONAL: (
        "where your pieces belonged",
        "when nothing is under attack, find your worst-placed piece and give it a better square",
    ),
    OPENING_GAP: (
        "the opening's basics",
        "in the opening, get your knights and bishops out and castle before going after anything",
    ),
    ENDGAME: (
        "endgame technique",
        "in the endgame, bring your king to the middle and push your passed pawns",
    ),
    TIME_PRESSURE: (
        "the clock",
        "play the easy moves quickly and keep your time for the hard ones",
    ),
}

# A mistake with less than this on the clock defaults to time pressure.
TIME_PRESSURE_MS = 10_000

# The tagger's names that are tactics (what a move creates, what a line does, the mating
# pattern), as against what a move merely is (a capture, a check, a quiet move).
TACTICAL = frozenset(TAG_ORDER[TAG_ORDER.index("fork") :])

_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def tactical(tags: Sequence[MotifTag]) -> str | None:
    """The first tactical motif among the tags, or None."""
    for tag in tags:
        if tag.name in TACTICAL:
            return tag.name
    return None


def wins_material(board: chess.Board, reply: chess.Move) -> bool:
    """Whether the reply takes something for less than it gives: a bigger piece, or one
    nobody can take back -- by the rules, so a guard pinned to its own king, or a king that
    would step into check, takes nothing back (`exchange.can_retake`)."""
    if not board.is_capture(reply) or reply not in board.legal_moves:
        return False
    victim = board.piece_at(reply.to_square)
    attacker = board.piece_at(reply.from_square)
    if attacker is None:
        return False
    taken = _VALUE.get(victim.piece_type, 1) if victim is not None else 1  # en passant
    if taken > _VALUE.get(attacker.piece_type, 0):
        return True
    return not exchange.can_retake(board, reply)


def classify(
    before: chess.Board,
    move: chess.Move,
    *,
    played_tags: Sequence[MotifTag] = (),
    available_tags: Sequence[MotifTag] = (),
    allowed_tags: Sequence[MotifTag] = (),
    reply: str | None = None,
    reply_mate: int | None = None,
    phase: str = "middlegame",
    left_book: bool = False,
    clock_ms: int | None = None,
    missed_chance: bool = False,
) -> tuple[str, str | None]:
    """The cause and the motif of a mistake.

    `before` is the position the player faced and `move` what they played; `played_tags`
    are the tags of that move, `available_tags` those of the position they faced (what
    they could have played), `allowed_tags` those of the position after the move (what
    the opponent now has); `reply` is the opponent's best answer and `reply_mate` its
    mate distance; `left_book` says the move was the first out of theory; `clock_ms` the
    player's clock when the move was made; `missed_chance` that it let the opponent's
    mistake go (a miss).

    A miss with a tactic on the board is the tactic never considered, even when the move
    also allowed one: the chance was the lesson. The simulated tactic-blind student, told
    "what you kept missing: what their moves threatened" for its misses, wrote down the
    threat habit and not its own (2026-09-26).
    """
    allowed = tactical(allowed_tags)
    available = tactical(available_tags)
    attempted = tactical(played_tags)
    if clock_ms is not None and clock_ms < TIME_PRESSURE_MS:
        return TIME_PRESSURE, allowed or available or attempted
    after = before.copy(stack=False)
    after.push(move)
    if reply_mate is not None and reply_mate > 0:
        return MISSED_THREAT, allowed or "mate"
    forcing = before.is_capture(move) or before.gives_check(move)
    if missed_chance and available is not None and not forcing:
        return NOT_CONSIDERED, available
    if forcing:
        # A capture or a check that comes out behind is a line miscalculated, even when the
        # reply it allows wins material: the player began it. Read as a threat missed, the
        # greedy student heard "what you kept missing: what their moves threatened" after
        # every game (the simulated students' round 5, 2026-09-26).
        return MISCALCULATED, attempted or allowed or available
    if allowed is not None:
        return MISSED_THREAT, allowed
    if reply is not None:
        try:
            reply_move = chess.Move.from_uci(reply)
        except ValueError:
            reply_move = None
        if reply_move is not None and wins_material(after, reply_move):
            return MISSED_THREAT, "hangingPiece"
    if attempted is not None:
        return MISCALCULATED, attempted
    if available is not None:
        return NOT_CONSIDERED, available
    if phase == "endgame":
        return ENDGAME, None
    if phase == "opening" or left_book:
        return OPENING_GAP, None
    return POSITIONAL, None


# What a player says about a mistake, when it says plainly what happened.
_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:time|clock|rush(?:ed|ing)?|hurr(?:y|ied)|in a hurry|too fast|quick(?:ly)?|"
            r"premove[sd]?|no time)\b",
            re.I,
        ),
        TIME_PRESSURE,
    ),
    (
        re.compile(
            r"\b(?:(?:i )?(?:thought|figured|calculated|expected|assumed|believed|saw)\b.*"
            r"\b(?:but|work|would|could|was|wasn'?t|didn'?t)|misjudg|miscalculat|"
            r"(?:saw|knew) (?:it|that|the (?:threat|idea|move|fork|pin|mate)))",
            re.I,
        ),
        MISCALCULATED,
    ),
    (
        re.compile(
            r"\b(?:didn'?t (?:see|notice|spot|consider|look|think)|missed|overlooked|"
            r"never (?:saw|considered|thought)|no idea|blind|forgot|"
            r"wasn'?t (?:looking|paying attention)|completely missed)\b",
            re.I,
        ),
        MISSED_THREAT,
    ),
)


def from_words(text: str) -> str | None:
    """The cause a player's own words name, or None when they do not say. "I missed it"
    is a threat missed; "I thought it worked" a line miscalculated; "I was rushing"
    time pressure."""
    for pattern, cause in _WORDS:
        if pattern.search(text):
            return cause
    return None
