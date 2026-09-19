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

from collections.abc import Sequence

import chess

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
    nobody can take back."""
    if not board.is_capture(reply) or reply not in board.legal_moves:
        return False
    victim = board.piece_at(reply.to_square)
    attacker = board.piece_at(reply.from_square)
    if attacker is None:
        return False
    taken = _VALUE.get(victim.piece_type, 1) if victim is not None else 1  # en passant
    if taken > _VALUE.get(attacker.piece_type, 0):
        return True
    after = board.copy(stack=False)
    after.push(reply)
    return not after.is_attacked_by(after.turn, reply.to_square)


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
) -> tuple[str, str | None]:
    """The cause and the motif of a mistake.

    `before` is the position the player faced and `move` what they played; `played_tags`
    are the tags of that move, `available_tags` those of the position they faced (what
    they could have played), `allowed_tags` those of the position after the move (what
    the opponent now has); `reply` is the opponent's best answer and `reply_mate` its
    mate distance; `left_book` says the move was the first out of theory; `clock_ms` the
    player's clock when the move was made.
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
    if before.is_capture(move) or before.gives_check(move):
        return MISCALCULATED, available
    if available is not None:
        return NOT_CONSIDERED, available
    if phase == "endgame":
        return ENDGAME, None
    if phase == "opening" or left_book:
        return OPENING_GAP, None
    return POSITIONAL, None
