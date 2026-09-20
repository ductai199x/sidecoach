"""The word on a sound move, written from the facts rather than by the model.

A best, great, excellent, good or book move that cost nothing gets a sentence: the
verdict in chess.com's vocabulary, then what the move does when the board says so --
the capture, the check, the castling, the promotion (describe_move), the tactic it
makes (the tagger's tag, described only when every square it names still holds the
piece it claims). Nothing else: no advice, no guess at what it allows, no numbers, no
question -- the rules the model was given for the same word. The opener never repeats
the last one.

Tai, 2026-09-20: "if I made the best move or great move, we can synthesize the response
without hitting the API". A routine call was 2.7k prompt tokens (mostly cached) and a
second's wait for "Nf3 was best, developing with a threat"; this is free and immediate,
and every word of it is true of the board.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import chess

from sidecoach.coach.facts import PIECE_NAMES, describe_tactic
from sidecoach.events import MotifTag

# The verdict, by the move's label; "{san}" is the move. What follows it, when the board
# gives something, is joined with " -- " or ", ".
OPENERS: dict[str, list[str]] = {
    "book": ["{san}: book", "{san}, still theory", "{san}: known territory"],
    "brilliant": ["{san}!! -- brilliant", "{san} -- what a find", "{san}: yes, that's the shot"],
    "great": ["{san}: great move", "{san} -- yes, the one move that works", "{san}! -- that's it"],
    "best": [
        "{san}: best",
        "{san} -- that's the move",
        "{san}, spot on",
        "{san}: exactly",
        "{san}: the engine's choice too",
    ],
    "excellent": ["{san}: excellent", "{san}: strong", "{san} -- very good", "{san}: nice"],
    "good": ["{san}: good", "{san}: fine", "{san}, solid", "{san} works", "{san}: sound"],
}
# The tags worth a clause, in the order one is chosen; the rest (a capture, a check, a
# quiet move) are said by the move itself.
TACTICS = (
    "fork",
    "skewer",
    "pin",
    "absolutePin",
    "discoveredAttack",
    "doubleCheck",
    "capturingDefender",
    "trappedPiece",
    "hangingPiece",
)


def what_it_does(board: chess.Board, move: chess.Move, tags: Sequence[MotifTag]) -> str | None:
    """A clause on the move, from the board: the mate, the check, the capture, the castling,
    the promotion, or the tactic a tag names when the squares bear it out; None when the
    move is quiet and no tag holds."""
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return "checkmate"
    for tag in tags:
        if tag.name in TACTICS and tag.squares:
            said = describe_tactic(after, tag.name, tag.squares)
            if said is not None:
                return said
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        return f"castling {side}"
    parts: list[str] = []
    if board.is_capture(move):
        taken = board.piece_at(move.to_square)
        what = (
            "the pawn en passant"
            if board.is_en_passant(move)
            else f"the {PIECE_NAMES[taken.piece_type]}"
            if taken
            else "a piece"
        )
        parts.append(f"taking {what} on {chess.square_name(move.to_square)}")
    if move.promotion:
        parts.append(f"promoting to a {PIECE_NAMES[move.promotion]}")
    if after.is_check():
        parts.append("with check")
    return ", ".join(parts) if parts else None


def verdict(
    board: chess.Board,
    move: chess.Move,
    label: str | None,
    tags: Sequence[MotifTag] = (),
    *,
    last: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """The sentence: the opener for the label (never `last`'s), then what the move does."""
    rng = rng or random.Random()
    san = board.san(move)
    openers = OPENERS.get(label or "", OPENERS["good"])
    # `last` is the previous sentence whole; its opener is what must not come again.
    choices = [o for o in openers if last is None or not last.startswith(o.format(san=san))]
    choices = choices or openers
    opener = rng.choice(choices).format(san=san)
    clause = what_it_does(board, move, tags)
    if clause is None:
        return opener + "."
    if clause == "checkmate":
        return f"{opener} -- checkmate."
    joiner = " -- " if clause.startswith("the ") else ", "
    return f"{opener}{joiner}{clause}."
