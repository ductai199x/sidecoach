"""The word on a sound move, written from the facts rather than by the model.

A best, great, excellent, good or book move that cost nothing gets a word of praise in
chess.com's vocabulary, then -- when the board shows one -- what the move teaches: the mate,
the tactic it makes (the tagger's tag, described only when every square it names still holds
the piece it claims), or the strategic idea behind it (positional.best_move_idea, from the
board's geometry), put as the principle to keep. Nothing else: no advice, no guess at what it
allows, no numbers, no question -- the rules the model was given for the same word.

Never the move itself, nor what it plainly does -- the capture, the check, the castling: the
player has just played it and sees it on the board. Tai, 2026-09-23: "the coach shouldn't be
starting the message with the move either since that's redundant ... either say something
instructive or don't say anything at all is okay or just 'That's a good move'". The praise
never repeats the last one, and a lesson taught in one of the last few words waits its turn.

Tai, 2026-09-20: "if I made the best move or great move, we can synthesize the response
without hitting the API". A routine call was 2.7k prompt tokens (mostly cached) and a
second's wait; this is free and immediate, and every word of it is true of the board.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import chess

from sidecoach.analysis import exchange, positional
from sidecoach.coach.bag import Bags
from sidecoach.coach.facts import describe_tactic
from sidecoach.coach.situation import shut_in
from sidecoach.events import MotifTag

# The praise, by the move's label. What the move teaches, when the board gives something,
# follows it after " -- ".
OPENERS: dict[str, list[str]] = {
    "book": ["Book", "Still theory", "Known territory"],
    "brilliant": ["Brilliant", "What a find", "Yes, that's the shot"],
    "great": ["Great move", "Yes, the one move that works", "That's it"],
    "best": ["Best", "That's the move", "Spot on", "Exactly", "The engine agrees"],
    "excellent": ["Excellent", "Strong", "Very good", "Nice"],
    "good": ["Good", "Fine", "Solid", "Sound"],
}
# A mate needs no label: the game is won.
MATE = ["Checkmate -- well played.", "And that's mate. Well played.", "Mate. Nicely done."]
# The tags worth a lesson, in the order one is chosen; the rest (a capture, a check, a quiet
# move) the player sees on the board.
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
# The idea behind a move (positional.best_move_idea's kinds) as the principle it follows:
# what the player should keep, not what the move did. Two wordings each, taken in turn.
PRINCIPLES: dict[str, tuple[str, str]] = {
    "castle": (
        "king safety first: a castled king is much harder to attack",
        "the king tucked away, and a rook on its way to the centre",
    ),
    "rescue": (
        "a piece in danger comes first",
        "getting the attacked piece to safety before anything else",
    ),
    "prophylaxis": (
        "it takes away your opponent's idea before it starts",
        "prophylaxis: stop their plan before pushing your own",
    ),
    "activateKing": (
        "in the endgame the king is a fighting piece, and it belongs in the centre",
        "in the endgame the king walks to the centre and fights",
    ),
    "develop": (
        "every piece you bring out is one more in the fight",
        "develop first, attack later",
    ),
    "outpostMove": (
        "a knight where no pawn can chase it is a strong piece",
        "an outpost: no enemy pawn can drive that knight away",
    ),
    "centralise": (
        "a piece in the centre reaches both sides of the board",
        "pieces work best from the middle",
    ),
    "takeFile": ("rooks belong on open files", "an open file is a road for your rook"),
    "rookBehindPasser": (
        "rooks belong behind passed pawns",
        "behind the passed pawn, the rook gains room with every step the pawn takes",
    ),
    "break": (
        "a pawn break opens lines for your pieces",
        "challenging their pawns opens the game for your pieces",
    ),
    "centre": (
        "pawns in the centre give your pieces room",
        "the fight is for the centre, and you joined it",
    ),
    "pawnStorm": (
        "pawns thrown at their king pry open lines for the attack",
        "the pawn storm: open lines toward their king",
    ),
    "pushPasser": ("passed pawns must be pushed", "a passed pawn gets stronger as it advances"),
    "activate": (
        "improving your worst piece is rarely wrong",
        "your most passive piece joins the game",
    ),
    "space": (
        "space on the flank cramps their pieces",
        "more space, less room for their pieces",
    ),
    "simplify": (
        "when you are ahead, trades bring the win closer",
        "ahead in material, every trade helps you",
    ),
}


def _threat(board: chess.Board, threats: Sequence[str]) -> chess.Move | None:
    """The opponent's first threat (the fact sheet's SAN, read after a pass), or None."""
    if not threats or board.is_check():
        return None
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    try:
        return passed.parse_san(threats[0])
    except ValueError:
        return None


def _pin_wins(after: chess.Board, tag: MotifTag) -> bool:
    """The pinned piece can be won as the board stands: a pin worth a word after the move.
    A pawn pinned to its king with nothing to take it is true and teaches nothing."""
    if len(tag.squares) < 2:
        return False
    return exchange.can_be_won(after, chess.parse_square(tag.squares[1]))


def lesson(
    board: chess.Board,
    move: chess.Move,
    tags: Sequence[MotifTag] = (),
    *,
    phase: str = "middlegame",
    threats: Sequence[str] = (),
    taught: Sequence[str] = (),
) -> tuple[str, list[str]] | None:
    """What a sound move teaches, as its kind and the ways to say it: the mate, the tactic a
    tag names when the squares bear it out, or the idea behind the move as its principle
    (two wordings). None when the board shows none, or when its kind is in `taught` (said
    in one of the last few words)."""
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        return "mate", ["checkmate"]
    for tag in tags:
        if tag.name in TACTICS and tag.squares and tag.name not in taught:
            if tag.name in ("pin", "absolutePin") and not _pin_wins(after, tag):
                continue
            said = describe_tactic(after, tag.name, tag.squares)
            if said is not None:
                return tag.name, [said]
    us = board.turn
    idea = positional.best_move_idea(
        board,
        move,
        us,
        phase=phase,
        worst=positional.worst_piece_square(board, us),
        threat=_threat(board, threats),
    )
    if idea is None or idea.kind not in PRINCIPLES or idea.kind in taught:
        return None
    if idea.kind in ("centre", "space", "centralise"):
        # A principle with nothing on the board behind it is no lesson ("the fight is for
        # the centre" after every pawn move, Tai, 2026-09-24): only what the move opens.
        opened = shut_in(board, move)
        if opened is None:
            return None
        home, _ = opened
        name = chess.piece_name(board.piece_type_at(home) or chess.PAWN)
        return idea.kind, [f"that lets your {name} out"]
    return idea.kind, list(PRINCIPLES[idea.kind])


def verdict(
    board: chess.Board,
    move: chess.Move,
    label: str | None,
    tags: Sequence[MotifTag] = (),
    *,
    phase: str = "middlegame",
    threats: Sequence[str] = (),
    taught: Sequence[str] = (),
    last: str | None = None,
    rng: random.Random | None = None,
    bags: Bags | None = None,
) -> tuple[str, str | None]:
    """The sentence -- the praise for the label, then what the move teaches -- and the kind
    of lesson it taught, None when it is praise alone. With `bags` (the live coach's) every
    wording is drawn from its bag, each once before any comes again (`coach/bag.py`);
    without, a `choice` that is never `last`'s praise. The praise is drawn before the
    wording, so the phone's port draws the same."""
    rng = rng or random.Random()
    found = lesson(board, move, tags, phase=phase, threats=threats, taught=taught)
    if found is not None and found[0] == "mate":
        if bags is not None:
            return bags.draw("mate", MATE), "mate"
        return rng.choice([m for m in MATE if m != last] or MATE), "mate"
    openers = OPENERS.get(label or "", OPENERS["good"])
    if bags is not None:
        opener = bags.draw(f"praise:{label or 'good'}", openers)
    else:
        # `last` is the previous sentence whole; its praise is what must not come again.
        choices = [o for o in openers if last is None or not last.startswith(o)]
        opener = rng.choice(choices or openers)
    if found is None:
        return opener + ".", None
    kind, clauses = found
    if bags is not None:
        clause = bags.draw(f"clause:{kind}", clauses)
    else:
        clause = rng.choice(clauses) if len(clauses) > 1 else clauses[0]
    return f"{opener} -- {clause}.", kind
