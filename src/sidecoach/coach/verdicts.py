"""The word on a sound move, written from the facts rather than by the model.

A best, great, excellent, good or book move that cost nothing gets a word of praise in
chess.com's vocabulary, then -- when the board shows one -- what the move teaches: the mate,
the tactic it makes (the tagger's tag, described only when every square it names still holds
the piece it claims), the mate it threatens, or the strategic idea behind it
(positional.best_move_idea, from the board's geometry), put as the principle to keep -- and
the principle only when it is the move's reason: never of a move that checks, mates along
the line, makes a tactic, or threatens something (`lesson`; Tai, 2026-09-27: "If it says
something it should be relevant"). Nothing else: no advice, no guess at what it allows, no
numbers, no question -- the rules the model was given for the same word.

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
import re
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
    "great": ["Great move", "Yes, you found it", "That's it"],
    "best": ["Best", "That's the move", "Spot on", "Exactly", "Just what the position wanted"],
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
    # Prophylaxis against a threat to take or to check: not their plan stopped before it
    # starts but their threat met ("prophylaxis" was said of Nc3 guarding e4 against Nxe4).
    "defend": ("their threat comes first", "first their threat, then your plan"),
}
# A move that threatens mate: the point is the mate, whatever else the move does.
MATE_THREAT = ("that threatens mate", "and now mate is threatened")
# The tags that say what a move is for when it is not a principle: the check, the mate along
# the line, the tactic it makes or starts. A move that carries one is a move with a concrete
# point, and a principle said of it was not the reason ("rooks belong on open files" of a
# rook check that led to mate, "passed pawns must be pushed" of the push that mated: Tai,
# 2026-09-27). The weaker geometric tags -- an interference, an x-ray, an exposed king --
# come with castling and rook lifts alike and do not count.
CONCRETE = frozenset(
    {
        "doubleCheck",
        "mate",
        "sacrifice",
        "deflection",
        "attraction",
        "clearance",
        "capturingDefender",
        "discoveredAttack",
        "fork",
        "skewer",
        "pin",
        "absolutePin",
        "attackingF2F7",
        "trappedPiece",
        "hangingPiece",
        "promotion",
        "underPromotion",
    }
)
# The kinds that are about a threat met: a move that saves a piece is their point.
_DEFENSIVE = ("rescue", "prophylaxis", "castle")
# The pawn moves whose principle is about pieces or a king to attack.
_PAWN_MOVES = ("break", "centre", "space", "pawnStorm")


# The terms of art in the wordings above: said to a player below 1400, the simulated
# 800-rated student listed "prophylaxis: stop their plan before pushing your own" among the
# things it did not understand (2026-09-26).
JARGON = re.compile(r"\b(?:prophylaxis|outpost|pawn storm)\b", re.I)


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


def _direct(board: chess.Board, threat: chess.Move) -> bool:
    """Their threat (read after a pass) takes something or gives check: a threat to meet, not
    a plan to forestall."""
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    return passed.is_capture(threat) or passed.gives_check(threat)


def threatens_mate(after: chess.Board) -> bool:
    """After a move that gives no check, the mover would mate in one were it their move
    again."""
    if after.is_check() or after.is_game_over():
        return False
    passed = after.copy(stack=False)
    passed.push(chess.Move.null())
    for reply in passed.legal_moves:
        passed.push(reply)
        mate = passed.is_checkmate()
        passed.pop()
        if mate:
            return True
    return False


def _threatens_piece(board: chess.Board, after: chess.Board, us: chess.Color) -> bool:
    """The move makes a threat to win a piece of theirs, a minor or more, that could not be
    won before it: the threat is its point."""
    for square, piece in after.piece_map().items():
        if piece.color == us or exchange.VALUE[piece.piece_type] < 3:
            continue
        if exchange.can_be_won(after, square) and not exchange.can_be_won(board, square):
            return True
    return False


def _lines_up(after: chess.Board, square: chess.Square, us: chess.Color) -> bool:
    """The piece that moved to `square`, a bishop, rook or queen, stands on a line with two of
    their pieces one behind the other, a minor or more each: a pin (the one behind worth
    more, or the king) or a skewer (the one in front worth more). Tagged or not, the line is
    the move's point: the annotator keeps a pin only when the line cashes it in. A pawn in
    front is a file's usual traffic, not a pin."""
    piece = after.piece_at(square)
    if piece is None or piece.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        return False
    steps = []
    if piece.piece_type != chess.BISHOP:
        steps += [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if piece.piece_type != chess.ROOK:
        steps += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for df, dr in steps:
        seen: list[chess.Piece] = []
        f, r = chess.square_file(square), chess.square_rank(square)
        while len(seen) < 2:
            f, r = f + df, r + dr
            if not (0 <= f < 8 and 0 <= r < 8):
                break
            other = after.piece_at(chess.square(f, r))
            if other is None:
                continue
            if other.color == us:
                break
            seen.append(other)
        if len(seen) < 2:
            continue
        front, back = (exchange.VALUE[p.piece_type] for p in seen)
        if front >= 3 and back >= 3 and front != back:
            return True
    return False


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
    tag names when the squares bear it out, the mate it threatens, or the idea behind the
    move as its principle (two wordings). None when the board shows none, or when its kind
    is in `taught` (said in one of the last few words).

    A principle only when it is the move's reason: never of a move that checks, mates along
    the line, makes or starts a tactic (`CONCRETE`), lines up on two of their pieces, or
    threatens mate or one of their pieces -- that is its point, and said or not, the
    principle beside it was noise (Tai, 2026-09-27: "If it says something it should be
    relevant")."""
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
    # A capture that wins something -- a recapture most often -- is about the material:
    # "that lets your bishop out" was said of bxc3 taking back a knight, "that threatens
    # mate" of Qxd4 taking a queen.
    won = exchange.exchange(board, move)
    if threatens_mate(after) and won < 3:
        return None if "mateThreat" in taught else ("mateThreat", list(MATE_THREAT))
    us = board.turn
    if (
        after.is_check()
        or any(tag.name in CONCRETE or tag.name.startswith("mateIn") for tag in tags)
        or _lines_up(after, move.to_square, us)
        or _threatens_piece(board, after, us)
        or won > 0
    ):
        return None
    threat = _threat(board, threats)
    idea = positional.best_move_idea(
        board,
        move,
        us,
        phase=phase,
        worst=positional.worst_piece_square(board, us),
        threat=threat,
    )
    if idea is None or idea.kind not in PRINCIPLES:
        return None
    if idea.kind not in _DEFENSIVE and any(tag.name == "defensiveMove" for tag in tags):
        return None  # it saves a piece: that is its point, not the rook's file
    kind = idea.kind
    if kind == "prophylaxis" and threat is not None and _direct(board, threat):
        kind = "defend"
    if kind in taught:
        return None
    if kind in _PAWN_MOVES and all(
        p.piece_type in (chess.KING, chess.PAWN) for p in board.piece_map().values()
    ):
        return None  # a pawn ending: no pieces for the lines to open, no king to storm
    if kind in ("centre", "space", "centralise"):
        # A principle with nothing on the board behind it is no lesson ("the fight is for
        # the centre" after every pawn move, Tai, 2026-09-24): only what the move opens.
        opened = shut_in(board, move)
        if opened is None:
            return None
        home, _ = opened
        name = chess.piece_name(board.piece_type_at(home) or chess.PAWN)
        return kind, [f"that lets your {name} out"]
    return kind, list(PRINCIPLES[kind])


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
    plain: bool = False,
    strong: bool = False,
) -> tuple[str, str | None]:
    """The sentence -- the praise for the label, then what the move teaches -- and the kind
    of lesson it taught, None when it is praise alone. With `bags` (the live coach's) every
    wording is drawn from its bag, each once before any comes again (`coach/bag.py`);
    without, a `choice` that is never `last`'s praise. The praise is drawn before the
    wording, so the phone's port draws the same. `plain` leaves out the wordings that name
    a term of art (a player below 1400: `JARGON`); `strong` (1800 and up) the principles
    -- "rooks belong on open files" was what the strong simulated students most often said
    they already knew -- keeping the tactics, which are the board's own."""
    rng = rng or random.Random()
    found = lesson(board, move, tags, phase=phase, threats=threats, taught=taught)
    if strong and found is not None and found[0] in PRINCIPLES:
        found = None
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
    if plain:
        clauses = [c for c in clauses if not JARGON.search(c)] or clauses
    if bags is not None:
        clause = bags.draw(f"clause:{kind}", clauses)
    else:
        clause = rng.choice(clauses) if len(clauses) > 1 else clauses[0]
    return f"{opener} -- {clause}.", kind
