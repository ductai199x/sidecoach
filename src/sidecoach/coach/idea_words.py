"""The idea of the best line (`coach/idea.py`) in the coach's words, one field per rung.

Every sentence here is built from the board and the engine's line, and says something true
about this position that helps the player find the answer without handing it over: the
goal and the pattern with why it works (the idea rung), then the target and the piece (the
focus rung), then the move played out (the move rung). A sentence that could be said of any
position is not written here (Tai, 2026-09-23: "we want instructive comments, not generic").
"""

from __future__ import annotations

import zlib
from collections.abc import Callable, Sequence

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach import idea as idea_of
from sidecoach.coach.idea import Idea, origin
from sidecoach.events import Analysis

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
# Counts in words to twenty: a digit is a figure the coach may not quote.
_NUMBER = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}  # fmt: skip
# Counts to thirty in words: a queen's squares, a pawn's way to queening.
_WORDS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
)


def in_words(n: int) -> str:
    """A count as a coach says it: "nine", "twenty-three"."""
    if 0 <= n < len(_WORDS):
        return _WORDS[n]
    if 20 < n < 30:
        return f"twenty-{_WORDS[n - 20]}"
    return str(n)


# The longest mate whose line the move rung reads out whole; past it, its first moves.
MATE_READ = 5
MATE_SHOWN = 5  # the plies of a longer mate read out, the first move with them
# The furthest ply of a win read out move by move: the player's third move.
LATER_READ = 4

# The pattern's name, as the coach names it.
PATTERN_NAMES: dict[str, str] = {
    "smotheredMate": "a smothered mate",
    "backRankMate": "a back-rank mate",
    "anastasiaMate": "an Anastasia mate",
    "arabianMate": "an Arabian mate",
    "hookMate": "a hook mate",
    "bodenMate": "Boden's mate",
    "doubleBishopMate": "a mate by two bishops",
    "dovetailMate": "a dovetail mate",
    "vukovicMate": "Vukovic's mate",
    "killBoxMate": "a box mate",
    "swallowstailMate": "a swallow's tail mate",
    "epauletteMate": "an epaulette mate",
    "triangleMate": "a triangle mate",
    "blindSwineMate": "a blind swine mate",
    "cornerMate": "a corner mate",
    "operaMate": "an opera mate",
    "pillsburysMate": "Pillsbury's mate",
    "morphysMate": "Morphy's mate",
    "balestraMate": "a Balestra mate",
    "mate": "a mating attack",
    "doubleCheck": "a double check",
    "fork": "a fork",
    "skewer": "a skewer",
    "discoveredAttack": "a discovered attack",
    "absolutePin": "a pin",
    "pin": "a pin",
    "trappedPiece": "a trapped piece",
    "capturingDefender": "removing the defender",
    "deflection": "a deflection",
    "attraction": "a decoy",
    "hangingPiece": "a loose piece",
    "overloading": "an overloaded defender",
}

# Why each pattern works, in general terms, naming none of the player's own pieces.
PATTERN_WHY: dict[str, str] = {
    "smotheredMate": "their king is hemmed in by its own pieces, so a check it can neither "
    "block nor step away from ends it",
    "backRankMate": "their king is shut in behind its own pawns, and a check along the back "
    "rank has nowhere to be escaped",
    "anastasiaMate": "one piece takes the king's escape squares while another mates it along "
    "the edge of the board",
    "arabianMate": "with the king in a corner, one piece guards its way out and another mates "
    "right beside it",
    "hookMate": "a protected piece next to the king cuts off its escape while the mate comes "
    "from close by",
    "bodenMate": "two long diagonals cross at the king, and its own pieces block its way out",
    "doubleBishopMate": "two diagonals hit the king and the squares it would run to",
    "dovetailMate": "the mating piece stands right next to the king, its retreat blocked by "
    "its own pieces",
    "vukovicMate": "one piece mates with the support of another beside the king on the edge",
    "killBoxMate": "the king is caught in a box of squares no escape leads out of",
    "swallowstailMate": "the check comes from right in front of the king, guarded, and the two "
    "squares behind it are blocked by its own pieces",
    "epauletteMate": "the king's own pieces stand on both sides of it like epaulettes, so a "
    "check along the line leaves it nowhere to go",
    "triangleMate": "two pieces stand next to the king in a triangle, each guarding the "
    "other and every square around it",
    "blindSwineMate": "two pieces side by side deep in their camp sweep the king's squares, "
    "each protecting the other",
    "cornerMate": "the king is shut in the corner, and a check it cannot step out of or "
    "block finishes it",
    "operaMate": "the check comes from right beside the king, guarded from a distance, "
    "and the king's own piece takes away its escape",
    "pillsburysMate": "the check comes down a file while the square the king would run to is "
    "covered from a diagonal",
    "morphysMate": "the check comes down a long diagonal while a file shuts off the king's escape",
    "balestraMate": "the check comes from a distance and every square the king could reach "
    "is already covered",
    "doubleCheck": "two pieces give check at once, so the king must move and they cannot "
    "answer anything else",
    "fork": "one piece attacks two of theirs at once, and they can save only one",
    "skewer": "a valuable piece is attacked on a line, and when it steps aside the piece "
    "behind it falls",
    "discoveredAttack": "one piece steps off a line with a threat of its own and uncovers an "
    "attack from the piece behind it: two threats at once",
    "absolutePin": "one of their pieces cannot move without exposing the king behind it, so "
    "it can be attacked again and again",
    "pin": "one of their pieces cannot move without losing what stands behind it, so it can "
    "be attacked again",
    "trappedPiece": "one of their pieces has no safe square left, so an attack on it wins it",
    "capturingDefender": "the piece that guards the target is taken first, and the target falls",
    "deflection": "a defender is pulled away from what it guards, and what it guarded falls",
    "attraction": "a piece is lured onto a square where it can be hit",
    "hangingPiece": "something of theirs is not defended at all",
    "overloading": "one piece with two jobs cannot do both",
}

# What each preparation does, said of the moves before the blow.
PREPARATION_WHY: dict[str, str] = {
    "sacrifice": "it starts by giving up material",
    "attraction": "their king or a defender is lured onto the wrong square first",
    "deflection": "a defender is pulled away first",
    "capturingDefender": "a defender is taken first",
    "clearance": "a line or a square is cleared first",
}

# What a preparation has done, said once the blow comes.
PREPARED: dict[str, str] = {
    "sacrifice": "you have given up material to open the way",
    "attraction": "a piece has been lured onto the wrong square",
    "deflection": "a defender has been pulled away",
    "capturingDefender": "a defender has been taken",
    "clearance": "a line has been cleared",
}
# The same, when it is the first move itself.
STARTS: dict[str, str] = {
    PREPARED["sacrifice"]: "by giving up material to open the way",
    PREPARED["attraction"]: "by luring a piece onto the wrong square",
    PREPARED["deflection"]: "by pulling a defender away",
    PREPARED["capturingDefender"]: "by taking a defender",
    PREPARED["clearance"]: "by clearing a line",
    "a trade has cleared the way": "with a trade",
}

# The question that sets the player looking, per pattern.
PATTERN_QUESTION: dict[str, str] = {
    "fork": "Which two of their pieces could one of yours hit at once, and from where?",
    "skewer": "Which line has two of their pieces on it, the more valuable in front?",
    "discoveredAttack": "Which of your pieces is standing in front of another's line?",
    "doubleCheck": "Which move uncovers a check while giving one itself?",
    "absolutePin": "Which of their pieces could you pin to their king -- and what can it "
    "no longer do then?",
    "pin": "Which of their pieces stands in front of a bigger one, on a line one of yours "
    "could take?",
    "trappedPiece": "Which of their pieces is short of squares?",
    "capturingDefender": "Which of their pieces is doing the guarding?",
    "deflection": "Which of their pieces is guarding two things at once?",
    "attraction": "Onto which square would you like to drag one of their pieces?",
    "hangingPiece": "Which of their pieces has no defender?",
}

# The plan's why, keyed by the positional idea's kind (`analysis/positional.py`).
PLAN_WHY: dict[str, str] = {
    "castle": "a king left in the centre gets caught when the position opens, and castling "
    "also brings a rook into play",
    "rescue": "a piece left attacked is simply lost, so it moves first -- ideally to a square "
    "where it does something",
    "prophylaxis": "their idea is worth stopping before it starts",
    "activateKing": "in an endgame the king is a fighting piece, not one to hide",
    "develop": "pieces still at home take no part in the game",
    "outpostMove": "a piece on a square no enemy pawn can attack stays there for good",
    "centralise": "a piece in the centre reaches both sides of the board",
    "takeFile": "a rook on an open file gets into their position",
    "rookBehindPasser": "a rook behind a passed pawn pushes it and grows stronger as it goes",
    "break": "a pawn break opens lines for your pieces",
    "centre": "the centre is where pieces have the most room",
    "pawnStorm": "pawns rolling at their king open lines toward it",
    "pushPasser": "a passed pawn is there to be pushed",
    "passerPawnEnding": "in a pawn ending the first passed pawn usually decides, and the "
    "king walks it home",
    "activate": "your least active piece is the one worth improving",
    "space": "more space gives your pieces room and cramps theirs",
    "simplify": "when you are ahead, trades make the extra material count",
}
PLAN_QUESTION: dict[str, str] = {
    "castle": "Is there any reason to keep your king in the centre?",
    "rescue": "Which of your pieces is attacked, and where is it safe and useful?",
    "prophylaxis": "If you passed, what would your opponent play?",
    "activateKing": "Where would your king do the most good?",
    "develop": "Which of your pieces is still at home, and where does it belong?",
    "outpostMove": "Which square in their half can no pawn of theirs ever attack?",
    "centralise": "Which of your pieces could reach the centre?",
    "takeFile": "Which file has no pawns on it?",
    "rookBehindPasser": "Where does your rook best support the passed pawn?",
    "break": "Which pawn move would open lines toward their weaknesses?",
    "centre": "What would make the centre yours?",
    "pawnStorm": "Which pawn can start rolling toward their king?",
    "pushPasser": "What stops your passed pawn from advancing?",
    "activate": "Which of your pieces has no job right now?",
    "space": "Where can you take space without leaving a weakness behind?",
    "simplify": "Which trade makes your extra material count?",
    "passerPawnEnding": "Which pawn move makes you a passed pawn -- and can their king catch it?",
}


def _sq(square: chess.Square | None) -> str:
    return chess.square_name(square) if square is not None else ""


def _named(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    return f"{PIECE_NAMES[piece.piece_type]} on {chess.square_name(square)}" if piece else ""


def pattern(idea: Idea) -> str | None:
    """The pattern's name: "a fork", "a smothered mate", "a mating attack"."""
    if idea.blow is not None:
        return PATTERN_NAMES.get(idea.blow.name)
    if idea.goal == "mate":
        return PATTERN_NAMES["mate"]
    return None


def goal(idea: Idea) -> str | None:
    """What the best line is for, naming no square, piece or move."""
    if idea.goal == "mate":
        n = idea.mate_in or 0
        if n == 1:
            return "you can mate in one move"
        return f"you can force mate in {_NUMBER.get(n, str(n))}" if n <= 5 else "you can force mate"
    if idea.goal == "draw":
        return {
            "perpetual": "you can save the game: check after check, and they cannot escape",
            "stalemate": "you can save the game with a stalemate",
        }.get(idea.draw or "", "you can hold the game: the same position comes round again")
    if idea.goal == "win" and idea.back:
        # Taking back what they just took: never "you can win a knight" for a trade.
        what = idea.won or "material"
        if idea.score is not None and idea.score < 0.4:
            return f"you can win back {what}, though you stay worse"
        return f"you can win back {what}"
    if idea.goal == "win" and idea.level:
        # Winning it back, not winning: the engine calls the game level after it.
        what = idea.won or "material"
        if idea.down and idea.score is not None and idea.score < 0.4:
            return f"you can win back {what}, though you stay worse"
        if idea.down:
            return f"you can win back {what}, and that levels the game"
        if idea.score is not None and idea.score < 0.4:
            return f"you can win {what}, though you stay worse"
        return f"you can win {what}, and that holds the game level"
    if idea.goal == "win":
        return f"you can win {idea.won}" if idea.won else "you can win material by force"
    if idea.goal == "promote":
        return "you can push a pawn through to promotion"
    if idea.goal == "attack":
        return (
            "you have a winning attack on their king"
            if idea.winning
            else "you have a strong attack on their king"
        )
    if idea.goal == "convert":
        return f"you can trade into {_ending(idea)}"
    if idea.goal == "king":
        return "you are winning, and your king leads the way"
    if idea.goal == "breakthrough":
        return "you can break through and make a pawn that runs"
    if idea.goal == "combination":
        return "you can give up material and win it back with a tactic"
    if idea.goal == "defend":
        lead = "you are winning, but " if idea.winning else ""
        if idea.prepared:
            what = "mate" if idea.threatened == "mate" else "to win material"
            return f"{lead}your opponent is preparing {what}: stop it before it comes"
        if idea.threatened == "mate":
            return f"{lead}your opponent threatens mate: meet it first"
        return f"{lead}your opponent threatens to win material: meet it first"
    if idea.goal == "plan" and idea.winning:
        # Won, with nothing to take and no mate: said so, then how -- never only "a quiet
        # move is wanted" in a position the engine calls won (Lichess 8KhlA, y6QH7).
        return "you are winning: no trick is needed here, only good technique"
    return None


# The first hint's first words when the best line is a tactic: said as one to find (Tai,
# 2026-09-25: "if there's a tactic that results in a material or big positional gain, the
# coach should mention that there is tactics to be exploited in the first hint"). Three ways
# of saying each, chosen by the position (`opener`), so the same words do not open every hint.
_TACTIC_LEADS = (
    "there is a tactic here that wins {what}",
    "you have a tactic that wins {what}",
    "this position has a tactic that wins {what}",
)
_MATE_ONE_LEADS = (
    "there is a mate in one here",
    "you can mate in one move",
    "you have a mate in one",
)
_MATE_LEADS = (
    "there is a forced mate in {n} here",
    "you can force mate in {n}",
    "you have a forced mate in {n}",
)
_MATE_LONG_LEADS = ("there is a forced mate here", "you can force mate", "you have a forced mate")


def opener(
    idea: Idea,
    goal_words: str | None,
    *,
    tactic: bool,
    seed: int,
    pick: Callable[[str, Sequence[str]], str] | None = None,
) -> str | None:
    """The first hint's lead: what the best line is for (`goal_words`, as `goal` or the
    caller made it), said as a tactic when `tactic` -- a line that wins material by force,
    not a piece simply left hanging or a recapture -- and a mate as one to find. `pick` (the
    live coach's bags, `coach/bag.py`) draws the words, each once before any comes again;
    without it `seed` picks them (`opener_seed`), the same every time for a position."""

    def choose(key: str, leads: Sequence[str]) -> str:
        return pick(key, leads) if pick is not None else leads[seed % len(leads)]

    if idea.goal == "mate" and goal_words is not None and "mate" in goal_words:
        n = idea.mate_in or 0
        if n == 1:
            return choose("lead:mate-one", _MATE_ONE_LEADS)
        if n <= 5:
            lead = choose("lead:mate", _MATE_LEADS)
            return lead.format(n=_NUMBER.get(n, str(n)))
        return choose("lead:mate-long", _MATE_LONG_LEADS)
    if not tactic or goal_words is None:
        return goal_words
    if goal_words.startswith("you can win "):
        what = goal_words[len("you can win ") :]
        if what == "material by force":
            what = "material"
        return choose("lead:tactic", _TACTIC_LEADS).format(what=what)
    if idea.goal == "promote":
        return "there is a tactic here that gets a pawn through to promotion"
    if idea.goal == "combination":
        score = idea.score or 0.0
        if score >= 0.8:
            return (
                "there is a tactic here that gives up material, wins it back and leaves you winning"
            )
        if score >= 0.62:
            return (
                "there is a tactic here that gives up material, wins it back and leaves you "
                "clearly better"
            )
        return "there is a tactic here that gives up material and wins it back"
    return goal_words


def opener_seed(board: chess.Board) -> int:
    """What picks the opener's words: the position's placement and side to move, the same
    on every platform (the move counters are not read off a screen)."""
    text = f"{board.board_fen()} {'w' if board.turn == chess.WHITE else 'b'}"
    return zlib.crc32(text.encode("ascii"))


def lesson(text: str | None, idea: Idea) -> str | None:
    """A pattern's lesson (`mechanism`, a defence's) said of what the hint just named:
    "that's a fork: ..." after the two pieces one move hits; of a mate, an attack or a draw
    still to come, "it's a back-rank mate: ..."; "it takes two sacrifices: ..."."""
    if not text:
        return text
    head = text.split(":", 1)[0]
    if head.endswith(" sacrifices"):
        return f"it takes {text}"
    lead = "it is" if idea.goal in ("mate", "attack", "draw") else "that is"
    return f"{lead} {text}"


def mechanism(idea: Idea) -> str | None:
    """The pattern and why it works here, square-free: the idea rung's teaching -- and what
    else makes it work: a guard that is pinned, a quiet first move, an in-between move."""
    text = _mechanism(idea)
    if text is None or idea.goal not in ("mate", "win", "promote", "combination", "attack"):
        return text
    extra: list[str] = []
    if idea.also is not None and idea.also in PATTERN_NAMES:
        extra.append(f"There is {PATTERN_NAMES[idea.also]} in it too")
    if idea.pinned_guard is not None:
        extra.append("It works because the piece that could take is pinned")
    if idea.starts_with == "quiet" and not idea.sacrifices:
        extra.append(
            "It starts with a quiet move: no check and no capture, but a threat they cannot meet"
        )
    if idea.intermezzo:
        extra.append("On the way comes an in-between move: a check before taking back")
    return ". ".join([text, *extra]) if extra else text


def _mechanism(idea: Idea) -> str | None:
    """The pattern and why it works here, square-free: the idea rung's teaching."""
    name = idea.blow.name if idea.blow is not None else None
    parts: list[str] = []
    if name in PATTERN_WHY:
        parts.append(f"{PATTERN_NAMES[name]}: {PATTERN_WHY[name]}")
    elif idea.goal == "mate":
        parts.append("a mating attack: their king runs out of squares")
    elif idea.goal == "promote":
        parts.append("a pawn on the run: once its way is clear it is too far advanced to be caught")
    elif idea.goal == "attack":
        parts.append(
            "an attack on the king: check after check drives it into the open, where your "
            "pieces can pick things off"
        )
    elif idea.goal == "convert":
        parts.append(
            "trading down: when you are winning, every trade takes away their chances, and "
            + (
                "a pawn ending is counted, not played -- your king and pawns get there first"
                if idea.ending == "pawn"
                else "the fewer pieces, the fewer tricks"
            )
        )
    elif idea.goal == "king" and idea.opposition:
        parts.append(
            "the opposition: with the kings face to face and one square between them, the "
            "side that must move gives way"
        )
    elif idea.goal == "king":
        parts.append("king activity: in an endgame the king is a fighting piece, not one to hide")
    elif idea.goal == "breakthrough":
        parts.append("a breakthrough: a pawn given up opens the way, and another pawn runs through")
    elif idea.goal == "draw" and idea.draw == "perpetual":
        parts.append(
            "a perpetual check: their king has nowhere to hide from the checks, and a "
            "position that repeats three times is a draw"
        )
    elif idea.goal == "draw" and idea.draw == "stalemate":
        parts.append("a stalemate: if they have no legal move and are not in check, it is a draw")
    elif idea.sacrifices and idea.goal in ("win", "combination"):
        # No pattern to name, but the sacrifice is the idea: said as the idea.
        count = len(idea.sacrifices)
        what = "a sacrifice" if count == 1 else f"{_NUMBER.get(count, str(count))} sacrifices"
        opened = "you give up material, and what it opens up wins more back"
        if idea.sacrifices[0][0] == 0:
            return f"{what}: {opened}. It starts straight away"
        ordinal = _ORDINAL.get(idea.sacrifices[0][0] // 2 + 1, "next")
        return f"{what}: {opened}. The first comes on your {ordinal} move"
    if not parts:
        return None
    text = parts[0]
    # When it comes, and what it costs: the sacrifices on the way say more than any other
    # preparation (Tai, 2026-09-23: "the coach is completely blind to" them).
    given = idea.sacrifices
    if given:
        count = len(given)
        ordinal = _ORDINAL.get(given[0][0] // 2 + 1, "next")
        sacs = "a sacrifice" if count == 1 else f"{_NUMBER.get(count, str(count))} sacrifices"
        if idea.blow_at > 1 and name in PATTERN_WHY:
            when = _ORDINAL.get(idea.blow_at, "next")
            return f"{text}. It comes on your {when} move, after {sacs}"
        if given[0][0] == 0:
            if count == 1:
                return f"{text}. It starts with a sacrifice: you give up material to open the way"
            return f"{text}. It takes {sacs}, the first straight away"
        return f"{text}. It takes {sacs}, the first on your {ordinal} move"
    # When it comes, and what prepares it: a second sentence. A sacrifice only as the line
    # counts one (`idea._sacrifices`), never on the tagger's looser word.
    before = None
    for prep in idea.preparation:
        if prep == "sacrifice":
            continue
        if prep in PREPARED and prep != name:
            before = PREPARED[prep]
            break
    if before is None:
        if idea.starts_with == "sacrifice":
            before = PREPARED["sacrifice"]
        elif idea.starts_with == "capture" and idea.blow_at > 1:
            before = "a trade has cleared the way"
    if (
        idea.goal in ("mate", "win", "promote", "attack")
        and idea.blow_at > 1
        and name in PATTERN_WHY
    ):
        when = f"It comes on your {_ORDINAL.get(idea.blow_at, 'next')} move"
        text += f". {when}, once {before}" if before else f". {when}"
    elif before is not None and before in STARTS:
        text += f". It starts {STARTS[before]}"
    elif (
        idea.goal in ("mate", "attack")
        and idea.starts_with == "check"
        and (idea.goal == "attack" or (idea.mate_in or 0) > 1)
    ):
        text += ". It starts with a check, so they have no time for anything else"
    return text


def question(idea: Idea, plan_kind: str | None) -> str:
    """The idea rung's question: the one that sets the player looking the right way."""
    name = idea.blow.name if idea.blow is not None else None
    if idea.goal == "convert":
        return "Which trade leaves them nothing to fight with?"
    if idea.goal == "king":
        if idea.opposition:
            return "Where does your king face theirs, so that they must give way?"
        return "Where does your king want to go, and what can it attack there?"
    if idea.goal == "breakthrough":
        return "Which pawn can you give up so that another gets through?"
    if idea.goal == "draw":
        if idea.draw == "perpetual":
            return "Which checks can you keep giving, however their king runs?"
        if idea.draw == "stalemate":
            return "What if your king had no move at all -- and you were not in check?"
        return "Which move keeps the position going round in a circle?"
    if idea.goal == "mate":
        if idea.mate_in == 1:
            return "Which check leaves their king no square at all?"
        if idea.starts_with == "sacrifice" or idea.sacrifices:
            return (
                "Go through your checks, the ones that give up material too: after which "
                "does their king run out of squares?"
            )
        return "Go through your checks one by one: after each, where can their king go?"
    if idea.goal in ("win", "promote") and name in PATTERN_QUESTION:
        return PATTERN_QUESTION[name]
    if idea.goal == "combination" and name in PATTERN_QUESTION:
        return "What could you give up to set it up? " + PATTERN_QUESTION[name]
    if idea.goal == "attack":
        if idea.starts_with == "sacrifice" or idea.sacrifices:
            return "Which sacrifice tears open the cover around their king?"
        return "Which check drives their king into the open?"
    if idea.goal == "win":
        return "Which of their pieces is loose or attacked more times than it is defended?"
    if idea.goal == "promote":
        return "What stands in your pawn's way, and how do you remove it?"
    if idea.goal == "defend":
        return (
            "If you passed, what would they play -- and how do you stop it without losing ground?"
        )
    if plan_kind and plan_kind in PLAN_QUESTION:
        return PLAN_QUESTION[plan_kind]
    if idea.starts_with in ("check", "capture", "promotion"):
        # A forcing answer with no plan behind it: which forcing move, not which idle piece.
        return "Go through your checks and captures: which leaves them the fewest good replies?"
    return "Which of your pieces has no job right now?"


NUDGE: dict[str, str] = {
    "fork": "Which square hits both?",
    "skewer": "Which check or attack lines them up?",
    "discoveredAttack": "Which move uncovers the line with a threat of its own?",
    "doubleCheck": "Which move gives check twice?",
    "absolutePin": "Which move pins it?",
    "pin": "Which move pins it?",
    "trappedPiece": "How do you attack it?",
    "hangingPiece": "How do you win it safely?",
    "capturingDefender": "Which capture removes the guard?",
    "deflection": "How do you pull it away?",
    "attraction": "What drags it there?",
}


def nudge(idea: Idea) -> str | None:
    """The focus rung's question: the one that leads from the target to the move."""
    name = idea.blow.name if idea.blow is not None else None
    if idea.goal == "mate":
        return "Which check is it?" if idea.mate_in == 1 else "Which check comes first?"
    if idea.goal == "draw":
        return "Which check starts it?" if idea.draw == "perpetual" else "Which move is it?"
    if idea.goal == "convert":
        return "Which trade is it?"
    if idea.goal == "king":
        return "Which square does it take from theirs?"
    if idea.goal == "breakthrough":
        return "Which pawn goes first?"
    if idea.goal == "attack":
        return "Which sacrifice opens it?" if idea.sacrifices else "Which check comes first?"
    if idea.goal == "promote":
        return "Which move clears its way?"
    if idea.goal == "defend":
        return "Which move stops it and keeps your pieces working?"
    if name in NUDGE:
        return NUDGE[name]
    if idea.goal == "win" and idea.gain_ply > 0:
        return "What has to happen first, before you can take it?"
    if idea.goal == "win":
        return "Which move wins it?"
    return None


def target(idea: Idea, board: chess.Board) -> str | None:
    """The target on the board, grounded: the focus rung. Squares of their pieces and
    their king; the player's piece is said separately (`piece`)."""
    if idea.goal in ("mate", "attack") and idea.target is not None:
        free = [chess.square_name(s) for s in idea.flight]
        if not free:
            room = "with no free square"
        elif len(free) == 1:
            room = f"whose only free square is {free[0]}"
        else:
            room = "with free squares on " + ", ".join(free[:-1]) + f" and {free[-1]}"
        # How many checks there are, while the number narrows the search.
        checks = (
            f"; you have {_NUMBER.get(idea.checks, str(idea.checks))} check"
            f"{'s' if idea.checks != 1 else ''}"
            if 0 < idea.checks <= 4
            else ""
        )
        return f"their king on {_sq(idea.target)}, {room}{checks}"
    if idea.goal == "defend" and idea.threat is not None:
        # In words, never as a move: below the move rung no move is named, theirs included.
        mover = board.piece_at(idea.threat.from_square)
        taken = board.piece_at(idea.threat.to_square)
        who = f"their {_named(board, idea.threat.from_square)}" if mover else "their threat"
        if idea.prepared and idea.follow is not None:
            if idea.threatened == "mate":
                return f"{who}, one move from threatening mate"
            aim = board.piece_at(idea.follow.to_square)
            if aim is not None and aim.color == board.turn:
                return f"{who}, one move from winning your {_named(board, idea.follow.to_square)}"
            return f"{who}, one move from winning material"
        if idea.threatened == "mate":
            return f"{who}, which threatens mate"
        if taken is not None and taken.color == board.turn:
            return f"{who}, which threatens to take your {_named(board, idea.threat.to_square)}"
        return f"{who}, which threatens to win material"
    if idea.goal == "promote" and idea.first is not None:
        pawn = _queening_pawn(idea, board)
        if board.piece_type_at(pawn) != chess.PAWN:
            return None
        rank = chess.square_rank(pawn)
        to_go = 7 - rank if board.turn == chess.WHITE else rank
        away = f"{_NUMBER.get(to_go, str(to_go))} square{'s' if to_go != 1 else ''}"
        text = f"your pawn on {_sq(pawn)}, {away} from queening"
        if idea.blow is not None and idea.blow_board is not None:
            held = [
                s
                for s in (chess.parse_square(x) for x in idea.blow.squares[1:])
                if (p := idea.blow_board.piece_at(s)) is not None
                and p.color != board.turn
                and p.piece_type != chess.KING
            ]
            if held:
                text += f"; their {_now(idea, board, held[0])} is the piece to deal with"
        return text
    blow, where = idea.blow, idea.blow_board
    if blow is not None and where is not None:
        squares = [chess.parse_square(s) for s in blow.squares]
        theirs = [s for s in squares if (p := where.piece_at(s)) and p.color != board.turn]
        later = " after your first move" if idea.blow_at > 1 else ""
        if blow.name == "fork" and idea.blow_move is not None:
            # The first square is where the forking piece lands; the targets follow, and they
            # stand where they are before the fork as after it.
            targets = [
                s
                for s in squares[1:]
                if (p := where.piece_at(s)) is not None and p.color != board.turn
            ]
            if len(targets) >= 2:
                listed = [f"their {_now(idea, board, s)}" for s in targets[:3]]
                named = (
                    listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + f" and {listed[-1]}"
                )
                both = "both" if len(listed) == 2 else "all of them"
                return f"{named}: one move can hit {both}{later}"
        if blow.name == "skewer" and len(squares) >= 3:
            return (
                f"their {_now(idea, board, squares[1])} and their "
                f"{_now(idea, board, squares[2])}: one line can hold both{later}"
            )
        # Every piece named where it stands now, and when the pattern comes about.
        if blow.name in ("pin", "absolutePin") and len(squares) >= 3:
            return (
                f"their {_now(idea, board, squares[1])} and their "
                f"{_now(idea, board, squares[2])} behind it: one line can pin them{later}"
            )
        if blow.name == "discoveredAttack" and len(squares) >= 3:
            return (
                f"their {_now(idea, board, squares[2])}, on a line one of your pieces is "
                f"blocking{later}"
            )
        if blow.name in ("trappedPiece", "hangingPiece") and theirs:
            state = "it has no safe square" if blow.name == "trappedPiece" else "nothing defends it"
            when = "after your first move, " if idea.blow_at > 1 else ""
            return f"their {_now(idea, board, theirs[0])}: {when}{state}"
    target_piece = board.piece_at(idea.target) if idea.target is not None else None
    if idea.target is not None and target_piece is not None and target_piece.color != board.turn:
        return f"their {_named(board, idea.target)}"
    return None


def piece(idea: Idea, board: chess.Board) -> str | None:
    """The player's piece that starts it, and the one that strikes when that differs."""
    if idea.first is None:
        return None
    moving = board.piece_at(idea.first.from_square)
    if moving is None:
        return None
    # With its square: "your knight" is two pieces on most boards.
    text = f"your {PIECE_NAMES[moving.piece_type]} on {_sq(idea.first.from_square)}"
    given_first = bool(idea.sacrifices) and idea.sacrifices[0][0] == 0
    if given_first and idea.sacrifices[0][1] in (PIECE_NAMES[moving.piece_type], "exchange"):
        text += " -- the piece you give up"
    if idea.blow_move is not None and idea.blow_board is not None and idea.blow_at > 1:
        striker = idea.blow_board.piece_at(idea.blow_move.from_square)
        if striker is not None and striker.piece_type != moving.piece_type:
            text += (", then" if given_first else " first, then") + (
                f" your {PIECE_NAMES[striker.piece_type]}"
            )
    return text


def why(idea: Idea, board: chess.Board, describe) -> str | None:  # noqa: ANN001
    """The move played out, correct by construction: the line to the mate, or to the blow
    and what it wins, the first move named as the sacrifice it is. `describe` is
    `facts.describe_tactic`."""
    text = _why(idea, board, describe)
    if text is not None and idea.pinned_guard is not None and idea.blow_board is not None:
        guard = idea.blow_board.piece_at(idea.pinned_guard)
        if guard is not None:
            text += (
                f"; their {PIECE_NAMES[guard.piece_type]} on {_sq(idea.pinned_guard)} is pinned "
                "and cannot take"
            )
    if text is not None and idea.first is not None and board.is_en_passant(idea.first):
        first = board.san(idea.first)
        if text.startswith(first):
            text = f"{first}, en passant{text[len(first) :]}"
    if text is None or idea.first is None or not idea.sacrifices or idea.sacrifices[0][0]:
        return text
    first = board.san(idea.first)
    kind = idea.sacrifices[0][1]
    sac = f"{'an' if kind == 'exchange' else 'a'} {kind} sacrifice"
    rest = text[len(first) :] if text.startswith(first) else None
    if rest is None:
        return text
    if rest.startswith(":"):
        return f"{first}, {sac}:{rest[1:]}"
    if rest.startswith(","):
        return f"{first}, {sac}{rest}"
    return f"{first}, {sac},{rest}"


def _why(idea: Idea, board: chess.Board, describe) -> str | None:  # noqa: ANN001
    if idea.first is None:
        return None
    first = board.san(idea.first)
    if idea.goal == "mate" and idea.mate_in:
        plies = [m.uci() for m in idea.line[: 2 * idea.mate_in - 1]]
        if idea.mate_in == 1:
            return f"{first} is mate"
        after = board.copy(stack=False)
        after.push(idea.first)
        if idea.mate_in == 2 and len(plies) == 3:
            reply = after.san(idea.line[1])
            after.push(idea.line[1])
            return f"{first}, and after {reply}, {after.san(idea.line[2]).rstrip('#')} is mate"
        if idea.mate_in > MATE_READ:
            # A mate in nine read out whole is sixteen moves no one follows by ear, and
            # stopped short of the mate besides (Tai's game, 2026-09-24).
            shown = plies[1:MATE_SHOWN]
            return f"{first} forces mate: {first} {_line(after, shown)}, and mate follows"
        return f"{first} forces mate: {first} {_line(after, plies[1:])}"
    if idea.goal in ("win", "combination"):
        sentence = None
        if idea.blow is not None and idea.blow_board is not None:
            sentence = describe(idea.blow_board, idea.blow.name, idea.blow.squares, idea.blow_move)
        won = f"you win {idea.won}" if idea.won else "you win material"
        if idea.goal == "combination":
            won = "the material comes back" + (
                ", and the position is won" if idea.winning else ", with the better game"
            )
        if idea.blow_at > 1 and idea.blow_move is not None and idea.blow_board is not None:
            # The moves between the first and the blow, then the blow, then what it does.
            after = board.copy(stack=False)
            after.push(idea.first)
            between = [m.uci() for m in idea.line[1 : 2 * (idea.blow_at - 1)]]
            blow = idea.blow_board.san(idea.blow_move)
            text = f"{first}, and after {_line(after, between)}, {blow}"
            return text + (f", and {sentence}" if sentence else "") + f"; {won}"
        if sentence is None and idea.goal == "win" and idea.gain_ply > LATER_READ:
            # Past two moves the line is not read out: "h4 -- after d5 g4 Nxh4 Bg5 Be7 Bxh4
            # Bxh4 g5 Nc6 Rxh4 Rxh4, Nxh4" was twelve plies (Tai's game g01, 2026-09-24).
            return f"{first}: {won}, a few moves on"
        if sentence is None and idea.goal == "win" and 0 < idea.gain_ply < len(idea.line):
            # No tactic, and the first move does not win it: the moves up to the one that does.
            after = board.copy(stack=False)
            after.push(idea.first)
            between = [m.uci() for m in idea.line[1 : idea.gain_ply]]
            for uci in between:
                after.push(chess.Move.from_uci(uci))
            taker = after.san(idea.line[idea.gain_ply])
            after = board.copy(stack=False)
            after.push(idea.first)
            return f"{first}, and after {_line(after, between)}, {taker}; {won}"
        return f"{first}: " + (f"{sentence}; {won}" if sentence else won)
    if idea.goal == "promote":
        pawn = _queening_pawn(idea, board)
        which = (
            "the pawn"
            if pawn == idea.first.from_square or board.piece_type_at(pawn) != chess.PAWN
            else f"your pawn on {_sq(pawn)}"
        )
        queens = f"{which} goes through to queen"
        sentence = None
        if idea.blow is not None and idea.blow_board is not None:
            sentence = describe(idea.blow_board, idea.blow.name, idea.blow.squares, idea.blow_move)
        if sentence and idea.blow_at > 1 and idea.blow_move is not None:
            after = board.copy(stack=False)
            after.push(idea.first)
            between = [m.uci() for m in idea.line[1 : 2 * (idea.blow_at - 1)]]
            blow = idea.blow_board.san(idea.blow_move)  # type: ignore[union-attr]
            return f"{first}, and after {_line(after, between)}, {blow}, and {sentence}; {queens}"
        if sentence:
            return f"{first}: {sentence}; {queens}"
        return f"{first}: {queens}"
    if idea.goal == "defend" and idea.threat is not None:
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        if idea.prepared and idea.follow is not None:
            then = passed.copy(stack=False)
            then.push(idea.threat)
            then.push(chess.Move.null())
            follow = then.san(idea.follow).rstrip("#")
            aim = "mate" if idea.threatened == "mate" else ""
            return f"{first} meets their idea: {passed.san(idea.threat)}, and then {follow}" + (
                f" {aim}" if aim else ""
            )
        return f"{first} stops {passed.san(idea.threat)}"
    if idea.goal == "convert":
        shown = [m.uci() for m in idea.line[:4]]
        ending = _ending(idea).replace("a won ", "the ")
        return f"{first} starts it: {_line(board, shown)}, and {ending} is won"
    if idea.goal == "king":
        if idea.opposition:
            return f"{first} takes the opposition: their king has to give way"
        return f"{first}: the king leads, and the rest follows"
    if idea.goal == "breakthrough" and idea.runner is not None:
        shown = [m.uci() for m in idea.line[:5]]
        return f"{first} breaks through: {_line(board, shown)}, and a pawn runs through"
    if idea.goal == "draw":
        shown = [m.uci() for m in idea.line[:6]]
        how = {"perpetual": "perpetual check", "stalemate": "stalemate"}.get(
            idea.draw or "", "a repetition"
        )
        return f"{first} holds the game: {_line(board, shown)}, and it is a draw by {how}"
    if idea.goal == "attack":
        shown = [m.uci() for m in idea.line[:7]]
        return f"{first} starts it: {_line(board, shown)}, and their king is in the open"
    return None


def _queening_pawn(idea: Idea, board: chess.Board) -> chess.Square:
    """Where the pawn that queens in the line stands now -- or the first of the player's
    pawns to reach the seventh -- followed back to the board in front of the player. The
    first move's square only when that is a pawn: a king's walk that lets a pawn run was
    read as "your pawn on f5", of the king (Lichess 00g9R)."""
    assert idea.first is not None
    us = board.turn
    seventh = 6 if us == chess.WHITE else 1
    boards = [board]
    reached: chess.Square | None = None
    for j, move in enumerate(idea.line):
        mover = boards[-1].piece_at(move.from_square)
        ours = j % 2 == 0 and mover is not None and mover.piece_type == chess.PAWN
        if ours and move.promotion:
            start = origin(board, boards, j, move.from_square)
            if start is not None:
                return start
            break
        if ours and reached is None and chess.square_rank(move.to_square) == seventh:
            reached = origin(board, boards, j, move.from_square)
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    if reached is not None:
        return reached
    return idea.first.from_square


def _now(idea: Idea, board: chess.Board, square: chess.Square) -> str:
    """Their piece standing on `square` when the blow comes, named where it stands now:
    the focus hint is read on the board in front of the player."""
    assert idea.blow_board is not None
    ply = 2 * (idea.blow_at - 1)
    boards = [board]
    for move in idea.line[:ply]:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    start = origin(board, boards, ply, square)
    blown = idea.blow_board.piece_at(square)
    piece = board.piece_at(start) if start is not None else None
    same = piece is not None and blown is not None and blown.piece_type == piece.piece_type
    if start is not None and same:
        return _named(board, start)
    # Not on the board yet (a pawn that queens on the way): its name, and no square.
    return PIECE_NAMES[blown.piece_type] if blown is not None else "piece"


def _line(board: chess.Board, plies: list[str]) -> str:
    text = san_line(board, plies, limit=len(plies))
    return " ".join(t for t in text.split() if not t.endswith(".") and not t.endswith("..."))


def _ending(idea: Idea) -> str:
    """The won ending by its kind: "a won rook ending"; one of no single kind is "a won
    ending" ("a won mixed ending" said nothing, Tai's game g05, 2026-09-24)."""
    kind = idea.ending
    if kind in (None, "mixed", "minor-piece"):
        return "a won ending"
    return f"a won {kind} ending"


def _yours(won: str) -> str:
    """What they win, said to the player: "their queen" from their side is "your queen"."""
    return won.replace("their ", "your ")


def allowed(board: chess.Board, analysis: Analysis, played: str) -> str | None:
    """What the move just played lets the opponent do: the engine's best line in the position
    it left (theirs to move), read by the same idea reader -- mate, material, a won ending,
    a winning attack, a pawn through. The half a lesson after a mistake was missing: "Rxa6
    was better" and nothing on Rxa1+ taking a rook (Tai's game g02, 13.Qxd5, 2026-09-24).
    None when the line takes nothing of the kind."""
    if not analysis.lines:
        return None
    idea = idea_of.read(board, analysis)
    if idea.first is None or idea.first not in board.legal_moves:
        return None
    first = board.san(idea.first)
    after = f"after {played}, "
    if idea.goal == "mate" and idea.mate_in:
        if idea.mate_in == 1:
            return after + f"{first.rstrip('#')} is mate"
        n = _NUMBER.get(idea.mate_in, str(idea.mate_in))
        return after + f"they have mate in {n}, starting with {first}"
    if idea.goal == "win" and idea.won and not idea.level and not idea.back:
        won = _yours(idea.won)
        if idea.gain_ply == 0:
            return after + f"{first} wins {won}"
        if idea.gain_ply > LATER_READ:
            return after + f"they win {won} a few moves on, starting with {first}"
        return after + f"they win {won}, starting with {first}"
    if idea.goal == "convert" and idea.ending:
        return after + f"{first} trades them into {_ending(idea)}"
    if idea.goal == "promote":
        return after + f"their pawn goes through to queen, starting with {first}"
    if idea.goal == "attack" and idea.winning:
        return after + f"they have a winning attack on your king, starting with {first}"
    return None
