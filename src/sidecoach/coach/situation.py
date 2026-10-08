"""The first hint: what is at stake, and the fact on the board that matters.

Tai, 2026-09-24, after reading Capablanca-Marshall move by move: the first hint was a principle
and a question ("no tactic here -- this is about fighting for the centre: what would make the
centre yours?"), and in a middlegame there are a million ways to fight for the centre while
only a few moves keep the game. The first hint is now a short lesson on the position:

- the stakes, from the engine's lines: you are in check; this is opening theory; only one move
  holds (and what even the next best one costs); only two or three keep your edge; nothing is
  urgent and several moves are fine; the game is won, or lost;
- the key fact, named on the board: their king's squares and your checks, their threat, the
  target, the plan and the pieces it needs, the concrete thing the best move is about;
- the trap, when there is one: the capture that looks free and is not.

It names pieces, squares and the opponent's moves -- never the player's move, nor the square it
goes to unless an enemy piece stands there (the capture's target is the lesson). A bare
principle is never the hint; the principle comes only after the fact it explains.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange, positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words, plans
from sidecoach.coach.idea import Idea
from sidecoach.events import Analysis, Line

GOOD_MARGIN = 0.04  # within this of the best move's expected score, a move is as good
ONLY_DROP = 0.15  # past the one good move, a drop this big: only one move holds
# Past two or three good moves, a drop this big: only they keep it (14.Qf3 in
# Capablanca-Marshall: d4 and Qf3 at 0.60, the rest at 0.50 and below).
FEW_DROP = 0.08
MANY = 4  # this many good moves: nothing is urgent
WON = 0.8
LOST = 0.2
CHOICES = 3  # the good moves named when several are fine, or in opening theory

_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
# Counts in words to twenty: a digit in the first hint is a figure the coach may not quote
# ("you have 10 checks": the pre-spend audit, 2026-09-25).
_NUMBER = {
    2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty",
}  # fmt: skip
_PAWN_MOVE = re.compile(r"[a-h](x[a-h])?[1-8]")

# The second hint's question for the best move's own idea, when the idea reader has none:
# from the fact the first hint named to the move, never the move itself.
POSITIONAL_NUDGE: dict[str, str] = {
    "castle": "Where is your king safest, and how does it get there in one move?",
    "rescue": "Where is it safe -- and where does it also do something?",
    "prophylaxis": "Which move takes that idea away without losing time?",
    "develop": "Which of them comes out first, and to which square?",
    "activateKing": "Which way should your king walk?",
    "outpostMove": "Which square is it, and which knight gets there?",
    "takeFile": "Which rook goes to it?",
    "rookBehindPasser": "Which square is behind it?",
    "break": "Is it time to strike, or does it need one more move of support?",
    "centre": "Which pawn move lets it out?",
    "centralise": "Which pawn move lets it out?",
    "space": "Which pawn move lets it out?",
    "activate": "Where would it do more?",
    "pushPasser": "How far can it go safely right now?",
    "passerPawnEnding": "How far can it go safely right now?",
    "pawnStorm": "Which pawn goes first?",
    "simplify": "Which trade is it?",
}


# Questions that fit any position with something to win, and so narrow nothing once the first
# hint has named the target: never said after it.
ANY_POSITION = frozenset(
    {
        "Which of their pieces is loose or attacked more times than it is defended?",
        "Which of your pieces has no job right now?",
        "Go through your checks and captures: which leaves them the fewest good replies?",
    }
)


def positional_nudge(board: chess.Board, kind: str, key: str) -> str | None:
    """The second hint's question for the best move's own idea (`POSITIONAL_NUDGE`), said of
    what the first hint named: one piece at home is "it", not "which of them"; space taken
    is not a piece let out."""
    if kind == "develop" and len(undeveloped(board, board.turn)) == 1:
        return "Which square does it belong on?"
    if kind == "space" and "shut in" not in key:
        return "Which pawn goes forward, and how far?"
    return POSITIONAL_NUDGE.get(kind)


_CENTRE = (chess.D4, chess.E4, chess.D5, chess.E5)


def centre_move(board: chess.Board, move: chess.Move) -> tuple[str, str] | None:
    """What a pawn move in the centre does, named on the board, for the second hint and the
    third: the central squares it now covers, or the file a pawn capture opens. "Fighting for
    the centre" said nothing (Tai, 2026-09-24); "f4 covers e5" says what f4 is for."""
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return None
    us = board.turn
    san = board.san(move)
    after = board.copy(stack=False)
    after.push(move)
    before = board.attacks_mask(move.from_square)
    covered = [sq for sq in _CENTRE if after.attacks_mask(move.to_square) & chess.BB_SQUARES[sq]]
    covered = [sq for sq in covered if not before & chess.BB_SQUARES[sq]]
    if covered:
        where = _listed([_sq(sq) for sq in covered])
        return f"Which pawn move covers {where}?", f"{san}: the pawn now covers {where}"
    if board.is_capture(move):
        file_ = chess.square_file(move.from_square)
        if (
            not board.pieces(chess.PAWN, us)
            & chess.BB_FILES[file_]
            & ~chess.BB_SQUARES[move.from_square]
        ):
            letter = "abcdefgh"[file_]
            return (
                "Which capture opens a file for your rooks?",
                f"{san}: the capture opens the {letter}-file for your rooks",
            )
    return None


def reach_why(board: chess.Board, move: chess.Move) -> str | None:
    """A piece brought to a better square, counted: "Bf6: your bishop goes from four squares
    to nine". "The move is Bf6: centralising a piece" said nothing (Karpov-Ribli)."""
    piece = board.piece_at(move.from_square)
    if piece is None or piece.piece_type in (chess.PAWN, chess.KING):
        return None
    us = board.turn
    before = chess.popcount(board.attacks_mask(move.from_square) & ~board.occupied_co[us])
    after = board.copy(stack=False)
    after.push(move)
    later = chess.popcount(after.attacks_mask(move.to_square) & ~after.occupied_co[us])
    if later <= before:
        return None
    was = "one square" if before == 1 else f"{idea_words.in_words(before)} squares"
    name = _NAMES[piece.piece_type]
    return f"{board.san(move)}: your {name} goes from {was} to {idea_words.in_words(later)}"


def _kings_apart(board: chess.Board, us: chess.Color) -> bool:
    """The kings stand on opposite wings (files a-c against f-h); a king on d or e has none."""
    ours, theirs = board.king(us), board.king(not us)
    if ours is None or theirs is None:
        return False
    sides = {_wing(ours), _wing(theirs)}
    return sides == {"on the queenside", "on the kingside"}


def _wing(square: chess.Square) -> str:
    file_ = chess.square_file(square)
    if file_ <= 2:
        return "on the queenside"
    if file_ >= 5:
        return "on the kingside"
    return "in the centre"


# The first hint said as a coach speaks (Tai, 2026-09-25: "make it sound a bit better and
# more natural"): the contractions of speech. A verb that ends its clause stays whole
# ("where it is.", never "where it's."), so "is" and "are" contract only before a word.
_CONTRACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b([Cc])annot\b"), r"\1an't"),
    (re.compile(r"\b([Ww])ill not\b"), r"\1on't"),
    (
        re.compile(r"\b(is|are|was|were|does|do|did|has|have|had|could|would|should) not\b", re.I),
        r"\1n't",
    ),
    (re.compile(r"\b([Ii]t|[Tt]here|[Tt]hat|[Ww]hat) is (?=\w)"), r"\1's "),
    (re.compile(r"\b([Yy]ou|[Tt]hey|[Ww]e) are (?=\w)"), r"\1're "),
    (re.compile(r"\bI am\b"), "I'm"),
)


def spoken(text: str) -> str:
    """`text` with the contractions a coach speaks with: "it's a trap", "can't take back",
    "you're lost"."""
    for pattern, short in _CONTRACTIONS:
        text = pattern.sub(short, text)
    return text


# The most sentences a first hint says. Past it, the details of the pattern go first, the
# least telling first: a second pattern in the line, the quiet first move, the in-between
# move, which move the blow comes on. The stakes, the prize, the key fact, the pattern and
# why it works, a sacrifice on the way, and the question stay (the first hint ran to seven,
# eight, ten sentences when every detail stacked up: the real-game review, 2026-09-26).
FIRST_HINT_SENTENCES = 5
_DETAILS = tuple(
    re.compile(pattern)
    for pattern in (
        r"There is (?:a|an) [^.?!]* in it too\.",
        r"It starts with a quiet move: no check and no capture, but a threat they cannot meet\.",
        r"On the way comes an in-between move: a check before taking back\.",
        r"It comes on your \w+ move(?:, once [^.?!]*)?\.",
    )
)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def within_budget(text: str, budget: int = FIRST_HINT_SENTENCES) -> str:
    """`text` with the pattern's details dropped, least telling first, until it says at most
    `budget` sentences -- or none is left to drop."""
    for detail in _DETAILS:
        if len(_SENTENCE_END.split(text.strip())) <= budget:
            break
        text = re.sub(r"\s{2,}", " ", detail.sub("", text)).strip()
    return text


def sentence(text: str) -> str:
    """`text` as a sentence: a capital first (never on a pawn's move: "hxg4 looks..."), a
    stop last."""
    text = text.strip()
    if not text:
        return ""
    if not _PAWN_MOVE.match(text):
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


@dataclass(frozen=True)
class Stakes:
    """How much rides on the move: its kind, how many moves are as good as the best, the
    best move's expected score, and the first line past the good ones."""

    kind: str  # "check", "book", "won", "lost", "only", "few", "many", or "open"
    good: int
    top: float
    next_line: Line | None
    legal: int
    choices: tuple[str, ...] = ()


def _sq(square: chess.Square) -> str:
    return chess.square_name(square)


def _named(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    return f"{_NAMES[piece.piece_type]} on {_sq(square)}" if piece else _sq(square)


def _listed(items: Sequence[str]) -> str:
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def read_stakes(board: chess.Board, analysis: Analysis) -> Stakes | None:
    """The stakes of the position, from the engine's lines."""
    lines = [line for line in analysis.lines if line.moves]
    if not lines:
        return None
    legal = board.legal_moves.count()
    top = lines[0].expected_score
    if lines[0].mate is not None and lines[0].mate > 0:
        good_lines = [line for line in lines if line.mate is not None and line.mate > 0]
    else:
        good_lines = [line for line in lines if top - line.expected_score <= GOOD_MARGIN]
    rest = [line for line in lines if line not in good_lines]
    next_line = rest[0] if rest else None
    good = len(good_lines)
    choices: list[str] = []
    for line in good_lines[:CHOICES]:
        try:
            move = chess.Move.from_uci(line.moves[0])
        except ValueError:
            continue
        if move in board.legal_moves:
            choices.append(board.san(move))
    drop = top - next_line.expected_score if next_line is not None else 0.0
    if board.is_check():
        kind = "check"
    elif analysis.in_book:
        kind = "book"
    elif top >= WON:
        kind = "won"
    elif top < LOST:
        kind = "lost"
    elif good == 1 and next_line is not None and drop >= ONLY_DROP:
        kind = "only"
    elif good <= 3 and next_line is not None and drop >= FEW_DROP:
        kind = "few"
    elif good >= MANY or (good == len(lines) and legal > len(lines)):
        kind = "many"
    else:
        kind = "open"
    return Stakes(kind, good, top, next_line, legal, tuple(choices))


def _what_it_gives(board: chess.Board, line: Line) -> str | None:
    """What the line's first move hands the opponent, read from the position it leaves by
    the same idea reader: "a knight", "your queen", "mate", "a winning attack"."""
    try:
        first = chess.Move.from_uci(line.moves[0])
    except (ValueError, IndexError):
        return None
    if first not in board.legal_moves or len(line.moves) < 2:
        return None
    # With the board's moves: a capture of the player's is the start of an exchange, and the
    # piece taken back for it is not "given" (`idea.exchange_owed`).
    after = board.copy()
    after.push(first)
    theirs = Analysis(
        game_id="",
        fen=after.fen(),
        ply=0,
        lines=[
            Line(
                moves=list(line.moves[1:]),
                depth=line.depth,
                expected_score=1.0 - line.expected_score,
                mate=-line.mate if line.mate is not None else None,
            )
        ],
        decisive=False,
        only_move=False,
    )
    idea = idea_of.read(after, theirs)
    if idea.goal == "mate":
        return "mate"
    if idea.goal == "win" and idea.won and not idea.level and not idea.back:
        return idea_words._yours(idea.won)
    if idea.goal == "attack" and idea.winning:
        return "a winning attack"
    if idea.goal == "promote":
        return "a new queen"
    if idea.goal == "convert":
        return "a won ending"
    return None


def _have(board: chess.Board, given: str) -> bool:
    """Whether the player has the pieces `given` names as lost: a knight promoted in the line
    and taken is not "a knight" the player has to lose (Lichess UkpFL, a rook ending)."""
    lost = given.split(" for ", 1)[0]
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        if _NAMES[piece_type] in lost and not board.pieces(piece_type, board.turn):
            return False
    return True


def edge_words(top: float) -> str:
    """What the best move keeps, by its expected score: "your edge", "the balance"."""
    return _edge(top)


def _edge(top: float) -> str:
    if top >= 0.6:
        return "your edge"
    if top >= 0.4:
        return "the balance"
    return "you in the game"


def _cost(board: chess.Board, stakes: Stakes) -> str:
    """What even the next best move costs: grounded in its line when the reader sees it,
    else in the score."""
    line = stakes.next_line
    if line is None:
        return ""
    given = _what_it_gives(board, line)
    if given == "mate":
        return "even the next best move gets you mated"
    if given == "a winning attack":
        return "even the next best move hands them a winning attack"
    if given == "a new queen":
        return "even the next best move lets them queen"
    if given == "a won ending":
        return "even the next best move hands them a won ending"
    if given is not None and _have(board, given):
        return f"even the next best move loses {given}"
    rest = line.expected_score
    if rest < 0.3:
        return "anything else loses"
    if rest < 0.45:
        return "anything else leaves you worse"
    if stakes.top >= 0.6 and rest < 0.55:
        return "anything else gives away your edge"
    return "anything else gives back much of what you have"


def _only(stakes: Stakes) -> bool:
    """One move holds: the next best drops by `ONLY_DROP` at least, and out of a win when
    the best one wins ("only one move keeps it that way" was said with the next at 0.93)."""
    line = stakes.next_line
    if stakes.good != 1 or line is None or stakes.top - line.expected_score < ONLY_DROP:
        return False
    return stakes.top < WON or line.expected_score < WON


def stakes_words(board: chess.Board, stakes: Stakes, opening: str | None) -> str | None:
    """The stakes as the first hint says them."""
    kind = stakes.kind
    if kind == "check":
        if stakes.legal == 1:
            # And whether it saves anything: "only one legal move" of a mated king was read
            # as "only one move holds; otherwise you are mated" (the test audit, 2026-09-26).
            if stakes.top < LOST:
                return (
                    "You're in check, and there is only one legal move -- it does not save the game"
                )
            return "You're in check, and there is only one legal move"
        if stakes.top < LOST:
            # "More than one way out is fine" was said of a lost king, every way out within a
            # hair of the best because all of them lose (the facts audit, 2026-10-01).
            return "You're in check and losing; the best way out makes them work for it"
        if _only(stakes):
            if stakes.top >= WON:
                return (
                    "You're in check but winning, and only one way out keeps it that way: "
                    f"{_cost(board, stakes)}"
                )
            return f"You're in check, and only one way out holds: {_cost(board, stakes)}"
        if stakes.top >= WON:
            return "You're in check but winning: get out of it without giving anything back"
        if stakes.good >= 2:
            return "You're in check, and more than one way out is fine"
        return "You're in check: choose the way out with care"
    if kind == "book":
        moves = _listed(stakes.choices) if stakes.choices else None
        name = f" -- the {opening}" if opening else ""
        if moves:
            return f"This is opening theory{name}; the main moves here are {moves}"
        return f"This is opening theory{name}"
    if kind == "won":
        if _only(stakes):
            return f"You're winning, but only one move keeps it that way: {_cost(board, stakes)}"
        return None
    if kind == "lost":
        return "You're losing here; the best move makes them work for it"
    if kind == "only":
        return f"Only one move holds here: {_cost(board, stakes)}"
    if kind == "few":
        if stakes.good == 1:
            return f"Only one move keeps {_edge(stakes.top)} here: {_cost(board, stakes)}"
        n = _NUMBER.get(stakes.good, str(stakes.good))
        return f"Only {n} moves keep {_edge(stakes.top)} here: {_cost(board, stakes)}"
    if kind == "many":
        moves = _listed(stakes.choices) if stakes.choices else None
        if moves:
            return f"Nothing is urgent: several moves keep {_edge(stakes.top)}, {moves} among them"
        return f"Nothing is urgent: several moves keep {_edge(stakes.top)}"
    return None  # a few moves close together: nothing worth saying about the stakes


def ranked_traps(traps: Sequence[Line], target: chess.Square | None) -> list[Line]:
    """The traps in the order the first hint wants them: the one that takes what the best line
    wins later ("not yet"), then those that walk into mate, then the worst first. Topalov-
    Kramnik, 36.Qxg3?? "allows mate in two" was not the first trap found."""

    def square(trap: Line) -> chess.Square | None:
        try:
            return chess.Move.from_uci(trap.moves[0]).to_square
        except (ValueError, IndexError):
            return None

    def rank(trap: Line) -> tuple[int, int, float]:
        on_target = target is not None and square(trap) == target
        mated = trap.mate is not None and trap.mate < 0
        return (0 if on_target else 1, 0 if mated else 1, trap.expected_score)

    return sorted(traps, key=rank)


def trap_words(
    board: chess.Board,
    trap: Line,
    target: chess.Square | None = None,
    top: float | None = None,
    best: chess.Move | None = None,
) -> str | None:
    """The capture that looks free and is not: "careful, though: hxg4 looks like it wins a
    knight, but after hxg4 Qh4, they get a winning attack". When it takes what the best line
    wins later (`target`), the two are one lesson: "their rook on a8 can be won, but not at
    once: Qxa8 right away runs into Nxh3+, and they have mate" (16.Re2 in Capablanca-Marshall;
    `in_order` when the hint has named the target already). `top` is the best move's expected
    score: a game already lost is not lost by the capture."""
    try:
        capture = chess.Move.from_uci(trap.moves[0])
    except (ValueError, IndexError):
        return None
    taken = board.piece_at(capture.to_square)
    if capture not in board.legal_moves or taken is None or len(trap.moves) < 2:
        return None
    san = board.san(capture)
    after = board.copy(stack=False)
    after.push(capture)
    try:
        reply = chess.Move.from_uci(trap.moves[1])
    except ValueError:
        return None
    if reply not in after.legal_moves:
        return None
    reply_san = after.san(reply)
    # Their reply, unless it would say where the solution goes: a reply onto the first move's
    # square, empty now, or written as the first move is ("after Rxh6 e5" where the solution
    # is e5; "after Qxf7 Rxf7" where it is Rxf7). The trap is said without it.
    shows = best is not None and (
        reply_san == board.san(best)
        or (reply.to_square == best.to_square and board.piece_at(best.to_square) is None)
    )
    given = _what_it_gives(board, trap)
    # A capture that comes out ahead or even in material, with no mate, attack or lost game
    # behind it, is a lesser move, not a trap (19...Bxf2 Qxf2 won the exchange and 21.Qxf2
    # Qxc1 traded evenly in Capablanca-Marshall).
    ahead = _net(board, trap.moves) >= 0
    if given == "mate":
        tail = "they have mate"
    elif given == "a winning attack":
        tail = "they get a winning attack"
    elif given is not None and not ahead:
        tail = f"they win {given}"
    elif trap.expected_score < 0.3 and (top is None or top >= 0.3):
        tail = "you are lost"
    elif _checks(after, trap.moves[1:]) >= 2:
        tail = "their checks begin, and your king has no cover"
    elif not ahead:
        tail = "it costs you more than it wins"
    elif (
        top is not None
        and top - trap.expected_score >= BALANCE_DROP
        and trap.expected_score < WORSE
    ):
        # The balance lost without material: the wrong recapture opening a file (23.exd5?
        # Re8 in Polugaevsky-Nezhmetdinov).
        tail = "you are left clearly worse"
    elif top is not None and top >= EDGE and trap.expected_score < EDGE - TRAP_EDGE_DROP:
        # Even in material but no longer better: "Not 17 Qxf2 Bg3 ... with an unclear
        # position" (Capablanca-Marshall).
        tail = "your edge is gone"
    else:
        return None
    what = _NAMES[taken.piece_type]
    if best is not None and best != capture and best.to_square == capture.to_square:
        # The same piece, taken with the wrong one of yours.
        after_it = f"{san}" if shows else f"{san} {reply_san}"
        return f"{RIGHT_PIECE}, though: {san} looks natural, but after {after_it}, {tail}"
    if target is not None and capture.to_square == target:
        runs = "a reply" if shows else reply_san
        return (
            f"their {_named(board, target)} can be won{NOT_YET}{san} right away runs into "
            f"{runs}, and {tail}"
        )
    after_it = f"{san}" if shows else f"{san} {reply_san}"
    return f"careful, though: {san} looks like it wins a {what}, but after {after_it}, {tail}"


# The trap that takes the target too soon, as `trap_words` says it: the target named, then
# the order ("their rook on b8 can be won, but not at once: Qxb8 right away runs into ...").
NOT_YET = ", but not at once: "
RIGHT_PIECE = "take back with the right piece"


def in_order(trap: str) -> str:
    """A trap that takes the target too soon, where the hint has named the target already:
    the order alone ("careful with the order, though: Qxb8 right away runs into Qd2+")."""
    if NOT_YET not in trap:
        return trap
    return "careful with the order, though: " + trap.split(NOT_YET, 1)[1]


def first_said(trap: str) -> str:
    """A trap said first, with nothing before it to turn from: "careful: ..."."""
    return trap.replace(", though: ", ": ", 1)


TRAP_NET_PLIES = 8  # how far a trap's line is followed to see what it nets
EDGE = 0.6  # a better game, by expected score
TRAP_EDGE_DROP = 0.05  # below the edge by this much after the capture: the edge is gone
BALANCE_DROP = 0.2  # the capture costs this much ...
WORSE = 0.4  # ... and leaves the player below this: clearly worse


def _net(board: chess.Board, moves: Sequence[str]) -> int:
    """What the line nets the side to move in material, in pawns, as it stands after their
    last reply within TRAP_NET_PLIES."""
    us = board.turn
    b = board.copy(stack=False)

    def balance() -> int:
        return sum(
            idea_of.VALUES.get(p.piece_type, 0) * (1 if p.color == us else -1)
            for p in b.piece_map().values()
        )

    start = balance()
    net = 0
    for j, uci in enumerate(moves[:TRAP_NET_PLIES]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in b.legal_moves:
            break
        b.push(move)
        if j % 2 == 1:
            net = balance() - start
    return net


def _checks(board: chess.Board, moves: Sequence[str]) -> int:
    """How many checks the side to move gives in its first four moves of `moves`."""
    b = board.copy(stack=False)
    given = 0
    for j, uci in enumerate(moves[:8]):
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in b.legal_moves:
            break
        if j % 2 == 0 and b.gives_check(move):
            given += 1
        b.push(move)
    return given


def undeveloped(board: chess.Board, us: chess.Color) -> list[chess.Square]:
    """The player's knights and bishops still on their home squares."""
    return positional.at_home(board, us)


def shut_in(board: chess.Board, move: chess.Move) -> tuple[chess.Square, chess.Square] | None:
    """A pawn move that lets a piece still at home out: the piece's square and the pawn's.
    "Your bishop on c1 is shut in behind the pawn on d2" -- what 15.d4 was for in
    Capablanca-Marshall ("development counts for more than material")."""
    us = board.turn
    if board.piece_type_at(move.from_square) != chess.PAWN:
        return None
    after = board.copy(stack=False)
    after.push(move)
    # Bishops only: a queen's diagonal opened is not what a pawn move is for (c4 "lets your
    # queen out" said nothing about the English).
    for sq in undeveloped(board, us):
        if board.piece_type_at(sq) != chess.BISHOP:
            continue
        before = chess.popcount(board.attacks_mask(sq) & ~board.occupied_co[us])
        later = chess.popcount(after.attacks_mask(sq) & ~after.occupied_co[us])
        if later - before >= 2:
            return sq, move.from_square
    return None


def _attacker(board: chess.Board, square: chess.Square) -> chess.Square | None:
    """The cheapest enemy piece that can take on `square` legally."""
    piece = board.piece_at(square)
    if piece is None:
        return None
    takers = exchange.capturers(board, not piece.color, square)
    if not takers:
        return None
    return min(takers, key=lambda s: exchange.VALUE.get(board.piece_type_at(s) or 0, 0))


def positional_key(
    board: chess.Board,
    best: chess.Move,
    kind: str,
    threat: chess.Move | None,
    phase: str,
) -> str | None:
    """The concrete situation behind the best move's own idea, named on the board -- the
    position, not the move: "your king is still on e1", "your bishop on c1 is shut in"."""
    us = board.turn
    piece = board.piece_at(best.from_square)
    if piece is None:
        return None
    name = _NAMES[piece.piece_type]
    frm = _sq(best.from_square)
    if kind == "castle":
        king = board.king(us)
        return f"Your king is still in the centre on {_sq(king)}" if king is not None else None
    if kind == "rescue":
        attacker = _attacker(board, best.from_square)
        if attacker is None:
            return None
        return f"Your {name} on {frm} is attacked by their {_named(board, attacker)}"
    if kind == "prophylaxis" and threat is not None:
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        if threat not in passed.legal_moves:
            return None
        if threat.to_square == best.to_square:
            # Your move takes the very square theirs wants: said by where their piece stands
            # ("their idea is b5" named the answer, b5, in a pawn ending).
            mover = passed.piece_at(threat.from_square)
            if mover is None:
                return None
            wants = (
                "wants to come forward"
                if mover.piece_type == chess.PAWN
                else "has a square in mind"
            )
            return (
                f"Their {_named(passed, threat.from_square)} {wants}: it is worth stopping "
                "before it comes"
            )
        return f"Their idea is {passed.san(threat)}: it is worth stopping before it comes"
    if kind == "develop":
        home = undeveloped(board, us)
        if not home:
            return None
        pieces = _listed([_named(board, s) for s in home])
        verb = "has" if len(home) == 1 else "have"
        return f"Your {pieces} {verb} not moved yet"
    if kind == "activateKing":
        return f"This is an ending: your king on {frm} belongs in the fight, not on the side"
    if kind == "outpostMove":
        return (
            "There is a square in their half that no pawn of theirs can ever attack, and your "
            f"knight on {frm} can get there"
        )
    if kind == "takeFile":
        on_file = chess.BB_FILES[chess.square_file(best.to_square)]
        file_ = "abcdefgh"[chess.square_file(best.to_square)]
        if board.pieces_mask(chess.ROOK, us) & on_file:
            # A rook already on it: "none of your rooks is there yet" was false (2026-10-01).
            return f"The {file_}-file has no pawns of yours on it"
        return f"The {file_}-file has no pawns of yours on it, and none of your rooks is there yet"
    if kind in ("pushPasser", "passerPawnEnding") and piece.piece_type == chess.PAWN:
        rank = chess.square_rank(best.from_square)
        to_go = 7 - rank if us == chess.WHITE else rank
        away = f"{idea_words.in_words(to_go)} square{'s' if to_go != 1 else ''}"
        return f"Your passed pawn on {frm} is {away} from queening"
    if kind == "rookBehindPasser":
        # Behind their passer the rook holds it back; it pushes only your own (it was said
        # "where it pushes it on" of their pawn, 2026-10-01).
        after = board.copy(stack=False)
        after.push(best)
        passers = [
            s
            for s in chess.SquareSet(after.pieces_mask(chess.PAWN, not us))
            if chess.square_file(s) == chess.square_file(best.to_square)
            and positional.is_passed(after, s)
        ]
        if passers and not any(
            chess.square_file(s) == chess.square_file(best.to_square)
            and positional.is_passed(after, s)
            for s in chess.SquareSet(after.pieces_mask(chess.PAWN, us))
        ):
            return (
                f"Your rook on {frm} belongs behind their passed pawn on {_sq(passers[0])}, "
                "where it holds it back"
            )
        return f"Your rook on {frm} belongs behind a passed pawn, where it pushes it on"
    if kind == "break":
        after = board.copy(stack=False)
        after.push(best)
        hit = [
            s
            for s in chess.SquareSet(after.attacks_mask(best.to_square))
            if after.piece_type_at(s) == chess.PAWN and after.color_at(s) != us
        ]
        if hit:
            return f"Your pawn on {frm} can strike at their pawn on {_sq(hit[0])}"
        return None
    if kind in ("centre", "space", "centralise", "activate"):
        opened = shut_in(board, best)
        if opened is not None:
            home, pawn = opened
            return f"Your {_named(board, home)} is still shut in behind the pawn on {_sq(pawn)}"
        if kind == "activate":
            reach = chess.popcount(board.attacks_mask(best.from_square) & ~board.occupied_co[us])
            return (
                f"Your {name} on {frm} is your least active piece: it reaches only {reach} squares"
            )
        if kind == "space" and piece.piece_type == chess.PAWN and phase != "endgame":
            # Where, not which pawn: 14.Qe1 in Johner-Nimzowitsch, a4 "White's last chance to
            # make a fight of it".
            return f"You can take space {_wing(best.from_square)}"
        return None
    if kind == "pawnStorm":
        king = board.king(not us)
        if king is None:
            return None
        if _kings_apart(board, us):
            return (
                f"Their king on {_sq(king)} is on the other wing from yours: your pawns can "
                "roll at it"
            )
        # A king in the centre has no wing ("the other wing from yours" of a king on e1).
        return (
            f"Their king on {_sq(king)} is {_wing(king)}, and yours is not: your pawns can "
            "roll at it"
        )
    if kind == "simplify":
        return "You are ahead in material: every trade brings the win closer"
    return None


def plan_key(board: chess.Board, plan: plans.LinePlan) -> str | None:
    """The line's plan, named on the board, as a lesson."""
    pieces = plans.pieces(plan, board)
    sq = _sq(plan.square) if plan.square is not None else None
    if plan.kind == "attack" and sq:
        who = f": {pieces} can join in" if pieces else ""
        return f"The plan is an attack on their king on {sq}{who}"
    if plan.kind == "storm" and sq:
        if _kings_apart(board, board.turn):
            return (
                f"The plan is a pawn storm on their king on {sq}: with the kings on opposite "
                "wings, it costs your own king nothing"
            )
        return f"The plan is a pawn storm on their king on {sq}, away from your own king"
    if plan.kind == "double" and plan.where:
        return (
            f"The {plan.detail or 'open'} {plan.where}-file is the way in: two of your heavy "
            "pieces belong on it"
        )
    if plan.kind == "outpost" and sq and plan.square is not None:
        who = f"{pieces}" if pieces else "a knight"
        if positional.pawn_safe(board, plan.square, board.turn):
            return f"{sq} is a square no pawn of theirs can ever attack, and {who} can reach it"
        # The line makes it one (a pawn of theirs that attacks it now is traded or passes
        # on the way): "no pawn of theirs can ever attack" was said with b6 hitting c5.
        return f"{sq} becomes a square no pawn of theirs can attack, and {who} can reach it"
    if plan.kind == "passer" and sq:
        return f"Your passed pawn on {sq} is the point: push it, and it ties their pieces down"
    if plan.kind == "break" and sq:
        return f"The pawn break on {sq} is the plan"
    if plan.kind == "wing" and plan.where:
        return f"Your pawns on the {plan.where} can take space, away from both kings"
    return None


def threat_key(board: chess.Board, threat: plans.MoveThreat) -> str | None:
    """The target of the threat the best move makes, named: the lesson is to hit it."""
    target = board.piece_at(threat.square)
    if threat.kind == "mate":
        return (
            f"Their king on {_sq(threat.square)} is short of squares: one quiet move can set "
            "up mate"
        )
    if target is None or target.color == board.turn:
        return None
    if threat.kind == "trade":
        return f"Their {_named(board, threat.square)} is worth trading off"
    return f"Their {_named(board, threat.square)} can be hit"


def defend_key(board: chess.Board, idea: Idea, best: chess.Move | None = None) -> str | None:
    """Their threat, named: the move they threaten and what it wins, or what it prepares.
    When their move goes to the square the best move takes first, it is said by their piece:
    "they are preparing e6" named the answer, e6 (Lichess vgdbX)."""
    if idea.threat is None:
        return None
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    if idea.threat not in passed.legal_moves:
        return None
    san = passed.san(idea.threat)
    if best is not None and idea.threat.to_square == best.to_square:
        mover = _named(passed, idea.threat.from_square)
        if idea.threatened == "mate":
            return f"Their {mover} is one move from a mating threat"
        return f"Their {mover} is one move from winning material"
    if idea.prepared and idea.follow is not None:
        then = passed.copy(stack=False)
        then.push(idea.threat)
        then.push(chess.Move.null())
        if idea.follow not in then.legal_moves:
            return None
        follow = then.san(idea.follow).rstrip("#+")
        if idea.threatened == "mate":
            return f"They are preparing {san}, and then {follow} would be mate"
        aim = board.piece_at(idea.follow.to_square)
        what = f"your {_named(board, idea.follow.to_square)}" if aim is not None else "material"
        return f"They are preparing {san}, and then {follow} would win {what}"
    if idea.threatened == "mate":
        return f"They threaten {san.rstrip('#')} mate"
    taken = board.piece_at(idea.threat.to_square)
    if taken is not None and taken.color == board.turn:
        return f"They threaten {san}, winning your {_named(board, idea.threat.to_square)}"
    return f"They threaten {san}"


def relevant_feature(
    board: chess.Board, facts_list: Sequence[positional.PositionalFact], pv: Sequence[chess.Move]
) -> str | None:
    """A static feature of the board that the best move is about: one whose squares the
    move itself touches. A square only a later move of the line touches sent the player to
    that move first (15.Bd2 in Johner-Nimzowitsch, beside a4)."""
    touched: set[str] = set()
    left: set[str] = set()
    for move in pv[:1]:
        touched |= {_sq(move.from_square), _sq(move.to_square)}
        left.add(_sq(move.from_square))
    for fact in facts_list:
        if fact.kind in _ASSETS and set(fact.squares) & left:
            continue
        if set(fact.squares) & touched:
            return fact.text
    return None


# Features that say a piece stands well where it is: not the point of a move that takes it away.
_ASSETS = frozenset({"openFile", "halfOpenFile", "outpost"})


def mate_key(board: chess.Board, idea: Idea) -> str | None:
    """Their king and its room, counted, and your checks: "their king on f1 has three free
    squares, and you have two checks". The squares themselves come with the second hint: one
    of them can be where your check lands (e2+ with e2 among the king's squares)."""
    if idea.target is None:
        return None
    room = len(idea.flight)
    squares = (
        "no free square"
        if room == 0
        else ("one free square" if room == 1 else f"{_NUMBER.get(room, str(room))} free squares")
    )
    # The checks are counted only for a line that starts with one: before a quiet first move
    # the count sends the search the wrong way (the audit, 2026-10-07).
    checks = idea.checks if idea.checks_first else 0
    said = (
        f", and you have {_NUMBER.get(checks, str(checks))} checks"
        if checks > 1
        else (", and you have one check" if checks == 1 else "")
    )
    return f"Their king on {_sq(idea.target)} has {squares}{said}"


def _checks_after(board: chess.Board, move: chess.Move) -> int:
    """The checks the opponent has once `move` is played."""
    after = board.copy(stack=False)
    after.push(move)
    return sum(1 for reply in after.legal_moves if after.gives_check(reply))


def evasion_why(board: chess.Board, best: chess.Move) -> str | None:
    """What the way out of check does, at the move rung: takes the checker, blocks, or steps
    away ("The move is Re4." said nothing). A king step says what sets its square apart only
    when something does: every legal king move lands where nothing attacks it, so "a square
    nothing of theirs can reach" read as a reason no other king move had (dev audit,
    2026-09-26)."""
    if not board.is_check():
        return None
    san = board.san(best)
    checkers = board.checkers()
    if best.to_square in checkers:
        return f"{san} takes the piece giving check"
    if board.piece_type_at(best.from_square) == chess.KING:
        others = [
            m
            for m in board.legal_moves
            if m != best and board.piece_type_at(m.from_square) == chess.KING
        ]
        if (
            others
            and not _checks_after(board, best)
            and all(_checks_after(board, m) for m in others)
        ):
            return f"{san} steps out of check to the one square where no check follows"
        return f"{san} steps out of check"
    return f"{san} blocks the check"


def _captured(board: chess.Board, line: Sequence[chess.Move], ply: int) -> chess.Square | None:
    """The square of the enemy piece the player's move at `ply` takes, as it stands now."""
    b = board.copy(stack=False)
    for j, move in enumerate(line):
        if move not in b.legal_moves:
            return None
        if j == ply:
            if not b.is_capture(move):
                return None
            # Followed back to where it stands now, through its own moves; an en passant
            # capture takes the pawn beside the square it lands on.
            at = move.to_square
            if b.is_en_passant(move):
                at = chess.square(
                    chess.square_file(move.to_square), chess.square_rank(move.from_square)
                )
            for k in range(j - 1, -1, -1):
                if k % 2 == 1 and line[k].to_square == at:
                    at = line[k].from_square
            piece = board.piece_at(at)
            return at if piece is not None and piece.color != board.turn else None
        b.push(move)
    return None


def win_key(board: chess.Board, idea: Idea) -> str | None:
    """What a win without a named pattern takes: "their pawn on d5 is the target"."""
    line = list(idea.line)
    for ply in (idea.gain_ply, *range(0, min(len(line), 8), 2)):
        sq = _captured(board, line, ply)
        if sq is not None:
            return f"Their {_named(board, sq)} is the target"
    return None


def convert_key(board: chess.Board, idea: Idea) -> str | None:
    """The trade that leaves the won ending: the first of theirs the player's line takes."""
    line = list(idea.line)
    for ply in range(0, min(len(line), 8), 2):
        sq = _captured(board, line, ply)
        if sq is not None:
            return f"Their {_named(board, sq)} is the piece to trade off"
    return None


def king_key(board: chess.Board, idea: Idea) -> str | None:
    """An ending the king decides: the kings face to face, or their pawns in its reach."""
    us = board.turn
    ours, theirs = board.king(us), board.king(not us)
    if ours is None or theirs is None:
        return None
    if idea.opposition:
        return f"The kings stand close: your king on {_sq(ours)}, theirs on {_sq(theirs)}"
    pawns = sorted(board.pieces(chess.PAWN, not us), key=lambda sq: chess.square_distance(ours, sq))
    if pawns:
        near = _listed([_sq(sq) for sq in pawns[:2]])
        return (
            f"Your king on {_sq(ours)} is your best piece here: their pawns on {near} are in "
            "its reach"
        )
    return f"Your king on {_sq(ours)} is your best piece here"


def check_key(board: chess.Board, prize: chess.Square | None = None) -> str | None:
    """The piece giving check -- and, when it is what the line wins (`prize`), said so once."""
    checkers = list(board.checkers())
    if len(checkers) == 1:
        if checkers[0] == prize:
            return f"Their {_named(board, checkers[0])} gives check, and it is what you win"
        return f"Their {_named(board, checkers[0])} gives check"
    if len(checkers) == 2:
        first, second = (_named(board, sq) for sq in checkers)
        return f"Their {first} and their {second} both give check: only a king move gets out"
    return None


def their_idea_key(
    board: chess.Board,
    threat: chess.Move | None,
    *,
    sharp: bool = False,
    best: chess.Move | None = None,
) -> str | None:
    """What they would do if you let them: the prophylactic question, answered on the board.
    Where every other move loses (`sharp`), whatever it is; else only a move that changes
    something -- a capture, a check, a pawn move: "if you let them, their best is Bd7" said
    nothing (Johner-Nimzowitsch, eight times). Never when it names the square of the `best`
    move: "If you let them, their best is h4." was the first hint where 11...Qh4+ was the
    answer (Wiel-Vaganian; the audit, 2026-10-07: a hint withholds the move's square)."""
    if threat is None or board.is_check():
        return None
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    if threat not in passed.legal_moves:
        return None
    if not (
        sharp
        or passed.is_capture(threat)
        or passed.gives_check(threat)
        or passed.piece_type_at(threat.from_square) == chess.PAWN
    ):
        return None
    said = passed.san(threat)
    if best is not None and chess.square_name(best.to_square) in said:
        return None
    return f"If you let them, their best is {said}"


def kings_key(board: chess.Board, best: chess.Move) -> str | None:
    """A pawn ending's king move with the kings close: the standoff, named (Lichess R8gGn,
    ...Ke6 taking the opposition)."""
    if any(p.piece_type not in (chess.KING, chess.PAWN) for p in board.piece_map().values()):
        return None
    if board.piece_type_at(best.from_square) != chess.KING:
        return None
    ours, theirs = board.king(board.turn), board.king(not board.turn)
    if ours is None or theirs is None or chess.square_distance(ours, theirs) > 3:
        return None
    return (
        f"The kings stand close, yours on {_sq(ours)} and theirs on {_sq(theirs)}: whichever "
        "has to give way loses ground"
    )


def passer_key(board: chess.Board, best: chess.Move) -> str | None:
    """Their passed pawn near queening, when the best move takes it: what makes the capture
    the point (Lichess UkpFL, Rxe3+ against a pawn two squares from e1)."""
    sq = best.to_square
    if not board.is_capture(best) or board.piece_type_at(sq) != chess.PAWN:
        return None
    if board.color_at(sq) == board.turn or not positional.is_passed(board, sq):
        return None
    rank = chess.square_rank(sq)
    to_go = rank if board.turn == chess.WHITE else 7 - rank
    if to_go > 3:
        return None
    squares = "one square" if to_go == 1 else f"{_NUMBER.get(to_go, str(to_go))} squares"
    return f"Their passed pawn on {_sq(sq)} is {squares} from queening"
