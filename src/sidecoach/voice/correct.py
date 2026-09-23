"""Chess-aware correction of a raw transcript, against the moves that are legal right now.

A speech recogniser hears English, not chess: "knight" comes back as "night" or "9", "e takes
d five" as "E-Tix D5", "the a-pawn" as "the upon". The fixes are of two kinds. First, a table
of known confusions -- homophones and glued tokens -- that is safe in any chess sentence.
Second, and the real one: every legal move on the board has a handful of spoken forms
("knight f three", "knight to f three", "bishop takes c six", "castle kingside"), and a span
of the transcript that matches one of them closely is rewritten as that move's SAN. The
board is the ground truth, so the correction can only ever produce a move that exists; the
rest of the sentence is left as the user said it, for the coach and its claim checker.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import chess

from sidecoach.voice.notation import speak_notation

_WORDS = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
}
_NUMBER = {v: k for k, v in _WORDS.items()}
# What a number sounds like after a file letter ("e for" is e4, "knight f tree" is f3).
_NUMBER_LIKE = {
    "for": "four",
    "fore": "four",
    "to": "two",
    "too": "two",
    "tree": "three",
    "free": "three",
    "won": "one",
    "ate": "eight",
    "sex": "six",
    "sevin": "seven",
}
# What a file letter sounds like before a number ("see six" is c6).
_LETTER_LIKE = {
    "eh": "a",
    "ay": "a",
    "bee": "b",
    "be": "b",
    "see": "c",
    "sea": "c",
    "dee": "d",
    "gee": "g",
    "aitch": "h",
    "age": "h",
}
_ANY_NUMBER = "|".join([*_WORDS.values(), *_WORDS.keys(), *_NUMBER_LIKE.keys()])

# Known confusions, applied before matching. Each is safe in any chess sentence.
_CONFUSIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(night|nite|nights)\b", re.I), "knight"),
    (re.compile(r"\b(nine|9)\b(?=\s+(?:on|to|from|takes|[a-h]\b))", re.I), "knight"),
    (re.compile(r"\bwith the (?:nine|9)\b", re.I), "with the knight"),
    (re.compile(r"\b(rock|rocks)\b(?=\s+(?:on|to|[a-h]\b|takes))", re.I), "rook"),
    (re.compile(r"\b(?:rooka|roka) ?([1-8])\b", re.I), r"rook a\1"),  # "Roka 1" is rook + a1
    (re.compile(r"\b(?:rook|rock) ?([a-h]) ?([1-8])\b", re.I), r"rook \1\2"),
    (re.compile(r"\bthe (?:upon|appawn|a-pawn|apon)\b", re.I), "the a pawn"),
    (re.compile(r"\bczech\b", re.I), "check"),
    (re.compile(r"\bcheque\b", re.I), "check"),
    (re.compile(r"\bking'?s? ?side\b", re.I), "kingside"),
    (re.compile(r"\bqueen'?s? ?side\b", re.I), "queenside"),
    (re.compile(r"\bcastle (?:short|shorts)\b", re.I), "castle kingside"),
    (re.compile(r"\bcastle long\b", re.I), "castle queenside"),
    (re.compile(r"\b([a-h])[- ]?(?:takes|tix|tex|tics|tx|x)\b", re.I), r"\1 takes"),
    (re.compile(r"\b([a-h])x([a-h])\s*(" + _ANY_NUMBER + r")\b", re.I), r"\1 takes \2 \3"),
    (re.compile(r"\bcaptures\b", re.I), "takes"),
]


def normalise(text: str) -> str:
    """Lowercase, punctuation stripped, confusions fixed, squares as 'e four'."""
    t = text.lower()
    t = re.sub(r"[^\w\s'-]", " ", t)
    t = t.replace("-", " ")
    for pattern, repl in _CONFUSIONS:
        t = pattern.sub(repl, t)
    # A file letter followed by something that sounds like a number is a square.
    t = re.sub(
        r"\b([a-h]) ?(" + "|".join(_NUMBER_LIKE) + r")\b",
        lambda m: f"{m.group(1)} {_NUMBER_LIKE[m.group(2)]}",
        t,
    )
    t = re.sub(
        r"\b(" + "|".join(_LETTER_LIKE) + r") (" + "|".join(_WORDS.values()) + r")\b",
        lambda m: f"{_LETTER_LIKE[m.group(1)]} {m.group(2)}",
        t,
    )
    t = re.sub(r"\b([a-h]) ?([1-8])\b", lambda m: f"{m.group(1)} {_WORDS[m.group(2)]}", t)
    return re.sub(r"\s+", " ", t).strip()


def spoken_forms(board: chess.Board) -> dict[str, str]:
    """Every way a legal move might be said -> its SAN."""
    forms: dict[str, str] = {}
    for move in board.legal_moves:
        san = board.san(move)
        base = speak_notation(san)  # "knight f three", "bishop takes c six", "castles kingside"
        variants = {base}
        stem = base.removesuffix(" check").removesuffix(" checkmate")
        variants.add(stem)
        if san.startswith("O-O"):
            side = "queenside" if san.startswith("O-O-O") else "kingside"
            variants |= {f"castle {side}", f"castles {side}", f"castling {side}", side}
        else:
            piece = chess.piece_name(board.piece_type_at(move.from_square) or chess.PAWN)
            to = chess.square_name(move.to_square)
            dest = f"{to[0]} {_WORDS[to[1]]}"
            if piece == "pawn":
                variants |= {f"pawn {dest}", f"pawn to {dest}", dest}
                if board.is_capture(move):
                    variants.add(f"pawn takes {dest}")
            else:
                variants |= {f"{piece} to {dest}", f"{piece} {dest}"}
                if board.is_capture(move):
                    variants |= {f"{piece} takes {dest}", f"{piece} takes on {dest}"}
            if board.is_capture(move) and piece != "pawn":
                victim = board.piece_type_at(move.to_square)
                if victim:
                    variants.add(f"{piece} takes {chess.piece_name(victim)}")
                    variants.add(f"{piece} takes the {chess.piece_name(victim)}")
        promotion = move.promotion
        if promotion:
            # What the pawn becomes is said after the square too: "e eight queen", "e eight
            # equals queen", as well as the notation's "e eight promotes to queen".
            name = chess.piece_name(promotion)
            head = stem.split(" promotes to ")[0]
            variants |= {f"{head} {name}", f"{head} equals {name}"}
        for v in variants:
            # A promotion that does not say to what is a queen's: the forms that name no piece
            # belong to it alone. The knight's, generated last, used to take them all, and
            # "e8 queen" came out as e8=N+ (2026-09-23).
            if promotion and promotion != chess.QUEEN and chess.piece_name(promotion) not in v:
                continue
            forms[v] = san
    return forms


@dataclass
class Correction:
    text: str  # the sentence with spoken moves rewritten as SAN
    moves: list[str] = field(default_factory=list)  # the SAN moves found, in order
    raw: str = ""


_MIN_RATIO = 0.86


def correct(text: str, board: chess.Board | None) -> Correction:
    """Fix known confusions and rewrite any spoken legal move as its SAN.

    Without a board only the confusion table applies. Matching is greedy over the longest
    spans first; a span needs a square (or a castling word) in it before it is compared, so
    ordinary sentences are never touched.
    """
    norm = normalise(text)
    if board is None:
        return Correction(text=norm, raw=text)
    forms = spoken_forms(board)
    tokens = norm.split()
    out: list[str] = []
    moves: list[str] = []
    i = 0
    while i < len(tokens):
        best: tuple[float, int, str] | None = None  # (ratio, span length, san)
        for n in range(min(5, len(tokens) - i), 0, -1):
            span = tokens[i : i + n]
            if not _plausible(span):
                continue
            if n == 2 and i > 0 and tokens[i - 1] == "on" and len(span[0]) == 1:
                continue  # "the pawn on d five" is a place, not a move there
            phrase = " ".join(span)
            if phrase in forms:
                best = (1.0, n, forms[phrase])
                break
            keys = _keys(span)
            for form, san in forms.items():
                ratio = difflib.SequenceMatcher(None, phrase, form).ratio()
                # Close is not enough: the piece and the square the person named must be the
                # form's. Fuzz may mend how the words around them were heard, never swap them
                # for another legal move -- "knight to f three" became Kf3 when only the king
                # could go there (Tai, 2026-09-23: what matches no legal move stays as words,
                # for the coach to read).
                if (
                    ratio >= _MIN_RATIO
                    and (best is None or ratio > best[0])
                    and _keys(form.split()) == keys
                ):
                    best = (ratio, n, san)
        if best is not None and best[0] >= _MIN_RATIO:
            out.append(best[2])
            moves.append(best[2])
            i += best[1]
        else:
            out.append(tokens[i])
            i += 1
    return Correction(text=" ".join(out), moves=moves, raw=text)


_SQUARE_WORDS = set(_WORDS.values())
_CASTLE_WORDS = {"castle", "castles", "castling", "kingside", "queenside"}
# The only words a spoken move is made of. A span with anything else in it is not compared,
# so "play knight to f three here" matches on "knight to f three" alone and never swallows
# the words around it.
_MOVE_VOCAB = (
    _SQUARE_WORDS
    | _CASTLE_WORDS
    | set("abcdefgh")
    | {
        "knight",
        "bishop",
        "rook",
        "queen",
        "king",
        "pawn",
        "takes",
        "to",
        "on",
        "the",
        "x",
        "check",
        "checkmate",
        "promotes",
        "promote",
        "equals",
    }
)


# The words that say which move: the piece, the square, the side castled to, what a pawn
# becomes -- and "on", which makes a square a place ("the pawn on e five") rather than where a
# move goes. A fuzzy match must keep all of them as they were said.
_KEY_WORDS = (
    _SQUARE_WORDS
    | set("abcdefgh")
    | {"knight", "bishop", "rook", "queen", "king", "pawn", "kingside", "queenside", "on"}
)


def _keys(words: list[str]) -> list[str]:
    return [w for w in words if w in _KEY_WORDS]


def _plausible(span: list[str]) -> bool:
    """A span worth comparing to the move forms: move words only, naming a square or castling."""
    if not all(w in _MOVE_VOCAB for w in span):
        return False
    if any(w in _CASTLE_WORDS for w in span):
        return True
    for a, b in zip(span, span[1:], strict=False):
        if len(a) == 1 and a in "abcdefgh" and b in _SQUARE_WORDS:
            return True
    return False
