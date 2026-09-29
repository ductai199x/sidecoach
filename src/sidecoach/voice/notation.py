"""Turn chess notation into words before it is spoken.

A text-to-speech voice reads "Nf3" as "enn eff three" and "O-O" as "oh oh"; the coach's
prose is full of such tokens. This converts the standard algebraic notation embedded in a
line -- piece moves, captures, castling, checks, promotions, en passant, and bare squares --
into their spoken form, and leaves the surrounding prose untouched. Only tokens shaped like
notation are matched, so an ordinary word or a number in the sentence is never mangled.

Two things the voice needed on top (Tai, 2026-09-29: "dxe5 pronounced like d-ai-ex-ee-five ...
anything involving the a-file would be pronounce 'er'"). A square or a pawn move written with
a capital, as the model starts a sentence ("G4 was a miss", "Dxe5 was better"), matched no
move and was spelled out letter by letter; it is read as the square or the pawn move it is.
And the file letter a on its own is the article to the voice's phonemizer (espeak: "uh"), so
"rook a four" was heard as "rook of fours": the voice is given "eh" instead, which the
phonemizer reads as the letter. That one is for the voice only (`voice=True`): the speech
correction matches what a person says against these words, and a person says "a four".
"""

from __future__ import annotations

import re

_PIECE = {
    "K": "king",
    "Q": "queen",
    "R": "rook",
    "B": "bishop",
    "N": "knight",
    "P": "pawn",
}
_RANK = {
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
}

# A SAN move: an optional piece, an optional from-file and/or from-rank for disambiguation, an
# optional capture, the destination square, an optional promotion, and an optional check or
# mate mark. Castling and game results are handled separately. Word-bounded so it matches a
# token, not a fragment of a longer word.
_MOVE = re.compile(
    r"\b([KQRBNP])?([a-h])?([1-8])?(x)?([a-h])([1-8])(=[QRBN])?([+#])?(?![A-Za-z0-9])"
)
# A square or a pawn move with a capital file letter, as a sentence's first word has it: "G4",
# "Dxe5", "Exd8=Q+". A capital B before a capture is a bishop's ("Bxc6" is SAN); straight before
# a rank it is the square. The whole token is checked, so an opening's code ("A40") is not one.
_CAPITAL = re.compile(r"\b([A-H])(?=(x[a-h])?[1-8](?:=[QRBN])?[+#]?(?![A-Za-z0-9]))")
# The file letter a as the voice reads it (see the module's note).
_VOICED = {"a": "eh"}
_CASTLE = re.compile(r"\b(O-O-O|O-O)([+#])?(?![A-Za-z0-9])")
_RESULT = re.compile(r"\b(1-0|0-1|1/2-1/2|½-½)\b")


def _file(letter: str, voice: bool) -> str:
    return _VOICED.get(letter, letter) if voice else letter


def _square(file: str, rank: str, voice: bool) -> str:
    return f"{_file(file, voice)} {_RANK[rank]}"


def _lower(match: re.Match[str]) -> str:
    letter = match.group(1)
    return letter if letter == "B" and match.group(2) else letter.lower()


def _mark(mark: str | None) -> str:
    if mark == "+":
        return " check"
    if mark == "#":
        return " checkmate"
    return ""


def _castle(match: re.Match[str]) -> str:
    side = "queenside" if match.group(1) == "O-O-O" else "kingside"
    return f"castles {side}{_mark(match.group(2))}"


def _move(match: re.Match[str], voice: bool) -> str:
    piece, from_file, from_rank, capture, to_file, to_rank, promo, mark = match.groups()
    # A bare square with none of a piece, capture or promotion is a pawn move (or a square the
    # prose just names); anything with only a destination and nothing else is still valid SAN.
    words: list[str] = []
    if piece:
        words.append(_PIECE[piece])
    if from_file:
        words.append(_file(from_file, voice))
    if from_rank:
        words.append(_RANK[from_rank])
    if capture:
        words.append("takes")
    words.append(_square(to_file, to_rank, voice))
    text = " ".join(words)
    if promo:
        text += f" promotes to {_PIECE[promo[1]]}"
    return text + _mark(mark)


def _result(match: re.Match[str]) -> str:
    return {
        "1-0": "white wins",
        "0-1": "black wins",
        "1/2-1/2": "draw",
        "½-½": "draw",
    }[match.group(1)]


def speak_notation(text: str, *, voice: bool = False) -> str:
    """`text` with every chess-notation token replaced by how it should be read aloud; `voice`,
    as the voice is given it (the file letter a as "eh").

    Order matters: results and castling are recognised before the general move pattern, which
    would otherwise swallow part of them, and a capital file letter is made the letter it is
    before it. Everything the patterns do not match is left as is.
    """
    text = _RESULT.sub(_result, text)
    text = _CASTLE.sub(_castle, text)
    text = _CAPITAL.sub(_lower, text)
    return _MOVE.sub(lambda m: _move(m, voice), text)
