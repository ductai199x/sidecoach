"""The AI coach's words as the review page draws them (docs/game-review-plan.md, decision 3):
each move, piece, square, file and chess term in a paragraph a link to the board (Tai,
2026-09-30: "hovering or tapping/clicking on the highlights must also interact with the board").

A move draws its arrow, coloured by whose it is -- the better move's line gold, the student's
blue, their opponent's red -- and lights its two squares, read in the positions the paragraph can
mean: before the move, after it, along the reply's line and the better line, and either side's
threat (a move for the side not to move). A piece on a square or a bare square lights its square;
a file spans it end to end (a double-ended arrow, as chess notation marks one); a coordinate that
is a move of those lines ("the move c4") is the move. A term carries its meaning, and what it
names on the board: the pawns of a pawn structure ("your doubled c-pawns", read in the position
after the move), the file it names ("the half-open d-file"), else the squares of the move's own
motif of that name when the review tagged one ("the fork" -> the fork's squares).
The paragraph comes back as its pieces: plain text as it is, a link as a dict the page draws.
"""

from __future__ import annotations

import re
from typing import Any

import chess

from sidecoach.events import ReviewedGame

LINE_PLIES = 6

# A term, its meaning, and the review's motif tags it names (their squares lit).
TERMS: dict[str, tuple[str, tuple[str, ...]]] = {
    "fork": ("One piece attacks two or more at once; only one can be saved.", ("fork",)),
    "pin": (
        "A piece that cannot move without exposing a more valuable piece behind it.",
        ("pin",),
    ),
    "skewer": (
        "A valuable piece attacked in a line: when it moves, the piece behind it falls.",
        ("skewer",),
    ),
    "discovered attack": (
        "Moving one piece uncovers an attack by another behind it.",
        ("discoveredAttack", "discoveredCheck"),
    ),
    "double attack": ("Two threats at once; only one can be met.", ("fork", "doubleCheck")),
    "back rank": (
        "The king's first rank: a king shut in by its own pawns can be mated there.",
        ("backRankMate",),
    ),
    "sacrifice": (
        "Material given up on purpose for something worth more: an attack, a mate, a won ending.",
        ("sacrifice",),
    ),
    "passed pawn": ("A pawn no enemy pawn can stop on its way to promotion.", ("advancedPawn",)),
    "isolated pawn": (
        "A pawn with no pawn of its side on the files beside it: only pieces can defend it.",
        (),
    ),
    "doubled pawns": ("Two pawns of one side on one file: they cannot protect each other.", ()),
    "backward pawn": (
        "A pawn left behind its neighbours, which no pawn can protect; the square in front of "
        "it is a hole.",
        (),
    ),
    "hanging pawns": (
        "Two pawns side by side with no pawns beside them: strong while they can advance, "
        "targets once stopped.",
        (),
    ),
    "hanging": ("Attacked and not defended: it can be taken for free.", ("hangingPiece",)),
    "trapped": ("A piece with no safe square left: it will be won.", ("trappedPiece",)),
    "outpost": ("A square no enemy pawn can attack: a home for a knight.", ()),
    "open file": ("A file with no pawns on it: a road for the rooks.", ()),
    "half-open file": ("A file with only the opponent's pawns on it: pressure for the rooks.", ()),
    "bishop pair": ("Both bishops against a bishop and a knight, or two knights.", ()),
    "bad bishop": ("A bishop hemmed in by its own pawns on its colour.", ()),
    "initiative": (
        "Making the threats: the other side has to answer instead of carrying out its plans.",
        (),
    ),
    "tempo": ("A move's worth of time.", ()),
    "zugzwang": ("Every move makes things worse, and a move must be made.", ("zugzwang",)),
    "prophylaxis": ("A move that stops the opponent's plan before it starts.", ()),
    "fianchetto": ("A bishop developed on the long diagonal, from b2, g2, b7 or g7.", ()),
    "king safety": ("How well the king is sheltered from checks and attacks.", ("exposedKing",)),
    "pawn break": ("A pawn advance that challenges the opponent's pawns and opens lines.", ()),
    "pawn majority": ("More pawns than the opponent on one wing: a passed pawn in the making.", ()),
    "promotion": (
        "A pawn reaching the last rank becomes a queen, or another piece.",
        ("promotion",),
    ),
    "deflection": ("Forcing a defender away from what it guards.", ("deflection",)),
    "decoy": ("Luring a piece onto a square where it can be hit.", ("attraction",)),
    "overloaded": ("A piece with more jobs to do than it can do at once.", ("overloading",)),
    "interference": ("A piece put between an enemy piece and what it defends.", ("interference",)),
    "in-between move": (
        "An unexpected move played before the expected reply.",
        ("intermezzo",),
    ),
    "quiet move": ("A move with no check or capture that still carries a threat.", ("quietMove",)),
    "counterplay": ("Threats of one's own that make the opponent defend.", ()),
    "closed centre": ("Centre pawns locked against each other: the play moves to the wings.", ()),
    "opposition": ("Kings facing each other one square apart: the side not to move has it.", ()),
    "weak square": ("A square no pawn can defend any more: a post for the opponent's pieces.", ()),
    "the exchange": ("A rook for a bishop or a knight.", ()),
}
# Each term's spellings: plurals, verbs, the American centre.
_SPELLINGS = {
    "fork": r"fork(?:s|ed|ing)?",
    "pin": r"pin(?:s|ned|ning)?",
    "skewer": r"skewer(?:s|ed|ing)?",
    "discovered attack": r"discovered (?:attack|check)s?",
    "double attack": r"double attacks?",
    "back rank": r"back[- ]rank",
    "sacrifice": r"sacrific(?:e|es|ed|ing)",
    "passed pawn": r"passed (?:pawns?|[a-h]-pawns?)|passers?",
    "isolated pawn": r"isolated (?:pawns?|[a-h]-pawns?)",
    "doubled pawns": r"doubled (?:pawns|[a-h]-pawns)",
    "backward pawn": r"backward (?:pawns?|[a-h]-pawns?)",
    "hanging pawns": r"hanging pawns",
    "hanging": r"hanging|loose",
    "trapped": r"trapped",
    "outpost": r"outposts?",
    "open file": r"open (?:[a-h]-)?files?",
    "half-open file": r"(?:half|semi)-open (?:[a-h]-)?files?",
    "bishop pair": r"bishop pair|pair of bishops",
    "bad bishop": r"bad bishops?",
    "initiative": r"initiative",
    "tempo": r"tempo|tempi",
    "zugzwang": r"zugzwang",
    "prophylaxis": r"prophyla(?:xis|ctic)",
    "fianchetto": r"fianchett(?:o|oed)",
    "king safety": r"king safety",
    "pawn break": r"pawn breaks?",
    "pawn majority": r"pawn majority|(?:queenside|kingside) majority",
    "promotion": r"promot(?:ion|e|es|ed|ing)|queening",
    "deflection": r"deflect(?:ion|s|ed|ing)?",
    "decoy": r"decoys?",
    "overloaded": r"overload(?:ed|ing)?",
    "interference": r"interference",
    "in-between move": r"in-between moves?|zwischenzug|intermezzo",
    "quiet move": r"quiet moves?",
    "counterplay": r"counterplay",
    "closed centre": r"closed cent(?:re|er)",
    "opposition": r"the opposition",
    "weak square": r"weak squares?|holes?",
    "the exchange": r"the exchange",
}
_TERM = re.compile(
    r"\b(?:"
    + "|".join(
        f"(?P<t{i}>{_SPELLINGS[name]})"
        for i, name in sorted(enumerate(TERMS), key=lambda p: -len(_SPELLINGS[p[1]]))
    )
    + r")\b",
    re.IGNORECASE,
)
_TERM_NAMES = list(TERMS)

_PIECE = (
    r"\b(?:(?:your opponent's|your|their|the|White's|Black's|White|Black)\s+)?"
    r"(?:(?:isolated|passed|backward|doubled|loose|hanging|pinned|trapped|advanced|extra|lone|"
    r"last|remaining|bad|good|light-squared|dark-squared|active|passive|strong|weak|exposed)\s+)"
    r"{0,2}(?:[a-h]-)?\b(?:king|queen|rook|bishop|knight|pawn)s?\s+(?:on|at)\s+(?P<sq>[a-h][1-8])\b"
)
# Moves, squares and files in their own case: "b" is a file, "B" a bishop.
_SAN = (
    r"(?-i:(?<![\w-])(?P<san>O-O-O|O-O|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?"
    r"|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][18]=[QRBN][+#]?)(?![\w-]))"
)
_COORD = r"(?-i:(?<![\w-])(?P<coord>[a-h][1-8])(?![\w-]))"
_FILE = r"(?-i:\b(?P<file>[a-h])-file\b)"
_LINKS = re.compile(f"(?P<piece>{_PIECE})|{_SAN}|{_FILE}|{_COORD}", re.IGNORECASE)


def _side(board: chess.Board, user: str | None) -> str:
    """A move's colour by whose it is: the student's blue, their opponent's red."""
    side = "white" if board.turn == chess.WHITE else "black"
    return "idea" if user is None or side == user else "threat"


def _arrow(board: chess.Board, move: chess.Move, role: str) -> dict[str, Any]:
    start, end = chess.square_name(move.from_square), chess.square_name(move.to_square)
    return {"kind": "move", "arrow": [start, end, role]}


# The pawn structures a term names, and whose pawns it means by the word before it.
_STRUCTURES = {"passed pawn", "isolated pawn", "doubled pawns", "backward pawn", "hanging pawns"}
_OWNER = re.compile(r"(your opponent's|your|their|White's|Black's)\s+(?:\w+\s+)?$", re.IGNORECASE)


def _owners(before: str, user: str | None) -> list[chess.Color]:
    """Whose pawns a term means, by the word before it ("your doubled pawns"); both sides'
    when it names no one."""
    found = _OWNER.search(before)
    word = found.group(1).lower() if found else ""
    us = None if user is None else chess.WHITE if user == "white" else chess.BLACK
    if word.startswith(("white", "black")):
        return [chess.WHITE if word.startswith("white") else chess.BLACK]
    if us is not None and word == "your":
        return [us]
    if us is not None and word in ("your opponent's", "their"):
        return [not us]
    return [chess.WHITE, chess.BLACK]


def _pawns(board: chess.Board, colour: chess.Color, structure: str, file: int | None) -> set[int]:
    """The pawns of `colour` in a structure: doubled (another of theirs on the file), isolated
    (none of theirs beside it), passed (no enemy pawn ahead on its file or beside it); a
    backward or hanging pawn only by the file named."""
    own = board.pieces(chess.PAWN, colour)
    enemy = board.pieces(chess.PAWN, not colour)
    files = [chess.square_file(s) for s in own]
    out = set()
    for square in own:
        f, rank = chess.square_file(square), chess.square_rank(square)
        if file is not None and f != file:
            continue
        ahead = range(rank + 1, 8) if colour == chess.WHITE else range(rank)
        if structure == "doubled pawns":
            fits = files.count(f) > 1
        elif structure == "isolated pawn":
            fits = not any(abs(g - f) == 1 for g in files)
        elif structure == "passed pawn":
            fits = not any(
                chess.square(g, r) in enemy for g in (f - 1, f, f + 1) if 0 <= g < 8 for r in ahead
            )
        else:
            fits = file is not None
        if fits:
            out.add(square)
    return out


def _moves(game: ReviewedGame, ply: int, user: str | None) -> dict[str, dict[str, Any]]:
    """The moves a paragraph on move `ply` names, by SAN (checks left off): the move, the better
    move's line (gold), the reply's line; the first naming wins."""
    move = game.moves[ply - 1]
    before = chess.Board(move.fen_before)
    out: dict[str, dict[str, Any]] = {}

    def walk(start: chess.Board, line: list[str], role: str | None) -> None:
        for board, played in _along(start, line):
            san = board.san(played).rstrip("+#")
            out.setdefault(san, _arrow(board, played, role or _side(board, user)))

    walk(before, [move.assessment.uci], None)
    walk(before, move.assessment.best[:LINE_PLIES], "best")
    reply = move.after.lines[0].moves[:LINE_PLIES] if move.after.lines else []
    walk(chess.Board(move.after.fen), reply, None)
    return out


def _along(start: chess.Board, line: list[str]) -> list[tuple[chess.Board, chess.Move]]:
    """Each move of a line with the position it is played from, as far as it stays legal."""
    board = start.copy(stack=False)
    out = []
    for uci in line:
        played = chess.Move.from_uci(uci)
        if played not in board.legal_moves:
            break
        out.append((board.copy(stack=False), played))
        board.push(played)
    return out


def _elsewhere(game: ReviewedGame, ply: int) -> list[chess.Board]:
    """The positions a move not of those lines is read in: before the move, after it, along
    both lines, and in each of them the side not to move's threat ("Nd2 threatens Nxe4")."""
    move = game.moves[ply - 1]
    boards = [chess.Board(move.fen_before), chess.Board(move.after.fen)]
    reply = move.after.lines[0].moves[:LINE_PLIES] if move.after.lines else []
    for start, line in ((boards[0], move.assessment.best[:LINE_PLIES]), (boards[1], reply)):
        for board, played in _along(start, line):
            after = board.copy(stack=False)
            after.push(played)
            boards.append(after)
    for board in list(boards):
        if not board.is_check():
            passed = board.copy(stack=False)
            passed.push(chess.Move.null())
            boards.append(passed)
    return boards


def _move(
    san: str,
    named: dict[str, dict[str, Any]],
    elsewhere: list[chess.Board],
    user: str | None,
) -> dict[str, Any] | None:
    """A move's link: its arrow and colour, from the game's lines or the first position it is
    legal in; None when it is legal nowhere the paragraph can mean."""
    known = named.get(san.rstrip("+#"))
    if known is not None:
        return {"text": san, **known}
    for board in elsewhere:
        try:
            move = board.parse_san(san)
        except ValueError:
            continue
        return {"text": san, **_arrow(board, move, _side(board, user))}
    return None


def links(
    text: str, game: ReviewedGame, ply: int, user: str | None, *, general: bool = False
) -> list[str | dict[str, Any]]:
    """`text` (a paragraph in the student's words, on move `ply`) as its pieces: plain text as
    it is, each link a dict -- "text", "kind" (move, piece, square, file, term) and what it does
    on the board ("squares" lit, a move's "arrow" [from, to, colour], a file's "span" [end,
    end]; a term's "meaning"). A `general` paragraph (the lesson) lights nothing of this game."""
    named = _moves(game, ply, user)
    elsewhere = _elsewhere(game, ply)
    tags = [] if general else game.moves[ply - 1].assessment.tags
    found: list[tuple[int, int, dict[str, Any]]] = []
    for match in _LINKS.finditer(text):
        link: dict[str, Any] | None = None
        if match.group("piece"):
            link = {"text": match.group(0), "kind": "piece", "squares": [match.group("sq")]}
        elif match.group("san"):
            link = _move(match.group("san"), named, elsewhere, user)
        elif match.group("file"):
            file = match.group("file").lower()
            link = {"text": match.group(0), "kind": "file", "span": [f"{file}1", f"{file}8"]}
        elif match.group("coord"):
            coord = match.group("coord")
            moved = _move(coord, named, [], user) if coord in named else None
            link = moved or {"text": coord, "kind": "square", "squares": [coord.lower()]}
        if link is not None:
            found.append((match.start(), match.end(), link))
    move = game.moves[ply - 1]
    shown = (chess.Board(move.after.fen), chess.Board(move.fen_before))
    for match in _TERM.finditer(text):
        name = next(_TERM_NAMES[int(k[1:])] for k, v in match.groupdict().items() if v)
        meaning, motifs = TERMS[name]
        link = {"text": match.group(0), "kind": "term", "meaning": meaning}
        on_file = re.search(r"\b([a-h])-(?:file|pawn)", match.group(0))
        if on_file and "file" in name:
            # A file the term names: "the half-open d-file".
            link["span"] = [f"{on_file.group(1)}1", f"{on_file.group(1)}8"]
        elif name in _STRUCTURES and not general:
            # Its pawns as the board shows them (after the move), else as they stood before.
            file = chess.FILE_NAMES.index(on_file.group(1)) if on_file else None
            owners = _owners(text[: match.start()], user)
            for board in shown:
                pawns = {sq for c in owners for sq in _pawns(board, c, name, file)}
                if pawns:
                    link["squares"] = sorted(chess.square_name(sq) for sq in pawns)
                    break
        else:
            squares = sorted({sq for tag in tags if tag.name in motifs for sq in tag.squares})
            if squares:
                link["squares"] = squares
        found.append((match.start(), match.end(), link))
    # The earliest link wins where two overlap ("the passed pawn on e5" is a piece, not a term).
    out: list[str | dict[str, Any]] = []
    at = 0
    for start, end, link in sorted(found, key=lambda f: (f[0], -f[1])):
        if start < at:
            continue
        if start > at:
            out.append(text[at:start])
        out.append(link)
        at = end
    if at < len(text):
        out.append(text[at:])
    return out


def linked(
    texts: dict[str, dict[str, str]], game: ReviewedGame, user: str | None
) -> dict[str, dict[str, list[str | dict[str, Any]]]]:
    """Every paragraph of the saved words (by ply, then part) as its links."""
    return {
        ply: {
            part: links(text, game, int(ply), user, general=part == "lesson")
            for part, text in parts.items()
        }
        for ply, parts in texts.items()
    }
