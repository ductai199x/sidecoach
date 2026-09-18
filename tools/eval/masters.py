"""Master games from the annotated books: align each book's move headings to the real
game and keep the master's explanation of every move as the judge's reference.

The books (gitignored `refs/`, OCR'd to markdown) keep every move as a token with the move
number and the destination square intact -- the piece glyph is mangled (`ctJf6` for Nf6,
`'iVc2` for Qc2, `.ifl` for Bf1, a ♘ glyph in the cleaner scans) and a rank digit can come
out as a letter (`WhS` for Kh8, `dS` for d5). The moves therefore come from a clean source:
pgnmentor.com's per-player collections, fetched once by surname and cached. A game is
located by both players' surnames and the year, and when several fit, by which one the
book's tokens align with best. Alignment is by move number + colour + destination square
under a small confusion map, verified against the game ply by ply; the prose between one
aligned token and the next is that move's annotation.

What comes out (all under gitignored data/eval/masters/):
  <book>/<game>.json   the game (headers, PGN moves) with, per ply, the master's prose
  report.md            per game: located or not, plies aligned, tokens unmatched
The prose is reference material for the judge only. It goes into no prompt, no repository
file and no C.

Run:  uv run python tools/eval/masters.py [--book nunn|fischer|mammoth] [--game N] [-v]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
REFS = ROOT / "refs"
OUT = ROOT / "data" / "eval" / "masters"
CACHE = Path(__file__).with_name(".cache")
UA = "sidecoach-eval-harness (research; contact taiducnguyen.drexel@gmail.com)"

BOOKS = {
    "nunn": REFS / "Understanding chess move by move" / "Understanding chess move by move.md",
    "fischer": REFS / "My 60 Memorable Games" / "My 60 Memorable Games.md",
    "mammoth": REFS
    / "The Mammoth Book of the World's Greatest Chess Games ._ New edn (Mammoth Books 200)"
    / "The Mammoth Book of the World's Greatest Chess Games ._ New edn (Mammoth Books 200).md",
}

# ---- the text -------------------------------------------------------------------------

GLYPHS = str.maketrans({"♘": "N", "♗": "B", "♕": "Q", "♖": "R", "♔": "K", "♙": ""})
# What the OCR makes of a rank digit. A letter may stand for more than one digit.
RANK_ALTS = {
    "1": "1lI|i",
    "2": "2Zz",
    "3": "3",
    "4": "4",
    "5": "5Ss",
    "6": "6G",
    "7": "7",
    "8": "8SsB",
}
_SQUARE = re.compile(r"([a-h])([1-8lI|iZzSsGB])")
_CASTLE = re.compile(r"[0Oo]\s*-\s*[0Oo](\s*-\s*[0Oo])?")
_DOTS = r"(?:\.\s*\.\s*\.|…|\.{3})"
# A move token at the start of a line, after markdown: "12 Nf3", "12 ... Nf6", "12 Nf3 Nf6".
_LINE_TOKEN = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*\s*)?(?:\|\s*)?(\d{1,3})(?:\*\*)?\s*(" + _DOTS + r")?\s*(.*)$"
)


def _clean(line: str) -> str:
    line = line.translate(GLYPHS)
    line = re.sub(r"\$\\Delta\s*([a-h][1-8])\$", r"N\1", line)  # $\Delta f3$ -> Nf3
    line = re.sub(r"\$\\hat\{\\\$\}\$", "", line)  # a piece glyph the OCR lost
    line = re.sub(r"\$[^$]*\$", "", line)
    line = line.replace("**", "").replace("*", "")
    line = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", line)  # markdown links
    line = re.sub(r"<[^>]+>", "", line)
    return line


_RANKISH = "[1-8lI|iZzSsGB]"
# A move word ends on its destination square (then maybe a promotion piece): a short thing
# like "ctJf6", "'iVxb5", "exd6", "e8=Q". Prose words with a letter that could pass for a
# rank ("symmetrical" -> a1) are ruled out by the length and the ending.
_MOVEWORD = re.compile(r"^.{0,5}[a-h]" + _RANKISH + r"(?:=?[QRBN])?$")
# The OCR sometimes glues the move to the first word of its note: "e5This symmetrical".
_GLUED = re.compile(r"^(.{0,5}[a-h]" + _RANKISH + r"[+#]?[!?]*)([A-Z][a-z]+.*)$")


def _unglue(word: str) -> str:
    m = _GLUED.match(word)
    return m.group(1) if m else word


def _move_bits(word: str) -> tuple[str, str] | None:
    """(kind, key) for a word that looks like a move: ("sq", "d5") or ("castle", "O-O")."""
    w = _unglue(word.strip()).strip("!?+#,;:.()")
    if _CASTLE.fullmatch(w.replace(" ", "")):
        return ("castle", "O-O-O" if w.count("0") + w.count("O") + w.count("o") == 3 else "O-O")
    if not _MOVEWORD.match(w):
        return None
    pairs = _SQUARE.findall(w)
    if not pairs:
        return None
    file, rank = pairs[-1]
    # A plain lowercase word ("does", "goes") is prose unless its rank is a real digit;
    # the OCR's letter-for-digit only comes with garbage around it (".ifl", "WhS").
    if not rank.isdigit() and w.isalpha() and w.islower():
        return None
    return ("sq", file + rank)


def _compatible(bits: tuple[str, str], move: chess.Move, board: chess.Board) -> bool:
    kind, key = bits
    if board.is_castling(move):
        return kind == "castle" and key == ("O-O-O" if board.is_queenside_castling(move) else "O-O")
    if kind != "sq":
        return False
    file, rank = key[0], key[1]
    true = chess.square_name(move.to_square)
    return file == true[0] and rank in RANK_ALTS[true[1]]


@dataclass
class Token:
    line: int  # index into the game's lines
    number: int
    black_hint: bool  # "12 ..." said black explicitly
    words: list[str]  # the move-looking words on the line, in order (one or two)


@dataclass
class BookGame:
    book: str
    number: int
    white: str
    black: str
    event: str
    year: str
    lines: list[str]
    tokens: list[Token] = field(default_factory=list)


def _tokens(lines: list[str]) -> list[Token]:
    """The move tokens, line by line. A line that starts with one ("12 Nf3", "12 ... Nf6",
    "| 12 | Nf3 | Nf6 |") yields every "number move [move]" group on it, so "1 d4 Nf6 2 Nf3
    g6" gives four moves; prose that merely mentions a move mid-sentence yields nothing."""
    out: list[Token] = []
    for i, raw in enumerate(lines):
        line = _clean(raw).replace("|", " ")
        line = re.sub(r"^\s*#{1,6}\s*", "", line)
        words = line.split()
        if not words or not re.fullmatch(r"\d{1,3}", words[0]):
            continue
        k = 0
        while k < len(words):
            w = words[k]
            if not re.fullmatch(r"\d{1,3}", w) or not 0 < int(w) <= 200:
                k += 1
                continue
            number, k = int(w), k + 1
            dots = False
            if k < len(words) and re.fullmatch(_DOTS, words[k]):
                dots, k = True, k + 1
            elif k < len(words) and re.match(_DOTS, words[k]):
                # "...Nf6" glued to the dots
                dots, words[k] = True, re.sub("^" + _DOTS, "", words[k])
            if k >= len(words) or _move_bits(words[k]) is None:
                continue
            group = [words[k]]
            k += 1
            if (
                not dots
                and k < len(words)
                and not re.fullmatch(r"\d{1,3}", words[k])
                and _move_bits(words[k]) is not None
            ):
                group.append(words[k])
                k += 1
            out.append(Token(i, number, dots, group))
    return out


_INITIALLED = re.compile(
    r"((?:[A-Za-z][a-z]?\.\s*)+(?:van der |van |de |von )?[A-Z][\w'-]+(?:\s[a-z]{2,4})?)\s+[–—-]\s+"
    r"((?:[A-Za-z][a-z]?\.\s*)+(?:van der |van |de |von )?[A-Z][\w'-]+)"
)


def _names(text: str) -> tuple[str, str] | None:
    text = _clean(text)
    text = re.sub(r"^\s*#+\s*", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)  # "[U.S.A.]"
    m = _INITIALLED.search(text)
    if m:
        # "A. Khalifman - E. Sveshnikov Russian Championship, Elista 1996 ..." on one line.
        return m.group(1).strip(), m.group(2).strip()
    parts = re.split(r"\s+[–—-]\s+|\s+v\.?\s+", text)
    if len(parts) != 2:
        return None
    a, b = (re.sub(r"\s+", " ", p).strip(" .:*") for p in parts)
    if not a or not b or len(a) > 40 or len(b) > 40 or any(ch.isdigit() for ch in a + b):
        return None
    return a, b


_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _games_nunn(text: str) -> list[BookGame]:
    lines = text.split("\n")
    starts: list[tuple[int, int, str, str, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(?:#{1,6}\s*)?\**\s*Game\s+(\d+)\b\**\s*(.*)$", line)
        if not m or int(m.group(1)) > 30:
            continue
        rest = m.group(2).strip()
        names = _names(rest.split("  ")[0]) if rest else None
        event = ""
        look = [rest] + [lines[j] for j in range(i + 1, min(i + 8, len(lines)))]
        if names is None:
            for cand in look[1:]:
                names = _names(cand)
                if names:
                    break
        if names is None:
            continue
        for cand in look:
            if _YEAR.search(cand):
                event = _clean(cand).strip(" *")
                break
        if starts and starts[-1][1] == int(m.group(1)):
            continue
        starts.append((i, int(m.group(1)), names[0], names[1], event))
    return _cut(lines, starts, "nunn")


def _games_fischer(text: str) -> list[BookGame]:
    lines = text.split("\n")
    starts: list[tuple[int, int, str, str, str]] = []
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(\d{1,2})\s+(.+)$", line)
        if not m or int(m.group(1)) > 60:
            continue
        names = _names(m.group(2))
        if names is None:
            continue
        event = ""
        for j in range(i + 1, min(i + 6, len(lines))):
            if _YEAR.search(lines[j]):
                event = _clean(lines[j]).strip(" *")
                break
        if starts and starts[-1][1] == int(m.group(1)):
            continue
        starts.append((i, int(m.group(1)), names[0], names[1], event))
    return _cut(lines, starts, "fischer")


def _games_mammoth(text: str) -> list[BookGame]:
    lines = text.split("\n")
    starts: list[tuple[int, int, str, str, str]] = []
    for i, line in enumerate(lines):
        if "Game" not in line or not line.startswith("#"):
            continue
        clean = _clean(line)
        m = re.match(r"^\s*#+\s*Game\s+(\d+)\s+(.*)$", clean)
        if not m:
            continue
        rest = m.group(2)
        # The players are the bold segment with the dash; the event the italic one.
        unlinked = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", re.sub(r"<[^>]+>", "", line))
        bold = [b for b in re.findall(r"\*\*([^*]+)\*\*", unlinked) if re.search(r"\s[–—-]\s", b)]
        it = re.search(r"\*([^*]*\d{4}[^*]*)\*", unlinked)
        event = it.group(1).strip() if it else ""
        names = _names(bold[0]) if bold else None
        if names is None:
            # 'The "Immortal Game" Adolf Anderssen – Lionel Kieseritzky London 1851 ...'
            rest = re.sub(r'^The\s+"[^"]*"\s*', "", rest)
            players = re.split(r"\s+(?=[A-Z][^\s]*\s+\d{4}\b)|\s{2,}", rest)[0]
            for cut in range(len(players), 10, -1):
                names = _names(players[:cut])
                if names:
                    break
        if names is None:
            names = _names(rest)
        if not event:
            # Two-line header: "Game 7 **Pillsbury – Lasker**" then "*St Petersburg 1895/6*".
            for j in range(i + 1, min(i + 4, len(lines))):
                it = re.search(r"\*([^*]*\d{4}[^*]*)\*", lines[j])
                if it:
                    event = it.group(1).strip()
                    break
        if names is None:
            continue
        starts.append((i, int(m.group(1)), names[0], names[1], event))
    return _cut(lines, starts, "mammoth")


def _cut(lines: list[str], starts, book: str) -> list[BookGame]:
    games: list[BookGame] = []
    for k, (i, number, white, black, event) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        year = _YEAR.search(event)
        game = BookGame(
            book, number, white, black, event, year.group(1) if year else "", lines[i:end]
        )
        game.tokens = _tokens(game.lines)
        games.append(game)
    return games


def book_games(book: str) -> list[BookGame]:
    text = BOOKS[book].read_text(encoding="utf-8")
    return {"nunn": _games_nunn, "fischer": _games_fischer, "mammoth": _games_mammoth}[book](text)


# ---- the moves ------------------------------------------------------------------------


def _ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def surname(name: str) -> str:
    """'J. van der Wiel' -> 'Wiel'; 'Bu Xiangzhi' -> 'Bu' is wrong but harmless (the whole
    name is also tried)."""
    words = [w for w in re.split(r"[\s.]+", _ascii(name)) if w]
    words = [w for w in words if len(w) > 1 or w.lower() not in ("v",)]
    return words[-1] if words else name


INDEX_URL = "https://www.pgnmentor.com/files.html"


def collection_index() -> list[str]:
    """Every player collection pgnmentor offers, from its file index (cached)."""
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / "files.html"
    if not cached.exists():
        req = urllib.request.Request(INDEX_URL, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=90) as response:  # noqa: S310
            cached.write_bytes(response.read())
    names = set(
        re.findall(
            r"players/([A-Za-z]+)\.zip", cached.read_text(encoding="utf-8", errors="replace")
        )
    )
    return sorted(names)


def _edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _collection_names(name: str) -> list[str]:
    """The collections that could hold this player's games, best match first: the surname
    as pgnmentor spells it ("Tim man" -> Timman, "Onishchuk" -> Onischuk, "M. Gurevich" ->
    GurevichM, "J. Polgar" -> PolgarJ)."""
    words = [w for w in re.split(r"[\s.]+", _ascii(name)) if w]
    initials = "".join(w[0] for w in words if len(w) == 1).upper()
    rest = [w for w in words if len(w) > 1]
    if not rest:
        return []
    forms = {
        "".join(rest).lower(),
        rest[-1].lower(),
        rest[0].lower(),  # Chinese names put the surname first: Bu Xiangzhi, Ye Jiangchuan
        "".join(rest[-2:]).lower(),
        "".join(w for w in rest if w.lower() not in ("van", "der", "de", "von")).lower(),
    }
    scored: list[tuple[int, str]] = []
    for coll in collection_index():
        base = coll.lower()
        # A family shares a surname; the initial picks the member (PolgarJ, GurevichM).
        family = re.fullmatch(r"([A-Za-z][a-z]+)([A-Z])", coll)
        stem = family.group(1).lower() if family else base
        for form in forms:
            if len(form) < 3 and form != base:
                continue
            if form == base:
                scored.append((0, coll))
            elif family and form == stem:
                scored.append((1 if not initials or family.group(2) == initials[0] else 3, coll))
            elif len(form) >= 6 and _edits(form, base) <= 2:
                scored.append((2, coll))
    scored.sort()
    seen: set[str] = set()
    return [c for _, c in scored if not (c in seen or seen.add(c))]


_PARSED: dict[str, list[chess.pgn.Game] | None] = {}


def fetch_collection(name: str) -> list[chess.pgn.Game] | None:
    """pgnmentor's collection for a surname, cached on disk and in memory; None when
    there is none."""
    if name in _PARSED:
        return _PARSED[name]
    _PARSED[name] = _fetch_collection(name)
    return _PARSED[name]


def _fetch_collection(name: str) -> list[chess.pgn.Game] | None:
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{name}.pgn"
    missing = CACHE / f"{name}.missing"
    if missing.exists():
        return None
    if not cached.exists():
        req = urllib.request.Request(
            f"https://www.pgnmentor.com/players/{name}.zip", headers={"User-Agent": UA}
        )
        try:
            time.sleep(0.5)  # a hobby site: one request every half second at most
            with urllib.request.urlopen(req, timeout=90) as response:  # noqa: S310
                blob = response.read()
            zf = zipfile.ZipFile(io.BytesIO(blob))
            pgn = next(n for n in zf.namelist() if n.lower().endswith(".pgn"))
            text = zf.read(pgn).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                missing.write_text("", encoding="utf-8")  # there is no such collection
            else:
                print(f"  pgnmentor {name}: HTTP {exc.code}; will retry next run", file=sys.stderr)
            return None
        except (urllib.error.URLError, zipfile.BadZipFile, StopIteration, TimeoutError) as exc:
            print(f"  pgnmentor {name}: {type(exc).__name__}; will retry next run", file=sys.stderr)
            return None
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cached.write_text(text, encoding="utf-8", newline="\n")
    games: list[chess.pgn.Game] = []
    stream = io.StringIO(cached.read_text(encoding="utf-8"))
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception:  # noqa: BLE001
            continue
        if game is None:
            break
        games.append(game)
    return games


def _matches(header: str, name: str) -> bool:
    """The PGN header names this player: by surname, allowing the book's spelling to
    differ a little from the collection's (Onishchuk / Onischuk, Gligorich / Gligoric)."""
    h, n = _ascii(header).lower(), _ascii(name).lower()
    if n in h:
        return True
    head = re.split(r"[,(]", h)[0].strip()  # "Onischuk, Alexander" -> "onischuk"
    words = [w for w in re.split(r"[\s.]+", n) if len(w) > 1]
    for s in {surname(name).lower(), words[0] if words else "", "".join(words)}:
        if len(s) < 2:
            continue
        if s == head or s in h.split() or (len(s) >= 5 and head.startswith(s[:-1])):
            return True
        if len(s) >= 6 and _edits(s, head) <= 2:
            return True
    return False


def candidates(game: BookGame) -> list[tuple[str, chess.pgn.Game]]:
    """Games in the players' collections with both names in the headers (either order) and
    the year, when the book gives one."""
    out: list[tuple[str, chess.pgn.Game]] = []
    seen: set[str] = set()
    found: set[tuple] = set()
    for who in (game.white, game.black):
        for coll in _collection_names(who):
            if coll in seen:
                continue
            seen.add(coll)
            games = fetch_collection(coll)
            if not games:
                continue
            for g in games:
                h = g.headers
                w, b = h.get("White", ""), h.get("Black", "")
                same = _matches(w, game.white) and _matches(b, game.black)
                swapped = _matches(w, game.black) and _matches(b, game.white)
                if not (same or swapped):
                    continue
                # The book's year may be a season ("1998/9") or a match spanning two years.
                if game.year and h.get("Date", "")[:4] not in (game.year, str(int(game.year) + 1)):
                    continue
                key = (w, b, h.get("Date", ""), h.get("Round", ""), len(list(g.mainline_moves())))
                if key in found:
                    continue
                found.add(key)
                out.append((coll, g))
    return out


# ---- the alignment --------------------------------------------------------------------


@dataclass
class Aligned:
    ply: int
    san: str
    uci: str
    fen_before: str
    token: int | None  # index into game.tokens
    annotation: str = ""


@dataclass
class Half:
    """One move-word of a token: "12 Nf3 Nf6" is two halves, white's and black's."""

    token: int
    line: int
    number: int
    colour: bool | None  # True white, False black, None either
    bits: tuple[str, str]


def _halves(tokens: list[Token]) -> list[Half]:
    out: list[Half] = []
    for k, t in enumerate(tokens):
        bits = [_move_bits(w) for w in t.words]
        if t.black_hint:
            if bits[0]:
                out.append(Half(k, t.line, t.number, False, bits[0]))
        elif len(bits) == 2:
            if bits[0]:
                out.append(Half(k, t.line, t.number, True, bits[0]))
            if bits[1]:
                out.append(Half(k, t.line, t.number, False, bits[1]))
        elif bits[0]:
            out.append(Half(k, t.line, t.number, None, bits[0]))
    return out


_SECTION = re.compile(r"^\s*#{1,6}\s*(?!\**\s*\d)")  # a heading that is not a move
_GAME_END = re.compile(
    r"^\s*\**\s*(1-0|0-1|1/2-1/2|½-½|Lessons from this game|White resigns|Black resigns)"
)


def _prose_end(lines: list[str], start: int) -> int:
    """Where a move's note must stop: the next section heading (the lessons, the next
    theme, the next game) or the game's result line."""
    for j in range(start, len(lines)):
        if _SECTION.match(lines[j]) or _GAME_END.match(lines[j]):
            return j
    return len(lines)


# What the OCR leaves in a note that is not the master's words: the move tables of the
# Fischer book (a row of moves, or its rule), diagram captions ("Position after 23...",
# the players' names in capitals), and strikethrough marks around a lost figurine.
_TABLE_LINE = re.compile(r"^\s*\|.*\|?\s*$")
_CAPTION = re.compile(r"^\s*(?:Position after\b.*|[A-Z][A-Z' .-]{2,}|Diagram\s*\d*)\s*$")
_PIECE_MOVE = re.compile(r"\b[KQRBN]x?[a-h][1-8]")
# A line that is a move number with its move lost to the scan: "Not 26", "26 ...", "28 .".
_DANGLING = re.compile(r"^\s*(?:Not|Or|If|On)?\s*\d{1,3}\s*(?:\.\.\.|\.)?\s*$")
# A move number whose move the scan lost, mid-line: followed by punctuation, another
# number, or the end of the line ("(if 25... 26 ) 26 27 28 29 mates").
_LOST_MOVE = re.compile(
    r"(?<![A-Za-z\d])\d{1,3}(?!\d)\s*(?:\.\.\.|\.)?\s*(?=[?!+#]|\d{1,3}(?!\d)|$|[,;])", re.M
)


# What the scan put where a piece figurine stood, per book, where it did so reliably:
# tallied over the tokens the alignment matched to a mainline move (whose true piece is
# known), kept at 75% agreement or better over at least eight cases. The rest of the
# figurines are lost ("xd8"), and the judge is told so.
PIECE_STANDINS: dict[str, dict[str, str]] = {
    "fischer": {"W": "Q", "w": "Q", "D": "N", "\u2756": "K", "\u2318": "R", "\u25a1": "R"},
    "nunn": {"t": "N", "C": "N", "i": "B", "M": "R", '"': "Q"},
    "mammoth": {},
}


def _restore_pieces(text: str, book: str) -> str:
    for ch, piece in PIECE_STANDINS.get(book, {}).items():
        text = re.sub(r"(?<![A-Za-z])" + re.escape(ch) + r"(x?[a-h][1-8])", piece + r"\1", text)
    return text


def _note(text: str, book: str = "") -> str | None:
    """The master's note as prose: table rows, captions and strike marks dropped, the
    scan's stand-ins for piece letters restored where they are reliable. None when nothing
    readable is left: a bare rule of a table, or a variation whose piece letters the OCR
    lost ("Not 18... xh7? 19 xf4."), which no judge can use."""
    kept = [
        line
        for line in text.splitlines()
        if not (_TABLE_LINE.match(line) or _CAPTION.match(line) or _DANGLING.match(line))
    ]
    text = "\n".join(kept).replace("~~", "")
    text = _restore_pieces(text, book)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    words = re.findall(r"[A-Za-z]{3,}", text)
    if len(words) < 3 and not _PIECE_MOVE.search(text):
        return None
    lost = len(_LOST_MOVE.findall(text))
    if lost >= 3 and lost * 4 > len(words):
        return None  # the variations are gone and little prose is left: unreadable
    if not words and not re.search(r"[!?#]", text):
        return None  # bare moves with no judgment on them: the next move's text, leaked
    return text or None


def align(game: BookGame, pgn: chess.pgn.Game) -> tuple[list[Aligned], int]:
    """Match the game's plies to the book's move-words, in order, keeping as many as
    possible: a longest-common-subsequence over plies x words where a ply and a word agree
    when the move number, the colour and the destination square (under the OCR's confusions)
    do. Variations and notes the OCR put out of order simply go unmatched. Returns the plies
    (with their token, when one matched) and how many matched."""
    board = pgn.board()
    plies: list[tuple[int, chess.Move, chess.Board, str]] = []
    for ply, move in enumerate(pgn.mainline_moves(), start=1):
        plies.append((ply, move, board.copy(stack=False), board.san(move)))
        board.push(move)
    halves = _halves(game.tokens)
    n, m = len(plies), len(halves)

    def ok(i: int, j: int) -> bool:
        ply, move, before, _ = plies[i]
        h = halves[j]
        white = ply % 2 == 1
        if h.number != (ply + 1) // 2 or (h.colour is not None and h.colour != white):
            return False
        return _compatible(h.bits, move, before)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            best = max(below[j], row[j + 1])
            if ok(i, j):
                best = max(best, below[j + 1] + 1)
            row[j] = best
    pairs: dict[int, int] = {}
    i = j = 0
    while i < n and j < m:
        if ok(i, j) and dp[i][j] == dp[i + 1][j + 1] + 1:
            pairs[i] = j
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    out: list[Aligned] = []
    for i, (ply, move, before, san) in enumerate(plies):
        j = pairs.get(i)
        out.append(
            Aligned(ply, san, move.uci(), before.fen(), halves[j].token if j is not None else None)
        )
    # The annotation of a ply: the lines after its word's line, up to the next matched word's
    # line. Two words on one line ("12 Nf3 Nf6"): the prose after belongs to the second.
    matched = sorted(pairs.items(), key=lambda kv: kv[1])
    for k, (i, j) in enumerate(matched):
        start = halves[j].line + 1
        if k + 1 < len(matched) and halves[matched[k + 1][1]].line == halves[j].line:
            continue
        end = halves[matched[k + 1][1]].line if k + 1 < len(matched) else len(game.lines)
        end = min(end, _prose_end(game.lines, start))
        text = "\n".join(game.lines[start:end])
        text = re.sub(r"!\[\]\([^)]*\)", "", text)  # images
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        cut = text.find("Lessons from this game")  # the book's closing list, mid-line
        if cut >= 0:
            text = text[:cut].rstrip()
        out[i].annotation = _note(_clean(text), game.book)
    return out, len(pairs)


# ---- the run --------------------------------------------------------------------------


def slug(game: BookGame) -> str:
    return f"{game.number:03d}-{surname(game.white).lower()}-{surname(game.black).lower()}"


def run(books: list[str], only: int | None, verbose: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Master games: alignment report", ""]
    totals = {"games": 0, "located": 0, "plies": 0, "annotated": 0}
    for book in books:
        games = book_games(book)
        report += [f"## {book}: {len(games)} games found in the text", ""]
        report.append(
            "| game | players | year | located | plies | aligned | annotated | tokens unmatched |"
        )
        report.append("|---|---|---|---|---|---|---|---|")
        for game in games:
            if only is not None and game.number != only:
                continue
            totals["games"] += 1
            best: tuple[float, str, chess.pgn.Game, list[Aligned], int] | None = None
            for coll, pgn in candidates(game):
                aligned, used = align(game, pgn)
                score = used / max(1, len(game.tokens))
                if best is None or score > best[0]:
                    best = (score, coll, pgn, aligned, used)
            if best is None or best[0] < 0.4:
                why = "no candidate" if best is None else f"best alignment {best[0]:.0%}"
                report.append(
                    f"| {game.number} | {game.white} – {game.black} | {game.year} | no ({why}) | "
                    f"| | | {len(game.tokens)} |"
                )
                if verbose:
                    print(f"[{book} {game.number}] {game.white} – {game.black} {game.year}: {why}")
                continue
            score, coll, pgn, aligned, used = best
            annotated = [a for a in aligned if a.annotation]
            totals["located"] += 1
            totals["plies"] += len(aligned)
            totals["annotated"] += len(annotated)
            item = {
                "book": book,
                "game": game.number,
                "white": game.white,
                "black": game.black,
                "event": game.event,
                "year": game.year,
                "source": f"pgnmentor:{coll}",
                "headers": dict(pgn.headers),
                "result": pgn.headers.get("Result", "*"),
                "tokens": len(game.tokens),
                "aligned": used,
                "plies": [
                    {
                        "ply": a.ply,
                        "san": a.san,
                        "uci": a.uci,
                        "fen_before": a.fen_before,
                        "annotation": a.annotation,
                    }
                    for a in aligned
                ],
            }
            (OUT / book).mkdir(exist_ok=True)
            (OUT / book / f"{slug(game)}.json").write_text(
                json.dumps(item, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            report.append(
                f"| {game.number} | {game.white} – {game.black} | {game.year} | "
                f"{pgn.headers.get('White')} – {pgn.headers.get('Black')}"
                f" {pgn.headers.get('Date', '')[:4]} "
                f"({coll}) | {len(aligned)} | {used} | {len(annotated)} |"
                f" {len(game.tokens) - used} |"
            )
            if verbose:
                print(
                    f"[{book} {game.number}] {game.white} – {game.black} {game.year}: "
                    f"{pgn.headers.get('White')} – {pgn.headers.get('Black')} "
                    f"plies {len(aligned)}, aligned {used}/{len(game.tokens)} tokens, "
                    f"annotated {len(annotated)}"
                )
        report.append("")
    report += [
        "## Totals",
        "",
        f"games {totals['games']}, located {totals['located']}, plies {totals['plies']}, "
        f"annotated plies {totals['annotated']}",
    ]
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[-3:]))
    print("report:", OUT / "report.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=list(BOOKS), action="append")
    ap.add_argument("--game", type=int)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--list", action="store_true", help="list the games found in the text")
    args = ap.parse_args()
    books = args.book or list(BOOKS)
    if args.list:
        for book in books:
            for g in book_games(book):
                print(
                    f"{book} {g.number:3d}  {g.white} – {g.black}  [{g.year}] {g.event[:50]}"
                    f"  tokens={len(g.tokens)}"
                )
        return
    run(books, args.game, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
