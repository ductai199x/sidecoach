"""Opening identification and the point where a game leaves known theory.

The data is the Lichess chess-openings tables (CC0): five TSV files `a.tsv` through
`e.tsv` in `data/openings`, columns eco, name, pgn, 3810 named lines in total.
`scripts/fetch_openings.py` downloads them. `data/` is gitignored, so the data may be
absent; `OpeningBook.load` then returns None and the caller does without opening facts.

There is no Polyglot `.bin` book, although DESIGN.md section 5.4 mentions one. The same
tables answer both questions: a move is *in book* while the game is still standing on a
position that some named line passes through, and the first move that leaves every line
is the deviation. One dataset, one index, no second file to fetch and keep in step. A
Polyglot book would also answer a different question (what a strong player would pick),
which Stockfish already answers better for coaching purposes.

Indexing is by position, not by move order: each line is replayed on a board and every
position along it is stored as an EPD, so 3810 lines become 7855 positions of which 3810
carry a name. Transpositions therefore resolve for free, and 1.d4 Nf6 2.c4 e6 3.Nc3 d5
is named the same as 1.d4 d5 2.c4 e6 3.Nc3 Nf6. Whether a move is in book is walked only
as far as the game's first deviation, so a game that wanders out of theory and back does
not silently become book again. Its name is Lichess's (scalachess `OpeningDb.search`): the
last named position the game reaches in its first 40 plies, so a game that leaves theory
and transposes back into a named line takes that line's name (Tai, 2026-10-02).

EPD rather than FEN because it drops the halfmove and fullmove counters, so two move
orders of unequal length still match, and it names an en passant square only when the
capture is actually available.

Parsing the tables costs about a second, too slow for a live session start, so the
compiled index is cached as JSON beside the TSVs and reloaded in a few milliseconds
whenever the source files still have the size and mtime the cache recorded.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from sidecoach.config import load_settings

log = logging.getLogger(__name__)

CACHE_FILE = "openings.cache.json"
CACHE_VERSION = 1
# How far into a game its opening is named: Lichess's SEARCH_MAX_PLIES.
NAME_PLIES = 40


@dataclass(frozen=True)
class Opening:
    """A named opening line matched against a game."""

    eco: str
    """ECO code, for example "C50"."""
    name: str
    """Lichess name, for example "Italian Game: Giuoco Pianissimo"."""
    ply: int
    """How many plies of the game the matched line covers."""


def _fingerprint(tables: Sequence[Path]) -> dict[str, list[int]]:
    """Size and mtime of each table, so a changed table invalidates the cache."""
    prints: dict[str, list[int]] = {}
    for table in tables:
        stat = table.stat()
        prints[table.name] = [stat.st_size, stat.st_mtime_ns]
    return prints


def _compile(tables: Sequence[Path]) -> tuple[dict[str, str], set[str]]:
    """Replay every line in the tables.

    Returns the EPD of each line's final position mapped to "eco\\tname", and the set of
    every EPD any line passes through, the start position included.
    """
    names: dict[str, str] = {}
    positions: set[str] = {chess.Board().epd()}
    for table in tables:
        text = table.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            fields = line.split("\t")
            if len(fields) != 3 or fields[0] == "eco":
                continue
            eco, name, pgn = fields
            board = chess.Board()
            walked: list[str] = []
            try:
                for token in pgn.split():
                    if token[0].isdigit():
                        continue
                    board.push_san(token)
                    walked.append(board.epd())
            except ValueError as exc:
                log.warning("%s line %d: %s (%s)", table.name, number, exc, name)
                continue
            if not walked:
                continue
            positions.update(walked)
            names.setdefault(walked[-1], f"{eco}\t{name}")
    return names, positions


class OpeningBook:
    """Prefix index over the Lichess opening lines, keyed by position."""

    def __init__(self, names: dict[str, str], positions: set[str]) -> None:
        self._names = names
        self._positions = positions

    def __len__(self) -> int:
        return len(self._names)

    @classmethod
    def load(cls, folder: Path | None = None) -> OpeningBook | None:
        """The book, or None when the data has not been fetched.

        `folder` defaults to `data/openings` under the settings data dir. Never raises
        because the data is missing, unreadable or corrupt: every failure logs and
        returns None, so a session without the tables simply has no opening facts.
        """
        try:
            root = Path(folder) if folder is not None else load_settings().data_dir / "openings"
            tables = sorted(root.glob("*.tsv")) if root.is_dir() else []
            if not tables:
                log.info("no opening tables in %s; run scripts/fetch_openings.py", root)
                return None

            cache = root / CACHE_FILE
            cached = cls._read_cache(cache, tables)
            if cached is not None:
                return cached

            names, positions = _compile(tables)
            if not names:
                log.warning("opening tables in %s held no usable lines", root)
                return None
            cls._write_cache(cache, tables, names, positions)
            return cls(names, positions)
        except OSError as exc:
            log.warning("could not load the opening book: %s", exc)
            return None

    @classmethod
    def _read_cache(cls, cache: Path, tables: Sequence[Path]) -> OpeningBook | None:
        """The cached index, or None when it is missing, stale or unreadable."""
        if not cache.is_file():
            return None
        try:
            blob: Any = json.loads(cache.read_text(encoding="utf-8"))
            if blob["version"] != CACHE_VERSION or blob["sources"] != _fingerprint(tables):
                return None
            names: dict[str, str] = blob["names"]
            positions: list[str] = blob["positions"]
            if not isinstance(names, dict) or not isinstance(positions, list):
                return None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.warning("ignoring the opening cache %s: %s", cache, exc)
            return None
        return cls(names, set(positions))

    @classmethod
    def _write_cache(
        cls, cache: Path, tables: Sequence[Path], names: dict[str, str], positions: set[str]
    ) -> None:
        """Write the compiled index beside the tables. A read-only folder is not an error."""
        blob = {
            "version": CACHE_VERSION,
            "sources": _fingerprint(tables),
            "names": names,
            "positions": sorted(positions),
        }
        scratch = cache.with_suffix(".tmp")
        try:
            scratch.write_text(json.dumps(blob), encoding="utf-8")
            scratch.replace(cache)
        except OSError as exc:
            log.warning("could not write the opening cache %s: %s", cache, exc)
            scratch.unlink(missing_ok=True)

    def _trace(self, moves: Sequence[chess.Move]) -> int | None:
        """Walk `moves` from the start position until they leave the book: the index of the
        first move that left every line, or None when the game never left."""
        board = chess.Board()
        for index, move in enumerate(moves):
            if move not in board.legal_moves:
                # Not a game from the start position: a caller handed us moves that do not
                # belong to this history, and it is in no book.
                return index
            board.push(move)
            if board.epd() not in self._positions:
                return index
        return None

    def identify(self, moves: Sequence[chess.Move]) -> Opening | None:
        """The opening as Lichess names it (scalachess `OpeningDb.search`): the last named
        position the game reaches within its first NAME_PLIES plies, whatever it passed
        through on the way -- a game that leaves theory and transposes back into a named
        line takes that line's name (Tai, 2026-10-02).

        The latest is the most specific name: after 1.e4 e5 2.Nf3 Nc6 3.Bc4 that is the
        Italian Game rather than the King's Pawn Game it also passed. None when no named
        position comes up in those plies, or the moves are not a game from the start.
        """
        board = chess.Board()
        found: Opening | None = None
        for index, move in enumerate(moves[:NAME_PLIES]):
            if move not in board.legal_moves:
                return found
            board.push(move)
            named = self._names.get(board.epd())
            if named is not None:
                eco, name = named.split("\t", 1)
                found = Opening(eco=eco, name=name, ply=index + 1)
        return found

    def in_book(self, moves: Sequence[chess.Move]) -> bool:
        """Whether `moves` is a prefix of at least one known line."""
        return self._trace(moves) is None

    def deviation(self, moves: Sequence[chess.Move]) -> int | None:
        """Index of the first move that leaves every known line, None if it never does."""
        return self._trace(moves)
