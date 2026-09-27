"""Puzzles for the player's weak spots (DESIGN section 5.7, M6).

The Lichess puzzle database (CC0) filtered into SQLite under the data directory,
`data/puzzles.sqlite`: a puzzle is a position, the solution in UCI, a rating, and the
themes Lichess tagged it with -- the same names our tagger uses for the tactics
(`analysis/motifs.py`), so a weak cell of the profile names the puzzles to practise.

A recommendation is made after a game's review from the profile's worst motif: a few
puzzles of that theme near the player's puzzle level, none of them solved lately, a
failed one back first (spaced repetition of the plainest kind). The player's level moves
with the outcomes, a step up on a solve and down on a fail, from a start of 1100 for a
chess.com player around 1200.

The outcome is read off the screen like everything else: a puzzle opened on Lichess is
a position the pipeline joins (GameStarted of kind "position"); its placement and side
to move name the puzzle in the store, and the moves the player then makes are checked
against the solution -- every solver's move right through the last is a solve, a wrong
one a fail -- and the attempt is recorded beside the motif.
"""

from __future__ import annotations

import contextlib
import csv
import random
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

from sidecoach.events import PuzzleOutcome, PuzzleRef

# The ratings a puzzle may have to be kept at all: the band the coach serves (chess.com
# 800-1600, DESIGN) with room to grow either way.
RATING_MIN = 600
RATING_MAX = 2000
# The window around the player's level a recommended puzzle comes from.
WINDOW = 150
# The level starts here for a new player and moves this much per outcome.
LEVEL_START = 1100
LEVEL_STEP = 25
# How many puzzles a recommendation carries.
COUNT = 3
# Themes that are tactics, in Lichess's names; the profile's motif names are the same but
# for the mates, which Lichess splits by length.
THEME_FOR_MOTIF: dict[str, list[str]] = {
    "mate": ["mateIn1", "mateIn2", "mateIn3"],
    "absolutePin": ["pin"],
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS puzzles (
    id TEXT PRIMARY KEY,
    fen TEXT NOT NULL,
    moves TEXT NOT NULL,
    rating INTEGER NOT NULL,
    popularity INTEGER NOT NULL DEFAULT 0,
    plays INTEGER NOT NULL DEFAULT 0,
    themes TEXT NOT NULL,
    position TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS puzzles_rating ON puzzles (rating);
CREATE INDEX IF NOT EXISTS puzzles_position ON puzzles (position);
"""


@dataclass(frozen=True)
class Puzzle:
    id: str
    fen: str  # the position before the opponent's first move, as Lichess stores it
    moves: tuple[str, ...]  # UCI: the opponent's move first, then the solution in turns
    rating: int
    popularity: int
    plays: int
    themes: tuple[str, ...]

    @property
    def url(self) -> str:
        return f"https://lichess.org/training/{self.id}"

    @property
    def solver_moves(self) -> tuple[str, ...]:
        """The moves the player must find: the second, fourth, ... of the line."""
        return self.moves[1::2]

    def position(self) -> str:
        """The position the solver faces: the placement and the side to move after the
        opponent's first move -- what the screen shows, and the key the store looks a
        puzzle up by (castling rights and counters do not survive a cold read)."""
        return position_key(self.fen, self.moves[0])

    def ref(self) -> PuzzleRef:
        return PuzzleRef(id=self.id, url=self.url, rating=self.rating, themes=list(self.themes))


def open_store(data_dir: Path) -> PuzzleStore | None:
    """The puzzle store under the data directory, filled once from the Lichess CSV
    beside it (data/puzzles/*.csv, the sample `tools/puzzles/fetch.py` writes or the
    whole database) when it is empty; None when there are no puzzles to be had."""
    store = PuzzleStore(data_dir / "puzzles.sqlite")
    if store.count() == 0:
        for csv_path in sorted((data_dir / "puzzles").glob("*.csv")):
            store.import_csv(csv_path)
    if store.count() == 0:
        store.close()
        return None
    return store


def position_key(fen: str, first_move: str | None = None) -> str:
    """`board_fen turn` of a position, after `first_move` when given."""
    board = chess.Board(fen)
    if first_move:
        board.push_uci(first_move)
    return f"{board.board_fen()} {'w' if board.turn else 'b'}"


def themes_for(motif: str) -> list[str]:
    return THEME_FOR_MOTIF.get(motif, [motif])


class PuzzleStore:
    """The puzzles on disk (or in memory), imported once from a Lichess CSV."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            str(path) if path is not None else ":memory:", check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def count(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM puzzles").fetchone()[0])

    def clear(self) -> None:
        """Every puzzle out; the store stays open for a fresh import."""
        with self._lock:
            self._db.execute("DELETE FROM puzzles")
            self._db.commit()
            with contextlib.suppress(sqlite3.OperationalError):
                self._db.execute("VACUUM")

    def import_csv(self, path: Path, *, limit: int | None = None) -> int:
        """Load a Lichess puzzle CSV (the database's columns: PuzzleId,FEN,Moves,Rating,
        RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags), keeping the
        puzzles in the rating band; returns how many were added."""
        added = 0
        with path.open(encoding="utf-8", newline="") as handle, self._lock:
            for row in csv.DictReader(handle):
                try:
                    rating = int(row["Rating"])
                # A row too short to hold a rating reads None there (the parity audit).
                except (KeyError, ValueError, TypeError):
                    continue
                if not RATING_MIN <= rating <= RATING_MAX:
                    continue
                moves = (row.get("Moves") or "").split()
                if len(moves) < 2:
                    continue
                try:
                    position = position_key(row["FEN"], moves[0])
                except (ValueError, TypeError):
                    continue
                cursor = self._db.execute(
                    "INSERT OR IGNORE INTO puzzles"
                    " (id, fen, moves, rating, popularity, plays, themes, position)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["PuzzleId"],
                        row["FEN"],
                        row["Moves"],
                        rating,
                        int(row.get("Popularity") or 0),
                        int(row.get("NbPlays") or 0),
                        row.get("Themes") or "",
                        position,
                    ),
                )
                added += cursor.rowcount
                if limit is not None and added >= limit:
                    break
            self._db.commit()
        return added

    @staticmethod
    def _puzzle(row: sqlite3.Row) -> Puzzle:
        return Puzzle(
            id=row["id"],
            fen=row["fen"],
            moves=tuple(row["moves"].split()),
            rating=int(row["rating"]),
            popularity=int(row["popularity"]),
            plays=int(row["plays"]),
            themes=tuple(row["themes"].split()),
        )

    def get(self, puzzle_id: str) -> Puzzle | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM puzzles WHERE id = ?", (puzzle_id,)).fetchone()
        return self._puzzle(row) if row is not None else None

    def at_position(self, key: str) -> list[Puzzle]:
        """The puzzles whose solver's position is `key` (see position_key)."""
        with self._lock:
            rows = self._db.execute("SELECT * FROM puzzles WHERE position = ?", (key,)).fetchall()
        return [self._puzzle(r) for r in rows]

    def pick(
        self,
        motif: str,
        level: int,
        *,
        count: int = COUNT,
        exclude: set[str] = frozenset(),  # type: ignore[assignment]
        first: list[str] = (),  # type: ignore[assignment]
        rng: random.Random | None = None,
    ) -> list[Puzzle]:
        """`count` puzzles of the motif's themes within WINDOW of `level`: the ids in
        `first` (failed lately, due again) ahead of a random draw among the rest, none in
        `exclude` (solved lately), the more played the better."""
        rng = rng or random.Random()
        themes = themes_for(motif)
        clauses = " OR ".join("(' ' || themes || ' ') LIKE ?" for _ in themes)
        params: list[object] = [f"% {t} %" for t in themes]
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM puzzles WHERE ({clauses}) AND rating BETWEEN ? AND ?"
                " ORDER BY plays DESC LIMIT 400",
                [*params, level - WINDOW, level + WINDOW],
            ).fetchall()
        pool = [self._puzzle(r) for r in rows if r["id"] not in exclude]
        chosen: list[Puzzle] = [p for p in pool if p.id in first][:count]
        rest = [p for p in pool if p.id not in first]
        rng.shuffle(rest)
        chosen += rest[: count - len(chosen)]
        return chosen


def puzzle_level(store: Any) -> int:
    """The player's puzzle level from the profile store (LEVEL_START to begin with)."""
    value = store.setting("puzzle_level")
    return int(value) if value else LEVEL_START


def record_attempt(
    store: Any, puzzle_id: str, motif: str | None, solved: bool, moves: list[str], ts: float
) -> PuzzleOutcome:
    """An attempt into the profile store: kept beside the motif, the level moved a step
    up on a solve and down on a fail. Shared by the profiler (a puzzle followed on the
    screen) and the puzzle page (one solved on its own board)."""
    step = LEVEL_STEP if solved else -LEVEL_STEP
    level = max(RATING_MIN, min(RATING_MAX, puzzle_level(store) + step))
    store.set_setting("puzzle_level", str(level))
    store.add_attempt(puzzle_id, ts, motif, solved, moves)
    return PuzzleOutcome(
        game_id=f"puzzle-{puzzle_id}", puzzle_id=puzzle_id, motif=motif, solved=solved, level=level
    )


class Attempt:
    """A puzzle as the player works through it on the screen: which solver moves came,
    and whether it is solved or failed yet."""

    def __init__(self, puzzle: Puzzle, motif: str | None) -> None:
        self.puzzle = puzzle
        self.motif = motif
        self.played: list[str] = []
        self.outcome: str | None = None  # "solved" | "failed" | None while under way

    def move(self, uci: str, *, by_user: bool) -> str | None:
        """A move on the board; the outcome once it is decided, else None."""
        if self.outcome is not None:
            return None
        if not by_user:
            return None  # the site's replies are the line's; nothing to judge
        expected = self.puzzle.solver_moves
        index = len(self.played)
        if index >= len(expected):
            return None
        self.played.append(uci)
        want = expected[index]
        # Lichess accepts any mating move where the line's is a mate; the board says.
        if uci != want and not self._also_mates(index, uci):
            self.outcome = "failed"
        elif len(self.played) == len(expected):
            self.outcome = "solved"
        return self.outcome

    def _also_mates(self, index: int, uci: str) -> bool:
        board = chess.Board(self.puzzle.fen)
        try:
            for move in self.puzzle.moves[: 1 + 2 * index]:
                board.push_uci(move)
            board.push_uci(uci)
        except ValueError:
            return False
        return board.is_checkmate()
