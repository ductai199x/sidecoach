"""SQLite storage for the games and the player model: one file, `<data_dir>/profile.sqlite`.

The record is written for questions to be asked of it -- by the code that builds the
review, and by the coach writing its own SQL (ask.py). So it is denormalised on
purpose: every move carries the facts a question might need (the phase, the expected
score after it from the player's side, the player's clock, the material, the tactics
that were on the board and the ones the move allowed), and one key, `move_id`, joins
everything. `SCHEMA_DOC` in ask.py is the same schema in words, with worked queries,
and the tests run those queries against a database built from a game.

Tables: games; moves, each with the engine's verdict once it arrives; evidence, one row
per mistake of the player's with its cause and motif (user-correctable); dialogue,
what the player and the coach said; sessions, with their summaries; library, the games
Game Review can open -- imported from chess.com, Lichess or a PGN -- and reviews, a game
reviewed whole with its summary and the coach's words on every move; twins, the watched
games that were imported too (the import stands for them). Recordings stay on disk.
Everything is written as it happens, so a session that ends abruptly loses at most the last
game's summary. `Store(None)` is in memory, for tests
and for replays, which must not teach the profile anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    user_side TEXT NOT NULL,
    start_fen TEXT NOT NULL,
    time_control TEXT,
    opponent TEXT,
    result TEXT,
    reason TEXT,
    opening TEXT,
    accuracy REAL,
    final_score REAL,
    pgn TEXT,
    final_fen TEXT
);
CREATE TABLE IF NOT EXISTS moves (
    move_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    segment INTEGER NOT NULL DEFAULT 0,
    ply INTEGER NOT NULL,
    move_number INTEGER NOT NULL,
    san TEXT NOT NULL,
    uci TEXT NOT NULL,
    by_user INTEGER NOT NULL,
    fen_before TEXT NOT NULL,
    fen_after TEXT NOT NULL,
    phase TEXT,
    think_ms INTEGER,
    clock_ms INTEGER,
    capture INTEGER NOT NULL DEFAULT 0,
    gives_check INTEGER NOT NULL DEFAULT 0,
    material INTEGER,
    ts REAL NOT NULL,
    label TEXT,
    severity TEXT,
    expected_loss REAL,
    score_after REAL,
    best_uci TEXT,
    best_san TEXT,
    accuracy REAL,
    tags TEXT,
    available_tags TEXT,
    allowed_tags TEXT,
    depth INTEGER,
    withdrawn INTEGER NOT NULL DEFAULT 0,
    UNIQUE (game_id, segment, ply)
);
CREATE TABLE IF NOT EXISTS evidence (
    move_id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    segment INTEGER NOT NULL DEFAULT 0,
    ply INTEGER NOT NULL,
    phase TEXT NOT NULL,
    expected_loss REAL NOT NULL,
    cause TEXT,
    motif TEXT,
    time_pressure INTEGER NOT NULL DEFAULT 0,
    self_report TEXT,
    corrected INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS dialogue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    move_id INTEGER,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    ply INTEGER,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    games INTEGER NOT NULL DEFAULT 0,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT NOT NULL,
    ts REAL NOT NULL,
    motif TEXT NOT NULL,
    reason TEXT,
    level INTEGER NOT NULL,
    puzzle_ids TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS puzzle_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    puzzle_id TEXT NOT NULL,
    ts REAL NOT NULL,
    motif TEXT,
    solved INTEGER NOT NULL,
    moves TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS library (
    key TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    added_at REAL NOT NULL,
    played_at REAL,
    white TEXT,
    black TEXT,
    white_elo INTEGER,
    black_elo INTEGER,
    result TEXT,
    time_class TEXT,
    user_side TEXT,
    pgn TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reviews (
    key TEXT NOT NULL,
    strength TEXT NOT NULL,
    version INTEGER NOT NULL,
    reviewed_at REAL NOT NULL,
    report TEXT NOT NULL,
    summary TEXT,
    words TEXT,
    PRIMARY KEY (key, strength)
);
-- Stats (docs/stats-plan.md; profile/stat_facts.py): a reviewed game's facts, kept when it is
-- reviewed so the Stats window adds them up at once.
CREATE TABLE IF NOT EXISTS stat_games (
    key TEXT PRIMARY KEY,
    strength TEXT NOT NULL,
    source TEXT,
    time_class TEXT,
    played_at REAL,
    user_side TEXT,
    bot INTEGER NOT NULL,
    result TEXT,
    opening TEXT,
    termination TEXT
);
CREATE TABLE IF NOT EXISTS stat_moves (
    key TEXT NOT NULL,
    ply INTEGER NOT NULL,
    side TEXT NOT NULL,
    fen TEXT NOT NULL,
    uci TEXT NOT NULL,
    san TEXT NOT NULL,
    phase TEXT NOT NULL,
    loss REAL NOT NULL,
    label TEXT,
    accuracy REAL,
    score REAL NOT NULL,
    chance TEXT,
    found INTEGER NOT NULL,
    allowed TEXT,
    structure TEXT NOT NULL,
    themes TEXT NOT NULL,
    endgame TEXT,
    spent_ms INTEGER,
    clock_ms INTEGER,
    PRIMARY KEY (key, ply)
);
-- A game Sidecoach watched that was imported too (profile/library.py `reconcile`): the import is
-- its record, the watched one stands aside in the library, the background review and Stats.
CREATE TABLE IF NOT EXISTS twins (
    watched TEXT PRIMARY KEY,
    imported TEXT NOT NULL,
    found_at REAL NOT NULL
);
-- A game the player removed (Store.remove_game), by its library key, for good: bringing games
-- in again leaves it out. `grp` is the key the player removed (an import takes the watched game
-- it stands for with it); `rows`, every row the game took with it, while it can still be put
-- back (UNDO_KEEP_S); `trace`, a watched game's own record, so its import is left out too.
CREATE TABLE IF NOT EXISTS removed (
    key TEXT PRIMARY KEY,
    removed_at REAL NOT NULL,
    grp TEXT NOT NULL,
    rows TEXT,
    trace TEXT
);
-- A review under way (ui/review_room.py `review_row`): each position kept as it is measured, so
-- a review stopped part way -- the app closed, a crash -- goes on from where it was (Tai,
-- 2026-09-30); gone once the review is saved. Only a search at the same depth and lines counts.
CREATE TABLE IF NOT EXISTS review_progress (
    key TEXT NOT NULL,
    strength TEXT NOT NULL,
    depth INTEGER NOT NULL,
    multipv INTEGER NOT NULL,
    fen TEXT NOT NULL,
    measured TEXT NOT NULL,
    PRIMARY KEY (key, strength, fen)
);
CREATE INDEX IF NOT EXISTS moves_game ON moves (game_id, withdrawn);
CREATE INDEX IF NOT EXISTS evidence_game ON evidence (game_id);
CREATE INDEX IF NOT EXISTS attempts_puzzle ON puzzle_attempts (puzzle_id, ts);
"""

# Read-only SQL from the coach: what it may do, checked by SQLite's authorizer. Reading
# tables and running functions is allowed; everything else -- writes, schema changes,
# pragmas, attaching files -- is refused before it runs.
_READ_OPS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
QUERY_ROWS = 200
QUERY_SECONDS = 2.0

# A game's rows: the library's tables by its library key; a watched game's own by its game id
# (its library key is WATCHED + the id).
WATCHED = "watched:"
LIBRARY_TABLES = ("library", "reviews", "review_progress", "stat_games", "stat_moves")
WATCHED_TABLES = ("evidence", "dialogue", "recommendations", "moves", "games")
RESTORABLE = frozenset({*LIBRARY_TABLES, *WATCHED_TABLES, "twins"})
# A removed game's rows are kept this long, for Undo (the page offers it for seconds).
UNDO_KEEP_S = 600.0
# The review strengths, weakest first: a stronger review stands for the game.
STRENGTH_RANK = {"fast": 1, "standard": 2, "deep": 3}
_RANK = (
    "CASE strength "
    + " ".join(f"WHEN '{name}' THEN {rank}" for name, rank in STRENGTH_RANK.items())
    + " ELSE 0 END"
)

# The watched games Stats leaves out: those whose import's facts are kept (the import counts).
_STOOD_FOR = "SELECT t.watched FROM twins t JOIN stat_games s ON s.key = t.imported"


class QueryRefused(ValueError):
    """A query that is not a single read-only statement, or that ran too long."""


@dataclass(frozen=True)
class GameRow:
    game_id: str
    session_id: str | None
    started_at: float
    ended_at: float | None
    user_side: str
    start_fen: str
    time_control: str | None
    opponent: str | None
    result: str | None
    reason: str | None
    opening: str | None
    accuracy: float | None
    final_score: float | None
    pgn: str | None
    final_fen: str | None


@dataclass(frozen=True)
class MoveRow:
    move_id: int
    game_id: str
    segment: int
    ply: int
    move_number: int
    san: str
    uci: str
    by_user: bool
    fen_before: str
    fen_after: str
    phase: str | None
    think_ms: int | None
    clock_ms: int | None
    capture: bool
    gives_check: bool
    material: int | None
    ts: float
    label: str | None
    severity: str | None
    expected_loss: float | None
    score_after: float | None
    best_uci: str | None
    best_san: str | None
    accuracy: float | None
    tags: list[str]
    available_tags: list[str]
    allowed_tags: list[str]
    depth: int | None
    withdrawn: bool


@dataclass(frozen=True)
class EvidenceRow:
    move_id: int
    game_id: str
    segment: int
    ply: int
    phase: str
    expected_loss: float
    cause: str | None
    motif: str | None
    time_pressure: bool
    self_report: str | None
    corrected: bool


@dataclass(frozen=True)
class CellRow:
    """One cell of the error taxonomy, computed from the record: mistakes (`errors`) by
    cause, motif and phase with the expected score they cost, and the occasions
    (`chances`) -- moves in a phase, moves with a tactic on the board -- with the cause
    None. A tactic's occasions count in `errors` those the player missed: a mistake that
    did not play the tactic on the board. `games`: the games its errors came from."""

    cause: str | None
    motif: str | None
    phase: str | None
    errors: int
    chances: int
    expected_loss: float
    games: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Trace:
    """What is kept for good of a watched game the player removed: when it began, their side,
    the result and the moves as read -- enough to know its import (`library.left_out_of`)."""

    started_at: float
    user_side: str
    result: str | None
    sans: list[str]


@dataclass(frozen=True)
class LibraryRow:
    """A game Game Review can open: where it came from (`source`: chesscom, lichess, pgn),
    who played it, and its PGN. `key` is the source and its own id ("chesscom:174300959956");
    `user_side` the side the player had, when their name for that site is known."""

    key: str
    source: str
    added_at: float
    pgn: str
    played_at: float | None = None
    white: str | None = None
    black: str | None = None
    white_elo: int | None = None
    black_elo: int | None = None
    result: str | None = None
    time_class: str | None = None
    user_side: str | None = None


@dataclass(frozen=True)
class ReviewRow:
    """A game reviewed whole at one strength: the review (`analysis.game_review`'s
    ReviewedGame, as JSON), its summary and the coach's words on the moves (JSON, written as
    they are prepared). `version` is the review's format: an older one is done again."""

    key: str
    strength: str
    version: int
    reviewed_at: float
    report: str
    summary: str | None
    words: str | None


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    started_at: float
    ended_at: float | None
    games: int
    summary: str | None


def _tags(value: str | None) -> list[str]:
    return json.loads(value) if value else []


def _ids(value: str | None) -> frozenset[str]:
    """A JSON array of game ids, its nulls left out."""
    return frozenset(g for g in json.loads(value or "[]") if g is not None)


def _move_row(row: sqlite3.Row) -> MoveRow:
    data = dict(row)
    for key in ("by_user", "withdrawn", "capture", "gives_check"):
        data[key] = bool(data[key])
    for key in ("tags", "available_tags", "allowed_tags"):
        data[key] = _tags(data[key])
    return MoveRow(**data)


def _evidence_row(row: sqlite3.Row) -> EvidenceRow:
    data = dict(row)
    data["time_pressure"] = bool(data["time_pressure"])
    data["corrected"] = bool(data["corrected"])
    return EvidenceRow(**data)


class Store:
    """The database. Every method is safe to call from any thread; the panel's engine
    thread writes and its server thread reads."""

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

    # ---- games ------------------------------------------------------------------------

    def start_game(
        self,
        game_id: str,
        started_at: float,
        user_side: str,
        start_fen: str,
        session_id: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO games"
                " (game_id, session_id, started_at, user_side, start_fen) VALUES (?, ?, ?, ?, ?)",
                (game_id, session_id, started_at, user_side, start_fen),
            )
            if session_id is not None:
                self._db.execute(
                    "UPDATE sessions SET games = games + 1 WHERE session_id = ?", (session_id,)
                )
            self._db.commit()

    def set_game(self, game_id: str, **fields: object) -> None:
        """Set columns of a game as they become known: the time control, the opponent."""
        if not fields:
            return
        assignments = ", ".join(f"{key} = ?" for key in fields)
        with self._lock:
            self._db.execute(
                f"UPDATE games SET {assignments} WHERE game_id = ?", (*fields.values(), game_id)
            )
            self._db.commit()

    def end_game(
        self,
        game_id: str,
        ended_at: float,
        result: str,
        reason: str | None,
        *,
        pgn: str | None = None,
        final_fen: str | None = None,
        accuracy: float | None = None,
        final_score: float | None = None,
        opening: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE games SET ended_at = ?, result = ?, reason = ?, pgn = ?, final_fen = ?,"
                " accuracy = ?, final_score = ?, opening = ? WHERE game_id = ?",
                (ended_at, result, reason, pgn, final_fen, accuracy, final_score, opening, game_id),
            )
            self._db.commit()

    def game(self, game_id: str) -> GameRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        return None if row is None else GameRow(**dict(row))

    def games(self, limit: int | None = 50, *, finished: bool = True) -> list[GameRow]:
        """The most recent games, newest first; every one when `limit` is None."""
        clause = "WHERE result IS NOT NULL" if finished else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM games {clause} ORDER BY started_at DESC LIMIT ?",
                (-1 if limit is None else limit,),
            ).fetchall()
        return [GameRow(**dict(row)) for row in rows]

    def delete_game(self, game_id: str) -> None:
        """Every row of a game: one followed provisionally that turned out not to be one."""
        with self._lock, self._db:
            for table in WATCHED_TABLES:
                self._db.execute(f"DELETE FROM {table} WHERE game_id = ?", (game_id,))

    def purge_unfinished(self, min_plies: int) -> int:
        """Games with no result and fewer than `min_plies` moves -- positions the app was
        closed on before they could qualify -- removed; returns how many."""
        with self._lock:
            rows = self._db.execute(
                "SELECT g.game_id FROM games g LEFT JOIN moves m ON m.game_id = g.game_id "
                "WHERE g.result IS NULL GROUP BY g.game_id HAVING COUNT(m.move_id) < ?",
                (min_plies,),
            ).fetchall()
        for row in rows:
            self.delete_game(str(row["game_id"]))
        return len(rows)

    def games_count(self) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM games WHERE result IS NOT NULL"
            ).fetchone()
        return int(row["n"])

    # ---- moves ------------------------------------------------------------------------

    def add_move(
        self,
        game_id: str,
        segment: int,
        ply: int,
        *,
        move_number: int,
        san: str,
        uci: str,
        by_user: bool,
        fen_before: str,
        fen_after: str,
        phase: str | None,
        think_ms: int | None,
        clock_ms: int | None,
        capture: bool,
        gives_check: bool,
        material: int | None,
        ts: float,
    ) -> int:
        """A move as it was played; its id. `segment` counts the position resyncs before
        it: the ply numbering restarts after each one, and the earlier moves stay on
        record. A move written twice at one place replaces the first (a takeback replayed)."""
        with self._lock:
            self._db.execute(
                "DELETE FROM moves WHERE game_id = ? AND segment = ? AND ply = ?",
                (game_id, segment, ply),
            )
            cursor = self._db.execute(
                "INSERT INTO moves (game_id, segment, ply, move_number, san, uci, by_user,"
                " fen_before, fen_after, phase, think_ms, clock_ms, capture, gives_check,"
                " material, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    segment,
                    ply,
                    move_number,
                    san,
                    uci,
                    int(by_user),
                    fen_before,
                    fen_after,
                    phase,
                    think_ms,
                    clock_ms,
                    int(capture),
                    int(gives_check),
                    material,
                    ts,
                ),
            )
            self._db.commit()
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def assess_move(
        self,
        move_id: int,
        *,
        label: str | None,
        severity: str,
        expected_loss: float,
        best_uci: str | None,
        best_san: str | None,
        accuracy: float | None,
        tags: Sequence[str],
        available_tags: Sequence[str],
        depth: int,
        phase: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE moves SET label = ?, severity = ?, expected_loss = ?, best_uci = ?,"
                " best_san = ?, accuracy = ?, tags = ?, available_tags = ?, depth = ?,"
                " phase = COALESCE(?, phase) WHERE move_id = ?",
                (
                    label,
                    severity,
                    expected_loss,
                    best_uci,
                    best_san,
                    accuracy,
                    json.dumps(list(tags)),
                    json.dumps(list(available_tags)),
                    depth,
                    phase,
                    move_id,
                ),
            )
            self._db.commit()

    def analyse_move(
        self, move_id: int, *, score_after: float | None, allowed_tags: Sequence[str]
    ) -> None:
        """What the position after the move holds: the expected score from the player's
        side, and the tactics the mover allowed."""
        with self._lock:
            self._db.execute(
                "UPDATE moves SET score_after = ?, allowed_tags = ? WHERE move_id = ?",
                (score_after, json.dumps(list(allowed_tags)), move_id),
            )
            self._db.commit()

    def withdraw_moves(self, game_id: str, segment: int, from_ply: int) -> None:
        """A takeback: the moves from `from_ply` on are out of the game, kept for the record."""
        with self._lock:
            self._db.execute(
                "UPDATE moves SET withdrawn = 1 WHERE game_id = ? AND segment = ? AND ply >= ?",
                (game_id, segment, from_ply),
            )
            self._db.execute(
                "DELETE FROM evidence WHERE game_id = ? AND segment = ? AND ply >= ?",
                (game_id, segment, from_ply),
            )
            self._db.commit()

    def moves(self, game_id: str, *, withdrawn: bool = False) -> list[MoveRow]:
        clause = "" if withdrawn else "AND withdrawn = 0"
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM moves WHERE game_id = ? {clause} ORDER BY segment, ply",
                (game_id,),
            ).fetchall()
        return [_move_row(row) for row in rows]

    def move(self, move_id: int) -> MoveRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM moves WHERE move_id = ?", (move_id,)).fetchone()
        return None if row is None else _move_row(row)

    # ---- dialogue -----------------------------------------------------------------------

    def add_dialogue(
        self,
        game_id: str | None,
        ts: float,
        role: str,
        kind: str,
        text: str,
        ply: int | None,
        move_id: int | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO dialogue (game_id, move_id, ts, role, kind, ply, text)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (game_id, move_id, ts, role, kind, ply, text),
            )
            self._db.commit()

    # ---- evidence -------------------------------------------------------------------------

    def add_evidence(
        self,
        move_id: int,
        *,
        game_id: str,
        segment: int,
        ply: int,
        phase: str,
        expected_loss: float,
        cause: str | None,
        motif: str | None,
        time_pressure: bool,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO evidence (move_id, game_id, segment, ply, phase,"
                " expected_loss, cause, motif, time_pressure) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    move_id,
                    game_id,
                    segment,
                    ply,
                    phase,
                    expected_loss,
                    cause,
                    motif,
                    int(time_pressure),
                ),
            )
            self._db.commit()

    def evidence(
        self, game_id: str | None = None, *, limit: int | None = None
    ) -> list[EvidenceRow]:
        """The evidence rows, newest game first, for one game or all of them."""
        where = "WHERE e.game_id = ?" if game_id is not None else ""
        cap = f"LIMIT {int(limit)}" if limit is not None else ""
        args: tuple[object, ...] = (game_id,) if game_id is not None else ()
        with self._lock:
            rows = self._db.execute(
                "SELECT e.* FROM evidence e JOIN games g ON g.game_id = e.game_id"
                f" {where} ORDER BY g.started_at DESC, e.segment, e.ply {cap}",
                args,
            ).fetchall()
        return [_evidence_row(row) for row in rows]

    def correct_evidence(
        self, game_id: str, segment: int, ply: int, cause: str | None, motif: str | None = None
    ) -> EvidenceRow | None:
        """The user's correction of a mistake's cause (and motif)."""
        key = (game_id, segment, ply)
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM evidence WHERE game_id = ? AND segment = ? AND ply = ?", key
            ).fetchone()
            if row is None:
                return None
            new_motif = row["motif"] if motif is None else motif
            self._db.execute(
                "UPDATE evidence SET cause = ?, motif = ?, corrected = 1"
                " WHERE game_id = ? AND segment = ? AND ply = ?",
                (cause, new_motif, *key),
            )
            self._db.commit()
            fresh = self._db.execute(
                "SELECT * FROM evidence WHERE game_id = ? AND segment = ? AND ply = ?", key
            ).fetchone()
        return _evidence_row(fresh)

    def self_report(self, move_id: int, text: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE evidence SET self_report = ? WHERE move_id = ?", (text, move_id)
            )
            self._db.commit()

    # ---- the taxonomy's cells, computed --------------------------------------------------

    def cells(self) -> list[CellRow]:
        """The error taxonomy as counts: mistakes by (cause, motif, phase) with the score
        they cost; the occasions with cause None -- every judged move of the player's in
        its phase (motif None), and every such move with a tactic on the board, per
        tactic, with those of its occasions the player missed (a mistake that did not
        play it). The motif of a mistake is the tactic it allowed as often as the one it
        let go, so the mistakes alone never say how often a tactic on the board was missed:
        "about 100% (5 of 4)" counted the forks walked into over the forks the player had
        (the review of 2026-10-01). Computed from the record, so a correction or a takeback
        needs no bookkeeping."""
        with self._lock:
            errors = self._db.execute(
                "SELECT cause, motif, phase, COUNT(*) AS n, SUM(expected_loss) AS loss,"
                " json_group_array(DISTINCT game_id) AS games"
                " FROM evidence GROUP BY cause, motif, phase"
            ).fetchall()
            phases = self._db.execute(
                "SELECT phase, COUNT(*) AS n FROM moves"
                " WHERE by_user = 1 AND withdrawn = 0 AND severity IS NOT NULL"
                " AND phase IS NOT NULL GROUP BY phase"
            ).fetchall()
            missed = (
                "(e.move_id IS NOT NULL AND NOT EXISTS"
                " (SELECT 1 FROM json_each(m.tags) p WHERE p.value = t.value))"
            )
            motifs = self._db.execute(
                "SELECT m.phase AS phase, t.value AS motif, COUNT(*) AS n,"
                f" SUM({missed}) AS missed,"
                f" json_group_array(DISTINCT CASE WHEN {missed} THEN m.game_id END) AS games"
                " FROM moves m JOIN json_each(m.available_tags) t"
                " LEFT JOIN evidence e ON e.move_id = m.move_id"
                " WHERE m.by_user = 1 AND m.withdrawn = 0 AND m.severity IS NOT NULL"
                " GROUP BY m.phase, t.value"
            ).fetchall()
        out = [
            CellRow(
                cause=r["cause"],
                motif=r["motif"],
                phase=r["phase"],
                errors=int(r["n"]),
                chances=0,
                expected_loss=float(r["loss"] or 0.0),
                games=_ids(r["games"]),
            )
            for r in errors
        ]
        out += [
            CellRow(
                cause=None,
                motif=None,
                phase=r["phase"],
                errors=0,
                chances=int(r["n"]),
                expected_loss=0.0,
            )
            for r in phases
        ]
        out += [
            CellRow(
                cause=None,
                motif=r["motif"],
                phase=r["phase"],
                errors=int(r["missed"] or 0),
                chances=int(r["n"]),
                expected_loss=0.0,
                games=_ids(r["games"]),
            )
            for r in motifs
        ]
        return sorted(out, key=lambda c: -c.expected_loss)

    # ---- sessions ---------------------------------------------------------------------

    def start_session(self, session_id: str, started_at: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, started_at),
            )
            self._db.commit()

    # ---- puzzles ------------------------------------------------------------------------

    def setting(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._db.commit()

    def add_recommendation(
        self, game_id: str, ts: float, motif: str, reason: str, level: int, puzzle_ids: list[str]
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO recommendations (game_id, ts, motif, reason, level, puzzle_ids)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (game_id, ts, motif, reason, level, " ".join(puzzle_ids)),
            )
            self._db.commit()

    def add_attempt(
        self, puzzle_id: str, ts: float, motif: str | None, solved: bool, moves: list[str]
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO puzzle_attempts (puzzle_id, ts, motif, solved, moves)"
                " VALUES (?, ?, ?, ?, ?)",
                (puzzle_id, ts, motif, int(solved), " ".join(moves)),
            )
            self._db.commit()

    def attempts(self, limit: int = 100) -> list[tuple[str, float, bool]]:
        """(puzzle_id, ts, solved) of the latest attempts, latest first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT puzzle_id, ts, solved FROM puzzle_attempts ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [(r["puzzle_id"], float(r["ts"]), bool(r["solved"])) for r in rows]

    def end_session(self, session_id: str, ended_at: float, summary: str | None) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE sessions SET ended_at = ?, summary = ? WHERE session_id = ?",
                (ended_at, summary, session_id),
            )
            self._db.commit()

    def session(self, session_id: str) -> SessionRow | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return None if row is None else SessionRow(**dict(row))

    def last_session(self, before: str | None = None) -> SessionRow | None:
        """The most recent session with a summary, other than `before`."""
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM sessions WHERE summary IS NOT NULL AND session_id IS NOT ?"
                " ORDER BY started_at DESC LIMIT 1",
                (before,),
            ).fetchone()
        return None if row is None else SessionRow(**dict(row))

    # ---- the review's library and its reviews ----------------------------------------------

    def add_library_game(self, row: LibraryRow) -> None:
        """A game imported for review; the same game imported again replaces its row."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO library (key, source, added_at, played_at, white, black,"
                " white_elo, black_elo, result, time_class, user_side, pgn)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    row.key,
                    row.source,
                    row.added_at,
                    row.played_at,
                    row.white,
                    row.black,
                    row.white_elo,
                    row.black_elo,
                    row.result,
                    row.time_class,
                    row.user_side,
                    row.pgn,
                ),
            )
            self._db.commit()

    def fill_sides(self, names: Sequence[str]) -> None:
        """The player's side in each imported game that has none, by their usernames (lower
        case): a game brought in before the username was set."""
        if not names:
            return
        marks = ",".join("?" * len(names))
        with self._lock:
            self._db.execute(
                f"UPDATE library SET user_side = CASE WHEN lower(white) IN ({marks}) THEN 'white'"
                f" WHEN lower(black) IN ({marks}) THEN 'black' END"
                f" WHERE user_side IS NULL AND (lower(white) IN ({marks})"
                f" OR lower(black) IN ({marks}))",
                (*names, *names, *names, *names),
            )
            self._db.commit()

    def set_time_class(self, key: str, time_class: str | None) -> None:
        with self._lock:
            self._db.execute("UPDATE library SET time_class = ? WHERE key = ?", (time_class, key))
            self._db.commit()

    def library(self) -> list[LibraryRow]:
        """Every imported game, the latest played first. Every one: the Games window listed 500
        while the background review and Stats counted all (the review of 2026-10-01), and the
        games past it could not be seen, opened or removed."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM library ORDER BY COALESCE(played_at, added_at) DESC"
            ).fetchall()
        return [LibraryRow(**dict(r)) for r in rows]

    def library_game(self, key: str) -> LibraryRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM library WHERE key = ?", (key,)).fetchone()
        return None if row is None else LibraryRow(**dict(row))

    def remove_game(self, key: str) -> list[str]:
        """A game out of the player's record: every row of it -- the library's, or the watched
        record with its moves and what the coach drew from it -- its review and its Stats, and
        the game it is paired with (an import and the watched game it stands for are one game).
        What it took is kept UNDO_KEEP_S for `restore_game`; its key for good, so bringing
        games in leaves it out. Returns the keys removed."""
        now = time.time()
        with self._lock, self._db:
            paired = self._db.execute(
                "SELECT watched, imported FROM twins WHERE watched = ? OR imported = ?", (key, key)
            ).fetchall()
            gone = {str(r["key"]) for r in self._db.execute("SELECT key FROM removed").fetchall()}
            # A game removed already (a second click) keeps what it took, for its Undo.
            with_pair = [key, *sorted({k for p in paired for k in p} - {key})]
            group = [k for k in with_pair if k not in gone]
            for k in group:
                rows = {table: self._take(table, "key", k) for table in LIBRARY_TABLES}
                rows["twins"] = self._take_twins(k)
                trace = None
                if k.startswith(WATCHED):
                    game_id = k[len(WATCHED) :]
                    trace = self._trace(game_id)
                    for table in WATCHED_TABLES:
                        rows[table] = self._take(table, "game_id", game_id)
                self._db.execute(
                    "INSERT OR REPLACE INTO removed (key, removed_at, grp, rows, trace)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (k, now, key, json.dumps(rows), json.dumps(trace) if trace else None),
                )
            # What older removals took can no longer be put back: only their keys stay.
            self._db.execute(
                "UPDATE removed SET rows = NULL WHERE removed_at < ?", (now - UNDO_KEEP_S,)
            )
        return group

    def restore_game(self, key: str) -> list[str]:
        """A game the player removed moments ago put back as it was, with the game it was
        paired with; nothing once its rows are no longer kept. Returns the keys put back."""
        now = time.time()
        with self._lock, self._db:
            found = self._db.execute(
                "SELECT key, rows FROM removed"
                " WHERE grp = ? AND rows IS NOT NULL AND removed_at >= ?",
                (key, now - UNDO_KEEP_S),
            ).fetchall()
            for row in found:
                for table, items in json.loads(row["rows"]).items():
                    if table not in RESTORABLE:
                        continue
                    for item in items:
                        cols = list(item)
                        self._db.execute(
                            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)})"
                            f" VALUES ({', '.join('?' * len(cols))})",
                            [item[c] for c in cols],
                        )
                self._db.execute("DELETE FROM removed WHERE key = ?", (row["key"],))
        return [str(r["key"]) for r in found]

    def removed_keys(self) -> set[str]:
        """Every game the player removed, by its library key."""
        with self._lock:
            rows = self._db.execute("SELECT key FROM removed").fetchall()
        return {str(r["key"]) for r in rows}

    def removed_traces(self) -> list[Trace]:
        """The watched games the player removed, as much of them as tells their import."""
        with self._lock:
            rows = self._db.execute(
                "SELECT trace FROM removed WHERE trace IS NOT NULL ORDER BY key"
            ).fetchall()
        return [Trace(**json.loads(r["trace"])) for r in rows]

    def is_removed(self, key: str) -> bool:
        with self._lock:
            row = self._db.execute("SELECT 1 FROM removed WHERE key = ?", (key,)).fetchone()
        return row is not None

    def in_record(self, key: str) -> bool:
        """Whether a game (by its library key) is in the player's record: its import in the
        library, or Sidecoach's record of the watched game -- gone once removed, or once the
        profile is reset."""
        if key.startswith(WATCHED):
            table, column, value = "games", "game_id", key[len(WATCHED) :]
        else:
            table, column, value = "library", "key", key
        with self._lock:
            row = self._db.execute(f"SELECT 1 FROM {table} WHERE {column} = ?", (value,)).fetchone()
        return row is not None

    def unremove(self, keys: Sequence[str]) -> None:
        """Games the player brought in by hand (a pasted PGN): no longer left out."""
        with self._lock, self._db:
            self._db.executemany("DELETE FROM removed WHERE key = ?", [(k,) for k in keys])

    def _take(self, table: str, column: str, value: str) -> list[dict[str, Any]]:
        """A table's rows of one game, as they were, and gone from it."""
        rows = self._db.execute(f"SELECT * FROM {table} WHERE {column} = ?", (value,)).fetchall()
        self._db.execute(f"DELETE FROM {table} WHERE {column} = ?", (value,))
        return [dict(r) for r in rows]

    def _take_twins(self, key: str) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM twins WHERE watched = ? OR imported = ?", (key, key)
        ).fetchall()
        self._db.execute("DELETE FROM twins WHERE watched = ? OR imported = ?", (key, key))
        return [dict(r) for r in rows]

    def _trace(self, game_id: str) -> dict[str, Any] | None:
        game = self._db.execute(
            "SELECT started_at, user_side, result FROM games WHERE game_id = ?", (game_id,)
        ).fetchone()
        if game is None:
            return None
        moves = self._db.execute(
            "SELECT san FROM moves WHERE game_id = ? AND withdrawn = 0 ORDER BY segment, ply",
            (game_id,),
        ).fetchall()
        return {
            "started_at": game["started_at"],
            "user_side": game["user_side"],
            "result": game["result"],
            "sans": [str(m["san"]).rstrip("+#") for m in moves],
        }

    def add_twin(self, watched: str, imported: str) -> None:
        """A watched game's import (library keys): the import stands for it from now on."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO twins (watched, imported, found_at) VALUES (?, ?, ?)",
                (watched, imported, time.time()),
            )
            self._db.commit()

    def twins(self) -> dict[str, str]:
        """The watched games that were imported too: watched key -> imported key."""
        with self._lock:
            rows = self._db.execute("SELECT watched, imported FROM twins").fetchall()
        return {str(r["watched"]): str(r["imported"]) for r in rows}

    # ---- Stats: a reviewed game's facts (profile/stat_facts.py) --------------------------------

    def save_stats(self, strength: str, game: dict[str, Any], moves: list[dict[str, Any]]) -> None:
        """A game's facts from its review at `strength`, in place of any before from a review
        no stronger: Stats keeps the strongest review's facts."""
        with self._lock:
            key = game["key"]
            kept = self._db.execute(
                f"SELECT {_RANK} AS rank FROM stat_games WHERE key = ?", (key,)
            ).fetchone()
            if kept is not None and kept["rank"] > STRENGTH_RANK.get(strength, 0):
                return
            self._db.execute("DELETE FROM stat_moves WHERE key = ?", (key,))
            self._db.execute(
                "INSERT OR REPLACE INTO stat_games (key, strength, source, time_class, played_at,"
                " user_side, bot, result, opening, termination) VALUES"
                " (:key, :strength, :source, :time_class, :played_at, :user_side, :bot, :result,"
                " :opening, :termination)",
                {**game, "strength": strength, "bot": int(game["bot"])},
            )
            self._db.executemany(
                "INSERT INTO stat_moves (key, ply, side, fen, uci, san, phase, loss, label,"
                " accuracy, score, chance, found, allowed, structure, themes, endgame, spent_ms,"
                " clock_ms)"
                " VALUES"
                " (:key, :ply, :side, :fen, :uci, :san, :phase, :loss, :label, :accuracy, :score,"
                " :chance, :found, :allowed, :structure, :themes, :endgame, :spent_ms, :clock_ms)",
                [{**m, "key": key, "found": int(m["found"])} for m in moves],
            )
            self._db.commit()

    def stat_games(self) -> list[dict[str, Any]]:
        """The reviewed games' facts, each game once: a watched game its import stands for is
        left out once the import's own facts are kept. Until then the watched game counts (the
        review of 2026-10-01: a game whose record differed from its import left Stats the moment
        the import came in, until the import was reviewed -- on the phone, until it was opened).
        The player's side and the time class are the library's as it is now, when it has the
        game: a game reviewed before the username was set kept no side and never reached Stats
        (the review of 2026-10-01)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT s.key, s.strength, s.source, s.played_at, s.bot, s.result, s.opening,"
                " s.termination, COALESCE(l.user_side, s.user_side) AS user_side,"
                " COALESCE(l.time_class, s.time_class) AS time_class"
                " FROM stat_games s LEFT JOIN library l ON l.key = s.key"
                f" WHERE s.key NOT IN ({_STOOD_FOR}) ORDER BY s.played_at"
            ).fetchall()
        return [{**dict(r), "bot": bool(r["bot"])} for r in rows]

    def stat_moves(self, keys: list[str]) -> list[dict[str, Any]]:
        """The facts of the games' moves, in order."""
        if not keys:
            return []
        marks = ",".join("?" * len(keys))
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM stat_moves WHERE key IN ({marks}) ORDER BY key, ply", keys
            ).fetchall()
        return [{**dict(r), "found": bool(r["found"])} for r in rows]

    def unreviewed(self) -> list[LibraryRow]:
        """The library's imported games no review covers yet, the latest played first."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM library WHERE key NOT IN (SELECT key FROM reviews)"
                " ORDER BY COALESCE(played_at, added_at) DESC"
            ).fetchall()
        return [LibraryRow(**dict(r)) for r in rows]

    def reviews_without_stats(self) -> list[ReviewRow]:
        """Each game's latest review whose facts are not kept yet (a watched game its import
        stands for needs none once the import's are kept: `stat_games`)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT r.* FROM reviews r WHERE r.key NOT IN (SELECT key FROM stat_games)"
                f" AND r.key NOT IN ({_STOOD_FOR})"
                " AND r.reviewed_at = (SELECT MAX(reviewed_at) FROM reviews WHERE key = r.key)"
            ).fetchall()
        return [ReviewRow(**dict(r)) for r in rows]

    def save_review(
        self,
        key: str,
        strength: str,
        version: int,
        reviewed_at: float,
        report: str,
        summary: str | None,
    ) -> bool:
        """A game reviewed whole; whether it was saved. A review of an older format is replaced,
        and its words go with it. One of this format saved already stays: another reviewer
        finished first (the review of 2026-10-01: the background review and a review window on
        one game at once -- the window saved first, the player paid for the AI coach's words,
        and the background's save wiped them). Nor is a game the player removed meanwhile
        saved. One statement, so no other connection's save or removal comes in between."""
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO reviews (key, strength, version, reviewed_at, report, summary, words)"
                " SELECT ?, ?, ?, ?, ?, ?, NULL WHERE NOT EXISTS"
                " (SELECT 1 FROM removed WHERE key = ?)"
                " ON CONFLICT (key, strength) DO UPDATE SET version = excluded.version,"
                " reviewed_at = excluded.reviewed_at, report = excluded.report,"
                " summary = excluded.summary, words = NULL"
                " WHERE reviews.version != excluded.version",
                (key, strength, version, reviewed_at, report, summary, key),
            )
            self._db.commit()
        return cursor.rowcount > 0

    def review(self, key: str, strength: str) -> ReviewRow | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM reviews WHERE key = ? AND strength = ?", (key, strength)
            ).fetchone()
        return None if row is None else ReviewRow(**dict(row))

    def saved_reviews(self) -> list[tuple[str, str]]:
        """Every saved review, by its game's key and its strength."""
        with self._lock:
            rows = self._db.execute("SELECT key, strength FROM reviews").fetchall()
        return [(str(r["key"]), str(r["strength"])) for r in rows]

    def set_review_summary(self, key: str, strength: str, summary: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE reviews SET summary = ? WHERE key = ? AND strength = ?",
                (summary, key, strength),
            )
            self._db.commit()

    def stat_strengths(self) -> dict[str, str]:
        """The strength of the review each game's Stats facts come from, by its key."""
        with self._lock:
            rows = self._db.execute("SELECT key, strength FROM stat_games").fetchall()
        return {str(r["key"]): str(r["strength"]) for r in rows}

    def best_review(self, key: str) -> ReviewRow | None:
        """The game's strongest review (deep, standard, fast), the latest of that strength: the
        one the review opens at, Your games shows and Stats counts. The latest was shown, and a
        Fast look after a Deep review lowered them all (the review of 2026-10-01)."""
        with self._lock:
            row = self._db.execute(
                f"SELECT * FROM reviews WHERE key = ? ORDER BY {_RANK} DESC, reviewed_at DESC"
                " LIMIT 1",
                (key,),
            ).fetchone()
        return None if row is None else ReviewRow(**dict(row))

    def add_progress(
        self, key: str, strength: str, depth: int, multipv: int, fen: str, measured: str
    ) -> None:
        """A position of a review under way, measured (committed at once: a kill keeps it)."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO review_progress"
                " (key, strength, depth, multipv, fen, measured) VALUES (?, ?, ?, ?, ?, ?)",
                (key, strength, depth, multipv, fen, measured),
            )
            self._db.commit()

    def progress(self, key: str, strength: str, depth: int, multipv: int) -> list[str]:
        """The positions of a review under way measured so far, at this depth and lines."""
        with self._lock:
            rows = self._db.execute(
                "SELECT measured FROM review_progress"
                " WHERE key = ? AND strength = ? AND depth = ? AND multipv = ?",
                (key, strength, depth, multipv),
            ).fetchall()
        return [str(r["measured"]) for r in rows]

    def clear_progress(self, key: str, strength: str | None = None) -> None:
        """A game's review at `strength` is saved: what a review of it at that strength kept
        part way goes -- every strength's when none is named (the game is gone). A Deep review
        stopped part way kept its positions only until the game's Standard review was saved or
        opened, and started over (the review of 2026-10-01)."""
        with self._lock:
            if strength is None:
                self._db.execute("DELETE FROM review_progress WHERE key = ?", (key,))
            else:
                self._db.execute(
                    "DELETE FROM review_progress WHERE key = ? AND strength = ?", (key, strength)
                )
            self._db.commit()

    def set_review_words(self, key: str, strength: str, words: str) -> None:
        """The coach's words on a review's moves, as far as they are prepared."""
        with self._lock:
            self._db.execute(
                "UPDATE reviews SET words = ? WHERE key = ? AND strength = ?",
                (words, key, strength),
            )
            self._db.commit()

    # ---- read-only SQL from the coach -----------------------------------------------------

    def query(
        self, sql: str, *, rows: int = QUERY_ROWS, seconds: float = QUERY_SECONDS
    ) -> tuple[list[str], list[tuple[object, ...]]]:
        """Run one read-only statement and return (column names, rows), at most `rows` of
        them. Anything but a single SELECT (or WITH ... SELECT) over the tables is
        refused by SQLite's authorizer before it runs; a statement still running after
        `seconds` is interrupted. Used by the coach's own queries (ask.py)."""
        text = sql.strip().rstrip(";").strip()
        if not text:
            raise QueryRefused("empty query")
        if ";" in text:
            raise QueryRefused("one statement at a time")
        head = text.split(None, 1)[0].upper()
        if head not in ("SELECT", "WITH"):
            raise QueryRefused("only SELECT queries are allowed")

        def authorize(action: int, *_: object) -> int:
            return sqlite3.SQLITE_OK if action in _READ_OPS else sqlite3.SQLITE_DENY

        with self._lock:
            timer = threading.Timer(seconds, self._db.interrupt)
            self._db.set_authorizer(authorize)
            try:
                timer.start()
                cursor = self._db.execute(text)
                names = [d[0] for d in cursor.description or ()]
                fetched = cursor.fetchmany(rows + 1)
            except sqlite3.DatabaseError as exc:
                raise QueryRefused(str(exc)) from exc
            finally:
                timer.cancel()
                self._db.set_authorizer(None)
        return names, [tuple(r) for r in fetched[:rows]]
