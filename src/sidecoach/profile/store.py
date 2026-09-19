"""SQLite storage for the games and the player model: one file, `<data_dir>/profile.sqlite`.

Tables (DESIGN section 7): games; moves, each with the engine's verdict once it arrives;
dialogue, what the player and the coach said; evidence, one row per mistake of the
player's with its cause and motif (user-correctable); cells, the Beta counts of the
error taxonomy; sessions, with their summaries. Recordings and PGNs stay on disk.

Everything is written as it happens, so a session that ends abruptly loses at most the
last game's summary. `Store(None)` is in memory, for tests and for replays, which must
not teach the profile anything.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    user_side TEXT NOT NULL,
    start_fen TEXT NOT NULL,
    result TEXT,
    reason TEXT,
    opening TEXT,
    accuracy REAL,
    pgn TEXT,
    final_fen TEXT
);
CREATE TABLE IF NOT EXISTS moves (
    game_id TEXT NOT NULL,
    segment INTEGER NOT NULL DEFAULT 0,
    ply INTEGER NOT NULL,
    san TEXT NOT NULL,
    uci TEXT NOT NULL,
    by_user INTEGER NOT NULL,
    fen_before TEXT NOT NULL,
    fen_after TEXT NOT NULL,
    think_ms INTEGER,
    ts REAL NOT NULL,
    label TEXT,
    severity TEXT,
    expected_loss REAL,
    best_uci TEXT,
    accuracy REAL,
    tags TEXT,
    depth INTEGER,
    withdrawn INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (game_id, segment, ply)
);
CREATE TABLE IF NOT EXISTS dialogue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    ply INTEGER,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    game_id TEXT NOT NULL,
    segment INTEGER NOT NULL DEFAULT 0,
    ply INTEGER NOT NULL,
    phase TEXT NOT NULL,
    expected_loss REAL NOT NULL,
    cause TEXT,
    motif TEXT,
    time_pressure INTEGER NOT NULL DEFAULT 0,
    self_report TEXT,
    corrected INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (game_id, segment, ply)
);
CREATE TABLE IF NOT EXISTS cells (
    cause TEXT NOT NULL DEFAULT '',
    motif TEXT NOT NULL DEFAULT '',
    phase TEXT NOT NULL DEFAULT '',
    errors INTEGER NOT NULL DEFAULT 0,
    chances INTEGER NOT NULL DEFAULT 0,
    expected_loss REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (cause, motif, phase)
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at REAL NOT NULL,
    ended_at REAL,
    games INTEGER NOT NULL DEFAULT 0,
    summary TEXT
);
"""


@dataclass(frozen=True)
class GameRow:
    game_id: str
    session_id: str | None
    started_at: float
    ended_at: float | None
    user_side: str
    start_fen: str
    result: str | None
    reason: str | None
    opening: str | None
    accuracy: float | None
    pgn: str | None
    final_fen: str | None


@dataclass(frozen=True)
class MoveRow:
    game_id: str
    segment: int
    ply: int
    san: str
    uci: str
    by_user: bool
    fen_before: str
    fen_after: str
    think_ms: int | None
    ts: float
    label: str | None
    severity: str | None
    expected_loss: float | None
    best_uci: str | None
    accuracy: float | None
    tags: list[str]
    depth: int | None
    withdrawn: bool


@dataclass(frozen=True)
class EvidenceRow:
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
    cause: str | None
    motif: str | None
    phase: str | None
    errors: int
    chances: int
    expected_loss: float


@dataclass(frozen=True)
class SessionRow:
    session_id: str
    started_at: float
    ended_at: float | None
    games: int
    summary: str | None


def _key(value: str | None) -> str:
    return value or ""


def _unkey(value: str) -> str | None:
    return value or None


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
        opening: str | None = None,
    ) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE games SET ended_at = ?, result = ?, reason = ?, pgn = ?, final_fen = ?,"
                " accuracy = ?, opening = ? WHERE game_id = ?",
                (ended_at, result, reason, pgn, final_fen, accuracy, opening, game_id),
            )
            self._db.commit()

    def set_user_side(self, game_id: str, user_side: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE games SET user_side = ? WHERE game_id = ?", (user_side, game_id)
            )
            self._db.commit()

    def game(self, game_id: str) -> GameRow | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
        return None if row is None else GameRow(**dict(row))

    def games(self, limit: int = 50, *, finished: bool = True) -> list[GameRow]:
        """The most recent games, newest first."""
        clause = "WHERE result IS NOT NULL" if finished else ""
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM games {clause} ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [GameRow(**dict(row)) for row in rows]

    # ---- moves ------------------------------------------------------------------------

    def add_move(
        self,
        game_id: str,
        segment: int,
        ply: int,
        san: str,
        uci: str,
        *,
        by_user: bool,
        fen_before: str,
        fen_after: str,
        think_ms: int | None,
        ts: float,
    ) -> None:
        """A move as it was played. `segment` counts the position resyncs before it: the
        ply numbering restarts after each one, and the earlier moves stay on record."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO moves (game_id, segment, ply, san, uci, by_user,"
                " fen_before, fen_after, think_ms, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    segment,
                    ply,
                    san,
                    uci,
                    int(by_user),
                    fen_before,
                    fen_after,
                    think_ms,
                    ts,
                ),
            )
            self._db.commit()

    def assess_move(
        self,
        game_id: str,
        segment: int,
        ply: int,
        *,
        label: str | None,
        severity: str,
        expected_loss: float,
        best_uci: str | None,
        accuracy: float | None,
        tags: list[str],
        depth: int,
    ) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE moves SET label = ?, severity = ?, expected_loss = ?, best_uci = ?,"
                " accuracy = ?, tags = ?, depth = ? WHERE game_id = ? AND segment = ? AND ply = ?",
                (
                    label,
                    severity,
                    expected_loss,
                    best_uci,
                    accuracy,
                    json.dumps(tags),
                    depth,
                    game_id,
                    segment,
                    ply,
                ),
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
                f"SELECT * FROM moves WHERE game_id = ? {clause} ORDER BY segment, ply", (game_id,)
            ).fetchall()
        out = []
        for row in rows:
            data = dict(row)
            data["by_user"] = bool(data["by_user"])
            data["withdrawn"] = bool(data["withdrawn"])
            data["tags"] = json.loads(data["tags"]) if data["tags"] else []
            out.append(MoveRow(**data))
        return out

    # ---- dialogue -----------------------------------------------------------------------

    def add_dialogue(
        self, game_id: str | None, ts: float, role: str, kind: str, text: str, ply: int | None
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO dialogue (game_id, ts, role, kind, ply, text)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (game_id, ts, role, kind, ply, text),
            )
            self._db.commit()

    # ---- evidence and cells ---------------------------------------------------------------

    def add_evidence(
        self,
        game_id: str,
        segment: int,
        ply: int,
        *,
        phase: str,
        expected_loss: float,
        cause: str | None,
        motif: str | None,
        time_pressure: bool,
    ) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO evidence (game_id, segment, ply, phase, expected_loss,"
                " cause, motif, time_pressure) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (game_id, segment, ply, phase, expected_loss, cause, motif, int(time_pressure)),
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
        out = []
        for row in rows:
            data = dict(row)
            data["time_pressure"] = bool(data["time_pressure"])
            data["corrected"] = bool(data["corrected"])
            out.append(EvidenceRow(**data))
        return out

    def correct_evidence(
        self, game_id: str, segment: int, ply: int, cause: str | None, motif: str | None = None
    ) -> EvidenceRow | None:
        """The user's correction of a mistake's cause (and motif): the cells move with it."""
        key = (game_id, segment, ply)
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM evidence WHERE game_id = ? AND segment = ? AND ply = ?", key
            ).fetchone()
            if row is None:
                return None
            old = dict(row)
            new_motif = old["motif"] if motif is None else motif
            self._db.execute(
                "UPDATE evidence SET cause = ?, motif = ?, corrected = 1"
                " WHERE game_id = ? AND segment = ? AND ply = ?",
                (cause, new_motif, *key),
            )
            self._bump(
                old["cause"], old["motif"], old["phase"], errors=-1, loss=-old["expected_loss"]
            )
            self._bump(cause, new_motif, old["phase"], errors=1, loss=old["expected_loss"])
            self._db.commit()
            fresh = self._db.execute(
                "SELECT * FROM evidence WHERE game_id = ? AND segment = ? AND ply = ?", key
            ).fetchone()
        data = dict(fresh)
        data["time_pressure"] = bool(data["time_pressure"])
        data["corrected"] = bool(data["corrected"])
        return EvidenceRow(**data)

    def self_report(self, game_id: str, segment: int, ply: int, text: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE evidence SET self_report = ? WHERE game_id = ? AND segment = ? AND ply = ?",
                (text, game_id, segment, ply),
            )
            self._db.commit()

    def _bump(
        self,
        cause: str | None,
        motif: str | None,
        phase: str | None,
        *,
        errors: int = 0,
        chances: int = 0,
        loss: float = 0.0,
    ) -> None:
        self._db.execute(
            "INSERT INTO cells (cause, motif, phase, errors, chances, expected_loss)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(cause, motif, phase) DO UPDATE SET errors = errors + excluded.errors,"
            " chances = chances + excluded.chances,"
            " expected_loss = expected_loss + excluded.expected_loss",
            (_key(cause), _key(motif), _key(phase), errors, chances, loss),
        )

    def bump_cell(
        self,
        cause: str | None,
        motif: str | None,
        phase: str | None,
        *,
        errors: int = 0,
        chances: int = 0,
        loss: float = 0.0,
    ) -> None:
        """Add to a cell's counts. Errors and chances are counted separately: an occasion
        (a tactic on the board, a move in a phase) is a chance, a mistake an error."""
        with self._lock:
            self._bump(cause, motif, phase, errors=errors, chances=chances, loss=loss)
            self._db.commit()

    def cells(self) -> list[CellRow]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM cells WHERE errors > 0 OR chances > 0 ORDER BY expected_loss DESC"
            ).fetchall()
        return [
            CellRow(
                cause=_unkey(row["cause"]),
                motif=_unkey(row["motif"]),
                phase=_unkey(row["phase"]),
                errors=row["errors"],
                chances=row["chances"],
                expected_loss=row["expected_loss"],
            )
            for row in rows
        ]

    # ---- sessions ---------------------------------------------------------------------

    def start_session(self, session_id: str, started_at: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO sessions (session_id, started_at) VALUES (?, ?)",
                (session_id, started_at),
            )
            self._db.commit()

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

    def games_count(self) -> int:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM games WHERE result IS NOT NULL"
            ).fetchone()
        return int(row["n"])
