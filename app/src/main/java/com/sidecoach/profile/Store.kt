package com.sidecoach.profile

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteException
import android.database.sqlite.SQLiteStatement
import android.os.CancellationSignal
import android.os.OperationCanceledException
import org.json.JSONArray
import java.io.File
import java.util.Timer
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.schedule
import kotlin.concurrent.withLock

/**
 * SQLite storage for the games and the player model: one file, `profile.sqlite` in the app's
 * files. `sidecoach/profile/store.py`.
 *
 * The record is written for questions to be asked of it -- by the code that builds the
 * review, and by the coach writing its own SQL (Ask.kt). So it is denormalised on purpose:
 * every move carries the facts a question might need (the phase, the expected score after it
 * from the player's side, the player's clock, the material, the tactics that were on the board
 * and the ones the move allowed), and one key, `move_id`, joins everything. `SCHEMA_DOC` in
 * Ask.kt is the same schema in words, with worked queries.
 *
 * Tables: games; moves, each with the engine's verdict once it arrives; evidence, one row per
 * mistake of the player's with its cause and motif (user-correctable); dialogue, what the
 * player and the coach said; sessions, with their summaries. Everything is written as it
 * happens, so a session that ends abruptly loses at most the last game's summary.
 * `Store(null)` is in memory.
 *
 * Android's SQLite, not Python's, underneath; two things differ and are met here. A query's
 * arguments are bound as strings, so integers are written into the SQL itself (they are the
 * code's own numbers, never text from outside) and only strings are bound. And there is no
 * authorizer for the coach's read-only SQL: `PRAGMA query_only` gives the same guarantee --
 * the database refuses every write while the coach's statement runs -- beside the same
 * statement checks the desktop makes first.
 */
class Store(val path: File?) {

    private val db: SQLiteDatabase
    private val lock = ReentrantLock()

    init {
        path?.parentFile?.mkdirs()
        db = if (path != null) SQLiteDatabase.openOrCreateDatabase(path, null) else SQLiteDatabase.create(null)
        lock.withLock {
            for (statement in SCHEMA.split(";").map { it.trim() }.filter { it.isNotEmpty() }) db.execSQL(statement)
        }
    }

    fun close() {
        lock.withLock { db.close() }
    }

    // ---- statements ------------------------------------------------------------------------

    /** A write, its arguments bound by type as Python's sqlite3 binds them. */
    private fun statement(sql: String, args: Array<out Any?>): SQLiteStatement {
        val statement = db.compileStatement(sql)
        for ((i, value) in args.withIndex()) {
            val at = i + 1
            when (value) {
                null -> statement.bindNull(at)
                is Boolean -> statement.bindLong(at, if (value) 1 else 0)
                is Int -> statement.bindLong(at, value.toLong())
                is Long -> statement.bindLong(at, value)
                is Double -> statement.bindDouble(at, value)
                is Float -> statement.bindDouble(at, value.toDouble())
                else -> statement.bindString(at, value.toString())
            }
        }
        return statement
    }

    private fun exec(sql: String, vararg args: Any?): Int =
        statement(sql, args).use { it.executeUpdateDelete() }

    private fun insert(sql: String, vararg args: Any?): Long =
        statement(sql, args).use { it.executeInsert() }

    /** A read; only strings are bound (see the class's note). */
    private fun <T> rows(sql: String, vararg args: String, map: (Cursor) -> T): List<T> {
        db.rawQuery(sql, if (args.isEmpty()) null else arrayOf(*args)).use { cursor ->
            val out = ArrayList<T>()
            while (cursor.moveToNext()) out.add(map(cursor))
            return out
        }
    }

    // ---- games ------------------------------------------------------------------------------

    fun startGame(gameId: String, startedAt: Double, userSide: String, startFen: String, sessionId: String? = null) {
        lock.withLock {
            exec(
                "INSERT OR REPLACE INTO games (game_id, session_id, started_at, user_side, start_fen)" +
                    " VALUES (?, ?, ?, ?, ?)",
                gameId, sessionId, startedAt, userSide, startFen,
            )
            if (sessionId != null) exec("UPDATE sessions SET games = games + 1 WHERE session_id = ?", sessionId)
        }
    }

    /** Set columns of a game as they become known: the time control, the opponent. */
    fun setGame(gameId: String, fields: Map<String, Any?>) {
        if (fields.isEmpty()) return
        val assignments = fields.keys.joinToString(", ") { "$it = ?" }
        lock.withLock { exec("UPDATE games SET $assignments WHERE game_id = ?", *fields.values.toTypedArray(), gameId) }
    }

    fun endGame(
        gameId: String,
        endedAt: Double,
        result: String,
        reason: String?,
        pgn: String? = null,
        finalFen: String? = null,
        accuracy: Double? = null,
        finalScore: Double? = null,
        opening: String? = null,
    ) {
        lock.withLock {
            exec(
                "UPDATE games SET ended_at = ?, result = ?, reason = ?, pgn = ?, final_fen = ?," +
                    " accuracy = ?, final_score = ?, opening = ? WHERE game_id = ?",
                endedAt, result, reason, pgn, finalFen, accuracy, finalScore, opening, gameId,
            )
        }
    }

    fun game(gameId: String): GameRow? = lock.withLock {
        rows("SELECT * FROM games WHERE game_id = ?", gameId, map = ::gameRow).firstOrNull()
    }

    /** The most recent games, newest first. */
    fun games(limit: Int = 50, finished: Boolean = true): List<GameRow> {
        val clause = if (finished) "WHERE result IS NOT NULL" else ""
        return lock.withLock {
            rows("SELECT * FROM games $clause ORDER BY started_at DESC LIMIT $limit", map = ::gameRow)
        }
    }

    /** Every row of a game: one followed provisionally that turned out not to be one. */
    fun deleteGame(gameId: String) {
        lock.withLock {
            db.beginTransaction()
            try {
                for (table in listOf("evidence", "dialogue", "recommendations", "moves", "games")) {
                    exec("DELETE FROM $table WHERE game_id = ?", gameId)
                }
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
        }
    }

    /**
     * Games with no result and fewer than [minPlies] moves -- positions the app was closed on
     * before they could qualify -- removed; returns how many.
     */
    fun purgeUnfinished(minPlies: Int): Int {
        val ids = lock.withLock {
            rows(
                "SELECT g.game_id FROM games g LEFT JOIN moves m ON m.game_id = g.game_id " +
                    "WHERE g.result IS NULL GROUP BY g.game_id HAVING COUNT(m.move_id) < $minPlies",
            ) { it.getString(0) }
        }
        for (id in ids) deleteGame(id)
        return ids.size
    }

    fun gamesCount(): Int = lock.withLock {
        rows("SELECT COUNT(*) AS n FROM games WHERE result IS NOT NULL") { it.getInt(0) }.first()
    }

    // ---- moves -------------------------------------------------------------------------------

    /**
     * A move as it was played; its id. [segment] counts the position resyncs before it: the
     * ply numbering restarts after each one, and the earlier moves stay on record. A move
     * written twice at one place replaces the first (a takeback replayed).
     */
    fun addMove(
        gameId: String,
        segment: Int,
        ply: Int,
        moveNumber: Int,
        san: String,
        uci: String,
        byUser: Boolean,
        fenBefore: String,
        fenAfter: String,
        phase: String?,
        thinkMs: Int?,
        clockMs: Int?,
        capture: Boolean,
        givesCheck: Boolean,
        material: Int?,
        ts: Double,
    ): Long = lock.withLock {
        exec("DELETE FROM moves WHERE game_id = ? AND segment = ? AND ply = ?", gameId, segment, ply)
        insert(
            "INSERT INTO moves (game_id, segment, ply, move_number, san, uci, by_user," +
                " fen_before, fen_after, phase, think_ms, clock_ms, capture, gives_check," +
                " material, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            gameId, segment, ply, moveNumber, san, uci, byUser, fenBefore, fenAfter, phase,
            thinkMs, clockMs, capture, givesCheck, material, ts,
        )
    }

    fun assessMove(
        moveId: Long,
        label: String?,
        severity: String,
        expectedLoss: Double,
        bestUci: String?,
        bestSan: String?,
        accuracy: Double?,
        tags: List<String>,
        availableTags: List<String>,
        depth: Int,
        phase: String? = null,
    ) {
        lock.withLock {
            exec(
                "UPDATE moves SET label = ?, severity = ?, expected_loss = ?, best_uci = ?," +
                    " best_san = ?, accuracy = ?, tags = ?, available_tags = ?, depth = ?," +
                    " phase = COALESCE(?, phase) WHERE move_id = ?",
                label, severity, expectedLoss, bestUci, bestSan, accuracy,
                jsonList(tags), jsonList(availableTags), depth, phase, moveId,
            )
        }
    }

    /**
     * What the position after the move holds: the expected score from the player's side, and
     * the tactics the mover allowed.
     */
    fun analyseMove(moveId: Long, scoreAfter: Double?, allowedTags: List<String>) {
        lock.withLock {
            exec("UPDATE moves SET score_after = ?, allowed_tags = ? WHERE move_id = ?", scoreAfter, jsonList(allowedTags), moveId)
        }
    }

    /** A takeback: the moves from [fromPly] on are out of the game, kept for the record. */
    fun withdrawMoves(gameId: String, segment: Int, fromPly: Int) {
        lock.withLock {
            exec("UPDATE moves SET withdrawn = 1 WHERE game_id = ? AND segment = ? AND ply >= ?", gameId, segment, fromPly)
            exec("DELETE FROM evidence WHERE game_id = ? AND segment = ? AND ply >= ?", gameId, segment, fromPly)
        }
    }

    fun moves(gameId: String, withdrawn: Boolean = false): List<MoveRow> {
        val clause = if (withdrawn) "" else "AND withdrawn = 0"
        return lock.withLock {
            rows("SELECT * FROM moves WHERE game_id = ? $clause ORDER BY segment, ply", gameId, map = ::moveRow)
        }
    }

    fun move(moveId: Long): MoveRow? = lock.withLock {
        rows("SELECT * FROM moves WHERE move_id = $moveId", map = ::moveRow).firstOrNull()
    }

    // ---- dialogue ------------------------------------------------------------------------------

    fun addDialogue(gameId: String?, ts: Double, role: String, kind: String, text: String, ply: Int?, moveId: Long? = null) {
        lock.withLock {
            exec(
                "INSERT INTO dialogue (game_id, move_id, ts, role, kind, ply, text) VALUES (?, ?, ?, ?, ?, ?, ?)",
                gameId, moveId, ts, role, kind, ply, text,
            )
        }
    }

    // ---- evidence ------------------------------------------------------------------------------

    fun addEvidence(
        moveId: Long,
        gameId: String,
        segment: Int,
        ply: Int,
        phase: String,
        expectedLoss: Double,
        cause: String?,
        motif: String?,
        timePressure: Boolean,
    ) {
        lock.withLock {
            exec(
                "INSERT OR REPLACE INTO evidence (move_id, game_id, segment, ply, phase," +
                    " expected_loss, cause, motif, time_pressure) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                moveId, gameId, segment, ply, phase, expectedLoss, cause, motif, timePressure,
            )
        }
    }

    /** The evidence rows, newest game first, for one game or all of them. */
    fun evidence(gameId: String? = null, limit: Int? = null): List<EvidenceRow> {
        val where = if (gameId != null) "WHERE e.game_id = ?" else ""
        val cap = if (limit != null) "LIMIT $limit" else ""
        val sql = "SELECT e.* FROM evidence e JOIN games g ON g.game_id = e.game_id" +
            " $where ORDER BY g.started_at DESC, e.segment, e.ply $cap"
        return lock.withLock {
            if (gameId != null) rows(sql, gameId, map = ::evidenceRow) else rows(sql, map = ::evidenceRow)
        }
    }

    /** The user's correction of a mistake's cause (and motif). */
    fun correctEvidence(gameId: String, segment: Int, ply: Int, cause: String?, motif: String? = null): EvidenceRow? =
        lock.withLock {
            val where = "WHERE game_id = ? AND segment = $segment AND ply = $ply"
            val row = rows("SELECT * FROM evidence $where", gameId, map = ::evidenceRow).firstOrNull()
                ?: return@withLock null
            val newMotif = motif ?: row.motif
            exec(
                "UPDATE evidence SET cause = ?, motif = ?, corrected = 1 WHERE game_id = ? AND segment = ? AND ply = ?",
                cause, newMotif, gameId, segment, ply,
            )
            rows("SELECT * FROM evidence $where", gameId, map = ::evidenceRow).first()
        }

    fun selfReport(moveId: Long, text: String) {
        lock.withLock { exec("UPDATE evidence SET self_report = ? WHERE move_id = ?", text, moveId) }
    }

    // ---- the taxonomy's cells, computed ------------------------------------------------------

    /**
     * The error taxonomy as counts: mistakes by (cause, motif, phase) with the score they cost;
     * the occasions with cause null -- every judged move of the player's in its phase (motif
     * null), and every such move with a tactic on the board, per tactic. Computed from the
     * record, so a correction or a takeback needs no bookkeeping.
     */
    fun cells(): List<CellRow> {
        val out = ArrayList<CellRow>()
        lock.withLock {
            out += rows(
                "SELECT cause, motif, phase, COUNT(*) AS n, SUM(expected_loss) AS loss" +
                    " FROM evidence GROUP BY cause, motif, phase",
            ) { c ->
                CellRow(
                    cause = c.stringOrNull("cause"),
                    motif = c.stringOrNull("motif"),
                    phase = c.stringOrNull("phase"),
                    errors = c.getInt(c.getColumnIndexOrThrow("n")),
                    chances = 0,
                    expectedLoss = c.doubleOrNull("loss") ?: 0.0,
                )
            }
            out += rows(
                "SELECT phase, COUNT(*) AS n FROM moves" +
                    " WHERE by_user = 1 AND withdrawn = 0 AND severity IS NOT NULL" +
                    " AND phase IS NOT NULL GROUP BY phase",
            ) { c ->
                CellRow(null, null, c.stringOrNull("phase"), 0, c.getInt(c.getColumnIndexOrThrow("n")), 0.0)
            }
            out += rows(
                "SELECT m.phase AS phase, t.value AS motif, COUNT(*) AS n" +
                    " FROM moves m, json_each(m.available_tags) t" +
                    " WHERE m.by_user = 1 AND m.withdrawn = 0 AND m.severity IS NOT NULL" +
                    " GROUP BY m.phase, t.value",
            ) { c ->
                CellRow(null, c.stringOrNull("motif"), c.stringOrNull("phase"), 0, c.getInt(c.getColumnIndexOrThrow("n")), 0.0)
            }
        }
        return out.sortedBy { -it.expectedLoss }
    }

    // ---- sessions -------------------------------------------------------------------------------

    fun startSession(sessionId: String, startedAt: Double) {
        lock.withLock { exec("INSERT OR REPLACE INTO sessions (session_id, started_at) VALUES (?, ?)", sessionId, startedAt) }
    }

    // ---- puzzles ---------------------------------------------------------------------------------

    fun setting(key: String): String? = lock.withLock {
        rows("SELECT value FROM settings WHERE key = ?", key) { it.stringOrNull("value") }.firstOrNull()
    }

    fun setSetting(key: String, value: String) {
        lock.withLock {
            exec(
                "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                key, value,
            )
        }
    }

    fun addRecommendation(gameId: String, ts: Double, motif: String, reason: String, level: Int, puzzleIds: List<String>) {
        lock.withLock {
            exec(
                "INSERT INTO recommendations (game_id, ts, motif, reason, level, puzzle_ids) VALUES (?, ?, ?, ?, ?, ?)",
                gameId, ts, motif, reason, level, puzzleIds.joinToString(" "),
            )
        }
    }

    /** The puzzles recommended lately, latest first. */
    fun recommendedIds(limit: Int = 20): List<String> = lock.withLock {
        rows("SELECT puzzle_ids FROM recommendations ORDER BY ts DESC LIMIT $limit") { it.getString(0) }
    }.flatMap { ids -> ids.split(Regex("\\s+")).filter { it.isNotEmpty() } }

    fun addAttempt(puzzleId: String, ts: Double, motif: String?, solved: Boolean, moves: List<String>) {
        lock.withLock {
            exec(
                "INSERT INTO puzzle_attempts (puzzle_id, ts, motif, solved, moves) VALUES (?, ?, ?, ?, ?)",
                puzzleId, ts, motif, solved, moves.joinToString(" "),
            )
        }
    }

    /** (puzzle_id, ts, solved) of the latest attempts, latest first. */
    fun attempts(limit: Int = 100): List<Triple<String, Double, Boolean>> = lock.withLock {
        rows("SELECT puzzle_id, ts, solved FROM puzzle_attempts ORDER BY ts DESC LIMIT $limit") {
            Triple(it.getString(0), it.getDouble(1), it.getInt(2) != 0)
        }
    }

    fun endSession(sessionId: String, endedAt: Double, summary: String?) {
        lock.withLock { exec("UPDATE sessions SET ended_at = ?, summary = ? WHERE session_id = ?", endedAt, summary, sessionId) }
    }

    fun session(sessionId: String): SessionRow? = lock.withLock {
        rows("SELECT * FROM sessions WHERE session_id = ?", sessionId, map = ::sessionRow).firstOrNull()
    }

    /** The most recent session with a summary, other than [before]. */
    fun lastSession(before: String? = null): SessionRow? = lock.withLock {
        // `IS NOT ?` with nothing bound is `IS NOT NULL`: every row, as Python's None binds.
        val sql = "SELECT * FROM sessions WHERE summary IS NOT NULL AND session_id IS NOT " +
            (if (before == null) "NULL" else "?") + " ORDER BY started_at DESC LIMIT 1"
        (if (before == null) rows(sql, map = ::sessionRow) else rows(sql, before, map = ::sessionRow)).firstOrNull()
    }

    // ---- read-only SQL from the coach ----------------------------------------------------------

    /**
     * Run one read-only statement and return (column names, rows), at most [maxRows] of them.
     * Anything but a single SELECT (or WITH ... SELECT) is refused before it runs, and the
     * database refuses any write while it runs; a statement still running after [seconds] is
     * interrupted. Used by the coach's own queries (Ask.kt).
     */
    fun query(sql: String, maxRows: Int = QUERY_ROWS, seconds: Double = QUERY_SECONDS): Pair<List<String>, List<List<Any?>>> {
        val text = sql.trim().trimEnd(';').trim()
        if (text.isEmpty()) throw QueryRefused("empty query")
        if (';' in text) throw QueryRefused("one statement at a time")
        val head = text.split(Regex("\\s+"), limit = 2)[0].uppercase()
        if (head != "SELECT" && head != "WITH") throw QueryRefused("only SELECT queries are allowed")
        return lock.withLock {
            val cancel = CancellationSignal()
            val timer = Timer("profile-query", true)
            db.execSQL("PRAGMA query_only = 1")
            try {
                timer.schedule((seconds * 1000).toLong()) { cancel.cancel() }
                db.rawQuery(text, null, cancel).use { cursor ->
                    val names = cursor.columnNames.toList()
                    val out = ArrayList<List<Any?>>()
                    while (out.size < maxRows + 1 && cursor.moveToNext()) {
                        out.add((0 until cursor.columnCount).map { i -> cell(cursor, i) })
                    }
                    names to out.take(maxRows)
                }
            } catch (exc: SQLiteException) {
                throw QueryRefused(exc.message ?: exc.toString())
            } catch (exc: OperationCanceledException) {
                throw QueryRefused("interrupted")
            } finally {
                timer.cancel()
                db.execSQL("PRAGMA query_only = 0")
            }
        }
    }

    private fun cell(cursor: Cursor, i: Int): Any? = when (cursor.getType(i)) {
        Cursor.FIELD_TYPE_NULL -> null
        Cursor.FIELD_TYPE_INTEGER -> cursor.getLong(i)
        Cursor.FIELD_TYPE_FLOAT -> cursor.getDouble(i)
        Cursor.FIELD_TYPE_BLOB -> cursor.getBlob(i)
        else -> cursor.getString(i)
    }

    companion object {
        const val QUERY_ROWS = 200
        const val QUERY_SECONDS = 2.0

        const val SCHEMA = """
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
CREATE INDEX IF NOT EXISTS moves_game ON moves (game_id, withdrawn);
CREATE INDEX IF NOT EXISTS evidence_game ON evidence (game_id);
CREATE INDEX IF NOT EXISTS attempts_puzzle ON puzzle_attempts (puzzle_id, ts)
"""

        /**
         * A list of strings as Python's `json.dumps` writes it -- `["capture", "check"]`, a space
         * after each comma, non-ASCII escaped -- since the coach's own SQL can read it back.
         */
        fun jsonList(items: List<String>): String = items.joinToString(", ", "[", "]") { pyJsonString(it) }

        private fun pyJsonString(s: String): String {
            val out = StringBuilder("\"")
            for (ch in s) {
                when {
                    ch == '"' -> out.append("\\\"")
                    ch == '\\' -> out.append("\\\\")
                    ch == '\n' -> out.append("\\n")
                    ch == '\r' -> out.append("\\r")
                    ch == '\t' -> out.append("\\t")
                    ch == '\b' -> out.append("\\b")
                    ch == '\u000C' -> out.append("\\f")
                    ch.code < 0x20 || ch.code > 0x7f -> out.append("\\u%04x".format(ch.code))
                    else -> out.append(ch)
                }
            }
            return out.append('"').toString()
        }

        private fun tags(value: String?): List<String> {
            if (value.isNullOrEmpty()) return emptyList()
            val array = JSONArray(value)
            return (0 until array.length()).map { array.getString(it) }
        }

        private fun Cursor.stringOrNull(name: String): String? {
            val i = getColumnIndexOrThrow(name)
            return if (isNull(i)) null else getString(i)
        }

        private fun Cursor.intOrNull(name: String): Int? {
            val i = getColumnIndexOrThrow(name)
            return if (isNull(i)) null else getInt(i)
        }

        private fun Cursor.doubleOrNull(name: String): Double? {
            val i = getColumnIndexOrThrow(name)
            return if (isNull(i)) null else getDouble(i)
        }

        private fun gameRow(c: Cursor) = GameRow(
            gameId = c.getString(c.getColumnIndexOrThrow("game_id")),
            sessionId = c.stringOrNull("session_id"),
            startedAt = c.getDouble(c.getColumnIndexOrThrow("started_at")),
            endedAt = c.doubleOrNull("ended_at"),
            userSide = c.getString(c.getColumnIndexOrThrow("user_side")),
            startFen = c.getString(c.getColumnIndexOrThrow("start_fen")),
            timeControl = c.stringOrNull("time_control"),
            opponent = c.stringOrNull("opponent"),
            result = c.stringOrNull("result"),
            reason = c.stringOrNull("reason"),
            opening = c.stringOrNull("opening"),
            accuracy = c.doubleOrNull("accuracy"),
            finalScore = c.doubleOrNull("final_score"),
            pgn = c.stringOrNull("pgn"),
            finalFen = c.stringOrNull("final_fen"),
        )

        private fun moveRow(c: Cursor) = MoveRow(
            moveId = c.getLong(c.getColumnIndexOrThrow("move_id")),
            gameId = c.getString(c.getColumnIndexOrThrow("game_id")),
            segment = c.getInt(c.getColumnIndexOrThrow("segment")),
            ply = c.getInt(c.getColumnIndexOrThrow("ply")),
            moveNumber = c.getInt(c.getColumnIndexOrThrow("move_number")),
            san = c.getString(c.getColumnIndexOrThrow("san")),
            uci = c.getString(c.getColumnIndexOrThrow("uci")),
            byUser = c.getInt(c.getColumnIndexOrThrow("by_user")) != 0,
            fenBefore = c.getString(c.getColumnIndexOrThrow("fen_before")),
            fenAfter = c.getString(c.getColumnIndexOrThrow("fen_after")),
            phase = c.stringOrNull("phase"),
            thinkMs = c.intOrNull("think_ms"),
            clockMs = c.intOrNull("clock_ms"),
            capture = c.getInt(c.getColumnIndexOrThrow("capture")) != 0,
            givesCheck = c.getInt(c.getColumnIndexOrThrow("gives_check")) != 0,
            material = c.intOrNull("material"),
            ts = c.getDouble(c.getColumnIndexOrThrow("ts")),
            label = c.stringOrNull("label"),
            severity = c.stringOrNull("severity"),
            expectedLoss = c.doubleOrNull("expected_loss"),
            scoreAfter = c.doubleOrNull("score_after"),
            bestUci = c.stringOrNull("best_uci"),
            bestSan = c.stringOrNull("best_san"),
            accuracy = c.doubleOrNull("accuracy"),
            tags = tags(c.stringOrNull("tags")),
            availableTags = tags(c.stringOrNull("available_tags")),
            allowedTags = tags(c.stringOrNull("allowed_tags")),
            depth = c.intOrNull("depth"),
            withdrawn = c.getInt(c.getColumnIndexOrThrow("withdrawn")) != 0,
        )

        private fun evidenceRow(c: Cursor) = EvidenceRow(
            moveId = c.getLong(c.getColumnIndexOrThrow("move_id")),
            gameId = c.getString(c.getColumnIndexOrThrow("game_id")),
            segment = c.getInt(c.getColumnIndexOrThrow("segment")),
            ply = c.getInt(c.getColumnIndexOrThrow("ply")),
            phase = c.getString(c.getColumnIndexOrThrow("phase")),
            expectedLoss = c.getDouble(c.getColumnIndexOrThrow("expected_loss")),
            cause = c.stringOrNull("cause"),
            motif = c.stringOrNull("motif"),
            timePressure = c.getInt(c.getColumnIndexOrThrow("time_pressure")) != 0,
            selfReport = c.stringOrNull("self_report"),
            corrected = c.getInt(c.getColumnIndexOrThrow("corrected")) != 0,
        )

        private fun sessionRow(c: Cursor) = SessionRow(
            sessionId = c.getString(c.getColumnIndexOrThrow("session_id")),
            startedAt = c.getDouble(c.getColumnIndexOrThrow("started_at")),
            endedAt = c.doubleOrNull("ended_at"),
            games = c.getInt(c.getColumnIndexOrThrow("games")),
            summary = c.stringOrNull("summary"),
        )
    }
}

/** A query that is not a single read-only statement, or that ran too long. */
class QueryRefused(message: String) : IllegalArgumentException(message)

data class GameRow(
    val gameId: String,
    val sessionId: String?,
    val startedAt: Double,
    val endedAt: Double?,
    val userSide: String,
    val startFen: String,
    val timeControl: String?,
    val opponent: String?,
    val result: String?,
    val reason: String?,
    val opening: String?,
    val accuracy: Double?,
    val finalScore: Double?,
    val pgn: String?,
    val finalFen: String?,
)

data class MoveRow(
    val moveId: Long,
    val gameId: String,
    val segment: Int,
    val ply: Int,
    val moveNumber: Int,
    val san: String,
    val uci: String,
    val byUser: Boolean,
    val fenBefore: String,
    val fenAfter: String,
    val phase: String?,
    val thinkMs: Int?,
    val clockMs: Int?,
    val capture: Boolean,
    val givesCheck: Boolean,
    val material: Int?,
    val ts: Double,
    val label: String?,
    val severity: String?,
    val expectedLoss: Double?,
    val scoreAfter: Double?,
    val bestUci: String?,
    val bestSan: String?,
    val accuracy: Double?,
    val tags: List<String>,
    val availableTags: List<String>,
    val allowedTags: List<String>,
    val depth: Int?,
    val withdrawn: Boolean,
)

data class EvidenceRow(
    val moveId: Long,
    val gameId: String,
    val segment: Int,
    val ply: Int,
    val phase: String,
    val expectedLoss: Double,
    val cause: String?,
    val motif: String?,
    val timePressure: Boolean,
    val selfReport: String?,
    val corrected: Boolean,
)

/**
 * One cell of the error taxonomy, computed from the record: mistakes ([errors]) by cause,
 * motif and phase with the expected score they cost, and the occasions ([chances]) -- moves
 * in a phase, moves with a tactic on the board -- with the cause null.
 */
data class CellRow(
    val cause: String?,
    val motif: String?,
    val phase: String?,
    val errors: Int,
    val chances: Int,
    val expectedLoss: Double,
)

data class SessionRow(
    val sessionId: String,
    val startedAt: Double,
    val endedAt: Double?,
    val games: Int,
    val summary: String?,
)
