package com.sidecoach.profile

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteException
import android.database.sqlite.SQLiteReadOnlyDatabaseException
import android.database.sqlite.SQLiteStatement
import android.os.CancellationSignal
import android.os.OperationCanceledException
import com.sidecoach.voice.Py
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.Locale
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
 * statement checks the desktop makes first, and what the authorizer denies is refused in its
 * words ([query]).
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

    /** The most recent games, newest first; every one when [limit] is null. */
    fun games(limit: Int? = 50, finished: Boolean = true): List<GameRow> {
        val clause = if (finished) "WHERE result IS NOT NULL" else ""
        return lock.withLock {
            rows("SELECT * FROM games $clause ORDER BY started_at DESC LIMIT ${limit ?: -1}", map = ::gameRow)
        }
    }

    /** Every row of a game: one followed provisionally that turned out not to be one. */
    fun deleteGame(gameId: String) {
        lock.withLock {
            db.beginTransaction()
            try {
                for (table in WATCHED_TABLES) exec("DELETE FROM $table WHERE game_id = ?", gameId)
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
                "SELECT cause, motif, phase, COUNT(*) AS n, SUM(expected_loss) AS loss," +
                    " json_group_array(DISTINCT game_id) AS games" +
                    " FROM evidence GROUP BY cause, motif, phase",
            ) { c ->
                CellRow(
                    cause = c.stringOrNull("cause"),
                    motif = c.stringOrNull("motif"),
                    phase = c.stringOrNull("phase"),
                    errors = c.getInt(c.getColumnIndexOrThrow("n")),
                    chances = 0,
                    expectedLoss = c.doubleOrNull("loss") ?: 0.0,
                    games = ids(c.stringOrNull("games")),
                )
            }
            out += rows(
                "SELECT phase, COUNT(*) AS n FROM moves" +
                    " WHERE by_user = 1 AND withdrawn = 0 AND severity IS NOT NULL" +
                    " AND phase IS NOT NULL GROUP BY phase",
            ) { c ->
                CellRow(null, null, c.stringOrNull("phase"), 0, c.getInt(c.getColumnIndexOrThrow("n")), 0.0)
            }
            // A tactic's occasions with those the player missed (a mistake that did not play it):
            // a mistake's own motif is the tactic it allowed as often as the one it let go, and
            // "about 100% (5 of 4)" counted the forks walked into over the forks the player had
            // (the review of 2026-10-01).
            val missed = "(e.move_id IS NOT NULL AND NOT EXISTS" +
                " (SELECT 1 FROM json_each(m.tags) p WHERE p.value = t.value))"
            out += rows(
                "SELECT m.phase AS phase, t.value AS motif, COUNT(*) AS n," +
                    " SUM($missed) AS missed," +
                    " json_group_array(DISTINCT CASE WHEN $missed THEN m.game_id END) AS games" +
                    " FROM moves m JOIN json_each(m.available_tags) t" +
                    " LEFT JOIN evidence e ON e.move_id = m.move_id" +
                    " WHERE m.by_user = 1 AND m.withdrawn = 0 AND m.severity IS NOT NULL" +
                    " GROUP BY m.phase, t.value",
            ) { c ->
                CellRow(
                    null,
                    c.stringOrNull("motif"),
                    c.stringOrNull("phase"),
                    c.intOrNull("missed") ?: 0,
                    c.getInt(c.getColumnIndexOrThrow("n")),
                    0.0,
                    ids(c.stringOrNull("games")),
                )
            }
        }
        return out.sortedBy { -it.expectedLoss }
    }

    /** A JSON array of game ids, its nulls left out (store.py `_ids`). */
    private fun ids(value: String?): Set<String> {
        val array = org.json.JSONArray(value ?: "[]")
        return (0 until array.length()).filter { !array.isNull(it) }.map { array.getString(it) }.toSet()
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

    // ---- the review's library and its reviews (store.py) ------------------------------------------

    /**
     * The player's side in each imported game that has none, by their usernames (lower case): a
     * game brought in before the username was set. `fill_sides`.
     */
    fun fillSides(names: List<String>) {
        if (names.isEmpty()) return
        val marks = names.joinToString(",") { "?" }
        lock.withLock {
            exec(
                "UPDATE library SET user_side = CASE WHEN lower(white) IN ($marks) THEN 'white'" +
                    " WHEN lower(black) IN ($marks) THEN 'black' END" +
                    " WHERE user_side IS NULL AND (lower(white) IN ($marks) OR lower(black) IN ($marks))",
                *(names + names + names + names).toTypedArray(),
            )
        }
    }

    fun setTimeClass(key: String, timeClass: String?) {
        lock.withLock { exec("UPDATE library SET time_class = ? WHERE key = ?", timeClass, key) }
    }

    /** A game imported for review; the same game imported again replaces its row. */
    fun addLibraryGame(row: LibraryRow) {
        lock.withLock {
            exec(
                "INSERT OR REPLACE INTO library (key, source, added_at, played_at, white, black," +
                    " white_elo, black_elo, result, time_class, user_side, pgn)" +
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                row.key, row.source, row.addedAt, row.playedAt, row.white, row.black,
                row.whiteElo, row.blackElo, row.result, row.timeClass, row.userSide, row.pgn,
            )
        }
    }

    /**
     * Every imported game, the latest played first. Every one: the Games window listed 500 while
     * Stats counted all (the review of 2026-10-01), and the games past it could not be seen,
     * opened or removed. A game that is a copy of another (pasted and imported: [addTwin],
     * `Library.pairCopies`) is that other's: once. `library`.
     */
    fun library(): List<LibraryRow> = lock.withLock {
        rows(
            "SELECT * FROM library WHERE key NOT IN (SELECT watched FROM twins)" +
                " ORDER BY COALESCE(played_at, added_at) DESC",
        ) { libraryRow(it) }
    }

    fun libraryGame(key: String): LibraryRow? = lock.withLock {
        rows("SELECT * FROM library WHERE key = ?", key) { libraryRow(it) }.firstOrNull()
    }

    // ---- a game removed from the player's record (store.py) ---------------------------------

    /**
     * A game out of the player's record: every row of it -- the library's, or the watched record
     * with its moves and what the coach drew from it -- its review and its Stats, and the game
     * it is paired with (an import and the watched game it stands for are one game). What it
     * took is kept [UNDO_KEEP_S] for [restoreGame]; its key for good, so bringing games in
     * leaves it out. Returns the keys removed. `remove_game`.
     */
    fun removeGame(key: String): List<String> = lock.withLock {
        val now = System.currentTimeMillis() / 1000.0
        db.beginTransaction()
        try {
            val paired = rows("SELECT watched, imported FROM twins WHERE watched = ? OR imported = ?", key, key) {
                listOf(it.getString(0), it.getString(1))
            }
            val gone = rows("SELECT key FROM removed") { it.getString(0) }.toSet()
            // A game removed already (a second click) keeps what it took, for its Undo.
            val group = (listOf(key) + (paired.flatten().toSortedSet() - key)).filter { it !in gone }
            for (k in group) {
                val taken = JSONObject()
                for (table in LIBRARY_TABLES) taken.put(table, take(table, "key", k))
                taken.put("twins", takeTwins(k))
                var trace: JSONObject? = null
                if (k.startsWith(WATCHED)) {
                    val gameId = k.removePrefix(WATCHED)
                    trace = trace(gameId)
                    for (table in WATCHED_TABLES) taken.put(table, take(table, "game_id", gameId))
                }
                exec(
                    "INSERT OR REPLACE INTO removed (key, removed_at, grp, rows, trace) VALUES (?, ?, ?, ?, ?)",
                    k, now, key, taken.toString(), trace?.toString(),
                )
            }
            // What older removals took can no longer be put back: only their keys stay.
            exec("UPDATE removed SET rows = NULL WHERE removed_at < ?", now - UNDO_KEEP_S)
            db.setTransactionSuccessful()
            group
        } finally {
            db.endTransaction()
        }
    }

    /**
     * A game the player removed moments ago put back as it was, with the game it was paired
     * with; nothing once its rows are no longer kept. Returns the keys put back. `restore_game`.
     */
    fun restoreGame(key: String): List<String> = lock.withLock {
        val now = System.currentTimeMillis() / 1000.0
        db.beginTransaction()
        try {
            val found = rows("SELECT key, rows, removed_at FROM removed WHERE grp = ? AND rows IS NOT NULL", key) {
                Triple(it.getString(0), it.getString(1), it.getDouble(2))
            }.filter { it.third >= now - UNDO_KEEP_S }
            for ((k, taken, _) in found) {
                val tables = JSONObject(taken)
                for (table in tables.keys()) {
                    if (table !in RESTORABLE) continue
                    val items = tables.getJSONArray(table)
                    for (i in 0 until items.length()) {
                        val item = items.getJSONObject(i)
                        val cols = item.keys().asSequence().toList()
                        val values = cols.map { c -> item.get(c).takeIf { it != JSONObject.NULL } }
                        exec(
                            "INSERT OR REPLACE INTO $table (${cols.joinToString(", ")}) VALUES (${cols.joinToString(", ") { "?" }})",
                            *values.toTypedArray(),
                        )
                    }
                }
                exec("DELETE FROM removed WHERE key = ?", k)
            }
            db.setTransactionSuccessful()
            found.map { it.first }
        } finally {
            db.endTransaction()
        }
    }

    /** Every game the player removed, by its library key. `removed_keys`. */
    fun removedKeys(): Set<String> = lock.withLock {
        rows("SELECT key FROM removed") { it.getString(0) }.toSet()
    }

    /** The watched games the player removed, as much of them as tells their import. `removed_traces`. */
    fun removedTraces(): List<Trace> = lock.withLock {
        rows("SELECT trace FROM removed WHERE trace IS NOT NULL ORDER BY key") { c ->
            val t = JSONObject(c.getString(0))
            val sans = t.getJSONArray("sans")
            Trace(
                startedAt = t.getDouble("started_at"),
                userSide = t.getString("user_side"),
                result = if (t.isNull("result")) null else t.getString("result"),
                sans = (0 until sans.length()).map { sans.getString(it) },
            )
        }
    }

    fun isRemoved(key: String): Boolean = lock.withLock {
        rows("SELECT 1 FROM removed WHERE key = ?", key) { true }.isNotEmpty()
    }

    /**
     * Whether a game (by its library key) is in the player's record: its import in the library,
     * or Sidecoach's record of the watched game -- gone once removed, or once the profile is
     * reset. `in_record`.
     */
    fun inRecord(key: String): Boolean = lock.withLock {
        if (key.startsWith(WATCHED)) {
            rows("SELECT 1 FROM games WHERE game_id = ?", key.removePrefix(WATCHED)) { true }.isNotEmpty()
        } else {
            rows("SELECT 1 FROM library WHERE key = ?", key) { true }.isNotEmpty()
        }
    }

    /** Games the player brought in by hand (a pasted PGN): no longer left out. `unremove`. */
    fun unremove(keys: List<String>) {
        lock.withLock { for (k in keys) exec("DELETE FROM removed WHERE key = ?", k) }
    }

    /** A table's rows of one game, as they were (each column's type kept), and gone from it. */
    private fun take(table: String, column: String, value: String): JSONArray {
        val out = cursorRows("SELECT * FROM $table WHERE $column = ?", value)
        exec("DELETE FROM $table WHERE $column = ?", value)
        return out
    }

    private fun takeTwins(key: String): JSONArray {
        val out = cursorRows("SELECT * FROM twins WHERE watched = ? OR imported = ?", key, key)
        exec("DELETE FROM twins WHERE watched = ? OR imported = ?", key, key)
        return out
    }

    private fun cursorRows(sql: String, vararg args: String): JSONArray {
        val out = JSONArray()
        db.rawQuery(sql, arrayOf(*args)).use { c ->
            while (c.moveToNext()) {
                val row = JSONObject()
                for (i in 0 until c.columnCount) {
                    row.put(
                        c.getColumnName(i),
                        when (c.getType(i)) {
                            Cursor.FIELD_TYPE_NULL -> JSONObject.NULL
                            Cursor.FIELD_TYPE_INTEGER -> c.getLong(i)
                            Cursor.FIELD_TYPE_FLOAT -> c.getDouble(i)
                            else -> c.getString(i)
                        },
                    )
                }
                out.put(row)
            }
        }
        return out
    }

    private fun trace(gameId: String): JSONObject? {
        val game = rows("SELECT started_at, user_side, result FROM games WHERE game_id = ?", gameId) { c ->
            Triple(c.getDouble(0), c.getString(1), if (c.isNull(2)) null else c.getString(2))
        }.firstOrNull() ?: return null
        val sans = rows("SELECT san FROM moves WHERE game_id = ? AND withdrawn = 0 ORDER BY segment, ply", gameId) {
            it.getString(0).trimEnd('+', '#')
        }
        return JSONObject()
            .put("started_at", game.first)
            .put("user_side", game.second)
            .put("result", game.third ?: JSONObject.NULL)
            .put("sans", JSONArray(sans))
    }

    /** A watched game's import (library keys): the import stands for it from now on. */
    fun addTwin(watched: String, imported: String) {
        lock.withLock {
            exec(
                "INSERT OR REPLACE INTO twins (watched, imported, found_at) VALUES (?, ?, ?)",
                watched, imported, System.currentTimeMillis() / 1000.0,
            )
        }
    }

    /** Every game that stood aside for [was] stands aside for [now] (a better copy came). `retarget_twins`. */
    fun retargetTwins(was: String, now: String) {
        lock.withLock { exec("UPDATE twins SET imported = ? WHERE imported = ?", now, was) }
    }

    /** The watched games that were imported too: watched key -> imported key. */
    fun twins(): Map<String, String> = lock.withLock {
        rows("SELECT watched, imported FROM twins") { it.getString(0) to it.getString(1) }.toMap(LinkedHashMap())
    }

    // ---- Stats: a reviewed game's facts (StatFacts.kt; store.py) ----------------------------------

    /**
     * A game's facts from its review at [strength], in place of any before from a review no
     * stronger: Stats keeps the strongest review's facts (a Fast look after a Deep review lowered
     * Stats, the review of 2026-10-01). `save_stats`.
     */
    fun saveStats(strength: String, game: GameFact, moves: List<MoveFact>) {
        lock.withLock {
            val kept = rows("SELECT $RANK AS rank FROM stat_games WHERE key = ?", game.key) { it.getInt(0) }.firstOrNull()
            if (kept != null && kept > (STRENGTH_RANK[strength] ?: 0)) return
            db.beginTransaction()
            try {
                exec("DELETE FROM stat_moves WHERE key = ?", game.key)
                exec(
                    "INSERT OR REPLACE INTO stat_games (key, strength, source, time_class, played_at," +
                        " user_side, bot, result, opening, termination) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    game.key, strength, game.source, game.timeClass, game.playedAt, game.userSide,
                    game.bot, game.result, game.opening, game.termination,
                )
                for (m in moves) {
                    exec(
                        "INSERT INTO stat_moves (key, ply, side, fen, uci, san, phase, loss, label," +
                            " accuracy, score, chance, found, allowed, structure, themes, endgame, spent_ms," +
                            " clock_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        game.key, m.ply, m.side, m.fen, m.uci, m.san, m.phase, m.loss, m.label,
                        m.accuracy, m.score, m.chance, m.found, m.allowed, m.structure, m.themes, m.endgame,
                        m.spentMs, m.clockMs,
                    )
                }
                // Raised with every save of facts, in the same commit ([statsRevision]).
                exec(
                    "INSERT INTO settings (key, value) VALUES (?, '1') ON CONFLICT(key)" +
                        " DO UPDATE SET value = CAST(value AS INTEGER) + 1",
                    STATS_REVISION_KEY,
                )
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
        }
    }

    /**
     * A count raised with every save of a game's Stats facts: what was read from them is stale
     * once it moves. The games' keys, strengths and sides alone missed facts made again at the
     * same strength -- rederive, the catch-up of a saved review -- and the coach kept a stale
     * "biggest cost" (the audit of 2026-10-02). `stats_revision`.
     */
    fun statsRevision(): String? = setting(STATS_REVISION_KEY)

    /**
     * The reviewed games' facts, each game once: a watched game its import stands for is left
     * out once the import's own facts are kept. Until then the watched game counts (the review of
     * 2026-10-01: a game whose record differed from its import left Stats the moment the import
     * came in, until the import was opened and reviewed). `stat_games`.
     */
    fun statGames(): List<StatGameRow> = lock.withLock {
        // The player's side and the time class are the library's as it is now, when it has the
        // game: a game reviewed before the username was set kept no side and never reached Stats
        // (the review of 2026-10-01).
        rows(
            "SELECT s.key, s.strength, s.source, s.played_at, s.bot, s.result, s.opening," +
                " s.termination, COALESCE(l.user_side, s.user_side) AS user_side," +
                " COALESCE(l.time_class, s.time_class) AS time_class" +
                " FROM stat_games s LEFT JOIN library l ON l.key = s.key" +
                " WHERE s.key NOT IN ($STOOD_FOR) ORDER BY s.played_at",
        ) { statGameRow(it) }
    }

    /** Each imported game's players' ratings, White's and Black's, by its key. `library_elos`. */
    fun libraryElos(): Map<String, Pair<Int?, Int?>> = lock.withLock {
        rows("SELECT key, white_elo, black_elo FROM library") { c ->
            c.getString(0) to ((if (c.isNull(1)) null else c.getInt(1)) to (if (c.isNull(2)) null else c.getInt(2)))
        }.toMap()
    }

    /** Each saved review's game shape (its summary's), by the game's key and strength. `review_shapes`. */
    fun reviewShapes(): Map<Pair<String, String>, String?> = lock.withLock {
        rows(
            "SELECT key, strength, json_extract(summary, '$.shape') AS shape FROM reviews WHERE json_valid(summary)",
        ) { c -> (c.getString(0) to c.getString(1)) to (if (c.isNull(2)) null else c.getString(2)) }.toMap()
    }

    /** The facts of the games' moves, in order. `stat_moves`. */
    fun statMoves(keys: List<String>): List<StatMoveRow> {
        if (keys.isEmpty()) return emptyList()
        val marks = keys.joinToString(",") { "?" }
        return lock.withLock {
            rows("SELECT * FROM stat_moves WHERE key IN ($marks) ORDER BY key, ply", *keys.toTypedArray()) { statMoveRow(it) }
        }
    }

    /**
     * Each game's strongest review ([bestReview]) whose facts are not kept yet (a watched game
     * its import stands for needs none once the import's are kept: [statGames]).
     * `reviews_without_stats`.
     */
    fun reviewsWithoutStats(): List<ReviewRow> = lock.withLock {
        rows(
            "SELECT r.* FROM reviews r WHERE r.key NOT IN (SELECT key FROM stat_games)" +
                " AND r.key NOT IN ($STOOD_FOR)" +
                " AND r.rowid = (SELECT rowid FROM reviews WHERE key = r.key" +
                " ORDER BY $RANK DESC, reviewed_at DESC LIMIT 1)",
        ) { reviewRow(it) }
    }

    /** The library's imported games no review covers yet, the latest played first. */
    fun unreviewed(): List<LibraryRow> = lock.withLock {
        rows(
            "SELECT * FROM library WHERE key NOT IN (SELECT key FROM reviews)" +
                " AND key NOT IN (SELECT watched FROM twins) ORDER BY COALESCE(played_at, added_at) DESC",
        ) { libraryRow(it) }
    }

    /**
     * A game reviewed whole; whether it was saved. A review of an older format is replaced, and
     * its words go with it. One of this format saved already stays: another reviewer finished
     * first (the review of 2026-10-01: the desktop's background review and a review window on
     * one game at once -- the window saved first, the player paid for the AI coach's words, and
     * the background's save wiped them). Nor is a game the player removed meanwhile saved. One
     * statement, so no other connection's save or removal comes in between. `save_review`.
     */
    fun saveReview(key: String, strength: String, version: Int, reviewedAt: Double, report: String, summary: String?): Boolean =
        lock.withLock {
            exec(
                "INSERT INTO reviews (key, strength, version, reviewed_at, report, summary, words)" +
                    " SELECT ?, ?, ?, ?, ?, ?, NULL WHERE NOT EXISTS (SELECT 1 FROM removed WHERE key = ?)" +
                    " ON CONFLICT (key, strength) DO UPDATE SET version = excluded.version," +
                    " reviewed_at = excluded.reviewed_at, report = excluded.report," +
                    " summary = excluded.summary, words = NULL WHERE reviews.version != excluded.version",
                key, strength, version, reviewedAt, report, summary, key,
            ) > 0
        }

    fun review(key: String, strength: String): ReviewRow? = lock.withLock {
        rows("SELECT * FROM reviews WHERE key = ? AND strength = ?", key, strength) { reviewRow(it) }.firstOrNull()
    }

    /**
     * The game's strongest review (deep, standard, fast), the latest of that strength: the one
     * the review opens at, Your games shows and Stats counts. The latest was shown, and a Fast
     * look after a Deep review lowered them all (the review of 2026-10-01). `best_review`.
     */
    fun bestReview(key: String): ReviewRow? = lock.withLock {
        rows("SELECT * FROM reviews WHERE key = ? ORDER BY $RANK DESC, reviewed_at DESC LIMIT 1", key) { reviewRow(it) }.firstOrNull()
    }

    /** Every review of the game, the strongest first. `reviews_of`. */
    fun reviewsOf(key: String): List<ReviewRow> = lock.withLock {
        rows("SELECT * FROM reviews WHERE key = ? ORDER BY $RANK DESC, reviewed_at DESC", key) { reviewRow(it) }
    }

    /** Every saved review, by its game's key and its strength. `saved_reviews`. */
    fun savedReviews(): List<Pair<String, String>> = lock.withLock {
        rows("SELECT key, strength FROM reviews") { it.getString(0) to it.getString(1) }
    }

    /** `set_review_summary`. */
    fun setReviewSummary(key: String, strength: String, summary: String) {
        lock.withLock { exec("UPDATE reviews SET summary = ? WHERE key = ? AND strength = ?", summary, key, strength) }
    }

    /**
     * A saved review's report and summary made again from what it kept ([ReviewRoom.rederive]):
     * its words -- the AI coach's, paid for -- stay as they are. Only the review read at
     * [reviewedAt]: one replaced meanwhile is left alone. Whether it was written.
     * `set_review_report`.
     */
    fun setReviewReport(key: String, strength: String, reviewedAt: Double, report: String, summary: String): Boolean =
        lock.withLock {
            exec(
                "UPDATE reviews SET report = ?, summary = ? WHERE key = ? AND strength = ? AND reviewed_at = ?",
                report, summary, key, strength, reviewedAt,
            ) > 0
        }

    /** The strength of the review each game's Stats facts come from, by its key. `stat_strengths`. */
    fun statStrengths(): Map<String, String> = lock.withLock {
        rows("SELECT key, strength FROM stat_games") { it.getString(0) to it.getString(1) }.toMap()
    }

    /** The coach's words on a review's moves, as far as they are prepared. */
    fun setReviewWords(key: String, strength: String, words: String) {
        lock.withLock { exec("UPDATE reviews SET words = ? WHERE key = ? AND strength = ?", words, key, strength) }
    }

    /** A position of a review under way, measured (committed at once: a kill keeps it). */
    fun addProgress(key: String, strength: String, depth: Int, multipv: Int, fen: String, measured: String) {
        lock.withLock {
            exec(
                "INSERT OR REPLACE INTO review_progress (key, strength, depth, multipv, fen, measured) VALUES (?, ?, ?, ?, ?, ?)",
                key, strength, depth, multipv, fen, measured,
            )
        }
    }

    /** The positions of a review under way measured so far, at this depth and lines. */
    fun progress(key: String, strength: String, depth: Int, multipv: Int): List<String> = lock.withLock {
        rows(
            "SELECT measured FROM review_progress WHERE key = ? AND strength = ? AND depth = $depth AND multipv = $multipv",
            key, strength,
        ) { it.getString(0) }
    }

    /**
     * A game's review at [strength] is saved: what a review of it at that strength kept part way
     * goes -- every strength's when none is named (the game is gone). A Deep review stopped part
     * way kept its positions only until the game's Standard review was saved or opened, and
     * started over (the review of 2026-10-01). `clear_progress`.
     */
    fun clearProgress(key: String, strength: String? = null) {
        lock.withLock {
            if (strength == null) {
                exec("DELETE FROM review_progress WHERE key = ?", key)
            } else {
                exec("DELETE FROM review_progress WHERE key = ? AND strength = ?", key, strength)
            }
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
     * Anything but a single SELECT (or WITH ... SELECT) over the tables is refused before it
     * runs, and the database refuses any write while it runs; a statement still running after
     * [seconds] is interrupted. Used by the coach's own queries (Ask.kt).
     *
     * The desktop's authorizer allows reading tables and running functions and nothing else,
     * and the model reads what it refused as "not authorized". The phone says the same where it
     * refuses the same (the parity audit, 2026-09-26): a write inside a WITH, which `query_only`
     * stops as "attempt to write a readonly database", and the pragmas' table-valued functions
     * (`pragma_table_info(...)`), which `query_only` lets run and the authorizer does not, so
     * they are refused before they run. SQLite's other refusals are its own message, as Python's
     * sqlite3 gives it, without the code and the statement Android adds.
     */
    fun query(sql: String, maxRows: Int = QUERY_ROWS, seconds: Double = QUERY_SECONDS): Pair<List<String>, List<List<Any?>>> {
        val text = Py.strip(Py.strip(sql).trimEnd(';'))
        if (text.isEmpty()) throw QueryRefused("empty query")
        if (';' in text) throw QueryRefused("one statement at a time")
        val head = Py.split(text)[0].uppercase(Locale.ROOT)
        if (head != "SELECT" && head != "WITH") throw QueryRefused("only SELECT queries are allowed")
        if (pragmaTable(text.replace(STRING_LITERAL, "''"))) throw QueryRefused(NOT_AUTHORIZED)
        return lock.withLock {
            val cancel = CancellationSignal()
            val timer = Timer("profile-query", true)
            db.execSQL("PRAGMA query_only = 1")
            try {
                timer.schedule((seconds * 1000).toLong()) { cancel.cancel() }
                // The names from the statement itself, never stepped; the rows from it capped
                // at one more than wanted: Android's cursor counts a whole result on its first
                // fill, and a query with no LIMIT ran into the time limit and was refused where
                // the desktop's fetchmany stops at its rows (the parity audit, 2026-09-29).
                // A statement that is no read -- a write inside a WITH, a syntax error -- is not
                // one SQLite takes as a subquery: it runs as written, and says what is wrong with it.
                val names = db.rawQuery(text, null, cancel).use { it.columnNames.toList() }
                val capped = try {
                    db.rawQuery("SELECT * FROM ($text) LIMIT ${maxRows + 1}", null, cancel)
                } catch (_: SQLiteException) {
                    db.rawQuery(text, null, cancel)
                }
                capped.use { cursor ->
                    val out = ArrayList<List<Any?>>()
                    while (out.size < maxRows + 1 && cursor.moveToNext()) {
                        out.add((0 until cursor.columnCount).map { i -> cell(cursor, i) })
                    }
                    names to out.take(maxRows)
                }
            } catch (exc: SQLiteReadOnlyDatabaseException) {
                throw QueryRefused(NOT_AUTHORIZED)
            } catch (exc: SQLiteException) {
                throw QueryRefused(sqliteMessage(exc))
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

    /**
     * SQLite's message, as Python's `str(exc)` gives it: Android appends the result code and,
     * for a statement that did not compile, the statement (" (code 1 SQLITE_ERROR): , while
     * compiling: ...").
     */
    private fun sqliteMessage(exc: SQLiteException): String {
        val message = exc.message ?: return exc.toString()
        val at = message.indexOf(" (code ")
        return if (at >= 0) message.substring(0, at) else message
    }

    companion object {
        const val QUERY_ROWS = 200
        const val QUERY_SECONDS = 2.0

        // A game's rows: the library's tables by its library key; a watched game's own by its
        // game id (its library key is WATCHED + the id).
        const val WATCHED = "watched:"
        val LIBRARY_TABLES = listOf("library", "reviews", "review_progress", "stat_games", "stat_moves")
        val WATCHED_TABLES = listOf("evidence", "dialogue", "recommendations", "moves", "games")
        val RESTORABLE = (LIBRARY_TABLES + WATCHED_TABLES + "twins").toSet()

        /** A removed game's rows are kept this long, for Undo (the page offers it for seconds). */
        var UNDO_KEEP_S = 600.0

        // The review strengths, weakest first: a stronger review stands for the game. `STRENGTH_RANK`.
        val STRENGTH_RANK = mapOf("fast" to 1, "standard" to 2, "deep" to 3)

        // The settings' row that counts the saves of Stats facts ([statsRevision]). `STATS_REVISION_KEY`.
        const val STATS_REVISION_KEY = "stats.revision"
        private val RANK = "CASE strength " + STRENGTH_RANK.entries.joinToString(" ") { "WHEN '${it.key}' THEN ${it.value}" } + " ELSE 0 END"

        /** The watched games Stats leaves out: those whose import's facts are kept (the import counts). */
        private const val STOOD_FOR = "SELECT t.watched FROM twins t JOIN stat_games s ON s.key = t.imported"

        /** What the desktop's authorizer answers a statement it denies (`SQLITE_DENY`). */
        private const val NOT_AUTHORIZED = "not authorized"

        /**
         * A pragma's table-valued function used as one -- read FROM, JOINed, or called with its
         * arguments -- and not a CTE of that name: `SELECT 1 AS pragma_count` is a column
         * the desktop's authorizer lets through (the parity audit, 2026-09-29).
         */
        private val PRAGMA_USED = Regex(
            "(?:\\b(?:FROM|JOIN)\\s+|,\\s*)(?:[\"`\\[]?main[\"`\\]]?\\s*\\.\\s*)?[\"`\\[]?(pragma_\\w+)" + "|\\b(pragma_\\w+)\\s*\\(",
            RegexOption.IGNORE_CASE,
        )

        /** A CTE's name (`WITH name AS (`, `, name(cols) AS (`). */
        private val CTE_NAME = Regex("(?:\\bWITH(?:\\s+RECURSIVE)?|,)\\s*(\\w+)\\s*(?:\\([^)]*\\))?\\s*AS\\s*\\(", RegexOption.IGNORE_CASE)

        /** Whether [sql] reads a pragma's table-valued function. */
        internal fun pragmaTable(sql: String): Boolean {
            val ctes = CTE_NAME.findAll(sql).map { it.groupValues[1].lowercase(Locale.ROOT) }.toSet()
            return PRAGMA_USED.findAll(sql).any { m ->
                val name = (m.groups[1] ?: m.groups[2])!!.value.lowercase(Locale.ROOT)
                name !in ctes
            }
        }

        /** A string literal ('it''s'), which is data, not a name: blanked before [PRAGMA_FUNCTION] looks. */
        private val STRING_LITERAL = Regex("'(?:[^']|'')*'")

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
CREATE TABLE IF NOT EXISTS twins (
    watched TEXT PRIMARY KEY,
    imported TEXT NOT NULL,
    found_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS removed (
    key TEXT PRIMARY KEY,
    removed_at REAL NOT NULL,
    grp TEXT NOT NULL,
    rows TEXT,
    trace TEXT
);
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
CREATE INDEX IF NOT EXISTS attempts_puzzle ON puzzle_attempts (puzzle_id, ts)
"""

        /**
         * A list of strings as Python's `json.dumps` writes it -- `["capture", "check"]`, a space
         * after each comma, non-ASCII escaped -- since the coach's own SQL can read it back.
         */
        /** `json.dumps(list(items))`: Python's separators, and every character outside ' '..'~' escaped. */
        fun jsonList(items: List<String>): String = items.joinToString(", ", "[", "]") { pyJsonString(it) }

        /** A string as `json.dumps` (`ensure_ascii`) writes it; DEL too (the parity audit, 2026-09-26). */
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
                    ch.code < 0x20 || ch.code >= 0x7f -> out.append(String.format(Locale.ROOT, "\\u%04x", ch.code))
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

        private fun libraryRow(c: Cursor) = LibraryRow(
            key = c.getString(c.getColumnIndexOrThrow("key")),
            source = c.getString(c.getColumnIndexOrThrow("source")),
            addedAt = c.getDouble(c.getColumnIndexOrThrow("added_at")),
            pgn = c.getString(c.getColumnIndexOrThrow("pgn")),
            playedAt = c.doubleOrNull("played_at"),
            white = c.stringOrNull("white"),
            black = c.stringOrNull("black"),
            whiteElo = c.intOrNull("white_elo"),
            blackElo = c.intOrNull("black_elo"),
            result = c.stringOrNull("result"),
            timeClass = c.stringOrNull("time_class"),
            userSide = c.stringOrNull("user_side"),
        )

        private fun reviewRow(c: Cursor) = ReviewRow(
            key = c.getString(c.getColumnIndexOrThrow("key")),
            strength = c.getString(c.getColumnIndexOrThrow("strength")),
            version = c.getInt(c.getColumnIndexOrThrow("version")),
            reviewedAt = c.getDouble(c.getColumnIndexOrThrow("reviewed_at")),
            report = c.getString(c.getColumnIndexOrThrow("report")),
            summary = c.stringOrNull("summary"),
            words = c.stringOrNull("words"),
        )

        private fun statGameRow(c: Cursor) = StatGameRow(
            key = c.getString(c.getColumnIndexOrThrow("key")),
            strength = c.getString(c.getColumnIndexOrThrow("strength")),
            source = c.stringOrNull("source"),
            timeClass = c.stringOrNull("time_class"),
            playedAt = c.doubleOrNull("played_at"),
            userSide = c.stringOrNull("user_side"),
            bot = c.getInt(c.getColumnIndexOrThrow("bot")) != 0,
            result = c.stringOrNull("result"),
            opening = c.stringOrNull("opening"),
            termination = c.stringOrNull("termination"),
        )

        private fun statMoveRow(c: Cursor) = StatMoveRow(
            key = c.getString(c.getColumnIndexOrThrow("key")),
            ply = c.getInt(c.getColumnIndexOrThrow("ply")),
            side = c.getString(c.getColumnIndexOrThrow("side")),
            fen = c.getString(c.getColumnIndexOrThrow("fen")),
            uci = c.getString(c.getColumnIndexOrThrow("uci")),
            san = c.getString(c.getColumnIndexOrThrow("san")),
            phase = c.getString(c.getColumnIndexOrThrow("phase")),
            loss = c.getDouble(c.getColumnIndexOrThrow("loss")),
            label = c.stringOrNull("label"),
            accuracy = c.doubleOrNull("accuracy"),
            score = c.getDouble(c.getColumnIndexOrThrow("score")),
            chance = c.stringOrNull("chance"),
            found = c.getInt(c.getColumnIndexOrThrow("found")) != 0,
            allowed = c.stringOrNull("allowed"),
            structure = c.getString(c.getColumnIndexOrThrow("structure")),
            themes = c.getString(c.getColumnIndexOrThrow("themes")),
            endgame = c.stringOrNull("endgame"),
            spentMs = c.intOrNull("spent_ms"),
            clockMs = c.intOrNull("clock_ms"),
        )

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

/**
 * What is kept for good of a watched game the player removed: when it began, their side, the
 * result and the moves as read -- enough to know its import ([Library.leftOutOf]). `Trace`.
 */
data class Trace(val startedAt: Double, val userSide: String, val result: String?, val sans: List<String>)

/**
 * A game Game Review can open: where it came from (`source`: chesscom, lichess, pgn, watched),
 * who played it, and its PGN. `key` is the source and its own id ("chesscom:174300959956");
 * `userSide` the side the player had, when their name for that site is known. `LibraryRow`.
 */
data class LibraryRow(
    val key: String,
    val source: String,
    val addedAt: Double,
    val pgn: String,
    val playedAt: Double? = null,
    val white: String? = null,
    val black: String? = null,
    val whiteElo: Int? = null,
    val blackElo: Int? = null,
    val result: String? = null,
    val timeClass: String? = null,
    val userSide: String? = null,
)

/** A game reviewed whole at one strength: the review (JSON), its summary and the coach's words. `ReviewRow`. */
data class ReviewRow(
    val key: String,
    val strength: String,
    val version: Int,
    val reviewedAt: Double,
    val report: String,
    val summary: String?,
    val words: String?,
)

/** A reviewed game's facts as Stats keeps them (`stat_games`): [GameFact] and the review's strength. */
data class StatGameRow(
    val key: String,
    val strength: String,
    val source: String?,
    val timeClass: String?,
    val playedAt: Double?,
    val userSide: String?,
    val bot: Boolean,
    val result: String?,
    val opening: String?,
    val termination: String?,
)

/** A move's facts as Stats keeps them (`stat_moves`): [MoveFact] and its game's key. */
data class StatMoveRow(
    val key: String,
    val ply: Int,
    val side: String,
    val fen: String,
    val uci: String,
    val san: String,
    val phase: String,
    val loss: Double,
    val label: String?,
    val accuracy: Double?,
    val score: Double,
    val chance: String?,
    val found: Boolean,
    val allowed: String?,
    val structure: String,
    val themes: String,
    val endgame: String?,
    val spentMs: Int?,
    val clockMs: Int?,
)

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
/**
 * One cell of the error taxonomy (store.py `CellRow`): mistakes (`errors`) by cause, motif and
 * phase with the expected score they cost, and the occasions (`chances`) with the cause null. A
 * tactic's occasions count in `errors` those the player missed. `games`: the games its errors
 * came from.
 */
data class CellRow(
    val cause: String?,
    val motif: String?,
    val phase: String?,
    val errors: Int,
    val chances: Int,
    val expectedLoss: Double,
    val games: Set<String> = emptySet(),
)

data class SessionRow(
    val sessionId: String,
    val startedAt: Double,
    val endedAt: Double?,
    val games: Int,
    val summary: String?,
)
