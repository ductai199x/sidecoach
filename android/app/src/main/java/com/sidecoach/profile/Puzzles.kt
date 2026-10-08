package com.sidecoach.profile

import android.database.sqlite.SQLiteDatabase
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.events.PuzzleRef
import java.io.File
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Puzzles for the player's weak spots (DESIGN section 5.7). `sidecoach/profile/puzzles.py`.
 *
 * The Lichess puzzle database (CC0) filtered into SQLite in the app's files, `puzzles.sqlite`:
 * a puzzle is a position, the solution in UCI, a rating, and the themes Lichess tagged it with
 * -- the same names the tagger uses for the tactics (Motifs), so a weak cell of the profile
 * names the puzzles to practise.
 *
 * A recommendation is made after a game's review from the profile's worst motif: a few puzzles
 * of that theme near the player's puzzle level, none of them solved lately, a failed one back
 * first (spaced repetition of the plainest kind). The player's level moves with the outcomes,
 * a step up on a solve and down on a fail, from a start of 1100.
 *
 * The outcome is read off the screen like everything else: a puzzle opened on Lichess is a
 * position the pipeline joins (GameStarted of kind "position"); its placement and side to move
 * name the puzzle in the store, and the moves the player then makes are checked against the
 * solution -- every solver's move right through the last is a solve, a wrong one a fail.
 */
object Puzzles {

    /** The ratings a puzzle may have to be kept at all. */
    const val RATING_MIN = 600
    const val RATING_MAX = 2000

    /** The window around the player's level a recommended puzzle comes from. */
    const val WINDOW = 150

    /** The level starts here for a new player and moves this much per outcome. */
    const val LEVEL_START = 1100
    const val LEVEL_STEP = 25

    /** How many puzzles a recommendation carries. */
    const val COUNT = 3

    /**
     * A puzzle's rating deviation where the CSV gives none (or the store predates the column):
     * the opponent's deviation in the player's rating update (Glicko).
     */
    const val DEVIATION_FALLBACK = 75

    /**
     * The store's version (SQLite's user_version): 1 once every puzzle has the deviation the CSV
     * gives it. A store made before the column (version 0) gets the column with the fallback,
     * then its deviations from the CSV beside it, once ([PuzzleStore.fillDeviations]).
     */
    const val DEVIATIONS_VERSION = 1

    /** A CSV row's RatingDeviation, or [DEVIATION_FALLBACK] where it has none or not a number. */
    fun deviation(text: String?): Int =
        if (text.isNullOrEmpty()) DEVIATION_FALLBACK else pyInt(text)?.toInt() ?: DEVIATION_FALLBACK

    /**
     * Themes that are tactics, in Lichess's names; the profile's motif names are the same but
     * for the mates, which Lichess splits by length.
     */
    val THEME_FOR_MOTIF: Map<String, List<String>> = mapOf(
        "mate" to listOf("mateIn1", "mateIn2", "mateIn3"),
        "absolutePin" to listOf("pin"),
    )

    /** `board_fen turn` of a position, after [firstMove] when given. */
    fun positionKey(fen: String, firstMove: String? = null): String {
        val board = Board(fen)
        if (!firstMove.isNullOrEmpty()) {
            val move = Move.fromUciOrNull(firstMove)
            if (move == null || !board.isLegal(move)) throw IllegalArgumentException("illegal $firstMove in $fen")
            board.push(move)
        }
        return "${board.boardFen()} ${if (board.turn) "w" else "b"}"
    }

    fun themesFor(motif: String): List<String> = THEME_FOR_MOTIF[motif] ?: listOf(motif)

    /**
     * The puzzle store in the app's files, filled once from the Lichess CSV beside it
     * (the CSV files in `puzzles/`, the sample Setup writes) when it is empty; null when there are no
     * puzzles to be had. A store made before the puzzles kept their rating deviation gets them
     * from the same CSV, once.
     */
    fun openStore(files: File): PuzzleStore? {
        val store = PuzzleStore(File(files, "puzzles.sqlite"))
        val csvs = File(files, "puzzles").listFiles { f: File -> f.name.endsWith(".csv") }?.sortedBy { it.name }.orEmpty()
        if (store.count() == 0) {
            for (csv in csvs) store.importCsv(csv)
        } else if (store.version() < DEVIATIONS_VERSION) {
            store.fillDeviations(csvs)
        }
        if (store.count() == 0) {
            store.close()
            return null
        }
        return store
    }

    /** The player's puzzle level from the profile store ([LEVEL_START] to begin with). */
    fun puzzleLevel(store: Store): Int = store.setting("puzzle_level")?.takeIf { it.isNotEmpty() }?.toInt() ?: LEVEL_START

    /**
     * An attempt into the profile store: kept beside the motif, the level moved a step up on a
     * solve and down on a fail. Shared by the profiler (a puzzle followed on the screen) and
     * the puzzle page (one solved on its own board).
     */
    fun recordAttempt(store: Store, puzzleId: String, motif: String?, solved: Boolean, moves: List<String>, ts: Double): PuzzleOutcome {
        val step = if (solved) LEVEL_STEP else -LEVEL_STEP
        val level = maxOf(RATING_MIN, minOf(RATING_MAX, puzzleLevel(store) + step))
        store.setSetting("puzzle_level", level.toString())
        store.addAttempt(puzzleId, ts, motif, solved, moves)
        return PuzzleOutcome(gameId = "puzzle-$puzzleId", puzzleId = puzzleId, motif = motif, solved = solved, level = level)
    }

    /** The records of a CSV text as `list(csv.reader(io.StringIO(text, newline="")))` reads them. */
    fun csvRows(text: String): List<List<String>> {
        val reader = CsvReader(java.io.StringReader(text))
        val rows = ArrayList<List<String>>()
        while (true) rows.add(reader.next() ?: break)
        return rows
    }

    /**
     * Python's `int(text)`: whitespace around it, a sign, decimal digits of any script with
     * single underscores between them; null where Python raises ValueError.
     */
    fun pyInt(text: String): Long? {
        val s = com.sidecoach.voice.Py.strip(text)
        var i = 0
        var negative = false
        if (s.startsWith("+") || s.startsWith("-")) {
            negative = s[0] == '-'
            i = 1
        }
        var value = 0L
        var afterDigit = false
        var digits = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            i += Character.charCount(cp)
            if (cp == '_'.code) {
                if (!afterDigit) return null
                afterDigit = false
                continue
            }
            val d = Character.digit(cp, 10)
            if (d < 0) return null
            // Python's int has no ceiling; past a Long's, the number is only ever out of range.
            value = if (value > (Long.MAX_VALUE - d) / 10) Long.MAX_VALUE else value * 10 + d
            afterDigit = true
            digits++
        }
        if (digits == 0 || !afterDigit) return null
        return if (negative) -value else value
    }
}

/**
 * Python's `csv.reader` (the excel dialect, not strict) over a text opened with `newline=""`:
 * `_csv.c`'s states, one character at a time, a record read on as far as a quoted field runs.
 * A line ends at "\n", "\r\n" or a lone "\r", as Python's file splits it; a blank line is an
 * empty record, which `csv.DictReader` skips; a quote is special only at a field's start, and
 * after a quoted field's closing quote whatever follows is kept as it is. Streamed: the whole
 * Lichess database is a gigabyte (the parity audit, 2026-09-26: the phone read the file whole,
 * dropped a lone "\r" and let a quote open anywhere).
 */
class CsvReader(input: java.io.Reader) {
    private val input = if (input is java.io.BufferedReader) input else java.io.BufferedReader(input)

    /** The character read after a "\r" that was not "\n", for the next line; -2 for none. */
    private var ahead = -2

    private val fields = ArrayList<String>()
    private val field = StringBuilder()
    private var state = START_RECORD

    private fun read(): Int {
        if (ahead != -2) {
            val c = ahead
            ahead = -2
            return c
        }
        return input.read()
    }

    /** The next record, or null at the end: `Reader_iternext`. */
    fun next(): List<String>? {
        fields.clear()
        field.setLength(0)
        state = START_RECORD
        do {
            var c = read()
            if (c == -1) {
                // The end: a record under way is saved, as the lenient reader does.
                if (field.isNotEmpty() || state == IN_QUOTED_FIELD) {
                    save()
                    return ArrayList(fields)
                }
                return null
            }
            // One line, as Python's file iterator gives it, then its end.
            while (true) {
                process(c)
                if (c == '\n'.code) break
                if (c == '\r'.code) {
                    val n = read()
                    if (n == '\n'.code) process(n) else if (n != -1) ahead = n
                    break
                }
                c = read()
                if (c == -1) break
            }
            process(EOL)
        } while (state != START_RECORD)
        return ArrayList(fields)
    }

    private fun save() {
        fields.add(field.toString())
        field.setLength(0)
    }

    private fun endOfLine(c: Int): Boolean = c == '\n'.code || c == '\r'.code || c == EOL

    /** `parse_process_char`. */
    private fun process(c: Int) {
        when (state) {
            START_RECORD -> when {
                c == EOL -> Unit // a blank line: an empty record
                c == '\n'.code || c == '\r'.code -> state = EAT_CRNL
                else -> startField(c)
            }
            START_FIELD -> startField(c)
            IN_FIELD -> when {
                endOfLine(c) -> {
                    save()
                    state = if (c == EOL) START_RECORD else EAT_CRNL
                }
                c == ','.code -> {
                    save()
                    state = START_FIELD
                }
                else -> field.append(c.toChar())
            }
            IN_QUOTED_FIELD -> when {
                c == EOL -> Unit // the field runs on to the next line
                c == '"'.code -> state = QUOTE_IN_QUOTED_FIELD
                else -> field.append(c.toChar())
            }
            QUOTE_IN_QUOTED_FIELD -> when {
                c == '"'.code -> {
                    field.append('"')
                    state = IN_QUOTED_FIELD
                }
                c == ','.code -> {
                    save()
                    state = START_FIELD
                }
                endOfLine(c) -> {
                    save()
                    state = if (c == EOL) START_RECORD else EAT_CRNL
                }
                else -> {
                    field.append(c.toChar())
                    state = IN_FIELD
                }
            }
            EAT_CRNL -> when {
                c == '\n'.code || c == '\r'.code -> Unit
                c == EOL -> state = START_RECORD
                else -> throw IllegalStateException("new-line character seen in unquoted field")
            }
        }
    }

    private fun startField(c: Int) {
        when {
            endOfLine(c) -> {
                save()
                state = if (c == EOL) START_RECORD else EAT_CRNL
            }
            c == '"'.code -> state = IN_QUOTED_FIELD
            c == ','.code -> {
                save()
                state = START_FIELD
            }
            else -> {
                field.append(c.toChar())
                state = IN_FIELD
            }
        }
    }

    private companion object {
        const val EOL = -3
        const val START_RECORD = 0
        const val START_FIELD = 1
        const val IN_FIELD = 2
        const val IN_QUOTED_FIELD = 3
        const val QUOTE_IN_QUOTED_FIELD = 4
        const val EAT_CRNL = 5
    }
}

data class Puzzle(
    val id: String,
    /** The position before the opponent's first move, as Lichess stores it. */
    val fen: String,
    /** UCI: the opponent's move first, then the solution in turns. */
    val moves: List<String>,
    val rating: Int,
    val popularity: Int,
    val plays: Int,
    val themes: List<String>,
    /** The rating's deviation, as Lichess holds it. */
    val deviation: Int = Puzzles.DEVIATION_FALLBACK,
) {
    val url: String get() = "https://lichess.org/training/$id"

    /** The moves the player must find: the second, fourth, ... of the line. */
    val solverMoves: List<String> get() = moves.filterIndexed { i, _ -> i % 2 == 1 }

    /**
     * The position the solver faces: the placement and the side to move after the opponent's
     * first move -- what the screen shows, and the key the store looks a puzzle up by.
     */
    fun position(): String = Puzzles.positionKey(fen, moves[0])

    fun ref(): PuzzleRef = PuzzleRef(id = id, url = url, rating = rating, themes = themes)
}

/** The puzzles on disk (or in memory), imported once from a Lichess CSV. */
class PuzzleStore(val path: File?) {

    private val db: SQLiteDatabase
    private val lock = ReentrantLock()

    init {
        path?.parentFile?.mkdirs()
        db = if (path != null) SQLiteDatabase.openOrCreateDatabase(path, null) else SQLiteDatabase.create(null)
        lock.withLock {
            val columns = columns()
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS puzzles (id TEXT PRIMARY KEY, fen TEXT NOT NULL, moves TEXT NOT NULL," +
                    " rating INTEGER NOT NULL, popularity INTEGER NOT NULL DEFAULT 0, plays INTEGER NOT NULL DEFAULT 0," +
                    " themes TEXT NOT NULL, position TEXT NOT NULL," +
                    " deviation INTEGER NOT NULL DEFAULT ${Puzzles.DEVIATION_FALLBACK})",
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS puzzles_rating ON puzzles (rating)")
            db.execSQL("CREATE INDEX IF NOT EXISTS puzzles_position ON puzzles (position)")
            if (columns.isEmpty()) {
                // A new store: whatever is imported into it carries its deviation.
                db.execSQL("PRAGMA user_version = ${Puzzles.DEVIATIONS_VERSION}")
            } else if ("deviation" !in columns) {
                // A store from before the column: the fallback until fillDeviations.
                try {
                    db.execSQL(
                        "ALTER TABLE puzzles ADD COLUMN deviation INTEGER NOT NULL DEFAULT ${Puzzles.DEVIATION_FALLBACK}",
                    )
                } catch (_: android.database.sqlite.SQLiteException) {
                    // another opener added it
                }
            }
        }
    }

    private fun columns(): Set<String> =
        db.rawQuery("PRAGMA table_info(puzzles)", null).use { c ->
            val out = HashSet<String>()
            while (c.moveToNext()) out.add(c.getString(c.getColumnIndexOrThrow("name")))
            out
        }

    fun version(): Int = lock.withLock {
        db.rawQuery("PRAGMA user_version", null).use { it.moveToFirst(); it.getInt(0) }
    }

    /**
     * Each puzzle's rating deviation from the Lichess CSVs it was imported from, in one
     * transaction with the store's version: a store from before the column gets them once, and
     * one interrupted on the way is filled again at the next open. Returns how many puzzles got
     * one; with no CSV to read, nothing changes (the fallback stays).
     */
    fun fillDeviations(files: List<File>): Int {
        if (files.isEmpty()) return 0
        var filled = 0
        lock.withLock {
            db.beginTransaction()
            try {
                val update = db.compileStatement("UPDATE puzzles SET deviation = ? WHERE id = ?")
                for (file in files) {
                    file.bufferedReader(Charsets.UTF_8).use { text ->
                        val records = CsvReader(text)
                        val header = records.next() ?: return@use
                        while (true) {
                            val values = records.next() ?: break
                            if (values.isEmpty()) continue
                            val row = HashMap<String, String?>()
                            for ((i, name) in header.withIndex()) row[name] = values.getOrNull(i)
                            update.clearBindings()
                            update.bindLong(1, Puzzles.deviation(row["RatingDeviation"]).toLong())
                            val id = row["PuzzleId"]
                            if (id == null) update.bindNull(2) else update.bindString(2, id)
                            filled += update.executeUpdateDelete()
                        }
                    }
                }
                update.close()
                db.execSQL("PRAGMA user_version = ${Puzzles.DEVIATIONS_VERSION}")
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
        }
        return filled
    }

    fun close() {
        lock.withLock { db.close() }
    }

    fun count(): Int = lock.withLock {
        db.rawQuery("SELECT COUNT(*) FROM puzzles", null).use { it.moveToFirst(); it.getInt(0) }
    }

    /** Every puzzle out; the store stays open for a fresh import. */
    fun clear() {
        lock.withLock {
            db.execSQL("DELETE FROM puzzles")
            runCatching { db.execSQL("VACUUM") }
        }
    }

    /**
     * Load a Lichess puzzle CSV (the database's columns: PuzzleId,FEN,Moves,Rating,
     * RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags), keeping the puzzles in
     * the rating band; returns how many were added.
     */
    fun importCsv(file: File, limit: Int? = null): Int {
        var added = 0
        file.bufferedReader(Charsets.UTF_8).use { text ->
            // `csv.DictReader`: the first record names the fields; a blank line is skipped; a
            // field a short row lacks is None (null), a long row's extra fields are not read.
            val records = CsvReader(text)
            val header = records.next() ?: return 0
            lock.withLock {
                db.beginTransaction()
                try {
                    val insert = db.compileStatement(
                        "INSERT OR IGNORE INTO puzzles (id, fen, moves, rating, popularity, plays, themes, position, deviation)" +
                            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    )
                    while (true) {
                        var values = records.next() ?: break
                        while (values.isEmpty()) values = records.next() ?: break
                        if (values.isEmpty()) break
                        val row = HashMap<String, String?>()
                        for ((i, name) in header.withIndex()) row[name] = values.getOrNull(i)
                        // A row too short to hold a rating (None) is skipped, as the desktop's
                        // import skips it on int(None)'s TypeError (desktop 1e2e10f).
                        val rating = row["Rating"]?.let { Puzzles.pyInt(it) } ?: continue
                        if (rating < Puzzles.RATING_MIN || rating > Puzzles.RATING_MAX) continue
                        val moves = com.sidecoach.voice.Py.split(row["Moves"] ?: "")
                        if (moves.size < 2) continue
                        val position = try {
                            Puzzles.positionKey(row["FEN"] ?: "", moves[0])
                        } catch (_: Exception) {
                            continue
                        }
                        insert.clearBindings()
                        insert.bindString(1, row["PuzzleId"] ?: "")
                        insert.bindString(2, row["FEN"] ?: "")
                        insert.bindString(3, row["Moves"] ?: "")
                        insert.bindLong(4, rating)
                        insert.bindLong(5, count(row["Popularity"]))
                        insert.bindLong(6, count(row["NbPlays"]))
                        insert.bindString(7, row["Themes"] ?: "")
                        insert.bindString(8, position)
                        insert.bindLong(9, Puzzles.deviation(row["RatingDeviation"]).toLong())
                        added += insert.executeUpdateDelete()
                        if (limit != null && added >= limit) break
                    }
                    insert.close()
                    db.setTransactionSuccessful()
                } finally {
                    db.endTransaction()
                }
            }
        }
        return added
    }

    /** `int(row.get(name) or 0)`: raises, as Python's does, on what is not a number. */
    private fun count(text: String?): Long {
        if (text.isNullOrEmpty()) return 0
        return Puzzles.pyInt(text) ?: throw NumberFormatException("invalid literal for int() with base 10: '$text'")
    }

    private fun puzzles(sql: String, args: Array<String>?): List<Puzzle> = lock.withLock {
        db.rawQuery(sql, args).use { c ->
            val out = ArrayList<Puzzle>()
            while (c.moveToNext()) {
                out.add(
                    Puzzle(
                        id = c.getString(c.getColumnIndexOrThrow("id")),
                        fen = c.getString(c.getColumnIndexOrThrow("fen")),
                        moves = c.getString(c.getColumnIndexOrThrow("moves")).split(Regex("\\s+")).filter { it.isNotEmpty() },
                        rating = c.getInt(c.getColumnIndexOrThrow("rating")),
                        popularity = c.getInt(c.getColumnIndexOrThrow("popularity")),
                        plays = c.getInt(c.getColumnIndexOrThrow("plays")),
                        themes = c.getString(c.getColumnIndexOrThrow("themes")).split(Regex("\\s+")).filter { it.isNotEmpty() },
                        deviation = c.getInt(c.getColumnIndexOrThrow("deviation")),
                    ),
                )
            }
            out
        }
    }

    fun get(puzzleId: String): Puzzle? = puzzles("SELECT * FROM puzzles WHERE id = ?", arrayOf(puzzleId)).firstOrNull()

    /** The puzzles whose solver's position is [key] (see [Puzzles.positionKey]). */
    fun atPosition(key: String): List<Puzzle> = puzzles("SELECT * FROM puzzles WHERE position = ?", arrayOf(key))

    /**
     * [count] puzzles of the motif's themes within WINDOW of [level]: the ids in [first]
     * (failed lately, due again) ahead of a draw among the rest, none in [exclude] (solved
     * lately), the more played the better. [shuffle] is the draw: Python's
     * `random.Random().shuffle`, injected so a test can fix it.
     */
    fun pick(
        motif: String,
        level: Int,
        count: Int = Puzzles.COUNT,
        exclude: Set<String> = emptySet(),
        first: List<String> = emptyList(),
        shuffle: (MutableList<Puzzle>) -> Unit = { it.shuffle() },
    ): List<Puzzle> {
        val themes = Puzzles.themesFor(motif)
        val clauses = themes.joinToString(" OR ") { "(' ' || themes || ' ') LIKE ?" }
        val params = themes.map { "% $it %" }.toTypedArray()
        val rows = puzzles(
            "SELECT * FROM puzzles WHERE ($clauses) AND rating BETWEEN ${level - Puzzles.WINDOW}" +
                " AND ${level + Puzzles.WINDOW} ORDER BY plays DESC LIMIT 400",
            params,
        )
        val pool = rows.filter { it.id !in exclude }
        val chosen = ArrayList(pool.filter { it.id in first }.take(count))
        val rest = pool.filter { it.id !in first }.toMutableList()
        shuffle(rest)
        chosen.addAll(rest.take(maxOf(0, count - chosen.size)))
        return chosen
    }
}

/**
 * A puzzle as the player works through it on the screen: which solver moves came, and whether
 * it is solved or failed yet.
 */
class Attempt(val puzzle: Puzzle, val motif: String?) {
    val played = ArrayList<String>()

    /** "solved" | "failed" | null while under way. */
    var outcome: String? = null
        private set

    /** A move on the board; the outcome once it is decided, else null. */
    fun move(uci: String, byUser: Boolean): String? {
        if (outcome != null) return null
        if (!byUser) return null  // the site's replies are the line's; nothing to judge
        val expected = puzzle.solverMoves
        val index = played.size
        if (index >= expected.size) return null
        played.add(uci)
        val want = expected[index]
        // Lichess accepts any mating move where the line's is a mate; the board says.
        if (uci != want && !alsoMates(index, uci)) {
            outcome = "failed"
        } else if (played.size == expected.size) {
            outcome = "solved"
        }
        return outcome
    }

    private fun alsoMates(index: Int, uci: String): Boolean {
        val board = Board(puzzle.fen)
        for (move in puzzle.moves.take(1 + 2 * index) + uci) {
            val parsed = Move.fromUciOrNull(move) ?: return false
            if (!board.isLegal(parsed)) return false
            board.push(parsed)
        }
        return board.isCheckmate()
    }
}
