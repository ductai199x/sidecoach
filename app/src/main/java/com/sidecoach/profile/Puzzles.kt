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
     * puzzles to be had.
     */
    fun openStore(files: File): PuzzleStore? {
        val store = PuzzleStore(File(files, "puzzles.sqlite"))
        if (store.count() == 0) {
            val csvs = File(files, "puzzles").listFiles { f: File -> f.name.endsWith(".csv") }?.sortedBy { it.name }.orEmpty()
            for (csv in csvs) store.importCsv(csv)
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

    /**
     * The rows of a CSV file as Python's `csv` module reads them: fields split on commas,
     * a quoted field may hold commas, newlines and doubled quotes.
     */
    fun csvRows(text: String): List<List<String>> {
        val rows = ArrayList<List<String>>()
        var row = ArrayList<String>()
        val field = StringBuilder()
        var quoted = false
        var i = 0
        var sawAny = false
        while (i < text.length) {
            val ch = text[i]
            if (quoted) {
                if (ch == '"') {
                    if (i + 1 < text.length && text[i + 1] == '"') {
                        field.append('"')
                        i++
                    } else {
                        quoted = false
                    }
                } else {
                    field.append(ch)
                }
            } else {
                when (ch) {
                    '"' -> quoted = true
                    ',' -> { row.add(field.toString()); field.setLength(0) }
                    '\r' -> Unit
                    '\n' -> { row.add(field.toString()); field.setLength(0); rows.add(row); row = ArrayList(); sawAny = false; i++; continue }
                    else -> field.append(ch)
                }
            }
            sawAny = true
            i++
        }
        if (sawAny || field.isNotEmpty()) {
            row.add(field.toString())
            rows.add(row)
        }
        return rows
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
            db.execSQL(
                "CREATE TABLE IF NOT EXISTS puzzles (id TEXT PRIMARY KEY, fen TEXT NOT NULL, moves TEXT NOT NULL," +
                    " rating INTEGER NOT NULL, popularity INTEGER NOT NULL DEFAULT 0, plays INTEGER NOT NULL DEFAULT 0," +
                    " themes TEXT NOT NULL, position TEXT NOT NULL)",
            )
            db.execSQL("CREATE INDEX IF NOT EXISTS puzzles_rating ON puzzles (rating)")
            db.execSQL("CREATE INDEX IF NOT EXISTS puzzles_position ON puzzles (position)")
        }
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
        val rows = Puzzles.csvRows(file.readText(Charsets.UTF_8))
        if (rows.isEmpty()) return 0
        val header = rows[0]
        var added = 0
        lock.withLock {
            db.beginTransaction()
            try {
                val insert = db.compileStatement(
                    "INSERT OR IGNORE INTO puzzles (id, fen, moves, rating, popularity, plays, themes, position)" +
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                )
                for (values in rows.drop(1)) {
                    val row = header.zip(values).toMap()
                    val rating = row["Rating"]?.toIntOrNull() ?: continue
                    if (rating < Puzzles.RATING_MIN || rating > Puzzles.RATING_MAX) continue
                    val moves = (row["Moves"] ?: "").split(Regex("\\s+")).filter { it.isNotEmpty() }
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
                    insert.bindLong(4, rating.toLong())
                    insert.bindLong(5, (row["Popularity"]?.takeIf { it.isNotEmpty() }?.toLong() ?: 0))
                    insert.bindLong(6, (row["NbPlays"]?.takeIf { it.isNotEmpty() }?.toLong() ?: 0))
                    insert.bindString(7, row["Themes"] ?: "")
                    insert.bindString(8, position)
                    added += insert.executeUpdateDelete()
                    if (limit != null && added >= limit) break
                }
                insert.close()
                db.setTransactionSuccessful()
            } finally {
                db.endTransaction()
            }
        }
        return added
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
