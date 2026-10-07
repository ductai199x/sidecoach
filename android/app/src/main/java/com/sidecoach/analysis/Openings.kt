package com.sidecoach.analysis

import android.util.Log
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * Opening identification and the point where a game leaves known theory.
 * `sidecoach/analysis/openings.py`.
 *
 * The data is the Lichess chess-openings tables (CC0): five TSV files `a.tsv` through `e.tsv`
 * in `openings` under the app's files, columns eco, name, pgn, 3810 named lines in total.
 * Setup fetches them, as the desktop's does; until then [OpeningBook.load] returns null and
 * the caller does without opening facts.
 *
 * The same tables answer both questions: a move is *in book* while the game is still standing
 * on a position that some named line passes through, and the first move that leaves every
 * line is the deviation. Indexing is by position, not by move order: each line is replayed
 * and every position along it stored as an EPD, so transpositions resolve for free. Whether a
 * move is in book is walked only as far as the game's first deviation, so a game that wanders
 * out of theory and back does not silently become book again. Its name is Lichess's (scalachess
 * `OpeningDb.search`): the last named position the game reaches in its first 40 plies, so a
 * game that leaves theory and transposes back into a named line takes that line's name (Tai,
 * 2026-10-02).
 *
 * EPD rather than FEN because it drops the halfmove and fullmove counters, so two move orders
 * of unequal length still match, and it names an en passant square only when the capture is
 * actually available.
 *
 * Parsing the tables takes a moment, so the compiled index is cached as JSON beside them and
 * reloaded whenever the source files still have the size and time the cache recorded.
 */

/** A named opening line matched against a game. */
data class Opening(
    /** ECO code, for example "C50". */
    val eco: String,
    /** Lichess name, for example "Italian Game: Giuoco Pianissimo". */
    val name: String,
    /** How many plies of the game the matched line covers. */
    val ply: Int,
)

/** What the annotator asks of a book: `annotator.Book`. */
interface Book {
    fun identify(moves: List<Move>): Opening?
    fun inBook(moves: List<Move>): Boolean
}

/** Prefix index over the Lichess opening lines, keyed by position. */
class OpeningBook(private val names: Map<String, String>, private val positions: Set<String>) : Book {

    val size: Int get() = names.size

    /** How many positions the lines pass through, the start position included. */
    val positionCount: Int get() = positions.size

    /**
     * Walk [moves] from the start position until they leave the book: the index of the first
     * move that left every line, null when the game never left.
     */
    private fun trace(moves: List<Move>): Int? {
        val board = Board()
        for ((index, move) in moves.withIndex()) {
            // Not a game from the start position: moves that do not belong to this history,
            // and it is in no book.
            if (!board.isLegal(move)) return index
            board.push(move)
            if (board.epd() !in positions) return index
        }
        return null
    }

    /**
     * The opening as Lichess names it (scalachess `OpeningDb.search`): the last named position
     * the game reaches within its first [NAME_PLIES] plies, whatever it passed through on the
     * way -- a game that leaves theory and transposes back into a named line takes that line's
     * name (Tai, 2026-10-02). The latest is the most specific name. Null when no named position
     * comes up in those plies, or the moves are not a game from the start.
     */
    override fun identify(moves: List<Move>): Opening? {
        val board = Board()
        var found: Opening? = null
        for ((index, move) in moves.take(NAME_PLIES).withIndex()) {
            if (!board.isLegal(move)) return found
            board.push(move)
            val named = names[board.epd()]
            if (named != null) {
                val (eco, name) = named.split("\t", limit = 2)
                found = Opening(eco = eco, name = name, ply = index + 1)
            }
        }
        return found
    }

    /** Whether [moves] is a prefix of at least one known line. */
    override fun inBook(moves: List<Move>): Boolean = trace(moves) == null

    /** Index of the first move that leaves every known line, null if it never does. */
    fun deviation(moves: List<Move>): Int? = trace(moves)

    companion object {
        private const val TAG = "sidecoach.openings"
        const val CACHE_FILE = "openings.cache.json"
        const val CACHE_VERSION = 1

        /** How far into a game its opening is named: Lichess's SEARCH_MAX_PLIES. */
        const val NAME_PLIES = 40

        /**
         * The book, or null when the data has not been fetched. Never throws because the
         * data is missing, unreadable or corrupt: every failure logs and returns null, so a
         * session without the tables simply has no opening facts.
         */
        fun load(folder: File): OpeningBook? {
            try {
                val tables = folder.listFiles { f: File -> f.isFile && f.name.endsWith(".tsv") }
                    ?.sortedBy { it.name }
                    .orEmpty()
                if (tables.isEmpty()) {
                    Log.i(TAG, "no opening tables in $folder; Setup fetches them")
                    return null
                }
                val cache = File(folder, CACHE_FILE)
                readCache(cache, tables)?.let { return it }
                val (names, positions) = compile(tables)
                if (names.isEmpty()) {
                    Log.w(TAG, "opening tables in $folder held no usable lines")
                    return null
                }
                writeCache(cache, tables, names, positions)
                return OpeningBook(names, positions)
            } catch (exc: IOException) {
                Log.w(TAG, "could not load the opening book: $exc")
                return null
            }
        }

        /** Size and time of each table, so a changed table invalidates the cache. */
        private fun fingerprint(tables: List<File>): JSONObject = JSONObject().apply {
            for (table in tables) put(table.name, JSONArray().put(table.length()).put(table.lastModified()))
        }

        /**
         * Replay every line in the tables: the EPD of each line's final position mapped to
         * "eco\tname", and every EPD any line passes through, the start position included.
         */
        fun compile(tables: List<File>): Pair<Map<String, String>, Set<String>> {
            val names = HashMap<String, String>()
            val positions = HashSet<String>().apply { add(Board().epd()) }
            for (table in tables) {
                val text = table.readText(Charsets.UTF_8)
                for ((number, line) in splitLines(text).withIndex()) {
                    val fields = line.split("\t")
                    if (fields.size != 3 || fields[0] == "eco") continue
                    val (eco, name, pgn) = fields
                    val board = Board()
                    val walked = ArrayList<String>()
                    var bad = false
                    for (token in splitWords(pgn)) {
                        if (token[0].isDigit()) continue
                        val move = board.parseSan(token)
                        if (move == null) {
                            Log.w(TAG, "${table.name} line ${number + 1}: cannot play $token ($name)")
                            bad = true
                            break
                        }
                        board.push(move)
                        walked.add(board.epd())
                    }
                    if (bad || walked.isEmpty()) continue
                    positions.addAll(walked)
                    names.putIfAbsent(walked.last(), "$eco\t$name")
                }
            }
            return names to positions
        }

        /** Python's line boundaries (`str.splitlines`), past the three a Kotlin `lines()` knows. */
        private val LINE_BREAKS = Regex("\r\n|[\n\r\u000B\u000C\u001C\u001D\u001E\u0085  ]")

        /** `text.splitlines()`: no empty line after a final break. */
        private fun splitLines(text: String): List<String> {
            if (text.isEmpty()) return emptyList()
            val lines = text.split(LINE_BREAKS)
            val last = text.last()
            return if (lines.isNotEmpty() && lines.last().isEmpty() && LINE_BREAKS.matches(last.toString())) {
                lines.dropLast(1)
            } else {
                lines
            }
        }

        /** `text.split()`: runs of Python's whitespace, no empty words. */
        private fun splitWords(text: String): List<String> {
            val words = ArrayList<String>()
            val word = StringBuilder()
            for (c in text) {
                if (c.isWhitespace() || c == '\u0085') {
                    if (word.isNotEmpty()) words.add(word.toString())
                    word.setLength(0)
                } else {
                    word.append(c)
                }
            }
            if (word.isNotEmpty()) words.add(word.toString())
            return words
        }

        /** The cached index, or null when it is missing, stale or unreadable. */
        private fun readCache(cache: File, tables: List<File>): OpeningBook? {
            if (!cache.isFile) return null
            return try {
                val blob = JSONObject(cache.readText(Charsets.UTF_8))
                if (blob.getInt("version") != CACHE_VERSION) return null
                if (blob.getJSONObject("sources").toString() != fingerprint(tables).toString()) return null
                val namesJson = blob.getJSONObject("names")
                val names = HashMap<String, String>(namesJson.length() * 2)
                for (key in namesJson.keys()) names[key] = namesJson.getString(key)
                val list = blob.getJSONArray("positions")
                val positions = HashSet<String>(list.length() * 2)
                for (i in 0 until list.length()) positions.add(list.getString(i))
                OpeningBook(names, positions)
            } catch (exc: Exception) {
                Log.w(TAG, "ignoring the opening cache $cache: $exc")
                null
            }
        }

        /** Write the compiled index beside the tables. A read-only folder is not an error. */
        private fun writeCache(cache: File, tables: List<File>, names: Map<String, String>, positions: Set<String>) {
            val blob = JSONObject().apply {
                put("version", CACHE_VERSION)
                put("sources", fingerprint(tables))
                put("names", JSONObject(names))
                put("positions", JSONArray(positions.sorted()))
            }
            val scratch = File(cache.parentFile, cache.nameWithoutExtension + ".tmp")
            try {
                scratch.writeText(blob.toString(), Charsets.UTF_8)
                if (!scratch.renameTo(cache)) throw IOException("rename to $cache failed")
            } catch (exc: IOException) {
                Log.w(TAG, "could not write the opening cache $cache: $exc")
                scratch.delete()
            }
        }
    }
}
