package com.sidecoach.profile

import com.sidecoach.chess.Board
import com.sidecoach.chess.PgnGame
import com.sidecoach.chess.PgnNode
import com.sidecoach.chess.PgnReader
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.llm.HttpReply
import com.sidecoach.llm.Transport
import com.sidecoach.llm.UrlTransport
import org.json.JSONObject
import java.io.IOException
import java.net.URLEncoder
import java.security.MessageDigest
import java.time.LocalDate
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter

/**
 * The games Game Review can open, brought in from where they were played (`profile/library.py`):
 * a pasted or opened PGN, the player's chess.com games through chess.com's official public data
 * (api.chess.com/pub), their Lichess games through Lichess's public API. Each becomes a
 * [LibraryRow]: who played, when, at what ratings and time control, which side was the player's
 * (their username for the site, from the settings), and the PGN.
 *
 * Only public data is read, one request at a time as chess.com asks, under a User-Agent that
 * names the app; nothing is sent to the player's accounts.
 */
object Library {
    const val USER_AGENT = "Sidecoach (a personal chess coach)"
    private const val CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/%s/games/archives"
    private const val LICHESS_GAMES = "https://lichess.org/api/games/user/%s"
    private const val TIMEOUT_MS = 20_000

    /** The site knows no player by that name: a username mistyped in Settings. `NoPlayer`. */
    class NoPlayer(message: String) : Exception(message)

    /** The site answered with an error (`httpx.HTTPStatusError`): said as "could not reach". */
    class HttpFailure(message: String) : IOException(message)

    /** Every game in a PGN text that has moves; a text that is no PGN gives none. `games_in`. */
    fun gamesIn(text: String): List<PgnGame> =
        PgnReader.games(text).filter { it.errors.isEmpty() && it.variations.isNotEmpty() }

    /**
     * A game as the library keeps it. Without a [key] of its own (a pasted PGN), the key is a
     * hash of its moves and players, so the same game pasted twice is one game. `row_for`.
     */
    fun rowFor(
        game: PgnGame,
        source: String,
        key: String? = null,
        userNames: Iterable<String> = emptyList(),
        timeClass: String? = null,
        now: Double? = null,
    ): LibraryRow {
        val headers = game.headers
        val pgn = game.toString()
        val names = userNames.map { it.trim() }.filter { it.isNotEmpty() }.map { it.lowercase() }.toSet()
        val white = headers["White"]
        val black = headers["Black"]
        val userSide = when {
            !white.isNullOrEmpty() && white.lowercase() in names -> "white"
            !black.isNullOrEmpty() && black.lowercase() in names -> "black"
            else -> null
        }
        val own = key ?: run {
            val movetext = game.mainlineMoves().joinToString(" ") { it.uci() }
            val text = "${white ?: "None"}|${black ?: "None"}|${headers["Date"] ?: "None"}|$movetext"
            val digest = MessageDigest.getInstance("SHA-1").digest(text.toByteArray(Charsets.UTF_8))
            "$source:" + digest.joinToString("") { "%02x".format(it) }.take(16)
        }
        return LibraryRow(
            key = own,
            source = source,
            addedAt = now ?: (System.currentTimeMillis() / 1000.0),
            pgn = pgn,
            playedAt = playedAt(game),
            white = white,
            black = black,
            whiteElo = int(headers["WhiteElo"]),
            blackElo = int(headers["BlackElo"]),
            result = headers["Result"],
            timeClass = timeClass?.takeIf { it.isNotEmpty() } ?: timeClassOf(headers["TimeControl"] ?: ""),
            userSide = userSide,
        )
    }

    /** The games of a pasted or opened PGN. `from_pgn`. */
    fun fromPgn(text: String, userNames: Iterable<String> = emptyList()): List<LibraryRow> {
        val names = userNames.toList()
        return gamesIn(text).map { rowFor(it, source = "pgn", userNames = names) }
    }

    /**
     * The player's standard games of the last [months] months on chess.com (their public
     * archive; the newest month can lag the site by up to a day, their documentation says).
     * `from_chesscom`.
     */
    fun fromChesscom(user: String, months: Int = 3, transport: Transport = UrlTransport()): List<LibraryRow> {
        val answer = get(transport, CHESSCOM_ARCHIVES.format(path(user.lowercase())))
        if (answer.status == 404) throw NoPlayer("chess.com has no player named \"$user\".")
        raiseForStatus(answer)
        val all = JSONObject(answer.text).optJSONArray("archives")
        val archives = if (all == null) emptyList() else (0 until all.length()).map { all.get(it).toString() }
        val rows = ArrayList<LibraryRow>()
        for (month in if (months > 0) archives.takeLast(months) else archives) {
            val got = get(transport, month)
            raiseForStatus(got)
            val games = JSONObject(got.text).optJSONArray("games") ?: continue
            for (i in 0 until games.length()) {
                val entry = games.getJSONObject(i)
                if (str(entry, "rules") != "chess" || str(entry, "pgn").isNullOrEmpty()) continue
                for (game in gamesIn(str(entry, "pgn")!!)) {
                    val url = if (!entry.has("url")) "" else str(entry, "url") ?: "None"
                    val gameId = url.trimEnd('/').substringAfterLast("/")
                    rows.add(
                        rowFor(
                            game,
                            source = "chesscom",
                            key = if (gameId.isNotEmpty()) "chesscom:$gameId" else null,
                            userNames = listOf(user),
                            timeClass = str(entry, "time_class"),
                        ),
                    )
                }
            }
        }
        return rows
    }

    /** The player's latest standard games on Lichess, with their clocks. `from_lichess`. */
    fun fromLichess(user: String, maxGames: Int = 60, transport: Transport = UrlTransport()): List<LibraryRow> {
        val params = linkedMapOf(
            "max" to maxGames.toString(),
            "clocks" to "true",
            "opening" to "true",
            "perfType" to "ultraBullet,bullet,blitz,rapid,classical,correspondence",
        )
        val query = params.entries.joinToString("&") { (k, v) -> k + "=" + URLEncoder.encode(v, "UTF-8") }
        val response = get(
            transport,
            LICHESS_GAMES.format(path(user)) + "?" + query,
            mapOf("Accept" to "application/x-chess-pgn"),
        )
        if (response.status == 404) throw NoPlayer("Lichess has no player named \"$user\".")
        raiseForStatus(response)
        val rows = ArrayList<LibraryRow>()
        for (game in gamesIn(response.text)) {
            if (game.headers.getOrDefault("Variant", "Standard") != "Standard") continue
            val site = game.headers.getOrDefault("Site", "")
            val gameId = if ("lichess.org/" in site) site.trimEnd('/').substringAfterLast("/") else ""
            rows.add(
                rowFor(
                    game,
                    source = "lichess",
                    key = if (gameId.isNotEmpty()) "lichess:$gameId" else null,
                    userNames = listOf(user),
                ),
            )
        }
        return rows
    }

    /** A game of the library by its key: imported, or one Sidecoach watched. `find`. */
    fun find(store: Store, key: String): LibraryRow? {
        if (key.startsWith("watched:")) {
            val game = store.game(key.removePrefix("watched:")) ?: return null
            if (game.result == null) return null
            return watchedRow(game, store.moves(game.gameId))
        }
        return store.libraryGame(key)
    }

    // A watched record is a game when it finished with a result and its moves play out from
    // where it began for this many plies at least: a puzzle's mating line recorded from the
    // starting position, or a game cut off when the board was turned round, is not one (Tai's
    // app, 09-28).
    const val WATCHED_PLIES = 10

    /**
     * The games Sidecoach watched to their end, as the library shows them: their moves as
     * recorded (withdrawn ones left out), the player named "You", the opponent as recorded. A
     * record with no result, or whose moves do not play out as a game, is left out. Every one
     * unless [limit]: the Games window listed the latest 200 while Stats counted all (the review
     * of 2026-10-01). `from_watched`.
     */
    fun fromWatched(store: Store, limit: Int? = null): List<LibraryRow> =
        store.games(limit).mapNotNull { watchedRow(it, store.moves(it.gameId)) }

    /** One watched record as the library shows it, or null when it is no game ([fromWatched]). */
    fun watchedRow(game: GameRow, moves: List<MoveRow>): LibraryRow? {
        if (moves.isEmpty() || game.result !in listOf("1-0", "0-1", "1/2-1/2")) return null
        val pgn = PgnGame()
        val board = Board(game.startFen)
        if (game.startFen != STARTING_FEN) pgn.setup(board)
        var node: PgnNode = pgn
        var played = 0
        for (row in moves.sortedWith(compareBy({ it.segment }, { it.ply }))) {
            // A resync's new line: the review stops where the record does.
            val move = board.parseSan(row.san) ?: break
            node = node.addVariation(move)
            board.push(move)
            played += 1
        }
        if (played < WATCHED_PLIES) return null
        val you = "You"
        val them = game.opponent?.takeIf { it.isNotEmpty() } ?: "Opponent"
        pgn.headers["White"] = if (game.userSide == "white") you else them
        pgn.headers["Black"] = if (game.userSide == "black") you else them
        pgn.headers["Result"] = game.result ?: "*"
        val started = LocalDateTime.ofEpochSecond(kotlin.math.floor(game.startedAt).toLong(), 0, ZoneOffset.UTC)
        pgn.headers["Date"] = DATE.format(started)
        if (!game.timeControl.isNullOrEmpty()) pgn.headers["TimeControl"] = game.timeControl
        return LibraryRow(
            key = "watched:${game.gameId}",
            source = "watched",
            addedAt = game.endedAt?.takeIf { it != 0.0 } ?: game.startedAt,
            pgn = pgn.toString(),
            playedAt = game.startedAt,
            white = pgn.headers["White"],
            black = pgn.headers["Black"],
            result = game.result,
            timeClass = timeClassOf(game.timeControl ?: ""),
            userSide = game.userSide,
        )
    }

    // A watched game and an import are one game when the import started at most TWIN_BEFORE_S
    // before Sidecoach began following it (it can join late, never early) and no later than
    // TWIN_AFTER_S after, and nearly every move Sidecoach recorded -- TWIN_SHARE of them,
    // TWIN_PLIES at least -- turns up in the import in order: Sidecoach can misread a move or
    // miss a few, so the record need not agree move for move (Tai, 2026-09-30: "sometimes our
    // app is a bit inaccurate").
    const val TWIN_BEFORE_S = 3 * 3600.0
    const val TWIN_AFTER_S = 600.0
    const val TWIN_SHARE = 0.9
    const val TWIN_PLIES = 12

    // An import that says when in the day it started (chess.com, Lichess); else a date alone, the
    // same day or the next: within DATE_WINDOW_S of the watched game's start. The timed window
    // lies inside that one, so [pairs] looks only at the imports played that near.
    private val TIMED = Regex("^\\[UTCTime \"", RegexOption.MULTILINE)
    const val DATE_WINDOW_S = 36 * 3600.0

    /**
     * Each game Sidecoach watched that was imported too, from chess.com, Lichess or a PGN: the
     * import is the game's record -- every move, the clocks, the names, the ratings -- and
     * stands for the watched one in the library, the reviews and Stats. Pairs found are kept in
     * the store; returns them all, watched key -> imported key. Every watched game, not the
     * latest 200 (the review of 2026-10-01), each against the imports played near it. `reconcile`.
     */
    fun reconcile(store: Store): Map<String, String> {
        val twins = store.twins().toMutableMap()
        val found = pairs(store.games(null), store.library(), twins) { store.moves(it.gameId) }
        for ((watched, imported) in found) {
            store.addTwin(watched, imported)
            twins[watched] = imported
        }
        return twins
    }

    /** The new pairs among watched records and imports, the known ones left as they are ([reconcile]). */
    fun pairs(
        watched: List<Pair<GameRow, List<MoveRow>>>,
        imported: List<LibraryRow>,
        known: Map<String, String>,
    ): LinkedHashMap<String, String> {
        val moves = watched.associate { (game, ms) -> game.gameId to ms }
        return pairs(watched.map { it.first }, imported, known) { moves.getValue(it.gameId) }
    }

    /** [pairs], each watched game's moves read only when an import was played near it. */
    fun pairs(
        games: List<GameRow>,
        imported: List<LibraryRow>,
        known: Map<String, String>,
        movesOf: (GameRow) -> List<MoveRow>,
    ): LinkedHashMap<String, String> {
        val out = LinkedHashMap<String, String>()
        val dated = imported.indices.filter { imported[it].playedAt != null }
            .sortedWith(compareBy({ imported[it].playedAt!! }, { it }))
        val stamps = DoubleArray(dated.size) { imported[dated[it]].playedAt!! }
        val undated = imported.indices.filter { imported[it].playedAt == null }
        val sansOf = HashMap<String, List<String>?>() // an import's moves, read once
        for (game in games) {
            val key = "watched:${game.gameId}"
            if (key in known) continue
            val lo = bisectLeft(stamps, game.startedAt - DATE_WINDOW_S)
            val hi = bisectRight(stamps, game.startedAt + DATE_WINDOW_S)
            // In the library's order, as every import was looked at before: the first best wins.
            val near = (dated.subList(lo, hi) + undated).sorted().map { imported[it] }.filter { mayBe(game, it) }
            if (near.isEmpty()) continue
            val seen = movesOf(game).sortedWith(compareBy({ it.segment }, { it.ply })).map { it.san.trimEnd('+', '#') }
            var best: Pair<Double, String>? = null
            for (row in near) {
                val played = sansOf.getOrPut(row.key) { sans(row.pgn) }
                val share = agreeing(seen, played ?: emptyList())
                if (share != null && (best == null || share > best.first)) best = share to row.key
            }
            if (best != null) out[key] = best.second
        }
        return out
    }

    /** Python's `bisect.bisect_left`: where [x] goes in sorted [a], before any equal to it. */
    private fun bisectLeft(a: DoubleArray, x: Double): Int {
        var lo = 0
        var hi = a.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (a[mid] < x) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** Python's `bisect.bisect_right`: where [x] goes in sorted [a], after any equal to it. */
    private fun bisectRight(a: DoubleArray, x: Double): Int {
        var lo = 0
        var hi = a.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (x < a[mid]) hi = mid else lo = mid + 1
        }
        return lo
    }

    /**
     * The games among [rows] the player removed ([Store.removeGame]), which bringing games in
     * again leaves out: by key, or -- a watched game removed before its import came in -- by
     * being that game, as [reconcile] knows an import. `left_out`.
     */
    fun leftOut(store: Store, rows: List<LibraryRow>): Set<String> =
        leftOutOf(rows, store.removedKeys(), store.removedTraces())

    /** `left_out_of`. */
    fun leftOutOf(rows: List<LibraryRow>, keys: Set<String>, traces: List<Trace>): Set<String> {
        val out = LinkedHashSet<String>()
        for (row in rows) {
            if (row.key in keys) {
                out.add(row.key)
                continue
            }
            var played: List<String>? = null
            for (trace in traces) {
                if (!mayBe(trace.startedAt, trace.userSide, trace.result, row)) continue
                if (played == null) played = sans(row.pgn) ?: emptyList()
                if (agreeing(trace.sans, played) != null) {
                    out.add(row.key)
                    break
                }
            }
        }
        return out
    }

    private fun mayBe(game: GameRow, row: LibraryRow): Boolean = mayBe(game.startedAt, game.userSide, game.result, row)

    /**
     * Whether an import can be the watched game at all: the same side, the same result when both
     * are known, started in the window (a date alone, the same day or the next). `_may_be`.
     */
    private fun mayBe(startedAt: Double, userSide: String, result: String?, row: LibraryRow): Boolean {
        if (row.userSide != null && row.userSide != userSide) return false
        if (!row.result.isNullOrEmpty() && !result.isNullOrEmpty() && row.result != "*" && row.result != result) {
            return false
        }
        val playedAt = row.playedAt ?: return true
        if (!TIMED.containsMatchIn(row.pgn)) return kotlin.math.abs(playedAt - startedAt) <= DATE_WINDOW_S
        return startedAt - TWIN_BEFORE_S <= playedAt && playedAt <= startedAt + TWIN_AFTER_S
    }

    private fun sans(pgn: String): List<String>? {
        val game = gamesIn(pgn).firstOrNull() ?: return null
        val board = game.board()
        return game.mainlineMoves().map { move ->
            board.san(move).trimEnd('+', '#').also { board.push(move) }
        }
    }

    /**
     * The share of the moves Sidecoach recorded that the import has in the same order, when it
     * is enough to call them one game; null otherwise. `_agreeing`.
     */
    fun agreeing(seen: List<String>, played: List<String>): Double? {
        if (seen.isEmpty() || played.isEmpty()) return null
        val matched = matchingBlocks(seen, played).sumOf { it[2] }
        val share = matched.toDouble() / seen.size
        val enough = matched >= minOf(TWIN_PLIES, played.size)
        return if (share >= TWIN_SHARE && enough) share else null
    }

    /**
     * `difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks()`, as (i, j,
     * size) triples ending with (len(a), len(b), 0): no junk, no popular elements.
     */
    fun matchingBlocks(a: List<String>, b: List<String>): List<IntArray> {
        val b2j = HashMap<String, MutableList<Int>>()
        for ((i, elt) in b.withIndex()) b2j.getOrPut(elt) { ArrayList() }.add(i)

        fun longest(alo: Int, ahi: Int, blo: Int, bhi: Int): IntArray {
            var besti = alo
            var bestj = blo
            var bestsize = 0
            var j2len = HashMap<Int, Int>()
            for (i in alo until ahi) {
                val newj2len = HashMap<Int, Int>()
                for (j in b2j[a[i]] ?: emptyList()) {
                    if (j < blo) continue
                    if (j >= bhi) break
                    val k = (j2len[j - 1] ?: 0) + 1
                    newj2len[j] = k
                    if (k > bestsize) {
                        besti = i - k + 1
                        bestj = j - k + 1
                        bestsize = k
                    }
                }
                j2len = newj2len
            }
            while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1]) {
                besti -= 1
                bestj -= 1
                bestsize += 1
            }
            while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize]) {
                bestsize += 1
            }
            return intArrayOf(besti, bestj, bestsize)
        }

        val queue = ArrayDeque<IntArray>()
        queue.addLast(intArrayOf(0, a.size, 0, b.size))
        val blocks = ArrayList<IntArray>()
        while (queue.isNotEmpty()) {
            val (alo, ahi, blo, bhi) = queue.removeLast()
            val x = longest(alo, ahi, blo, bhi)
            val (i, j, k) = x
            if (k > 0) {
                blocks.add(x)
                if (alo < i && blo < j) queue.addLast(intArrayOf(alo, i, blo, j))
                if (i + k < ahi && j + k < bhi) queue.addLast(intArrayOf(i + k, ahi, j + k, bhi))
            }
        }
        blocks.sortWith(compareBy<IntArray>({ it[0] }, { it[1] }, { it[2] }))
        var i1 = 0
        var j1 = 0
        var k1 = 0
        val out = ArrayList<IntArray>()
        for ((i2, j2, k2) in blocks) {
            if (i1 + k1 == i2 && j1 + k1 == j2) {
                k1 += k2
            } else {
                if (k1 > 0) out.add(intArrayOf(i1, j1, k1))
                i1 = i2
                j1 = j2
                k1 = k2
            }
        }
        if (k1 > 0) out.add(intArrayOf(i1, j1, k1))
        out.add(intArrayOf(a.size, b.size, 0))
        return out
    }

    private val CONTROL = Regex("^(\\d+)(?:\\+(\\d+))?$")

    // As Sidecoach words the clock it read at a game's start (profile/Profiler.kt): "10 min", "30 s".
    private val WATCHED_CONTROL = Regex("^(\\d+(?:\\.\\d+)?) (min|s)$")

    /**
     * The time class of a PGN `TimeControl` ("600+5"), or of the clock Sidecoach read ("10
     * min"), by its expected length (the base plus forty increments): bullet under 3 minutes,
     * blitz under 10, rapid under 60, classical beyond; "1/86400" and the like are
     * correspondence ("daily"). `time_class_of`.
     */
    fun timeClassOf(control: String): String? {
        if (control.startsWith("1/")) return "daily"
        val watched = WATCHED_CONTROL.matchEntire(control)
        val match = CONTROL.matchEntire(control)
        val seconds = when {
            watched != null -> watched.groupValues[1].toDouble() * (if (watched.groupValues[2] == "min") 60 else 1)
            match != null ->
                match.groupValues[1].toDouble() + 40 * (match.groupValues[2].ifEmpty { "0" }.toDouble())
            else -> return null
        }
        return when {
            seconds < 180 -> "bullet"
            seconds < 600 -> "blitz"
            seconds < 3600 -> "rapid"
            else -> "classical"
        }
    }

    private val DATE = DateTimeFormatter.ofPattern("yyyy.MM.dd")

    // `time.strptime(..., "%Y.%m.%d %H:%M:%S")`: its patterns for each field, the whole text.
    private val STAMP = Regex(
        "^(\\d\\d\\d\\d)\\.(1[0-2]|0[1-9]|[1-9])\\.(3[01]|[12]\\d|0[1-9]|[1-9]| [1-9])\\s+" +
            "(2[0-3]|[0-1]\\d|\\d):([0-5]\\d|\\d):(6[0-1]|[0-5]\\d|\\d)$",
        RegexOption.IGNORE_CASE,
    )

    /**
     * When the game was played, from its UTC date and time (chess.com, Lichess), else its date
     * alone -- a day on the calendar, taken at noon here so it shows as that day; null when the
     * PGN does not say. `_played_at`.
     */
    fun playedAt(game: PgnGame): Double? {
        val headers = game.headers
        val utcDate = headers["UTCDate"]
        val utcTime = headers["UTCTime"]
        if (!utcDate.isNullOrEmpty() && !utcTime.isNullOrEmpty()) {
            val (day, seconds) = stamp("$utcDate ${utcTime.take(8)}") ?: return null
            return (day.toEpochDay() * 86400 + seconds).toDouble()
        }
        val (day, seconds) = stamp("${headers["Date"] ?: ""} 12:00:00") ?: return null
        val noon = day.atStartOfDay(ZoneId.systemDefault()).plusSeconds(seconds.toLong())
        return noon.toEpochSecond().toDouble()
    }

    /** A date and the seconds into it, as `strptime` reads them; null where it raises. */
    private fun stamp(text: String): Pair<LocalDate, Int>? {
        val m = STAMP.matchEntire(text) ?: return null
        val g = m.groupValues
        val day = try {
            LocalDate.of(g[1].toInt(), g[2].toInt(), g[3].trim().toInt())
        } catch (_: java.time.DateTimeException) {
            return null
        }
        return day to g[4].toInt() * 3600 + g[5].toInt() * 60 + g[6].toInt()
    }

    private fun int(value: String?): Int? =
        if (value == null || value == "" || value == "?") null else value.trim().toIntOrNull()

    private fun path(segment: String): String = URLEncoder.encode(segment, "UTF-8").replace("+", "%20")

    private fun str(entry: JSONObject, name: String): String? =
        if (entry.isNull(name)) null else entry.opt(name)?.toString()

    private fun get(transport: Transport, url: String, extra: Map<String, String> = emptyMap()): HttpReply =
        transport.get(url, mapOf("User-Agent" to USER_AGENT) + extra, TIMEOUT_MS)

    private fun raiseForStatus(reply: HttpReply) {
        if (reply.status >= 400) throw HttpFailure("${reply.status} ${reply.reason ?: ""}".trim())
    }
}
