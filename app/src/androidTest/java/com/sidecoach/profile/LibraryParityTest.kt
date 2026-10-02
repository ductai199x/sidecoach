package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.llm.HttpReply
import com.sidecoach.llm.Transport
import org.json.JSONArray
import org.json.JSONObject
import org.junit.AfterClass
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.time.LocalDateTime
import java.time.ZoneId
import java.time.ZoneOffset

/**
 * The games library as the desktop's (profile/Library.kt): `tools/make_library_fixtures.py`
 * records what `profile/library.py` makes of pasted PGNs under several usernames, time controls,
 * difflib's matching blocks, chess.com's and Lichess's answers, and stores of watched records
 * and imports (`from_watched`'s rows, `reconcile`'s pairs).
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class LibraryParityTest {

    @Test
    fun pastedGamesAreTheDesktopsRows() {
        val failures = ArrayList<String>()
        var rows = 0
        val cases = fixture.getJSONArray("pgn")
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val names = strings(case.getJSONArray("names"))
            val got = Library.fromPgn(case.getString("text"), names)
            val want = case.getJSONArray("rows")
            val label = "${case.getString("name")} $names"
            if (got.size != want.length()) {
                failures.add("$label: ${got.size} rows, python ${want.length()}")
                continue
            }
            for (r in got.indices) {
                rows += 1
                val w = want.getJSONObject(r)
                if (w.has("utc_noon")) {
                    val noon = LocalDateTime.ofEpochSecond(w.getDouble("utc_noon").toLong(), 0, ZoneOffset.UTC)
                    w.put("played_at", noon.atZone(ZoneId.systemDefault()).toEpochSecond().toDouble())
                }
                compare("$label #$r", got[r], w, failures)
            }
        }
        assertTrue("$rows rows; ${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun timeClassesAreTheDesktops() {
        val cases = fixture.getJSONArray("time_classes")
        val failures = (0 until cases.length()).map { cases.getJSONArray(it) }.mapNotNull {
            val control = it.getString(0)
            val want = if (it.isNull(1)) null else it.getString(1)
            val got = Library.timeClassOf(control)
            if (got != want) "'$control': mine $got, python $want" else null
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun matchingBlocksAreDifflibs() {
        val cases = fixture.getJSONArray("blocks")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val a = strings(case.getJSONArray("a"))
            val b = strings(case.getJSONArray("b"))
            val want = case.getJSONArray("blocks").let { bs -> (0 until bs.length()).map { strings(bs.getJSONArray(it)) } }
            val got = Library.matchingBlocks(a, b).map { block -> block.map { it.toString() } }
            if (got != want) failures.add("$a / $b\n  mine:   $got\n  python: $want")
            val share = if (case.isNull("agreeing")) null else case.getDouble("agreeing")
            val mine = Library.agreeing(a, b)
            if (mine != share) failures.add("$a / $b: agreeing mine $mine, python $share")
        }
        assertTrue("${cases.length()} cases; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun chesscomAnswersGiveTheDesktopsRows() {
        val cases = fixture.getJSONArray("chesscom")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val site = Site(case.getJSONObject("answers"))
            site(case, failures) { Library.fromChesscom(case.getString("user"), months = case.getInt("months"), transport = site) }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun lichessAnswersGiveTheDesktopsRows() {
        val cases = fixture.getJSONArray("lichess")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val site = Site(case.getJSONObject("answers"))
            site(case, failures) { Library.fromLichess(case.getString("user"), transport = site) }
            if (site.headers.any { it["Accept"] != "application/x-chess-pgn" || "Sidecoach" !in (it["User-Agent"] ?: "") }) {
                failures.add("${case.getString("name")}: asked without the PGN Accept or the User-Agent")
            }
            if (site.queries.any { "clocks=true" !in it }) failures.add("${case.getString("name")}: no clocks asked")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun watchedGamesAndTheirTwinsAreTheDesktops() {
        val stores = fixture.getJSONArray("stores")
        val failures = ArrayList<String>()
        var twins = 0
        for (s in 0 until stores.length()) {
            val store = stores.getJSONObject(s)
            val games = store.getJSONArray("games").let { gs -> (0 until gs.length()).map { gameRow(gs.getJSONObject(it)) } }
            val moves = store.getJSONObject("moves")
            val watched = games.map { g ->
                val ms = moves.getJSONArray(g.gameId)
                g to (0 until ms.length()).map { moveRow(ms.getJSONObject(it)) }
            }
            val gotWatched = watched.mapNotNull { (g, ms) -> Library.watchedRow(g, ms) }
            val wantWatched = store.getJSONArray("watched")
            if (gotWatched.size != wantWatched.length()) {
                failures.add("store $s: ${gotWatched.map { it.key }} watched, python ${wantWatched.length()}")
            } else {
                for (r in gotWatched.indices) compare("store $s watched #$r", gotWatched[r], wantWatched.getJSONObject(r), failures)
            }
            val library = store.getJSONArray("library").let { ls -> (0 until ls.length()).map { libraryRow(ls.getJSONObject(it)) } }
            val got = Library.pairs(watched, library, emptyMap())
            val want = store.getJSONObject("twins").let { t -> t.keys().asSequence().associateWith { t.getString(it) } }
            twins += want.size
            if (got != want) failures.add("store $s twins\n  mine:   $got\n  python: $want")
        }
        assertTrue("$twins twins; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    /**
     * Games removed as the player removes them (Tai, 2026-10-01), replayed on the phone's own
     * store: what each removal took (a second click nothing, an import its watched game), the
     * keys and traces kept for good, which games bringing them in again leaves out, what is
     * left, and what one Undo puts back -- as the desktop's Store.remove_game and left_out_of.
     */
    @Test
    fun removalsAreTheDesktops() {
        val stores = fixture.getJSONArray("stores")
        val failures = ArrayList<String>()
        var asked = 0
        for (s in 0 until stores.length()) {
            val case = stores.getJSONObject(s)
            val removal = case.getJSONObject("removal")
            val before = removal.getJSONObject("before")
            val library = case.getJSONArray("library").let { ls -> (0 until ls.length()).map { libraryRow(ls.getJSONObject(it)) } }
            fun check(what: String, mine: Any, theirs: Any) {
                if (mine != theirs) failures.add("store $s $what\n  mine:   $mine\n  python: $theirs")
            }
            val store = Store(null)
            try {
                val games = before.getJSONArray("games")
                for (i in 0 until games.length()) {
                    val g = gameRow(games.getJSONObject(i))
                    store.startGame(g.gameId, g.startedAt, g.userSide, g.startFen, g.sessionId)
                    store.setGame(g.gameId, mapOf("time_control" to g.timeControl, "opponent" to g.opponent))
                    if (g.result != null) store.endGame(g.gameId, g.endedAt ?: 0.0, g.result, g.reason)
                    val moves = before.getJSONObject("moves").getJSONArray(g.gameId)
                    for (m in 0 until moves.length()) {
                        val mv = moveRow(moves.getJSONObject(m))
                        store.addMove(
                            mv.gameId, mv.segment, mv.ply, mv.moveNumber, mv.san, mv.uci, mv.byUser, mv.fenBefore,
                            mv.fenAfter, null, null, null, false, false, null, mv.ts,
                        )
                    }
                }
                for (row in library) store.addLibraryGame(row)
                val twins = before.getJSONObject("twins")
                for (w in twins.keys()) store.addTwin(w, twins.getString(w))

                val keys = strings(removal.getJSONArray("asked"))
                asked += keys.size
                val want = removal.getJSONArray("removed").let { r -> (0 until r.length()).map { strings(r.getJSONArray(it)) } }
                check("removed", keys.map { store.removeGame(it) }, want)
                check("keys kept", store.removedKeys().sorted(), strings(removal.getJSONArray("keys")))
                val traces = store.removedTraces()
                val wantTraces = removal.getJSONArray("traces").let { t ->
                    (0 until t.length()).map { t.getJSONObject(it) }.map { o ->
                        Trace(o.getDouble("started_at"), o.getString("user_side"), optString(o, "result"), strings(o.getJSONArray("sans")))
                    }
                }
                check("traces", traces, wantTraces)
                check("left out", Library.leftOutOf(library, store.removedKeys(), traces).sorted(), strings(removal.getJSONArray("left_out")))
                check("games after", store.games(200, finished = false).map { it.gameId }.sorted(), strings(removal.getJSONArray("games_after")))
                check("library after", store.library().map { it.key }.sorted(), strings(removal.getJSONArray("library_after")))
                if (keys.isNotEmpty()) {
                    check("restored", store.restoreGame(keys[0]), strings(removal.getJSONArray("restored")))
                    check("games restored", store.games(200, finished = false).map { it.gameId }.sorted(), strings(removal.getJSONArray("games_restored")))
                    check("library restored", store.library().map { it.key }.sorted(), strings(removal.getJSONArray("library_restored")))
                    val wantTwins = removal.getJSONObject("twins_restored").let { t -> t.keys().asSequence().associateWith { t.getString(it) } }
                    check("twins restored", store.twins(), wantTwins)
                }
            } finally {
                store.close()
            }
        }
        assertTrue("$asked removals; ${failures.size} failures:\n" + failures.take(20).joinToString("\n"), failures.isEmpty())
    }

    /** A site's answers by URL (the query aside), every URL asked kept, as the fixture's `Site`. */
    private class Site(private val answers: JSONObject) : Transport {
        val asked = ArrayList<String>()
        val queries = ArrayList<String>()
        val headers = ArrayList<Map<String, String>>()

        override fun get(url: String, headers: Map<String, String>, timeoutMs: Int): HttpReply {
            val bare = url.substringBefore("?")
            asked.add(bare)
            queries.add(url.substringAfter("?", ""))
            this.headers.add(headers)
            val answer = answers.optJSONArray(bare) ?: return HttpReply(404, "{}")
            return HttpReply(answer.getInt(0), answer.getString(1))
        }

        override fun post(url: String, headers: Map<String, String>, body: String, timeoutMs: Int): HttpReply =
            throw UnsupportedOperationException()

        override fun postLines(
            url: String,
            headers: Map<String, String>,
            body: String,
            timeoutMs: Int,
            onLine: (String) -> Boolean,
        ): HttpReply = throw UnsupportedOperationException()

        operator fun invoke(case: JSONObject, failures: MutableList<String>, fetch: () -> List<LibraryRow>) {
            val name = case.getString("name")
            try {
                val rows = fetch()
                val want = case.optJSONArray("rows")
                if (want == null) failures.add("$name: rows, python raises")
                else if (rows.size != want.length()) failures.add("$name: ${rows.size} rows, python ${want.length()}")
                else for (r in rows.indices) compare("$name #$r", rows[r], want.getJSONObject(r), failures)
            } catch (e: Library.NoPlayer) {
                if (case.optString("no_player") != e.message) failures.add("$name: no player '${e.message}', python ${case.opt("no_player")}")
            } catch (e: Library.HttpFailure) {
                if (!case.optBoolean("http_error")) failures.add("$name: http failure ${e.message}")
            }
            val want = strings(case.getJSONArray("asked"))
            if (asked != want) failures.add("$name: asked $asked\n  python: $want")
        }
    }

    companion object {
        /**
         * The 10 MB fixture, read once for the class (each of its seven tests read it again) and
         * let go after it: the suite's other classes share the 192 MB heap.
         */
        private var read: JSONObject? = null

        private val fixture: JSONObject
            get() = read ?: InstrumentationRegistry.getInstrumentation().context.assets.open("profile/library.json")
                .bufferedReader().use { JSONObject(it.readText()) }.also { read = it }

        @JvmStatic
        @AfterClass
        fun letGo() {
            read = null
        }

        fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.get(it).toString() }

        private fun optString(o: JSONObject, name: String): String? = if (o.isNull(name)) null else o.getString(name)
        private fun optDouble(o: JSONObject, name: String): Double? = if (o.isNull(name)) null else o.getDouble(name)
        private fun optInt(o: JSONObject, name: String): Int? = if (o.isNull(name)) null else o.getInt(name)

        fun compare(label: String, row: LibraryRow, want: JSONObject, failures: MutableList<String>) {
            val mine = mapOf(
                "key" to row.key, "source" to row.source, "pgn" to row.pgn, "played_at" to row.playedAt,
                "white" to row.white, "black" to row.black, "white_elo" to row.whiteElo, "black_elo" to row.blackElo,
                "result" to row.result, "time_class" to row.timeClass, "user_side" to row.userSide,
            )
            for ((field, value) in mine) {
                val theirs: Any? = when (field) {
                    "played_at" -> optDouble(want, field)
                    "white_elo", "black_elo" -> optInt(want, field)
                    else -> optString(want, field)
                }
                if (value != theirs) failures.add("$label: $field\n  mine:   $value\n  python: $theirs")
            }
        }

        fun gameRow(o: JSONObject) = GameRow(
            gameId = o.getString("game_id"),
            sessionId = optString(o, "session_id"),
            startedAt = o.getDouble("started_at"),
            endedAt = optDouble(o, "ended_at"),
            userSide = o.getString("user_side"),
            startFen = o.getString("start_fen"),
            timeControl = optString(o, "time_control"),
            opponent = optString(o, "opponent"),
            result = optString(o, "result"),
            reason = optString(o, "reason"),
            opening = optString(o, "opening"),
            accuracy = optDouble(o, "accuracy"),
            finalScore = optDouble(o, "final_score"),
            pgn = optString(o, "pgn"),
            finalFen = optString(o, "final_fen"),
        )

        fun moveRow(o: JSONObject) = MoveRow(
            moveId = o.getLong("move_id"), gameId = o.getString("game_id"), segment = o.getInt("segment"),
            ply = o.getInt("ply"), moveNumber = o.getInt("move_number"), san = o.getString("san"),
            uci = o.getString("uci"), byUser = o.getBoolean("by_user"), fenBefore = o.getString("fen_before"),
            fenAfter = o.getString("fen_after"), phase = null, thinkMs = null, clockMs = null, capture = false,
            givesCheck = false, material = null, ts = o.getDouble("ts"), label = null, severity = null,
            expectedLoss = null, scoreAfter = null, bestUci = null, bestSan = null, accuracy = null,
            tags = emptyList(), availableTags = emptyList(), allowedTags = emptyList(), depth = null, withdrawn = false,
        )

        fun libraryRow(o: JSONObject) = LibraryRow(
            key = o.getString("key"), source = o.getString("source"), addedAt = o.getDouble("added_at"),
            pgn = o.getString("pgn"), playedAt = optDouble(o, "played_at"), white = optString(o, "white"),
            black = optString(o, "black"), whiteElo = optInt(o, "white_elo"), blackElo = optInt(o, "black_elo"),
            result = optString(o, "result"), timeClass = optString(o, "time_class"), userSide = optString(o, "user_side"),
        )
    }
}
