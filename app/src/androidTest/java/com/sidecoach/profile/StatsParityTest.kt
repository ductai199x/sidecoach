package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.events.MotifTag
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.ui.StatsApi
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Stats as the desktop's (profile/StatFacts.kt, profile/Stats.kt, the Store's Stats section,
 * ui/StatsApi.kt): `tools/make_stats_fixtures.py` builds a desktop store op by op -- library rows,
 * the seven real reviews of `reviews.json` kept under many rows and headers, random facts saved
 * whole, twins, reviews, the puzzle level -- and records what it then holds, the `/api/stats`
 * body under many filters, the `/api/stats/positions` body for every selector, and the parts
 * (`kind_of`, `chance_in`, the boards' structure, themes and endgame, `bot_game`, `grade`,
 * `fmean`...). Here the same ops build a phone store and everything is held to the desktop's,
 * numbers by value and objects in the desktop's key order. The puzzles an area to work on gets
 * are a random draw on both, so they are held to the area's pool instead.
 */
@RunWith(AndroidJUnit4::class)
class StatsParityTest {

    @Test
    fun theKeptFactsAreTheDesktops() {
        val rows = fixture.getJSONObject("rows")
        val failures = ArrayList<String>()
        val store = Store(File(files, "profile.sqlite"))
        try {
            same("stat_games", JSONArray(store.statGames().map { json(it) }), rows.getJSONArray("stat_games"), failures)
            val keys = strings(rows.getJSONArray("keys"))
            same("stat_moves", JSONArray(store.statMoves(keys).map { json(it) }), rows.getJSONArray("stat_moves"), failures)
            val pending = store.reviewsWithoutStats().map { json(it) }.sortedBy { it.getString("key") + "|" + it.getString("strength") }
            val want = rows.getJSONArray("reviews_without_stats").let { a -> (0 until a.length()).map { a.getJSONObject(it) } }
                .sortedBy { it.getString("key") + "|" + it.getString("strength") }
            same("reviews_without_stats", JSONArray(pending), JSONArray(want), failures)
        } finally {
            store.close()
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theStatsAreTheDesktops() {
        val api = StatsApi(files)
        val now = fixture.getDouble("now")
        val cases = fixture.getJSONArray("stats")
        val failures = ArrayList<String>()
        var areas = 0
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val query = query(case.getJSONObject("query"))
            val mine = api.stats(query, now)
            val want = case.getJSONObject("body")
            same("$query", mine, want, failures, skip = setOf("puzzles"))
            // The puzzles: as many as the desktop drew, each from the area's pool.
            val work = mine.getJSONObject("stats").getJSONArray("work")
            val theirs = want.getJSONObject("stats").getJSONArray("work")
            val pools = case.getJSONArray("pools")
            for (j in 0 until minOf(work.length(), theirs.length())) {
                areas += 1
                val picked = work.getJSONObject(j).getJSONArray("puzzles")
                val pool = pools.getJSONObject(j)
                val wanted = theirs.getJSONObject(j).getJSONArray("puzzles").length()
                if (picked.length() != wanted) failures.add("$query work[$j].puzzles: ${picked.length()}, python $wanted")
                val ids = HashSet<String>()
                for (k in 0 until picked.length()) {
                    val p = picked.getJSONObject(k)
                    val id = p.getString("id")
                    if (!ids.add(id)) failures.add("$query work[$j].puzzles: $id twice")
                    if (!pool.has(id) || pool.getInt(id) != p.getInt("rating")) failures.add("$query work[$j].puzzles: $p not in the pool")
                }
            }
        }
        assertTrue("${cases.length()} filters, $areas areas; ${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun thePositionsAreTheDesktops() {
        val api = StatsApi(files)
        val now = fixture.getDouble("now")
        val cases = fixture.getJSONArray("positions")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val query = query(case.getJSONObject("query"))
            same("$query", api.positions(query, now), case.getJSONObject("body"), failures)
        }
        assertTrue("${cases.length()} queries; ${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun thePartsAreTheDesktops() {
        val units = fixture.getJSONObject("units")
        val failures = ArrayList<String>()
        fun check(name: String, mine: Any?, theirs: Any?) {
            val a = if (theirs == JSONObject.NULL) null else theirs
            val equal = if (mine is Number && a is Number) mine.toDouble() == a.toDouble() else mine == a
            if (!equal) failures.add("$name: mine $mine, python $a")
        }
        for ((args, want) in pairs(units.getJSONArray("kind_of"))) {
            val names = strings(args as JSONArray)
            check("kind_of $names", StatFacts.kindOf(names.map { MotifTag(it, listOf("e4")) }), want)
        }
        for ((args, want) in pairs(units.getJSONArray("chance_in"))) {
            val analysis = args as JSONObject
            check("chance_in ${analysis.getString("fen")} ${analysis.getJSONArray("tags")}", StatFacts.chanceIn(ReviewJson.analysis(analysis)), want)
        }
        val boards = units.getJSONArray("boards")
        for (i in 0 until boards.length()) {
            val case = boards.getJSONArray(i)
            val fen = case.getString(0)
            val board = Board(fen)
            check("structure_of $fen", StatFacts.structureOf(board), case.getString(1))
            check("themes_of $fen", StatFacts.themesOf(board), strings(case.getJSONArray(2)))
            check("endgame_of $fen", StatFacts.endgameOf(board), case.getString(3))
        }
        for ((score, want) in pairs(units.getJSONArray("standing"))) {
            check("standing $score", StatFacts.standing((score as Number).toDouble()), want)
        }
        for ((headers, want) in pairs(units.getJSONArray("bot_game"))) {
            val h = headers as JSONObject
            check("bot_game $h", StatFacts.botGame(h.keys().asSequence().associateWith { h.getString(it) }), want)
        }
        for ((opening, want) in pairs(units.getJSONArray("family"))) {
            check("family $opening", Stats.family(opening as? String), want)
        }
        for ((text, want) in pairs(units.getJSONArray("termination_kind"))) {
            check("termination_kind $text", Stats.terminationKind(text as? String), want)
        }
        val grades = units.getJSONArray("grade")
        for (i in 0 until grades.length()) {
            val g = grades.getJSONArray(i)
            val l = g.getJSONObject(2)
            val line = Stats.PeerLine(l.getDouble("a"), l.getDouble("b"), l.getDouble("r2"), l.getDouble("lo"), l.getDouble("hi"))
            val (peer, kind) = Stats.grade(g.getDouble(0), g.getInt(1), line, g.getDouble(3))
            check("grade $g peer", peer, g.getDouble(4))
            check("grade $g kind", kind, g.get(5))
        }
        for ((rating, want) in pairs(units.getJSONArray("lichess"))) {
            check("lichess $rating", Performance.lichess((rating as Number).toDouble()), want)
        }
        for ((values, want) in pairs(units.getJSONArray("fmean"))) {
            val list = (values as JSONArray).let { v -> (0 until v.length()).map { v.getDouble(it) } }
            check("fmean $list", Stats.fmean(list), want)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }

    companion object {
        private val instrumentation = InstrumentationRegistry.getInstrumentation()

        private fun asset(name: String): String =
            instrumentation.context.assets.open(name).bufferedReader(Charsets.UTF_8).use { it.readText() }

        private val fixture: JSONObject by lazy { JSONObject(asset("profile/stats.json")) }

        private val reviews: List<ReviewedGame> by lazy {
            val games = JSONObject(asset("profile/reviews.json")).getJSONArray("games")
            (0 until games.length()).map { ReviewJson.reviewed(games.getJSONObject(it).getJSONObject("reviewed")) }
        }

        /**
         * The app's files as the desktop's data folder was: the profile store built op by op as
         * the desktop's was, and the puzzles of `puzzles.csv` to import. Built once for the class.
         */
        private val files: File by lazy {
            val dir = File(instrumentation.targetContext.cacheDir, "stats-parity").apply {
                deleteRecursively()
                mkdirs()
            }
            File(dir, "puzzles").mkdirs()
            instrumentation.context.assets.open("profile/puzzles.csv").use { input ->
                File(dir, "puzzles/puzzles.csv").outputStream().use { input.copyTo(it) }
            }
            val store = Store(File(dir, "profile.sqlite"))
            try {
                val ops = fixture.getJSONArray("ops")
                for (i in 0 until ops.length()) replay(store, ops.getJSONObject(i))
            } finally {
                store.close()
            }
            dir
        }

        private fun replay(store: Store, op: JSONObject) {
            when (val kind = op.getString("op")) {
                "library" -> store.addLibraryGame(libraryRow(op.getJSONObject("row")))
                "keep" -> {
                    val base = reviews[op.getInt("review")]
                    val reviewed = if (op.isNull("headers")) base else {
                        val h = op.getJSONObject("headers")
                        ReviewedGame(
                            headers = h.keys().asSequence().associateWithTo(LinkedHashMap()) { h.getString(it) },
                            start = base.start,
                            moves = base.moves,
                            strength = base.strength,
                            depth = base.depth,
                            seconds = base.seconds,
                        )
                    }
                    StatFacts.keepStats(store, op.getString("strength"), reviewed, libraryRow(op.getJSONObject("row")))
                }
                "save" -> {
                    val moves = op.getJSONArray("moves")
                    store.saveStats(op.getString("strength"), gameFact(op.getJSONObject("game")), (0 until moves.length()).map { moveFact(moves.getJSONObject(it)) })
                }
                "twin" -> store.addTwin(op.getString("watched"), op.getString("imported"))
                "review" -> store.saveReview(
                    op.getString("key"),
                    op.getString("strength"),
                    op.getInt("version"),
                    op.getDouble("reviewed_at"),
                    op.getString("report"),
                    str(op, "summary"),
                )
                "setting" -> store.setSetting(op.getString("key"), op.getString("value"))
                else -> throw AssertionError("no op $kind")
            }
        }

        private fun str(o: JSONObject, k: String): String? = if (o.isNull(k)) null else o.getString(k)
        private fun int(o: JSONObject, k: String): Int? = if (o.isNull(k)) null else o.getInt(k)
        private fun dbl(o: JSONObject, k: String): Double? = if (o.isNull(k)) null else o.getDouble(k)
        private fun nul(value: Any?): Any = value ?: JSONObject.NULL

        private fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

        private fun pairs(a: JSONArray): List<Pair<Any?, Any?>> =
            (0 until a.length()).map { a.getJSONArray(it) }.map { (if (it.isNull(0)) null else it.get(0)) to it.get(1) }

        private fun query(o: JSONObject): Map<String, String> = o.keys().asSequence().associateWith { o.getString(it) }

        private fun libraryRow(o: JSONObject) = LibraryRow(
            key = o.getString("key"),
            source = o.getString("source"),
            addedAt = o.getDouble("added_at"),
            pgn = o.getString("pgn"),
            playedAt = dbl(o, "played_at"),
            white = str(o, "white"),
            black = str(o, "black"),
            whiteElo = int(o, "white_elo"),
            blackElo = int(o, "black_elo"),
            result = str(o, "result"),
            timeClass = str(o, "time_class"),
            userSide = str(o, "user_side"),
        )

        private fun gameFact(o: JSONObject) = GameFact(
            key = o.getString("key"),
            source = o.getString("source"),
            timeClass = str(o, "time_class"),
            playedAt = dbl(o, "played_at"),
            userSide = str(o, "user_side"),
            bot = o.getBoolean("bot"),
            result = str(o, "result"),
            opening = str(o, "opening"),
            termination = str(o, "termination"),
        )

        private fun moveFact(o: JSONObject) = MoveFact(
            ply = o.getInt("ply"),
            side = o.getString("side"),
            fen = o.getString("fen"),
            uci = o.getString("uci"),
            san = o.getString("san"),
            phase = o.getString("phase"),
            loss = o.getDouble("loss"),
            label = str(o, "label"),
            accuracy = dbl(o, "accuracy"),
            score = o.getDouble("score"),
            chance = str(o, "chance"),
            found = o.getBoolean("found"),
            allowed = str(o, "allowed"),
            structure = o.getString("structure"),
            themes = o.getString("themes"),
            endgame = str(o, "endgame"),
            spentMs = int(o, "spent_ms"),
            clockMs = int(o, "clock_ms"),
        )

        /** `stat_games()`'s row: the table's columns in order, `bot` a truth. */
        private fun json(g: StatGameRow): JSONObject = JSONObject()
            .put("key", g.key)
            .put("strength", g.strength)
            .put("source", nul(g.source))
            .put("time_class", nul(g.timeClass))
            .put("played_at", nul(g.playedAt))
            .put("user_side", nul(g.userSide))
            .put("bot", g.bot)
            .put("result", nul(g.result))
            .put("opening", nul(g.opening))
            .put("termination", nul(g.termination))

        /** `stat_moves()`'s row: the table's columns in order, `found` a truth. */
        private fun json(m: StatMoveRow): JSONObject = JSONObject()
            .put("key", m.key)
            .put("ply", m.ply)
            .put("side", m.side)
            .put("fen", m.fen)
            .put("uci", m.uci)
            .put("san", m.san)
            .put("phase", m.phase)
            .put("loss", m.loss)
            .put("label", nul(m.label))
            .put("accuracy", nul(m.accuracy))
            .put("score", m.score)
            .put("chance", nul(m.chance))
            .put("found", m.found)
            .put("allowed", nul(m.allowed))
            .put("structure", m.structure)
            .put("themes", m.themes)
            .put("endgame", nul(m.endgame))
            .put("spent_ms", nul(m.spentMs))
            .put("clock_ms", nul(m.clockMs))

        private fun json(r: ReviewRow): JSONObject = JSONObject()
            .put("key", r.key)
            .put("strength", r.strength)
            .put("version", r.version)
            .put("reviewed_at", r.reviewedAt)
            .put("report", r.report)
            .put("summary", nul(r.summary))
            .put("words", nul(r.words))

        /**
         * The two JSON values the same: numbers by value, an object's keys in the desktop's order
         * (the page reads some in order: the results, the sections), [skip]'s keys left out.
         */
        fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>, skip: Set<String> = emptySet()) {
            if (failures.size >= 40) return
            val a = if (mine == JSONObject.NULL) null else mine
            val b = if (theirs == JSONObject.NULL) null else theirs
            when {
                a is JSONObject && b is JSONObject -> {
                    val mineKeys = a.keys().asSequence().filter { it !in skip }.toList()
                    val theirKeys = b.keys().asSequence().filter { it !in skip }.toList()
                    if (mineKeys.toSet() == theirKeys.toSet() && mineKeys != theirKeys) {
                        failures.add("$path: keys in the order $mineKeys, python $theirKeys")
                    }
                    for (k in (mineKeys + theirKeys).distinct()) {
                        if (!a.has(k) || !b.has(k)) {
                            failures.add("$path.$k: ${if (a.has(k)) "only mine" else "only python's"}")
                        } else {
                            same("$path.$k", a.get(k), b.get(k), failures, skip)
                        }
                    }
                }
                a is JSONArray && b is JSONArray -> {
                    if (a.length() != b.length()) failures.add("$path: ${a.length()} items, python ${b.length()}\n  mine:   ${a.toString().take(600)}\n  python: ${b.toString().take(600)}")
                    else for (i in 0 until a.length()) same("$path[$i]", a.get(i), b.get(i), failures, skip)
                }
                a is Number && b is Number -> if (a.toDouble() != b.toDouble()) failures.add("$path: mine $a, python $b")
                else -> if (a != b) failures.add("$path: mine $a, python $b")
            }
        }
    }
}
