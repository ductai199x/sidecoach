package com.sidecoach.profile

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.events.Analysis
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.ProfileSnapshot
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.events.Recommendation
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.events.UserUtterance
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The profile stage held to `sidecoach/profile/`: four real games' events (judged by the
 * desktop's annotator on the real Stockfish, with the tagger and the book) replayed into the
 * Kotlin profiler, and everything compared with what the Python did
 * (`tools/make_profile_fixtures.py`) -- every event it emitted to every event it was fed, the
 * session's summary, the next session's snapshot, every row of every table, the taxonomy's
 * cells; and on the ask side the system message, the SQL found in replies, the nine worked
 * queries run and rendered, the queries refused, and two conversations with a scripted model.
 */
@RunWith(AndroidJUnit4::class)
class ProfileTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    private fun fixture(): JSONObject {
        val text = instrumentation.context.assets.open("profile/profile.json").bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    private fun asset(name: String, into: File): File {
        instrumentation.context.assets.open(name).use { input -> into.outputStream().use { input.copyTo(it) } }
        return into
    }

    @Test
    fun theStageMatchesThePythonEventForEventRowForRow() {
        val data = fixture()
        val dir = File(instrumentation.targetContext.cacheDir, "profile-test").apply { deleteRecursively(); mkdirs() }
        val puzzles = PuzzleStore(null)
        val imported = puzzles.importCsv(asset("profile/puzzles.csv", File(dir, "puzzles.csv")))
        assertEquals("puzzles imported", data.getInt("puzzles_imported"), imported)

        val dbFile = File(dir, "profile.sqlite")
        val store = Store(dbFile)
        val profiler = Profiler(store, sessionId = data.getString("session"), ts = 1000.0, puzzles = puzzles, shuffle = {})
        same("first snapshot", data.getJSONObject("first_snapshot"), dump(profiler.snapshot()))

        val steps = data.getJSONArray("steps")
        var ts = 1000.0
        for (i in 0 until steps.length()) {
            val step = steps.getJSONObject(i)
            ts = step.getDouble("ts")
            val out = if (step.has("call")) {
                when (step.getString("call")) {
                    "tick" -> profiler.tick(ts)
                    "flush" -> profiler.flush(ts)
                    else -> error("unknown call")
                }
            } else {
                profiler.onEvent(event(step.getJSONObject("event")), ts)
            }
            val where = "step $i ${step.optJSONObject("event")?.optString("type") ?: step.optString("call")}"
            same(where, step.getJSONArray("out"), JSONArray(out.map { dump(it) }))
        }

        val summary = profiler.endSession(ts + 5.0)
        assertEquals("the session's summary", data.optString("summary").ifEmpty { null }, summary)
        val second = Profiler(store, sessionId = "session-two", ts = ts + 10.0, puzzles = puzzles, shuffle = {})
        same("the next session's snapshot", data.getJSONObject("second_snapshot"), dump(second.snapshot()))
        same("the cells", data.getJSONArray("cells"), JSONArray(store.cells().map { cell(it) }))

        // Every row of every table, as the Python wrote them.
        val tables = data.getJSONObject("tables")
        val raw = SQLiteDatabase.openDatabase(dbFile.path, null, SQLiteDatabase.OPEN_READONLY)
        raw.use { db ->
            for (table in tables.keys()) {
                val want = tables.getJSONObject(table)
                db.rawQuery("SELECT * FROM $table ORDER BY rowid", null).use { c ->
                    same("$table columns", want.getJSONArray("columns"), JSONArray(c.columnNames.toList()))
                    val rows = JSONArray()
                    while (c.moveToNext()) rows.put(JSONArray((0 until c.columnCount).map { value(c, it) }))
                    same("$table rows", want.getJSONArray("rows"), rows)
                }
            }
        }

        // The ask side.
        assertEquals("the system message", data.getString("system"), Ask.systemMessage())
        val extracts = data.getJSONArray("extracts")
        for (i in 0 until extracts.length()) {
            val e = extracts.getJSONObject(i)
            assertEquals("extract $i", if (e.isNull("sql")) null else e.getString("sql"), Ask.extractSql(e.getString("text")))
        }
        val queries = data.getJSONArray("queries")
        for (i in 0 until queries.length()) {
            val q = queries.getJSONObject(i)
            val (columns, rows) = store.query(q.getString("sql"))
            same("query $i columns", q.getJSONArray("columns"), JSONArray(columns))
            same("query $i rows", q.getJSONArray("rows"), JSONArray(rows.map { JSONArray(it) }))
            assertEquals("query $i rendered: ${q.getString("question")}", q.getString("rendered"), Ask.renderRows(columns, rows))
        }
        // What is refused, and in the words the model reads ("Query refused: {exc}."): the
        // authorizer's "not authorized" for a write inside a WITH and the pragmas' functions,
        // SQLite's own message for the rest (parity pv-02).
        val refused = data.getJSONArray("refused")
        for (i in 0 until refused.length()) {
            val r = refused.getJSONObject(i)
            val error = try {
                store.query(r.getString("sql"))
                null
            } catch (exc: QueryRefused) {
                exc.message
            }
            assertEquals("refused: '${r.getString("sql")}'", r.getBoolean("refused"), error != null)
            assertEquals("refused: '${r.getString("sql")}' says", if (r.isNull("error")) null else r.getString("error"), error)
        }
        // Still writable after a refusal: the read-only switch was put back.
        store.setSetting("after", "refusals")
        assertEquals("refusals", store.setting("after"))
        // A runaway query is interrupted, and the row cap holds.
        val bounded = data.getJSONObject("bounded")
        val runaway = try {
            store.query(bounded.getString("runaway"), seconds = 0.2)
            null
        } catch (exc: QueryRefused) {
            exc.message
        }
        assertEquals("the runaway query", bounded.getString("runaway_error"), runaway)
        assertEquals("the row cap", bounded.getInt("capped_rows"), store.query(bounded.getString("capped"), maxRows = 10).second.size)
        // A query with no LIMIT stops at the cap, its names its own (the parity audit, 2026-09-29).
        val (columns, rows) = store.query(bounded.getString("unbounded"), maxRows = 10)
        assertEquals("unbounded names", bounded.getJSONArray("unbounded_columns").toString(), org.json.JSONArray(columns).toString())
        assertEquals("unbounded rows", bounded.getJSONArray("unbounded_rows").toString(), org.json.JSONArray(rows.map { org.json.JSONArray(it) }).toString())
        // Values as Python's str() shows them in the rows: infinities, wide floats, blobs, a cut text.
        val renders = data.getJSONArray("renders")
        for (i in 0 until renders.length()) {
            val r = renders.getJSONObject(i)
            val (columns, rows) = store.query(r.getString("sql"))
            assertEquals("rendered: ${r.getString("sql")}", r.getString("rendered"), Ask.renderRows(columns, rows))
        }
        val cellValues = data.getJSONArray("cell_values")
        for (i in 0 until cellValues.length()) {
            val c = cellValues.getJSONObject(i)
            val value = when (val v = c.getString("value")) {
                "inf" -> Double.POSITIVE_INFINITY
                "-inf" -> Double.NEGATIVE_INFINITY
                "nan" -> Double.NaN
                else -> v.toDouble()
            }
            assertEquals("the cell of ${c.getString("value")}", c.getString("cell"), Ask.cell(value))
        }

        val conversations = data.getJSONArray("conversations")
        for (i in 0 until conversations.length()) {
            val want = conversations.getJSONObject(i)
            val provider = Scripted(strings(want.getJSONArray("replies")))
            val answer = Ask.answer("How am I doing?", store, provider)
            assertEquals("conversation $i text", want.getString("text"), answer.text)
            assertEquals("conversation $i calls", want.getInt("calls"), answer.calls)
            assertEquals("conversation $i failed", want.getBoolean("failed"), answer.failed)
            val exchanges = JSONArray(answer.exchanges.map { e ->
                e as Ask.Exchange
                JSONObject().put("sql", e.sql).put("columns", JSONArray(e.columns))
                    .put("rows", JSONArray(e.rows.map { JSONArray(it) })).put("refused", e.error != null)
            })
            // A refused query's rows and columns are empty on both sides.
            same("conversation $i exchanges", want.getJSONArray("exchanges"), exchanges)
            val seen = JSONArray(provider.seen.map { messages ->
                JSONArray(messages.map { JSONObject().put("role", it.role).put("content", it.content) })
            })
            sameMessages("conversation $i messages", want.getJSONArray("messages"), seen)
        }
        store.close()
    }

    // ---- the events in -------------------------------------------------------------------------

    private fun event(e: JSONObject): Event = when (val type = e.getString("type")) {
        "GameStarted" -> GameStarted(
            gameId = e.getString("game_id"),
            userSide = side(e.getString("user_side")),
            startFen = e.getString("start_fen"),
            kind = e.getString("kind"),
        )
        "ClockObservation" -> ClockObservation(
            whiteMs = e.optIntOrNull("white_ms"),
            blackMs = e.optIntOrNull("black_ms"),
            running = if (e.isNull("running")) null else side(e.getString("running")),
        )
        "MovePlayed" -> MovePlayed(
            gameId = e.getString("game_id"),
            ply = e.getInt("ply"),
            san = e.getString("san"),
            uci = e.getString("uci"),
            fenBefore = e.getString("fen_before"),
            fenAfter = e.getString("fen_after"),
            byUser = e.getBoolean("by_user"),
            thinkMs = e.optIntOrNull("think_ms"),
        )
        "MoveAssessment" -> MoveAssessment(
            gameId = e.getString("game_id"),
            ply = e.getInt("ply"),
            uci = e.getString("uci"),
            expectedLoss = e.getDouble("expected_loss"),
            severity = Severity.valueOf(e.getString("severity").uppercase()),
            best = strings(e.getJSONArray("best")),
            tags = tags(e.getJSONArray("tags")),
            depth = e.getInt("depth"),
            label = if (e.isNull("label")) null else Label.valueOf(e.getString("label").uppercase()),
            accuracy = if (e.isNull("accuracy")) null else e.getDouble("accuracy"),
        )
        "Analysis" -> Analysis(
            gameId = e.getString("game_id"),
            fen = e.getString("fen"),
            ply = e.getInt("ply"),
            lines = (0 until e.getJSONArray("lines").length()).map { i ->
                val l = e.getJSONArray("lines").getJSONObject(i)
                val wdl = l.optJSONArray("wdl")
                Line(
                    moves = strings(l.getJSONArray("moves")),
                    depth = l.getInt("depth"),
                    expectedScore = l.getDouble("expected_score"),
                    scoreCp = l.optIntOrNull("score_cp"),
                    mate = l.optIntOrNull("mate"),
                    wdl = if (wdl == null) null else Triple(wdl.getInt(0), wdl.getInt(1), wdl.getInt(2)),
                )
            },
            decisive = e.getBoolean("decisive"),
            onlyMove = e.getBoolean("only_move"),
            threats = strings(e.getJSONArray("threats")),
            tags = tags(e.getJSONArray("tags")),
            opening = if (e.isNull("opening")) null else e.getString("opening"),
            provisional = e.optBoolean("provisional"),
        )
        "CoachMessage" -> CoachMessage(
            text = e.getString("text"),
            level = DisclosureLevel.entries.first { it.value == e.getInt("level") },
            trigger = Trigger.entries.first { it.slug == e.getString("trigger") },
            gameId = if (e.isNull("game_id")) null else e.getString("game_id"),
            ply = e.optIntOrNull("ply"),
        )
        "UserUtterance" -> UserUtterance(e.getString("text"), e.getString("source"))
        "Takeback" -> Takeback(e.getString("game_id"), e.getInt("plies"), e.getString("fen"))
        "Resync" -> Resync(e.getString("game_id"), e.getString("fen"), e.getString("reason"))
        "GameEnded" -> GameEnded(
            gameId = e.getString("game_id"),
            result = e.getString("result"),
            reason = if (e.isNull("reason")) null else e.getString("reason"),
            pgn = if (e.isNull("pgn")) null else e.getString("pgn"),
            finalFen = if (e.isNull("final_fen")) null else e.getString("final_fen"),
        )
        else -> error("unknown event $type")
    }

    // ---- the events out, as the Python dumps them -----------------------------------------------

    private fun dump(event: Event): JSONObject = when (event) {
        is ProfileEvidence -> JSONObject()
            .put("game_id", event.gameId).put("ply", event.ply).put("phase", event.phase)
            .put("expected_loss", event.expectedLoss).put("motif", event.motif ?: JSONObject.NULL)
            .put("cause", event.cause ?: JSONObject.NULL).put("time_pressure", event.timePressure)
            .put("type", "ProfileEvidence")
        is GameReview -> JSONObject()
            .put("game_id", event.gameId).put("user_side", event.userSide.name.lowercase())
            .put("result", event.result).put("reason", event.reason ?: JSONObject.NULL)
            .put("opening", event.opening ?: JSONObject.NULL).put("moves", event.moves)
            .put("accuracy", event.accuracy ?: JSONObject.NULL)
            .put("tallies", JSONObject(event.tallies as Map<*, *>))
            .put("turning_points", JSONArray(event.turningPoints.map { tp ->
                JSONObject().put("ply", tp.ply).put("san", tp.san).put("label", tp.label)
                    .put("expected_loss", tp.expectedLoss).put("segment", tp.segment)
                    .put("best_san", tp.bestSan ?: JSONObject.NULL).put("cause", tp.cause ?: JSONObject.NULL)
                    .put("motif", tp.motif ?: JSONObject.NULL).put("phase", tp.phase)
                    .put("fen_before", tp.fenBefore ?: JSONObject.NULL)
            }))
            .put("phases", JSONArray(event.phases.map {
                JSONObject().put("phase", it.phase).put("moves", it.moves)
                    .put("expected_loss", it.expectedLoss).put("mistakes", it.mistakes)
            }))
            .put("causes", JSONObject(event.causes as Map<*, *>))
            .put("lines", JSONArray(event.lines))
            .put("type", "GameReview")
        is Recommendation -> JSONObject()
            .put("game_id", event.gameId).put("motif", event.motif).put("motif_words", event.motifWords)
            .put("reason", event.reason).put("level", event.level)
            .put("puzzles", JSONArray(event.puzzles.map {
                JSONObject().put("id", it.id).put("url", it.url).put("rating", it.rating).put("themes", JSONArray(it.themes))
            }))
            .put("type", "Recommendation")
        is PuzzleOutcome -> JSONObject()
            .put("game_id", event.gameId).put("puzzle_id", event.puzzleId).put("motif", event.motif ?: JSONObject.NULL)
            .put("solved", event.solved).put("level", event.level).put("type", "PuzzleOutcome")
        is ProfileSnapshot -> JSONObject()
            .put("games", event.games)
            .put("cells", JSONArray(event.cells.map {
                JSONObject().put("cause", it.cause ?: JSONObject.NULL).put("motif", it.motif ?: JSONObject.NULL)
                    .put("phase", it.phase ?: JSONObject.NULL).put("errors", it.errors).put("chances", it.chances)
                    .put("expected_loss", it.expectedLoss)
            }))
            .put("insights", JSONArray(event.insights)).put("last_session", event.lastSession ?: JSONObject.NULL)
            .put("type", "ProfileSnapshot")
        else -> JSONObject().put("type", event::class.java.simpleName)
    }

    private fun cell(c: CellRow): JSONObject = JSONObject()
        .put("cause", c.cause ?: JSONObject.NULL).put("motif", c.motif ?: JSONObject.NULL)
        .put("phase", c.phase ?: JSONObject.NULL).put("errors", c.errors).put("chances", c.chances)
        .put("expected_loss", c.expectedLoss)

    // ---- comparison ------------------------------------------------------------------------------

    /** Equal as JSON values, numbers to within 1e-9 (an int and a float of it are equal). */
    private fun same(where: String, want: Any?, got: Any?) {
        when {
            want == null || want == JSONObject.NULL -> if (!(got == null || got == JSONObject.NULL)) fail("$where: want null, got $got")
            want is JSONObject -> {
                if (got !is JSONObject) fail("$where: want object, got $got")
                got as JSONObject
                val keys = want.keys().asSequence().toSet()
                assertEquals("$where keys", keys, got.keys().asSequence().toSet())
                for (k in keys) same("$where.$k", want.opt(k), got.opt(k))
            }
            want is JSONArray -> {
                if (got !is JSONArray) fail("$where: want array $want, got $got")
                got as JSONArray
                assertEquals("$where length ($want vs $got)", want.length(), got.length())
                for (i in 0 until want.length()) same("$where[$i]", want.opt(i), got.opt(i))
            }
            want is Number -> {
                if (got !is Number) fail("$where: want $want, got $got")
                assertEquals(where, want.toDouble(), (got as Number).toDouble(), 1e-9)
            }
            else -> assertEquals(where, want, got)
        }
    }

    /** Messages compared as text, a refused query's words included (parity pv-02). */
    private fun sameMessages(where: String, want: JSONArray, got: JSONArray) {
        assertEquals("$where calls", want.length(), got.length())
        for (i in 0 until want.length()) {
            val w = want.getJSONArray(i)
            val g = got.getJSONArray(i)
            assertEquals("$where[$i] length", w.length(), g.length())
            for (j in 0 until w.length()) {
                val wc = w.getJSONObject(j).getString("content")
                val gc = g.getJSONObject(j).getString("content")
                assertEquals("$where[$i][$j]", wc, gc)
                assertEquals("$where[$i][$j] role", w.getJSONObject(j).getString("role"), g.getJSONObject(j).getString("role"))
            }
        }
    }

    private fun value(c: Cursor, i: Int): Any? = when (c.getType(i)) {
        Cursor.FIELD_TYPE_NULL -> JSONObject.NULL
        Cursor.FIELD_TYPE_INTEGER -> c.getLong(i)
        Cursor.FIELD_TYPE_FLOAT -> c.getDouble(i)
        else -> c.getString(i)
    }

    /**
     * `tests/test_profile.py`'s profiler tests as the desktop ran them: every call on a profiler
     * (the events in and what came out, a tick, a flush, a correction, the session's end, the
     * snapshot) replayed here, and every table of every store compared at the end -- the review
     * waiting for the verdicts, the grace period and the close, the next game flushing the last,
     * a resync's segment, a correction, the words after a lesson, a position never recorded, a
     * game joined from the middle, the turn the clock settles, an end not seen.
     */
    @Test
    fun theDesktopTestsScenariosReplayAsTheyDid() {
        val scenarios = fixture().getJSONArray("scenarios")
        assertTrue(scenarios.length() > 0)
        for (s in 0 until scenarios.length()) {
            val scenario = scenarios.getJSONObject(s)
            val name = scenario.getString("name")
            val stores = ArrayList<Store>()
            val profilers = ArrayList<Profiler>()
            val steps = scenario.getJSONArray("steps")
            for (i in 0 until steps.length()) {
                val step = steps.getJSONObject(i)
                val where = "$name, step $i"
                when {
                    step.has("new_store") -> stores.add(Store(null))
                    step.has("new_profiler") -> profilers.add(
                        Profiler(
                            stores[step.getInt("store")],
                            sessionId = if (step.isNull("session")) null else step.getString("session"),
                            ts = step.getDouble("ts"),
                        ),
                    )
                    step.has("event") -> {
                        val event = step.getJSONObject("event")
                        val out = profilers[step.getInt("profiler")].onEvent(event(event), step.getDouble("ts"))
                        same("$where ${event.getString("type")}", step.getJSONArray("out"), JSONArray(out.map { dump(it) }))
                    }
                    else -> {
                        val profiler = profilers[step.getInt("profiler")]
                        when (val call = step.getString("call")) {
                            "tick" -> same("$where tick", step.getJSONArray("out"), JSONArray(profiler.tick(step.getDouble("ts")).map { dump(it) }))
                            "flush" -> same("$where flush", step.getJSONArray("out"), JSONArray(profiler.flush(step.getDouble("ts")).map { dump(it) }))
                            "snapshot" -> same("$where snapshot", step.getJSONArray("out"), JSONArray(listOf(dump(profiler.snapshot()))))
                            "end_session" -> assertEquals(
                                "$where end_session",
                                if (step.isNull("summary")) null else step.getString("summary"),
                                profiler.endSession(step.getDouble("ts")),
                            )
                            "correct" -> {
                                val args = step.getJSONArray("args")
                                val cause = if (args.isNull(3)) null else args.getString(3)
                                assertEquals("$where correct", step.getBoolean("ok"), profiler.correct(args.getString(0), args.getInt(1), args.getInt(2), cause))
                            }
                            else -> fail("$where: unknown call $call")
                        }
                    }
                }
            }
            val tables = scenario.getJSONArray("tables")
            assertEquals("$name stores", tables.length(), stores.size)
            for ((k, store) in stores.withIndex()) {
                val want = tables.getJSONObject(k)
                for (table in want.keys()) {
                    val (columns, rows) = store.query("SELECT * FROM $table ORDER BY rowid", maxRows = 100_000)
                    same("$name store $k $table columns", want.getJSONObject(table).getJSONArray("columns"), JSONArray(columns))
                    same("$name store $k $table rows", want.getJSONObject(table).getJSONArray("rows"), JSONArray(rows.map { JSONArray(it) }))
                }
                store.close()
            }
        }
    }

    /**
     * Floats added as Python 3.12's sum() adds them, compensated: a session of 64.7, 97.4 and
     * 55.4 had its mean accuracy said 72 on the desktop and 73 on the phone (the parity audit,
     * 2026-09-29).
     */
    @Test
    fun floatsAddUpAsPythonAddsThem() {
        val accuracies = listOf(64.7, 97.4, 55.4)
        assertEquals(217.5, com.sidecoach.voice.Py.sum(accuracies), 0.0)
        assertEquals("72", Insights.fixed(com.sidecoach.voice.Py.sum(accuracies) / 3, 0))
        assertEquals(0.30000000000000004, com.sidecoach.voice.Py.sum(listOf(0.1, 0.1, 0.1)), 0.0) // Python's own
        assertEquals(0.0, com.sidecoach.voice.Py.sum(emptyList()), 0.0)
    }

    /**
     * What the cells add up to in words, cell rows in and lines out as the desktop wrote them:
     * nothing while the counts are small; the biggest cost a gap, and a cost spread across kinds
     * of position a habit; the motif missed most often; the phase that costs most per move, and
     * none when the busy phases cost nothing (`test_insights_need_evidence_before_they_speak`,
     * `test_insights_name_the_biggest_cost_and_whether_it_is_a_gap` and
     * `test_a_cost_spread_across_positions_is_a_habit` in tests/test_profile.py, and cases beside
     * them: the habit and the cost per move were in no line the games of the fixture made; the
     * parity audit, 2026-09-29).
     */
    @Test
    fun theInsightsMatchThePython() {
        val cases = fixture().getJSONArray("insights")
        val said = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val rows = case.getJSONArray("rows").let { a ->
                (0 until a.length()).map { k ->
                    val r = a.getJSONObject(k)
                    CellRow(
                        cause = if (r.isNull("cause")) null else r.getString("cause"),
                        motif = if (r.isNull("motif")) null else r.getString("motif"),
                        phase = if (r.isNull("phase")) null else r.getString("phase"),
                        errors = r.getInt("errors"),
                        chances = r.getInt("chances"),
                        expectedLoss = r.getDouble("expected_loss"),
                    )
                }
            }
            val lines = Insights.insights(rows, case.getInt("games"))
            assertEquals("insights #$i", strings(case.getJSONArray("lines")), lines)
            said.addAll(lines)
        }
        // The lines the fixture is there for.
        assertTrue(said.toString(), said.any { it.startsWith("That looks like a habit: ") })
        assertTrue(said.toString(), said.any { it.startsWith("That looks like a gap: ") })
        assertTrue(said.toString(), said.any { " costs you most per move (" in it })
    }

    /**
     * `causes.wins_material` for every capture of the puzzle sample's first positions and the
     * desktop test's own -- taking back read by the rules (`exchange.can_retake`), so a guard
     * pinned to its own king takes nothing back -- and `causes.classify` on the pinned-guard
     * blunder of the desktop test; and `tests/test_profile.py`'s cause cases (time pressure
     * first, a tactic allowed, a chance let go, a line miscalculated, a piece left to be taken,
     * a tactic never considered, the phase naming the rest), what counts as a tactic, the
     * causes read from the player's words, and the motifs in words (f2/f7, x-ray).
     */
    @Test
    fun winsMaterialAndTheCauseMatchThePython() {
        val causes = fixture().getJSONObject("causes")
        val wins = causes.getJSONArray("wins_material")
        val failures = ArrayList<String>()
        for (i in 0 until wins.length()) {
            val o = wins.getJSONObject(i)
            val board = com.sidecoach.chess.Board(o.getString("fen"))
            val mine = Causes.winsMaterial(board, com.sidecoach.chess.Move.fromUci(o.getString("reply")))
            if (mine != o.getBoolean("wins") && failures.size < 8) {
                failures.add("${o.getString("fen")} ${o.getString("reply")}: kotlin $mine, python ${o.getBoolean("wins")}")
            }
        }
        val classified = causes.getJSONArray("classify")
        for (i in 0 until classified.length()) {
            val o = classified.getJSONObject(i)
            val board = com.sidecoach.chess.Board(o.getString("fen"))
            val (cause, motif) = Causes.classify(
                board, com.sidecoach.chess.Move.fromUci(o.getString("move")), reply = o.getString("reply"),
            )
            val want = o.getString("cause") to (if (o.isNull("motif")) null else o.getString("motif"))
            if ((cause to motif) != want) failures.add("${o.getString("fen")}: kotlin ${cause to motif}, python $want")
        }
        val tables = fixture().getJSONObject("cause_tables")
        val cases = tables.getJSONArray("classify")
        for (i in 0 until cases.length()) {
            val o = cases.getJSONObject(i)
            val a = o.getJSONObject("args")
            fun tagsOf(key: String) = if (a.has(key)) tags(a.getJSONArray(key)) else emptyList()
            val got = Causes.classify(
                com.sidecoach.chess.Board(o.getString("fen")),
                com.sidecoach.chess.Move.fromUci(o.getString("move")),
                playedTags = tagsOf("played_tags"),
                availableTags = tagsOf("available_tags"),
                allowedTags = tagsOf("allowed_tags"),
                reply = if (a.has("reply")) a.getString("reply") else null,
                replyMate = if (a.has("reply_mate")) a.getInt("reply_mate") else null,
                phase = a.optString("phase", "middlegame"),
                leftBook = a.optBoolean("left_book", false),
                clockMs = if (a.has("clock_ms")) a.getInt("clock_ms") else null,
                missedChance = a.optBoolean("missed_chance", false),
            )
            val want = o.getString("cause") to (if (o.isNull("motif")) null else o.getString("motif"))
            if (got != want) failures.add("classify ${o.getString("move")} $a: kotlin $got, python $want")
        }
        val tactical = tables.getJSONArray("tactical")
        for (i in 0 until tactical.length()) {
            val o = tactical.getJSONObject(i)
            val want = if (o.isNull("motif")) null else o.getString("motif")
            val got = Causes.tactical(tags(o.getJSONArray("tags")))
            if (got != want) failures.add("tactical ${o.getJSONArray("tags")}: kotlin $got, python $want")
        }
        assertEquals("the tactics", strings(tables.getJSONArray("tactical_set")).toSet(), Causes.TACTICAL)
        val words = tables.getJSONArray("from_words")
        for (i in 0 until words.length()) {
            val o = words.getJSONObject(i)
            val want = if (o.isNull("cause")) null else o.getString("cause")
            if (Causes.fromWords(o.getString("text")) != want) failures.add("from_words '${o.getString("text")}': python $want")
        }
        val motifs = tables.getJSONArray("motif_words")
        for (i in 0 until motifs.length()) {
            val o = motifs.getJSONObject(i)
            if (Insights.motifWords(o.getString("motif")) != o.getString("words")) failures.add("motif_words ${o.getString("motif")}")
        }
        assertTrue("${wins.length()} captures: " + failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * The review's first sentence says how the game ended in words (`review.HOW_IT_ENDED`,
     * desktop d446f43): "lost by resignation or on time", "lost on time", "won in 17 moves" for
     * a game over on the board -- never the reason verbatim after "by".
     */
    @Test
    fun theReviewSaysHowTheGameEndedInPlainWords() {
        val endings = fixture().getJSONArray("endings")
        assertTrue(endings.length() > 0)
        for (i in 0 until endings.length()) {
            val e = endings.getJSONObject(i)
            val review = GameReview(
                gameId = "g",
                userSide = side(e.getString("user_side")),
                result = e.getString("result"),
                reason = if (e.isNull("reason")) null else e.getString("reason"),
            )
            assertEquals("${e.getString("result")} ${e.opt("reason")}", e.getString("line"), Review.linesFor(review, 17)[0])
        }
    }

    /** A list in the record as `json.dumps` writes it: DEL and everything past ASCII escaped (parity pv-04). */
    @Test
    fun theRecordsListsAreWrittenAsPythonWritesThem() {
        val cases = fixture().getJSONArray("json_lists")
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val items = strings(case.getJSONArray("items"))
            assertEquals("$items", case.getString("json"), Store.jsonList(items))
        }
    }

    private class Scripted(replies: List<String>) : LLMProvider {
        override val name = "scripted"
        private val replies = ArrayDeque(replies)
        val seen = ArrayList<List<ChatMessage>>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(system: String, messages: List<ChatMessage>, maxTokens: Int, temperature: Double, thinking: Boolean): Completion {
            seen.add(ArrayList(messages))
            return Completion(text = replies.removeFirstOrNull() ?: "```sql\nSELECT 1\n```")
        }
    }

    private companion object {
        fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

        fun tags(a: JSONArray): List<MotifTag> = (0 until a.length()).map { i ->
            val t = a.getJSONObject(i)
            MotifTag(t.getString("name"), strings(t.getJSONArray("squares")), if (t.isNull("move")) null else t.getString("move"))
        }

        fun side(name: String) = if (name == "white") Pieces.Side.WHITE else Pieces.Side.BLACK

        fun JSONObject.optIntOrNull(key: String): Int? = if (!has(key) || isNull(key)) null else getInt(key)
    }
}
