package com.sidecoach.ui

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.coach.CoachRunner
import com.sidecoach.coach.ModelCall
import com.sidecoach.coach.Spoken
import com.sidecoach.events.Analysis
import com.sidecoach.events.BoardUnread
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.Pieces
import com.sidecoach.voice.TtsProvider
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The panel held to the desktop's answers, scenario by scenario: `tools/make_panel_fixtures.py`
 * feeds a scripted game through the desktop's `PanelEngine` and records every line it wrote and
 * the state it kept, the lines it writes for the coach's messages, `log.elide` over awkward
 * whitespace, `outcome_result` and `end_game`'s
 * refusal, what `Settings.model_validate` says of saves that do not fit, and what the server
 * answers a route it does not serve. The phone is given the same inputs and must answer the same.
 */
@RunWith(AndroidJUnit4::class)
class PanelParityTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext
    private val fixture: JSONObject by lazy {
        val context = InstrumentationRegistry.getInstrumentation().context
        JSONObject(context.assets.open("panel/panel.json").bufferedReader().use { it.readText() })
    }
    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (!same(mine, theirs) && failures.size < 12) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
    }

    /** The game's events and the engine's word on them, written down line for line. */
    @Test
    fun everyEventIsWrittenDownAsTheDesktopWritesIt() {
        val steps = fixture.getJSONArray("dispatch")
        var opening: String? = null
        // The desktop's engine is a fresh one; the phone's is the app's, which other tests have
        // written on (a GameStarted leaves the badges as they were, as the desktop's does).
        PanelEngine.set("labels" to emptyMap<String, String>(), "game_id" to null, "user_side" to null, "clock" to null)
        for (i in 0 until steps.length()) {
            val step = steps.getJSONObject(i)
            val event = eventOf(step.getJSONObject("event"))
            val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
            if (event is MoveAssessment || event is Analysis) {
                opening = PanelEngine.onEngineEvents(listOf(event), opening)
            } else {
                PanelEngine.onGameEvent(event)
            }
            val polled = PanelEngine.poll(cursor)
            val lines = polled.getJSONArray("lines")
            val want = step.getJSONArray("records")
            val label = "step $i ${step.getJSONObject("event").getString("type")}"
            check("$label: records", lines.length(), want.length())
            for (k in 0 until minOf(lines.length(), want.length())) {
                val mine = lines.getJSONObject(k)
                val theirs = want.getJSONObject(k)
                for (key in listOf("chan", "level", "text", "data")) check("$label #$k $key", mine.opt(key), theirs.opt(key))
            }
            val state = polled.getJSONObject("state")
            val kept = step.getJSONObject("state")
            for (key in kept.keys()) check("$label: state $key", state.opt(key), kept.opt(key))
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** A voice that is busy or not as a case says, and says nothing (`FakeTTS`). */
    private class FakeTts : TtsProvider {
        override val name = "fake"
        @Volatile override var busy = false
        override fun healthcheck(): Pair<Boolean, String> = true to "fake"
        override fun say(text: String, voice: String?, owner: String) = Unit
        override fun stop(owner: String?) = Unit
    }

    /**
     * The coach's messages as the panel logs them, each with what it was checked against: "model
     * reply rejected; the quick answer stands", "draft rejected → template", "cut short", "not
     * spoken: board moved on", "not spoken: voice busy", "cut in", the leak and the form noted,
     * each model reply's figures, "LLM call failed" (`panel._coach_meta`, the phone's
     * `CoachRunner.drain`; the parity audit, 2026-09-29). Each message goes into an offline
     * coach's record, with the board's ply and the voice as the case sets them.
     */
    @Test
    fun theCoachsLinesAreWrittenAsTheDesktopWritesThem() {
        val prefs = app.getSharedPreferences("settings", Context.MODE_PRIVATE)
        val saved = prefs.getString("json", null)
        val speaker = PanelVoice.speaker
        val voicing = PanelVoice.voicing
        val paced = PanelVoice.coach
        val profiler = PanelEngine.profiler
        var runner: CoachRunner? = null
        try {
            // An offline coach: no model to reach from a test.
            val settings = SettingsStore(app)
            val values = settings.values()
            values.getJSONObject("llm").put("provider", "none")
            settings.save(values)
            val built = CoachRunner(SettingsStore(app))
            runner = built
            val coach = built.coach!!
            val voice = FakeTts()
            PanelVoice.speaker = voice
            PanelVoice.coach = null // the pace: not a fast game, as the desktop's fresh coach
            PanelEngine.profiler = null
            if (PanelVoice.muted) PanelVoice.setMuted(false)
            val cases = fixture.getJSONArray("coach_meta")
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val label = case.getString("label")
                PanelEngine.set("ply" to case.getInt("ply_now"))
                voice.busy = case.getBoolean("busy")
                PanelVoice.voicing = case.optJSONArray("voicing")?.let { it.getString(0) to (if (it.isNull(1)) null else it.getInt(1)) }
                coach.spoken.add(spokenOf(case.getJSONObject("spoken")))
                if (!case.isNull("failed")) {
                    coach.speaker.failures += 1
                    coach.speaker.lastError = case.getString("failed")
                }
                val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
                built.drain()
                val lines = PanelEngine.poll(cursor).getJSONArray("lines").let { a ->
                    (0 until a.length()).map { a.getJSONObject(it) }.filter { it.getString("chan") == "coach" }
                }
                val want = case.getJSONArray("records")
                check("$label: records ${lines.map { it.getString("text") }}", lines.size, want.length())
                for (k in 0 until minOf(lines.size, want.length())) {
                    for (key in listOf("chan", "level", "text", "data")) {
                        check("$label #$k $key", lines[k].opt(key), want.getJSONObject(k).opt(key))
                    }
                }
            }
        } finally {
            runner?.close()
            PanelVoice.speaker = speaker
            PanelVoice.voicing = voicing
            PanelVoice.coach = paced
            PanelEngine.profiler = profiler
            if (saved == null) prefs.edit().remove("json").commit() else prefs.edit().putString("json", saved).commit()
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** A coach message and what it was checked against, from the desktop's case. */
    private fun spokenOf(o: JSONObject): Spoken {
        val level = DisclosureLevel.of(o.getInt("level"))
        val message = CoachMessage(
            text = o.getString("text"),
            level = level,
            trigger = Trigger.entries.first { it.slug == o.getString("trigger") },
            gameId = "g1",
            ply = if (o.isNull("ply")) null else o.getInt("ply"),
            fen = if (o.isNull("fen")) null else o.getString("fen"),
        )
        val calls = o.getJSONArray("calls").let { a ->
            (0 until a.length()).map { k ->
                val c = a.getJSONObject(k)
                ModelCall(c.getString("raw"), statsOf(c.getJSONObject("stats")), c.getInt("candidates"), c.getInt("passed"), c.getString("prompt"))
            }
        }
        return Spoken(
            message = message,
            level = level,
            problems = strings(o.getJSONArray("problems")),
            attempts = o.getInt("attempts"),
            calls = calls,
            streamed = o.getBoolean("streamed"),
            leaks = strings(o.getJSONArray("leaks")),
            notes = strings(o.getJSONArray("notes")),
        )
    }

    /** A call's stats as the provider gives them: numbers, text, and a list of text. */
    private fun statsOf(o: JSONObject): Map<String, Any?> {
        val out = LinkedHashMap<String, Any?>()
        for (key in o.keys()) {
            val value = o.get(key)
            out[key] = when (value) {
                JSONObject.NULL -> null
                is JSONArray -> strings(value)
                else -> value
            }
        }
        return out
    }

    /** `log.elide`: Python's whitespace run together, the cut counted in code points. */
    @Test
    fun linesAreElidedAsPythonElidesThem() {
        val cases = fixture.getJSONArray("elide")
        for (i in 0 until cases.length()) {
            val one = cases.getJSONObject(i)
            check("elide #$i", PanelEngine.elide(one.getString("text"), one.getInt("limit")), one.getString("out"))
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * How a game ends on the user's word, from their side; an outcome the panel does not know
     * is refused, and one it knows waits for the session's loop.
     */
    @Test
    fun anEndIsScoredFromTheUsersSideAndAnUnknownOneRefused() {
        val outcomes = fixture.getJSONObject("outcomes")
        val table = outcomes.getJSONArray("table")
        for (i in 0 until table.length()) {
            val one = table.getJSONObject(i)
            val side = if (one.isNull("side")) null else one.getString("side")
            val (result, reason) = PanelEngine.outcomeResult(one.getString("outcome"), side)
            check("outcome ${one.getString("outcome")} for $side", JSONArray(listOf(result, reason)), one.getJSONArray("result"))
        }
        PanelEngine.commands.clear()
        val refused = runCatching { PanelEngine.endGame("rage-quit") }.exceptionOrNull()
        check("refusal", refused?.message, outcomes.getString("refused"))
        check("nothing queued for a refusal", PanelEngine.commands.size, 0)
        PanelEngine.endGame("draw")
        check("the end queued", PanelEngine.commands.poll(), "end" to "draw")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * A save checked field by field as pydantic checks it: the phone's own settings with one
     * or more fields changed, and either pydantic's errors -- type, where, words, what came,
     * the bound -- or the values it saves.
     */
    @Test
    fun aSaveThatDoesNotFitIsRefusedByFieldAsPydanticRefusesIt() {
        val store = SettingsStore(app)
        val schema = JSONObject(store.schema)
        val cases = fixture.getJSONArray("settings")
        for (i in 0 until cases.length()) {
            val one = cases.getJSONObject(i)
            val label = one.getString("label")
            val body = store.defaults()
            val changes = one.getJSONObject("changes")
            for (dotted in changes.keys()) {
                val value = changes.opt(dotted)
                if ('.' in dotted) {
                    body.getJSONObject(dotted.substringBefore('.')).put(dotted.substringAfter('.'), value)
                } else {
                    body.put(dotted, value)
                }
            }
            val checked = SettingsCheck.validate(schema, store.defaults(), body)
            if (one.has("errors")) {
                check("$label errors", checked.errors, one.getJSONArray("errors"))
            } else {
                check("$label errors", checked.errors.length(), 0)
                val saved = one.getJSONObject("saved")
                for (dotted in saved.keys()) {
                    val section = checked.values.optJSONObject(dotted.substringBefore('.'))
                    val key = dotted.substringAfter('.')
                    val want = saved.opt(dotted)
                    if (want == "<dropped>") {
                        check("$label $dotted dropped", section?.has(key), false)
                    } else {
                        check("$label $dotted", section?.opt(key), want)
                    }
                }
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * Every route the desktop's server serves is served here; one it does not serve, or a
     * method a route does not take, is answered as the desktop answers it -- over HTTP, through
     * the local server the page talks to.
     */
    @Test
    fun theRoutesAreTheDesktopsAndTheRestAnsweredAsItAnswers() {
        val served = fixture.getJSONArray("served")
        for (i in 0 until served.length()) {
            val one = served.getJSONObject(i)
            val path = one.getString("path").replace("{puzzle_id}", "abc").replace("{key}", "abc")
                .replace("{set_id}", "abc")
            check("served ${one.getString("method")} $path", PanelApi.serves(one.getString("method"), path), true)
        }
        val api = PanelApi(
            "test",
            SetupApi.shared(app),
            SettingsApi(SettingsStore(app)),
            files = app.filesDir,
            rooms = { PuzzleRooms.shared(app) },
            dumps = File(app.cacheDir, "dumps"),
        )
        val server = LocalServer(app.assets, api = api)
        val port = server.start()
        PanelEngine.commands.clear()
        try {
            val probes = fixture.getJSONArray("routes")
            for (i in 0 until probes.length()) {
                val probe = probes.getJSONObject(i)
                val method = probe.getString("method")
                val path = probe.getString("path")
                val connection = URL("http://127.0.0.1:$port$path").openConnection() as HttpURLConnection
                connection.requestMethod = method
                val body = probe.optJSONObject("json")
                if (body != null) {
                    connection.doOutput = true
                    connection.setRequestProperty("Content-Type", "application/json")
                    connection.outputStream.use { it.write(body.toString().toByteArray()) }
                }
                val status = connection.responseCode
                val text = (if (status < 400) connection.inputStream else connection.errorStream)
                    ?.bufferedReader()?.use { it.readText() } ?: ""
                connection.disconnect()
                check("$method $path status", status, probe.getInt("status"))
                val want = probe.get("body")
                check("$method $path body", if (want is String) text else JSONObject(text), want)
            }
        } finally {
            server.stop()
            PanelEngine.commands.clear()
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    // ---- the events, from the desktop's JSON --------------------------------------------------

    private fun eventOf(o: JSONObject): Event = when (o.getString("type")) {
        "GameStarted" -> GameStarted(o.getString("game_id"), side(o.getString("user_side"))!!, o.getString("start_fen"), o.getString("kind"))
        "MovePlayed" -> MovePlayed(
            o.getString("game_id"), o.getInt("ply"), o.getString("san"), o.getString("uci"),
            o.getString("fen_before"), o.getString("fen_after"), o.getBoolean("by_user"),
            if (o.isNull("think_ms")) null else o.getInt("think_ms"),
        )
        "MoveAssessment" -> MoveAssessment(
            gameId = o.getString("game_id"),
            ply = o.getInt("ply"),
            uci = o.getString("uci"),
            expectedLoss = o.getDouble("expected_loss"),
            severity = Severity.valueOf(o.getString("severity").uppercase()),
            best = strings(o.getJSONArray("best")),
            tags = o.getJSONArray("tags").let { a -> (0 until a.length()).map { MotifTag(a.getJSONObject(it).getString("name")) } },
            depth = o.getInt("depth"),
            label = if (o.isNull("label")) null else Label.valueOf(o.getString("label").uppercase()),
        )
        "Analysis" -> Analysis(
            gameId = o.getString("game_id"),
            fen = o.getString("fen"),
            ply = o.getInt("ply"),
            lines = o.getJSONArray("lines").let { a ->
                (0 until a.length()).map { k ->
                    val line = a.getJSONObject(k)
                    Line(strings(line.getJSONArray("moves")), line.getInt("depth"), line.getDouble("expected_score"))
                }
            },
            decisive = o.getBoolean("decisive"),
            onlyMove = o.getBoolean("only_move"),
            threats = strings(o.getJSONArray("threats")),
            opening = if (o.isNull("opening")) null else o.getString("opening"),
            provisional = o.optBoolean("provisional"),
        )
        "Takeback" -> Takeback(o.getString("game_id"), o.getInt("plies"), o.getString("fen"))
        "Resync" -> Resync(o.getString("game_id"), o.getString("fen"), o.getString("reason"))
        "Unexplained" -> Unexplained(o.getString("game_id"), o.getString("fen"), o.getString("reason"))
        "BoardUnread" -> BoardUnread(o.getString("reason"))
        "GameEnded" -> GameEnded(
            o.getString("game_id"), o.getString("result"),
            if (o.isNull("reason")) null else o.getString("reason"),
            if (o.isNull("pgn")) null else o.getString("pgn"),
        )
        "ClockObservation" -> ClockObservation(
            if (o.isNull("white_ms")) null else o.getInt("white_ms"),
            if (o.isNull("black_ms")) null else o.getInt("black_ms"),
            side(if (o.isNull("running")) null else o.getString("running")),
        )
        "ScreenStateChanged" -> ScreenStateChanged(ScreenState.valueOf(o.getString("state").uppercase()))
        else -> throw IllegalArgumentException("no event ${o.getString("type")}")
    }

    private fun side(name: String?): Pieces.Side? = name?.let { Pieces.Side.valueOf(it.uppercase()) }

    private fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

    companion object {
        /** Two JSON values alike: numbers by value (an int and a whole float are one number). */
        fun same(a: Any?, b: Any?): Boolean {
            val x = if (a == JSONObject.NULL) null else a
            val y = if (b == JSONObject.NULL) null else b
            return when {
                x == null || y == null -> x == null && y == null
                x is Number && y is Number -> x.toDouble() == y.toDouble()
                x is JSONObject && y is JSONObject ->
                    x.keys().asSequence().toSet() == y.keys().asSequence().toSet() &&
                        x.keys().asSequence().all { same(x.opt(it), y.opt(it)) }
                x is JSONArray && y is JSONArray ->
                    x.length() == y.length() && (0 until x.length()).all { same(x.opt(it), y.opt(it)) }
                else -> x == y
            }
        }
    }
}
