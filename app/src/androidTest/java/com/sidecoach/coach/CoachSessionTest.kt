package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileSnapshot
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the coach says, and every prompt it says it from, held to the desktop's.
 *
 * The generator played a real game against the real Stockfish with a recorded model -- a
 * fixed list of replies answered by call index -- and wrote down every call's system message
 * and user turn along with every message that came out, its level, trigger, board marks and
 * what the checks found. Here the same events are replayed into [Coach] with the same
 * recorded model, and after every step the messages, the checks, the counters and the
 * prompts must be the same, once without a verifier and once with one.
 *
 * The draws are the same too: the desktop's `random.choice` was patched with a counter, and
 * the same counter is injected here, so which template of a rotating list is said and which
 * candidate is spoken are compared in the order they were drawn.
 */
@RunWith(AndroidJUnit4::class)
class CoachSessionTest {

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine == theirs || failures.size >= 4) return
        if (mine is String && theirs is String) {
            // Two long texts: the window around the first character that differs says more
            // than two screenfuls of the same prose.
            var at = 0
            while (at < mine.length && at < theirs.length && mine[at] == theirs[at]) at += 1
            val from = maxOf(0, at - 120)
            failures.add(
                "$what: differ at $at of ${mine.length}/${theirs.length}\n" +
                    "  --- kotlin ---\n" + mine.substring(from, minOf(mine.length, at + 160)) +
                    "\n  --- python ---\n" + theirs.substring(from, minOf(theirs.length, at + 160))
            )
            return
        }
        failures.add("$what:\n  --- kotlin ---\n$mine\n  --- python ---\n$theirs")
    }

    private class Call(
        val kind: String,
        val index: Int,
        val system: String,
        val messages: List<ChatMessage>,
        val maxTokens: Int,
        val temperature: Double,
        val reply: String,
    )

    /**
     * The verifier: answers by what it was asked about, since [Speaker] checks its
     * candidates in a thread pool and their calls are in no fixed order.
     */
    private class RecordedVerifier(val log: MutableList<Call>) : LLMProvider {
        override val name = "recorded-verifier"

        override fun healthcheck(): Pair<Boolean, String> = true to "recorded"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            val content = messages.joinToString("\n") { it.content }
            val text = when {
                "Unsupported statements:" in content -> VERIFY_REPAIR
                VERIFY_FAULTED_PHRASE in content -> VERIFY_FAULT
                else -> VERIFY_CLEAN
            }
            synchronized(log) { log.add(Call("verify", -1, system, messages, maxTokens, temperature, text)) }
            return Completion(text = text, stats = mapOf("recorded" to true))
        }
    }

    /** A model that answers by call index, and writes down every call it was given. */
    private class Recorded(
        val replies: List<String>,
        val tag: String,
        val log: MutableList<Call>,
    ) : LLMProvider {
        override val name = "recorded"
        private var n = 0

        override fun healthcheck(): Pair<Boolean, String> = true to "recorded"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            val index = n++
            val text = replies[index % replies.size]
            log.add(Call(tag, index, system, messages, maxTokens, temperature, text))
            return Completion(text = text, stats = mapOf("recorded" to true))
        }
    }

    @Test
    fun everyThingTheCoachSaysMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        Corrections.load(app.assets)
        val text = context.assets.open("chess/session.json").bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val began = System.currentTimeMillis()

        val runs = root.getJSONArray("runs")
        var steps = 0
        var calls = 0
        for (r in 0 until runs.length()) {
            val one = runs.getJSONObject(r)
            val withVerifier = one.getBoolean("with_verifier")
            val label = when {
                one.optBoolean("blitz") -> "blitz"
                withVerifier -> "verified"
                else -> "plain"
            }
            val replies = strings(one.getJSONArray("replies"))
            val claimVerdicts = one.getJSONArray("claim_verdicts").let { all ->
                (0 until all.length()).map { strings(all.getJSONArray(it)) }
            }

            val draws = intArrayOf(0)
            val log = ArrayList<Call>()
            val provider = Recorded(replies, "coach", log)
            val verifier = if (withVerifier) RecordedVerifier(log) else null
            var claimIndex = 0
            val coach = Coach(
                provider,
                draftCheck = { draft, board, after ->
                    checkDraft(draft, board, after).map { "${it.text}: ${it.reason}" }
                },
                claims = { _, _, _ -> claimVerdicts[claimIndex++ % claimVerdicts.size] },
                verifier = verifier,
                summariseEvery = 0,
                compactRecord = false,
                persona = "warm_direct",
                rating = 1000 to 1400,
                chooser = { list -> list[draws[0]++ % list.size] },
                sample = { passed -> passed[draws[0]++ % passed.size].second },
            )
            coach.syncClaims = true
            coach.syncSummaries = true

            var seenSpoken = 0
            var seenCalls = 0
            val script = one.getJSONArray("steps")
            for (i in 0 until script.length()) {
                val step = script.getJSONObject(i)
                steps += 1
                apply(coach, step)
                coach.speaker.runPending()
                val out = coach.poll()
                val where = "$label #$i ${step.getString("op")}"
                check(
                    "$where messages",
                    out.map { showMessage(it) },
                    step.getJSONArray("messages").let { all ->
                        (0 until all.length()).map { wantMessage(all.getJSONObject(it)) }
                    },
                )
                val fresh = coach.spoken.drop(seenSpoken)
                seenSpoken = coach.spoken.size
                check(
                    "$where spoken",
                    fresh.map { showSpoken(it) },
                    step.getJSONArray("spoken").let { all ->
                        (0 until all.length()).map { wantSpoken(all.getJSONObject(it)) }
                    },
                )
                val all = log.drop(seenCalls)
                seenCalls = log.size
                calls += all.size
                val freshCalls = all.filter { it.kind != "verify" }
                checkCalls(
                    "$where verify call",
                    all.filter { it.kind == "verify" }
                        .sortedBy { c -> c.messages.joinToString("\n") { it.content } },
                    step.getJSONArray("verify_calls"),
                )
                val wanted = step.getJSONArray("calls")
                checkCalls("$where call", freshCalls, wanted)
                check("$where fallbacks", coach.speaker.fallbacks, step.getInt("fallbacks"))
                check("$where failures", coach.speaker.failures, step.getInt("failures"))
                check("$where dropped", coach.speaker.dropped, step.getInt("dropped"))
                check(
                    "$where last reason",
                    coach.policy.lastReason,
                    if (step.isNull("last_reason")) null else step.getString("last_reason"),
                )
                check("$where level", coach.conversation.level.value, step.getInt("level"))
                // The draws, in the order they were made: a template chosen out of turn
                // shows up here before it shows up as prose.
                check("$where draws", draws[0], step.getInt("draws"))
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${runs.length()} runs, $steps steps, $calls model calls, $took ms")
        assertTrue(failures.joinToString("\n\n"), failures.isEmpty())
    }

    private fun checkCalls(what: String, mine: List<Call>, theirs: JSONArray) {
        check("$what count", mine.size, theirs.length())
        for (k in mine.indices) {
            if (k >= theirs.length()) break
            val want = theirs.getJSONObject(k)
            val call = mine[k]
            check("$what $k kind", call.kind, want.getString("kind"))
            if (call.kind != "verify") check("$what $k index", call.index, want.getInt("index"))
            check("$what $k system", call.system, want.getString("system"))
            check("$what $k max tokens", call.maxTokens, want.getInt("max_tokens"))
            check("$what $k temperature", call.temperature, want.getDouble("temperature"))
            val wantMessages = want.getJSONArray("messages")
            check("$what $k turns", call.messages.size, wantMessages.length())
            for (m in call.messages.indices) {
                if (m >= wantMessages.length()) break
                val turn = wantMessages.getJSONObject(m)
                check("$what $k turn $m role", call.messages[m].role, turn.getString("role"))
                check(
                    "$what $k turn $m content",
                    call.messages[m].content,
                    turn.getString("content"),
                )
            }
        }
    }

    private fun apply(coach: Coach, step: JSONObject) {
        val ts = step.getDouble("ts")
        when (val op = step.getString("op")) {
            "profile" -> coach.onEvent(
                ProfileSnapshot(
                    games = step.getInt("games"),
                    insights = strings(step.getJSONArray("insights")),
                    lastSession = if (step.isNull("last_session")) {
                        null
                    } else {
                        step.getString("last_session")
                    },
                ),
                ts,
            )

            "new_game" -> coach.onEvent(
                GameStarted(
                    gameId = GAME_ID,
                    userSide = if (step.getString("user_side") == "white") {
                        Pieces.Side.WHITE
                    } else {
                        Pieces.Side.BLACK
                    },
                    startFen = step.getString("start_fen"),
                    kind = step.getString("kind"),
                ),
                ts,
            )

            "clock" -> coach.onEvent(
                ClockObservation(
                    whiteMs = if (step.isNull("white_ms")) null else step.getInt("white_ms"),
                    blackMs = if (step.isNull("black_ms")) null else step.getInt("black_ms"),
                ),
                ts,
            )

            "move" -> coach.onEvent(
                MovePlayed(
                    gameId = GAME_ID,
                    ply = step.getInt("ply"),
                    san = step.getString("san"),
                    uci = step.getString("uci"),
                    fenBefore = step.getString("fen_before"),
                    fenAfter = step.getString("fen_after"),
                    byUser = step.getBoolean("by_user"),
                    thinkMs = if (step.isNull("think_ms")) null else step.getInt("think_ms"),
                ),
                ts,
            )

            "analysis" -> {
                val a = step.getJSONObject("analysis")
                coach.onEvent(
                    Analysis(
                        gameId = GAME_ID,
                        fen = a.getString("fen"),
                        ply = a.getInt("ply"),
                        lines = lines(a.getJSONArray("lines")),
                        decisive = a.getBoolean("decisive"),
                        onlyMove = a.getBoolean("only_move"),
                        threats = strings(a.getJSONArray("threats")),
                        traps = lines(a.getJSONArray("traps")),
                        tags = tags(a.getJSONArray("tags")),
                        opening = if (a.isNull("opening")) null else a.getString("opening"),
                        inBook = a.getBoolean("in_book"),
                    ),
                    ts,
                )
            }

            "assessment" -> {
                val a = step.getJSONObject("assessment")
                coach.onEvent(
                    MoveAssessment(
                        gameId = GAME_ID,
                        ply = a.getInt("ply"),
                        uci = a.getString("uci"),
                        expectedLoss = a.getDouble("expected_loss"),
                        severity = Severity.valueOf(a.getString("severity").uppercase()),
                        best = strings(a.getJSONArray("best")),
                        tags = tags(a.getJSONArray("tags")),
                        depth = a.getInt("depth"),
                        label = if (a.isNull("label")) {
                            null
                        } else {
                            Label.valueOf(a.getString("label").uppercase())
                        },
                        accuracy = if (a.isNull("accuracy")) null else a.getDouble("accuracy"),
                    ),
                    ts,
                )
            }

            "ask" -> coach.ask(step.getString("text"), ts)

            "tick" -> coach.tick(ts)

            "takeback" -> coach.onEvent(
                Takeback(GAME_ID, step.getInt("plies"), step.getString("fen")),
                ts,
            )

            "ended" -> coach.onEvent(
                GameEnded(
                    GAME_ID,
                    step.getString("result"),
                    if (step.isNull("reason")) null else step.getString("reason"),
                ),
                ts,
            )

            "review" -> coach.onEvent(
                GameReview(
                    gameId = GAME_ID,
                    userSide = Pieces.Side.WHITE,
                    result = "1-0",
                    lines = strings(step.getJSONArray("lines")),
                ),
                ts,
            )

            else -> throw IllegalArgumentException("unknown op '$op'")
        }
    }

    // --- what came out, as one comparable string ----------------------------------------------

    private fun showMessage(m: com.sidecoach.events.CoachMessage): String =
        listOf(
            m.text,
            m.level.value.toString(),
            m.trigger.slug,
            m.gameId ?: "-",
            m.ply?.toString() ?: "-",
            m.fen ?: "-",
            m.annotations.joinToString(";") {
                "${it.kind}|${it.squares.joinToString(",")}|${it.color}|${it.opacity}|${it.label}"
            },
        ).joinToString("\u0001")

    private fun wantMessage(o: JSONObject): String =
        listOf(
            o.getString("text"),
            o.getInt("level").toString(),
            o.getString("trigger"),
            if (o.isNull("game_id")) "-" else o.getString("game_id"),
            if (o.isNull("ply")) "-" else o.getInt("ply").toString(),
            if (o.isNull("fen")) "-" else o.getString("fen"),
            o.getJSONArray("annotations").let { all ->
                (0 until all.length()).joinToString(";") { i ->
                    val a = all.getJSONObject(i)
                    "${a.getString("kind")}|" +
                        strings(a.getJSONArray("squares")).joinToString(",") + "|" +
                        "${a.getString("color")}|${a.getDouble("opacity")}|" +
                        (if (a.isNull("label")) "null" else a.getString("label"))
                }
            },
        ).joinToString("\u0001")

    private fun showSpoken(s: Spoken): String =
        listOf(
            s.message.text,
            s.level.value.toString(),
            s.problems.joinToString(";"),
            s.attempts.toString(),
            s.recorded.toString(),
            s.streamed.toString(),
            s.repaired.toString(),
            s.verified.joinToString(";") { it.joinToString(",") },
            s.calls.joinToString("\u0002") { "${it.candidates}/${it.passed}\u0003${it.prompt}" },
        ).joinToString("\u0001")

    private fun wantSpoken(o: JSONObject): String =
        listOf(
            o.getString("text"),
            o.getInt("level").toString(),
            strings(o.getJSONArray("problems")).joinToString(";"),
            o.getInt("attempts").toString(),
            o.getBoolean("recorded").toString(),
            o.getBoolean("streamed").toString(),
            o.getBoolean("repaired").toString(),
            o.getJSONArray("verified").let { all ->
                (0 until all.length()).joinToString(";") {
                    strings(all.getJSONArray(it)).joinToString(",")
                }
            },
            o.getJSONArray("calls").let { all ->
                (0 until all.length()).joinToString("\u0002") { i ->
                    val c = all.getJSONObject(i)
                    "${c.getInt("candidates")}/${c.getInt("passed")}\u0003${c.getString("prompt")}"
                }
            },
        ).joinToString("\u0001")

    private fun lines(array: JSONArray): List<Line> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        Line(
            moves = strings(o.getJSONArray("moves")),
            depth = o.getInt("depth"),
            expectedScore = o.getDouble("expected_score"),
            scoreCp = if (o.isNull("score_cp")) null else o.getInt("score_cp"),
            mate = if (o.isNull("mate")) null else o.getInt("mate"),
            wdl = if (o.isNull("wdl")) {
                null
            } else {
                o.getJSONArray("wdl").let { Triple(it.getInt(0), it.getInt(1), it.getInt(2)) }
            },
        )
    }

    private fun tags(array: JSONArray): List<MotifTag> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        MotifTag(
            name = o.getString("name"),
            squares = strings(o.getJSONArray("squares")),
            move = if (o.isNull("move")) null else o.getString("move"),
        )
    }

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    companion object {
        private const val TAG = "sidecoach.coach"
        private const val GAME_ID = "parity"

        // What the recorded verifier answers, keyed on the call's own text.
        private const val VERIFY_CLEAN = "{\"unsupported\": []}"
        private const val VERIFY_FAULT = "{\"unsupported\": [\"the bishop is doing real work\"]}"
        private const val VERIFY_REPAIR = "{\"reply\": \"That was fine. The centre holds.\"}"
        private const val VERIFY_FAULTED_PHRASE = "the bishop is doing real work"
    }
}
