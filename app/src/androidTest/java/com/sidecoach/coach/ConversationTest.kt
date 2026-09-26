package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The harness the model reads, held to the desktop's, after every single step.
 *
 * The generator played a real game against the real Stockfish, the real tagger and the real
 * opening book and wrote down every event and the whole rendered message after each one.
 * Here the same events are replayed into [Conversation] and the text compared -- the record
 * with its verdicts, the position now at every disclosure level, the facts, the inventory,
 * the tally -- through a takeback, a re-read of the board, a fold of the dialogue, a dialog,
 * a review and the end of the game, once plain and once with the compact record.
 *
 * The system message of `coach/prompt.py` is compared too, for every persona and rating.
 */
@RunWith(AndroidJUnit4::class)
class ConversationTest {

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs && failures.size < 4) {
            failures.add("$what:\n  --- kotlin ---\n$mine\n  --- python ---\n$theirs")
        }
    }

    @Test
    fun everyStepOfTheHarnessMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/conversation.json").bufferedReader()
            .use { it.readText() }
        val root = JSONObject(text)
        val began = System.currentTimeMillis()

        checkPrompt(root.getJSONObject("prompt"))

        val runs = root.getJSONArray("runs")
        var steps = 0
        for (r in 0 until runs.length()) {
            val one = runs.getJSONObject(r)
            val compact = one.getBoolean("compact")
            val convo = Conversation()
            convo.compactRecord = compact
            val label = if (compact) "compact" else "plain"
            val script = one.getJSONArray("steps")
            for (i in 0 until script.length()) {
                val step = script.getJSONObject(i)
                steps += 1
                apply(convo, step, "$label #$i")
                check("$label #$i ${step.getString("op")} render", convo.render(), step.getString("render"))
                check(
                    "$label #$i ${step.getString("op")} drained",
                    convo.drainNew().map { it.kind },
                    strings(step.getJSONArray("drained")),
                )
                check("$label #$i active", convo.activeRecords().size, step.getInt("active"))
                check("$label #$i judged", convo.judgedRecords().size, step.getInt("judged"))
            }
            // The messages the coach sends are the render, once, as a user message.
            val messages = convo.messages()
            check("$label messages", messages.size, 1)
            check("$label message role", messages[0].role, "user")
            check("$label message content", messages[0].content, convo.render())

            val last = convo.activeRecords().last()
            val plies = one.getJSONObject("last_ply")
            check("$label last ply short", convo.renderPly(last, full = false), plies.getString("short"))
            val levels = plies.getJSONObject("levels")
            for (key in levels.keys()) {
                check(
                    "$label last ply at level $key",
                    convo.renderPly(last, DisclosureLevel.of(key.toInt())),
                    levels.getString(key),
                )
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${runs.length()} runs, $steps steps, $took ms")
        assertTrue(failures.joinToString("\n\n"), failures.isEmpty())
    }

    private fun apply(convo: Conversation, step: JSONObject, where: String) {
        when (val op = step.getString("op")) {
            "new_game" -> convo.newGame(
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
                control = step.getString("control"),
                profile = strings(step.getJSONArray("profile")),
            )

            "move" -> convo.move(
                MovePlayed(
                    gameId = GAME_ID,
                    ply = step.getInt("ply"),
                    san = step.getString("san"),
                    uci = step.getString("uci"),
                    fenBefore = step.getString("fen_before"),
                    fenAfter = step.getString("fen_after"),
                    byUser = step.getBoolean("by_user"),
                    thinkMs = if (step.isNull("think_ms")) null else step.getInt("think_ms"),
                )
            )

            "clock" -> convo.clock(
                if (step.isNull("white_ms")) null else step.getInt("white_ms"),
                if (step.isNull("black_ms")) null else step.getInt("black_ms"),
            )

            "assess" -> convo.assess(
                MoveAssessment(
                    gameId = GAME_ID,
                    ply = step.getInt("ply"),
                    uci = step.getString("uci"),
                    expectedLoss = step.getDouble("expected_loss"),
                    severity = Severity.valueOf(step.getString("severity").uppercase()),
                    best = strings(step.getJSONArray("best")),
                    tags = tags(step.getJSONArray("tags")),
                    depth = step.getInt("depth"),
                    label = if (step.isNull("label")) {
                        null
                    } else {
                        Label.valueOf(step.getString("label").uppercase())
                    },
                    accuracy = if (step.isNull("accuracy")) null else step.getDouble("accuracy"),
                )
            )

            "analysed" -> convo.analysed(
                Analysis(
                    gameId = GAME_ID,
                    fen = step.getString("fen"),
                    ply = step.getInt("ply"),
                    lines = lines(step.getJSONArray("lines")),
                    decisive = step.getBoolean("decisive"),
                    onlyMove = step.getBoolean("only_move"),
                    threats = strings(step.getJSONArray("threats")),
                    tags = tags(step.getJSONArray("tags")),
                    opening = if (step.isNull("opening")) null else step.getString("opening"),
                ),
                factsFrom(step.getJSONObject("facts")),
            )

            "level" -> convo.level = DisclosureLevel.of(step.getInt("level"))

            "note" -> convo.note(step.getString("text"))

            "review" -> convo.review(strings(step.getJSONArray("lines")))

            "user_said" -> convo.userSaid(
                step.getString("text"),
                if (step.isNull("verdicts")) null else strings(step.getJSONArray("verdicts")),
            )

            "coach_said" -> convo.coachSaid(step.getString("text"), nod = step.getBoolean("nod"))

            "takeback" -> convo.takeback(
                Takeback(GAME_ID, step.getInt("plies"), step.getString("fen"))
            )

            "resync" -> {
                val fen = step.getString("fen")
                convo.resync(Resync(GAME_ID, fen, step.getString("reason")), Board(fen))
            }

            "covered" -> convo.covered()

            "ended" -> convo.ended(
                GameEnded(
                    GAME_ID,
                    step.getString("result"),
                    if (step.isNull("reason")) null else step.getString("reason"),
                )
            )

            "fold" -> {
                val before = convo.summary
                val pending = convo.foldable(step.getInt("keep"))
                check(
                    "$where foldable",
                    pending.map { (after, entry) -> "$after|${entry.kind}|${entry.text}" },
                    step.getJSONArray("pending").let { all ->
                        (0 until all.length()).map { k ->
                            val o = all.getJSONObject(k)
                            "${o.getString("after")}|${o.getString("kind")}|${o.getString("text")}"
                        }
                    },
                )
                check(
                    "$where summary request",
                    summaryRequest(pending, before),
                    step.getString("summary_request"),
                )
                convo.fold(pending.map { it.second }, step.getString("summary"))
            }

            else -> throw IllegalArgumentException("unknown op '$op' at $where")
        }
    }

    private fun checkPrompt(prompt: JSONObject) {
        // C ships as an app asset: the shipped copy must be the one the desktop reads.
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        check("shipped corrections", Corrections.load(app.assets), prompt.getString("corrections"))
        Corrections.set(prompt.getString("corrections"))
        check("PERSONA", PERSONA, prompt.getString("persona_default"))
        check("CONTRACT", CONTRACT, prompt.getString("contract"))
        check("RULES", RULES, prompt.getString("rules"))
        check("REPLY_FORMAT", REPLY_FORMAT, prompt.getString("reply_format"))
        check("DEFAULT_PERSONA", DEFAULT_PERSONA, prompt.getString("default_persona"))
        val personas = prompt.getJSONObject("personas")
        check("persona names", PERSONAS.keys.toList(), personas.keys().asSequence().toList())
        for (name in personas.keys()) {
            val pair = personas.getJSONArray(name)
            check("persona $name label", PERSONAS[name]?.first, pair.getString(0))
            check("persona $name text", PERSONAS[name]?.second, pair.getString(1))
        }
        val cases = prompt.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val one = cases.getJSONObject(i)
            val persona = if (one.isNull("persona")) null else one.getString("persona")
            val rating = if (one.isNull("rating")) {
                null
            } else {
                one.getJSONArray("rating").let { it.getInt(0) to it.getInt(1) }
            }
            val withC = one.getBoolean("with_corrections")
            val label = "prompt '$persona' $rating c=$withC"
            check("$label persona text", personaText(persona, rating), one.getString("persona_text"))
            check("$label system", systemMessage(withC, persona, rating), one.getString("system"))
        }
    }

    // --- rebuilding what the generator wrote down ---------------------------------------------

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

    private fun factsFrom(o: JSONObject) = PositionFacts(
        fen = o.getString("fen"),
        sideToMove = o.getString("side_to_move"),
        userSide = nullable(o, "user_side"),
        phase = o.getString("phase"),
        material = nullable(o, "material"),
        lastMove = nullable(o, "last_move"),
        decisive = o.getBoolean("decisive"),
        onlyMove = o.getBoolean("only_move"),
        nature = nullable(o, "nature"),
        goal = nullable(o, "goal"),
        theme = nullable(o, "theme"),
        mechanism = nullable(o, "mechanism"),
        region = nullable(o, "region"),
        target = nullable(o, "target"),
        piece = nullable(o, "piece"),
        why = nullable(o, "why"),
        question = nullable(o, "question"),
        nudge = nullable(o, "nudge"),
        focus = strings(o.getJSONArray("focus")),
        tactic = nullable(o, "tactic"),
        plan = nullable(o, "plan"),
        ideas = strings(o.getJSONArray("ideas")),
        candidates = strings(o.getJSONArray("candidates")),
        threats = strings(o.getJSONArray("threats")),
        line = strings(o.getJSONArray("line")),
        userClaims = o.getJSONArray("user_claims").let { all ->
            (0 until all.length()).map { k ->
                val entry = all.getJSONObject(k)
                entry.keys().asSequence().associateWith { key ->
                    if (entry.isNull(key)) null else entry.opt(key)
                }
            }
        },
        // The first hint and its parts: the sheet carries the whole of it (desktop 717cc9c).
        stakes = nullable(o, "stakes"),
        key = nullable(o, "key"),
        trap = nullable(o, "trap"),
        choices = strings(o.getJSONArray("choices")),
        situation = nullable(o, "situation"),
    )

    private fun nullable(o: JSONObject, key: String): String? =
        if (o.isNull(key)) null else o.getString(key)

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    companion object {
        private const val TAG = "sidecoach.coach"
        private const val GAME_ID = "parity"
    }
}
