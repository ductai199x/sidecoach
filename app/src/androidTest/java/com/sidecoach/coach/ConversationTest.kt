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
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
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

    /** `new_game` then the move from [fenBefore], by the player or not; the move's record. */
    private fun oneMove(fenBefore: String, san: String, ply: Int, byUser: Boolean, userSide: Pieces.Side): Pair<Conversation, PlyRecord> {
        val convo = Conversation()
        convo.newGame(GameStarted(gameId = "g", userSide = userSide, startFen = Board().fen()))
        val board = Board(fenBefore)
        val move = board.parseSan(san)!!
        board.push(move)
        val record = convo.move(
            MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = fenBefore, fenAfter = board.fen(), byUser = byUser)
        )
        return convo to record
    }

    /**
     * After 10...O-O (a mistake) the engine's line was 10...Nxe4 11. Qxd4 Nf6 12. O-O: it ends on
     * the move played, and the verdict lost the line entirely. The line is dropped only when its
     * first move is the one played (`test_conversation.py`, TG-13).
     */
    @Test
    fun thePreferredLineIsDroppedOnlyWhenItsFirstMoveWasPlayed() {
        val (convo, record) = oneMove("r1bqk2r/2p1bppp/p2p1n2/1p1P4/3pP3/1B6/PPP1NPPP/R1BQK2R b KQkq - 1 10", "O-O", 20, false, Pieces.Side.WHITE)
        val uci = record.move.uci
        convo.assess(MoveAssessment(gameId = "g", ply = 20, uci = uci, expectedLoss = 0.13, severity = Severity.MISTAKE, best = listOf("f6e4", "d1d4", "e4f6", "e1g1"), label = Label.MISTAKE))
        assertTrue("the engine preferred 10... Nxe4 11. Qxd4 Nf6 12. O-O" in convo.renderPly(record))
        convo.assess(MoveAssessment(gameId = "g", ply = 20, uci = uci, expectedLoss = 0.01, severity = Severity.NONE, best = listOf(uci, "a2a4"), label = Label.GOOD))
        assertTrue("the engine preferred" !in convo.renderPly(record))
    }

    /**
     * Playing Black, "he will just move away to the right" (Kh1-g2) was answered "their King
     * cannot move to the right" (Tai's puzzle, 2026-09-28): the Setting says which file is on
     * which hand, a question that says left or right carries it, and the moves in question are
     * given their way (`test_conversation.py`).
     */
    @Test
    fun leftAndRightAreSaidAsThePlayerSeesThem() {
        for ((side, right) in listOf(Pieces.Side.WHITE to "h", Pieces.Side.BLACK to "a")) {
            assertTrue(side.name, "'to the right' means toward the $right-file" in hands(side))
        }
        for (said in listOf("he will just move away to the right, no?", "on my left side", "rightwards")) {
            assertTrue(said, SIDEWAYS.containsMatchIn(said))
        }
        for (said in listOf("is that right?", "right, so what now?", "what's left to do?")) {
            assertTrue(said, !SIDEWAYS.containsMatchIn(said))
        }
        val board = Board("6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4")
        assertEquals(
            "Kg2 (h1 to g2) goes to their right; Rg1+ (f1 to g1) goes to their left",
            ways(board, listOf("Kg2", "Rg1+"), Pieces.Side.BLACK),
        )
        assertEquals("e4 (e2 to e4) goes along the file", ways(Board(), listOf("e4"), Pieces.Side.WHITE))
        assertEquals("", ways(board, listOf("Rxf1+"), Pieces.Side.BLACK))
    }

    /**
     * After the player's capture, with the opponent to move: the facts heading closes its bracket
     * and says whose plan they are; the weaker lines say whose; the material notes the recapture
     * the engine expects (`test_conversation.py`, TG-13).
     */
    @Test
    fun thePositionSectionsOnTheOpponentsTurn() {
        val (convo, record) = oneMove("4k3/2n5/4N3/3P4/8/8/8/4K3 b - - 0 40", "Nxe6", 79, true, Pieces.Side.BLACK)
        convo.assess(MoveAssessment(gameId = "g", ply = 79, uci = record.move.uci, expectedLoss = 0.0, severity = Severity.NONE, best = listOf(record.move.uci), label = Label.BEST))
        val board = Board(record.move.fenAfter)
        val analysis = Analysis(
            gameId = "g", fen = board.fen(), ply = 79,
            lines = listOf(
                Line(moves = listOf("d5e6", "e8e7"), depth = 20, expectedScore = 0.5),
                Line(moves = listOf("e1e2", "e6d4"), depth = 20, expectedScore = 0.1),
                Line(moves = listOf("e1d1", "e6d4"), depth = 20, expectedScore = 0.1),
                Line(moves = listOf("e1f1", "e6d4"), depth = 20, expectedScore = 0.1),
            ),
            decisive = true, onlyMove = true,
        )
        convo.analysed(analysis, factsFromAnalysis(board, analysis, user = com.sidecoach.chess.BLACK))
        val text = convo.renderPly(record, DisclosureLevel.IDEA)
        assertTrue(text, "Material: the player is up a knight for a pawn before the expected recapture dxe6, then the player is down a pawn." in text)
        assertTrue(text, "## Facts (the opponent to move; the opponent's plan and ideas)\n" in text)
        assertTrue(text, "decisive: yes\nonly move: yes\n" in text)
        assertTrue(text, "  weaker for the opponent:\n" in text)
        assertTrue(text, "## Standing\nIt is about equal" in text)
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

            "analysed" -> {
                val facts = factsFrom(step.getJSONObject("facts"))
                convo.analysed(
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
                    facts,
                )
                // The facts as the lesson after the move reads them: an idea about a piece gone
                // from its square left out, a threat the move stopped too said (TG-09).
                val faced = step.getJSONObject("faced")
                val board = Board(step.getString("fen"))
                fun boardOf(key: String) = if (faced.isNull(key)) null else Board(faced.getString(key))
                check("$where faced", convo.facedText(facts, board, boardOf("after"), boardOf("played")), faced.getString("text"))
                check("$where faced bare", convo.facedText(facts, board), faced.getString("text_bare"))
            }

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
                // The desktop's own `summary_request`, as the messages the summariser is sent.
                check(
                    "$where summary request",
                    summaryRequest(pending, before).map { "${it.role}|${it.content}" },
                    step.getJSONArray("summary_request").let { all ->
                        (0 until all.length()).map { k ->
                            val o = all.getJSONObject(k)
                            "${o.getString("role")}|${o.getString("content")}"
                        }
                    },
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
        check("CONTRACT", CONTRACT, prompt.getString("contract"))
        check("RULES", RULES, prompt.getString("rules"))
        check("DEFAULT_PERSONA", DEFAULT_PERSONA, prompt.getString("default_persona"))
        val personas = prompt.getJSONObject("personas")
        // The order as an array: a JSON object's keys come back in the parser's order, which
        // org.json does not promise to keep (the parity audit cc-04).
        val order = prompt.getJSONArray("persona_order")
        check("persona names", PERSONAS.keys.toList(), (0 until order.length()).map { order.getString(it) })
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
        good = o.optJSONArray("good")?.let { strings(it) } ?: emptyList(),
        threats = strings(o.getJSONArray("threats")),
        line = strings(o.getJSONArray("line")),
        // The first hint and its parts: the sheet carries the whole of it (desktop 717cc9c).
        stakes = nullable(o, "stakes"),
        key = nullable(o, "key"),
        trap = nullable(o, "trap"),
        choices = strings(o.getJSONArray("choices")),
        situation = nullable(o, "situation"),
        principle = nullable(o, "principle"),
    )

    /**
     * Qxe5 takes their queen and Rxe5 takes back: with the new position not yet analysed, the
     * engine's own line for the move names the recapture. Without it "up a queen for a pawn"
     * was said, and the model called the queen trade a queen won (the dev audit, 2026-09-26).
     * test_conversation's `test_a_capture_the_engine_takes_back_is_said_as_such_before_the_position_is_read`.
     */
    @Test
    fun aCaptureTheEngineTakesBackIsSaidAsSuchBeforeThePositionIsRead() {
        val before = Board("1r5r/pbp3p1/5nkp/4Qq2/8/2N2P2/PP2R1PP/R1B3K1 b - - 6 19")
        val move = before.parseSan("Qxe5")!!
        val after = before.copy()
        after.push(move)
        val record = PlyRecord(
            MovePlayed(
                gameId = "g",
                ply = 38,
                san = "Qxe5",
                uci = move.uci(),
                fenBefore = before.fen(),
                fenAfter = after.fen(),
                byUser = true,
            ),
        )
        record.assessment = MoveAssessment(
            gameId = "g",
            ply = 38,
            uci = move.uci(),
            expectedLoss = 0.0,
            severity = Severity.NONE,
            best = listOf(move.uci(), "e2e5", "h8e8"),
            label = Label.GREAT,
        )
        val found = Conversation().expectedRecapture(record, before, after)
        assertTrue("$found", found != null && found.first == "Rxe5")
    }

    /** `tests.fakes.ScriptedProvider(then=...)`: answers [then], and keeps every call's messages. */
    private class Summariser(private val then: String) : LLMProvider {
        override val name = "scripted"
        val prompts = ArrayList<List<ChatMessage>>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            prompts.add(messages.toList())
            return Completion(text = then)
        }
    }

    /**
     * At a puzzle, what was said before the first move hung under "Game so far" with no
     * position, read as said after a move never made (the deep prompt review, 2026-10-01;
     * `test_what_was_said_before_any_move_is_said_at_the_start_position`).
     */
    @Test
    fun whatWasSaidBeforeAnyMoveIsSaidAtTheStartPosition() {
        val convo = Conversation()
        convo.newGame(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = Board().fen()))
        convo.userSaid("hint")
        convo.coachSaid("Look at the f-file.")
        val board = Board()
        for ((i, san) in listOf("e4", "e5").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            convo.move(
                MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before, fenAfter = board.fen(), byUser = ply % 2 == 1),
            )
        }
        val record = convo.render().substringAfter("# Game so far\n")
        assertTrue(record, record.startsWith("At the start position, before any move:\n    PLAYER said: "))
    }

    /**
     * Dialogue compaction: what was said earlier than the last `keep` plies is folded into one
     * summary at the top of the record; the summariser is called once and sees only the dialogue,
     * each line with the move it followed; the last plies keep their words; with nothing older to
     * fold, no call is made. The summary is the game's own: the next game starts without it, and
     * one written for entries of the game before is dropped -- it opened the next game's record
     * (`test_the_earlier_dialogue_folds_into_a_summary_and_the_last_plies_stay_verbatim` in
     * tests/test_conversation.py; the parity audit, 2026-09-29).
     */
    @Test
    fun theEarlierDialogueFoldsIntoASummaryAndTheLastPliesStayVerbatim() {
        val convo = Conversation()
        convo.newGame(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = Board().fen()))
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6", "Bb5", "a6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            convo.move(
                MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before, fenAfter = board.fen(), byUser = ply % 2 == 1),
            )
            if (ply == 1) convo.coachSaid("A fine first move.")
            if (ply == 2) {
                convo.userSaid("what now?")
                convo.coachSaid("Develop a piece.")
            }
            if (ply == 5) convo.coachSaid("The Ruy Lopez.")
        }
        val summariser = Summariser("The coach praised 1. e4 and told the player to develop.")
        assertTrue(summarise(convo, summariser, keep = 2))
        val prompt = summariser.prompts.last().last().content
        assertTrue(prompt, "after 1. e4, COACH: A fine first move." in prompt)
        assertTrue(prompt, "after 1... e5, PLAYER: \"what now?\"" in prompt)
        assertTrue("within the last two plies: kept verbatim", "The Ruy Lopez" !in prompt)
        val text = convo.render()
        assertTrue(text, "Earlier dialogue, summarised: The coach praised 1. e4" in text)
        assertTrue(text, "A fine first move." !in text && "Develop a piece." !in text)
        assertTrue(text, "YOU (the coach) said: \"The Ruy Lopez.\"" in text)
        // Nothing older to fold the second time: no call is made.
        assertTrue(!summarise(convo, summariser, keep = 2))
        assertEquals(1, summariser.prompts.size)
        // The summary is the game's own.
        val pending = convo.foldable(1).map { it.second }
        assertTrue(pending.isNotEmpty())
        convo.newGame(GameStarted(gameId = "g2", userSide = Pieces.Side.WHITE, startFen = Board().fen()))
        assertTrue("Earlier dialogue" !in convo.render())
        assertTrue(!convo.fold(pending, "The last game's dialogue.") && convo.summary == "")
    }

    private fun nullable(o: JSONObject, key: String): String? =
        if (o.isNull(key)) null else o.getString(key)

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    companion object {
        private const val TAG = "sidecoach.coach"
        private const val GAME_ID = "parity"
    }
}
