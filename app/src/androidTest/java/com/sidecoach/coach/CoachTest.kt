package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.profile.Causes
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * When the coach speaks: the desktop's `tests/test_coach.py` for desktop 88b84a6, driven the
 * way the desktop drives it -- a scripted model, the events by hand. A question asked before
 * the engine has looked is answered when it has; no decisive question while the player plays
 * out the line they found, and one again when the opponent leaves it; and the next puzzle's
 * question after a long one.
 */
@RunWith(AndroidJUnit4::class)
class CoachTest {

    /** Answers from a script, and remembers every prompt it was given. */
    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"

        /** What it says next; a test may set them afresh, as the desktop's `provider.replies = ...`. */
        val replies = ArrayDeque(replies)
        val prompts = ArrayList<Pair<String, List<ChatMessage>>>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            synchronized(prompts) { prompts.add(system to messages.toList()) }
            val text = synchronized(replies) { replies.removeFirstOrNull() } ?: "Keep looking."
            return Completion(text = text)
        }
    }

    private fun coach(replies: List<String>): Pair<Coach, ScriptedProvider> {
        val provider = ScriptedProvider(replies)
        val coach = Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        return coach to provider
    }

    private fun start(coach: Coach, userSide: Pieces.Side = Pieces.Side.WHITE) {
        coach.onEvent(GameStarted(gameId = "g", userSide = userSide, startFen = Board().fen()), 0.0)
    }

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    /**
     * A question asked before the engine has looked at the position on the board is told to
     * wait a moment and answered when the analysis lands, never from an older position (Tai,
     * 2026-09-24: a hint pressed as a puzzle appeared got "I have not seen enough of this game
     * yet to say. Give me a move or two.").
     */
    @Test
    fun askingBeforeTheEngineHasLookedIsAnsweredWhenItHas() {
        val (coach, _) = coach(emptyList())
        start(coach)
        coach.ask("hint", 1.0)
        val out = messages(coach)
        assertEquals(listOf("One moment -- still looking at this position."), out.map { it.text })
        coach.ask("hint", 1.5) // asked again while waiting: one "moment" is enough
        assertEquals(emptyList<String>(), messages(coach).map { it.text })
        val start = Board()
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = start.fen(),
                ply = 0,
                lines = listOf(
                    Line(moves = listOf("e2e4", "e7e5", "g1f3"), depth = 18, expectedScore = 0.55),
                    Line(moves = listOf("d2d4", "d7d5"), depth = 18, expectedScore = 0.54),
                ),
                decisive = false,
                onlyMove = false,
            ),
            3.0,
        )
        val answered = messages(coach)
        assertTrue(
            answered.map { it.trigger }.toString(),
            answered.isNotEmpty() && answered.all { it.trigger == Trigger.USER_ASK },
        )
        assertTrue(answered[0].text, "One moment" !in answered[0].text)
    }

    private fun puzzle(coach: Coach, fen: String = QD1): Board {
        coach.onEvent(
            GameStarted(gameId = "g", userSide = Pieces.Side.BLACK, startFen = fen, kind = "position"),
            0.0,
        )
        return Board(fen)
    }

    private fun analysed(coach: Coach, board: Board, ply: Int, line: List<String>, ts: Double, mate: Int?) {
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = ply,
                lines = listOf(
                    Line(moves = line, depth = 18, expectedScore = 1.0, mate = mate),
                    Line(moves = listOf("h6h5"), depth = 18, expectedScore = 0.1),
                ),
                decisive = true,
                onlyMove = true,
            ),
            ts,
        )
    }

    private fun played(coach: Coach, board: Board, ply: Int, uci: String, byUser: Boolean, ts: Double) {
        val move = Move.fromUci(uci)
        val before = board.fen()
        val san = board.san(move)
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g",
                ply = ply,
                san = san,
                uci = uci,
                fenBefore = before,
                fenAfter = board.fen(),
                byUser = byUser,
            ),
            ts,
        )
    }

    /**
     * The puzzle appears, the player presses Hint before the engine has looked, and the hint
     * comes when it has: no puzzle question was asked, as on Tai's desktop.
     */
    private fun hintThen(coach: Coach, board: Board) {
        coach.ask("hint", 0.5)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        val got = messages(coach)
        assertTrue(
            got.map { it.trigger }.toString(),
            got.isNotEmpty() && got.all { it.trigger == Trigger.USER_ASK },
        )
    }

    /**
     * After 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 the coach asked "You can mate in one move ... Which
     * check leaves their king no square at all?" -- a fresh question in the middle of the
     * combination the player was carrying out (Tai's desktop, 2026-09-24).
     */
    @Test
    fun noNewQuestionWhileThePlayerPlaysOutTheLineTheyFound() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        hintThen(coach, board)
        for ((i, uci) in QD1_LINE.take(4).withIndex()) {
            val ply = i + 1
            played(coach, board, ply, uci, ply % 2 == 1, 2.0 + ply)
            if (ply % 2 == 0) {
                analysed(coach, board, ply, QD1_LINE.drop(ply), 2.5 + ply, (5 - ply) / 2)
            }
        }
        val asked = messages(coach).filter { it.trigger == Trigger.DECISIVE_QUESTION }.map { it.text }
        assertEquals(asked.toString(), emptyList<String>(), asked)
    }

    @Test
    fun anOpponentWhoLeavesTheLineBringsTheQuestionBack() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        hintThen(coach, board)
        played(coach, board, 1, "c2d1", true, 3.0)
        played(coach, board, 2, "a5e1", false, 4.0) // blocks instead of taking: off the line
        analysed(coach, board, 2, listOf("d1e1"), 4.5, 1)
        val asked = messages(coach).filter { it.trigger == Trigger.DECISIVE_QUESTION }
        assertEquals(1, asked.size)
    }

    /**
     * The decisive stretch pointed at in one puzzle said nothing about the next: move numbers
     * start again, and a new puzzle's first position fell inside the last one's stretch, so its
     * question was never asked (Tai's desktop, 2026-09-24).
     */
    @Test
    fun theNextPuzzleGetsItsQuestionAfterALongOne() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach, "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
        for ((i, uci) in listOf("g8f8", "a1a2", "f8e7", "a2a3").withIndex()) {
            val ply = i + 1
            played(coach, board, ply, uci, ply % 2 == 1, ply.toDouble())
        }
        analysed(coach, board, 4, listOf("e7d6", "a3a6"), 5.0, null)
        assertEquals(listOf(Trigger.DECISIVE_QUESTION), messages(coach).map { it.trigger })
        coach.onEvent(
            Resync(gameId = "g", fen = QD1, reason = "position adopted after 30 unexplained cells"),
            20.0,
        )
        analysed(coach, Board(QD1), 0, QD1_LINE, 21.0, 3)
        assertEquals(listOf(Trigger.DECISIVE_QUESTION), messages(coach).map { it.trigger })
    }

    /** White, the user, to move with Qxf7# on the board, after black's Nf6 (`_play_to_scholar`). */
    private fun playToScholar(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = move.uci(),
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        assertTrue(coach.userToMove && coach.board.fen() == SCHOLAR)
    }

    /** `_mate_analysis`: Qxf7# on the scholar's board. */
    private fun mateAnalysis(ply: Int, decisive: Boolean = true) = Analysis(
        gameId = "g",
        fen = SCHOLAR,
        ply = ply,
        lines = listOf(
            Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
            Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
        ),
        decisive = decisive,
        onlyMove = true,
        tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
    )

    /** The directive of the last model call: the harness after its "# Task" line. */
    private fun directive(provider: ScriptedProvider): String {
        val messages = synchronized(provider.prompts) { provider.prompts.last().second }
        assertEquals(1, messages.size)
        return messages[0].content.substringAfterLast("# Task\n")
    }

    /** `_bishop_check_instead_of_mate`: the user plays Bxf7+ where Qxf7 was mate, and it is judged. */
    private fun bishopCheckInsteadOfMate(coach: Coach) {
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Bxf7+")!!
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g",
                ply = 7,
                san = "Bxf7+",
                uci = move.uci(),
                fenBefore = SCHOLAR,
                fenAfter = board.fen(),
                byUser = true,
            ),
            10.0,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g",
                ply = 7,
                uci = move.uci(),
                expectedLoss = 0.45,
                severity = Severity.BLUNDER,
                best = listOf("h5f7"),
                label = Label.MISS,
            ),
            11.0,
        )
        messages(coach)
    }

    /**
     * "Why was that bad?" is about a position that is past: the directive states the verdict,
     * the move the engine preferred and the idea, and says nothing is withheld -- and what it
     * cost only when they ask it: "It cost you 22% of the game." opened every answer, and Tai
     * had said the figure is for when he asks (2026-09-20; desktop 6d9ba6f). The phone's
     * directive names the better move as "the engine preferred Qxf7#" (the desktop's "Qxf7# was
     * better" came with commits not ported).
     */
    @Test
    fun aQuestionAboutTheLastMoveGetsItsVerdictWithoutTheFigure() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("why was that bad?", 12.0)
        val out = messages(coach)
        val directive = directive(provider)
        assertTrue(directive, "They ask about the move they already played, Bxf7+" in directive)
        assertTrue(directive, "the engine preferred Qxf7#" in directive)
        assertTrue(directive, "nothing about it is withheld" in directive)
        assertTrue(directive, "45%" !in directive && "no figures" in directive)
        // Spoken from the position the move was played from, with everything shown.
        assertTrue(out.isNotEmpty() && out.last().level == DisclosureLevel.LINE)
        assertEquals(SCHOLAR, out.last().fen)
        // The fallback is the verdict itself, not a hint -- and without the figure.
        assertTrue(out.last().text, out.last().text.startsWith("Qxf7# was better"))
    }

    /**
     * Asked what it cost, the figure is said (desktop 6d9ba6f). The desktop's "how much did that
     * cost me?" reaches the answer about the last move through `_asked_own_move` (f733318, not
     * ported); on the phone the question about the last move carries it.
     */
    @Test
    fun whatAMoveCostIsSaidWhenTheyAskIt() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("was that a mistake? how much did it cost me?", 12.0)
        val out = messages(coach)
        val directive = directive(provider)
        assertTrue(directive, "cost 45% of the game -- they asked" in directive)
        assertTrue(out.isNotEmpty())
        assertTrue(out.last().text, out.last().text.startsWith("It cost you 45% of the game. Qxf7# was better"))
    }

    @Test
    fun inBlitzAMistakeGetsAWordThatDoesNotCallItFine() {
        // Blitz keeps only the word after the move (the policy), so a blunder comes to the
        // directive as POST_MOVE: no lesson and no question, but not "fine" either.
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val position = coach.positions.getValue(6)
        assertNotNull(position.facts)
        val board = Board(SCHOLAR)
        val move = board.parseSan("a3")!!
        board.push(move)
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "a3",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = board.fen(),
            byUser = true,
        )
        val blunder = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.5,
            severity = Severity.BLUNDER,
            best = listOf("h5f7"),
            label = Label.BLUNDER,
        )
        val (instruction, fallback) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), blunder, played,
        )
        assertTrue(fallback, fallback in COST_YOU.map { it.replace("{best}", "Qxf7#") + "." })
        assertTrue(instruction, "which was a blunder" in instruction && "fine" !in instruction)
        assertTrue(instruction, "no lesson, no question" in instruction)
        // A move a little behind the best is still fine.
        val good = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.03,
            severity = Severity.NONE,
            best = listOf("h5f7"),
            label = Label.GOOD,
        )
        val (_, fine) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        // The better move and what it does, and the pattern to keep.
        assertTrue(fine, FINE_TIGHTER.any { fine.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(fine, "it is mate" in fine)
        assertTrue(fine, KEEP_LEADS.any { "${it.replaceFirstChar { c -> c.uppercase() }}:" in fine })
        // Said once, the lesson is not said again a move later: the verdict and the better move
        // stand alone (Nova bot games, 2026-09-25: 27% of lessons came again within three moves).
        coach.conversation.coachSaid(fine)
        val (_, again) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        assertTrue(again, FINE_TIGHTER.any { again.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(again, "it is mate" !in again)
        assertTrue(again, KEEP_LEADS.none { "${it.replaceFirstChar { c -> c.uppercase() }}:" in again })
        assertTrue(instruction, "Do not open with the move they played" in instruction)
    }

    @Test
    fun aLessonIsSaidOfTheBoardThePlayerSeesAndNotTwice() {
        // 4.d3 had just let the bishop out: "still shut in behind the pawn on d2" was the
        // position before the move (Nova bot games, 2026-09-25).
        assertTrue(stale("your bishop on c1 is still shut in behind the pawn on d2", "d2d3"))
        assertFalse(stale("your bishop on c1 is still shut in behind the pawn on d2", "b1c3"))
        // A line is about the moves to come, and stands.
        assertFalse(stale("after Qg2, Rdg1, and the queen on g2 is trapped", "g2g3"))
        val said = "That works, though d4 was a touch better -- your bishop on c1 is still shut in behind " +
            "the pawn on d2. The lesson to keep: the centre is where pieces have the most room."
        val text = "Playable -- d4 was a little more precise -- your bishop on c1 is still shut in behind " +
            "the pawn on d2. Worth remembering: the centre is where pieces have the most room."
        assertEquals("Playable -- d4 was a little more precise.", fresh(text, said))
        assertEquals(text, fresh(text, "Nice."))
        var opener = "Playable -- d4 was a little more precise"
        assertEquals("$opener.", fresh(text, said, opener))
        // An opener with its own " -- " and no reason after it: the better move is not a reason,
        // and stays when the same move was just said ("Not that one." -- Nova bot games).
        opener = "Not that one -- Ng3 was better"
        val just = "That one cost you: Ng3 was better. After c6, they win a pawn."
        assertEquals("$opener.", fresh("$opener.", just.lowercase(), opener))
        assertEquals("$opener.", fresh("$opener -- it wins a pawn.", "$just It wins a pawn.", opener))
    }

    /**
     * `test_offline_coach.py`: four decisive moments in one game got the same sentence four times
     * (a master-game pass, 2026-09-21): each list is a bag, every wording once before any comes
     * again (Tai, 2026-09-25), and never the last one said.
     */
    @Test
    fun theTemplatesRotateAndNeverRepeatTheLast() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        for ((key, options) in listOf("cost" to COST_YOU, "fine" to FINE_TIGHTER, "keep" to KEEP_LEADS)) {
            val said = List(options.size * 8) { coach.bags.draw(key, options) }
            assertTrue("$key: $said", said.zipWithNext().all { (a, b) -> a != b })
            for (start in said.indices step options.size) {
                assertEquals(key, options.sorted(), said.subList(start, start + options.size).sorted())
            }
        }
        // The praise on a fine move too, the same bag whether a lesson follows or not.
        val said = List(10) { coach.verdictWord("best") }
        assertEquals(OPENERS.getValue("best").map { "$it." }.sorted(), said.take(5).sorted())
        // A list of one is said again, rather than nothing.
        assertEquals("only", coach.bags.draw("one", listOf("only")))
        assertEquals("only", coach.bags.draw("one", listOf("only")))
    }

    /**
     * `test_offline_coach.py`: the first hint is written into every position's facts and said in
     * few: its lead ("There is a mate in one here") is peeked from its bag as it is written and
     * taken only when it is said, so the hints never asked for do not use up the bag.
     */
    @Test
    fun theFirstHintsLeadIsDrawnWhenTheHintIsSaid() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(coach)
        for (ts in listOf(7.0, 7.5)) coach.onEvent(mateAnalysis(6, decisive = false), ts)  // twice; nothing asked
        messages(coach)
        val position = coach.positions.getValue(6)
        val lead = position.leads.getValue("lead:mate-one")
        assertEquals(IdeaWords.MATE_ONE_LEADS.sorted(), coach.bags.left("lead:mate-one").sorted())
        coach.ask("hint", 8.0)
        val out = messages(coach)
        assertEquals(out.map { it.text }.toString(), 1, out.size)
        assertTrue("$lead / ${out[0].text}", out[0].text.lowercase().startsWith(Situation.spoken(lead).lowercase()))
        val left = coach.bags.left("lead:mate-one")
        assertTrue(lead !in left && left.size == IdeaWords.MATE_ONE_LEADS.size - 1)
        assertEquals(emptyMap<String, String>(), position.leads)
    }

    /** `italian`: 1.e4 e5 2.Nf3 Nc6, White (the player) to move, the position analysed. */
    private fun italian(coach: Coach) {
        start(coach)
        val board = Board()
        for ((i, san) in listOf("e4", "e5", "Nf3", "Nc6").withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(
                    gameId = "g",
                    ply = ply,
                    san = san,
                    uci = move.uci(),
                    fenBefore = before,
                    fenAfter = board.fen(),
                    byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = 4,
                lines = listOf(Line(moves = listOf("f1b5"), depth = 12, expectedScore = 0.55)),
                decisive = false,
                onlyMove = false,
                tags = emptyList(),
            ),
            5.0,
        )
        messages(coach)
    }

    /**
     * "I'll play Qd1, blocking the check" answered the templates' hint, and got the hint back word
     * for word, three times (the students' round 5): a move proposed is examined aloud as a
     * question about it is; and a template already said since the last move is not said again
     * (desktop adc4b26).
     */
    @Test
    fun aMoveProposedInAStatementIsExaminedAloudAndNothingIsSaidTwice() {
        val engine = EngineLike { board, _ ->
            if (board.turn == WHITE) {
                listOf(Line(moves = listOf("f1b5"), depth = 14, expectedScore = 0.55))
            } else {
                listOf(Line(moves = listOf("f8c5", "d2d3"), depth = 14, expectedScore = 0.48))
            }
        }
        // The planner reads their words into a probe (first call); the model answers (second).
        val (coach, _) = coach(listOf("{\"probes\": [{\"label\": \"your Bc4\", \"moves\": [\"Bc4\"]}]}", "[p=1.0] Fine."))
        coach.engine = engine
        italian(coach)
        coach.ask("I'll play Bc4 and aim at f7.", 6.0)
        val out = messages(coach).map { it.text }
        assertTrue(out.toString(), out.isNotEmpty() && out[0].startsWith("your Bc4: after 3. Bc4 it is about equal"))
        // Offline, a statement naming no move gets the template once, and not again.
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        italian(offline)
        offline.ask("I think my position is fine.", 7.0)
        val first = messages(offline).map { it.text }
        offline.ask("I think my position is fine.", 8.0)
        val again = messages(offline).map { it.text }
        assertTrue("$first / $again", first.isNotEmpty() && again.intersect(first.toSet()).isEmpty())
    }

    /**
     * Each lesson taught its own moment and a principle of its own, and the threat-blind
     * student's notebook held five unrelated rules (the simulated students, 2026-09-26): the
     * second mistake of one cause in the session says the pattern, and every second one after
     * (desktop e7b5915).
     */
    @Test
    fun aCauseThatComesBackIsNamedWithTheHabitThatFixesIt() {
        val (coach, _) = coach(emptyList())
        start(coach)

        fun placed(ply: Int, cause: String, gameId: String = "g"): List<String> {
            coach.onEvent(
                ProfileEvidence(gameId = gameId, ply = ply, phase = "middlegame", expectedLoss = 0.2, cause = cause),
                ply.toDouble(),
            )
            return messages(coach).map { it.text }
        }

        assertEquals(emptyList<String>(), placed(9, Causes.MISSED_THREAT))
        assertEquals(emptyList<String>(), placed(13, Causes.POSITIONAL))
        assertEquals(emptyList<String>(), placed(21, Causes.MISSED_THREAT, gameId = "old")) // another game's
        val second = placed(25, Causes.MISSED_THREAT)
        assertTrue(second.toString(), second.size == 1 && "twice now" in second[0])
        assertTrue(second[0], "what their moves threatened" in second[0])
        assertTrue(second[0], "look at what their last move attacks" in second[0])
        assertEquals(emptyList<String>(), placed(31, Causes.MISSED_THREAT))
        val fourth = placed(37, Causes.MISSED_THREAT)
        assertTrue(fourth.toString(), "4 times now" in fourth[0])
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        // The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
        private const val QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
        private val QD1_LINE = listOf("c2d1", "d4d1", "d8d1", "a5e1", "d1e1")
    }
}
