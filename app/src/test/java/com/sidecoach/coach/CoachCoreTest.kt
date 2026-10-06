package com.sidecoach.coach

import com.sidecoach.analysis.DEFAULT_THRESHOLDS
import com.sidecoach.analysis.aboutAsGood
import com.sidecoach.analysis.clearlyWorse
import com.sidecoach.analysis.gradedGood
import com.sidecoach.analysis.labelMove
import com.sidecoach.chess.Board
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Move
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
import com.sidecoach.events.PuzzleRef
import com.sidecoach.events.Recommendation
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.events.TurnPending
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
import java.util.concurrent.CountDownLatch

/**
 * The coach's core logic, from the deep review of `coach.py` (2026-10-01), as the desktop's
 * `tests/test_coach_core.py` and `tests/test_alternatives.py` check it: what a takeback leaves,
 * what a combination is, how the player's words and the model's replies are kept to their moment,
 * and the best moves answered from real engine analyses (androidTest/assets/chess/alternatives).
 * Logic only: time is the policy's `ts` and the events; the one worker is held on a latch.
 */
class CoachCoreTest {

    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"
        val replies = ArrayDeque(replies)
        val prompts = ArrayList<Pair<String, List<ChatMessage>>>()

        /** Raise as a server that is down does (`ScriptedProvider(fail=True)`). */
        @Volatile var fail = false

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            synchronized(prompts) { prompts.add(system to messages.toList()) }
            if (fail) throw RuntimeException("server down")
            val text = synchronized(replies) { replies.removeFirstOrNull() } ?: "Keep looking."
            return Completion(text = text)
        }
    }

    private fun policy() = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0))

    private fun scripted(replies: List<String>, claims: ClaimsCheck? = null): Pair<Coach, ScriptedProvider> {
        val provider = ScriptedProvider(replies)
        return Coach(provider, policy = policy(), claims = claims) to provider
    }

    private fun offlineCoach(): Coach = Coach(OfflineProvider(), policy = policy())

    private fun start(coach: Coach, side: Pieces.Side = Pieces.Side.WHITE) {
        coach.onEvent(GameStarted(gameId = "g", userSide = side, startFen = Board().fen()), 0.0)
    }

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    private fun stuck(coach: Coach): List<CoachMessage> {
        val since = coach.policy.stuckFrom
        assertNotNull(since)
        coach.tick(since!! + coach.policy.stuckThreshold())
        return messages(coach)
    }

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
                    gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before,
                    fenAfter = board.fen(), byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
        }
        assertTrue(coach.userToMove && coach.board.fen() == SCHOLAR)
    }

    private fun mateAnalysis(ply: Int, decisive: Boolean = true) = Analysis(
        gameId = "g", fen = SCHOLAR, ply = ply,
        lines = listOf(
            Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
            Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
        ),
        decisive = decisive, onlyMove = true,
        tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
    )

    private fun played(coach: Coach, board: Board, ply: Int, uci: String, byUser: Boolean, ts: Double) {
        val move = Move.fromUci(uci)
        val before = board.fen()
        val san = board.san(move)
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g", ply = ply, san = san, uci = uci, fenBefore = before,
                fenAfter = board.fen(), byUser = byUser,
            ),
            ts,
        )
    }

    private fun judged(coach: Coach, ply: Int, uci: String, ts: Double) {
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = ply, uci = uci, expectedLoss = 0.0, severity = Severity.NONE,
                best = listOf(uci), label = Label.BEST,
            ),
            ts,
        )
    }

    /** The user's [san] from SCHOLAR (ply 7) at [ts], its verdict a second later. */
    private fun judgedAtScholar(coach: Coach, san: String, loss: Double, severity: Severity, label: Label, ts: Double) {
        val board = Board(SCHOLAR)
        val move = board.parseSan(san)!!
        board.push(move)
        coach.onEvent(
            MovePlayed(
                gameId = "g", ply = 7, san = san, uci = move.uci(), fenBefore = SCHOLAR,
                fenAfter = board.fen(), byUser = true,
            ),
            ts,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = 7, uci = move.uci(), expectedLoss = loss, severity = severity,
                best = listOf("h5f7"), label = label,
            ),
            ts + 1,
        )
    }

    private fun mateMissed(coach: Coach, ts: Double = 10.0) =
        judgedAtScholar(coach, "Bxf7+", 0.45, Severity.BLUNDER, Label.MISS, ts)

    private fun lookedLine(coach: Coach, board: Board, ply: Int, moves: List<String>, ts: Double) {
        coach.onEvent(
            Analysis(
                gameId = "g", fen = board.fen(), ply = ply,
                lines = listOf(Line(moves = moves, depth = 18, expectedScore = 0.5)),
                decisive = false, onlyMove = false,
            ),
            ts,
        )
    }

    private fun looked(coach: Coach, board: Board, ply: Int, ucis: List<String>, ts: Double) {
        coach.onEvent(
            Analysis(
                gameId = "g", fen = board.fen(), ply = ply,
                lines = ucis.mapIndexed { i, u -> Line(moves = listOf(u), depth = 18, expectedScore = 0.5 - i / 10.0) },
                decisive = false, onlyMove = false,
            ),
            ts,
        )
    }

    private fun directiveOf(provider: ScriptedProvider, index: Int = -1): String {
        val last = synchronized(provider.prompts) {
            provider.prompts[if (index < 0) provider.prompts.size + index else index].second
        }
        assertEquals(1, last.size)
        return last[0].content.substringAfterLast("# Task\n")
    }

    private fun harnessOf(provider: ScriptedProvider): String =
        synchronized(provider.prompts) { provider.prompts.last().second }[0].content

    private fun levels(out: List<CoachMessage>) = out.map { it.level }

    // 1. A takeback leaves the position at the hints it had

    @Test
    fun aTakebackLeavesThePositionAtTheHintsItHadBeforeTheWithdrawnMove() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        assertEquals(listOf(DisclosureLevel.IDEA), levels(stuck(coach)))
        mateMissed(coach, 20.0)
        messages(coach)
        assertEquals(DisclosureLevel.LINE, coach.positions[6]!!.disclosure.level)
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 30.0)
        val position = coach.positions[6]!!
        assertEquals("the idea was said, the rest not", DisclosureLevel.IDEA, position.disclosure.level)
        assertEquals(listOf(DisclosureLevel.FOCUS), levels(stuck(coach)))
    }

    @Test
    fun aTakebackWithNoHintSaidLeavesTheFirstHintTheIdea() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateMissed(coach, 20.0)
        messages(coach)
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 30.0)
        assertEquals(DisclosureLevel.SILENT, coach.positions[6]!!.disclosure.level)
        coach.ask("hint", 31.0)
        val answers = messages(coach)
        assertEquals(listOf(Trigger.USER_ASK), answers.map { it.trigger })
        assertEquals("the first hint is the idea, not the answer", DisclosureLevel.IDEA, answers[0].level)
    }

    @Test
    fun aTakebackKeepsWhatThePlayerAskedForBeforeTheMove() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.ask("just tell me the move", 8.0)
        messages(coach)
        assertEquals(DisclosureLevel.MOVE, coach.positions[6]!!.disclosure.level)
        mateMissed(coach, 20.0)
        messages(coach)
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 30.0)
        assertEquals(DisclosureLevel.MOVE, coach.positions[6]!!.disclosure.level)
    }

    // 2. Ordinary words open nothing

    @Test
    fun ordinaryWordsOpenNeitherTheMoveNorTheLineNorAHint() {
        val board = Board(SCHOLAR)
        for (said in listOf(
            "which move should I avoid?",
            "which move is the opponent threatening?",
            "the answer is Nf3, right?",
            "what do I do about their threat?",
            "what should I do about the bishop?",
            "what's next for me after castling?",
            "what comes next in the plan after I castle?",
            "I'm not stuck, just thinking",
            "I am not really stuck, I am looking at the queen",
            "I don't feel stuck",
        )) {
            assertEquals(said, null, unlockFor(said, board, listOf("Qxf7#")))
        }
        for ((said, unlock) in listOf(
            "show me the move" to Unlock.ASK_MOVE,
            "tell me the move" to Unlock.ASK_MOVE,
            "what's the best move?" to Unlock.ASK_MOVE,
            "what should I play?" to Unlock.ASK_MOVE,
            "what's the line?" to Unlock.ASK_LINE,
            "what comes next?" to Unlock.ASK_LINE,
            "I'm stuck" to Unlock.HINT,
            "give me a hint" to Unlock.HINT,
        )) {
            assertEquals(said, unlock, unlockFor(said, board, listOf("Qxf7#")))
        }
    }

    // 4. A reply is the moment it was asked in

    @Test
    fun aReplyCheckedWhileANewGameStartsIsStillTheOldGames() {
        val release = CountDownLatch(1)
        val (coach, _) = scripted(
            listOf("Yes, Qxf7 is mate."),
            claims = ClaimsCheck { _, _, _ ->
                release.await() // the worker's check waits for the new game
                emptyList()
            },
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.ask("Is Qxf7 mate?", 8.0)
        coach.onEvent(GameStarted(gameId = "h", userSide = Pieces.Side.WHITE, startFen = Board().fen()), 9.0)
        release.countDown()
        for (thread in Thread.getAllStackTraces().keys) if (thread.name == "sidecoach-claims") thread.join()
        val answers = messages(coach)
        assertEquals(listOf("Yes, Qxf7 is mate."), answers.map { it.text })
        assertEquals("the game it was asked in", "g", answers[0].gameId)
        assertTrue("no turn of the new game", coach.conversation.entries.none { it.kind == "coach" })
    }

    // That move after a bot's instant reply is the player's own

    private fun askedAfterTheReply(replyTs: Double): Pair<Coach, ScriptedProvider> {
        val (coach, provider) = scripted(listOf("Qxf7 was mate on the spot.", "It gave the bishop away for a pawn."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateMissed(coach, 10.0)
        messages(coach)
        val board = Board(SCHOLAR)
        board.push(board.parseSan("Bxf7+")!!)
        played(coach, board, 8, "e8f7", false, replyTs)
        looked(coach, board, 8, listOf("h5f3", "h5h4"), replyTs + 0.5)
        coach.ask("why was that bad?", replyTs + 2.0)
        messages(coach)
        return coach to provider
    }

    /**
     * `test_what_that_means_after_a_reply_is_the_models_to_decide_from_both_positions`. Not
     * decided by a threshold of two seconds in code any more (Tai, 2026-10-02: "why does instant
     * mean 2 seconds"): the model gets the sheet of the position the player faced and the one
     * after the reply, and how long the reply took, in the guidelines, fast or slow.
     */
    @Test
    fun whatThatMeansAfterAReplyIsTheModelsToDecideFromBothPositions() {
        for ((replyTs, took) in listOf(10.5 to "0.5 seconds", 40.0 to "30.0 seconds")) {
            val (coach, provider) = askedAfterTheReply(replyTs)
            val harness = harnessOf(provider)
            assertTrue("$replyTs", "The position they faced, before Bxf7+" in harness)
            assertTrue("# Position now" in harness)
            val directive = directiveOf(provider)
            assertTrue(
                "$replyTs: $directive",
                "The opponent's last move, Kxf7, came $took after the player's Bxf7+" in directive,
            )
            assertTrue("\"That\", \"it\" or \"that move\" may mean their own last move" in directive)
            assertFalse("They ask about the move they already played" in directive)
            assertTrue(coach.userToMove)
        }
    }

    // The best moves, and a hint asked once the move was given

    private fun load(name: String): Pair<Board, Analysis> {
        val json = fixture("chess/alternatives/$name.json")
        val lines = json.getJSONArray("lines")
        val analysis = Analysis(
            gameId = "g", fen = json.getString("fen"), ply = 0,
            lines = (0 until lines.length()).map { i ->
                val line = lines.getJSONObject(i)
                val moves = line.getJSONArray("moves")
                Line(
                    moves = (0 until moves.length()).map { moves.getString(it) },
                    depth = line.getInt("depth"),
                    expectedScore = line.getDouble("expected_score"),
                    mate = if (line.isNull("mate")) null else line.getInt("mate"),
                )
            },
            decisive = json.getBoolean("decisive"), onlyMove = json.getBoolean("only_move"),
        )
        return Board(analysis.fen) to analysis
    }

    private fun coachAt(name: String, replies: List<String>? = null): Pair<Coach, ScriptedProvider> {
        val (board, analysis) = load(name)
        val provider = ScriptedProvider(replies ?: emptyList())
        val coach = if (replies != null) Coach(provider, policy = policy()) else offlineCoach()
        val side = if (board.turn == com.sidecoach.chess.WHITE) Pieces.Side.WHITE else Pieces.Side.BLACK
        coach.onEvent(GameStarted(gameId = "g", userSide = side, startFen = board.fen(), kind = "position"), 0.0)
        coach.onEvent(analysis, 1.0)
        return coach to provider
    }

    /** The label the review gives a move this far behind the best, all else plain. */
    private fun gradeOf(loss: Double): Label = labelMove(
        loss, DEFAULT_THRESHOLDS, isBest = loss == 0.0, inBook = false, decisive = false,
        sacrifice = false, beforeScore = 0.5, opponentBlundered = false,
    )

    /**
     * "About equally good" is what the move labels grade Best or Excellent, "good" what they grade
     * Good (Tai, 2026-10-02: the grading bands, not 0.03 of our own): Bc5 and Nf6 about as good as
     * each other; Be7, 0.029 behind, and d6 good, a little behind; h6 an inaccuracy.
     * `test_a_quiet_position_has_moves_about_as_good_good_ones_and_none_clearly_worse`.
     */
    @Test
    fun aQuietPositionHasMovesAboutAsGoodGoodOnesAndNoneClearlyWorse() {
        val (board, analysis) = load("italian")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        assertEquals(listOf("Bc5", "Nf6"), facts.candidates)
        assertEquals(listOf("Be7", "d6"), facts.good)
        val top = analysis.lines[0].expectedScore
        val labels = analysis.lines.map { gradeOf(top - it.expectedScore) }
        assertEquals(listOf(Label.BEST, Label.EXCELLENT, Label.GOOD, Label.GOOD), labels.take(4))
        assertEquals(Label.INACCURACY, labels[4])
        for ((line, label) in analysis.lines.zip(labels)) {
            val loss = top - line.expectedScore
            assertEquals(label in listOf(Label.BEST, Label.EXCELLENT), aboutAsGood(loss))
            assertEquals(label in listOf(Label.BEST, Label.EXCELLENT, Label.GOOD), gradedGood(loss))
        }
        assertEquals(emptyList<Worse>(), worseMoves(board, analysis, board.turn))
        assertEquals("Bc5 and Nf6 are both fine here -- about equally good.", fineSaid(facts.candidates))
        assertEquals("Be7 and d6 are good too, a little behind.", goodSaid(facts.good))
    }

    /** `test_clearly_worse_is_a_mistake_or_worse_as_the_labels_grade_it`. */
    @Test
    fun clearlyWorseIsAMistakeOrWorseAsTheLabelsGradeIt() {
        for (loss in listOf(0.0, 0.03, 0.05, 0.0999, 0.10, 0.2, 0.5)) {
            assertEquals(loss.toString(), gradeOf(loss) in listOf(Label.MISTAKE, Label.BLUNDER), clearlyWorse(loss))
        }
    }

    @Test
    fun oneClearBestMoveIsTheOnlyFineOneAndTheOthersPassUpTheMate() {
        val (board, analysis) = load("scholar")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        assertEquals(listOf("Qxf7#"), facts.candidates)
        val worse = worseMoves(board, analysis, board.turn)
        assertEquals(listOf("Qd1", "Qe2"), worse.map { it.san })
        assertTrue(worse.all { it.wrong == "passes up the mate" && clearlyWorse(it.drop) })
        assertEquals("Qd1 passes up the mate; Qe2 passes up the mate", worseText(worse))
        assertEquals("Qxf7# is the move here.", fineSaid(facts.candidates))
    }

    @Test
    fun aMoveThatMissesAWonQueenSaysSo() {
        val (board, analysis) = load("queen_hangs")
        val worse = worseMoves(board, analysis, board.turn, limit = 6)
        assertEquals("Nc3", worse[0].san)
        assertTrue(worse.all { it.wrong == "passes up winning a queen and a pawn" })
        assertEquals(emptyList<Worse>(), worseMoves(Board(analysis.fen), analysis, BLACK))
    }

    @Test
    fun askedForTheBestMovesEveryMoveAboutAsGoodIsNamed() {
        for (said in listOf(
            "what are the best moves?",
            "any other good moves?",
            "what else works?",
            "which moves are fine here?",
        )) {
            assertEquals(said, Unlock.ASK_MOVE, unlockFor(said, load("italian").first, emptyList()))
            val (coach, _) = coachAt("italian")
            coach.ask(said, 2.0)
            val answer = messages(coach)
            assertEquals(said, 1, answer.size)
            assertEquals(
                said,
                "Bc5 and Nf6 are both fine here -- about equally good. " +
                    "Be7 and d6 are good too, a little behind.",
                answer[0].text,
            )
            assertEquals(DisclosureLevel.MOVE, answer[0].level)
        }
    }

    @Test
    fun theModelIsToldToReadEveryLineAndToNameTheMovesThatAreFine() {
        val (coach, provider) = coachAt("italian", listOf("Bc5 and Nf6 are both fine."))
        coach.ask("what are the best moves?", 2.0)
        messages(coach)
        val directive = directiveOf(provider)
        // The AI's guidelines, whatever the words: every line, every move about level named, and
        // the moves about as good as the facts list them (no keyword picks this).
        assertTrue(directive, "read all of the engine lines" in directive)
        assertTrue(directive, "The moves about equally good here: Bc5, Nf6." in directive)
        assertTrue(directive, "Good too, a little behind the best: Be7, d6." in directive)
        assertTrue(directive, "name every move about level with the best" in directive)
        assertTrue(directive, "no percentages, scores or move numbers" in directive)
    }

    @Test
    fun askedWhetherAMoveIsAlsoGoodTheModelGetsTheFineMoves() {
        val (coach, provider) = coachAt("italian", listOf("Yes, Be7 is good too."))
        coach.ask("is Be7 also good?", 2.0)
        messages(coach)
        val directive = directiveOf(provider)
        assertTrue(directive, "or whether another move works too" in directive)
        assertTrue(directive, "The moves about equally good here: Bc5, Nf6." in directive)
        assertTrue(directive, "Good too, a little behind the best: Be7, d6." in directive)
        assertTrue(directive, "read all of the engine lines" in directive)
        // Offline, at the move level, the template says what it is: good, a little behind.
        val (offline, _) = coachAt("italian")
        offline.ask("what are the best moves?", 2.0)
        messages(offline)
        offline.ask("is Be7 also good?", 3.0)
        assertEquals("Yes, Be7 is good too, a little behind Bc5.", messages(offline).single().text)
    }

    @Test
    fun oneClearBestMoveIsNamedWithWhatGoesWrongWithTheOthers() {
        val (coach, _) = coachAt("scholar")
        coach.ask("what are the best moves?", 2.0)
        assertEquals(
            "Qxf7# is the move here. Qd1 is clearly worse: it passes up the mate. " +
                "Qe2 is clearly worse: it passes up the mate.",
            messages(coach)[0].text,
        )
        val (again, provider) = coachAt("scholar", listOf("Qxf7 is mate; the others pass it up."))
        again.ask("what are the best moves?", 2.0)
        messages(again)
        val directive = directiveOf(provider)
        assertFalse(directive, "The moves about equally good here" in directive) // one clear best
        assertTrue(
            directive,
            "Clearly worse here, and what goes wrong with them: Qd1 passes up the mate" in directive,
        )
        assertTrue(directive, "Qe2 passes up the mate" in directive)
    }

    @Test
    fun theBestMoveAskedInTheSingularNamesTheNearEqualOnesToo() {
        val (coach, _) = coachAt("italian")
        coach.ask("what's the best move?", 2.0)
        val text = messages(coach)[0].text
        assertTrue(text, "Bc5" in text && "Nf6 is about as good." in text && "Be7" !in text)
        val (_, provider) = coachAt("italian", listOf("Bc5.")).also { (c, _) ->
            c.ask("what's the best move?", 2.0)
            messages(c)
        }
        assertTrue("The moves about equally good here: Bc5, Nf6." in directiveOf(provider))
        val (_, single) = coachAt("scholar", listOf("Qxf7 mates.")).also { (c, _) ->
            c.ask("what's the best move?", 2.0)
            messages(c)
        }
        assertFalse("about equally good here" in directiveOf(single)) // one clear best
    }

    /**
     * Be7 is 0.029 behind the best line, graded Good: "is Be7 a good move?" was refused as not among
     * the engine's best moves. It is a good move -- and called the best, it is not (Tai, 2026-10-02).
     * `test_the_claim_check_says_a_good_move_is_good_and_not_the_best`.
     */
    @Test
    fun theClaimCheckSaysAGoodMoveIsGoodAndNotTheBest() {
        val (board, analysis) = load("italian")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        fun verdict(move: String, text: String, engine: ClaimEngine? = null): Pair<ClaimResult, String> {
            val v = verifyClaims(
                listOf(Claim(ClaimKind.BEST, null, null, move, text)),
                board,
                facts = if (engine == null) facts else null,
                engine = engine,
            ).single()
            return v.result to v.reason
        }
        assertEquals(ClaimResult.TRUE, verdict("Nf6", "Nf6 is a good move").first)
        val good = verdict("Be7", "Be7 is a good move")
        assertEquals(
            ClaimResult.TRUE to "Be7 is a good move here, a little behind the engine's best (Bc5 and Nf6).",
            good,
        )
        val best = verdict("Be7", "Be7 is the best move")
        assertTrue(best.second, best.first == ClaimResult.FALSE && "is a good move here, a little behind" in best.second)
        assertEquals(ClaimResult.TRUE, verdict("d6", "d6 is a good move").first)
        assertEquals("an inaccuracy is no good move", ClaimResult.FALSE, verdict("h6", "h6 is a good move").first)
        // The engine, asked when the facts are not this position's: the fixture's lines, in any order.
        assertEquals(good, verdict("Be7", "Be7 is a good move", ClaimEngine { _, _ -> analysis.lines.reversed() }))
    }

    @Test
    fun aHintAskedAfterTheMoveWasGivenSaysWhyItBeatsWhatIsWorse() {
        val (coach, _) = coachAt("scholar")
        val said = ArrayList<CoachMessage>()
        for (n in 0 until 4) {
            coach.ask("hint", 2.0 + n)
            said.add(messages(coach)[0])
        }
        assertEquals(
            listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS, DisclosureLevel.MOVE, DisclosureLevel.MOVE),
            levels(said),
        )
        assertTrue("Qxf7" in said[2].text && "clearly worse" !in said[2].text)
        assertTrue(said[3].text, "Qd1 is clearly worse: it passes up the mate." in said[3].text)
        // And to the model, as a guideline: said by what the reply gave (the third names the move).
        val (again, provider) = coachAt(
            "scholar",
            listOf("a", "b", "Qxf7 is mate.", "Qxf7 is mate; Qd1 passes it up."),
        )
        for (n in 0 until 4) {
            again.ask("hint", 2.0 + n)
            messages(again)
        }
        val directive = directiveOf(provider)
        assertTrue(directive, "has been given already, in your turns above: it is Qxf7#" in directive)
        assertTrue(
            directive,
            "Clearly worse here, and what goes wrong with them: Qd1 passes up the mate" in directive,
        )
        assertTrue(directive, "instead of something that looks as good but is clearly worse" in directive)
        assertFalse("has been given already" in directiveOf(provider, 2))
    }

    // 5, 13. A lesson on a withdrawn move is shown all the same, and is no turn of the game now

    /**
     * `test_a_lesson_on_a_withdrawn_move_is_shown_in_its_place_not_recorded_not_spoken`. The deep
     * coach.py review (2026-10-01) dropped the lesson on the move taken back. Tai, 2026-10-02:
     * "There is no need to drop any replies, the text message should come out in the same correct
     * order. The only thing that might be good is to interrupt the tts": it is shown, marked late
     * (the voice does not say it), and is no turn of the game now.
     */
    @Test
    fun aLessonOnAWithdrawnMoveIsShownInItsPlaceNotRecordedNotSpoken() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateMissed(coach, 10.0) // the lesson is queued
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 12.0)
        val shown = messages(coach)
        assertTrue(shown.size == 1 && shown[0].trigger == Trigger.POST_MISTAKE)
        assertTrue("for the voice: not spoken", coach.spoken.last().late)
        assertEquals("no turn of the moment now", emptyList<String>(), coach.saidSinceMove())
    }

    /**
     * The offline coach, the same mistake six times over (taken back each time): every lesson is
     * a lesson, none asks what they expected or saw (Tai, 2026-10-02: "very rude and super
     * unwelcoming"; `test_no_lesson_on_a_mistake_asks_what_the_player_expected`).
     */
    @Test
    fun noOfflineLessonAsksWhatThePlayerExpected() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        messages(coach)
        val lessons = ArrayList<String>()
        for (n in 0 until 6) {
            mateMissed(coach, 10.0 + 10 * n)
            lessons += messages(coach).filter { it.trigger == Trigger.POST_MISTAKE }.map { it.text }
            coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 15.0 + 10 * n)
            messages(coach)
        }
        assertEquals(lessons.toString(), 6, lessons.size)
        val asking = Regex("expect|what did you|did you see|were you", RegexOption.IGNORE_CASE)
        for (text in lessons) assertFalse(text, asking.containsMatchIn(text))
    }

    // 6. A question does not delete the lesson on a mistake or the review

    private fun requests(coach: Coach, vararg triggers: Trigger): List<CoachRequest> {
        val position = coach.positions[6]!!
        return triggers.map { trigger ->
            CoachRequest(
                trigger = trigger, level = DisclosureLevel.IDEA, system = "",
                messages = listOf(ChatMessage("user", "x")), facts = position.facts!!,
                board = position.board, fallback = trigger.slug, ply = 6, gameId = "g",
            )
        }
    }

    @Test
    fun aQuestionDropsNothingQueued() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val speaker = Speaker(OfflineProvider())
        for (request in requests(
            coach, Trigger.STUCK, Trigger.POST_MISTAKE, Trigger.REVIEW, Trigger.POST_MOVE, Trigger.USER_ASK,
        )) {
            speaker.submit(request)
        }
        // Nothing is dropped (Tai, 2026-10-02); the stuck hint, said first, stays first.
        assertEquals(5, speaker.pending)
        speaker.runPending()
        assertEquals(
            listOf(
                Trigger.STUCK,
                Trigger.USER_ASK,
                Trigger.POST_MISTAKE,
                Trigger.REVIEW,
                Trigger.POST_MOVE,
            ),
            speaker.poll().map { it.message.trigger },
        )
    }

    // 7. A failure while answering is counted, and the question is answered

    @Test
    fun aFailureBuildingAReplyIsCountedAndTheTemplateIsSaid() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        // The model's own failure is `speakNow`'s; what `answer` guards is the rest of the building:
        // here the draft check breaks, after the model has answered.
        val broken = Speaker(
            ScriptedProvider(listOf("[p=1.0] Qxf7 is mate.")),
            draftCheck = DraftCheck { _, _, _, _ -> throw IllegalStateException("boom") },
        )
        broken.submit(requests(coach, Trigger.USER_ASK)[0])
        broken.runPending()
        val said = broken.poll()
        assertEquals(listOf("user_ask"), said.map { it.message.text })
        assertEquals(1, broken.failures)
        assertEquals("reply: IllegalStateException: boom", broken.lastError)
        assertEquals(listOf("the reply failed: IllegalStateException: boom"), said[0].problems)
    }

    @Test
    fun aFailureOnTheClaimsThreadIsCountedAndTheQuestionAnswered() {
        val (coach, _) = scripted(
            listOf("Yes, Qxf7 is mate."),
            claims = ClaimsCheck { _, _, _ -> throw IllegalStateException("boom") },
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.ask("Is Qxf7 mate?", 8.0)
        for (thread in Thread.getAllStackTraces().keys) if (thread.name == "sidecoach-claims") thread.join()
        // The claims check's own failure is caught (no verdicts) and the model answers; it is
        // counted and kept for the log, never swallowed
        // (`test_a_failed_claims_check_is_counted_and_the_question_answered_without_it`).
        assertEquals(listOf("Yes, Qxf7 is mate."), messages(coach).map { it.text })
        assertEquals(1, coach.speaker.failures)
        assertEquals("claims check: IllegalStateException: boom", coach.speaker.lastError)
    }

    // 8. The pattern and the practice are unprompted too

    @Test
    fun thePatternAndThePracticeKeepQuietInQuietModeAndInBullet() {
        val recommended = Recommendation(
            gameId = "g", motif = "fork", motifWords = "fork", reason = "forks cost you most", level = 1500,
            puzzles = listOf(PuzzleRef(id = "p1", url = "https://example.org/p1", rating = 1500)),
        )
        fun evidence(coach: Coach) {
            for (ply in listOf(9, 13, 17)) {
                coach.onEvent(
                    ProfileEvidence(
                        gameId = "g", ply = ply, phase = "middlegame", expectedLoss = 0.2,
                        cause = Causes.MISSED_THREAT,
                    ),
                    ply.toDouble(),
                )
            }
        }
        for (how in listOf("quiet", "bullet")) {
            val coach = offlineCoach()
            start(coach)
            if (how == "quiet") coach.quiet = true else coach.policy.timeControl = TimeControl.BULLET
            evidence(coach)
            coach.onEvent(recommended, 20.0)
            assertEquals(how, emptyList<CoachMessage>(), messages(coach))
        }
        val coach = offlineCoach()
        start(coach)
        evidence(coach)
        coach.onEvent(recommended, 20.0)
        val said = messages(coach).map { it.text }
        assertTrue(said.toString(), said.size == 2 && "Something worth working on" in said[0] && "For practice" in said[1])
    }

    // 9. Two questions while the position is read: both are answered

    @Test
    fun twoQuestionsHeldForTheEngineAreBothAnsweredInOrder() {
        val coach = offlineCoach()
        start(coach)
        coach.ask("what's the best move?", 1.0)
        coach.ask("what is my opponent threatening?", 2.0)
        assertEquals(
            listOf("One moment -- still looking at this position."),
            messages(coach).map { it.text },
        )
        val (board, analysis) = load("scholar")
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = board.fen(), kind = "position"), 3.0)
        coach.onEvent(analysis, 4.0)
        val answered = messages(coach)
        assertEquals("the last question did not overwrite the first", 2, answered.size)
        assertTrue("Qxf7" in answered[0].text)
    }

    /**
     * The reconciler cannot yet say whose move it is ([TurnPending], the desktop's
     * docs/reconciler-model.md section 6): nothing unasked is said, and a question waits --
     * answered for the wrong side it would teach the wrong move (Tai, 2026-10-05: the turn order
     * should never fail). Once the side to move is known again the question is answered.
     * `test_a_question_while_the_side_to_move_is_not_known_waits_for_it`.
     */
    @Test
    fun aQuestionWhileTheSideToMoveIsNotKnownWaitsForIt() {
        val (board, analysis) = load("scholar")
        val coach = offlineCoach()
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = board.fen(), kind = "position"), 1.0)
        coach.onEvent(analysis, 2.0)
        messages(coach)
        coach.onEvent(TurnPending(gameId = "g", pending = true), 3.0)
        coach.ask("what's the best move?", 4.0)
        assertEquals(listOf("One moment -- following the moves on the board."), messages(coach).map { it.text })
        coach.tick(120.0) // stuck long past a hint, were the side to move known
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.onEvent(TurnPending(gameId = "g", pending = false), 121.0)
        val answered = messages(coach)
        assertTrue(answered.toString(), answered.size == 1 && "Qxf7" in answered[0].text)
    }

    // 14. The review's sentence names the labels it counts

    @Test
    fun theReviewSentenceNamesEveryLabelItCounts() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        judgedAtScholar(coach, "Qxf7#", 0.0, Severity.NONE, Label.BEST, 10.0)
        coach.ask("how did I do overall?", 12.0)
        assertTrue(messages(coach).last().text.startsWith("1 of your 1 moves were best."))
    }

    // 3. A combination is a forcing sequence or one plan, not any run of best moves

    private fun withPlans(plans: Map<Int, String?>): Coach {
        val coach = offlineCoach()
        start(coach)
        val board = Board()
        val steps = listOf("e2e4", "e7e5", "g1f3", "b8c6")
        for ((ply, uci) in steps.withIndex()) {
            val mine = ply % 2 == 0
            if (mine) {
                lookedLine(coach, board, ply, listOf(uci), 100.0 * ply)
                val position = coach.positions[ply]!!
                position.facts = position.facts!!.copy(
                    plan = plans[ply], theme = null,
                    nature = "quiet -- no tactic to find; the move improves your position",
                )
                if (ply == 0) {
                    assertEquals(listOf(DisclosureLevel.IDEA), levels(stuck(coach)))
                    assertEquals(listOf(DisclosureLevel.FOCUS), levels(stuck(coach)))
                }
            }
            played(coach, board, ply + 1, uci, mine, 100.0 * ply + 50)
            if (mine) judged(coach, ply + 1, uci, 100.0 * ply + 51)
            messages(coach)
        }
        lookedLine(coach, board, 4, listOf("f1b5"), 500.0)
        val position = coach.positions[4]!!
        position.facts = position.facts!!.copy(
            plan = plans[4], nature = "quiet -- no tactic to find; the move improves your position",
        )
        return coach
    }

    @Test
    fun aBestMoveOnAnotherPlanIsANewCombinationAndGetsItsStuckHint() {
        val coach = withPlans(
            mapOf(0 to "fighting for the centre", 2 to "finishing your development", 4 to "getting your king safe")
        )
        assertEquals(listOf(DisclosureLevel.IDEA), levels(stuck(coach)))
    }

    @Test
    fun theSamePlanAcrossBestMovesIsStillAnotherPosition() {
        val coach = withPlans(
            mapOf(
                0 to "finishing your development",
                2 to "finishing your development",
                4 to "finishing your development",
            )
        )
        // Tai, 2026-10-02: the opponent moved, so it is another position, and a hint comes.
        assertEquals(listOf(DisclosureLevel.IDEA), levels(stuck(coach)))
    }

    @Test
    fun aForcingLineFollowedIsOneCombinationAndLeftIsANewOne() {
        for (followed in listOf(true, false)) {
            val coach = offlineCoach()
            start(coach)
            val board = Board()
            lookedLine(coach, board, 0, listOf("e2e4", "e7e5", "g1f3"), 0.0)
            val first = coach.positions[0]!!
            first.facts = first.facts!!.copy(
                nature = "forcing -- the answer is a check, a capture or a promotion", plan = null, theme = null,
            )
            stuck(coach)
            stuck(coach)
            played(coach, board, 1, "e2e4", true, 50.0)
            judged(coach, 1, "e2e4", 50.5)
            played(coach, board, 2, if (followed) "e7e5" else "c7c5", false, 51.0)
            messages(coach)
            lookedLine(coach, board, 2, listOf("g1f3"), 52.0)
            val second = coach.positions[2]!!
            second.facts = second.facts!!.copy(plan = null, theme = null)
            if (followed) {
                assertEquals(emptyList<CoachMessage>(), stuck(coach))
            } else {
                assertEquals(listOf(DisclosureLevel.IDEA), levels(stuck(coach)))
            }
        }
    }

    // With the AI coach on, the model reads the question: no keyword picks its directive, forces
    // or removes its reply, or sets the rung (Tai, 2026-10-02). The keyword rules are the offline
    // coach's. `tests/test_ai_reads_the_question.py`.

    /** `test_a_question_the_keywords_would_miss_reaches_the_model_with_the_whole_sheet`. */
    @Test
    fun aQuestionTheKeywordsWouldMissReachesTheModelWithTheWholeSheet() {
        for (said in listOf("whats the play here lol", "ok so what now", "gimme the good stuff", "thoughts?", "idk man")) {
            val reply = "Qxf7 is mate: their king has no square. Bxf7+ looks as good and passes it up."
            val (coach, provider) = scripted(listOf(reply))
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
            assertEquals("nothing unasked in a quiet position", emptyList<CoachMessage>(), messages(coach))
            coach.ask(said, 8.0)
            val out = messages(coach)
            // Nothing refuses or blanks it: the model's own words, once, whatever they asked.
            assertEquals(said, listOf(reply), out.map { it.text })
            assertTrue(said, coach.spoken.last().problems.isEmpty() && coach.speaker.fallbacks == 0)
            assertEquals(said, 1, provider.prompts.size)
            val harness = harnessOf(provider)
            // The whole sheet, whatever the rung: every engine line, the facts, the answer among
            // them.
            val lines = harness.substringAfter("## Engine lines").substringBefore("\n\n")
            assertTrue(said, "Qxf7#" in lines && "Bxf7+" in lines)
            assertTrue(said, "goal: the player can mate in one move" in harness)
            assertTrue(said, "PLAYER said: \"$said\"" in harness)
            val directive = directiveOf(provider)
            // The guidelines, whatever the words, and the ladder as it stands -- never a branch
            // chosen by a keyword.
            assertTrue(said, AI_GUIDELINES in directive)
            assertTrue(said, "nothing yet" in directive)
            assertTrue(said, "They ask" !in directive && "They asked" !in directive)
        }
    }

    /** `test_the_ladder_follows_what_the_reply_said_not_what_the_question_was_guessed_to_mean`. */
    @Test
    fun theLadderFollowsWhatTheReplySaidNotWhatTheQuestionWasGuessedToMean() {
        val (coach, provider) = scripted(
            listOf(
                "Their king is nearly boxed in: look for a check that ends it.",
                "Qxf7# is mate.",
                "It is still Qxf7#, taking the pawn that guards their king.",
            ),
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        val position = coach.positions[6]!!
        // A question the model reads -- not the move or the line asked for outright, which open
        // them exactly ([askedOutright], 2026-10-03) -- climbs the rung its reply said.
        coach.ask("what's going on here?", 8.0)
        var out = messages(coach)
        assertEquals("the reply gave no move", DisclosureLevel.IDEA, out[0].level)
        assertEquals(DisclosureLevel.IDEA, position.disclosure.level)
        coach.ask("hint", 9.0)
        out = messages(coach)
        assertTrue("the idea was said", "nothing yet" !in directiveOf(provider, 1))
        assertEquals(DisclosureLevel.MOVE, out[0].level)
        assertEquals(DisclosureLevel.MOVE, position.disclosure.level)
        // Said, it stays said: the ladder never falls, and the move is given again with why.
        coach.ask("another hint", 10.0)
        messages(coach)
        assertTrue(
            "The move has been given already, in your turns above: it is Qxf7#" in directiveOf(provider, 2),
        )
        assertEquals(DisclosureLevel.MOVE, position.disclosure.level)
    }

    /** `test_a_question_about_a_move_asked_the_model_is_not_withheld_by_the_rung`. */
    @Test
    fun aQuestionAboutAMoveAskedTheModelIsNotWithheldByTheRung() {
        val (coach, provider) = scripted(listOf("Qxf7# ends it at once; nothing follows."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("show me the line", 8.0)
        val out = messages(coach)
        assertTrue(out.isNotEmpty() && out[0].text.startsWith("Qxf7#"))
        assertTrue(coach.spoken.last().problems.isEmpty())
        assertTrue("The line, or what happens next" in directiveOf(provider))
    }

    /** `test_the_offline_coach_keeps_its_keyword_rules`. */
    @Test
    fun theOfflineCoachKeepsItsKeywordRules() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("whats the play here lol", 8.0)
        val plain = messages(coach).single()
        assertTrue(plain.level == DisclosureLevel.IDEA && "Qxf7" !in plain.text)
        coach.ask("what should I play?", 9.0)
        val move = messages(coach).single()
        assertTrue(move.level == DisclosureLevel.MOVE && "Qxf7" in move.text)
    }

    /** `test_a_reply_after_the_player_moved_again_is_late_but_shown_in_order`. */
    @Test
    fun aReplyAfterThePlayerMovedAgainIsLateButShownInOrder() {
        val (coach, _) = scripted(listOf("Qxf7# is mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("what should I play?", 8.0) // the request is built; the model has not answered
        assertFalse(coach.movedOn(coach.currentMoment()))
        judgedAtScholar(coach, "Qxf7#", 0.0, Severity.NONE, Label.BEST, 9.0) // the player moves
        val out = messages(coach).filter { it.trigger == Trigger.USER_ASK }
        assertEquals("shown, in its place", listOf("Qxf7# is mate at once."), out.map { it.text })
        val asked = coach.spoken.first { it.message.trigger == Trigger.USER_ASK }
        assertTrue("for the voice: not spoken", asked.late)
        // The opponent's reply alone is no new moment.
        val moment = coach.currentMoment()
        val board = Board(coach.board.fen())
        played(coach, board, 8, "e8e7", false, 11.0)
        assertFalse(coach.movedOn(moment))
    }

    /**
     * `test_a_question_held_for_the_engine_is_answered_after_the_player_moves`: "One moment" was a
     * promise, and the player moving before the engine's look threw the question away (Tai,
     * 2026-10-02: no reply is dropped).
     */
    @Test
    fun aQuestionHeldForTheEngineIsAnsweredAfterThePlayerMoves() {
        val (coach, provider) = scripted(listOf("Develop the knight and keep the centre."))
        playToScholar(coach)
        coach.ask("what's the plan here?", 7.0) // no analysis yet: held
        assertTrue(messages(coach).single().text.startsWith("One moment"))
        val board = Board(coach.board.fen())
        played(coach, board, 7, "b1c3", true, 8.0)
        played(coach, board, 8, "a7a6", false, 9.0)
        assertTrue("nothing answered before the position is read", synchronized(provider.prompts) { provider.prompts.isEmpty() })
        looked(coach, board, 8, listOf("d2d3"), 10.0)
        val out = messages(coach).filter { it.trigger == Trigger.USER_ASK }
        assertEquals(listOf("Develop the knight and keep the centre."), out.map { it.text })
        assertTrue("PLAYER said: \"what's the plan here?\"" in harnessOf(provider))
    }

    /**
     * A reply asked for before the player moved and landing after it is shown (late), and it is
     * this game's -- but about the position they moved from: written in the record, not said at
     * the position now. Recorded there, the next question was told "do not repeat the verdict",
     * and its "?" made the next words its answer.
     * `test_a_reply_that_lands_after_the_players_move_is_the_records_not_said_here`.
     */
    @Test
    fun aReplyThatLandsAfterThePlayersMoveIsTheRecordsNotSaidHere() {
        val reply = "Look at f7: what does their king have left?"
        val (coach, provider) = scripted(listOf(reply, "The bishop check gave the king a square."))
        playToScholar(coach)
        coach.quiet = true // no word on the move: the late reply is the only coach turn
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("what should I aim for?", 8.0) // built; the model has not answered yet
        mateMissed(coach, 9.0) // the player moves first
        val board = Board(SCHOLAR)
        board.push(board.parseSan("Bxf7+")!!)
        looked(coach, board, 7, listOf("e8e7"), 9.5)
        val out = messages(coach).filter { it.trigger == Trigger.USER_ASK }
        assertEquals("shown, in its place", listOf(reply), out.map { it.text })
        assertTrue(coach.spoken.first().late)
        assertEquals("not said at the position now", emptyList<String>(), coach.saidSinceMove())
        assertFalse(coach.lastTurnAsked() || coach.answeringOurQuestion("the king"))
        assertTrue("in the record", reply in coach.conversation.render().substringBefore("# Position now"))
        coach.ask("why was that bad?", 10.0)
        messages(coach)
        val directive = directiveOf(provider)
        assertFalse("You have spoken since the last move already" in directive)
        assertFalse("Your last turn above ended with a question" in directive)
    }

    /**
     * The fallback's principle was counted as said for every hint asked of the AI, though the
     * fallback is said only when the reply fails. Offline the template is the answer, and is
     * counted. `test_a_question_to_the_ai_charges_no_rule_its_fallback_holds`.
     */
    @Test
    fun aQuestionToTheAiChargesNoRuleItsFallbackHolds() {
        val (coach, _) = scripted(listOf("Their king is nearly out of squares."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("hint", 8.0)
        assertEquals(listOf("Their king is nearly out of squares."), messages(coach).map { it.text })
        assertTrue(coach.rules.isEmpty() && coach.lastRule == null)
        val offline = offlineCoach()
        playToScholar(offline)
        offline.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(offline)
        offline.ask("hint", 8.0)
        val said = messages(offline).single()
        assertTrue("their king runs out of squares" in said.text && offline.rules.isNotEmpty())
    }

    /** `said_level`: what a reply said, from its words. */
    @Test
    fun theRungAReplySaidIsReadFromItsWords() {
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "opening",
            target = "the opponent's king on e8", piece = "the player's queen on h5",
            goal = "the player can mate in one move", candidates = listOf("Qxf7#"), line = listOf("Qxf7#"),
        )
        assertEquals(DisclosureLevel.MOVE, saidLevel("Qxf7 is mate.", facts))
        assertEquals(DisclosureLevel.FOCUS, saidLevel("Their king on e8 has no square left.", facts))
        assertEquals(DisclosureLevel.SILENT, saidLevel("Look for a check.", facts))
        val long = facts.copy(line = listOf("Qxf7#", "Kd8"))
        assertEquals(DisclosureLevel.LINE, saidLevel("Qxf7 and then Kd8 is forced.", long))
        // A pawn move is named only as a whole move: "e4" is not named by "Nxe4" or "Qe4" (the
        // phone read it as a square and rated such replies MOVE; the round-3 parity audit).
        val pawn = facts.copy(target = null, piece = null, candidates = listOf("e4"), line = listOf("e4", "e5"))
        assertEquals(DisclosureLevel.SILENT, saidLevel("Nxe4 wins a pawn.", pawn))
        assertEquals(DisclosureLevel.SILENT, saidLevel("Qe4 is fine.", pawn))
        assertEquals(DisclosureLevel.MOVE, saidLevel("e4 grabs the centre.", pawn))
        assertEquals(DisclosureLevel.LINE, saidLevel("After e4 e5 it is level.", pawn))
        val disclosure = Disclosure()
        assertEquals(DisclosureLevel.IDEA, disclosure.said(DisclosureLevel.IDEA))
        assertEquals(null, disclosure.said(DisclosureLevel.IDEA))
        assertEquals(DisclosureLevel.MOVE, disclosure.said(DisclosureLevel.MOVE))
        assertEquals("never lower after higher", null, disclosure.said(DisclosureLevel.FOCUS))
        assertEquals(DisclosureLevel.MOVE, disclosure.level)
        assertEquals(null, Disclosure().said(DisclosureLevel.SILENT))
    }

    /**
     * Two words of six letters shared with the goal or theme read as the idea (a magic number, Tai
     * 2026-10-02): now the reply's squares, moves, pieces, mate and promotion are read against what
     * the idea's facts name, and words alone climb nothing.
     * `test_the_rung_is_what_the_reply_names_never_a_count_of_shared_words`.
     */
    @Test
    fun theRungIsWhatTheReplyNamesNeverACountOfSharedWords() {
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "opening",
            goal = "you can mate in one move",
            theme = "a mating attack",
            mechanism = "a mating attack: their king runs out of squares",
            key = "Their king on e8 has one free square, and you have three checks",
            target = "their king on e8, whose only free square is e7; you have three checks",
            piece = "your queen on h5",
            candidates = listOf("Qxf7#"), line = listOf("Qxf7#"),
        )
        assertEquals(DisclosureLevel.SILENT, saidLevel("Every attack needs squares.", facts))
        assertEquals(DisclosureLevel.SILENT, saidLevel("You are winning, and the material is level.", facts))
        assertEquals(DisclosureLevel.IDEA, saidLevel("Their king is short of air.", facts))
        assertEquals(DisclosureLevel.IDEA, saidLevel("There is a mate here somewhere.", facts))
        assertEquals(DisclosureLevel.FOCUS, saidLevel("Your queen can do it.", facts))
        assertEquals(DisclosureLevel.SILENT, saidLevel("Your knight is fine where it is.", facts))
        val trap = facts.copy(key = null, trap = "hxg4 looks like it wins, but Qh5 mates")
        assertEquals(DisclosureLevel.IDEA, saidLevel("Careful with hxg4.", trap))
    }

    /**
     * "am I winning?" answered "You are winning, and the material is level." raised the ladder to
     * the idea, and the stuck hint after it was the second hint, the idea never said (Tai,
     * 2026-10-02: the ladder rises only when the reply hints).
     * `test_an_answer_that_names_nothing_of_the_idea_leaves_the_ladder_where_it_was`.
     */
    @Test
    fun anAnswerThatNamesNothingOfTheIdeaLeavesTheLadderWhereItWas() {
        val (coach, provider) = scripted(
            listOf("You are winning, and the material is level.", "Their king is nearly out of squares."),
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        val position = coach.positions[6]!! // the position the player faces
        coach.ask("am I winning?", 8.0)
        assertEquals(listOf("You are winning, and the material is level."), messages(coach).map { it.text })
        assertEquals("nothing of the idea was said", DisclosureLevel.SILENT, position.disclosure.level)
        val hint = stuck(coach).single()
        assertTrue(hint.trigger == Trigger.STUCK && hint.level == DisclosureLevel.IDEA)
        assertEquals(2, provider.prompts.size)
    }

    /**
     * A failed AI reply fell back to nothing when a pattern read the player's words as a statement
     * after the coach had spoken (Tai, 2026-10-02: something true is still said; no pattern blanks a
     * fallback). `test_a_failed_reply_after_a_nod_is_never_silence`.
     */
    @Test
    fun aFailedReplyAfterANodIsNeverSilence() {
        val (coach, provider) = scripted(listOf("Their king on e8 has one square, e7."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("hint", 8.0)
        assertEquals(listOf("Their king on e8 has one square, e7."), messages(coach).map { it.text })
        provider.fail = true // the server goes down: every reply from here fails
        coach.ask("ok I see it now", 9.0) // a nod: no question in it
        val nod = messages(coach).single()
        assertTrue("the template, not silence", nod.trigger == Trigger.USER_ASK && nod.text.isNotEmpty())
    }

    /**
     * The template of a failed AI reply was blanked when the coach had said it since the move: asked
     * twice, the second answer was nothing. `test_a_failed_reply_whose_template_repeats_is_still_said`.
     */
    @Test
    fun aFailedReplyWhoseTemplateRepeatsIsStillSaid() {
        val (coach, provider) = scripted(emptyList())
        provider.fail = true
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("what's the material?", 10.0)
        val first = messages(coach).single()
        coach.ask("what's the material?", 11.0)
        val again = messages(coach).single()
        assertTrue("said again rather than nothing", first.text.isNotEmpty() && again.text == first.text)
    }

    /**
     * With no template to fall back on, a failed reply to a question says honestly that no answer
     * came, rather than nothing. `test_a_failed_reply_with_no_template_says_so`.
     */
    @Test
    fun aFailedReplyWithNoTemplateSaysSo() {
        val provider = ScriptedProvider(emptyList()).also { it.fail = true }
        val speaker = Speaker(provider)
        fun request(trigger: Trigger) = CoachRequest(
            trigger = trigger, level = DisclosureLevel.IDEA, system = "",
            messages = listOf(ChatMessage("user", "# Task\nAnswer them.")),
            facts = PositionFacts(fen = Board().fen(), sideToMove = "white", phase = "opening"),
            board = Board(), fallback = "", ply = 0, gameId = "g",
        )
        assertEquals(NOT_ANSWERED, speaker.speakNow(request(Trigger.USER_ASK)).message.text)
        assertEquals("an unasked remark may stay unsaid", "", speaker.speakNow(request(Trigger.POST_MOVE)).message.text)
    }

    /**
     * Tai, 2026-10-02 (T8): "the stuck hint does not need to be voiced but the order should still be
     * preserved". The hint waiting when the question comes is shown first, in its place, and the
     * voice goes to the answer: its sink reads it as not to be spoken, and the panel is told so
     * ([Spoken.unvoiced]). A hint with no question behind it is spoken as ever.
     * `test_a_stuck_hint_a_question_came_behind_is_shown_first_and_not_spoken`.
     */
    @Test
    fun aStuckHintAQuestionCameBehindIsShownFirstAndNotSpoken() {
        val (coach, _) = scripted(listOf("Look at f7.", "Qxf7 is mate."))
        val sinks = HashMap<Trigger, () -> Boolean>()
        coach.streamSink = { trigger, _, movedOn, _ ->
            sinks[trigger] = movedOn // what the voice will ask when the first sentence comes
            null
        }
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val since = coach.policy.stuckFrom!!
        coach.tick(since + coach.policy.stuckThreshold()) // the stuck hint, queued, not yet run
        assertFalse("nothing behind it yet: spoken", sinks.getValue(Trigger.STUCK)())
        coach.ask("what should I play?", since + coach.policy.stuckThreshold() + 1)
        assertTrue("a question came behind it: not spoken", sinks.getValue(Trigger.STUCK)())
        assertFalse("the answer is spoken", sinks.getValue(Trigger.USER_ASK)())
        val out = messages(coach)
        assertEquals("in order, none dropped", listOf(Trigger.STUCK, Trigger.USER_ASK), out.map { it.trigger })
        val (hint, answer) = coach.spoken.takeLast(2)
        assertTrue(hint.unvoiced && !answer.unvoiced)
    }

    // 10. What the log keeps is bounded

    @Test
    fun theLogKeepsThePromptsOfTheLastRepliesOnly() {
        val coach = offlineCoach()
        for (n in 0 until SPOKEN_PROMPTS_KEPT + 10) {
            coach.speaker.submitReady(
                Spoken(
                    message = CoachMessage(text = "m$n", level = DisclosureLevel.IDEA, trigger = Trigger.USER_ASK),
                    level = DisclosureLevel.IDEA, problems = emptyList(), attempts = 1,
                    calls = listOf(ModelCall("raw", emptyMap(), 1, 1, "prompt $n")),
                ),
                now = true,
            )
        }
        assertEquals(SPOKEN_PROMPTS_KEPT + 10, coach.poll().size)
        val kept = coach.spoken.map { it.calls[0].prompt }
        assertEquals(List(10) { "" }, kept.take(10))
        assertEquals("prompt 10", kept[10])
        assertEquals("prompt ${SPOKEN_PROMPTS_KEPT + 9}", kept.last())
    }

    // Asked with no game watched (Tai, 2026-10-02): the AI coach answers, told so; the offline
    // coach keeps its plain line. `test_coach_core.py`.

    @Test
    fun afterAStopTheAiCoachAnswersAndIsToldNoGameIsWatched() {
        val (coach, provider) = coachAt("italian", listOf("Your bishop on c4 eyes f7."))
        coach.pause(3.0) // the panel's Stop
        assertFalse(coach.watching)
        coach.ask("why that bishop?", 4.0)
        assertEquals(listOf("Your bishop on c4 eyes f7."), messages(coach).map { it.text })
        assertTrue("told no game is being watched", NOT_WATCHING in harnessOf(provider))
        // Once a spell: a second question finds it said already, and not again.
        provider.replies.addLast("Develop and castle.")
        coach.ask("and then?", 5.0)
        messages(coach)
        assertEquals(1, harnessOf(provider).split(NOT_WATCHING).size - 1)
    }

    @Test
    fun withNoPositionTheAiCoachAnswersAndTheOfflineOneSaysSo() {
        val (coach, provider) = scripted(listOf("Learn the Italian: quick development."))
        coach.spawn = { work, _ -> work() } // the question's worker, run here
        coach.watching = false // never started
        coach.ask("what opening should I learn?", 1.0)
        val answer = messages(coach)
        assertEquals(listOf("Learn the Italian: quick development."), answer.map { it.text })
        assertEquals(Trigger.USER_ASK, answer[0].trigger)
        assertTrue(NO_BOARD_UNWATCHED in directiveOf(provider))
        assertTrue("\"what opening should I learn?\"" in harnessOf(provider))
        assertFalse("no board shown", "FEN:" in harnessOf(provider))
        // A session running, the board not read yet: the model is told that instead.
        coach.watching = true
        provider.replies.addLast("Wait for the board.")
        coach.ask("what now?", 2.0)
        assertEquals(listOf("Wait for the board."), messages(coach).map { it.text })
        assertTrue(NO_BOARD_YET in directiveOf(provider))
        // A failed call is answered all the same, with the coach's own words.
        val down = object : LLMProvider {
            override val name = "down"
            override fun healthcheck(): Pair<Boolean, String> = false to "down"
            override fun chat(
                system: String,
                messages: List<ChatMessage>,
                maxTokens: Int,
                temperature: Double,
                thinking: Boolean,
            ): Completion = throw java.io.IOException("Connection refused")
        }
        val failing = Coach(down, policy = policy())
        failing.spawn = { work, _ -> work() }
        failing.ask("still there?", 3.0)
        assertEquals(
            listOf("I could not work that out just now; ask me again in a moment."),
            messages(failing).map { it.text },
        )
        // The offline coach: its plain line, no model.
        val offline = offlineCoach()
        offline.watching = false
        offline.ask("what opening should I learn?", 1.0)
        assertEquals(listOf("I have not seen a position to talk about yet."), messages(offline).map { it.text })
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
