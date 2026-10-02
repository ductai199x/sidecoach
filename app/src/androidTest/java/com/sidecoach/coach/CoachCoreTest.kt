package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
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
import java.util.concurrent.CountDownLatch

/**
 * The coach's core logic, from the deep review of `coach.py` (2026-10-01), as the desktop's
 * `tests/test_coach_core.py` and `tests/test_alternatives.py` check it: what a takeback leaves,
 * what a combination is, how the player's words and the model's replies are kept to their moment,
 * and the best moves answered from real engine analyses (androidTest/assets/chess/alternatives).
 * Logic only: time is the policy's `ts` and the events; the one worker is held on a latch.
 */
@RunWith(AndroidJUnit4::class)
class CoachCoreTest {

    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"
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

    private fun directiveOf(provider: ScriptedProvider): String {
        val last = synchronized(provider.prompts) { provider.prompts.last().second }
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

    @Test
    fun thatMoveAfterAnInstantReplyIsThePlayersOwnLastMove() {
        val (_, provider) = askedAfterTheReply(10.5)
        assertTrue("The position they faced, before Bxf7+" in harnessOf(provider))
        val directive = directiveOf(provider)
        assertTrue(directive, "They ask about the move they already played, Bxf7+" in directive)
        assertTrue(directive, "By \"that\" they mean Bxf7+" in directive && "Kxf7" in directive)
    }

    @Test
    fun thatMoveAfterASlowReplyStaysWithTheModel() {
        val (_, provider) = askedAfterTheReply(40.0)
        assertFalse("They ask about the move they already played" in directiveOf(provider))
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

    @Test
    fun aQuietPositionHasSeveralMovesAboutAsGoodAndNoneClearlyWorse() {
        val (board, analysis) = load("italian")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        assertEquals(listOf("Bc5", "Nf6", "Be7"), facts.candidates)
        assertEquals(emptyList<Worse>(), worseMoves(board, analysis, board.turn))
        assertEquals("Bc5, Nf6 and Be7 are all fine here -- about equally good.", fineSaid(facts.candidates))
    }

    @Test
    fun oneClearBestMoveIsTheOnlyFineOneAndTheOthersPassUpTheMate() {
        val (board, analysis) = load("scholar")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        assertEquals(listOf("Qxf7#"), facts.candidates)
        val worse = worseMoves(board, analysis, board.turn)
        assertEquals(listOf("Qd1", "Qe2"), worse.map { it.san })
        assertTrue(worse.all { it.wrong == "passes up the mate" && it.drop >= WORSE_MARGIN })
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
            assertTrue(said, "Bc5, Nf6 and Be7 are all fine here -- about equally good." in answer[0].text)
            assertEquals(DisclosureLevel.MOVE, answer[0].level)
        }
    }

    @Test
    fun theModelIsToldToReadEveryLineAndToNameTheMovesThatAreFine() {
        val (coach, provider) = coachAt("italian", listOf("Bc5, Nf6 and Be7 are all fine."))
        coach.ask("what are the best moves?", 2.0)
        messages(coach)
        val directive = directiveOf(provider)
        assertTrue(directive, "all of them and not only line 1" in directive)
        assertTrue(directive, "all fine: Bc5, Nf6, Be7" in directive)
        assertTrue(directive, "name every one" in directive && "No figures" in directive)
    }

    @Test
    fun askedWhetherAMoveIsAlsoGoodTheModelGetsTheFineMoves() {
        val (coach, provider) = coachAt("italian", listOf("Yes, Be7 is fine too."))
        coach.ask("is Be7 also good?", 2.0)
        messages(coach)
        val directive = directiveOf(provider)
        assertTrue(directive, "whether another move is also good" in directive)
        assertTrue(directive, "all fine: Bc5, Nf6, Be7" in directive)
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
        assertTrue(directive, "One move stands out: Qxf7#" in directive)
        assertTrue(directive, "Qd1 passes up the mate; Qe2 passes up the mate" in directive)
    }

    @Test
    fun theBestMoveAskedInTheSingularNamesTheNearEqualOnesToo() {
        val (coach, _) = coachAt("italian")
        coach.ask("what's the best move?", 2.0)
        val text = messages(coach)[0].text
        assertTrue(text, "Bc5" in text && "Nf6 and Be7 are about as good." in text)
        val (_, provider) = coachAt("italian", listOf("Bc5.")).also { (c, _) ->
            c.ask("what's the best move?", 2.0)
            messages(c)
        }
        assertTrue("3 moves are about equally good here (Bc5, Nf6, Be7)" in directiveOf(provider))
        val (_, single) = coachAt("scholar", listOf("Qxf7 mates.")).also { (c, _) ->
            c.ask("what's the best move?", 2.0)
            messages(c)
        }
        assertFalse("about equally good" in directiveOf(single))
    }

    @Test
    fun theClaimCheckAcceptsEveryMoveAboutAsGoodAsTheBestAsSanctioned() {
        val (board, analysis) = load("italian")
        val facts = factsFromAnalysis(board, analysis, lastMove = null, user = board.turn)
        assertTrue("Be7" in facts.candidates)
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
        val (again, provider) = coachAt("scholar", listOf("a", "b", "c", "Qxf7 is mate; Qd1 passes it up."))
        for (n in 0 until 4) {
            again.ask("hint", 2.0 + n)
            messages(again)
        }
        val directive = directiveOf(provider)
        assertTrue(directive, "has been given already: it is Qxf7#" in directive)
        assertTrue(directive, "Clearly worse here, and what goes wrong: Qd1 passes up the mate" in directive)
        assertTrue(directive, "instead of something that looks as good but is clearly worse" in directive)
    }

    // 5, 13. A lesson on a withdrawn move is dropped, and its question was never asked

    @Test
    fun aLessonOnAWithdrawnMoveIsNeitherShownNorSpoken() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateMissed(coach, 10.0) // the lesson is queued
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 12.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        assertTrue(coach.spoken.isEmpty() && coach.withdrawn.size == 1)
        assertTrue(coach.policy.lastReason!!.contains("no longer has"))
    }

    @Test
    fun aQuestionThatEndsADroppedLessonIsAskedAgainNotMarkedAsked() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.mistakes = ELICIT_EVERY - 1
        mateMissed(coach, 10.0)
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 12.0)
        messages(coach)
        coach.mistakes = ELICIT_EVERY
        assertTrue("the lesson was never said: nor was the question", coach.elicitDue(100.0))
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
    fun aQuestionOvertakesARemarkButNotTheLessonOrTheReview() {
        val coach = offlineCoach()
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val speaker = Speaker(OfflineProvider())
        for (request in requests(
            coach, Trigger.STUCK, Trigger.POST_MISTAKE, Trigger.REVIEW, Trigger.POST_MOVE, Trigger.USER_ASK,
        )) {
            speaker.submit(request)
        }
        assertEquals(2, speaker.dropped)
        assertEquals(3, speaker.pending)
        speaker.runPending()
        assertEquals(
            listOf(Trigger.USER_ASK, Trigger.POST_MISTAKE, Trigger.REVIEW),
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
        // The claims check's own failure is caught (no verdicts) and the model answers.
        assertEquals(listOf("Yes, Qxf7 is mate."), messages(coach).map { it.text })
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
        assertTrue(said.toString(), said.size == 2 && "three times" in said[0] && "For practice" in said[1])
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
    fun theSamePlanAcrossBestMovesIsOneCombination() {
        val coach = withPlans(
            mapOf(
                0 to "finishing your development",
                2 to "finishing your development",
                4 to "finishing your development",
            )
        )
        assertEquals(emptyList<CoachMessage>(), stuck(coach))
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

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
