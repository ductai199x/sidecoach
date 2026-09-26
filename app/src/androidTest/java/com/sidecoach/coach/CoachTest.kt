package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.KING
import com.sidecoach.chess.Piece
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.Resync
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

    /** `_given`: the messages the model was given on the [index]-th call (the last by default). */
    private fun given(provider: ScriptedProvider, index: Int = -1): List<ChatMessage> =
        synchronized(provider.prompts) {
            provider.prompts[if (index < 0) provider.prompts.size + index else index].second
        }

    /** `_harness`: the one user message, the harness with the task at its end. */
    private fun harness(messages: List<ChatMessage>): String {
        assertTrue(messages.size == 1 && messages[0].role == "user")
        return messages[0].content
    }

    /** `_directive` of a call's messages. */
    private fun directiveOf(messages: List<ChatMessage>): String =
        harness(messages).substringAfterLast("# Task\n")

    /** `_current_move_message`: the 'Position now' section, the position the player faces. */
    private fun currentMoveMessage(messages: List<ChatMessage>): String {
        val text = harness(messages)
        assertTrue("no position in the harness", "# Position now" in text)
        return text.substringAfter("# Position now").substringBefore("# Task")
    }

    /**
     * The first hint whole is on the sheet, the least the model's says, and the idea level names
     * the key fact on the board, as the first hint does (parity pass, 2026-09-25; desktop
     * 717cc9c, `test_a_decisive_moment_earns_the_idea_and_names_no_move`).
     */
    @Test
    fun aDecisiveMomentsSheetCarriesTheFirstHintWhole() {
        val (coach, provider) = coach(
            listOf("You can mate right now: their king on e8 is short of squares. Which check does it?")
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.DECISIVE_QUESTION)
        assertEquals(DisclosureLevel.IDEA, out[0].level)
        val messages = given(provider, 0)
        val current = currentMoveMessage(messages)
        assertTrue(current, "the first hint in the board's own words" in current)
        assertTrue(current, "Their king on e8 has one free square" in current)
        val directive = directiveOf(messages)
        assertTrue(directive, "the idea: what is at stake and the key fact on the board" in directive)
        assertTrue(directive, "no square" !in directive)
        assertTrue("named the key square: it stands", out[0].text.startsWith("You can mate right now"))
    }

    /**
     * The template is the floor (parity pass, 2026-09-25): a first hint that names nothing of the
     * key fact -- true, and thinner than the free coach's -- is sent back, and the template speaks
     * when the second is no better (desktop 717cc9c).
     */
    @Test
    fun aReplyThatSaysLessThanTheTemplateIsNotSaid() {
        val thin = "Your king is safe for now, is it not?"
        val (coach, provider) = coach(listOf(thin, "Think about checks. Which one works?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = messages(coach)
        assertTrue(out[0].text, out[0].text != thin && "e8" in out[0].text)
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 2 && coach.speaker.fallbacks == 1)
        assertTrue(spoken.problems.toString(), spoken.problems.any { it.startsWith("left out") })
        val retry = given(provider, 1)
        assertTrue("left out" in retry.last().content && "e8" in retry.last().content)
    }

    /**
     * The lesson is taught from the position they faced, which the record keeps as one line: its
     * sheet goes with the request, and the template's lesson as the floor, whose better move the
     * reply must name (parity pass, 2026-09-25; desktop 717cc9c,
     * `test_feedback_after_the_users_move_may_say_everything_about_that_position`).
     */
    @Test
    fun theLessonAfterAMoveCarriesThePositionTheyFaced() {
        val (coach, provider) = coach(
            listOf(
                "Their king on e8 is nearly out of squares.",
                // Never opening with the move just played (desktop f25217c).
                "Qxf7 was mate on the spot, sharper than the bishop. Do you see why?",
            )
        )
        bishopCheckInsteadOfMate(coach)
        val messages = given(provider)
        val content = messages.last().content
        val faced = content.substringAfter("# The position they faced, before Bxf7+").substringBefore("# Task")
        assertTrue(content, "# The position they faced, before Bxf7+" in content)
        assertTrue(faced, "goal: the player can mate in one move" in faced)
        assertTrue("The lesson in the board's own words" in directiveOf(messages))
        assertEquals(emptyList<String>(), coach.spoken.last().problems)
        // The hints' questions are for the side to move: never on the opponent's sheet, where
        // labelled "to the player" they were the opponent's; nor on the faced sheet, a position
        // already moved from (parity pass 2; desktop 2219e11).
        val now = currentMoveMessage(messages).substringBefore("# The position they faced")
        assertTrue(now, "(to the player, as it stands)" !in now.substringAfter("## Facts"))
        assertTrue(faced, "a question that sets them looking" !in faced)
        // Asked why it was bad, the answer is taught from the same position, and names the move.
        coach.ask("why was that bad?", 12.0)
        messages(coach)
        val asked = given(provider).last().content
        assertTrue(asked, "# The position they faced, before Bxf7+" in asked)
        val said = coach.spoken.last().message.text
        assertTrue(said, said.isNotEmpty() && "Qxf7" in said)
    }

    /**
     * "What should I be aiming for?" earns the hint's rung, and its template -- the first hint, a
     * lesson on the position -- goes out at once, as for a bare hint; the model builds on it with
     * the hint's directive, not the generic answer (parity pass 2, 2026-09-25; desktop 2219e11).
     */
    @Test
    fun aQuestionAboutThePlanIsAnsweredByTheFirstHintAtOnce() {
        val (coach, provider) = coach(listOf("The king on e8 has one square left. Which check takes it away?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.ask("what should I be aiming for here?", 8.0)
        val out = messages(coach)
        assertTrue(out[0].text, out[0].level == DisclosureLevel.IDEA && "e8" in out[0].text)
        // The hint is out at once, in the template's words; the model adds to it (desktop
        // af1bae2).
        val directive = directiveOf(given(provider, 0))
        assertTrue(directive, "have just been shown it" in directive && "Do not repeat it" in directive)
        assertTrue(directive, "be honest about their own claim" !in directive)
    }

    /**
     * The runtime audit: with the voice on, the reply was checked sentence by sentence only, and
     * "Mate is close." went out whole. Now the stream holds until what the reply must name has
     * come, says nothing when it never does, and stops at the task's length (desktop af1bae2).
     */
    @Test
    fun withTheVoiceOnNothingGoesOutBeforeTheKeyFactAndNotPastTheLength() {
        val heard = ArrayList<String>()
        var reader = FirstCandidate({ heard.add(it); true }, { emptyList() }, mustName = listOf("e8"))
        reader.feed("Stay calm. Look carefully at the whole board. ")
        assertEquals("held: the key fact has not come", emptyList<String>(), heard)
        reader.finish()
        assertTrue(heard.isEmpty() && reader.problems.isNotEmpty() && reader.problems[0].startsWith("left out"))

        heard.clear()
        reader = FirstCandidate({ heard.add(it); true }, { emptyList() }, mustName = listOf("e8"))
        reader.feed("Mate is close. Their king on e8 has one square. Which check takes it? ")
        reader.finish()
        assertTrue(heard.toString(), heard.joinToString(" ").startsWith("Mate is close. Their king on e8"))

        heard.clear()
        reader = FirstCandidate({ heard.add(it); true }, { emptyList() }, limit = 1)
        reader.feed("One. Two. Three. Four. Five. ")
        reader.finish()
        assertEquals("the limit and its slack, no more", 3, heard.sumOf { t -> t.count { it == '.' } })
    }

    /** A late verdict's lesson must not give away the answer in the position faced now (af1bae2). */
    @Test
    fun aLateLessonDoesNotGiveAwayTheMoveTheyFaceNow() {
        val request = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7", nowAnswer = listOf("Qxf7#"))
        val faults = Speaker(ScriptedProvider(emptyList())).problemsOf("And now Qxf7# is mate on the spot.", request)
        assertTrue(faults.toString(), "gives away Qxf7# in the position they face now" in faults)
    }

    /**
     * Every lesson names the better move, and skipping any reply that did let "That was a
     * blunder: a3 was better" through for a good move (the runtime audit; desktop af1bae2).
     */
    @Test
    fun theVerdictWordsAreCheckedClauseByClause() {
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val good = afterMoveRequest(
            ruy, "O-O", "b5a4", severity = Severity.NONE, label = Label.GOOD, expectedLoss = 0.04,
        )
        assertEquals(
            listOf("called O-O a slip; its verdict is good"),
            verdictProblems("That was a blunder: Ba4 was better.", good),
        )
        assertTrue(verdictProblems("O-O was the best move here, though Ba4 was close.", good).isNotEmpty())
        assertEquals(emptyList<String>(), verdictProblems("Fine, but Ba4 was the best move here.", good))
    }

    /**
     * The verdict comes first, from the best line alone; the analysis of the position the move
     * left about a second later, and with it what the move allowed. The app holds the lesson for
     * it (at most [LESSON_WAIT_S]); the game moving on releases it at once (desktop 98c4d40).
     */
    @Test
    fun aMistakesLessonWaitsForWhatTheMoveAllowed() {
        val coach = Coach(
            ScriptedProvider(emptyList()),
            policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
            lessonWaitS = LESSON_WAIT_S,
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Bxf7+")!!
        board.push(move)
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "Bxf7+",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = board.fen(),
            byUser = true,
        )
        val verdict = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.45,
            severity = Severity.BLUNDER,
            best = listOf("h5f7"),
            label = Label.MISS,
        )
        coach.onEvent(played, 10.0)
        coach.onEvent(verdict, 10.1)
        assertEquals("held for the analysis of the position it left", emptyList<CoachMessage>(), messages(coach))
        coach.tick(10.1 + LESSON_WAIT_S / 2)
        assertEquals("still inside the wait", emptyList<CoachMessage>(), messages(coach))
        coach.tick(10.1 + LESSON_WAIT_S)
        var out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MISTAKE)
        // The analysis landing releases it at once.
        coach.onEvent(Takeback(gameId = "g", plies = 1, fen = SCHOLAR), 12.0)
        messages(coach)
        coach.onEvent(played, 13.0)
        coach.onEvent(verdict, 13.1)
        val after = Analysis(
            gameId = "g",
            fen = board.fen(),
            ply = 7,
            lines = listOf(Line(moves = listOf("e8e7"), depth = 12, expectedScore = 0.1)),
            decisive = false,
            onlyMove = true,
        )
        coach.onEvent(after, 13.5)
        out = messages(coach)
        assertEquals(listOf(Trigger.POST_MISTAKE), out.map { it.trigger })
    }

    /**
     * The second audit: "ahead by about 1.2 pawns" was no figure to the old check; "you took on
     * d5" and "that d5 pawn" said the capture back; "which was a inaccuracy" (desktop 8438193).
     */
    @Test
    fun figuresOrdinalsCapturesAndArticles() {
        assertTrue(figureSaid("You are ahead by about 1.2 pawns."))
        assertTrue(figureSaid("It cost you 15% of the game."))
        assertFalse(figureSaid("Your rook on the 7th rank and Nf3 were strong."))
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val trade = afterMoveRequest(ruy, "Bxc6", "e1g1")
        assertTrue(restated("You took on c6, giving up the bishop pair.", trade).isNotEmpty())
        assertTrue(restated("Winning that c6 knight gives up the bishop pair.", trade).isNotEmpty())
        assertEquals(
            listOf("an inaccuracy", "a miss", "best", "a book move"),
            listOf("inaccuracy", "miss", "best", "book").map { rated(it) },
        )
    }

    /** Their own move is theirs even after the reply, and a threat gets its answer (af1bae2). */
    @Test
    fun theirOwnMoveIsTheirsEvenAfterTheReplyAndAThreatGetsItsAnswer() {
        for (said in listOf("what should I have played?", "did I just blunder?", "was my move bad?")) {
            assertTrue(said, MY_MOVE.containsMatchIn(said))
        }
        assertFalse("'that' is the last move played", MY_MOVE.containsMatchIn("why was that bad?"))
        for (said in listOf("what is their threat?", "what does my opponent threaten?", "am I in danger?")) {
            assertTrue(said, ASK_THREAT.containsMatchIn(said))
        }
        val (coach, provider) = coach(listOf("Nothing is threatened right now."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what is their threat?", 8.0)
        messages(coach)
        assertTrue("They ask what the opponent threatens" in directiveOf(given(provider, 0)))
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
     * The verdict and the better move came back in every reply, and "I understand: I should play
     * Nxc3" got the lesson again and "Which White piece is still at home on g1?" (the AI's
     * sessions, 2026-09-26; desktop 9e49622). The desktop's words ("I see, I should have seen the
     * mate.", "Why does the bishop check not work?") reach the answer about the last move through
     * `_asked_own_move` (f733318, not ported); on the phone the words about the last move do.
     */
    @Test
    fun aFollowUpGetsWhatItAsksAndANodOneSentence() {
        val (coach, provider) = coach(listOf("...", "...", "...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(coach.lock) {
            coach.conversation.coachSaid("That one cost you: Qxf7# was better -- it is mate.")
        }
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("I see now what was better: the queen mates.", 12.0)
        messages(coach)
        var directive = directive(provider)
        assertTrue(directive, "do not repeat either" in directive)
        assertTrue(directive, "They ask nothing: reply in one short sentence" in directive)
        assertTrue(directive, "then one question" !in directive)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("Why was that bad?", 13.0)
        messages(coach)
        directive = directive(provider)
        assertTrue(directive, "Answer only what they ask now" in directive && "do not repeat either" in directive)
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
        // Once a game after that: said every second time it came to "10 times now" in a session
        // (desktop 9e49622).
        assertEquals(emptyList<String>(), placed(31, Causes.MISSED_THREAT))
        assertEquals(emptyList<String>(), placed(37, Causes.MISSED_THREAT))
        coach.gameId = "g2" // the next game of the session
        val fifth = placed(9, Causes.MISSED_THREAT, gameId = "g2")
        assertTrue(fifth.toString(), "5 times now" in fifth[0])
        assertEquals(emptyList<String>(), placed(15, Causes.MISSED_THREAT, gameId = "g2"))
        // A player of 1800 and up hears the pattern without the textbook habit.
        coach.strongPlayer = true
        coach.gameId = "g3"
        val strong = placed(9, Causes.MISSED_THREAT, gameId = "g3")[0]
        assertTrue(strong, "what their moves threatened" in strong && "look at what" !in strong)
    }

    /**
     * A 2000 player was told "you kept missing endgame technique: bring your King to the middle
     * and push your passed pawns" (the AI's sessions, 2026-09-26; desktop 216dcb1).
     */
    @Test
    fun aStrongPlayersReviewNamesThePatternWithoutTheTextbookHabit() {
        val (coach, _) = coach(emptyList())
        coach.strongPlayer = true
        start(coach)
        val pattern = "What you kept missing: endgame technique -- 2 of your 3 mistakes. The habit that fixes " +
            "it: in the endgame, bring your king to the middle and push your passed pawns."
        coach.onEvent(
            GameReview(
                gameId = "g",
                userSide = Pieces.Side.WHITE,
                result = "1-0",
                moves = 40,
                lines = listOf("You played white and won in 40 moves.", pattern),
            ),
            1.0,
        )
        val said = messages(coach).joinToString(" ") { it.text }
        assertTrue(said, "endgame technique -- 2 of your 3 mistakes." in said && "passed pawns" !in said)
        // What the model is shown of the review says it the same way.
        assertTrue("passed pawns" !in coach.conversation.render())
    }

    /**
     * The AI said "Qxf7+ is illegal here" and "Kf7 is illegal because f7 is controlled by the King
     * on e8" of legal moves (its own sessions, checked against the boards; desktop 403e675).
     */
    @Test
    fun aLegalMoveCalledIllegalIsAFault() {
        val board = Board("2r3k1/4bpp1/p3p2p/qp1bn1NQ/8/1P1BB3/P4PPP/3R2K1 w - - 0 20")
        assertEquals(
            listOf("Qxf7: called illegal, but it is a legal move here"),
            falseIllegal("No: Qxf7+ is illegal here.", board),
        )
        // A claim inside a line is about a later position; a true one is no fault.
        assertEquals(emptyList<String>(), falseIllegal("After Bh7+ Kh8, Qxf7+ is illegal.", board))
        assertEquals(emptyList<String>(), falseIllegal("Qh4 is not legal: the pawn blocks it.", Board()))
    }

    /** `_after_move_request`: a lesson's request on [fen] after [san], the engine preferring [bestUci]. */
    private fun afterMoveRequest(
        fen: String,
        san: String,
        bestUci: String,
        nowAnswer: List<String> = emptyList(),
        severity: Severity = Severity.MISTAKE,
        label: Label = Label.MISTAKE,
        expectedLoss: Double = 0.2,
    ): CoachRequest {
        val board = Board(fen)
        return CoachRequest(
            trigger = Trigger.POST_MISTAKE,
            level = DisclosureLevel.LINE,
            system = "",
            messages = emptyList(),
            facts = PositionFacts(fen = fen, sideToMove = "white", phase = "opening"),
            board = board,
            fallback = "",
            ply = 7,
            gameId = "g",
            playedSan = san,
            assessment = MoveAssessment(
                gameId = "g",
                ply = 7,
                uci = board.parseSan(san)!!.uci(),
                expectedLoss = expectedLoss,
                severity = severity,
                best = listOf(bestUci),
                label = label,
            ),
            nowAnswer = nowAnswer,
        )
    }

    /**
     * The owner's rules after a move, checked in code (the eval audit, 2026-09-25): never open
     * with the move just played, never say what it plainly did unless the better move does the
     * same, and no figure in a remark nobody asked for (desktop f25217c).
     */
    @Test
    fun aWordAfterTheMoveNeverRestatesItNorQuotesAFigure() {
        val ruy = "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"
        val trade = afterMoveRequest(ruy, "Bxc6", "e1g1")
        assertEquals(listOf("opens with Bxc6, the move just played"), restated("Bxc6 gives up the bishop pair.", trade))
        assertEquals(
            listOf("says what Bxc6 plainly did: the capture on c6"),
            restated("Taking the knight on c6 gives up the bishop pair.", trade),
        )
        assertEquals(
            listOf("a figure in a remark nobody asked for: say it in words"),
            restated("It cost you 15% of the game.", trade),
        )
        // The better move castles: saying so is the lesson, not a restatement.
        assertEquals(
            emptyList<String>(),
            restated("Castling first was better: the king is safe, the rook joins in.", trade),
        )
        // The better move takes on the same square with check: the words are about it.
        val miss = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7")
        assertEquals(emptyList<String>(), restated("Taking the pawn on f7 with check was right, but with the queen.", miss))
    }

    /**
     * "Say at least this" and "two or three sentences" asked for both: a first hint that runs
     * longer sets the length (the eval audit, 2026-09-25; desktop f25217c).
     */
    @Test
    fun theFirstHintSetsTheLengthOfTheHint() {
        val short = PositionFacts(fen = SCHOLAR, sideToMove = "white", phase = "opening", situation = "A. B.")
        assertEquals(IDEA_DIRECTIVE, ideaDirective(short))
        val long = short.copy(situation = "One. Two. Three. Four. Five.")
        assertTrue("Five sentences at most." in ideaDirective(long))
        assertEquals(5, directiveLimits(ideaDirective(long)).first)
        val longest = short.copy(situation = List(9) { "Yes." }.joinToString(" "))
        assertEquals(6, directiveLimits(ideaDirective(longest)).first)
    }

    /**
     * A trap the template names ("Bxf7+ looks natural, but after Kxf7 ...") is a line from the
     * root: its boards are frames, so saying it is not faulted as unplayable (desktop f25217c).
     */
    @Test
    fun theDraftCheckerSeesEveryLineTheFactsWereReadFrom() {
        val base = mateAnalysis(6)
        val analysis = Analysis(
            gameId = base.gameId,
            fen = base.fen,
            ply = base.ply,
            lines = base.lines,
            decisive = base.decisive,
            onlyMove = base.onlyMove,
            traps = listOf(Line(moves = listOf("c4f7", "e8e7"), depth = 12, expectedScore = 0.4)),
            tags = base.tags,
        )
        val position = CoachPosition(ply = 6, board = Board(SCHOLAR), analysis = analysis)
        val (coach, _) = coach(emptyList())
        val frames = coach.analysisFrames(position)
        assertTrue(frames.any { it.pieceAt(parseSquare("e7")) == Piece(KING, BLACK) })
        assertTrue(frames.any { it.pieceAt(parseSquare("f7")) == Piece(QUEEN, WHITE) })
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        // The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
        private const val QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
        private val QD1_LINE = listOf("c2d1", "d4d1", "d8d1", "a5e1", "d1e1")
    }
}
