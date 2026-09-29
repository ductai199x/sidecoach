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
import com.sidecoach.events.GameEnded
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
import com.sidecoach.llm.StreamCallback
import com.sidecoach.llm.ToolsProvider
import org.json.JSONArray
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

    /**
     * `test_an_early_analysis_answers_a_question_waiting_on_it` (desktop 316026b). Tai,
     * 2026-09-28: "first usable answer is depth 12 ... the computation continues until max depth
     * of 18". A question asked before the engine has looked is answered from the early
     * analysis, the full depth still to come.
     */
    @Test
    fun anEarlyAnalysisAnswersAQuestionWaitingOnIt() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.ask("hint", 6.5)
        assertEquals(listOf("One moment -- still looking at this position."), messages(coach).map { it.text })
        coach.onEvent(mateAnalysis(6, provisional = true), 7.0)
        val answered = messages(coach)
        assertTrue(
            answered.map { it.trigger }.toString(),
            answered.isNotEmpty() && answered.all { it.trigger == Trigger.USER_ASK },
        )
    }

    /**
     * `test_the_coachs_own_words_wait_for_the_full_depth` (desktop 316026b): the decisive
     * question is the coach's own: an early analysis asks nothing, the full depth does -- and an
     * early one that comes after it replaces nothing.
     */
    @Test
    fun theCoachsOwnWordsWaitForTheFullDepth() {
        val (coach, _) = coach(
            listOf("You can mate right now: their king on e8 is short of squares. Which check does it?")
        )
        playToScholar(coach)
        val early = mateAnalysis(6, provisional = true)
        coach.onEvent(early, 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.onEvent(mateAnalysis(6), 8.0)
        val out = messages(coach)
        assertEquals(listOf(Trigger.DECISIVE_QUESTION), out.map { it.trigger })
        coach.onEvent(early, 9.0)
        val analysis = coach.positions.getValue(6).analysis
        assertTrue(analysis != null && !analysis.provisional)
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

    /**
     * A reply off the engine's line brought the idea back: the stuck hint's "a tactic that wins a
     * knight ... a pin" at a puzzle's start came again as the question after the move was found
     * (Tai's phone, 2026-09-29: "Each level can only be said once and if the level higher was
     * said then the lower one can't be said"). The next hint unasked is the focus
     * (`test_no_new_question_while_the_line_is_played_out_nor_a_hint_said_twice`).
     */
    @Test
    fun aHintIsNeverSaidTwiceAlongTheCombination() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        val first = messages(coach)
        assertEquals(listOf(Trigger.DECISIVE_QUESTION to DisclosureLevel.IDEA), first.map { it.trigger to it.level })
        played(coach, board, 1, "c2d1", true, 3.0)
        played(coach, board, 2, "a5e1", false, 4.0) // blocks instead of taking: off the line
        analysed(coach, board, 2, listOf("d1e1"), 4.5, 1)
        assertEquals(emptyList<CoachMessage>(), messages(coach).filter { it.trigger != Trigger.POST_MOVE })
        coach.tick(4.0 + coach.policy.stuckThreshold() + 1)
        val stuck = messages(coach).filter { it.trigger == Trigger.STUCK }
        assertEquals(stuck.toString(), listOf(DisclosureLevel.FOCUS), stuck.map { it.level })
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

    /**
     * The counts of what was said unasked were kept by ply, and every puzzle starts at ply 0:
     * after one puzzle's question and focus there, the next puzzle's log said "gave the idea and
     * the focus here" before a word of it (Tai's phone, 2026-09-29).
     */
    @Test
    fun aNewPuzzleGetsItsHintsWhateverTheLastOneHad() {
        val (coach, _) = coach(emptyList())
        val board = puzzle(coach)
        analysed(coach, board, 0, QD1_LINE, 1.0, 3)
        coach.tick(1.0 + coach.policy.stuckThreshold() + 1)
        assertEquals(listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS), messages(coach).map { it.level })
        val next = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
        coach.onEvent(Resync(gameId = "g", fen = next, reason = "adopted"), 100.0)
        analysed(coach, Board(next), 0, listOf("g8f8"), 101.0, null)
        coach.tick(101.0 + coach.policy.stuckThreshold() + 1)
        assertEquals(listOf(DisclosureLevel.IDEA, DisclosureLevel.FOCUS), messages(coach).map { it.level })
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

    /**
     * `_mate_analysis`: Qxf7# on the scholar's board; `provisional`, as the desktop's
     * `model_copy(update={"provisional": True})` makes it.
     */
    private fun mateAnalysis(ply: Int, decisive: Boolean = true, provisional: Boolean = false) = Analysis(
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
        provisional = provisional,
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
        // Everything, the answer included (Tai, 2026-09-26: the model sees engine lines at hints;
        // the reply is checked; desktop 9e2e8e2).
        assertTrue("the model sees every move", "Legal moves" in current && "Qxf7#" in current)
        assertTrue(current, "line 1's first move is the answer to the position they face" in current)
        assertTrue(current, "On the board (complete lists)" in current)
        // The rung's facts -- the goal and the pattern -- stand apart from those past it.
        val facts = current.substringAfter("## Facts (the player to move)").substringBefore("## Beyond this level")
        assertTrue(facts, "goal: the player can mate in one move" in facts)
        assertTrue(facts, "pattern: a mating attack" in facts)
        for (secret in listOf("Qxf7", "queen", "h5f7", "candidates")) assertTrue(secret, secret !in facts)
        val beyond = current.substringAfter("## Beyond this level").substringBefore("## On the board")
        assertTrue(beyond, "piece: the player's queen on h5" in beyond && "Qxf7" in beyond)
        // What the level keeps back is said once, in the Task, with why the model sees it all.
        assertTrue(current, "## Disclosure" !in current)
        val directive = directiveOf(messages)
        assertTrue(directive, "You see the answer so that the hint can be precise" in directive)
        // The first hint's facts are on the sheet to teach from (parity, 2026-09-25; the prompt
        // audit, 2026-09-28), and the idea level names the key fact on the board.
        assertTrue(current, "what a first hint rests on" in current)
        assertTrue(current, "Their king on e8 has one free square" in current)
        assertTrue(directive, "the idea: what is at stake and the key fact" in directive)
        assertTrue(directive, "no square" !in directive)
        assertTrue("named the key square: it stands", out[0].text.startsWith("You can mate right now"))
    }

    /**
     * A first hint that names nothing of the key fact is thinner than the free coach's; it was sent
     * back and the template spoken (parity pass, 2026-09-25). Its form is the prompt's to ask for,
     * and a slip of it is minor (Tai, 2026-09-28: "we don't need to throw anything out"): it is
     * said, and noted for the log (desktop 3ff672b).
     */
    @Test
    fun aReplyThinnerThanTheTemplateIsSaidAndNoted() {
        val thin = "Your king is safe for now, is it not?"
        val (coach, _) = coach(listOf(thin))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = messages(coach)
        assertEquals(thin, out[0].text)
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 1 && coach.speaker.fallbacks == 0 && spoken.problems.isEmpty())
        assertTrue(spoken.notes.toString(), spoken.notes.any { it.startsWith("left out") })
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
        // The better move the reply must name is the directive's -- not the template's lesson as
        // the least the reply says, which came back word for word (the prompt audit, 2026-09-28).
        assertTrue("The better move was Qxf7#" in directiveOf(messages))
        assertTrue("says at least this" !in directiveOf(messages))
        assertEquals(emptyList<String>(), coach.spoken.last().problems)
        // The hints' questions are for the side to move: never on the opponent's sheet, where
        // labelled "to the player" they were the opponent's; nor on the faced sheet, a position
        // already moved from (parity pass 2; desktop 2219e11).
        val now = currentMoveMessage(messages).substringBefore("# The position they faced")
        assertTrue(now, "(to the player, as it stands)" !in now.substringAfter("## Facts"))
        assertTrue(faced, "a question that sets them looking" !in faced)
    }

    /**
     * "What should I be aiming for?" earns the hint's rung and the hint's directive, not the
     * generic answer (parity pass 2, 2026-09-25; desktop 2219e11); asked of the AI coach, the
     * model answers it, with nothing said before it (Tai, 2026-09-28; desktop 3ff672b).
     */
    @Test
    fun aQuestionAboutThePlanIsAnsweredWithTheFirstHint() {
        val reply = "The king on e8 has one square left. Which check takes it away?"
        val (coach, provider) = coach(listOf(reply))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        coach.ask("what should I be aiming for here?", 8.0)
        val out = messages(coach)
        assertTrue(out.map { it.text }.toString(), out.map { it.text } == listOf(reply) && out[0].level == DisclosureLevel.IDEA)
        val directive = directiveOf(given(provider, 0))
        assertTrue(directive, "They asked what to aim for: answer with the hint at this rung." in directive)
        assertTrue(directive, "be honest about their own claim" !in directive)
    }

    /**
     * Tai, 2026-09-29: "Each level can only be said once and if the level higher was said then
     * the lower one can't be said." Once a question had the move, no stuck hint follows: the
     * focus is below it (on 2026-09-28 one did, and both replies went for "d5: your bishop on c8
     * is still shut in", twice). A stuck template stays at the focus all the same
     * (`test_no_hint_unasked_comes_after_the_move_was_given`).
     */
    @Test
    fun noHintUnaskedComesAfterTheMoveWasGiven() {
        val (coach, provider) = coach(listOf("Qxf7# is mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what should I play?", 8.0)
        assertEquals(listOf("Qxf7# is mate at once."), messages(coach).map { it.text })
        for (ts in listOf(17.0, 40.0, 90.0)) coach.tick(ts)
        assertEquals(emptyList<CoachMessage>(), messages(coach))
        assertEquals(1, provider.prompts.size)
        val position = coach.positions.getValue(6) // the position on the board (`coach._current`)
        assertNotNull(position.facts)
        val template = coach.instruction(
            Trigger.STUCK, position, DisclosureLevel.MOVE, null, emptyList(), null, null,
        ).second
        assertTrue("the template is the focus's, not the move's: $template", "Qxf7" !in template)
    }

    /**
     * The runtime audit: with the voice on, an eight-sentence homily went out whole. The stream
     * stops at the task's length and its slack (desktop af1bae2, 3ff672b).
     */
    @Test
    fun withTheVoiceOnNothingGoesOutPastTheLength() {
        val heard = ArrayList<String>()
        val reader = FirstCandidate({ heard.add(it); true }, { emptyList() }, limit = 1)
        reader.feed("One. Two. Three. Four. Five. ")
        reader.finish()
        assertEquals("the limit and its slack, no more", 3, heard.sumOf { t -> t.count { it == '.' } })
    }

    /**
     * A late verdict's lesson that gives away the answer in the position faced now is noted
     * (af1bae2) -- logged, never a reason to refuse the reply (Tai, 2026-09-26; desktop 9e2e8e2).
     */
    @Test
    fun aLateLessonDoesNotGiveAwayTheMoveTheyFaceNow() {
        val request = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7", nowAnswer = listOf("Qxf7#"))
        val speaker = Speaker(ScriptedProvider(emptyList()))
        val said = "And now Qxf7# is mate on the spot."
        assertTrue("gives away Qxf7# in the position they face now" in speaker.leaks(said, request))
        assertTrue(speaker.problemsOf(said, request).none { it.startsWith("gives away") })
    }

    /**
     * A reply that gives the answer away is logged, never refused (Tai, 2026-09-26: the leak
     * check is "a debugging tool and not something used to reject response"; desktop 9e2e8e2).
     */
    @Test
    fun aLeakedDraftIsSpokenAndTheLeakLogged() {
        val (coach, provider) = coach(
            listOf("Their king on e8 is boxed in: your queen can end it on f7. Which check is it?")
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        val out = messages(coach)
        assertTrue(out.size == 1 && out[0].text.startsWith("Their king on e8 is boxed in"))
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 1 && spoken.problems.isEmpty() && coach.speaker.fallbacks == 0)
        assertTrue(spoken.leaks.toString(), "withheld square f7" in spoken.leaks && "withheld piece queen" in spoken.leaks)
        assertEquals("no second call for a leak", 1, provider.prompts.size)
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

    /**
     * The parity pilot: "**Rc1** was better", and a closing question wrapped in ** failed the
     * question check twice, so the template spoke (desktop de68772).
     */
    @Test
    fun theModelsMarkdownNeverReachesTheVoiceOrTheChecks() {
        assertEquals("Their threat is ...f5, on e4. Why?", plain("Their threat is **...f5**, on `e4`. **Why?**"))
        assertEquals("Plan\npush the pawn\ntrade", plain("# Plan\n- push the pawn\n* trade"))
        assertEquals(
            listOf(1.0 to "Rc1 was better. Do you see why?"),
            parseCandidates("**Rc1** was better. **Do you see why?**"),
        )
    }

    /**
     * Round 2: "Stockfish gives White about 0.8 pawns" in seven answers (desktop 47138dd; the
     * review's costliest losses are held by the conversation fixture). Noted, never refused: a
     * reply's form is the prompt's to ask for (Tai, 2026-09-28; desktop 3ff672b).
     */
    @Test
    fun theCoachNeverNamesTheEngine() {
        val request = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7")
        val speaker = Speaker(ScriptedProvider(emptyList()))
        val said = "Qxf7 was better; Stockfish liked it more."
        val notes = speaker.notes(said, request)
        assertTrue(notes.toString(), "named the engine: say it as the coach who saw the game" in notes)
        assertEquals(emptyList<String>(), speaker.problemsOf(said, request))
        val asked = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7", userText = "what does the engine say?")
        assertTrue(
            "named the engine: say it as the coach who saw the game" !in
                speaker.notes("Qxf7 was better; the engine liked it more.", asked)
        )
    }

    /**
     * Hint, Best move and Line. Asked of the AI coach, the model answers, with its marks and
     * nothing said before it: its first words come within a second or two (Tai, 2026-09-28: "it's
     * already almost instant"). Offline the template for the level goes out at once, the move and
     * the line asked for bare are the template's alone, and no model is asked (desktop 3ff672b).
     */
    @Test
    fun theButtonsAreTheModelsToAnswerAndTheTemplatesOffline() {
        val (coach, provider) = coach(listOf("Qxf7# ends it: the queen takes with check and mate."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what should I play", 8.0)
        assertEquals("nothing before the model's answer", emptyList<CoachMessage>(), coach.poll())
        val out = messages(coach)
        assertEquals(listOf("Qxf7# ends it: the queen takes with check and mate."), out.map { it.text })
        assertTrue(out[0].trigger == Trigger.USER_ASK && out[0].level == DisclosureLevel.MOVE)
        assertTrue(out[0].annotations.any { it.kind == "arrow" })
        val directive = directiveOf(given(provider, 0))
        assertTrue(directive, "They asked what to play: tell them plainly it is Qxf7#" in directive)
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(offline)
        offline.onEvent(mateAnalysis(6, decisive = false), 7.0)
        offline.ask("what should I play", 8.0)
        val quick = offline.poll() // at once
        assertEquals(listOf("Qxf7# is mate."), quick.map { it.text })
        assertEquals(DisclosureLevel.MOVE, quick[0].level)
        assertTrue(quick[0].annotations.any { it.kind == "arrow" })
        assertEquals(emptyList<CoachMessage>(), messages(offline))
    }

    /**
     * The material, the opening by name, who stands better. Offline the template says them
     * exactly, the fact itself, never the hint for the position ("How many pieces am I up?" was
     * once answered "Something forcing is on..."). Asked of the AI coach, the model answers (Tai,
     * 2026-09-28); and a question that only mentions the opening is not a question about its name:
     * "what should I play against this opening? give me a line" got "The opening was the A40
     * Queen's Pawn Game.", and is the line's to answer (desktop 4365dfa, 3ff672b).
     */
    @Test
    fun aQuestionOnlyAboutAFactIsTheModelsAndOfflineTheTemplates() {
        for (said in listOf("how is my opening?", "why am I losing?", "am I winning? what should I do?")) {
            assertFalse(said, factQuestion(said))
        }
        val (coach, provider) = coach(listOf("The model's words.", "Qxf7# -- it is mate at once."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what opening is this?", 8.0)
        assertEquals(listOf("The model's words."), messages(coach).map { it.text })
        assertTrue("The opening by the name the record gives it." in directiveOf(given(provider, 0)))
        coach.ask("what should I play against this opening? give me a line", 9.0)
        assertEquals(listOf("Qxf7# -- it is mate at once."), messages(coach).map { it.text })
        val directive = directiveOf(given(provider, 1))
        assertTrue(directive, "They asked to see the line: give it -- Qxf7#" in directive)
        assertTrue(directive, "The opening by the name" !in directive)
        val offline = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(offline)
        offline.onEvent(mateAnalysis(6, decisive = false), 7.0)
        for ((i, pair) in listOf(
            "who's up material?" to listOf("Material:"),
            "how many pieces am I up?" to listOf("Material:"),
            "am I winning?" to listOf("Right now you are winning"),
            "what opening is this?" to listOf("We left the opening book early.", "The opening was the"),
        ).withIndex()) {
            val (said, opens) = pair
            assertTrue(said, factQuestion(said))
            offline.ask(said, (8 + i).toDouble())
            val out = messages(offline)
            assertTrue("$said: $out", out.isNotEmpty() && opens.any { out.last().text.startsWith(it) })
        }
    }

    /**
     * The sheet's threats are the side NOT to move's: on the opponent's turn, the player's own.
     * "What are they threatening?" there is answered with their best move now (desktop 2ec1747).
     */
    @Test
    fun theirThreatOnTheirTurnIsTheirBestMoveNotThePlayers() {
        val (coach, provider) = coach(listOf("", ""))
        start(coach, Pieces.Side.BLACK)
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
                    byUser = ply % 2 == 0,
                ),
                ply.toDouble(),
            )
        }
        assertFalse(coach.userToMove)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("what are they threatening?", 8.0)
        val out = messages(coach)
        assertTrue(directiveOf(given(provider, 0)), "it is the opponent's move" in directiveOf(given(provider, 0)))
        assertTrue(out.toString(), out.isNotEmpty() && out.last().text.startsWith("It's their move, and their best is Qxf7#"))
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
        // And as the pilot asked it, answered from the threats, never examined as a named line
        // (desktop 653a473, whose commit names these three).
        for (said in listOf("What is he threatening?", "what are they up to", "whats the threat")) {
            assertTrue(said, ASK_THREAT.containsMatchIn(said) && !mayCarryClaims(said))
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
     * had said the figure is for when he asks (2026-09-20; desktop 6d9ba6f). The better move is
     * named as "Qxf7# was better", never "the engine preferred" (desktop 47138dd).
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
        // Taught from the position the move was played from, as the lesson after it was.
        val asked = given(provider).last().content
        assertTrue(asked, "# The position they faced, before Bxf7+" in asked)
        val directive = directive(provider)
        assertTrue(directive, "They ask about the move they already played, Bxf7+" in directive)
        assertTrue(directive, "Qxf7# was better" in directive)
        assertTrue(directive, "from a position that is past, so all of it may be said" in directive)
        assertTrue(directive, "45%" !in directive && "No figures" in directive)
        // Spoken from the position the move was played from, with everything shown.
        assertTrue(out.isNotEmpty() && out.last().level == DisclosureLevel.LINE)
        assertEquals(SCHOLAR, out.last().fen)
        // The fallback is the verdict itself, not a hint -- and without the figure.
        assertTrue(out.last().text, out.last().text.startsWith("Qxf7# was better"))
    }

    /**
     * The verdict and the better move came back in every reply, and "I understand: I should play
     * Nxc3" got the lesson again and "Which White piece is still at home on g1?" (the AI's
     * sessions, 2026-09-26; desktop 9e49622). With a model the words are the model's to read, with
     * the sheet of the position the move was played from beside them (Tai's puzzle, 2026-09-28).
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
        coach.ask("I see, I should have seen the mate.", 12.0)
        messages(coach)
        var directive = directive(provider)
        assertTrue(directive, "do not say it again in other words" in directive)
        assertTrue(directive, "They ask nothing: reply in one short sentence" in directive)
        assertTrue(directive, "then one question" !in directive)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("Why does the bishop check not work?", 13.0)
        messages(coach)
        val request = synchronized(provider.prompts) { provider.prompts.last().second[0].content }
        assertTrue(request, "# The position they faced, before Bxf7+" in request)
        directive = directive(provider)
        assertTrue(directive, "Answer them directly" in directive && "do not say it again" in directive)
    }

    /**
     * The follow-up's directive says not to repeat the better move, and the check that a reply
     * names it refused every reply that obeyed; after two, the template said the verdict again
     * (the prompt audit, 2026-09-28; desktop test of the same name). The floor, the lesson again,
     * goes with it.
     */
    @Test
    fun aFollowUpAboutTheirMoveNeedNotNameTheBetterMoveAgain() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(coach.lock) {
            coach.conversation.coachSaid("That one cost you: Qxf7# was better -- it is mate.")
        }
        val reply = "A check that leads nowhere only hands their king a free move."
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf(reply, reply))
        }
        coach.ask("Why does the bishop check not work?", 13.0)
        val out = messages(coach)
        val directive = directive(provider)
        assertTrue(directive, "says at least this" !in directive)
        assertTrue(out.isNotEmpty())
        assertEquals(reply, out.last().text)
    }

    /**
     * Asked what it cost, the figure is said (desktop 6d9ba6f), in the desktop's words, which reach
     * the answer about the last move through askedOwnMove (desktop f733318).
     */
    @Test
    fun whatAMoveCostIsSaidWhenTheyAskIt() {
        val (coach, provider) = coach(listOf("...", "..."))
        bishopCheckInsteadOfMate(coach)
        synchronized(provider.replies) {
            provider.replies.clear()
            provider.replies.addAll(listOf("", ""))
        }
        coach.ask("how much did that cost me?", 12.0)
        val out = messages(coach)
        val directive = directive(provider)
        assertTrue(directive, "it cost 45% of the game: they asked" in directive)
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
        coach.messages += RULE_SPACING // past the decisive question's rule of thumb (desktop 3d9568d)
        val rulesBefore = coach.rulesSaid()
        val (_, fine) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        // The better move and what it does -- and no rule of thumb: a fine move is not a lesson
        // (Tai, 2026-09-27: "the coach is a teacher, not a nagging annoying bot"; desktop f983cd0).
        assertTrue(fine, FINE_TIGHTER.any { fine.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(fine, "it is mate" in fine)
        assertTrue("${coach.rulesSaid()} $fine", coach.rulesSaid() == rulesBefore && !ANNOUNCED.containsMatchIn(fine))
        // Said once, the lesson is not said again a move later: the verdict and the better move
        // stand alone (Nova bot games, 2026-09-25: 27% of lessons came again within three moves).
        coach.conversation.coachSaid(fine)
        val (_, again) = coach.instruction(
            Trigger.POST_MOVE, position, DisclosureLevel.LINE, null, emptyList(), good, played,
        )
        assertTrue(again, FINE_TIGHTER.any { again.startsWith(it.replace("{best}", "Qxf7#")) })
        assertTrue(again, "it is mate" !in again && coach.rulesSaid() == rulesBefore)
        assertTrue(instruction, "tells them nothing: do not open with it" in instruction)
    }

    /**
     * `test_a_rule_of_thumb_cools_down_across_games_and_puzzles` (desktop 3d9568d). Tai,
     * 2026-09-27, after a run of puzzles: "it keeps on nagging me about some lessons like 'A rule
     * to take with you' ... there has to be a cooldown of some number of messages before it can
     * appear again". A rule is not said again for [RULE_COOLDOWN] of the coach's messages,
     * whatever game or puzzle they fall in, and no two rules come within [RULE_SPACING] messages;
     * the model is told to leave it out while it cools.
     */
    @Test
    fun aRuleOfThumbCoolsDownAcrossGamesAndPuzzles() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val position = coach.positions.getValue(6)
        val board = Board(SCHOLAR)
        val move = board.parseSan("a3")!!
        val played = MovePlayed(
            gameId = "g",
            ply = 7,
            san = "a3",
            uci = move.uci(),
            fenBefore = SCHOLAR,
            fenAfter = SCHOLAR,
            byUser = true,
        )
        val mistake = MoveAssessment(
            gameId = "g",
            ply = 7,
            uci = move.uci(),
            expectedLoss = 0.5,
            severity = Severity.MISTAKE,
            best = listOf("h5f7"),
            label = Label.MISTAKE,
        )

        fun lesson(): Pair<String, String> {
            coach.conversation.entries.clear() // a new puzzle: nothing said in this game yet
            return coach.instruction(
                Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, null, emptyList(), mistake, played,
            )
        }

        // The rules said from here on, in the order said (the desktop spies on `_rule_said`).
        val baseline = coach.rulesSaid()
        fun saidRules(): List<String> =
            coach.rulesSaid().entries.filter { (rule, at) -> baseline[rule] != at }.sortedBy { it.value }.map { it.key }

        fun hasRule(text: String) = saidRules().any { it.lowercase().trimEnd('.') in text.lowercase() }

        // The decisive question's first hint said its principle: no rule for a few messages.
        var (instruction, soon) = lesson()
        assertTrue(soon, !hasRule(soon) && RULE_HELD in instruction)
        coach.messages += RULE_SPACING
        val (firstInstruction, first) = lesson()
        assertTrue(first, hasRule(first) && RULE_HELD !in firstInstruction)
        // Said as a sentence of its own, never announced (Tai, 2026-09-27: "nagging stuff like
        // 'Remember this' 'Keep this one'"; desktop f983cd0).
        assertTrue(first, sentenceOf(saidRules().last()) in first && !ANNOUNCED.containsMatchIn(first))
        assertTrue(firstInstruction, "pattern to remember" in firstInstruction)
        repeat(RULE_COOLDOWN - 1) {
            coach.messages += 1
            val (held, again) = lesson()
            instruction = held
            assertTrue(again, !hasRule(again))
            // Held, the directive asks for no rule at all, not for one and then for none: the
            // local model gave one anyway in 2 of 17 lessons (the free check, desktop 5b3d4bf).
            assertTrue(instruction, RULE_HELD in instruction)
            assertTrue(instruction, "pattern to remember" !in instruction && "pattern to keep" !in instruction)
        }
        coach.messages += 1
        val (_, back) = lesson()
        assertTrue("back after the cooldown: $back", hasRule(back))
        // Any two rules keep their distance, and a sound move's principle is a rule too.
        assertFalse(coach.ruleOk("a different rule"))
        assertTrue("takeFile" in coach.verdictTaught())
        coach.messages += RULE_SPACING
        assertTrue(coach.ruleOk("a different rule"))
        coach.taughtByVerdict("takeFile")
        assertTrue("takeFile" in coach.verdictTaught() && !coach.ruleOk(null))
        coach.messages += RULE_SPACING
        assertTrue("takeFile" in coach.verdictTaught() && "castle" !in coach.verdictTaught())
        coach.messages += RULE_COOLDOWN
        assertTrue(coach.ruleOk("takeFile")) // cooled; the last few lessons' memory is its own
    }

    /**
     * `test_a_hints_principle_cools_down_too` (desktop 3d9568d): the first hint's principle ("A
     * move that attacks something they must then defend keeps you a step ahead: ...") came 218
     * times in the master games' hints, 59 of them again in the same game: it is a rule of thumb
     * like the lesson's, left out while it cools.
     */
    @Test
    fun aHintsPrincipleCoolsDownToo() {
        val (coach, _) = coach(emptyList())
        val rule = "A move that attacks something they must then defend keeps you a step ahead."
        val hint = "You can take the initiative. $rule Their pawn on d3 can be hit."
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "middlegame", situation = hint, principle = rule,
        )
        var (said, note) = coach.coolHint(facts, hint)
        assertTrue(said == hint && note.isEmpty())
        coach.messages += RULE_COOLDOWN - 1
        val cooled = coach.coolHint(facts, hint)
        said = cooled.first
        note = cooled.second
        assertEquals("You can take the initiative. Their pawn on d3 can be hit.", said)
        assertTrue(note, rule in note && "heard it lately" in note)
        coach.messages += 1
        assertEquals(hint to "", coach.coolHint(facts, hint))
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
            "the pawn on d2."
        val rule = "The centre is where pieces have the most room."
        val text = "Playable -- d4 was a little more precise -- your bishop on c1 is still shut in behind " +
            "the pawn on d2. $rule"
        // The reason said lately goes; a rule of thumb is not `fresh`'s to hold -- one said lately
        // is never written in ([Coach.ruleOk]), so a rule that is here stays (desktop f983cd0).
        assertEquals("Playable -- d4 was a little more precise. $rule", fresh(text, said))
        assertEquals(text, fresh(text, "Nice."))
        var opener = "Playable -- d4 was a little more precise"
        assertEquals("$opener. $rule", fresh(text, said, opener))
        // An opener with its own " -- " and no reason after it: the better move is not a reason,
        // and stays when the same move was just said ("Not that one." -- Nova bot games).
        opener = "Not that one -- Ng3 was better"
        val just = "That one cost you: Ng3 was better. After c6, they win a pawn."
        assertEquals("$opener.", fresh("$opener.", just.lowercase(), opener))
        assertEquals("$opener.", fresh("$opener -- it wins a pawn.", "$just It wins a pawn.", opener))
    }

    /**
     * Desktop f363ea0 (which adds no test of its own): asked "why was that bad?", an answer that
     * would name nothing but the better move gives the engine's line from the position the move
     * was played from -- "g6 was better." alone answered nothing, 13 of 95 answers in the
     * desktop's offline sweep of bot games (2026-09-27). An answer with a reason, or a line that
     * does not start with the better move, stands as it is.
     */
    @Test
    fun anAnswerThatNamesOnlyTheBetterMoveGivesTheLine() {
        val facts = PositionFacts(
            fen = SCHOLAR, sideToMove = "white", phase = "middlegame",
            line = listOf("g6", "O-O-O", "Bg7", "Qf3", "Ne8", "Nf6"),
        )
        val opener = "g6 was better"
        assertEquals("g6 was better -- the line runs g6 O-O-O Bg7 Qf3 Ne8.", withLine("g6 was better.", opener, "g6", facts))
        val reasoned = "g6 was better -- it keeps the long diagonal."
        assertEquals(reasoned, withLine(reasoned, opener, "g6", facts))
        assertEquals("h6 was better.", withLine("h6 was better.", "h6 was better", "h6", facts))
        assertEquals("g6 was better.", withLine("g6 was better.", opener, "g6", null))
    }

    /**
     * `test_offline_coach.py`: four decisive moments in one game got the same sentence four times
     * (a master-game pass, 2026-09-21): each list is a bag, every wording once before any comes
     * again (Tai, 2026-09-25), and never the last one said.
     */
    @Test
    fun theTemplatesRotateAndNeverRepeatTheLast() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        for ((key, options) in listOf("cost" to COST_YOU, "fine" to FINE_TIGHTER)) {
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
     * for word, three times (the students' round 5): a move proposed is examined as a question
     * about it is, and the model answers from the examination; and a template already said since
     * the last move is not said again (desktop adc4b26, 3ff672b).
     */
    @Test
    fun aMoveProposedInAStatementIsExaminedAndNothingIsSaidTwice() {
        val engine = EngineLike { board, _ ->
            if (board.turn == WHITE) {
                listOf(Line(moves = listOf("f1b5"), depth = 14, expectedScore = 0.55))
            } else {
                listOf(Line(moves = listOf("f8c5", "d2d3"), depth = 14, expectedScore = 0.48))
            }
        }
        val (coach, provider) = coach(listOf("[p=1.0] Fine."))
        coach.engine = engine
        italian(coach)
        coach.ask("I'll play Bc4 and aim at f7.", 6.0)
        assertEquals(listOf("Fine."), messages(coach).map { it.text })
        val directive = directiveOf(given(provider))
        assertTrue(directive, "Engine on 3. Bc4, which the player asked about" in directive)
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
     * third mistake of one cause in the session says the pattern once, with its habit. Said the
     * second time and then in every game, "this keeps happening -- 5 times now" nagged (Tai's
     * phone, 2026-09-27); the review sums it up after (desktop f983cd0).
     */
    @Test
    fun aCauseThatComesBackIsNamedOnceWithTheHabitThatFixesIt() {
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
        assertEquals("twice is not a pattern yet", emptyList<String>(), placed(25, Causes.MISSED_THREAT))
        val third = placed(31, Causes.MISSED_THREAT)
        assertTrue(third.toString(), third.size == 1 && "three times" in third[0])
        assertTrue(third[0], "what their moves threatened" in third[0])
        assertTrue(third[0], "look at what their last move attacks" in third[0])
        // Once a session: not again in this game, nor in the next.
        assertEquals(emptyList<String>(), placed(37, Causes.MISSED_THREAT))
        coach.gameId = "g2"
        assertEquals(emptyList<String>(), placed(9, Causes.MISSED_THREAT, gameId = "g2"))
        assertEquals(emptyList<String>(), placed(15, Causes.MISSED_THREAT, gameId = "g2"))

        // A player of 1800 and up hears the pattern without the textbook habit.
        val (strongCoach, _) = coach(emptyList())
        start(strongCoach)
        strongCoach.strongPlayer = true
        for (ply in listOf(9, 13, 17)) {
            strongCoach.onEvent(
                ProfileEvidence(gameId = "g", ply = ply, phase = "middlegame", expectedLoss = 0.2, cause = Causes.MISSED_THREAT),
                ply.toDouble(),
            )
        }
        val strong = messages(strongCoach).map { it.text }
        assertTrue(strong.toString(), strong.size == 1 && "what their moves threatened" in strong[0])
        assertTrue(strong[0], "look at what" !in strong[0])
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
        userText: String? = null,
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
            userText = userText,
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
     * same, and no figure in a remark nobody asked for -- the last [NO_FIGURES]' note, where the
     * directive carries it ([Speaker.notes]) (desktop f25217c).
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
        // The better move castles: saying so is the lesson, not a restatement.
        assertEquals(
            emptyList<String>(),
            restated("Castling first was better: the king is safe, the rook joins in.", trade),
        )
        // The better move takes on the same square with check: the words are about it.
        val miss = afterMoveRequest(SCHOLAR, "Bxf7+", "h5f7")
        assertEquals(emptyList<String>(), restated("Taking the pawn on f7 with check was right, but with the queen.", miss))
        // The check said of the move played is a restatement; a check in the better line is not
        // (the benchmark's smoke run, 2026-09-25: "after Bg4 your knight jumps in with check";
        // desktop 1117852).
        val f5 = "rnbqkbnr/ppppp1pp/8/5p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        val check = afterMoveRequest(f5, "Qh5+", "e4f5")
        val said = "says what Qh5 plainly did: the check"
        assertTrue(said in restated("Sure, Qh5+ gives check, but exf5 wins a pawn.", check))
        assertTrue(said in restated("That gave check, but exf5 wins a pawn.", check))
        assertEquals(emptyList<String>(), restated("exf5 wins a pawn, and after Nf6 your queen comes in with check.", check))
    }

    /**
     * The first hint on the sheet is facts to teach from, not the least a reply says, so the
     * length is the directive's own: one more for the question it offers on top ("then, if you
     * like, one question"), which it does not require (the prompt audit, 2026-09-28).
     */
    @Test
    fun theFirstHintAsksTwoOrThreeSentencesAndOffersAQuestion() {
        assertEquals(4 to null, directiveLimits(IDEA_DIRECTIVE))
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

    /**
     * `test_verdicts._played`: a sound move of the player's from the scholar's-mate position,
     * [san], labelled [label], losing [loss]; what the coach said.
     */
    private fun soundMove(coach: Coach, san: String, label: Label, loss: Double = 0.01): List<String> {
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val board = Board(SCHOLAR)
        val move = board.parseSan(san)!!
        board.push(move)
        coach.onEvent(
            MovePlayed(gameId = "g", ply = 7, san = san, uci = move.uci(), fenBefore = SCHOLAR, fenAfter = board.fen(), byUser = true),
            8.0,
        )
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = 7, uci = move.uci(), expectedLoss = loss, severity = Severity.NONE,
                best = listOf("h5f7"), label = label,
            ),
            8.5,
        )
        return messages(coach).map { it.text }
    }

    /**
     * Who writes the word on a sound move, by its label (Tai, 2026-09-20 and 2026-09-26: "They need
     * to go through the api"): theory, the engine's own move and an excellent one get the board's
     * sentence and no call (the routine model's, with the local verdicts off); a good move goes to
     * the model, which names the better one; a great or brilliant move to the MAIN model at its
     * full effort, told to show some delight, the board's line standing in when the reply fails.
     * The routine model is a separate one here, so a delight asked of it would show.
     * `test_verdicts.py`: the coach writes a sound move's word itself; a good move goes to the
     * model; a great or brilliant move gets the model's delight.
     */
    @Test
    fun aSoundMovesWordIsRoutedByItsLabel() {
        fun routed(replies: List<String>, localVerdicts: Boolean = true): Triple<Coach, ScriptedProvider, ScriptedProvider> {
            val main = ScriptedProvider(replies)
            val routine = ScriptedProvider(replies)
            val coach = Coach(
                main,
                policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
                routineProvider = routine,
                localVerdicts = localVerdicts,
            )
            return Triple(coach, main, routine)
        }
        // The board's own word: no call at all.
        for ((san, label) in listOf("Qxf7#" to Label.BEST, "Nc3" to Label.EXCELLENT, "Nc3" to Label.BOOK)) {
            val (coach, main, routine) = routed(listOf("never used"))
            val texts = soundMove(coach, san, label)
            assertTrue("$label: $texts", texts.size == 1 && !texts[0].startsWith(san.trimEnd('#', '+')))
            assertTrue("$label: no model", main.prompts.isEmpty() && routine.prompts.isEmpty())
            if (label == Label.BEST) assertTrue(texts[0], texts[0] in MATE)
            if (label == Label.EXCELLENT) {
                assertTrue(texts[0], texts[0].split(" -- ")[0].trimEnd('.') in OPENERS.getValue("excellent"))
            }
        }
        // The local verdicts off, the routine model writes it.
        run {
            val (coach, main, routine) = routed(listOf("[p=1.0] Mate on the board -- well played."), localVerdicts = false)
            assertEquals(listOf("Mate on the board -- well played."), soundMove(coach, "Qxf7#", Label.BEST))
            assertTrue(main.prompts.isEmpty() && routine.prompts.size == 1)
        }
        // A good move: the main model names the better one, the label said in its sense.
        run {
            val (coach, main, routine) = routed(listOf("[p=1.0] Fine, but Qxf7# was there: the mate."))
            assertEquals(listOf("Fine, but Qxf7# was there: the mate."), soundMove(coach, "Nc3", Label.GOOD, loss = 0.03))
            assertTrue(main.prompts.size == 1 && routine.prompts.isEmpty())
            val task = directive(main)
            assertTrue(task, "which was good: sound, a little behind Qxf7#" in task)
            assertTrue(task, "tells them nothing: do not open with it" in task)
        }
        // Great and brilliant: the main model, with delight; the board's line if it fails.
        for (label in listOf(Label.GREAT, Label.BRILLIANT)) {
            val (coach, main, routine) = routed(listOf("[p=1.0] Yes -- the one move, and it ends the game."))
            assertEquals(listOf("Yes -- the one move, and it ends the game."), soundMove(coach, "Qxf7#", label, loss = 0.0))
            assertTrue("$label: the main model", main.prompts.size == 1 && routine.prompts.isEmpty())
            val task = directive(main)
            assertTrue(task, "Show some delight" in task && label.name.lowercase() in task)
            assertTrue(task, "tells them nothing: do not open with it" in task)
            val (failing, _, _) = routed(listOf("", ""))
            val fallback = soundMove(failing, "Qxf7#", label, loss = 0.0)
            assertTrue("$label: $fallback", fallback.size == 1 && fallback[0] in MATE)
        }
    }

    /** `_mate_in_one_played`: the user plays Qxf7#, the engine's own move, after the scholar's position. */
    private fun mateInOnePlayed(coach: Coach, gameId: String = "g", ply: Int = 7, start: String = SCHOLAR, ts: Double = 10.0) {
        val board = Board(start)
        val move = board.parseSan("Qxf7#")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = gameId, ply = ply, san = "Qxf7#", uci = move.uci(), fenBefore = start, fenAfter = board.fen(), byUser = true), ts)
        coach.onEvent(
            MoveAssessment(gameId = gameId, ply = ply, uci = move.uci(), expectedLoss = 0.0, severity = Severity.NONE, best = listOf("h5f7"), label = Label.BEST),
            ts + 0.5,
        )
    }

    /** `_play` of the given moves from the start, the user White, each at its ply's second. */
    private fun playFromStart(coach: Coach, sans: List<String>, thinkMs: Int? = null): Board {
        start(coach)
        val board = Board()
        for ((i, san) in sans.withIndex()) {
            val ply = i + 1
            val move = board.parseSan(san)!!
            val before = board.fen()
            board.push(move)
            coach.onEvent(
                MovePlayed(gameId = "g", ply = ply, san = san, uci = move.uci(), fenBefore = before, fenAfter = board.fen(), byUser = ply % 2 == 1, thinkMs = thinkMs),
                ply.toDouble(),
            )
        }
        return board
    }

    /** Naming the move yourself earns the candidate level (`test_coach.py`). */
    @Test
    fun namingTheMoveYourselfEarnsTheCandidateLevel() {
        val (coach, _) = coach(listOf("What do you think?", "Yes, Qxf7 is mate."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        coach.ask("Is Qxf7 mate?", 8.0)
        assertEquals(DisclosureLevel.MOVE, messages(coach)[0].level)
        val board = Board(SCHOLAR)
        assertEquals(Unlock.USER_NAMED_MOVE, unlockFor("Is Qxf7 mate?", board, listOf("Qxf7#")))
        assertEquals("not the candidate", null, unlockFor("What about Bxf7?", board, listOf("Qxf7#")))
        assertEquals(Unlock.ASK_LINE, unlockFor("show me the line", board, emptyList()))
        assertEquals(Unlock.HINT, unlockFor("HINT please", board, emptyList()))
        assertEquals(Unlock.ASK_MOVE, unlockFor("what should I play here", board, emptyList()))
        assertEquals(null, unlockFor("how are you", board, emptyList()))
    }

    /**
     * The model offers a leaky reply and a clean one in one distribution: the checked one is kept
     * at the first attempt; and a stump ("Given"), clean by default, never wins -- both calls are
     * kept for the log (`test_coach.py`, `test_coach_verbalized.py`).
     */
    @Test
    fun theCheckedCandidateIsKeptAndAStumpNeverWins() {
        val checker = DraftCheck { text, _, _, _ -> if ("rook on f7" in text) listOf("the rook on f7 is not there") else emptyList() }
        fun checked(replies: List<String>): Pair<Coach, List<CoachMessage>> {
            val coach = Coach(ScriptedProvider(replies), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)), draftCheck = checker)
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6), 7.0)
            return coach to messages(coach)
        }
        var (coach, out) = checked(listOf("[p=0.6] Your rook on f7 is strong.\n[p=0.4] Their king on e8 is short of squares, is it not?"))
        assertEquals("Their king on e8 is short of squares, is it not?", out[0].text)
        assertTrue(coach.spoken.last().attempts == 1 && coach.speaker.fallbacks == 0)
        val first = "[p=0.6] Your rook on f7 is strong. What next?\n[p=0.4] Given"
        val second = "[p=0.7] Their king on e8 is short of squares. What is their threat?"
        val stump = checked(listOf(first, second))
        coach = stump.first
        out = stump.second
        assertEquals("Their king on e8 is short of squares. What is their threat?", out[0].text)
        val spoken = coach.spoken.last()
        assertEquals(2, spoken.attempts)
        assertEquals(listOf(first, second), spoken.calls.map { it.raw })
        assertEquals(listOf(1 to 0, 1 to 1), spoken.calls.map { it.candidates to it.passed })
    }

    /**
     * "hint" carries no claim: prompted to be honest about "their claim", the model answered its
     * own previous message as though the player had said it (`test_coach_verbalized.py`).
     */
    @Test
    fun aBareHintIsAskedForAsAHintNotAsAClaim() {
        for (said in listOf("hint", "Hint", "give me a hint", "a clue please", "help", "any tips?")) {
            assertTrue(said, ASK_HINT.matchesAt(said, 0))
        }
        for (said in listOf("what should I play", "is Nf3 a good idea", "help me with the knight")) {
            assertFalse(said, ASK_HINT.matchesAt(said, 0))
        }
        val (coach, provider) = coach(listOf("Where is their king weakest?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.hint(8.0)
        messages(coach)
        val prompt = given(provider, 0).last().content
        assertTrue(prompt, "They asked for a hint" in prompt)
        assertTrue(prompt, "their own claim" !in prompt && "anything said earlier" in prompt)
    }

    /** A re-read of the board is a message and the numbering restarts (`test_coach.py`). */
    @Test
    fun aRereadOfTheBoardIsAMessageAndTheNumberingRestarts() {
        val (coach, provider) = coach(listOf("Understood."))
        playToScholar(coach)
        val fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4"
        coach.onEvent(Resync(gameId = "g", fen = fen, reason = "position adopted after 3 unexplained cells"), 20.0)
        val board = Board(fen)
        val move = board.parseSan("d6")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = "g", ply = 1, san = "d6", uci = move.uci(), fenBefore = fen, fenAfter = board.fen(), byUser = false), 21.0)
        coach.onEvent(
            Analysis(gameId = "g", fen = board.fen(), ply = 1, lines = listOf(Line(moves = listOf("c3d5"), depth = 12, expectedScore = 0.6)), decisive = false, onlyMove = false),
            22.0,
        )
        coach.ask("what now?", 23.0)
        messages(coach)
        val text = harness(given(provider))
        assertTrue(text, "    EVENT: The board was re-read" in text && "Moves may have gone unseen" in text)
        // The numbering follows the position the board was read at, not a ply count.
        assertTrue("4... d6 by the opponent" in currentMoveMessage(given(provider)))
    }

    /**
     * The claims in the player's words are checked off the caller's thread -- the caller never
     * waits -- and only when there are any: a bare hint never runs the check (`test_coach.py`).
     */
    @Test
    fun claimsAreCheckedOffTheCallersThreadAndOnlyWhenThereAreAny() {
        assertFalse(mayCarryClaims("hint"))
        assertFalse(mayCarryClaims("what should I play"))
        assertFalse(mayCarryClaims("show me the line"))
        assertTrue(mayCarryClaims("is Nf3 good here"))
        assertTrue(mayCarryClaims("my knight is hanging"))
        assertTrue(mayCarryClaims("can I castle"))
        val threads = java.util.Collections.synchronizedList(ArrayList<String>())
        val started = java.util.concurrent.CountDownLatch(1)
        val done = java.util.concurrent.CountDownLatch(1)
        val provider = ScriptedProvider(listOf("Not quite. What is your most active piece already eyeing?"))
        val coach = Coach(
            provider,
            policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)),
            claims = ClaimsCheck { _, _, _ ->
                threads.add(Thread.currentThread().name)
                started.countDown()
                Thread.sleep(200)
                listOf("'Nf3 is good': not the strongest, Qxf7# mates").also { done.countDown() }
            },
        )
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        val begun = System.nanoTime()
        coach.ask("is Nf3 good here", 8.0)
        assertTrue("the caller never waits on the check", System.nanoTime() - begun < 100_000_000L)
        assertTrue(started.await(2, java.util.concurrent.TimeUnit.SECONDS))
        assertTrue(done.await(2, java.util.concurrent.TimeUnit.SECONDS))
        Thread.sleep(200) // the worker hands the request over after the check
        assertTrue(threads.toString(), threads.isNotEmpty() && threads[0] != Thread.currentThread().name)
        assertEquals(listOf("Not quite. What is your most active piece already eyeing?"), messages(coach).map { it.text })
        assertTrue("Their claims, checked against the board" in given(provider, 0).last().content)
        coach.hint(9.0) // a bare hint never runs the check
        messages(coach)
        assertEquals(1, threads.size)
    }

    /**
     * "How did I do overall?" was answered "you were down a queen" to a player who had won one:
     * the material is the user's side's, a finished game withholds nothing, the overall verdict
     * is drawn from the tally, and when the model's review fails the template is the record
     * (`test_coach.py`).
     */
    @Test
    fun materialIsAlwaysFromTheUsersSideAndTheReviewDrawsOnTheTally() {
        val (coach, provider) = coach(listOf("You kept it clean: nothing lost, and the finish was sharp."))
        val board = playFromStart(coach, listOf("e4", "e5", "Qh5", "Nc6", "Qxe5+"))
        coach.onEvent(
            Analysis(gameId = "g", fen = board.fen(), ply = 5, lines = listOf(Line(moves = listOf("f8e7"), depth = 12, expectedScore = 0.4)), decisive = false, onlyMove = false),
            6.0,
        )
        val facts = coach.positions[5]!!.facts!!
        assertEquals("from the user's side, not the mover's", "you are up a pawn", facts.material)
        assertTrue(facts.userSide == "white" && facts.sideToMove == "black")
        assertTrue("Material: the player is up a pawn." in coach.conversation.renderPly(coach.conversation.records.last(), DisclosureLevel.LINE))
        for ((ply, pair) in mapOf(1 to ("e2e4" to Label.BOOK), 3 to ("d1h5" to Label.INACCURACY), 5 to ("h5e5" to Label.BEST))) {
            val (uci, label) = pair
            val inaccuracy = label == Label.INACCURACY
            coach.onEvent(
                MoveAssessment(
                    gameId = "g", ply = ply, uci = uci, expectedLoss = if (inaccuracy) 0.03 else 0.0,
                    severity = if (inaccuracy) Severity.INACCURACY else Severity.NONE, best = listOf(uci), label = label,
                ),
                7.0,
            )
        }
        messages(coach)
        coach.onEvent(GameEnded(gameId = "g", result = "1-0", reason = "resigned", pgn = "", finalFen = board.fen()), 8.0)
        coach.ask("how did I do overall this game?", 9.0)
        val out = messages(coach)
        assertTrue(out.isNotEmpty() && out.last().level == DisclosureLevel.LINE)
        val directive = directiveOf(given(provider))
        assertTrue(directive, "They asked how they did overall" in directive)
        assertTrue(directive, "do not infer the course of the game from the position" in directive)
        val text = harness(given(provider))
        assertTrue(text, "The player's moves so far (3 judged): 1 best, 1 book, 1 inaccuracy." in text)
        assertTrue(text, "Ground lost most at: 2. Qh5 (inaccuracy)" in text)
        assertTrue(text, "    EVENT: Game over: 1-0" in text)
        for (words in listOf("what do I keep missing?", "where did I lose it?", "how am I playing so far?")) {
            assertTrue(words, ASK_REVIEW.containsMatchIn(words))
        }
        assertFalse(ASK_REVIEW.containsMatchIn("what's the plan so far?"))
        provider.replies.clear()
        provider.replies.addAll(listOf("", ""))
        coach.ask("what do I keep missing?", 10.0)
        assertEquals(
            "1 of your 3 moves were best or excellent. You lost the most ground at 2. Qh5 (inaccuracy).",
            messages(coach).last().text,
        )
    }

    /**
     * The model gives nothing usable, twice: the template for the rung speaks in its place, once,
     * and the failed calls are in the log (`test_coach.py`, desktop 3ff672b).
     */
    @Test
    fun aHintTheModelFailsTwiceIsTheTemplates() {
        val (coach, provider) = coach(listOf("", ""))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        coach.ask("hint", 8.0)
        val out = messages(coach)
        // The idea template: the goal and the pattern.
        assertTrue(out.map { it.text }.toString(), out.size == 1 && "mate in one" in out[0].text.split(".")[0])
        // Nothing was said before the model was asked: their words are the last turn.
        val first = given(provider, 0)
        val said = currentMoveMessage(first).substringAfter("## Said here").trim('\n').lines()
        assertEquals("    PLAYER said: \"hint\"", said.last())
        val directive = directiveOf(first)
        assertTrue(directive, "They asked for a hint." in directive)
        val spoken = coach.spoken.last()
        assertTrue(spoken.attempts == 2 && spoken.problems.isNotEmpty() && coach.speaker.fallbacks == 1)
        assertEquals(listOf(out[0].text), coach.conversation.entries.filter { it.role == "assistant" }.map { it.text })
    }

    /** A question about a fine last move says it was fine and invents no fault (`test_coach.py`). */
    @Test
    fun aQuestionAboutAFineLastMoveSaysItWasFine() {
        val (coach, provider) = coach(listOf("...", ""))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        messages(coach)
        mateInOnePlayed(coach, ts = 10.0)
        messages(coach)
        coach.ask("was that a mistake?", 12.0)
        messages(coach)
        val directive = directiveOf(given(provider))
        assertTrue(directive, "its verdict was best, and it cost nothing" in directive && "invent no fault" in directive)
    }

    /**
     * Asked why the opponent played their last move, the coach names it: a move already played
     * may be discussed in full, never a leak (`test_coach.py`).
     */
    @Test
    fun aMoveOfTheGameSoFarIsNeverALeak() {
        val (coach, _) = coach(listOf("[p=1.0] Nf6 develops with a threat; what does it leave behind?"))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("why did he play that?", 8.0)
        val out = messages(coach)
        assertTrue(out.isNotEmpty() && out.last().text.startsWith("Nf6 develops"))
        assertEquals(DisclosureLevel.IDEA, out.last().level)
    }

    /**
     * A question about the player over their games goes to the record's answerer off the coach's
     * thread; the answer is the coach's turn, the lookup noted; a question about this game stays
     * with the game (`test_coach.py`).
     */
    @Test
    fun aQuestionAboutThePlayerOverTheirGamesGoesToTheRecord() {
        val (coach, _) = coach(listOf("Keep looking."))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val asked = java.util.Collections.synchronizedList(ArrayList<String>())
        coach.profileAsk = { question ->
            asked.add(question)
            ProfileAnswer(
                text = "Over 9 games you left the opening ahead 7 times and were behind by move 25.",
                exchanges = listOf("SELECT 1"),
                calls = 2,
            )
        }
        coach.ask("why am I always losing in the middlegame?", 7.0)
        // The answer lands on the lookup's own thread: gathered until it is out.
        val said = ArrayList<CoachMessage>()
        val deadline = System.nanoTime() + 5_000_000_000L
        while (said.none { it.text.startsWith("Over 9 games") } && System.nanoTime() < deadline) {
            said.addAll(messages(coach))
            Thread.sleep(20)
        }
        assertEquals(listOf("why am I always losing in the middlegame?"), asked.toList())
        assertTrue(said.isNotEmpty() && said.last().text.startsWith("Over 9 games") && said.last().trigger == Trigger.USER_ASK)
        assertEquals("answer", coach.profileAnswers[0].first)
        val record = coach.conversation.render()
        assertTrue("PLAYER said: \"why am I always losing in the middlegame?\"" in record)
        assertTrue("looked the question up in the record (1 query)" in record)
        coach.ask("what should I play here?", 8.0)
        assertEquals(1, asked.size)
    }

    /**
     * Every third mistake ends by asking what the player saw, in the coach's own words, while the
     * player answers; one that goes unanswered is the last of the game (`test_coach.py`).
     */
    @Test
    fun everyThirdMistakeEndsByAskingWhatThePlayerSaw() {
        val (coach, _) = coach(emptyList())
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 6.5)
        val position = coach.positions[6]!!
        val played = MovePlayed(gameId = "g", ply = 7, san = "a3", uci = "a2a3", fenBefore = SCHOLAR, fenAfter = SCHOLAR, byUser = true)
        val assessment = MoveAssessment(gameId = "g", ply = 7, uci = "a2a3", expectedLoss = 0.5, severity = Severity.BLUNDER, best = listOf("h5f7"), label = Label.BLUNDER)
        var (instruction, fallback) = coach.instruction(Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, null, emptyList(), assessment, played)
        assertTrue("expected" !in instruction && "expecting" !in fallback)
        val asked = coach.instruction(Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, null, emptyList(), assessment, played, elicit = true)
        instruction = asked.first
        fallback = asked.second
        assertTrue(instruction, "what they expected their move to do" in instruction)
        assertTrue(instruction, "misjudge" !in instruction && "altogether" !in instruction)
        assertTrue(fallback, fallback.endsWith("What were you expecting that move to do?"))
        val due = ArrayList<Boolean>()
        for (n in 1..2 * ELICIT_EVERY) {
            coach.mistakes = n
            due.add(coach.elicitDue(10.0 * n))
            if (due.last()) coach.ask("I thought the queen would have to move", 10.0 * n + ELICIT_ANSWER_S / 2)
        }
        assertEquals((1..2 * ELICIT_EVERY).map { it % ELICIT_EVERY == 0 }, due)
        coach.mistakes = 3 * ELICIT_EVERY
        assertTrue("answered: asked again", coach.elicitDue(1000.0))
        coach.mistakes = 4 * ELICIT_EVERY
        assertFalse("unanswered: not again this game", coach.elicitDue(2000.0))
        coach.mistakes = 5 * ELICIT_EVERY
        coach.ask("hint", 2000.0 + ELICIT_ANSWER_S + 1)
        assertFalse("a minute later is not an answer", coach.elicitDue(3000.0))
        start(coach)
        coach.mistakes = ELICIT_EVERY
        assertTrue("a new game starts afresh", coach.elicitDue(4001.0))
    }

    /**
     * A puzzle or an analysis board: the coach hears it is a position with no history; the mating
     * move on it gets no word and the end no summary, and nothing calls the model; a move that
     * does not end it gets its word as in a game (`test_coach.py`).
     */
    @Test
    fun aPositionIsNotAGameAndAPuzzleSolvedByMateGetsNoWord() {
        val (told, _) = coach(emptyList())
        val setup = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"
        told.onEvent(GameStarted(gameId = "p", userSide = Pieces.Side.BLACK, startFen = setup, kind = "position"), 0.0)
        val record = told.conversation.render()
        assertTrue(record, "You coach Black." in record && "This is a position, not a game from the start" in record)
        assertTrue(record, "joined in progress" !in record)

        val (coach, provider) = coach(emptyList())
        for (n in 0 until 2) {
            val id = "p$n"
            coach.onEvent(GameStarted(gameId = id, userSide = Pieces.Side.WHITE, startFen = SCHOLAR, kind = "position"), 10.0 * n)
            coach.onEvent(mateAnalysis(0).let { Analysis(gameId = id, fen = it.fen, ply = 0, lines = it.lines, decisive = it.decisive, onlyMove = it.onlyMove, tags = it.tags) }, 10.0 * n + 0.5)
            messages(coach) // the decisive-moment nudge on the position itself
            provider.prompts.clear()
            mateInOnePlayed(coach, gameId = id, ply = 1, ts = 10.0 * n + 1.0)
            assertEquals("the mating move is not commented on", emptyList<CoachMessage>(), messages(coach))
            coach.onEvent(GameEnded(gameId = id, result = "1-0", reason = "game over on board", pgn = "1. Qxf7#"), 10.0 * n + 2.0)
            assertEquals("no summary of a puzzle", emptyList<CoachMessage>(), messages(coach))
        }
        assertTrue("no model call at all", provider.prompts.isEmpty())
        coach.onEvent(GameStarted(gameId = "q", userSide = Pieces.Side.WHITE, startFen = SCHOLAR, kind = "position"), 50.0)
        coach.onEvent(mateAnalysis(0).let { Analysis(gameId = "q", fen = it.fen, ply = 0, lines = it.lines, decisive = it.decisive, onlyMove = it.onlyMove, tags = it.tags) }, 50.5)
        messages(coach)
        val board = Board(SCHOLAR)
        val move = board.parseSan("Nf3")!!
        board.push(move)
        coach.onEvent(MovePlayed(gameId = "q", ply = 1, san = "Nf3", uci = move.uci(), fenBefore = SCHOLAR, fenAfter = board.fen(), byUser = true), 51.0)
        coach.onEvent(
            MoveAssessment(gameId = "q", ply = 1, uci = move.uci(), expectedLoss = 0.5, severity = Severity.BLUNDER, best = listOf("h5f7"), label = Label.BLUNDER),
            51.5,
        )
        assertEquals(listOf(Trigger.POST_MISTAKE), messages(coach).map { it.trigger })
    }

    /**
     * The user's think times set the pace: in a fast game the word on a move is the template, at
     * once, with no model call; what the player asks still goes to the model; a player who slows
     * down -- in the game, or thinking longer than the pace in the position -- gets the model back
     * (Tai, 2026-09-22; `test_coach.py`).
     */
    @Test
    fun aFastGameGetsTheTemplatesAndSlowingDownBringsTheModelBack() {
        val (coach, provider) = coach(listOf("never used for a move", "[p=1.0] Asked and answered."))
        coach.localVerdicts = false
        val board = playFromStart(coach, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), thinkMs = (FAST_PACE_S * 1000 / 3).toInt())
        assertTrue("two quick moves of the user's and the game counts as fast", coach.fastPace)
        assertTrue(coach.userToMove && coach.fastNow(6.0 + 1.0))
        assertFalse("thinking longer than the pace: slowed", coach.fastNow(6.0 + FAST_PACE_S + 1))
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        mateInOnePlayed(coach)
        var out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MOVE && out[0].text.isNotEmpty())
        assertTrue("no model call in a fast game", provider.prompts.isEmpty())
        coach.ask("what was that about", 12.0)
        out = messages(coach)
        assertTrue(out.isNotEmpty() && out.last().trigger == Trigger.USER_ASK)
        assertTrue("the ask goes to the model", provider.prompts.isNotEmpty())
        for (ply in listOf(8, 9)) {
            coach.onEvent(
                MovePlayed(gameId = "g", ply = ply, san = "a3", uci = "a2a3", fenBefore = board.fen(), fenAfter = board.fen(), byUser = true, thinkMs = (FAST_PACE_S * 3000).toInt()),
                20.0 + ply,
            )
        }
        assertFalse(coach.fastPace)
    }

    /**
     * A puzzle's last moves come quick -- the line found, played out -- and carried over they
     * made the next puzzle fast from its first move: a find after two minutes' thought was only
     * shown (Tai's puzzles, 2026-09-27: "sometimes it's just completely muted"; desktop 8af8695,
     * `test_the_pace_is_the_games_own`).
     */
    @Test
    fun thePaceIsTheGamesOwn() {
        val (coach, _) = coach(emptyList())
        playFromStart(coach, listOf("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), thinkMs = 1000)
        assertTrue(coach.fastPace)
        coach.onEvent(GameStarted(gameId = "h", userSide = Pieces.Side.WHITE, startFen = Board().fen()), 10.0)
        assertTrue(coach.paceS == null && !coach.fastPace)
    }

    /**
     * No model at all (llm.provider none): the factory's offline provider; a decisive moment
     * gets the question template, the player's best move the verdict from the board, a typed
     * question the hint template -- and nothing is ever called (`test_offline_coach.py`).
     */
    @Test
    fun offlineTheTemplatesSpeakAndNothingIsCalled() {
        // The factory gives the offline provider, and it reports itself (the desktop's usage
        // ledger has no phone counterpart: nothing is priced).
        val none = com.sidecoach.llm.makeProvider(com.sidecoach.llm.LLMSettings(provider = "none"))
        assertTrue(none is OfflineProvider && none.offline)
        val (ok, detail) = none.healthcheck()
        assertTrue(detail, ok && "offline" in detail)
        assertTrue((none as OfflineProvider).withEffort("low") === none)
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6), 7.0)
        var out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.DECISIVE_QUESTION && "?" in out[0].text)
        assertEquals(1, coach.speaker.fallbacks)
        coach.ask("hint", 8.0)
        out = messages(coach)
        assertTrue("a hint from the templates", out.size == 1 && out[0].text.isNotEmpty())
        assertEquals(out[0].fen, Board(out[0].fen).fen())
        mateInOnePlayed(coach)
        out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.POST_MOVE)
        assertTrue(out[0].text, out[0].text in MATE && "Qxf7#" !in out[0].text)
        assertEquals("nothing was ever called", 0, coach.speaker.failures)
    }

    /**
     * 9...Nxd4 in a Dragon drew "This is a moment to slow down": when the opponent has just
     * captured and the engine's move takes back there, no decisive question; a zwischenzug as
     * the best move keeps it (`test_offline_coach.py`).
     */
    @Test
    fun noDecisiveQuestionOnAnObviousRecapture() {
        val coach = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
        val board = playFromStart(coach, listOf("e4", "c5", "Nf3", "d6", "d4", "cxd4"))
        fun analysed(first: String, second: String, ts: Double) = coach.onEvent(
            Analysis(
                gameId = "g", fen = board.fen(), ply = 6,
                lines = listOf(Line(moves = listOf(first), depth = 12, expectedScore = 0.52), Line(moves = listOf(second), depth = 12, expectedScore = 0.35)),
                decisive = true, onlyMove = false,
            ),
            ts,
        )
        analysed("f3d4", "c2c3", 7.0)
        assertEquals("the recapture is the move; no question", emptyList<CoachMessage>(), messages(coach))
        analysed("f1b5", "f3d4", 8.0)
        val out = messages(coach)
        assertTrue(out.size == 1 && out[0].trigger == Trigger.DECISIVE_QUESTION)
    }

    /**
     * Game 2 answered "how did I do overall?" with game 1's sentences when the model failed
     * (the parity audit, 2026-09-26; desktop c5c488f).
     */
    @Test
    fun aNewGameForgetsTheLastGamesReview() {
        val (coach, _) = coach(emptyList())
        start(coach)
        coach.reviewLines = listOf("You played white and won by checkmate in 4 moves.")
        start(coach)
        assertEquals(emptyList<String>(), coach.reviewLines)
    }

    /**
     * Saving a new rating rewords the system message and the templates alike: the register was
     * kept until the coach was rebuilt (desktop c5c488f). PanelWarm's Save calls [Coach.reword].
     */
    @Test
    fun aSavedRatingChangesTheTemplatesRegister() {
        val coach = Coach(ScriptedProvider(emptyList()), rating = 800 to 1100)
        assertTrue(coach.plainWords && !coach.strongPlayer)
        coach.reword(null, 2000 to 2300)
        assertTrue(!coach.plainWords && coach.strongPlayer)
        assertTrue("2000" in coach.system)
    }

    /** `test_engine_tools.FirstLegal`: the first legal moves by UCI, a line of four, a little ahead. */
    private val firstLegal = EngineLike { board, depth ->
        val replay = board.copy(withStack = false)
        val moves = ArrayList<String>()
        repeat(4) {
            val legal = replay.legalMoves().sortedBy { it.uci() }
            if (legal.isEmpty()) return@repeat
            moves.add(legal[0].uci())
            replay.push(legal[0])
        }
        if (moves.isEmpty()) emptyList() else listOf(Line(moves = moves, depth = depth ?: 14, expectedScore = 0.6))
    }

    /**
     * Asked "ok and after that?" with the line already given, the engine is asked about the
     * line's end before the model answers (Luna at no effort gave the same line back, Tai's
     * puzzle; `test_engine_tools.py`).
     */
    @Test
    fun theLineAskedForAgainBringsTheEngineOnItsEnd() {
        val (coach, provider) = coach(listOf("Qxf7# -- it is mate at once.", "Qxf7# -- it is mate at once."))
        coach.engine = firstLegal
        playToScholar(coach)
        coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
        messages(coach)
        coach.ask("show me the line", 8.0)
        messages(coach)
        assertFalse(LINE_GIVEN in directive(provider))
        coach.ask("ok and after that?", 9.0)
        messages(coach)
        val task = directive(provider)
        assertTrue(task, LINE_GIVEN in task)
        assertTrue(task, "The engine, asked about the line's end:" in task)
        assertTrue(task, "The game is over there: checkmate, White wins." in task)
    }

    /**
     * Luna with a reasoning effort takes no tools on chat completions: the 400 left the coach
     * silent on every question. A refused request is asked again without the tools; a server
     * that is down is not (`test_engine_tools.py`).
     */
    @Test
    fun aServerThatRefusesTheToolsIsAskedWithoutThem() {
        class Refusing(val error: String) : ToolsProvider {
            override val name = "refusing"
            val seen = ArrayList<Boolean>() // whether each call carried the tools

            override fun healthcheck(): Pair<Boolean, String> = true to "refusing"

            override fun chat(
                system: String,
                messages: List<ChatMessage>,
                maxTokens: Int,
                temperature: Double,
                thinking: Boolean,
            ): Completion {
                synchronized(seen) { seen.add(false) }
                return Completion(text = "The pawn on f7 is defended only by the King.")
            }

            override fun chatTools(
                system: String,
                messages: List<ChatMessage>,
                tools: JSONArray,
                turns: JSONArray,
                maxTokens: Int,
                temperature: Double,
                toolChoice: String?,
                onDelta: StreamCallback?,
            ): Completion {
                synchronized(seen) { seen.add(true) }
                throw RuntimeException(error)
            }
        }
        for ((error, answered) in listOf("400: tools not supported" to true, "503: down" to false)) {
            val provider = Refusing(error)
            val coach = Coach(provider, policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))
            coach.engine = firstLegal
            playToScholar(coach)
            coach.onEvent(mateAnalysis(6, decisive = false), 7.0)
            messages(coach)
            synchronized(provider.seen) { provider.seen.clear() }
            coach.ask("is f7 weak?", 8.0)
            val said = messages(coach).map { it.text }
            assertEquals(error, answered, "The pawn on f7 is defended only by the King." in said)
            assertEquals(error, answered, synchronized(provider.seen) { false in provider.seen })
        }
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        // The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
        private const val QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
        private val QD1_LINE = listOf("c2d1", "d4d1", "d8d1", "a5e1", "d1e1")

        /** The leads that announced a rule, gone for good (desktop f983cd0). */
        private val ANNOUNCED = Regex(
            "(keep this one|the lesson to keep|worth remembering|a rule to take with you|" +
                "the thing to remember|remember this):",
            RegexOption.IGNORE_CASE,
        )

        private fun sentenceOf(rule: String): String = rule[0].uppercaseChar() + rule.substring(1).trimEnd('.') + "."
    }
}
