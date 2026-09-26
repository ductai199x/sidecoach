package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Severity
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the player says about the game in their own words: the desktop's
 * `tests/test_chat_after_move.py`, the cases of the commits ported (desktop 797094c onwards).
 * The file's first case, `_asked_own_move` (desktop f733318), is not on the phone: it builds on
 * desktop commits not ported (the runtime audit's `_last_own_move(mine=)`, `_ASK_THREAT`).
 */
@RunWith(AndroidJUnit4::class)
class ChatAfterMoveTest {

    /** Answers from a script, and remembers every prompt it was given. */
    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"
        private val replies = ArrayDeque(replies)
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

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    /** `_played`: a game from the start, every move judged, one of them a blunder. */
    private fun played(coach: Coach, ucis: List<String>, mistakeAt: Int) {
        val board = Board()
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = board.fen()), 0.0)
        for ((i, uci) in ucis.withIndex()) {
            val ply = i + 1
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
                    byUser = ply % 2 == 1,
                ),
                ply.toDouble(),
            )
            coach.onEvent(
                MoveAssessment(
                    gameId = "g",
                    ply = ply,
                    uci = uci,
                    expectedLoss = if (ply == mistakeAt) 0.3 else 0.0,
                    severity = if (ply == mistakeAt) Severity.BLUNDER else Severity.NONE,
                    best = listOf(uci),
                    label = if (ply == mistakeAt) Label.BLUNDER else Label.BEST,
                ),
                ply + 0.5,
            )
        }
    }

    private fun analysed(coach: Coach, ucis: List<String>, ply: Int, best: String, score: Double, ts: Double) {
        val board = Board()
        for (uci in ucis.take(ply)) board.push(Move.fromUci(uci))
        coach.onEvent(
            Analysis(
                gameId = "g",
                fen = board.fen(),
                ply = ply,
                lines = listOf(Line(moves = listOf(best), depth = 12, expectedScore = score)),
                decisive = false,
                onlyMove = false,
                tags = emptyList(),
            ),
            ts,
        )
    }

    /**
     * Named in a statement or before the question, the pieces came back as the whole answer --
     * where each can go, what each attacks -- and the question went unanswered (the simulated
     * students, 2026-09-26; desktop 797094c). The phone says a piece examined as the object has
     * it ("the player's knight on f3 stands there; it can go to ..."): the desktop's
     * "Your knight on f3 can go to" came with f733318, not ported.
     */
    @Test
    fun whatAQuestionNamesIsExaminedAloudAndWhatAStatementNamesIsNot() {
        assertEquals(
            "Why is h5 their reply?",
            questionPart("I see it now: my rook on e1 stays active. Why is h5 their reply?"),
        )
        assertEquals("where can my knight go", questionPart("where can my knight go"))
        assertEquals("", questionPart("I see the fork now: 19.Nf6 hits the queen and rook."))
        assertEquals("After 12...h5 what then?", questionPart("After 12...h5 what then?"))

        val (coach, _) = coach(emptyList())
        val ucis = listOf("e2e4", "e7e5", "g1f3", "b8c6")
        played(coach, ucis, mistakeAt = 3)
        analysed(coach, ucis, 4, "f1c4", 0.55, 4.5)
        messages(coach)
        coach.ask("I see it now: my knight on f3 attacks the pawn on e5.", 5.0)
        val statement = messages(coach).map { it.text }
        assertTrue(statement.toString(), statement.none { "can go to" in it })
        coach.ask("Where can my knight on f3 go?", 6.0)
        val question = messages(coach).map { it.text }
        assertTrue(
            question.toString(),
            question.any { "the player's knight on f3 stands there; it can go to" in it },
        )
    }

    /**
     * Asked about their own last move, the coach answers from the position it was played from
     * -- but "b5 is empty" was said with the player's bishop on b5 (the students' round 6; desktop
     * f85c2bf): pieces and squares are read on the board as it stands. The desktop's question
     * ("Is my bishop on b5 safe there?") reaches the position before the move through
     * `_asked_own_move` (f733318, not ported); on the phone a question about the last move
     * ("was that a mistake ...") reaches it, and the desktop's own question is asked too.
     */
    @Test
    fun aPieceAskedAboutIsReadOnTheBoardAsItStands() {
        val (coach, _) = coach(emptyList())
        val ucis = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")
        played(coach, ucis, mistakeAt = 5)
        for (ply in 4..5) analysed(coach, ucis, ply, if (ply == 5) "a7a6" else "f1b5", 0.5, ply + 0.7)
        messages(coach)
        assertTrue(!coach.userToMove)
        coach.ask("Was that a mistake, putting my bishop on b5?", 6.0)
        var said = messages(coach).joinToString(" ") { it.text }
        assertTrue(said, "the player's bishop on b5 stands there" in said && "b5 is empty" !in said)
        coach.ask("Is my bishop on b5 safe there?", 7.0)
        said = messages(coach).joinToString(" ") { it.text }
        assertTrue(said, "the player's bishop on b5 stands there" in said && "b5 is empty" !in said)
    }

    /** The directive of the last model call: the harness after its "# Task" line. */
    private fun directive(provider: ScriptedProvider): String {
        val messages = synchronized(provider.prompts) { provider.prompts.last().second }
        assertEquals(1, messages.size)
        return messages[0].content.substringAfterLast("# Task\n")
    }

    /**
     * Every answer to the coach's question had earned a new question and none was resolved (the
     * simulated students: "what Nc4 was expected to do was asked but not answered"; desktop
     * cf1e2a0). On the desktop the player's words on the opponent's turn go to the own-move
     * directive (`_asked_own_move`, f733318, not ported); here they go to the generic one, which
     * closes the question the same way.
     */
    @Test
    fun anAnswerToTheCoachsQuestionClosesItWithoutAnother() {
        val (coach, provider) = coach(listOf("You have it: the knight was loose.", "Right."))
        val ucis = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5")
        played(coach, ucis, mistakeAt = 7)
        analysed(coach, ucis, 7, "d7d5", 0.45, 7.5)
        messages(coach)
        synchronized(coach.lock) { coach.conversation.coachSaid("What were you expecting that move to do?") }
        assertTrue(coach.answeringOurQuestion("I wanted to attack f7 with the knight."))
        assertTrue(!coach.answeringOurQuestion("Why is that bad?"))
        coach.ask("I wanted to attack f7 with the knight.", 8.0)
        messages(coach)
        val directive = directive(provider)
        assertTrue(directive, "They are answering the question you just asked" in directive)
        assertTrue(directive, "then one question" !in directive)
        // Once answered, the thread is closed: words after the coach's reply answer nothing.
        assertTrue(!coach.answeringOurQuestion("ok"))
    }
}
