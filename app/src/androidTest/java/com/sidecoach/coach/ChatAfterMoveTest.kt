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
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the player says about the game in their own words: the desktop's
 * `tests/test_chat_after_move.py`, the cases of the commits ported (desktop 797094c onwards).
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

    /** `_offline`: the coach with no model -- its templates are the answer. */
    private fun offline() = Coach(OfflineProvider(), policy = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0)))

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
     * students, 2026-09-26; desktop 797094c). A piece examined is said to the player: "Your knight
     * on f3 can go to ..." (desktop f733318).
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

        val ucis = listOf("e2e4", "e7e5", "g1f3", "b8c6")
        fun italian(coach: Coach) {
            played(coach, ucis, mistakeAt = 3)
            analysed(coach, ucis, 4, "f1c4", 0.55, 4.5)
            messages(coach)
        }

        // Offline the examination is the answer: a question's pieces are read out, a statement's
        // not (desktop 9e49622 moved this half to the coach without a model).
        var c = offline()
        italian(c)
        c.ask("I see it now: my knight on f3 attacks the pawn on e5.", 5.0)
        val statement = messages(c).map { it.text }
        assertTrue(statement.toString(), statement.none { "can go to" in it })
        c.ask("Where can my knight on f3 go?", 6.0)
        val question = messages(c).map { it.text }
        assertTrue(question.toString(), question.any { "Your knight on f3 can go to" in it })
        // With a model to answer in words, a piece's inventory said first is noise ("Your king on
        // g8 can go to f7, g7, h8; ..." before the AI's own answer; desktop 9e49622).
        c = coach(listOf("[p=1.0] It can go to d4, e5, g5 or h4.")).first
        italian(c)
        c.ask("Where can my knight on f3 go?", 6.0)
        val modelled = messages(c).map { it.text }
        assertTrue(modelled.toString(), modelled.none { "Your knight on f3 can go to" in it })
    }

    /**
     * Asked about their own last move, the coach answers from the position it was played from
     * -- but "b5 is empty" was said with the player's bishop on b5 (the students' round 6; desktop
     * f85c2bf): pieces and squares are read on the board as it stands. "Is my bishop on b5 safe
     * there?", said while the opponent is to move, is about the move just played (desktop
     * f733318).
     */
    @Test
    fun aPieceAskedAboutIsReadOnTheBoardAsItStands() {
        val coach = offline() // the inventory said aloud: the coach without a model (desktop 9e49622)
        val ucis = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")
        played(coach, ucis, mistakeAt = 5)
        for (ply in 4..5) analysed(coach, ucis, ply, if (ply == 5) "a7a6" else "f1b5", 0.5, ply + 0.7)
        messages(coach)
        assertTrue(!coach.userToMove)
        coach.ask("Is my bishop on b5 safe there?", 6.0)
        val said = messages(coach).joinToString(" ") { it.text }
        assertTrue(said, "Your bishop on b5" in said && "b5 is empty" !in said)
    }

    /**
     * Offline, what the player says while the opponent is to move is about the move they just
     * played: "I didn't even notice they were aiming for my rook" after 9.Nf3? was answered for
     * the side to move -- the student's own knight (a simulated student's session, 2026-09-26;
     * desktop f733318). With a model only a question about the move in so many words is: the
     * model reads the rest from the record, and the guess made for it answered "how would this
     * position continues?" four times as a question about the move just made (Tai's puzzle,
     * 2026-09-28).
     */
    @Test
    fun chatOnTheOpponentsTurnIsAboutThePlayersMove() {
        val moves = listOf("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "f3g5")
        for (withModel in listOf(false, true)) {
            val coach = if (withModel) coach(emptyList()).first else offline()
            played(coach, moves, mistakeAt = 7)
            assertTrue(!coach.userToMove) // the player's 4.Ng5 is the last move
            val own = coach.askedOwnMove("I didn't even notice they were aiming for my rook")
            assertEquals(if (withModel) null else 7, own?.move?.ply)
            // About the opponent's side of the board, the game or how it goes on: not the move.
            assertEquals(null, coach.askedOwnMove("what are they threatening?"))
            assertEquals(null, coach.askedOwnMove("how did I do this game?"))
            assertEquals(null, coach.askedOwnMove("how would this position continue?"))
            // About the move in so many words.
            assertEquals(7, coach.askedOwnMove("why was my last move bad?")?.move?.ply)
        }
    }

    @Test
    fun chatOnThePlayersOwnTurnIsAboutThePosition() {
        for (withModel in listOf(false, true)) {
            val coach = if (withModel) coach(emptyList()).first else offline()
            played(coach, listOf("e2e4", "e7e5", "g1f3", "b8c6"), mistakeAt = 3)
            assertTrue(coach.userToMove)
            assertEquals(null, coach.askedOwnMove("I didn't even notice they were aiming for my rook"))
            // Asked in so many words, it is still about the player's move.
            assertTrue(coach.askedOwnMove("why was my last move bad?") != null)
        }
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
     * cf1e2a0). With a model the player's words on the opponent's turn go to the generic
     * directive, which closes the question the same way.
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
