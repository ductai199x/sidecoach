package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Analysis
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Severity
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Words that can mean two things (Tai, 2026-10-02: "Look for all of these weird ambiguous stuff and
 * make sure the AI coach have enough guidelines to be able to ask clarification questions and the
 * offline coach can still respond correctly"), as the desktop's `tests/test_ambiguity.py` and
 * `test_the_other_good_moves_on_their_turn_are_the_players_last_turns` check them. Logic on
 * events, no timing.
 */
class AmbiguityTest {

    private class ScriptedProvider(replies: List<String>) : LLMProvider {
        override val name = "scripted"
        val replies = ArrayDeque(replies)
        val prompts = ArrayList<List<ChatMessage>>()

        override fun healthcheck(): Pair<Boolean, String> = true to "scripted"

        override fun chat(
            system: String,
            messages: List<ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): Completion {
            synchronized(prompts) { prompts.add(messages.toList()) }
            val text = synchronized(replies) { replies.removeFirstOrNull() } ?: "Keep looking."
            return Completion(text = text)
        }
    }

    private fun policy() = InterjectionPolicy(PolicySettings(stuckSeconds = 10.0))

    private fun offlineCoach(): Coach = Coach(OfflineProvider(), policy = policy())

    private fun messages(coach: Coach): List<CoachMessage> {
        coach.speaker.runPending()
        return coach.poll()
    }

    private fun answer(coach: Coach, words: String, ts: Double): String {
        coach.ask(words, ts)
        return messages(coach).single().text
    }

    private fun directiveOf(provider: ScriptedProvider): String =
        synchronized(provider.prompts) { provider.prompts.last() }[0].content.substringAfterLast("# Task\n")

    private fun playToScholar(coach: Coach) {
        coach.onEvent(GameStarted(gameId = "g", userSide = Pieces.Side.WHITE, startFen = Board().fen()), 0.0)
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
        coach.quiet = true // no word on the moves: the answers are the only messages
        coach.onEvent(
            Analysis(
                gameId = "g", fen = SCHOLAR, ply = 6,
                lines = listOf(
                    Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
                    Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
                ),
                decisive = false, onlyMove = true,
                tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
            ),
            7.0,
        )
    }

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

    private fun verdict(coach: Coach, ply: Int, uci: String, severity: Severity, label: Label, ts: Double) {
        val loss = when (severity) {
            Severity.NONE -> 0.0
            Severity.MISTAKE -> 0.2
            else -> 0.45
        }
        coach.onEvent(
            MoveAssessment(
                gameId = "g", ply = ply, uci = uci, expectedLoss = loss, severity = severity,
                best = listOf(if (ply == 7) "h5f7" else "e8e7"), label = label,
            ),
            ts,
        )
    }

    /** Bxf7+ (judged [mine]) and Ke7 (judged [theirs]), each position read: the player to move. */
    private fun bishopCheckAndKe7(coach: Coach, mine: Severity, theirs: Severity) {
        playToScholar(coach)
        val board = Board(SCHOLAR)
        played(coach, board, 7, "c4f7", true, 10.0)
        verdict(coach, 7, "c4f7", mine, if (mine == Severity.NONE) Label.BEST else Label.MISS, 10.5)
        looked(coach, board, 7, listOf("e8e7"), 10.6)
        played(coach, board, 8, "e8e7", false, 11.0)
        val label = when (theirs) {
            Severity.NONE -> Label.BEST
            Severity.MISTAKE -> Label.MISTAKE
            else -> Label.BLUNDER
        }
        verdict(coach, 8, "e8e7", theirs, label, 11.5)
        looked(coach, board, 8, listOf("h5e5"), 11.6)
        messages(coach)
    }

    /** `test_that_after_the_reply_is_the_move_that_went_wrong_or_asked`. */
    @Test
    fun thatAfterTheReplyIsTheMoveThatWentWrongOrAsked() {
        val cases = listOf(
            Triple(Severity.BLUNDER, Severity.NONE, "Qxf7# was better"),
            Triple(Severity.NONE, Severity.MISTAKE, "Their Ke7 was a mistake."),
            Triple(
                Severity.NONE, Severity.NONE,
                "Neither was a mistake: your Bxf7+ was best, and their Ke7 was best.",
            ),
            Triple(Severity.BLUNDER, Severity.MISTAKE, "Your Bxf7+, or their Ke7?"),
        )
        for ((mine, theirs, said) in cases) {
            for (words in listOf("why was that bad?", "was that a mistake?")) {
                val coach = offlineCoach()
                bishopCheckAndKe7(coach, mine, theirs)
                val got = answer(coach, words, 12.0)
                assertTrue("$words $mine $theirs: $got", got.startsWith(said))
            }
        }
    }

    /** `test_whose_move_said_in_the_words_settles_it`. */
    @Test
    fun whoseMoveSaidInTheWordsSettlesIt() {
        for ((words, said) in listOf(
            "why was my move bad?" to "Qxf7# was better",
            "was their move a mistake?" to "Their Ke7 was a mistake.",
            "why was Bxf7+ bad?" to "Qxf7# was better",
        )) {
            val coach = offlineCoach()
            bishopCheckAndKe7(coach, Severity.BLUNDER, Severity.MISTAKE)
            val got = answer(coach, words, 12.0)
            assertTrue("$words: $got", got.startsWith(said))
        }
    }

    /** `test_the_answer_to_which_move_is_read_with_the_question`. */
    @Test
    fun theAnswerToWhichMoveIsReadWithTheQuestion() {
        for ((reply, said) in listOf(
            "mine" to "Qxf7# was better",
            "theirs" to "Their Ke7 was a mistake.",
            "Ke7" to "Their Ke7 was a mistake.",
        )) {
            val coach = offlineCoach()
            bishopCheckAndKe7(coach, Severity.BLUNDER, Severity.MISTAKE)
            assertEquals("Your Bxf7+, or their Ke7?", answer(coach, "why was that bad?", 12.0))
            val got = answer(coach, reply, 13.0)
            assertTrue("$reply: $got", got.startsWith(said))
            assertNull(coach.clarifying)
        }
        val coach = offlineCoach()
        bishopCheckAndKe7(coach, Severity.BLUNDER, Severity.MISTAKE)
        answer(coach, "why was that bad?", 12.0)
        assertTrue(!answer(coach, "what are they threatening?", 13.0).startsWith("Qxf7# was better"))
        assertNull("a question of its own ends the wait for the answer", coach.clarifying)
    }

    /** `test_the_ai_reads_the_words_itself_with_a_guideline_to_ask`. */
    @Test
    fun theAiReadsTheWordsItselfWithAGuidelineToAsk() {
        val provider = ScriptedProvider(listOf("Your Bxf7+, or their Ke7?"))
        val coach = Coach(provider, policy = policy())
        bishopCheckAndKe7(coach, Severity.BLUNDER, Severity.MISTAKE)
        coach.ask("why was that bad?", 12.0)
        assertEquals(listOf("Your Bxf7+, or their Ke7?"), messages(coach).map { it.text })
        assertTrue(AI_GUIDELINES in directiveOf(provider))
        for (words in listOf(
            "First, what the words refer to.",
            "say in a few words which it is",
            "do not guess: reply with one short question that names the readings",
            "Never ask when every reading gets the same answer.",
        )) {
            assertTrue(words, words in AI_GUIDELINES)
        }
        assertNull("the offline coach's question is the offline coach's", coach.clarifying)
    }

    /** `test_the_other_good_moves_on_their_turn_are_the_players_last_turns`. */
    @Test
    fun theOtherGoodMovesOnTheirTurnAreThePlayersLastTurns() {
        val instead = "Qxf7# was the move there. Yours, Bxf7+, was a miss. " +
            "Bxf7+ was clearly worse there: it passes up the mate."
        for (said in listOf(
            "any other good moves?",
            "does anything else work too?",
            "what else could I have played?",
        )) {
            val offline = offlineCoach()
            playToScholar(offline)
            val board = Board(SCHOLAR)
            played(offline, board, 7, "c4f7", true, 10.0)
            verdict(offline, 7, "c4f7", Severity.BLUNDER, Label.MISS, 10.5)
            assertEquals(said, instead, answer(offline, said, 11.0))
            val provider = ScriptedProvider(listOf("Qxf7# would have mated."))
            val coach = Coach(provider, policy = policy())
            playToScholar(coach)
            val after = Board(SCHOLAR)
            played(coach, after, 7, "c4f7", true, 10.0)
            verdict(coach, 7, "c4f7", Severity.BLUNDER, Label.MISS, 10.5)
            looked(coach, after, 7, listOf("e8e7"), 10.6)
            assertEquals(listOf("Qxf7# would have mated."), messages(coach.also { it.ask(said, 11.0) }).map { it.text })
            val directive = directiveOf(provider)
            assertTrue(AI_GUIDELINES in directive && "what they could have played instead" in directive)
            assertTrue(
                directive,
                "Before Bxf7+, the player's last move, the moves about equally good were: Qxf7#; " +
                    "Bxf7+ was a miss." in directive,
            )
            assertTrue(
                "Clearly worse there, and what goes wrong with them: Bxf7+ passes up the mate" in directive,
            )
        }
        // The opponent's move asked about is theirs, not the player's last turn's.
        val coach = offlineCoach()
        playToScholar(coach)
        val board = Board(SCHOLAR)
        played(coach, board, 7, "c4f7", true, 10.0)
        verdict(coach, 7, "c4f7", Severity.BLUNDER, Label.MISS, 10.5)
        looked(coach, board, 7, listOf("e8e7"), 10.6)
        val got = answer(coach, "what is their best move?", 11.0)
        assertTrue(got, got.startsWith("It's their move") && "Ke7" in got)
    }

    companion object {
        const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    }
}
