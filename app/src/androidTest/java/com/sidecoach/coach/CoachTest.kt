package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
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
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
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
    private fun mateAnalysis(ply: Int) = Analysis(
        gameId = "g",
        fen = SCHOLAR,
        ply = ply,
        lines = listOf(
            Line(moves = listOf("h5f7"), depth = 12, expectedScore = 1.0, mate = 1),
            Line(moves = listOf("c4f7"), depth = 12, expectedScore = 0.55),
        ),
        decisive = true,
        onlyMove = true,
        tags = listOf(MotifTag(name = "mateIn1", squares = listOf("f7", "e8"), move = "h5f7")),
    )

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
    }

    companion object {
        // 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
        private const val SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

        // The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
        private const val QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
        private val QD1_LINE = listOf("c2d1", "d4d1", "d8d1", "a5e1", "d1e1")
    }
}
