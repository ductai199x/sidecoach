package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.MotifTag
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The causes of a mistake and the review's pattern line, case for case with the cases the
 * desktop's `tests/test_profile.py` gained from 797094c on. Pure functions, no store: it runs
 * without the device's SQLite.
 */
@RunWith(AndroidJUnit4::class)
class CausesReviewTest {

    /** `_user_move`: a judged move with the desktop test's defaults. */
    private fun userMove(
        ply: Int,
        san: String,
        uci: String,
        fenBefore: String,
        label: String? = "good",
        severity: String = "none",
        expectedLoss: Double = 0.01,
        bestUci: String? = null,
        accuracy: Double? = 90.0,
        phase: String = "middlegame",
        cause: String? = null,
    ) = Review.UserMove(
        ply = ply,
        san = san,
        uci = uci,
        fenBefore = fenBefore,
        label = label,
        severity = severity,
        expectedLoss = expectedLoss,
        bestUci = bestUci,
        accuracy = accuracy,
        phase = phase,
        cause = cause,
    )

    /**
     * Even when the move also allowed one: the tactic-blind student was told it kept missing
     * their threats and wrote down the threat habit (the simulated students; desktop b9be768).
     */
    @Test
    fun aChanceLetGoWithATacticOnTheBoardIsTheTacticNeverConsidered() {
        val board = Board()
        val move = Move.fromUci("e2e4")
        var found = Causes.classify(board, move, availableTags = listOf(FORK), allowedTags = listOf(FORK), missedChance = true)
        assertEquals(Causes.NOT_CONSIDERED to "fork", found)
        // Not a miss: the tactic allowed comes first, as before.
        found = Causes.classify(board, move, availableTags = listOf(FORK), allowedTags = listOf(FORK))
        assertEquals(Causes.MISSED_THREAT, found.first)
        // A mate allowed is the threat whatever else.
        found = Causes.classify(board, move, availableTags = listOf(FORK), replyMate = 1, missedChance = true)
        assertEquals(Causes.MISSED_THREAT, found.first)
    }

    /**
     * Even when the reply it allows wins material: the player began it. As a threat missed, the
     * greedy student heard "what their moves threatened" after every game (desktop adc4b26).
     */
    @Test
    fun aCaptureOrCheckThatComesOutBehindIsALineMiscalculated() {
        // 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 4.Nxe5?, allowing the fork ...Qg5.
        val board = Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        var found = Causes.classify(board, Move.fromUci("f3e5"), allowedTags = listOf(FORK))
        assertEquals(Causes.MISCALCULATED to "fork", found)
        // A mate allowed is still the threat.
        found = Causes.classify(board, Move.fromUci("f3e5"), replyMate = 2)
        assertEquals(Causes.MISSED_THREAT, found.first)
    }

    /**
     * Asked "What do I keep missing?", the reviews named the move that cost most and never the
     * pattern (the simulated students, 2026-09-26). Second, after the result, so the offline
     * review's first four lines carry it.
     */
    @Test
    fun aCauseBehindTwoMistakesIsNamedSecondWithTheHabitThatFixesIt() {
        val afterE4 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

        fun missed(ply: Int, loss: Double, cause: String = Causes.MISSED_THREAT) = userMove(
            ply, "Qh5", "d1h5", afterE4, label = "mistake", severity = "mistake",
            expectedLoss = loss, cause = cause,
        )

        fun review(moves: List<Review.UserMove>) = Review.buildReview(
            gameId = "g",
            userSide = Pieces.Side.WHITE,
            result = "0-1",
            reason = null,
            opening = "King's Pawn",
            moves = moves,
            fullmoves = 30,
        )

        var lines = review(listOf(missed(3, 0.2), missed(9, 0.1), missed(15, 0.3, Causes.POSITIONAL))).lines
        assertEquals(
            "What you kept missing: what their moves threatened -- 2 of your 3 mistakes. The habit" +
                " that fixes it: before each move, look at what their last move attacks, and at the" +
                " captures and checks your move would leave them.",
            lines[1],
        )
        // One mistake of each kind is no pattern.
        lines = review(listOf(missed(3, 0.2), missed(15, 0.3, Causes.POSITIONAL))).lines
        assertTrue(lines.toString(), lines.none { it.startsWith("What you kept missing") })
    }

    companion object {
        private val FORK = MotifTag(name = "fork", squares = listOf("e5", "c6", "g6"))
    }
}
