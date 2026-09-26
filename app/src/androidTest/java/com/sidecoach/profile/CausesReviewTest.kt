package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The review's pattern line, case for case with the desktop's `tests/test_profile.py`
 * (desktop 797094c). Pure functions, no store: it runs without the device's SQLite.
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
}
