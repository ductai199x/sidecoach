package com.sidecoach.profile

import com.sidecoach.events.GameReview
import com.sidecoach.vision.Pieces
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The review's first sentence counts the moves in agreement: "1 move", not "1 moves" (the
 * audit, 2026-10-07: grammar in the templates; `test_the_review_says_how_the_game_ended_in_plain_words`).
 * The device test `ProfileTest.theReviewSaysHowTheGameEndedInPlainWords` holds the endings at 17
 * moves.
 */
class ReviewLinesTest {
    @Test
    fun oneMoveIsSaidInTheSingular() {
        val mated = GameReview(gameId = "g", userSide = Pieces.Side.WHITE, result = "1-0", reason = "checkmate")
        assertEquals("You played white and won by checkmate in 1 move.", Review.linesFor(mated, 1)[0])
        val unknown = GameReview(gameId = "g", userSide = Pieces.Side.WHITE, result = "*", reason = null)
        assertEquals("You played white for 1 move; I did not see how it ended.", Review.linesFor(unknown, 1)[0])
        assertEquals("You played white and won by checkmate in 17 moves.", Review.linesFor(mated, 17)[0])
    }
}
