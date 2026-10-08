package com.sidecoach.ui

import com.sidecoach.profile.Puzzle
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The puzzle page's result line ("Solved · your puzzle level 1125 (+25)"): the session's state
 * carries `result` -- null while the puzzle is played, then solved or not, the level after it
 * and how far it moved, as the room recorded it -- and the page reads it from the JSON the room
 * serves. `tests/test_puzzle_mode.py` (`test_a_mate_puzzle_and_giving_up_on_it`,
 * `test_a_solved_puzzle_shows_the_level_it_moved_to`); the room over the API is checked on the
 * device by PuzzleModeTest.
 */
class PuzzleResultTest {
    // Black's king steps to the corner; White mates on the back rank.
    private val mate = Puzzle(
        id = "mate1",
        fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
        moves = listOf("g8h8", "a1a8"),
        rating = 900,
        popularity = 70,
        plays = 95,
        themes = listOf("mateIn1", "backRankMate"),
    )

    @Test
    fun theResultComesOnceRecordedAndIsServedAsTheDesktopServesIt() {
        val s = PuzzleSession(mate, "mate")
        val playing = JSONObject(s.state())
        assertTrue("no result line while the puzzle is played", playing.has("result") && playing.isNull("result"))
        assertTrue(s.move("a1a8").isNotEmpty() && s.status == "solved")
        assertTrue("none until the room has recorded it", JSONObject(s.state()).isNull("result"))
        s.recordedAs(true, 1990, 2000)
        val result = JSONObject(s.state()).getJSONObject("result")
        assertEquals("""{"solved":true,"level":2000,"change":10}""", result.toString())
    }

    @Test
    fun aGivenUpPuzzleShowsTheLevelDown() {
        val s = PuzzleSession(mate, "mate")
        s.giveUp()
        s.recordedAs(false, 1100, 1075)
        val result = JSONObject(s.state()).getJSONObject("result")
        assertEquals("""{"solved":false,"level":1075,"change":-25}""", result.toString())
    }

    /**
     * Lichess: a hint shown, the round is not rated. The page's chips send these words; a question
     * of the player's own words does not count, nor a chip once the puzzle is over
     * (`test_hint_best_move_or_line_while_playing_makes_the_attempt_unrated`).
     */
    @Test
    fun hintBestMoveOrLineWhilePlayingMakesTheAttemptUnrated() {
        for (chip in listOf("hint", "what should I play", "show me the line", "  Hint ")) {
            val s = PuzzleSession(mate, "mate")
            s.asked(chip)
            assertTrue(chip, s.assisted)
        }
        val s = PuzzleSession(mate, "mate")
        s.asked("why does the rook go there?")
        assertTrue(!s.assisted)
        assertTrue(s.move("a1a8").isNotEmpty() && s.done)
        s.asked("hint")
        assertTrue("after the end: a review, not help", !s.assisted)
    }
}
