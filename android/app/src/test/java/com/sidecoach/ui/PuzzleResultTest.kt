package com.sidecoach.ui

import com.sidecoach.profile.Puzzle
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The puzzle page's result line ("Solved · your puzzle rating 1712? (+212)"): the session's
 * state carries `result` -- null while the puzzle is played, then solved or not, the rating after
 * it, how far it moved and whether it is still provisional, as the room recorded it -- and the
 * page reads it from the JSON the room serves. `tests/test_puzzle_mode.py` (`test_a_mate_puzzle_and_giving_up_on_it`,
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
        s.recordedAs(true, 1500, 1712, true)
        val result = JSONObject(s.state()).getJSONObject("result")
        assertEquals("""{"solved":true,"level":1712,"change":212,"provisional":true,"unrated":null}""", result.toString())
    }

    /**
     * Why the rating stayed, for the result line ("Not rated: you used a hint"): the help used,
     * or a puzzle tried before (`test_a_puzzle_solved_after_a_hint_leaves_the_rating_as_it_was`,
     * `test_a_puzzle_tried_before_says_why_its_rating_stays`).
     */
    @Test
    fun theResultSaysWhyTheRatingStayed() {
        val s = PuzzleSession(mate, "mate")
        s.asked("hint")
        assertTrue(s.move("a1a8").isNotEmpty())
        s.recordedAs(true, 1500, 1500, true, s.help)
        assertEquals("""{"solved":true,"level":1500,"change":0,"provisional":true,"unrated":"hint"}""", JSONObject(s.state()).getJSONObject("result").toString())
        val again = PuzzleSession(mate, "mate")
        again.recordedAs(true, 1500, 1500, true, "repeat")
        assertEquals("repeat", JSONObject(again.state()).getJSONObject("result").getString("unrated"))
    }

    /** The first help used names it (`test_the_result_names_the_first_help_used`). */
    @Test
    fun theResultNamesTheFirstHelpUsed() {
        val cases = listOf(
            listOf("hint") to "hint",
            listOf("what should I play", "hint") to "best_move",
            listOf("  Show me the line ") to "line",
            listOf("why does the knight go there?") to null,
        )
        for ((chips, used) in cases) {
            val s = PuzzleSession(mate, "mate")
            for (chip in chips) s.asked(chip)
            assertEquals(chips.toString(), used, s.help)
        }
    }

    @Test
    fun aGivenUpPuzzleShowsTheRatingDown() {
        val s = PuzzleSession(mate, "mate")
        s.giveUp()
        s.recordedAs(false, 1612, 1609, false)
        val result = JSONObject(s.state()).getJSONObject("result")
        assertEquals("""{"solved":false,"level":1609,"change":-3,"provisional":false,"unrated":null}""", result.toString())
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
