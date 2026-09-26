package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The fact sheet, case for case with the cases the desktop's `tests/test_facts.py` gained from
 * 4cc6f11 on (the rest of the sheet is held to the Python's own output by CoachPrimitivesTest).
 */
@RunWith(AndroidJUnit4::class)
class FactsTest {

    /**
     * "A quiet improvement -- which of your pieces has no job right now?" was the hint at the
     * check positions of seven of ten AI sessions, and the students did not understand it (the
     * simulated students' round 4; desktop 4cc6f11).
     */
    @Test
    fun inCheckTheQuestionIsTheCheck() {
        // 1.e4 e5 2.Nf3 d6 3.Bb5+: Black answers the check with the quiet ...c6.
        val fen = "rnbqkbnr/ppp2ppp/3p4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 3"
        val analysis = Analysis(
            gameId = "g",
            fen = fen,
            ply = 5,
            lines = listOf(Line(moves = listOf("c7c6"), depth = 12, expectedScore = 0.45)),
            decisive = false,
            onlyMove = false,
        )
        val facts = factsFromAnalysis(Board(fen), analysis)
        assertEquals(IN_CHECK_QUESTION, facts.question)
        assertEquals(IN_CHECK_NATURE, facts.nature)
    }
}
