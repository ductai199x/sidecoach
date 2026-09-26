package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * `tests/test_effects_legal.py`'s case of desktop 00894e0: the notes a line is explained by
 * ([lineNotes]). The rest of what a move does is held to the desktop's answer position by
 * position in `chess/coach.json` ([CoachPrimitivesTest]).
 */
@RunWith(AndroidJUnit4::class)
class EffectsTest {

    /**
     * A reply to check says whether only a king move got out of it, or it was one of several
     * ways out (the bench's gold labels: "the check forced the King away" where a block was
     * legal).
     */
    @Test
    fun aLineMoveByMoveSaysWhatEachMoveDoesAndWhatACheckForces() {
        val board = Board("8/p1q1k2K/2p1p3/3p4/3P1R1P/6r1/1r6/8 w - - 0 49")
        var notes = lineNotes(board, listOf("Rf1", "Qd6", "Rf7+", "Kxf7"))
        assertTrue(notes, notes.startsWith("Rf1 (gets the rook out of attack), Qd6, Rf7+ (gives check"))
        assertTrue(notes, notes.endsWith("Kxf7 (only a king move gets out of check; takes the rook on f7)"))
        val pinned = Board("6rk/5prq/1p1p1n2/p1pP1B1p/P1P2P1n/2P1R1P1/2Q4K/2B2NR1 b - - 0 35")
        notes = lineNotes(pinned, listOf("f6g4", "f5g4", "h7c2")) // UCI, as the engine gives it
        assertTrue(notes, "Bxg4 (one of 3 ways out of check; takes the knight on g4" in notes)
        assertTrue(notes, notes.split(", ").last().startsWith("Qxc2+ (takes the queen on c2"))
        assertEquals("Rf1 (gets the rook out of attack)", lineNotes(board, listOf("Rf1", "Ke9")))
    }
}
