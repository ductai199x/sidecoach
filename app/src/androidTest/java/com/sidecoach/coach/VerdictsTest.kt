package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The word on a sound move, pitched to the player: the cases the desktop's
 * `tests/test_verdicts.py` gained from cf1e2a0 on. The desktop's tests replace `lesson` with one
 * that returns the principle; here the boards are real ones whose lesson is that principle, as
 * the coach fixture (`coach.json`) records it and CoachPrimitivesTest holds the Kotlin to.
 */
@RunWith(AndroidJUnit4::class)
class VerdictsTest {

    /**
     * The simulated 800-rated student listed "prophylaxis: stop their plan before pushing your
     * own" among what it did not understand; every principle keeps a plain wording (desktop
     * cf1e2a0).
     */
    @Test
    fun aPlayerBelow1400HearsNoTermOfArt() {
        for ((kind, words) in PRINCIPLES) {
            assertTrue(kind, words.any { !JARGON.containsMatchIn(it) })
        }
        // White's king to f4 takes e5 from their king: the lesson is prophylaxis.
        val board = Board(PROPHYLAXIS)
        val move = Move.fromUci("e3f4")
        assertTrue(lesson(board, move, phase = "endgame", threats = listOf("Ke5"))?.first == "prophylaxis")
        fun said(i: Int, plain: Boolean) = verdict(
            board, move, "good", phase = "endgame", threats = listOf("Ke5"),
            pick = { it[i % it.size] }, plain = plain,
        ).first
        val plain = (0 until 20).map { said(it, plain = true) }.toSet()
        assertTrue(plain.toString(), plain.none { "prophylaxis" in it })
        val rich = (0 until 40).map { said(it, plain = false) }.toSet()
        assertTrue(rich.toString(), rich.any { "prophylaxis" in it })
    }

    companion object {
        // From the coach fixture's positions: an endgame where Kf4 is prophylaxis (their Ke5
        // threatened), and an opening where ...Rg8 takes the open file.
        const val PROPHYLAXIS = "8/8/p3k3/1p3p1p/1P6/P3K1PP/8/8 w - - 1 41"
        const val TAKE_FILE = "2kr1b1r/ppq2p2/2p1pp2/8/2B5/3P2Q1/PPPB1PP1/R4RK1 b - - 1 1"
    }
}
