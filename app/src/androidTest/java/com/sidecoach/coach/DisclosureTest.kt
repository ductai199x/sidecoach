package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.events.DisclosureLevel
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the leak check withholds, and what the facts show at each level: the desktop's
 * `tests/test_disclosure.py`, the cases its commits since 6f70e2e changed or added. The parity
 * fixture (coach.json) holds [leakCheck] and [visibleFacts] to the desktop's over real positions;
 * these are the owner's rules said as cases.
 */
@RunWith(AndroidJUnit4::class)
class DisclosureTest {

    private fun facts() = PositionFacts(
        fen = START,
        sideToMove = "white",
        phase = "opening",
        theme = "loose piece",
        region = "around f7",
        piece = "your knight",
        candidates = listOf("Ng5"),
        threats = listOf("Qxd4"),
        line = listOf("Ng5", "d5", "Nxf7"),
    )

    /** Their threat is the first hint's key fact as often as not: words from the idea level (af1bae2). */
    @Test
    fun visibleFactsHideMovesBelowTheMoveLevel() {
        val bundle = visibleFacts(facts(), DisclosureLevel.FOCUS)
        assertEquals("loose piece", bundle["theme"])
        for (hidden in listOf("candidates", "line")) assertTrue(hidden, hidden !in bundle)
        val idea = visibleFacts(facts(), DisclosureLevel.IDEA)
        for (hidden in listOf("region", "piece", "candidates")) assertTrue(hidden, hidden !in idea)
        assertEquals(listOf("Qxd4"), idea["threats"])
    }

    /**
     * Tai, 2026-09-25: "leaking is not a criminal offense ... clamp down egregious offense but be
     * smart about it". The moves to find and the squares they go to are withheld; a plan, a
     * weakness, another move or square may be named (desktop af1bae2).
     */
    @Test
    fun leakCheckWithholdsTheAnswerAndNothingElseBelowTheMove() {
        assertEquals(
            emptyList<String>(),
            leakCheck("Maybe d3 or Nc3 here, then a plan on the queenside?", facts(), DisclosureLevel.IDEA),
        )
        val leaks = leakCheck("Try Ng5, then Nxf7.", facts(), DisclosureLevel.IDEA)
        assertTrue(leaks.toString(), "move Ng5" in leaks && "move Nxf7" in leaks)
        assertTrue("square g5" in leakCheck("A knight on g5 would be strong.", facts(), DisclosureLevel.IDEA))
        // The opponent's move in the line is not the player's answer.
        assertEquals(emptyList<String>(), leakCheck("If they push d5, the centre opens.", facts(), DisclosureLevel.IDEA))
    }

    @Test
    fun leakCheckAllowsSquaresAndPiecesTheUserNamed() {
        val said = "Is my bishop on c4 attacking f7? And what is my knight doing?"
        assertEquals(emptyList<String>(), leakCheck("Yes, your bishop hits f7.", facts(), DisclosureLevel.IDEA, said = said))
        assertEquals(emptyList<String>(), leakCheck("Your knight is well placed.", facts(), DisclosureLevel.IDEA, said = said))
        // An answer square the user did not name is still withheld (desktop af1bae2).
        assertTrue("square g5" in leakCheck("Look at g5.", facts(), DisclosureLevel.IDEA, said = said))
    }

    /**
     * A move said with its number is the record (the review names "17. Ng5" where ground was
     * lost); a pawn on another square than the one to move is a fact, not the hint (the eval
     * audit's gate, 2026-09-25; desktop d17b39b).
     */
    @Test
    fun theRecordAndOtherPiecesOfAKindAreNoLeak() {
        assertEquals(emptyList<String>(), leakCheck("You lost the most ground at 17. Ng5 (mistake).", facts(), DisclosureLevel.IDEA))
        val f = facts().copy(piece = "the player's pawn on e4")
        assertEquals(emptyList<String>(), leakCheck("They threaten Ne6, attacking the pawn on d2.", f, DisclosureLevel.IDEA))
        assertTrue("piece pawn" in leakCheck("Your pawn on e4 is the one.", f, DisclosureLevel.IDEA))
        assertTrue("piece pawn" in leakCheck("Think about your pawn.", f, DisclosureLevel.IDEA))
    }

    companion object {
        private const val START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    }
}
