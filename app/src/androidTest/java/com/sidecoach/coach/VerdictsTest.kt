package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.MotifTag
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
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

    /**
     * The strong simulated students most often said they already knew "rooks belong on open
     * files" (75 times): from 1800 the praise stands alone (desktop 6d9ba6f). The board is the
     * desktop's own open file (`OPEN_FILE`, desktop 3d9568d): the one this test had, ...Rg8, also
     * hits the queen on g3, and a move that threatens to win a piece teaches that, not the file.
     */
    @Test
    fun aPlayerOf1800AndUpHearsNoTextbookPrinciple() {
        val board = Board(OPEN_FILE)
        val move = Move.fromUci("a1d1")
        var (text, kind) = verdict(board, move, "good", phase = "endgame", pick = { it[1 % it.size] }, strong = true)
        assertTrue(text, kind == null && "rook" !in text)
        val rich = verdict(board, move, "good", phase = "endgame", pick = { it[1 % it.size] })
        text = rich.first
        kind = rich.second
        assertTrue(text, kind == "takeFile")
    }

    /**
     * `test_a_principle_only_when_it_is_the_moves_reason` (desktop 3d9568d). Tai, 2026-09-27:
     * "rooks on open files" of a rook that checked, set up mate, pinned or skewered; "passed pawns
     * must be pushed" of the push that mated. "If it says something it should be relevant". A
     * move with a concrete point gets that point or the praise alone.
     */
    @Test
    fun aPrincipleOnlyWhenItIsTheMovesReason() {
        // Rd1 on the back rank with no luft threatens Rd8 mate: that is the lesson, not the file.
        var board = Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        val rook = Move.fromUci("a1d1")
        assertEquals("mateThreat" to MATE_THREAT, lesson(board, rook, phase = "endgame"))
        assertNull(lesson(board, rook, phase = "endgame", taught = listOf("mateThreat")))
        // The same file, with a check: the check is its point.
        board = Board("3k4/5pp1/7p/8/8/8/5PPP/R5K1 w - - 0 1")
        assertTrue(board.givesCheck(rook))
        assertNull(lesson(board, rook, phase = "endgame"))
        // A line that mates (the annotator's mate tag), a sacrifice, a deflection: no principle.
        board = Board(OPEN_FILE)
        for (name in listOf("mate", "mateIn3", "sacrifice", "deflection", "clearance", "pin")) {
            val tag = MotifTag(name = name, squares = listOf("d1"))
            assertNull(name, lesson(board, rook, listOf(tag), phase = "endgame"))
        }
        // A rook that lines up on a knight in front of a rook (a pin the line does not cash in,
        // so untagged): the pin is its point.
        board = Board("3r2k1/5pp1/3n3p/8/8/8/5PPP/R5K1 w - - 0 1")
        assertNull(lesson(board, rook, phase = "endgame"))
        // A move that makes a threat to win a piece: the threat is its point.
        board = Board("6k1/5pp1/7p/3n4/8/8/5PPP/R5K1 w - - 0 1")
        assertNull(lesson(board, rook, phase = "endgame"))
        // A capture that wins material is about the material: bxc3 taking back the knight was
        // "that lets your bishop out".
        board = Board("rnbqkb1r/ppp1pp1p/6p1/8/3PP3/2n5/PP3PPP/R1BQKBNR w KQkq - 0 6")
        assertNull(lesson(board, Move.fromUci("b2c3"), phase = "opening"))
    }

    /** `test_a_threat_met_is_said_as_one_and_a_pawn_ending_has_no_pieces_to_free` (desktop 3d9568d). */
    @Test
    fun aThreatMetIsSaidAsOneAndAPawnEndingHasNoPiecesToFree() {
        // Nc3 guards e4 against Nxe4: their threat met, not "prophylaxis".
        var board = Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        var said = lesson(board, Move.fromUci("b1c3"), phase = "opening", threats = listOf("Nxe4"))
        assertTrue(said.toString(), said != null && said.first in listOf("defend", "develop"))
        // A pawn break in a pawn ending: no pieces for the lines to open.
        board = Board("8/5k2/8/4p3/3P4/8/5K2/8 w - - 0 1")
        said = lesson(board, Move.fromUci("d4d5"), phase = "endgame")
        assertTrue(said.toString(), said == null || said.first !in listOf("break", "centre", "space", "pawnStorm"))
    }

    companion object {
        // From the coach fixture's positions: an endgame where Kf4 is prophylaxis (their Ke5
        // threatened).
        const val PROPHYLAXIS = "8/8/p3k3/1p3p1p/1P6/P3K1PP/8/8 w - - 1 41"

        /** A rook to the open d-file, the king with a square to go to: the file is the point. */
        const val OPEN_FILE = "6k1/5pp1/7p/8/8/8/5PPP/R5K1 w - - 0 1"
    }
}
