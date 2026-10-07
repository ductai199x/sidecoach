package com.sidecoach.state

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.Piece
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `tests/test_en_passant_seen.py` (desktop 754cfc9, 6c571dd), case for case: a position read from pixels
 * learns what the site's last-move highlight shows -- a pawn that has just advanced two squares
 * can be taken in passing (Tai's chess.com puzzle, 2026-09-24: after c2-c4 the coach did not see
 * bxc3 e.p. and recommended Bd7).
 */
class EnPassantSeenTest {

    // Tai's puzzle, Black to move after c2-c4, with the black pawn on b4.
    private fun placement(): Board = Board("2r1k2r/2q1bppp/p2p4/1b1Pp3/1pP2P2/1P1Q4/P2BB1PP/R4RK1 w - - 0 1")

    private val c2 = parseSquare("c2")
    private val c3 = parseSquare("c3")
    private val c4 = parseSquare("c4")

    @Test
    fun theHighlightedDoubleStepGivesTheEnPassantSquare() {
        val board = placement()
        assertEquals(LastMoveSeen(WHITE, c3), lastMoveSeen(board, setOf(c2, c4)))
        val adopted = adoptPlacement(board, BLACK, setOf(c2, c4))
        assertNotNull(adopted)
        val (position, turnUnknown) = adopted!!
        assertEquals(BLACK, position.turn)
        assertFalse(turnUnknown)
        assertEquals(c3, position.epSquare)
        assertTrue(position.isLegal(Move.fromUci("b4c3")))
        assertTrue(position.fen().contains(" c3 "))
    }

    @Test
    fun noHighlightNoEnPassant() {
        val adopted = adoptPlacement(placement(), BLACK)
        assertNotNull(adopted)
        assertNull(adopted!!.first.epSquare)
    }

    @Test
    fun aOneSquareAdvanceSaysWhoseMoveItIsAndNoEnPassant() {
        val board = placement()
        assertEquals(LastMoveSeen(WHITE), lastMoveSeen(board, setOf(c3, c4)))
        // The start square occupied: no move from it.
        board.setPieceAt(c2, Piece(PAWN, WHITE))
        assertNull(lastMoveSeen(board, setOf(c2, c4)))
    }

    @Test
    fun aPieceMoveHighlightedSettlesTheSideToMove() {
        // Lichess 000rO after the setup move Ka5 (a6-a5): both sides could be to move on the
        // placement alone; the highlight says White moved, so Black is to move.
        val board = Board("3R4/8/8/KB2b3/1p6/1P2k3/3p4/8 w - - 0 1")
        val adopted = adoptPlacement(board, WHITE, setOf(parseSquare("a6"), parseSquare("a5")))
        assertNotNull(adopted)
        assertEquals(BLACK, adopted!!.first.turn)
        assertFalse(adopted.second)
    }
}
