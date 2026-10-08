package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.WHITE
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * `tests/test_outcome.py` (desktop 70931ab), case for case: where a line leads
 * (`coach/outcome.py`), what it takes and gives, piece by piece.
 */
class OutcomeTest {

    private fun uci(board: Board, sans: String): List<String> {
        val line = board.copy()
        val out = ArrayList<String>()
        for (san in sans.split(" ")) {
            val move = line.parseSan(san)!!
            out.add(move.uci())
            line.push(move)
        }
        return out
    }

    @Test
    fun theExchangeIsNamedNotCountedInPawns() {
        // Be6 Kd8 Bxd7 Qxd7 wins a rook for a bishop. Said "the player wins two pawns", the model
        // added two pawns to the exchange (the locked test set's audit, 2026-09-26).
        val board = Board("2k2b1r/ppqr2p1/3p3p/2pB1Q2/3pP3/3P3P/PPP2PP1/3R1RK1 w - - 4 17")
        val said = lineOutcome(board, uci(board, "Be6 Kd8 Bxd7 Qxd7"), WHITE)
        assertTrue(said, said.startsWith("the player wins the exchange: a rook for a bishop"))
        assertFalse(said, "two pawns" in said)
    }

    @Test
    fun aPlainWinNamesThePiecesAndAPromotionKeepsTheSum() {
        val board = Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
        val won = lineOutcome(board, uci(board, "exd5"), WHITE)
        assertTrue(won, won.startsWith("the player wins a pawn"))
        val queening = Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        val said = lineOutcome(queening, uci(queening, "a8=Q+"), WHITE)
        // In words (the audit, 2026-10-07: figures in the templates; `test_a_sum_in_pawns_is_said_in_words`).
        assertTrue(said, said.startsWith("the player wins material worth eight pawns"))
        // Past eight too: a queen made with a pawn taken on the way.
        val taking = Board("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        val more = lineOutcome(taking, uci(taking, "axb8=Q+"), WHITE)
        assertTrue(more, "material worth eleven pawns" in more)
    }
}
