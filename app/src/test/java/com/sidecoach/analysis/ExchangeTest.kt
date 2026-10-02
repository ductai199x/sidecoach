package com.sidecoach.analysis

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareName
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Who can take, who can take back, and what a capture nets, held to `analysis/exchange.py`.
 *
 * Every tactic the tagger names and every "hanging" or "defended" the coach says rests on
 * these answers, read by legal moves: a piece pinned to its own king takes nothing, a king
 * never takes back onto a guarded square, and an exchange is played out least valuable piece
 * first. The fixture (`tools/make_exchange_fixtures.py`) asks every square of random games,
 * puzzle positions and the desktop's own test positions, for both colours, and every capture
 * the exchange it starts.
 */
class ExchangeTest {

    @Test
    fun everySquareAndEveryExchangeMatchesThePython() {
        val text = TestAssets.text("chess/exchange.json")
        val positions = JSONObject(text).getJSONArray("positions")
        val failures = ArrayList<String>()
        var squares = 0
        var exchanges = 0
        val began = System.currentTimeMillis()
        for (i in 0 until positions.length()) {
            val case = positions.getJSONObject(i)
            val fen = case.getString("fen")
            val board = Board(fen)
            val rows = case.getJSONArray("squares")
            for (sq in 0 until 64) {
                val row = rows.getJSONArray(sq)
                val where = "$fen ${squareName(sq)}"
                var k = 0
                for (colour in booleanArrayOf(WHITE, BLACK)) {
                    val side = if (colour) "white" else "black"
                    same(failures, "$where captures on ($side)", strings(row.getJSONArray(k)),
                        Exchange.capturesOn(board, colour, sq).map { it.uci() })
                    same(failures, "$where capturers ($side)", ints(row.getJSONArray(k + 1)),
                        Exchange.capturers(board, colour, sq))
                    same(failures, "$where defenders ($side)", ints(row.getJSONArray(k + 2)),
                        Exchange.defenders(board, colour, sq))
                    same(failures, "$where best capture ($side)", row.getInt(k + 3),
                        Exchange.bestCapture(board, colour, sq))
                    k += 4
                }
                same(failures, "$where can be won", row.getBoolean(8), Exchange.canBeWon(board, sq))
                same(failures, "$where hanging", row.getBoolean(9), Exchange.hanging(board, sq))
                squares += 1
            }
            val captures = case.getJSONArray("captures")
            for (c in 0 until captures.length()) {
                val o = captures.getJSONObject(c)
                val view = Board(o.getString("fen"))
                val move = Move.fromUci(o.getString("move"))
                val where = "${o.getString("fen")} ${o.getString("move")}"
                same(failures, "$where exchange", o.getInt("exchange"), Exchange.exchange(view, move))
                same(failures, "$where can retake", o.getBoolean("retake"), Exchange.canRetake(view, move))
                exchanges += 1
            }
            // The board handed in is never changed.
            same(failures, "$fen left as it was", fen, board.fen())
        }
        val took = System.currentTimeMillis() - began
        TestLog.i(TAG, "${positions.length()} positions, $squares squares, $exchanges exchanges in $took ms")
        assertTrue(failures.take(8).joinToString("\n"), failures.isEmpty())
    }

    private fun same(failures: MutableList<String>, what: String, want: Any?, got: Any?) {
        if (want == got) return
        if (failures.size < 8) failures.add("$what: kotlin $got, python $want")
    }

    private companion object {
        const val TAG = "sidecoach.analysis"

        fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

        fun ints(a: JSONArray): List<Int> = (0 until a.length()).map { a.getInt(it) }
    }

    /** `test_a_side_in_check_answers_it_before_its_pieces_can_be_taken` (the facts audit, 2026-10-01). */
    @Test
    fun aSideInCheckAnswersItBeforeItsPiecesCanBeTaken() {
        // White is in check from the d1 rook: a null move left Rf1xf7 illegal, and the defended
        // rook on f7 read as hanging. After Kc2 or Ka2 it is defended.
        var board = Board("8/pk1r1Rp1/1p2p2p/4P1bP/4N3/8/PP6/1K1r1R2 w - - 5 34")
        assertFalse(Exchange.hanging(board, parseSquare("f7")))
        assertFalse(Exchange.canBeWon(board, parseSquare("f7")))
        // Bxh7+: Black answers the check, and the queen on a5 still guards the bishop on g5
        // (hxg5 still wins it for a pawn: it can be won, but it is not hanging).
        board = Board("r1b2rk1/3n1ppB/p2pp3/q5b1/2PQ3P/2N5/PP4PR/3K2N1 b - - 0 17")
        assertFalse(Exchange.hanging(board, parseSquare("g5")))
        assertTrue(Exchange.canBeWon(board, parseSquare("g5")))
        // A knight check that forks the king and an unguarded rook: it hangs after every answer.
        board = Board("r3k3/2N5/8/8/8/8/8/4K3 b - - 0 1")
        assertTrue(Exchange.hanging(board, parseSquare("a8")))
        assertTrue(Exchange.canBeWon(board, parseSquare("a8")))
    }
}
