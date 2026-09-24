package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

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
@RunWith(AndroidJUnit4::class)
class ExchangeTest {

    @Test
    fun everySquareAndEveryExchangeMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/exchange.json").bufferedReader().use { it.readText() }
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
        Log.i(TAG, "${positions.length()} positions, $squares squares, $exchanges exchanges in $took ms")
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
}
