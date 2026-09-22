package com.sidecoach.chess

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The chess core held to python-chess, position by position.
 *
 * `tools/make_chess_fixtures.py` records, for 685 positions -- the awkward ones and
 * samples from sixty random games -- every legal move in the order the Python yields it,
 * its SAN, the position it leads to, and what the Python makes of the position: check,
 * mate, stalemate, material, the result, and whether it is valid with either side to move.
 * The reconciler depends on all of it, so the port must agree on all of it.
 */
@RunWith(AndroidJUnit4::class)
class ChessTest {

    private fun fixtures(): JSONObject {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/positions.json").bufferedReader().use { it.readText() }
        return JSONObject(text)
    }

    @Test
    fun everyPositionReadsAsThePythonReadsIt() {
        val data = fixtures().getJSONArray("positions")
        val failures = ArrayList<String>()
        var moves = 0
        val began = System.currentTimeMillis()
        for (i in 0 until data.length()) {
            val want = data.getJSONObject(i)
            val fen = want.getString("fen")
            val board = Board(fen)
            // python-chess keeps the square even when the FEN does not print it.
            if (!want.isNull("ep")) board.epSquare = parseSquare(want.getString("ep"))

            fun check(what: String, mine: Any?, theirs: Any?) {
                if (mine != theirs && failures.size < 20) {
                    failures.add("$fen: $what is $mine, python says $theirs")
                }
            }

            check("the fen round-trip", board.fen(), fen)
            check("board_fen", board.boardFen(), want.getString("board_fen"))
            check("check", board.isCheck(), want.getBoolean("check"))
            check("checkmate", board.isCheckmate(), want.getBoolean("checkmate"))
            check("stalemate", board.isStalemate(), want.getBoolean("stalemate"))
            check("insufficient material", board.isInsufficientMaterial(), want.getBoolean("insufficient"))
            check("game over", board.isGameOver(), want.getBoolean("game_over"))
            check("result", board.result(), want.getString("result"))

            val validity = want.getJSONObject("valid")
            for ((name, turn) in listOf("white" to WHITE, "black" to BLACK)) {
                val trial = board.copy(withStack = false)
                trial.turn = turn
                check("valid with $name to move", trial.isValid(), validity.getBoolean(name))
            }

            val wantMoves = want.getJSONArray("moves")
            val wantSans = want.getJSONArray("sans")
            val wantAfter = want.getJSONArray("after")
            val mine = board.legalMoves()
            moves += mine.size
            if (mine.size != wantMoves.length()) {
                if (failures.size < 20) {
                    failures.add("$fen: ${mine.size} legal moves, python has ${wantMoves.length()}")
                }
                continue
            }
            for (m in mine.indices) {
                if (mine[m].uci() != wantMoves.getString(m)) {
                    if (failures.size < 20) {
                        failures.add(
                            "$fen: move $m is ${mine[m].uci()}, python's is ${wantMoves.getString(m)}",
                        )
                    }
                    break
                }
                val san = board.san(mine[m])
                if (san != wantSans.getString(m)) {
                    if (failures.size < 20) {
                        failures.add("$fen: ${mine[m].uci()} reads $san, python writes ${wantSans.getString(m)}")
                    }
                }
                val after = board.copy(withStack = false)
                after.push(mine[m])
                if (after.fen() != wantAfter.getString(m)) {
                    if (failures.size < 20) {
                        failures.add(
                            "$fen: after ${mine[m].uci()} the fen is ${after.fen()}, " +
                                "python's is ${wantAfter.getString(m)}",
                        )
                    }
                }
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i("sidecoach.chess", "${data.length()} positions, $moves legal moves, checked in $took ms")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun pushAndPopLeaveThePositionWhereItWas() {
        val data = fixtures().getJSONArray("positions")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val fen = data.getJSONObject(i).getString("fen")
            val board = Board(fen)
            for (move in board.legalMoves()) {
                board.push(move)
                board.pop()
                if (board.fen() != fen && failures.size < 10) {
                    failures.add("$fen: ${move.uci()} left ${board.fen()}")
                }
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theMoveCountsMatchToDepth() {
        val data = fixtures().getJSONArray("perft")
        val failures = ArrayList<String>()
        for (i in 0 until data.length()) {
            val o = data.getJSONObject(i)
            val fen = o.getString("fen")
            val depth = o.getInt("depth")
            val want = o.getLong("nodes")
            val began = System.currentTimeMillis()
            val mine = perft(Board(fen), depth)
            val took = System.currentTimeMillis() - began
            Log.i("sidecoach.chess", "perft($depth) of $fen: $mine in $took ms")
            if (mine != want) failures.add("$fen: perft($depth) is $mine, python counts $want")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun perft(board: Board, depth: Int): Long {
        if (depth == 0) return 1
        var total = 0L
        for (move in board.legalMoves()) {
            board.push(move)
            total += perft(board, depth - 1)
            board.pop()
        }
        return total
    }
}
