package com.sidecoach.chess

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The chess core held to python-chess, position by position.
 *
 * `tools/make_chess_fixtures.py` records, for 685 positions -- the awkward ones and
 * samples from sixty random games -- every legal move in the order the Python yields it,
 * its SAN, the position it leads to, and what the Python makes of the position: check,
 * mate, stalemate, material, the result, and whether it is valid with either side to move.
 * The reconciler depends on all of it, so the port must agree on all of it. And the rest the
 * desktop leans on: `parse_san`'s failure classes, the draw rules ply by ply, the null move,
 * moves the generator never yields (king-takes-rook castling), and FENs with odd fields.
 */
class ChessTest {

    private fun fixtures(): JSONObject {
        val text = TestAssets.text("chess/positions.json")
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
        TestLog.i("sidecoach.chess", "${data.length()} positions, $moves legal moves, checked in $took ms")
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
            TestLog.i("sidecoach.chess", "perft($depth) of $fen: $mine in $took ms")
            if (mine != want) failures.add("$fen: perft($depth) is $mine, python counts $want")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** `parse_san`: the move, or which of python-chess's three exceptions the text raises. */
    @Test
    fun sanParsesAsPythonChessParsesIt() {
        val cases = fixtures().getJSONArray("parse_san")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val (move, failure) = Board(case.getString("fen")).parseSanFully(case.getString("san"))
            val mine = "${move?.uci()} ${failure?.name?.lowercase()}"
            val theirs = "${case.str("uci")} ${case.str("failure")}"
            if (mine != theirs) failures.add("${case.getString("fen")} '${case.getString("san")}': $mine, python $theirs")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** Repetitions, the fifty- and seventy-five-move rules, claimable or not, ply by ply. */
    @Test
    fun theDrawRulesMatchPlyByPly() {
        val games = fixtures().getJSONArray("draws")
        val failures = ArrayList<String>()
        for (g in 0 until games.length()) {
            val game = games.getJSONObject(g)
            val board = Board(game.getString("fen"))
            val moves = game.getJSONArray("moves")
            val steps = game.getJSONArray("steps")
            for (i in 0 until moves.length()) {
                board.push(Move.fromUci(moves.getString(i)))
                val want = steps.getJSONObject(i)
                val mine = listOf(
                    board.fen(), board.isRepetition(), board.isRepetition(2),
                    board.canClaimThreefoldRepetition(), board.isFiftyMoves(), board.canClaimFiftyMoves(),
                    board.isFivefoldRepetition(), board.isSeventyfiveMoves(), board.isGameOver(),
                    board.isGameOver(claimDraw = true), board.result(claimDraw = true),
                )
                val theirs = listOf(
                    want.getString("fen"), want.getBoolean("repetition"), want.getBoolean("repetition2"),
                    want.getBoolean("threefold"), want.getBoolean("fifty"), want.getBoolean("claim_fifty"),
                    want.getBoolean("fivefold"), want.getBoolean("seventyfive"), want.getBoolean("game_over"),
                    want.getBoolean("game_over_claim"), want.getString("result_claim"),
                )
                if (mine != theirs) failures.add("${game.getString("fen")} ply ${i + 1}: $mine, python $theirs")
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** The null move: pushed, written with no suffix whatever it leaves, and popped. */
    @Test
    fun theNullMoveIsPushedAndWrittenAsPythonChessDoes() {
        val cases = fixtures().getJSONArray("nulls")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val san = board.san(Move.NULL)
            board.push(Move.NULL)
            val mine = listOf(san, board.fen(), board.moveStack.map { it.uci() })
            board.pop()
            val stack = case.getJSONArray("stack")
            val theirs = listOf(case.getString("san"), case.getString("after"), List(stack.length()) { stack.getString(it) })
            if (mine != theirs || board.fen() != case.getString("popped")) {
                failures.add("${case.getString("fen")}: $mine then ${board.fen()}, python $theirs then ${case.getString("popped")}")
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * Moves the generator never yields -- the king taking its own rook, a drop, a king that
     * promotes -- judged as python-chess judges them, and the legal ones written and pushed.
     */
    @Test
    fun movesAreLegalAsPythonChessJudgesThem() {
        val cases = fixtures().getJSONArray("legality")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val board = Board(case.getString("fen"))
            val move = Move.fromUciOrNull(case.getString("uci"))!!
            val legal = board.isLegal(move)
            val mine = mutableListOf<Any?>(board.isPseudoLegal(move), legal, null, null, null)
            if (legal) {
                val after = board.copy(withStack = false)
                mine[2] = board.san(move)
                after.push(move)
                mine[3] = after.fen()
                mine[4] = after.moveStack.last().uci()
            }
            val theirs = listOf(
                case.getBoolean("pseudo_legal"), case.getBoolean("legal"),
                case.str("san"), case.str("after"), case.str("stacked"),
            )
            if (mine != theirs) failures.add("${case.getString("fen")} ${case.getString("uci")}: $mine, python $theirs")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** FENs with odd fields read as python-chess reads them, and the ones it refuses refused. */
    @Test
    fun fensParseAsPythonChessParsesThem() {
        val cases = fixtures().getJSONArray("fens")
        val failures = ArrayList<String>()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val given = case.getString("given")
            val board = try {
                Board(given)
            } catch (exc: IllegalArgumentException) {
                if (case.isNull("error")) failures.add("'$given' refused ($exc), python reads it")
                else if (exc.message != case.getString("error")) failures.add("'$given': '${exc.message}', python '${case.getString("error")}'")
                continue
            }
            if (!case.isNull("error")) {
                failures.add("'$given' read as ${board.fen()}, python refuses it: ${case.getString("error")}")
                continue
            }
            val mine = listOf(
                board.fen(), board.fen(enPassant = "fen"), board.castlingRights.toString(16),
                board.promoted.toString(16), board.isValid(), board.ply(),
            )
            val theirs = listOf(
                case.getString("fen"), case.getString("fen_ep"), case.getString("castling_rights"),
                case.getString("promoted"), case.getBoolean("valid"), case.getInt("ply"),
            )
            if (mine != theirs) failures.add("'$given': $mine, python $theirs")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** A string field, null when JSON has null there (`optString` would say "null"). */
    private fun JSONObject.str(name: String): String? = if (isNull(name)) null else getString(name)

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
