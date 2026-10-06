package com.sidecoach.state

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.Board
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.TurnPending
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import com.sidecoach.vision.Squares
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The reconciler held to the desktop's, frame for frame.
 *
 * `tools/make_reconcile_fixtures.py` renders games with the desktop's own renderer, reads
 * every frame with the desktop's own recogniser, and runs the desktop's `Reconciler` over
 * the readings -- writing down both the readings and every event it emitted on every frame.
 * The port gets the same readings and must emit the same events, in the same order, on the
 * same frames: a move that settles a frame late is a failure here, which is the point.
 */
class ReconcileTest {

    @Test
    fun everyGameIsFollowedAsThePythonFollowsIt() {
        val text = TestAssets.text("chess/reconcile.json")
        val runs = JSONObject(text).getJSONArray("runs")
        val failures = ArrayList<String>()
        for (r in 0 until runs.length()) {
            val run = runs.getJSONObject(r)
            val name = run.getString("name")
            val bottom = if (run.getString("bottom") == "white") Side.WHITE else Side.BLACK
            val reconciler = Reconciler(bottom = bottom, userSide = bottom)
            val frames = run.getJSONArray("frames")
            var moves = 0
            val began = System.currentTimeMillis()
            for (f in 0 until frames.length()) {
                val frame = frames.getJSONObject(f)
                val readings = readings(frame.getJSONArray("readings"))
                val events = reconciler.update(readings, frame.getDouble("ts"))
                val mine = events.map { line(it) }
                val theirs = frame.getJSONArray("events").let { a ->
                    List(a.length()) { a.getString(it) }
                }
                moves += mine.count { it.startsWith("MovePlayed") }
                if (mine != theirs) {
                    failures.add(
                        "$name frame $f:\n  kotlin: ${mine.joinToString(" | ")}" +
                            "\n  python: ${theirs.joinToString(" | ")}",
                    )
                    if (failures.size >= 6) break
                }
            }
            val took = System.currentTimeMillis() - began
            TestLog.i(
                "sidecoach.state",
                "$name: ${frames.length()} frames, $moves moves, $took ms — ${reconciler.pgn()}",
            )
            if (reconciler.pgn() != run.getString("pgn")) {
                failures.add("$name: pgn is '${reconciler.pgn()}', python's is '${run.getString("pgn")}'")
            }
            if (reconciler.board.fen() != run.getString("final_fen")) {
                failures.add(
                    "$name: final fen is ${reconciler.board.fen()}, python's is ${run.getString("final_fen")}",
                )
            }
            if (failures.size >= 6) break
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** Every cell read exactly from [board], White at the bottom: a clean frame of it. */
    private fun clean(board: Board): Readings = (0 until 64).map { i ->
        val piece = board.pieceAt(Squares.squareAt(i / 8, i % 8, true))
        if (piece == null) {
            CellReading(Kind.EMPTY)
        } else {
            CellReading(Kind.PIECE, symbol = piece.symbol(), colour = if (piece.colour) Side.WHITE else Side.BLACK)
        }
    }.toMutableList()

    /**
     * `test_browsing_anywhere_in_a_long_game_is_browsing` (the state review, 2026-10-01): a
     * click anywhere in the move list shows that position, however far back -- browsing, the
     * game kept, and stepping forward through the list tells nothing. Only the last forty plies
     * were looked through, and the start: the position after ten plies of sixty was adopted by
     * a resync, the game numbered again from it (after the end, a new game begun), and stepping
     * forward told the game's own moves a second time.
     */
    @Test
    fun browsingAnywhereInALongGameIsBrowsing() {
        for (finished in listOf(false, true)) {
            val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE)
            rec.update(clean(Board()), 0.0)
            val played = Board()
            for (san in LONG_GAME) played.push(played.parseSan(san)!!)
            assertEquals(60, played.moveStack.size)
            rec.board = played.copy()
            val first = rec.gameId
            if (finished) rec.end("1/2-1/2", "draw agreed")
            val shown = Board()
            for (move in played.moveStack.take(10)) shown.push(move)
            val events = ArrayList<Event>()
            var ts = 1.0
            repeat(12) { ts += 0.1; events += rec.update(clean(shown), ts) }
            for (move in played.moveStack.subList(10, 16)) {
                shown.push(move)
                repeat(3) { ts += 0.1; events += rec.update(clean(shown), ts) }
            }
            val told = events.filter { it is GameStarted || it is GameEnded || it is Resync || it is Takeback || it is MovePlayed }
            assertTrue("finished=$finished: ${told.map { line(it) }}", told.isEmpty())
            assertEquals(16, rec.viewing)
            assertEquals(first, rec.gameId)
            assertEquals(played.moveStack.map { it.uci() }, rec.board.moveStack.map { it.uci() })
            assertEquals(finished, rec.finished)
            repeat(2) { ts += 0.1; rec.update(clean(played), ts) }
            assertNull(rec.viewing)
        }
    }

    /**
     * `test_position_after_a_finished_game_starts_another_game` (the state review, 2026-10-01):
     * a position adopted after the game ended begins another, and that start is all -- a Resync
     * of the same position after it had the engine measure it twice and the coach note a board
     * re-read that was never read before.
     */
    @Test
    fun aPositionAfterAFinishedGameIsAStartAlone() {
        val rec = Reconciler(bottom = Side.WHITE, userSide = Side.WHITE, resyncHoldFrames = 3)
        val board = Board()
        rec.update(clean(board), 0.0)
        var ts = 0.0
        for (san in listOf("f3", "e5", "g4", "Qh4#")) {
            board.push(board.parseSan(san)!!)
            repeat(2) { ts += 0.1; rec.update(clean(board), ts) }
        }
        assertTrue(rec.finished)
        val first = rec.gameId
        val later = Board(MIDDLEGAME)
        val events = ArrayList<Event>()
        for (i in 0 until 6) events += rec.update(clean(later), 2.0 + 0.1 * i)
        val kinds = events.filter { it is GameStarted || it is Resync }.map { it::class.java.simpleName }
        assertEquals(listOf("GameStarted"), kinds)
        val started = events.filterIsInstance<GameStarted>().single()
        assertEquals("position", started.kind)
        assertEquals(later.boardFen(), Board(started.startFen).boardFen())
        assertEquals(listOf(0), events.filterIsInstance<PositionChanged>().map { it.ply })
        assertTrue(rec.gameId != first && !rec.finished)
    }

    private fun readings(array: org.json.JSONArray): Readings = (0 until 64).map { i ->
        val o = array.getJSONObject(i)
        CellReading(
            kind = when (o.getString("kind")) {
                "empty" -> Kind.EMPTY
                "piece" -> Kind.PIECE
                else -> Kind.UNKNOWN
            },
            symbol = if (o.isNull("symbol")) null else o.getString("symbol"),
            colour = if (o.isNull("colour")) {
                null
            } else if (o.getString("colour") == "white") {
                Side.WHITE
            } else {
                Side.BLACK
            },
            settled = o.getBoolean("settled"),
            hover = o.getBoolean("hover"),
            covered = o.getBoolean("covered"),
            tint = if (o.isNull("tint")) null else o.getString("tint"),
            drawn = o.getBoolean("drawn"),
            highlighted = o.getBoolean("highlighted"),
            guess = if (o.isNull("guess")) null else o.getString("guess"),
        )
    }.toMutableList()

    /** The same line `make_reconcile_fixtures.py` writes; the game id is random, so it is out. */
    private fun line(event: Event): String = when (event) {
        is GameStarted ->
            "GameStarted ${event.kind} ${side(event.userSide)} ${event.startFen}"
        is PositionChanged -> "PositionChanged ${event.ply} ${event.fen}"
        is MovePlayed ->
            "MovePlayed ${event.ply} ${event.san} ${event.uci} " +
                "${if (event.byUser) "user" else "them"} ${event.thinkMs ?: "None"}"
        is Takeback -> "Takeback ${event.plies} ${event.fen}"
        is TurnPending -> "TurnPending ${if (event.pending) "True" else "False"}"
        is Resync -> "Resync ${event.reason} | ${event.fen}"
        is Unexplained -> "Unexplained ${event.reason} | ${event.fen}" + if (event.told) " | told" else ""
        is GameEnded ->
            "GameEnded ${event.result} ${event.reason} | ${event.pgn} | ${event.finalFen}"
        else -> event::class.java.simpleName
    }

    private fun side(side: Pieces.Side): String = if (side == Side.WHITE) "white" else "black"

    companion object {
        /** A Ruy Lopez of 60 plies (the desktop's `LONG_GAME`). */
        val LONG_GAME = listOf(
            "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3",
            "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7", "c4", "c6", "cxb5", "axb5", "Nc3", "Bb7",
            "Bg5", "b4", "Nb1", "h6", "Bh4", "c5", "dxe5", "Nxe4", "Bxe7", "Qxe7", "exd6", "Qf6",
            "Nbd2", "Nxd6", "Nc4", "Nxc4", "Bxc4", "Nb6", "Ne5", "Rae8", "Bxf7+", "Rxf7", "Nxf7",
            "Rxe1+", "Qxe1", "Kxf7", "Qe3", "Qg5", "Qxg5", "hxg5", "b3", "Ke6", "a3", "Kd6",
        )

        /** The desktop's `MIDDLEGAME`. */
        const val MIDDLEGAME = "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"
    }
}
