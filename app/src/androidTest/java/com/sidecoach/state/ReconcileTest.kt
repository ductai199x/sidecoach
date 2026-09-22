package com.sidecoach.state

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The reconciler held to the desktop's, frame for frame.
 *
 * `tools/make_reconcile_fixtures.py` renders games with the desktop's own renderer, reads
 * every frame with the desktop's own recogniser, and runs the desktop's `Reconciler` over
 * the readings -- writing down both the readings and every event it emitted on every frame.
 * The port gets the same readings and must emit the same events, in the same order, on the
 * same frames: a move that settles a frame late is a failure here, which is the point.
 */
@RunWith(AndroidJUnit4::class)
class ReconcileTest {

    @Test
    fun everyGameIsFollowedAsThePythonFollowsIt() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/reconcile.json").bufferedReader().use { it.readText() }
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
            Log.i(
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

    private fun readings(array: org.json.JSONArray): Readings = Array(64) { i ->
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
    }

    /** The same line `make_reconcile_fixtures.py` writes; the game id is random, so it is out. */
    private fun line(event: Event): String = when (event) {
        is GameStarted ->
            "GameStarted ${event.kind} ${side(event.userSide)} ${event.startFen}"
        is PositionChanged -> "PositionChanged ${event.ply} ${event.fen}"
        is MovePlayed ->
            "MovePlayed ${event.ply} ${event.san} ${event.uci} " +
                "${if (event.byUser) "user" else "them"} ${event.thinkMs}"
        is Takeback -> "Takeback ${event.plies} ${event.fen}"
        is Resync -> "Resync ${event.reason} | ${event.fen}"
        is GameEnded ->
            "GameEnded ${event.result} ${event.reason} | ${event.pgn} | ${event.finalFen}"
        else -> event::class.java.simpleName
    }

    private fun side(side: Pieces.Side): String = if (side == Side.WHITE) "white" else "black"
}
