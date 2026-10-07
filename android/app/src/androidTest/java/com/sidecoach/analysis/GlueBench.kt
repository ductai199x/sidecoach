package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Analysis
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.ui.SettingsStore
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Before and after, end to end (the desktop's tools/engine/bench_glue.py): the app's engine glue
 * ([Assessing], its analyser built from the settings as the app builds it) fed three of the
 * player's games (assets/bench/games.txt, the desktop bench's), move by move, each move from idle
 * -- every answer of the move before in, nothing owed. Per move: how long until its verdict,
 * until the first word on the position it made, until that position is told whole; and the
 * engines' CPU. The same file builds against any checkout (`main` as installed, or this branch).
 * A tool, run in a copy of the app (apart1):
 *
 *   adb shell am instrument -w -e bench 1 -e class com.sidecoach.analysis.GlueBench \
 *       com.sidecoach.test/androidx.test.runner.SidecoachRunnerApart1
 *
 * Writes glue.json to that copy's external files; progress in logcat (tag sidecoach.bench).
 */
@InProcess(1)
@RunWith(AndroidJUnit4::class)
class GlueBench {
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun glue() {
        assumeTrue("a tool: -e bench 1", InstrumentationRegistry.getArguments().getString("bench") == "1")
        val outside = app.getExternalFilesDir(null)!!
        val net = File(app.filesDir, EngineAssets.NET)
        if (net.length() != EngineAssets.NET_BYTES) File(outside, EngineAssets.NET).copyTo(net, overwrite = true)
        val games = InstrumentationRegistry.getInstrumentation().context.assets
            .open("bench/games.txt").bufferedReader().readLines().filter { it.isNotBlank() }
            .map { it.trim().split(" ") }
        val settings = SettingsStore(app)
        Log.i(TAG, "engine settings: ${settings.engine()}")
        val rows = JSONArray()
        for ((g, moves) in games.withIndex()) {
            val assessing = Assessing.fromSettings(app, settings)
            val analyser = assessing.analyser ?: error("no engine: ${assessing.detail}")
            val gameId = "bench$g"
            val board = Board()
            assessing.submit(GameStarted(gameId = gameId, userSide = Pieces.Side.WHITE, startFen = STARTING_FEN))
            settle(assessing, analyser, 0, System.nanoTime(), null)
            for ((i, uci) in moves.withIndex()) {
                val ply = i + 1
                val move = Move.fromUci(uci)
                val before = board.fen()
                val san = board.san(move)
                board.push(move)
                val cpu0 = assessing.engineCpuMs()
                val row = JSONObject().put("game", g).put("ply", ply)
                val t0 = System.nanoTime()
                assessing.submit(
                    MovePlayed(
                        gameId = gameId, ply = ply, san = san, uci = uci, fenBefore = before,
                        fenAfter = board.fen(), byUser = ply % 2 == 1,
                    ),
                )
                settle(assessing, analyser, ply, t0, row)
                row.put("settled", (System.nanoTime() - t0) / 1e9)
                row.put("cpu", (assessing.engineCpuMs() - cpu0) / 1000.0)
                rows.put(row)
                Log.i(TAG, "game $g ply $ply: $row")
            }
            assessing.close()
        }
        File(outside, "glue.json").writeText(JSONObject().put("model", android.os.Build.MODEL).put("rows", rows).toString())
        Log.i(TAG, "glue: ${rows.length()} moves")
    }

    /** Until nothing is owed: the moments the move at [ply]'s answers came, into [row]. */
    private fun settle(assessing: Assessing, analyser: Analyser, ply: Int, since: Long, row: JSONObject?) {
        while (System.nanoTime() - since < PATIENCE_NS) {
            val now = (System.nanoTime() - since) / 1e9
            // Read before the drain: a word is put before nothing is owed, so once this is 0 the
            // drain below holds the last of them.
            val owed = analyser.pending
            for (event in assessing.drain()) {
                if (row == null) continue
                when (event) {
                    is MoveAssessment -> if (event.ply == ply && !row.has("verdict")) {
                        row.put("verdict", now).put("verdict_depth", event.depth)
                    }
                    is Analysis -> if (event.ply == ply) {
                        if (!row.has("first")) row.put("first", now)
                        if (!event.provisional) {
                            // The depth the position's lines reached: the shallowest of them.
                            row.put("whole", now).put("whole_depth", event.lines.minOfOrNull { it.depth } ?: 0)
                        }
                    }
                    else -> Unit
                }
            }
            if (owed == 0) return
            Thread.sleep(2)
        }
        error("ply $ply: not settled in ${PATIENCE_NS / 1_000_000_000} s")
    }

    private companion object {
        const val TAG = "sidecoach.bench"
        const val PATIENCE_NS = 600_000_000_000L // a bound on a hang, never the expected path
    }
}
