package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.chess.Board
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The phone's searches, before and after the redesign, on 300 positions (assets/bench/
 * tail_fens.txt: the 150 slowest of the desktop's 8,886 and 150 at random), each cold (the hash
 * cleared first), one thread, one at a time: the phone's search now -- six lines to its depth --
 * and the redesign's -- one line to its depth, and six lines to depth 12 -- and the threats.
 * Wall time and the engine's CPU. A tool, run in a copy of the app (apart1):
 *
 *   adb shell am instrument -w -e bench 1 -e class com.sidecoach.analysis.EngineTailBench \
 *       com.sidecoach.test/androidx.test.runner.SidecoachRunnerApart1
 *
 * Writes tail.json to that copy's external files; progress in logcat (tag sidecoach.bench).
 */
@InProcess(1)
@RunWith(AndroidJUnit4::class)
class EngineTailBench {
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun tail() {
        assumeTrue("a tool: -e bench 1", InstrumentationRegistry.getArguments().getString("bench") == "1")
        val outside = app.getExternalFilesDir(null)!!
        val net = File(app.filesDir, EngineAssets.NET)
        if (net.length() != EngineAssets.NET_BYTES) File(outside, EngineAssets.NET).copyTo(net, overwrite = true)
        val binary = Engine.binaryIn(app.applicationInfo.nativeLibraryDir)
        val depth = InstrumentationRegistry.getArguments().getString("depth")?.toInt() ?: 14
        val fens = InstrumentationRegistry.getInstrumentation().context.assets
            .open("bench/tail_fens.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        val send = Engine::class.java.getDeclaredMethod("send", String::class.java).apply { isAccessible = true }
        val engine = Engine(binary, net, threads = 1, hashMb = 64, multipv = 6)
        engine.open()
        engine.analyse(Board(), depth = 1)

        fun timed(board: Board, d: Int, lines: Int): JSONObject {
            send.invoke(engine, "setoption name Clear Hash")
            val c0 = (engine.cpuMs() ?: 0)
            val t0 = System.nanoTime()
            val found = engine.analyse(board, depth = d, multipv = lines)
            return JSONObject()
                .put("wall", (System.nanoTime() - t0) / 1e9)
                .put("cpu", ((engine.cpuMs() ?: 0) - c0) / 1000.0)
                .put("depth", found.firstOrNull()?.depth ?: 0)
        }

        val out = JSONObject().put("depth", depth).put("model", android.os.Build.MODEL)
        val rows = JSONArray()
        // `-e one 1`: one line to the depth alone, on the 150 random positions (the second half).
        val oneOnly = InstrumentationRegistry.getArguments().getString("one") == "1"
        val name = if (oneOnly) "tail-one-$depth.json" else "tail.json"
        try {
            for ((i, fen) in fens.withIndex()) {
                val board = Board(fen)
                val row = JSONObject().put("fen", fen)
                if (oneOnly) {
                    if (i < 150) continue
                    row.put("one", timed(board, depth, 1))
                    rows.put(row)
                    out.put("rows", rows)
                    File(outside, name).writeText(out.toString())
                    continue
                }
                row.put("six", timed(board, depth, 6)) // before: the phone's search now
                row.put("one", timed(board, depth, 1)) // after: one line to the depth
                row.put("six12", timed(board, 12, 6)) // after: six lines to 12, the hint
                send.invoke(engine, "setoption name Clear Hash")
                val t0 = System.nanoTime()
                threatMoves(engine, board, depth)
                row.put("threats", JSONObject().put("wall", (System.nanoTime() - t0) / 1e9))
                rows.put(row)
                if (i % 10 == 9) {
                    Log.i(TAG, "tail ${i + 1}/${fens.size}")
                    out.put("rows", rows)
                    File(outside, name).writeText(out.toString())
                }
            }
        } finally {
            engine.close()
        }
        out.put("rows", rows)
        File(outside, name).writeText(out.toString())
        Log.i(TAG, "done")
    }

    companion object {
        private const val TAG = "sidecoach.bench"
    }
}
