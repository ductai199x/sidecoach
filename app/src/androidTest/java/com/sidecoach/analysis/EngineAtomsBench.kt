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
import java.util.concurrent.CountDownLatch
import kotlin.concurrent.thread

/**
 * The engine's atoms on the phone, as the desktop's tools/engine/atoms*.py measure them: the
 * same 60 positions (assets/bench/atoms_fens.txt), each searched cold (the hash cleared first)
 * to the phone's depth with two lines and with six on one thread, six on two threads, and its
 * threats; then one to four six-line one-thread searches at once. Wall time, the engine
 * process's CPU, and when depth 12 came. A tool, not a test: run in a copy of the app
 * (apart1), never the app itself, with the network pushed beside it:
 *
 *   adb push nn-1a298aa575a0.nnue /sdcard/Android/data/com.sidecoach.apart1/files/
 *   adb shell am instrument -w -e bench 1 -e class com.sidecoach.analysis.EngineAtomsBench \
 *       com.sidecoach.test/androidx.test.runner.SidecoachRunnerApart1
 *
 * Writes atoms.json to that folder; progress in logcat (tag sidecoach.bench).
 */
@InProcess(1)
@RunWith(AndroidJUnit4::class)
class EngineAtomsBench {
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun atoms() {
        assumeTrue("a tool: -e bench 1", InstrumentationRegistry.getArguments().getString("bench") == "1")
        val outside = app.getExternalFilesDir(null)!!
        val net = File(app.filesDir, EngineAssets.NET)
        if (net.length() != EngineAssets.NET_BYTES) File(outside, EngineAssets.NET).copyTo(net, overwrite = true)
        val binary = Engine.binaryIn(app.applicationInfo.nativeLibraryDir)
        val depth = InstrumentationRegistry.getArguments().getString("depth")?.toInt() ?: 14
        val fens = InstrumentationRegistry.getInstrumentation().context.assets
            .open("bench/atoms_fens.txt").bufferedReader().readLines().filter { it.isNotBlank() }
        val out = JSONObject().put("depth", depth).put("model", android.os.Build.MODEL)
        val send = Engine::class.java.getDeclaredMethod("send", String::class.java).apply { isAccessible = true }

        fun open(threads: Int, lines: Int): Engine = Engine(binary, net, threads = threads, hashMb = 64, multipv = lines).also {
            it.open()
            it.analyse(Board(), depth = 1)
        }

        fun cold(engine: Engine) = send.invoke(engine, "setoption name Clear Hash")

        fun search(engine: Engine, board: Board, lines: Int): JSONObject {
            cold(engine)
            var at12 = -1.0
            val c0 = engine.cpuMs()
            val t0 = System.nanoTime()
            val found = engine.analyse(board, depth = depth, multipv = lines, onDepth = { got ->
                if (at12 < 0 && got.isNotEmpty() && got[0].depth >= 12) at12 = (System.nanoTime() - t0) / 1e9
            })
            val wall = (System.nanoTime() - t0) / 1e9
            check(found.isNotEmpty() && found[0].depth == depth) { "searched to ${found.firstOrNull()?.depth}" }
            return JSONObject().put("wall", wall).put("cpu", (engine.cpuMs() - c0) / 1000.0).put("at12", at12)
        }

        // A: each variant on an engine of its own.
        val variants = linkedMapOf("2x1" to open(1, 2), "6x1" to open(1, 6), "6x2" to open(2, 6))
        val threatsEngine = open(1, 6)
        val alone = JSONArray()
        try {
            for ((i, fen) in fens.withIndex()) {
                val board = Board(fen)
                val row = JSONObject().put("fen", fen)
                val names = variants.keys.toList()
                for (name in names.drop(i % names.size) + names.take(i % names.size)) {
                    row.put(name, search(variants.getValue(name), board, name.substringBefore("x").toInt()))
                }
                cold(threatsEngine)
                val t0 = System.nanoTime()
                threatMoves(threatsEngine, board, depth)
                row.put("threats", JSONObject().put("wall", (System.nanoTime() - t0) / 1e9))
                alone.put(row)
                Log.i(TAG, "alone ${i + 1}/${fens.size}")
                out.put("alone", alone)
                File(outside, "atoms.json").writeText(out.toString())
            }
        } finally {
            variants.values.forEach { it.close() }
            threatsEngine.close()
        }
        // B: k six-line one-thread searches at once.
        val together = JSONArray()
        for (k in listOf(1, 2, 3, 4)) {
            val engines = List(k) { open(1, 6) }
            try {
                for (batch in fens.chunked(k)) {
                    val start = CountDownLatch(1)
                    val results = arrayOfNulls<JSONObject>(batch.size)
                    val workers = batch.mapIndexed { j, fen ->
                        thread {
                            start.await()
                            results[j] = search(engines[j], Board(fen), 6).put("fen", fen).put("k", k)
                        }
                    }
                    start.countDown()
                    workers.forEach { it.join() }
                    results.forEach { together.put(it) }
                }
            } finally {
                engines.forEach { it.close() }
            }
            Log.i(TAG, "side by side k=$k done")
            out.put("side_by_side", together)
            File(outside, "atoms.json").writeText(out.toString())
        }
        Log.i(TAG, "done")
    }

    companion object {
        private const val TAG = "sidecoach.bench"
    }
}
