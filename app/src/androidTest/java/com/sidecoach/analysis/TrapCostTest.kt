package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import org.json.JSONObject
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * What the traps ([Traps.trapLines], desktop fc30fa7) cost per analysed position, beside the rest
 * of [GameAnnotator.measure]: the main search (depth 18, six lines, the best carried on) and the
 * threat search, with the engine as the app builds it from its default settings (two threads,
 * 64 MB, a 5-second cap). Not a parity test and not part of the suite: it runs only when asked,
 *
 *     adb -s <serial> shell am instrument -w -e bench true \
 *         -e class com.sidecoach.analysis.TrapCostTest com.sidecoach.test/androidx.test.runner.AndroidJUnitRunner
 *
 * and logs the numbers under the `sidecoach.analysis` tag. The positions are the coach fixture's
 * master-game and Lichess positions, analysed one after another on one engine, as a game is.
 */
@RunWith(AndroidJUnit4::class)
class TrapCostTest {

    /** The engine, counting its searches. */
    private class Counting(binary: File, net: File) :
        Engine(binary, net, threads = 2, hashMb = 64, multipv = 6, pvPlies = 12, maxSeconds = 5.0) {
        var calls = 0

        override fun analyse(
            board: Board,
            depth: Int?,
            timeS: Double?,
            multipv: Int?,
            pvPlies: Int?,
            carryBest: Boolean,
            onDepth: ((List<Line>) -> Unit)?,
        ): List<Line> {
            calls += 1
            return super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        }
    }

    @Test
    fun whatTheTrapsCostPerPosition() {
        val args = InstrumentationRegistry.getArguments()
        assumeTrue("a benchmark: run with -e bench true", args.getString("bench") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        assumeTrue("the engine or its network is missing", binary.exists() && net != null)
        val limit = args.getString("positions")?.toIntOrNull() ?: 40

        val text = InstrumentationRegistry.getInstrumentation().context.assets
            .open("chess/coach.json").bufferedReader().use { it.readText() }
        val cases = JSONObject(text).getJSONArray("cases")
        val fens = ArrayList<String>()
        for (source in listOf("masters", "lichess")) {
            var taken = 0
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                if (!case.getString("source").startsWith(source) || taken >= limit / 2) continue
                fens.add(case.getString("fen"))
                taken += 1
            }
        }

        val main = ArrayList<Long>()
        val threats = ArrayList<Long>()
        val traps = ArrayList<Long>()
        val searches = ArrayList<Int>()
        var found = 0
        Counting(binary, net!!).use { engine ->
            engine.open()
            for (fen in fens) {
                val board = Board(fen)
                val t0 = System.nanoTime()
                val lines = engine.analyse(board, depth = 18, multipv = 6, carryBest = true)
                val t1 = System.nanoTime()
                threatMoves(engine, board, 18)
                val t2 = System.nanoTime()
                val before = engine.calls
                val trapped = Traps.trapLines(engine, board, lines)
                val t3 = System.nanoTime()
                main.add((t1 - t0) / 1_000_000)
                threats.add((t2 - t1) / 1_000_000)
                traps.add((t3 - t2) / 1_000_000)
                searches.add(engine.calls - before)
                if (trapped.isNotEmpty()) found += 1
            }
        }

        fun stats(xs: List<Long>): String {
            val sorted = xs.sorted()
            val mean = xs.sum() / xs.size
            return "mean $mean ms, median ${sorted[sorted.size / 2]} ms, max ${sorted.last()} ms"
        }
        val without = main.indices.map { main[it] + threats[it] }
        val with = main.indices.map { main[it] + threats[it] + traps[it] }
        val searched = searches.count { it > 0 }
        val summary = "${fens.size} positions: main search ${stats(main)}; threats ${stats(threats)}; " +
            "traps ${stats(traps)}, ${searches.sum()} searches at depth ${Traps.TRAP_DEPTH} in " +
            "$searched positions, a trap found in $found; measure() without traps ${stats(without)}, " +
            "with ${stats(with)}; traps add ${100 * traps.sum() / without.sum()}% in all, " +
            "${if (searched > 0) traps.sum() / searched else 0} ms per position that searches"
        Log.i(TAG, summary)
        for (i in fens.indices) {
            Log.i(TAG, "${fens[i]}: main ${main[i]} threats ${threats[i]} traps ${traps[i]} (${searches[i]} searches)")
        }
    }

    companion object {
        private const val TAG = "sidecoach.analysis"
    }
}
