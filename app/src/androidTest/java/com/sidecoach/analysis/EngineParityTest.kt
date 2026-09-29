package com.sidecoach.analysis

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * What the engine is told, and what is made of its answers, held to the desktop's
 * `sidecoach/analysis/engine.py` driving Stockfish through python-chess
 * (`tools/make_engine_fixtures.py`).
 *
 * The engine is scripted: a shell script that announces Stockfish 19's options and answers
 * each search with what the desktop's scripted engine answered to the same `position` command
 * and number of lines -- a different command is no answer at all. So the lines come back as the
 * desktop's did only when the phone sent the game's moves as python-chess sends them
 * (`position startpos moves ...`, the repetitions in them; the FEN alone after a null move),
 * and every line the phone sent is compared with the desktop's: the options only when they
 * change, a new game before the first search. Twice: an engine that gives win-draw-loss, and
 * one that does not, whose numbers then come from python-chess's model.
 *
 * Two differences are the phone's own and taken out before comparing: it waits for the engine
 * to be ready once the options are set (the network loads then), and it keeps a search's time
 * itself (`go` without `movetime`), so that the voice's pause does not count against it.
 */
@RunWith(AndroidJUnit4::class)
class EngineParityTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    /** The Engine the app builds, started as the shell running [script] instead of Stockfish. */
    private class Scripted(private val script: File) :
        Engine(File("/system/bin/sh"), null, threads = 2, hashMb = 64, multipv = 3, pvPlies = 12, maxSeconds = 5.0) {
        override fun launch(): Process =
            ProcessBuilder("/system/bin/sh", script.absolutePath).redirectErrorStream(true).start()
    }

    /** The scripted engine: every line it reads logged, each search answered as the desktop's was. */
    private fun script(persona: JSONObject, log: File, into: File): File {
        val out = ArrayList<String>()
        out.add("LOG='${log.absolutePath}'")
        out.add("mpv=1")
        out.add("pos=")
        out.add("while IFS= read -r line; do")
        out.add("  echo \"\$line\" >> \"\$LOG\"")
        out.add("  case \"\$line\" in")
        out.add("    uci)")
        out.add("      echo 'id name Scripted'")
        out.add("      echo 'id author sidecoach'")
        val options = persona.getJSONArray("options")
        for (i in 0 until options.length()) out.add("      echo '${options.getString(i)}'")
        out.add("      echo 'uciok'")
        out.add("      ;;")
        out.add("    isready) echo 'readyok' ;;")
        out.add("    'setoption name MultiPV value '*) mpv=\${line##* } ;;")
        out.add("    position*) pos=\$line ;;")
        out.add("    go*)")
        out.add("      case \"\$mpv|\$pos\" in")
        val replies = persona.getJSONObject("replies")
        for (key in replies.keys()) {
            out.add("        '$key')")
            val reply = replies.getJSONArray(key)
            for (i in 0 until reply.length()) out.add("          echo '${reply.getString(i)}'")
            out.add("          ;;")
        }
        out.add("        *) echo 'bestmove (none)' ;;")
        out.add("      esac")
        out.add("      ;;")
        out.add("    quit) exit 0 ;;")
        out.add("  esac")
        out.add("done")
        into.writeText(out.joinToString("\n") + "\n")
        return into
    }

    @Test
    fun whatTheEngineIsToldAndWhatComesBackMatchTheDesktop() {
        val text = instrumentation.context.assets.open("chess/engine.json").bufferedReader().use { it.readText() }
        val data = JSONObject(text)
        val cache = instrumentation.targetContext.cacheDir
        val failures = ArrayList<String>()
        for (name in listOf("wdl", "nowdl")) {
            val persona = data.getJSONObject(name)
            val log = File(cache, "scripted-$name.log").apply { delete() }
            val engine = Scripted(script(persona, log, File(cache, "scripted-$name.sh")))
            val scenarios = persona.getJSONArray("scenarios")
            for (i in 0 until scenarios.length()) {
                val scenario = scenarios.getJSONObject(i)
                val board = Board(scenario.getString("fen"))
                val moves = scenario.getJSONArray("moves")
                for (m in 0 until moves.length()) board.push(Move.fromUci(moves.getString(m)))
                val given = if (scenario.getBoolean("stack")) board else board.copy(withStack = false)
                // A best line carried on to MIN_BEST_PLIES (`carry_best`: the parity audit,
                // 2026-09-29) is searched on from its end, and those searches are compared too.
                val lines = engine.analyse(
                    given,
                    depth = scenario.getInt("depth"),
                    multipv = scenario.getInt("multipv"),
                    carryBest = scenario.optBoolean("carry_best"),
                )
                val want = lines(scenario.getJSONArray("lines"))
                if (!same(lines, want)) failures.add("$name, ${scenario.getString("name")}: $lines, desktop $want")
            }
            engine.close()
            // The phone's own `isready` at the start (the network loads then), and the desktop's
            // `movetime` (the phone keeps the time itself), taken out.
            val phone = log.readLines().toMutableList().apply { remove("isready") }
            val sent = persona.getJSONArray("sent")
            val desktop = List(sent.length()) { sent.getString(it).replace(Regex(" movetime \\d+"), "") }
            if (phone != desktop) {
                failures.add("$name: the phone sent\n  ${phone.joinToString("\n  ")}\nthe desktop\n  ${desktop.joinToString("\n  ")}")
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun lines(array: JSONArray): List<Line> = List(array.length()) { i ->
        val line = array.getJSONObject(i)
        val moves = line.getJSONArray("moves")
        val wdl = line.optJSONArray("wdl")
        Line(
            moves = List(moves.length()) { moves.getString(it) },
            depth = line.getInt("depth"),
            expectedScore = line.getDouble("expected_score"),
            scoreCp = if (line.isNull("score_cp")) null else line.getInt("score_cp"),
            mate = if (line.isNull("mate")) null else line.getInt("mate"),
            wdl = if (wdl == null) null else Triple(wdl.getInt(0), wdl.getInt(1), wdl.getInt(2)),
        )
    }

    /** Equal, the expected score to within the last bits two `exp`s may differ by. */
    private fun same(mine: List<Line>, theirs: List<Line>): Boolean =
        mine.size == theirs.size && mine.zip(theirs).all { (a, b) ->
            a.copy(expectedScore = 0.0) == b.copy(expectedScore = 0.0) &&
                Math.abs(a.expectedScore - b.expectedScore) < 1e-12
        }
}
