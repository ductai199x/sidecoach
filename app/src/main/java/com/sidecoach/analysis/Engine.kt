package com.sidecoach.analysis

import android.os.SystemClock
import android.util.Log
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Line
import java.io.BufferedReader
import java.io.File
import java.io.IOException
import java.io.InputStreamReader
import java.io.Writer
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class EngineUnavailable(message: String) : RuntimeException(message)

/**
 * Stockfish over UCI: one persistent process, MultiPV lines, real win-draw-loss numbers.
 * A port of `sidecoach/analysis/engine.py`.
 *
 * The desktop drives Stockfish through python-chess; there is no such library here, so the
 * protocol is spoken directly. Everything else is the desktop's: the same options, the same
 * restart-once rule, the same line for each MultiPV index at the last depth reached, and
 * scores from the mover's point of view in expected score, never centipawns.
 *
 * Three things are the phone's own. The executable lives in the APK's library directory,
 * because that is the only place an Android app may execute a file from. The network is a
 * separate 98 MB file the app fetches, because embedding it would put 98 MB in the APK for
 * every ABI; it is handed over through the EvalFile option. And the engine can be paused
 * ([pause]) while the voice synthesises.
 */
open class Engine(
    private val binary: File,
    private val net: File?,
    var threads: Int = 2,
    var hashMb: Int = 64,
    var multipv: Int = 6,
    var pvPlies: Int = DEFAULT_PV_PLIES,
    /** How a line's score becomes an expected score; see [expectedScore]. */
    var scoreModel: String = DEFAULT_SCORE_MODEL,
    /**
     * No single position may hold up a game. Some positions are pathologically slow to reach
     * a given depth, and one of those in a post-game review stalls the whole run.
     */
    var maxSeconds: Double? = 5.0,
) : AutoCloseable {

    private var process: Process? = null

    /** The engine process's id while it runs, read once when it starts ([processId]). */
    @Volatile private var enginePid: Int? = null
    private var writer: Writer? = null
    private var reader: BufferedReader? = null
    private val lock = ReentrantLock()
    private val options = HashSet<String>()
    private var currentMultipv = -1

    /** What the engine calls itself, for the log. */
    var name: String = binary.name
        private set

    /**
     * The engine's own CPU time in milliseconds, or 0 while it is not running. It is a
     * separate process, so the app's /proc/self/stat does not count it and a profile that
     * reads only that one sees the vision and none of the search.
     */
    fun cpuMs(): Long {
        val pid = enginePid ?: return 0
        return try {
            val stat = File("/proc/$pid/stat").readText()
            val after = stat.substring(stat.lastIndexOf(')') + 2).split(" ")
            (after[11].toLong() + after[12].toLong()) * 10
        } catch (exc: Exception) {
            0
        }
    }

    fun open() {
        if (process != null) return
        if (!binary.exists()) throw EngineUnavailable("no engine at $binary")
        val started = try {
            ProcessBuilder(binary.absolutePath)
                .redirectErrorStream(true)
                .directory(binary.parentFile)
                .start()
        } catch (exc: IOException) {
            throw EngineUnavailable("cannot start $binary: $exc")
        }
        process = started
        enginePid = processId(started)
        writer = started.outputStream.bufferedWriter()
        reader = BufferedReader(InputStreamReader(started.inputStream))
        options.clear()
        send("uci")
        readUntil("uciok") { line ->
            if (line.startsWith("option name ")) {
                options.add(line.removePrefix("option name ").substringBefore(" type").trim())
            } else if (line.startsWith("id name ")) {
                name = line.removePrefix("id name ").trim()
            }
        }
        // The network first: without it the engine answers `isready` and then dies on the
        // first search.
        if (net != null && "EvalFile" in options) {
            setOption("EvalFile", net.absolutePath)
        }
        if ("Threads" in options) setOption("Threads", threads.toString())
        if ("Hash" in options) setOption("Hash", hashMb.toString())
        // Real win-draw-loss numbers beat a model fitted to centipawns; ask for them when
        // the engine has them.
        if ("UCI_ShowWDL" in options) setOption("UCI_ShowWDL", "true")
        currentMultipv = -1
        send("isready")
        readUntil("readyok") {}
        Log.i(TAG, "$name ready: ${threads} threads, ${hashMb} MB, net ${net?.name ?: "embedded"}")
        // An engine started while the voice holds the old one paused (a restart) waits too.
        synchronized(clock) { if (paused) signal(SIGSTOP) }
    }

    /** Threads and hash on the running engine (the settings saved mid-session). */
    fun reconfigure(threads: Int? = null, hashMb: Int? = null) {
        if (threads != null) this.threads = threads
        if (hashMb != null) this.hashMb = hashMb
        if (process == null) return
        runCatching {
            if ("Threads" in options) setOption("Threads", this.threads.toString())
            if ("Hash" in options) setOption("Hash", this.hashMb.toString())
        }
    }

    override fun close() {
        val current = process ?: return
        // A stopped process reads nothing, `quit` included: it is let go on first.
        synchronized(clock) {
            if (paused) signal(SIGCONT)
            enginePid = null
            budgetMs = 0
            clock.notifyAll()
        }
        process = null
        runCatching { send("quit") }  // a dead engine needs no goodbye
        runCatching { current.waitFor() }
        runCatching { writer?.close() }
        runCatching { reader?.close() }
        runCatching { current.destroy() }
        writer = null
        reader = null
    }

    // The voice's turn

    /**
     * Guards the pause and the running search's time: [paused], [budgetMs], [ranMs],
     * [runningSince], [searchNumber].
     */
    private val clock = Object()
    private var paused = false

    /** The running search's time limit in milliseconds, 0 when none is kept. */
    private var budgetMs = 0L

    /** The time the search has run, up to [runningSince]. */
    private var ranMs = 0L

    /** When the search last ran on, 0 while it is paused or none is timed. */
    private var runningSince = 0L
    private var searchNumber = 0

    /**
     * Hold the engine still while the voice synthesises (Tai, 2026-09-23: Kokoro or Stockfish
     * at work, not both -- they share the phone's big cores, and with both at work each runs at
     * half speed). The process is stopped where it stands, never ended: its hash, which the
     * next searches draw on, is kept whole (Tai: pause, never quit). A search's time limit
     * counts only the time the engine runs ([watch]). Safe from any thread, and a no-op when
     * already paused.
     */
    fun pause() {
        synchronized(clock) {
            if (paused) return
            paused = true
            if (runningSince != 0L) {
                ranMs += SystemClock.elapsedRealtime() - runningSince
                runningSince = 0
            }
            signal(SIGSTOP)
            clock.notifyAll()
        }
    }

    /** Let a [pause]d engine run on from where it stopped. A no-op when it is not paused. */
    fun resume() {
        synchronized(clock) {
            if (!paused) return
            paused = false
            signal(SIGCONT)
            if (budgetMs != 0L) runningSince = SystemClock.elapsedRealtime()
            clock.notifyAll()
        }
    }

    private fun signal(which: Int) {
        val pid = enginePid ?: return
        runCatching { android.os.Process.sendSignal(pid, which) }
    }

    /** Keep a search's time limit of [ms] from now: [watch] stops it when the time has run. */
    private fun arm(ms: Long) {
        val number = synchronized(clock) {
            budgetMs = ms
            ranMs = 0
            runningSince = if (paused) 0 else SystemClock.elapsedRealtime()
            ++searchNumber
        }
        Thread({ watch(number) }, "engine-clock").apply {
            isDaemon = true
            start()
        }
    }

    private fun disarm() {
        synchronized(clock) {
            budgetMs = 0
            runningSince = 0
            clock.notifyAll()
        }
    }

    /**
     * The time limit of search [number]: `stop` once it has run its time, the pauses not
     * counted. The engine's own `movetime` would count them, and cut a search short the moment
     * it resumed; `stop` ends the search as `movetime` does, with its best move.
     */
    private fun watch(number: Int) {
        synchronized(clock) {
            while (searchNumber == number && budgetMs != 0L) {
                if (paused) {
                    clock.wait()
                    continue
                }
                val left = budgetMs - ranMs - (SystemClock.elapsedRealtime() - runningSince)
                if (left <= 0) {
                    runCatching { send("stop") }
                    budgetMs = 0
                    return
                }
                clock.wait(left)
            }
        }
    }

    /**
     * The best lines for `board`, best first, from the side to move's point of view. A
     * finished position has no lines: there is nothing to play.
     */
    open fun analyse(
        board: Board,
        depth: Int? = null,
        timeS: Double? = null,
        multipv: Int? = null,
        pvPlies: Int? = null,
    ): List<Line> {
        // Over, a claimable draw included, as python-chess's `is_game_over(claim_draw=True)`:
        // with the game's moves on the board, a threefold repetition can be claimed.
        if (board.isGameOver(claimDraw = true)) return emptyList()
        if (depth == null && timeS == null) throw IllegalArgumentException("analyse needs a depth or a time limit")
        var capped = timeS
        val cap = maxSeconds
        if (depth != null && cap != null) capped = if (timeS == null) cap else minOf(timeS, cap)
        var wanted = multipv ?: this.multipv
        wanted = maxOf(1, minOf(wanted, board.legalMoves().size))
        val infos = lock.withLock { analyseWithRestart(board, depth, capped, wanted) }
        val plies = pvPlies ?: this.pvPlies
        return infos.map { line(it, plies) }.filter { it.moves.isNotEmpty() }
    }

    private fun analyseWithRestart(
        board: Board,
        depth: Int?,
        timeS: Double?,
        multipv: Int,
    ): List<Info> {
        for (attempt in 1..2) {
            open()
            try {
                return search(board, depth, timeS, multipv)
            } catch (exc: IOException) {
                close()
                if (attempt == 2) throw EngineUnavailable("engine died twice on ${board.fen()}: $exc")
            }
        }
        throw AssertionError("unreachable")
    }

    /** One `info` line's numbers, kept per MultiPV index at the last depth reached. */
    private class Info(
        val depth: Int,
        val multipv: Int,
        val scoreCp: Int?,
        val mate: Int?,
        val wdl: Triple<Int, Int, Int>?,
        val pv: List<String>,
    )

    private fun search(board: Board, depth: Int?, timeS: Double?, multipv: Int): List<Info> {
        if (multipv != currentMultipv) {
            setOption("MultiPV", multipv.toString())
            currentMultipv = multipv
        }
        send("position fen ${board.fen()}")
        val go = StringBuilder("go")
        if (depth != null) go.append(" depth ").append(depth)
        // The time limit is the desktop's `movetime`, kept here ([watch]) so that a pause
        // does not count against it.
        send(go.toString())
        if (timeS != null) arm((timeS * 1000).toLong())
        val best = HashMap<Int, Info>()
        try {
            readUntil("bestmove") { line ->
                if (line.startsWith("info ") && " pv " in " $line ") {
                    parseInfo(line)?.let { info ->
                        val kept = best[info.multipv]
                        // The last word on each line wins, which at a fixed depth is the
                        // deepest: python-chess keeps the same.
                        if (kept == null || info.depth >= kept.depth) best[info.multipv] = info
                    }
                }
            }
        } finally {
            if (timeS != null) disarm()
        }
        return best.entries.sortedBy { it.key }.map { it.value }
    }

    private fun parseInfo(line: String): Info? {
        val words = line.split(" ")
        var depth = 0
        var multipv = 1
        var scoreCp: Int? = null
        var mate: Int? = null
        var wdl: Triple<Int, Int, Int>? = null
        var pv: List<String> = emptyList()
        var i = 1
        while (i < words.size) {
            when (words[i]) {
                "depth" -> { depth = words.getOrNull(i + 1)?.toIntOrNull() ?: 0; i += 2 }
                "multipv" -> { multipv = words.getOrNull(i + 1)?.toIntOrNull() ?: 1; i += 2 }
                "score" -> {
                    when (words.getOrNull(i + 1)) {
                        "cp" -> { scoreCp = words.getOrNull(i + 2)?.toIntOrNull(); i += 3 }
                        "mate" -> { mate = words.getOrNull(i + 2)?.toIntOrNull(); i += 3 }
                        else -> i += 2
                    }
                }
                "wdl" -> {
                    val w = words.getOrNull(i + 1)?.toIntOrNull()
                    val d = words.getOrNull(i + 2)?.toIntOrNull()
                    val l = words.getOrNull(i + 3)?.toIntOrNull()
                    if (w != null && d != null && l != null) wdl = Triple(w, d, l)
                    i += 4
                }
                "pv" -> { pv = words.drop(i + 1); i = words.size }
                else -> i += 1
            }
        }
        if (pv.isEmpty()) return null
        return Info(depth, multipv, scoreCp, mate, wdl, pv)
    }

    private fun line(info: Info, pvPlies: Int): Line = Line(
        moves = info.pv.take(pvPlies),
        depth = info.depth,
        expectedScore = expectedScore(scoreModel, info.scoreCp, info.mate, info.wdl),
        scoreCp = info.scoreCp,
        mate = info.mate,
        wdl = info.wdl,
    )

    // The protocol

    private fun setOption(name: String, value: String) {
        send("setoption name $name value $value")
    }

    private fun send(command: String) {
        val out = writer ?: throw IOException("the engine is not running")
        out.write(command)
        out.write("\n")
        out.flush()
    }

    private inline fun readUntil(token: String, body: (String) -> Unit) {
        val input = reader ?: throw IOException("the engine is not running")
        while (true) {
            val line = input.readLine() ?: throw IOException("the engine closed its output")
            if (line.startsWith(token)) return
            body(line)
        }
    }

    companion object {
        private const val TAG = "sidecoach.engine"

        /** A principal variation can run thirty plies; nobody reads that far. */
        const val DEFAULT_PV_PLIES = 12
        const val DEFAULT_SCORE_MODEL = "human"

        /** The engine as the packager leaves it: an executable in the library directory. */
        fun binaryIn(nativeLibraryDir: String): File = File(nativeLibraryDir, "libstockfish.so")

        /** Linux's stop and continue signals ([pause], [resume]); `android.os.Process` names neither. */
        private const val SIGSTOP = 19
        private const val SIGCONT = 18

        /**
         * A process's id. Android's `Process` has no `pid()` on every release this app runs
         * on; the field behind it has been called `pid` since the beginning.
         */
        private fun processId(process: Process): Int? = runCatching {
            val field = process.javaClass.getDeclaredField("pid")
            field.isAccessible = true
            field.getInt(process)
        }.getOrNull()
    }
}

/**
 * A line of UCI moves as a person reads it: `24. Nxe5 Bxe5 25. Qh5`; without `numbered`, the
 * bare moves. Anything the engine cannot play from this position stops the line.
 */
fun sanLine(board: Board, moves: List<String>, limit: Int? = null, numbered: Boolean = true): String {
    val replay = board.copy(withStack = false)
    val parts = ArrayList<String>()
    for ((i, uci) in moves.take(limit ?: moves.size).withIndex()) {
        val move = runCatching { Move.fromUci(uci) }.getOrNull() ?: break
        if (!replay.isLegal(move)) break
        if (numbered && replay.turn == WHITE) {
            parts.add("${replay.fullmoveNumber}.")
        } else if (numbered && i == 0) {
            parts.add("${replay.fullmoveNumber}...")
        }
        parts.add(replay.san(move))
        replay.push(move)
    }
    return parts.joinToString(" ")
}
