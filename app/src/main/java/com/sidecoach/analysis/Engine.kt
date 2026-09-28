package com.sidecoach.analysis

import android.os.SystemClock
import android.util.Log
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
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

/** A search inside [Engine.interruptible] was stopped by [Engine.interrupt]. */
class Interrupted : RuntimeException()

/**
 * Stockfish over UCI: one persistent process, MultiPV lines, real win-draw-loss numbers.
 * A port of `sidecoach/analysis/engine.py`.
 *
 * The desktop drives Stockfish through python-chess; there is no such library here, so the
 * protocol is spoken directly, as python-chess speaks it: the options sent only when they
 * change, a new game before the first search, and the position as the game's moves from where
 * they began, so the engine sees the repetitions in them (`EngineParityTest` holds the phone to
 * the desktop's transcript). Everything else is the desktop's: the same restart-once rule, the
 * same line for each MultiPV index at the last depth reached, and scores from the mover's point
 * of view in expected score, never centipawns.
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

    /**
     * Each option's value as the engine holds it: the default it announced, then what was
     * set. python-chess's `config`, which `_setoption` holds a value against and sends only
     * when it differs ([configure]).
     */
    private val config = HashMap<String, String>()

    /**
     * No search asked of this process yet: python-chess's `first_game`, on which its first
     * analysis tells the engine a new game begins (`ucinewgame`) and waits for it to be ready.
     */
    private var firstGame = true

    /**
     * Inside [interruptible]: the search running, which [interrupt] stops, and whether it has.
     * Readable by a scripted engine, as the desktop's tests read `_interruptible` and `_stopped`.
     */
    @Volatile protected var interruptibleNow = false
        private set
    @Volatile protected var stopped = false
        private set

    /**
     * A search inside [interruptible] is between its `go` and its `bestmove`, so a `stop` from
     * [interrupt] reaches that search and no other: python-chess's `_search` and its `stop`,
     * which is sent only while the search runs. Guarded by [stopping].
     */
    private var searching = false
    private val stopping = Any()

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

    /** The engine's process, started: the executable in the library directory. */
    protected open fun launch(): Process =
        ProcessBuilder(binary.absolutePath)
            .redirectErrorStream(true)
            .directory(binary.parentFile)
            .start()

    fun open() {
        if (process != null) return
        if (!binary.exists()) throw EngineUnavailable("no engine at $binary")
        val started = try {
            launch()
        } catch (exc: IOException) {
            throw EngineUnavailable("cannot start $binary: $exc")
        }
        process = started
        enginePid = processId(started)
        writer = started.outputStream.bufferedWriter()
        reader = BufferedReader(InputStreamReader(started.inputStream))
        options.clear()
        config.clear()
        firstGame = true
        send("uci")
        readUntil("uciok") { line ->
            if (line.startsWith("option name ")) {
                val option = parseOption(line)
                options.add(option.first)
                option.second?.let { config[option.first] = it }
            } else if (line.startsWith("id name ")) {
                name = line.removePrefix("id name ").trim()
            }
        }
        // The network first: without it the engine answers `isready` and then dies on the
        // first search.
        if (net != null && "EvalFile" in options) {
            setOption("EvalFile", net.absolutePath)
        }
        if ("Threads" in options) configure("Threads", threads.toString())
        if ("Hash" in options) configure("Hash", hashMb.toString())
        // Real win-draw-loss numbers beat a model fitted to centipawns; ask for them when the
        // engine has them, and fall back to python-chess's model when it does not ([line]).
        if ("UCI_ShowWDL" in options) configure("UCI_ShowWDL", "true")
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
            if ("Threads" in options) configure("Threads", this.threads.toString())
            if ("Hash" in options) configure("Hash", this.hashMb.toString())
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
     *
     * `carryBest`: a best line shorter than [MIN_BEST_PLIES] is carried on to it -- for the
     * coach, who reads the line for its point; a verdict needs only the score.
     * `onDepth`: handed each depth's lines as the search finishes it, the search going on.
     */
    open fun analyse(
        board: Board,
        depth: Int? = null,
        timeS: Double? = null,
        multipv: Int? = null,
        pvPlies: Int? = null,
        carryBest: Boolean = false,
        onDepth: ((List<Line>) -> Unit)? = null,
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
        val plies = pvPlies ?: this.pvPlies

        fun asLines(infos: List<Info>, carry: Boolean): MutableList<Line> = infos
            .mapIndexed { i, info -> line(info, board, if (carry && i == 0) maxOf(plies, MIN_BEST_PLIES) else plies) }
            .filter { it.moves.isNotEmpty() }
            .toMutableList()

        val depths: ((List<Info>) -> Unit)? = if (onDepth == null) null else { infos -> onDepth(asLines(infos, false)) }
        val infos = lock.withLock { analyseWithRestart(board, depth, capped, wanted, depths) }
        val lines = asLines(infos, carryBest)
        if (carryBest && lines.isNotEmpty() && lines[0].moves.size < MIN_BEST_PLIES) {
            lines[0] = carriedOn(board, lines[0], MIN_BEST_PLIES)
        }
        return lines
    }

    /**
     * `line` carried on by searches from its end until it is `plies` long or the game is over
     * there; its score and depth stay the first search's.
     */
    private fun carriedOn(board: Board, line: Line, plies: Int): Line {
        val moves = ArrayList(line.moves)
        val end = board.copy(withStack = false)
        for (uci in moves) end.push(Move.fromUci(uci))
        repeat(EXTEND_TRIES) {
            if (moves.size >= plies || end.isGameOver(claimDraw = true)) return line.copy(moves = moves)
            val infos = lock.withLock { analyseWithRestart(end, EXTEND_DEPTH, EXTEND_SECONDS, 1) }
            val more = infos.firstOrNull()?.pv ?: emptyList()
            if (more.isEmpty()) return line.copy(moves = moves)
            for (uci in more.take(plies - moves.size)) {
                moves.add(uci)
                end.push(Move.fromUci(uci))
            }
        }
        return line.copy(moves = moves)
    }

    /**
     * Searches that another thread may stop ([interrupt]): each throws [Interrupted] once
     * stopped, and so does every search after it here -- a look ahead the game has overtaken is
     * let go at once, not searched to its end first.
     */
    fun <T> interruptible(body: () -> T): T {
        stopped = false
        interruptibleNow = true
        try {
            return body()
        } finally {
            interruptibleNow = false
        }
    }

    /**
     * Stop the search running inside [interruptible], and the ones after it there. Safe from
     * any thread, and while the engine is [pause]d: the `stop` waits in the pipe and ends the
     * search the moment the engine runs on.
     */
    fun interrupt() {
        synchronized(stopping) {
            stopped = true
            if (searching) runCatching { send("stop") }  // it may have ended just now
        }
    }

    private fun analyseWithRestart(
        board: Board,
        depth: Int?,
        timeS: Double?,
        multipv: Int,
        onDepth: ((List<Info>) -> Unit)? = null,
    ): List<Info> {
        for (attempt in 1..2) {
            open()
            try {
                return search(board, depth, timeS, multipv, onDepth)
            } catch (exc: IOException) {
                close()
                if (attempt == 2) throw EngineUnavailable("engine died twice on ${board.fen()}: $exc")
            }
        }
        throw AssertionError("unreachable")
    }

    /**
     * The numbers of one `info` line, or of one MultiPV index once its lines are merged as
     * python-chess merges them (`AnalysisResult.post`): each line's fields update the index's,
     * the last word on each field wins, and a field a line does not carry stays. A line that
     * names no index is the first's -- so `info depth 21 currmove e2e4 currmovenumber 1`, which
     * Stockfish sends once a search has run three seconds, moves the best line's depth on while
     * its moves stay the last depth's, as the desktop reports it. Null is a field not given.
     */
    private class Info {
        var depth: Int? = null
        var multipv: Int? = null

        /** A score was given: [scoreCp] or [mate], the other null (python-chess's one `score`). */
        var score = false
        var scoreCp: Int? = null
        var mate: Int? = null
        var wdl: Triple<Int, Int, Int>? = null
        var pv: List<String>? = null
    }

    /**
     * `analyse`'s search, read as it goes (`_streamed`): each depth handed to [onDepth] once its
     * last line is in (Stockfish reports a depth's lines best first), and, inside
     * [interruptible], stopped by [interrupt].
     */
    private fun search(
        board: Board,
        depth: Int?,
        timeS: Double?,
        multipv: Int,
        onDepth: ((List<Info>) -> Unit)? = null,
    ): List<Info> {
        val stoppable = interruptibleNow
        if (stoppable && stopped) throw Interrupted()
        // What python-chess's analysis sends before the search: pondering off and analysis
        // mode on where the engine has them, the number of lines, and -- on the process's
        // first search -- a new game, waited on until the engine is ready.
        if ("Ponder" in options) configure("Ponder", "false")
        if ("UCI_AnalyseMode" in options) configure("UCI_AnalyseMode", "true")
        if ("MultiPV" in options || multipv > 1) configure("MultiPV", multipv.toString())
        if (firstGame) {
            send("ucinewgame")
            firstGame = false
            send("isready")
            readUntil("readyok") {}
        }
        position(board)
        val go = StringBuilder("go")
        if (depth != null) go.append(" depth ").append(depth)
        // The time limit is the desktop's `movetime`, kept here ([watch]) so that a pause
        // does not count against it.
        send(go.toString())
        if (timeS != null) arm((timeS * 1000).toLong())
        if (stoppable) {
            synchronized(stopping) {
                searching = true
                if (stopped) runCatching { send("stop") }  // interrupted before the search was in reach
            }
        }
        // python-chess's `AnalysisResult.multipv`: one merged record per index, the first there
        // from the start; its moves read against the position sent (`self.engine.board`).
        val root = board.copy(withStack = false)
        val infos = arrayListOf(Info())
        // Each index's last line as it came, not merged (python-chess yields the line itself),
        // and the deepest depth handed on.
        val latest = java.util.TreeMap<Int, Info>()
        var done = 0
        try {
            readUntil("bestmove") { line ->
                val words = words(line)
                if (words.firstOrNull() == "info") {
                    val info = parseInfo(words, root)
                    post(infos, info)
                    val rank = info.multipv ?: 1
                    val at = info.depth
                    if (at != null && !info.pv.isNullOrEmpty()) {
                        latest[rank] = info
                        if (onDepth != null && rank == multipv && at > done) {
                            done = at
                            onDepth(latest.values.toList())
                        }
                    }
                }
            }
        } finally {
            if (stoppable) synchronized(stopping) { searching = false }
            if (timeS != null) disarm()
        }
        if (stoppable && stopped) throw Interrupted()
        return infos
    }

    /**
     * `AnalysisResult.post`: the line's fields into its index's record, the records grown to
     * the index. Python's negative index for a MultiPV of 0 or less counts from the end.
     */
    private fun post(infos: MutableList<Info>, info: Info) {
        val wanted = info.multipv ?: 1
        while (infos.size < wanted) infos.add(Info())
        val at = if (wanted >= 1) wanted - 1 else infos.size + wanted - 1
        if (at !in infos.indices) return
        val into = infos[at]
        info.depth?.let { into.depth = it }
        info.multipv?.let { into.multipv = it }
        if (info.score) {
            into.score = true
            into.scoreCp = info.scoreCp
            into.mate = info.mate
        }
        info.wdl?.let { into.wdl = it }
        info.pv?.let { into.pv = it }
    }

    /**
     * python-chess's `_parse_uci_info`, for the fields the desktop reads (depth, the MultiPV
     * index, the score, win-draw-loss, the moves), every other token taken as it takes it.
     * A field that does not parse is not given, and the words it took stay taken; a line of
     * moves stops at the first word that is no move, and one with an illegal move in it is no
     * line at all (`_create_variation_line`, `push_uci`), the words after `pv` read again.
     */
    private fun parseInfo(words: List<String>, root: Board): Info {
        val info = Info()
        var i = 1
        fun next(): String = if (i < words.size) words[i++] else ""
        while (i < words.size) {
            when (next()) {
                "string" -> i = words.size
                "depth" -> next().toIntOrNull()?.let { info.depth = it }
                "multipv" -> next().toIntOrNull()?.let { info.multipv = it }
                "seldepth", "nodes", "currmovenumber", "hashfull", "nps", "tbhits", "cpuload",
                "time", "ebf", "currmove" -> next()
                "score" -> {
                    val kind = next()
                    val value = next()
                    if (i < words.size && (words[i] == "lowerbound" || words[i] == "upperbound")) i++
                    val number = value.toIntOrNull()
                    if (number != null && (kind == "cp" || kind == "mate")) {
                        info.score = true
                        info.scoreCp = if (kind == "cp") number else null
                        info.mate = if (kind == "mate") number else null
                    }
                }
                "currline" -> if (next().toIntOrNull() != null) {
                    variation(root, words, i)?.let { i = it.second }
                }
                "refutation" -> {
                    val refuted = root.copy(withStack = false)
                    val pushed = runCatching { refuted.push(refuted.parseUci(next())) }.isSuccess
                    if (pushed) variation(refuted, words, i)?.let { i = it.second }
                }
                "pv" -> variation(root, words, i)?.let { (moves, after) ->
                    info.pv = moves
                    i = after
                }
                "wdl" -> {
                    val wins = next().toIntOrNull()
                    val draws = next().toIntOrNull()
                    val losses = next().toIntOrNull()
                    if (wins != null && draws != null && losses != null) info.wdl = Triple(wins, draws, losses)
                }
                else -> Unit
            }
        }
        return info
    }

    /**
     * `engine._line`: a line with no score has no moves either, and is dropped; the numbers
     * come from the engine when it gives win-draw-loss, else from python-chess's model of
     * them ([WDL_MODEL]) at the board's ply.
     */
    private fun line(info: Info, board: Board, pvPlies: Int): Line {
        if (!info.score) return Line(moves = emptyList(), depth = 0, expectedScore = 0.5)
        val wdl = info.wdl ?: modelWdl(info.scoreCp, info.mate, board.ply())
        return Line(
            moves = (info.pv ?: emptyList()).take(pvPlies),
            depth = info.depth ?: 0,
            expectedScore = expectedScore(scoreModel, info.scoreCp, info.mate, wdl),
            scoreCp = info.scoreCp,
            mate = info.mate,
            wdl = wdl,
        )
    }

    // The protocol

    /**
     * python-chess's `_position`: the game's moves from where they began (`startpos` when that
     * is the standard start), so the engine sees the repetitions they hold; the position
     * alone, en passant square and all, when the history holds a null move.
     */
    private fun position(board: Board) {
        // Standard chess, as python-chess selects it when the engine offers the choice.
        if ("UCI_Variant" in options) configure("UCI_Variant", "chess")
        if ("UCI_Chess960" in options) configure("UCI_Chess960", "false")
        send(positionCommand(board))
    }

    private fun setOption(name: String, value: String) {
        send("setoption name $name value $value")
    }

    /** python-chess's `_setoption`: sent only when the engine does not already hold it. */
    private fun configure(name: String, value: String) {
        if (config[name] == value) return
        setOption(name, value)
        config[name] = value
    }

    private fun send(command: String) {
        val out = writer ?: throw IOException("the engine is not running")
        // Whole, under the writer's own lock: the `stop` of [interrupt] and of [watch] can come
        // from two threads at once, and two halves of one line are no command.
        synchronized(out) {
            out.write(command)
            out.write("\n")
            out.flush()
        }
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

        /**
         * The best line is read for what it is for -- the mate, the piece won, the pawn queened
         * -- and a puzzle's point can come seven moves in (a queen won on ply 11, a pawn
         * queening on ply 15, in a sample of Lichess puzzles). Stockfish's own line often stops
         * short (a hash hit cuts it to three plies), so the best line is kept to this length,
         * and a shorter one carried on by searching from its end, whatever the other lines are
         * cut to (Tai, 2026-09-23: "at least 8 to 10 ply since some puzzles and hard positions
         * are really deep").
         */
        const val MIN_BEST_PLIES = 16

        /** How deep, and for how long at most, each search that carries the line on looks. */
        const val EXTEND_DEPTH = 14
        const val EXTEND_SECONDS = 0.5

        /** How many searches may carry one line on. */
        const val EXTEND_TRIES = 3

        /** python-chess's model for turning centipawns into win-draw-loss when an engine will not. */
        const val WDL_MODEL = "sf"
        const val DEFAULT_SCORE_MODEL = "human"

        /**
         * The `position` command python-chess sends for [board] (`UciProtocol._position`):
         * `startpos` when the moves began at the standard start, else the FEN they began
         * from, then the moves -- the whole stack, so the engine sees the repetitions in it.
         * A history holding a null move cannot be sent as moves, so the board's own FEN goes
         * alone. The FEN names the en passant square whenever there is one
         * (`en_passant="fen"`), legal capture or not.
         */
        fun positionCommand(board: Board): String {
            val builder = StringBuilder("position")
            val safeHistory = board.moveStack.none { it.isNull }
            val root = if (safeHistory) board.root() else board
            val fen = root.fen(enPassant = "fen")
            if (fen == STARTING_FEN) {
                builder.append(" startpos")
            } else {
                builder.append(" fen ").append(fen)
            }
            if (!safeHistory) {
                Log.w(TAG, "Not transmitting history with null moves to UCI engine")
            } else if (board.moveStack.isNotEmpty()) {
                builder.append(" moves")
                for (move in board.moveStack) builder.append(' ').append(move.uci())
            }
            return builder.toString()
        }

        /**
         * `Score.wdl(model="sf", ply=ply)`, what the desktop falls back to when the engine
         * gives no win-draw-loss of its own: python-chess's fit of Stockfish 16.1 for a
         * centipawn score, and a mate a win or a loss outright.
         */
        fun modelWdl(scoreCp: Int?, mate: Int?, ply: Int): Triple<Int, Int, Int>? {
            if (mate != null) return if (mate > 0) Triple(1000, 0, 0) else Triple(0, 0, 1000)
            val cp = scoreCp ?: return null
            val wins = sf161Wins(cp, ply)
            val losses = sf161Wins(-cp, ply)
            return Triple(wins, 1000 - wins - losses, losses)
        }

        /** python-chess's `_sf16_1_wins`. */
        private fun sf161Wins(cp: Int, ply: Int): Int {
            // https://github.com/official-stockfish/Stockfish/blob/sf_16.1/src/uci.cpp#L48
            val normalizeToPawnValue = 356
            // https://github.com/official-stockfish/Stockfish/blob/sf_16.1/src/uci.cpp#L383-L384
            val m = minOf(120.0, maxOf(8.0, ply / 2.0 + 1)) / 32
            val a = (((-1.06249702 * m + 7.42016937) * m + 0.89425629) * m) + 348.60356174
            val b = (((-5.33122190 * m + 39.57831533) * m + -90.84473771) * m) + 123.40620748
            val x = minOf(4000.0, maxOf(cp * normalizeToPawnValue / 100.0, -4000.0))
            return (0.5 + 1000 / (1 + Math.exp((a - x) / b))).toInt()
        }

        /** python-chess's `UCI_REGEX`, matched at a word's start as `re.match` matches it. */
        private val UCI_REGEX = Regex("[a-h][1-8][a-h][1-8][pnbrqk]?|[PNBRQK]@[a-h][1-8]|0000\\z")

        /** `str.split()`: the words of an engine's line, runs of whitespace between them. */
        private fun words(line: String): List<String> {
            val out = ArrayList<String>()
            var start = -1
            for (i in line.indices) {
                if (line[i].isWhitespace()) {
                    if (start >= 0) out.add(line.substring(start, i))
                    start = -1
                } else if (start < 0) {
                    start = i
                }
            }
            if (start >= 0) out.add(line.substring(start))
            return out
        }

        /**
         * `_create_variation_line`: the moves from word [start] on, played out from [board],
         * up to the first word that is no move, and the index of that word; null when a move
         * cannot be played there (python-chess's `push_uci` raising).
         */
        private fun variation(board: Board, words: List<String>, start: Int): Pair<List<String>, Int>? {
            val walk = board.copy(withStack = false)
            val moves = ArrayList<String>()
            var at = start
            while (at < words.size && UCI_REGEX.matchesAt(words[at], 0)) {
                val move = try {
                    walk.parseUci(words[at])
                } catch (exc: IllegalArgumentException) {
                    return null
                }
                walk.push(move)
                moves.add(move.uci())
                at += 1
            }
            return moves to at
        }

        private val OPTION_KEYWORDS = Regex("\\s*\\b(name|type|default|min|max|var)\\b\\s*")

        /**
         * An `option` line's name and its default as python-chess holds it (`Option.parse`):
         * a check option true or false, a spin option its number, a button none. The line is
         * cut at the keywords as python-chess cuts it, so a name may hold spaces.
         */
        private fun parseOption(line: String): Pair<String, String?> {
            val arg = line.removePrefix("option").trim()
            val tokens = ArrayList<String>()
            var last = 0
            for (match in OPTION_KEYWORDS.findAll(arg)) {
                tokens.add(arg.substring(last, match.range.first))
                tokens.add(match.groupValues[1])
                last = match.range.last + 1
            }
            tokens.add(arg.substring(last))
            val parts = linkedMapOf("name" to "", "type" to "", "default" to "", "min" to "", "max" to "")
            var current: String? = null
            for (token in tokens) {
                if (token == "var" || (token in parts && parts[token] == "")) {
                    current = token
                } else if (current == "var") {
                    continue
                } else if (current != null) {
                    parts[current] = token
                }
            }
            val default = parts.getValue("default")
            val value = when (parts.getValue("type")) {
                "check" -> if (default.isNotEmpty() && default != "false") "true" else "false"
                "spin" -> default.toIntOrNull()?.toString()
                "button", "reset", "save" -> null
                else -> default
            }
            return parts.getValue("name") to value
        }

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
