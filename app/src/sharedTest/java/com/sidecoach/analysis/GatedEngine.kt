package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import java.io.File
import java.util.Collections
import java.util.concurrent.CountDownLatch

/**
 * What a quiet position gets from a fake engine given nothing to say for it: a level line, its
 * move the position's first legal one ([ScriptEngine.analyse]; the desktop's `QUIET`).
 */
internal val QUIET = listOf(Line(moves = listOf("a2a3"), depth = 1, expectedScore = 0.5))

/**
 * An engine that answers from a script (the desktop's `FakeEngine`, tests/fakes.py): the lines
 * for a position's FEN, else [default], as many as asked for, and nothing for a finished game.
 * It remembers every position it was asked about, with the moves that led there, the depth and
 * the number of lines asked for, and never starts a process.
 */
internal open class ScriptEngine(
    private val answers: Map<String, List<Line>> = emptyMap(),
    private val default: List<Line> = QUIET,
) : Engine(File("fake"), null) {
    val calls: MutableList<String> = Collections.synchronizedList(ArrayList())
    val histories: MutableList<List<String>> = Collections.synchronizedList(ArrayList())
    val depths: MutableList<Int?> = Collections.synchronizedList(ArrayList())
    val multipvs: MutableList<Int> = Collections.synchronizedList(ArrayList())

    override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
        synchronized(calls) {
            calls.add(board.fen())
            histories.add(board.moveStack.map { it.uci() })
            depths.add(depth)
            multipvs.add(multipv ?: 1)
        }
        if (gameOver(board)) return emptyList()
        var lines = answers[board.fen()] ?: default
        if (lines === QUIET) {
            // A line the position can play: a search of the position after its move (a
            // candidate's own) reads on from it.
            val first = board.legalMoves().minBy { it.uci() }
            lines = listOf(QUIET[0].copy(moves = listOf(first.uci())))
        }
        return if (multipv != null) lines.take(multipv) else lines
    }

    /** The searches asked of it, with how many lines each: (FEN, lines), in order. */
    fun asked(): List<Pair<String, Int>> = synchronized(calls) { calls.zip(multipvs) }

    /** Never starts a process. */
    override fun open() = Unit

    override fun close() = Unit
}

/**
 * A fake engine whose searches of the positions in [gated] (FENs) wait until the test lets them
 * go ([release]) or the analyser stops them ([interrupt], throwing [Interrupted], as a real one
 * stops); every other search, and every depth it finishes (onDepth, from [EARLY_DEPTH] to the
 * depth asked for), is answered at once. [entered] counts down when a gated search begins. So a
 * test acts mid-search by the engine's state, never by a clock (the desktop's `GatedEngine`).
 */
internal class GatedEngine(
    answers: Map<String, List<Line>> = emptyMap(),
    gated: Collection<String> = emptyList(),
    default: List<Line> = QUIET,
) : ScriptEngine(answers, default) {
    private val gated = gated.toSet()
    val entered: Map<String, CountDownLatch> = gated.associateWith { CountDownLatch(1) }
    private val released: MutableSet<String> = Collections.synchronizedSet(HashSet())

    /** Told when a gated search begins or is stopped: a test's watch woken for it ([AnalyserWatch.poke]). */
    @Volatile var poke: (() -> Unit)? = null

    /** The gated searches the analyser stopped. */
    val halted: MutableList<String> = Collections.synchronizedList(ArrayList())
    private val gate = Object()

    fun release(fen: String) {
        synchronized(gate) {
            released.add(fen)
            gate.notifyAll()
        }
    }

    override fun interrupt() {
        super.interrupt()
        synchronized(gate) { gate.notifyAll() }
    }

    override fun analyse(board: Board, depth: Int?, timeS: Double?, multipv: Int?, pvPlies: Int?, carryBest: Boolean, onDepth: ((List<Line>) -> Unit)?): List<Line> {
        val fen = board.fen()
        if (fen in gated) {
            entered.getValue(fen).countDown()
            poke?.invoke()
            synchronized(gate) {
                val deadline = System.nanoTime() + HANG_MS * 1_000_000
                while (fen !in released) {
                    if (interruptibleNow && stopped) {
                        halted.add(fen)
                        poke?.invoke()
                        throw Interrupted()
                    }
                    val left = (deadline - System.nanoTime()) / 1_000_000
                    // An exception, not an Error: on a worker's thread an Error ends the test process.
                    if (left <= 0) throw IllegalStateException("the search of $fen was never let go")
                    gate.wait(left)
                }
            }
        }
        var lines = super.analyse(board, depth, timeS, multipv, pvPlies, carryBest, onDepth)
        if (depth != null && depth > 0) lines = lines.map { it.copy(depth = depth) }
        if (onDepth != null && lines.isNotEmpty() && depth != null) {
            for (d in EARLY_DEPTH until depth) onDepth(lines.map { it.copy(depth = d) })
        }
        return lines
    }
}
