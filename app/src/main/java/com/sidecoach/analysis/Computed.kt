package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import kotlin.math.abs
import kotlin.math.max

/**
 * What the engine has computed this game, for whoever reads it. A port of
 * `sidecoach/analysis/computed.py`.
 *
 * The analyser is the producer: it measures every position of the game, the one the player
 * faces early and then deeper, and the positions ahead of the game while it would otherwise
 * idle. Each depth it finishes goes in here, and a reader takes the newest word on a position at
 * any time (Tai, 2026-09-28: "stockfish is a reactive producer ... everything else just consumes
 * it"). The coach's engine is such a reader ([ComputedFirst]): it searches only a position that
 * is not here, and puts what it found here too (Tai, 2026-09-29: "our harness must first look at
 * what was already computed, then if that's not computed only then invoke a call to stockfish to
 * ask for analysis over that specific line").
 *
 * A move on a line the model was shown is read from that line, before any position is looked up
 * ([restOfLine], the coach's own step): a separate search of the position after it -- the
 * look-ahead's, deeper or not -- can answer differently, and a second search after 64. Bd2
 * answered it with Kh6 where the lines had Rxh3, and the model, shown both, said Bd2 avoided
 * Rxh3 (the dev audit, 2026-09-26). Only positions measured are kept here: the lines passing
 * through a position went stale as the search deepened, a move refuted at depth 18 still read
 * off its line at depth 12 (the code review, 2026-09-29).
 *
 * The analyser's thread and the coach's both write; a lock keeps each write whole.
 */

/**
 * The position as a key: the pieces, the side to move, the castling rights and the en passant
 * square -- the FEN without its clocks, which a board read off the screen guesses.
 */
fun positionKey(board: Board): String = board.fen().split(" ").take(4).joinToString(" ")

/**
 * The rest of [line] once its first [plies] are played, as a search from there would give it:
 * from the side to move there, a mate counted in the moves still to come, and the depth what
 * the search had left.
 */
fun lineAfter(line: Line, plies: Int): Line {
    val turned = plies % 2 == 1
    var mate: Int? = null
    val m = line.mate
    if (m != null && m != 0) {
        // Whose mate it is, and how many of the mating side's moves the plies played; a mate
        // past its count (a line carried on beyond it) is no mate this line knows.
        val moverMates = m > 0
        val played = if (moverMates) (plies + 1) / 2 else plies / 2
        val left = abs(m) - played
        if (left > 0) mate = if (moverMates != turned) left else -left
    }
    val cp = line.scoreCp
    val wdl = line.wdl
    return Line(
        moves = line.moves.subList(plies, line.moves.size).toList(),
        depth = max(1, line.depth - plies),
        expectedScore = if (turned) 1.0 - line.expectedScore else line.expectedScore,
        scoreCp = if (turned && cp != null) -cp else cp,
        mate = mate,
        wdl = if (turned && wdl != null) Triple(wdl.third, wdl.second, wdl.first) else wdl,
    )
}

/**
 * The rest of the first of [lines] that begins with [moves] (UCI) and goes on past them, from
 * the side to move after them; null when none does.
 */
fun restOfLine(lines: List<Line>, moves: List<String>): Line? {
    val n = moves.size
    if (n == 0) return null
    for (line in lines) {
        if (line.moves.size > n && line.moves.subList(0, n) == moves) return lineAfter(line, n)
    }
    return null
}

/** The engine's lines of each position measured. */
class Computed {
    private val lock = Any()

    /** Measured whole, the newest depth: the lines a coach's sheet shows. */
    private val whole = HashMap<String, List<Line>>()

    /** Measured for a verdict only: the best line alone, at the full depth. */
    private val thin = HashMap<String, List<Line>>()

    /** A new game: none of this is about it. */
    fun clear() {
        synchronized(lock) {
            whole.clear()
            thin.clear()
        }
    }

    /**
     * The lines of [board], as a search just gave them. A shallower search than the one here
     * already is left out: the deeper word stands.
     */
    fun add(board: Board, lines: List<Line>, thin: Boolean = false) {
        if (lines.isEmpty()) return
        val key = positionKey(board)
        synchronized(lock) {
            val store = if (thin) this.thin else whole
            val here = store[key]
            if (here == null || here[0].depth <= lines[0].depth) store[key] = lines.toList()
        }
    }

    /**
     * What is known of [board]: its lines measured whole, else its verdict's line; none when it
     * was not measured. Measured whole first, as the sheet shows it.
     */
    fun lines(board: Board): List<Line> {
        val key = positionKey(board)
        synchronized(lock) {
            return (whole[key]?.takeIf { it.isNotEmpty() } ?: thin[key] ?: emptyList()).toList()
        }
    }
}

/**
 * The coach's engine: a position computed read ([Computed]), a search for the rest. The engine
 * starts at its first search ([Engine.analyse] opens it), so a coach whose questions the analyser
 * has answered never starts a second Stockfish; closed, it searches no more -- a question still
 * being answered as the coach closes would start a process nobody closes. [search] is the
 * engine's search (the desktop's `Searcher`), and [onClose] its close.
 */
class ComputedFirst(
    private val search: (Board, Int?) -> List<Line>,
    val computed: Computed = Computed(),
    private val onClose: () -> Unit = {},
) : AutoCloseable {
    constructor(engine: Engine, computed: Computed = Computed()) :
        this({ board, depth -> engine.analyse(board, depth = depth) }, computed, engine::close)

    /** Positions searched, not found computed. */
    var searches = 0
        private set

    @Volatile
    var closed = false
        private set

    fun analyse(board: Board, depth: Int?): List<Line> {
        val known = computed.lines(board)
        if (known.isNotEmpty() || closed) return known
        searches += 1
        val lines = search(board, depth)
        computed.add(board, lines)
        return lines
    }

    override fun close() {
        closed = true
        onClose()
    }
}
