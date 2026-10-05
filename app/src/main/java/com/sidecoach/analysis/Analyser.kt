package com.sidecoach.analysis

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.events.Event
import com.sidecoach.events.Line
import java.util.PriorityQueue
import java.util.TreeMap
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow

/**
 * The engine service for a game followed live: every position searched once, on a pool of
 * single-thread Stockfish workers, and nothing searched that will not be read. A port of
 * `sidecoach/analysis/analyser.py`.
 *
 * The engine audit of 2026-10-04 (a replay of a 38-minute session) found the one-engine analyser
 * searching 258 of 624 positions two to six times, throwing away 211 s of searching by
 * interrupting it, and searching again from depth 1 the very positions its look-ahead had been
 * searching when the game reached them; and four threads costing 7.7 times the processor time of
 * one for the same depth-18 answer (Tai, 2026-10-04: "a lot of work is duplicated, interrupted,
 * and some are just discarded ... the code is not very smart about all of it").
 *
 * So, here:
 *
 * - A position is a node, by its cache key (the FEN and the moves a repetition could reach back
 *   over). Its search is the canonical one -- the full depth, every line, one thread -- made
 *   once; every depth it finishes is a reader's at once (the first usable one is depth 12 of that
 *   same search, not a search of its own).
 * - The game's positions are its path. While the player thinks, the workers search ahead: the
 *   best moves from the position on the board, the best reply to each, and so on to [lookAhead]
 *   plies (Tai, 2026-09-27; 2026-10-04: "we absolutely need look-ahead").
 * - A position searched ahead that the game reaches is promoted: its search, done or under way,
 *   becomes the game's, never begun again. A search is stopped only when its position is on no
 *   path any more: off the game and out of the look-ahead, nothing will read it.
 * - One worker is kept for the game's own positions, so a position the game reaches never waits
 *   behind the look-ahead and never stops it.
 * - A move is judged from the position before's lines when it is among them (85% of moves), from
 *   the position after's search otherwise -- which is the game's next position, searched anyway.
 * - The side searches (threats, traps, declined recaptures, what a quiet move threatens) are made
 *   for the game's positions, once their own search is done; a position searched ahead gets them
 *   when the game reaches it.
 *
 * Events go out as they did: an `Analysis` for each depth of a game position (provisional until
 * the full depth and its side searches are in), and a `MoveAssessment` for each move, in the
 * order of the moves.
 */

/**
 * The look-ahead searches the positions the game is likely to reach, likeliest first, while it
 * waits (Tai, 2026-10-04: "an adaptive, dynamic system that is able to maximize both comp eff and
 * the chance of catching the opponent's move"). A position's chance is the product of the chances
 * of the moves leading there. A move's chance comes from the position's own lines: a line that
 * loses `loss` of expected score against the best is played with a chance in proportion to
 * exp(-loss / T) -- a forced position puts nearly all of it on the one move, a quiet one spreads
 * it -- and T, how closely a side's moves follow the engine, is learned from this game's own moves,
 * each side's, as they are played. Nothing is trained and nothing is kept between games (Tai,
 * 2026-10-04: "move accuracy needs to be dependent on the game ... a static model ... is going to
 * drift").
 *
 * Where T starts in a game, and how many moves' worth that start counts for: a little, so the
 * game's own moves decide it within a few.
 */
const val T_START = 0.05
const val T_START_WEIGHT = 2.0

/** 0.005 .. ~1.5, the values T is fitted over. */
val T_GRID: List<Double> = (0 until 20).map { 0.005 * 1.35.pow(it) }

/** The share of moves played outside the lines searched, at the start of a game (learned too). */
const val OUTSIDE_START = 0.15

/**
 * A position is searched ahead while its chance of being reached is at least REACH, at most
 * LOOK_AHEAD plies from the board (tuned by simulation, tools/engine/simulate.py).
 */
const val REACH = 0.15
const val LOOK_AHEAD = 4

/**
 * How many workers search side by side, one thread each: the desktop's three (Tai, 2026-10-04);
 * the phone's two ([PHONE_WORKERS]), the two cores its one engine of two threads took before.
 */
const val WORKERS = 3
const val PHONE_WORKERS = 2

/**
 * The positions kept, searched, for the game and for any game after it (a puzzle tried again is
 * answered at once); the oldest go past this.
 */
const val KEEP_NODES = 20000

/**
 * An analyser ready to take moves, its workers' engines started, one thread each ([make] is handed
 * the threads). An engine that will not start is said here, the others closed, rather than
 * failing every search after. `open_analyser`.
 */
fun openAnalyser(
    depth: Int,
    multipv: Int,
    workers: Int = WORKERS,
    thresholds: Thresholds = Thresholds(),
    lookAhead: Int = LOOK_AHEAD,
    tagger: Tagger? = null,
    book: Book? = null,
    make: (threads: Int) -> Engine,
): Analyser {
    val engines = List(maxOf(1, workers)) { make(1) }
    try {
        for (engine in engines) engine.open()
    } catch (exc: Exception) {
        for (engine in engines) runCatching { engine.close() }
        throw exc
    }
    val annotator = GameAnnotator(engines[0], depth = depth, multipv = multipv, thresholds = thresholds, tagger = tagger, book = book)
    return Analyser(annotator, engines = engines, lookAhead = lookAhead).also { it.start() }
}

class Analyser(
    val annotator: GameAnnotator,
    /**
     * One per worker (open or not), the first kept for the game's own positions when there are
     * several; the annotator's own engine alone when not given.
     */
    engines: List<Engine>? = null,
    /** Plies searched ahead of the game; none unless asked (a live session asks for [LOOK_AHEAD]). */
    val lookAhead: Int = 0,
) : AutoCloseable {

    val engines: List<Engine> = if (engines.isNullOrEmpty()) listOf(annotator.engine) else engines.toList()

    /** One position: its board (with the moves that led there, which the engine reads for repetitions), and what has been found of it. */
    internal class Node(val key: String, val board: Board) {
        var measured: Measured? = null  // its own search, at the full depth
        var whole: Measured? = null  // with its side searches
        var partial: List<Line>? = null  // the deepest depth so far, while its search runs
        var failed = false
    }

    internal class Played(val ply: Int, val before: String, val after: String, val move: Move, val byUser: Boolean)

    internal class Job(val kind: String, val key: String, val era: Int)

    /** A move of this game among the lines of the position it came from: their losses, and which it was (null: outside them). */
    internal class Choice(val losses: List<Double>, val index: Int?)

    /** What [told] holds of a ply: its position, the deepest depth told, whether whole. */
    private data class Told(val key: String, val depth: Int, val whole: Boolean)

    internal val lock = Object()
    internal var nodes = LinkedHashMap<String, Node>()
    private var gameId: String? = null
    internal val path = TreeMap<Int, String>()  // ply -> the game's position at it
    private val plyOf = HashMap<String, Int>()
    private val moves = HashMap<Int, Played>()
    private var current = 0
    internal var nextVerdict: Int? = null
    private var told = HashMap<Int, Told>()
    internal var ahead = LinkedHashMap<String, Pair<Double, Int>>()  // key -> (-chance, plies on)
    private var user: Boolean? = null  // the player's colour, once the game says it

    /**
     * This game's moves, each side's (true: the player's), as what the look-ahead's chances learn
     * from; each side's T fitted to them ([fit]).
     */
    private var played = hashMapOf(true to ArrayList<Choice>(), false to ArrayList<Choice>())
    internal var t = hashMapOf(true to T_START, false to T_START)
    var reach = REACH
    private val running = HashMap<Int, Job>()  // by worker: the job a stop can reach
    internal val claimed = HashMap<Int, Job>()  // by worker: the job it took, running or about to
    private var era = 0

    /** What the workers have finished; a test reads it as the desktop's `_results`. */
    internal val results = ConcurrentLinkedQueue<Event>()
    private val threads = ArrayList<Thread>()
    private var closed = false

    // What it did, for the logs and the replay's measure.
    @Volatile var searches = 0  // positions searched to the full depth
        private set
    @Volatile var sideSearches = 0
        private set
    @Volatile var looked = 0  // positions searched ahead of the game
        private set
    @Volatile var aheadHits = 0  // of them, reached by the game
        private set
    @Volatile var stopped = 0  // searches let go because nothing would read them
        private set
    @Volatile var withdrawn = 0
        private set
    @Volatile var failures = 0
        private set
    @Volatile var lastError: String? = null
        private set

    /**
     * What a test watches the workers by, as the desktop's tests wrap `_cond.notify_all` and
     * `_results.put`: told after each word put in [results] and after each change of state the
     * workers are woken for. Null in the app, where nothing is told and nothing else changes.
     */
    internal interface Watch {
        fun said(event: Event) {}
        fun changed() {}

        /** At the end of [withdraw], what was under way for the moves taken back let go. */
        fun withdrew(gameId: String, ply: Int) {}
    }

    @Volatile internal var watch: Watch? = null

    // ---- the caller's side -------------------------------------------------------------

    fun start() {
        if (threads.isNotEmpty()) return
        for ((index, engine) in engines.withIndex()) {
            threads.add(
                Thread({ work(index, engine) }, "sidecoach-analyser-$index").also {
                    it.isDaemon = true
                    it.start()
                },
            )
        }
    }

    override fun close() {
        synchronized(lock) {
            closed = true
            for (index in running.keys.toList()) engines[index].interrupt()
            wake()
        }
        for (thread in threads) thread.join(5000)
        threads.clear()
        for (engine in uniqueEngines()) engine.close()
    }

    /**
     * What is still owed to the game: its positions not measured whole, and its moves not judged.
     * The look-ahead owes nothing.
     */
    val pending: Int
        get() = synchronized(lock) {
            var owed = path.values.count { !wholeTold(it) }
            val next = nextVerdict
            if (next != null) owed += moves.keys.count { it >= next }
            owed
        }

    fun poll(): List<Event> {
        val out = ArrayList<Event>()
        while (true) out.add(results.poll() ?: break)
        return out
    }

    /**
     * A position the game stands on with no move to it: its start, or a resync. [user]: the
     * player's colour, when the game says it (the look-ahead's widths follow it).
     */
    fun submitPosition(gameId: String, board: Board, ply: Int, user: Boolean? = null) {
        synchronized(lock) {
            if (this.gameId != gameId) newGame(gameId)
            if (user != null) this.user = user
            val key = node(board).key
            setPath(ply, key)
            current = ply
            // Told again: a position submitted is one the player faces anew (a takeback to the
            // start, a re-read), whatever was told of it before.
            told.remove(ply)
            reached(key, ply)
            reframe()
            wake()
        }
    }

    fun submitMove(gameId: String, before: Board, move: Move, ply: Int, byUser: Boolean = true) {
        synchronized(lock) {
            if (this.gameId != gameId) newGame(gameId)
            if (!before.isLegal(move)) {
                lastError = "${move.uci()} is not legal in ${before.fen()}"
                failures += 1
                return
            }
            // Whose move it was says the player's colour, every move.
            user = if (byUser) before.turn else !before.turn
            val source = node(before.copy())
            if (path[ply - 1] != source.key) {
                setPath(ply - 1, source.key)
                reached(source.key, ply - 1)
            }
            val after = before.copy()
            after.push(move)
            val target = node(after)
            if (target.key in ahead) {
                aheadHits += 1
                annotator.aheadHits += 1
            }
            setPath(ply, target.key)
            moves[ply] = Played(ply, source.key, target.key, move, byUser)
            val next = nextVerdict
            if (next == null || ply < next) nextVerdict = ply
            current = ply
            // The verdict first: the word on the move is the news, the position after it next.
            judge()
            reached(target.key, ply)
            reframe()
            wake()
        }
    }

    /**
     * The game's moves after [ply] taken back: nothing more is said of them, and a search made
     * only for them stops -- unless the look-ahead from the position taken back to wants it,
     * which it often does (the move taken back was a likely one).
     */
    fun withdraw(gameId: String, ply: Int) {
        synchronized(lock) {
            if (gameId != this.gameId) return
            val gone = path.keys.filter { it > ply }
            withdrawn += gone.count { it in moves }
            for (p in gone) {
                val key = path.remove(p)!!
                if (plyOf[key] == p) plyOf.remove(key)
                moves.remove(p)
                told.remove(p)
                annotator.forgetAssessed(p)
            }
            val next = nextVerdict
            if (next != null && next > ply + 1) nextVerdict = ply + 1
            if (moves.isEmpty()) nextVerdict = null
            current = ply
            reframe()
            wake()
        }
        watch?.withdrew(gameId, ply)
    }

    /**
     * A new game, or the board re-read: the game's path and look-ahead let go, and what is under
     * way stops; what was measured is kept, for a position seen again.
     */
    fun reset() {
        synchronized(lock) {
            gameId = null
            path.clear()
            plyOf.clear()
            moves.clear()
            told.clear()
            ahead.clear()
            nextVerdict = null
            current = 0
            stopUnwanted()
            annotator.reset()
            wake()
        }
    }

    /**
     * The game's positions measured whole -- their own search and their side searches -- in the
     * order of the game: what its review need not search again ([Assessing.keepForReview]).
     * Nothing for a game this is not on.
     */
    fun measuredPath(gameId: String): List<Measured> = synchronized(lock) {
        if (gameId != this.gameId) return emptyList()
        path.values.mapNotNull { nodes[it]?.whole }
    }

    /**
     * The settings a search depends on changed: nothing measured before counts, and a search
     * under way is not kept ([GameAnnotator.forgetMeasured]).
     */
    fun forgetMeasured() {
        synchronized(lock) {
            era += 1
            val kept = LinkedHashMap<String, Node>()
            for ((k, n) in nodes) if (k in plyOf) kept[k] = Node(k, n.board)
            nodes = kept
            told = HashMap(told.mapValues { (_, v) -> Told(v.key, 0, false) })
            for (index in running.keys.toList()) engines[index].interrupt()
            annotator.forgetMeasured()
            wake()
        }
    }

    /** Hash, the lines, their length or the scoring saved mid-session: on every worker's engine. */
    fun reconfigure(hashMb: Int? = null, multipv: Int? = null, pvPlies: Int? = null, scoreModel: String? = null) {
        for (engine in uniqueEngines()) {
            if (hashMb != null) engine.reconfigure(hashMb = hashMb)
            if (multipv != null) engine.multipv = multipv
            if (pvPlies != null) engine.pvPlies = pvPlies
            if (scoreModel != null) engine.scoreModel = scoreModel
        }
    }

    // ---- the graph ---------------------------------------------------------------------

    /** Another game: what the look-ahead learned of how its sides play starts over. */
    private fun newGame(gameId: String) {
        this.gameId = gameId
        played = hashMapOf(true to ArrayList(), false to ArrayList())
        t = hashMapOf(true to T_START, false to T_START)
    }

    /**
     * The game's position at [ply] is [key]: the one it held before, if another, is no longer
     * the game's -- nothing more is told of it at this ply, and its search stops unless the
     * look-ahead wants it.
     */
    private fun setPath(ply: Int, key: String) {
        val old = path[ply]
        if (old != null && old != key) {
            if (plyOf[old] == ply) plyOf.remove(old)
            told.remove(ply)
        }
        path[ply] = key
        plyOf[key] = ply
    }

    internal fun node(board: Board): Node {
        val key = cacheKey(board)
        nodes[key]?.let { return it }
        if (nodes.size >= KEEP_NODES) forgetOldest()
        return Node(key, board.copy()).also { nodes[key] = it }
    }

    private fun forgetOldest() {
        val keep = HashSet(path.values) + ahead.keys + claimed.values.map { it.key }
        for (key in nodes.keys.take(KEEP_NODES / 10)) {
            if (key !in keep) nodes.remove(key)
        }
    }

    private fun alive(key: String): Boolean = key in plyOf || key in ahead

    /**
     * The look-ahead from the position the game is on: the positions whose chance of being
     * reached is at least [reach], at most [lookAhead] plies on, each with its chance -- grown from
     * searched positions only; a search for a position on no path stops.
     */
    private fun reframe() {
        val next = LinkedHashMap<String, Pair<Double, Int>>()
        // Python's heap of (-chance, distance, key): the likeliest first, ties by distance, then key.
        val frontier = PriorityQueue<Triple<Double, Int, String>>(
            compareBy<Triple<Double, Int, String>>({ it.first }, { it.second }, { it.third }),
        )
        path[current]?.let { frontier.add(Triple(-1.0, 0, it)) }
        while (frontier.isNotEmpty()) {
            val (negative, distance, key) = frontier.poll()!!
            val node = nodes[key] ?: continue
            val measured = node.measured ?: continue
            if (distance >= lookAhead) continue
            val mover = isPlayer(node.board, distance)
            val odds = odds(mover, measured.lines)
            for ((line, odd) in measured.lines.zip(odds)) {
                val chance = -negative * odd
                if (chance < reach) continue
                val move = line.moves.firstOrNull()?.let { Move.fromUciOrNull(it) } ?: continue
                if (!node.board.isLegal(move)) continue
                val child = node.board.copy()
                child.push(move)
                if (child.isGameOver()) continue
                val childNode = node(child)
                if (childNode.key in next || childNode.key in plyOf) continue
                next[childNode.key] = -chance to distance + 1
                frontier.add(Triple(-chance, distance + 1, childNode.key))
            }
        }
        ahead = next
        stopUnwanted()
    }

    /**
     * Whether the player is to move in [board]; while their colour is unknown, the side to move
     * on the board is taken for theirs.
     */
    private fun isPlayer(board: Board, distance: Int): Boolean {
        val user = user ?: return distance % 2 == 0
        return board.turn == user
    }

    /**
     * The chance that each of [lines] is the move played, by the player or the opponent: in
     * proportion to exp(-loss / T) of the line's loss against the best, T this game's for that
     * side; the share of moves this game played outside the lines left out.
     */
    internal fun odds(player: Boolean, lines: List<Line>): List<Double> {
        if (lines.isEmpty()) return emptyList()
        val best = lines[0].expectedScore
        val side = t.getValue(player)
        val weights = lines.map { exp(-maxOf(0.0, best - it.expectedScore) / side) }
        val total = weights.sum()
        val choices = played.getValue(player)
        val outside = (OUTSIDE_START * T_START_WEIGHT + choices.count { it.index == null }) /
            (T_START_WEIGHT + choices.size)
        return weights.map { (1.0 - outside) * it / total }
    }

    /**
     * A move played: which of the lines of the position it came from it was, kept for the side
     * that played it, and that side's T fitted again.
     */
    internal fun learn(move: Played, lines: List<Line>) {
        if (lines.isEmpty()) return
        val uci = move.move.uci()
        val best = lines[0].expectedScore
        val losses = lines.map { maxOf(0.0, best - it.expectedScore) }
        val index = lines.indexOfFirst { it.moves.isNotEmpty() && it.moves[0] == uci }.takeIf { it >= 0 }
        val choices = played.getValue(move.byUser)
        choices.add(Choice(losses, index))
        t[move.byUser] = fit(choices)
    }

    // ---- the workers -------------------------------------------------------------------

    /**
     * The next job for worker [index]: the game's newest position not searched, then the side
     * searches of the game's positions, then -- unless this is the game's own worker -- the
     * look-ahead, likeliest first.
     */
    internal fun pick(index: Int): Job? {
        val busy = claimed.values.map { it.kind to it.key }.toSet()
        for (ply in path.descendingKeySet()) {
            val node = nodes[path[ply]] ?: continue
            if (node.measured == null && !node.failed && (MAIN to node.key) !in busy) return Job(MAIN, node.key, era)
        }
        if (annotator.withThreats) {
            for (ply in path.descendingKeySet()) {
                val node = nodes[path[ply]] ?: continue
                if (node.measured != null && node.whole == null && !node.failed && (SIDE to node.key) !in busy) {
                    return Job(SIDE, node.key, era)
                }
            }
        }
        if (index == 0 && engines.size > 1) return null  // the game's own worker
        val likeliest = ahead.entries.sortedWith(compareBy({ it.value.first }, { it.value.second }))
        for ((key, _) in likeliest) {
            val node = nodes[key]
            val wanted = node != null && node.measured == null && !node.failed
            if (wanted && (MAIN to key) !in busy) return Job(MAIN, key, era)
        }
        return null
    }

    private fun work(index: Int, engine: Engine) {
        while (true) {
            val job: Job
            val board: Board
            val measured: Measured?
            synchronized(lock) {
                var picked: Job? = null
                while (!closed) {
                    picked = pick(index)
                    if (picked != null) break
                    lock.wait()
                }
                if (closed || picked == null) return
                job = picked
                // Taken the moment it is picked: no other worker picks it meanwhile.
                claimed[index] = job
                val node = nodes.getValue(job.key)
                board = node.board.copy()
                measured = node.measured
            }
            val found = try {
                run(index, engine, job, board, measured)
            } catch (exc: Interrupted) {
                null
            } catch (exc: Exception) {
                // A failed search must never end a session.
                synchronized(lock) {
                    lastError = "${exc::class.java.simpleName}: ${exc.message}"
                    failures += 1
                    nodes[job.key]?.failed = true
                }
                null
            }
            synchronized(lock) {
                running.remove(index)
                claimed.remove(index)
                if (found != null && job.era == era && job.key in nodes) {
                    if (job.kind == MAIN) searched(job.key, found) else sided(job.key, found)
                }
                wake()
            }
        }
    }

    /**
     * One job's search, stoppable from the moment it is running ([stopUnwanted]); null when it
     * was no longer wanted by then.
     */
    private fun run(index: Int, engine: Engine, job: Job, board: Board, measured: Measured?): Measured? =
        engine.interruptible {
            synchronized(lock) {
                if (closed || job.era != era || !wanted(job)) return@interruptible null
                running[index] = job  // inside the stoppable block: a stop reaches it
            }
            if (job.kind == MAIN) {
                annotator.search(board, engine = engine, onDepth = { lines -> depthDone(job.key, job.era, lines) })
            } else {
                annotator.withSide(board, measured!!, engine = engine)
            }
        }

    private fun wanted(job: Job): Boolean {
        val node = nodes[job.key] ?: return false
        if (job.kind == MAIN) return node.measured == null && alive(job.key)
        return node.whole == null && job.key in plyOf
    }

    private fun stopUnwanted() {
        for ((index, job) in running.entries.toList()) {
            val wanted = if (job.kind == MAIN) alive(job.key) else job.key in plyOf
            if (!wanted) {
                stopped += 1
                engines[index].interrupt()
            }
        }
    }

    // ---- what comes back ---------------------------------------------------------------

    /** A depth of a search finished, the search going on. */
    private fun depthDone(key: String, era: Int, lines: List<Line>) {
        synchronized(lock) {
            val node = nodes[key] ?: return
            if (era != this.era || lines.isEmpty()) return
            node.partial = lines
            val ply = plyOf[key] ?: return
            tellPartial(ply, node)
        }
    }

    private fun searched(key: String, measured: Measured) {
        val node = nodes.getValue(key)
        node.measured = measured
        node.partial = null
        searches += 1
        if (key in ahead && key !in plyOf) looked += 1
        if (!annotator.withThreats) node.whole = measured
        judge()  // the verdict first: the word on the move, then the position after it
        plyOf[key]?.let { tell(it, node) }
        reframe()
    }

    private fun sided(key: String, whole: Measured) {
        val node = nodes.getValue(key)
        node.whole = whole
        sideSearches += 1
        plyOf[key]?.let { tell(it, node) }
    }

    /** The game is on this position at [ply]: what is known of it is told at once. */
    private fun reached(key: String, ply: Int) {
        val node = nodes.getValue(key)
        val was = told[ply]
        if (was != null && was.key != key) told.remove(ply)
        if (node.measured != null) {
            tell(ply, node)
        } else if (node.partial != null) {
            tellPartial(ply, node)
        }
    }

    private fun wholeTold(key: String): Boolean {
        val ply = plyOf[key] ?: return false
        val was = told[ply] ?: return false
        return was.key == key && was.whole
    }

    private fun tellPartial(ply: Int, node: Node) {
        val lines = node.partial ?: emptyList()
        val depth = if (lines.isEmpty()) 0 else lines[0].depth
        if (depth < EARLY_DEPTH || depth >= annotator.depth) return
        val was = told[ply]
        if (was != null && was.key == node.key && was.depth >= depth) return
        told[ply] = Told(node.key, depth, false)
        val measured = annotator.measuredLines(node.board, lines)
        put(annotator.analysisFrom(gameId ?: "", node.board, ply, measured, provisional = true))
    }

    /** The full depth told: whole once its side searches are in, provisional before. */
    private fun tell(ply: Int, node: Node) {
        val found = node.whole ?: node.measured ?: return
        val game = gameId ?: return
        val whole = node.whole != null
        val was = told[ply]
        if (was != null && was.key == node.key && (was.whole || (!whole && was.depth >= found.depth))) return
        told[ply] = Told(node.key, found.depth, whole)
        put(annotator.analysisFrom(game, node.board, ply, found, provisional = !whole))
    }

    /**
     * Each move judged as soon as it can be, in the order of the moves: the move before is what
     * a verdict reads ("the opponent just erred", "a recapture").
     */
    private fun judge() {
        while (true) {
            val at = nextVerdict ?: return
            val move = moves[at] ?: return
            val before = nodes.getValue(move.before)
            val after = nodes.getValue(move.after)
            val beforeMeasured = before.measured
            val own = beforeMeasured != null && annotator.judgeable(beforeMeasured, move.move)
            if (before.failed || (!own && after.failed)) {
                // A search that failed judges nothing: this move goes without a verdict, and the
                // moves after it are judged all the same -- it held them all back.
                nextVerdict = at + 1
                continue
            }
            if (beforeMeasured == null) return
            if (!own && after.measured == null) return  // not among the lines: the position after is needed
            try {
                val verdict = annotator.assessFrom(gameId ?: "", before.board, move.move, move.ply, beforeMeasured, after.measured)
                put(verdict)
                learn(move, beforeMeasured.lines)
            } catch (exc: Exception) {
                // A move that cannot be judged is skipped.
                lastError = "${exc::class.java.simpleName}: ${exc.message}"
                failures += 1
            }
            nextVerdict = at + 1
        }
    }

    private fun put(event: Event) {
        results.add(event)
        watch?.said(event)
    }

    /** The workers woken (under [lock]), and a test's watch told. */
    private fun wake() {
        lock.notifyAll()
        watch?.changed()
    }

    private fun uniqueEngines(): List<Engine> {
        val seen = ArrayList<Engine>()
        for (engine in engines) if (seen.none { it === engine }) seen.add(engine)
        return seen
    }

    companion object {
        private const val MAIN = "main"
        private const val SIDE = "side"

        /**
         * The T that makes this game's moves likeliest (each the line it was among its lines,
         * chance in proportion to exp(-loss / T)), the start counting as [T_START_WEIGHT] moves
         * played at the start's T; over [T_GRID]. Moves outside the lines say nothing of it.
         */
        private fun fit(played: List<Choice>): Double {
            val chosen = played.filter { it.index != null }
            fun likelihood(t: Double): Double {
                var total = 0.0
                for (choice in chosen) {
                    val logs = choice.losses.map { -it / t }
                    val top = logs.max()
                    total += logs[choice.index!!] - top - ln(logs.sumOf { exp(it - top) })
                }
                // The start: as if T_START_WEIGHT moves had shown it, a penalty for straying from it.
                return total - T_START_WEIGHT * ln(t / T_START).pow(2)
            }
            return T_GRID.maxBy { likelihood(it) }
        }
    }
}
