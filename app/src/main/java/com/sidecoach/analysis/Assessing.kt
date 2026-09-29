package com.sidecoach.analysis

import android.content.Context
import android.util.Log
import com.sidecoach.ui.SettingsStore
import java.io.File
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Event
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.Analysis

/**
 * The game's events into the engine, and the engine's verdicts back out: the `_Assessing`
 * stage of the desktop's `cli.py`.
 *
 * A move is submitted with the position it was played from, which the event carries. A game
 * that begins from the middle -- a puzzle, an analysis board, a game joined late -- or jumps
 * there on a resync submits the position instead: the coach's hint and best move need the
 * position's analysis before any move is played.
 */
class Assessing(
    val analyser: Analyser?,
    /** A few words on the engine's state, for the panel's log: on, with its depth, or why not. */
    val detail: String = if (analyser != null) "Stockfish, depth ${analyser.annotator.depth}" else "off",
) {

    /** The position each pending ply was played from, so a verdict can be read out. */
    private val from = HashMap<Int, String>()

    /**
     * The game as it stands now: its id, its last ply, and each ply's move and the position
     * after it -- what a takeback withdraws, and what [drain] holds the engine's words against.
     */
    private var gameId: String? = null
    private var ply = 0
    private val line = HashMap<Int, String>()
    private val fens = HashMap<Int, String>()

    /** Each ply's move as this glue holds the game now, for the tests. */
    internal val playedLine: Map<Int, String> get() = HashMap(line)

    /**
     * The game's board with the moves that led to it, for the opening book: the opening is a
     * fact about the moves, and the annotator asks with them. `_Assessing._game`.
     */
    private var game: Board? = null

    val on: Boolean get() = analyser != null

    /**
     * Where the assessment lines go when set (a puzzle room's log): `_Assessing.sink`. Left
     * null, they go to logcat as before.
     */
    @Volatile var sink: ((String) -> Unit)? = null

    /** The engine process's CPU time in milliseconds; 0 when there is no engine. */
    fun engineCpuMs(): Long = analyser?.annotator?.engine?.cpuMs() ?: 0

    fun submit(event: Event) {
        val worker = analyser ?: return
        when (event) {
            is MovePlayed -> {
                val before = historyTo(event.fenBefore)
                val move = Move.fromUci(event.uci)
                worker.submitMove(event.gameId, before, move, event.ply, byUser = event.byUser)
                from[event.ply] = event.fenBefore
                before.push(move)
                game = before
                gameId = event.gameId
                ply = event.ply
                line[event.ply] = event.uci
                fens[event.ply] = event.fenAfter
            }
            is Takeback -> {
                // Moves withdrawn: the engine lets go of them at once, and what it still says
                // about them is not passed on ([drain]).
                val back = maxOf(0, ply - event.plies)
                worker.withdraw(event.gameId, back)
                for (gone in line.keys.filter { it > back }) {
                    line.remove(gone)
                    fens.remove(gone)
                    from.remove(gone)
                }
                ply = back
                game = historyTo(event.fen)
            }
            is GameStarted -> {
                // The engine's queue is the last game's: dropped, so this game is measured
                // from its first move.
                worker.reset()
                from.clear()
                forget(event.gameId)
                game = Board(event.startFen)
                if (event.startFen != STARTING_FEN) {
                    worker.submitPosition(event.gameId, Board(event.startFen), 0)
                }
            }
            is Resync -> {
                worker.reset()
                from.clear()
                forget(event.gameId)
                // A position adopted, not played to: it has no history to look up.
                game = Board(event.fen)
                worker.submitPosition(event.gameId, Board(event.fen), 0)
            }
            else -> Unit
        }
    }

    /** A game begins, or its numbering starts again (a resync): no plies stand yet. */
    private fun forget(id: String) {
        gameId = id
        ply = 0
        line.clear()
        fens.clear()
    }

    /**
     * The engine's word on a move or a position the game has since taken back: a ply past the
     * game's last, or one whose move or position is no longer the game's. Passed on, a
     * withdrawn move's verdict was taken for the replacement played at its ply.
     */
    private fun stale(event: Event): Boolean {
        val (id, at) = when (event) {
            is MoveAssessment -> event.gameId to event.ply
            is Analysis -> event.gameId to event.ply
            else -> return false
        }
        if (id != gameId) return false
        if (at > ply) return true
        if (event is MoveAssessment) {
            val played = line[at]
            return played != null && played != event.uci
        }
        val fen = fens[at] ?: return false
        return fen.split(" ").take(4) != (event as Analysis).fen.split(" ").take(4)
    }

    /**
     * The game's board standing on [fen], with the moves that led there: the next move, or a
     * takeback's position further back along them. A position the game never stood on starts
     * again from itself, with no history -- as a resync does. `_Assessing._history_to`.
     */
    private fun historyTo(fen: String): Board {
        val target = Board(fen)
        val walked = game?.copy() ?: return target
        val want = target.epd()
        while (true) {
            if (walked.epd() == want) return walked
            if (walked.moveStack.isEmpty()) return target
            walked.pop()
        }
    }

    /** Nothing queued or being measured: every verdict that will come has come. */
    val idle: Boolean get() = analyser == null || analyser.pending == 0

    /** Let the engine catch up at the end of a game, so nothing is lost on the way out. */
    fun wait(timeoutS: Double = 30.0): List<Event> {
        val worker = analyser ?: return emptyList()
        val drained = ArrayList<Event>()
        val deadline = System.nanoTime() + (timeoutS * 1e9).toLong()
        while (worker.pending != 0 && System.nanoTime() < deadline) {
            drained += drain()
            Thread.sleep(50)
        }
        return drained + drain()
    }

    fun reset() {
        analyser?.reset()
        from.clear()
    }

    /** What the engine has finished, logged and handed back for the coach. */
    fun drain(): List<Event> {
        val worker = analyser ?: return emptyList()
        val drained = worker.poll().filter { !stale(it) }
        for (event in drained) {
            if (event is MoveAssessment) {
                val fen = from.remove(event.ply)
                val sink = sink
                if (sink == null) {
                    Log.i(TAG, text(event, fen))
                } else if (fen != null) {
                    sink("ply ${event.ply} ${assessmentText(event, Board(fen))}")
                }
            }
        }
        return drained
    }

    fun close() {
        analyser?.close()
    }

    private fun text(assessment: MoveAssessment, fen: String?): String {
        val board = fen?.let { Board(it) }
        val played = board?.let { b ->
            runCatching { b.san(Move.fromUci(assessment.uci)) }.getOrNull()
        } ?: assessment.uci
        val best = board?.let { sanLine(it, assessment.best, 4) } ?: assessment.best.joinToString(" ")
        val word = assessment.label?.name?.lowercase() ?: assessment.severity.name.lowercase()
        // `f"{loss:.3f}"`: half to even, and a point whatever the phone's language (a
        // locale's `%.3f` writes a comma on a German or French phone).
        val loss = fixed(assessment.expectedLoss, 3)
        return "${assessment.ply}. $played — $word, cost $loss, depth ${assessment.depth}" +
            if (best.isNotEmpty() && assessment.label != Label.BEST) ", best $best" else ""
    }

    companion object {
        private const val TAG = "sidecoach.analysis"

        /**
         * One line about a played move: what it cost, what was better, what it did.
         * `cli._assessment_text`, as the room's log has it (`_plain`: no console markup).
         * The best move is named only for a move already played, which is what the
         * disclosure rules allow.
         */
        fun assessmentText(assessment: MoveAssessment, before: Board): String {
            val parts = arrayListOf("loss ${fixed2(assessment.expectedLoss)}")
            val severity = assessment.severity.name.lowercase()
            if (severity != "none") {
                parts.add(severity)
                val best = sanLine(before, assessment.best, 4)
                if (best.isNotEmpty()) parts.add("better: $best")
            }
            val tags = assessment.tags.map { it.name }.distinct().joinToString(", ")
            if (tags.isNotEmpty()) parts.add(tags)
            return parts.joinToString("  ")
        }

        private fun fixed2(value: Double): String = fixed(value, 2)

        /** `f"{value:.Nf}"`: the float's exact value rounded half to even, never the locale's. */
        private fun fixed(value: Double, places: Int): String =
            java.math.BigDecimal(value).setScale(places, java.math.RoundingMode.HALF_EVEN).toPlainString()

        /**
         * `cli._Assessing(settings, enabled)` with `cli._make_analyser`: the engine running
         * and built from the settings, or none, and the words for which. The desktop points
         * a missing engine at `sidecoach doctor`; the phone's binary ships in the APK, so
         * what can be missing is the network, and Setup is where it comes from.
         */
        /**
         * A second Stockfish for the coach, as `cli._named_engine` builds it: for the moves
         * the player names and the model's own questions to the engine (`EngineTools`), the
         * analyser's being on the game's positions. Half the threads, at most 128 MB, three
         * lines, three seconds a position. Built, not started: the coach starts it at its
         * first search, so a session nobody asks anything pays nothing for it. Null when
         * there is no network yet.
         */
        fun coachEngine(context: Context, settings: SettingsStore): Engine? {
            val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
            val net = EngineAssets.found(context) ?: return null
            if (!binary.exists()) return null
            val config = settings.engine()
            return Engine(
                binary,
                net,
                threads = maxOf(1, config.threads / 2),
                hashMb = minOf(128, config.hashMb),
                multipv = 3,
                pvPlies = config.pvPlies,
                scoreModel = config.scoreModel,
                maxSeconds = 3.0,
            )
        }

        fun fromSettings(context: Context, settings: SettingsStore, enabled: Boolean = true): Assessing {
            if (!enabled) return Assessing(null, "off")
            val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
            val net = EngineAssets.found(context)
            if (!binary.exists() || net == null) {
                return Assessing(null, "no Stockfish network yet (see Setup)")
            }
            val config = settings.engine()
            val analyser = try {
                val engine = Engine(
                    binary,
                    net,
                    threads = config.threads,
                    hashMb = config.hashMb,
                    multipv = config.multipv,
                    pvPlies = config.pvPlies,
                    scoreModel = config.scoreModel,
                    maxSeconds = config.maxSeconds,
                )
                // `open_analyser`: the engine started when the analyser is built, so an engine
                // that will not start is said here rather than failing every search after.
                engine.open()
                Analyser(
                    GameAnnotator(
                        engine,
                        depth = config.depth,
                        multipv = config.multipv,
                        thresholds = settings.thresholds(),
                        // `_load_tagger` and `_load_book`: the motifs always, the book when
                        // Setup has fetched its tables.
                        tagger = Motifs,
                        book = OpeningBook.load(File(context.filesDir, "openings")),
                    ),
                    lookAhead = LOOK_AHEAD,
                ).also { it.start() }
            } catch (exc: Exception) {
                Log.w(TAG, "the engine did not start: $exc")
                return Assessing(null, "Stockfish did not start: ${exc.message}")
            }
            return Assessing(analyser, "Stockfish, depth ${config.depth}")
        }
    }
}
