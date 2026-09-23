package com.sidecoach.analysis

import android.content.Context
import android.util.Log
import com.sidecoach.ui.SettingsStore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.events.Event
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Label
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Resync

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

    val on: Boolean get() = analyser != null

    /** The engine process's CPU time in milliseconds; 0 when there is no engine. */
    fun engineCpuMs(): Long = analyser?.annotator?.engine?.cpuMs() ?: 0

    fun submit(event: Event) {
        val worker = analyser ?: return
        when (event) {
            is MovePlayed -> {
                worker.submitMove(
                    event.gameId, Board(event.fenBefore), Move.fromUci(event.uci), event.ply,
                )
                from[event.ply] = event.fenBefore
            }
            is GameStarted -> {
                // The engine's queue is the last game's: dropped, so this game is measured
                // from its first move.
                worker.reset()
                from.clear()
                if (event.startFen != STARTING_FEN) {
                    worker.submitPosition(event.gameId, Board(event.startFen), 0)
                }
            }
            is Resync -> {
                worker.reset()
                from.clear()
                worker.submitPosition(event.gameId, Board(event.fen), 0)
            }
            else -> Unit
        }
    }

    fun reset() {
        analyser?.reset()
        from.clear()
    }

    /** What the engine has finished, logged and handed back for the coach. */
    fun drain(): List<Event> {
        val worker = analyser ?: return emptyList()
        val drained = worker.poll()
        for (event in drained) {
            if (event is MoveAssessment) {
                val fen = from.remove(event.ply)
                Log.i(TAG, text(event, fen))
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
        val loss = "%.3f".format(assessment.expectedLoss)
        return "${assessment.ply}. $played — $word, cost $loss, depth ${assessment.depth}" +
            if (best.isNotEmpty() && assessment.label != Label.BEST) ", best $best" else ""
    }

    companion object {
        private const val TAG = "sidecoach.analysis"

        /**
         * `cli._Assessing(settings, enabled)` with `cli._make_analyser`: the engine running
         * and built from the settings, or none, and the words for which. The desktop points
         * a missing engine at `sidecoach doctor`; the phone's binary ships in the APK, so
         * what can be missing is the network, and Setup is where it comes from.
         */
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
                Analyser(
                    GameAnnotator(
                        engine,
                        depth = config.depth,
                        multipv = config.multipv,
                        thresholds = settings.thresholds(),
                    ),
                ).also { it.start() }
            } catch (exc: Exception) {
                Log.w(TAG, "the engine did not start: $exc")
                return Assessing(null, "Stockfish did not start: ${exc.message}")
            }
            return Assessing(analyser, "Stockfish, depth ${config.depth}")
        }
    }
}
