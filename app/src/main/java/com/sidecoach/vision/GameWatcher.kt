package com.sidecoach.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Takeback
import com.sidecoach.analysis.Analyser
import com.sidecoach.analysis.Assessing
import com.sidecoach.analysis.Engine
import com.sidecoach.analysis.EngineAssets
import com.sidecoach.analysis.GameAnnotator
import com.sidecoach.events.Label
import com.sidecoach.events.MoveAssessment
import com.sidecoach.state.GamePipeline
import com.sidecoach.ui.PanelEngine
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * A frame of the screen, watched: the board found and kept, its theme learned from the start
 * position, every square named, and the whole folded into a legal game.
 *
 * This is the desktop's live loop (`cli.py`'s watcher and `ui/panel.py`'s) rather than a
 * reader of positions: a frame goes through the change gate -- gated on the board's own
 * region, so a board that has not moved costs nothing however busy the rest of the screen
 * is -- and then through the pipeline, which answers with the game's events.
 */
class GameWatcher(context: Context, themeDirectory: File? = null) : AutoCloseable {

    private val embedder: CellEmbedder? = runCatching { CellEmbedder.load(context) }
        .onFailure { Log.w(TAG, "the piece net did not load: $it") }
        .getOrNull()

    private val store = themeDirectory?.let { ThemeStore(it) }
    private val pipeline: GamePipeline? = embedder?.let { one ->
        GamePipeline(one, store).apply {
            // The clocks, when the templates are there: a missing reader costs the clocks
            // and nothing else, so it is not an error.
            val templates = GlyphTemplates.load(context.assets)
            if (templates == null) {
                Log.i(TAG, "no clock templates: the clocks will not be read")
            } else {
                clock = ClockReader(templates)
            }
        }
    }
    private val gate = ChangeGate()
    private var frameId = 0
    private var reads = 0

    /** The last clock the vision read, for the screen. */
    var clocks: ClockObservation? = null
        private set

    /**
     * The engine, when it has a network to load. Without one the game is still watched and
     * followed; it is only unassessed, and the screen says so.
     */
    private val assessing: Assessing = Assessing(
        run {
            val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
            val net = EngineAssets.found(context)
            if (!binary.exists() || net == null) {
                Log.i(TAG, "no engine yet: binary=${binary.exists()} net=${net != null}")
                null
            } else {
                runCatching {
                    val engine = Engine(binary, net, threads = ENGINE_THREADS, hashMb = ENGINE_HASH_MB)
                    val analyser = Analyser(GameAnnotator(engine, depth = ENGINE_DEPTH))
                    analyser.start()
                    analyser
                }.onFailure { Log.w(TAG, "the engine did not start: $it") }.getOrNull()
            }
        },
    )

    /** What one frame yielded, for the screen and the log. */
    data class Watch(
        val boardFound: Boolean,
        val region: Region? = null,
        val cellPx: Double = 0.0,
        val fit: Double = 0.0,
        val verified: Boolean = false,
        val changed: Boolean = true,
        /** The game as it stands: null until a position has been recognised. */
        val fen: String? = null,
        val ply: Int = 0,
        val pgn: String = "",
        val moves: List<String> = emptyList(),
        val userSide: String? = null,
        val state: ScreenState = ScreenState.UNKNOWN,
        /** What the engine has said about the moves so far, newest last. */
        val verdicts: List<String> = emptyList(),
        val engineOn: Boolean = false,
        /** Whether the reconciler has a game it is following. */
        val following: Boolean = false,
        val finished: String? = null,
        /** Cells the net had to embed on this frame: 64 at a bootstrap, a handful after. */
        val fresh: Int = 0,
        val unsure: Int = 0,
        val detectMs: Long = 0,
        val readMs: Long = 0,
    )

    private var state: ScreenState = ScreenState.UNKNOWN
    private val spoken = ArrayList<String>()
    private val verdicts = ArrayList<String>()

    fun read(bitmap: Bitmap, ts: Double): Watch? {
        val pipeline = pipeline ?: return null
        val bgr = toBgr(bitmap) ?: return null
        try {
            var began = SystemClock.elapsedRealtime()
            val pixels = FramePixels.of(bgr)
            // The desktop moves the gate onto the board before deciding a frame changed.
            gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
            val changed = gate.changed(pixels, ts)
            val gateMs = SystemClock.elapsedRealtime() - began
            began = SystemClock.elapsedRealtime()
            val events = pipeline.process(bgr, ts, changed, frameId++, pixels)
            val took = SystemClock.elapsedRealtime() - began
            for (event in events) {
                note(event)
                PanelEngine.onGameEvent(event)
                assessing.submit(event)
            }
            for (event in assessing.drain()) {
                note(event)
                PanelEngine.onEngineEvent(event)
            }
            if (++reads % 5 == 1) {
                Log.i(
                    TAG,
                    "frame $reads: gate $gateMs ms, board ${pipeline.detectMs} ms, " +
                        "cells ${pipeline.classifyMs} ms, cold ${pipeline.coldMs} ms, " +
                        "all $took ms, changed=$changed",
                )
            }
            val board = pipeline.board
            val rec = pipeline.reconciler
            return Watch(
                boardFound = board != null,
                region = board?.region,
                cellPx = board?.cellPx ?: 0.0,
                fit = board?.fit ?: 0.0,
                verified = pipeline.tracker.verified,
                changed = changed,
                fen = if (rec != null && rec.started) rec.board.fen() else null,
                ply = rec?.board?.moveStack?.size ?: 0,
                pgn = rec?.pgn() ?: "",
                moves = ArrayList(spoken),
                verdicts = ArrayList(verdicts),
                engineOn = assessing.on,
                following = rec != null && rec.started,
                userSide = pipeline.bottom?.name?.lowercase(),
                state = state,
                finished = rec?.result,
                fresh = pipeline.model?.lastFresh ?: 0,
                unsure = pipeline.readings?.count { it.kind == Pieces.Kind.UNKNOWN } ?: 0,
                detectMs = pipeline.detectMs,
                readMs = took,
            )
        } finally {
            bgr.release()
        }
    }

    /** The engine process's CPU time in milliseconds; 0 when there is no engine. */
    fun engineCpuMs(): Long = assessing.engineCpuMs()

    /** The events, as a line in the log and a line on the screen. */
    private fun note(event: Event) {
        when (event) {
            is MoveAssessment -> {
                val word = event.label?.name?.lowercase() ?: event.severity.name.lowercase()
                verdicts.add("${event.ply}. $word (${"%.2f".format(event.expectedLoss)})")
                if (verdicts.size > 20) verdicts.removeAt(0)
            }
            is GameStarted -> {
                spoken.clear()
                verdicts.clear()
                Log.i(TAG, "game started (${event.kind}), ${event.userSide} at the bottom")
            }
            is MovePlayed -> {
                spoken.add(event.san)
                if (spoken.size > 40) spoken.removeAt(0)
                Log.i(
                    TAG,
                    "${event.ply}. ${event.san} (${event.uci}) " +
                        "${if (event.byUser) "you" else "them"}, ${event.thinkMs ?: 0} ms",
                )
            }
            is ClockObservation -> {
                clocks = event
                Log.i(
                    TAG,
                    "clock: white ${event.whiteMs} black ${event.blackMs}, " +
                        "running ${event.running ?: "-"}",
                )
            }
            is Takeback -> Log.i(TAG, "takeback of ${event.plies} plies")
            is Resync -> Log.i(TAG, "resync: ${event.reason}")
            is GameEnded -> Log.i(TAG, "game over ${event.result}: ${event.reason} — ${event.pgn}")
            is ScreenStateChanged -> {
                state = event.state
                Log.i(TAG, "screen: ${event.state}")
            }
            else -> Unit
        }
    }

    private fun toBgr(bitmap: Bitmap): Mat? {
        if (!loaded) return null
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }

    override fun close() {
        assessing.close()
        embedder?.close()
    }

    companion object {
        private const val TAG = "sidecoach.game"

        /**
         * What the engine gets on a phone. The desktop's defaults are four threads, 256 MB
         * and depth 18; a phone has fewer cores to spare and a thermal budget, and the depth
         * is what the loss is measured at on both sides of a move, so it stays the desktop's.
         */
        private const val ENGINE_THREADS = 2
        private const val ENGINE_HASH_MB = 64
        private const val ENGINE_DEPTH = 18

        /** OpenCV's native library, loaded once for the process. */
        val loaded: Boolean by lazy {
            OpenCVLoader.initLocal().also { if (!it) Log.e(TAG, "OpenCV did not load") }
        }
    }
}
