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
import com.sidecoach.analysis.Assessing
import com.sidecoach.events.Label
import com.sidecoach.events.MoveAssessment
import com.sidecoach.coach.CoachRunner
import com.sidecoach.chess.STARTING_FEN
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
class GameWatcher(
    context: Context,
    themeDirectory: File? = null,
    /**
     * The net, the engine and the coach the warm-up built (`PanelWarm.forSession`). They
     * belong to the app, not to a session: a watcher uses them and leaves them open, as the
     * desktop's session uses `_assessing` and `_coaching` and Stop keeps them.
     */
    private val embedder: CellEmbedder?,
    private val assessing: Assessing,
    private val coaching: CoachRunner,
) : AutoCloseable {

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

    init {
        PanelEngine.onAsk = { text ->
            coaching.ask(text, android.os.SystemClock.elapsedRealtime() / 1000.0)
        }
        PanelEngine.onEnd = { outcome -> command("end", outcome) }
        PanelEngine.onFlip = { command("flip") }
        PanelEngine.onAnalyze = { command("analyze") }
    }

    /**
     * What the page asked for between frames. The panel's buttons are pressed from another
     * thread; the vision's own thread drains them where `panel._drain_commands` drains its
     * queue, so their events join the frame's.
     */
    private val commands = ArrayDeque<Pair<String, String>>()

    /**
     * The last frame read, and the board on it: `latest` and `latest_geom` in the desktop's
     * loop. A command that needs pixels -- Flip, Re-read -- is pressed from the panel, and
     * the app reads nothing while its own window is in front, so the frame it works from is
     * the last one of the game's screen. Kept by swapping, never copied.
     */
    private var lastFrame: Mat? = null
    private var lastGeometry: BoardGeometry? = null

    /** The panel's board shown turned around; `panel._view_flip`. */
    private var viewFlip = false

    /** The side a cold read assumes at the bottom once the player has chosen one. */
    private var forcedBottom: Pieces.Side? = null

    /** The last clock the vision read, for the screen. */
    var clocks: ClockObservation? = null
        private set

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
            val geometry = pipeline.board?.let { BoardGeometry(it.region, it.cellPx) }
            events += drainCommands(pipeline, bgr, geometry, ts)
            dispatch(events, ts)
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
            // This frame becomes `latest`; the one it replaces is released. One frame is
            // held, never a copy per tick.
            val previous = lastFrame
            lastFrame = bgr
            lastGeometry = pipeline.board?.let { BoardGeometry(it.region, it.cellPx) }
            if (previous !== bgr) previous?.release()
        }
    }

    /**
     * The events of a frame -- or of a command drained between frames -- through the log,
     * the page, the engine and the coach, in the order `panel`'s loop runs them.
     */
    private fun dispatch(events: List<Event>, ts: Double) {
        for (event in events) {
            note(event)
            PanelEngine.onGameEvent(event)
            assessing.submit(event)
        }
        val assessed = assessing.drain()
        for (event in assessed) {
            note(event)
            PanelEngine.onEngineEvent(event)
        }
        // The coach, in the order the desktop's loop runs it: the game's events, then
        // the engine's word on them. The tick and the drain are the loop's own, and
        // happen whether or not this frame was read (see `coachTick`).
        coaching.events(events, ts)
        coaching.events(assessed, ts)
    }

    /** Queue one of the panel's commands for the vision thread. */
    private fun command(name: String, payload: String = "") {
        synchronized(commands) { commands.addLast(name to payload) }
    }

    /** `panel._drain_commands`, for the three the phone has. */
    private fun drainCommands(
        pipeline: GamePipeline,
        frame: Mat?,
        geometry: BoardGeometry?,
        ts: Double,
    ): MutableList<Event> {
        val events = mutableListOf<Event>()
        while (true) {
            val (name, payload) = synchronized(commands) { commands.removeFirstOrNull() } ?: break
            when (name) {
                "end" -> events += endGame(pipeline, payload, ts)
                "flip" -> events += flip(pipeline, frame, geometry, ts)
                "analyze" -> events += analyse(pipeline, frame, geometry, ts)
            }
        }
        return events
    }

    /** `panel._end`: the game ends on the user's word, or is forgotten altogether. */
    private fun endGame(pipeline: GamePipeline, outcome: String, ts: Double): MutableList<Event> {
        val rec = pipeline.reconciler
        if (rec == null || !rec.started) {
            PanelEngine.emit("user", "warn", "no game to end")
            return mutableListOf()
        }
        val (result, reason) = outcomeResult(outcome, PanelEngine.userSide())
        val events = pipeline.endGame(result, reason, ts)
        if (events.isNotEmpty()) {
            PanelEngine.emit("user", "info", "you ended the game: $reason ($result)")
        }
        if (outcome == "forget") {
            pipeline.reset()
            PanelEngine.forgotten()
            PanelEngine.emit("game", "info", "forgot the game; waiting for a start position")
        }
        return events
    }

    /**
     * `panel._flip`. Flip means three things.
     *
     * A game seen from its start position has its orientation from the pieces themselves, so
     * flipping only turns the panel's board around (a view preference); re-reading an
     * unchanged screen the wrong way up would resync to a rotated, bogus position. A game
     * joined from the middle -- a puzzle, an analysis board, a resync -- had its orientation
     * from the labels or from the game before, which can be wrong: flipping re-reads the
     * screen with the other side at the bottom and begins again from that. With no game
     * running, flipping chooses the side the next cold read assumes at the bottom.
     */
    private fun flip(
        pipeline: GamePipeline,
        frame: Mat?,
        geometry: BoardGeometry?,
        ts: Double,
    ): MutableList<Event> {
        val rec = pipeline.reconciler
        if (rec != null && rec.started && rec.baseFen == STARTING_FEN) {
            viewFlip = !viewFlip
            PanelEngine.set("view_flip" to viewFlip)
            PanelEngine.emit(
                "user", "info", "view flipped (the game is still read from the screen as is)",
            )
            return mutableListOf()
        }
        val current = forcedBottom ?: pipeline.bottom ?: Pieces.Side.WHITE
        val target = if (current == Pieces.Side.WHITE) Pieces.Side.BLACK else Pieces.Side.WHITE
        val name = target.name.lowercase()
        forcedBottom = target
        pipeline.forcedBottom = target
        PanelEngine.set("bottom" to name)
        if (rec != null && rec.started) {
            val events = if (frame != null && geometry != null) {
                pipeline.reorient(frame, geometry, target, ts)
            } else {
                mutableListOf()
            }
            if (events.isNotEmpty()) {
                PanelEngine.emit("user", "info", "board re-read with $name at the bottom")
                return events
            }
            // Mid-move, or unreadable now: forget the game; the next quiet frame is read
            // cold with this side at the bottom.
            val forgotten = endGame(pipeline, "forget", ts)
            PanelEngine.emit("user", "info", "re-reading the board with $name at the bottom")
            return forgotten
        }
        PanelEngine.emit("user", "info", "analysis orientation: $name at the bottom")
        return mutableListOf()
    }

    /** `panel._analyze`: read the position on the board now, cold. */
    private fun analyse(
        pipeline: GamePipeline,
        frame: Mat?,
        geometry: BoardGeometry?,
        ts: Double,
    ): MutableList<Event> {
        if (frame == null || geometry == null) {
            PanelEngine.emit("vision", "warn", "no board on screen to analyse yet")
            return mutableListOf()
        }
        val events = runCatching { pipeline.coldStart(frame, geometry, forcedBottom, ts) }
            .onFailure { PanelEngine.emit("vision", "error", "analyse failed: $it") }
            .getOrNull() ?: return mutableListOf()
        if (events.isEmpty()) {
            PanelEngine.emit(
                "vision",
                "warn",
                "could not read the position cold: no cached style match, coordinates off " +
                    "(use Flip), or the board is mid-move",
            )
        } else {
            PanelEngine.emit("vision", "info", "analysing the position on the board (cold start)")
        }
        return events
    }

    /**
     * Time passing for the coach, and whatever it has finished saying.
     *
     * Once a tick, whether or not a frame was read: the app does not read the screen while
     * its own window is in front, and a question asked from that window still wants its
     * answer.
     */
    fun coachTick(ts: Double) {
        val pipeline = pipeline
        if (pipeline != null) {
            val events = drainCommands(pipeline, lastFrame, lastGeometry, ts)
            if (events.isNotEmpty()) dispatch(events, ts)
        }
        coaching.tick(ts)
        coaching.drain()
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
        PanelEngine.onAsk = null
        PanelEngine.onEnd = null
        PanelEngine.onFlip = null
        PanelEngine.onAnalyze = null
        lastFrame?.release()
        lastFrame = null
    }

    companion object {
        private const val TAG = "sidecoach.game"

        /** The PGN result and the reason for an outcome, from the user's side. */
        fun outcomeResult(outcome: String, userSide: String?): Pair<String, String> {
            val userWhite = userSide != "black"
            return when (outcome) {
                "resigned" ->
                    (if (userWhite) "0-1" else "1-0") to "user resigned or lost on time"
                "won" ->
                    (if (userWhite) "1-0" else "0-1") to "opponent resigned or lost on time"
                "draw" -> "1/2-1/2" to "draw agreed"
                else -> "*" to "abandoned by the user"
            }
        }

        /** OpenCV's native library, loaded once for the process. */
        val loaded: Boolean by lazy {
            OpenCVLoader.initLocal().also { if (!it) Log.e(TAG, "OpenCV did not load") }
        }
    }
}
