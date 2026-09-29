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
import com.sidecoach.events.Unexplained
import com.sidecoach.events.BoardUnread
import com.sidecoach.events.UserUtterance
import com.sidecoach.profile.Profiler
import com.sidecoach.profile.PuzzleStore
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Store
import com.sidecoach.analysis.Assessing
import com.sidecoach.events.MoveAssessment
import com.sidecoach.coach.CoachRunner
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.state.GamePipeline
import com.sidecoach.ui.PanelEngine
import com.sidecoach.ui.PanelVoice
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * A frame of the screen, watched: the board found and kept, its theme learned from the start
 * position, every square named, and the whole folded into a legal game.
 *
 * This is the desktop's live loop (`cli.py`'s watcher and `ui/panel.py`'s) rather than a
 * reader of positions: a frame goes through the change gate -- gated on the board's own
 * region, so a board that has not moved costs nothing however busy the rest of the screen
 * is -- and then through the pipeline, which answers with the game's events. What the loop
 * learns goes to the panel's log as `panel._run` writes it: the board locked and lost, the
 * start position, a degraded reading, a tick that stalled; and the status strip, once a second.
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
    /**
     * The player model's file: a live session teaches the profile on disk; null keeps it in
     * memory, forgotten with the session, as the desktop's replay does (and a test).
     */
    profilePath: File? = File(context.filesDir, "profile.sqlite"),
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

    /**
     * The player model: a session teaches the profile on disk (`panel._run`). Practice is a
     * nicety, so the session runs without puzzles when there are none.
     */
    private val profileStore = Store(profilePath)
    private val puzzles: PuzzleStore? = try {
        Puzzles.openStore(context.filesDir)
    } catch (exc: Exception) {
        PanelEngine.emit("sys", "warn", "puzzles unavailable: ${exc.message ?: ""}")
        null
    }
    private val profiler = Profiler(profileStore, ts = System.currentTimeMillis() / 1000.0, puzzles = puzzles)

    /** Whether this tick's frame went through the loop's pass already (see `coachTick`). */
    private var dispatched = false

    /** The word that a joined position has become a game, said once per game. */
    private var promotionLogged = false

    // What `panel._run` keeps across its ticks: the vision's last transitions, the opening the
    // engine last named, and the numbers the status strip and the stall warning are made of.
    private var locked = false
    private var verified = false
    private var bootstraps = 0
    private var degraded = false
    private var opening: String? = null
    private var ticks = 0
    private val pipelineMs = ArrayList<Double>()
    private var lastHealth = SystemClock.elapsedRealtime() / 1000.0
    private var lastTs = SystemClock.elapsedRealtime() / 1000.0
    private var tickEnd = System.nanoTime()
    private var tGrab = 0L
    private var tBegun = 0L
    private var tVision = 0L
    private var tickFrame = 0

    init {
        PanelEngine.profiler = profiler
        PanelEngine.emit("sys", "info", "profile: ${profileStore.path?.path ?: "in memory (replay)"}")
        if (puzzles != null) {
            PanelEngine.emit(
                "sys",
                "info",
                "puzzles: ${puzzles.count()} from ${puzzles.path?.path}, level ${profiler.puzzleLevel()}",
            )
        } else {
            PanelEngine.emit("sys", "warn", "puzzles: none (get them in Set up, for practice after a game)")
        }
        val first = profiler.snapshot()
        PanelEngine.set("profile" to PanelEngine.profileState(first))
        coaching.events(listOf(first), SystemClock.elapsedRealtime() / 1000.0)
        // A question about the player over their games: the coach's model writes the queries
        // itself (profile/Ask), read-only over this store.
        coaching.useRecord(profileStore)
        PanelEngine.set("status" to "waiting for a board", "coach_on" to coaching.on, "analysis_on" to assessing.on)
    }

    /**
     * The last frame read, and the board on it: `latest` and `latest_geom` in the desktop's
     * loop. A command that needs pixels -- Flip, Re-read -- is pressed from the panel, and
     * the app reads nothing while its own window is in front, so the frame it works from is
     * the last one of the game's screen. Kept by swapping, never copied.
     */
    private var lastFrame: Mat? = null
    private var lastGeometry: BoardGeometry? = null

    /** The last frame's pixels as the gate reads them, for a tick that brings no new frame. */
    private var lastPixels: FramePixels? = null

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
        val coachOn: Boolean = false,
        val coachPending: Int = 0,
        /** Whether the pipeline has learned the board and follows what is on it. */
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

    /**
     * One tick of the desktop's loop over [bitmap], the newest frame -- or over the last one
     * again when [bitmap] is null: the display sent nothing since, so the screen is what it
     * was. The tick is the same either way (the desktop reads the same pixels ten times a
     * second too); the phone only skips converting, comparing and measuring pixels it has
     * already converted, compared and measured.
     */
    fun read(bitmap: Bitmap?, ts: Double): Watch? {
        val pipeline = pipeline ?: return null
        tGrab = System.nanoTime()
        val same = bitmap == null
        val bgr = (if (same) lastFrame else toBgr(bitmap!!)) ?: return null
        try {
            var began = SystemClock.elapsedRealtime()
            lastTs = ts
            val pixels = lastPixels.takeIf { same } ?: FramePixels.of(bgr)
            // The desktop moves the gate onto the board before deciding a frame changed.
            gate.region = ChangeGate.boardGate(pipeline.board, pixels.width, pixels.height)
            val changed = if (same) gate.changedAgain(pixels, ts) else gate.changed(pixels, ts)
            lastPixels = pixels
            val gateMs = SystemClock.elapsedRealtime() - began
            // The frames a Dump writes out: those that changed, in memory only.
            if (changed) PanelEngine.frames.push(ts, bgr)
            began = SystemClock.elapsedRealtime()
            tBegun = System.nanoTime()
            tickFrame = frameId++
            val events: MutableList<Event> = try {
                pipeline.process(bgr, ts, changed, tickFrame, pixels, sameFrame = same)
            } catch (exc: Exception) {  // a pipeline bug must not end the panel
                PanelEngine.emit("sys", "error", "pipeline error at frame $tickFrame: ${PanelEngine.pyRepr(exc)}")
                mutableListOf()
            }
            tVision = System.nanoTime()
            pipelineMs.add((tVision - tBegun) / 1e6)
            ticks += 1
            val took = SystemClock.elapsedRealtime() - began
            // The last board locked, kept when the lock drops: Flip and Re-read after a dialog
            // or a scroll re-read it there (`panel.py`'s latest_geom; the parity audit,
            // 2026-09-29: the phone said "no board on screen" where the desktop re-read it).
            pipeline.board?.let { lastGeometry = BoardGeometry(it.region, it.cellPx) }
            events += drainCommands(pipeline, bgr, lastGeometry, ts)
            transitions(pipeline)
            degraded = PanelEngine.checkDegraded(pipeline.readings, degraded)
            dispatch(events, ts)
            dispatched = true
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
                coachOn = coaching.on,
                coachPending = coaching.pending,
                // `panel._refresh_state`: following once the theme is learned.
                following = pipeline.model != null,
                userSide = pipeline.bottom?.name?.lowercase(),
                state = state,
                finished = rec?.result,
                fresh = pipeline.model?.lastEmbedded ?: 0,
                unsure = pipeline.readings?.count { it.kind == Pieces.Kind.UNKNOWN } ?: 0,
                detectMs = pipeline.detectMs,
                readMs = took,
            )
        } finally {
            // This frame becomes `latest`; the one it replaces is released. One frame is
            // held, never a copy per tick.
            val previous = lastFrame
            lastFrame = bgr
            pipeline.board?.let { lastGeometry = BoardGeometry(it.region, it.cellPx) }
            if (previous !== bgr) previous?.release()
        }
    }

    /**
     * Vision transitions, as `panel._run` logs them: only the changes, never every frame --
     * the board locked or lost, the lock verified or in doubt, a start position learned.
     */
    private fun transitions(pipeline: GamePipeline) {
        if ((pipeline.board != null) != locked) {
            locked = pipeline.board != null
            PanelEngine.set("locked" to locked)
            val b = pipeline.board
            if (locked && b != null) {
                val r = b.region
                PanelEngine.emit(
                    "vision",
                    "info",
                    "board locked at (${r.x},${r.y}) ${r.w}x${r.h}, cell ${PanelEngine.format1(b.cellPx)}px, " +
                        "fit ${PanelEngine.format2(b.fit)}",
                    JSONObject().apply {
                        put("region", JSONArray(listOf(r.x, r.y, r.w, r.h)))
                        put("cell_px", round(b.cellPx, 2))
                        put("fit", round(b.fit, 3))
                    },
                )
            } else {
                PanelEngine.emit("vision", "warn", "board lost")
            }
        }
        if (locked && pipeline.tracker.verified != verified) {
            verified = pipeline.tracker.verified
            PanelEngine.emit("vision", "debug", if (verified) "lock verified" else "lock in doubt")
        }
        if (pipeline.bootstraps != bootstraps) {
            bootstraps = pipeline.bootstraps
            val bottom = pipeline.bottom?.name?.lowercase() ?: "?"
            val labels = pipeline.model?.rankLabels?.size ?: 0
            PanelEngine.emit(
                "vision",
                "info",
                "start position seen, $bottom at the bottom; theme learned" +
                    "${if (store != null) " and cached" else ""}, $labels/8 rank labels",
            )
        }
    }

    /**
     * The events of a frame -- or of a command drained between frames -- through the log,
     * the page, the engine, the profile and the coach, in the order `panel`'s loop runs them.
     */
    private fun dispatch(events: List<Event>, ts: Double) {
        for (event in events) {
            note(event)
            PanelEngine.onGameEvent(event)
            assessing.submit(event)
        }
        // Idle read before the drain: then everything the engine will say about this game is
        // in what the drain returns, and a finished game can be reviewed now -- a failed
        // search never sends its verdict, and the review would otherwise wait out the grace
        // period for it.
        val idle = assessing.idle
        val assessed = assessing.drain()
        for (event in assessed) note(event)
        opening = PanelEngine.onEngineEvents(assessed, opening)
        val profiled = PanelEngine.profile(profiler, events + assessed, tick = true, flush = idle)
        // The coach, in the order the desktop's loop runs it: the game's events, the engine's
        // word on them, then the profile's. The tick and the drain are the loop's own, and
        // happen whether or not this frame was read (see `coachTick`).
        coaching.events(events, ts)
        coaching.events(assessed, ts)
        coaching.events(profiled, ts)
    }

    /**
     * `panel._drain_commands`: an ask, and the game's three buttons, from the engine's queue
     * (`PanelEngine.commands`) -- what was pressed before this session started too.
     */
    private fun drainCommands(
        pipeline: GamePipeline,
        frame: Mat?,
        geometry: BoardGeometry?,
        ts: Double,
    ): MutableList<Event> {
        val events = mutableListOf<Event>()
        while (true) {
            val (name, payload) = PanelEngine.commands.poll() ?: break
            when (name) {
                "ask", "ask_spoken" -> if (coaching.on) {
                    // Spoken or typed, the ask is a turn of the conversation on the page.
                    val spoken = name == "ask_spoken"
                    val text = payload.toString()
                    PanelEngine.heard(text, spoken)
                    coaching.ask(text, ts)
                    PanelEngine.recordDialogue(UserUtterance(text, source = if (spoken) "voice" else "text"))
                } else {
                    PanelEngine.emit("user", "warn", "no coach to ask: $payload")
                }
                "end" -> events += endGame(pipeline, payload.toString(), ts)
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
        val (result, reason) = PanelEngine.outcomeResult(outcome, PanelEngine.userSide())
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
     * running, the board on the screen is read that way; with none to read, flipping chooses
     * the side the next cold read assumes when the labels say nothing.
     */
    private fun flip(
        pipeline: GamePipeline,
        frame: Mat?,
        geometry: BoardGeometry?,
        ts: Double,
    ): MutableList<Event> {
        val rec = pipeline.reconciler
        if (rec != null && rec.started && rec.baseFen == STARTING_FEN) {
            PanelEngine.toggleViewFlip()
            PanelEngine.emit(
                "user", "info", "view flipped (the game is still read from the screen as is)",
            )
            return mutableListOf()
        }
        // The side the board is read with, when a game is: the labels may have turned it
        // since the last Flip.
        val running = if (rec != null && rec.started) pipeline.bottom else null
        val current = running ?: forcedBottom ?: pipeline.bottom ?: Pieces.Side.WHITE
        val target = current.other
        val name = target.name.lowercase()
        forcedBottom = target
        pipeline.forcedBottom = target
        PanelEngine.set("bottom" to name)
        // The board on the screen is read that way now, whether a game was running or it
        // was the one left unread for its side -- and its labels learn the side from it.
        val events = if (frame != null && geometry != null) {
            pipeline.reorient(frame, geometry, target, ts, told = true)
        } else {
            mutableListOf()
        }
        if (events.isNotEmpty()) {
            PanelEngine.emit("user", "info", "board re-read with $name at the bottom")
            return events
        }
        if (rec != null && rec.started) {
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
            .onFailure { PanelEngine.emit("vision", "error", "analyse failed: ${PanelEngine.pyRepr(it)}") }
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
     * Time passing for the coach, and whatever it has finished saying; then the page's state,
     * the status strip once a second, and a word on a tick that stalled.
     *
     * Once a tick, whether or not a frame was read: the app does not read the screen while
     * its own window is in front, and a question asked from that window still wants its
     * answer.
     */
    fun coachTick(ts: Double) {
        val pipeline = pipeline
        val read = dispatched
        if (pipeline != null) {
            val events = drainCommands(pipeline, lastFrame, lastGeometry, ts)
            // The loop's pass over the engine and the profile, once a tick: a frame read this
            // tick has had it, so only commands' events go through again.
            if (events.isNotEmpty() || !dispatched) dispatch(events, ts)
        }
        dispatched = false
        PanelVoice.loopTs = ts
        coaching.tick(ts)
        coaching.drain()
        PanelVoice.sayHeldWord(coaching.coach, ts)
        if (pipeline == null) return
        refreshState(pipeline)
        val now = SystemClock.elapsedRealtime() / 1000.0
        if (now - lastHealth >= PanelEngine.HEALTH_EVERY_S) {
            PanelEngine.health(pipeline, coaching, ticks, pipelineMs, now - lastHealth, lastTs)
            ticks = 0
            pipelineMs.clear()
            lastHealth = now
        }
        val tDone = System.nanoTime()
        if (read && (tDone - tickEnd) / 1e9 > PanelEngine.STALL_WARN_S) {
            PanelEngine.emit(
                "sys",
                "warn",
                "tick $tickFrame took ${seconds(tDone - tickEnd)}s: capture " +
                    "${seconds(tGrab - tickEnd)}s, recorder ${seconds(tBegun - tGrab)}s, vision " +
                    "${seconds(tVision - tBegun)}s, engine/coach ${seconds(tDone - tVision)}s",
            )
        }
        if (read) tickEnd = tDone
    }

    /**
     * `panel._refresh_state`: the position, the move list and the last move's highlight from
     * the game's own moves as the desktop reads them every tick -- a position adopted without a
     * new game (a puzzle after an unfinished one) or a takeback shows only what the game now
     * holds (Tai, 2026-09-24: the last puzzle's moves and its gxh3 highlight stayed on the
     * next) -- the side at the bottom, and the status. A joined position that has run long
     * enough is a game of the record (the profile's call, `Profiler.PROVISIONAL_PLIES`): the
     * page says so once.
     */
    private fun refreshState(pipeline: GamePipeline) {
        val rec = pipeline.reconciler
        val fen = rec?.board?.fen()
        val ucis = rec?.board?.moveStack?.map { it.uci() } ?: emptyList()
        val (sans, fens) = if (rec != null) sansOf(rec.baseFen, rec.board.moveStack) else Pair(emptyList(), emptyList())
        val aGame = rec != null && (rec.kind == "game" || profiler.followingAGame)
        if (aGame && rec.kind != "game" && !promotionLogged) {
            promotionLogged = true
            PanelEngine.emit(
                "game",
                "info",
                "${ucis.size} plies on the joined position: a game joined late, recorded " +
                    "from the join and reviewed at the end",
            )
        }
        val status = when {
            pipeline.model == null -> "waiting for a start"
            rec != null && rec.finished ->
                "game over ${rec.result ?: ""} · waiting for the next game".replace("  ", " ")
            aGame -> "following a game"
            else -> "following a position"
        }
        PanelEngine.set(
            "fen" to fen,
            "moves" to sans,
            "fens" to fens,
            "ucis" to ucis,
            "last_move" to ucis.lastOrNull(),
            "ply" to ucis.size,
            "following" to (pipeline.model != null),
            "bottom" to pipeline.bottom?.name?.lowercase(),
            "status" to status,
        )
    }

    /** `panel._sans`: the game's moves as SAN, and the FEN after each, from its base. */
    private fun sansOf(baseFen: String, moves: List<com.sidecoach.chess.Move>): Pair<List<String>, List<String>> {
        val board = com.sidecoach.chess.Board(baseFen)
        val out = ArrayList<String>()
        val fens = ArrayList<String>()
        for (move in moves) {
            // A resync base can make SAN ambiguous: show the UCI then, as the desktop does.
            out.add(runCatching { board.san(move) }.getOrElse { move.uci() })
            board.push(move)
            fens.add(board.fen())
        }
        return Pair(out, fens)
    }

    /** The engine process's CPU time in milliseconds; 0 when there is no engine. */
    fun engineCpuMs(): Long = assessing.engineCpuMs()

    /** The events, as a line in the log and a line on the screen. */
    private fun note(event: Event) {
        when (event) {
            is MoveAssessment -> {
                val word = event.label?.name?.lowercase() ?: event.severity.name.lowercase()
                // `f"{x:.2f}"`: half to even, never the device's locale (a comma in Germany).
                verdicts.add("${event.ply}. $word (${PanelEngine.format2(event.expectedLoss)})")
                if (verdicts.size > 20) verdicts.removeAt(0)
            }
            is GameStarted -> {
                spoken.clear()
                verdicts.clear()
                promotionLogged = false
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
            is Unexplained -> Log.i(TAG, "board unexplained: ${event.reason}; the game is kept")
            is BoardUnread -> Log.i(TAG, "board unread: ${event.reason}")
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

    /**
     * The end of a session: `panel._run`'s `finally`, on the vision thread. A Stop waits a few
     * seconds for the engine's backlog, so the review has every verdict; the coach's replies
     * in flight are waited for; the strip is brought up to date once more; Stockfish and the
     * coach stay for the next Start (only the screen stops), the engine's game dropped; the
     * session is summarised for the next one's coach, and the stores are closed.
     */
    override fun close() {
        try {
            val late = assessing.wait(5.0)
            coaching.events(late, SystemClock.elapsedRealtime() / 1000.0)
            coaching.events(PanelEngine.profile(profiler, late, flush = true), SystemClock.elapsedRealtime() / 1000.0)
            coaching.wait()
            // The strip once more, with the replies that came after the last tick.
            pipeline?.let { one -> runCatching { PanelEngine.health(one, coaching, ticks, pipelineMs, 1.0, lastTs) } }
            assessing.reset()
            try {
                val summary = profiler.endSession(System.currentTimeMillis() / 1000.0)
                if (!summary.isNullOrEmpty()) PanelEngine.emit("game", "info", "session: $summary")
            } catch (exc: Exception) {  // the profile must not block the shutdown
                PanelEngine.emit("sys", "error", "profile error at close: ${PanelEngine.pyRepr(exc)}")
            }
        } finally {
            PanelEngine.profiler = null
            coaching.useRecord(null)
            profileStore.close()
            puzzles?.close()
            lastFrame?.release()
            lastFrame = null
        }
    }

    companion object {
        private const val TAG = "sidecoach.game"

        /** `f"{x:.1f}"` of a span of nanoseconds, in seconds. */
        private fun seconds(nanos: Long): String = PanelEngine.format1(nanos / 1e9)

        /** Python's `round(x, places)`. */
        private fun round(value: Double, places: Int): Double =
            BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toDouble()

        /** OpenCV's native library, loaded once for the process. */
        val loaded: Boolean by lazy {
            OpenCVLoader.initLocal().also { if (!it) Log.e(TAG, "OpenCV did not load") }
        }
    }
}
