package com.sidecoach.state

import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.Piece
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareDistance
import com.sidecoach.events.BoardUnread
import com.sidecoach.events.Event
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.vision.Badge
import com.sidecoach.vision.BoardCandidate
import com.sidecoach.vision.BoardDetect
import com.sidecoach.vision.BoardGeometry
import com.sidecoach.vision.BoardRefine
import com.sidecoach.vision.BoardTracker
import com.sidecoach.vision.CellEmbedder
import com.sidecoach.vision.Coordinates
import com.sidecoach.vision.FramePixels
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import com.sidecoach.vision.Region
import com.sidecoach.vision.ResultBadges
import com.sidecoach.vision.ClockReader
import com.sidecoach.vision.Squares
import com.sidecoach.vision.ThemeCache
import com.sidecoach.vision.ThemeModel
import com.sidecoach.vision.ThemeStore
import android.util.Log
import org.opencv.core.Mat

/** What a read shows: the placement and the squares the site highlights. */
private typealias View = Pair<String, Set<Int>>

/**
 * Frame to events: tracker, start-position bootstrap, recogniser, reconciler.
 * A port of `sidecoach/state/pipeline.py`.
 *
 * The recogniser is rebuilt from every new start position it sees, so a board theme or piece
 * set changed between games is harmless, and the orientation is decided per game.
 *
 * One piece of the desktop's pipeline is not here and is absent rather than faked: the app's
 * own-window exclusion, which on Android is `SidecoachApp.inFront` deciding not to read at
 * all.
 */
class GamePipeline(
    private val embedder: CellEmbedder,
    val cache: ThemeStore? = null,
) {

    companion object {
        /**
         * A cover of this many cells that stays this long over a live game is the game-over
         * dialog. Chess.com's covers about thirty-five cells; a promotion picker four.
         */
        const val DIALOG_CELLS = 12
        const val DIALOG_SECONDS = 2.0

        /**
         * The site's result badges on the two kings end the game once they have been read on
         * this many consecutive frames with the same result; a story of moves still pending is
         * given this many more frames to settle first, so the record keeps the last move.
         */
        const val BADGE_FRAMES = 2
        const val BADGE_PATIENCE = 12

        /**
         * Clocks. A reading is bottom/top relative to the board; the side at the bottom is the
         * user, so the bottom box is the user's clock. chess.com draws white's box light and
         * black's dark, but the board orientation is the game's own truth, so the colours are
         * keyed from `bottom` rather than the box tint. Everything here tolerates a missing,
         * hidden, or misread clock by producing nothing: a bad clock costs the clock, never
         * the session.
         */
        const val CLOCK_MIN_CONFIDENCE = 0.4

        /** Three hours; anything above is a misread, not a clock. */
        const val CLOCK_MAX_MS = 3 * 60 * 60 * 1000

        /** A loser whose clock read this little at the end lost on time. */
        const val FLAG_MS = 1000

        /**
         * A frame that a move (or the start position) needs to be confirmed by is taken this
         * soon after the one that showed it, not at the capture's own rate (Tai, 2026-10-02:
         * latency first, and "very selectively" -- one early frame per move).
         * `CONFIRM_INTERVAL_S`.
         */
        const val CONFIRM_INTERVAL_S = 1.0 / 30

        /**
         * A resync or a cold start that left the side to move open takes it from the clock
         * that is running, when the clock was read this recently.
         */
        const val CLOCK_TURN_FRESH_S = 3.0

        /**
         * A game joined from the middle is read cold from the theme cache: tried this often,
         * and adopted once the same quiet, complete position has been read this many times, no
         * two reads further apart than the expiry.
         */
        const val COLD_START_EVERY_S = 0.5
        const val COLD_START_HOLD = 3
        const val COLD_START_EXPIRE_S = 3.0

        /** How recent a board read just before the adopted one must be to be its start, the
         * site's opening move between them (`COLD_BEFORE_S`). */
        const val COLD_BEFORE_S = 2.0

        /** The distinct boards the flip check keeps: the first board, and reads past it mid-slide. */
        const val FLIP_BOARDS_KEPT = 4

        /**
         * A board that has gone this long without being read, with no game followed on it, is
         * explained to the person once ([BoardUnread]) rather than left in silence.
         */
        const val COLD_NOTICE_S = 6.0

        /**
         * A board no move explains has its coordinate labels read this often; two reads this
         * far apart saying the other side is at the bottom turn the game round ([flipIfTurned]).
         */
        const val FLIP_CHECK_EVERY_S = 0.25

        /**
         * A game whose board the recogniser no longer reads -- this many cells unknown (a new
         * theme tints them all), or this many frames in a row no story explains (a new piece
         * set read as other pieces) -- has its look learned again from its own position, when
         * it has read broken the same way for RELEARN_HOLD_S, tried this often ([relearnLook]).
         */
        const val RELEARN_UNKNOWN_CELLS = 32
        const val RELEARN_UNEXPLAINED_FRAMES = 2
        const val RELEARN_EVERY_S = 0.5
        const val RELEARN_HOLD_S = 0.5

        /**
         * A game whose look was lost is read on once the readings are whole again: at most this
         * many cells unreadable (`pipeline.LOOK_BACK_MAX_UNREAD`).
         */
        const val LOOK_BACK_MAX_UNREAD = 2

        private const val TAG = "sidecoach.pipeline"
    }

    var startFramesNeeded = 2
    val tracker = BoardTracker()

    /**
     * The side at the bottom for a board read cold whose coordinate labels say nothing, when
     * set (the user's Flip); labels that read win over it.
     */
    var forcedBottom: Side? = null
    var board: BoardCandidate? = null
    var bootstraps = 0

    /** The clock reader, when the app has its templates; without one there are no clocks. */
    var clock: ClockReader? = null

    /**
     * One game's reading: everything [reset] forgets, each field back at its default. What it
     * keeps -- the settings, the tracker and its lock, the side the person told, the clock
     * reader, the cache and the count of recognisers built -- is the pipeline's own
     * (`pipeline._RESET_KEEPS`, desktop e0ea564, where `reset` had listed twenty fields).
     */
    private class Game {
        var model: ThemeModel? = null
        var reconciler: Reconciler? = null
        var bottom: Side? = null

        /**
         * The last clock observation emitted, so an unchanged reading is not repeated, and
         * when it was read.
         */
        var lastClock: Triple<Int?, Int?, Side?>? = null
        var lastClockTs: Double? = null

        var startSeen = 0
        var startOnScreen = false
        var readingsValue: Readings? = null

        /**
         * The geometry the readings were taken with, and whether a changed frame went unread
         * because the lock was in doubt: either means the next verified frame is read again.
         */
        var readingsKey: Pair<Region, Double>? = null
        var unreadChange = false
        var coldTried: Double? = null
        var coldCandidate: Triple<View, Int, Double>? = null

        /**
         * The read behind the candidate, and the read of the board before it with when it was
         * last seen: the start an opening move may have been played from ([autoColdStart]).
         */
        var coldCandidateRead: ColdRead? = null
        var coldBoards: List<Pair<ColdRead, Double>> = emptyList()

        /**
         * The last cold read and what it was read from (the lock and the side told), so pixels
         * that have not changed are not read again; since when no cold read has succeeded on
         * this lock, and whether the person has been told.
         */
        var coldLast: Pair<Triple<Region, Double, Side?>, ColdRead?>? = null
        var coldSince: Double? = null
        var coldNoticed = false

        /** The lock the start check last ran on with no model: unchanged pixels on it look the same. */
        var startChecked: Pair<Region, Double>? = null

        /**
         * The side the labels last said was at the bottom of an unexplained board, when, and
         * the position then read that way round.
         */
        var flipSeen: Triple<Side, Double, View?>? = null

        /**
         * The last few distinct boards the flip check read, and when: among them a new puzzle's
         * brief first board, the start its opening move is played from ([flipIfTurned]).
         */
        var flipBoards: List<Pair<ColdRead, Double>> = emptyList()
        var screenState: ScreenState = ScreenState.UNKNOWN
        var dialogSince: Double? = null
        var covered = false
        var badges: Pair<String, Int>? = null
        var badgeFrame: Triple<String, Int, Int>? = null

        /** When the look was last learned again mid-game, or tried ([relearnLook]). */
        var relearnTried: Double? = null

        /** The lock and the readings the look was last read broken with, and since when. */
        var relearnSeen: Pair<Any, Double>? = null

        /**
         * Since when the game's board has shown a look that could not be learned again, and
         * whether the person has been told ([relearnLook]).
         */
        var lookLost: Pair<Double, Boolean>? = null
    }

    private var game = Game()

    var model: ThemeModel?
        get() = game.model
        set(value) { game.model = value }
    var reconciler: Reconciler?
        get() = game.reconciler
        set(value) { game.reconciler = value }
    var bottom: Side?
        get() = game.bottom
        set(value) { game.bottom = value }

    /** `internal` for the desktop's tests, which set `_last_clock` as a clock reader would have left it. */
    internal var lastClock: Triple<Int?, Int?, Side?>?
        get() = game.lastClock
        set(value) { game.lastClock = value }
    internal var lastClockTs: Double?
        get() = game.lastClockTs
        set(value) { game.lastClockTs = value }

    /** The current frame's pixels when the caller already read them out of the Mat. */
    private var framePixels: FramePixels? = null

    /**
     * The last changed frame's grey image, for a tick that brings the same pixels again: the
     * display sent nothing, or the change gate calls them unchanged (the desktop's
     * `tracker.gray`).
     */
    private var lastGray: com.sidecoach.vision.Plane? = null

    /** What the last frame cost, by stage, for the log. */
    var detectMs = 0L
        private set
    var classifyMs = 0L
        private set
    var coldMs = 0L
        private set

    /** What the recogniser read from the last changed frame. */
    val readings: Readings? get() = game.readingsValue

    /**
     * Whether the last frame showed something that the next frame confirms: a move seen once
     * ([Reconciler.pendingStory]), or the start position seen fewer times than a game needs to
     * begin. The capture takes that next frame early ([CONFIRM_INTERVAL_S]).
     * `awaiting_confirmation`.
     */
    val awaitingConfirmation: Boolean
        get() = reconciler?.pendingStory
            ?: (game.startOnScreen && game.startSeen in 1 until startFramesNeeded)

    /**
     * Events for one frame, stamped with its time and id.
     *
     * `changed` is false when the gated part of the frame did not change since the last stored
     * frame; readings are then reused, but every tick still counts for the reconciler's timers
     * and for the start-position count.
     */
    fun process(
        frame: Mat,
        ts: Double,
        changed: Boolean = true,
        frameId: Int? = null,
        pixels: FramePixels? = null,
        /** The same pixels as the last call's frame: its grey image is the last one. */
        sameFrame: Boolean = false,
    ): MutableList<Event> {
        framePixels = pixels
        var began = android.os.SystemClock.elapsedRealtime()
        // A still tick (the gate's `changed` false) is read as the last changed frame's pixels:
        // not turned grey again, and what the tracker computed on them stands (round 3 audit,
        // E10; the desktop's tracker keeps its grey image the same way).
        val gray = lastGray.takeIf { sameFrame || !changed }
            ?: BoardDetect.toGray(frame).also { lastGray = it }
        board = tracker.update(gray, changed)
        detectMs = android.os.SystemClock.elapsedRealtime() - began
        classifyMs = 0
        coldMs = 0
        val candidate = board
        if (candidate == null || !tracker.verified) {
            // No board, or a lock the frame did not validate: readings taken from an
            // unverified geometry would be noise for the reconciler. Whatever changed in this
            // frame is still unread.
            game.unreadChange = game.unreadChange || changed
            reconciler?.boardUnseen()
            // A board that comes back is waited for, and explained, afresh.
            game.coldLast = null
            game.coldSince = null
            game.coldNoticed = false
            game.startChecked = null
            return stamp(emitScreenState(boardPresent = false), ts, frameId)
        }
        val geometry = BoardGeometry(candidate.region, candidate.cellPx)
        val key = candidate.region to Math.round(candidate.cellPx * 100.0) / 100.0
        val newPixels = changed || game.unreadChange
        // These pixels are not the ones last read cold, whether or not this tick reads.
        if (newPixels) game.coldLast = null
        var fresh = changed || game.unreadChange || game.readingsValue == null || key != game.readingsKey
        game.unreadChange = false
        val events = mutableListOf<Event>()
        // This tick carries new pixels: the moment to sample the clock.
        if (changed) events.addAll(readClock(frame, ts))
        var rebuilt = false
        if (fresh) {
            val current = model
            if (current != null) {
                // The current recogniser reading the frame as this game's start position means
                // the user browsed back to it, or the next game has the same theme and
                // orientation; the reconciler tells those apart. Only a start-shaped frame it
                // cannot read that way, a new theme or the colours swapped, needs a fresh
                // recogniser.
                began = android.os.SystemClock.elapsedRealtime()
                game.readingsValue = current.classifyFrame(frame, geometry, embedder)
                classifyMs = android.os.SystemClock.elapsedRealtime() - began
                game.readingsKey = key
                game.startOnScreen = ThemeModel.startShapedReadings(game.readingsValue!!) &&
                    !readsAsStart(game.readingsValue!!)
            } else if (newPixels || key != game.startChecked) {
                // With no model, unchanged pixels on the same lock look the same: not checked
                // again on every tick a board sits there unread (a quarter of a second a
                // frame on the phone).
                began = android.os.SystemClock.elapsedRealtime()
                game.startOnScreen = ThemeModel.looksLikeStartPosition(frame, geometry)
                classifyMs = android.os.SystemClock.elapsedRealtime() - began
                game.startChecked = key
            }
        }
        // On every verified tick, changed or not: a new look must stay a moment.
        if (model != null && !game.startOnScreen) events.addAll(relearnLook(frame, geometry, ts))
        game.startSeen = if (game.startOnScreen) game.startSeen + 1 else 0
        if (shouldBootstrap()) {
            // Decided on every verified tick, changed or not, so that a start position stored
            // on one tick and left unchanged for the next few still counts.
            val before = bootstraps
            events.addAll(bootstrap(frame, geometry))
            rebuilt = bootstraps != before
            if (rebuilt && model != null) {
                game.readingsValue = model!!.classifyFrame(frame, geometry, embedder)
                game.readingsKey = key
                // The new recogniser reads this frame as its own start position.
                game.startOnScreen = false
            }
        }
        if (model == null && fresh && !game.startOnScreen) {
            began = android.os.SystemClock.elapsedRealtime()
            events.addAll(autoColdStart(frame, geometry, ts))
            coldMs = android.os.SystemClock.elapsedRealtime() - began
        }
        val current = model
        val rec = reconciler
        val currentReadings = game.readingsValue
        if (current == null || rec == null || currentReadings == null) {
            events.addAll(emitScreenState(boardPresent = true))
            return stamp(events, ts, frameId)
        }
        if (game.lookLost != null) {
            // The board's look changed and no learned set reads it: the game is not followed on
            // readings of other pieces ([relearnLook]), only told, until its look is back.
            events.addAll(emitScreenState(boardPresent = true))
            return stamp(events, ts, frameId)
        }
        val wasStarted = rec.started
        if (fresh) game.badgeFrame = badgedKings(frame, geometry)
        events.addAll(resyncReoriented(rec.update(currentReadings, ts), frame, geometry, ts))
        if (events.none { it is Resync || it is GameStarted }) events.addAll(flipIfTurned(frame, geometry, ts, fresh))
        events.addAll(turnFromClock(ts))
        if (fresh) events.addAll(endOnBadges())
        events.addAll(dialogOverTheBoard(game.readingsValue!!, ts))
        if (wasStarted && !rebuilt && fresh &&
            events.any { it is GameStarted } &&
            ThemeModel.startShapedReadings(game.readingsValue!!)
        ) {
            // The reconciler saw the start position come back with the same orientation. The
            // theme may still have changed between games, so rebuild the recogniser from this
            // frame and keep the reconciler it already reset.
            rebuildModel(frame, geometry)
        }
        events.addAll(emitScreenState(boardPresent = true))
        return stamp(events, ts, frameId)
    }

    /**
     * A dialog that covers the board of a live game and stays.
     *
     * Chess.com puts its game-over dialog over the board the moment a game ends; other sites,
     * and other moments, put other things there. So a cover is never taken as a result on its
     * own. When the mating move was read once, clean, and the cover came before a second frame
     * could confirm it, the cover is that confirmation. Otherwise the screen state becomes
     * COVERED: the game stays open and the coach is quiet until the board is back.
     */
    private fun dialogOverTheBoard(readings: Readings, ts: Double): MutableList<Event> {
        val rec = reconciler
        if (rec == null || !rec.started || rec.finished) {
            game.dialogSince = null
            game.covered = false
            return mutableListOf()
        }
        val coveredCells = readings.count { it.covered }
        if (coveredCells < DIALOG_CELLS) {
            game.dialogSince = null
            game.covered = false
            return mutableListOf()
        }
        if (game.dialogSince == null) {
            game.dialogSince = ts
            return mutableListOf()
        }
        if (ts - game.dialogSince!! < DIALOG_SECONDS) return mutableListOf()
        if (rec.pendingTerminal() != null) {
            val events = rec.commitPendingTerminal()
            if (events.isNotEmpty()) {
                game.dialogSince = null
                return events
            }
        }
        game.covered = true
        return mutableListOf()
    }

    /**
     * The site's result badges on the kings: the result and the two squares they sit on, or
     * null.
     *
     * The kings are looked for where the reconciler has them and, when the last frame left
     * something unexplained, on the neighbouring squares as well: the mate often lands with
     * the mated king's own move, and the site stamps the badge and the check glow on that king
     * in the same frame, so its cell reads as an unsettled piece of no known kind and the
     * reconciler holds as for a drag. A badge names the king's square: the reading there is
     * replaced by a settled king, and the story confirms on the clean frames instead of
     * waiting for the dialog. Covered cells are never read.
     */
    private fun badgedKings(frame: Mat, geometry: BoardGeometry): Triple<String, Int, Int>? {
        val rec = reconciler
        val readings = game.readingsValue
        val bottomSide = bottom
        if (rec == null || !rec.started || rec.finished || bottomSide == null) return null
        if (readings == null) return null
        val wide = rec.pendingStory || rec.blocked || rec.unexplained

        fun readingAt(square: Int): CellReading {
            val (r, c) = Squares.rowCol(square, bottomSide.color)
            return readings[r * 8 + c]
        }

        fun candidates(colour: Boolean): List<Int> {
            val king = rec.board.king(colour) ?: return emptyList()
            val out = arrayListOf(king)
            if (wide) {
                for (sq in 0 until 64) if (sq != king && squareDistance(sq, king) == 1) out.add(sq)
            }
            return out
        }

        val pixels by lazy { framePixels ?: FramePixels.of(frame) }

        fun badged(colour: Boolean): List<Pair<Int, Badge>> {
            val found = ArrayList<Pair<Int, Badge>>()
            for (sq in candidates(colour)) {
                val reading = readingAt(sq)
                if (reading.covered || reading.kind == Kind.EMPTY) continue
                val badge = ResultBadges.badgeOn(pixels, geometry, sq, bottomSide)
                if (badge != null) found.add(sq to badge)
            }
            return found
        }

        val whites = badged(WHITE)
        if (whites.isEmpty()) return null
        val blacks = badged(BLACK)
        val pairs = ArrayList<Triple<String, Int, Int>>()
        for ((wk, wb) in whites) {
            for ((bk, bb) in blacks) {
                val result = ResultBadges.resultOf(wb, bb)
                if (wk != bk && result != null) pairs.add(Triple(result, wk, bk))
            }
        }
        if (pairs.size != 1) return null
        val (result, wk, bk) = pairs[0]
        for ((square, symbol, colour) in listOf(
            Triple(wk, "K", Side.WHITE), Triple(bk, "k", Side.BLACK),
        )) {
            val (r, c) = Squares.rowCol(square, bottomSide.color)
            val reading = readings[r * 8 + c]
            if (reading.kind == Kind.EMPTY || reading.covered) continue
            if (reading.kind == Kind.PIECE && reading.symbol == symbol && reading.settled) continue
            // What is drawn on the square is the badge, and it names the king: not a mark to
            // wait out, not a drag.
            readings[r * 8 + c] = reading.copy(
                kind = Kind.PIECE,
                symbol = symbol,
                colour = colour,
                confidence = 1.0,
                settled = true,
                hover = false,
                drawn = false,
                guess = symbol,
            )
        }
        return Triple(result, wk, bk)
    }

    /**
     * The game ends on the site's result badges when the board could not end it: the pair must
     * agree with itself for [BADGE_FRAMES] frames. A mating story still pending that the badges
     * confirm is committed first, so the record has the move; a frame the record has not caught
     * up with is given a few frames to be told. The reason is what the pipeline knows: a mate
     * on the board, a clock at zero; otherwise a resignation.
     */
    private fun endOnBadges(): MutableList<Event> {
        val rec = reconciler
        val frame = game.badgeFrame
        if (rec == null || !rec.started || rec.finished || frame == null) {
            game.badges = null
            return mutableListOf()
        }
        val result = frame.first
        val seen = if (game.badges != null && game.badges!!.first == result) game.badges!!.second + 1 else 1
        game.badges = result to seen
        if (seen < BADGE_FRAMES) return mutableListOf()
        val pending = rec.pendingTerminal()
        if (pending != null && pending.second.result() == result) {
            val events = rec.commitPendingTerminal()
            if (events.isNotEmpty()) return events
        }
        if (!rec.agrees && seen < BADGE_PATIENCE) {
            // The screen is ahead of the record -- the mating move is on the board and not
            // told yet -- so the record gets a few frames to catch up before the badges end
            // the game without the move.
            return mutableListOf()
        }
        return rec.end(result, endReason(rec.board, result))
    }

    /** Why the game ended, from what the pipeline knows. `internal` for the desktop's tests. */
    internal fun endReason(board: Board, result: String): String {
        if (board.isCheckmate()) return "checkmate"
        val seen = lastClock
        if (seen != null) {
            val loserMs = if (result == "1-0") seen.second else seen.first
            if (loserMs != null && loserMs <= FLAG_MS) return "lost on time"
        }
        return "resignation"
    }

    /**
     * A clock observation for this frame, or nothing.
     *
     * Never raises, never reports a clock the board is not oriented for, and repeats nothing:
     * a missing, hidden, or misread clock simply produces no event, so the coach's cadence
     * falls back to its untimed defaults rather than breaking. `internal` for the desktop's
     * tests of the clock path, which call `_read_clock`.
     */
    internal fun readClock(frame: Mat, ts: Double): MutableList<Event> {
        val reader = clock ?: return mutableListOf()
        val side = bottom ?: return mutableListOf()
        val where = board ?: return mutableListOf()
        val observation = try {
            clockObservation(reader.read(frame, where, ts), side)
        } catch (_: Exception) {
            // A clock must never take the pipeline down.
            return mutableListOf()
        } ?: return mutableListOf()
        val key = Triple(observation.whiteMs, observation.blackMs, observation.running)
        lastClockTs = ts
        if (key == lastClock) return mutableListOf()
        lastClock = key
        return mutableListOf(observation)
    }

    /**
     * A side to move left open by a resync or a cold start, settled by the clock that is
     * running -- read on this tick or a moment ago.
     */
    private fun turnFromClock(ts: Double): MutableList<Event> {
        val rec = reconciler ?: return mutableListOf()
        val seen = lastClock
        if (!rec.turnUnknown || seen == null) return mutableListOf()
        val running = seen.third ?: return mutableListOf()
        val read = lastClockTs ?: return mutableListOf()
        if (ts - read > CLOCK_TURN_FRESH_S) return mutableListOf()
        return rec.turnFromClock(running.color)
    }

    /** End the game on the user's word, stamped like every other event. */
    fun endGame(result: String, reason: String, ts: Double, frameId: Int? = null): MutableList<Event> {
        val rec = reconciler ?: return mutableListOf()
        return stamp(rec.end(result, reason), ts, frameId)
    }

    /**
     * The session's end (`end_unseen`): the game under way, its end never seen, ends with the
     * result its final position proves ([Reconciler.endUnseen]), stamped like every other event.
     */
    fun endUnseen(ts: Double, frameId: Int? = null): MutableList<Event> {
        val rec = reconciler ?: return mutableListOf()
        return stamp(rec.endUnseen(), ts, frameId)
    }

    /**
     * Forget the game and the recogniser; the next start position bootstraps again. Every
     * field of the game returns to its default ([Game]); what the pipeline keeps, it keeps.
     */
    fun reset() {
        game = Game()
        clock?.reset()
    }

    /** The last clock observation and the reader's memory dropped: the board's sides changed under them. */
    private fun forgetClock() {
        lastClock = null
        lastClockTs = null
        clock?.reset()
    }

    /**
     * A position resync checked against the coordinate labels. A resync keeps the orientation
     * the game had; a puzzle page shows each puzzle from the solver's side, so the next puzzle
     * can come up the other way round and read as a position turned on its head. When the
     * labels say the other side is at the bottom, the board is read again that way and a new
     * game begun from it, in place of the resync's events. A position adopted after a finished
     * game is a game begun, with no Resync ([Reconciler]'s position resync), and is checked the
     * same way.
     */
    private fun resyncReoriented(
        events: MutableList<Event>,
        frame: Mat,
        geometry: BoardGeometry,
        ts: Double,
    ): MutableList<Event> {
        val current = model ?: return events
        if (events.none { it is Resync || (it is GameStarted && it.kind == "position") }) return events
        val labelled = Coordinates.readBottomSide(frame, geometry, current.rankLabels)
        if (labelled == null || labelled == bottom) return events
        // A resync after a finished game began a new one on its head: that start goes unsaid,
        // and the game to end is the finished one, already ended.
        val began = events.any { it is GameStarted }
        val read = reread(labelled) ?: coldRead(frame, geometry, labelled) ?: return events
        // The opening move the right way up: the resync found it on its head, and its first
        // board turned round is the start (`_turned_start`); else the boards the flip check read
        // meanwhile ([startAmong]).
        val before = turnedStart(events, read) ?: startAmong(game.flipBoards, read, ts)
        game.flipBoards = emptyList()
        val reoriented = reorientRead(read, ts, endPrevious = !began, before = before)
        if (reoriented.isEmpty()) return events
        learnLabels(frame, geometry, told = false)
        // Everything the resync said goes with it: its start, its positions and the move it
        // found between two boards on their heads -- kept, that move was a "1... Nd5" no one
        // played (Tai's Pixel, 0.1.72). What came before it stays.
        val cut = events.indexOfFirst { it is Resync || (it is GameStarted && it.kind == "position") }
        val kept = events.subList(0, cut).filterNot { it is PositionChanged }.toMutableList()
        kept.addAll(reoriented)
        return kept
    }

    /**
     * The first board of a resync adopted on its head, turned the right way round (square s at
     * 63 - s, the colours right), when one legal move joins it to [read]: the start its opening
     * move is played from, or null (`_turned_start`).
     */
    private fun turnedStart(events: List<Event>, read: ColdRead): Pair<Board, Set<Int>>? {
        val found = events.firstOrNull { it is MovePlayed } as MovePlayed? ?: return null
        val head = Board(found.fenBefore)
        val placed = Board(null)
        for ((square, piece) in head.pieceMap()) placed.setPieceAt(63 - square, piece)
        // No highlight: the site's last-move squares on the board on its head say nothing of
        // the turn there, and the one move is looked for with either side to move.
        val lit = highlightedSquares(bySquare(read.readings, read.bottom))
        Reconciler.oneMoveBetween(placed, emptySet(), read.placement.boardFen(), read.bottom.color, lit) ?: return null
        return placed to emptySet()
    }

    /**
     * A board no move explains whose coordinate labels say the other side is at the bottom:
     * the next puzzle, shown from its solver's side. It is read that way at once, as a new
     * game, from the readings already taken ([reorient]).
     *
     * It used to wait for a resync to adopt the board on its head -- ten still frames, a
     * second and a half on the phone -- and then read all 64 cells again the right way round,
     * another second and a half; and a board that on its head was no legal position waited
     * for the long unexplained-board note, which after it stopped being a Resync (the
     * takeback fix) turned nothing round at all (Tai's puzzles on the Pixel, 2026-09-23: five
     * to ten seconds, and stuck). The labels cost a few milliseconds; two reads a moment
     * apart must agree, so a piece dragged over the edge is not taken for the board turning.
     * The labels are read even after a Flip: the side the person chose is for a board they say
     * nothing about, and a Flip that stood over them read the next puzzle from the other side
     * on its head, where no move was ever found (Tai, 2026-09-27: "The board white or black
     * should be automatically read ... every time there's a sync or resync").
     */
    private fun flipIfTurned(frame: Mat, geometry: BoardGeometry, ts: Double, changed: Boolean): MutableList<Event> {
        val rec = reconciler
        val current = model
        if (rec == null || current == null || rec.unexplainedFor < 1) {
            game.flipSeen = null
            game.flipBoards = emptyList()
            return mutableListOf()
        }
        val seen = game.flipSeen
        // The same pixels, looked at a moment ago. A changed frame is always looked at: the next
        // puzzle's first board, up for one tick right after a look at the last one, was skipped,
        // and its opening move with it (the puzzle sweep, 2026-10-03).
        if (seen != null && !changed && ts - seen.second < FLIP_CHECK_EVERY_S) return mutableListOf()
        val labelled = Coordinates.readBottomSide(frame, geometry, current.rankLabels)
        if (labelled == null || labelled == bottom) {
            game.flipSeen = null
            return mutableListOf()
        }
        // The position must read the same twice too: a piece still sliding in reads as two
        // empty squares, and a board adopted then lacks it for good -- the next puzzle's setup
        // move caught mid-slide at two frames a second, and nothing after it was recorded
        // (Tai, 2026-09-24: "sometimes there is no move list whatsoever"). Two reads after the
        // first look at the labels must agree; the board the first look reads is kept only as
        // what came before: read only at the second look, a new puzzle's first board, up for a
        // third of a second, was gone before it was ever read (Tai's puzzles, 2026-10-03: the
        // first move missed whenever the next puzzle was the other side's).
        val read = reread(labelled) ?: coldRead(frame, geometry, labelled)
        val placed = read?.let { view(it) }
        val last = game.flipBoards.lastOrNull()?.let { view(it.first) }
        if (read != null && last != placed) {
            // Kept as a board shown: a new puzzle's first board is the start its opening move is
            // played from. A read of nothing -- a piece caught sliding -- keeps what was read before
            // it, and the few last boards are kept, so one read mid-slide as a board without the
            // piece never stands in for the first board (0.1.70, Tai's dump 20261003-171151).
            game.flipBoards = game.flipBoards.takeLast(FLIP_BOARDS_KEPT - 1) + (read to ts)
        }
        if (seen == null || seen.first != labelled) {
            game.flipSeen = Triple(labelled, ts, null)
            return mutableListOf()
        }
        if (placed == null || placed != seen.third) {
            game.flipSeen = Triple(labelled, ts, placed)
            return mutableListOf()
        }
        game.flipSeen = null
        val before = startAmong(game.flipBoards, read!!, ts)
        game.flipBoards = emptyList()
        val events = reorientRead(read, ts, before = before)
        if (events.isNotEmpty()) learnLabels(frame, geometry, told = false)
        return events
    }

    /**
     * Of [boards], read before [read], the newest that one legal move joins to it, seen at most
     * [COLD_BEFORE_S] ago the same way up: the start a new puzzle's opening move is played from
     * ([Reconciler.startFrom]), or null. The newest, not the last: a frame caught mid-slide, read
     * as a board without the moving piece, came between (`_start_among`).
     */
    private fun startAmong(boards: List<Pair<ColdRead, Double>>, read: ColdRead, ts: Double): Pair<Board, Set<Int>>? {
        val target = read.placement.boardFen()
        val lit = highlightedSquares(bySquare(read.readings, read.bottom))
        for ((prior, seenAt) in boards.asReversed()) {
            if (ts - seenAt > COLD_BEFORE_S || prior.bottom != read.bottom) continue
            if (prior.placement.boardFen() == target) continue
            val highlighted = highlightedSquares(bySquare(prior.readings, prior.bottom))
            if (Reconciler.oneMoveBetween(prior.placement, highlighted, target, read.bottom.color, lit) != null) {
                return prior.placement to highlighted
            }
        }
        return null
    }

    /**
     * The frame's own readings taken with [side] at the bottom: the recogniser read cells, not
     * squares, so turning the board round needs no second read of its 64 cells -- a second
     * and a half on the phone. Null when the readings are not this frame's, or not a quiet,
     * complete, legal position that way round.
     */
    private fun reread(side: Side): ColdRead? {
        val current = model ?: return null
        val readings = game.readingsValue ?: return null
        val lock = board ?: return null
        if (game.readingsKey != (lock.region to Math.round(lock.cellPx * 100.0) / 100.0)) return null
        val placement = placementFromReadings(readings, side) ?: return null
        if (adoptPlacement(placement) == null) return null
        return ColdRead(current, side, readings, placement)
    }

    /**
     * Read the board again with `bottom` at the bottom and begin again from what it shows: the
     * panel's Flip on a game joined from the middle, or a resync whose labels say the other
     * side. A game under way ends first. Nothing happens when the board cannot be read cleanly
     * now (a piece in flight): the caller may set [forcedBottom] and forget the game, and the
     * automatic cold start reads it that way on the next quiet frame. The frame's own readings
     * serve when they can ([reread]); a cold read from the theme cache otherwise. [told]: the
     * person said which side is at the bottom (Flip), and the labels learn from them where
     * they did not say it themselves ([learnLabels]).
     */
    fun reorient(
        frame: Mat,
        geometry: BoardGeometry,
        bottom: Side,
        ts: Double,
        endPrevious: Boolean = true,
        told: Boolean = false,
    ): MutableList<Event> {
        val read = reread(bottom) ?: coldRead(frame, geometry, bottom) ?: return mutableListOf()
        val events = reorientRead(read, ts, endPrevious)
        if (events.isNotEmpty()) learnLabels(frame, geometry, told)
        return events
    }

    /**
     * The coordinate labels learned from a board just adopted, when they say no side there --
     * learned where a start position was seen through a lock off its board, or not learned at
     * all -- and the set saved with them, so the next board of the set reads its own side.
     * Only a side known for this board teaches: told for it ([told], a Flip), or read from its
     * 8 ([Coordinates.eights]). A Flip left standing from the last puzzle would teach the next
     * one, shown from the other side, upside down, and every read after it. Labels that read a
     * side are kept, the Flip's or the other: learned as they read their start back, they are
     * right, and a Flip by mistake would have taught every board of the set upside down.
     */
    private fun learnLabels(frame: Mat, geometry: BoardGeometry, told: Boolean) {
        val current = model ?: return
        val side = bottom ?: return
        if (Coordinates.readLabels(frame, geometry, current.rankLabels) != null) return
        if (!told && Coordinates.eights(frame, geometry) != side) return
        val labels = Coordinates.learnRankLabels(frame, geometry, side)
        if (labels.isEmpty()) return
        current.rankLabels = labels
        val store = cache ?: return
        for ((key, cached) in store.candidates(current.lightBg, current.darkBg)) {
            if (cached.exemplarVectors.contentDeepEquals(current.exemplarVectors)) {
                store.save(current, key)
                return
            }
        }
    }

    /** `reorient` with its read already taken (the desktop's `read=`): [ColdRead] is private. */
    private fun reorientRead(
        read: ColdRead,
        ts: Double,
        endPrevious: Boolean = true,
        before: Pair<Board, Set<Int>>? = null,
    ): MutableList<Event> {
        val events = if (endPrevious) endInProgress("board reoriented") else mutableListOf()
        events.addAll(adoptCold(read, ts, before))
        return events
    }

    private fun endInProgress(reason: String): MutableList<Event> {
        val previous = reconciler
        if (previous == null || !previous.inProgress || previous.finished) return mutableListOf()
        return mutableListOf(previous.replacedEvent(reason))
    }

    /**
     * The screen state from what the pipeline already knows: whether a board is locked, and
     * where its game sits in its life. DESIGN's first slice reads it from board presence and
     * the game lifecycle (running clocks refine live-versus-analysis later); there is no
     * separate classifier. `internal` for the desktop's tests.
     */
    internal fun deriveScreenState(boardPresent: Boolean): ScreenState {
        if (!boardPresent) return ScreenState.UNKNOWN  // a menu, the lobby, or scrolled off
        val rec = reconciler
        if (rec == null || !rec.started) return ScreenState.UNKNOWN  // no game recognised yet
        if (rec.finished) return ScreenState.GAME_OVER
        if (game.covered) return ScreenState.COVERED
        return ScreenState.LIVE_GAME
    }

    private fun emitScreenState(boardPresent: Boolean): MutableList<Event> {
        val state = deriveScreenState(boardPresent)
        if (state == game.screenState) return mutableListOf()
        game.screenState = state
        return mutableListOf(ScreenStateChanged(state))
    }

    private fun stamp(events: MutableList<Event>, ts: Double, frameId: Int?): MutableList<Event> {
        for (event in events) {
            event.ts = ts
            event.sourceFrameId = frameId
        }
        return events
    }

    /**
     * Whether the recogniser reads the frame as the start position it was built on. Nothing
     * may contradict the start, bar one cell for a badge or a lifted piece; and most of the
     * board must be readable at all, since a new theme reads as unknown almost everywhere
     * without contradicting anything. `internal` for the desktop's test of a stray reading.
     */
    internal fun readsAsStart(readings: Readings, at: Side? = null): Boolean =
        readsAs(readings, null, checkNotNull(at ?: bottom))

    /** [readsAsStart]'s rule for any [position] (the start position when null). `_reads_as`. */
    private fun readsAs(readings: Readings, position: Board?, bottomSide: Side): Boolean {
        val start = position ?: Board()
        var contradictions = 0
        var unknown = 0
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val reading = readings[r * 8 + c]
                if (reading.kind == Kind.UNKNOWN) { unknown += 1; continue }
                val piece = start.pieceAt(Squares.squareAt(r, c, bottomSide.color))
                val bad = if (piece == null) {
                    reading.kind == Kind.PIECE && reading.symbol != null
                } else {
                    reading.kind == Kind.EMPTY ||
                        (reading.symbol != null && reading.symbol != piece.symbol())
                }
                if (bad) contradictions += 1
            }
        }
        return unknown < 32 && contradictions <= 1
    }

    private fun shouldBootstrap(): Boolean = game.startOnScreen && game.startSeen >= startFramesNeeded

    /**
     * A recogniser learned from the start position on screen, and the orientation with it: a
     * new game, whose clock is not the last one's. `internal` for the test that a rebuild
     * forgets the clock.
     */
    internal fun rebuildModel(frame: Mat, geometry: BoardGeometry): Boolean {
        val lock = board
        val gray = lastGray
        if (lock != null && gray != null && lockOffBoard(gray, lock)) {
            // The board moved under the lock -- a page finishing its layout slid it 10 px on
            // Tai's desktop (2026-10-05) -- and the lock is re-fitted only every few ticks: a set
            // learned now is learned on squares cut beside the board, every piece of every later
            // board a tenth of a cell off its learned place, read as one in flight. The lock is
            // re-fitted on the next tick, and the start learned once it sits on the board.
            tracker.refitNext()
            return false
        }
        val whiteAtBottom = ThemeModel.orientationFromStart(ThemeModel.occupancy(frame, geometry))
        val built = ThemeModel.fromStartPosition(frame, geometry, whiteAtBottom, embedder)
        game.startSeen = 0
        if (built == null) return false  // something covered a piece; the next start frame tries
        val side = Side.of(whiteAtBottom)
        bottom = side
        model = built
        built.rankLabels = Coordinates.learnRankLabels(frame, geometry, side)
        game.coldSince = null
        game.coldNoticed = false
        game.lookLost = null
        forgetClock()
        reconciler?.let {
            it.bottom = side
            it.userSide = side
        }
        bootstraps += 1
        model = keepLook(built, frame, geometry, side, Board())
        return true
    }

    /**
     * A look just learned from [frame] showing [position]: the set to read with. A cached set
     * that reads the frame as [position], every piece at rest, is this look already
     * ([learnedSet]): it is read with, and nothing is saved. Any other look is saved beside the
     * cached sets -- never over one. Replacing the set that read the board left Tai with none
     * (2026-10-05): chess.com's puzzles page shows small start positions whose pieces sit lower
     * and larger in their squares; his set read them, so the set learned from them was saved
     * over it, and on every real board after it each piece read as one in flight -- no puzzle
     * adopted, through five restarts. Last, and never in the way: an exception here -- a damaged
     * set among the cached ones -- left the rebuild half done (round 2, finding 9); the theme
     * simply goes unkept. `pipeline._keep_look`.
     */
    private fun keepLook(learned: ThemeModel, frame: Mat, geometry: BoardGeometry, side: Side, position: Board): ThemeModel {
        val store = cache ?: return learned
        try {
            val cached = learnedSet(store, learned, frame, geometry, side, position)
            if (cached != null) {
                if (cached.rankLabels.isEmpty()) cached.rankLabels = learned.rankLabels
                return cached
            }
            store.save(learned)
        } catch (e: Exception) {
            Log.w(TAG, "the learned theme was not cached", e) // as the desktop logs it
        }
        return learned
    }

    /**
     * Whether a re-fit of the lock finds the board where the lock is, to within the tracker's
     * deadband, with the fit the tracker asks of a lock (`pipeline._on_its_board`).
     */
    /**
     * Whether a re-fit on this frame finds the board, validating, more than the tracker's
     * deadband from the lock: a lock that trails a board just moved. A re-fit that does not
     * validate says nothing against the lock (a badge, a cover): it stands.
     * `pipeline._lock_off_board`.
     */
    private fun lockOffBoard(gray: com.sidecoach.vision.Plane, lock: BoardCandidate): Boolean {
        val refit = BoardRefine.refine(gray, lock).board
        val shift = maxOf(
            kotlin.math.abs(refit.region.x - lock.region.x),
            kotlin.math.abs(refit.region.y - lock.region.y),
            kotlin.math.abs(refit.region.x + refit.region.w - lock.region.x - lock.region.w),
            kotlin.math.abs(refit.region.y + refit.region.h - lock.region.y - lock.region.h),
        )
        return refit.fit >= tracker.minFit && shift > tracker.refitDeadbandPx
    }

    private fun onItsBoard(gray: com.sidecoach.vision.Plane, lock: BoardCandidate): Boolean {
        val refit = BoardRefine.refine(gray, lock).board
        val shift = maxOf(
            kotlin.math.abs(refit.region.x - lock.region.x),
            kotlin.math.abs(refit.region.y - lock.region.y),
            kotlin.math.abs(refit.region.x + refit.region.w - lock.region.x - lock.region.w),
            kotlin.math.abs(refit.region.y + refit.region.h - lock.region.y - lock.region.h),
        )
        return refit.fit >= tracker.minFit && shift <= tracker.refitDeadbandPx
    }

    /**
     * A board whose look changed mid-game: read on with a whole learned set that reads the game's
     * position, or a [BoardUnread] ("look") when none does (`pipeline._relearn_look`).
     *
     * A look is never learned mid-game. A set learned from a game's position knew only the kinds
     * that position held, and one theme and piece set could be learned partly again and again,
     * the sets disagreeing: one hard square had a rook endgame learned as the look, and the next
     * puzzle's knights were read as bishops, or not at all, and whole puzzles went unread, in
     * silence (Tai's Pixel, 2026-10-03: five such sets for one board). A set is learned whole,
     * at a start position, and nowhere else (Tai: "bring it up to the user as an error and do
     * not try to get a partial read").
     *
     * A look changed is a new theme, or a new piece set: on a board no story explains, at least
     * half the pieces of the game's position -- or one legal move on -- not read as their own
     * kind, landed ([lookChanged]). A square or two read badly is no new look. A cached set that
     * reads the position, every piece landed, is taken; otherwise the person is told once, after
     * [COLD_NOTICE_S], and the game goes unread until its look is back or a new game starts.
     */
    private fun relearnLook(frame: Mat, geometry: BoardGeometry, ts: Double): MutableList<Event> {
        val none = mutableListOf<Event>()
        val rec = reconciler ?: return none
        val current = model ?: return none
        val side = bottom ?: return none
        val readings = game.readingsValue ?: return none
        if (!rec.started || rec.finished || ThemeModel.startShapedReadings(readings)) return none
        val unknown = readings.count { it.kind == Kind.UNKNOWN }
        // Sampled only when it may matter: this runs on every still tick.
        var sampled = if (unknown >= RELEARN_UNKNOWN_CELLS) ThemeCache.samplePalette(frame, geometry) else null
        // A new theme is new square colours: half the board unknown on the colours the
        // recogniser knows is a lock a few pixels off its board, before the re-fit caught up
        // (the desktop's geometry sweep, seed 175).
        val newTheme = unknown >= RELEARN_UNKNOWN_CELLS && sampled != null &&
            !ThemeCache.sameTheme(sampled.first, sampled.second, current.lightBg, current.darkBg)
        if (game.lookLost != null) {
            // The game is not followed while its look is lost, so its stories say nothing: the
            // look is back when the readings are whole again.
            if (newTheme || unlanded(readings) || unreadable(readings) > LOOK_BACK_MAX_UNREAD) {
                return lookUnread(ts)
            }
            Log.i(TAG, "the board's look is back: the game is read on")
            game.lookLost = null
            game.relearnSeen = null
            return none
        }
        // The game unread: a new theme, or frames no story explains (a new set read as other
        // pieces). Or not yet unread but about to be: a new set read as the right pieces that no
        // longer land on their squares ([unlanded]; round 3, the desktop's look-change sweep).
        val stalled = newTheme || rec.unexplainedFor >= RELEARN_UNEXPLAINED_FRAMES
        if (!stalled && !unlanded(readings)) {
            game.lookLost = null // the game is read
            game.relearnSeen = null
            return none
        }
        if (readings.any { it.hover }) { // a piece dragged
            game.relearnSeen = null
            return none
        }
        // Only a look that stays broken, read the same on one lock for RELEARN_HOLD_S: a piece
        // in flight, a premove's ghost or a lock a few pixels off for a frame pass (the
        // desktop's stress sweep, seeds 59, 60, 194; its geometry sweep, 133).
        val seen: Any = game.readingsKey to readings.map { it.kind to it.symbol }
        val since = game.relearnSeen
        if (since == null || since.first != seen) {
            game.relearnSeen = seen to ts
            return none
        }
        if (ts - since.second < RELEARN_HOLD_S) return none
        val tried = game.relearnTried
        if (tried != null && ts - tried < RELEARN_EVERY_S) return none
        game.relearnTried = ts
        val whiteAtBottom = side == Side.WHITE
        // Read only on a lock that sits on its board to the pixel (the desktop's geometry sweep,
        // seed 25: a board mid-nudge read as a new theme).
        val lock = board ?: return none
        val gray = lastGray ?: return none
        if (!onItsBoard(gray, lock)) return none
        // The frame must show pieces exactly where the game's position, or one legal move on
        // from it, puts them, as the look-free occupancy reads them: another board, a dialog or a
        // piece in flight shows none of them, and is no look of this game's.
        val base = rec.board.copy(withStack = false)
        val positions = mutableListOf(base)
        for (move in base.legalMoves()) {
            if (move.promotion != 0 && move.promotion != com.sidecoach.chess.QUEEN) continue
            positions.add(base.copy(withStack = false).also { it.push(move) })
        }
        val occupied = ThemeModel.occupancy(frame, geometry).map { it.fraction >= Pieces.PIECE_MIN_FRACTION }
        if (sampled == null) sampled = ThemeCache.samplePalette(frame, geometry)
        val shown = positions.firstOrNull { position ->
            (0 until 64).none { i ->
                occupied[i] != (position.pieceAt(Squares.squareAt(i / 8, i % 8, whiteAtBottom)) != null)
            }
        }
        if (shown == null) {
            if (stalled && newTheme) return lookUnread(ts)
            return none // nothing says the look changed (another board, a dialog, a story)
        }
        if (!newTheme && !lookChanged(readings, shown, whiteAtBottom)) {
            return none // a square or two read badly: the reconciler's to explain, no new look
        }
        // A whole set cached for these colours that reads the position, every piece landed: the
        // look switched to one learned before.
        val palette = sampled
        val store = cache
        if (store != null && palette != null) {
            for ((_, cached) in store.candidates(palette.first, palette.second)) {
                val cachedReadings = cached.classifyFrame(frame, geometry, embedder)
                if (readsLanded(cachedReadings, shown, whiteAtBottom)) {
                    model = cached
                    game.readingsValue = cachedReadings
                    game.lookLost = null
                    Log.i(TAG, "the look changed mid-game to a learned set; read on with it")
                    return none
                }
            }
        }
        return lookUnread(ts)
    }

    /**
     * The game's board in a look no learned set reads: the game goes unread, and the person is
     * told once, as of a board unread at the start ([COLD_NOTICE_S]) (`pipeline._look_unread`).
     */
    private fun lookUnread(ts: Double): MutableList<Event> {
        val (lostSince, told) = game.lookLost ?: (ts to false)
        game.lookLost = lostSince to (told || ts - lostSince >= COLD_NOTICE_S)
        if (told || ts - lostSince < COLD_NOTICE_S) return mutableListOf()
        Log.w(TAG, "the board's look changed mid-game and no learned set reads it")
        return mutableListOf(BoardUnread("look"))
    }

    /** The cells read as nothing known: unknown, or a piece of no kind or off its square. */
    private fun unreadable(readings: Readings): Int = readings.count {
        it.kind == Kind.UNKNOWN || (it.kind == Kind.PIECE && (it.symbol == null || !it.settled))
    }

    /**
     * Whether at least half the pieces of [position] are not read as their own kind, landed on
     * their squares -- or a kind reads all off its squares ([unlanded]): a new piece set. A square
     * or two read badly is no new look (`pipeline._look_changed`).
     */
    private fun lookChanged(readings: Readings, position: Board, whiteAtBottom: Boolean): Boolean {
        var pieces = 0
        var wrong = 0
        for (i in 0 until 64) {
            val piece = position.pieceAt(Squares.squareAt(i / 8, i % 8, whiteAtBottom)) ?: continue
            pieces += 1
            val reading = readings[i]
            if (!(reading.kind == Kind.PIECE && reading.symbol == piece.symbol() && reading.settled)) wrong += 1
        }
        return pieces > 0 && (2 * wrong >= pieces || unlanded(readings))
    }

    /**
     * Whether a kind of which two or more pieces stand on the board reads every one of them off
     * its square (unsettled): what a new piece set the recogniser still names right looks like
     * (its landing references are the old set's), while something drawn over the board -- an
     * arrow, a mark, a highlight -- spoils squares, not every piece of a kind
     * (`pipeline._unlanded`, round 3).
     */
    private fun unlanded(readings: Readings): Boolean {
        val landed = HashMap<String, MutableList<Boolean>>()
        for (reading in readings) {
            val symbol = reading.symbol
            if (reading.kind == Kind.PIECE && symbol != null) landed.getOrPut(symbol) { ArrayList() }.add(reading.settled)
        }
        return landed.values.any { it.size >= 2 && it.none { settled -> settled } }
    }

    /**
     * Whether [readings] show [position] on every cell: each of its pieces of its own kind,
     * landed on its square, and nothing on the squares it leaves empty (`pipeline._reads_landed`).
     */
    private fun readsLanded(readings: Readings, position: Board, whiteAtBottom: Boolean): Boolean {
        for (i in 0 until 64) {
            val reading = readings[i]
            val piece = position.pieceAt(Squares.squareAt(i / 8, i % 8, whiteAtBottom))
            if (piece == null) {
                if (reading.kind != Kind.EMPTY) return false
            } else if (!(reading.kind == Kind.PIECE && reading.symbol == piece.symbol() && reading.settled)) {
                return false
            }
        }
        return true
    }

    /**
     * The cached set this frame shows again -- the same pieces drawn the same way on the same
     * board, learned before -- or null for a look not seen yet. Each set cached on these board
     * colours reads the frame; one that reads it as [position] (the start position by default)
     * quiet and whole, every piece of its kind and at rest where that set's pieces rest, is this
     * look. Pieces drawn otherwise -- the same set larger, or lower in their squares -- read as
     * their kinds but off their resting places: another look, kept beside ([keepLook]).
     * `pipeline._learned_set`.
     */
    private fun learnedSet(
        store: ThemeStore,
        model: ThemeModel,
        frame: Mat,
        geometry: BoardGeometry,
        side: Side,
        position: Board? = null,
    ): ThemeModel? {
        val start = position ?: Board()
        for ((_, cached) in store.candidates(model.lightBg, model.darkBg)) {
            val placement = placementFromReadings(cached.classifyFrame(frame, geometry, embedder), side)
            if (placement != null && placement.boardFen() == start.boardFen()) return cached
        }
        return null
    }

    /**
     * The board read cold from the theme cache: a cached recogniser for its colours, the
     * orientation ([bottom], else the coordinate labels, else [fallback] -- the side a Flip
     * chose, for a board whose labels say nothing), the readings and the placement they make
     * -- or null when no learned set gives all of those.
     *
     * One board's colours can carry several learned sets. Each reads the board; a set whose
     * read is not a quiet, complete, legal position is passed over (a set for other pieces
     * misreads them: a king taken for a bishop), and of those that give one the surest read
     * wins, so a set learned for other pieces on the same colours never stands in for the
     * right one (Tai's Pixel, 2026-09-23: a test board's set, cached under chess.com's green,
     * read his puzzle's black pawns as white).
     */
    private fun coldRead(
        frame: Mat,
        geometry: BoardGeometry,
        bottom: Side?,
        asked: Boolean = false,
        fallback: Side? = null,
    ): ColdRead? {
        val store = cache ?: return null
        val sampled = com.sidecoach.vision.ThemeCache.samplePalette(frame, geometry) ?: return null
        var best: Pair<Double, ColdRead>? = null
        for ((_, cached) in store.candidates(sampled.first, sampled.second)) {
            // Coordinates off, or not this set's; the caller may say the side.
            val side = bottom ?: Coordinates.readBottomSide(frame, geometry, cached.rankLabels)
                ?: fallback ?: continue
            val readings = cached.classifyFrame(frame, geometry, embedder)
            val placement = placementFromReadings(readings, side, asked = asked) ?: continue
            if (adoptPlacement(placement) == null) continue
            val fit = fit(readings)
            if (best == null || fit > best.first) best = fit to ColdRead(cached, side, readings, placement)
        }
        return best?.second
    }

    /**
     * How surely a set read a board: the mean margin of its piece readings over the next
     * nearest kind. The right set reads its own pieces well clear of any other kind.
     */
    private fun fit(readings: Readings): Double {
        var sum = 0.0
        var n = 0
        for (cell in readings) {
            if (cell.kind == Kind.PIECE) { sum += cell.confidence; n += 1 }
        }
        return if (n == 0) 0.0 else sum / n
    }

    /**
     * A board that has stayed unread for [COLD_NOTICE_S] is explained, once while it stays:
     * the person saw nothing happen and could not know why (Tai's puzzle on the Pixel,
     * 2026-09-23). Nothing is said when a learned set does read the board and only a passing
     * thing stands in the way -- a piece held, a selected piece's dots -- and the wait starts
     * again.
     */
    private fun coldNotice(frame: Mat, geometry: BoardGeometry, ts: Double): MutableList<Event> {
        val since = game.coldSince
        if (since == null) {
            game.coldSince = ts
            return mutableListOf()
        }
        if (game.coldNoticed || ts - since < COLD_NOTICE_S) return mutableListOf()
        val reason = unreadReason(frame, geometry)
        if (reason == null) {
            game.coldSince = ts
            return mutableListOf()
        }
        game.coldNoticed = true
        return mutableListOf(BoardUnread(reason))
    }

    /**
     * Why a board goes unread: "orientation" when a learned set reads it as a position but
     * the coordinates say nothing (turned off), so Flip must; "unlearned" when no learned set
     * reads it as one -- its colours never seen, or its pieces not the ones the sets on those
     * colours learned -- so a start position shown once teaches it; null when a learned set
     * reads it with its orientation and only a passing thing (a piece held, the dots of a
     * selected piece) keeps the read from being clean.
     */
    private fun unreadReason(frame: Mat, geometry: BoardGeometry): String? {
        val store = cache ?: return null
        // The two square colours cannot be told apart: not a board to read.
        val sampled = com.sidecoach.vision.ThemeCache.samplePalette(frame, geometry) ?: return null
        for ((_, cached) in store.candidates(sampled.first, sampled.second)) {
            val readings = cached.classifyFrame(frame, geometry, embedder)
            val told = Coordinates.readBottomSide(frame, geometry, cached.rankLabels) ?: forcedBottom
            for (side in if (told != null) listOf(told) else listOf(Side.WHITE, Side.BLACK)) {
                if (adoptPlacement(loosePlacement(bySquare(readings, side))) != null) {
                    return if (told != null) null else "orientation"
                }
            }
        }
        return "unlearned"
    }

    internal class ColdRead(
        val model: ThemeModel,
        val bottom: Side,
        val readings: Readings,
        val placement: Board,
    )

    /**
     * A board is adopted only once every read that confirms it shows the same view: the same
     * pieces and the same highlighted squares. A site's move lights its squares before the
     * piece leaves, so a read taken as the next puzzle's setup move began matched the one
     * before it square for square; the board was adopted with the piece still on its origin,
     * the highlight said the other side had just moved, and the setup move that followed could
     * not be explained (Tai's desktop, 2026-09-24: 1...Qxh7 2.Qb1 invented, dump
     * 20260924-165745).
     */
    private fun view(read: ColdRead): View =
        read.placement.boardFen() to highlightedSquares(bySquare(read.readings, read.bottom))

    /**
     * A game joined from the middle, without being asked. With the theme in the cache the
     * board reads cold; the position is adopted once it has read the same, quiet and complete,
     * on [COLD_START_HOLD] tries [COLD_START_EVERY_S] apart, so a board mid-animation or a page
     * passing by is never taken for a game. A start position is left to the bootstrap.
     *
     * Pixels that have not changed since the last read, on the same lock with the same side
     * told, read the same, so that read stands for this try: a board left unread -- a puzzle
     * of a set not learned -- costs nothing while it sits there. New pixels drop it
     * ([process]), on a tick that tries or not. A board that stays unread is explained once
     * ([coldNotice]).
     */
    private fun autoColdStart(frame: Mat, geometry: BoardGeometry, ts: Double): MutableList<Event> {
        if (cache == null) return mutableListOf()
        val tried = game.coldTried
        if (tried != null && ts - tried < COLD_START_EVERY_S) return mutableListOf()
        game.coldTried = ts
        val lock = checkNotNull(board)
        val seen = Triple(lock.region, Math.round(lock.cellPx * 100.0) / 100.0, forcedBottom)
        val last = game.coldLast
        val read = if (last == null || last.first != seen) {
            coldRead(frame, geometry, null, fallback = forcedBottom).also { game.coldLast = seen to it }
        } else {
            last.second
        }
        // Nothing read cleanly this time; the candidate stands until it expires.
        if (read == null) return coldNotice(frame, geometry, ts)
        game.coldSince = null
        val view = view(read)
        val candidate = game.coldCandidate
        game.coldCandidate = if (candidate != null && candidate.first == view &&
            ts - candidate.third <= COLD_START_EXPIRE_S
        ) {
            Triple(view, candidate.second + 1, ts)
        } else {
            val behind = game.coldCandidateRead
            if (candidate != null && behind != null) {
                // Another board read moments ago: a new puzzle's first board, perhaps, before the
                // site's opening move ([Reconciler.startFrom]'s `before`); the last few kept
                // ([startAmong]).
                game.coldBoards = game.coldBoards.takeLast(FLIP_BOARDS_KEPT - 1) + (behind to candidate.third)
            }
            Triple(view, 1, ts)
        }
        game.coldCandidateRead = read
        if (game.coldCandidate!!.second < COLD_START_HOLD) return mutableListOf()
        game.coldCandidate = null
        val before = startAmong(game.coldBoards, read, ts)
        game.coldBoards = emptyList()
        game.coldCandidateRead = null
        val events = adoptCold(read, ts, before)
        if (events.isNotEmpty()) learnLabels(frame, geometry, told = false)
        return events
    }

    /**
     * Recognise the board cold from the theme cache and begin a game from the position on it --
     * a puzzle or a mid-game board, with no move-0 bootstrap.
     */
    fun coldStart(
        frame: Mat,
        geometry: BoardGeometry,
        bottom: Side? = null,
        ts: Double? = null,
    ): MutableList<Event> {
        val read = coldRead(frame, geometry, bottom, asked = true) ?: return mutableListOf()
        return adoptCold(read, ts)
    }

    /** `internal` for the test that a board turned round forgets the clock. */
    internal fun adoptCold(read: ColdRead, ts: Double?, before: Pair<Board, Set<Int>>? = null): MutableList<Event> {
        val reconciler = Reconciler(bottom = read.bottom, userSide = read.bottom)
        // The site's last-move highlight, for what a cold read cannot know: a pawn that has
        // just advanced two squares, and whose move it is ([lastMoveSeen]).
        val highlighted = highlightedSquares(bySquare(read.readings, read.bottom))
        val events = reconciler.startFrom(read.placement, ts, highlighted, before)
        if (events.isEmpty()) return mutableListOf()  // not a legal position; adopt nothing
        // Only now, with a legal position begun, is the pipeline switched onto this theme.
        if (read.bottom != bottom) {
            // The clock read the other way up named the other side as running, and the side to
            // move a cold read leaves open was settled from it on the next tick (round 2,
            // finding 3): read afresh, as a new recogniser's clock is.
            forgetClock()
        }
        model = read.model
        bottom = read.bottom
        game.coldSince = null
        game.coldNoticed = false
        game.lookLost = null
        game.readingsValue = read.readings
        game.readingsKey = board?.let { it.region to Math.round(it.cellPx * 100.0) / 100.0 }
        this.reconciler = reconciler
        return events
    }

    /**
     * The classified board as a bare placement, or null unless it is a fully quiet, fully
     * readable position with both kings on it.
     *
     * A cold read has no move history to lean on, so it must not adopt a board caught mid-move:
     * a lifted piece empties its square and floats over another with the drag's hover outline,
     * so a single frame read then would drop or misplace a piece. Every square must therefore
     * be a settled piece or a settled empty with no hover and no unreadable cell.
     *
     * [asked]: the player asked for this read ("Re-read the screen"), so a mark on an empty
     * square -- a legal-move dot around a piece left selected -- is the empty square it is
     * ([Pieces.markOnEmpty]), as long as nothing on the board is being dragged. Reads nobody
     * asked for keep treating it as no evidence: a phone may draw no outline around a dragged
     * piece, and a drag held still must not be adopted.
     */
    private fun placementFromReadings(readings: Readings, bottom: Side, asked: Boolean = false): Board? {
        if (asked && readings.any { it.hover }) return null
        val board = Board(null)
        for (row in 0 until 8) {
            for (col in 0 until 8) {
                val reading = readings[row * 8 + col]
                if (asked && Pieces.markOnEmpty(reading)) continue
                if (reading.hover || reading.kind == Kind.UNKNOWN || !reading.settled) return null
                if (reading.kind == Kind.PIECE) {
                    val symbol = reading.symbol ?: return null  // no known kind: not a clean read
                    board.setPieceAt(
                        Squares.squareAt(row, col, bottom.color), Piece.fromSymbol(symbol),
                    )
                }
            }
        }
        if (board.king(WHITE) == null || board.king(BLACK) == null) return null
        return board
    }

    private fun bootstrap(frame: Mat, geometry: BoardGeometry): MutableList<Event> {
        val previous = reconciler
        if (!rebuildModel(frame, geometry)) return mutableListOf()
        val side = bottom!!
        if (previous != null && !previous.inProgress && !previous.finished) {
            // The game has not begun: the board was flipped, or re-themed, while the start
            // position was up. The same game goes on, seen from the other side.
            return previous.reorient(side)
        }
        val events = endInProgress("new game started")
        reconciler = Reconciler(bottom = side, userSide = side)
        return events
    }
}
