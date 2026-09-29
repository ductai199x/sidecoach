package com.sidecoach.ui

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.text.style.ReplacementSpan
import android.text.style.StyleSpan
import android.util.TypedValue
import android.view.GestureDetector
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.sidecoach.R
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The coach's words in a box over other apps, the way Live Caption shows speech (Tai,
 * 2026-09-23: "behave like Live Caption"; the desktop's `routes/captions` is its port). While you
 * play in another app the voice is all the panel gives you; this puts its words on screen too.
 *
 * The box comes up with the coach's words the moment the coach has them, newest at the bottom,
 * the earlier ones scrolling up; the voice follows as soon as the phone has made it, and the box
 * stays up while it speaks (Tai, 2026-09-25: "just output the text asap and then the voice can
 * come whenever" -- the words used to wait for their audio, seconds on the phone). It fades a
 * few seconds after, or stays up with the last words when the setting keeps it on screen.
 * Touch and hold to move it anywhere; double-tap to see more lines; drag the bar at its foot
 * for more or fewer; tap for the game's moves. Its buttons -- mute, and the panel's three quick
 * pills -- are always up (Tai, 2026-09-29: "they are handy"). Drag it onto the cross at the
 * bottom of the screen to turn captions off. Taps
 * anywhere else go to the app underneath. It keeps off the chess board -- the vision reads the
 * board from the screen, and a box over it would hide squares -- and it is not shown while
 * Sidecoach's own panel is in front, which shows the words already.
 *
 * Everything runs on the main thread; the public calls are safe from any.
 */
object Captions {
    private const val PREFS = "captions"
    private const val LINES = 2
    private const val EXPANDED_LINES = 6
    private const val MAX_LINES = 10
    /** How long the box stays after the voice stops, as Live Caption's does. */
    internal const val HIDE_AFTER_MS = 5000L

    /** How long the moves a tap shows stay, after the tap or the last touch on them. */
    internal const val CONTROLS_MS = 4000L

    /**
     * [HIDE_AFTER_MS] and [CONTROLS_MS] as the box keeps them. Only a test changes them: one
     * about the box under a finger, not about its fade, whose slow emulator took longer than
     * the five seconds between showing the words and touching them, and touched the home
     * screen the faded box had left (CaptionsGestureTest, 2026-09-27).
     */
    @Volatile internal var hideAfterMs = HIDE_AFTER_MS
    @Volatile internal var controlsMs = CONTROLS_MS
    private const val TICK_MS = 80L
    private const val KEEP_CHARS = 1200

    private val main = Handler(Looper.getMainLooper())
    private var app: Context? = null

    @Volatile private var on = false
    @Volatile private var front = false

    /** Sidecoach quit: no box until its panel is opened again ([quit]). */
    @Volatile private var quitted = false
    @Volatile private var keep = false

    /** The box up now is the settings' sample line, which fades whatever [keep] says. */
    private var previewing = false

    /**
     * How the box looks, from the settings (Tai, 2026-09-23: the user tunes the text colour,
     * the background colour, its opacity and the text size).
     */
    data class Style(
        val text: Int = 0xFFFFFFFF.toInt(),
        val background: Int = 0xFF1F1F1F.toInt(),
        val opacityPercent: Int = 92,
        val sizeSp: Float = 17f,
    ) {
        /** The background with its opacity. */
        val fill: Int get() = ((opacityPercent.coerceIn(0, 100) * 255 / 100) shl 24) or (background and 0xFFFFFF)
    }

    @Volatile private var style = Style()

    /** The settings' look, applied now to a box that is up and to every one after. */
    fun style(look: Style) {
        style = look
        main.post { box?.restyle(look) }
    }

    /** Where the board is on the screen, in pixels, from the vision; null when none is seen. */
    @Volatile var board: Rect? = null

    // The words: what has been said, and the line being said now.
    private val said = StringBuilder()
    private var saying = ""
    private var sayingFrom = 0L
    private var sayingMs = 0L

    private var box: Box? = null
    private var target: Target? = null
    private var expanded = false

    /** The setting: captions on or off. [ask] opens Android's page to allow the overlay. */
    fun configure(context: Context, enabled: Boolean, ask: Boolean = false) {
        app = context.applicationContext
        on = enabled
        main.post {
            if (!enabled) {
                drop()
                return@post
            }
            val ctx = app ?: return@post
            if (ask && !Settings.canDrawOverlays(ctx)) {
                runCatching {
                    ctx.startActivity(
                        Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:${ctx.packageName}"))
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
                    )
                }
            }
            restore()
        }
    }

    /**
     * The setting: the box stays on screen between the coach's lines, with the last words in it
     * (Tai, 2026-09-24: "the caption box is always on screen if true and if false then fade away
     * after some time"). Off, it fades [HIDE_AFTER_MS] after the voice stops.
     */
    fun keep(stay: Boolean) {
        keep = stay
        main.post {
            if (stay) {
                main.removeCallbacks(hideLater)
                restore()
            } else if (front) {
                drop()
            } else if (box != null && saying.isEmpty()) {
                hideSoon()
            }
        }
    }

    /** Sidecoach's own panel in front (true) or not: its page shows the words already. */
    fun panelInFront(inFront: Boolean) {
        front = inFront
        if (inFront) quitted = false
        main.post { if (inFront && !keep) drop() else restore() }
    }

    /**
     * Sidecoach quit -- its panel finished, or swiped from the recents: the box goes with it,
     * kept on screen or not, and stays gone, words or not, until the panel is opened again
     * (Tai, 2026-09-29: "the live caption box doesn't automatically disappear/close when I quit the app on mobile"; always on screen had
     * brought it back as the panel went).
     */
    fun quit() {
        quitted = true
        main.post { drop() }
    }

    /**
     * Always on screen: the box up, with the words said so far or none yet, whatever is in
     * front (Tai, 2026-09-29: "is it always on screen or not????").
     */
    private fun restore() {
        if (!keep || !on || quitted || box != null) return
        if (ensureBox()) box?.setText(said.toString())
    }

    /**
     * The voice has started on a line already shown: the box stays up while it speaks, [speechMs]
     * about, then fades as it would (or stays, kept on screen).
     */
    fun voiced(speechMs: Long) {
        if (!on || front) return
        main.post {
            if (box == null || saying.isNotEmpty()) return@post
            hideSoon(speechMs)
        }
    }

    /** The voice muted or not, wherever that was said: the box's speaker follows. */
    fun muted(on: Boolean) {
        main.post { box?.showMuted(on) }
    }

    /**
     * A line of the coach's, shown whole the moment it comes: its voice follows. While the panel
     * is in front, which shows it already, it is kept for when the panel is not: a box kept on
     * screen comes back with it (Tai, 2026-09-29: "I don't see shit on my screen" -- an answer
     * asked for on the panel, the panel left, and the kept box had nothing to show). Each
     * message starts a line of its own; [continues], a later sentence of a reply written sentence
     * by sentence, runs on from the one before (Tai, 2026-09-29: "the text do not have any new
     * lines what so ever so it's always really hard to read").
     */
    fun show(text: String, continues: Boolean = false) {
        if (!on) return
        main.post { begin(text, 0, continues = continues) }
    }

    /**
     * A sample line over whatever is in front, Sidecoach's own settings page included: a change
     * to the captions shows the moment it is saved (Tai, 2026-09-23: "applies instantly on
     * save"). It starts a fresh box, and fades as any line does.
     */
    fun preview() {
        if (!on) return
        main.post {
            drop()
            said.setLength(0)
            begin(PREVIEW, 0, force = true)
        }
    }

    private const val PREVIEW = "This is how captions look. Spot on -- rooks belong on open files."

    // ---- the words ---------------------------------------------------------------------------

    private fun begin(text: String, speechMs: Long, force: Boolean = false, continues: Boolean = false) {
        if (!on || quitted) return
        if (front && !force && !keep) {
            if (!previewing) {
                finish()
                if (said.isNotEmpty()) said.append(if (continues) " " else "\n")
                said.append(line(text))
                trim()
            }
            return
        }
        previewing = force
        finish()
        if (said.isNotEmpty()) said.append(if (continues) " " else "\n")
        trim()
        saying = line(text)
        sayingFrom = SystemClock.uptimeMillis()
        sayingMs = speechMs
        // A preview over Sidecoach's own page lets every tap through, Save included.
        if (!ensureBox(touchable = !force)) {
            finish()
            return
        }
        main.removeCallbacks(hideLater)
        main.removeCallbacks(tick)
        box?.let { if (it.movesShown) it.showMoves() }
        tick.run()
    }

    /**
     * A move or a square as the coach writes it (SAN, castling, a bare square, or a square or a
     * pawn move a sentence starts with a capital: "G4 was a miss"), and a piece named: the words
     * a player looks for, set off in the box -- the moves and squares in [accent] and bold, the
     * pieces bold (Tai, 2026-09-29: "why are the pieces and their names and square names and SAN
     * annotations not highlighted"). The desktop's `captions.runs`.
     */
    private const val MOVE = """(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|[A-H](?:x[a-h])?[1-8])[+#]?"""
    private const val PIECE = """(?:[Kk]ing|[Qq]ueen|[Rr]ook|[Bb]ishop|[Kk]night|[Pp]awn)s?"""
    private val TOKENS = Regex("""(?<![A-Za-z0-9])(?:($MOVE)|($PIECE))(?![A-Za-z0-9])""")

    internal fun styled(text: String, fill: Int): CharSequence {
        val out = SpannableStringBuilder(text)
        val colour = accent(fill)
        for (m in TOKENS.findAll(text)) {
            val from = m.range.first
            val to = m.range.last + 1
            out.setSpan(StyleSpan(Typeface.BOLD), from, to, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
            if (m.groups[1] != null) out.setSpan(ForegroundColorSpan(colour), from, to, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        return out
    }

    /** The colour moves and squares stand out in, against the box's [fill]: a warm gold on a
     * dark box, a deep amber on a light one (the desktop's `captions.accent`). */
    internal fun accent(fill: Int): Int {
        val luma = 0.2126 * ((fill shr 16) and 255) + 0.7152 * ((fill shr 8) and 255) + 0.0722 * (fill and 255)
        return if (luma < 140) 0xFFFFD479.toInt() else 0xFF8A5300.toInt()
    }

    /** The coach writes a dash as two hyphens; read, it is a dash. */
    private fun line(text: String): String = text.trim().replace(" -- ", " — ")

    /** The line being said, all of it, into what has been said. */
    private fun finish() {
        if (saying.isEmpty()) return
        said.append(saying)
        saying = ""
        main.removeCallbacks(tick)
    }

    private fun trim() {
        if (said.length <= KEEP_CHARS) return
        val cut = said.indexOf(" ", said.length - KEEP_CHARS).let { if (it < 0) said.length - KEEP_CHARS else it + 1 }
        said.delete(0, cut)
    }

    /** How much of the line is out by now: whole words, at the pace the voice speaks them. */
    internal fun revealed(text: String, elapsedMs: Long, speechMs: Long): Int {
        if (speechMs <= 0 || elapsedMs >= speechMs) return text.length
        val upTo = (text.length * elapsedMs / speechMs).toInt()
        if (upTo <= 0) return 0
        val end = text.indexOf(' ', upTo)
        return if (end < 0) text.length else end
    }

    private val tick: Runnable = object : Runnable {
        override fun run() {
            val view = box ?: return
            val n = revealed(saying, SystemClock.uptimeMillis() - sayingFrom, sayingMs)
            view.setText(said.toString() + saying.substring(0, n))
            if (n < saying.length) {
                main.postDelayed(this, TICK_MS)
            } else {
                finish()
                hideSoon()
            }
        }
    }

    private val hideLater = Runnable { hide() }

    private fun hideSoon(speaking: Long = 0) {
        main.removeCallbacks(hideLater)
        if (keep) return
        main.postDelayed(hideLater, speaking + hideAfterMs)
    }

    /** A touch on the box ends: the fade waits its full time again, once the words are out. */
    private fun rest() {
        if (saying.isEmpty()) hideSoon()
    }

    // ---- the window --------------------------------------------------------------------------

    private fun ensureBox(touchable: Boolean = true): Boolean {
        box?.let { return true }
        val ctx = app ?: return false
        if (!Settings.canDrawOverlays(ctx)) return false
        return runCatching {
            val made = Box(ctx, touchable)
            made.attach()
            box = made
            true
        }.getOrElse { false }
    }

    private fun hide() {
        val view = box ?: return
        if (view.busy) {
            hideSoon()
            return
        }
        view.fadeOut { if (box === view) drop() }
    }

    /** Where the box is on the screen now, for the tests; null when it is not shown. */
    internal fun shownAt(): Rect? = box?.bounds()

    /** Where the box's window really is on the screen, for the tests. */
    internal fun windowAt(): Rect? = box?.onScreen()

    /**
     * Whether the box has been drawn and has faded fully in, for the tests: until then the
     * system has no picture of it to send touches to, and a tap lands on what is under it.
     */
    internal fun drawn(): Boolean = box?.let { it.frames > 0 && it.alpha >= 1f } ?: false

    /** Where each of the box's buttons is on the screen, by name, for the tests. */
    internal fun buttonsAt(): Map<String, Rect> = box?.buttonBounds() ?: emptyMap()

    /** Whether the game's moves are up under the words, for the tests. */
    internal fun movesUp(): Boolean = box?.movesShown ?: false

    /** The words the box shows, for the tests; null when it is not shown. */
    internal fun wordsShown(): String? = box?.shownText()

    /** Whether captions are on. */
    internal val enabled: Boolean get() = on

    private fun drop() {
        main.removeCallbacks(tick)
        main.removeCallbacks(hideLater)
        finish()
        // The sample line is no word of the coach's: a box kept on screen does not bring it back.
        if (previewing) said.setLength(0)
        previewing = false
        box?.detach()
        box = null
        target?.detach()
        target = null
    }

    /** Captions off from the box itself: the setting follows, as the page's switch would. */
    private fun turnOff() {
        on = false
        drop()
        val ctx = app ?: return
        Thread({
            runCatching {
                val store = SettingsStore(ctx)
                val values = store.values()
                values.optJSONObject("voice")?.put("captions", false)
                store.save(values)
                PanelEngine.emit("sys", "info", "captions off; Settings > The coach's voice turns them on again")
            }
        }, "captions-off").start()
    }

    private fun prefs(ctx: Context) = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    /** The screen, in pixels. */
    private fun screen(ctx: Context): Rect = ctx.getSystemService(WindowManager::class.java).currentWindowMetrics.bounds

    /**
     * The cross a box is carried onto to turn captions off: its centre this far above the foot of
     * the screen, and how near it the finger must be let go --
     * the finger, not the box. The box's centre near a cross 100 dp up made every box parked low
     * a dismissal, whatever the reach (Tai, 2026-09-29: "sometime I need to reposition it but I
     * can't since I would accidentally dismiss it"; then, the reach cut from 110 to 56 dp, "the
     * dismiss thing is STILL TOO FAR UP. It's like nothing have changed whatsoever!").
     */
    private const val CROSS_LIFT_DP = 40f
    private const val CROSS_REACH_DP = 44f

    /** The centre of the cross a box is dropped on to turn captions off, at the screen's foot. */
    internal fun crossCentre(ctx: Context): Pair<Int, Int> {
        val screen = screen(ctx)
        return screen.centerX() to screen.bottom - dp(ctx, CROSS_LIFT_DP)
    }

    /**
     * Where a box of [w] x [h] at ([x], [y]) goes so it does not cover the board: above it or
     * below it, whichever is nearer and has the room, else the side with more room. Kept on
     * the screen either way.
     */
    internal fun clear(x: Int, y: Int, w: Int, h: Int, screen: Rect, board: Rect?, margin: Int): Pair<Int, Int> {
        val cx = x.coerceIn(screen.left, max(screen.left, screen.right - w))
        var cy = y.coerceIn(screen.top, max(screen.top, screen.bottom - h))
        if (board != null) {
            val over = Rect(cx, cy, cx + w, cy + h)
            val guard = Rect(board.left - margin, board.top - margin, board.right + margin, board.bottom + margin)
            if (Rect.intersects(over, guard)) {
                val above = guard.top - h
                val below = guard.bottom
                val roomAbove = above >= screen.top
                val roomBelow = below + h <= screen.bottom
                val middle = cy + h / 2
                cy = when {
                    roomAbove && roomBelow -> if (middle < board.centerY()) above else below
                    roomAbove -> above
                    roomBelow -> below
                    guard.top - screen.top >= screen.bottom - guard.bottom -> screen.top
                    else -> screen.bottom - h
                }
            }
        }
        return cx to cy
    }

    private fun dp(ctx: Context, value: Float): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value, ctx.resources.displayMetrics).roundToInt()

    private fun overlayParams(w: Int, h: Int): WindowManager.LayoutParams = WindowManager.LayoutParams(
        w,
        h,
        WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
        WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
            WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL or
            WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
        PixelFormat.TRANSLUCENT,
    ).apply { gravity = Gravity.TOP or Gravity.START }

    /** The caption box: its buttons, the words, the moves a tap shows, and the bar that sets its
     * height. */
    @SuppressLint("ViewConstructor", "ClickableViewAccessibility")
    private class Box(private val ctx: Context, touchable: Boolean = true) : LinearLayout(ctx) {
        private val windows = ctx.getSystemService(WindowManager::class.java)
        private val params: WindowManager.LayoutParams
        private val words = TextView(ctx)
        private val buttons = LinearLayout(ctx)
        private val mute = ImageView(ctx)
        private val moves = TextView(ctx)
        private val strip = HorizontalScrollView(ctx)
        private val bar = FrameLayout(ctx)
        private var lines = prefs(ctx).getInt("lines", LINES).coerceIn(1, MAX_LINES)
        private var lineHeight: Int
        private var attached = false

        /** A finger on the box: it neither fades nor moves on its own meanwhile. */
        var busy = false
            private set

        /** The frames drawn so far (`drawn`, for the tests). */
        var frames = 0
            private set

        override fun dispatchDraw(canvas: android.graphics.Canvas) {
            super.dispatchDraw(canvas)
            frames++
        }

        private var moving = false

        /** The finger has travelled past the touch slop since it came down. */
        private var dragged = false
        private var downX = 0f
        private var downY = 0f
        private var startX = 0
        private var startY = 0

        init {
            orientation = VERTICAL
            val pad = dp(ctx, 16f)
            setPadding(pad, dp(ctx, 12f), pad, dp(ctx, 4f))
            background = GradientDrawable().apply {
                cornerRadius = dp(ctx, 20f).toFloat()
                setColor(style.fill)
            }
            elevation = dp(ctx, 6f).toFloat()

            // Mute, as an icon, then the panel's quick pills (Tai, 2026-09-23): a hint, the best
            // move or the line without leaving the game, the answer spoken and captioned here.
            buttons.orientation = HORIZONTAL
            buttons.gravity = Gravity.CENTER_VERTICAL
            val round = dp(ctx, 34f)
            val inset = dp(ctx, 7f)
            mute.setPadding(inset, inset, inset, inset)
            mute.background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0x33FFFFFF)
            }
            mute.tag = "mute"
            mute.setOnClickListener { toggleMute() }
            showMuted(PanelVoice.muted)
            buttons.addView(mute, LayoutParams(round, round).apply { marginEnd = dp(ctx, 8f) })
            for ((name, label) in listOf("hint" to "Hint", "move" to "Best move", "line" to "Line")) {
                buttons.addView(chip(label).apply {
                    tag = name
                    setOnClickListener { quick(name) }
                })
            }
            addView(buttons, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(ctx, 10f)
            })

            words.setTextColor(style.text)
            words.setTextSize(TypedValue.COMPLEX_UNIT_SP, style.sizeSp)
            words.setLineSpacing(0f, 1.15f)
            lineHeight = words.lineHeight
            addView(words, LayoutParams(LayoutParams.MATCH_PARENT, lineHeight * shownLines()))

            // The game's moves, one line under the words, the newest in view: the panel's move
            // list as the phone lays it out, with chess.com's badges (Tai, 2026-09-23).
            moves.setTextColor(style.text)
            moves.setTextSize(TypedValue.COMPLEX_UNIT_SP, style.sizeSp - 2f)
            moves.isSingleLine = true
            strip.isHorizontalScrollBarEnabled = false
            strip.visibility = GONE
            strip.addView(moves)
            strip.setOnTouchListener { _, event ->
                // Swiping through the moves keeps them up.
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> main.removeCallbacks(hideMoves)
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        main.postDelayed(hideMoves, controlsMs)
                        rest()
                    }
                }
                false
            }
            addView(strip, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT).apply {
                topMargin = dp(ctx, 8f)
            })

            val grip = View(ctx).apply {
                background = GradientDrawable().apply {
                    cornerRadius = dp(ctx, 2f).toFloat()
                    setColor(0x73FFFFFF)
                }
            }
            bar.addView(grip, FrameLayout.LayoutParams(dp(ctx, 32f), dp(ctx, 4f), Gravity.CENTER))
            bar.setOnTouchListener { _, event -> resize(event) }
            addView(bar, LayoutParams(dp(ctx, 72f), dp(ctx, 20f)).apply { gravity = Gravity.CENTER_HORIZONTAL })

            val screen = screen(ctx)
            val width = min(screen.width() - dp(ctx, 32f), max(screen.width() * 4 / 5, dp(ctx, 320f)))
            params = overlayParams(width, WindowManager.LayoutParams.WRAP_CONTENT)
            if (!touchable) params.flags = params.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
            val saved = prefs(ctx)
            val x = saved.getInt("x", screen.left + (screen.width() - width) / 2)
            val y = saved.getInt("y", screen.bottom - dp(ctx, 220f))
            val (cx, cy) = clear(x, y, width, boxHeight(), screen, board, dp(ctx, 8f))
            params.x = cx
            params.y = cy

            val gestures = GestureDetector(ctx, object : GestureDetector.SimpleOnGestureListener() {
                override fun onDown(e: MotionEvent) = true
                override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                    toggleMoves()
                    return true
                }
                override fun onDoubleTap(e: MotionEvent): Boolean {
                    expanded = !expanded
                    setHeightTo(shownLines())
                    return true
                }
                override fun onLongPress(e: MotionEvent) = startMoving()
            })
            // A finger that moves drags the box at once; waiting for a hold first lost the
            // drag whenever a real finger drifted a few pixels before the hold was up (Tai on
            // the Pixel, 2026-09-23: "I can't drag the box"). A tap and a double-tap stay
            // where they are, so they are still told apart.
            val slop = android.view.ViewConfiguration.get(ctx).scaledTouchSlop
            setOnTouchListener { _, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        busy = true
                        dragged = false
                        downX = event.rawX
                        downY = event.rawY
                        startX = params.x
                        startY = params.y
                    }
                    MotionEvent.ACTION_MOVE -> {
                        if (hypot((event.rawX - downX).toDouble(), (event.rawY - downY).toDouble()) > slop) {
                            dragged = true
                            if (!moving) startMoving()
                        }
                        if (moving) {
                            params.x = startX + (event.rawX - downX).roundToInt()
                            params.y = startY + (event.rawY - downY).roundToInt()
                            windows.updateViewLayout(this, params)
                            target?.near(event.rawX.roundToInt() to event.rawY.roundToInt())
                        }
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        busy = false
                        if (moving) {
                            moving = false
                            drop(event.rawX.roundToInt() to event.rawY.roundToInt())
                        } else {
                            rest()
                        }
                    }
                }
                gestures.onTouchEvent(event)
                true
            }
        }

        /** The height the window was last asked for, until it is laid out at it. */
        private var asked = 0

        /**
         * The box's height about to change -- more lines, the moves, a text size.
         * Low on the screen it grows upward, its foot where it was, as Live Caption's does; it
         * stays on the screen either way. The new height is measured here and the new place
         * asked for before the window grows, so the two come in one step: moved once it had
         * grown, the window stayed where it was on the screen while the system sent its touches
         * to the new place, and a tap on the buttons' row landed on the words.
         */
        fun reflow() {
            if (!attached || moving || height == 0) return
            val before = if (asked > 0) asked else height
            measure(
                MeasureSpec.makeMeasureSpec(params.width, MeasureSpec.EXACTLY),
                MeasureSpec.makeMeasureSpec(0, MeasureSpec.UNSPECIFIED),
            )
            val after = measuredHeight
            if (after == before) return
            asked = after
            val screen = screen(ctx)
            var y = params.y
            if (y + before / 2 > screen.centerY()) y -= after - before
            y = y.coerceIn(screen.top, max(screen.top, screen.bottom - after))
            if (y != params.y) {
                params.y = y
                windows.updateViewLayout(this, params)
            }
        }

        override fun onLayout(changed: Boolean, l: Int, t: Int, r: Int, b: Int) {
            super.onLayout(changed, l, t, r, b)
            asked = 0
        }

        private fun startMoving() {
            if (moving) return
            moving = true
            performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
            target = Target(ctx).also { it.attach() }
        }

        private fun chip(label: String) = TextView(ctx).apply {
            text = label
            setTextColor(0xFFFFFFFF.toInt())
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
            val h = dp(ctx, 12f)
            val v = dp(ctx, 7f)
            setPadding(h, v, h, v)
            background = GradientDrawable().apply {
                cornerRadius = dp(ctx, 16f).toFloat()
                setColor(0x33FFFFFF)
            }
            layoutParams = LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
                marginEnd = dp(ctx, 8f)
            }
        }

        private fun shownLines() = if (expanded) max(lines, EXPANDED_LINES) else lines

        /** The box's height before it is laid out: the padding, the buttons' row, the words and
         * the bar. */
        private fun boxHeight() =
            paddingTop + paddingBottom + dp(ctx, 34f + 10f) + lineHeight * shownLines() + dp(ctx, 20f)

        fun bounds(): Rect = Rect(params.x, params.y, params.x + width, params.y + height)

        /** Where the window is on the screen, as the system shows it. */
        fun onScreen(): Rect {
            val at = IntArray(2)
            getLocationOnScreen(at)
            return Rect(at[0], at[1], at[0] + width, at[1] + height)
        }

        fun attach() {
            alpha = 0f
            windows.addView(this, params)
            attached = true
            animate().alpha(1f).setDuration(180).start()
        }

        fun detach() {
            if (!attached) return
            attached = false
            animate().cancel()
            runCatching { windows.removeView(this) }
        }

        fun fadeOut(then: () -> Unit) {
            animate().alpha(0f).setDuration(350).withEndAction(then).start()
        }

        fun shownText(): String = words.text.toString()

        fun setText(text: String) {
            words.text = styled(text, style.fill)
            words.post {
                val layout = words.layout ?: return@post
                val bottom = layout.getLineBottom(max(0, layout.lineCount - 1))
                words.scrollTo(0, max(0, bottom - words.height))
            }
        }

        private fun setHeightTo(n: Int) {
            words.layoutParams = words.layoutParams.apply { height = lineHeight * n }
            words.requestLayout()
            setText(words.text.toString())
            reflow()
        }

        /** The settings' look, on the box that is up. */
        fun restyle(look: Style) {
            (background as GradientDrawable).setColor(look.fill)
            words.setTextColor(look.text)
            words.setTextSize(TypedValue.COMPLEX_UNIT_SP, look.sizeSp)
            moves.setTextColor(look.text)
            moves.setTextSize(TypedValue.COMPLEX_UNIT_SP, look.sizeSp - 2f)
            lineHeight = words.lineHeight
            setHeightTo(shownLines())
        }

        /** The move list, from the panel's state; hidden while the game has no moves. */
        fun showMoves() {
            val (sans, labels) = PanelEngine.moveList()
            if (sans.isEmpty()) {
                strip.visibility = GONE
                reflow()
                return
            }
            moves.text = moveText(sans, labels)
            strip.visibility = VISIBLE
            reflow()
            strip.post { strip.scrollTo(moves.width, 0) }
        }

        private fun moveText(sans: List<String>, labels: Map<String, String>): CharSequence {
            val out = SpannableStringBuilder()
            for ((i, san) in sans.withIndex()) {
                val ply = i + 1
                if (i % 2 == 0) {
                    if (i > 0) out.append("   ")
                    val from = out.length
                    out.append("${i / 2 + 1}. ")
                    out.setSpan(ForegroundColorSpan(0xFF9A9A9A.toInt()), from, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                } else {
                    out.append("  ")
                }
                val from = out.length
                out.append(san)
                if (ply == sans.size) {
                    out.setSpan(BackgroundColorSpan(0x33FFFFFF), from, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    out.setSpan(StyleSpan(Typeface.BOLD), from, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }
                val badge = BADGES[labels[ply.toString()]]
                if (badge != null) {
                    out.append(" ")
                    val at = out.length
                    out.append("\u2009")
                    out.setSpan(Badge(badge.first, badge.second, dp(ctx, 16f)), at, out.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }
            }
            return out
        }

        /** The speaker as the voice stands: struck through while it is muted. */
        fun showMuted(muted: Boolean) {
            mute.setImageResource(if (muted) R.drawable.ic_voice_off else R.drawable.ic_voice_on)
            mute.contentDescription = if (muted) "Unmute the coach's voice" else "Mute the coach's voice"
        }

        fun buttonBounds(): Map<String, Rect> {
            val at = IntArray(2)
            return (0 until buttons.childCount).associate { i ->
                val button = buttons.getChildAt(i)
                button.getLocationOnScreen(at)
                button.tag as String to Rect(at[0], at[1], at[0] + button.width, at[1] + button.height)
            }
        }

        private fun toggleMoves() {
            main.removeCallbacks(hideMoves)
            if (movesShown) {
                strip.visibility = GONE
                reflow()
            } else {
                showMoves()
                main.postDelayed(hideMoves, controlsMs)
            }
            rest()
        }

        private val hideMoves = Runnable {
            strip.visibility = GONE
            reflow()
        }

        /** The moves up now, for a line that comes while they are. */
        val movesShown: Boolean get() = strip.visibility == VISIBLE

        private fun toggleMute() {
            val now = !PanelVoice.muted
            Thread({ runCatching { PanelVoice.setMuted(now) } }, "captions-mute").start()
            showMuted(now)
            rest()
        }

        /** A quick pill: what the panel's own asks (`store.quick`); the box stays its full time. */
        private fun quick(name: String) {
            PanelEngine.ask(PanelApi.QUICK.getValue(name))
            rest()
        }

        private var resizeFrom = 0f
        private var resizeLines = 0

        private fun resize(event: MotionEvent): Boolean {
            when (event.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    busy = true
                    resizeFrom = event.rawY
                    resizeLines = shownLines()
                }
                MotionEvent.ACTION_MOVE -> {
                    val n = (resizeLines + ((event.rawY - resizeFrom) / lineHeight).roundToInt()).coerceIn(1, MAX_LINES)
                    if (n != shownLines()) {
                        lines = n
                        expanded = false
                        setHeightTo(n)
                    }
                }
                MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                    busy = false
                    prefs(ctx).edit().putInt("lines", lines).apply()
                    rest()
                }
            }
            return true
        }

        /** The box let go with the finger at [at]: on the cross, captions off; anywhere else,
         * kept there, off the board. */
        private fun drop(at: Pair<Int, Int>) {
            val bin = target
            target = null
            bin?.detach()
            // Only a box carried onto the cross turns captions off. A box can sit within the
            // cross's reach where it was put, and a touch held without moving -- the start of a
            // move, or a tap a busy phone was late to deliver -- turned them off where it was let
            // go (the gesture test's screenshot, 2026-09-26).
            if (bin != null && dragged && bin.over(at)) {
                turnOff()
                return
            }
            val (x, y) = clear(params.x, params.y, width, height, screen(ctx), board, dp(ctx, 8f))
            params.x = x
            params.y = y
            windows.updateViewLayout(this, params)
            prefs(ctx).edit().putInt("x", x).putInt("y", y).apply()
            rest()
        }
    }

    /**
     * chess.com's move-quality badges, as the panel's move list draws them (`panel.css`): the
     * colour and the mark, for the labels worth one; good and book stay plain.
     */
    private val BADGES = mapOf(
        "brilliant" to (0xFF1BACA6.toInt() to "!!"),
        "great" to (0xFF5B8BB0.toInt() to "!"),
        "best" to (0xFF81B64C.toInt() to "\u2605"),
        "inaccuracy" to (0xFFF7C631.toInt() to "?!"),
        "mistake" to (0xFFFFA459.toInt() to "?"),
        "miss" to (0xFFFF7769.toInt() to "\u2715"),
        "blunder" to (0xFFFA412D.toInt() to "??"),
    )

    /** A badge in the move list: a coloured disc with its mark, the size of a letter. */
    private class Badge(private val colour: Int, private val mark: String, private val size: Int) : ReplacementSpan() {
        private val disc = Paint(Paint.ANTI_ALIAS_FLAG)
        private val ink = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFFFFFFF.toInt()
            typeface = Typeface.DEFAULT_BOLD
            textAlign = Paint.Align.CENTER
        }

        override fun getSize(paint: Paint, text: CharSequence?, start: Int, end: Int, fm: Paint.FontMetricsInt?): Int = size

        override fun draw(
            canvas: Canvas,
            text: CharSequence?,
            start: Int,
            end: Int,
            x: Float,
            top: Int,
            y: Int,
            bottom: Int,
            paint: Paint,
        ) {
            val r = size / 2f
            val cy = (top + bottom) / 2f
            disc.color = colour
            canvas.drawCircle(x + r, cy, r, disc)
            ink.textSize = size * if (mark.length > 1) 0.55f else 0.68f
            val base = cy - (ink.descent() + ink.ascent()) / 2
            canvas.drawText(mark, x + r, base, ink)
        }
    }

    /** The cross at the foot of the screen a box is dropped on to turn captions off. */
    @SuppressLint("ViewConstructor")
    private class Target(private val ctx: Context) : FrameLayout(ctx) {
        private val windows = ctx.getSystemService(WindowManager::class.java)
        private val size = dp(ctx, 56f)

        /** The window has room around the circle for it to grow when a box is over it. */
        private val room = dp(ctx, 96f)
        private val params = overlayParams(room, room)
        private val circle = TextView(ctx)
        private var hot = false
        private val centreX: Int
        private val centreY: Int

        init {
            circle.text = "✕"
            circle.gravity = Gravity.CENTER
            circle.setTextColor(0xFFFFFFFF.toInt())
            circle.setTextSize(TypedValue.COMPLEX_UNIT_SP, 22f)
            circle.background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(0x99000000.toInt())
            }
            addView(circle, LayoutParams(size, size, Gravity.CENTER))
            val (x, y) = crossCentre(ctx)
            centreX = x
            centreY = y
            params.x = x - room / 2
            params.y = y - room / 2
            params.flags = params.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        }

        fun attach() {
            alpha = 0f
            windows.addView(this, params)
            animate().alpha(1f).setDuration(150).start()
        }

        fun detach() {
            runCatching { windows.removeView(this) }
        }

        fun over(at: Pair<Int, Int>): Boolean =
            hypot((at.first - centreX).toDouble(), (at.second - centreY).toDouble()) < dp(ctx, CROSS_REACH_DP)

        fun near(at: Pair<Int, Int>) {
            val now = over(at)
            if (now == hot) return
            hot = now
            if (now) performHapticFeedback(HapticFeedbackConstants.CLOCK_TICK)
            (circle.background as GradientDrawable).setColor(if (now) 0xE6B3261E.toInt() else 0x99000000.toInt())
            circle.animate().scaleX(if (now) 1.3f else 1f).scaleY(if (now) 1.3f else 1f).setDuration(120).start()
        }
    }
}
