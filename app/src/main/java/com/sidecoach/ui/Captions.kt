package com.sidecoach.ui

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.graphics.Rect
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.Settings
import android.util.TypedValue
import android.view.GestureDetector
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import com.sidecoach.MainActivity
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The coach's words in a box over other apps, the way Live Caption shows speech (Tai,
 * 2026-09-23: "behave like Live Caption"; the phone's own, the desktop has none). While you
 * play in another app the voice is all the panel gives you; this puts its words on screen too.
 *
 * The box comes up when the coach starts to speak and its words appear as they are said,
 * newest at the bottom, the earlier ones scrolling up; it fades a few seconds after the voice
 * stops. A line the voice does not say (muted, or a fast game) appears whole. Touch and hold
 * to move it anywhere; double-tap to see more lines; drag the bar at its foot for more or
 * fewer; tap for its buttons; drag it onto the cross at the bottom of the screen to turn
 * captions off. Taps anywhere else go to the app underneath. It keeps off the chess board --
 * the vision reads the board from the screen, and a box over it would hide squares -- and it
 * is not shown while Sidecoach's own panel is in front, which shows the words already.
 *
 * Everything runs on the main thread; the public calls are safe from any.
 */
object Captions {
    private const val PREFS = "captions"
    private const val TEXT_SP = 17f
    private const val LINES = 2
    private const val EXPANDED_LINES = 6
    private const val MAX_LINES = 10
    /** How long the box stays after the voice stops, as Live Caption's does. */
    private const val HIDE_AFTER_MS = 5000L
    private const val CONTROLS_MS = 4000L
    private const val TICK_MS = 80L
    private const val KEEP_CHARS = 1200

    private val main = Handler(Looper.getMainLooper())
    private var app: Context? = null

    @Volatile private var on = false
    @Volatile private var front = false

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
        }
    }

    /** Sidecoach's own panel in front (true) or not: its page shows the words already. */
    fun panelInFront(inFront: Boolean) {
        front = inFront
        if (inFront) main.post { drop() }
    }

    /** A line the voice starts to say now, and about how long it speaks for. */
    fun say(text: String, speechMs: Long) {
        if (!on || front) return
        main.post { begin(text, speechMs) }
    }

    /** A line the voice will not say (muted, a fast game): shown whole. */
    fun show(text: String) {
        if (!on || front) return
        main.post { begin(text, 0) }
    }

    // ---- the words ---------------------------------------------------------------------------

    private fun begin(text: String, speechMs: Long) {
        if (!on || front) return
        val fresh = box == null
        finish()
        if (said.isNotEmpty()) said.append(if (fresh) "\n" else " ")
        trim()
        // The coach writes a dash as two hyphens; read, it is a dash.
        saying = text.trim().replace(" -- ", " — ")
        sayingFrom = SystemClock.uptimeMillis()
        sayingMs = speechMs
        if (!ensureBox()) {
            finish()
            return
        }
        main.removeCallbacks(hideLater)
        main.removeCallbacks(tick)
        tick.run()
    }

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

    private fun hideSoon() {
        main.removeCallbacks(hideLater)
        main.postDelayed(hideLater, HIDE_AFTER_MS)
    }

    /** A touch on the box ends: the fade waits its full time again, once the words are out. */
    private fun rest() {
        if (saying.isEmpty()) hideSoon()
    }

    // ---- the window --------------------------------------------------------------------------

    private fun ensureBox(): Boolean {
        box?.let { return true }
        val ctx = app ?: return false
        if (!Settings.canDrawOverlays(ctx)) return false
        return runCatching {
            val made = Box(ctx)
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

    /** Whether captions are on. */
    internal val enabled: Boolean get() = on

    private fun drop() {
        main.removeCallbacks(tick)
        main.removeCallbacks(hideLater)
        finish()
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

    /** The caption box: the words, the buttons a tap shows, and the bar that sets its height. */
    @SuppressLint("ViewConstructor", "ClickableViewAccessibility")
    private class Box(private val ctx: Context) : LinearLayout(ctx) {
        private val windows = ctx.getSystemService(WindowManager::class.java)
        private val params: WindowManager.LayoutParams
        private val words = TextView(ctx)
        private val buttons = LinearLayout(ctx)
        private val mute = chip("Mute voice")
        private val bar = FrameLayout(ctx)
        private var lines = prefs(ctx).getInt("lines", LINES).coerceIn(1, MAX_LINES)
        private val lineHeight: Int
        private var attached = false

        /** A finger on the box: it neither fades nor moves on its own meanwhile. */
        var busy = false
            private set

        private var moving = false
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
                setColor(0xEB1F1F1F.toInt())
            }
            elevation = dp(ctx, 6f).toFloat()

            buttons.orientation = HORIZONTAL
            buttons.visibility = GONE
            buttons.addView(chip("Open Sidecoach").apply { setOnClickListener { openApp() } })
            buttons.addView(mute.apply { setOnClickListener { toggleMute() } })
            addView(buttons, LayoutParams(LayoutParams.WRAP_CONTENT, LayoutParams.WRAP_CONTENT).apply {
                bottomMargin = dp(ctx, 10f)
            })

            words.setTextColor(0xFFFFFFFF.toInt())
            words.setTextSize(TypedValue.COMPLEX_UNIT_SP, TEXT_SP)
            words.setLineSpacing(0f, 1.15f)
            lineHeight = words.lineHeight
            addView(words, LayoutParams(LayoutParams.MATCH_PARENT, lineHeight * shownLines()))

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
            val saved = prefs(ctx)
            val x = saved.getInt("x", screen.left + (screen.width() - width) / 2)
            val y = saved.getInt("y", screen.bottom - dp(ctx, 220f))
            val (cx, cy) = clear(x, y, width, boxHeight(), screen, board, dp(ctx, 8f))
            params.x = cx
            params.y = cy

            // More lines or the buttons make the box taller. Low on the screen it grows upward, its
            // foot where it was, as Live Caption's does; it stays on the screen either way.
            addOnLayoutChangeListener { _, _, top, _, bottom, _, oldTop, _, oldBottom ->
                val grew = (bottom - top) - (oldBottom - oldTop)
                if (grew == 0 || oldBottom == oldTop || moving) return@addOnLayoutChangeListener
                val screen = screen(ctx)
                var y = params.y
                if (y + (oldBottom - oldTop) / 2 > screen.centerY()) y -= grew
                y = y.coerceIn(screen.top, max(screen.top, screen.bottom - (bottom - top)))
                if (y != params.y) {
                    params.y = y
                    windows.updateViewLayout(this, params)
                }
            }

            val gestures = GestureDetector(ctx, object : GestureDetector.SimpleOnGestureListener() {
                override fun onDown(e: MotionEvent) = true
                override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                    toggleButtons()
                    return true
                }
                override fun onDoubleTap(e: MotionEvent): Boolean {
                    expanded = !expanded
                    setHeightTo(shownLines())
                    return true
                }
                override fun onLongPress(e: MotionEvent) {
                    moving = true
                    performHapticFeedback(HapticFeedbackConstants.LONG_PRESS)
                    target = Target(ctx).also { it.attach() }
                }
            })
            setOnTouchListener { _, event ->
                when (event.actionMasked) {
                    MotionEvent.ACTION_DOWN -> {
                        busy = true
                        downX = event.rawX
                        downY = event.rawY
                        startX = params.x
                        startY = params.y
                    }
                    MotionEvent.ACTION_MOVE -> if (moving) {
                        params.x = startX + (event.rawX - downX).roundToInt()
                        params.y = startY + (event.rawY - downY).roundToInt()
                        windows.updateViewLayout(this, params)
                        target?.near(centre())
                    }
                    MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                        busy = false
                        if (moving) {
                            moving = false
                            drop(centre())
                        } else {
                            rest()
                        }
                    }
                }
                gestures.onTouchEvent(event)
                true
            }
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

        private fun boxHeight() = paddingTop + paddingBottom + lineHeight * shownLines() + dp(ctx, 20f)

        private fun centre(): Pair<Int, Int> = (params.x + width / 2) to (params.y + height / 2)

        fun bounds(): Rect = Rect(params.x, params.y, params.x + width, params.y + height)

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

        fun setText(text: String) {
            words.text = text
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
        }

        private fun toggleButtons() {
            val show = buttons.visibility != VISIBLE
            mute.text = if (PanelVoice.muted) "Unmute voice" else "Mute voice"
            buttons.visibility = if (show) VISIBLE else GONE
            main.removeCallbacks(hideButtons)
            if (show) main.postDelayed(hideButtons, CONTROLS_MS)
            rest()
        }

        private val hideButtons = Runnable { buttons.visibility = GONE }

        private fun openApp() {
            runCatching {
                ctx.startActivity(
                    Intent(ctx, MainActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_REORDER_TO_FRONT),
                )
            }
        }

        private fun toggleMute() {
            val now = !PanelVoice.muted
            Thread({ runCatching { PanelVoice.setMuted(now) } }, "captions-mute").start()
            mute.text = if (now) "Unmute voice" else "Mute voice"
            main.removeCallbacks(hideButtons)
            main.postDelayed(hideButtons, CONTROLS_MS)
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

        private fun drop(at: Pair<Int, Int>) {
            val bin = target
            target = null
            bin?.detach()
            if (bin != null && bin.over(at)) {
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
            val screen = screen(ctx)
            params.x = screen.left + (screen.width() - room) / 2
            params.y = screen.bottom - room - dp(ctx, 52f)
            params.flags = params.flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
            centreX = params.x + room / 2
            centreY = params.y + room / 2
        }

        fun attach() {
            alpha = 0f
            windows.addView(this, params)
            animate().alpha(1f).setDuration(150).start()
        }

        fun detach() {
            runCatching { windows.removeView(this) }
        }

        fun over(at: Pair<Int, Int>): Boolean = hypot((at.first - centreX).toDouble(), (at.second - centreY).toDouble()) < dp(ctx, 110f)

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
