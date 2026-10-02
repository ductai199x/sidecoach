package com.sidecoach.ui

import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import android.view.InputDevice
import android.view.MotionEvent
import android.view.ViewConfiguration
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * The caption box under a finger: its touches injected as the screen would send them, the
 * overlay window real. Its buttons -- mute and the panel's quick pills, each doing what it does
 * there -- are up from the start. A double-tap shows more lines and a second one fewer; a tap
 * shows the game's moves, and left alone they go after their four seconds; touch, hold and drag
 * moves the box, and dropping it on the cross turns captions off.
 * Each state is screenshotted into the app's files (`captions/`) for a person to look at.
 *
 * The box's own time is the test's ([StepClock]): nothing fades or times out while the gestures
 * run, and the moves' timeout is checked at its shipped [Captions.CONTROLS_MS] to the
 * millisecond. Android's own gesture timing -- a long press, the double-tap window -- is real,
 * and the test waits for what it decides (the box taken up by a hold, a tap confirmed), not for
 * a guess at how long that takes.
 */
@RunWith(AndroidJUnit4::class)
class CaptionsGestureTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val automation = instrumentation.uiAutomation
    private val shots = File(app.getExternalFilesDir(null), "captions").apply { mkdirs() }

    /** The screenshots' PNGs, written while the test goes on: a dozen of them took seconds. */
    private val writer = Executors.newSingleThreadExecutor()

    private fun touch(action: Int, x: Int, y: Int, down: Long) {
        val event = MotionEvent.obtain(down, SystemClock.uptimeMillis(), action, x.toFloat(), y.toFloat(), 0)
        event.source = InputDevice.SOURCE_TOUCHSCREEN
        automation.injectInputEvent(event, true)
        event.recycle()
    }

    /**
     * [count] taps as a finger makes them: each down and up [TAP_MS] apart and the taps
     * [GAP_MS] apart, stamped with those times as the touchscreen stamps its events, and sent
     * without waiting for the app to handle each, as the touchscreen sends them. Sent one by
     * one, each waiting on the app, a double-tap on a starved emulator went out as two taps
     * 650 ms apart -- the first down handled 255 ms after it was sent -- and Android rightly
     * took them for two single taps: the buttons on, then off (2026-09-27, the box's own log).
     * A finger's double-tap reaches the app 80 ms apart however busy the app is.
     */
    private fun taps(x: Int, y: Int, count: Int) {
        val start = SystemClock.uptimeMillis()
        for (i in 0 until count) {
            val down = start + i * (TAP_MS + GAP_MS)
            send(MotionEvent.ACTION_DOWN, x, y, down, down)
            send(MotionEvent.ACTION_UP, x, y, down, down + TAP_MS)
        }
    }

    private fun tap(x: Int, y: Int) = taps(x, y, 1)

    private fun doubleTap(x: Int, y: Int) = taps(x, y, 2)

    /**
     * One touch event at [at], queued for the screen without waiting for the app; or, [handled],
     * returning once the app has handled it (and so every event queued before it).
     */
    private fun send(action: Int, x: Int, y: Int, down: Long, at: Long, handled: Boolean = false) {
        val event = MotionEvent.obtain(down, at, action, x.toFloat(), y.toFloat(), 0)
        event.source = InputDevice.SOURCE_TOUCHSCREEN
        automation.injectInputEvent(event, handled)
        event.recycle()
    }

    /** The screen as it is now, for a person to look at; its PNG is written meanwhile. */
    private fun shot(name: String) {
        val image = automation.takeScreenshot() ?: return
        writer.execute {
            File(shots, "$name.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
            image.recycle()
            Log.i(TAG, "screenshot $name")
        }
    }

    /**
     * Where the box is, as the screen shows it -- not where it asked to be: the two parted once
     * (a grown box stayed put on the screen while its touches went to the new place), and a
     * test reading the asked-for place passed.
     */
    private fun box(): Rect? {
        var at: Rect? = null
        instrumentation.runOnMainSync { at = Captions.windowAt() }
        return at
    }

    /** The box shown where it asked to be. */
    private fun settled(): Boolean {
        var asked: Rect? = null
        var shown: Rect? = null
        instrumentation.runOnMainSync {
            asked = Captions.shownAt()
            shown = Captions.windowAt()
        }
        return asked != null && asked == shown
    }

    /**
     * The box at rest: what the main thread was given is done (a layout, a fade-in, a drag's
     * last move), and the window is where it asked to be.
     */
    private fun still(what: String) {
        instrumentation.waitForIdleSync()
        assertTrue("$what: the box is not where it asked to be", waitFor { settled() })
    }

    /**
     * Whether [what] comes true within [ms]: a state the box is to reach, never a bound on
     * how soon. Ten seconds, since the box answers on its own thread, and an emulator starved
     * by its host answered a tap three to four seconds late (2026-09-27, its four cores kept
     * busy by sixteen loops); a box that is quick returns at once.
     */
    private fun waitFor(ms: Long = 10_000, what: () -> Boolean): Boolean {
        val until = SystemClock.uptimeMillis() + ms
        while (SystemClock.uptimeMillis() < until) {
            if (what()) return true
            SystemClock.sleep(20)
        }
        return what()
    }

    /**
     * A tap's double-tap window over: the main thread has handled the touches, then Android's
     * double-tap timeout has passed and what it posted for it (a single tap confirmed) has run.
     * Only then can the test say a tap was not taken for half of a double-tap, or the next tap
     * not make one with it.
     */
    private fun windowOver() {
        instrumentation.waitForIdleSync()
        SystemClock.sleep(ViewConfiguration.getDoubleTapTimeout().toLong())
        instrumentation.waitForIdleSync()
    }

    private fun moving(): Boolean {
        var now = false
        instrumentation.runOnMainSync { now = Captions.moving() }
        return now
    }

    /**
     * Hold at [from] until the box is taken up by the hold -- Android's long press, a real
     * finger drifting a few pixels meanwhile when [drift] -- then drag in steps to [to]; [before]
     * runs with the finger still down and the box under it. Without [hold], a plain drag: the
     * finger moves at once.
     */
    private fun holdAndDrag(
        from: Pair<Int, Int>,
        to: Pair<Int, Int>,
        hold: Boolean = true,
        drift: Boolean = false,
        before: () -> Unit = {},
    ) {
        val down = SystemClock.uptimeMillis()
        touch(MotionEvent.ACTION_DOWN, from.first, from.second, down)
        if (hold) {
            var i = 0
            assertTrue("a hold did not take the box up", waitFor {
                if (drift) {
                    i++
                    touch(MotionEvent.ACTION_MOVE, from.first + (i % 3) * 4, from.second + (i % 2) * 5, down)
                }
                moving()
            })
        }
        // The drag at a finger's pace, a frame between moves, stamped with those times as the
        // touchscreen stamps them and queued without waiting on the app, as [taps] sends a tap.
        val steps = 20
        val moved = SystemClock.uptimeMillis()
        for (i in 1..steps) {
            val x = from.first + (to.first - from.first) * i / steps
            val y = from.second + (to.second - from.second) * i / steps
            send(MotionEvent.ACTION_MOVE, x, y, down, moved + i * FRAME_MS)
        }
        instrumentation.waitForIdleSync()
        if (from != to) assertTrue("the box did not follow the finger", waitFor { moving() && settled() })
        before()
        send(MotionEvent.ACTION_UP, to.first, to.second, down, maxOf(SystemClock.uptimeMillis(), moved + (steps + 1) * FRAME_MS), handled = true)
    }

    /** A shell command's output, as the shell user runs it. */
    private fun shell(command: String): String =
        android.os.ParcelFileDescriptor.AutoCloseInputStream(automation.executeShellCommand(command))
            .bufferedReader().use { it.readText().trim() }

    @Test
    fun theBoxAnswersTapsDoubleTapsAndADragToTheCross() {
        // Nothing of the system's over the screen: a crash dialog ("Bluetooth keeps stopping",
        // which the emulators show after a restart under load) sits above every overlay and is
        // touch-modal, and the lock screen hides overlays altogether, so the touches injected
        // for the box went elsewhere and the double-tap "showed no more lines" (2026-09-26, on
        // main too; the screenshots showed the dialog, then after a reboot the lock screen). The
        // screen is woken and unlocked, the dialog closed and new ones kept off for the test;
        // the setting goes back after.
        val power = app.getSystemService(android.os.PowerManager::class.java)
        val keyguard = app.getSystemService(android.app.KeyguardManager::class.java)
        fun awake() = power.isInteractive && !keyguard.isKeyguardLocked
        var asked = 0L
        assertTrue("the screen did not wake and unlock", waitFor(20_000) {
            if (!awake() && SystemClock.uptimeMillis() - asked >= 1000) {
                shell("input keyevent KEYCODE_WAKEUP")
                shell("wm dismiss-keyguard")
                asked = SystemClock.uptimeMillis()
            }
            awake()
        })
        assertTrue(
            "the screen is not awake and unlocked (interactive ${power.isInteractive}, " +
                "locked ${keyguard.isKeyguardLocked})",
            awake(),
        )
        val hidden = shell("settings get global hide_error_dialogs")
        shell("settings put global hide_error_dialogs 1")
        shell("am broadcast -a android.intent.action.CLOSE_SYSTEM_DIALOGS")
        try {
            answersTapsDoubleTapsAndADrag()
        } finally {
            Captions.reset()
            // The move list this test put on the page is no game's.
            PanelEngine.set("moves" to emptyList<String>(), "labels" to emptyMap<String, String>())
            if (hidden == "null" || hidden.isEmpty()) {
                shell("settings delete global hide_error_dialogs")
            } else {
                shell("settings put global hide_error_dialogs $hidden")
            }
            writer.shutdown()
            assertTrue("the screenshots were not written", writer.awaitTermination(60, TimeUnit.SECONDS))
        }
    }

    private fun answersTapsDoubleTapsAndADrag() {
        automation.executeShellCommand("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow").close()
        Launcher.inFront(instrumentation)
        app.getSharedPreferences("captions", 0).edit().clear().commit()
        // A fresh start: a test before this one may have closed the panel (Captions.quit).
        Captions.reset()
        instrumentation.waitForIdleSync()
        // The box's clock is the test's: the box fades five seconds after the words are out, and
        // its moves go four seconds after a tap, and a starved emulator took longer than that
        // between showing the words and the double-tap (5.7 and 10 s with its four cores kept
        // busy, 2026-09-27: the box had faded, the taps went to the home screen, and the
        // double-tap "showed no more lines"). Its time moves only where the moves' timeout is
        // checked; the fade is CaptionsKeepTest's.
        val clock = StepClock(instrumentation)
        Captions.clock = clock
        Captions.configure(app, true)
        Captions.show(
            "Spot on -- rooks belong on open files. Your rook now looks straight down the file at " +
                "their queen. Now, what is your opponent's most dangerous idea, and which of your " +
                "pieces is doing the least?",
        )
        val shownAt = SystemClock.uptimeMillis()
        assertTrue("no caption box", waitFor { box() != null })
        // Drawn and faded in: a box laid out but not yet drawn has no picture on the screen for
        // the system to send touches to, and a starved emulator took seconds to draw it (the
        // screenshot showed the home screen alone while the taps went to it).
        assertTrue("the box was never drawn", waitFor(15_000) {
            var drawn = false
            instrumentation.runOnMainSync { drawn = Captions.drawn() }
            drawn
        })
        // ...and on the screen: woken from sleep, a starved emulator's display showed black for
        // seconds after it said it was on, and the taps sent meanwhile came so late that the
        // first became a long press (the screenshot showed the cross and no box).
        assertTrue("the screen never showed the box", waitFor(20_000) {
            val at = box() ?: return@waitFor false
            val image = automation.takeScreenshot() ?: return@waitFor false
            val pixel = image.getPixel(at.centerX().coerceIn(0, image.width - 1), at.centerY().coerceIn(0, image.height - 1))
            image.recycle()
            pixel and 0xFFFFFF != 0
        })
        still("shown")
        val first = box()!!
        shot("1_shown")
        // The buttons up with the words, no tap needed (Tai, 2026-09-29: "they are handy").
        var buttons: Map<String, Rect> = emptyMap()
        instrumentation.runOnMainSync { buttons = Captions.buttonsAt() }
        assertEquals(listOf("mute", "hint", "move", "line"), buttons.keys.toList())
        assertTrue("the buttons are not laid out: $buttons", buttonsShown())

        // Double-tap: more lines, growing upward from the foot of the screen.
        val (cx, cy) = first.centerX() to first.centerY()
        Log.i(TAG, "double-tap ${SystemClock.uptimeMillis() - shownAt} ms after the line was shown; box ${box()}")
        assertNotNull("the box went before the double-tap, ${SystemClock.uptimeMillis() - shownAt} ms after the line", box())
        doubleTap(cx, cy)
        // Taller with more lines, and no moves: two single taps would make it taller too, for a
        // moment, with the moves a tap shows -- so the double-tap window is let close first.
        val grew = waitFor { (box()?.height() ?: 0) > first.height() } &&
            run { windowOver(); true } &&
            (box()?.height() ?: 0) > first.height() && !movesShown()
        if (!grew) shot("1b_no_more_lines")
        assertTrue(
            "a double-tap showed no more lines (box ${box()}, moves ${movesShown()}, interactive " +
                "${app.getSystemService(android.os.PowerManager::class.java).isInteractive}, locked " +
                "${app.getSystemService(android.app.KeyguardManager::class.java).isKeyguardLocked}, " +
                "focus ${focus()})",
            grew,
        )
        still("taller")
        val tall = box()!!
        assertTrue("it grew downward: $first -> $tall", tall.bottom <= first.bottom + 2)
        shot("2_expanded")
        doubleTap(tall.centerX(), tall.centerY())
        val fewer = waitFor { (box()?.height() ?: 0) == first.height() } &&
            run { windowOver(); true } &&
            (box()?.height() ?: 0) == first.height() && !movesShown()
        if (!fewer) shot("2b_kept_lines")
        assertTrue(
            "a second double-tap kept the lines (box ${box()}, moves ${movesShown()}, first $first, tall $tall)",
            fewer,
        )

        // A tap: the game's moves under the words.
        PanelEngine.set(
            "moves" to listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+", "Nc3", "Nxe4", "O-O"),
            "labels" to mapOf("1" to "book", "3" to "book", "5" to "best", "7" to "best", "9" to "best", "11" to "inaccuracy", "13" to "mistake", "15" to "brilliant"),
        )
        instrumentation.waitForIdleSync()
        tap(cx, cy)
        val movesFrom = SystemClock.uptimeMillis()
        assertTrue("a tap showed no moves", waitFor { movesShown() && (box()?.height() ?: 0) > first.height() })
        still("with its moves")
        val withMoves = box()!!
        assertTrue("the moves grew it downward: $first -> $withMoves", withMoves.bottom <= first.bottom + 2)
        shot("3_moves")

        // The row: mute as an icon, then the panel's quick pills, each asking what its pill
        // on the panel asks.
        instrumentation.runOnMainSync { buttons = Captions.buttonsAt() }
        // A session running, so the pills' asks are queued for its loop (`engine._commands`).
        val wasRunning = PanelEngine.running
        PanelEngine.commands.clear()
        PanelEngine.set("running" to true)
        val pills = mutableListOf<String>()
        try {
            for (name in listOf("hint", "move", "line")) {
                val at = buttons.getValue(name)
                var shown: Map<String, Rect> = emptyMap()
                instrumentation.runOnMainSync { shown = Captions.buttonsAt() }
                Log.i(TAG, "pill $name ${SystemClock.uptimeMillis() - movesFrom} ms after the moves were asked for; row ${shown[name]}")
                tap(at.centerX(), at.centerY())
                windowOver()
            }
            assertTrue("the pills asked $pills", waitFor {
                while (true) pills.add(PanelEngine.commands.poll()?.second?.toString() ?: break)
                pills == listOf("hint", "what should I play", "show me the line")
            })
        } finally {
            PanelEngine.set("running" to wasRunning)
            PanelEngine.commands.clear()
        }
        val wasMuted = PanelVoice.muted
        val speaker = buttons.getValue("mute")
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not mute", waitFor { PanelVoice.muted != wasMuted })
        instrumentation.waitForIdleSync()
        shot("3b_muted")
        windowOver()
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not unmute", waitFor { PanelVoice.muted == wasMuted })
        // The voice's state after the last press, not the setting before the first: a test
        // process with no warm-up starts unmuted whatever the phone's settings say, and with them
        // muted this failed once and passed after (the settings review, 2026-10-01).
        assertTrue("the setting is the last press's", waitFor { SettingsStore(app).muted() == wasMuted })
        assertTrue("the moves went while the buttons were used", movesShown())
        windowOver()
        val withRow = box()!!
        tap(withRow.left + withRow.width() * 3 / 4, withRow.bottom - 40)
        val hid = SystemClock.uptimeMillis()
        val gone = waitFor { (box()?.height() ?: 0) == first.height() && !movesShown() }
        if (!gone) shot("3c_kept_moves")
        assertTrue(
            "a second tap kept the moves (${SystemClock.uptimeMillis() - hid} ms; box ${box()}, " +
                "moves ${movesShown()}, first $first)",
            gone,
        )
        assertTrue("the buttons went with the moves", buttonsShown())

        // Left alone, the moves go by themselves after their shipped four seconds of the box's
        // clock: still up a millisecond before, gone at them. Nothing a finger does after the tap
        // counts from any other time: the tap is the last touch.
        windowOver()
        tap(cx, cy)
        assertTrue("a tap showed no moves again", waitFor { movesShown() && (box()?.height() ?: 0) > first.height() })
        windowOver()
        clock.advance(Captions.CONTROLS_MS - 1)
        assertTrue("the moves went before their time", movesShown())
        clock.advance(1)
        assertTrue("the moves did not go at their time", waitFor { !movesShown() && (box()?.height() ?: 0) == first.height() })

        // A touch held and let go where it is: the box stays, and so do the captions. With the
        // cross's reach at 110 dp the box's first place was within it, and a hold alone turned
        // them off.
        windowOver()
        still("before the hold")
        val still = box()!!
        holdAndDrag(still.centerX() to still.centerY(), still.centerX() to still.centerY())
        instrumentation.waitForIdleSync()
        assertTrue("a hold let go in place turned captions off", Captions.enabled && box() != null)

        // Touch, hold and drag: the box goes where it is let go, and stays there.
        windowOver()
        still("before the drag")
        val here = box()!!
        val screen = app.resources.displayMetrics
        holdAndDrag(here.centerX() to here.centerY(), screen.widthPixels / 2 to screen.heightPixels / 3) {
            shot("4_dragging")
        }
        assertTrue("the box did not move", waitFor { (box()?.centerY() ?: Int.MAX_VALUE) < screen.heightPixels / 2 })
        still("moved")
        shot("5_moved")
        val moved = box()!!
        val saved = app.getSharedPreferences("captions", 0)
        assertTrue("the place was not kept", saved.getInt("y", -1) == moved.top)

        // A finger that just drags, and one that drifts while it holds: both move the box
        // (on the Pixel a real finger's drift lost the hold, and with it the drag).
        holdAndDrag(moved.centerX() to moved.centerY(), moved.centerX() to moved.centerY() + 300, hold = false)
        assertTrue("a plain drag did not move the box", waitFor { (box()?.centerY() ?: 0) > moved.centerY() + 200 })
        still("dragged")
        val dragged = box()!!
        holdAndDrag(dragged.centerX() to dragged.centerY(), dragged.centerX() to dragged.centerY() - 300, drift = true)
        assertTrue("a drifting hold did not move the box", waitFor { (box()?.centerY() ?: Int.MAX_VALUE) < dragged.centerY() - 200 })
        still("held and dragged")
        val back = box()!!

        // Low on the screen, just above the cross: moved there, captions still on (the reach
        // used to be 110 dp, and a box put there was turned off; Tai, 2026-09-29).
        val (crossX, crossY) = Captions.crossCentre(app)
        val above = (90 * screen.density).toInt()
        holdAndDrag(back.centerX() to back.centerY(), crossX to crossY - above)
        still("let go above the cross")
        assertTrue("captions went off above the cross", Captions.enabled && box() != null)
        val low = box()!!

        // All the way down, beside the cross: the box parked at the very foot of the screen,
        // captions still on -- the finger let go away from the cross, whatever the box covers
        // (Tai, 2026-09-29: "the dismiss thing is STILL TOO FAR UP").
        holdAndDrag(low.centerX() to low.centerY(), screen.widthPixels / 5 to screen.heightPixels - (8 * screen.density).toInt())
        still("parked at the foot")
        assertTrue("captions went off with the box parked at the foot", Captions.enabled && box() != null)
        val foot = box()!!
        assertTrue("the box did not go to the foot: $foot", foot.bottom > screen.heightPixels - (140 * screen.density).toInt())

        // Onto the cross at the foot of the screen: captions off.
        holdAndDrag(foot.centerX() to foot.centerY(), crossX to crossY) {
            shot("6_over_the_cross")
        }
        assertTrue("the box is still up", waitFor { box() == null })
        assertFalse("captions are still on", Captions.enabled)
        assertTrue(
            "the setting did not follow",
            waitFor { !SettingsStore(app).captions() },
        )
        assertNull(box())
        instrumentation.waitForIdleSync()
        shot("7_off")

        // The settings page's switch puts them back, in the look the settings give.
        Captions.configure(app, true)
        assertTrue(Captions.enabled)
        Captions.style(Captions.Style(text = 0xFFFFE066.toInt(), background = 0xFF0B2A4A.toInt(), opacityPercent = 70, sizeSp = 22f))
        Captions.show("Bigger, yellow words on a see-through blue box: the settings' look.")
        assertTrue("no styled box", waitFor { box() != null })
        still("styled")
        assertTrue("bigger words did not make a taller box", box()!!.height() > first.height())
        shot("8_styled")

        // Saved on Sidecoach's own settings page: a sample line over it at once.
        Captions.panelInFront(true)
        assertTrue("the box stayed over the panel", waitFor { box() == null })
        Captions.style(Captions.Style())
        Captions.preview()
        assertTrue("no preview over the settings page", waitFor { box() != null })
        instrumentation.waitForIdleSync()
        shot("9_preview")
        Captions.panelInFront(false)
        assertNotNull(Captions)
    }

    /** Whether the box's row of buttons is up (laid out with a width). */
    private fun buttonsShown(): Boolean {
        var at: Map<String, Rect> = emptyMap()
        instrumentation.runOnMainSync { at = Captions.buttonsAt() }
        return at.values.any { it.width() > 0 }
    }

    /** Whether the game's moves are up under the words. */
    private fun movesShown(): Boolean {
        var up = false
        instrumentation.runOnMainSync { up = Captions.movesUp() }
        return up
    }

    /** The window focused now, from `dumpsys window` (no shell here to pipe it through grep). */
    private fun focus(): String =
        shell("dumpsys window").lines().firstOrNull { "mCurrentFocus" in it }?.trim() ?: "unknown"

    companion object {
        private const val TAG = "sidecoach.captions"

        /** A finger's tap: down to up. */
        private const val TAP_MS = 50L

        /** Between a double-tap's two taps, up to down. */
        private const val GAP_MS = 80L

        /** Between two moves of a dragging finger: a frame. */
        private const val FRAME_MS = 16L
    }
}
