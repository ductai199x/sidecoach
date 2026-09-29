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

/**
 * The caption box under a finger: its touches injected as the screen would send them, the
 * overlay window real. Its buttons -- mute and the panel's quick pills, each doing what it does
 * there -- are up from the start. A double-tap shows more lines and a second one fewer; a tap
 * shows the game's moves, and left alone they go after their four seconds; touch, hold and drag
 * moves the box, and dropping it on the cross turns captions off.
 * Each state is screenshotted into the app's files (`captions/`) for a person to look at.
 */
@RunWith(AndroidJUnit4::class)
class CaptionsGestureTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val automation = instrumentation.uiAutomation
    private val shots = File(app.getExternalFilesDir(null), "captions").apply { mkdirs() }

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

    /** One touch event at [at], queued for the screen without waiting for the app. */
    private fun send(action: Int, x: Int, y: Int, down: Long, at: Long) {
        val event = MotionEvent.obtain(down, at, action, x.toFloat(), y.toFloat(), 0)
        event.source = InputDevice.SOURCE_TOUCHSCREEN
        automation.injectInputEvent(event, false)
        event.recycle()
    }

    private fun shot(name: String) {
        val image = automation.takeScreenshot() ?: return
        File(shots, "$name.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
        Log.i(TAG, "screenshot $name")
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
     * Whether [what] comes true within [ms]: a state the box is to reach, never a bound on
     * how soon. Ten seconds, since the box answers on its own thread, and an emulator starved
     * by its host answered a tap three to four seconds late (2026-09-27, its four cores kept
     * busy by sixteen loops); a box that is quick returns at once.
     */
    private fun waitFor(ms: Long = 10_000, what: () -> Boolean): Boolean {
        val until = SystemClock.uptimeMillis() + ms
        while (SystemClock.uptimeMillis() < until) {
            if (what()) return true
            Thread.sleep(50)
        }
        return what()
    }

    /**
     * Hold at [from] for [holdMs] -- a real finger drifting a few pixels meanwhile when
     * [drift] -- then drag in steps to [to]; [before] runs with the finger still down.
     */
    private fun holdAndDrag(
        from: Pair<Int, Int>,
        to: Pair<Int, Int>,
        holdMs: Long = ViewConfiguration.getLongPressTimeout() + 300L,
        drift: Boolean = false,
        before: () -> Unit = {},
    ) {
        val down = SystemClock.uptimeMillis()
        touch(MotionEvent.ACTION_DOWN, from.first, from.second, down)
        if (drift) {
            for (i in 1..(holdMs / 50).toInt()) {
                touch(MotionEvent.ACTION_MOVE, from.first + (i % 3) * 4, from.second + (i % 2) * 5, down)
                Thread.sleep(50)
            }
        } else {
            Thread.sleep(holdMs)
        }
        val steps = 20
        for (i in 1..steps) {
            val x = from.first + (to.first - from.first) * i / steps
            val y = from.second + (to.second - from.second) * i / steps
            touch(MotionEvent.ACTION_MOVE, x, y, down)
            Thread.sleep(16)
        }
        Thread.sleep(300)
        before()
        touch(MotionEvent.ACTION_UP, to.first, to.second, down)
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
        val awake = SystemClock.uptimeMillis() + 20_000
        while (!(power.isInteractive && !keyguard.isKeyguardLocked) && SystemClock.uptimeMillis() < awake) {
            shell("input keyevent KEYCODE_WAKEUP")
            shell("wm dismiss-keyguard")
            Thread.sleep(1000)
        }
        assertTrue(
            "the screen is not awake and unlocked (interactive ${power.isInteractive}, " +
                "locked ${keyguard.isKeyguardLocked})",
            power.isInteractive && !keyguard.isKeyguardLocked,
        )
        val hidden = shell("settings get global hide_error_dialogs")
        shell("settings put global hide_error_dialogs 1")
        shell("am broadcast -a android.intent.action.CLOSE_SYSTEM_DIALOGS")
        // The box fades five seconds after the words are out, and its moves go four seconds
        // after a tap: this test is about the box under a finger, and a starved emulator took
        // longer than that between showing the words and the double-tap (5.7 and 10 s with its
        // four cores kept busy, 2026-09-27: the box had faded, the taps went to the home screen,
        // and the double-tap "showed no more lines"). So the box and its moves stay while the
        // gestures run; the fade is CaptionsKeepTest's, and the moves' own timeout is checked
        // below at its real length.
        Captions.hideAfterMs = LONG_MS
        Captions.controlsMs = LONG_MS
        try {
            answersTapsDoubleTapsAndADrag()
        } finally {
            Captions.hideAfterMs = Captions.HIDE_AFTER_MS
            Captions.controlsMs = Captions.CONTROLS_MS
            // The move list this test put on the page is no game's.
            PanelEngine.set("moves" to emptyList<String>(), "labels" to emptyMap<String, String>())
            if (hidden == "null" || hidden.isEmpty()) {
                shell("settings delete global hide_error_dialogs")
            } else {
                shell("settings put global hide_error_dialogs $hidden")
            }
        }
    }

    private fun answersTapsDoubleTapsAndADrag() {
        automation.executeShellCommand("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow").close()
        automation.executeShellCommand("input keyevent KEYCODE_HOME").close()
        Thread.sleep(1500)
        app.getSharedPreferences("captions", 0).edit().clear().commit()
        Captions.board = null
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
        Thread.sleep(400)
        assertTrue("the box is not where it asked to be", waitFor { settled() })
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
        // moment, with the moves a tap shows.
        val grew = waitFor { (box()?.height() ?: 0) > first.height() } &&
            run { Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 200L); true } &&
            (box()?.height() ?: 0) > first.height() && !movesShown()
        if (!grew) shot("1b_no_more_lines")
        assertTrue(
            "a double-tap showed no more lines (box ${box()}, moves ${movesShown()}, interactive " +
                "${app.getSystemService(android.os.PowerManager::class.java).isInteractive}, locked " +
                "${app.getSystemService(android.app.KeyguardManager::class.java).isKeyguardLocked}, " +
                "focus ${focus()})",
            grew,
        )
        assertTrue("the taller box is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
        val tall = box()!!
        assertTrue("it grew downward: $first -> $tall", tall.bottom <= first.bottom + 2)
        shot("2_expanded")
        doubleTap(tall.centerX(), tall.centerY())
        val fewer = waitFor { (box()?.height() ?: 0) == first.height() } &&
            run { Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 200L); true } &&
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
        Thread.sleep(400)
        tap(cx, cy)
        val movesFrom = SystemClock.uptimeMillis()
        assertTrue("a tap showed no moves", waitFor { movesShown() && (box()?.height() ?: 0) > first.height() })
        assertTrue("the box with its moves is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
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
        val asked = mutableListOf<String>()
        try {
            for (name in listOf("hint", "move", "line")) {
                val at = buttons.getValue(name)
                var shown: Map<String, Rect> = emptyMap()
                instrumentation.runOnMainSync { shown = Captions.buttonsAt() }
                Log.i(TAG, "pill $name ${SystemClock.uptimeMillis() - movesFrom} ms after the moves were asked for; row ${shown[name]}")
                tap(at.centerX(), at.centerY())
                Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
            }
            assertTrue("the pills asked $asked", waitFor {
                while (true) asked.add(PanelEngine.commands.poll()?.second?.toString() ?: break)
                asked == listOf("hint", "what should I play", "show me the line")
            })
        } finally {
            PanelEngine.set("running" to wasRunning)
            PanelEngine.commands.clear()
        }
        val wasMuted = PanelVoice.muted
        val speaker = buttons.getValue("mute")
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not mute", waitFor { PanelVoice.muted != wasMuted })
        Thread.sleep(300)
        shot("3b_muted")
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not unmute", waitFor { PanelVoice.muted == wasMuted })
        assertTrue("the moves went while the buttons were used", movesShown())
        Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
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

        // Left alone, the moves go by themselves after their real four seconds: not before
        // (the tap is handled after it is sent, so they last at least that long from here),
        // and not long after, however slow the emulator.
        Captions.controlsMs = Captions.CONTROLS_MS
        Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
        val tapped = SystemClock.uptimeMillis()
        tap(cx, cy)
        assertTrue("a tap showed no moves again", waitFor { (box()?.height() ?: 0) > first.height() })
        val early = Captions.CONTROLS_MS - 500 - (SystemClock.uptimeMillis() - tapped)
        if (early > 0) Thread.sleep(early)
        assertTrue(
            "the moves went within ${SystemClock.uptimeMillis() - tapped} ms of the tap",
            (box()?.height() ?: 0) > first.height(),
        )
        assertTrue("the moves did not go by themselves", waitFor(15_000) { (box()?.height() ?: 0) == first.height() })
        Log.i(TAG, "the moves went by themselves ${SystemClock.uptimeMillis() - tapped} ms after the tap")
        Captions.controlsMs = LONG_MS

        // A touch held and let go where it is: the box stays, and so do the captions. With the
        // cross's reach at 110 dp the box's first place was within it, and a hold alone turned
        // them off.
        Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
        val still = box()!!
        holdAndDrag(still.centerX() to still.centerY(), still.centerX() to still.centerY())
        Thread.sleep(300)
        assertTrue("a hold let go in place turned captions off", Captions.enabled && box() != null)

        // Touch, hold and drag: the box goes where it is let go, and stays there.
        Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
        val here = box()!!
        val screen = app.resources.displayMetrics
        holdAndDrag(here.centerX() to here.centerY(), screen.widthPixels / 2 to screen.heightPixels / 3) {
            shot("4_dragging")
        }
        assertTrue("the box did not move", waitFor { (box()?.centerY() ?: Int.MAX_VALUE) < screen.heightPixels / 2 })
        assertTrue("the moved box is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
        shot("5_moved")
        val moved = box()!!
        val saved = app.getSharedPreferences("captions", 0)
        assertTrue("the place was not kept", saved.getInt("y", -1) == moved.top)

        // A finger that just drags, and one that drifts while it holds: both move the box
        // (on the Pixel a real finger's drift lost the hold, and with it the drag).
        holdAndDrag(moved.centerX() to moved.centerY(), moved.centerX() to moved.centerY() + 300, holdMs = 0)
        assertTrue("a plain drag did not move the box", waitFor { (box()?.centerY() ?: 0) > moved.centerY() + 200 })
        assertTrue("the dragged box is not where it asked to be", waitFor { settled() })
        val dragged = box()!!
        holdAndDrag(dragged.centerX() to dragged.centerY(), dragged.centerX() to dragged.centerY() - 300, drift = true)
        assertTrue("a drifting hold did not move the box", waitFor { (box()?.centerY() ?: Int.MAX_VALUE) < dragged.centerY() - 200 })
        assertTrue("the box held and dragged is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
        val back = box()!!

        // Low on the screen, just above the cross: moved there, captions still on (the reach
        // used to be 110 dp, and a box put there was turned off; Tai, 2026-09-29).
        val (crossX, crossY) = Captions.crossCentre(app)
        val above = (90 * screen.density).toInt()
        holdAndDrag(back.centerX() to back.centerY(), crossX to crossY - above)
        assertTrue("a box let go above the cross turned captions off", waitFor { settled() })
        assertTrue("captions went off above the cross", Captions.enabled && box() != null)
        Thread.sleep(300)
        val low = box()!!

        // Onto the cross at the foot of the screen: captions off.
        holdAndDrag(low.centerX() to low.centerY(), crossX to crossY) {
            shot("6_over_the_cross")
        }
        assertTrue("the box is still up", waitFor { box() == null })
        assertFalse("captions are still on", Captions.enabled)
        assertTrue(
            "the setting did not follow",
            waitFor { !SettingsStore(app).captions() },
        )
        assertNull(box())
        Thread.sleep(300)
        shot("7_off")

        // The settings page's switch puts them back, in the look the settings give.
        Captions.configure(app, true)
        assertTrue(Captions.enabled)
        Captions.style(Captions.Style(text = 0xFFFFE066.toInt(), background = 0xFF0B2A4A.toInt(), opacityPercent = 70, sizeSp = 22f))
        Captions.show("Bigger, yellow words on a see-through blue box: the settings' look.")
        assertTrue("no styled box", waitFor { box() != null })
        assertTrue("the styled box is not where it asked to be", waitFor { settled() })
        Thread.sleep(400)
        assertTrue("bigger words did not make a taller box", box()!!.height() > first.height())
        shot("8_styled")

        // Saved on Sidecoach's own settings page: a sample line over it at once.
        Captions.panelInFront(true)
        assertTrue("the box stayed over the panel", waitFor { box() == null })
        Captions.style(Captions.Style())
        // The real fade again, so the sample line leaves the screen as it does in use.
        Captions.hideAfterMs = Captions.HIDE_AFTER_MS
        Captions.controlsMs = Captions.CONTROLS_MS
        Captions.preview()
        assertTrue("no preview over the settings page", waitFor { box() != null })
        Thread.sleep(400)
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

        /** The box's fade and its moves' timeout while the gestures run: longer than any run. */
        private const val LONG_MS = 600_000L
    }
}
