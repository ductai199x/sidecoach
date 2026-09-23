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
 * overlay window real. A double-tap shows more lines and a second one fewer; a tap shows the
 * buttons -- mute and the panel's quick pills, each doing what it does there; touch, hold and
 * drag moves the box, and dropping it on the cross turns captions off.
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

    private fun tap(x: Int, y: Int) {
        val down = SystemClock.uptimeMillis()
        touch(MotionEvent.ACTION_DOWN, x, y, down)
        touch(MotionEvent.ACTION_UP, x, y, down)
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

    private fun waitFor(ms: Long = 3000, what: () -> Boolean): Boolean {
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

    @Test
    fun theBoxAnswersTapsDoubleTapsAndADragToTheCross() {
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
        assertTrue("no caption box", waitFor { box() != null })
        Thread.sleep(400)
        assertTrue("the box is not where it asked to be", waitFor { settled() })
        val first = box()!!
        shot("1_shown")

        // Double-tap: more lines, growing upward from the foot of the screen.
        val (cx, cy) = first.centerX() to first.centerY()
        tap(cx, cy)
        Thread.sleep(80)
        tap(cx, cy)
        assertTrue("a double-tap showed no more lines", waitFor { (box()?.height() ?: 0) > first.height() })
        assertTrue("the taller box is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
        val tall = box()!!
        assertTrue("it grew downward: $first -> $tall", tall.bottom <= first.bottom + 2)
        shot("2_expanded")
        tap(tall.centerX(), tall.centerY())
        Thread.sleep(80)
        tap(tall.centerX(), tall.centerY())
        assertTrue("a second double-tap kept the lines", waitFor { (box()?.height() ?: 0) == first.height() })

        // A tap: the buttons, and the game's moves under the words.
        PanelEngine.set(
            "moves" to listOf("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "cxd4", "Bb4+", "Nc3", "Nxe4", "O-O"),
            "labels" to mapOf("1" to "book", "3" to "book", "5" to "best", "7" to "best", "9" to "best", "11" to "inaccuracy", "13" to "mistake", "15" to "brilliant"),
        )
        Thread.sleep(400)
        tap(cx, cy)
        assertTrue("a tap showed no buttons", waitFor { (box()?.height() ?: 0) > first.height() })
        assertTrue("the box with its buttons is not where it asked to be", waitFor { settled() })
        Thread.sleep(300)
        val withRow = box()!!
        assertTrue("the buttons grew it downward: $first -> $withRow", withRow.bottom <= first.bottom + 2)
        shot("3_buttons")

        // The row: mute as an icon, then the panel's quick pills, each asking what its pill
        // on the panel asks.
        var buttons: Map<String, Rect> = emptyMap()
        instrumentation.runOnMainSync { buttons = Captions.buttonsAt() }
        assertEquals(listOf("mute", "hint", "move", "line"), buttons.keys.toList())
        val asked = mutableListOf<String>()
        val session = PanelEngine.onAsk
        PanelEngine.onAsk = { text, _ -> synchronized(asked) { asked.add(text) } }
        try {
            for (name in listOf("hint", "move", "line")) {
                val at = buttons.getValue(name)
                tap(at.centerX(), at.centerY())
                Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
            }
            assertTrue("the pills asked ${synchronized(asked) { asked.toList() }}", waitFor {
                synchronized(asked) { asked.toList() } == listOf("hint", "what should I play", "show me the line")
            })
        } finally {
            PanelEngine.onAsk = session
        }
        val wasMuted = PanelVoice.muted
        val speaker = buttons.getValue("mute")
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not mute", waitFor { PanelVoice.muted != wasMuted })
        Thread.sleep(300)
        shot("3b_muted")
        tap(speaker.centerX(), speaker.centerY())
        assertTrue("the icon did not unmute", waitFor { PanelVoice.muted == wasMuted })
        instrumentation.runOnMainSync { buttons = Captions.buttonsAt() }
        assertTrue("the buttons went while in use", buttons.isNotEmpty())
        Thread.sleep(ViewConfiguration.getDoubleTapTimeout() + 100L)
        val withButtons = box()!!
        tap(withButtons.left + withButtons.width() * 3 / 4, withButtons.bottom - 40)
        assertTrue("a second tap kept the buttons", waitFor { (box()?.height() ?: 0) == first.height() })

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

        // Onto the cross at the foot of the screen: captions off.
        holdAndDrag(back.centerX() to back.centerY(), screen.widthPixels / 2 to screen.heightPixels - 250) {
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
        Captions.preview()
        assertTrue("no preview over the settings page", waitFor { box() != null })
        Thread.sleep(400)
        shot("9_preview")
        Captions.panelInFront(false)
        assertNotNull(Captions)
    }

    companion object {
        private const val TAG = "sidecoach.captions"
    }
}
