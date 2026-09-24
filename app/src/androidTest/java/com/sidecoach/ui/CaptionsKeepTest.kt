package com.sidecoach.ui

import android.graphics.Rect
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The setting that keeps the caption box on screen (Tai, 2026-09-24: "the caption box is
 * always on screen if true and if false then fade away after some time"). Off, the box fades a
 * few seconds after the words are out; on, it stays, steps aside while Sidecoach's own panel is
 * in front, and comes back with the words when the panel goes; turned off again, it fades.
 */
@RunWith(AndroidJUnit4::class)
class CaptionsKeepTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext
    private val automation = instrumentation.uiAutomation

    private fun box(): Rect? {
        var at: Rect? = null
        instrumentation.runOnMainSync { at = Captions.windowAt() }
        return at
    }

    private fun waitFor(ms: Long, what: () -> Boolean): Boolean {
        val until = SystemClock.uptimeMillis() + ms
        while (SystemClock.uptimeMillis() < until) {
            if (what()) return true
            Thread.sleep(50)
        }
        return what()
    }

    @Test
    fun theBoxFadesUnlessTheSettingKeepsItOnScreen() {
        automation.executeShellCommand("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow").close()
        automation.executeShellCommand("input keyevent KEYCODE_HOME").close()
        Thread.sleep(1500)
        app.getSharedPreferences("captions", 0).edit().clear().commit()
        Captions.board = null
        Captions.panelInFront(false)
        Captions.configure(app, true)
        try {
            // Off: the words, then the fade.
            Captions.keep(false)
            Captions.show("Spot on -- rooks belong on open files.")
            assertTrue("no caption box", waitFor(3000) { box() != null })
            assertTrue("the box did not fade", waitFor(9000) { box() == null })

            // On: still up long after the fade would have come.
            Captions.keep(true)
            Captions.show("Their knight on e4 attacks your queen on d6.")
            assertTrue("no caption box", waitFor(3000) { box() != null })
            Thread.sleep(8000)
            assertNotNull("the kept box faded", box())

            // Sidecoach's own panel in front: out of the way; gone from the front: back.
            Captions.panelInFront(true)
            assertTrue("the box stayed over the panel", waitFor(3000) { box() == null })
            Captions.panelInFront(false)
            assertTrue("the kept box did not come back", waitFor(3000) { box() != null })

            // Turned off again: it fades as before.
            Captions.keep(false)
            assertTrue("the box did not fade once the setting was off", waitFor(9000) { box() == null })
        } finally {
            Captions.keep(false)
            Captions.configure(app, false)
        }
    }
}
