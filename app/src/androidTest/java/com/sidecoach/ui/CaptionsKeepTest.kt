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
        // A fresh start: a test before this one may have closed the panel (Captions.quit).
        Captions.reset()
        // The fade a second after the words, not five: the test is about whether the box fades
        // or stays, not how long the shipped fade waits; the test waited out five of them.
        Captions.hideAfterMs = FADE_MS
        Captions.configure(app, true)
        try {
            // Off: the words, then the fade.
            Captions.keep(false)
            Captions.show("Spot on -- rooks belong on open files.")
            assertTrue("no caption box", waitFor(3000) { box() != null })
            assertTrue("the box did not fade", waitFor(9000) { box() == null })

            // Each message on a line of its own; a reply's later sentences run on (Tai,
            // 2026-09-29: "the text do not have any new lines what so ever").
            Captions.keep(true)
            Captions.show("Nice move.")
            Captions.show("Your bishop is loose.")
            Captions.show("What defends it?", continues = true)
            var lines: String? = null
            assertTrue("the messages did not keep lines of their own: $lines", waitFor(3000) {
                instrumentation.runOnMainSync { lines = Captions.wordsShown() }
                lines?.endsWith("Nice move.\nYour bishop is loose. What defends it?") == true
            })
            Captions.keep(false)
            assertTrue("the box did not fade", waitFor(9000) { box() == null })

            // On: still up long after the fade would have come.
            Captions.keep(true)
            Captions.show("Their knight on e4 attacks your queen on d6.")
            assertTrue("no caption box", waitFor(3000) { box() != null })
            Thread.sleep(3 * FADE_MS)
            assertNotNull("the kept box faded", box())

            // Always on screen means always (Tai, 2026-09-29: "is it always on screen or
            // not????"): over Sidecoach's own panel too, and an answer asked for there shows in
            // it at once -- it used to step aside and drop the words, and he saw nothing.
            Captions.panelInFront(true)
            Thread.sleep(1000)
            assertNotNull("the kept box left for the panel", box())
            Captions.show("Play Be3+. Your bishop from f4 checks the king.")
            var words: String? = null
            assertTrue("the kept box did not show the answer: $words", waitFor(3000) {
                instrumentation.runOnMainSync { words = Captions.wordsShown() }
                words?.endsWith("Play Be3+. Your bishop from f4 checks the king.") == true
            })
            Captions.panelInFront(false)
            assertNotNull("the kept box went when the panel did", box())

            // Sidecoach quit: the kept box goes, and a line said after does not bring it back;
            // the panel opened again, it is back (Tai, 2026-09-29: "the live caption box doesn't automatically disappear/close when I quit the app on mobile").
            Captions.quit()
            assertTrue("the kept box stayed after Sidecoach quit", waitFor(3000) { box() == null })
            Captions.show("Their rook takes on e1.")
            Thread.sleep(1000)
            assertTrue("a line brought the box back after Sidecoach quit", box() == null)
            Captions.panelInFront(true)
            assertTrue("the kept box did not come back with the panel", waitFor(3000) { box() != null })
            Captions.panelInFront(false)

            // Captions turned on with the box kept: up at once, before the coach says anything.
            Captions.configure(app, false)
            assertTrue("the box stayed with captions off", waitFor(3000) { box() == null })
            Captions.configure(app, true)
            assertTrue("the kept box did not come up with captions", waitFor(3000) { box() != null })

            // Turned off again: it fades as before.
            Captions.keep(false)
            assertTrue("the box did not fade once the setting was off", waitFor(9000) { box() == null })
        } finally {
            Captions.keep(false)
            Captions.configure(app, false)
            Captions.hideAfterMs = Captions.HIDE_AFTER_MS
        }
    }

    /**
     * The sample line over a box kept on screen fades as any sample does, and the kept box comes
     * back touchable, without it; a real line after a sample does not run on from it; a line
     * that comes while the box fades gets a box; a held word said later brings the faded box
     * back (voice findings 3, 8 and 20).
     */
    @Test
    fun theSampleLineLeavesATouchableBoxAndALineDuringTheFadeIsShown() {
        automation.executeShellCommand("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow").close()
        automation.executeShellCommand("input keyevent KEYCODE_HOME").close()
        Thread.sleep(1500)
        app.getSharedPreferences("captions", 0).edit().clear().commit()
        Captions.reset()
        Captions.hideAfterMs = FADE_MS
        Captions.configure(app, true)
        try {
            // Kept on screen, a look changed: the sample, then the kept box, touchable, empty of it.
            Captions.keep(true)
            Captions.preview()
            assertTrue("no sample box", waitFor(3000) { box() != null })
            var touchable: Boolean? = null
            assertTrue("the kept box stayed untouchable", waitFor(9000) {
                instrumentation.runOnMainSync { touchable = Captions.touchable() }
                touchable == true
            })
            var words: String? = null
            instrumentation.runOnMainSync { words = Captions.wordsShown() }
            assertTrue("the sample stayed: $words", words?.contains("This is how captions look") != true)
            // A real line straight after a sample: the sample is not above it.
            Captions.preview()
            Captions.show("Nice move.") // posted after the sample: its order, not a wait
            assertTrue("the line did not show", waitFor(3000) {
                instrumentation.runOnMainSync { words = Captions.wordsShown() }
                words == "Nice move."
            })
            instrumentation.runOnMainSync { touchable = Captions.touchable() }
            assertTrue("the line's box is untouchable", touchable == true)
            // Not kept: a line that comes while the box fades out is shown.
            Captions.keep(false)
            Captions.show("Your bishop is loose.")
            assertTrue(waitFor(3000) { box() != null })
            // The fade begun and a line given in one step on the main thread: by the box's
            // state, not a sleep timed into the 350 ms fade (round 3 audit, 2026-10-02).
            var fading = false
            instrumentation.runOnMainSync {
                Captions.fadeNow()
                fading = Captions.fadingNow()
                Captions.showNow("What defends it?")
            }
            assertTrue("the box was not fading", fading)
            var shownNow: String? = null
            instrumentation.runOnMainSync { shownNow = Captions.wordsShown() }
            assertTrue("the line during the fade was not shown: $shownNow", shownNow?.endsWith("What defends it?") == true)
            instrumentation.runOnMainSync { fading = Captions.fadingNow() }
            assertTrue("the line's box is the fading one", !fading)
            assertNotNull("the line during the fade went with the box", box())
            assertTrue("the box did not fade", waitFor(9000) { box() == null })
            // A held word said now: the box comes back with the words.
            Captions.again()
            assertTrue("the faded box did not come back", waitFor(3000) { box() != null })
            // The voice still speaking: the box waits for it, and fades once it is done (voice
            // finding 9: a flat five seconds faded it under a long line still being said).
            val voice = object : com.sidecoach.voice.TtsProvider {
                override val name = "speaking"
                @Volatile override var busy = true
                override fun healthcheck() = true to "fake"
                override fun say(text: String, voice: String?, owner: String) = Unit
                override fun stop(owner: String?) = Unit
            }
            PanelVoice.speaker = voice
            instrumentation.runOnMainSync {
                Captions.fadeNow()
                fading = Captions.fadingNow()
            }
            assertTrue("the box faded while the voice spoke", !fading)
            voice.busy = false
            instrumentation.runOnMainSync {
                Captions.fadeNow()
                fading = Captions.fadingNow()
            }
            assertTrue("the box did not fade once the voice was done", fading)
        } finally {
            PanelVoice.speaker = null
            Captions.keep(false)
            Captions.configure(app, false)
            Captions.hideAfterMs = Captions.HIDE_AFTER_MS
        }
    }

    private companion object {
        /** The fade's wait in this test: a second, against the five the box ships with. */
        const val FADE_MS = 1000L
    }
}
