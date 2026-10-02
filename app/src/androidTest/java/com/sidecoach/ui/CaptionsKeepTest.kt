package com.sidecoach.ui

import android.graphics.Rect
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The setting that keeps the caption box on screen (Tai, 2026-09-24: "the caption box is
 * always on screen if true and if false then fade away after some time"). Off, the box fades
 * [Captions.HIDE_AFTER_MS] after the words are out; on, it stays, steps aside while Sidecoach's
 * own panel is in front, and comes back with the words when the panel goes; turned off again, it
 * fades.
 *
 * The box's time is the test's ([StepClock]): the fade is checked at its shipped five seconds to
 * the millisecond -- up a millisecond before, fading at it -- and "still up long after" is eight
 * seconds of that clock, none of them waited. What is waited for is the window itself: a box put
 * up or taken down, on the main thread and in the window manager, a state with a deadline that is
 * only a safety net.
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

    private fun <T> onMain(read: () -> T): T {
        var got: T? = null
        instrumentation.runOnMainSync { got = read() }
        @Suppress("UNCHECKED_CAST")
        return got as T
    }

    /** Whether [what] comes true; the deadline is a safety net, never a bound on how soon. */
    private fun waitFor(what: () -> Boolean): Boolean {
        val until = SystemClock.uptimeMillis() + SAFETY_MS
        while (SystemClock.uptimeMillis() < until) {
            if (what()) return true
            SystemClock.sleep(20)
        }
        return what()
    }

    /** What the main thread was given is done: a posted show, keep or panel change has run. */
    private fun settle() = instrumentation.waitForIdleSync()

    /**
     * The fade, at its time: up and not fading a millisecond before [Captions.HIDE_AFTER_MS] on
     * [clock], fading at it, gone once the fade-out is over.
     */
    private fun fadesAtItsTime(clock: StepClock, what: String) {
        clock.advance(Captions.HIDE_AFTER_MS - 1)
        assertNotNull("$what: gone before its time", box())
        assertEquals("$what: fading before its time", false, onMain { Captions.fading() })
        clock.advance(1)
        assertTrue("$what: did not fade at its time", waitFor { box() == null })
    }

    private fun fresh(): StepClock {
        automation.executeShellCommand("appops set ${app.packageName} SYSTEM_ALERT_WINDOW allow").close()
        Launcher.inFront(instrumentation)
        app.getSharedPreferences("captions", 0).edit().clear().commit()
        // A fresh start: a test before this one may have closed the panel (Captions.quit).
        Captions.reset()
        settle()
        return StepClock(instrumentation).also { Captions.clock = it }
    }

    @Test
    fun theBoxFadesUnlessTheSettingKeepsItOnScreen() {
        val clock = fresh()
        Captions.configure(app, true)
        try {
            // Off: the words, then the fade.
            Captions.keep(false)
            Captions.show("Spot on -- rooks belong on open files.")
            assertTrue("no caption box", waitFor { box() != null })
            fadesAtItsTime(clock, "the words")

            // Each message on a line of its own; a reply's later sentences run on (Tai,
            // 2026-09-29: "the text do not have any new lines what so ever").
            Captions.keep(true)
            Captions.show("Nice move.")
            Captions.show("Your bishop is loose.")
            Captions.show("What defends it?", continues = true)
            var lines: String? = null
            assertTrue("the messages did not keep lines of their own: $lines", waitFor {
                lines = onMain { Captions.wordsShown() }
                lines?.endsWith("Nice move.\nYour bishop is loose. What defends it?") == true
            })
            Captions.keep(false)
            fadesAtItsTime(clock, "the box let go")

            // On: still up long after the fade would have come.
            Captions.keep(true)
            Captions.show("Their knight on e4 attacks your queen on d6.")
            assertTrue("no caption box", waitFor { box() != null })
            clock.advance(KEPT_MS)
            assertNotNull("the kept box faded", box())

            // Always on screen means always (Tai, 2026-09-29: "is it always on screen or
            // not????"): over Sidecoach's own panel too, and an answer asked for there shows in
            // it at once -- it used to step aside and drop the words, and he saw nothing.
            Captions.panelInFront(true)
            settle()
            assertNotNull("the kept box left for the panel", box())
            Captions.show("Play Be3+. Your bishop from f4 checks the king.")
            var words: String? = null
            assertTrue("the kept box did not show the answer: $words", waitFor {
                words = onMain { Captions.wordsShown() }
                words?.endsWith("Play Be3+. Your bishop from f4 checks the king.") == true
            })
            Captions.panelInFront(false)
            settle()
            assertNotNull("the kept box went when the panel did", box())

            // Sidecoach quit: the kept box goes, and a line said after does not bring it back;
            // the panel opened again, it is back (Tai, 2026-09-29: "the live caption box doesn't automatically disappear/close when I quit the app on mobile").
            Captions.quit()
            assertTrue("the kept box stayed after Sidecoach quit", waitFor { box() == null })
            Captions.show("Their rook takes on e1.")
            settle()
            assertNull("a line brought the box back after Sidecoach quit", box())
            Captions.panelInFront(true)
            assertTrue("the kept box did not come back with the panel", waitFor { box() != null })
            Captions.panelInFront(false)

            // Captions turned on with the box kept: up at once, before the coach says anything.
            Captions.configure(app, false)
            assertTrue("the box stayed with captions off", waitFor { box() == null })
            Captions.configure(app, true)
            assertTrue("the kept box did not come up with captions", waitFor { box() != null })

            // Turned off again: it fades as before.
            Captions.keep(false)
            fadesAtItsTime(clock, "the setting turned off")
        } finally {
            Captions.keep(false)
            Captions.configure(app, false)
            Captions.reset()
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
        val clock = fresh()
        Captions.configure(app, true)
        try {
            // Kept on screen, a look changed: the sample, then the kept box, touchable, empty of it.
            Captions.keep(true)
            Captions.preview()
            assertTrue("no sample box", waitFor { box() != null })
            assertEquals("the sample's box takes touches", false, onMain { Captions.touchable() })
            clock.advance(Captions.HIDE_AFTER_MS)
            assertTrue("the kept box stayed untouchable", waitFor { onMain { Captions.touchable() } == true })
            var words: String? = onMain { Captions.wordsShown() }
            assertTrue("the sample stayed: $words", words?.contains("This is how captions look") != true)
            // A real line straight after a sample: the sample is not above it.
            Captions.preview()
            assertTrue("no sample", waitFor { onMain { Captions.wordsShown() }?.contains("This is how captions look") == true })
            Captions.show("Nice move.")
            assertTrue("the line did not show", waitFor {
                words = onMain { Captions.wordsShown() }
                words == "Nice move."
            })
            assertEquals("the line's box is untouchable", true, onMain { Captions.touchable() })
            // Not kept: a line that comes while the box fades out is shown.
            Captions.keep(false)
            Captions.show("Your bishop is loose.")
            assertTrue(waitFor { onMain { Captions.wordsShown() }?.endsWith("Your bishop is loose.") == true })
            clock.advance(Captions.HIDE_AFTER_MS)
            assertEquals("the box is not fading out", true, onMain { Captions.fading() })
            Captions.show("What defends it?")
            // The fade-out the line came in has run its course: the line's box is up all the same.
            assertTrue("the line during the fade got no box", waitFor {
                onMain { Captions.fading() } == false && onMain { Captions.wordsShown() }?.endsWith("What defends it?") == true
            })
            settle()
            assertNotNull("the line during the fade went with the box", box())
            fadesAtItsTime(clock, "the line during the fade")
            // A held word said now: the box comes back with the words.
            Captions.again()
            assertTrue("the faded box did not come back", waitFor { box() != null })
        } finally {
            Captions.keep(false)
            Captions.configure(app, false)
            Captions.reset()
        }
    }

    private companion object {
        /** Long after the fade would have come: the eight seconds this test used to wait. */
        const val KEPT_MS = 8000L

        /** A state's safety net: an emulator starved by its host answered seconds late. */
        const val SAFETY_MS = 10_000L
    }
}
