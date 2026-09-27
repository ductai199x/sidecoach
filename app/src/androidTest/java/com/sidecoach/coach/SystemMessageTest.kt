package com.sidecoach.coach

import android.app.Instrumentation
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.SidecoachApp
import com.sidecoach.llm.OfflineProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * C, the lessons from earlier games, at the end of every coach's system message: loaded by the
 * app's own start, before any coach is built, and appended only when it has content
 * (`tests/test_prompt.py`). Nothing on the phone ever loaded it, and every model call went
 * without the lessons (the parity audit cc-01, 2026-09-26).
 */
@RunWith(AndroidJUnit4::class)
class SystemMessageTest {

    @Test
    fun theAppsStartLoadsTheLessonsAndOnlyWhatHasContentIsAppended() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        assertTrue(context.applicationContext is SidecoachApp)
        val shipped = context.assets.open("coach/corrections.md").bufferedReader(Charsets.UTF_8)
            .use { it.readText() }.replace("\r\n", "\n").replace("\r", "\n").trim()
        assertTrue("the shipped lessons are empty", shipped.isNotEmpty())
        try {
            // The app started afresh, with nothing loaded: its start alone must load C.
            Corrections.set("")
            instrumentation.runOnMainSync {
                val app = Instrumentation.newApplication(SidecoachApp::class.java, context)
                instrumentation.callApplicationOnCreate(app)
            }
            assertEquals(shipped, Corrections.text())
            val system = Coach(OfflineProvider()).system
            assertTrue(system, system.endsWith("in front of you.\n$shipped"))
            assertTrue(system.indexOf(REPLY_FORMAT) < system.indexOf("Lessons from earlier games"))

            // Empty, C adds nothing; with content, it is the message's last part.
            Corrections.set("")
            assertFalse(systemMessage().contains("Lessons from earlier games"))
            val one = "When asked about the opening: state its name as the message gives it."
            Corrections.set(one)
            assertTrue(systemMessage().endsWith("in front of you.\n$one"))
            assertFalse(systemMessage(withCorrections = false).contains("Lessons from earlier games"))
        } finally {
            Corrections.set(shipped)
        }
    }
}
