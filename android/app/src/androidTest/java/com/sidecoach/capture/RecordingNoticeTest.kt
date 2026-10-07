package com.sidecoach.capture

import android.app.AlertDialog
import android.graphics.Bitmap
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.MainActivity
import com.sidecoach.ui.PanelWarm
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The chess apps named before a session when they can be told they are being recorded: the
 * check over the apps, the answer kept per version, the words, the real lookup, and the
 * question itself over the panel (screenshotted into the app's files, `recording_notice/`).
 */
@RunWith(AndroidJUnit4::class)
class RecordingNoticeTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val app = instrumentation.targetContext

    private fun clear() {
        app.getSharedPreferences("recording_notice", 0).edit().clear().commit()
    }

    @Before fun before() = clear()

    @After fun after() = clear()

    /**
     * A button of [dialog] clicked, and its listener run: the dialog hands a click to its
     * listener as a message on the main thread's queue, so a step queued after the click runs
     * once the listener has (one queue, in order).
     */
    private fun click(dialog: AlertDialog, which: Int) {
        instrumentation.runOnMainSync { dialog.getButton(which).performClick() }
        instrumentation.runOnMainSync {}
    }

    private fun showing(dialog: AlertDialog): Boolean {
        var up = false
        instrumentation.runOnMainSync { up = dialog.isShowing && dialog.findViewById<TextView>(android.R.id.message) != null }
        return up
    }

    private fun lookup(vararg apps: Triple<String, List<String>, Long>) = RecordingNotice.Lookup { pkg ->
        apps.firstOrNull { it.first == pkg }?.let { it.second to it.third }
    }

    @Test
    fun onlyTheChessAppsThatDeclareItAreNamed() {
        val found = RecordingNotice.watchers(
            lookup(
                Triple("com.chess", listOf("android.permission.INTERNET", RecordingNotice.PERMISSION), 280070L),
                Triple("org.lichess.mobileV2", listOf("android.permission.INTERNET"), 5L),
                Triple("com.example.notchess", listOf(RecordingNotice.PERMISSION), 1L),
            ),
        )
        assertEquals(listOf(RecordingNotice.Watcher("com.chess", "Chess.com", 280070L)), found)
        // chess.com as it is on Tai's Pixel (4.10.17): DETECT_SCREEN_CAPTURE, the screenshot
        // notice, is not the recording one.
        assertTrue(
            RecordingNotice.watchers(
                lookup(Triple("com.chess", listOf("android.permission.DETECT_SCREEN_CAPTURE"), 280070L)),
            ).isEmpty(),
        )
        // What the log says of it at each Start.
        assertEquals(
            "Chess.com (com.chess 280070) cannot be told",
            RecordingNotice.survey(lookup(Triple("com.chess", listOf("android.permission.DETECT_SCREEN_CAPTURE"), 280070L))),
        )
        assertEquals("none of the chess apps it knows is installed", RecordingNotice.survey(lookup()))
    }

    @Test
    fun anAnswerHoldsUntilTheAppUpdates() {
        val v1 = lookup(Triple("com.chess", listOf(RecordingNotice.PERMISSION), 1L))
        val first = RecordingNotice.unanswered(app, v1)
        assertEquals(1, first.size)
        RecordingNotice.agree(app, first)
        // No wait: `apply` puts the answer in the process's preferences before it returns (only
        // the write to disk is later), and every read here is of those.
        assertTrue("asked again at the same version", RecordingNotice.unanswered(app, v1).isEmpty())
        val v2 = lookup(Triple("com.chess", listOf(RecordingNotice.PERMISSION), 2L))
        assertEquals("not asked again after an update", 1, RecordingNotice.unanswered(app, v2).size)
    }

    @Test
    fun theWordsNameTheAppsOnce() {
        val chess = RecordingNotice.Watcher("com.chess", "Chess.com", 1L)
        val lichess = RecordingNotice.Watcher("org.lichess.mobileapp", "Lichess", 1L)
        val lichess2 = RecordingNotice.Watcher("org.lichess.mobileV2", "Lichess", 1L)
        val (title, message) = RecordingNotice.words(listOf(chess))
        assertEquals("Chess.com can tell it is being recorded", title)
        assertTrue(message, message.startsWith("Chess.com has asked Android to be told when its screen is recorded."))
        assertTrue(message, "It is not told which app is recording, and it gets nothing of what is recorded." in message)
        assertEquals("Chess.com and Lichess can tell they are being recorded", RecordingNotice.words(listOf(chess, lichess)).first)
        assertEquals("Lichess can tell it is being recorded", RecordingNotice.words(listOf(lichess, lichess2)).first)
    }

    @Test
    fun theRealLookupReadsAnAppsDeclaredPermissions() {
        val real = RecordingNotice.installed(app)
        val own = real.find(app.packageName)
        assertNotNull("the lookup could not read Sidecoach itself", own)
        assertTrue(own!!.first.toString(), "android.permission.SYSTEM_ALERT_WINDOW" in own.first)
        assertNull("not installed here", real.find("com.example.nothing.here"))
    }

    /**
     * The question over the panel, Cancel and Continue. Every step is on the main thread:
     * `show()` has put the dialog's window up when the call that asked returns, and a button's
     * listener has run once a step queued after its click has ([click]) -- so each is asserted
     * at once, never polled for.
     *
     * The panel is opened for the question alone: its warm-up (the voice, the recogniser, the
     * piece net, the vision, the engines) is said to have begun already ([PanelWarm.started]).
     * It took seconds of the device's cores that the question has no need of, it ran on into
     * the classes after this one, changing the app's shared state under them (it muted the
     * voice of PanelVoicePolicyTest), and its models' memory is what could have the WebView's
     * renderer killed and the window rebuilt under the question. What the launch costs now is
     * the WebView drawing the page (its in-process GPU thread and the RenderThread).
     */
    @Test
    fun theQuestionContinuesOnlyOnContinue() {
        val warming = PanelWarm.started
        PanelWarm.started = true
        try {
            question()
        } finally {
            PanelWarm.started = warming
        }
    }

    private fun question() {
        val watchers = listOf(RecordingNotice.Watcher("com.chess", "Chess.com", 7L))
        var proceeded = 0
        var cancelled = 0
        var dialog: AlertDialog? = null
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            try {
                var host: android.app.Activity? = null
                scenario.onActivity {
                    host = it
                    dialog = RecordingNotice.ask(it, watchers, proceed = { proceeded++ }, cancelled = { cancelled++ })
                }
                val shown = dialog!!
                assertTrue("the question did not come up", showing(shown))
                var message = ""
                instrumentation.runOnMainSync { message = shown.findViewById<TextView>(android.R.id.message).text.toString() }
                assertEquals(RecordingNotice.words(watchers).second, message)
                val shots = File(app.getExternalFilesDir(null), "recording_notice").apply { mkdirs() }
                instrumentation.uiAutomation.takeScreenshot()?.let { image ->
                    File(shots, "question.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
                }
                // Cancel: nothing shared, nothing remembered.
                click(shown, AlertDialog.BUTTON_NEGATIVE)
                assertEquals(0, proceeded)
                assertEquals(1, cancelled)
                val lookup = RecordingNotice.Lookup { pkg -> if (pkg == "com.chess") listOf(RecordingNotice.PERMISSION) to 7L else null }
                assertEquals(1, RecordingNotice.unanswered(app, lookup).size)
                // Continue: on to the share, and not asked again at this version.
                scenario.onActivity {
                    dialog = RecordingNotice.ask(it, watchers, proceed = { proceeded++ }, cancelled = { cancelled++ })
                }
                assertTrue("the question did not come up again", showing(dialog!!))
                click(dialog!!, AlertDialog.BUTTON_POSITIVE)
                // The answer took the question down: its dismissal handled now, never after the
                // window has gone (handled then, it ended the whole run, 2026-10-02).
                instrumentation.waitForIdleSync()
                instrumentation.runOnMainSync { assertTrue("the question stayed up after its answer", dialog?.isShowing == false) }
                assertEquals(1, proceeded)
                assertEquals(1, cancelled)
                assertTrue(RecordingNotice.unanswered(app, lookup).isEmpty())
                // Both questions were asked over the one window the test opened.
                scenario.onActivity { assertTrue("the window was rebuilt under the question", it === host) }
            } finally {
                // Whatever failed, no question left up when the window closes: its dismissal
                // handled after the window was gone ended the whole run.
                instrumentation.runOnMainSync { dialog?.takeIf { it.isShowing }?.dismiss() }
                instrumentation.waitForIdleSync()
            }
        }
    }
}
