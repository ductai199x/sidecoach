package com.sidecoach.capture

import android.app.AlertDialog
import android.graphics.Bitmap
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.MainActivity
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
        Thread.sleep(100)
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

    @Test
    fun theQuestionContinuesOnlyOnContinue() {
        val watchers = listOf(RecordingNotice.Watcher("com.chess", "Chess.com", 7L))
        var proceeded = 0
        var cancelled = 0
        var dialog: AlertDialog? = null
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            scenario.onActivity {
                dialog = RecordingNotice.ask(it, watchers, proceed = { proceeded++ }, cancelled = { cancelled++ })
            }
            Thread.sleep(800)
            val shown = dialog!!
            var message = ""
            instrumentation.runOnMainSync { message = shown.findViewById<TextView>(android.R.id.message).text.toString() }
            assertEquals(RecordingNotice.words(watchers).second, message)
            val shots = File(app.getExternalFilesDir(null), "recording_notice").apply { mkdirs() }
            instrumentation.uiAutomation.takeScreenshot()?.let { image ->
                File(shots, "question.png").outputStream().use { image.compress(Bitmap.CompressFormat.PNG, 100, it) }
            }
            // Cancel: nothing shared, nothing remembered.
            instrumentation.runOnMainSync { shown.getButton(AlertDialog.BUTTON_NEGATIVE).performClick() }
            Thread.sleep(300)
            assertEquals(0, proceeded)
            assertEquals(1, cancelled)
            val lookup = RecordingNotice.Lookup { pkg -> if (pkg == "com.chess") listOf(RecordingNotice.PERMISSION) to 7L else null }
            assertEquals(1, RecordingNotice.unanswered(app, lookup).size)
            // Continue: on to the share, and not asked again at this version.
            scenario.onActivity {
                dialog = RecordingNotice.ask(it, watchers, proceed = { proceeded++ }, cancelled = { cancelled++ })
            }
            Thread.sleep(500)
            instrumentation.runOnMainSync { dialog!!.getButton(AlertDialog.BUTTON_POSITIVE).performClick() }
            Thread.sleep(300)
            assertEquals(1, proceeded)
            assertEquals(1, cancelled)
            assertTrue(RecordingNotice.unanswered(app, lookup).isEmpty())
        }
    }
}
