package com.sidecoach.voice

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Collections
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * The phone's voice speaking a line in pieces ([Kokoro.plan]): each piece made while the one
 * before plays, the engine paused around every piece and never left paused, the line's start
 * told once, and a stop that ends a line part-way.
 */
@RunWith(AndroidJUnit4::class)
class KokoroTtsTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun tts(): KokoroTts? {
        val dir = File(app.filesDir, "models/kokoro")
        if (!File(dir, KokoroTts.ONNX_FILE).exists() || !File(dir, KokoroTts.VOICES_FILE).exists()) return null
        return KokoroTts(app, dir).also {
            assertTrue(it.load())
            // The graph is loaded once, whatever the voice does
            // (`test_kokoro_loads_once_and_a_voice_change_reloads_nothing`).
            val model = MODEL.get(it)
            assertTrue(it.load() && it.load())
            it.voice = "bm_george"
            it.voice = "af_heart"
            assertTrue(it.load())
            assertSame("the graph must be loaded exactly once", model, MODEL.get(it))
            assertTrue(it.healthcheck().second, it.healthcheck().second.contains("voice af_heart, cpu,"))
            it.voice = "af_sarah"
            it.prime()
        }
    }

    private fun quiet(tts: KokoroTts, seconds: Int): Boolean {
        val until = SystemClock.elapsedRealtime() + seconds * 1000L
        while (tts.busy) {
            if (SystemClock.elapsedRealtime() > until) return false
            Thread.sleep(20)
        }
        return true
    }

    @Test
    fun aLongLineIsSpokenInPiecesWhileTheRestIsMade() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val events = Collections.synchronizedList(ArrayList<String>())
        val times = Collections.synchronizedList(ArrayList<Long>())
        val heard = ArrayList<String>()
        tts!!.onSynthesis = { on ->
            events.add(if (on) "make" else "made")
            times.add(SystemClock.elapsedRealtime())
        }
        tts.onPlay = { line ->
            events.add("play")
            times.add(SystemClock.elapsedRealtime())
            synchronized(heard) { heard.add(line) }
        }
        // Long enough to be cut in two (past Kokoro.WHOLE_BELOW_S of speech): the second piece is
        // made while the first plays. The line used to be four pieces, ten seconds of speech made
        // and then played out in real time, for the same order of events.
        val line = "That was fine. The centre holds, and the bishop is doing real work on the long diagonal."
        val pieces = Kokoro.plan(KokoroTokenizer(app), Py.strip(Notation.speak(line, voice = true)), 1.0f, "en-us")
        assertTrue("the line is not cut: ${pieces.map { it.first }}", pieces.size > 1)
        val began = SystemClock.elapsedRealtime()
        tts.say(line)
        assertTrue("the line never finished", quiet(tts, 60))
        tts.close()
        val made = events.filter { it != "play" }
        assertEquals("a pause and a resume around every piece", List(pieces.size) { listOf("make", "made") }.flatten(), made)
        assertEquals("the line's start told once", listOf(line), heard)
        val play = events.indexOf("play")
        assertTrue("played before its first piece was made: $events", play > 1 && events[1] == "made")
        Log.i(
            TAG,
            "${pieces.size} pieces; first sound ${times[play] - began} ms after the line, the whole line made " +
                "${times[events.lastIndexOf("made")] - began} ms after it: $events",
        )
    }

    @Test
    fun aStopEndsALineMidWay() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val firstMade = CountDownLatch(1)
        val events = Collections.synchronizedList(ArrayList<String>())
        tts!!.onSynthesis = { on ->
            events.add(if (on) "make" else "made")
            if (!on) firstMade.countDown()
        }
        val line = "The knight on f3 guards the pawn on e5, and the bishop on c4 eyes f7. " +
            "The knight on f3 guards the pawn on e5, and the bishop on c4 eyes f7. " +
            "The knight on f3 guards the pawn on e5, and the bishop on c4 eyes f7."
        val pieces = Kokoro.plan(KokoroTokenizer(app), Py.strip(Notation.speak(line, voice = true)), 1.0f, "en-us")
        assertTrue("the line is not cut into three or more", pieces.size >= 3)
        tts.say(line)
        assertTrue("no piece was made", firstMade.await(30, TimeUnit.SECONDS))
        tts.stop()
        assertTrue("the line went on after the stop", quiet(tts, 20))
        tts.close()
        val made = events.count { it == "made" }
        assertEquals("a pause left standing: $events", events.count { it == "make" }, made)
        assertTrue("every piece was made after the stop: $events", made < pieces.size)
        Log.i(TAG, "stopped after $made of ${pieces.size} pieces")
    }

    /**
     * The first run's warm-up loads the voice before Setup has fetched it; once the files arrive
     * it is loaded again, not left dead until a restart; a closed voice never is
     * (`test_the_voice_loads_once_setup_has_fetched_its_files`, voice finding 4).
     */
    @Test
    fun theVoiceIsLoadedAgainOnceSetupHasFetchedItsFiles() {
        val dir = File(app.cacheDir, "kokoro-latch-${System.nanoTime()}").apply { mkdirs() }
        try {
            val tts = KokoroTts(app, dir)
            assertTrue(!tts.load()) // the warm-up, before Setup
            assertTrue("nothing changed: not tried again", !(ARRIVED.invoke(tts) as Boolean))
            File(dir, KokoroTts.ONNX_FILE).writeBytes(byteArrayOf(1))
            File(dir, KokoroTts.VOICES_FILE).writeBytes(byteArrayOf(1))
            assertTrue("the files came: tried again", ARRIVED.invoke(tts) as Boolean)
            assertTrue(!tts.load()) // stand-ins, not a model: tried, and failed on them
            assertTrue("the same stand-ins: not tried again", !(ARRIVED.invoke(tts) as Boolean))
            tts.close()
            File(dir, KokoroTts.ONNX_FILE).writeBytes(byteArrayOf(1, 2))
            assertTrue("closed: never again", !(ARRIVED.invoke(tts) as Boolean))
        } finally {
            dir.listFiles()?.forEach { it.delete() }
            dir.delete()
        }
    }

    companion object {
        /** `KokoroTts.filesArrived`: whether a failed load would be tried again now. */
        private val ARRIVED = KokoroTts::class.java.getDeclaredMethod("filesArrived").apply { isAccessible = true }

        private const val TAG = "sidecoach.voice"

        /** The loaded graph, to see that it is the same one after more loads and voice changes. */
        private val MODEL = KokoroTts::class.java.getDeclaredField("kokoro").apply { isAccessible = true }
    }
}
