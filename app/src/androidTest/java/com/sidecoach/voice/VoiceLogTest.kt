package com.sidecoach.voice

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.ui.PanelEngine
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.BlockingQueue
import java.util.concurrent.LinkedBlockingQueue

/**
 * The voice's warnings reach the panel's log, as the desktop's `_VoiceLogHandler` forwards every
 * warning and error of the `sidecoach.voice` logger (parity pv-01): a voice, a recogniser or a
 * gate that did not load, and an utterance the handler failed on, each said on the voice channel
 * in the desktop's words. And [Py.repr], which the log's `%r` and `{x!r}` go through, against
 * what Python's `repr` answered for the same strings.
 */
@RunWith(AndroidJUnit4::class)
class VoiceLogTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun linesSince(cursor: Int): List<JSONObject> {
        val lines = PanelEngine.poll(cursor).getJSONArray("lines")
        return (0 until lines.length()).map { lines.getJSONObject(it) }
    }

    private fun cursor(): Int = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")

    private fun emptyDir(name: String): File =
        File(app.cacheDir, "voicelog-$name").apply { mkdirs() }

    @Test
    fun aWarningAndAnErrorOfTheVoiceAreInThePanelsLog() {
        val before = cursor()
        VoiceLog.warn("Kokoro playback failed: no output")
        VoiceLog.error("something broke")
        val lines = linesSince(before).filter { it.getString("chan") == "voice" }
        assertEquals(listOf("warn", "error"), lines.map { it.getString("level") })
        assertEquals(listOf("Kokoro playback failed: no output", "something broke"), lines.map { it.getString("text") })
    }

    @Test
    fun aVoiceThatDoesNotLoadSaysSoInTheLog() {
        val before = cursor()
        val tts = KokoroTts(app, emptyDir("kokoro"))
        assertFalse(tts.load())
        // Silent from then on: a line is not even queued.
        tts.say("Nothing to say it with.")
        assertFalse(tts.busy)
        val lines = linesSince(before).filter { it.getString("chan") == "voice" && it.getString("text").startsWith("Kokoro") }
        assertEquals(1, lines.size)
        assertEquals("warn", lines[0].getString("level"))
        assertTrue(lines[0].getString("text"), lines[0].getString("text").startsWith("Kokoro voice unavailable, going silent: "))
    }

    @Test
    fun aRecogniserAndAGateThatDoNotLoadSaySoInTheLog() {
        val before = cursor()
        val stt = ParakeetStt(app, emptyDir("parakeet"))
        assertFalse(stt.load())
        assertEquals("", stt.transcribe(FloatArray(SAMPLE_RATE / 2)))
        val gate = SileroGate(File(emptyDir("silero"), SileroGate.FILE))
        assertFalse(gate.load())
        val texts = linesSince(before).filter { it.getString("chan") == "voice" && it.getString("level") == "warn" }
            .map { it.getString("text") }.filter { it.startsWith("Parakeet") || it.startsWith("Silero") }
        // A recogniser not downloaded yet is looked for again at each use, as the desktop's is
        // (`ParakeetSTT.load`, voice finding 4: Setup fetches it after the first warm-up), so the
        // load and the transcription each say so; the gate, tried once, says so once.
        assertEquals(texts.toString(), 3, texts.size)
        assertEquals("Parakeet unavailable, speech recognition is off: not downloaded yet (Setup, in the menu)", texts[0])
        assertEquals(texts[0], texts[1])
        assertTrue(texts[2], texts[2].startsWith("Silero VAD unavailable: "))
    }

    /** A microphone whose frames the test puts in. */
    private class FedMic : FrameSource {
        override val frames: BlockingQueue<FloatArray> = LinkedBlockingQueue()
        override val detail = "fed by the test"
        override val isOpen = true
        override fun open() = true
        override fun close() = Unit
    }

    private class Quiet : VoiceGate {
        override fun load() = true
        override fun healthcheck() = true to "quiet"
        override fun reset() = Unit
        override fun probability(input: FloatArray) = 0f
    }

    @Test
    fun anUtteranceTheHandlerFailsOnIsSaidAndListeningGoesOn() {
        val before = cursor()
        val heard = LinkedBlockingQueue<Double>()
        var first = true
        val mic = FedMic()
        val listener = Listener(mic, Quiet(), onUtterance = { _, seconds ->
            if (first) {
                first = false
                throw IllegalStateException("boom")
            }
            heard.put(seconds)
        })
        listener.setMode("ptt")
        repeat(2) {
            listener.press()
            repeat(16) { mic.frames.put(FloatArray(HOP)) }
            val until = System.currentTimeMillis() + 2000
            while (mic.frames.isNotEmpty() && System.currentTimeMillis() < until) Thread.sleep(10)
            Thread.sleep(50)
            listener.release()
        }
        assertTrue("the second utterance was not handed over", heard.poll(3, java.util.concurrent.TimeUnit.SECONDS) != null)
        listener.setMode("off")
        val texts = linesSince(before).filter { it.getString("chan") == "voice" }.map { it.getString("text") }
            .filter { it.startsWith("utterance") }
        assertEquals(listOf("utterance handler failed: boom"), texts)
    }

    @Test
    fun reprWritesAStringAsPythonDoes() {
        // What `repr` answered on the desktop (Python 3.12), string by string.
        val cases = listOf(
            "Qxf7#: mate." to "'Qxf7#: mate.'",
            "it's" to "\"it's\"",
            "say \"hi\"" to "'say \"hi\"'",
            "both ' and \"" to "'both \\' and \"'",
            "tab\there\nnew\rret" to "'tab\\there\\nnew\\rret'",
            "a\u0080b c d​e😀fg" to "'a\\x80b\\xa0c\\u2028d\\u200be😀f\\ue000g'",
            "back\\slash" to "'back\\\\slash'",
            "x\u007fy\u0001" to "'x\\x7fy\\x01'",
            "½-½ é" to "'½-½ é'",
            "͸" to "'\\u0378'",
        )
        for ((text, expected) in cases) assertEquals(text, expected, Py.repr(text))
    }
}
