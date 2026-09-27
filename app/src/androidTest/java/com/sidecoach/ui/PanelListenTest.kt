package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.voice.FrameSource
import com.sidecoach.voice.HOP
import com.sidecoach.voice.Listener
import com.sidecoach.voice.Recogniser
import com.sidecoach.voice.VoiceGate
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.json.JSONObject
import java.util.concurrent.BlockingQueue
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * The panel's hearing, as the desktop's tests have it (`tests/test_panel_engine.py`'s
 * dictation tests, `tests/test_voice_hearing.py`'s listener test): what is said reaches the text
 * box as it is recognised and is asked of the coach a second after the utterance ends, unless
 * more is said, which appends, or the person types; and an utterance that comes to nothing is
 * said, so a click does not hold the words before it.
 */
@RunWith(AndroidJUnit4::class)
class PanelListenTest {

    /** Speech recognition that reads the words off the clip's length: one word per 0.4 s. */
    private class WordsByLength : Recogniser {
        /** A clip of exactly [held] samples waits for [gate] first (a slow partial). */
        @Volatile var gate = CountDownLatch(0)
        @Volatile var held = -1
        override val loaded = true
        override val loadSeconds: Double? = 0.0
        override fun healthcheck() = true to "fake"
        override fun load() = true
        override fun prime(): Double? = 0.0
        override fun close() = Unit
        override fun transcribe(audio: FloatArray): String {
            if (audio.size == held) gate.await(2, TimeUnit.SECONDS)
            return List(maxOf(1, (audio.size / 16000.0 / 0.4).toInt())) { "hello" }.joinToString(" ")
        }
    }

    /** What was asked of the session's loop (`engine._commands`). */
    private val asked = PanelEngine.commands
    private var recogniser = WordsByLength()

    @Before
    fun setUp() {
        PanelListen.cancelDictation()
        PanelEngine.set("fen" to null)
        PanelListen.sendDelayS = 0.3
        recogniser = WordsByLength()
        PanelListen.stt = recogniser
        // A session: the ask is queued for its loop.
        asked.clear()
        PanelEngine.set("running" to true)
    }

    @After
    fun tearDown() {
        PanelEngine.set("running" to false)
        asked.clear()
        PanelListen.sendDelayS = PanelListen.DICTATION_SEND_DELAY_S
        PanelListen.cancelDictation()
    }

    private fun dictation(): JSONObject =
        PanelEngine.poll(Int.MAX_VALUE).getJSONObject("state").getJSONObject("dictation")

    private fun assertDictation(text: String, live: Boolean, sends: Boolean?) {
        val d = dictation()
        assertEquals(d.toString(), text, d.getString("text"))
        assertEquals(d.toString(), live, d.getBoolean("live"))
        if (sends != null) assertEquals(d.toString(), sends, !d.isNull("sends_at"))
    }

    private fun silence(seconds: Double) = FloatArray((16000 * seconds).toInt())

    @Test
    fun theDictationStreamsIntoTheBoxAndSendsAfterASecondOfQuiet() {
        assertDictation("", live = false, sends = false)
        // The key goes down: the box says it is listening.
        PanelListen.dictationHold()
        assertTrue(dictation().getBoolean("live"))
        // Partials while the key is held: the words so far, live.
        PanelListen.onPartial(silence(1.0))
        val deadline = System.currentTimeMillis() + 2000
        while (dictation().getString("text") != "hello hello" && System.currentTimeMillis() < deadline) Thread.sleep(20)
        assertDictation("hello hello", live = true, sends = false)
        // The key comes up: the final result replaces the partial and the send is armed.
        PanelListen.onUtterance(silence(2.0), 2.0)
        assertDictation("hello hello hello hello hello", live = false, sends = true)
        assertTrue(asked.isEmpty())
        // Pressed again before it sends: it waits, and the next utterance appends.
        PanelListen.dictationHold()
        assertTrue(dictation().isNull("sends_at"))
        PanelListen.onUtterance(silence(0.5), 0.5)
        assertEquals("hello hello hello hello hello hello", dictation().getString("text"))
        // ...and after the quiet, one ask with everything, and the box is clear.
        assertEquals("ask_spoken" to "hello hello hello hello hello hello", asked.poll(2, TimeUnit.SECONDS))
        assertDictation("", live = false, sends = false)
        // A partial still being recognised when its utterance ends is dropped: the final
        // result has those words, and they must not show twice.
        val slow = CountDownLatch(1)
        recogniser.gate = slow
        recogniser.held = 16000
        PanelListen.onPartial(silence(1.0)) // blocks in transcribe
        PanelListen.onUtterance(silence(2.0), 2.0)
        slow.countDown()
        Thread.sleep(200)
        assertEquals("hello hello hello hello hello", dictation().getString("text"))
        PanelListen.cancelDictation()
        // No 'heard:' line at info level: the words are on the page, not in the terminal.
        val lines = PanelEngine.poll(0).getJSONArray("lines")
        for (i in 0 until lines.length()) {
            val line = lines.getJSONObject(i)
            assertTrue(!(line.getString("text").startsWith("heard:") && line.getString("level") == "info"))
        }
    }

    @Test
    fun typingOverADictationKeepsTheWordsAndStopsTheSend() {
        PanelListen.onUtterance(silence(1.0), 1.0)
        assertTrue(!dictation().isNull("sends_at"))
        val cancelled = PanelListen.cancelDictation()
        assertEquals(true, cancelled.getBoolean("ok"))
        assertEquals("hello hello", cancelled.getString("text"))
        assertDictation("", live = false, sends = false)
        Thread.sleep(1200)
        assertTrue(asked.isEmpty())
    }

    @Test
    fun aClickAfterTheWordsDoesNotHoldTheirSend() {
        PanelListen.onUtterance(silence(1.0), 1.0)
        PanelListen.onSpeechStart() // the click begins
        assertDictation("hello hello", live = true, sends = false)
        PanelListen.onDropped() // ...and is dropped
        assertDictation("hello hello", live = false, sends = true)
        assertEquals("ask_spoken" to "hello hello", asked.poll(2, TimeUnit.SECONDS))
        // With nothing said before it, the box simply stops listening.
        PanelListen.onSpeechStart()
        PanelListen.onDropped()
        assertDictation("", live = false, sends = false)
    }

    @Test
    fun anAskBeforeStartIsAnsweredAtOnceNotKeptForALaterSession() {
        PanelEngine.set("running" to false)
        val before = PanelEngine.poll(0).getInt("cursor")
        PanelEngine.ask("what should I play")
        assertTrue("an ask with no session was queued", asked.isEmpty())
        val lines = PanelEngine.poll(before).getJSONArray("lines")
        val texts = (0 until lines.length()).map { lines.getJSONObject(it) }
        assertTrue(texts.any { it.getString("text") == "you: what should I play" })
        val reply = texts.first { it.getString("chan") == "coach" && "Press Start first" in it.getString("text") }
        assertTrue(reply.getJSONObject("data").getString("full").startsWith("Press Start first"))
    }

    /**
     * The settings page's live test (`test_the_microphone_test_keeps_what_it_hears_and_asks_no_
     * coach`): while it is on, an utterance is transcribed and corrected into the test's list,
     * the coach is not asked, and the listening mode it interrupted comes back when it stops.
     */
    @Test
    fun theMicrophoneTestKeepsWhatItHearsAndAsksNoCoach() {
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        InstrumentationRegistry.getInstrumentation().uiAutomation
            .grantRuntimePermission(app.packageName, android.Manifest.permission.RECORD_AUDIO)
        PanelListen.build(app)
        PanelListen.stt = recogniser
        PanelEngine.set("fen" to com.sidecoach.chess.STARTING_FEN)
        val mode = PanelListen.listener?.mode
        val status = PanelListen.hearStatus()
        assertTrue(!status.getBoolean("on") && status.getJSONArray("heard").length() == 0)
        val started = PanelListen.setHearTest(true)
        org.junit.Assume.assumeTrue("no microphone here: $started", started.getBoolean("ok"))
        try {
            assertTrue(started.getBoolean("on") && PanelListen.listener?.mode != "off")
            PanelListen.hear(silence(1.0))
            val heard = PanelListen.hearStatus()
            assertTrue(heard.getBoolean("ok") && heard.has("level"))
            val first = heard.getJSONArray("heard").getJSONObject(0)
            assertEquals("hello hello", first.getString("heard"))
            assertTrue(first.getString("raw").isNotEmpty() && first.getLong("stt_ms") >= 0)
            assertTrue("nothing was asked of the coach", asked.isEmpty())
        } finally {
            assertTrue(!PanelListen.setHearTest(false).getBoolean("on"))
            assertEquals(mode, PanelListen.listener?.mode)
            PanelEngine.set("fen" to null)
        }
    }

    /** A voice gate that answers a script of probabilities, one per frame. */
    private class ScriptedGate(probs: List<Double>) : VoiceGate {
        private val probs = ArrayDeque(probs)
        override fun load() = true
        override fun healthcheck() = true to "scripted"
        override fun reset() = Unit
        override fun probability(input: FloatArray): Float = (probs.removeFirstOrNull() ?: 0.0).toFloat()
    }

    /** A microphone whose frames the test puts in. */
    private class FedMic : FrameSource {
        override val frames: BlockingQueue<FloatArray> = LinkedBlockingQueue()
        override val detail = "fed by the test"
        override val isOpen = true
        override fun open() = true
        override fun close() = Unit
    }

    @Test
    fun anUtteranceThatComesToNothingIsSaid() {
        val starts = ArrayList<String>()
        val dropped = java.util.concurrent.CopyOnWriteArrayList<String>()
        val heard = ArrayList<Double>()
        val mic = FedMic()
        // Two confident frames start speech; the silence after it is a click, not an ask.
        val gate = ScriptedGate(listOf(0.9, 0.9) + List(20) { 0.1 })
        val listener = Listener(
            mic,
            gate,
            onUtterance = { _, seconds -> heard.add(seconds) },
            onSpeechStart = { starts.add("start") },
            onDropped = { dropped.add("dropped") },
        )
        listener.setMode("vad")
        repeat(22) { mic.frames.put(FloatArray(HOP)) }
        val deadline = System.currentTimeMillis() + 3000
        while (dropped.isEmpty() && System.currentTimeMillis() < deadline) Thread.sleep(10)
        assertEquals(listOf("start"), starts)
        assertEquals(listOf("dropped"), dropped.toList())
        assertTrue(heard.isEmpty())
        // Push-to-talk: a press and a release with nothing kept in between.
        listener.setMode("ptt")
        listener.press()
        listener.release()
        assertEquals(listOf("dropped", "dropped"), dropped.toList())
        assertTrue(heard.isEmpty())
        listener.setMode("off")
    }
}
