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
        /** Counted down when a held clip begins to be recognised: the test acts by this state. */
        @Volatile var working = CountDownLatch(1)
        override val loaded = true
        override val loadSeconds: Double? = 0.0
        override fun healthcheck() = true to "fake"
        override fun load() = true
        override fun prime(): Double? = 0.0
        override fun close() = Unit
        override fun transcribe(audio: FloatArray): String {
            if (audio.size == held) {
                working.countDown()
                gate.await(2, TimeUnit.SECONDS)
            }
            return List(maxOf(1, (audio.size / 16000.0 / 0.4).toInt())) { "hello" }.joinToString(" ")
        }
    }

    /** What was asked of the session's loop (`engine._commands`). */
    private val asked = PanelEngine.commands
    private var recogniser = WordsByLength()
    private val realTimer = PanelListen.makeTimer

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
        PanelListen.makeTimer = realTimer
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

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    /** The microphone allowed, as the person allows it once (the phone's own). */
    private fun allowMicrophone() {
        InstrumentationRegistry.getInstrumentation().uiAutomation
            .grantRuntimePermission(app.packageName, android.Manifest.permission.RECORD_AUDIO)
    }

    /** A 16 kHz mono 16-bit WAV of the desktop's tests, as floats (`server.hear` reads it so). */
    private fun clip(name: String): FloatArray {
        val raw = InstrumentationRegistry.getInstrumentation().context
            .assets.open("voice/$name").use { it.readBytes() }
        val wav = java.nio.ByteBuffer.wrap(raw).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var at = 12
        while (String(raw, at, 4, Charsets.US_ASCII) != "data") at += 8 + wav.getInt(at + 4)
        val start = at + 8
        return FloatArray(wav.getInt(at + 4) / 2) { wav.getShort(start + 2 * it) / 32768.0f }
    }

    /**
     * The settings page's live test (`test_the_microphone_test_keeps_what_it_hears_and_asks_no_coach`):
     * while it is on, an utterance is transcribed and corrected into the test's list, the coach is
     * not asked, and the listening mode it interrupted comes back when it stops.
     */
    @Test
    fun theMicrophoneTestKeepsWhatItHearsAndAsksNoCoach() {
        allowMicrophone()
        PanelListen.build(app)
        PanelListen.setListen("off")
        PanelListen.stt = object : Recogniser {
            override val loaded = true
            override val loadSeconds: Double? = 0.0
            override fun healthcheck() = true to "fake"
            override fun load() = true
            override fun prime(): Double? = 0.0
            override fun close() = Unit
            override fun transcribe(audio: FloatArray) = "Night F3"
        }
        PanelEngine.set("fen" to com.sidecoach.chess.STARTING_FEN)
        val status = PanelListen.hearStatus()
        assertEquals(false, status.getBoolean("on"))
        assertEquals(0, status.getJSONArray("heard").length())
        val started = PanelListen.setHearTest(true)
        // `pytest.skip("no microphone here")`.
        org.junit.Assume.assumeTrue("no microphone here: $started", started.getBoolean("ok"))
        try {
            assertTrue(started.toString(), started.getBoolean("on"))
            assertEquals("vad", PanelListen.listener!!.mode)
            // An utterance while the test is on: kept for the page, corrected, and not an ask.
            PanelListen.hear(silence(1.0))
            val heard = PanelListen.hearStatus()
            val first = heard.getJSONArray("heard").getJSONObject(0)
            assertEquals("corrected to the legal move", "Nf3", first.getString("heard"))
            assertEquals("Night F3", first.getString("raw"))
            assertTrue(first.getLong("stt_ms") >= 0 && heard.has("level") && heard.getBoolean("ok"))
            assertTrue("nothing was asked of the coach", asked.isEmpty())
        } finally {
            val stopped = PanelListen.setHearTest(false)
            assertEquals(false, stopped.getBoolean("on"))
            assertEquals("off", PanelListen.listener!!.mode)
            PanelEngine.set("fen" to null)
        }
    }

    /**
     * Listening saved while the microphone test runs (`test_listening_saved_during_the_microphone_test_is_kept`):
     * the test gave back the mode it found when it ended, undoing the save, and the save switched
     * the test's own listening, so it went deaf (the settings review, 2026-10-01). The test keeps
     * listening, and gives back the mode saved.
     */
    @Test
    fun listeningSavedDuringTheMicrophoneTestIsKept() {
        allowMicrophone()
        PanelListen.build(app)
        val store = SettingsStore(app)
        val saved = store.values()
        PanelListen.stt = object : Recogniser {
            override val loaded = true
            override val loadSeconds: Double? = 0.0
            override fun healthcheck() = true to "fake"
            override fun load() = true
            override fun prime(): Double? = 0.0
            override fun close() = Unit
            override fun transcribe(audio: FloatArray) = ""
        }
        try {
            store.save(store.values().also { it.getJSONObject("voice").put("listen", "ptt") })
            PanelListen.setListen("ptt")
            val started = PanelListen.setHearTest(true)
            org.junit.Assume.assumeTrue("no microphone here: $started", started.getBoolean("ok"))
            assertEquals("vad", PanelListen.listener!!.mode)
            val off = JSONObject().put("voice", JSONObject().put("listen", "off"))
            val res = SettingsApi(store).update(JSONObject().put("settings", off))
            assertTrue(res.toString(), res.getBoolean("ok"))
            assertEquals("the test still hears", "vad", PanelListen.listener!!.mode)
            assertEquals("the panel shows the mode saved", "off", PanelEngine.poll(0).getJSONObject("state").getString("listen"))
            PanelListen.setHearTest(false)
            assertEquals("off", PanelListen.listener!!.mode)
            assertEquals("off", store.listen())
        } finally {
            if (PanelListen.hearStatus().getBoolean("on")) PanelListen.setHearTest(false)
            PanelListen.setListen("off")
            store.save(saved)
        }
    }

    /** A microphone test whose page went away ends at the panel's next poll
     * (`test_a_microphone_test_whose_window_closed_ends_at_the_next_poll`, voice finding 10). */
    @Test
    fun aMicrophoneTestWhosePageWentAwayEndsAtTheNextPoll() {
        allowMicrophone()
        PanelListen.build(app)
        PanelListen.stt = object : Recogniser {
            override val loaded = true
            override val loadSeconds: Double? = 0.0
            override fun healthcheck() = true to "fake"
            override fun load() = true
            override fun prime(): Double? = 0.0
            override fun close() = Unit
            override fun transcribe(audio: FloatArray) = ""
        }
        try {
            PanelListen.setListen("ptt")
            val started = PanelListen.setHearTest(true)
            org.junit.Assume.assumeTrue("no microphone here: $started", started.getBoolean("ok"))
            PanelEngine.poll(0) // the page still polling
            assertTrue(PanelListen.hearTestRunning)
            PanelListen.hearTestSeenAgo(PanelListen.HEAR_TEST_TIMEOUT_S + 1) // the page went away
            PanelEngine.poll(0)
            assertTrue(!PanelListen.hearTestRunning)
            assertEquals("ptt", PanelListen.listener!!.mode)
        } finally {
            if (PanelListen.hearTestRunning) PanelListen.setHearTest(false)
            PanelListen.setListen("off")
        }
    }

    /**
     * `test_a_spoken_move_reaches_the_coach_as_san`, with the phone's recogniser: a real clip
     * is heard, corrected to the legal move, put in the text box and asked of the coach after the
     * quiet; the recogniser was loaded once and says where.
     */
    @Test
    fun aSpokenMoveReachesTheCoachAsSan() {
        val recogniser = com.sidecoach.voice.ParakeetStt(app, PanelListen.modelDir(app))
        org.junit.Assume.assumeTrue("no Parakeet (tools/test.ps1 pushes it)", recogniser.load())
        PanelListen.stt = recogniser
        PanelEngine.set("fen" to com.sidecoach.chess.STARTING_FEN)
        val cursor = PanelEngine.poll(Int.MAX_VALUE).getInt("cursor")
        PanelListen.hear(clip("knight_f3.wav"))
        val lines = PanelEngine.poll(cursor).getJSONArray("lines")
        val heard = (0 until lines.length()).map { lines.getJSONObject(it) }
            .filter { it.getString("chan") == "voice" && it.getString("text").startsWith("heard:") }
        assertTrue("nothing heard", heard.isNotEmpty())
        assertEquals("heard: Nf3", heard.last().getString("text"))
        assertEquals("Nf3", heard.last().getJSONObject("data").getJSONArray("moves").getString(0))
        assertTrue(heard.last().getJSONObject("data").getLong("stt_ms") < 2000)
        // The corrected text is in the text box, and asked of the coach after the quiet.
        assertEquals("Nf3", dictation().getString("text"))
        assertEquals("ask_spoken" to "Nf3", asked.poll(3, TimeUnit.SECONDS))
        assertTrue(recogniser.loaded)
        assertEquals("parakeet-tdt_ctc-110m, cpu", recogniser.healthcheck().second)
        recogniser.close()
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

    /**
     * A microphone's frames, put in by the test, read by the listener; when there are none, the
     * listener's wait for the next ends only when the test says ([idle]), never on a clock
     * (tests/fakes.py `SteppedFrames`). [idle] returning means every frame put in was read and
     * handled; [free] lets the listener go.
     */
    private class SteppedFrames : LinkedBlockingQueue<FloatArray>() {
        private val go = java.util.concurrent.Semaphore(0)
        private val waiting = java.util.concurrent.Semaphore(0)
        private var parked = false
        @Volatile private var free = false

        override fun poll(timeout: Long, unit: TimeUnit): FloatArray? {
            super.poll()?.let { return it }
            if (free) return null
            waiting.release() // every frame read and handled: the listener waits here
            go.tryAcquire(HANG_S, TimeUnit.SECONDS)
            return null
        }

        private fun park() = assertTrue("the listener never came for a frame", waiting.tryAcquire(HANG_S, TimeUnit.SECONDS))

        fun idle() {
            if (!parked) {
                park()
                parked = true
            }
            go.release()
            park()
        }

        fun free() {
            free = true
            go.release()
        }

        companion object {
            const val HANG_S = 30L
        }
    }

    /** A voice gate that reads a frame's own value as its speech probability, and counts. */
    private class LevelGate : VoiceGate {
        @Volatile var asked = 0
        override fun load() = true
        override fun healthcheck() = true to "level"
        override fun reset() = Unit
        override fun probability(input: FloatArray): Float {
            asked += 1
            return input[0]
        }
    }

    /**
     * The echo guard (Tai, 2026-10-02; `test_the_microphone_is_ignored_while_the_coach_speaks_but_not_a_held_key`):
     * while the coach's voice plays, what the microphone hears is ignored -- no utterance starts,
     * nothing of it is kept for the pre-roll, and an utterance begun before the voice ends as the
     * player's words alone. Listening resumes with the first frame after the voice. A held
     * push-to-talk key is heard all the same. Before, the start bar was only raised to 0.85.
     */
    @Test
    fun theMicrophoneIsIgnoredWhileTheCoachSpeaksButNotAHeldKey() {
        // The frame's value is its speech probability: the echo crosses the raised bar of 0.85.
        val echo = 0.875f
        val voice = 0.75f
        val quiet = 0f
        val steps = SteppedFrames()
        val mic = object : FrameSource {
            override val frames: BlockingQueue<FloatArray> = steps
            override val detail = "stepped by the test"
            override val isOpen = true
            override fun open() = true
            override fun close() = Unit
        }
        fun frames(value: Float, n: Int) = repeat(n) { steps.put(FloatArray(HOP) { value }) }
        fun count(said: FloatArray, value: Float) = said.count { it == value }
        val speaking = java.util.concurrent.atomic.AtomicBoolean(false)
        val starts = java.util.concurrent.CopyOnWriteArrayList<String>()
        val heard = LinkedBlockingQueue<FloatArray>()
        val gate = LevelGate()
        val listener = Listener(
            mic,
            gate,
            onUtterance = { audio, _ -> heard.put(audio) },
            onSpeechStart = { starts.add("start") },
            coachSpeaking = { speaking.get() },
        )
        listener.setMode("vad")
        try {
            // The coach speaks; the microphone hears it loud and long: nothing starts.
            speaking.set(true)
            frames(echo, 40)
            steps.idle()
            assertEquals(emptyList<String>(), starts.toList())
            assertTrue(heard.isEmpty())
            assertEquals(0, gate.asked)
            // The voice stops: the very next frames are heard, with no echo in their pre-roll.
            speaking.set(false)
            frames(voice, 12)
            frames(quiet, 14)
            steps.idle()
            assertEquals(listOf("start"), starts.toList())
            var said = heard.poll(SteppedFrames.HANG_S, TimeUnit.SECONDS)!!
            assertEquals(0, count(said, echo))
            assertTrue(count(said, voice) > 0)
            // The player is talking when a line begins: the utterance ends as their words alone.
            frames(voice, 12)
            steps.idle()
            assertEquals(listOf("start", "start"), starts.toList())
            speaking.set(true)
            frames(echo, 30)
            steps.idle()
            said = heard.poll(SteppedFrames.HANG_S, TimeUnit.SECONDS)!!
            assertEquals(0, count(said, echo))
            assertEquals(12 * HOP, count(said, voice))
            assertEquals(listOf("start", "start"), starts.toList())
            // Push-to-talk: the coach's words before the press are not kept (the pre-roll); what
            // is said while the key is held is, whatever the voice is doing.
            listener.setMode("ptt")
            frames(echo, com.sidecoach.voice.PRE_ROLL_FRAMES)
            steps.idle()
            listener.press()
            frames(voice, 20)
            steps.idle()
            listener.release()
            said = heard.poll(SteppedFrames.HANG_S, TimeUnit.SECONDS)!!
            assertEquals(0, count(said, echo))
            assertEquals(20 * HOP, count(said, voice))
        } finally {
            steps.free()
            listener.setMode("off")
        }
    }

    /**
     * The key pressed again while the first half was being recognised: that recognition ended
     * after the press and armed the second of quiet, and the first half went alone while the
     * player held the key (`test_a_press_while_the_last_words_are_recognised_keeps_one_question`,
     * voice finding 5). Driven by the recogniser's state and a timer the test fires: it slept to
     * land the press and waited past the real second of quiet (round 3 audit, 2026-10-02).
     */
    @Test
    fun aPressWhileTheLastWordsAreRecognisedKeepsOneQuestion() {
        val armed = java.util.concurrent.CopyOnWriteArrayList<Pair<java.util.concurrent.CompletableFuture<Unit>, () -> Unit>>()
        PanelListen.makeTimer = { _, send ->
            java.util.concurrent.CompletableFuture<Unit>().also { armed.add(it to send) }
        }
        val slow = CountDownLatch(1)
        recogniser.gate = slow
        recogniser.held = 16000
        PanelListen.dictationHold() // the key down
        val first = Thread { PanelListen.onUtterance(silence(1.0), 1.0) } // up: recognising ...
        first.start()
        assertTrue(recogniser.working.await(5, TimeUnit.SECONDS))
        PanelListen.dictationHold() // ... and down again while it works
        slow.countDown()
        first.join(5000)
        assertTrue(dictation().isNull("sends_at"))
        assertTrue("a send armed while the key is held", armed.none { !it.first.isCancelled })
        assertTrue(asked.isEmpty())
        PanelListen.onUtterance(silence(0.5), 0.5) // up
        val live = armed.filter { !it.first.isCancelled }
        assertEquals(1, live.size)
        live[0].second() // the second of quiet
        assertEquals("ask_spoken" to "hello hello hello", asked.poll())
    }

    /**
     * Push-to-talk ends at the longest utterance, and a mode switch with the key down lets go of
     * it (`test_push_to_talk_stops_at_the_longest_utterance_and_a_mode_switch_lets_go`, voice
     * finding 7).
     */
    @Test
    fun pushToTalkStopsAtTheLongestUtteranceAndAModeSwitchLetsGo() {
        val heard = java.util.concurrent.CopyOnWriteArrayList<Double>()
        val mic = FedMic()
        val listener = Listener(mic, ScriptedGate(emptyList()), onUtterance = { audio, _ -> heard.add(audio.size / 16000.0) })
        listener.setMode("ptt")
        try {
            listener.press()
            repeat(((com.sidecoach.voice.MAX_UTTERANCE_S + 5) * 16000 / HOP).toInt()) { mic.frames.put(FloatArray(HOP) { 0.01f }) }
            val deadline = System.currentTimeMillis() + 5000
            while ((heard.isEmpty() || mic.frames.isNotEmpty()) && System.currentTimeMillis() < deadline) Thread.sleep(10)
            assertEquals(1, heard.size)
            assertTrue("${heard[0]}", heard[0] <= com.sidecoach.voice.MAX_UTTERANCE_S + 0.1)
            listener.release()
            listener.press()
            listener.setMode("vad")
            listener.setMode("ptt")
            // The fresh press is taken, by the listener's own state as the desktop asserts it: a
            // second's speech, a 100 ms sleep and a release raced the listener (round 3 audit).
            listener.press()
            assertTrue("the fresh press was ignored", listener.pressTaken)
        } finally {
            listener.setMode("off")
        }
    }

    /** A microphone whose input failed is reopened (`test_a_microphone_whose_stream_died_is_reopened`,
     * voice finding 17). */
    @Test
    fun aMicrophoneWhoseInputFailedIsReopened() {
        val mic = object : FrameSource {
            override val frames: BlockingQueue<FloatArray> = LinkedBlockingQueue()
            override val detail = "fed by the test"
            override val isOpen = true
            @Volatile var opens = 0
            @Volatile var dead = false
            override fun open(): Boolean {
                opens += 1
                dead = false
                return true
            }
            override fun close() = Unit
            override fun stalled() = dead
        }
        val listener = Listener(mic, ScriptedGate(emptyList()), onUtterance = { _, _ -> })
        listener.setMode("ptt")
        try {
            assertEquals(1, mic.opens)
            mic.dead = true // the input failed
            val deadline = System.currentTimeMillis() + (com.sidecoach.voice.MIC_REOPEN_S * 1000).toLong() + 3000
            while (mic.opens < 2 && System.currentTimeMillis() < deadline) Thread.sleep(20)
            assertEquals(2, mic.opens)
        } finally {
            listener.setMode("off")
        }
    }
}
