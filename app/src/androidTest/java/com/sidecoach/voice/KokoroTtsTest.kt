package com.sidecoach.voice

import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
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
@Lane(Lane.Kind.VOICE)
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

    /**
     * An output that takes every sample at once: a line plays as fast as it is made, where the
     * phone's plays it out in real time. What is under test -- the pieces, the pauses around
     * them, the first sound after the first piece -- happens as it does on the phone; the ten
     * seconds of speech are not waited out. ([aStopEndsALineMidWay] plays through the phone's
     * own output.)
     */
    private class InstantOutput : KokoroTts.Output {
        @Volatile var taken = 0
            private set

        override val underrunCount: Int = 0

        override fun write(samples: FloatArray, at: Int, n: Int): Int {
            taken += n
            return n
        }

        override fun cut() = Unit

        override fun release() = Unit
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
        val line = "That was fine. The centre holds, and the bishop is doing real work; your knight is " +
            "still waiting on b1, though, and it wants to come to c3 or d2 before you castle."
        val pieces = Kokoro.plan(KokoroTokenizer(app), Py.strip(Notation.speak(line, voice = true)), 1.0f, "en-us")
        assertTrue("the line is not cut: ${pieces.map { it.first }}", pieces.size > 1)
        val output = InstantOutput()
        tts.makeOutput = { output }
        val began = SystemClock.elapsedRealtime()
        tts.say(line)
        assertTrue("the line never finished", quiet(tts, 60))
        tts.close()
        assertTrue("nothing was played", output.taken > 0)
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
     * An output that holds every write until the test opens it ([open]): the line playing stays
     * playing, and what the voice does meanwhile -- the next line made -- is seen, not timed.
     */
    private class GatedOutput : KokoroTts.Output {
        val open = CountDownLatch(1)
        override val underrunCount: Int = 0

        override fun write(samples: FloatArray, at: Int, n: Int): Int {
            open.await(30, TimeUnit.SECONDS)
            return n
        }

        override fun cut() = Unit

        override fun release() = Unit
    }

    /** The voice noting each line made ([made], one count per line: its last piece made) and
     * each line whose audio started ([played]). */
    private class Watched(val tts: KokoroTts) {
        val made = java.util.concurrent.Semaphore(0)
        val started = java.util.concurrent.Semaphore(0)
        val played: MutableList<String> = Collections.synchronizedList(ArrayList())

        init {
            tts.onSynthesis = { on -> if (!on) made.release() }
            tts.onPlay = { line ->
                played.add(line)
                started.release()
            }
        }
    }

    /**
     * Tai, 2026-10-03: "zero waits between sentences". The output holds the first line: the
     * second is made meanwhile, before the first has played, and follows it at once
     * (`test_the_next_line_is_made_while_the_one_before_plays`).
     */
    @Test
    fun theNextLineIsMadeWhileTheOneBeforePlays() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val output = GatedOutput()
        tts!!.makeOutput = { output }
        val watched = Watched(tts)
        tts.say("One.")
        tts.say("Two.")
        assertTrue("the first never started", watched.started.tryAcquire(30, TimeUnit.SECONDS))
        assertTrue("the second was not made while the first plays", watched.made.tryAcquire(2, 30, TimeUnit.SECONDS))
        assertEquals(listOf("One."), watched.played.toList())
        assertEquals("the second made, waiting behind the first", 1, tts.waiting)
        output.open.countDown()
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("One.", "Two."), watched.played.toList())
    }

    /**
     * The model writes a reply's next sentence after the first has begun to play: it is made the
     * moment it comes, not once the first has ended
     * (`test_a_line_that_comes_while_one_plays_is_made_before_it_ends`).
     */
    @Test
    fun aLineThatComesWhileOnePlaysIsMadeBeforeItEnds() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val output = GatedOutput()
        tts!!.makeOutput = { output }
        val watched = Watched(tts)
        tts.say("One.")
        assertTrue("the first never started", watched.started.tryAcquire(30, TimeUnit.SECONDS))
        assertTrue(watched.made.tryAcquire(30, TimeUnit.SECONDS))
        tts.say("Two.") // comes while "One." plays
        assertTrue("made while the first plays", watched.made.tryAcquire(30, TimeUnit.SECONDS))
        assertEquals(listOf("One."), watched.played.toList())
        output.open.countDown()
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("One.", "Two."), watched.played.toList())
    }

    /**
     * The pace plan's [KokoroTts.dropWaiting]: the lines behind the one playing -- the one made
     * ahead too -- go; the one playing is said to its end
     * (`test_dropping_what_waits_leaves_the_line_playing`).
     */
    @Test
    fun droppingWhatWaitsLeavesTheLinePlaying() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val output = GatedOutput()
        tts!!.makeOutput = { output }
        val watched = Watched(tts)
        tts.say("One.")
        tts.say("Two.")
        tts.say("Three.")
        assertTrue("the first never started", watched.started.tryAcquire(30, TimeUnit.SECONDS))
        assertTrue("the second was not made ahead", watched.made.tryAcquire(2, 30, TimeUnit.SECONDS))
        tts.dropWaiting()
        assertEquals("nothing waits", 0, tts.waiting)
        output.open.countDown()
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("One."), watched.played.toList())
    }

    /**
     * A fast line's word, held to the line's end, is made while it waits ([KokoroTts.prepare]):
     * said -- even while it is still being made -- its pieces go to the player without being made
     * again
     * (`test_a_line_prepared_ahead_is_not_made_again`).
     */
    @Test
    fun aLinePreparedAheadIsNotMadeAgain() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        tts!!.makeOutput = { InstantOutput() }
        val watched = Watched(tts)
        tts.prepare("Well found.")
        // Said while it is still being prepared: waited for, never made by the voice's worker.
        tts.say("Well found.")
        assertTrue("never said", watched.started.tryAcquire(30, TimeUnit.SECONDS))
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("Well found."), watched.played.toList())
        assertEquals("made again by the worker", 0, watched.made.availablePermits())
    }

    /**
     * A kept phrase ([KokoroTts.keep]: the short praise, made at the warm-up) plays at once when
     * said: nothing of it is made again (Tai, 2026-10-03: the voice "truly real time").
     */
    @Test
    fun aKeptPhraseIsNotMadeWhenSaid() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        tts!!.makeOutput = { InstantOutput() }
        tts.keep(listOf("Spot on.")).join()
        val watched = Watched(tts)
        tts.say("Spot on.")
        assertTrue("never said", watched.started.tryAcquire(30, TimeUnit.SECONDS))
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("Spot on."), watched.played.toList())
        assertEquals("made again", 0, watched.made.availablePermits())
    }

    /**
     * The faster graph put in the app's external files (`tools\install.ps1 -Voice`) is moved into
     * the model folder at the next load, and runs: copied whole, then the original removed.
     */
    @Test
    fun theFasterGraphPutBesideTheAppIsAdoptedAndRuns() {
        val models = File(app.filesDir, "models/kokoro")
        val full = File(models, KokoroTts.ONNX_FILE)
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", full.exists())
        val dir = File(app.cacheDir, "kokoro-adopt-${System.nanoTime()}").apply { mkdirs() }
        val outside = File(app.getExternalFilesDir(null), KokoroTts.FAST_FILE)
        try {
            File(models, KokoroTts.VOICES_FILE).copyTo(File(dir, KokoroTts.VOICES_FILE))
            // Any graph stands in for the faster one: what is under test is that it is taken.
            outside.parentFile!!.mkdirs()
            full.copyTo(outside, overwrite = true)
            val tts = KokoroTts(app, dir)
            assertTrue("the adopted graph did not load", tts.load())
            assertTrue("not moved into the model folder", File(dir, KokoroTts.FAST_FILE).length() == full.length())
            assertTrue("the original was left outside", !outside.exists())
            tts.close()
        } finally {
            outside.delete()
            dir.deleteRecursively()
        }
    }

    /**
     * The voice with its first line held in synthesis until [go] is let go ([synthesising] says
     * it began), and the lines whose audio starts, in order: the lines queued behind it wait
     * their turn.
     */
    private class Held(val tts: KokoroTts) {
        val synthesising = CountDownLatch(1)
        val go = CountDownLatch(1)
        private val lock = Object()
        private val heard = ArrayList<String>()
        val played: List<String> get() = synchronized(lock) { heard.toList() }

        init {
            tts.onSynthesis = { on ->
                if (on && synthesising.count > 0) {
                    synthesising.countDown()
                    assertTrue("the test never let the line go", go.await(30, TimeUnit.SECONDS))
                }
            }
            tts.onPlay = { line ->
                synchronized(lock) {
                    heard.add(line)
                    lock.notifyAll()
                }
            }
        }

        /** Waits until [line]'s audio has started (the voice's own signal); 30 s a safety net. */
        fun heard(line: String) {
            val until = SystemClock.elapsedRealtime() + 30_000
            synchronized(lock) {
                while (line !in heard) {
                    val left = until - SystemClock.elapsedRealtime()
                    assertTrue("$line was never said: $heard", left > 0)
                    lock.wait(left)
                }
            }
        }
    }

    /**
     * Round 3 audit, E5 (Tai, 2026-10-02): a card is being said, a coach line is queued after
     * it, and the review is closed. Before, the room's stop stopped the voice only when a room had
     * given the last line -- here the coach had, so the card played on after the review closed.
     * Now the review's own lines stop, the one being made and a later one, and the coach's line is
     * said. `test_a_rooms_stop_cuts_its_own_line_and_leaves_the_coachs_queued_one`.
     */
    @Test
    fun aRoomsStopCutsItsOwnLineAndLeavesTheCoachsQueuedOne() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val held = Held(tts!!)
        tts.say("Card one.", owner = "review")
        assertTrue("the card was never made", held.synthesising.await(30, TimeUnit.SECONDS))
        tts.say("Coach line.")
        tts.say("Card two.", owner = "review")
        tts.stop("review")
        held.go.countDown()
        held.heard("Coach line.") // after the cut card, the line queued behind it
        assertTrue("the voice never went quiet", quiet(tts, 30))
        tts.close()
        assertEquals(listOf("Coach line."), held.played)
    }

    /**
     * The other order: the coach is speaking and a card is queued after it, then the review is
     * closed. Before, the card having given the last line, the stop cut the coach off; now the
     * coach's line is said and the card never is.
     * `test_a_rooms_stop_leaves_the_coachs_playing_line_and_drops_its_queued_card`.
     */
    @Test
    fun aRoomsStopLeavesTheCoachsPlayingLineAndDropsItsQueuedCard() {
        val tts = tts()
        assumeTrue("no Kokoro (tools/test.ps1 pushes it)", tts != null)
        val held = Held(tts!!)
        tts.say("Coach line.")
        assertTrue("the line was never made", held.synthesising.await(30, TimeUnit.SECONDS))
        tts.say("Card one.", owner = "review")
        tts.stop("review")
        held.go.countDown()
        held.heard("Coach line.")
        assertTrue("the voice never went quiet", quiet(tts, 30)) // the coach's line, the last
        assertEquals(listOf("Coach line."), held.played)
        // A card said after the room's stop is said.
        tts.say("Card two.", owner = "review")
        held.heard("Card two.")
        tts.close()
        assertEquals(listOf("Coach line.", "Card two."), held.played)
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
