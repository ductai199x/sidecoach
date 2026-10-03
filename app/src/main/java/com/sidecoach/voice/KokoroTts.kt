package com.sidecoach.voice

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue

/**
 * The coach's lines spoken with Kokoro. `sidecoach/voice/kokoro.py`'s `KokoroTTS`: non-blocking,
 * and it never raises out of [say] or [stop].
 *
 * The graph is loaded once (by [load], or on the first line) and lives as long as the provider;
 * a voice is only a style chosen at synthesis time, so changing it reloads nothing. [say]
 * queues a line and returns at once; one worker thread synthesises in order, as the desktop's
 * does, and the next line waits until this one has been handed to the output.
 *
 * The phone's own (Tai, 2026-09-23): a line is spoken in pieces ([Kokoro.plan]), and a player
 * thread plays each piece while the worker makes the next, so the first words come about twice
 * as soon. The first piece waits until enough of the line is made that the rest arrives before
 * it is needed, so a line never stops mid-way. While a piece is being made the engine is
 * paused ([onSynthesis]): the two share the phone's big cores, and with both at work each runs
 * at half speed.
 *
 * Where the desktop plays: one output stream opened at the first line and kept open, so a
 * Bluetooth headset does not swallow the first syllables while its link wakes (Tai,
 * 2026-09-20). The desktop keeps silence flowing through its stream between lines; on the
 * phone the track is kept open and playing and Android plays the silence itself when nothing
 * is written, which costs no work and lets the audio path rest when the coach has been quiet
 * for a while -- a phone's battery pays for every second the audio hardware stays awake.
 * The track names no device, so Android moves it to the output it routes to now (the desktop
 * follows the system's default output by Windows' notification); one it lost is opened afresh
 * mid-line ([writePiece]).
 */
/** What a line's samples are written to: the voice's [AudioTrack], or a test's. */
fun interface PcmSink {
    fun write(samples: FloatArray, at: Int, n: Int): Int
}

class KokoroTts(
    private val context: Context,
    private val modelDir: File,
    /** The coach's voice: a style, not a model -- changing it reloads nothing. */
    @Volatile var voice: String = "af_sarah",
    @Volatile var speed: Float = 1.0f,
    private val lang: String = "en-us",
) : TtsProvider {
    override val name = "kokoro"

    /** The graph run: the faster one ([FAST_FILE]) when it is there, else the full-precision one. */
    private val onnxPath: File
        get() = File(modelDir, FAST_FILE).takeIf { it.exists() } ?: File(modelDir, ONNX_FILE)
    val voicesPath: File get() = File(modelDir, VOICES_FILE)

    @Volatile private var kokoro: Kokoro? = null
    @Volatile private var failed = false
    @Volatile private var closed = false
    @Volatile private var failedFiles: List<Pair<Long, Long>?> = emptyList()
    private val loadLock = Any()
    @Volatile var loadSeconds: Double? = null
        private set

    /**
     * Called with the line as given to [say] the moment its audio starts to play, for whoever
     * measures the way from a move to the first sound (the panel).
     */
    @Volatile var onPlay: ((String) -> Unit)? = null

    /**
     * Called with true as each piece of a line starts being synthesised and false when it is
     * done, for whoever pauses the engine meanwhile (the panel): Kokoro or Stockfish at work,
     * not both (Tai, 2026-09-23).
     */
    @Volatile var onSynthesis: ((Boolean) -> Unit)? = null

    /**
     * Called with the line as given to [say] and about how long it speaks, in milliseconds, the
     * moment its audio starts: for the captions, which show its words as they are said.
     */
    @Volatile var onCaption: ((String, Long) -> Unit)? = null

    /**
     * A queued line's owner, and the stops it came after: everything's ([generation]) and its
     * owner's own ([cut]). A stop since then cancels it. `kokoro._Stamp`.
     */
    private data class Stamp(val owner: String, val generation: Int, val cut: Int)

    private data class Line(
        val spoken: String,
        val given: String,
        val voice: String?,
        val stamp: Stamp,
        val queuedAt: Long = SystemClock.elapsedRealtime(),
        val id: Long = 0,
    ) {
        /**
         * When this line may start to play, in elapsed-realtime milliseconds: its own, as the
         * next line is made while it plays (one value for "the line being made" was overwritten
         * by the next). Under [lock].
         */
        var startAt: Long = Long.MAX_VALUE
    }

    /** Numbers the lines queued (`Line.id`). */
    private val lineIds = java.util.concurrent.atomic.AtomicLong()

    /**
     * The lines the worker has taken and the player has not begun, by id, with their owner: the
     * next line made while one plays among them (`waiting`). Under [lock].
     */
    private val taken = LinkedHashMap<Long, String>()

    /** Taken lines dropped by [dropWaiting]: never played. Under [lock]. */
    private val dropped = HashSet<Long>()

    /** The line playing now (its id and owner), null between lines. Under [lock]. */
    private var playingLine: Pair<Long, String>? = null

    /** Lines made ahead by [prepare] (a fast line's word), by their words, in pieces. Under [lock]. */
    private val prepared = LinkedHashMap<String, List<FloatArray>>()

    /** Lines being made ahead by [prepare], by their words: counted down once filed. Under [lock]. */
    private val preparing = HashMap<String, CountDownLatch>()

    private val queue = LinkedBlockingQueue<Line>()
    private var worker: Thread? = null

    /**
     * A piece of a line's audio on its way to the output: the line with its first piece (whose
     * playing waits for [startAt]), and with its last the latch the worker waits on.
     */
    private class Chunk(
        val samples: FloatArray,
        val mine: Stamp,
        val first: Line?,
        val done: CountDownLatch?,
        /** With the first piece: the line's speech, what is made and the rest's estimate. */
        val speechMs: Long = 0,
        /** The line it is a piece of (`Line.id`). */
        val lineId: Long = 0,
    )

    private val chunks = LinkedBlockingQueue<Chunk>()
    private var player: Thread? = null

    /**
     * Bumped on every stop of everything, so a synthesis in flight knows its line was
     * cancelled; [cuts] the same, an owner's own stops (a room's), which cancel that owner's
     * lines and no one else's. Under [lock].
     */
    private var generation = 0
    private val cuts = HashMap<String, Int>()

    /** The line being synthesised or played (its stamp), null between lines. Under [lock]. */
    private var current: Stamp? = null
    private val lock = Object()

    /** Whether a stop since the line was queued cancelled it. Under [lock]. */
    private fun stale(stamp: Stamp): Boolean =
        stamp.generation != generation || stamp.cut != (cuts[stamp.owner] ?: 0)

    private fun staleNow(stamp: Stamp): Boolean = synchronized(lock) { stale(stamp) }

    /** A stop since the line was queued, or [dropWaiting] since it was taken. Under [lock]. */
    private fun gone(stamp: Stamp, id: Long): Boolean = stale(stamp) || id in dropped

    private fun goneNow(stamp: Stamp, id: Long): Boolean = synchronized(lock) { gone(stamp, id) }
    @Volatile private var playing = false

    /** Milliseconds of synthesis a millisecond of speech takes now, as the last piece took. */
    @Volatile private var pace = PACE_GUESS

    /**
     * Where the samples go: the phone's audio output ([TrackOutput], an AudioTrack kept open
     * and playing), made at the first line. A test hands in its own ([makeOutput]): one that
     * plays a piece when the test lets it, so that what happens while a piece plays -- the next
     * one made -- is seen, not timed.
     */
    private var output: Output? = null

    /** How the output is made: the phone's own unless a test says otherwise. */
    @Volatile internal var makeOutput: () -> Output = { TrackOutput.open() }

    /** Synthesis holds it shared; [close] alone, so the model is never freed under a line. */
    private val using = java.util.concurrent.locks.ReentrantReadWriteLock()

    val loaded: Boolean get() = kokoro != null

    /**
     * Whether the voice can speak, checked without loading the model: the two model files must
     * be present; the model itself is loaded once, by [load] or on the first line.
     */
    override fun healthcheck(): Pair<Boolean, String> {
        val missing = listOf(onnxPath, voicesPath).filter { !it.exists() }.map { it.name }
        if (missing.isNotEmpty()) return false to "model files missing from $modelDir: ${missing.joinToString(", ")}"
        val where = if (kokoro != null) "cpu" else "cpu (not loaded yet)"
        return true to "kokoro-onnx, voice $voice, $where, speaker: ${speakerName()}"
    }

    /** The output the voice plays through, by name: where Android routes the track now. */
    fun speakerName(): String {
        val routed = (output as? TrackOutput)?.track?.routedDevice
        if (routed != null) return routed.productName?.toString()?.ifEmpty { null } ?: "the phone's output"
        return "the phone's media output (follows a headset or a speaker)"
    }

    /** Load the graph once. Returns whether the voice is usable; safe from any thread. */
    fun load(): Boolean {
        synchronized(loadLock) {
            if (kokoro != null) return true
            if (failed && !filesArrived()) return false
            failed = false
            adoptFast()
            return try {
                val began = SystemClock.elapsedRealtime()
                kokoro = Kokoro.load(context, onnxPath, voicesPath, THREADS)
                loadSeconds = (SystemClock.elapsedRealtime() - began) / 1000.0
                true
            } catch (exc: Throwable) { // any load failure degrades to silence
                VoiceLog.warn("Kokoro voice unavailable, going silent: ${VoiceLog.said(exc)}")
                failed = true
                failedFiles = modelFiles()
                false
            }
        }
    }

    /**
     * The faster graph ([FAST_FILE]) put in the app's external files (at their top, where adb
     * may write: `tools\install.ps1 -Voice`) is moved into the model folder: copied, checked, and
     * only then the original removed, so nothing is lost if the move fails.
     */
    private fun adoptFast() {
        // At the folder's top: one adb's shell makes inside it is the shell's, closed to the app.
        val outside = File(context.getExternalFilesDir(null) ?: return, FAST_FILE)
        if (!outside.exists()) return
        val inside = File(modelDir, FAST_FILE)
        val partial = File(modelDir, "$FAST_FILE.part")
        runCatching {
            modelDir.mkdirs()
            outside.copyTo(partial, overwrite = true)
            if (partial.length() != outside.length()) error("copied ${partial.length()} of ${outside.length()} bytes")
            if (!partial.renameTo(inside)) error("could not rename into place")
            outside.delete()
            Log.i(TAG, "faster voice adopted: ${inside.length() / 1_000_000} MB")
        }.onFailure {
            partial.delete()
            VoiceLog.warn("the faster voice could not be adopted, the full one runs: ${VoiceLog.said(it)}")
        }
    }

    /** The model files as they are now (size and time), null for a missing one. */
    private fun modelFiles(): List<Pair<Long, Long>?> =
        listOf(onnxPath, voicesPath).map { if (it.exists()) it.length() to it.lastModified() else null }

    /**
     * A load that failed is tried again once the model files have changed since: the first
     * run's warm-up ran before Setup fetched them, and the voice stayed dead until a restart
     * (voice finding 4). A closed voice ([close]) never is.
     */
    private fun filesArrived(): Boolean {
        if (closed) return false
        val now = modelFiles()
        return null !in now && now != failedFiles
    }

    /**
     * One short line synthesised and thrown away, so the first spoken one does not pay the
     * runtime's warm-up. The seconds it took, null if the voice did not load.
     */
    fun prime(): Double? {
        if (!load()) return null
        val began = SystemClock.elapsedRealtime()
        val shared = using.readLock()
        shared.lock()
        try {
            runCatching { kokoro?.create("Ready.", voice, speed, lang) }
        } finally {
            shared.unlock()
        }
        return (SystemClock.elapsedRealtime() - began) / 1000.0
    }

    /**
     * Queue a line to be spoken. Notation in it is read aloud in words. [voice] speaks this line
     * in another voice (a preview) without changing the coach's.
     */
    override fun say(text: String, voice: String?, owner: String) {
        if (failed && !filesArrived()) return
        val spoken = Py.strip(Notation.speak(text, voice = true))
        if (spoken.isEmpty()) return
        ensureWorker()
        val stamp = synchronized(lock) { Stamp(owner, generation, cuts[owner] ?: 0) }
        synchronized(lock) {
            queue.put(Line(spoken, text, voice, stamp, id = lineIds.incrementAndGet()))
            lock.notifyAll() // the worker, waiting for a line ([run])
        }
    }

    /** Whether a line is playing or waiting to play. */
    override val busy: Boolean get() = playing || queue.isNotEmpty()

    /**
     * How many lines wait behind the one playing: the queued, and the one taken and made while
     * it plays. Before anything plays, the line being made is the one to play, not a waiting one.
     */
    override val waiting: Int
        get() = synchronized(lock) {
            val live = taken.keys.count { it !in dropped } // a dropped line never plays
            val ahead = live - (if (playingLine == null && live > 0) 1 else 0)
            queue.size + ahead
        }

    /**
     * Drop [owner]'s lines waiting behind the one playing -- the one made ahead too -- and leave
     * the one playing to finish: what waits is no longer the news (the pace plan). `drop_waiting`.
     */
    override fun dropWaiting(owner: String) {
        synchronized(lock) {
            val playingNow = playingLine?.first
            for ((id, who) in taken) if (who == owner && id != playingNow) dropped.add(id)
            lock.notifyAll()
        }
        queue.removeIf { it.stamp.owner == owner }
    }

    /**
     * Lines said often and word for word -- the short praise of a move ("Spot on.", "Exactly.") --
     * made once, in the background, and kept as long as the voice: said, they play at once, with
     * nothing to make (Tai, 2026-10-03: the voice "truly real time"). Kept per voice: another
     * voice makes them again when [keep] is called for it. The thread making them is returned.
     */
    fun keep(texts: List<String>): Thread {
        val forVoice = voice
        return Thread({
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            for (text in texts) {
                if (voice != forVoice) return@Thread // another voice since: its own keep runs
                val spoken = Py.strip(Notation.speak(text, voice = true))
                val key = "$forVoice|$spoken"
                if (spoken.isEmpty() || synchronized(lock) { key in kept }) continue
                val shared = using.readLock()
                shared.lock()
                try {
                    val model = kokoro ?: return@Thread
                    val (style, pieces) = model.pieces(spoken, forVoice, speed, lang)
                    val made = pieces.map { piece -> model.synthesize(piece, style, speed) }
                    synchronized(lock) { kept[key] = made }
                } catch (exc: Exception) { // a phrase not kept is made when said
                    Log.d(TAG, "could not keep ${Py.repr(spoken)}: ${VoiceLog.said(exc)}")
                } finally {
                    shared.unlock()
                }
            }
            Log.i(TAG, "kept ${texts.size} phrases for $forVoice")
        }, "kokoro-keep").apply {
            isDaemon = true
            start()
        }
    }

    /** [keep]'s phrases, by "voice|spoken text". Under [lock]. */
    private val kept = HashMap<String, List<FloatArray>>()

    /**
     * Make [text] now, in its pieces, off the voice's threads, to play at once when it is said (a
     * fast line's word, held to the line's end). Nothing when the voice is not loaded. `prepare`.
     */
    override fun prepare(text: String) {
        val spoken = Py.strip(Notation.speak(text, voice = true))
        if (spoken.isEmpty() || kokoro == null) return
        val done = CountDownLatch(1)
        synchronized(lock) {
            if (spoken in prepared || spoken in preparing) return
            preparing[spoken] = done
        }
        Thread({
            val shared = using.readLock()
            shared.lock()
            try {
                val model = kokoro ?: return@Thread
                val (style, pieces) = model.pieces(spoken, voice, speed, lang)
                val made = pieces.map { piece -> model.synthesize(piece, style, speed) }
                synchronized(lock) {
                    prepared[spoken] = made
                    while (prepared.size > 4) prepared.remove(prepared.keys.first())
                }
            } catch (exc: Exception) { // only an early start lost
                Log.d(TAG, "could not prepare ${Py.repr(spoken)}: ${VoiceLog.said(exc)}")
            } finally {
                shared.unlock()
                synchronized(lock) { preparing.remove(spoken) }
                done.countDown()
            }
        }, "kokoro-prepare").apply { isDaemon = true }.start()
    }

    /**
     * Drop everything queued and silence anything now playing. With [owner], only that owner's
     * lines, queued or playing: a room's stop (a review closed, the puzzles' voice off) left
     * the coach's lines alone only when the room had given the last line, and cut them off when
     * it had (round 3 audit, E5; Tai, 2026-10-02).
     */
    override fun stop(owner: String?) {
        val cutNow: Boolean
        synchronized(lock) {
            if (owner == null) generation += 1 else cuts[owner] = (cuts[owner] ?: 0) + 1
            // What is playing is cut when it is the owner's: the line playing, or before anything
            // plays the line being made. The next line is made while one plays, so the line being
            // made can be a room's while the coach's plays: that one plays on.
            val playingNow = playingLine
            val now = current
            cutNow = owner == null ||
                playingNow?.second == owner ||
                (playingNow == null && now != null && now.owner == owner)
            lock.notifyAll()
        }
        // The queue's own lock: the lines kept stay in their order.
        if (owner == null) queue.clear() else queue.removeIf { it.stamp.owner == owner }
        // The pieces handed to the player: everyone's, or only the owner's (a line of another
        // owner's, made ahead, plays on).
        if (owner == null) {
            while (true) chunks.poll()?.done?.countDown() ?: break
        } else {
            val gone = chunks.filter { it.mine.owner == owner }
            chunks.removeAll(gone.toSet())
            gone.forEach { it.done?.countDown() }
        }
        if (!cutNow) return
        runCatching { output?.cut() }
    }

    /**
     * Let go of everything: the lines queued, the worker, the output and the model (a rebuilt
     * voice replaces this one; its native memory is no collector's). Waits for a line being
     * synthesised; nothing speaks or loads afterwards.
     */
    fun close() {
        closed = true
        failed = true
        stop()
        synchronized(lock) {
            worker?.interrupt()
            player?.interrupt()
        }
        val alone = using.writeLock()
        alone.lock()
        try {
            val out = output
            output = null
            runCatching { out?.release() }
            synchronized(loadLock) {
                runCatching { kokoro?.close() }
                kokoro = null
            }
        } finally {
            alone.unlock()
        }
    }

    private fun ensureWorker() {
        synchronized(lock) {
            if (player?.isAlive != true) {
                player = Thread({ play() }, "kokoro-play").apply {
                    isDaemon = true
                    start()
                }
            }
            if (worker?.isAlive == true) return
            worker = Thread({ run() }, "kokoro-tts").apply {
                isDaemon = true
                start()
            }
        }
    }

    /**
     * The worker: each line made in turn, one line ahead of the player -- the next is made while
     * one plays, and a line that comes while one plays is taken the moment it comes -- so a
     * sentence follows the one before with no wait (Tai, 2026-10-03: "zero waits between
     * sentences"; the next was made only once the one before had played, 0.7 to 1.8 s of
     * silence on the Pixel). Never more than one ahead: before the next is taken, the line before
     * the one just made has played.
     */
    private fun run() {
        var before: CountDownLatch? = null // the line before the one just made: played?
        while (true) {
            // Busy from the moment a line is taken up, through synthesis, until it has been
            // handed to the output -- so a caller asking `busy` mid-synthesis gets the truth.
            // Marked busy before the line leaves the queue: taken first, a look between the two
            // saw neither, and a caller waiting for the voice to go quiet went on at once.
            val line = try {
                synchronized(lock) {
                    var next: Line? = null
                    while (next == null) {
                        while (queue.isEmpty()) lock.wait()
                        playing = true
                        next = queue.poll() // null: a stop emptied it meanwhile
                        if (next == null) playing = (before?.count ?: 0L) > 0L
                    }
                    next
                }
            } catch (_: InterruptedException) {
                return
            }
            synchronized(lock) {
                current = line.stamp
                taken[line.id] = line.stamp.owner
            }
            try {
                val done = speakOne(line)
                if (done == null) synchronized(lock) { taken.remove(line.id) }
                before?.await()
                before = done
                // Nothing behind it: busy until it has played, a line that comes meanwhile taken
                // at once.
                while (queue.isEmpty() && done != null && done.count > 0L) {
                    done.await(AHEAD_LOOK_MS, java.util.concurrent.TimeUnit.MILLISECONDS)
                }
            } catch (_: InterruptedException) {
                return
            } finally {
                synchronized(lock) { current = null }
                playing = queue.isNotEmpty() || (before?.count ?: 0L) > 0L
            }
        }
    }

    /**
     * Make [line] and hand its pieces to the player; its latch, counted down once its last piece
     * has been written to the output, or null when nothing of it was handed over.
     */
    private fun speakOne(line: Line): CountDownLatch? {
        if (!load()) return null // degraded to silence
        val model = kokoro ?: return null
        val mine = line.stamp
        if (goneNow(mine, line.id)) return null // stopped, or dropped, while it waited its turn
        val done = CountDownLatch(1)
        // Said while it is still being made ahead ([prepare]): waited for, never made twice.
        val making = synchronized(lock) { if (line.voice == null) preparing[line.spoken] else null }
        making?.await(PREPARE_WAIT_MS, java.util.concurrent.TimeUnit.MILLISECONDS)
        val ready = synchronized(lock) {
            if (line.voice == null) kept["$voice|${line.spoken}"] ?: prepared.remove(line.spoken) else null
        }
        if (ready != null) {
            // Made ahead ([prepare]): its pieces go to the player at once, to play when its turn
            // comes.
            for ((i, samples) in ready.withIndex()) {
                val last = i == ready.size - 1
                val out = if (last) samples.copyOf(samples.size + (Kokoro.SAMPLE_RATE * TAIL_SILENCE_S).toInt()) else samples
                val speechMs = if (i == 0) ready.sumOf { it.size } * 1000L / Kokoro.SAMPLE_RATE else 0L
                chunks.put(Chunk(out, mine, if (i == 0) line else null, if (last) done else null, speechMs, line.id))
            }
            synchronized(lock) {
                line.startAt = 0
                lock.notifyAll()
            }
            return done
        }
        var handed = false
        val shared = using.readLock()
        shared.lock()
        val began = SystemClock.elapsedRealtime()
        var speech = 0L
        val timings = ArrayList<String>()
        try {
            if (kokoro == null) return null // closed meanwhile
            val (style, pieces) = model.pieces(line.spoken, line.voice ?: voice, speed, lang)
            // The speech each piece will make, in milliseconds, from its phonemes.
            val expected = pieces.map { (phonemes, pause) ->
                phonemes.codePointCount(0, phonemes.length) * Kokoro.MS_PER_PHONEME / speed + pause * 1000
            }
            var made = 0.0
            for ((i, piece) in pieces.withIndex()) {
                // A stop, or a drop, meanwhile means this line was cancelled.
                if (goneNow(mine, line.id)) return null
                val from = SystemClock.elapsedRealtime()
                val samples = try {
                    runCatching { onSynthesis?.invoke(true) }
                    model.synthesize(piece, style, speed)
                } finally {
                    runCatching { onSynthesis?.invoke(false) }
                }
                val took = SystemClock.elapsedRealtime() - from
                val ms = samples.size * 1000.0 / Kokoro.SAMPLE_RATE
                if (ms > 0) pace = took / ms
                made += ms
                speech += ms.toLong()
                timings.add("$took/${ms.toLong()}")
                val last = i == pieces.size - 1
                // Playing may start once every piece still to make will be ready before the
                // speech ahead of it runs out: the next one's synthesis against what is made,
                // the one after against that and the next one's speech, and so on.
                var hold = 0.0
                var making = 0.0
                var ahead = made
                for (j in i + 1 until pieces.size) {
                    making += pace * expected[j] * MARGIN + PASS_MS
                    hold = maxOf(hold, making - ahead)
                    ahead += expected[j]
                }
                val out = if (last) samples.copyOf(samples.size + (Kokoro.SAMPLE_RATE * TAIL_SILENCE_S).toInt()) else samples
                val speechMs = if (i == 0) (made + expected.drop(1).sum()).toLong() else 0L
                chunks.put(Chunk(out, mine, if (i == 0) line else null, if (last) done else null, speechMs, line.id))
                handed = last
                synchronized(lock) {
                    line.startAt = SystemClock.elapsedRealtime() + hold.toLong()
                    lock.notifyAll()
                }
            }
        } catch (exc: Exception) { // a bad line must not kill the voice
            VoiceLog.warn("Kokoro synthesis failed for ${Py.repr(line.spoken)}: ${VoiceLog.said(exc)}")
            return null
        } finally {
            shared.unlock()
            if (!handed) synchronized(lock) {
                // What was made of a line that failed part-way plays now; a stopped line's
                // pieces the player drops.
                line.startAt = 0
                lock.notifyAll()
            }
        }
        // Where a line's time goes, for the logcat: its wait in the queue, its synthesis, and
        // how long it speaks (a synthesis slower than the speech is the voice falling behind).
        Log.i(
            TAG,
            "line: waited ${began - line.queuedAt} ms, synthesised in ${SystemClock.elapsedRealtime() - began} ms, " +
                "$speech ms of speech, in ${timings.size} piece${if (timings.size == 1) "" else "s"} " +
                "(ms made in / ms of speech: ${timings.joinToString(" ")})",
        )
        // The worker goes on to the next line while this one plays ([run]).
        return if (handed) done else null
    }

    /**
     * The player: each piece written to the output in order, the first of a line once its
     * [startAt] has come, anything of a stopped line dropped.
     */
    private fun play() {
        // The output's underruns when a line began: any more by its end were gaps in it.
        var underruns = 0
        while (true) {
            val chunk = try {
                chunks.take()
            } catch (_: InterruptedException) {
                return
            }
            try {
                val line = chunk.first
                if (line != null) {
                    synchronized(lock) {
                        while (!gone(chunk.mine, chunk.lineId)) {
                            val wait = line.startAt - SystemClock.elapsedRealtime()
                            if (wait <= 0) break
                            lock.wait(wait)
                        }
                    }
                }
                // The track opened before the cancel check, and the piece written in slices, each
                // checked: a stop landing while the track opened, or during a piece, was ignored
                // and the line played over a mute or a barge-in (voice finding 13).
                val out = openOutput()
                if (goneNow(chunk.mine, chunk.lineId)) continue
                if (line != null) {
                    synchronized(lock) {
                        taken.remove(chunk.lineId)
                        playingLine = chunk.lineId to chunk.mine.owner
                    }
                    underruns = out.underrunCount
                    Log.i(TAG, "line: first sound ${SystemClock.elapsedRealtime() - line.queuedAt} ms after it was queued")
                    runCatching { onPlay?.invoke(line.given.ifEmpty { line.spoken }) }
                    runCatching { onCaption?.invoke(line.given.ifEmpty { line.spoken }, chunk.speechMs) }
                }
                writePiece(
                    chunk.samples,
                    output = { PcmSink(out::write) },
                    reopen = { PcmSink(reopenOutput(out)::write) },
                    stopped = { goneNow(chunk.mine, chunk.lineId) },
                )
                if (chunk.done != null) Log.i(TAG, "line: played with ${(output ?: out).underrunCount - underruns} gaps")
            } catch (_: InterruptedException) {
                return
            } catch (exc: Exception) { // audio errors are not fatal
                VoiceLog.warn("Kokoro playback failed: ${VoiceLog.said(exc)}")
            } finally {
                if (chunk.done != null) {
                    synchronized(lock) {
                        if (playingLine?.first == chunk.lineId) playingLine = null
                        taken.remove(chunk.lineId)
                        dropped.remove(chunk.lineId)
                    }
                    chunk.done.countDown()
                }
            }
        }
    }

    /**
     * The output [dead] dropped and a new one opened, on wherever Android routes the voice now.
     * Android moves a playing track to a new output itself (headphones plugged in, a Bluetooth
     * headset connected: the track asks for no device, so it follows the system's routing); a
     * track it could not move, or one the audio server lost, answered every write with an error
     * and the voice stayed silent until a restart (Tai, 2026-10-02: the voice follows the output
     * without a restart).
     */
    @Synchronized
    private fun reopenOutput(dead: Output): Output {
        if (output === dead) {
            output = null
            runCatching { dead.release() }
            VoiceLog.warn("the voice's output track died; opened afresh on the current output")
        }
        return openOutput()
    }

    /** The open output, opened at the first line and kept playing. */
    @Synchronized
    private fun openOutput(): Output {
        output?.let { return it }
        return makeOutput().also { output = it }
    }

    /** Where a line's samples are written, a slice at a time ([play]). */
    internal interface Output {
        /** The gaps the output has played so far (each a moment it had nothing to play). */
        val underrunCount: Int

        /** [n] samples from [at], returned once they are taken: how many were. */
        fun write(samples: FloatArray, at: Int, n: Int): Int

        /** Whatever is queued to play, silenced now; the output stays open. */
        fun cut()

        fun release()
    }

    /** The phone's audio output: an AudioTrack, opened once and kept playing. */
    internal class TrackOutput(val track: AudioTrack) : Output {
        override val underrunCount: Int get() = track.underrunCount

        override fun write(samples: FloatArray, at: Int, n: Int): Int =
            track.write(samples, at, n, AudioTrack.WRITE_BLOCKING)

        override fun cut() {
            track.pause()
            track.flush()
            track.play()
        }

        override fun release() = track.release()

        companion object {
            fun open(): TrackOutput {
                val minimum = AudioTrack.getMinBufferSize(
                    Kokoro.SAMPLE_RATE,
                    AudioFormat.CHANNEL_OUT_MONO,
                    AudioFormat.ENCODING_PCM_FLOAT,
                )
                val made = AudioTrack.Builder()
                    .setAudioAttributes(
                        // Media, so the voice follows the one media volume, as music and videos
                        // do. As USAGE_ASSISTANT it had a volume of its own, which the volume keys
                        // change only while it speaks: Tai's Pixel had it at 1 of 15 on the
                        // speaker (-43 dB) and the coach was barely heard (2026-10-02).
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build(),
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(Kokoro.SAMPLE_RATE)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                            .build(),
                    )
                    .setBufferSizeInBytes(maxOf(minimum, Kokoro.SAMPLE_RATE / 5 * 4))
                    .setTransferMode(AudioTrack.MODE_STREAM)
                    .setSessionId(AudioManager.AUDIO_SESSION_ID_GENERATE)
                    .build()
                made.play()
                return TrackOutput(made)
            }
        }
    }

    companion object {
        private const val TAG = "sidecoach.voice"

        /** Samples written at a time, a stop checked between: 50 ms at 24 kHz. */
        internal const val WRITE_SLICE = 1200

        /**
         * A piece written to the output in slices of [WRITE_SLICE], [stopped] checked before each:
         * an output that answers a write with [AudioTrack.ERROR_DEAD_OBJECT] (Android could not
         * move the track to a new route, or the audio server restarted) is opened afresh once
         * ([reopen]) and the rest of the piece goes there, mid-line. The samples written.
         */
        internal fun writePiece(samples: FloatArray, output: () -> PcmSink, reopen: () -> PcmSink, stopped: () -> Boolean): Int {
            var out = output()
            var reopened = false
            var at = 0
            while (at < samples.size) {
                if (stopped()) break
                val n = minOf(WRITE_SLICE, samples.size - at)
                val wrote = out.write(samples, at, n)
                if (wrote == AudioTrack.ERROR_DEAD_OBJECT && !reopened) {
                    reopened = true
                    out = reopen()
                    continue
                }
                if (wrote <= 0) break
                at += wrote
            }
            return at
        }
        /**
         * The full-precision model, as the desktop runs it (Tai, 2026-09-23): on the Pixel's ARM
         * cores it speaks twice as fast as the int8 one (0.61 of real time against 1.20), whose
         * quantised convolutions onnxruntime does not speed up there, and its audio is the
         * desktop's to the sample.
         */
        const val ONNX_FILE = "kokoro-v1.0.onnx"

        /**
         * Kokoro with the decoder's convolutions in 8 bits but its last stage's (in fp16: in 8 bits
         * it buzzed, near 11 kHz), and its instance norms fused: 0.31 of real time on the Pixel
         * where the full-precision graph runs 0.70, its speech as close to fp32 as the graph Tai
         * could not tell from fp32 in a blind A/B (2026-10-03). Made by `tools/kokoro_fast.py` in
         * the desktop repo.
         */
        const val FAST_FILE = "kokoro-v1.0-q8.onnx"
        const val VOICES_FILE = "voices-v1.0.bin"

        /**
         * Silence appended to every line: a breath between one sentence and the next, and what
         * a Bluetooth headset a few hundred milliseconds behind clips instead of the words
         * (Tai, 2026-09-19).
         */
        const val TAIL_SILENCE_S = 0.35

        /**
         * The graph's threads: the phone's big cores, not the two small ones. Five speak at 0.49
         * of real time on the Pixel 10 Pro XL, four at 0.56 to 0.61; the engine is paused while
         * they work.
         */
        const val THREADS = 5

        /** [pace] before any piece has been timed: the Pixel's, with room to spare. */
        private const val PACE_GUESS = 0.6

        /**
         * How much longer than its estimate the next piece is allowed to take: the speech a
         * piece makes is guessed from its phonemes to within 6% in half the coach's lines.
         */
        private const val MARGIN = 1.15

        /** The time a pass takes whatever its length, in milliseconds. */
        private const val PASS_MS = 80.0

        /**
         * While the last line plays with nothing behind it, how often the worker looks for a line
         * that has come: one that comes is taken and made at once, while the line plays.
         */
        private const val AHEAD_LOOK_MS = 20L

        /** A line said while it is still being made ahead waits this long at most for that
         * making, rather than being made a second time (`PREPARE_WAIT_S`). */
        private const val PREPARE_WAIT_MS = 5000L
    }
}
