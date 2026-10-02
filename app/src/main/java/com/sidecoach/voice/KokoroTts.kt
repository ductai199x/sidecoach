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
 */
class KokoroTts(
    private val context: Context,
    private val modelDir: File,
    /** The coach's voice: a style, not a model -- changing it reloads nothing. */
    @Volatile var voice: String = "af_sarah",
    @Volatile var speed: Float = 1.0f,
    private val lang: String = "en-us",
) : TtsProvider {
    override val name = "kokoro"

    private val onnxPath: File get() = File(modelDir, ONNX_FILE)
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

    private data class Line(
        val spoken: String,
        val given: String,
        val voice: String?,
        val queuedAt: Long = SystemClock.elapsedRealtime(),
    )

    private val queue = LinkedBlockingQueue<Line>()
    private var worker: Thread? = null

    /**
     * A piece of a line's audio on its way to the output: the line with its first piece (whose
     * playing waits for [startAt]), and with its last the latch the worker waits on.
     */
    private class Chunk(
        val samples: FloatArray,
        val mine: Int,
        val first: Line?,
        val done: CountDownLatch?,
        /** With the first piece: the line's speech, what is made and the rest's estimate. */
        val speechMs: Long = 0,
    )

    private val chunks = LinkedBlockingQueue<Chunk>()
    private var player: Thread? = null

    /** Bumped on every stop, so a synthesis in flight knows its line was cancelled. */
    private var generation = 0
    private val lock = Object()
    @Volatile private var playing = false

    /** When the line being made may start to play, in elapsed-realtime milliseconds. Under [lock]. */
    private var startAt = Long.MAX_VALUE

    /** Milliseconds of synthesis a millisecond of speech takes now, as the last piece took. */
    @Volatile private var pace = PACE_GUESS

    private var track: AudioTrack? = null

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
        val routed = track?.routedDevice
        if (routed != null) return routed.productName?.toString()?.ifEmpty { null } ?: "the phone's output"
        return "the phone's media output (follows a headset or a speaker)"
    }

    /** Load the graph once. Returns whether the voice is usable; safe from any thread. */
    fun load(): Boolean {
        synchronized(loadLock) {
            if (kokoro != null) return true
            if (failed && !filesArrived()) return false
            failed = false
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
    override fun say(text: String, voice: String?) {
        if (failed && !filesArrived()) return
        val spoken = Py.strip(Notation.speak(text, voice = true))
        if (spoken.isEmpty()) return
        ensureWorker()
        queue.put(Line(spoken, text, voice))
    }

    /** Whether a line is playing or waiting to play. */
    override val busy: Boolean get() = playing || queue.isNotEmpty()

    /** Drop everything queued and silence anything now playing. */
    override fun stop() {
        synchronized(lock) {
            generation += 1
            lock.notifyAll()
        }
        queue.clear()
        while (true) chunks.poll()?.done?.countDown() ?: break
        runCatching {
            track?.let {
                it.pause()
                it.flush()
                it.play()
            }
        }
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
            val t = track
            track = null
            runCatching { t?.release() }
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

    private fun run() {
        while (true) {
            val line = try {
                queue.take()
            } catch (_: InterruptedException) {
                return
            }
            // Busy from the moment a line is taken up, through synthesis, until it has been
            // handed to the output -- so a caller asking `busy` mid-synthesis gets the truth.
            playing = true
            try {
                speakOne(line)
            } finally {
                playing = false
            }
        }
    }

    private fun speakOne(line: Line) {
        if (!load()) return // degraded to silence
        val model = kokoro ?: return
        val mine = synchronized(lock) { generation }
        val done = CountDownLatch(1)
        var handed = false
        val shared = using.readLock()
        shared.lock()
        val began = SystemClock.elapsedRealtime()
        var speech = 0L
        val timings = ArrayList<String>()
        try {
            if (kokoro == null) return // closed meanwhile
            val (style, pieces) = model.pieces(line.spoken, line.voice ?: voice, speed, lang)
            // The speech each piece will make, in milliseconds, from its phonemes.
            val expected = pieces.map { (phonemes, pause) ->
                phonemes.codePointCount(0, phonemes.length) * Kokoro.MS_PER_PHONEME / speed + pause * 1000
            }
            synchronized(lock) { startAt = Long.MAX_VALUE }
            var made = 0.0
            for ((i, piece) in pieces.withIndex()) {
                // A stop meanwhile means this line was cancelled.
                if (synchronized(lock) { generation } != mine) return
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
                chunks.put(Chunk(out, mine, if (i == 0) line else null, if (last) done else null, speechMs))
                handed = last
                synchronized(lock) {
                    startAt = SystemClock.elapsedRealtime() + hold.toLong()
                    lock.notifyAll()
                }
            }
        } catch (exc: Exception) { // a bad line must not kill the voice
            VoiceLog.warn("Kokoro synthesis failed for ${Py.repr(line.spoken)}: ${VoiceLog.said(exc)}")
            return
        } finally {
            shared.unlock()
            if (!handed) synchronized(lock) {
                // What was made of a line that failed part-way plays now; a stopped line's
                // pieces the player drops.
                startAt = 0
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
        // The next line waits until this one has been handed to the output, as on the desktop.
        try {
            done.await()
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
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
                        while (generation == chunk.mine) {
                            val wait = startAt - SystemClock.elapsedRealtime()
                            if (wait <= 0) break
                            lock.wait(wait)
                        }
                    }
                }
                // The track opened before the cancel check, and the piece written in slices, each
                // checked: a stop landing while the track opened, or during a piece, was ignored
                // and the line played over a mute or a barge-in (voice finding 13).
                val out = openTrack()
                if (synchronized(lock) { generation } != chunk.mine) continue
                if (line != null) {
                    underruns = out.underrunCount
                    Log.i(TAG, "line: first sound ${SystemClock.elapsedRealtime() - line.queuedAt} ms after it was queued")
                    runCatching { onPlay?.invoke(line.given.ifEmpty { line.spoken }) }
                    runCatching { onCaption?.invoke(line.given.ifEmpty { line.spoken }, chunk.speechMs) }
                }
                var at = 0
                while (at < chunk.samples.size) {
                    if (synchronized(lock) { generation } != chunk.mine) break
                    val n = minOf(WRITE_SLICE, chunk.samples.size - at)
                    val wrote = out.write(chunk.samples, at, n, AudioTrack.WRITE_BLOCKING)
                    if (wrote <= 0) break
                    at += wrote
                }
                if (chunk.done != null) Log.i(TAG, "line: played with ${out.underrunCount - underruns} gaps")
            } catch (_: InterruptedException) {
                return
            } catch (exc: Exception) { // audio errors are not fatal
                VoiceLog.warn("Kokoro playback failed: ${VoiceLog.said(exc)}")
            } finally {
                chunk.done?.countDown()
            }
        }
    }

    /** The open output track, opened at the first line and kept playing. */
    @Synchronized
    private fun openTrack(): AudioTrack {
        track?.let { return it }
        val minimum = AudioTrack.getMinBufferSize(
            Kokoro.SAMPLE_RATE,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        val made = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_ASSISTANT)
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
        track = made
        return made
    }

    companion object {
        private const val TAG = "sidecoach.voice"

        /** Samples written at a time, a stop checked between: 50 ms at 24 kHz. */
        private const val WRITE_SLICE = 1200
        /**
         * The full-precision model, as the desktop runs it (Tai, 2026-09-23): on the Pixel's ARM
         * cores it speaks twice as fast as the int8 one (0.61 of real time against 1.20), whose
         * quantised convolutions onnxruntime does not speed up there, and its audio is the
         * desktop's to the sample.
         */
        const val ONNX_FILE = "kokoro-v1.0.onnx"
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
    }
}
