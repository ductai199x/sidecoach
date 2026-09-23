package com.sidecoach.voice

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.util.concurrent.LinkedBlockingQueue

/**
 * The coach's lines spoken with Kokoro. `sidecoach/voice/kokoro.py`'s `KokoroTTS`: non-blocking,
 * and it never raises out of [say] or [stop].
 *
 * The graph is loaded once (by [load], or on the first line) and lives as long as the provider;
 * a voice is only a style chosen at synthesis time, so changing it reloads nothing. [say]
 * queues a line and returns at once; one worker thread synthesises and plays in order.
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
) {
    val name = "kokoro"

    private val onnxPath: File get() = File(modelDir, ONNX_FILE)
    val voicesPath: File get() = File(modelDir, VOICES_FILE)

    @Volatile private var kokoro: Kokoro? = null
    @Volatile private var failed = false
    private val loadLock = Any()
    @Volatile var loadSeconds: Double? = null
        private set

    /**
     * Called with the line as given to [say] the moment its audio starts to play, for whoever
     * measures the way from a move to the first sound (the panel).
     */
    @Volatile var onPlay: ((String) -> Unit)? = null

    private data class Line(
        val spoken: String,
        val given: String,
        val voice: String?,
        val queuedAt: Long = SystemClock.elapsedRealtime(),
    )

    private val queue = LinkedBlockingQueue<Line>()
    private var worker: Thread? = null

    /** Bumped on every stop, so a synthesis in flight knows its line was cancelled. */
    private var generation = 0
    private val lock = Any()
    @Volatile private var playing = false

    private var track: AudioTrack? = null

    /** Synthesis holds it shared; [close] alone, so the model is never freed under a line. */
    private val using = java.util.concurrent.locks.ReentrantReadWriteLock()

    val loaded: Boolean get() = kokoro != null

    /**
     * Whether the voice can speak, checked without loading the model: the two model files must
     * be present; the model itself is loaded once, by [load] or on the first line.
     */
    fun healthcheck(): Pair<Boolean, String> {
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
            if (failed) return false
            return try {
                val began = SystemClock.elapsedRealtime()
                kokoro = Kokoro.load(context, onnxPath, voicesPath, THREADS)
                loadSeconds = (SystemClock.elapsedRealtime() - began) / 1000.0
                true
            } catch (exc: Throwable) { // any load failure degrades to silence
                Log.w(TAG, "Kokoro voice unavailable, going silent: $exc")
                failed = true
                false
            }
        }
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
    fun say(text: String, voice: String? = null) {
        if (failed) return
        val spoken = Py.strip(Notation.speak(text))
        if (spoken.isEmpty()) return
        ensureWorker()
        queue.put(Line(spoken, text, voice))
    }

    /** Whether a line is playing or waiting to play. */
    val busy: Boolean get() = playing || queue.isNotEmpty()

    /** Drop everything queued and silence anything now playing. */
    fun stop() {
        synchronized(lock) { generation += 1 }
        queue.clear()
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
        failed = true
        stop()
        synchronized(lock) { worker?.interrupt() }
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
        val shared = using.readLock()
        shared.lock()
        val began = SystemClock.elapsedRealtime()
        val samples = try {
            if (kokoro == null) return // closed meanwhile
            model.create(line.spoken, line.voice ?: voice, speed, lang)
        } catch (exc: Exception) { // a bad line must not kill the voice
            Log.w(TAG, "Kokoro synthesis failed for \"${line.spoken}\": $exc")
            return
        } finally {
            shared.unlock()
        }
        val took = SystemClock.elapsedRealtime() - began
        // Where a line's time goes, for the logcat: its wait in the queue, its synthesis, and
        // how long it speaks (a synthesis slower than the speech is the voice falling behind).
        Log.i(
            TAG,
            "line: waited ${began - line.queuedAt} ms, synthesised in $took ms, " +
                "${samples.size * 1000L / Kokoro.SAMPLE_RATE} ms of speech",
        )
        // A stop between synthesis and playback means this line was cancelled.
        if (synchronized(lock) { generation } != mine) return
        runCatching { onPlay?.invoke(line.given.ifEmpty { line.spoken }) }
        try {
            val padded = samples.copyOf(samples.size + (Kokoro.SAMPLE_RATE * TAIL_SILENCE_S).toInt())
            val out = openTrack()
            out.write(padded, 0, padded.size, AudioTrack.WRITE_BLOCKING)
        } catch (exc: Exception) { // audio errors are not fatal
            Log.w(TAG, "Kokoro playback failed: $exc")
        }
    }

    /** The open output track, opened at the first line and kept playing. */
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

        /** The graph's threads: the phone's big cores, not all eight. */
        const val THREADS = 4
    }
}
