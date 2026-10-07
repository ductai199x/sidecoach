package com.sidecoach.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Voice activity: when the user starts talking (barge-in) and when they have finished.
 * `sidecoach/voice/vad.py`.
 *
 * Silero VAD, always on the CPU -- a 2 MB model that costs under a millisecond per 32 ms frame.
 * [SileroGate] turns 16 kHz audio into a speech probability per frame, carrying the model's state
 * across frames. [Endpointer] is the state machine on top: speech has started after a couple of
 * confident frames (cut the coach off), and an utterance has ended after enough trailing silence.
 */
const val SAMPLE_RATE = 16000

/** Samples per frame: 32 ms at 16 kHz, the Silero v5 window. */
const val HOP = 512

/** Samples of the previous frame the model wants in front of each window. */
const val CONTEXT = 64

/** Speech probability per 32 ms frame, with the model's recurrent state carried along. */
class SileroGate(private val model: File) : VoiceGate {
    private var session: OrtSession? = null
    private var state = FloatArray(2 * 1 * 128)
    private var context = FloatArray(CONTEXT)
    private var failed: String? = null
    private val lock = Any()
    private val env = OrtEnvironment.getEnvironment()

    override fun healthcheck(): Pair<Boolean, String> {
        failed?.let { return false to "silero vad failed to load: $it" }
        if (!model.exists()) return false to "silero vad not downloaded yet (Setup)"
        return true to "silero vad, cpu"
    }

    override fun load(): Boolean {
        synchronized(lock) {
            if (session != null) return true
            if (failed != null) return false
            return try {
                val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(1) }
                session = env.createSession(model.path, options)
                true
            } catch (exc: Exception) { // no VAD means push-to-talk only
                failed = "${exc::class.java.simpleName}: ${exc.message}"
                VoiceLog.warn("Silero VAD unavailable: ${VoiceLog.said(exc)}")
                false
            }
        }
    }

    override fun reset() {
        state = FloatArray(2 * 1 * 128)
        context = FloatArray(CONTEXT)
    }

    /** Speech probability for one [HOP]-sample frame (float32, 16 kHz). */
    override fun probability(input: FloatArray): Float {
        if (!load()) return 0f
        val model = session ?: return 0f
        val frame = when {
            input.size > HOP -> input.copyOf(HOP)
            input.size < HOP -> input.copyOf(HOP) // zero-padded, as np.pad does
            else -> input
        }
        val window = FloatArray(CONTEXT + HOP)
        context.copyInto(window, 0)
        frame.copyInto(window, CONTEXT)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(window), longArrayOf(1, window.size.toLong())).use { windowTensor ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(state), longArrayOf(2, 1, 128)).use { stateTensor ->
                OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(SAMPLE_RATE.toLong())), longArrayOf(1)).use { srTensor ->
                    model.run(mapOf("input" to windowTensor, "state" to stateTensor, "sr" to srTensor)).use { result ->
                        val out = (result.get("output").get() as OnnxTensor).floatBuffer.get(0)
                        val next = (result.get("stateN").get() as OnnxTensor).floatBuffer
                        state = FloatArray(next.remaining()).also { next.get(it) }
                        context = frame.copyOfRange(HOP - CONTEXT, HOP)
                        return out
                    }
                }
            }
        }
    }

    fun close() {
        synchronized(lock) {
            runCatching { session?.close() }
            session = null
        }
    }

    companion object {
        const val FILE = "silero_vad.onnx"
    }
}

/**
 * Start and end of an utterance from a stream of per-frame speech probabilities.
 *
 * Speech starts when [startFrames] consecutive frames exceed [start] (two frames = 64 ms: quick
 * enough for a barge-in, long enough to ignore a click). It ends after [endMs] of frames below
 * [end]. While the coach is speaking, the mic hears the speakers, so the start threshold is
 * raised to [startWhileSpeaking] -- a headset makes this moot. (The listener no longer feeds it the
 * coach's voice: since 2026-10-02 it ignores the microphone while the voice plays, [Listener].)
 */
class Endpointer(
    val start: Double = 0.5,
    val end: Double = 0.35,
    val startFrames: Int = 2,
    val endMs: Int = 400,
    val startWhileSpeaking: Double = 0.85,
    val minUtteranceMs: Int = 250,
    val frameMs: Int = 32,
) {
    /** Inside an utterance. */
    var speaking = false
        private set
    private var run = 0
    private var silenceMs = 0
    private var lengthMs = 0

    /** Advance one frame: "start", "end", "drop" (too short to be an ask), or null. */
    fun feed(prob: Double, coachSpeaking: Boolean = false): String? {
        if (!speaking) {
            val threshold = if (coachSpeaking) startWhileSpeaking else start
            run = if (prob >= threshold) run + 1 else 0
            if (run >= startFrames) {
                speaking = true
                silenceMs = 0
                lengthMs = run * frameMs
                run = 0
                return "start"
            }
            return null
        }
        lengthMs += frameMs
        if (prob < end) {
            silenceMs += frameMs
            if (silenceMs >= endMs) {
                speaking = false
                val spoken = lengthMs - silenceMs
                lengthMs = 0
                silenceMs = 0
                return if (spoken >= minUtteranceMs) "end" else "drop"
            }
        } else {
            silenceMs = 0
        }
        return null
    }

    fun reset() {
        speaking = false
        run = 0
        silenceMs = 0
        lengthMs = 0
    }
}
