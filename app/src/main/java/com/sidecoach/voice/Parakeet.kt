package com.sidecoach.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer

/**
 * NVIDIA Parakeet-TDT-0.6B v3 as onnx-asr 0.12 runs it for the desktop (`NemoConformerTdt`):
 * the waveform to 128 mel features (onnx-asr's own `nemo128.onnx`), the encoder, then the TDT
 * decoding -- the joint network asked frame by frame for a token and a duration, the frames it
 * says to skip skipped -- and the tokens joined into text. int8 on the phone, where the desktop
 * runs the full-precision graphs (Tai, 2026-09-23).
 */
class Parakeet private constructor(
    private val env: OrtEnvironment,
    private val preprocessor: OrtSession,
    private val encoder: OrtSession,
    private val decoderJoint: OrtSession,
    private val vocab: Map<Int, String>,
    private val blank: Int,
    private val maxTokensPerStep: Int,
) : AutoCloseable {

    private val vocabSize = vocab.size

    /**
     * `TextResultsAsrAdapter.recognize(waveform, sample_rate=16000)` for one float32 mono clip at
     * 16 kHz: the text, as `_decode_tokens` joins it.
     */
    fun recognize(waveform: FloatArray): String = recognizeTokens(waveform).first

    /** The text, and the token pieces it was joined from (`TimestampedResult.tokens`). */
    fun recognizeTokens(waveform: FloatArray): Pair<String, List<String>> {
        val (features, featuresLens) = preprocess(waveform)
        val (encoded, length, dim) = encode(features, featuresLens)
        val tokens = decode(encoded, length, dim)
        return decodeTokens(tokens) to tokens.map { vocab.getValue(it) }
    }

    /** `OnnxPreprocessor.__call__`: the waveform to features and their length. */
    private fun preprocess(waveform: FloatArray): Pair<OnnxTensor, OnnxTensor> {
        OnnxTensor.createTensor(env, FloatBuffer.wrap(waveform), longArrayOf(1, waveform.size.toLong())).use { waves ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(longArrayOf(waveform.size.toLong())), longArrayOf(1)).use { lens ->
                preprocessor.run(mapOf("waveforms" to waves, "waveforms_lens" to lens)).use { result ->
                    val features = result.get("features").get() as OnnxTensor
                    val featuresLens = result.get("features_lens").get() as OnnxTensor
                    return copy(features) to copy(featuresLens)
                }
            }
        }
    }

    private fun copy(tensor: OnnxTensor): OnnxTensor {
        val shape = tensor.info.shape
        return when (tensor.info.type) {
            ai.onnxruntime.OnnxJavaType.FLOAT -> {
                val buffer = tensor.floatBuffer
                OnnxTensor.createTensor(env, FloatBuffer.wrap(FloatArray(buffer.remaining()).also { buffer.get(it) }), shape)
            }
            else -> {
                val buffer = tensor.longBuffer
                OnnxTensor.createTensor(env, LongBuffer.wrap(LongArray(buffer.remaining()).also { buffer.get(it) }), shape)
            }
        }
    }

    /**
     * `NemoConformerRnnt._encode`: the features through the encoder, transposed to time-major --
     * the encodings as [T][D] flattened, the length that counts, and D.
     */
    private fun encode(features: OnnxTensor, lens: OnnxTensor): Triple<FloatArray, Int, Int> {
        features.use { f ->
            lens.use { l ->
                encoder.run(mapOf("audio_signal" to f, "length" to l)).use { result ->
                    val out = result.get("outputs").get() as OnnxTensor
                    val encodedLengths = result.get("encoded_lengths").get() as OnnxTensor
                    val shape = out.info.shape // [1, D, T]
                    val dim = shape[1].toInt()
                    val frames = shape[2].toInt()
                    val raw = FloatArray(dim * frames).also { out.floatBuffer.get(it) }
                    val transposed = FloatArray(frames * dim)
                    for (d in 0 until dim) {
                        val row = d * frames
                        for (t in 0 until frames) transposed[t * dim + d] = raw[row + t]
                    }
                    val length = encodedLengths.longBuffer.get(0).toInt()
                    return Triple(transposed, minOf(length, frames), dim)
                }
            }
        }
    }

    /**
     * `_AsrWithTransducerDecoding._decoding` with the TDT step (`NemoConformerTdt._decode`): at
     * each frame the joint network gives the next token's logits and a duration; a token that is
     * not blank is kept and the state moves on; the duration, when positive, says how many frames
     * to jump; otherwise a blank, or [maxTokensPerStep] tokens in one frame, moves one frame on.
     */
    private fun decode(encodings: FloatArray, length: Int, dim: Int): List<Int> {
        val stateShape1 = longArrayOf(2, 1, STATE_DIM.toLong())
        var state1 = FloatArray(2 * STATE_DIM)
        var state2 = FloatArray(2 * STATE_DIM)
        val tokens = ArrayList<Int>()
        var t = 0
        var emitted = 0
        val frame = FloatArray(dim)
        while (t < length) {
            System.arraycopy(encodings, t * dim, frame, 0, dim)
            val previous = if (tokens.isEmpty()) blank else tokens.last()
            val (output, next1, next2) = joint(frame, dim, previous, state1, state2, stateShape1)
            var token = 0
            var bestLogit = Float.NEGATIVE_INFINITY
            for (k in 0 until vocabSize) {
                if (output[k] > bestLogit) {
                    bestLogit = output[k]
                    token = k
                }
            }
            var step = 0
            var bestStep = Float.NEGATIVE_INFINITY
            for (k in vocabSize until output.size) {
                if (output[k] > bestStep) {
                    bestStep = output[k]
                    step = k - vocabSize
                }
            }
            if (token != blank) {
                state1 = next1
                state2 = next2
                tokens.add(token)
                emitted += 1
            }
            if (step > 0) {
                t += step
                emitted = 0
            } else if (token == blank || emitted == maxTokensPerStep) {
                t += 1
                emitted = 0
            }
        }
        return tokens
    }

    /** One call of the joint network: the output vector, and the two states it leaves. */
    private fun joint(
        frame: FloatArray,
        dim: Int,
        previous: Int,
        state1: FloatArray,
        state2: FloatArray,
        stateShape: LongArray,
    ): Triple<FloatArray, FloatArray, FloatArray> {
        val inputs = HashMap<String, OnnxTensor>()
        try {
            inputs["encoder_outputs"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(frame), longArrayOf(1, dim.toLong(), 1))
            inputs["targets"] = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(previous)), longArrayOf(1, 1))
            inputs["target_length"] = OnnxTensor.createTensor(env, IntBuffer.wrap(intArrayOf(1)), longArrayOf(1))
            inputs["input_states_1"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(state1), stateShape)
            inputs["input_states_2"] = OnnxTensor.createTensor(env, FloatBuffer.wrap(state2), stateShape)
            decoderJoint.run(inputs, setOf("outputs", "output_states_1", "output_states_2")).use { result ->
                val outputs = (result.get("outputs").get() as OnnxTensor).floatBuffer
                val s1 = (result.get("output_states_1").get() as OnnxTensor).floatBuffer
                val s2 = (result.get("output_states_2").get() as OnnxTensor).floatBuffer
                return Triple(
                    FloatArray(outputs.remaining()).also { outputs.get(it) },
                    FloatArray(s1.remaining()).also { s1.get(it) },
                    FloatArray(s2.remaining()).also { s2.get(it) },
                )
            }
        } finally {
            for (tensor in inputs.values) tensor.close()
        }
    }

    /** `_AsrWithDecoding._decode_tokens`: the pieces joined, the word-piece spaces tidied. */
    private fun decodeTokens(ids: List<Int>): String {
        val joined = ids.joinToString("") { vocab.getValue(it) }
        return DECODE_SPACE_PATTERN.replace(joined) { m -> if (m.groups[1] != null) " " else "" }
    }

    override fun close() {
        runCatching { preprocessor.close() }
        runCatching { encoder.close() }
        runCatching { decoderJoint.close() }
    }

    companion object {
        private const val TAG = "sidecoach.voice"
        private const val STATE_DIM = 640
        private val DECODE_SPACE_PATTERN = Regex("""\A\s|\s\B|(\s)\b""")

        const val ENCODER = "encoder-model.int8.onnx"
        const val DECODER_JOINT = "decoder_joint-model.int8.onnx"
        const val VOCAB = "vocab.txt"
        const val CONFIG = "config.json"
        val FILES = listOf(ENCODER, DECODER_JOINT, VOCAB, CONFIG)

        /**
         * The model from [dir] (Setup's files), the preprocessor from the app's assets (the one
         * onnx-asr ships and uses, not the repository's copy).
         */
        fun load(context: Context, dir: File, threads: Int): Parakeet {
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(threads) }
            val began = SystemClock.elapsedRealtime()
            val preprocessorBytes = context.assets.open("asr/nemo128.onnx").use { it.readBytes() }
            val preprocessor = env.createSession(preprocessorBytes, options)
            val encoder = env.createSession(File(dir, ENCODER).path, options)
            val decoderJoint = env.createSession(File(dir, DECODER_JOINT).path, options)
            // `_AsrWithDecoding.__init__`: "token id" per line, the word-piece mark as a space.
            val vocab = LinkedHashMap<Int, String>()
            File(dir, VOCAB).readLines(Charsets.UTF_8).forEach { line ->
                val cut = line.trimEnd('\n').split(" ")
                vocab[cut[1].toInt()] = cut[0].replace("▁", " ")
            }
            val blank = vocab.entries.firstOrNull { it.value == "<blk>" }?.key
                ?: throw IllegalStateException("no <blk> in $VOCAB")
            val config = JSONObject(File(dir, CONFIG).readText(Charsets.UTF_8))
            Log.i(TAG, "parakeet loaded in ${SystemClock.elapsedRealtime() - began} ms")
            return Parakeet(env, preprocessor, encoder, decoderJoint, vocab, blank, config.optInt("max_tokens_per_step", 10))
        }
    }
}
