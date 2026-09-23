package com.sidecoach.voice

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.DataInputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.zip.ZipFile
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Kokoro-82M, as kokoro-onnx 0.6.1 runs it for the desktop (`Kokoro.create`): the text
 * phonemized with espeak-ng (see [Phonemizer]), the phonemes cut into batches the model takes in
 * one pass (`chunker.split_phonemes`), each batch synthesised with the voice's style for its
 * length, its silence trimmed (`trim.trim`, librosa's) and the pause its punctuation calls for
 * added, the batches joined. The models the app runs have no duration output, so the timed pause
 * insertion (`pauses.insert`) has nothing to work with and leaves the audio as it is, as it does
 * on the desktop.
 */
class Kokoro(
    private val session: OrtSession,
    private val voicesPath: File,
    val tokenizer: KokoroTokenizer,
) : AutoCloseable {

    private val env = OrtEnvironment.getEnvironment()
    private val styles = HashMap<String, FloatArray>()

    /** Voice name -> its style table, 510 rows of 256, read from the voices file on first use. */
    private fun voiceStyle(name: String): FloatArray = synchronized(styles) {
        styles.getOrPut(name) { readVoice(voicesPath, name) }
    }

    fun voices(): List<String> = Voices.names(voicesPath).sorted()

    // ---- synthesis -------------------------------------------------------------------------

    /**
     * `Kokoro.create(text, voice, speed, lang)` with its defaults: trimmed batches, a quarter
     * second after a sentence and a tenth after a clause, not continuous. Mono float samples at
     * [SAMPLE_RATE].
     */
    fun create(
        text: String,
        voice: String,
        speed: Float = 1.0f,
        lang: String = "en-us",
        sentencePause: Double = 0.25,
        clausePause: Double = 0.1,
    ): FloatArray {
        val (style, batches) = prepare(text, voice, speed, lang, sentencePause, clausePause)
        val parts = batches.map { (phonemes, pause) -> createBatch(phonemes, style, speed, pause) }
        val out = FloatArray(parts.sumOf { it.size })
        var at = 0
        for (part in parts) {
            part.copyInto(out, at)
            at += part.size
        }
        return out
    }

    /** `Kokoro._prepare`: the voice's style, and the text as batches, each with the pause after it. */
    fun prepare(
        text: String,
        voice: String,
        speed: Float,
        lang: String,
        sentencePause: Double,
        clausePause: Double,
    ): Pair<FloatArray, List<Pair<String, Double>>> {
        if (speed < 0.5f || speed > 2.0f) throw IllegalArgumentException("Speed should be between 0.5 and 2.0, got $speed")
        if (voice !in Voices.names(voicesPath)) throw IllegalArgumentException("Voice $voice not found in available voices")
        val style = voiceStyle(voice)
        var phonemes = tokenizer.phonemize(text, lang)
        // Newlines are not in the vocabulary, so without this the lines of a text run together
        // with no gap for the model to pause on.
        phonemes = Py.split(phonemes).joinToString(" ")
        val batched = splitPhonemes(phonemes, MAX_PHONEME_LENGTH)
        if (batched.isEmpty()) throw IllegalArgumentException("Nothing to synthesize, \"$text\" produced no phonemes")
        // Trimming removes the silence the model puts at a batch end, so the pause the
        // punctuation calls for is added back when joining.
        val batches = batched.dropLast(1).map { it to pauseAfter(it, sentencePause, clausePause) }
        return style to (batches + (batched.last() to 0.0))
    }

    /** `Kokoro._create_batch` with trimming: synthesise, trim the silence, pause after. */
    private fun createBatch(phonemes: String, style: FloatArray, speed: Float, pause: Double): FloatArray {
        val tokens = tokenizer.tokenize(phonemes)
        if (tokens.isEmpty()) throw IllegalArgumentException("No phonemes of \"$phonemes\" are in the model vocabulary")
        val audio = infer(tokens, styleFor(style, tokens.size), speed)
        // Trim leading and trailing silence for a more natural sound concatenation.
        val (start, end) = trim(audio)
        val trimmed = audio.copyOfRange(start, end)
        if (pause == 0.0) return trimmed
        val silence = (pause * SAMPLE_RATE).toInt()
        return trimmed.copyOf(trimmed.size + silence)
    }

    /** `Kokoro._style_for`: one style vector per phoneme count, so n phonemes use row n - 1. */
    private fun styleFor(voice: FloatArray, length: Int): FloatArray {
        val rows = voice.size / STYLE_DIM
        val row = min(length, rows) - 1
        return voice.copyOfRange(row * STYLE_DIM, (row + 1) * STYLE_DIM)
    }

    /** `Kokoro._infer`: one window of tokens through the model, the pads around it. */
    fun infer(tokens: LongArray, style: FloatArray, speed: Float): FloatArray {
        val padded = LongArray(tokens.size + 2)
        tokens.copyInto(padded, 1)
        OnnxTensor.createTensor(env, LongBuffer.wrap(padded), longArrayOf(1, padded.size.toLong())).use { tokenTensor ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(style), longArrayOf(1, STYLE_DIM.toLong())).use { styleTensor ->
                OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(speed)), longArrayOf(1)).use { speedTensor ->
                    val inputs = mapOf("tokens" to tokenTensor, "style" to styleTensor, "speed" to speedTensor)
                    session.run(inputs).use { result ->
                        val audio = result[0] as OnnxTensor
                        val buffer = audio.floatBuffer
                        return FloatArray(buffer.remaining()).also { buffer.get(it) }
                    }
                }
            }
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        const val SAMPLE_RATE = 24000
        const val MAX_PHONEME_LENGTH = 510
        private const val STYLE_DIM = 256

        const val SENTENCE_MARKS = ".!?…"
        const val CLAUSE_MARKS = ",;:"

        /** The session for [model], and the vocabulary from the assets. */
        fun load(context: Context, model: File, voices: File, threads: Int): Kokoro {
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(threads)
            val session = env.createSession(model.path, options)
            return Kokoro(session, voices, KokoroTokenizer(context))
        }

        // ---- the chunker (`chunker.py`) ------------------------------------------------------

        /** Least to most disruptive place to cut: sentence, clause, word. */
        private val BOUNDARIES = listOf(
            Regex("""(?<=[.!?…])\s+"""),
            Regex("""(?<=[,;:])\s+"""),
            Regex("""\s+"""),
        )

        private fun atoms(phonemes: String, maxLength: Int, level: Int, out: MutableList<String>) {
            val codes = phonemes.codePoints().toArray()
            if (codes.size <= maxLength) {
                if (phonemes.isNotEmpty()) out.add(phonemes)
                return
            }
            for (index in level until BOUNDARIES.size) {
                val pieces = BOUNDARIES[index].split(phonemes)
                if (pieces.size > 1) {
                    for (piece in pieces) atoms(Py.strip(piece), maxLength, index + 1, out)
                    return
                }
            }
            // A single unbroken run longer than the context: sliced rather than dropped.
            var start = 0
            while (start < codes.size) {
                val end = min(start + maxLength, codes.size)
                out.add(String(codes, start, end - start))
                start += maxLength
            }
        }

        /** Group consecutive atoms into batches within [limit], as index ranges. */
        private fun pack(lengths: List<Int>, limit: Int): List<Pair<Int, Int>> {
            val batches = ArrayList<Pair<Int, Int>>()
            var start = 0
            var size = 0
            for ((index, length) in lengths.withIndex()) {
                val candidate = if (index == start) length else size + 1 + length
                if (candidate > limit && index > start) {
                    batches.add(start to index)
                    start = index
                    size = length
                } else {
                    size = candidate
                }
            }
            if (lengths.isNotEmpty()) batches.add(start to lengths.size)
            return batches
        }

        /**
         * `split_phonemes`: batches of at most [maxLength], balanced -- the smallest limit that
         * still needs no extra batch, so no short remainder is spoken at a different rate.
         */
        fun splitPhonemes(phonemes: String, maxLength: Int = MAX_PHONEME_LENGTH): List<String> {
            val atoms = ArrayList<String>()
            atoms(Py.strip(phonemes), maxLength, 0, atoms)
            if (atoms.isEmpty()) return emptyList()
            val lengths = atoms.map { it.codePointCount(0, it.length) }
            val fewest = pack(lengths, maxLength).size
            var low = lengths.max()
            var high = maxLength
            while (low < high) {
                val middle = (low + high) / 2
                if (pack(lengths, middle).size <= fewest) high = middle else low = middle + 1
            }
            return pack(lengths, low).map { (start, end) -> atoms.subList(start, end).joinToString(" ") }
        }

        /** `pause_after`: the silence a batch ending with this text should be followed by. */
        fun pauseAfter(phonemes: String, sentence: Double, clause: Double): Double {
            val stripped = phonemes.trimEnd { Py.isSpace(it) }
            val mark = if (stripped.isEmpty()) "" else stripped.substring(stripped.length - 1)
            // Python's `"" in SENTENCE_MARKS` holds, as `mark in ...` does for any mark there.
            if (SENTENCE_MARKS.contains(mark)) return sentence
            if (CLAUSE_MARKS.contains(mark)) return clause
            return 0.0
        }

        // ---- trimming (`trim.trim`, librosa's) ---------------------------------------------

        /**
         * The non-silent interval of [y]: frames of 2048 samples every 512, centred with zero
         * padding, their RMS in dB against the loudest frame, those above -60 dB kept.
         * `trim(y)` with its defaults returns `y[start:end]`.
         */
        fun trim(y: FloatArray, topDb: Double = 60.0, frameLength: Int = 2048, hopLength: Int = 512): Pair<Int, Int> {
            val pad = frameLength / 2
            val padded = FloatArray(y.size + 2 * pad)
            y.copyInto(padded, pad)
            if (padded.size < frameLength) {
                throw IllegalArgumentException("Input is too short (n=${padded.size}) for frame_length=$frameLength")
            }
            val frames = (padded.size - frameLength) / hopLength + 1
            // rms: sqrt(mean(x^2)) per frame, in float32 as librosa's dtype has it.
            val rms = FloatArray(frames)
            for (f in 0 until frames) {
                var sum = 0.0
                val base = f * hopLength
                for (i in 0 until frameLength) {
                    val v = padded[base + i]
                    sum += (v * v).toDouble()
                }
                rms[f] = sqrt((sum / frameLength).toFloat())
            }
            // amplitude_to_db(rms, ref=np.max, top_db=None) > -top_db
            var ref = 0f
            for (v in rms) ref = max(ref, kotlin.math.abs(v))
            val refPower = ref * ref
            val refDb = 10.0 * log10(max(1e-10, refPower.toDouble()))
            var first = -1
            var last = -1
            for (f in 0 until frames) {
                val power = rms[f] * rms[f]
                val db = 10.0 * log10(max(1e-10, power.toDouble())) - refDb
                if (db > -topDb) {
                    if (first < 0) first = f
                    last = f
                }
            }
            if (first < 0) return 0 to 0
            val start = first * hopLength
            val end = min(y.size, (last + 1) * hopLength)
            return start to end
        }

        // ---- the voices file (`np.load` of an .npz) -----------------------------------------

        /** One voice's array, float32 little-endian, flattened: `np.load(path)[name]`. */
        fun readVoice(path: File, name: String): FloatArray = ZipFile(path).use { zip ->
            val entry = zip.getEntry("$name.npy") ?: throw IllegalArgumentException("Voice $name not found in available voices")
            DataInputStream(zip.getInputStream(entry).buffered()).use { input -> readNpy(input) }
        }

        /** An .npy array of '<f4' in C order, as a flat array. */
        private fun readNpy(input: DataInputStream): FloatArray {
            val magic = ByteArray(6)
            input.readFully(magic)
            if (magic[0] != 0x93.toByte() || String(magic, 1, 5, Charsets.US_ASCII) != "NUMPY") {
                throw IllegalArgumentException("not an .npy file")
            }
            val major = input.readUnsignedByte()
            input.readUnsignedByte()
            val headerLength = if (major == 1) {
                val b = ByteArray(2).also { input.readFully(it) }
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xffff
            } else {
                val b = ByteArray(4).also { input.readFully(it) }
                ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).int
            }
            val header = ByteArray(headerLength).also { input.readFully(it) }.toString(Charsets.US_ASCII)
            if ("'<f4'" !in header || "'fortran_order': False" !in header) {
                throw IllegalArgumentException("unexpected voice array: $header")
            }
            val shape = Regex("""'shape': \(([^)]*)\)""").find(header)?.groupValues?.get(1)
                ?.split(",")?.map { it.trim() }?.filter { it.isNotEmpty() }?.map { it.toInt() }
                ?: throw IllegalArgumentException("no shape in $header")
            val count = shape.fold(1) { a, b -> a * b }
            val bytes = ByteArray(count * 4).also { input.readFully(it) }
            val out = FloatArray(count)
            ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
            return out
        }
    }
}

/**
 * kokoro-onnx's `Tokenizer`: text to phonemes through phonemizer, and phonemes to the model's
 * token ids with the vocabulary kokoro-onnx ships (`DEFAULT_VOCAB`, in the app's assets).
 */
class KokoroTokenizer(private val context: Context) {

    /** Phoneme character to token id. */
    val vocab: Map<Char, Long> = run {
        val text = context.assets.open("kokoro/vocab.json").use { it.readBytes().toString(Charsets.UTF_8) }
        val table = JSONObject(text).getJSONObject("vocab")
        val out = HashMap<Char, Long>()
        for (key in table.keys()) out[key.single()] = table.getLong(key)
        out
    }

    /** `Tokenizer.phonemize(text, lang)`: the phonemes the vocabulary knows, stripped. */
    fun phonemize(text: String, lang: String = "en-us"): String {
        val phonemes = Phonemizer.phonemize(context, Py.strip(text), lang)
        return Py.strip(phonemes.filter { it in vocab })
    }

    /** `Tokenizer.tokenize`: the token ids of the phonemes, anything outside the vocabulary dropped. */
    fun tokenize(phonemes: String, limit: Int? = Kokoro.MAX_PHONEME_LENGTH): LongArray {
        if (limit != null && phonemes.codePointCount(0, phonemes.length) > limit) {
            throw IllegalArgumentException("text is too long, must be less than $limit phonemes")
        }
        return phonemes.mapNotNull { vocab[it] }.toLongArray()
    }
}
