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
        val style = checked(voice, speed)
        val batches = batches(phonemes(tokenizer, text, lang), sentencePause, clausePause)
        if (batches.isEmpty()) throw IllegalArgumentException("Nothing to synthesize, \"$text\" produced no phonemes")
        return style to batches
    }

    /** `Kokoro._prepare`'s checks: the speed in range and the voice on offer; its style. */
    private fun checked(voice: String, speed: Float): FloatArray {
        if (speed < 0.5f || speed > 2.0f) throw IllegalArgumentException("Speed should be between 0.5 and 2.0, got $speed")
        if (voice !in Voices.names(voicesPath)) throw IllegalArgumentException("Voice $voice not found in available voices")
        return voiceStyle(voice)
    }

    /**
     * The line as the phone speaks it ([plan]): the voice's style and the pieces, each with
     * the pause after it, checked as [prepare] checks. Each piece is synthesised as a batch is
     * ([synthesize]).
     */
    fun pieces(
        text: String,
        voice: String,
        speed: Float,
        lang: String,
        sentencePause: Double = 0.25,
        clausePause: Double = 0.1,
    ): Pair<FloatArray, List<Pair<String, Double>>> {
        val style = checked(voice, speed)
        val pieces = plan(tokenizer, text, speed, lang, sentencePause, clausePause)
        if (pieces.isEmpty()) throw IllegalArgumentException("Nothing to synthesize, \"$text\" produced no phonemes")
        return style to pieces
    }

    /** `Kokoro._create_batch` with trimming, for one of [pieces]. */
    fun synthesize(piece: Pair<String, Double>, style: FloatArray, speed: Float): FloatArray =
        createBatch(piece.first, style, speed, piece.second)

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

        /**
         * `Kokoro._prepare`'s phonemes: the tokenizer's, whitespace runs made single spaces.
         * Newlines are not in the vocabulary, so without this the lines of a text run together
         * with no gap for the model to pause on.
         */
        fun phonemes(tokenizer: KokoroTokenizer, text: String, lang: String): String =
            Py.split(tokenizer.phonemize(text, lang)).joinToString(" ")

        /**
         * `Kokoro._prepare`'s batches, each with the pause after it: trimming removes the
         * silence the model puts at a batch end, so the pause the punctuation calls for is
         * added back when joining.
         */
        fun batches(phonemes: String, sentencePause: Double, clausePause: Double): List<Pair<String, Double>> {
            val batched = splitPhonemes(phonemes, MAX_PHONEME_LENGTH)
            if (batched.isEmpty()) return emptyList()
            return batched.dropLast(1).map { it to pauseAfter(it, sentencePause, clausePause) } + (batched.last() to 0.0)
        }

        /** Speech a phoneme makes at speed 1, in milliseconds (55.8 over the coach's lines). */
        const val MS_PER_PHONEME = 56.0

        /** A line with less speech than this, in seconds, is spoken whole. */
        const val WHOLE_BELOW_S = 3.5

        /** The shortest first piece, in phonemes: about a second of speech. */
        const val FIRST_AT_LEAST = 16

        /** Python's `\s` in a `str` pattern: the characters `str.isspace` holds ([Py.isSpace]). */
        private const val SPACE = """[\s\u001C-\u001F\u0085\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]"""
        private val DASH = Regex("""$SPACE+(?:--|—|–)$SPACE+""")
        private val CLAUSE = Regex("""(?<=[.!?…,;:])$SPACE+""")

        private fun length(s: String): Int = s.codePointCount(0, s.length)

        /**
         * The pieces the phone speaks a line in (`tools/phone_speech.py`; Tai, 2026-09-23: the
         * phone only, the desktop speaks whole lines). On the desktop a line is one pass on the
         * GPU in a tenth of a second; on the phone's CPU a pass takes about half as long as the
         * speech, so a whole line starts only once all of it is made. In pieces, each is spoken
         * while the next is made.
         *
         * A line of under [WHOLE_BELOW_S] of speech stays whole, batched as kokoro-onnx batches
         * it: the desktop's own audio (in Tai's blind test the short lines sounded better
         * whole). A longer line is cut where the text pauses -- after a stop, a colon, a
         * semicolon or a comma in its phonemes, and at a dash in its text, which the
         * phonemizer drops, so each dash-separated part is phonemized on its own -- and its
         * clauses merged into few pieces: the first until it has [FIRST_AT_LEAST] phonemes,
         * each after that up to twice the one before, and a tail of less than half the piece
         * before joins it. A cut never falls between two words of a clause; a clause too long
         * for one pass is cut as kokoro-onnx cuts it.
         *
         * Over the coach's lines at the Pixel's speed: the first sound 1.14 s after the line
         * on average against 2.02 whole (p90 1.59 against 3.39), naturalness unchanged
         * (UTMOS22 +0.003 on the cut lines), 9 of 779 words heard differently.
         */
        fun plan(
            tokenizer: KokoroTokenizer,
            text: String,
            speed: Float,
            lang: String,
            sentencePause: Double = 0.25,
            clausePause: Double = 0.1,
        ): List<Pair<String, Double>> {
            val whole = phonemes(tokenizer, text, lang)
            if (length(whole) * MS_PER_PHONEME / 1000 / speed < WHOLE_BELOW_S) return batches(whole, sentencePause, clausePause)
            val atoms = ArrayList<String>()
            val parts = DASH.split(text)
            for (part in parts) {
                if (Py.strip(part).isEmpty()) continue
                // A text with no dash is its one part, whose phonemes are the line's.
                val phonemes = if (parts.size == 1) whole else phonemes(tokenizer, part, lang)
                for (atom in CLAUSE.split(phonemes)) {
                    if (Py.strip(atom).isEmpty()) continue
                    if (length(atom) > MAX_PHONEME_LENGTH) atoms.addAll(splitPhonemes(atom)) else atoms.add(atom)
                }
            }
            if (atoms.isEmpty()) return batches(whole, sentencePause, clausePause)
            val sizes = atoms.map { length(it) }
            val pieces = ArrayList<StringBuilder>()
            val lengths = ArrayList<Int>()
            val first = StringBuilder(atoms[0])
            var size = sizes[0]
            var i = 1
            while (size < FIRST_AT_LEAST && i < atoms.size && size + 1 + sizes[i] <= MAX_PHONEME_LENGTH) {
                first.append(' ').append(atoms[i])
                size += 1 + sizes[i]
                i += 1
            }
            pieces.add(first)
            lengths.add(size)
            while (i < atoms.size) {
                val piece = StringBuilder(atoms[i])
                size = sizes[i]
                i += 1
                val limit = minOf(MAX_PHONEME_LENGTH, 2 * lengths.last())
                while (i < atoms.size && size + 1 + sizes[i] <= limit) {
                    piece.append(' ').append(atoms[i])
                    size += 1 + sizes[i]
                    i += 1
                }
                pieces.add(piece)
                lengths.add(size)
            }
            val n = pieces.size
            if (n > 1 && lengths[n - 1] < lengths[n - 2] / 2 && lengths[n - 2] + 1 + lengths[n - 1] <= MAX_PHONEME_LENGTH) {
                // A tail too short to stand alone joins the piece before it.
                val last = pieces.removeAt(n - 1)
                pieces[n - 2].append(' ').append(last)
            }
            val out = pieces.map { it.toString() }
            return out.dropLast(1).map { it to pauseAfter(it, sentencePause, clausePause) } + (out.last() to 0.0)
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
