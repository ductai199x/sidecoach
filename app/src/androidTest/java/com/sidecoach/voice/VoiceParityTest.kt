package com.sidecoach.voice

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The voice's front end against the desktop's own (`tools/make_voice_fixtures.py`): notation read
 * out, phonemizer's phonemes, Kokoro's phonemes, tokens, batches and pauses, for every coach line
 * of the recorded sessions and the edge cases; the silence trims; the long strings' batches; and
 * Kokoro's audio within a tolerance.
 */
@RunWith(AndroidJUnit4::class)
class VoiceParityTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext
    private val fixtures: JSONObject by lazy {
        val context = InstrumentationRegistry.getInstrumentation().context
        JSONObject(context.assets.open("voice/voice.json").bufferedReader(Charsets.UTF_8).use { it.readText() })
    }
    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) failures.add("$what\n  kotlin: $mine\n  python: $theirs")
    }

    private fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }

    private fun doubles(array: JSONArray): List<Double> = (0 until array.length()).map { array.getDouble(it) }

    private fun longs(array: JSONArray): List<Long> = (0 until array.length()).map { array.getLong(it) }

    private fun floats(base64: String): FloatArray {
        val bytes = Base64.decode(base64, Base64.DEFAULT)
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun theFrontEndSaysWhatTheDesktopSays() {
        val tokenizer = KokoroTokenizer(app)
        val texts = fixtures.getJSONArray("texts")
        val began = System.currentTimeMillis()
        for (i in 0 until texts.length()) {
            val case = texts.getJSONObject(i)
            val text = case.getString("text")
            val spoken = Py.strip(Notation.speak(text))
            check("#$i spoken form of \"$text\"", spoken, case.getString("spoken"))
            for ((key, input) in listOf("given" to text, "said" to case.getString("spoken"))) {
                val want = case.getJSONObject(key)
                val where = "#$i $key \"$input\""
                check("$where raw", Phonemizer.phonemize(app, input, "en-us"), want.getString("raw"))
                val phonemes = tokenizer.phonemize(input, "en-us")
                check("$where phonemes", phonemes, want.getString("phonemes"))
                check("$where tokens", tokenizer.tokenize(phonemes, null).toList(), longs(want.getJSONArray("tokens")))
                val batches = Kokoro.splitPhonemes(Py.split(phonemes).joinToString(" "))
                check("$where batches", batches, strings(want.getJSONArray("batches")))
                val pauses = batches.dropLast(1).map { Kokoro.pauseAfter(it, 0.25, 0.1) } + if (batches.isEmpty()) emptyList() else listOf(0.0)
                check("$where pauses", pauses, doubles(want.getJSONArray("pauses")))
            }
        }
        Log.i(TAG, "${texts.length()} texts through the front end in ${System.currentTimeMillis() - began} ms")

        val trims = fixtures.getJSONArray("trims")
        for (i in 0 until trims.length()) {
            val case = trims.getJSONObject(i)
            check("trim #$i", Kokoro.trim(floats(case.getString("y"))), case.getInt("start") to case.getInt("end"))
        }
        val long = fixtures.getJSONArray("long")
        for (i in 0 until long.length()) {
            val case = long.getJSONObject(i)
            check("long #$i batches", Kokoro.splitPhonemes(case.getString("phonemes")), strings(case.getJSONArray("batches")))
        }
        assertTrue(failures.take(20).joinToString("\n\n") + "\n(${failures.size} failures)", failures.isEmpty())
    }

    @Test
    fun aLineIsSpokenInThePiecesThePlanSays() {
        val tokenizer = KokoroTokenizer(app)
        val plans = fixtures.getJSONArray("plans")
        var cut = 0
        for (i in 0 until plans.length()) {
            val case = plans.getJSONObject(i)
            val text = case.getString("text")
            val speed = case.getDouble("speed")
            val want = strings(case.getJSONArray("pieces"))
            val mine = Kokoro.plan(tokenizer, text, speed.toFloat(), "en-us")
            check("plan #$i \"$text\" at $speed pieces", mine.map { it.first }, want)
            check("plan #$i \"$text\" at $speed pauses", mine.map { it.second }, doubles(case.getJSONArray("pauses")))
            if (want.size > 1) cut += 1
        }
        Log.i(TAG, "${plans.length()} plans, $cut in pieces")
        assertTrue(failures.take(20).joinToString("\n\n") + "\n(${failures.size} failures)", failures.isEmpty())
    }

    @Test
    fun kokoroSpeaksAsTheDesktopDoes() {
        val dir = File(app.filesDir, "models/kokoro")
        val model = File(dir, KokoroTts.ONNX_FILE)
        val voices = File(dir, KokoroTts.VOICES_FILE)
        assumeTrue("no Kokoro in $dir (tools/test.ps1 pushes it)", model.exists() && voices.exists())
        assumeTrue("no audio fixtures", fixtures.has("audio"))
        val began = System.currentTimeMillis()
        val kokoro = Kokoro.load(app, model, voices, KokoroTts.THREADS)
        Log.i(TAG, "Kokoro loaded in ${System.currentTimeMillis() - began} ms")
        kokoro.use {
            val audio = fixtures.getJSONArray("audio")
            for (i in 0 until audio.length()) {
                val case = audio.getJSONObject(i)
                val theirs = floats(case.getString("samples"))
                val start = System.currentTimeMillis()
                val mine = it.create(case.getString("text"), "af_sarah", 1.0f, "en-us")
                val took = System.currentTimeMillis() - start
                // The trim cuts at 512-sample hops, and a frame whose level sits on its -60 dB line
                // can fall either side between two runtimes' last bits: an edge may move by a
                // hop. So the two are compared where they line up best, a hop either way.
                var correlation = -1.0
                var worst = 0.0
                for (shift in intArrayOf(-HOP, 0, HOP)) {
                    val (c, w) = compare(mine, theirs, shift)
                    if (c > correlation) {
                        correlation = c
                        worst = w
                    }
                }
                val seconds = mine.size / Kokoro.SAMPLE_RATE.toDouble()
                Log.i(
                    TAG,
                    "\"${case.getString("text")}\": ${mine.size} samples (python ${theirs.size}), " +
                        "correlation ${"%.5f".format(correlation)}, worst ${"%.4f".format(worst)}, " +
                        "${took} ms for ${"%.2f".format(seconds)} s (RTF ${"%.2f".format(took / 1000.0 / seconds)})",
                )
                check("audio #$i length within a hop at each end", abs(mine.size - theirs.size) <= 2 * HOP, true)
                check("audio #$i correlation >= 0.98", correlation >= 0.98, true)
            }
        }
        assertTrue(failures.joinToString("\n\n"), failures.isEmpty())
    }

    /** Correlation and worst difference of [mine] against [theirs], [mine] moved by [shift]. */
    private fun compare(mine: FloatArray, theirs: FloatArray, shift: Int): Pair<Double, Double> {
        var dot = 0.0
        var a2 = 0.0
        var b2 = 0.0
        var worst = 0.0
        for (k in theirs.indices) {
            val j = k + shift
            if (j < 0 || j >= mine.size) continue
            dot += mine[j].toDouble() * theirs[k]
            a2 += mine[j].toDouble() * mine[j]
            b2 += theirs[k].toDouble() * theirs[k]
            worst = maxOf(worst, abs(mine[j] - theirs[k]).toDouble())
        }
        return dot / sqrt(a2 * b2) to worst
    }

    companion object {
        private const val TAG = "sidecoach.voice"

        /** librosa's trim hop, as kokoro-onnx trims with it. */
        private const val HOP = 512
    }
}
