package com.sidecoach.voice

import android.util.Base64
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
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

/**
 * The listening against the desktop's own (`tools/make_stt_fixtures.py`): the chess correction (every
 * string of `tests/test_voice_correct.py` among them), every spoken form of the legal moves, and
 * difflib's ratio, exactly; Silero's probabilities within a tolerance and the endpointer's events
 * exactly; Parakeet int8's transcript and tokens for speech clips.
 */
@RunWith(AndroidJUnit4::class)
class SttParityTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext
    private val fixtures: JSONObject by lazy {
        val context = InstrumentationRegistry.getInstrumentation().context
        JSONObject(context.assets.open("voice/stt.json").bufferedReader(Charsets.UTF_8).use { it.readText() })
    }
    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) failures.add("$what\n  kotlin: $mine\n  python: $theirs")
    }

    private fun strings(array: JSONArray): List<String> = (0 until array.length()).map { array.getString(it) }

    private fun floats(base64: String): FloatArray {
        val bytes = Base64.decode(base64, Base64.DEFAULT)
        val out = FloatArray(bytes.size / 4)
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    @Test
    fun theCorrectionReadsMovesAsTheDesktopDoes() {
        val corrections = fixtures.getJSONArray("corrections")
        val began = System.currentTimeMillis()
        for (i in 0 until corrections.length()) {
            val case = corrections.getJSONObject(i)
            val board = if (case.isNull("fen")) null else Board(case.getString("fen"))
            val fixed = Correct.correct(case.getString("raw"), board)
            val where = "#$i \"${case.getString("raw")}\" on ${if (case.isNull("fen")) "no board" else case.getString("fen")}"
            check("$where text", fixed.text, case.getString("text"))
            check("$where moves", fixed.moves, strings(case.getJSONArray("moves")))
        }
        Log.i(TAG, "${corrections.length()} corrections in ${System.currentTimeMillis() - began} ms")
        val normalised = fixtures.getJSONArray("normalised")
        for (i in 0 until normalised.length()) {
            val case = normalised.getJSONObject(i)
            check("normalise \"${case.getString("raw")}\"", Correct.normalise(case.getString("raw")), case.getString("norm"))
        }
        // Every spoken form of every legal move, and the move it names.
        val forms = fixtures.getJSONArray("forms")
        for (i in 0 until forms.length()) {
            val case = forms.getJSONObject(i)
            val theirs = case.getJSONObject("forms").let { o -> o.keys().asSequence().associateWith { o.getString(it) } }
            check("spoken forms on ${case.getString("fen")}", Correct.spokenForms(Board(case.getString("fen"))).toMap(), theirs)
        }
        val ratios = fixtures.getJSONArray("ratios")
        for (i in 0 until ratios.length()) {
            val case = ratios.getJSONObject(i)
            val mine = SequenceMatcher(case.getString("a"), case.getString("b")).ratio()
            check("ratio \"${case.getString("a")}\" ~ \"${case.getString("b")}\"", mine, case.getDouble("ratio"))
        }
        assertTrue(failures.take(20).joinToString("\n\n") + "\n(${failures.size} failures)", failures.isEmpty())
    }

    @Test
    fun theGateAndTheEndpointerHearWhatTheDesktopHears() {
        val silero = File(app.filesDir, "models/silero/${SileroGate.FILE}")
        assumeTrue("no Silero in ${silero.parent} (tools/test.ps1 pushes it)", silero.exists())
        val gate = SileroGate(silero)
        assertTrue(gate.healthcheck().second, gate.load())
        val clips = fixtures.getJSONArray("clips")
        var worst = 0.0
        for (i in 0 until clips.length()) {
            val clip = clips.getJSONObject(i)
            val audio = floats(clip.getString("samples"))
            val probs = clip.getJSONArray("probs")
            gate.reset()
            var start = 0
            var k = 0
            while (start < audio.size) {
                val mine = gate.probability(audio.copyOfRange(start, minOf(start + HOP, audio.size)))
                worst = maxOf(worst, abs(mine - probs.getDouble(k)))
                start += HOP
                k += 1
            }
            check("clip #$i frames", k, probs.length())
            // The endpointer is plain logic: on the desktop's probabilities, the desktop's events.
            val endpointer = Endpointer()
            val events = (0 until probs.length()).map { endpointer.feed(probs.getDouble(it)) }
            val theirs = clips.getJSONObject(i).getJSONArray("events")
            check("clip #$i events", events, (0 until theirs.length()).map { if (theirs.isNull(it)) null else theirs.getString(it) })
        }
        Log.i(TAG, "silero: worst probability difference ${"%.6f".format(worst)}")
        check("silero within 1e-3", worst <= 1e-3, true)
        gate.close()
        assertTrue(failures.joinToString("\n\n"), failures.isEmpty())
    }

    @Test
    fun parakeetHearsWhatTheDesktopHears() {
        val dir = com.sidecoach.ui.PanelListen.modelDir(app)
        assumeTrue("no Parakeet in $dir (tools/test.ps1 pushes it)", Parakeet.FILES.all { File(dir, it).exists() })
        val began = System.currentTimeMillis()
        val parakeet = Parakeet.load(app, dir, 4)
        Log.i(TAG, "parakeet loaded in ${System.currentTimeMillis() - began} ms")
        parakeet.use {
            val clips = fixtures.getJSONArray("clips")
            for (i in 0 until clips.length()) {
                val clip = clips.getJSONObject(i)
                val audio = floats(clip.getString("samples"))
                val start = System.currentTimeMillis()
                val (text, tokens) = it.recognizeTokens(audio)
                val took = System.currentTimeMillis() - start
                Log.i(TAG, "\"${clip.getString("said")}\" -> \"${Py.strip(text)}\" in $took ms for ${"%.1f".format(audio.size / 16000.0)} s")
                check("clip #$i text", Py.strip(text), clip.getString("text"))
                check("clip #$i tokens", tokens, strings(clip.getJSONArray("tokens")))
            }
        }
        assertTrue(failures.joinToString("\n\n"), failures.isEmpty())
    }

    companion object {
        private const val TAG = "sidecoach.voice"
    }
}
