package com.sidecoach.voice

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * How fast the voice can be made on this phone: the Kokoro graph's variants (fp32 and fp16,
 * spinning threads or not, thread counts, XNNPACK) timed on the coach's own kind of lines, each
 * cut into the pieces the phone speaks, in rounds that alternate the order. A tool, not a test:
 * it runs only when asked (`-e bench 1`), in the `apart1` copy of the app, so a coaching session
 * live in the app goes on (Tai, 2026-10-03: the voice made slower than it speaks with a stream
 * playing beside it).
 *
 *   adb push kokoro-v1.0.fp16.onnx /sdcard/Android/data/com.sidecoach.apart1/files/
 *   (and kokoro-v1.0.onnx, voices-v1.0.bin)
 *   adb shell am instrument -w -e bench 1 -e class com.sidecoach.voice.KokoroBench \
 *       com.sidecoach.test/androidx.test.runner.SidecoachRunnerApart1
 *
 * Each variant's line: the time made over the speech made (below 1 keeps up), and the first
 * piece's time, the wait for the first sound. Also written to `kokoro.tsv` there.
 */
@RunWith(AndroidJUnit4::class)
class KokoroBench {
    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    /** [priority]: the creating thread's (Android's `Process.setThreadPriority`), which the graph's
     * thread pool, made with the session, inherits; 0 is the default. */
    private class Variant(
        val name: String,
        val model: String,
        val options: () -> OrtSession.SessionOptions,
        val priority: Int = 0,
    )

    private fun threads(n: Int, spin: Boolean) = {
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(n)
            if (!spin) addConfigEntry("session.intra_op.allow_spinning", "0")
        }
    }

    private fun xnnpack(n: Int) = {
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            addConfigEntry("session.intra_op.allow_spinning", "0")
            addXnnpack(mapOf("intra_op_num_threads" to n.toString()))
        }
    }

    /** The graph on the phone's GPU through Vulkan (Dawn, built into ONNX Runtime 1.30), the CPU
     * taking what it cannot run. */
    private fun webgpu() = {
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(5)
            // The Java API has no add for it: its generic add, private, by reflection (a tool).
            OrtSession.SessionOptions::class.java
                .getDeclaredMethod("addExecutionProvider", String::class.java, Map::class.java)
                .apply { isAccessible = true }
                .invoke(this, "WebGPU", emptyMap<String, String>())
        }
    }

    /** NNAPI: the phone's accelerators, in [fp16] or full precision. */
    private fun nnapi(fp16: Boolean) = {
        OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(5)
            val flags = java.util.EnumSet.noneOf(ai.onnxruntime.providers.NNAPIFlags::class.java)
            if (fp16) flags.add(ai.onnxruntime.providers.NNAPIFlags.USE_FP16)
            addNnapi(flags)
        }
    }

    private val lines = listOf(
        "Excellent: the only move that keeps the extra pawn.",
        "Use your king on f6 against the pawn on h4; the quiet approach wins it.",
        "Which square lets you do that while keeping the rook behind the passed pawn?",
        "That was fine. The centre holds, and the bishop is doing real work on the long diagonal.",
    )

    @Test
    @com.sidecoach.InProcess(1) // the copy of the app, never the app a session may be live in
    fun timeTheVariants() {
        assumeTrue("a tool: -e bench 1", InstrumentationRegistry.getArguments().getString("bench") == "1")
        // The copy's own folder: one the shell makes inside it is the shell's, closed to the app.
        val dir = app.getExternalFilesDir(null)!!
        val voices = File(dir, "voices-v1.0.bin")
        assumeTrue("no voices in $dir", voices.exists())
        val only = InstrumentationRegistry.getArguments().getString("variants")?.split(";")?.toSet()
        val variants = listOf(
            Variant("fp32 t5 (now)", "kokoro-v1.0.onnx", threads(5, spin = true)),
            Variant("fp32 t5 nospin", "kokoro-v1.0.onnx", threads(5, spin = false)),
            Variant("fp32 t4 nospin", "kokoro-v1.0.onnx", threads(4, spin = false)),
            Variant("fp32 xnnpack5", "kokoro-v1.0.onnx", xnnpack(5)),
            Variant("fp16 t5", "kokoro-v1.0.fp16.onnx", threads(5, spin = true)),
            Variant("fp16 t5 nospin", "kokoro-v1.0.fp16.onnx", threads(5, spin = false)),
            Variant("fp16 xnnpack5", "kokoro-v1.0.fp16.onnx", xnnpack(5)),
            Variant("dec-int8 t5", "kokoro-dec-int8.onnx", threads(5, spin = true)),
            Variant("stft-int8 t5", "kokoro-stft-int8.onnx", threads(5, spin = true)),
            Variant("int8 stage2-fp32", "no-stage2.onnx", threads(5, spin = true)),
            Variant("q stage2-fp32", "kokoro-q-nostage2.onnx", threads(5, spin = true)),
            Variant("q s2convs2-fp32", "kokoro-q-nos2c2.onnx", threads(5, spin = true)),
            Variant("fp32 webgpu", "kokoro-v1.0.onnx", webgpu()),
            Variant("fp32 nnapi", "kokoro-v1.0.onnx", nnapi(fp16 = false)),
            Variant("fp32 nnapi fp16", "kokoro-v1.0.onnx", nnapi(fp16 = true)),
            Variant("fp32 t5 urgent", "kokoro-v1.0.onnx", threads(5, spin = true), android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY),
            Variant("fp16 t5 urgent", "kokoro-v1.0.fp16.onnx", threads(5, spin = true), android.os.Process.THREAD_PRIORITY_URGENT_DISPLAY),
            Variant("fp16 t5 audio", "kokoro-v1.0.fp16.onnx", threads(5, spin = true), android.os.Process.THREAD_PRIORITY_AUDIO),
        ).filter { File(dir, it.model).exists() && (only == null || it.name in only) }
        if (InstrumentationRegistry.getArguments().getString("profile") == "1") {
            // Where the time goes on this phone: ONNX Runtime's own profile of the shipped graph
            // over the lines, written beside the models (`kokoro_profile.json`).
            val options = threads(5, spin = true)()
            options.enableProfiling(File(dir, "kokoro_profile").path)
            val graph = InstrumentationRegistry.getArguments().getString("model") ?: "kokoro-v1.0.onnx"
            val session = env().createSession(File(dir, graph).path, options)
            Kokoro(session, voices, KokoroTokenizer(app)).use { model ->
                speak(model, "Ready to play.")
                for (line in lines) speak(model, line)
                val written = session.endProfiling()
                File(written).copyTo(File(dir, "kokoro_profile.json"), overwrite = true)
            }
            return
        }
        val rounds = InstrumentationRegistry.getArguments().getString("rounds")?.toInt() ?: 2
        val out = StringBuilder("round\tvariant\tload_ms\tmade_ms\tspeech_ms\tpace\tfirst_piece_ms\n")
        val env = OrtEnvironment.getEnvironment()
        val tokenizer = KokoroTokenizer(app)
        for (round in 0 until rounds) {
            val order = if (round % 2 == 0) variants else variants.reversed()
            for (variant in order) {
                val loading = SystemClock.elapsedRealtime()
                // Each variant on a thread of its own, at its priority: the session's pool is
                // made from it and inherits it.
                var session: OrtSession? = null
                val maker = Thread {
                    android.os.Process.setThreadPriority(variant.priority)
                    session = runCatching { env.createSession(File(dir, variant.model).path, variant.options()) }
                        .onFailure { Log.w(TAG, "${variant.name}: could not load: $it") }
                        .getOrNull()
                    val made = session ?: return@Thread
                    timed(variant, made, loading, voices, tokenizer, round, out)
                }
                maker.start()
                maker.join()
                if (session == null) continue
                System.gc()
            }
        }
        File(dir, "kokoro.tsv").writeText(out.toString())
    }

    private fun timed(
        variant: Variant,
        session: OrtSession,
        loading: Long,
        voices: File,
        tokenizer: KokoroTokenizer,
        round: Int,
        out: StringBuilder,
    ) {
        val loadMs = SystemClock.elapsedRealtime() - loading
        Kokoro(session, voices, tokenizer).use { model ->
            // One line first, unmeasured: the first pass sets up the graph's buffers.
            speak(model, "Ready to play.")
            var made = 0L
            var speech = 0.0
            val firsts = ArrayList<Long>()
            for (line in lines) {
                val (m, s, first) = speak(model, line)
                made += m
                speech += s
                firsts.add(first)
            }
            val pace = made / speech
            val row = "$round\t${variant.name}\t$loadMs\t$made\t${speech.toLong()}\t${"%.2f".format(pace)}\t${firsts.joinToString(",")}"
            Log.i(TAG, row)
            out.append(row).append('\n')
        }
    }

    /** The line made in its pieces: the time taken, the speech made, the first piece's time. */
    private fun speak(model: Kokoro, line: String): Triple<Long, Double, Long> {
        val spoken = Py.strip(Notation.speak(line, voice = true))
        val (style, pieces) = model.pieces(spoken, "bm_george", 1.0f, "en-us")
        var made = 0L
        var speech = 0.0
        var first = -1L
        for (piece in pieces) {
            val from = SystemClock.elapsedRealtime()
            val samples = model.synthesize(piece, style, 1.0f)
            val took = SystemClock.elapsedRealtime() - from
            if (first < 0) first = took
            made += took
            speech += samples.size * 1000.0 / Kokoro.SAMPLE_RATE
        }
        return Triple(made, speech, first)
    }

    private fun env(): OrtEnvironment = OrtEnvironment.getEnvironment()

    companion object {
        private const val TAG = "sidecoach.bench"
    }
}
