package com.sidecoach.bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.SystemClock
import android.util.Log
import java.nio.FloatBuffer

/**
 * Spike B, the piece net: how long the phone takes to embed a whole board.
 *
 * The desktop reads a position by cutting the board into 64 cells, scaling each to 224
 * and pushing the batch through `yolo26n-embed.onnx` (6 MB, frozen, no training of ours);
 * the 1280-long vectors are matched against exemplars. Every frame that changes pays for
 * it, so this number decides whether the vision port is possible at all, and at what
 * rate. The desktop's 3090 does a batch in a few milliseconds; a laptop CPU in ~300 ms.
 *
 * Two ways of running it are timed, because they are the two the phone offers: the CPU
 * with XNNPACK, and NNAPI where the device still has it (deprecated since Android 15,
 * and Tensor's accelerator is reached through it).
 */
object EmbedderBench {

    private const val TAG = "sidecoach.bench"
    private const val MODEL = "yolo26n-embed.onnx"
    private const val CELLS = 64
    private const val SIDE = 224
    private const val DIM = 1280

    data class Result(
        val provider: String,
        val loadMs: Long,
        val firstMs: Long,
        val medianMs: Long,
        val bestMs: Long,
        val worstMs: Long,
        val runs: Int,
        val error: String? = null,
    ) {
        override fun toString(): String =
            if (error != null) "$provider: $error"
            else "$provider: load ${loadMs} ms, first ${firstMs} ms, " +
                "median ${medianMs} ms, best ${bestMs}, worst ${worstMs} ($runs runs)"
    }

    /** Both providers, each loaded fresh and run [runs] times over a full board. */
    fun run(context: Context, runs: Int = 10): List<Result> {
        val bytes = context.assets.open(MODEL).use { it.readBytes() }
        // One board's worth of pixels, the shape the net wants: (64, 3, 224, 224). The
        // values are noise, not a board -- the net's cost does not depend on them.
        val input = FloatArray(CELLS * 3 * SIDE * SIDE) { (it % 255) / 255f }
        return listOf(
            time("cpu (xnnpack)", bytes, input, runs) { /* the default */ },
            time("nnapi", bytes, input, runs) { it.addNnapi() },
        )
    }

    private fun time(
        provider: String,
        model: ByteArray,
        input: FloatArray,
        runs: Int,
        configure: (OrtSession.SessionOptions) -> Unit,
    ): Result {
        val env = OrtEnvironment.getEnvironment()
        var session: OrtSession? = null
        return try {
            val began = SystemClock.elapsedRealtime()
            val options = OrtSession.SessionOptions()
            // Two threads: the engine and the coach want the rest, as on the desktop.
            options.setIntraOpNumThreads(2)
            configure(options)
            session = env.createSession(model, options)
            val loadMs = SystemClock.elapsedRealtime() - began
            val name = session.inputNames.first()
            val shape = longArrayOf(CELLS.toLong(), 3, SIDE.toLong(), SIDE.toLong())
            val times = ArrayList<Long>(runs)
            repeat(runs) {
                val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape)
                val at = SystemClock.elapsedRealtime()
                tensor.use { t ->
                    session.run(mapOf(name to t)).use { out ->
                        // Touch the output so nothing is optimised away, and check the
                        // shape is the 1280 the matcher expects.
                        val info = out[0].info
                        check(info.toString().contains("$DIM")) { "unexpected output: $info" }
                    }
                }
                times.add(SystemClock.elapsedRealtime() - at)
            }
            val sorted = times.sorted()
            Result(
                provider = provider,
                loadMs = loadMs,
                firstMs = times.first(),
                medianMs = sorted[sorted.size / 2],
                bestMs = sorted.first(),
                worstMs = sorted.last(),
                runs = runs,
            ).also { Log.i(TAG, it.toString()) }
        } catch (exc: Throwable) {
            Result(provider, 0, 0, 0, 0, 0, 0, "${exc::class.simpleName}: ${exc.message}")
                .also { Log.w(TAG, it.toString()) }
        } finally {
            runCatching { session?.close() }
        }
    }
}
