package com.sidecoach.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * The frozen embedding net the square matcher reads cells with, ported from
 * `sidecoach/vision/embedding.py`.
 *
 * Ultralytics' YOLO26n-cls cut before its ImageNet linear, nothing of ours trained: a cell
 * crop in, a 1280-long unit vector out. The pieces on screen are named by cosine similarity
 * to the exemplars taken from the start position, so the crop must be prepared exactly as
 * the desktop prepares it -- BGR bytes as floats 0..255 at [INPUT] square, the channel flip
 * and the scaling inside the graph.
 *
 * The input goes into a **direct** buffer, and that is not a detail: handed a heap
 * FloatBuffer, onnxruntime's Java API copies it into native memory the slow way, and a
 * board's 9.6 million floats took 4.8 seconds against 0.1 for the inference itself
 * (measured on the emulator, 2026-09-22). The buffer is kept and refilled.
 */
class CellEmbedder private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
) : AutoCloseable {

    private var direct: FloatBuffer? = null
    private var directCells = 0
    private val interleaved = FloatArray(INPUT * INPUT * 3)
    private val planar = FloatArray(INPUT * INPUT * 3)

    /** Unit embeddings, one row of [DIM] per crop, in the order given. */
    fun embed(crops: List<Mat>): Array<FloatArray> {
        if (crops.isEmpty()) return emptyArray()
        val n = crops.size
        val plane = INPUT * INPUT
        val buffer = bufferFor(n)
        buffer.clear()
        for (crop in crops) {
            val sized = prepare(crop)
            sized.get(0, 0, interleaved)  // BGR, interleaved, row after row
            // The graph wants planes, not interleaved pixels: (n, 3, h, w).
            for (p in 0 until plane) {
                planar[p] = interleaved[p * 3]
                planar[plane + p] = interleaved[p * 3 + 1]
                planar[2 * plane + p] = interleaved[p * 3 + 2]
            }
            buffer.put(planar)  // bulk, straight into native memory
            if (sized !== crop) sized.release()
        }
        buffer.rewind()
        val shape = longArrayOf(n.toLong(), 3, INPUT.toLong(), INPUT.toLong())
        OnnxTensor.createTensor(env, buffer, shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                @Suppress("UNCHECKED_CAST")
                return result[0].value as Array<FloatArray>
            }
        }
    }

    private fun bufferFor(cells: Int): FloatBuffer {
        val held = direct
        if (held != null && directCells == cells) return held
        val floats = cells * 3 * INPUT * INPUT
        val fresh = ByteBuffer.allocateDirect(floats * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        direct = fresh
        directCells = cells
        return fresh
    }

    /** A cell crop at the net's input size, as float BGR. */
    private fun prepare(crop: Mat): Mat {
        val resized: Mat
        if (crop.cols() == INPUT && crop.rows() == INPUT) {
            resized = crop
        } else {
            resized = Mat()
            val growing = crop.cols() < INPUT
            Imgproc.resize(
                crop,
                resized,
                Size(INPUT.toDouble(), INPUT.toDouble()),
                0.0,
                0.0,
                if (growing) Imgproc.INTER_CUBIC else Imgproc.INTER_AREA,
            )
        }
        val floats = Mat()
        resized.convertTo(floats, CvType.CV_32FC3)
        if (resized !== crop) resized.release()
        return floats
    }

    override fun close() {
        runCatching { session.close() }
        direct = null
    }

    companion object {
        /**
         * The cell crop is scaled to this before the net: it was trained at 224, and the
         * read is best there (the desktop measured 0.22% wrong at 224, 0.43% at 128 and
         * 160, 2% at 96). The graph's spatial input is fixed, so a smaller one means a
         * fresh export.
         */
        const val INPUT = 224
        const val DIM = 1280

        /** An exemplar this similar names the cell; below it, nothing on the start position
         * looks like what is there. */
        const val MIN_SIMILARITY = 0.55f

        /** The lead the nearest exemplar's kind needs over the nearest of another kind. */
        const val MIN_MARGIN = 0.03f

        private const val ASSET = "yolo26n-embed.onnx"

        fun load(context: Context, threads: Int = 2): CellEmbedder {
            val bytes = context.assets.open(ASSET).use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(threads)
            val session = env.createSession(bytes, options)
            return CellEmbedder(env, session, session.inputNames.first())
        }
    }
}
