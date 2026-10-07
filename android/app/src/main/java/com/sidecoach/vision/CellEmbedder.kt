package com.sidecoach.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.opencv.core.Core
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
    private val planeFloats = FloatArray(INPUT * INPUT)
    private var keyBytes = ByteArray(0)
    private val planes = ArrayList<Mat>(3)

    /**
     * Where the net runs, in the runtime's words: a CPU session, chosen on purpose
     * (`voice.runtime.providers_for` says "cpu (as configured)" for the same choice).
     */
    val deviceDetail: String = "cpu (as configured)"

    /**
     * One start-position-sized batch through the net, so the first real one does not pay
     * the runtime's warm-up; `embedding.CellEmbedder.prime`. Returns the seconds it took.
     */
    fun prime(): Double {
        val begun = System.nanoTime()
        val blank = Mat.zeros(INPUT, INPUT, CvType.CV_8UC3)
        try {
            embed(List(64) { blank })
        } finally {
            blank.release()
        }
        return (System.nanoTime() - begun) / 1e9
    }

    /**
     * Embeddings by the crop's bytes (`CellEmbedder.embed`, desktop a3479bd): a piece that
     * lands on a square was seen there, or on another square of its colour, a moment ago --
     * the same pixels, the same vector, and no run of the net for it. The newest [CACHED] are
     * kept. A vector is never changed once made, so one serves every crop with those pixels.
     */
    private val seen = object : LinkedHashMap<Digest, FloatArray>(CACHED, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<Digest, FloatArray>?): Boolean =
            size > CACHED
    }

    /**
     * Unit embeddings, one row of [DIM] per crop, in the order given; a crop seen before is
     * not run through the net again.
     */
    fun embed(crops: List<Mat>): Array<FloatArray> {
        if (crops.isEmpty()) return emptyArray()
        // The crops are 8-bit BGR, as the cells are cut; anything else is run, not kept. Each is
        // read into one buffer for its key: a new array a crop was a large object each.
        val keys = crops.map { crop ->
            if (crop.depth() != CvType.CV_8U) return@map null
            val size = (crop.total() * crop.elemSize()).toInt()
            if (keyBytes.size != size) keyBytes = ByteArray(size)
            crop.get(0, 0, keyBytes)
            Digest.of(keyBytes) { it.longs(crop.rows().toLong(), crop.cols().toLong(), crop.type().toLong()) }
        }
        val out = arrayOfNulls<FloatArray>(crops.size)
        val todo = ArrayList<Int>()
        synchronized(seen) {
            for ((i, key) in keys.withIndex()) {
                val hit = key?.let { seen[it] }
                if (hit == null) todo.add(i) else out[i] = hit
            }
        }
        if (todo.isNotEmpty()) {
            val fresh = run(todo.map { crops[it] })
            synchronized(seen) {
                for ((j, i) in todo.withIndex()) {
                    out[i] = fresh[j]
                    keys[i]?.let { seen[it] = fresh[j] }
                }
            }
        }
        return Array(crops.size) { out[it]!! }
    }

    /** The crops through the net, in one batch. */
    private fun run(crops: List<Mat>): Array<FloatArray> {
        val n = crops.size
        val buffer = bufferFor(n)
        for (crop in crops) {
            val sized = prepare(crop)
            // The graph wants planes, not interleaved pixels: (n, 3, h, w), B then G then R.
            // OpenCV splits them natively; a loop here over every float of the board was a
            // good part of a full read's time on the phone.
            planes.clear()
            Core.split(sized, planes)
            for (channel in planes) {
                channel.get(0, 0, planeFloats)
                buffer.put(planeFloats)  // bulk, straight into native memory
                channel.release()
            }
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

    /**
     * The input buffer for [cells] crops, cleared and limited to them. With the cache above a
     * batch is only the cells not seen before, and its size changes from one frame to the next:
     * a buffer up to twice the size asked for serves, so a frame of one or two new cells does not
     * make a new one. A larger one is let go -- on Android a direct buffer lives on the Java heap,
     * and the start position's 64 cells are 38 MB of it.
     */
    private fun bufferFor(cells: Int): FloatBuffer {
        val floats = cells * 3 * INPUT * INPUT
        val held = direct
        if (held != null && directCells >= cells && directCells <= maxOf(2 * cells, 4)) {
            held.clear()
            held.limit(floats)
            return held
        }
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

        /**
         * Embeddings kept by the crop's content, 5 kB each: a few boards' worth of pieces. The
         * desktop keeps 2048; the phone's heap is a fifth of the desktop's memory, so fewer.
         */
        const val CACHED = 512

        /** An exemplar this similar names the cell; below it, nothing on the start position
         * looks like what is there. */
        const val MIN_SIMILARITY = 0.55f

        /** The lead the nearest exemplar's kind needs over the nearest of another kind. */
        const val MIN_MARGIN = 0.03f

        private const val ASSET = "yolo26n-embed.onnx"

        /**
         * Four threads: the net measured on the Pixel 10 Pro XL (2026-09-23) read a full board
         * of 64 cells in 488 ms on two and 298 on four; six were no steadier, XNNPACK slower,
         * and NNAPI no faster than the CPU. Stockfish keeps its own two.
         */
        fun load(context: Context, threads: Int = 4): CellEmbedder {
            val bytes = context.assets.open(ASSET).use { it.readBytes() }
            val env = OrtEnvironment.getEnvironment()
            val options = OrtSession.SessionOptions()
            options.setIntraOpNumThreads(threads)
            val session = env.createSession(bytes, options)
            return CellEmbedder(env, session, session.inputNames.first())
        }
    }
}
