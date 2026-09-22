package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** Where the time goes when a board is embedded: the crops, the scaling, the shuffle into
 * planes, and the net itself. A read that costs seconds is not a coach. */
@RunWith(AndroidJUnit4::class)
class EmbedStagesTest {

    @Test
    fun theStagesOfAnEmbedAreTimed() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixture = Fixtures.all().first { it.name == "italian-green" }
        val frame = Fixtures.mat(fixture)
        val geometry = BoardGeometry(Region(fixture.x, fixture.y, fixture.side, fixture.side), fixture.cell.toDouble())

        var t = System.nanoTime()
        val crops = ArrayList<Mat>(64)
        for (r in 0 until 8) for (c in 0 until 8) crops.add(geometry.crop(frame, r, c, 2 * Squares.CELL))
        val cropMs = (System.nanoTime() - t) / 1_000_000

        t = System.nanoTime()
        val scaled = crops.map { crop ->
            val out = Mat()
            Imgproc.resize(crop, out, Size(224.0, 224.0), 0.0, 0.0, Imgproc.INTER_CUBIC)
            val floats = Mat()
            out.convertTo(floats, CvType.CV_32FC3)
            out.release()
            floats
        }
        val scaleMs = (System.nanoTime() - t) / 1_000_000

        t = System.nanoTime()
        val plane = 224 * 224
        val pixels = FloatArray(plane * 3)
        val buffer = FloatArray(64 * 3 * plane)
        for ((i, mat) in scaled.withIndex()) {
            mat.get(0, 0, pixels)
            val base = i * 3 * plane
            for (p in 0 until plane) {
                buffer[base + p] = pixels[p * 3]
                buffer[base + plane + p] = pixels[p * 3 + 1]
                buffer[base + 2 * plane + p] = pixels[p * 3 + 2]
            }
        }
        val shuffleMs = (System.nanoTime() - t) / 1_000_000

        CellEmbedder.load(context).use { embedder ->
            t = System.nanoTime()
            embedder.embed(crops)
            val firstMs = (System.nanoTime() - t) / 1_000_000
            t = System.nanoTime()
            embedder.embed(crops)
            val secondMs = (System.nanoTime() - t) / 1_000_000
            Log.i(
                "sidecoach.bench",
                "64 cells: crops $cropMs ms, scale+convert $scaleMs ms, shuffle $shuffleMs ms, " +
                    "whole embed first $firstMs ms, again $secondMs ms",
            )
        }
        crops.forEach { it.release() }
        scaled.forEach { it.release() }
        frame.release()
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue(OpenCVLoader.initLocal())
        }
    }
}
