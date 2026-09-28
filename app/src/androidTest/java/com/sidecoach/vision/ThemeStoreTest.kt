package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * A learned set on disk: its landing crops, its palette and its start cells' embeddings (at
 * half precision, as the desktop keeps them) travel with it and read back as they were; a
 * file older than the oldest version read -- the phone's first format, with no landing crops, or
 * the ones before the palette was the desktop's and the embeddings were of the crop at one
 * resolution -- is ignored and the set learned again. The versions still read, 4 on, are
 * `ThemeMigrationTest`'s.
 */
@RunWith(AndroidJUnit4::class)
class ThemeStoreTest {

    @Test
    fun aLearnedSetReadsBackAsItWasAndAnOlderFileIsRelearned() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = Fixtures.all().associateBy { it.name }
        val start = fixtures.getValue("start-green")
        val italian = fixtures.getValue("italian-green")
        val geometry = BoardGeometry(Region(start.x, start.y, start.side, start.side), start.cell.toDouble())
        val directory = File(context.cacheDir, "themes-store-test").apply { deleteRecursively(); mkdirs() }
        CellEmbedder.load(context).use { embedder ->
            val startFrame = Fixtures.mat(start)
            val model = ThemeModel.fromStartPosition(startFrame, geometry, true, embedder)
            startFrame.release()
            assertNotNull("no model from the start frame", model)
            model!!
            // A crop per kind and square colour; a king and a queen start on one colour, and
            // that crop stands for both.
            val symbols = listOf("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
            assertEquals(symbols.flatMap { listOf(it to true, it to false) }.toSet(), model.landingCrops.keys)
            for (symbol in listOf("Q", "K", "q", "k")) {
                assertTrue(model.landingCrops.getValue(symbol to true) === model.landingCrops.getValue(symbol to false))
            }
            for (vector in model.exemplarVectors) {
                for (v in vector) assertEquals("the vectors are held at half precision", halfPrecision(v), v, 0f)
            }

            // Written and read back: the crops pixel for pixel, the palette and the vectors exactly.
            val store = ThemeStore(directory)
            val key = store.save(model)!!
            val back = store.candidates(model.lightBg, model.darkBg).single().second
            assertEquals(model.landingCrops.keys, back.landingCrops.keys)
            for ((kind, crop) in model.landingCrops) {
                val read = back.landingCrops.getValue(kind)
                assertEquals(crop.rows, read.rows)
                assertArrayEquals(crop.pixels, read.pixels)
            }
            assertEquals(model.palette.size, back.palette.size)
            for (i in model.palette.indices) assertArrayEquals(model.palette[i], back.palette[i], 0.0)
            assertEquals(model.exemplarVectors.size, back.exemplarVectors.size)
            for (i in model.exemplarVectors.indices) {
                assertArrayEquals(model.exemplarVectors[i], back.exemplarVectors[i], 0f)
            }
            val frame = Fixtures.mat(italian)
            val readings = back.classifyFrame(frame, geometry, embedder)
            frame.release()
            assertEquals(italian.symbols, readings.map { it.symbol ?: "" })

            // The same file stamped with an older version: not read, so the set is learned again.
            val file = File(directory, "theme-$key.bin")
            val bytes = file.readBytes()
            for (older in listOf(1, 2, 3)) {
                bytes[0] = 0; bytes[1] = 0; bytes[2] = 0; bytes[3] = older.toByte()
                file.writeBytes(bytes)
                assertTrue(
                    "a version $older file is ignored",
                    ThemeStore(directory).candidates(model.lightBg, model.darkBg).isEmpty(),
                )
            }
        }
        directory.deleteRecursively()
    }

    companion object {
        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
