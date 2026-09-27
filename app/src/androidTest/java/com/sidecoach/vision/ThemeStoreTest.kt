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
 * file of another version -- the phone's first format, with no landing crops, or the one
 * before the palette was the desktop's -- is ignored and the set learned again, as the
 * desktop's `theme_cache._read` ignores one.
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
            assertEquals(setOf("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"), model.landingCrops.keys)
            for (vector in model.exemplarVectors) {
                for (v in vector) assertEquals("the vectors are held at half precision", halfPrecision(v), v, 0f)
            }

            // Written and read back: the crops pixel for pixel, the palette and the vectors exactly.
            val store = ThemeStore(directory)
            val key = store.save(model)!!
            val back = store.candidates(model.lightBg, model.darkBg).single().second
            assertEquals(model.landingCrops.keys, back.landingCrops.keys)
            for ((symbol, crop) in model.landingCrops) {
                val read = back.landingCrops.getValue(symbol)
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
            for (older in listOf(1, 2)) {
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
