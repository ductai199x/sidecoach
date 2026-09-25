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
 * A learned set on disk: its landing crops travel with it (the desktop's cache version 10), and
 * a set the phone learned before it kept them -- the version 1 file -- still reads, its pieces
 * landing by the centroid until the set is learned again from a start position.
 */
@RunWith(AndroidJUnit4::class)
class ThemeStoreTest {

    @Test
    fun theLandingCropsTravelWithTheSetAndAnOlderSetStillReads() {
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

            // Written and read back: the crops, pixel for pixel.
            val store = ThemeStore(directory)
            val key = store.save(model)!!
            val back = store.candidates(model.lightBg, model.darkBg).single().second
            assertEquals(model.landingCrops.keys, back.landingCrops.keys)
            for ((symbol, crop) in model.landingCrops) {
                val read = back.landingCrops.getValue(symbol)
                assertEquals(crop.rows, read.rows)
                assertArrayEquals(crop.pixels, read.pixels)
            }

            // The same set as the phone wrote it before the crops: version 1, nothing after the
            // rank labels.
            val bare = ThemeModel(
                model.exemplars, model.lightBg, model.darkBg, model.pieceColours, model.palette,
                model.exemplarVectors, model.exemplarLabels, model.whiteAtBottom,
            ).apply { rankLabels = model.rankLabels }
            store.save(bare, key)
            val file = File(directory, "theme-$key.bin")
            val bytes = file.readBytes()
            val old = bytes.copyOfRange(0, bytes.size - 4)  // the crops' count, zero
            old[0] = 0; old[1] = 0; old[2] = 0; old[3] = ThemeCache.CACHE_VERSION_NO_CROPS.toByte()
            file.writeBytes(old)
            val older = ThemeStore(directory).candidates(model.lightBg, model.darkBg).single().second
            assertTrue("a version 1 set has no crops", older.landingCrops.isEmpty())
            val frame = Fixtures.mat(italian)
            val readings = older.classifyFrame(frame, geometry, embedder)
            frame.release()
            assertEquals(italian.symbols, readings.map { it.symbol ?: "" })
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
