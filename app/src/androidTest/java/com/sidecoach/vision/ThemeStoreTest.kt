package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.TestNets
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import java.io.DataOutputStream
import java.io.File

/**
 * A learned set on disk: its landing crops, its palette and its start cells' embeddings (at
 * half precision, as the desktop keeps them) travel with it and read back as they were; a
 * file older than the oldest version read -- the phone's first format, with no landing crops, or
 * the ones before the palette was the desktop's and the embeddings were of the crop at one
 * resolution -- is ignored and the set learned again. The versions still read, 4 on, are
 * `ThemeMigrationTest`'s.
 */
@Lane(Lane.Kind.VISION)
@InProcess(2)
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
        CellEmbedder.load(context, threads = TestNets.THREADS).use { embedder ->
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
            assertNotNull("the set is stored", store.save(model))
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

            // A shade of rendering noise still matches (the nearest palette within tolerance),
            // and an unseen palette is a miss (`test_theme_cache_round_trips_a_learned_model`;
            // the parity audit, 2026-09-29).
            val shade = store.get(DoubleArray(3) { model.lightBg[it] + 4 }, DoubleArray(3) { model.darkBg[it] - 3 })
            assertNotNull("a palette a few levels off is the same theme", shade)
            assertNull(
                "an unseen palette is a miss",
                store.get(doubleArrayOf(10.0, 10.0, 10.0), doubleArrayOf(250.0, 250.0, 250.0)),
            )

            // A file of an older version is not read, so the set is learned again:
            // ThemeMigrationTest#aFileOlderThanTheOldestNewerThanTodayOrCutShortIsRelearned.
        }
        directory.deleteRecursively()
    }

    @Test
    fun aSaveThatFailsKeepsTheSetItWouldReplaceAndDamagedFilesReadAsNothing() {
        // Round 2, finding 9 (desktop `test_a_save_cut_short_leaves_the_theme_it_replaces`,
        // `test_a_damaged_theme_file_reads_as_nothing`): a theme file was written in place, so a
        // save cut short lost the set it replaced.
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val start = Fixtures.all().first { it.name == "start-green" }
        val geometry = BoardGeometry(Region(start.x, start.y, start.side, start.side), start.cell.toDouble())
        val directory = File(context.cacheDir, "themes-store-damage").apply { deleteRecursively(); mkdirs() }
        CellEmbedder.load(context, threads = TestNets.THREADS).use { embedder ->
            val frame = Fixtures.mat(start)
            val model = checkNotNull(ThemeModel.fromStartPosition(frame, geometry, true, embedder))
            frame.release()
            val store = ThemeStore(directory)
            val key = checkNotNull(store.save(model))
            val saved = File(directory, "theme-$key.bin")
            val bytes = saved.readBytes()
            // The part file cannot be written (a directory stands where it would go): the save
            // fails, says so, and the set it would replace is whole.
            File(directory, "theme-$key.bin.part").mkdirs()
            assertNull("a save that fails says so", store.save(model, key))
            assertArrayEquals(bytes, saved.readBytes())
            assertNotNull(ThemeStore(directory).get(model.lightBg, model.darkBg))

            // Cut short or with a byte flipped anywhere, a file reads as a set or as nothing.
            val damaged = File(context.cacheDir, "themes-store-damaged")
            val random = java.util.Random(1)
            // Made one at a time, the same cases in the same order: the four hundred copies made
            // up front were 150 MB of the test process's 192 MB heap, and ran it out.
            val cases = (1 until bytes.size step maxOf(1, bytes.size / 200)).asSequence().map { bytes.copyOf(it) } +
                (0 until 200).asSequence().map { bytes.copyOf().also { b -> val at = random.nextInt(b.size); b[at] = (b[at].toInt() xor 0xFF).toByte() } }
            for (case in cases) {
                damaged.deleteRecursively()
                damaged.mkdirs()
                File(damaged, "theme-$key.bin").writeBytes(case)
                ThemeStore(damaged).candidates(model.lightBg, model.darkBg)  // must not throw
            }
            damaged.deleteRecursively()
        }
        directory.deleteRecursively()
    }

    /**
     * A length read from a damaged theme file -- a mask's bits, the embeddings' count and size,
     * a landing crop's rows and columns -- asked for more memory than there is, and the store
     * caught the OutOfMemoryError (round 2, finding 9). Each length is held to the bytes left in
     * the file before anything is allocated by it, so a file that says it holds two billion of
     * something reads as nothing, and no Error is ever raised (none is caught any more).
     */
    @Test
    fun aDamagedLengthIsRefusedBeforeItIsAllocated() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val directory = File(context.cacheDir, "themes-store-lengths")
        val light = doubleArrayOf(181.0, 217.0, 240.0)
        val dark = doubleArrayOf(99.0, 136.0, 181.0)
        fun header(out: DataOutputStream) {
            out.writeInt(ThemeCache.CACHE_VERSION)
            for (v in light) out.writeDouble(v)
            for (v in dark) out.writeDouble(v)
            out.writeBoolean(true)
        }
        val cases = mapOf<String, (DataOutputStream) -> Unit>(
            "mask bits" to { out ->
                header(out)
                out.writeInt(1) // one exemplar
                out.writeUTF("P")
                out.writeInt(Int.MAX_VALUE) // its tone's bits
            },
            "embeddings" to { out ->
                header(out)
                repeat(3) { out.writeInt(0) } // no exemplars, piece colours or palette
                out.writeInt(Int.MAX_VALUE) // vectors
                out.writeInt(Int.MAX_VALUE) // numbers each
            },
            "landing crop" to { out ->
                header(out)
                repeat(3) { out.writeInt(0) }
                repeat(2) { out.writeInt(0) } // no vectors
                repeat(2) { out.writeInt(0) } // no labels, no rank labels
                out.writeInt(1) // one crop
                out.writeUTF("P")
                out.writeBoolean(true)
                out.writeInt(1_000_000) // rows
                out.writeInt(1_000_000) // columns
            },
        )
        for ((name, write) in cases) {
            directory.deleteRecursively()
            directory.mkdirs()
            DataOutputStream(File(directory, "theme-damaged-length.bin").outputStream()).use(write)
            assertTrue("$name: read as a set", ThemeStore(directory).candidates(light, dark).isEmpty())
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
