package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import com.sidecoach.Lane
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

/**
 * A learned set on disk from any version the phone has written since 0.1.18 loads as today's
 * model (`ThemeCache.OLDEST_VERSION`): 0.1.19 took the file from version 4 to 6 in one
 * release, every set on Tai's phone was ignored, and a game joined mid-way said it had not
 * learned the pieces (2026-09-27).
 *
 * `assets/vision/themes/theme-v<N>.bin` is one set, learned from the boards fixture
 * `start-green` with OpenCV's generator seeded, and written by the writer of version N itself
 * (`tools/ThemeFilesMaker.kt`, run in a checkout of that version: 2857f14 for 4, 094a093 for 5,
 * bc23731 for 6); beside it, what that version made of it. Each must load, be the same set as
 * today's model learned from the same frame -- its landing crop of a kind, where it kept one per
 * kind, standing for both square colours -- read `italian-green` as that version and today's
 * model read it, and be written again at today's version.
 */
@Lane(Lane.Kind.VISION)
@RunWith(AndroidJUnit4::class)
class ThemeMigrationTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val context = instrumentation.targetContext
    private val symbols = listOf("P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")

    @Test
    fun everyVersionFromTheOldestLoadsAsTodaysModelAndReadsTheBoardAsItDid() {
        val fixtures = Fixtures.all().associateBy { it.name }
        val start = fixtures.getValue("start-green")
        val italian = fixtures.getValue("italian-green")
        val geometry = BoardGeometry(Region(start.x, start.y, start.side, start.side), start.cell.toDouble())
        CellEmbedder.load(context).use { embedder ->
            // Today's model of the same set: the same frame, the same seed.
            val startFrame = Fixtures.mat(start)
            Core.setRNGSeed(SEED)
            val today = ThemeModel.fromStartPosition(startFrame, geometry, true, embedder)!!
            startFrame.release()
            val board = Fixtures.mat(italian)
            try {
                val todayReads = today.classifyFrame(board, geometry, embedder).map { it.symbol ?: "" }
                assertEquals("today's model reads the board", italian.symbols, todayReads)
                for (version in ThemeCache.OLDEST_VERSION..ThemeCache.CACHE_VERSION) {
                    val bytes = themeFile(version)
                    val made = JSONObject(asset("vision/themes/theme-v$version.json").decodeToString())
                    assertEquals("the file is of version $version", version, ByteBuffer.wrap(bytes).int)
                    val directory = File(context.cacheDir, "theme-migration-$version").apply { deleteRecursively(); mkdirs() }
                    File(directory, "theme-${made.getString("key")}.bin").writeBytes(bytes)

                    // Found by its palette, as a board read cold finds it, and loaded.
                    val found = ThemeStore(directory).candidates(today.lightBg, today.darkBg)
                    assertEquals("version $version: the set is found and loads", 1, found.size)
                    val model = found.single().second
                    sameSet("version $version", today, model)

                    // Every kind on both square colours. Where the version kept one crop a kind,
                    // it stands for both, and it is today's crop on the colour it was taken from.
                    assertEquals("version $version: the crops' keys", symbols.flatMap { listOf(it to true, it to false) }.toSet(), model.landingCrops.keys)
                    val crops = made.getJSONObject("crops")
                    for (symbol in symbols) {
                        val light = model.landingCrops.getValue(symbol to true)
                        val dark = model.landingCrops.getValue(symbol to false)
                        if (version >= 6) {
                            assertEquals(crops.getLong("($symbol, true)"), sum(light))
                            assertEquals(crops.getLong("($symbol, false)"), sum(dark))
                        } else {
                            assertArrayEquals("version $version: $symbol's crop on both colours", light.pixels, dark.pixels)
                            assertEquals("version $version: $symbol's crop as written", crops.getLong(symbol), sum(light))
                            val theirs = listOf(true, false).map { today.landingCrops.getValue(symbol to it).pixels }
                            assertTrue(
                                "version $version: $symbol's crop is today's on one colour",
                                theirs.any { it.contentEquals(light.pixels) },
                            )
                        }
                    }
                    assertEquals(made.getDouble("landing_px"), model.landingPx, 0.0)

                    // The board read as the version that wrote it read it, and as today's model does.
                    val reads = model.classifyFrame(board, geometry, embedder).map { it.symbol ?: "" }
                    val theirs = made.getJSONArray("reads").let { a -> (0 until a.length()).map { a.getString(it) } }
                    assertEquals("version $version: the board as that version read it", theirs, reads)
                    assertEquals("version $version: the board as today's model reads it", todayReads, reads)

                    // Written again, it is today's version, and reads back as it was.
                    val again = File(context.cacheDir, "theme-migration-again-$version").apply { deleteRecursively(); mkdirs() }
                    val key = ThemeStore(again).save(model)!!
                    val written = File(again, "theme-$key.bin").readBytes()
                    assertEquals("written at today's version", ThemeCache.CACHE_VERSION, ByteBuffer.wrap(written).int)
                    val back = ThemeStore(again).candidates(model.lightBg, model.darkBg).single().second
                    sameSet("version $version written again", model, back)
                    assertEquals(model.landingCrops.keys, back.landingCrops.keys)
                    for ((kind, crop) in model.landingCrops) {
                        assertArrayEquals(crop.pixels, back.landingCrops.getValue(kind).pixels)
                    }
                    directory.deleteRecursively()
                    again.deleteRecursively()
                }
            } finally {
                board.release()
            }
        }
    }

    @Test
    fun aFileOlderThanTheOldestNewerThanTodayOrCutShortIsRelearned() {
        val bytes = themeFile(ThemeCache.OLDEST_VERSION)
        val made = JSONObject(asset("vision/themes/theme-v${ThemeCache.OLDEST_VERSION}.json").decodeToString())
        val light = made.getJSONArray("light_bg").let { a -> DoubleArray(3) { a.getDouble(it) } }
        val dark = made.getJSONArray("dark_bg").let { a -> DoubleArray(3) { a.getDouble(it) } }
        val directory = File(context.cacheDir, "theme-migration-refused").apply { deleteRecursively(); mkdirs() }
        val file = File(directory, "theme-${made.getString("key")}.bin")
        for (version in listOf(1, 2, 3, ThemeCache.CACHE_VERSION + 1)) {
            val stamped = bytes.copyOf()
            ByteBuffer.wrap(stamped).putInt(version)
            file.writeBytes(stamped)
            assertTrue("a version $version file is not read", ThemeStore(directory).candidates(light, dark).isEmpty())
        }
        for (version in ThemeCache.OLDEST_VERSION..ThemeCache.CACHE_VERSION) {
            val whole = themeFile(version)
            file.writeBytes(whole.copyOf(whole.size / 2))
            assertTrue("a version $version file cut short is not read", ThemeStore(directory).candidates(light, dark).isEmpty())
        }
        directory.deleteRecursively()
    }

    /**
     * The guard: the reader takes today's version and every one back to 4, the version 0.1.18
     * wrote, and there is a file of each, written by its own writer. A version raised without
     * reading the one before it fails the test above on that file; one raised without a file
     * written at the version it retires fails here.
     */
    @Test
    fun theReaderKeepsEveryVersionBackToTheOneOf0118() {
        assertEquals("sets learned by 0.1.18 (version 4) are kept", 4, ThemeCache.OLDEST_VERSION)
        for (version in 4..ThemeCache.CACHE_VERSION) {
            assertTrue("version $version is read", ThemeCache.reads(version))
            themeFile(version)
        }
        assertFalse(ThemeCache.reads(3))
        assertFalse(ThemeCache.reads(ThemeCache.CACHE_VERSION + 1))
    }

    /** The two models are the same learned set: everything but the landing crops. */
    private fun sameSet(what: String, want: ThemeModel, got: ThemeModel) {
        assertArrayEquals("$what: the light squares", want.lightBg, got.lightBg, 0.0)
        assertArrayEquals("$what: the dark squares", want.darkBg, got.darkBg, 0.0)
        assertEquals("$what: the orientation", want.whiteAtBottom, got.whiteAtBottom)
        assertEquals("$what: the exemplars", want.exemplars.map { it.symbol }, got.exemplars.map { it.symbol })
        for ((a, b) in want.exemplars.zip(got.exemplars)) {
            assertArrayEquals("$what: ${a.symbol}'s tone", a.tone, b.tone)
            assertEquals("$what: ${a.symbol}'s brightness", a.brightness, b.brightness, 0.0)
        }
        assertEquals("$what: the sides' colours", want.pieceColours.size, got.pieceColours.size)
        for ((a, b) in want.pieceColours.zip(got.pieceColours)) {
            assertArrayEquals("$what: a fill", a.first, b.first, 0.0)
            assertEquals("$what: an outline", a.second?.toList(), b.second?.toList())
        }
        assertEquals("$what: the palette", want.palette.map { it.toList() }, got.palette.map { it.toList() })
        assertEquals("$what: the start cells", want.exemplarLabels, got.exemplarLabels)
        assertEquals("$what: the embeddings", want.exemplarVectors.size, got.exemplarVectors.size)
        for (i in want.exemplarVectors.indices) {
            assertArrayEquals("$what: embedding $i", want.exemplarVectors[i], got.exemplarVectors[i], 0f)
        }
        assertEquals("$what: the rank labels", want.rankLabels.keys, got.rankLabels.keys)
        for ((rank, glyph) in want.rankLabels) assertArrayEquals(glyph, got.rankLabels.getValue(rank))
    }

    /** The theme file of [version]; there must be one for every version read. */
    private fun themeFile(version: Int): ByteArray = try {
        asset("vision/themes/theme-v$version.bin")
    } catch (e: IOException) {
        throw AssertionError(
            "no theme file of version $version: the reader loads every version back to " +
                "ThemeCache.OLDEST_VERSION, so a version retired by a new one needs its file, written " +
                "by its own writer (tools/ThemeFilesMaker.kt), in assets/vision/themes/",
        )
    }

    private fun asset(path: String): ByteArray = instrumentation.context.assets.open(path).use { it.readBytes() }

    private fun sum(crop: LandingCrop): Long = crop.pixels.sumOf { (it.toInt() and 0xFF).toLong() }

    companion object {
        /** The seed `tools/ThemeFilesMaker.kt` learned the set with. */
        private const val SEED = 20260927

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
