package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.TestNets
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import kotlin.math.abs

/**
 * The recogniser held to the desktop's, over every piece set it knows: `tools/make_cell_fixtures.py`
 * learns each set from its start position on the desktop and records the model -- the square
 * colours, the set's fills and outlines, the k-means palette, the exemplars -- and every field of
 * every cell it reads on a middlegame dressed with a highlight, a check, a drag and arrows (and,
 * since the parity audit of 2026-09-29, the move's badge over its piece, a green or teal tint, a
 * hover outline a pixel thin, and one set on a 641 px board: a fractional pitch), and on
 * a few boards under the game-over dialog with legal-move dots; with them the theme cache's key and
 * sampled colours, and the rank labels and the side they read at the bottom. The phone learns each
 * set from the same frame, its OpenCV generator seeded as the desktop's was for the k-means, and
 * must answer the same.
 *
 * Exact where the arithmetic is the same; within a tolerance where it cannot be: the k-means
 * centres (sums in another order), the net's similarities (another onnxruntime), the phase
 * correlation (another OpenCV).
 */
@Lane(Lane.Kind.VISION)
@InProcess(2)
@RunWith(AndroidJUnit4::class)
class CellsParityTest {

    @Test
    fun everySetIsLearnedAndReadAsTheDesktopDoes() {
        val context = InstrumentationRegistry.getInstrumentation()
        val index = JSONObject(
            context.context.assets.open("cells/index.json").bufferedReader().use { it.readText() },
        )
        val seed = index.getInt("seed")
        val cell = index.getInt("cell")
        val margin = index.getInt("margin")
        val geometry = BoardGeometry(Region(margin, margin, 8 * cell, 8 * cell), cell.toDouble())
        val failures = ArrayList<String>()
        var cellsCompared = 0
        var sets = 0
        var learnMs = 0L
        var readMs = 0L
        CellEmbedder.load(context.targetContext, threads = TestNets.THREADS).use { embedder ->
            val cases = index.getJSONArray("cases")
            for (k in 0 until cases.length()) {
                val case = cases.getJSONObject(k)
                sets++
                // A case on a board of its own size (a fractional pitch, 641 px: the parity
                // audit, 2026-09-29) carries its geometry.
                val region = case.optJSONArray("region")
                val here = if (region == null) {
                    geometry
                } else {
                    BoardGeometry(
                        Region(region.getInt(0), region.getInt(1), region.getInt(2), region.getInt(3)),
                        case.getDouble("cell_px"),
                    )
                }
                val name = "${case.getString("style")}/${case.getString("theme")}" +
                    (if (region == null) "" else " at ${region.getInt(2)} px")
                val start = Fixtures.asset("cells/${case.getString("start")}")
                Core.setRNGSeed(seed)
                var began = System.nanoTime()
                val model = ThemeModel.fromStartPosition(start, here, true, embedder)
                learnMs += (System.nanoTime() - began) / 1_000_000
                start.release()
                if (model == null) {
                    failures.add("$name: no model from the start position")
                    continue
                }
                compareModel(name, model, case.getJSONObject("model"), failures)
                compareCacheAndLabels(name, model, here, case, failures)
                val boards = case.getJSONArray("boards")
                for (b in 0 until boards.length()) {
                    val board = boards.getJSONObject(b)
                    val frame = Fixtures.asset("cells/${board.getString("file")}")
                    began = System.nanoTime()
                    val readings = model.classifyFrame(frame, here, embedder)
                    readMs += (System.nanoTime() - began) / 1_000_000
                    val cells = board.getJSONArray("cells")
                    for (i in 0 until 64) {
                        val sil = model.silAt(i)!!
                        val where = "$name ${board.getString("what")} cell ${i / 8},${i % 8}"
                        compareCell(where, model, readings[i], sil, cells.getJSONArray(i), failures)
                        cellsCompared++
                    }
                    frame.release()
                }
            }
            Log.i(
                TAG,
                "$sets of ${cases.length()} sets, $cellsCompared cells: ${failures.size} differences " +
                    "(learned in $learnMs ms, read in $readMs ms)",
            )
        }
        assertTrue(
            "${failures.size} differences from the desktop:\n" + failures.take(60).joinToString("\n"),
            failures.isEmpty(),
        )
    }

    private fun compareModel(name: String, model: ThemeModel, want: JSONObject, failures: MutableList<String>) {
        close("$name light square", model.lightBg, want.getJSONArray("light_bg"), EXACT, failures)
        close("$name dark square", model.darkBg, want.getJSONArray("dark_bg"), EXACT, failures)
        val colours = want.getJSONArray("piece_colours")
        if (colours.length() != model.pieceColours.size) {
            failures.add("$name: ${model.pieceColours.size} sides' colours, desktop ${colours.length()}")
        } else {
            for (s in 0 until colours.length()) {
                val pair = colours.getJSONArray(s)
                close("$name side $s fill", model.pieceColours[s].first, pair.getJSONArray(0), EXACT, failures)
                val edge = model.pieceColours[s].second
                if (pair.isNull(1) != (edge == null)) {
                    failures.add("$name side $s outline: phone $edge, desktop ${pair.opt(1)}")
                } else if (edge != null) {
                    close("$name side $s outline", edge, pair.getJSONArray(1), EXACT, failures)
                }
            }
        }
        // The palette as a set: every desktop centre has a phone centre by it, and no more.
        val palette = want.getJSONArray("palette")
        val theirs = (0 until palette.length()).map { p -> DoubleArray(3) { palette.getJSONArray(p).getDouble(it) } }
        val unmatched = theirs.filter { t -> model.palette.none { distance(it, t) <= PALETTE } }
        if (unmatched.isNotEmpty() || theirs.size != model.palette.size) {
            failures.add(
                "$name palette: phone ${model.palette.size} colours ${show(model.palette)}, " +
                    "desktop ${theirs.size} ${show(theirs)}",
            )
        }
        val exemplars = want.getJSONObject("exemplars")
        for (symbol in exemplars.keys()) {
            val e = exemplars.getJSONArray(symbol)
            val mine = model.exemplars.firstOrNull { it.symbol == symbol }
            if (mine == null) {
                failures.add("$name: no exemplar $symbol")
                continue
            }
            val cropSums = listOf(true, false).map { light ->
                val crop = model.landingCrops[symbol to light]
                (crop?.pixels?.sumOf { (it.toInt() and 0xFF).toLong() } ?: -1L).toDouble()
            }
            val got = listOf(mine.brightness, mine.tone.count { it }.toDouble()) + cropSums
            for (f in got.indices) {
                if (abs(got[f] - e.getDouble(f)) > EXACT) {
                    failures.add("$name exemplar $symbol ${EXEMPLAR_FIELDS[f]}: phone ${got[f]}, desktop ${e.getDouble(f)}")
                }
            }
        }
    }

    /**
     * The theme cache's key for the learned square colours and the colours it samples from the
     * dressed board, and the rank labels learned from the start position with the side they
     * read at the bottom of the dressed board.
     */
    private fun compareCacheAndLabels(
        name: String,
        model: ThemeModel,
        geometry: BoardGeometry,
        case: JSONObject,
        failures: MutableList<String>,
    ) {
        val key = ThemeCache.paletteKey(model.lightBg, model.darkBg)
        if (key != case.getString("palette_key")) {
            failures.add("$name palette key: phone $key, desktop ${case.getString("palette_key")}")
        }
        val dressed = Fixtures.asset("cells/${case.getString("dressed")}")
        val start = Fixtures.asset("cells/${case.getString("start")}")
        try {
            val sampled = ThemeCache.samplePalette(dressed, geometry)
            if (case.isNull("sampled") != (sampled == null)) {
                failures.add("$name sampled colours: phone ${sampled?.toList()}, desktop ${case.opt("sampled")}")
            } else if (sampled != null) {
                val want = case.getJSONArray("sampled")
                close("$name sampled light", sampled.first, want.getJSONArray(0), EXACT, failures)
                close("$name sampled dark", sampled.second, want.getJSONArray(1), EXACT, failures)
            }
            val labels = Coordinates.learnRankLabels(start, geometry, Pieces.Side.WHITE)
            val learned = labels.entries.sortedBy { it.key }.associate { (rank, glyph) -> rank.toString() to glyph.count { it } }
            val want = case.getJSONObject("rank_labels")
            val theirs = want.keys().asSequence().associateWith { want.getInt(it) }
            if (learned != theirs) failures.add("$name rank labels: phone $learned, desktop $theirs")
            val bottom = Coordinates.readBottomSide(dressed, geometry, labels)?.name?.lowercase()
            val wantBottom = case.optString("bottom_read").takeUnless { case.isNull("bottom_read") }
            if (bottom != wantBottom) failures.add("$name bottom read: phone $bottom, desktop $wantBottom")
        } finally {
            dressed.release()
            start.release()
        }
    }

    private fun compareCell(
        where: String,
        model: ThemeModel,
        reading: Pieces.CellReading,
        sil: Sil,
        want: JSONArray,
        failures: MutableList<String>,
    ) {
        val differs = ArrayList<String>()
        fun same(field: String, mine: Any?, theirs: Any?) {
            if (mine != theirs) differs.add("$field phone $mine desktop $theirs")
        }
        fun near(field: String, mine: Double, theirs: Double, tolerance: Double) {
            if (abs(mine - theirs) > tolerance) differs.add("$field phone $mine desktop $theirs")
        }
        same("kind", reading.kind.name.lowercase(), want.getString(0))
        same("symbol", reading.symbol, want.optString(1).takeUnless { want.isNull(1) })
        same("colour", reading.colour?.name?.lowercase(), want.optString(2).takeUnless { want.isNull(2) })
        near("confidence", reading.confidence, want.getDouble(3), NET)
        near("fg_fraction", reading.fgFraction, want.getDouble(4), EXACT)
        near("luminance", reading.luminance, want.getDouble(5), EXACT)
        same("highlighted", reading.highlighted, want.getBoolean(6))
        val background = want.getJSONArray(7)
        for (ch in 0 until 3) near("background[$ch]", reading.background[ch], background.getDouble(ch), EXACT)
        near("brightness", reading.brightness, want.getDouble(8), EXACT)
        same("settled", reading.settled, want.getBoolean(9))
        same("hover", reading.hover, want.getBoolean(10))
        near("offset", reading.offset, want.getDouble(11), PHASE)
        same("guess", reading.guess, want.optString(12).takeUnless { want.isNull(12) })
        same("covered", reading.covered, want.getBoolean(13))
        same("tint", reading.tint, want.optString(14).takeUnless { want.isNull(14) })
        same("drawn", reading.drawn, want.getBoolean(15))
        near("overlay", sil.overlay, want.getDouble(16), OVERLAY)
        if (!want.isNull(17)) {
            val fill = want.getJSONArray(17)
            val side = model.fillSide(sil, fill.getString(0))
            same("fill side of ${fill.getString(0)}", side?.name?.lowercase(), fill.optString(1).takeUnless { fill.isNull(1) })
        }
        if (differs.isNotEmpty()) failures.add("$where: ${differs.joinToString("; ")}")
    }

    private fun close(what: String, mine: DoubleArray, theirs: JSONArray, tolerance: Double, failures: MutableList<String>) {
        for (i in mine.indices) {
            if (abs(mine[i] - theirs.getDouble(i)) > tolerance) {
                failures.add("$what: phone ${mine.toList()}, desktop $theirs")
                return
            }
        }
    }

    private fun distance(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in 0 until 3) sum += (a[i] - b[i]) * (a[i] - b[i])
        return kotlin.math.sqrt(sum)
    }

    private fun show(colours: List<DoubleArray>): String =
        colours.joinToString(" ", "[", "]") { c -> c.joinToString(",") { "%.1f".format(it) } }

    companion object {
        private const val TAG = "sidecoach.vision"

        /** The same arithmetic on the same pixels: only the JSON's six decimals between them. */
        private const val EXACT = 2e-6

        /** k-means centres, summed in another order on another OpenCV. */
        private const val PALETTE = 0.05

        /** A share of the cell's pixels: one pixel of 2304 is 4e-4. */
        private const val OVERLAY = 1e-6

        /** The net's margins, from another onnxruntime. */
        private const val NET = 0.01

        /** The phase correlation's shift, a share of the cell, on another OpenCV. */
        private const val PHASE = 0.002

        private val EXEMPLAR_FIELDS = listOf("brightness", "tone", "light-square crop sum", "dark-square crop sum")

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
