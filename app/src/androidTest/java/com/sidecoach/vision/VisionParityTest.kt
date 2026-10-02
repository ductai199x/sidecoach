package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.InProcess
import com.sidecoach.Lane
import com.sidecoach.Parallel
import com.sidecoach.chess.Board
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import kotlin.math.abs

/**
 * The rest of the vision held to the desktop's answers, one table per part
 * (`tools/make_vision_fixtures.py`): the change gate on runs of frames and the margin it puts
 * round a board; the result badges on the live games' crops and on rendered boards; the board
 * detector and the tracker on small pages -- a board in place, nudged, slid, resized, under the
 * dialog, flush with the frame's corner at a half-pixel pitch, beside another board, a larger
 * checkerboard, two themes that are hard to read, nothing at all -- and the board moved as
 * desktop 17c45a6's scenario test moves it: the cells readable at the old lock and the new
 * place, and the tracker's probe finding it again.
 */
@Lane(Lane.Kind.FREE)
@InProcess(3)
@RunWith(AndroidJUnit4::class)
class VisionParityTest {

    @Test
    fun theGateCountsChangesAndPadsTheBoardAsTheDesktopDoes() {
        val gate = index.getJSONObject("gate")
        val size = gate.getJSONArray("frame")
        val width = size.getInt(0)
        val height = size.getInt(1)
        val failures = ArrayList<String>()
        val scenarios = gate.getJSONArray("scenarios")
        for (s in 0 until scenarios.length()) {
            val scenario = scenarios.getJSONObject(s)
            val keyframe = if (scenario.isNull("keyframe")) null else scenario.getDouble("keyframe")
            val one = ChangeGate(keyframeIntervalS = keyframe)
            one.region = if (scenario.isNull("region")) null else region(scenario.getJSONArray("region"))
            val steps = scenario.getJSONArray("steps")
            val want = scenario.getJSONArray("changed")
            val got = (0 until steps.length()).map { i ->
                val step = steps.getJSONArray(i)
                one.changed(gateFrame(width, height, step.getJSONArray(1)), step.getDouble(0))
            }
            val theirs = (0 until want.length()).map { want.getBoolean(it) }
            if (got != theirs) failures.add("${scenario.getString("what")}: phone $got, desktop $theirs")
        }
        val boards = gate.getJSONArray("boards")
        for (b in 0 until boards.length()) {
            val case = boards.getJSONObject(b)
            val frame = case.getJSONArray("frame")
            val board = BoardCandidate(region(case.getJSONArray("region")), case.getDouble("cell"), 1.0, 50.0)
            val got = ChangeGate.boardGate(board, frame.getInt(0), frame.getInt(1))
            val want = if (case.isNull("gate")) null else region(case.getJSONArray("gate"))
            if (got != want) failures.add("board gate for $board: phone $got, desktop $want")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun theResultBadgesReadAsTheDesktopReadsThem() {
        val badges = index.getJSONObject("badges")
        val failures = ArrayList<String>()
        val crops = badges.getJSONArray("crops")
        for (c in 0 until crops.length()) {
            val crop = crops.getJSONObject(c)
            val mat = Fixtures.asset("vision/${crop.getString("file")}")
            val frame = FramePixels.of(mat)
            mat.release()
            val corner = crop.getJSONArray("corner")
            val cx = corner.getDouble(0)
            val cy = corner.getDouble(1)
            val cell = crop.getDouble("cell")
            val name = crop.getString("file")
            val background = ResultBadges.squareColour(frame, cx, cy, cell)
            near("$name background", background, crop.getJSONArray("background"), failures)
            val disc = ResultBadges.findDisc(frame, cx, cy, cell, background)
            val wantDisc = if (crop.isNull("disc")) null else crop.getJSONArray("disc")
            if ((disc == null) != (wantDisc == null)) {
                failures.add("$name disc: phone ${disc?.let { it.first.toList() }}, desktop $wantDisc")
            } else if (disc != null && wantDisc != null) {
                near("$name disc colour", disc.first, wantDisc.getJSONArray(0), failures)
                if (abs(disc.second - wantDisc.getDouble(1)) > EXACT || abs(disc.third - wantDisc.getDouble(2)) > EXACT) {
                    failures.add("$name disc shares: phone ${disc.second}, ${disc.third}, desktop $wantDisc")
                }
            }
            val badge = ResultBadges.badgeAt(frame, cx, cy, cell)?.name?.lowercase()
            val want = crop.optString("badge").takeUnless { crop.isNull("badge") }
            if (badge != want) failures.add("$name: phone $badge, desktop $want")
        }
        val boards = badges.getJSONArray("boards")
        for (b in 0 until boards.length()) {
            val case = boards.getJSONObject(b)
            val name = case.getString("file")
            val mat = Fixtures.asset("vision/$name")
            val frame = FramePixels.of(mat)
            mat.release()
            val cell = case.getInt("cell")
            val margin = case.getInt("margin")
            val geometry = BoardGeometry(Region(margin, margin, 8 * cell, 8 * cell), cell.toDouble())
            val board = Board(case.getString("fen"))
            val bottom = if (case.getString("bottom") == "white") Pieces.Side.WHITE else Pieces.Side.BLACK
            val kings = case.getJSONObject("kings")
            val read = ArrayList<Badge?>(2)
            for ((colour, key) in listOf(true to "white", false to "black")) {
                val badge = ResultBadges.badgeOn(frame, geometry, board.king(colour)!!, bottom)
                read.add(badge)
                val got = badge?.name?.lowercase()
                val want = kings.optString(key).takeUnless { kings.isNull(key) }
                if (got != want) failures.add("$name $key king: phone $got, desktop $want")
            }
            val result = ResultBadges.resultOf(read[0], read[1])
            val want = case.optString("result").takeUnless { case.isNull("result") }
            if (result != want) failures.add("$name result: phone $result, desktop $want")
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    @Test
    fun boardsAreFoundAndFollowedAsTheDesktopFindsThem() {
        val boards = index.getJSONObject("boards")
        val failures = ArrayList<String>()
        val grays = HashMap<String, Plane>()
        fun gray(file: String): Plane = grays.getOrPut(file) {
            val mat = Fixtures.asset("vision/$file")
            BoardDetect.toGray(mat).also { mat.release() }
        }
        val detections = boards.getJSONArray("detections")
        for (d in 0 until detections.length()) checkDetection(detections.getJSONObject(d), ::gray, failures)
        val runs = boards.getJSONArray("runs")
        for (r in 0 until runs.length()) checkRun(runs.getJSONObject(r), ::gray, failures)
        // The cells a lock reads in its own colours, at the old place and the new
        // (`readable_cells`): what gives a moved board away.
        val readable = boards.getJSONArray("readable")
        for (k in 0 until readable.length()) {
            val case = readable.getJSONObject(k)
            val file = case.getString("file")
            val region = region(case.getJSONArray("region"))
            val c = case.getJSONArray("colours")
            val colours = Colours(c.getDouble(0), c.getDouble(1), c.getDouble(2))
            val got = BoardRefine.readableCells(gray(file), region, case.getDouble("cell"), colours)
            if (got != case.getInt("readable")) {
                failures.add("$file readable at $region: phone $got, desktop ${case.getInt("readable")}")
            }
        }
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * The desktop's tests of its vision work on boards at the frame's edge (desktop 80b1194,
     * 44b112f, 1271cf5, 0c957b5, 2669d9c; `tools/make_vision_fixtures.py`'s `edges`), on their
     * own 1920 by 1200 pages: boards flush with the right or bottom edge found where they are
     * (test_board_detect); a candidate two thirds of a cell off a board near the bottom
     * refused, its fit zero beside the board's (test_board_detect); boards flush with an edge,
     * or a cell from it, followed through a nudge; a board resized to near the bottom locked
     * where it is; a board gone and back at another size found within the full search's first
     * wait; a tracker asking for 0.95 searching on the first frame and the sixth; a wood board
     * near the bottom nudged an eighth of a cell, re-fitted before its colours are read again,
     * and a game board moved past a re-fit's reach with an empty board beside it (desktop
     * dcbbeaf's test_tracker). Every candidate, fit and tracker state as the desktop's. A page is
     * read once per case and let go after it: a 1920 by 1200 grey plane is nine megabytes.
     */
    @Test
    fun boardsAtTheFramesEdgeAreFoundAndFollowedAsTheDesktopFindsThem() {
        val failures = replayEach(index.getJSONObject("edges"))
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * A section's detections and runs, two cases side by side, each reading its own pages and
     * letting them go after it: a 1920 by 1200 grey plane is nine megabytes, and two cases'
     * pages at once fit the heap. Each case its own detector and tracker, as before.
     */
    private fun replayEach(section: JSONObject): List<String> {
        val failures = java.util.Collections.synchronizedList(ArrayList<String>())
        val detections = section.getJSONArray("detections")
        val runs = section.getJSONArray("runs")
        val cases: List<((String) -> Plane) -> Unit> =
            (0 until detections.length()).map { d -> { gray: (String) -> Plane -> checkDetection(detections.getJSONObject(d), gray, failures) } } +
                (0 until runs.length()).map { r -> { gray: (String) -> Plane -> checkRun(runs.getJSONObject(r), gray, failures) } }
        Parallel.forEach(cases, threads = 2) { case ->
            val grays = HashMap<String, Plane>()
            case { file ->
                grays.getOrPut(file) {
                    val mat = Fixtures.asset("vision/$file")
                    BoardDetect.toGray(mat).also { mat.release() }
                }
            }
        }
        return failures
    }

    /**
     * More of the desktop's detector and tracker tests on their own pages (the parity audit,
     * 2026-09-29; `tools/make_vision_fixtures.py`'s `desk`): two boards half a cell or more apart
     * both found and closer ones refused (`test_boards_half_a_cell_or_more_apart_are_both_found`,
     * `test_boards_closer_than_half_a_cell_read_as_one_pattern`), the real chess.com page
     * (`test_a_real_chesscom_page_yields_its_board`), a 37 px cell at (17, 333)
     * (`test_detects_odd_cell_size_and_position`); a re-fit adopted on its boundary against a
     * better fit (`test_refit_wins_on_boundary_strength_not_on_the_fit_score`), an inverted
     * theme's palette kept through a nudge (`test_locked_fit_keeps_polarity_on_an_inverted_theme`),
     * a dialog beside a page edge (`test_tracker_does_not_slide_under_a_dialog_beside_a_page_edge`)
     * and a cover near one square colour over half the board, the lock's colours held
     * (`test_fresh_palette_does_not_relock_from_a_half_covered_board`). A page is let go after
     * its case, as the edge pages are.
     */
    @Test
    fun theDesktopsDetectorAndTrackerTestsGoTheSameWay() {
        val failures = replayEach(index.getJSONObject("desk"))
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** What the detector finds on a page, and the fit at the regions the desktop read. */
    private fun checkDetection(case: JSONObject, gray: (String) -> Plane, failures: MutableList<String>) {
        val file = case.getString("file")
        val found = BoardDetect.detect(gray(file))
        val want = case.getJSONArray("boards")
        val got = found.map { show(it) }
        val theirs = (0 until want.length()).map { show(want.getJSONArray(it)) }
        if (got != theirs) failures.add("$file detected: phone $got, desktop $theirs")
        val fits = case.getJSONArray("fits")
        for (f in 0 until fits.length()) {
            val at = fits.getJSONArray(f)
            val region = Region(at.getInt(0), at.getInt(1), at.getInt(2), at.getInt(3))
            val (fit, contrast) = BoardRefine.fitScore(gray(file), region, at.getDouble(4))
            if (abs(fit - at.getDouble(5)) > EXACT || abs(contrast - at.getDouble(6)) > EXACT) {
                failures.add("$file fit at $region: phone $fit, $contrast, desktop ${at.getDouble(5)}, ${at.getDouble(6)}")
            }
        }
    }

    /**
     * A tracker fed the run's frames: its lock after every update, as the desktop's -- and, in a
     * run that records them (`colours`), the square colours it holds.
     */
    private fun checkRun(run: JSONObject, gray: (String) -> Plane, failures: MutableList<String>) {
        val settings = run.getJSONObject("settings")
        val tracker = BoardTracker(
            minFit = settings.optDouble("min_fit", 0.85),
            maxMisses = settings.optInt("max_misses", 3),
            refineEvery = settings.optInt("refine_every", 10),
        )
        val colours = run.optBoolean("colours")
        val frames = run.getJSONArray("frames")
        val states = run.getJSONArray("states")
        // The change gate's answers (round 3 audit, E10): a still tick hands the tracker the last
        // changed frame's grey image, as the pipeline does.
        val changed = run.optJSONArray("changed")
        var last: Plane? = null
        for (i in 0 until frames.length()) {
            val still = changed != null && !changed.getBoolean(i)
            val plane = last.takeIf { still } ?: gray(frames.getString(i)).also { last = it }
            val board = tracker.update(plane, !still)
            val state = states.getJSONArray(i)
            var got = "locked=${tracker.locked} verified=${tracker.verified} detections=${tracker.detections} " +
                "probes=${tracker.probes} " + (board?.let { show(it) } ?: "none")
            var want = "locked=${state.getBoolean(0)} verified=${state.getBoolean(1)} detections=${state.getInt(2)} " +
                "probes=${state.getInt(4)} " + (if (state.isNull(3)) "none" else show(state.getJSONArray(3)))
            if (colours) {
                got += " colours " + (tracker.colours?.let { "%.6f %.6f %.6f".format(it.light, it.dark, it.tolerance) } ?: "none")
                want += " colours " + if (state.isNull(5)) {
                    "none"
                } else {
                    state.getJSONArray(5).let { "%.6f %.6f %.6f".format(it.getDouble(0), it.getDouble(1), it.getDouble(2)) }
                }
            }
            if (got != want) {
                failures.add("${run.getString("what")}, update ${i + 1} (${frames.getString(i)}): phone $got, desktop $want")
                return
            }
        }
    }

    /** A flat grey frame with each rectangle (x, y, w, h, levels) raised by some levels. */
    private fun gateFrame(width: Int, height: Int, rects: JSONArray): FramePixels {
        val bytes = ByteArray(width * height * 3) { 100.toByte() }
        for (k in 0 until rects.length()) {
            val rect = rects.getJSONArray(k)
            val value = (100 + rect.getInt(4)).coerceIn(0, 255).toByte()
            for (y in rect.getInt(1) until minOf(height, rect.getInt(1) + rect.getInt(3))) {
                for (x in rect.getInt(0) until minOf(width, rect.getInt(0) + rect.getInt(2))) {
                    for (c in 0 until 3) bytes[(y * width + x) * 3 + c] = value
                }
            }
        }
        return FramePixels(bytes, width, height)
    }

    private fun region(a: JSONArray) = Region(a.getInt(0), a.getInt(1), a.getInt(2), a.getInt(3))

    /**
     * A board as both sides print it: the region, the pitch and the fit to six places, the
     * occupied and readable cells.
     */
    private fun show(b: BoardCandidate): String =
        "(${b.region.x},${b.region.y},${b.region.w},${b.region.h}) cell ${"%.6f".format(b.cellPx)} " +
            "fit ${"%.6f".format(b.fit)} contrast ${"%.6f".format(b.contrast)} occupied ${b.occupied} " +
            "readable ${b.readable}"

    private fun show(a: JSONArray): String =
        "(${a.getInt(0)},${a.getInt(1)},${a.getInt(2)},${a.getInt(3)}) cell ${"%.6f".format(a.getDouble(4))} " +
            "fit ${"%.6f".format(a.getDouble(5))} contrast ${"%.6f".format(a.getDouble(6))} occupied ${a.getInt(7)} " +
            "readable ${a.getInt(8)}"

    private fun near(what: String, mine: DoubleArray, theirs: JSONArray, failures: MutableList<String>) {
        for (i in mine.indices) {
            if (abs(mine[i] - theirs.getDouble(i)) > EXACT) {
                failures.add("$what: phone ${mine.toList()}, desktop $theirs")
                return
            }
        }
    }

    companion object {
        /** The same arithmetic: only the JSON's six decimals between them. */
        private const val EXACT = 2e-6

        /** The fixture's index, read once for the class rather than once a test. */
        private val index: JSONObject by lazy {
            val assets = InstrumentationRegistry.getInstrumentation().context.assets
            JSONObject(assets.open("vision/index.json").bufferedReader().use { it.readText() })
        }

        @JvmStatic
        @BeforeClass
        fun loadOpenCv() {
            assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        }
    }
}
