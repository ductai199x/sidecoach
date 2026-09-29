package com.sidecoach.vision

import android.content.res.AssetManager
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Read the two chess.com clocks drawn next to the board: `sidecoach/vision/clock.py`.
 *
 * What the recordings show (720p copies of a 1440p screen, 43-pixel cells):
 *
 * - Each clock is a rounded box right-aligned with the board's right edge, one about 0.14
 *   cell above the board and one the same distance below it. The box is about 1.56 cells
 *   wide and 0.47 tall. The digits are 0.23 cell tall, right-aligned inside the box with
 *   about 0.12 cell of padding, and a small clock icon sits at the left of the box. The
 *   format is M:SS or MM:SS, with tenths under a minute (0:59.9). No clock in the six
 *   recordings fell under eleven minutes, so the tenths dot is taken from the colon's lower
 *   dot, and a tenths separator matched as a colon is repaired in [parseClock].
 * - White's box is light with dark text and black's box dark with light text, whichever side
 *   is to move. The box colours therefore say which clock is white's, not which one runs.
 * - The running clock's icon animates: its hand sweeps round in bursts, once a second. The
 *   idle icon never changes a pixel. The digits of the running clock change once a second.
 * - The digit font is chess.com's own and the same in every recording. Glyphs are cut out of
 *   the box by column projection, scaled to a fixed line height, and classified by
 *   normalised correlation against templates learned from the recordings -- the desktop's
 *   `clock_templates.npz`, shipped here as the asset `vision/clock_templates.json`.
 *
 * The reader is a pure vision component: it reports the top and bottom clocks as drawn,
 * which box is light, and which clock is running. The pipeline knows which side sits at the
 * bottom and maps top and bottom to colours itself. Nothing about whose turn it is comes
 * from here: the reconciler reads the turn off the board.
 *
 * Geometry is relative to the board's cell size, and every glyph is scaled to a fixed line
 * height before classification, so the reader does not depend on the board or screen size.
 *
 * A bad frame must never take a live session down, so [ClockReader.read] catches everything
 * and returns null.
 */

const val GLYPH_H = 20
const val GLYPH_W = 16

/** How far from the board's edge a clock box may sit, in cells. */
const val SEARCH_CELLS = 1.3

/** A box is a run of near-constant colour along the column just inside its right edge. */
val BOX_HEIGHT_CELLS = 0.15 to 0.9
val BOX_WIDTH_CELLS = 0.6 to 3.5

/**
 * The box interior is flat colour; a dark box on a dark page differs from it by only ten
 * grey levels, so the tolerances stay well under that.
 */
const val ROW_STEP_TOLERANCE = 4.0
const val FILL_TOLERANCE = 4.0
const val MIN_CONTRAST = 60.0
const val INK_ON = 0.35

/** A glyph run wider than this fraction of the line height holds two touching glyphs. */
const val MAX_GLYPH_ASPECT = 0.85

private val CLOCK_RE = Regex("^(?:(\\d+):)?(\\d{1,2}):(\\d\\d)(?:\\.(\\d))?\$")

/** Which of the two boxes: the desktop's `Which` literal. */
enum class Which(val slug: String) {
    TOP("top"),
    BOTTOM("bottom"),
}

data class ClockReading(
    /** What the clock above the board shows, in milliseconds. */
    val topMs: Int?,
    /** What the clock below the board shows. */
    val bottomMs: Int?,
    /** Which one is ticking, if either. */
    val running: Which?,
    /** 0..1, how sure the digits were read. */
    val confidence: Double,
    /** Which box has the light background: white's clock, on every chess.com theme seen. */
    val light: Which? = null,
)

/** Where a clock box was found. Frame coordinates, [right] exclusive. */
data class ClockBox(
    val top: Int,
    val bottom: Int,
    val left: Int,
    val right: Int,
    val background: Double,
    val text: Double,
) {
    val light: Boolean get() = background > text

    val region: Region get() = Region(left, top, right - left, bottom - top)
}

/**
 * The glyphs of one box: column spans inside the ink map, the text line's rows, and the
 * column spans of whatever sits left of the digits (the clock icon).
 */
class Segmentation(
    val glyphs: List<Pair<Int, Int>>,
    val line: Pair<Int, Int>,
    val icon: List<Pair<Int, Int>>,
)

/** `int(round(x))`: Python rounds half to even, and a box edge lands on .5 often enough. */
internal fun pyRound(value: Double): Int = Math.rint(value).toInt()

/** Maximal runs of true in a boolean array as [start, end) spans. */
internal fun runsOf(on: BooleanArray): List<Pair<Int, Int>> {
    val out = ArrayList<Pair<Int, Int>>()
    var start = -1
    for (i in on.indices) {
        if (on[i] && start < 0) start = i
        if (!on[i] && start >= 0) {
            out.add(start to i)
            start = -1
        }
    }
    if (start >= 0) out.add(start to on.size)
    return out
}

/** `np.median`: the middle value, or the mean of the two middles. */
internal fun medianOf(values: DoubleArray, from: Int = 0, to: Int = values.size): Double {
    val n = to - from
    val copy = values.copyOfRange(from, to)
    copy.sort()
    return if (n % 2 == 1) copy[n / 2] else (copy[n / 2 - 1] + copy[n / 2]) / 2.0
}

/** `np.percentile` with the default linear interpolation. */
internal fun percentileOf(values: DoubleArray, q: Double): Double {
    val copy = values.copyOf()
    copy.sort()
    val position = q / 100.0 * (copy.size - 1)
    val lower = kotlin.math.floor(position).toInt()
    val upper = kotlin.math.ceil(position).toInt()
    if (lower == upper) return copy[lower]
    return copy[lower] + (copy[upper] - copy[lower]) * (position - lower)
}

/** Leftmost column, scanning left from [start], still within tolerance of the colour. */
internal fun fillExtent(gray: Plane, row: Int, start: Int, background: Double): Int {
    for (i in start downTo 0) {
        if (abs(gray[row, i] - background) > FILL_TOLERANCE) return i + 1
    }
    return 0
}

/**
 * The clock box whose right edge is at [xRight], between the given rows, if any.
 *
 * The column band just inside the right edge is the box's padding: along it, the box is a
 * run of near-constant colour of plausible height, bounded by the page above and below. The
 * run must extend left far enough to be a box, and must hold high-contrast ink of a
 * plausible amount, the digits. Among candidates the one with the most ink wins.
 */
fun findBox(gray: Plane, xRight: Double, rows: Pair<Int, Int>, cell: Double): ClockBox? {
    val height = gray.height
    val width = gray.width
    val r0 = max(0, rows.first)
    val r1 = min(height, rows.second)
    val c0 = pyRound(xRight - 0.10 * cell)
    val c1 = pyRound(xRight - 0.03 * cell)
    if (c1 - c0 < 1 || c0 < 0 || c1 > width || r1 - r0 < 4) return null
    // The median ignores a rounded corner in one column.
    val profile = DoubleArray(r1 - r0)
    val cut = DoubleArray(c1 - c0)
    for (row in 0 until r1 - r0) {
        for (col in c0 until c1) cut[col - c0] = gray[r0 + row, col].toDouble()
        profile[row] = medianOf(cut)
    }
    // steady[i] says rows i and i + 1 match, so a run of it from s to e - 1 is the row span
    // [s, e + 1) of one flat colour.
    val steady = BooleanArray(profile.size - 1) { abs(profile[it + 1] - profile[it]) <= ROW_STEP_TOLERANCE }
    var bestScore = 0.0
    var best: ClockBox? = null
    for ((s, e) in runsOf(steady)) {
        val a = s
        val b = e + 1
        val span = b - a
        if (span < BOX_HEIGHT_CELLS.first * cell || span > BOX_HEIGHT_CELLS.second * cell) continue
        val background = medianOf(profile, a, b)
        val top = r0 + a
        val bottom = r0 + b
        // The leftmost row reaching furthest left is the box's edge: the icon and the digits
        // stop the scan early on the rows they cross, and no row can run past the edge,
        // where the page is a different colour.
        var left = Int.MAX_VALUE
        for (row in top until bottom) left = min(left, fillExtent(gray, row, c1 - 1, background))
        val boxWidth = c1 - left
        if (boxWidth < BOX_WIDTH_CELLS.first * cell || boxWidth > BOX_WIDTH_CELLS.second * cell) {
            continue
        }
        val patch = DoubleArray((bottom - top) * (c1 - left))
        var at = 0
        for (row in top until bottom) {
            for (col in left until c1) patch[at++] = gray[row, col].toDouble()
        }
        val text = percentileOf(patch, if (background > 127) 1.0 else 99.0)
        if (abs(text - background) < MIN_CONTRAST) continue
        val threshold = 0.5 * abs(text - background)
        var inkCount = 0
        for (value in patch) if (abs(value - background) > threshold) inkCount += 1
        val fraction = inkCount.toDouble() / patch.size
        if (fraction <= 0.01 || fraction >= 0.4) continue
        val score = inkCount.toDouble()
        if (best == null || score > bestScore) {
            bestScore = score
            best = ClockBox(top, bottom, left, pyRound(xRight), background, text)
        }
    }
    return best
}

/** Text intensity inside the box, 0 for background to 1 for the text colour. */
fun inkMap(gray: Plane, box: ClockBox, cell: Double): Plane {
    val right = pyRound(box.right - 0.03 * cell)
    val w = right - box.left
    val h = box.bottom - box.top
    val data = FloatArray(max(0, w) * max(0, h))
    val span = (box.text - box.background).toFloat()
    var at = 0
    for (row in box.top until box.bottom) {
        for (col in box.left until right) {
            val value = (gray[row, col] - box.background.toFloat()) / span
            data[at++] = value.coerceIn(0.0f, 1.0f)
        }
    }
    return Plane(w, h, data)
}

/**
 * Cut the ink map into glyph column spans.
 *
 * The digits are the rightmost group of ink columns; a gap wider than half the box height
 * separates them from the icon. Within the group, the text line's rows are where the ink is,
 * and glyph spans are runs of columns holding at least half a pixel of ink over those rows.
 * Anti-aliasing can close the one-pixel gap between two round digits, so a run wider than a
 * digit is split at its weakest interior column.
 */
fun segment(ink: Plane): Segmentation? {
    val boxH = ink.height
    val on = BooleanArray(ink.width) { col ->
        var most = Float.NEGATIVE_INFINITY
        for (row in 0 until ink.height) most = max(most, ink[row, col])
        most > INK_ON
    }
    val runs = runsOf(on)
    if (runs.isEmpty()) return null
    val group = ArrayList<Pair<Int, Int>>()
    group.add(runs.last())
    for (index in runs.size - 2 downTo 0) {
        val (a, b) = runs[index]
        if (group[0].first - b > 0.45 * boxH) break
        group.add(0, a to b)
    }
    val icon = runs.subList(0, runs.size - group.size).toList()
    val g0 = group.first().first
    val g1 = group.last().second
    var t0 = -1
    var t1 = -1
    for (row in 0 until ink.height) {
        var most = Float.NEGATIVE_INFINITY
        for (col in g0 until g1) most = max(most, ink[row, col])
        if (most > INK_ON) {
            if (t0 < 0) t0 = row
            t1 = row + 1
        }
    }
    if (t0 < 0) return null
    val lineH = t1 - t0
    if (lineH < 4) return null
    val colsum = DoubleArray(ink.width)
    for (col in 0 until ink.width) {
        var total = 0.0
        for (row in t0 until t1) total += ink[row, col]
        colsum[col] = total
    }
    val strong = BooleanArray(ink.width) { it in g0 until g1 && colsum[it] > 0.5 }
    val glyphs = ArrayList<Pair<Int, Int>>()
    val pending = ArrayDeque(runsOf(strong))
    while (pending.isNotEmpty()) {
        val (a, b) = pending.removeFirst()
        if (b - a > MAX_GLYPH_ASPECT * lineH) {
            val lo = a + pyRound(0.3 * lineH)
            val hi = b - pyRound(0.3 * lineH)
            if (hi - lo >= 1) {
                var k = lo
                var least = Double.POSITIVE_INFINITY
                for (i in lo until hi) {
                    if (colsum[i] < least) {
                        least = colsum[i]
                        k = i
                    }
                }
                pending.addFirst(k to b)
                pending.addFirst(a to k)
                continue
            }
        }
        glyphs.add(a to b)
    }
    return Segmentation(glyphs, t0 to t1, icon)
}

/**
 * One glyph scaled to the fixed line height and centred on the canvas.
 *
 * The width scales with the height, so a narrow 1 stays narrow and a colon stays a pair of
 * dots; the rows are the whole text line, so a dot stays at the baseline.
 */
fun glyphCanvas(ink: Plane, line: Pair<Int, Int>, span: Pair<Int, Int>): FloatArray {
    val (t0, t1) = line
    val (a, b) = span
    val scale = GLYPH_H.toDouble() / (t1 - t0)
    val w = min(GLYPH_W, max(1, pyRound((b - a) * scale)))
    val crop = Mat(t1 - t0, b - a, CvType.CV_32F)
    val row = FloatArray(b - a)
    for (y in t0 until t1) {
        for (x in a until b) row[x - a] = ink[y, x]
        crop.put(y - t0, 0, row)
    }
    val resized = Mat()
    val interpolation = if (scale < 1) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR
    Imgproc.resize(crop, resized, Size(w.toDouble(), GLYPH_H.toDouble()), 0.0, 0.0, interpolation)
    val canvas = FloatArray(GLYPH_H * GLYPH_W)
    val off = (GLYPH_W - w) / 2
    val line32 = FloatArray(w)
    for (y in 0 until GLYPH_H) {
        resized.get(y, 0, line32)
        for (x in 0 until w) canvas[y * GLYPH_W + off + x] = line32[x]
    }
    crop.release()
    resized.release()
    return canvas
}

private fun normalised(canvas: FloatArray): FloatArray {
    var mean = 0.0
    for (value in canvas) mean += value
    mean /= canvas.size
    val out = FloatArray(canvas.size) { (canvas[it] - mean).toFloat() }
    var norm = 0.0
    for (value in out) norm += value.toDouble() * value
    val scale = (sqrt(norm) + 1e-6).toFloat()
    for (i in out.indices) out[i] = out[i] / scale
    return out
}

/**
 * Mean glyph canvases per label, several per label when the rendering varies (white text on
 * a dark box is bolder than dark text on a light one).
 */
class GlyphTemplates(
    val labels: List<String>,
    /** (N, GLYPH_H * GLYPH_W) floats in 0..1, row after row. */
    val canvases: List<FloatArray>,
    /** (N,) glyph width as a fraction of the line height. */
    val advances: FloatArray,
) {
    private val unit: List<FloatArray> = canvases.map { normalised(it) }

    /** (label, correlation with the best template, margin over the best other label). */
    fun classify(canvas: FloatArray): Triple<String, Double, Double> {
        val flat = normalised(canvas)
        val scores = DoubleArray(unit.size) { i ->
            var total = 0.0
            val one = unit[i]
            for (k in one.indices) total += one[k].toDouble() * flat[k]
            total
        }
        // `np.argsort(...)[::-1]`: ascending, ties in index order, then reversed.
        val order = scores.indices.sortedWith(
            compareBy<Int> { scores[it] }.thenBy { it }
        ).asReversed()
        val best = order[0]
        val label = labels[best]
        var margin = 1.0
        for (i in order.drop(1)) {
            if (labels[i] != label) {
                margin = scores[best] - scores[i]
                break
            }
        }
        return Triple(label, scores[best], margin)
    }

    companion object {
        const val ASSET = "vision/clock_templates.json"

        /**
         * The templates as the app ships them. The desktop keeps them in a `.npz`; the same
         * numbers are written out as JSON by `tools/make_clock_fixtures.py`.
         */
        fun load(assets: AssetManager, path: String = ASSET): GlyphTemplates? = try {
            val text = assets.open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }
            of(JSONObject(text))
        } catch (_: Exception) {
            null
        }

        fun of(json: JSONObject): GlyphTemplates {
            val labelsJson = json.getJSONArray("labels")
            val labels = (0 until labelsJson.length()).map { labelsJson.getString(it) }
            val canvasesJson = json.getJSONArray("canvases")
            val canvases = (0 until canvasesJson.length()).map { i ->
                val one = canvasesJson.getJSONArray(i)
                FloatArray(one.length()) { (one.getInt(it) / 255.0).toFloat() }
            }
            val advancesJson = json.getJSONArray("advances")
            val advances = FloatArray(advancesJson.length()) {
                advancesJson.getDouble(it).toFloat()
            }
            return GlyphTemplates(labels, canvases, advances)
        }
    }
}

/**
 * Milliseconds shown by a clock string such as 15:00, 1:02:03 or 0:59.9; null if it is not
 * a clock.
 *
 * A tenths dot read as a colon is repaired: seconds always have two digits, so a lone digit
 * after the last separator is tenths whichever glyph was matched.
 */
fun parseClock(raw: String): Int? {
    var text = raw
    if (text.count { it == ':' } > 1 && text.substringAfterLast(":").length == 1) {
        text = text.substringBeforeLast(":") + "." + text.substringAfterLast(":")
    }
    val match = CLOCK_RE.matchEntire(text) ?: return null
    val hours = match.groupValues[1].ifEmpty { null }
    val minutes = match.groupValues[2].toInt()
    val seconds = match.groupValues[3].toInt()
    val tenths = match.groupValues[4].ifEmpty { null }
    if (seconds >= 60 || (hours != null && minutes >= 60)) return null
    var total = minutes * 60 + seconds
    if (hours != null) total += hours.toInt() * 3600
    return total * 1000 + (tenths?.toInt()?.times(100) ?: 0)
}

/** One clock box read: the text, its value, and how sure the glyphs were. */
class BoxReading(
    val box: ClockBox,
    val text: String,
    val ms: Int?,
    val confidence: Double,
    val glyphs: List<Triple<String, Double, Double>>,
    val icon: Region?,
)

internal fun glyphConfidence(score: Double, margin: Double): Double =
    min((score - 0.5) / 0.3, margin / 0.12).coerceIn(0.0, 1.0)

fun readBox(gray: Plane, box: ClockBox, cell: Double, templates: GlyphTemplates): BoxReading? {
    val ink = inkMap(gray, box, cell)
    val seg = segment(ink) ?: return null
    val glyphs = seg.glyphs.map { templates.classify(glyphCanvas(ink, seg.line, it)) }
    val text = glyphs.joinToString("") { it.first }
    // `min()` of nothing raises in Python, and the whole frame's reading is abandoned.
    val confidence = glyphs.minOf { glyphConfidence(it.second, it.third) }
    var icon: Region? = null
    if (seg.icon.isNotEmpty()) {
        val a = seg.icon.first().first
        val b = seg.icon.last().second
        icon = Region(box.left + a, box.top, b - a, box.bottom - box.top)
    }
    return BoxReading(box, text, parseClock(text), confidence, glyphs, icon)
}

private class BoxState {
    var ms: Int? = null
    var decreased = Double.NEGATIVE_INFINITY
    var increased = Double.NEGATIVE_INFINITY
    var iconMoved = Double.NEGATIVE_INFINITY
    var icon: IntArray? = null
    var iconW = 0
    var iconH = 0
}

/**
 * Whether the icon patch changed by more than a pixel's jitter, tolerating a one pixel shift
 * of the located box.
 */
internal fun iconChanged(
    previous: IntArray,
    current: IntArray,
    w: Int,
    h: Int,
    previousW: Int,
    previousH: Int,
): Boolean {
    if (previousW != w || previousH != h || min(w, h) < 4) return false
    var best = Int.MAX_VALUE
    for (dy in -1..1) {
        for (dx in -1..1) {
            var count = 0
            for (y in 1 until h - 1) {
                for (x in 1 until w - 1) {
                    val a = previous[(y + dy) * w + (x + dx)]
                    val b = current[y * w + x]
                    if (abs(a - b) > 24) count += 1
                }
            }
            best = min(best, count)
        }
    }
    return best >= 3
}

/**
 * Reads both clocks near a tracked board, frame by frame.
 *
 * [ClockReading.running] is decided from motion, never from the box colours, which are the
 * sides. The running clock's icon is a face whose hand sweeps round, so its pixels change
 * from frame to frame while the idle icon's never do; and the running clock's value falls
 * once a second, while a clock that just rose has taken its increment and stopped, which
 * makes the other one the running clock. Evidence expires after [staleAfterS], so a game
 * that has not begun, or that has ended, reports no running clock.
 *
 * Whose turn it is comes from the board, not from here. This is a reading for the record and
 * for cross-checking, and nothing in the game state depends on it.
 *
 * A reader without templates (the asset is missing or unreadable) reads nothing rather than
 * raising, so a broken install costs the clocks and not the session.
 *
 * Open, as the desktop's class is, for the tests that give the pipeline a reader of their own
 * (`tests/test_clock_pipeline.py`'s `_StubReader`).
 */
open class ClockReader(
    val templates: GlyphTemplates? = null,
    val minConfidence: Double = 0.35,
    val staleAfterS: Double = 3.0,
    val handoverHoldsS: Double = 1.2,
) {
    var last: MutableMap<Which, BoxReading?> =
        mutableMapOf(Which.TOP to null, Which.BOTTOM to null)
        private set

    private var state = mutableMapOf(Which.TOP to BoxState(), Which.BOTTOM to BoxState())
    private var calls = 0

    open fun reset() {
        state = mutableMapOf(Which.TOP to BoxState(), Which.BOTTOM to BoxState())
        last = mutableMapOf(Which.TOP to null, Which.BOTTOM to null)
    }

    /**
     * Both clocks for this frame, or null when no clock is drawn near the board (an untimed
     * game, a lobby, the board scrolled so the clocks are off screen). Never raises on a bad
     * frame.
     */
    open fun read(frame: Mat, board: BoardCandidate, ts: Double? = null): ClockReading? = try {
        readOrThrow(frame, board, ts)
    } catch (_: Exception) {
        // A bad frame must never take the session down.
        null
    }

    private fun readOrThrow(frame: Mat, board: BoardCandidate, ts: Double?): ClockReading? {
        if (templates == null || frame.empty()) return null
        calls += 1
        val now = ts ?: (calls * 0.1)
        val (x, y, w, h) = board.region
        val cell = board.cellPx
        if (cell < 4.0 || cell > 4096.0 || w <= 0 || h <= 0) return null
        val reach = pyRound(SEARCH_CELLS * cell)
        val readings = mutableMapOf(
            Which.TOP to readSide(frame, x + w, (y - reach) to (y - 1), cell),
            Which.BOTTOM to readSide(frame, x + w, (y + h + 1) to (y + h + reach), cell),
        )
        last = readings
        if (readings.values.all { it == null }) return null
        for ((which, reading) in readings) observe(which, reading, frame, now)
        val running = running(now)
        var light: Which? = null
        val top = readings[Which.TOP]
        val bottom = readings[Which.BOTTOM]
        if (top != null && bottom != null && top.box.light != bottom.box.light) {
            light = if (top.box.light) Which.TOP else Which.BOTTOM
        } else if (top != null && bottom == null) {
            light = if (top.box.light) Which.TOP else null
        } else if (bottom != null && top == null) {
            light = if (bottom.box.light) Which.BOTTOM else null
        }
        val present = readings.values.filterNotNull()
        val confidence = present.minOf { it.confidence }
        return ClockReading(
            topMs = value(top),
            bottomMs = value(bottom),
            running = running,
            confidence = confidence,
            light = light,
        )
    }

    private fun value(reading: BoxReading?): Int? {
        if (reading == null || reading.confidence < minConfidence) return null
        return reading.ms
    }

    private fun readSide(
        frame: Mat,
        xRight: Int,
        rows: Pair<Int, Int>,
        cell: Double,
    ): BoxReading? {
        val height = frame.rows()
        val width = frame.cols()
        val r0 = max(0, rows.first)
        val r1 = min(height, rows.second)
        val c0 = max(0, (xRight - BOX_WIDTH_CELLS.second * cell).toInt() - 2)
        val c1 = min(width, pyRound(xRight.toDouble()) + 1)
        if (r1 - r0 < 4 || c1 - c0 < 4) return null
        val templates = this.templates ?: return null
        val gray = grayOf(frame, c0, r0, c1 - c0, r1 - r0)
        val box = findBox(gray, (xRight - c0).toDouble(), 0 to (r1 - r0), cell) ?: return null
        val reading = readBox(gray, box, cell, templates) ?: return null
        val shifted = ClockBox(
            box.top + r0,
            box.bottom + r0,
            box.left + c0,
            box.right + c0,
            box.background,
            box.text,
        )
        var icon = reading.icon
        if (icon != null) icon = Region(icon.x + c0, icon.y + r0, icon.w, icon.h)
        return BoxReading(shifted, reading.text, reading.ms, reading.confidence, reading.glyphs, icon)
    }

    private fun observe(which: Which, reading: BoxReading?, frame: Mat, now: Double) {
        val state = this.state.getValue(which)
        if (reading == null) {
            state.icon = null
            return
        }
        val ms = value(reading)
        if (ms != null) {
            val before = state.ms
            if (before != null) {
                if (ms < before) state.decreased = now else if (ms > before) state.increased = now
            }
            state.ms = ms
        }
        val icon = reading.icon
        if (icon == null || icon.w < 3 || icon.h < 3) {
            state.icon = null
            return
        }
        val patch = grayBytes(frame, icon.x, icon.y, icon.w, icon.h)
        val had = state.icon
        if (had != null && iconChanged(had, patch, icon.w, icon.h, state.iconW, state.iconH)) {
            state.iconMoved = now
        }
        state.icon = patch
        state.iconW = icon.w
        state.iconH = icon.h
    }

    private fun running(now: Double): Which? {
        val top = state.getValue(Which.TOP)
        val bottom = state.getValue(Which.BOTTOM)
        // A clock that rose took its increment: that side has just moved and stopped, and the
        // other one started. For a moment after, the stopped clock's icon is still settling,
        // so the handover outranks the icons rather than competing with them.
        val handover = max(top.increased, bottom.increased)
        if (now - handover <= handoverHoldsS) {
            return if (top.increased >= bottom.increased) Which.BOTTOM else Which.TOP
        }
        val evidence = mapOf(
            Which.TOP to maxOf(top.decreased, top.iconMoved, bottom.increased),
            Which.BOTTOM to maxOf(bottom.decreased, bottom.iconMoved, top.increased),
        )
        // `max(dict, key=...)` keeps the first key on a tie, and the map is in insertion order.
        var which = Which.TOP
        var most = evidence.getValue(Which.TOP)
        if (evidence.getValue(Which.BOTTOM) > most) {
            which = Which.BOTTOM
            most = evidence.getValue(Which.BOTTOM)
        }
        if (now - most > staleAfterS) return null
        return which
    }

    companion object {
        /** A grey plane of the frame's rectangle, the floats the desktop's `astype` gives. */
        internal fun grayOf(frame: Mat, x: Int, y: Int, w: Int, h: Int): Plane {
            val strip = Mat(frame, Rect(x, y, w, h))
            val gray = Mat()
            if (strip.channels() == 3) {
                Imgproc.cvtColor(strip, gray, Imgproc.COLOR_BGR2GRAY)
            } else {
                strip.copyTo(gray)
            }
            val bytes = ByteArray(w * h)
            gray.get(0, 0, bytes)
            strip.release()
            gray.release()
            return Plane(w, h, FloatArray(w * h) { (bytes[it].toInt() and 0xFF).toFloat() })
        }

        /** The same rectangle as grey bytes, for the icon patches. */
        internal fun grayBytes(frame: Mat, x: Int, y: Int, w: Int, h: Int): IntArray {
            val strip = Mat(frame, Rect(x, y, w, h))
            val gray = Mat()
            if (strip.channels() == 3) {
                Imgproc.cvtColor(strip, gray, Imgproc.COLOR_BGR2GRAY)
            } else {
                strip.copyTo(gray)
            }
            val bytes = ByteArray(w * h)
            gray.get(0, 0, bytes)
            strip.release()
            gray.release()
            return IntArray(w * h) { bytes[it].toInt() and 0xFF }
        }
    }
}
