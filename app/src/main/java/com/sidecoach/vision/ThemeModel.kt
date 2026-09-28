package com.sidecoach.vision

import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.core.TermCriteria
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * A piece kind as the start position showed it: its tone mask (see [Sil.tone]), fused over the
 * piece's cells, for the fill vote ([ThemeModel.fillSide]), and the median brightness of its
 * fill, for the brightness vote.
 */
class Exemplar(
    val symbol: String,
    val tone: BooleanArray,
    val brightness: Double,
)

/**
 * A piece kind's start-position crop in grey: the reference the phase landing measures a cell's
 * shift against (`ThemeModel.landing_crops`).
 */
class LandingCrop(val rows: Int, val cols: Int, val pixels: ByteArray) {
    companion object {
        /** A cell's BGR crop in grey, as `cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)`. */
        fun of(crop: Mat): LandingCrop {
            val grey = Mat()
            Imgproc.cvtColor(crop, grey, Imgproc.COLOR_BGR2GRAY)
            val pixels = ByteArray(grey.rows() * grey.cols())
            grey.get(0, 0, pixels)
            val out = LandingCrop(grey.rows(), grey.cols(), pixels)
            grey.release()
            return out
        }
    }
}

/**
 * Per-theme piece recognition bootstrapped from the start position: the `ThemeModel` of
 * `sidecoach/vision/pieces.py`, ported.
 *
 * The desktop's default matcher since 2026-09-19 is the embedding: the 64 cells of the
 * start position embedded by a frozen net, a later cell named by its nearest one. The
 * silhouette still decides everything that is not a piece -- empty squares, tints, hover
 * outlines, covers and drawings -- and still casts the brightness vote for a piece's side.
 */
class ThemeModel(
    val exemplars: List<Exemplar>,
    val lightBg: DoubleArray,
    val darkBg: DoubleArray,
    /** The set's colours: what a saturated patch over a square may be before it is a drawing. */
    val pieceColours: PieceColours,
    /** Every colour the set draws with, the palette a drawing over a square is told from. */
    val palette: List<DoubleArray>,
    /** The start position's 64 cells as the net embeds them, and what each held. */
    val exemplarVectors: Array<FloatArray>,
    val exemplarLabels: List<String>,
    val whiteAtBottom: Boolean,
    /**
     * Each piece kind's start-position crop in grey on each square colour, keyed by (symbol, on
     * a light square): the reference the phase landing measures a cell's shift against (an
     * instance off the labelled edge cells).
     */
    val landingCrops: Map<Pair<String, Boolean>, LandingCrop> = emptyMap(),
    /** The cell size the landing crops were taken at, for [phaseShift] to meet a resize. */
    val landingPx: Double = 0.0,
) {

    /** The last frame's cells: the crop, its silhouette, the cover flag, the cell size, the reading. */
    private class Cell(
        val crop: ByteArray,
        val sil: Sil,
        val covered: Boolean,
        val cellPx: Double,
        val reading: Pieces.CellReading,
    )

    private val cells = HashMap<Int, Cell>()

    /** The silhouette cell [i] of the last frame read was taken as, for the parity tests. */
    internal fun silAt(i: Int): Sil? = cells[i]?.sil

    /**
     * The rank-digit exemplars from the board's left edge, learned with the theme: how a
     * board read cold tells which side sits at the bottom (`vision/coordinates.py`).
     */
    var rankLabels: Map<Int, BooleanArray> = emptyMap()

    /** Cells the last frame had to embed: what the reading costs, for the log. */
    var lastFresh: Int = 64

    /** How many of those went through the net: the ones with a piece's worth of ink. */
    var lastEmbedded: Int = 64
        private set

    /**
     * The hue of a cell's background shift: "none", "move", "red", or "foreign". A
     * translucent tint keeps its hue whatever the square colour, so the hue is read from
     * the direction of the shift rather than from the blended colour itself.
     */
    fun highlightKind(background: DoubleArray, row: Int, col: Int): String {
        val expected = if (Squares.isLight(row, col)) lightBg else darkBg
        val shift = DoubleArray(3) { background[it] - expected[it] }
        if (norm(shift) <= Pieces.HIGHLIGHT_TOLERANCE) return "none"
        val blue = shift[0]
        val green = shift[1]
        val red = shift[2]
        if (green < -10.0 && red > green + 25.0 && blue <= 5.0) return "red"
        if (blue <= 5.0 && green >= -15.0) return "move"
        return "foreign"
    }

    /**
     * Whether what shifted a cell's background is a readable tint, and its hue.
     *
     * A tint blends one colour over the whole square, so the ring just inside the border
     * and the deeper corner blocks move away from the square colour in the same direction,
     * and no corner pixel is left at the square colour. A hover outline thick enough to
     * fill the ring, a capture ring, or the edge of something drawn over the square shifts
     * the ring while the corners stay clean, at least in part: that is foreign, and the
     * cell cannot be read.
     */
    fun tintKind(sil: Sil, row: Int, col: Int): Pair<Pieces.TintKind, String?> {
        val expected = if (Squares.isLight(row, col)) lightBg else darkBg
        val shallow = DoubleArray(3) { sil.background[it] - expected[it] }
        val shallowNorm = norm(shallow)
        if (shallowNorm <= Pieces.HIGHLIGHT_TOLERANCE) return Pieces.TintKind.NONE to null
        val corners = sil.cornerPixels.size / 3
        if (corners == 0) return Pieces.TintKind.FOREIGN to null
        var clean = 0
        var shiftedCount = 0
        for (i in 0 until corners) {
            val s0 = sil.cornerPixels[i * 3].toDouble() - expected[0]
            val s1 = sil.cornerPixels[i * 3 + 1].toDouble() - expected[1]
            val s2 = sil.cornerPixels[i * 3 + 2].toDouble() - expected[2]
            var squares = 0.0
            squares += s0 * s0
            squares += s1 * s1
            squares += s2 * s2
            val magnitude = sqrt(squares)
            if (magnitude <= Pieces.TINT_MIN_CORNER_SHIFT) {
                clean++
                continue
            }
            var dot = 0.0
            dot += s0 * shallow[0]
            dot += s1 * shallow[1]
            dot += s2 * shallow[2]
            if (dot / (max(magnitude, 1e-6) * shallowNorm) >= Pieces.TINT_MIN_COSINE) shiftedCount++
        }
        val known = clean + shiftedCount
        // Corner pixels that are neither clean nor shifted like the ring belong to a piece,
        // a badge or a label and are not counted, so a big piece on a highlighted square is
        // still a highlighted square.
        if (known < Pieces.CORNER_MIN_KNOWN * corners) return Pieces.TintKind.FOREIGN to null
        if (clean >= Pieces.CORNER_CLEAN_MAX * known) return Pieces.TintKind.FOREIGN to null
        val hue = when (highlightKind(sil.background, row, col)) {
            "move" -> "yellow"
            "red" -> "red"
            else -> "other"
        }
        return Pieces.TintKind.TINT to hue
    }

    /** Cells under an opaque overlay, and the cells its edges reach into. */
    fun coveredCells(sils: List<Sil>): BooleanArray {
        val backgrounds = Array(64) { sils[it].background }
        val expected = Array(64) { if (Squares.isLight(it / 8, it % 8)) lightBg else darkBg }
        val shifts = DoubleArray(64) { i ->
            norm(DoubleArray(3) { k -> backgrounds[i][k] - expected[i][k] })
        }
        val shifted = BooleanArray(64) { shifts[it] > Pieces.HIGHLIGHT_TOLERANCE }
        val opaque = Covers.opaqueCover(backgrounds, shifted)
        val translucent = Covers.translucentCover(backgrounds, expected, shifted)
        val covered = Covers.coverBody(BooleanArray(64) { opaque[it] || translucent[it] }, shifts)
        val reached = BooleanArray(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                if (covered[r * 8 + c]) continue
                outer@ for (rr in max(0, r - 1)..min(7, r + 1)) {
                    for (cc in max(0, c - 1)..min(7, c + 1)) {
                        if (covered[rr * 8 + cc] &&
                            Covers.overlayReaches(sils[r * 8 + c], backgrounds[rr * 8 + cc])
                        ) {
                            reached[r * 8 + c] = true
                            break@outer
                        }
                    }
                }
            }
        }
        return BooleanArray(64) { covered[it] || reached[it] }
    }

    /**
     * The 64 cells of a frame, read. Between one frame and the next most cells do not
     * change a pixel, and a cell that did not change reads the same: its silhouette and
     * its reading are reused, which is most of a frame's cost -- and on a phone, most of
     * the heat. Only the cells that changed are embedded, and in one batch.
     */
    fun classifyFrame(
        frame: Mat,
        geometry: BoardGeometry,
        embedder: CellEmbedder,
    ): MutableList<Pieces.CellReading> {
        val crops = ArrayList<Mat>(64)
        val bytes = ArrayList<ByteArray>(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val crop = geometry.crop(frame, r, c, 2 * Squares.CELL)
                crops.add(crop)
                val raw = ByteArray(crop.rows() * crop.cols() * 3)
                crop.get(0, 0, raw)
                bytes.add(raw)
            }
        }
        try {
            val sils = ArrayList<Sil>(64)
            for (i in 0 until 64) {
                val last = cells[i]
                sils.add(
                    if (last != null && last.crop.contentEquals(bytes[i])) last.sil
                    else SilhouetteReader.of(crops[i], Squares.CELL, pieceColours, palette),
                )
            }
            val covered = coveredCells(sils)
            val cellPx = geometry.cellPx
            val fresh = ArrayList<Int>(64)
            for (i in 0 until 64) {
                val last = cells[i]
                val same = last != null && last.sil === sils[i] && last.covered == covered[i] &&
                    last.cellPx == cellPx && last.crop.contentEquals(bytes[i])
                if (!same) fresh.add(i)
            }
            lastFresh = fresh.size
            val vectors = HashMap<Int, FloatArray>()
            // The net embeds, in one batch, the cells that changed with a piece's worth of ink:
            // only those are named by it ([classify]). An empty square or a hint dot is the
            // silhouette's to read, and embedding it anyway was most of a full read's cost --
            // a second on the phone for a board that was mostly empty squares.
            val inked = fresh.filter { sils[it].fraction >= Pieces.PIECE_MIN_FRACTION }
            lastEmbedded = inked.size
            if (inked.isNotEmpty()) {
                // Each at one resolution, whatever the board's size (`Pieces.canonical`).
                val seen = inked.map { Pieces.canonical(crops[it]) }
                val embedded = try {
                    embedder.embed(seen)
                } finally {
                    seen.forEach { it.release() }
                }
                for ((k, i) in inked.withIndex()) vectors[i] = embedded[k]
            }
            val readings = ArrayList<Pieces.CellReading>(64)
            for (i in 0 until 64) {
                val last = cells[i]
                if (!fresh.contains(i) && last != null) {
                    readings.add(last.reading)
                    continue
                }
                val reading = classify(sils[i], i / 8, i % 8, covered[i], cellPx, vectors[i], crops[i])
                cells[i] = Cell(bytes[i], sils[i], covered[i], cellPx, reading)
                readings.add(reading)
            }

            var tinted = 0
            for (reading in readings) if (reading.tint != null) tinted++
            if (tinted >= Pieces.THEME_MIN_TINTED) {
                // Highlights, marks and queues tint a few squares; a tint over half the
                // board or more is a theme this recogniser was not built on, and the
                // silhouettes it reads on it are not evidence.
                for (i in 0 until 64) {
                    val reading = readings[i]
                    if (reading.tint == null) continue
                    readings[i] = reading.copy(
                        kind = Pieces.Kind.UNKNOWN, symbol = null, colour = null, confidence = 0.0,
                        highlighted = false, settled = false, guess = null, tint = null,
                    )
                }
            }
            var coveredCount = 0
            for (v in covered) if (v) coveredCount++
            if (coveredCount < Pieces.TINT_MIN_CELLS || !Covers.hasBlock(covered)) {
                // A strip of covered cells is a tooltip over one rank or file, not a
                // dialog: the user may well be dragging next to it, and that drag must
                // stay visible.
                return readings
            }
            // Next to a dialog-sized overlay, a cell that does not read cleanly is a cell
            // the overlay's edge cuts through. Nothing is dragged while a dialog is up.
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val reading = readings[r * 8 + c]
                    val junk = reading.hover ||
                        (reading.kind == Pieces.Kind.PIECE && (!reading.settled || reading.symbol == null))
                    if (!junk || covered[r * 8 + c]) continue
                    var beside = false
                    for (rr in max(0, r - 1)..min(7, r + 1)) {
                        for (cc in max(0, c - 1)..min(7, c + 1)) if (covered[rr * 8 + cc]) beside = true
                    }
                    if (beside) {
                        readings[r * 8 + c] = reading.copy(
                            kind = Pieces.Kind.UNKNOWN, symbol = null, colour = null,
                            confidence = 0.0, highlighted = false, settled = false, hover = false,
                            offset = 0.0, guess = null, covered = true,
                        )
                    }
                }
            }
            return readings
        } finally {
            crops.forEach { it.release() }
        }
    }

    /** One cell, named: `ThemeModel._classify`, the embedding path. */
    fun classify(
        sil: Sil,
        row: Int,
        col: Int,
        covered: Boolean,
        cellPx: Double?,
        vector: FloatArray?,
        crop: Mat? = null,
    ): Pieces.CellReading {
        val (kindOfTint, tint) = tintKind(sil, row, col)
        // Red is the colour of marks, checks and premove queues, never of a move.
        val highlighted = kindOfTint == Pieces.TintKind.TINT && tint != "red"
        if (covered) {
            // An opaque overlay confirmed by the neighbours hides this cell: whatever the
            // silhouette shows is not the board, no evidence either way, and never a hover.
            return reading(sil, Pieces.Kind.UNKNOWN, covered = true)
        }
        if (kindOfTint == Pieces.TintKind.FOREIGN) {
            // A hover outline thick enough to fill the ring, a capture ring, an edge: the
            // silhouette is not the board, and a commit must wait until it is gone.
            return reading(sil, Pieces.Kind.UNKNOWN, hover = true)
        }
        if (sil.overlay > Pieces.OVERLAY_MAX) {
            // An arrow or a mark drawn over the square: an arrowhead over a black pawn
            // reads as a white piece, so the cell waits.
            return reading(sil, Pieces.Kind.UNKNOWN, drawn = true)
        }
        if (vector != null && sil.fraction >= Pieces.PIECE_MIN_FRACTION) {
            // Nothing is nothing, whatever tint it wears, and a hint dot is no evidence:
            // the embedding is asked only where a piece's worth of foreground is.
            return fromEmbedding(sil, vector, highlighted, tint, cellPx, crop, Squares.isLight(row, col))
        }
        if (sil.fraction <= Pieces.EMPTY_MAX_FRACTION) {
            return reading(
                sil, Pieces.Kind.EMPTY, confidence = 1.0, highlighted = highlighted,
                settled = true, hover = sil.hover, tint = tint,
            )
        }
        return reading(sil, Pieces.Kind.UNKNOWN, highlighted = highlighted, hover = sil.hover, tint = tint)
    }

    /**
     * The cell named by the nearest of the start position's cells in embedding space. The
     * side is voted ([voteSide]); once it is, the kind may be named by either side's cells
     * ([kindScores]). Where the piece sits is the phase correlation against the piece's own
     * start crop on a square of the cell's colour ([light]).
     */
    private fun fromEmbedding(
        sil: Sil,
        vector: FloatArray,
        highlighted: Boolean,
        tint: String?,
        cellPx: Double?,
        crop: Mat? = null,
        light: Boolean = true,
    ): Pieces.CellReading {
        val sims = FloatArray(exemplarVectors.size)
        for (i in exemplarVectors.indices) {
            var dot = 0f
            val row = exemplarVectors[i]
            for (k in row.indices) dot += row[k] * vector[k]
            sims[i] = dot
        }
        val side = voteSide(sil, sims)
        val scored = if (side != null) kindScores(sims) else sims
        val allowed = exemplarLabels.indices.filter { i ->
            val label = exemplarLabels[i]
            label == "empty" || side == null || label[0].isUpperCase() == side.color
        }
        val order = allowed.sortedByDescending { scored[it] }
        val best = order.first()
        val label = exemplarLabels[best]
        val similarity = scored[best]
        val runnerIndex = order.drop(1).firstOrNull { exemplarLabels[it] != label }
        val other = runnerIndex?.let { scored[it] } ?: -1f
        val margin = similarity - other
        if (similarity < CellEmbedder.MIN_SIMILARITY) {
            // Nothing on the start position looks like this, yet a piece's worth of
            // foreground is there: a king under the check glow, a piece under a label or
            // an arrow. Occupied, by something of no sure kind.
            return reading(
                sil, Pieces.Kind.PIECE, colour = side, confidence = 0.5, highlighted = highlighted,
                settled = false, hover = sil.hover,
                guess = if (label != "empty") label else null, tint = tint,
            )
        }
        if (label == "empty") {
            if (margin < CellEmbedder.MIN_MARGIN) {
                return reading(sil, Pieces.Kind.UNKNOWN, highlighted = highlighted, hover = sil.hover, tint = tint)
            }
            return reading(
                sil, Pieces.Kind.EMPTY, confidence = margin.toDouble(), highlighted = highlighted,
                settled = true, hover = sil.hover, tint = tint,
            )
        }
        val colour = Pieces.Side.of(label[0].isUpperCase())
        val reference = landingCrops[label to light]
        val offset = if (crop != null && reference != null) phaseShift(reference, crop, landingPx, cellPx ?: 0.0) else 0.0
        val settled = offset <= Pieces.settleTolerance(cellPx) && !sil.hover
        if (margin < CellEmbedder.MIN_MARGIN && runnerIndex != null && exemplarLabels[runnerIndex] == "empty") {
            // As near an empty square as a piece: something on the square that is neither,
            // a hint, a label, a mark's edge. No evidence, and no drag either.
            return reading(
                sil, Pieces.Kind.UNKNOWN, highlighted = highlighted, hover = sil.hover,
                guess = label, tint = tint,
            )
        }
        return reading(
            sil, Pieces.Kind.PIECE, symbol = label, colour = colour, confidence = margin.toDouble(),
            highlighted = highlighted, settled = settled, hover = sil.hover, offset = offset, tint = tint,
        )
    }

    /**
     * `pieces._phase_shift`: how far the piece in [crop] (BGR) sits from where it sits in
     * [reference] (grey), as a share of the cell, by phase correlation under a Hanning window.
     * When the two were taken at cell sizes ([referencePx], [cropPx]) more than
     * [Pieces.LANDING_SAME_SIZE] apart, they are also compared brought down to the coarser of
     * the two sizes, and the surer of the two correlations (the higher peak) gives the shift
     * (see [Pieces.LANDING_SAME_SIZE]).
     */
    private fun phaseShift(reference: LandingCrop, crop: Mat, referencePx: Double, cropPx: Double): Double {
        val ref = Mat(reference.rows, reference.cols, CvType.CV_8UC1)
        ref.put(0, 0, reference.pixels)
        val grey = Mat()
        Imgproc.cvtColor(crop, grey, Imgproc.COLOR_BGR2GRAY)
        if (grey.rows() != ref.rows() || grey.cols() != ref.cols()) {
            Imgproc.resize(grey, grey, Size(ref.cols().toDouble(), ref.rows().toDouble()))
        }
        val tries = ArrayList<Pair<Mat, Mat>>(2)
        tries.add(ref to grey)
        if (referencePx > 0 && cropPx > 0) {
            val coarser = min(referencePx, cropPx)
            val finer = max(referencePx, cropPx)
            val size = pyRound(coarser)
            if (finer > coarser * (1.0 + Pieces.LANDING_SAME_SIZE) && 8 <= size && size < reference.rows) {
                val small = Size(size.toDouble(), size.toDouble())
                val smallRef = Mat()
                Imgproc.resize(ref, smallRef, small, 0.0, 0.0, Imgproc.INTER_AREA)
                val smallGrey = Mat()
                Imgproc.resize(grey, smallGrey, small, 0.0, 0.0, Imgproc.INTER_AREA)
                tries.add(smallRef to smallGrey)
            }
        }
        var shift = 0.0
        var surest = Double.NEGATIVE_INFINITY
        for ((one, cell) in tries) {
            val window = Mat()
            Imgproc.createHanningWindow(window, Size(one.cols().toDouble(), one.rows().toDouble()), CvType.CV_32F)
            val a = Mat()
            one.convertTo(a, CvType.CV_32F)
            val b = Mat()
            cell.convertTo(b, CvType.CV_32F)
            val response = DoubleArray(1)
            val found = Imgproc.phaseCorrelate(a, b, window, response)
            for (m in listOf(window, a, b)) m.release()
            if (response[0] > surest) {
                shift = hypot(found.x, found.y) / one.rows()
                surest = response[0]
            }
        }
        for ((one, cell) in tries) {
            one.release()
            cell.release()
        }
        return shift
    }

    /**
     * The piece's side by a vote of three signals, each cast only when sure: the embedding
     * (its nearest white and nearest black COLOUR_TIE apart), the brightness of the
     * silhouette's fill, and the fill's own pixels under the named kind's exemplar mask.
     * Each is wrong somewhere on its own, and in every case seen the other two were right.
     */
    private fun voteSide(sil: Sil, sims: FloatArray): Pieces.Side? {
        val votes = ArrayList<Pieces.Side>(3)
        var bestWhite = -1f
        var bestBlack = -1f
        for (i in exemplarLabels.indices) {
            val label = exemplarLabels[i]
            if (label == "empty") continue
            if (label[0].isUpperCase()) bestWhite = max(bestWhite, sims[i])
            else bestBlack = max(bestBlack, sims[i])
        }
        if (abs(bestWhite - bestBlack) >= Pieces.COLOUR_TIE) {
            votes.add(if (bestWhite > bestBlack) Pieces.Side.WHITE else Pieces.Side.BLACK)
        }
        sideOf(sil)?.let { votes.add(it) }
        var nearest = "empty"
        var nearestSim = Float.NEGATIVE_INFINITY
        for (i in exemplarLabels.indices) {
            if (sims[i] > nearestSim) { nearestSim = sims[i]; nearest = exemplarLabels[i] }
        }
        if (nearest != "empty") fillSide(sil, nearest)?.let { votes.add(it) }
        val white = votes.count { it == Pieces.Side.WHITE }
        val black = votes.size - white
        if (white == black) return null
        return if (white > black) Pieces.Side.WHITE else Pieces.Side.BLACK
    }

    /**
     * Each cell's similarity raised to the best of its kind's, either side's: what a piece is,
     * is its shape, once its side is voted (`_kind_scores`). At [Pieces.CANONICAL_PX] the
     * square under a piece weighs more in its embedding, and the start position shows each
     * side's king and queen on one square colour only: a black queen on a light square came
     * nearer the black bishop learned on one than the black queen learned on a dark square,
     * and a mated king under the check glow nearer a rook of its own side than its own king
     * (the desktop's stress seeds 469, 584, 766). The other side shows the king and queen on
     * the other square colour. Learning them again on the other colour, their crops' squares
     * recoloured, named a black queen on g2 (688), and a set never learned read as a position
     * with two kings (the desktop's tests/test_cold_start.py).
     */
    private fun kindScores(sims: FloatArray): FloatArray {
        val scored = sims.copyOf()
        val kinds = LinkedHashMap<String, MutableList<Int>>()
        for (i in exemplarLabels.indices) {
            val label = exemplarLabels[i]
            if (label != "empty") kinds.getOrPut(label.lowercase()) { ArrayList() }.add(i)
        }
        for (members in kinds.values) {
            var best = Float.NEGATIVE_INFINITY
            for (i in members) best = max(best, sims[i])
            for (i in members) scored[i] = best
        }
        return scored
    }

    /**
     * The side whose fill the pixels under the named kind's fill show, against the two
     * sides' fills learned at the start. Null when the mask is too small to read or the
     * colour lies about as near one fill as the other.
     */
    internal fun fillSide(sil: Sil, label: String): Pieces.Side? {
        if (pieceColours.size < 2) return null
        val masks = exemplars.filter { it.symbol.uppercase() == label.uppercase() }.map { it.tone }
        if (masks.isEmpty()) return null
        val mask = if (masks.size == 1) masks[0] else BooleanArray(masks[0].size) { masks[0][it] && masks[1][it] }
        val core = erode(mask, Squares.CELL, Pieces.FILL_SIDE_ERODE)
        var on = 0
        for (v in core) if (v) on++
        if (on < Pieces.FILL_SIDE_MIN_PIXELS) return null
        val bs = ArrayList<Float>(on)
        val gs = ArrayList<Float>(on)
        val rs = ArrayList<Float>(on)
        for (i in core.indices) {
            if (!core[i]) continue
            bs.add(sil.inner[i * 3]); gs.add(sil.inner[i * 3 + 1]); rs.add(sil.inner[i * 3 + 2])
        }
        val colour = doubleArrayOf(median(bs), median(gs), median(rs))
        val white = distance(colour, pieceColours[0].first)
        val black = distance(colour, pieceColours[1].first)
        if (min(white, black) * Pieces.FILL_SIDE_RATIO > max(white, black)) return null
        return if (white < black) Pieces.Side.WHITE else Pieces.Side.BLACK
    }

    /** The side whose exemplars' brightness the piece's falls among (see SIDE_MARGIN). */
    private fun sideOf(sil: Sil): Pieces.Side? {
        val inside = ArrayList<Pieces.Side>(2)
        for (side in listOf(Pieces.Side.WHITE, Pieces.Side.BLACK)) {
            val values = exemplars
                .filter { it.symbol[0].isUpperCase() == side.color }
                .map { it.brightness }
            if (values.isEmpty()) continue
            if (values.min() - Pieces.SIDE_MARGIN <= sil.brightness &&
                sil.brightness <= values.max() + Pieces.SIDE_MARGIN
            ) {
                inside.add(side)
            }
        }
        return if (inside.size == 1) inside[0] else null
    }

    /** The position as FEN placement, from a board read in screen order. */
    fun placement(readings: List<Pieces.CellReading>): String {
        val squares = arrayOfNulls<String>(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val symbol = readings[r * 8 + c].symbol
                if (symbol != null) squares[Squares.squareAt(r, c, whiteAtBottom)] = symbol
            }
        }
        val out = StringBuilder()
        for (rank in 7 downTo 0) {
            var empty = 0
            for (file in 0 until 8) {
                val piece = squares[rank * 8 + file]
                if (piece == null) {
                    empty++
                } else {
                    if (empty > 0) { out.append(empty); empty = 0 }
                    out.append(piece)
                }
            }
            if (empty > 0) out.append(empty)
            if (rank > 0) out.append('/')
        }
        return out.toString()
    }

    private fun reading(
        sil: Sil,
        kind: Pieces.Kind,
        symbol: String? = null,
        colour: Pieces.Side? = null,
        confidence: Double = 0.0,
        highlighted: Boolean = false,
        settled: Boolean = false,
        hover: Boolean = false,
        offset: Double = 0.0,
        guess: String? = null,
        covered: Boolean = false,
        tint: String? = null,
        drawn: Boolean = false,
    ) = Pieces.CellReading(
        kind = kind,
        symbol = symbol,
        colour = colour,
        confidence = confidence,
        fgFraction = sil.fraction,
        luminance = sil.luminance,
        highlighted = highlighted,
        background = sil.background,
        brightness = sil.brightness,
        settled = settled,
        hover = hover,
        offset = offset,
        guess = guess,
        covered = covered,
        tint = tint,
        drawn = drawn,
    )

    /**
     * `cv2.erode(mask, ones(3, 3), iterations=times)`, by OpenCV itself: outside the patch
     * counts as set (the default border for an erosion), so a mask reaching the patch's edge
     * is not eaten from it. The phone's own erosion ate it, and 38 of 88 sets' fill masks
     * differed from the desktop's by up to 352 pixels (the parity audit, 2026-09-26).
     */
    private fun erode(mask: BooleanArray, size: Int, times: Int): BooleanArray {
        val src = Mat(size, size, CvType.CV_8U)
        src.put(0, 0, ByteArray(mask.size) { if (mask[it]) 1 else 0 })
        val kernel = Mat.ones(3, 3, CvType.CV_8U)
        val out = Mat()
        Imgproc.erode(src, out, kernel, Point(-1.0, -1.0), times)
        val bytes = ByteArray(mask.size)
        out.get(0, 0, bytes)
        src.release()
        kernel.release()
        out.release()
        return BooleanArray(mask.size) { bytes[it].toInt() != 0 }
    }

    companion object {
        private const val TAG = "sidecoach.vision"

        /** 8x8 foreground fractions, the cheap occupancy signal (`pieces.occupancy`). */
        fun occupancy(frame: Mat, geometry: BoardGeometry): List<Sil> {
            val out = ArrayList<Sil>(64)
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val crop = geometry.crop(frame, r, c, 2 * Squares.CELL)
                    out.add(SilhouetteReader.of(crop))
                    crop.release()
                }
            }
            return out
        }

        /**
         * Rows 0, 1, 6 and 7 occupied and rows 2 to 5 empty, allowing one stray cell
         * (`pieces._start_shaped`): one move away from the start already differs in two,
         * so the allowance is a single cell for a badge, a hover outline or a cursor.
         */
        fun startShaped(cells: List<Sil>): Boolean {
            var wrong = 0
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val occupied = cells[r * 8 + c].fraction >= Pieces.PIECE_MIN_FRACTION
                    if (occupied != (r <= 1 || r >= 6)) wrong++
                }
            }
            return wrong <= 1
        }

        /** Whether the frame's occupancy has the shape of the start position. */
        fun looksLikeStartPosition(frame: Mat, geometry: BoardGeometry): Boolean =
            startShaped(occupancy(frame, geometry))

        /**
         * [looksLikeStartPosition] from readings already taken, by the same rule, so a frame
         * that has been classified need not be silhouetted a second time
         * (`pieces.start_shaped`).
         */
        fun startShapedReadings(readings: List<Pieces.CellReading>): Boolean {
            var wrong = 0
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val occupied = readings[r * 8 + c].fgFraction >= Pieces.PIECE_MIN_FRACTION
                    if (occupied != (r <= 1 || r >= 6)) wrong++
                }
            }
            return wrong <= 1
        }

        /**
         * Which colour sits at the bottom of a start position: the brighter pieces are
         * white (`pieces.orientation_from_start`).
         */
        fun orientationFromStart(cells: List<Sil>): Boolean {
            val top = ArrayList<Float>(16)
            val bottom = ArrayList<Float>(16)
            for (c in 0 until 8) {
                top.add(cells[c].luminance.toFloat())
                top.add(cells[8 + c].luminance.toFloat())
                bottom.add(cells[48 + c].luminance.toFloat())
                bottom.add(cells[56 + c].luminance.toFloat())
            }
            return median(bottom) > median(top)
        }

        /** The start position in screen order for the side at the bottom: 64 labels. */
        fun startLabels(whiteAtBottom: Boolean): List<String> {
            val back = listOf("r", "n", "b", "q", "k", "b", "n", "r")
            return (0 until 64).map { i ->
                val square = Squares.squareAt(i / 8, i % 8, whiteAtBottom)
                when (square / 8) {
                    0 -> back[square % 8].uppercase()
                    1 -> "P"
                    6 -> "p"
                    7 -> back[square % 8]
                    else -> "empty"
                }
            }
        }

        /**
         * The recogniser for this game, or null when the start frame is not clean enough: a
         * cursor's outline over a piece, a piece hidden (the embeddings say so,
         * [startReadsAsStart]), or square colours that cannot be told apart. The net is
         * required (a phone without its model file learns nothing: see
         * [CellEmbedder.load]). The kings and queens have one cell each, and what a start
         * frame shows is every piece's reference for the game.
         *
         * Two passes: the first learns the set's colours from every piece, the second reads
         * every piece with those colours, as every later frame will be read.
         */
        fun fromStartPosition(
            frame: Mat,
            geometry: BoardGeometry,
            whiteAtBottom: Boolean,
            embedder: CellEmbedder,
        ): ThemeModel? {
            val labels = startLabels(whiteAtBottom)
            val crops = ArrayList<Mat>(64)
            for (r in 0 until 8) for (c in 0 until 8) crops.add(geometry.crop(frame, r, c, 2 * Squares.CELL))
            try {
                // In the order the Python's dicts keep: the piece kinds as the cells first
                // show them. The palette clusters the ink in that order.
                val first = LinkedHashMap<String, MutableList<Sil>>()
                for (i in 0 until 64) {
                    if (labels[i] == "empty") continue
                    first.getOrPut(labels[i]) { ArrayList() }.add(SilhouetteReader.of(crops[i]))
                }
                val colours = pieceColoursOf(first)
                val palette = setPalette(first)

                val samples = LinkedHashMap<String, MutableList<Sil>>()
                val lightBgs = ArrayList<DoubleArray>()
                val darkBgs = ArrayList<DoubleArray>()
                for (i in 0 until 64) {
                    val sil = SilhouetteReader.of(crops[i], Squares.CELL, colours, palette)
                    if (labels[i] == "empty") {
                        (if (Squares.isLight(i / 8, i % 8)) lightBgs else darkBgs).add(sil.background)
                        continue
                    }
                    if (sil.hover) {
                        Log.i(TAG, "the cursor's outline is on a piece; waiting for a clean start frame")
                        return null
                    }
                    samples.getOrPut(labels[i]) { ArrayList() }.add(sil)
                }
                if (lightBgs.isEmpty() || darkBgs.isEmpty()) return null
                val lightBg = medianColour(lightBgs)
                val darkBg = medianColour(darkBgs)
                if (distance(lightBg, darkBg) <= Pieces.COVER_TOLERANCE) {
                    return null  // the two square colours cannot be told apart
                }
                // The phase landing's references: a crop per piece kind and square colour, from
                // a cell off the labelled left column and bottom row where it can be. Against a
                // crop on the other colour a knight at rest read 0.138 of a cell off, against
                // one on its own 0.010 (the desktop's stress sweep, seed 138: the flat set on
                // green, 2026-09-27). A king and a queen start on one colour: that crop stands
                // for both.
                val landingCrops = LinkedHashMap<Pair<String, Boolean>, LandingCrop>()
                for (r in 0 until 8) {
                    for (c in 0 until 8) {
                        val symbol = labels[r * 8 + c]
                        if (symbol == "empty") continue
                        val key = symbol to Squares.isLight(r, c)
                        if (key in landingCrops && (c == 0 || r == 7)) continue
                        landingCrops[key] = LandingCrop.of(crops[r * 8 + c])
                    }
                }
                for ((key, crop) in landingCrops.entries.toList()) {
                    landingCrops.putIfAbsent(key.first to !key.second, crop)
                }
                // Embedded at one resolution, as every later read is (`Pieces.canonical`); the
                // landing crops above keep the full detail.
                val seen = crops.map { Pieces.canonical(it) }
                val vectors = try {
                    embedder.embed(seen)
                } finally {
                    seen.forEach { it.release() }
                }
                if (!startReadsAsStart(vectors, labels)) {
                    Log.i(TAG, "a piece is hidden or spoiled; the next start frame gets a chance")
                    return null
                }
                return ThemeModel(
                    exemplars = samples.map { (symbol, sils) -> fuse(symbol, sils) },
                    lightBg = lightBg,
                    darkBg = darkBg,
                    pieceColours = colours,
                    palette = palette,
                    // Held at half precision, as the desktop holds them (`astype(np.float16)`)
                    // and every similarity is taken against them.
                    exemplarVectors = Array(vectors.size) { v ->
                        FloatArray(vectors[v].size) { halfPrecision(vectors[v][it]) }
                    },
                    exemplarLabels = labels,
                    whiteAtBottom = whiteAtBottom,
                    landingCrops = landingCrops,
                    landingPx = geometry.cellPx,
                )
            } finally {
                crops.forEach { it.release() }
            }
        }

        /**
         * Whether every piece cell of a start frame shows its piece, by the embeddings: a
         * cell hidden under a cursor or a badge, or erased, is nearer an empty square than
         * any piece of its own colour, and would be that piece's exemplar for the whole
         * game (`pieces._start_reads_as_start`).
         */
        fun startReadsAsStart(vectors: Array<FloatArray>, labels: List<String>): Boolean {
            val empties = labels.indices.filter { labels[it] == "empty" }
            if (empties.isEmpty()) return true
            for (i in labels.indices) {
                val label = labels[i]
                if (label == "empty") continue
                val own = labels.indices.filter {
                    it != i && labels[it] != "empty" &&
                        labels[it][0].isUpperCase() == label[0].isUpperCase()
                }
                if (own.isEmpty()) continue
                var nearestEmpty = Float.NEGATIVE_INFINITY
                for (j in empties) nearestEmpty = max(nearestEmpty, dot(vectors[i], vectors[j]))
                var nearestOwn = Float.NEGATIVE_INFINITY
                for (j in own) nearestOwn = max(nearestOwn, dot(vectors[i], vectors[j]))
                if (nearestEmpty > nearestOwn) return false
            }
            return true
        }

        /**
         * One exemplar from the silhouettes of the cells holding this piece kind: the tone
         * pixels at least half of them share (the union of a pair), the median brightness.
         */
        private fun fuse(symbol: String, sils: List<Sil>): Exemplar {
            val tone = BooleanArray(Squares.CELL * Squares.CELL) { i -> sils.count { it.tone[i] } * 2 >= sils.size }
            return Exemplar(symbol, tone, medianOfDoubles(sils.map { it.brightness }))
        }

        /**
         * Each side's fill and outline, pooled over its sixteen pieces: every piece's ink
         * is two tones, and pooled by luminance the heavier cluster is the fill. Pooled,
         * because a piece whose fill is the square's own colour shows only its outline
         * there, and read on its own would name the outline its fill.
         */
        private fun pieceColoursOf(samples: Map<String, List<Sil>>): PieceColours {
            val out = ArrayList<Pair<DoubleArray, DoubleArray?>>(2)
            for (white in listOf(true, false)) {
                val tones = ArrayList<Pair<DoubleArray, Double>>()
                for ((symbol, sils) in samples) {
                    if (symbol[0].isUpperCase() != white) continue
                    for (sil in sils) tones.addAll(sil.tones)
                }
                if (tones.isEmpty()) continue
                val lum = tones.map { 0.114 * it.first[0] + 0.587 * it.first[1] + 0.299 * it.first[2] }
                val weight = tones.map { it.second }
                val groups: List<List<Int>> = if (tones.size >= 2 && lum.max() - lum.min() > 1) {
                    val split = weightedMedian(lum, weight)
                    var above = lum.indices.filter { lum[it] > split }
                    var below = lum.indices.filter { lum[it] <= split }
                    if (above.isEmpty() || below.isEmpty()) {
                        above = lum.indices.filter { lum[it] >= split }
                        below = lum.indices.filter { lum[it] < split }
                    }
                    listOf(above, below)
                } else {
                    listOf(lum.indices.toList(), emptyList())
                }
                val sorted = groups.sortedByDescending { g -> if (g.isEmpty()) -1.0 else g.sumOf { weight[it] } }
                val fill = pooledColour(tones, weight, sorted[0])
                val edge = if (sorted[1].isEmpty()) null else pooledColour(tones, weight, sorted[1])
                out.add(fill to edge)
            }
            return out
        }

        private fun pooledColour(
            tones: List<Pair<DoubleArray, Double>>,
            weight: List<Double>,
            group: List<Int>,
        ): DoubleArray = DoubleArray(3) { ch ->
            weightedMedian(group.map { tones[it].first[ch] }, group.map { weight[it] })
        }

        /**
         * Every colour the set draws with: the ink of all its pieces on the start frame,
         * clustered (`pieces._set_palette`). A set's fills and outlines are two colours a
         * side; its details may be more (a pink emblem and red lettering on a navy piece, a
         * gold crown on a teal king), and each is a colour a drawing is told from.
         *
         * The ink is taken in the Python's order -- the piece kinds as the start position
         * first shows them, their cells in screen order, each silhouette's pixels row by row
         * -- and thinned the way numpy thins it, so the same pixels are clustered. k-means
         * seeds itself from OpenCV's random generator, the calling thread's, as the desktop's
         * does. The phone's tones stood in for this until the parity audit (2026-09-26): a
         * detail colour that is neither tone read as a drawing on every board, three cells of
         * every anarcandy board among them.
         */
        internal fun setPalette(samples: Map<String, List<Sil>>): List<DoubleArray> {
            var count = 0
            for (sils in samples.values) for (sil in sils) for (on in sil.mask) if (on) count++
            if (count == 0) return emptyList()
            val ink = FloatArray(count * 3)
            var at = 0
            for (sils in samples.values) {
                for (sil in sils) {
                    for (i in sil.mask.indices) {
                        if (!sil.mask[i]) continue
                        ink[at++] = sil.inner[i * 3]
                        ink[at++] = sil.inner[i * 3 + 1]
                        ink[at++] = sil.inner[i * 3 + 2]
                    }
                }
            }
            val pixels: FloatArray
            if (count > PALETTE_SAMPLE) {
                val picked = linspaceIndices(count, PALETTE_SAMPLE)
                pixels = FloatArray(picked.size * 3)
                for ((k, i) in picked.withIndex()) {
                    pixels[k * 3] = ink[i * 3]
                    pixels[k * 3 + 1] = ink[i * 3 + 1]
                    pixels[k * 3 + 2] = ink[i * 3 + 2]
                }
            } else {
                pixels = ink
            }
            val n = pixels.size / 3
            val k = min(PALETTE_COLOURS, n)
            if (k < 1) return emptyList()
            val data = Mat(n, 3, CvType.CV_32F)
            data.put(0, 0, pixels)
            val labels = Mat()
            val centres = Mat()
            try {
                val criteria = TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 20, 1.0)
                Core.kmeans(data, k, labels, criteria, 3, Core.KMEANS_PP_CENTERS, centres)
                val assigned = IntArray(n)
                labels.get(0, 0, assigned)
                val counts = IntArray(k)
                for (label in assigned) counts[label]++
                val centre = FloatArray(k * 3)
                centres.get(0, 0, centre)
                val out = ArrayList<DoubleArray>(k)
                for (c in 0 until k) {
                    if (counts[c] < PALETTE_MIN_SHARE * n) continue
                    out.add(DoubleArray(3) { centre[c * 3 + it].toDouble() })
                }
                return out
            } finally {
                data.release()
                labels.release()
                centres.release()
            }
        }

        /**
         * Clusters of the set's ink, and the share of the ink a cluster must hold to be a
         * colour of the set: a queen's crown is one piece of thirty-two, a gold highlight a
         * sliver of every gold piece, and both were drawings at eight clusters and one percent.
         */
        const val PALETTE_COLOURS = 16
        const val PALETTE_MIN_SHARE = 0.002
        const val PALETTE_SAMPLE = 30000 // ink pixels clustered, at most

        private fun medianColour(colours: List<DoubleArray>): DoubleArray =
            DoubleArray(3) { ch -> medianOfDoubles(colours.map { it[ch] }) }

        /** numpy's median of float64 values, kept in doubles. */
        private fun medianOfDoubles(values: List<Double>): Double {
            if (values.isEmpty()) return 0.0
            val sorted = values.sorted()
            val n = sorted.size
            return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }

        private fun weightedMedian(values: List<Double>, weights: List<Double>): Double {
            if (values.isEmpty()) return 0.0
            val order = values.indices.sortedBy { values[it] }
            val total = weights.sum()
            var running = 0.0
            for (i in order) {
                running += weights[i]
                if (running >= total / 2) return values[i]
            }
            return values[order.last()]
        }

        private fun dot(a: FloatArray, b: FloatArray): Float {
            var sum = 0f
            for (k in a.indices) sum += a[k] * b[k]
            return sum
        }
    }
}

private fun norm(v: DoubleArray): Double {
    var sum = 0.0
    for (x in v) sum += x * x
    return sqrt(sum)
}

private fun distance(a: DoubleArray, b: DoubleArray): Double {
    var sum = 0.0
    for (i in 0 until 3) {
        val d = a[i] - b[i]
        sum += d * d
    }
    return sqrt(sum)
}

private fun median(values: List<Float>): Double {
    if (values.isEmpty()) return 0.0
    val sorted = values.sorted()
    val n = sorted.size
    return if (n % 2 == 1) sorted[n / 2].toDouble()
    else (sorted[n / 2 - 1].toDouble() + sorted[n / 2].toDouble()) / 2.0
}
